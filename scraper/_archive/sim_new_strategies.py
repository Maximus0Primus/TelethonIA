"""
v142 — Ex-ante measurement of 4 candidate strategies against historical paper trades.

Uses eval_history (persisted per-poll tick data, v138+) as ground-truth price series.
Replays each closed paper trade under the 4 hypotheses and compares to actual PnL:

  1. BOND_FAST_TP40_SL25_T20       — gate: liq < $3K, exits: TP40/SL25/trail10@15/T20
  2. TIME_DECAY_V2                  — TP decays 80→10% over 30min, BE arms at t=5min
  3. PEAK_TRAIL_V2                  — 4-tier trail (12/20/35/50%) by peak height
  4. MOMENTUM_CONFIRM_ENTRY         — delay 60s, enter only if price up ≥2%

Output: per-strategy WR / avg PnL / sum PnL (on actual position_usd), paired with
the actual paper result on the same trade set. No DB writes. Read-only.

Usage:
    python scraper/sim_new_strategies.py              # default 14 days
    python scraper/sim_new_strategies.py --days 7
    python scraper/sim_new_strategies.py --out out.json
"""

import argparse
import json
import os
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client


SCRAPER_DIR = Path(__file__).resolve().parent
load_dotenv(SCRAPER_DIR / ".env")


# ---- Strategy params (editable) -------------------------------------------

BOND_FAST = {
    "max_liq": 3_000,
    "tp_mult": 1.40,
    "sl_mult": 0.75,
    "trail_pct": 0.10,
    "trail_act": 0.15,
    "timeout_min": 20,
}

TIME_DECAY_V2 = {
    "tp_start": 1.80,     # +80% at t=0
    "tp_mid": 1.40,       # +40% at t=5min
    "tp_late": 1.00,      # any profit at t=15min (effectively BE-or-profit)
    "sl_start": 0.70,     # -30% first 5 min
    "be_minute": 5,       # BE armed after t=5min regardless of peak
    "timeout_min": 30,
    "decay_breakpoints": [(0, 1.80), (5, 1.40), (15, 1.00), (30, 1.00)],
}

PEAK_TRAIL_V2 = {
    "tiers": [  # (peak_mult_gte, trail_pct)
        (1.30, 0.12),
        (1.80, 0.20),
        (3.00, 0.35),
        (6.00, 0.50),
    ],
    "trail_act": 0.20,  # trail arms at +20% peak
    "sl_mult": 0.50,
    "timeout_min": 60,
}

MOMENTUM_CONFIRM = {
    "delay_min": 1.0,     # 60s
    "threshold": 1.02,    # +2%
    # After confirm, apply FAST_TP80_SL25 baseline:
    "tp_mult": 1.80,
    "sl_mult": 0.75,
    "timeout_min": 30,
}


# ---- Replay primitives -----------------------------------------------------

def _parse_ts(s):
    if isinstance(s, datetime):
        return s
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _iter_ticks(trade):
    """Yield (minute_from_open, decision_price, exec_price, high_at_poll) per poll.
    eval_history format (v138): [{"t": iso, "d": decision, "e": exec, "h": high}, ...]
    """
    eh = trade.get("eval_history") or []
    if not eh:
        return
    created = _parse_ts(trade["created_at"])
    entry = float(trade.get("entry_price") or 0)
    if entry <= 0:
        return
    for poll in eh:
        try:
            t = _parse_ts(poll["t"])
            mins = (t - created).total_seconds() / 60.0
            d = float(poll.get("d") or 0) or None
            e = float(poll.get("e") or 0) or None
            h = float(poll.get("h") or 0) or None
            price = e or d
            if price is None or price <= 0:
                continue
            yield mins, price, (h if h and h > 0 else price)
        except (KeyError, TypeError, ValueError):
            continue


def _simulate_bond_fast(trade):
    """Skip if liq >= 3000. Else replay TP40/SL25/trail10@15/T20 via eval_history."""
    liq = float(trade.get("rt_liquidity_usd") or 0)
    is_bonding = bool(trade.get("rt_is_pump_fun"))
    if not is_bonding and liq >= BOND_FAST["max_liq"]:
        return {"taken": False}

    entry = float(trade["entry_price"])
    tp = entry * BOND_FAST["tp_mult"]
    sl = entry * BOND_FAST["sl_mult"]
    trail_act_p = entry * (1 + BOND_FAST["trail_act"])
    trail_pct = BOND_FAST["trail_pct"]
    timeout = BOND_FAST["timeout_min"]

    high = entry
    trail_armed = False
    for mins, price, h in _iter_ticks(trade):
        high = max(high, h, price)
        if mins >= timeout:
            return {"taken": True, "status": "timeout", "pnl_pct": (price / entry) - 1, "exit_min": mins}
        if price <= sl:
            return {"taken": True, "status": "sl_hit", "pnl_pct": (sl / entry) - 1, "exit_min": mins}
        if price >= tp:
            return {"taken": True, "status": "tp_hit", "pnl_pct": (tp / entry) - 1, "exit_min": mins}
        if high >= trail_act_p:
            trail_armed = True
        if trail_armed:
            trigger = high * (1 - trail_pct)
            if price <= trigger and trigger > entry:
                return {"taken": True, "status": "trail_stop", "pnl_pct": (trigger / entry) - 1, "exit_min": mins}
    # No tick hit an exit → use last known price (trade may still be open mid-replay)
    return {"taken": True, "status": "noexit", "pnl_pct": 0.0, "exit_min": 0}


def _simulate_time_decay_v2(trade):
    """TP step-schedule (80→40→0% over 30min), SL→entry at t=5min regardless of peak."""
    entry = float(trade["entry_price"])
    timeout = TIME_DECAY_V2["timeout_min"]
    be_minute = TIME_DECAY_V2["be_minute"]
    sl_start = entry * TIME_DECAY_V2["sl_start"]

    def _tp_at(mins):
        # Piecewise-linear between breakpoints
        bps = TIME_DECAY_V2["decay_breakpoints"]
        for i in range(len(bps) - 1):
            m1, v1 = bps[i]
            m2, v2 = bps[i + 1]
            if m1 <= mins <= m2:
                if m2 == m1:
                    return v2
                return v1 + (v2 - v1) * (mins - m1) / (m2 - m1)
        return bps[-1][1]

    for mins, price, h in _iter_ticks(trade):
        if mins >= timeout:
            return {"taken": True, "status": "timeout", "pnl_pct": (price / entry) - 1, "exit_min": mins}
        effective_sl = max(sl_start, entry if mins >= be_minute else 0.0)
        if effective_sl > 0 and price <= effective_sl:
            status = "sl_hit" if effective_sl == sl_start else "be_stop"
            return {"taken": True, "status": status, "pnl_pct": (effective_sl / entry) - 1, "exit_min": mins}
        tp_mult = _tp_at(mins)
        if tp_mult > 1.0:
            tp_px = entry * tp_mult
            if price >= tp_px:
                return {"taken": True, "status": "tp_hit", "pnl_pct": (tp_px / entry) - 1, "exit_min": mins}
        elif tp_mult <= 1.0 and mins >= 15 and price >= entry:
            # Late phase: take any profit or breakeven if positive
            return {"taken": True, "status": "tp_late", "pnl_pct": (price / entry) - 1, "exit_min": mins}
    return {"taken": True, "status": "noexit", "pnl_pct": 0.0, "exit_min": 0}


def _simulate_peak_trail_v2(trade):
    """Multi-tier trail. Wider trail for higher peak. Pure trail, no TP."""
    entry = float(trade["entry_price"])
    tiers = sorted(PEAK_TRAIL_V2["tiers"])
    sl = entry * PEAK_TRAIL_V2["sl_mult"]
    timeout = PEAK_TRAIL_V2["timeout_min"]
    trail_act_p = entry * (1 + PEAK_TRAIL_V2["trail_act"])

    def _trail_for(peak):
        # Select the trail_pct for the highest peak_mult that is ≤ current peak / entry
        ratio = peak / entry if entry > 0 else 1.0
        best_pct = tiers[0][1]
        for mult, pct in tiers:
            if ratio >= mult:
                best_pct = pct
            else:
                break
        return best_pct

    high = entry
    trail_armed = False
    for mins, price, h in _iter_ticks(trade):
        high = max(high, h, price)
        if mins >= timeout:
            return {"taken": True, "status": "timeout", "pnl_pct": (price / entry) - 1, "exit_min": mins}
        if price <= sl:
            return {"taken": True, "status": "sl_hit", "pnl_pct": (sl / entry) - 1, "exit_min": mins}
        if high >= trail_act_p:
            trail_armed = True
        if trail_armed:
            trail_pct = _trail_for(high)
            trigger = high * (1 - trail_pct)
            if price <= trigger and trigger > entry:
                return {"taken": True, "status": "trail_stop", "pnl_pct": (trigger / entry) - 1, "exit_min": mins}
    return {"taken": True, "status": "noexit", "pnl_pct": 0.0, "exit_min": 0}


def _simulate_momentum_confirm(trade):
    """Skip if at t≈60s price < entry*1.02. Else apply FAST_TP80_SL25 from t=60s."""
    entry = float(trade["entry_price"])
    delay = MOMENTUM_CONFIRM["delay_min"]
    threshold = MOMENTUM_CONFIRM["threshold"]

    # Find tick closest to t=60s
    confirm_price = None
    confirm_high = entry
    for mins, price, h in _iter_ticks(trade):
        confirm_high = max(confirm_high, h)
        if mins >= delay:
            confirm_price = price
            break

    if confirm_price is None:
        return {"taken": False, "reason": "no_tick_post_delay"}
    if confirm_price < entry * threshold:
        return {"taken": False, "reason": "price_not_up_2pct"}

    # Confirmed. Simulate from t=delay using confirm_price as new entry.
    new_entry = confirm_price
    tp = new_entry * MOMENTUM_CONFIRM["tp_mult"]
    sl = new_entry * MOMENTUM_CONFIRM["sl_mult"]
    timeout = MOMENTUM_CONFIRM["timeout_min"]

    for mins, price, h in _iter_ticks(trade):
        if mins < delay:
            continue
        elapsed = mins - delay
        if elapsed >= timeout:
            return {"taken": True, "status": "timeout", "pnl_pct": (price / new_entry) - 1, "exit_min": elapsed,
                    "confirm_entry": new_entry}
        if price <= sl:
            return {"taken": True, "status": "sl_hit", "pnl_pct": (sl / new_entry) - 1, "exit_min": elapsed,
                    "confirm_entry": new_entry}
        if price >= tp:
            return {"taken": True, "status": "tp_hit", "pnl_pct": (tp / new_entry) - 1, "exit_min": elapsed,
                    "confirm_entry": new_entry}
    return {"taken": True, "status": "noexit", "pnl_pct": 0.0, "exit_min": 0, "confirm_entry": new_entry}


# ---- Aggregation -----------------------------------------------------------

def _summarize(label, results, trades):
    """results: list of dicts aligned with trades. Compute N, WR, avg/sum pnl."""
    taken = [(r, t) for r, t in zip(results, trades) if r.get("taken")]
    skipped = len(trades) - len(taken)
    if not taken:
        return {"label": label, "n_taken": 0, "n_skipped": skipped}
    pnl_pcts = [r["pnl_pct"] for r, _ in taken]
    winners = sum(1 for p in pnl_pcts if p > 0)
    # For pnl_usd, use actual position_usd from the trade. For skipped trades in
    # MOMENTUM_CONFIRM, pnl_usd=0 (no position taken).
    pnl_usd = sum((r["pnl_pct"]) * float(t.get("position_usd") or 0) for r, t in taken)
    statuses = defaultdict(int)
    for r, _ in taken:
        statuses[r.get("status", "?")] += 1
    return {
        "label": label,
        "n_trades_total": len(trades),
        "n_taken": len(taken),
        "n_skipped": skipped,
        "wr": round(winners / len(taken), 3) if taken else 0,
        "avg_pnl_pct": round(statistics.mean(pnl_pcts), 4),
        "median_pnl_pct": round(statistics.median(pnl_pcts), 4),
        "sum_pnl_usd": round(pnl_usd, 2),
        "p95_pnl_pct": round(sorted(pnl_pcts)[max(0, min(len(pnl_pcts) - 1, int(len(pnl_pcts) * 0.95)))], 4),
        "p05_pnl_pct": round(sorted(pnl_pcts)[max(0, min(len(pnl_pcts) - 1, int(len(pnl_pcts) * 0.05)))], 4),
        "by_status": dict(statuses),
    }


def _actual(trade):
    """Actual paper result."""
    pnl = trade.get("pnl_pct")
    if pnl is None:
        return {"taken": False}
    return {"taken": True, "status": trade.get("status", "?"), "pnl_pct": float(pnl)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--out", type=str, default=str(SCRAPER_DIR / "sim_new_strategies.json"))
    parser.add_argument("--min_N_history", type=int, default=3,
                        help="Minimum eval_history polls required to include a trade")
    args = parser.parse_args()

    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    client = create_client(url, key)

    since = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()
    print(f"=== sim_new_strategies — since {since} ({args.days} days) ===\n")

    # Fetch closed, non-shadow paper trades with eval_history
    resp = (
        client.table("paper_trades")
        .select("id,symbol,token_address,strategy,source,status,pnl_pct,pnl_usd,"
                "entry_price,exit_price,position_usd,rt_liquidity_usd,rt_is_pump_fun,"
                "created_at,exit_at,eval_history")
        .neq("status", "open")
        .neq("status", "closing")
        .eq("is_shadow", False)
        .gte("created_at", since)
        .not_.is_("eval_history", "null")
        .limit(5000)
        .execute()
    )
    rows = resp.data or []
    # Filter to trades with enough tick history
    trades = [r for r in rows if isinstance(r.get("eval_history"), list) and len(r["eval_history"]) >= args.min_N_history]
    print(f"[COUNTS] fetched={len(rows)} with_history>={args.min_N_history}={len(trades)}\n")
    if not trades:
        print("No trades with sufficient eval_history. Try --min_N_history 1 or --days 30.")
        return

    # Run simulations
    actual = [_actual(t) for t in trades]
    bond = [_simulate_bond_fast(t) for t in trades]
    td2 = [_simulate_time_decay_v2(t) for t in trades]
    pt2 = [_simulate_peak_trail_v2(t) for t in trades]
    mc = [_simulate_momentum_confirm(t) for t in trades]

    summaries = [
        _summarize("ACTUAL (paper)", actual, trades),
        _summarize("BOND_FAST_TP40_SL25_T20", bond, trades),
        _summarize("TIME_DECAY_V2", td2, trades),
        _summarize("PEAK_TRAIL_V2", pt2, trades),
        _summarize("MOMENTUM_CONFIRM (+ FAST_TP80_SL25)", mc, trades),
    ]

    # Print table
    print(f"{'Strategy':<38} {'N_taken':>8} {'N_skip':>7} {'WR':>6} {'avg':>8} {'p05':>8} {'p95':>8} {'sumPnL$':>10}")
    print("-" * 100)
    for s in summaries:
        if s["n_taken"] == 0:
            print(f"{s['label']:<38} {'0':>8} {s.get('n_skipped', 0):>7} {'—':>6} {'—':>8} {'—':>8} {'—':>8} {'0':>10}")
            continue
        print(f"{s['label']:<38} {s['n_taken']:>8} {s['n_skipped']:>7} "
              f"{s['wr']:>6.1%} {s['avg_pnl_pct']:>+8.1%} {s['p05_pnl_pct']:>+8.1%} "
              f"{s['p95_pnl_pct']:>+8.1%} {s['sum_pnl_usd']:>+10.2f}")

    print("\n=== Paired delta vs ACTUAL (same trade set, taken by both) ===")
    base = actual
    for variant_label, variant in [("BOND_FAST", bond), ("TIME_DECAY_V2", td2),
                                    ("PEAK_TRAIL_V2", pt2), ("MOMENTUM_CONFIRM", mc)]:
        deltas = []
        for a, v in zip(base, variant):
            if a.get("taken") and v.get("taken"):
                deltas.append(v["pnl_pct"] - a["pnl_pct"])
        if not deltas:
            print(f"{variant_label}: no paired trades")
            continue
        winners = sum(1 for d in deltas if d > 0)
        print(f"{variant_label:<22} N={len(deltas):<5} mean_delta={statistics.mean(deltas):+.2%} "
              f"median={statistics.median(deltas):+.2%} wr_improve={winners/len(deltas):.1%}")

    # Status breakdown for each strat
    print("\n=== Exit-type distribution ===")
    for s in summaries:
        if s["n_taken"] == 0:
            continue
        bs = s.get("by_status", {})
        total = sum(bs.values())
        parts = " ".join(f"{k}={v}({100*v/total:.0f}%)" for k, v in sorted(bs.items(), key=lambda x: -x[1]))
        print(f"{s['label']:<38} {parts}")

    # Save JSON
    out_path = Path(args.out)
    out_path.write_text(json.dumps({
        "since": since,
        "days": args.days,
        "n_trades": len(trades),
        "min_N_history": args.min_N_history,
        "summaries": summaries,
        "params": {
            "BOND_FAST": BOND_FAST,
            "TIME_DECAY_V2": TIME_DECAY_V2,
            "PEAK_TRAIL_V2": PEAK_TRAIL_V2,
            "MOMENTUM_CONFIRM": MOMENTUM_CONFIRM,
        },
    }, indent=2, default=str))
    print(f"\n[OK] JSON written to {out_path}")


if __name__ == "__main__":
    main()
