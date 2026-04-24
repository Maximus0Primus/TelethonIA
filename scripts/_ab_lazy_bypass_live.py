"""A/B: chiffrer l'impact de LAZY throttle sur live FAST/BE strats.

Pour chaque live trade fermé avec eval_history disponible, replay l'exit logic:
  - Variant A (actuel): LAZY throttle actif → skip eval si <180s depuis dernière.
  - Variant B (bypass):  LAZY désactivé → eval à chaque tick.

Comparaison:
  - pnl_live_real (ce que live a actually booked, avec Jupiter Ultra fill réel)
  - pnl_replay_A  (replay avec throttle — doit matcher live pour les LAZY)
  - pnl_replay_B  (replay sans throttle — ce que live ferait si on retirait LAZY)

Delta B vs A = impact net du fix "live bypass LAZY" sur les 14 derniers jours.
Positif = le fix gagne de l'argent. Négatif = le throttle te fait gagner.
"""
import os
import sys
import statistics as st
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))

from supabase import create_client
from paper_trader import (
    _evaluate_trade_exit,
    _last_eval_ts,
    LAZY_STRATEGIES,
    SELL_FEE_BPS,
    SELL_SLIPPAGE_BPS,
)

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

SINCE = os.environ.get("AB_SINCE",
    (datetime.now(timezone.utc) - timedelta(days=14)).isoformat())


def fetch_all(tbl, sel, **f):
    out, step, off = [], 1000, 0
    while True:
        q = sb.table(tbl).select(sel)
        for k, v in f.items():
            if k.startswith("gte_"): q = q.gte(k[4:], v)
            elif k.startswith("eq_"): q = q.eq(k[3:], v)
        r = q.range(off, off+step-1).execute()
        if not r.data: break
        out.extend(r.data)
        if len(r.data) < step: break
        off += step
    return out


def replay(trade: dict, eval_history: list, bypass_lazy: bool) -> dict | None:
    """Replay one trade's eval_history with/without LAZY throttle."""
    if not eval_history:
        return None
    trade_id = str(trade["id"])
    _last_eval_ts.pop(trade_id, None)

    # Use paper-accurate slip/fee (what _paper_sim_ev computes in live_trader)
    sell_slip = 1 - SELL_SLIPPAGE_BPS / 10_000

    # Rebuild a mutable fake trade for the replay so we don't corrupt caller data
    fake = dict(trade)
    fake["high_price_seen"] = float(trade["entry_price"])

    # Force symmetry mode
    if bypass_lazy:
        # Use live_sync entry_source to activate the existing v144.6 bypass
        # (equivalent to adding `source==rt_live` bypass, without mutating code)
        fake["entry_source"] = "live_sync"
    else:
        # Keep real entry_source ("ultra") to activate LAZY throttle as in prod
        fake["entry_source"] = trade.get("entry_source") or "ultra"

    last_ev = None
    for poll in eval_history:
        try:
            dec_p = poll.get("d")
            exec_p = poll.get("e")
            t_iso = poll.get("t")
            if dec_p is None or exec_p is None or not t_iso:
                continue
            t = datetime.fromisoformat(t_iso.replace("Z", "+00:00"))
        except Exception:
            continue

        ev = _evaluate_trade_exit(
            fake, exec_p, t, sell_slip,
            sell_fee_bps=SELL_FEE_BPS,
            decision_price=dec_p,
        )
        if ev is None:
            continue

        # Propagate high_price_seen update across polls (like prod)
        if ev.get("high_price_seen") is not None:
            nh = ev["high_price_seen"]
            if nh > float(fake.get("high_price_seen") or 0):
                fake["high_price_seen"] = nh

        # Exit fired → stop replay
        if ev.get("status") is not None:
            last_ev = ev
            break

    _last_eval_ts.pop(trade_id, None)

    if last_ev is None:
        # No exit during recorded polls — treat as timeout at last tick
        last_poll = eval_history[-1]
        last_price = float(last_poll.get("e") or 0)
        entry = float(trade["entry_price"] or 0)
        pnl_pct = (last_price / entry - 1) if entry > 0 else 0
        return {"status": "timeout_replay", "pnl_pct": pnl_pct, "exit_price": last_price}

    return last_ev


def main():
    print(f"A/B LAZY throttle on live trades — window since {SINCE}")
    print(f"LAZY strategies: {sorted(LAZY_STRATEGIES)}")

    rows = fetch_all(
        "paper_trades",
        "id,symbol,strategy,source,status,entry_source,entry_price,sl_price,tp_price,"
        "dex_spot_price_at_entry,horizon_minutes,position_usd,tranche_label,pnl_pct,"
        "pnl_usd,paper_sim_pnl_pct,eval_history,created_at,exit_at,rt_is_pump_fun",
        gte_created_at=SINCE,
    )
    closed = [
        r for r in rows
        if r.get("source") == "rt_live"
        and r.get("status") in ("sl_hit", "trail_stop", "tp_hit", "timeout", "be_stop")
        and isinstance(r.get("eval_history"), list)
        and len(r["eval_history"]) >= 1
    ]
    print(f"  {len(closed)} closed live trades with eval_history")

    per_strat = defaultdict(list)  # strategy -> list of dict per trade
    for tr in closed:
        a = replay(tr, tr["eval_history"], bypass_lazy=False)
        b = replay(tr, tr["eval_history"], bypass_lazy=True)
        if a is None or b is None:
            continue
        pnl_real_usd = float(tr.get("pnl_usd") or 0)
        pos_usd = float(tr.get("position_usd") or 0)
        a_usd = round(pos_usd * float(a.get("pnl_pct") or 0), 4)
        b_usd = round(pos_usd * float(b.get("pnl_pct") or 0), 4)
        per_strat[tr["strategy"]].append({
            "symbol": tr["symbol"],
            "pnl_real_usd": pnl_real_usd,
            "pnl_a_usd": a_usd,          # LAZY on (as live runs today)
            "pnl_b_usd": b_usd,          # LAZY bypass (proposed fix)
            "delta_usd": b_usd - a_usd,  # Impact of bypass
            "status_a": a.get("status"),
            "status_b": b.get("status"),
            "is_pump": tr.get("rt_is_pump_fun"),
        })

    # Summary
    print(f"\n{'Strategy':<25}{'N':>5}{'LAZY?':>7}"
          f"{'$real':>10}{'$A (lazy)':>12}{'$B (bypass)':>14}"
          f"{'Δ$B-A':>10}{'Δ%':>8}")
    print("-" * 100)

    total = {"real": 0.0, "a": 0.0, "b": 0.0, "delta": 0.0, "n": 0}
    for strat in sorted(per_strat.keys()):
        rows_s = per_strat[strat]
        n = len(rows_s)
        real = sum(r["pnl_real_usd"] for r in rows_s)
        a = sum(r["pnl_a_usd"] for r in rows_s)
        b = sum(r["pnl_b_usd"] for r in rows_s)
        delta = b - a
        delta_pct = (delta / abs(a) * 100) if abs(a) > 0.01 else 0
        is_lazy = "LAZY" if strat in LAZY_STRATEGIES else "-"
        print(f"{strat:<25}{n:>5}{is_lazy:>7}"
              f"{real:>+10.2f}{a:>+12.2f}{b:>+14.2f}"
              f"{delta:>+10.2f}{delta_pct:>+7.1f}%")
        total["real"] += real
        total["a"] += a
        total["b"] += b
        total["delta"] += delta
        total["n"] += n

    print("-" * 100)
    print(f"{'TOTAL':<25}{total['n']:>5}{'':>7}"
          f"{total['real']:>+10.2f}{total['a']:>+12.2f}{total['b']:>+14.2f}"
          f"{total['delta']:>+10.2f}")

    # Per-trade detail for LAZY strats (where the fix matters)
    print(f"\n=== Per-trade LAZY strat detail (where bypass changes behavior) ===")
    changed = []
    for strat, rows_s in per_strat.items():
        if strat not in LAZY_STRATEGIES:
            continue
        for r in rows_s:
            if abs(r["delta_usd"]) > 0.01:
                changed.append((strat, r))

    if not changed:
        print("  (no LAZY trade changed behavior — throttle never mattered in this window)")
    else:
        print(f"  {len(changed)} LAZY trade(s) where bypass changes outcome:")
        for strat, r in sorted(changed, key=lambda x: x[1]["delta_usd"]):
            tag = "[pump]" if r["is_pump"] else ""
            print(f"    {r['symbol']:<12}{strat:<25}"
                  f"A={r['pnl_a_usd']:+.3f}$ B={r['pnl_b_usd']:+.3f}$ "
                  f"ΔB-A={r['delta_usd']:+.3f}$ "
                  f"[{r['status_a']}→{r['status_b']}] {tag}")

    # Decision guidance
    print("\n=== Decision guidance ===")
    lazy_delta = sum(
        r["delta_usd"]
        for strat, rows_s in per_strat.items() if strat in LAZY_STRATEGIES
        for r in rows_s
    )
    lazy_n = sum(len(rows_s) for strat, rows_s in per_strat.items() if strat in LAZY_STRATEGIES)
    days = 14
    print(f"  LAZY strats: {lazy_n} trades over {days}d")
    print(f"  Bypass impact: {lazy_delta:+.2f}$ total ({lazy_delta/days:+.2f}$/day)")
    if lazy_delta < -1:
        print(f"  → LAZY throttle pays {-lazy_delta:.2f}$/{days}d. Keeping it is +EV (short term).")
        print(f"  → But you trade a strategy that differs from config. Scaling will erase this.")
    elif lazy_delta > 1:
        print(f"  → Bypass pays {lazy_delta:.2f}$/{days}d. Fix is +EV.")
    else:
        print(f"  → Impact is noise (<$1/14d). Fix for coherence, no $ cost.")


if __name__ == "__main__":
    main()
