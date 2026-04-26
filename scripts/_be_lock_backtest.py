"""Backtest BE+LOCK variants against classic BE on existing price_ticks.

Approach: for every closed paper trade of BE25_TP80_SL30 (and ETH equivalent),
replay the price_ticks from entry through horizon, applying both:
  - Classic BE25 logic (current production)
  - BE25_LOCK10 / LOCK15 variants (new logic)
Then compare outcomes head-to-head: avg pnl, WR, $/trade, exit reason mix.

Pure offline replay — uses live paper_trader._evaluate_trade_exit so the slip
factor matches production. Does NOT touch DB.

Usage:
    python scripts/_be_lock_backtest.py --since 2026-04-17 --chain solana
    python scripts/_be_lock_backtest.py --chain ethereum --base ETH_BE25_TP80_SL40_T2H
"""
import os
import sys
import argparse
from collections import defaultdict
from statistics import mean, median
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))

from supabase import create_client
from paper_trader import _evaluate_trade_exit


def replay_strategy(trade_template: dict, ticks: list, strategy: str,
                    tp_mult: float, sl_mult: float, horizon_min: int,
                    be_activation: float = 0.25) -> dict:
    """Replay a single trade through ticks under a strategy variant.

    trade_template: base dict carrying entry_price, position_usd, chain, etc.
    ticks: list of {"price": float, "ts": datetime}, sorted ascending.
    strategy: name (drives BE / BE+LOCK regex match in _evaluate_trade_exit).
    Returns {"status", "exit_price", "pnl_pct", "exit_minutes", "high_seen"}.
    """
    if not ticks:
        return {"status": "no_ticks", "pnl_pct": 0, "exit_minutes": 0}

    entry_price = float(trade_template["entry_price"])
    entry_ts = trade_template["created_at"]
    if isinstance(entry_ts, str):
        entry_ts = datetime.fromisoformat(entry_ts.replace("Z", "+00:00"))

    tp_price = entry_price * tp_mult
    sl_price = entry_price * sl_mult
    high_seen = entry_price

    for tick in ticks:
        ts = tick["ts"]
        price = float(tick["price"])
        elapsed_min = (ts - entry_ts).total_seconds() / 60
        if elapsed_min < 0:
            continue
        if elapsed_min > horizon_min:
            return {
                "status": "timeout",
                "exit_price": price,
                "pnl_pct": (price / entry_price) - 1,
                "exit_minutes": int(elapsed_min),
                "high_seen": high_seen,
            }
        if price > high_seen:
            high_seen = price

        # Build a synthetic trade dict for _evaluate_trade_exit.
        # IMPORTANT: created_at must be ISO string (eval calls .replace on it).
        synth_trade = dict(trade_template)
        synth_trade["strategy"] = strategy
        synth_trade["tp_price"] = tp_price
        synth_trade["sl_price"] = sl_price
        synth_trade["entry_price"] = entry_price
        synth_trade["high_price_seen"] = high_seen
        synth_trade["created_at"] = entry_ts.isoformat()
        synth_trade["horizon_minutes"] = horizon_min
        synth_trade["tranche_label"] = "main"
        # be_activation read from regex capture, no need to inject

        ev = _evaluate_trade_exit(
            synth_trade, price, ts,
            sell_slip_factor=1.0,  # we want raw price comparison
            sell_fee_bps=0,
            decision_price=price,
        )
        if ev and ev.get("status"):
            return {
                "status": ev["status"],
                "exit_price": ev.get("exit_price", price),
                "pnl_pct": ev.get("pnl_pct", (price / entry_price) - 1),
                "exit_minutes": int(elapsed_min),
                "high_seen": high_seen,
            }

    # Ran out of ticks before horizon
    last = ticks[-1]
    last_price = float(last["price"])
    last_elapsed = (last["ts"] - entry_ts).total_seconds() / 60
    return {
        "status": "no_horizon_reached",
        "exit_price": last_price,
        "pnl_pct": (last_price / entry_price) - 1,
        "exit_minutes": int(last_elapsed),
        "high_seen": high_seen,
    }


def fetch_ticks(sb, token_address: str, chain: str, since_iso: str,
                until_iso: str) -> list:
    """Fetch price_ticks for a single token in a time range."""
    out = []
    step, off = 1000, 0
    while True:
        q = (sb.table("price_ticks")
             .select("price_usd, fetched_at")
             .eq("token_address", token_address)
             .gte("fetched_at", since_iso)
             .lte("fetched_at", until_iso)
             .order("fetched_at", desc=False))
        r = q.range(off, off + step - 1).execute()
        if not r.data:
            break
        for t in r.data:
            ts = t["fetched_at"]
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            out.append({"price": float(t["price_usd"]), "ts": ts})
        if len(r.data) < step:
            break
        off += step
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-04-17", help="Trade cutoff")
    ap.add_argument("--chain", default="solana", choices=["solana", "ethereum"])
    ap.add_argument("--base", default="BE25_TP80_SL30",
                    help="Baseline strategy whose closed trades we replay")
    ap.add_argument("--tp", type=float, default=1.80, help="TP mult for variants")
    ap.add_argument("--sl", type=float, default=0.70, help="SL mult for variants")
    ap.add_argument("--horizon", type=int, default=30, help="Horizon minutes")
    ap.add_argument("--max-trades", type=int, default=200, help="Cap for speed")
    args = ap.parse_args()

    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

    # Fetch baseline trades
    rows = []
    step, off = 1000, 0
    while len(rows) < args.max_trades * 2:  # over-fetch in case of empty ticks
        r = (sb.table("paper_trades")
             .select("id, token_address, entry_price, position_usd, "
                     "chain, status, pnl_pct, created_at, exit_at, kol_group")
             .eq("strategy", args.base)
             .eq("chain", args.chain)
             .neq("status", "open").neq("status", "closing")
             .gte("created_at", args.since)
             .order("created_at", desc=False)
             .range(off, off + step - 1)
             .execute())
        if not r.data:
            break
        rows.extend(r.data)
        if len(r.data) < step:
            break
        off += step

    rows = rows[:args.max_trades]
    print(f"Backtesting {len(rows)} {args.base} trades ({args.chain}, since {args.since})")
    print(f"Variants tested: BE25 (baseline) | BE25_LOCK5 | BE25_LOCK10 | BE25_LOCK15 | BE25_LOCK20")
    print()

    # Build variant configs
    chain_pfx = "ETH_" if args.chain == "ethereum" else ""
    variants = [
        ("baseline", f"{chain_pfx}BE25_TP80_SL30"),  # classic BE
        ("LOCK5",    f"{chain_pfx}BE25_LOCK5_TP80_SL30"),
        ("LOCK10",   f"{chain_pfx}BE25_LOCK10_TP80_SL30"),
        ("LOCK15",   f"{chain_pfx}BE25_LOCK15_TP80_SL30"),
        ("LOCK20",   f"{chain_pfx}BE25_LOCK20_TP80_SL30"),
    ]

    results = {label: [] for label, _ in variants}

    for i, t in enumerate(rows):
        if i % 25 == 0:
            print(f"  [{i}/{len(rows)}] processing...")
        if not t.get("entry_price") or not t.get("token_address"):
            continue
        # Time window for ticks: from entry to entry + horizon + buffer
        entry_ts = datetime.fromisoformat(t["created_at"].replace("Z", "+00:00"))
        until = (entry_ts.timestamp() + (args.horizon + 5) * 60)
        until_iso = datetime.fromtimestamp(until, tz=timezone.utc).isoformat()
        since_iso = entry_ts.isoformat()
        ticks = fetch_ticks(sb, t["token_address"], args.chain, since_iso, until_iso)
        if len(ticks) < 2:
            continue

        for label, strat_name in variants:
            res = replay_strategy(
                t, ticks, strat_name,
                tp_mult=args.tp, sl_mult=args.sl, horizon_min=args.horizon,
            )
            results[label].append(res)

    # Aggregate per variant
    print()
    print(f"{'Variant':<12} {'N':>5} {'avg_pnl%':>10} {'med%':>8} {'WR%':>6} "
          f"{'tp_hit':>7} {'sl_hit':>7} {'be_stop':>8} {'timeout':>8}")
    print("-" * 80)
    for label, _ in variants:
        recs = results[label]
        if not recs:
            print(f"{label:<12} 0 (no data)")
            continue
        n = len(recs)
        pnl_avg = mean(r["pnl_pct"] for r in recs) * 100
        pnl_med = median(r["pnl_pct"] for r in recs) * 100
        wr = sum(1 for r in recs if r["pnl_pct"] > 0) / n * 100
        statuses = defaultdict(int)
        for r in recs:
            statuses[r.get("status", "?")] += 1
        print(f"{label:<12} {n:>5} {pnl_avg:>+9.2f}% {pnl_med:>+7.2f}% "
              f"{wr:>5.0f}% {statuses['tp_hit']:>7} {statuses['sl_hit']:>7} "
              f"{statuses['be_stop']:>8} {statuses['timeout']:>8}")

    # Pairwise delta vs baseline (per-trade matched)
    print()
    print("=== Pairwise vs baseline (matched per-trade delta) ===")
    base_pnls = [r["pnl_pct"] for r in results["baseline"]]
    for label, _ in variants:
        if label == "baseline":
            continue
        var_pnls = [r["pnl_pct"] for r in results[label]]
        n = min(len(base_pnls), len(var_pnls))
        if n == 0:
            continue
        deltas = [var_pnls[i] - base_pnls[i] for i in range(n)]
        avg_delta = mean(deltas) * 100
        positive = sum(1 for d in deltas if d > 0)
        negative = sum(1 for d in deltas if d < 0)
        zero = n - positive - negative
        print(f"{label:<12} N={n:>4}  delta_avg={avg_delta:>+6.2f}pp  "
              f"better={positive:>4} worse={negative:>4} same={zero:>4}")


if __name__ == "__main__":
    main()
