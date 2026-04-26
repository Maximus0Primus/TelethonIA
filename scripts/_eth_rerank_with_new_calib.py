"""Re-rank ETH paper strategies under the v14e.28 recalibrated slip model.

Pulls all closed ETH paper trades, recomputes expected pnl_pct using the new
gas+slip formula (was: 200 bps + $7.50 gas -> now 100 bps + $1.50 gas), and
aggregates per strategy. Shows the delta vs the old-calibration PnL stored
in DB, so you can see which strategies got promoted/demoted by the recal.

Usage:
    python scripts/_eth_rerank_with_new_calib.py
    python scripts/_eth_rerank_with_new_calib.py --since 2026-04-01 --min-n 10
"""
import os, sys, argparse
from collections import defaultdict
from statistics import mean, median

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client

# Old calibration (pre v14e.28)
OLD_GAS_USD = 7.50
OLD_BUY_BPS = 200
OLD_SELL_BPS = 200

# New calibration (v14e.28, Apr 26 empirical)
from strategies import (
    ETH_GAS_COST_USD_PER_SIDE as NEW_GAS_USD,
    ETH_BUY_SLIPPAGE_BPS as NEW_BUY_BPS,
    ETH_SELL_SLIPPAGE_BPS as NEW_SELL_BPS,
)


def slip_bps(pos_usd: float, gas_usd: float, base_slip_bps: int) -> int:
    """Mirror paper_trader._evm_slip_bps_with_gas: slip + gas-as-bps, clamped."""
    pos = max(float(pos_usd or 0), 1.0)
    gas_bps = int((gas_usd / pos) * 10_000)
    return max(50, min(2000, base_slip_bps + gas_bps))


def reprice(entry: float, exit_p: float, pos_usd: float,
            gas_usd: float, buy_bps: int, sell_bps: int) -> float:
    """Apply buy slip on entry, sell slip on exit. Return net pnl_pct."""
    if not entry or not exit_p or entry <= 0:
        return 0.0
    buy_slip = slip_bps(pos_usd, gas_usd, buy_bps) / 10_000
    sell_slip = slip_bps(pos_usd, gas_usd, sell_bps) / 10_000
    eff_entry = entry * (1 + buy_slip)
    eff_exit = exit_p * (1 - sell_slip)
    return eff_exit / eff_entry - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-04-01", help="ISO date for trade cutoff")
    ap.add_argument("--min-n", type=int, default=5, help="Min trades to report")
    ap.add_argument("--source", default="all", choices=["rt", "rt_live", "all"],
                    help="Trade source filter (default all)")
    args = ap.parse_args()

    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

    print(f"Re-ranking ETH strategies under v14e.28 recalibration")
    print(f"  Old: gas ${OLD_GAS_USD}/side, slip {OLD_BUY_BPS}/{OLD_SELL_BPS} bps")
    print(f"  New: gas ${NEW_GAS_USD}/side, slip {NEW_BUY_BPS}/{NEW_SELL_BPS} bps")
    print(f"  Since: {args.since} | min N: {args.min_n}")
    print()

    # Pull all closed ETH paper trades
    rows = []
    step, off = 1000, 0
    while True:
        q = (sb.table("paper_trades")
             .select("strategy, source, position_usd, entry_price, exit_price, "
                     "pnl_pct, status, created_at, chain")
             .eq("chain", "ethereum")
             .neq("status", "open")
             .neq("status", "closing")
             .gte("created_at", args.since))
        if args.source != "all":
            q = q.eq("source", args.source)
        r = q.range(off, off + step - 1).execute()
        if not r.data:
            break
        rows.extend(r.data)
        if len(r.data) < step:
            break
        off += step

    print(f"Loaded {len(rows)} closed ETH trades")

    # Filter trades with usable price data
    usable = [r for r in rows if r.get("entry_price") and r.get("exit_price")
              and float(r["entry_price"]) > 0 and float(r["exit_price"]) > 0]
    print(f"Usable (entry+exit prices present): {len(usable)}")
    print()

    # Aggregate per strategy
    by_strat = defaultdict(list)  # strat -> list of (old_pnl, new_pnl, position_usd)
    for r in usable:
        strat = r["strategy"]
        pos = float(r.get("position_usd") or 0)
        entry = float(r["entry_price"])
        exit_p = float(r["exit_price"])
        old_pnl = float(r.get("pnl_pct") or 0)  # what's in DB (old calib)
        new_pnl = reprice(entry, exit_p, pos, NEW_GAS_USD, NEW_BUY_BPS, NEW_SELL_BPS)
        # ALSO compute what the OLD calib would have given (for fair comparison —
        # because some old DB rows might have been saved with intermediate slip values,
        # or 0 slip if the row predates the EVM fee model).
        old_recomputed = reprice(entry, exit_p, pos, OLD_GAS_USD, OLD_BUY_BPS, OLD_SELL_BPS)
        by_strat[strat].append({
            "old_db": old_pnl,
            "old_recomputed": old_recomputed,
            "new": new_pnl,
            "pos": pos,
        })

    rows_out = []
    for strat, data in by_strat.items():
        n = len(data)
        if n < args.min_n:
            continue
        avg_old_db = mean(d["old_db"] for d in data) * 100
        avg_old_rec = mean(d["old_recomputed"] for d in data) * 100
        avg_new = mean(d["new"] for d in data) * 100
        delta_recal = avg_new - avg_old_rec
        wr_new = sum(1 for d in data if d["new"] > 0) / n * 100
        wr_old = sum(1 for d in data if d["old_db"] > 0) / n * 100
        med_pos = median(d["pos"] for d in data) if data else 0
        # Aggregated $/trade-equivalent (avg pnl × median pos)
        usd_per_trade_new = (avg_new / 100) * med_pos
        rows_out.append({
            "strategy": strat,
            "n": n,
            "avg_pnl_old_db": avg_old_db,
            "avg_pnl_old_rec": avg_old_rec,
            "avg_pnl_new": avg_new,
            "delta_recal_pp": delta_recal,
            "wr_new": wr_new,
            "wr_old": wr_old,
            "med_pos": med_pos,
            "usd_per_trade_new": usd_per_trade_new,
        })

    rows_out.sort(key=lambda r: r["avg_pnl_new"], reverse=True)

    print(f"{'Strategy':<28} {'N':>4} | {'old_db%':>8} {'old_rec%':>9} {'new%':>7} {'d_pp':>6} | "
          f"{'WR_old':>6} {'WR_new':>6} | {'pos$':>5} {'$/tr':>6}")
    print("-" * 110)
    for r in rows_out[:30]:
        print(f"{r['strategy']:<28} {r['n']:>4} | "
              f"{r['avg_pnl_old_db']:>+7.1f}% {r['avg_pnl_old_rec']:>+8.1f}% "
              f"{r['avg_pnl_new']:>+6.1f}% {r['delta_recal_pp']:>+5.1f}pp | "
              f"{r['wr_old']:>5.0f}% {r['wr_new']:>5.0f}% | "
              f"${r['med_pos']:>3.0f} ${r['usd_per_trade_new']:>+4.2f}")

    print()
    print("Legend:")
    print("  old_db%   = avg pnl_pct stored in DB (mix of old calib + 0-slip legacy)")
    print("  old_rec%  = avg pnl_pct recomputed under OLD calib (gas $7.50, slip 200bps)")
    print("  new%      = avg pnl_pct under NEW v14e.28 calib (gas $1.50, slip 100bps)")
    print("  d_pp      = recal effect on avg pnl (positive = strategy improved)")
    print("  pos$      = median position size (gas amortization depends on this)")
    print("  $/tr      = expected $/trade at median position under new calib")
    print()
    # Best/worst recal effects
    sorted_by_delta = sorted(rows_out, key=lambda r: r["delta_recal_pp"], reverse=True)
    if sorted_by_delta:
        print("Top 3 strategies most HELPED by recalibration (smaller positions benefit most):")
        for r in sorted_by_delta[:3]:
            print(f"  {r['strategy']:<28} N={r['n']:>3} pos=${r['med_pos']:>3.0f} "
                  f"old={r['avg_pnl_old_rec']:>+6.1f}% -> new={r['avg_pnl_new']:>+6.1f}% "
                  f"({r['delta_recal_pp']:>+5.1f}pp)")


if __name__ == "__main__":
    main()
