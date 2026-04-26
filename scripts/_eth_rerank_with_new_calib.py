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
    ap.add_argument("--position", type=float, default=0,
                    help="Override position size to simulate live $/trade. Default 0 = use actual paper position. Try 50 for live small-size.")
    ap.add_argument("--by-kol", action="store_true",
                    help="Also break down by (kol_group, strategy). Identifies KOLs gutting the avg.")
    ap.add_argument("--by-age", action="store_true",
                    help="Also break down by token age band (AGE12/AGE24/AGE48). Tests if 12-48h aged tokens beat 0-12h on ETH.")
    ap.add_argument("--all-filters", action="store_true",
                    help="Combined filter pass: AGE<12h + exclude top 5 worst KOLs. Most realistic live projection.")
    ap.add_argument("--max-age-h", type=float, default=12.0,
                    help="Max token age (hours) for --all-filters (default 12).")
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
                     "pnl_pct, status, created_at, chain, kol_group, "
                     "rt_token_age_hours")
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

    # Aggregate per strategy (and per kol/age if requested)
    by_strat = defaultdict(list)
    by_kol_strat = defaultdict(list)  # (kol, strat) -> list
    by_age_strat = defaultdict(list)  # (age_band, strat) -> list
    for r in usable:
        strat = r["strategy"]
        kol = (r.get("kol_group") or "unknown").lower()
        age_h = float(r.get("rt_token_age_hours") or 0)
        # Disjoint bands matching strategies.py AGE24/AGE48 convention
        if age_h < 12:
            age_band = "AGE0_12"
        elif age_h < 24:
            age_band = "AGE12_24"
        elif age_h < 48:
            age_band = "AGE24_48"
        else:
            age_band = "AGE48_PLUS"
        pos = args.position if args.position > 0 else float(r.get("position_usd") or 0)
        entry = float(r["entry_price"])
        exit_p = float(r["exit_price"])
        old_pnl = float(r.get("pnl_pct") or 0)
        new_pnl = reprice(entry, exit_p, pos, NEW_GAS_USD, NEW_BUY_BPS, NEW_SELL_BPS)
        old_recomputed = reprice(entry, exit_p, pos, OLD_GAS_USD, OLD_BUY_BPS, OLD_SELL_BPS)
        record = {
            "old_db": old_pnl,
            "old_recomputed": old_recomputed,
            "new": new_pnl,
            "pos": pos,
            "kol": kol,
            "age_band": age_band,
            "age_h": age_h,
        }
        by_strat[strat].append(record)
        by_kol_strat[(kol, strat)].append(record)
        by_age_strat[(age_band, strat)].append(record)

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
        print("Top 3 strategies most HELPED by recalibration:")
        for r in sorted_by_delta[:3]:
            print(f"  {r['strategy']:<28} N={r['n']:>3} pos=${r['med_pos']:>3.0f} "
                  f"old={r['avg_pnl_old_rec']:>+6.1f}% -> new={r['avg_pnl_new']:>+6.1f}% "
                  f"({r['delta_recal_pp']:>+5.1f}pp)")

    # Per-KOL breakdown — identifies who's gutting the strats
    if args.by_kol:
        print()
        print("=" * 110)
        print("PER-KOL BREAKDOWN — find the destroyers")
        print("=" * 110)
        # Aggregate by kol only (across all strats)
        by_kol = defaultdict(list)
        for (kol, strat), records in by_kol_strat.items():
            for rec in records:
                by_kol[kol].append(rec)

        kol_rows = []
        for kol, recs in by_kol.items():
            n = len(recs)
            if n < 3:  # min N=3 for KOL view (smaller threshold)
                continue
            avg_new = mean(d["new"] for d in recs) * 100
            avg_old = mean(d["old_recomputed"] for d in recs) * 100
            wr = sum(1 for d in recs if d["new"] > 0) / n * 100
            kol_rows.append({
                "kol": kol, "n": n, "avg_new": avg_new,
                "avg_old": avg_old, "wr": wr,
            })
        kol_rows.sort(key=lambda r: r["avg_new"])

        print()
        print("WORST KOLs on ETH (sorted ascending by new%):")
        print(f"{'KOL':<28} {'N':>4} | {'old_rec%':>9} {'new%':>7} | {'WR':>5}")
        print("-" * 75)
        for r in kol_rows[:10]:
            print(f"{r['kol']:<28} {r['n']:>4} | "
                  f"{r['avg_old']:>+8.1f}% {r['avg_new']:>+6.1f}% | "
                  f"{r['wr']:>4.0f}%")
        print()
        print("BEST KOLs on ETH:")
        print(f"{'KOL':<28} {'N':>4} | {'old_rec%':>9} {'new%':>7} | {'WR':>5}")
        print("-" * 75)
        for r in kol_rows[-10:][::-1]:
            print(f"{r['kol']:<28} {r['n']:>4} | "
                  f"{r['avg_old']:>+8.1f}% {r['avg_new']:>+6.1f}% | "
                  f"{r['wr']:>4.0f}%")

        # Show how excluding worst KOLs changes top strats
        print()
        print("=" * 110)
        worst_kols = {r["kol"] for r in kol_rows[:5]}  # bottom 5 KOLs
        print(f"Top strats EXCLUDING worst 5 KOLs: {sorted(worst_kols)}")
        print("=" * 110)
        clean_strat = defaultdict(list)
        for strat, records in by_strat.items():
            for rec in records:
                if rec["kol"] not in worst_kols:
                    clean_strat[strat].append(rec)
        clean_rows = []
        for strat, recs in clean_strat.items():
            n = len(recs)
            if n < args.min_n:
                continue
            avg_new = mean(d["new"] for d in recs) * 100
            avg_old = mean(d["old_recomputed"] for d in recs) * 100
            wr = sum(1 for d in recs if d["new"] > 0) / n * 100
            med_pos = median(d["pos"] for d in recs)
            usd_per_trade = (avg_new / 100) * med_pos
            clean_rows.append({
                "strategy": strat, "n": n, "avg_new": avg_new,
                "avg_old": avg_old, "wr": wr, "med_pos": med_pos,
                "usd_per_trade": usd_per_trade,
            })
        clean_rows.sort(key=lambda r: r["avg_new"], reverse=True)
        print(f"{'Strategy':<28} {'N':>4} | {'old%':>7} {'new%':>7} | {'WR':>5} | "
              f"{'pos$':>5} {'$/tr':>6}")
        print("-" * 90)
        for r in clean_rows[:15]:
            print(f"{r['strategy']:<28} {r['n']:>4} | "
                  f"{r['avg_old']:>+6.1f}% {r['avg_new']:>+6.1f}% | "
                  f"{r['wr']:>4.0f}% | "
                  f"${r['med_pos']:>3.0f} ${r['usd_per_trade']:>+4.2f}")

    # Per-age-band breakdown (per top-3 strategies)
    if args.by_age:
        print()
        print("=" * 110)
        print("PER-AGE-BAND BREAKDOWN — does the 12-48h universe beat 0-12h on ETH?")
        print("=" * 110)
        # Aggregate all ETH trades by age_band only (across all strats)
        by_age_only = defaultdict(list)
        for (band, _), recs in by_age_strat.items():
            by_age_only[band].extend(recs)
        print()
        print("All ETH strats × age_band:")
        print(f"{'age_band':<14} {'N':>5} | {'new pnl%':>9} | {'WR':>5}")
        print("-" * 50)
        for band in ("AGE0_12", "AGE12_24", "AGE24_48", "AGE48_PLUS"):
            recs = by_age_only.get(band, [])
            n = len(recs)
            if n == 0:
                print(f"{band:<14} {n:>5} | (no trades)")
                continue
            avg_new = mean(d["new"] for d in recs) * 100
            wr = sum(1 for d in recs if d["new"] > 0) / n * 100
            print(f"{band:<14} {n:>5} | {avg_new:>+8.1f}% | {wr:>4.0f}%")

        # Top 3 strats by age band
        print()
        print("Top strategies broken down by age_band (min N=5/cell):")
        # Pick top 5 strats overall by new pnl
        top_strats = [r["strategy"] for r in rows_out[:5]]
        print(f"{'Strategy':<28} {'AGE0_12':>16} {'AGE12_24':>16} {'AGE24_48':>16} {'AGE48+':>16}")
        print("-" * 100)
        for strat in top_strats:
            cells = []
            for band in ("AGE0_12", "AGE12_24", "AGE24_48", "AGE48_PLUS"):
                recs = by_age_strat.get((band, strat), [])
                n = len(recs)
                if n < 3:
                    cells.append(f"  N={n:<3} -")
                else:
                    avg = mean(d["new"] for d in recs) * 100
                    cells.append(f" N={n:<3} {avg:>+5.1f}%")
            print(f"{strat:<28} {cells[0]:>16} {cells[1]:>16} {cells[2]:>16} {cells[3]:>16}")

    # Combined filters — most realistic live projection
    if args.all_filters:
        print()
        print("=" * 110)
        print(f"COMBINED FILTERS: age<{args.max_age_h:.0f}h + exclude worst 5 KOLs")
        print(f"(position={'$' + str(int(args.position)) if args.position else 'paper actual'}, gas+slip = v14e.28 calibration)")
        print("=" * 110)

        # Determine worst 5 KOLs from earlier breakdown
        kol_perf = defaultdict(list)
        for strat, records in by_strat.items():
            for rec in records:
                kol_perf[rec["kol"]].append(rec["new"])
        kol_avg = [(k, mean(v) * 100, len(v)) for k, v in kol_perf.items() if len(v) >= 3]
        kol_avg.sort(key=lambda x: x[1])  # ascending = worst first
        worst5 = {k for k, _, _ in kol_avg[:5]}
        print(f"Worst 5 KOLs excluded: {sorted(worst5)}")
        print()

        clean_strat = defaultdict(list)
        for strat, records in by_strat.items():
            for rec in records:
                if rec["kol"] in worst5:
                    continue
                if rec["age_h"] > args.max_age_h:
                    continue
                clean_strat[strat].append(rec)

        clean_rows = []
        for strat, recs in clean_strat.items():
            n = len(recs)
            if n < args.min_n:
                continue
            avg_new = mean(d["new"] for d in recs) * 100
            wr = sum(1 for d in recs if d["new"] > 0) / n * 100
            med_pos = median(d["pos"] for d in recs)
            usd_per_trade = (avg_new / 100) * med_pos
            clean_rows.append({
                "strategy": strat, "n": n, "avg_new": avg_new,
                "wr": wr, "med_pos": med_pos,
                "usd_per_trade": usd_per_trade,
            })
        clean_rows.sort(key=lambda r: r["avg_new"], reverse=True)
        print(f"{'Strategy':<28} {'N':>4} | {'new pnl%':>9} | {'WR':>5} | "
              f"{'pos$':>5} {'$/tr':>6}")
        print("-" * 78)
        for r in clean_rows[:15]:
            print(f"{r['strategy']:<28} {r['n']:>4} | "
                  f"{r['avg_new']:>+8.1f}% | "
                  f"{r['wr']:>4.0f}% | "
                  f"${r['med_pos']:>3.0f} ${r['usd_per_trade']:>+4.2f}")


if __name__ == "__main__":
    main()
