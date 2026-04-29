"""Slippage sensitivity analyzer — post-mega-sweep robustness check.

v14e.34 — addresses the core lesson from the JSONB drift bug: strategies whose
$/day flips sign or rank when slip moves from 100 bps -> 600 bps are FRAGILE
and shouldn't be promoted, even if they win at the median 225.

Methodology
-----------
For each top-N candidate strategy (read from --top-csv or --strats):
  1. Pull its closed paper_trades on Solana from the last --days
  2. For each slip level in SLIP_GRID, recompute pnl_pct as:
        new_entry  = dex_spot_price_at_entry * (1 + slip_bps / 10000)
        new_pnl    = (exit_price * (1 - sell_slip / 10000)) / new_entry - 1
     (sell_slip stays as the persisted column = empirical dynamic value)
  3. Aggregate: total $/day, avg %, WR, N
  4. Compute robustness ratio: $/day at WORST slip / $/day at BEST slip
  5. Rank stability: does the strat's rank in the sorted top change?

A strat with $/day @ 100 = +$30 but $/day @ 600 = -$15 is FRAGILE — its win
in the sim assumes optimistic slip that won't hold in production.

Output
------
data/slip_sensitivity_<ts>.csv with columns:
  strategy, n_trades, days_observed, slip_50_dpd, slip_100_dpd, slip_225_dpd,
  slip_400_dpd, slip_600_dpd, slip_800_dpd, robustness_ratio, sign_flip,
  rank_at_100, rank_at_600, rank_drop

A summary is printed: top 10 most ROBUST + top 5 most FRAGILE.

Usage
-----
  # Standalone (auto-pick top strats from recent paper data)
  python scripts/_strat_slip_sensitivity.py --days 14 --top 20

  # Wired to mega-sweep (post-analysis step)
  python scripts/_strat_slip_sensitivity.py \
      --top-csv _mega_sweep_top_robust.csv --top 20 --days 14
"""
import argparse
import csv
import io
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv
from supabase import create_client

# Force UTF-8 stdout (Windows cp1252 default breaks on unicode arrows)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

SLIP_GRID = [50, 100, 225, 400, 600, 800]


def fetch_strat_trades(strategy: str, since_iso: str, chain: str = "solana", limit: int = 5000):
    """Pull closed paper trades for one strategy, only rows usable for recalc."""
    out, step, off = [], 1000, 0
    while off < limit:
        r = (
            sb.table("paper_trades")
            .select("id,strategy,token_address,kol_group,created_at,position_usd,"
                    "dex_spot_price_at_entry,exit_price,sell_slippage_bps,rt_liquidity_usd,"
                    "is_shadow,source,status,chain,pnl_pct,pnl_usd")
            .eq("strategy", strategy)
            .eq("chain", chain)
            .neq("status", "open")
            .neq("source", "rt_live")
            .gte("created_at", since_iso)
            .not_.is_("dex_spot_price_at_entry", "null")
            .not_.is_("exit_price", "null")
            .range(off, off + step - 1)
            .execute()
        )
        if not r.data:
            break
        out.extend(r.data)
        if len(r.data) < step:
            break
        off += step
    return out


def slip_mult_for(liq_usd: float) -> float:
    """Mirrors paper_trader.py:1212-1218 liquidity-aware slip multiplier."""
    if not liq_usd or liq_usd <= 0:
        return 3.0  # lds = 0.1 -> mult = 1 + 2*0.9 = 2.8 (clamp)
    lds = max(0.1, min(1.0, liq_usd / 50_000.0))
    return 1.0 + 2.0 * (1.0 - lds)


def recalc_pnl(trade: dict, base_buy_slip_bps: int) -> float | None:
    """Recompute pnl_pct under a given base buy slip. Returns None if data missing.

    v14e.45 bug fix: previously this function applied sell_slip a SECOND time
    via `(exit_price * (1 - sell_slip/10000))`, but exit_price stored in DB is
    ALREADY net of sell slip (cf. paper_trader.py:1953
    `exit_price = sl_price * _dynamic_sell_slip_factor(...)`). The double
    counting was inflating fragility — strats that flipped sign at slip 600bps
    were partially an artifact of the second sell penalty. Now we vary BUY
    slip only and keep the realized exit_price as-is. This means the script
    measures buy-slip sensitivity, not sell-slip sensitivity. To test sell
    slip independently, would need dex_spot_price_at_exit (not stored).
    """
    spot = trade.get("dex_spot_price_at_entry")
    exitp = trade.get("exit_price")
    if not spot or not exitp or spot <= 0:
        return None
    mult = slip_mult_for(float(trade.get("rt_liquidity_usd") or 0))
    new_buy_bps = base_buy_slip_bps * mult
    new_entry = float(spot) * (1.0 + new_buy_bps / 10_000.0)
    return float(exitp) / new_entry - 1.0


def aggregate_strat(trades: list[dict]) -> dict:
    """Per-slip $/day + WR + N for one strategy. Returns dict keyed by slip_bps."""
    if not trades:
        return {}
    # Date span (days observed)
    dates = sorted({t["created_at"][:10] for t in trades if t.get("created_at")})
    days = max(1, len(dates))
    out = {"n": len(trades), "days": days, "by_slip": {}}
    for slip in SLIP_GRID:
        pnl_pcts = []
        pnl_usds = []
        for t in trades:
            p = recalc_pnl(t, slip)
            if p is None:
                continue
            pnl_pcts.append(p)
            pos = float(t.get("position_usd") or 0)
            pnl_usds.append(p * pos)
        if not pnl_pcts:
            continue
        wins = sum(1 for p in pnl_pcts if p > 0)
        out["by_slip"][slip] = {
            "n": len(pnl_pcts),
            "avg_pct": sum(pnl_pcts) / len(pnl_pcts),
            "wr": wins / len(pnl_pcts),
            "total_usd": sum(pnl_usds),
            "dpd_usd": sum(pnl_usds) / days,
        }
    return out


def discover_top_strats(since_iso: str, n: int) -> list[str]:
    """If no --top-csv provided, pick top-N by raw $/day from paper_trades."""
    r = (
        sb.table("paper_trades")
        .select("strategy,pnl_usd,created_at")
        .eq("chain", "solana")
        .neq("status", "open")
        .neq("source", "rt_live")
        .gte("created_at", since_iso)
        .limit(50000)
        .execute()
    )
    rows = r.data or []
    by_strat = defaultdict(lambda: {"pnl": 0.0, "dates": set()})
    for x in rows:
        s = x["strategy"]
        by_strat[s]["pnl"] += float(x.get("pnl_usd") or 0)
        if x.get("created_at"):
            by_strat[s]["dates"].add(x["created_at"][:10])
    ranked = sorted(
        by_strat.items(),
        key=lambda kv: kv[1]["pnl"] / max(1, len(kv[1]["dates"])),
        reverse=True,
    )
    return [s for s, _ in ranked[:n]]


def load_strats_from_csv(path: str, n: int) -> list[str]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = row.get("strategy") or row.get("Strategy") or row.get("strat")
            if s and s not in out:
                out.append(s)
            if len(out) >= n:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14, help="Lookback window for paper_trades")
    ap.add_argument("--top", type=int, default=20, help="How many strats to test")
    ap.add_argument("--top-csv", default="", help="Optional CSV with strategy column (e.g. _mega_sweep_top_robust.csv)")
    ap.add_argument("--out", default="", help="Output CSV path (default: data/slip_sensitivity_<ts>.csv)")
    ap.add_argument("--persist-run-id", default="",
                    help="If set, UPDATE mega_sweep_runs.robustness_ratio for this run_id × strategy. "
                         "Closes the v14e.45 latent gap where the column was always NULL.")
    ap.add_argument("--chain", default="solana",
                    help="Chain to test (default solana). ETH workflow passes --chain ethereum.")
    args = ap.parse_args()

    since = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()
    print(f"Window: last {args.days}d (since {since})")

    if args.top_csv and os.path.exists(args.top_csv):
        strats = load_strats_from_csv(args.top_csv, args.top)
        print(f"Loaded {len(strats)} strats from {args.top_csv}")
    else:
        strats = discover_top_strats(since, args.top)
        print(f"Auto-discovered {len(strats)} top strats by $/day")

    if not strats:
        print("No strategies to test, exit.")
        return

    rows = []
    for i, s in enumerate(strats, 1):
        trades = fetch_strat_trades(s, since, chain=args.chain)
        agg = aggregate_strat(trades)
        if not agg.get("by_slip"):
            print(f"[{i}/{len(strats)}] {s}: no usable trades, skip")
            continue
        by = agg["by_slip"]
        dpd_at = {sl: by.get(sl, {}).get("dpd_usd", 0.0) for sl in SLIP_GRID}
        # Robustness: ratio worst/best, accounting for sign
        best = max(dpd_at.values())
        worst = min(dpd_at.values())
        if best <= 0:
            robust = 0.0  # all losing — degenerate
        elif worst < 0:
            robust = -1.0  # sign flip
        else:
            robust = worst / best
        sign_flip = (worst < 0 and best > 0)
        row = {
            "strategy": s,
            "n_trades": agg["n"],
            "days_observed": agg["days"],
            **{f"slip_{sl}_dpd": round(dpd_at[sl], 2) for sl in SLIP_GRID},
            **{f"slip_{sl}_wr": round(by.get(sl, {}).get("wr", 0) * 100, 1) for sl in SLIP_GRID},
            "robustness_ratio": round(robust, 3),
            "sign_flip": int(sign_flip),
        }
        rows.append(row)
        print(f"[{i}/{len(strats)}] {s}: dpd@100={dpd_at[100]:+.1f} dpd@600={dpd_at[600]:+.1f} robust={row['robustness_ratio']} flip={sign_flip}")

    if not rows:
        print("No results, exit.")
        return

    # Rank stability: who's #1 at slip=100 vs slip=600 ?
    ranked_100 = sorted(rows, key=lambda r: r["slip_100_dpd"], reverse=True)
    ranked_600 = sorted(rows, key=lambda r: r["slip_600_dpd"], reverse=True)
    rank_100 = {r["strategy"]: i + 1 for i, r in enumerate(ranked_100)}
    rank_600 = {r["strategy"]: i + 1 for i, r in enumerate(ranked_600)}
    for r in rows:
        r["rank_at_100"] = rank_100[r["strategy"]]
        r["rank_at_600"] = rank_600[r["strategy"]]
        r["rank_drop"] = r["rank_at_600"] - r["rank_at_100"]

    out = args.out or os.path.join(
        os.path.dirname(__file__), "..", "data",
        f"slip_sensitivity_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved -> {out}")

    # v14e.45: round-trip robustness_ratio back into mega_sweep_runs.
    # Latent gap depuis création table — la colonne était toujours NULL malgré
    # is_top_robust=true. Fix: UPDATE par (run_id, strategy) après le CSV.
    if args.persist_run_id:
        try:
            url = os.environ.get("SUPABASE_URL")
            key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
            if not (url and key):
                print(f"[persist] SUPABASE_URL/KEY missing — skipping DB UPDATE")
            else:
                from supabase import create_client
                sb = create_client(url, key)
                n_updated = 0
                for r in rows:
                    res = sb.table("mega_sweep_runs").update({
                        "robustness_ratio": r["robustness_ratio"],
                    }).eq("run_id", args.persist_run_id) \
                      .eq("strategy", r["strategy"]).execute()
                    if res.data:
                        n_updated += len(res.data)
                print(f"[persist] UPDATEd {n_updated} mega_sweep_runs rows "
                      f"(run_id={args.persist_run_id}) with robustness_ratio")
        except Exception as e:
            print(f"[persist] DB UPDATE FAILED: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("MOST ROBUST (positive $/day at every slip level)")
    print("=" * 80)
    robust_only = [r for r in rows if not r["sign_flip"] and r["robustness_ratio"] > 0]
    robust_only.sort(key=lambda r: r["robustness_ratio"], reverse=True)
    for r in robust_only[:10]:
        print(f"  {r['strategy']:<32}  robust={r['robustness_ratio']:.2f}  "
              f"dpd@225={r['slip_225_dpd']:+.1f}  dpd@600={r['slip_600_dpd']:+.1f}  "
              f"rank100->600: {r['rank_at_100']}->{r['rank_at_600']}")

    print("\n" + "=" * 80)
    print("MOST FRAGILE (sign flips: positive at low slip, negative at high)")
    print("=" * 80)
    fragile = [r for r in rows if r["sign_flip"]]
    fragile.sort(key=lambda r: r["slip_100_dpd"] - r["slip_600_dpd"], reverse=True)
    for r in fragile[:5]:
        print(f"  {r['strategy']:<32}  dpd@100={r['slip_100_dpd']:+.1f}  "
              f"dpd@600={r['slip_600_dpd']:+.1f}  delta={r['slip_100_dpd'] - r['slip_600_dpd']:+.1f}")
        print(f"      ! DO NOT promote — winning under optimistic slip only")


if __name__ == "__main__":
    main()
