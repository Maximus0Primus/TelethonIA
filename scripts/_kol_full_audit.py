"""Full KOL × strategy × chain audit (slip-honest).

For every KOL × chain pair, computes:
  - aggregate: N, WR, avg%, $/d, total $
  - per-strategy breakdown across the active main strategies
  - chain-aware verdict (allow / observe / blacklist)
  - best strategy per (kol, chain) when sample is sufficient

Uses pnl_usd_recalc / pnl_pct_recalc when present (post-v14e.34 honest slip),
falls back to pnl_usd / pnl_pct otherwise.

Output:
  data/kol_audit_<ts>.csv          — flat per-(kol, chain, strategy) rows
  data/kol_audit_summary_<ts>.csv  — one row per (kol, chain) with verdict
  console: blacklist / whitelist / chain-split / observe lists

Usage:
  python scripts/_kol_full_audit.py [--days 14] [--min-n 5]
"""
import argparse
import csv
import io
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from statistics import median

from dotenv import load_dotenv
from supabase import create_client

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

# Active main strategies on each chain, derived from recent paper data:
# (we audit only against strategies that ACTUALLY ran on each chain).
SOL_MAIN_STRATS = {
    "TP50_SL15", "FAST_TP50_SL30", "FAST_TP40_SL30", "FAST_TP100_SL20",
    "FAST_TP50_SL30_HYST", "FAST_TP80_SL25", "BE25_TP80_SL30",
    "BE25_TP80_SL30_S30_HYST", "FAST45_TP40_SL30_S30",
    "BE25_LOCK10_TP100_SL30_S40", "BE25_LOCK10_TP100_SL30_NZ_S40",
    "BE25_TP80_SL30_NZS30_HYST", "AGE24_FAST_TP50_SL30",
    "AGE48_FAST_TP50_SL30", "AGE72_FAST_TP50_SL30",
    "BE15_TP70_SL50_NZ", "HIGHSCORE_TP200_SL40", "NOZEROLIQ_TP200_SL40",
}
ETH_MAIN_STRATS = {
    "ETH_TP300_SL50_4H", "ETH_BE50_TP150_SL40_T2H", "ETH_FAST_TP100_SL20",
    "ETH_TP80_SL40_T2H", "ETH_BE50_LOCK20_TP150_SL40", "ETH_SLOW4H_TP100_SL50",
    "ETH_BE25_LOCK10_TP100_SL30", "ETH_FAST_TP100_SL50", "ETH_FAST_TP500_SL40_60M",
    "ETH_FAST60_TP100_SL50", "ETH_FAST_TP40_SL30", "ETH_BE50_LOCK25_TP200_SL40_4H",
    "ETH_BE20_TP100_SL50", "ETH_BE20_TP80_SL40_T2H", "ETH_BE30_TP100_SL40",
    "ETH_BE50_TP150_SL50",
}


def fetch_all(query):
    out, step, off = [], 1000, 0
    while True:
        r = query.range(off, off + step - 1).execute()
        if not r.data:
            break
        out.extend(r.data)
        if len(r.data) < step:
            break
        off += step
    return out


def fetch_trades(since_iso: str):
    rows = fetch_all(
        sb.table("paper_trades")
        .select("kol_group,chain,strategy,token_address,created_at,pnl_pct,pnl_usd,pnl_pct_recalc,pnl_usd_recalc")
        .neq("status", "open")
        .neq("source", "rt_live")
        .eq("is_shadow", False)
        .gte("created_at", since_iso)
    )
    return rows


def best_strat_per_kol(strat_breakdown: dict, min_n: int) -> tuple[str, float] | None:
    """Pick the strategy with best total $ per (kol, chain) requiring min_n trades."""
    eligible = [
        (s, agg) for s, agg in strat_breakdown.items()
        if agg["n"] >= min_n
    ]
    if not eligible:
        return None
    s, agg = max(eligible, key=lambda kv: kv[1]["total"])
    return s, agg["total"]


def verdict_for(agg: dict) -> str:
    """Per-(kol, chain) classification.

    Rules (sample-size adaptive):
      - N<5  -> 'unknown'  (insufficient)
      - N>=5 and total<=-100 and WR<25%  -> 'BLACKLIST'  (chronic loss)
      - N>=10 and total>=+200 and WR>=40% -> 'WHITELIST' (consistent gain)
      - N>=5 and abs(dpd)<10                 -> 'observe' (neutral)
      - else -> 'observe'
    """
    n = agg["n"]
    total = agg["total"]
    wr = agg["wr"]
    dpd = agg["dpd"]
    if n < 5:
        return "unknown"
    if total <= -100 and wr < 25:
        return "BLACKLIST"
    if total >= 200 and wr >= 40 and n >= 10:
        return "WHITELIST"
    if n >= 10 and total <= -50 and wr < 30:
        return "blacklist_lite"
    if abs(dpd) < 10:
        return "observe"
    return "observe"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14, help="Lookback window")
    ap.add_argument("--min-n", type=int, default=5, help="Min trades for verdict")
    ap.add_argument("--strat-min-n", type=int, default=3, help="Min trades for best-strat selection")
    ap.add_argument("--out-dir", default="", help="Output dir (default: data/)")
    args = ap.parse_args()

    since = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()
    print(f"Window: last {args.days}d (since {since[:19]}Z)")
    rows = fetch_trades(since)
    print(f"Fetched {len(rows):,} main paper trades (non-rt_live, closed)")

    # Build per-(kol, chain, strategy) buckets and per-(kol, chain) totals
    cell = defaultdict(lambda: {"n": 0, "wins": 0, "pcts": [], "usds": [], "dates": set(), "tokens": set()})
    rollup = defaultdict(lambda: {"n": 0, "wins": 0, "pcts": [], "usds": [], "dates": set(), "tokens": set(), "by_strat": {}})

    for r in rows:
        kol = r.get("kol_group")
        chain = r.get("chain") or "solana"
        strat = r.get("strategy")
        if not kol or not strat:
            continue
        pnl_pct = r.get("pnl_pct_recalc") if r.get("pnl_pct_recalc") is not None else r.get("pnl_pct")
        pnl_usd = r.get("pnl_usd_recalc") if r.get("pnl_usd_recalc") is not None else r.get("pnl_usd")
        if pnl_pct is None or pnl_usd is None:
            continue
        d = (r.get("created_at") or "")[:10]
        tok = r.get("token_address")

        for bucket in (cell[(kol, chain, strat)], rollup[(kol, chain)]):
            bucket["n"] += 1
            if pnl_pct > 0:
                bucket["wins"] += 1
            bucket["pcts"].append(float(pnl_pct))
            bucket["usds"].append(float(pnl_usd))
            if d:
                bucket["dates"].add(d)
            if tok:
                bucket["tokens"].add(tok)

    # Aggregate stats
    def stats(b):
        if not b["n"]:
            return None
        days = max(1, len(b["dates"]))
        return {
            "n": b["n"],
            "tokens": len(b["tokens"]),
            "active_days": days,
            "avg_pct": sum(b["pcts"]) / len(b["pcts"]) * 100,
            "median_pct": median(b["pcts"]) * 100,
            "total": sum(b["usds"]),
            "dpd": sum(b["usds"]) / days,
            "wr": 100.0 * b["wins"] / b["n"],
        }

    # Roll up best strat per (kol, chain)
    summary_rows = []
    cell_rows = []
    for (kol, chain), b in rollup.items():
        s = stats(b)
        if s is None:
            continue
        # Per-strategy breakdown
        strat_breakdown = {}
        for (k2, c2, st), b2 in cell.items():
            if k2 == kol and c2 == chain:
                ss = stats(b2)
                if ss is not None:
                    strat_breakdown[st] = ss
                    cell_rows.append({
                        "kol_group": kol, "chain": chain, "strategy": st,
                        **{f"{k}": round(v, 2) if isinstance(v, float) else v for k, v in ss.items()},
                    })
        best = best_strat_per_kol(strat_breakdown, args.strat_min_n)
        verdict = verdict_for(s)
        summary_rows.append({
            "kol_group": kol,
            "chain": chain,
            "n": s["n"],
            "tokens": s["tokens"],
            "active_days": s["active_days"],
            "wr": round(s["wr"], 1),
            "avg_pct": round(s["avg_pct"], 2),
            "median_pct": round(s["median_pct"], 2),
            "total_usd": round(s["total"], 2),
            "dpd_usd": round(s["dpd"], 2),
            "verdict": verdict,
            "best_strat": best[0] if best else None,
            "best_strat_total": round(best[1], 2) if best else None,
        })

    # Sort summary by verdict severity then total $
    verdict_order = {"BLACKLIST": 0, "blacklist_lite": 1, "observe": 2, "unknown": 3, "WHITELIST": 4}
    summary_rows.sort(key=lambda r: (verdict_order.get(r["verdict"], 99), r["total_usd"]))

    out_dir = args.out_dir or os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_csv = os.path.join(out_dir, f"kol_audit_summary_{ts}.csv")
    cell_csv = os.path.join(out_dir, f"kol_audit_{ts}.csv")
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    with open(cell_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cell_rows[0].keys()))
        w.writeheader()
        w.writerows(cell_rows)
    print(f"\nSaved -> {summary_csv}")
    print(f"Saved -> {cell_csv}")

    # Console summary
    by_verdict = defaultdict(list)
    for r in summary_rows:
        by_verdict[r["verdict"]].append(r)

    def print_section(title, rows, fmt="full"):
        print("\n" + "=" * 110)
        print(title)
        print("=" * 110)
        if not rows:
            print("  (none)")
            return
        for r in rows:
            if fmt == "full":
                print(f"  {r['kol_group']:<26} {r['chain']:<9} N={r['n']:>4} tok={r['tokens']:>3} "
                      f"WR={r['wr']:>5.1f}% avg={r['avg_pct']:>+6.1f}% med={r['median_pct']:>+6.1f}% "
                      f"$/d={r['dpd_usd']:>+8.1f} tot={r['total_usd']:>+9.1f}  best={r['best_strat'] or '-'}")

    print_section("BLACKLIST candidates (chronic loss, N>=5)", by_verdict["BLACKLIST"])
    print_section("BLACKLIST_LITE (less severe but still negative)", by_verdict["blacklist_lite"])
    print_section("WHITELIST (consistent winners, N>=10, total>=+$200, WR>=40%)", by_verdict["WHITELIST"])
    print_section("OBSERVE (neutral or insufficient confidence)", [r for r in by_verdict["observe"] if r["n"] >= 10])

    # Chain-split detection (KOL profitable on one chain, loss on other)
    by_kol = defaultdict(dict)
    for r in summary_rows:
        by_kol[r["kol_group"]][r["chain"]] = r
    chain_split = []
    for kol, chains in by_kol.items():
        if len(chains) < 2:
            continue
        sol = chains.get("solana", {})
        eth = chains.get("ethereum", {})
        if not sol or not eth:
            continue
        if (sol.get("verdict") == "BLACKLIST" and eth.get("verdict") in ("WHITELIST", "observe")) \
           or (eth.get("verdict") == "BLACKLIST" and sol.get("verdict") in ("WHITELIST", "observe")):
            chain_split.append((kol, sol, eth))

    print("\n" + "=" * 110)
    print("CHAIN-SPLIT (allow on one chain, blacklist on other)")
    print("=" * 110)
    for kol, sol, eth in chain_split:
        print(f"  {kol:<26} SOL: {sol['verdict']:<14} N={sol['n']:>3} tot={sol['total_usd']:>+8.0f} | "
              f"ETH: {eth['verdict']:<14} N={eth['n']:>3} tot={eth['total_usd']:>+8.0f}")

    # Aggregate impact
    print("\n" + "=" * 70)
    print("AGGREGATE IMPACT")
    print("=" * 70)
    bl = sum(r["total_usd"] for r in by_verdict["BLACKLIST"])
    bll = sum(r["total_usd"] for r in by_verdict["blacklist_lite"])
    wl = sum(r["total_usd"] for r in by_verdict["WHITELIST"])
    print(f"  BLACKLIST (recover by removing):           ${-bl:>+9.0f} over {args.days}d")
    print(f"  BLACKLIST_LITE (recover by removing):      ${-bll:>+9.0f} over {args.days}d")
    print(f"  WHITELIST (current contribution):          ${wl:>+9.0f} over {args.days}d")
    print(f"  Net delta if blacklist applied:            ${-(bl+bll)+wl:>+9.0f} (was ${bl+bll+wl:+,.0f})")


if __name__ == "__main__":
    main()
