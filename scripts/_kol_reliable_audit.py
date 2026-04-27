"""KOL audit with statistical rigor.

For each (kol, chain) over the lookback window:
  1. Aggregate stats (N, WR, avg_pct, total$, $/d)
  2. Bootstrap 95% CI on mean pnl_pct (10K resamples) — fat-tail safe
  3. One-sided sign test p-value (H0: median≥0)
  4. Per-strategy breakdown: how many strategies were positive vs negative
  5. Best strategy on that (kol, chain) and its $ result

VERDICT TIERS:
  RELIABLE_BLACKLIST  — Upper 95% CI of mean < 0 AND best_strat_total <= 0
                        AND N >= 30. Statistically pourris on EVERY strategy
                        they touch with sufficient confidence.

  PROBABLE_BLACKLIST  — same as above but N in [15, 29]. Suggestive but
                        not yet rock-solid. Watch.

  RELIABLE_WINNER     — Lower 95% CI of mean > 0 AND N >= 30 AND best_strat
                        total > 0 AND WR >= 40%.

  INCONCLUSIVE        — N < 15, OR CI bracket includes 0. Keep allowed.

  CHAIN_SPLIT         — kol has reliable verdict on one chain that differs
                        from the other chain.

Outputs:
  data/kol_reliable_<ts>.csv         — flat (kol, chain) summary
  data/kol_reliable_strats_<ts>.csv  — per (kol, chain, strategy) breakdown
  console: each tier listed with confidence intervals shown

Usage:
  python scripts/_kol_reliable_audit.py [--days 14] [--bootstrap 10000]
"""
import argparse
import csv
import io
import os
import random
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

random.seed(42)


def _load_artifact_strats() -> set[str]:
    """v14e.38: load _AUTO_DEPRECATED + _is_artifact_family check from strategies.py
    so the audit can drop trail/dip/split rows whose pnl is sim-only fiction."""
    try:
        from strategies import _AUTO_DEPRECATED, _is_artifact_family  # noqa
        return set(_AUTO_DEPRECATED), _is_artifact_family
    except Exception:
        return set(), lambda _name: False


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
    return fetch_all(
        sb.table("paper_trades")
        .select("kol_group,chain,strategy,token_address,created_at,pnl_pct,pnl_usd,pnl_pct_recalc,pnl_usd_recalc")
        .neq("status", "open")
        .neq("source", "rt_live")
        .eq("is_shadow", False)
        .gte("created_at", since_iso)
    )


def bootstrap_mean_ci(values: list[float], n_boot: int = 10000, alpha: float = 0.05) -> tuple[float, float, float]:
    """Returns (mean, lower_ci, upper_ci) at (1-alpha) confidence."""
    n = len(values)
    if n < 2:
        return (values[0] if n else 0.0, float("-inf"), float("inf"))
    means = []
    for _ in range(n_boot):
        sample = [values[random.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(n_boot * alpha / 2)]
    hi = means[int(n_boot * (1 - alpha / 2))]
    return (sum(values) / n, lo, hi)


def sign_test_pvalue(values: list[float]) -> float:
    """One-sided sign test, H0: median >= 0. Returns p of seeing this many or fewer
    positives if true median were 0. Lower p => more reliably negative."""
    n = len(values)
    if n == 0:
        return 1.0
    pos = sum(1 for v in values if v > 0)
    # Two-sided binomial (cdf at min(pos, n-pos)) under p=0.5
    # Using normal approximation: z = (pos - n/2) / sqrt(n/4)
    import math
    if n < 5:
        return 1.0
    z = (pos - n / 2) / math.sqrt(n / 4)
    # one-sided lower (H1: pos < n/2)
    p = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return p  # small p when pos << n/2 → reliably negative


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--reliable-min-n", type=int, default=30)
    ap.add_argument("--probable-min-n", type=int, default=15)
    ap.add_argument("--strat-min-n", type=int, default=3)
    ap.add_argument("--exclude-artifact-strats", action="store_true",
                    help="Drop rows whose strategy is in _AUTO_DEPRECATED (DTRAIL/PTRAIL/"
                         "TRAIL/SPLIT_/DIP30_/BOND_/TD2_/MCAP_MID_DTRAIL). Removes shadow "
                         "winnings that cannot translate to live (47x slip + reconciler 65% "
                         "early close). Recommended for any KOL verdict post-v14e.36.")
    ap.add_argument("--since-recalc", action="store_true",
                    help="Only count trades after Apr 25 23:23 UTC (v14e.24 commit). Avoids "
                         "the pre-recalc slip regime where pnl_pct was computed with BUY=10. "
                         "Result is more conservative (fewer trades) but slip-uniform.")
    args = ap.parse_args()

    since = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()
    if args.since_recalc:
        recalc_cutoff = "2026-04-25T23:23:00+00:00"
        if recalc_cutoff > since:
            since = recalc_cutoff
    print(f"Window: {args.days}d (since {since[:19]}Z)  |  bootstrap: {args.bootstrap:,}  |  reliable_min_n: {args.reliable_min_n}")

    artifact_set, is_artifact = _load_artifact_strats()
    if args.exclude_artifact_strats:
        print(f"Filtering out {len(artifact_set)} artifact strategies (DTRAIL/DIP/SPLIT/BOND/TD2)")

    rows = fetch_trades(since)
    if args.exclude_artifact_strats:
        before = len(rows)
        rows = [r for r in rows if r.get("strategy") and not is_artifact(r["strategy"])]
        print(f"Trades fetched: {before:,} -> {len(rows):,} after artifact filter ({before-len(rows):,} dropped)\n")
    else:
        print(f"Trades fetched: {len(rows):,}\n")

    # buckets[(kol, chain, strat)] -> list of pnl tuples
    buckets = defaultdict(list)
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
        tok = r.get("token_address") or ""
        buckets[(kol, chain, strat)].append((float(pnl_pct), float(pnl_usd), d, tok))

    # Per-strategy stats and per (kol, chain) rollup
    strat_rows = []
    rollup = defaultdict(lambda: {"pcts": [], "usds": [], "tokens": set(), "dates": set(), "by_strat": {}})

    for (kol, chain, strat), recs in buckets.items():
        pcts = [x[0] for x in recs]
        usds = [x[1] for x in recs]
        toks = {x[3] for x in recs}
        dates = {x[2] for x in recs if x[2]}
        n = len(pcts)
        wins = sum(1 for p in pcts if p > 0)
        s = {
            "n": n,
            "tokens": len(toks),
            "wr": 100.0 * wins / n,
            "avg_pct": sum(pcts) / n * 100,
            "total_usd": sum(usds),
        }
        strat_rows.append({"kol_group": kol, "chain": chain, "strategy": strat, **{k: round(v, 2) if isinstance(v, float) else v for k, v in s.items()}})
        # rollup
        rb = rollup[(kol, chain)]
        rb["pcts"].extend(pcts)
        rb["usds"].extend(usds)
        rb["tokens"].update(toks)
        rb["dates"].update(dates)
        rb["by_strat"][strat] = s

    # Compute rollup verdict per (kol, chain)
    summary = []
    for (kol, chain), b in rollup.items():
        n = len(b["pcts"])
        if n == 0:
            continue
        wins = sum(1 for p in b["pcts"] if p > 0)
        wr = 100.0 * wins / n
        days = max(1, len(b["dates"]))
        total = sum(b["usds"])
        dpd = total / days

        mean_pct, lo_pct, hi_pct = bootstrap_mean_ci(b["pcts"], args.bootstrap)
        sign_p = sign_test_pvalue(b["pcts"])

        # Per-strategy: count positive vs negative strats
        strat_breakdown = b["by_strat"]
        n_pos_strat = sum(1 for s in strat_breakdown.values() if s["total_usd"] > 0 and s["n"] >= args.strat_min_n)
        n_neg_strat = sum(1 for s in strat_breakdown.values() if s["total_usd"] < 0 and s["n"] >= args.strat_min_n)
        eligible_strats = [s for s in strat_breakdown.values() if s["n"] >= args.strat_min_n]
        # Best strategy: max total$
        best_total = max((s["total_usd"] for s in eligible_strats), default=None)
        best_strat_name = next((name for name, s in strat_breakdown.items() if eligible_strats and s["total_usd"] == best_total and s["n"] >= args.strat_min_n), None)

        # Verdict
        verdict = "INCONCLUSIVE"
        confidence = ""
        if n >= args.reliable_min_n and hi_pct < 0 and (best_total is None or best_total <= 0):
            verdict = "RELIABLE_BLACKLIST"
            confidence = f"hi_CI={hi_pct*100:+.1f}%, best_strat=${(best_total or 0):+.0f}"
        elif n >= args.probable_min_n and hi_pct < 0 and (best_total is None or best_total <= 0):
            verdict = "PROBABLE_BLACKLIST"
            confidence = f"hi_CI={hi_pct*100:+.1f}%, best_strat=${(best_total or 0):+.0f}"
        elif n >= args.reliable_min_n and lo_pct > 0 and wr >= 40 and (best_total or 0) > 0:
            verdict = "RELIABLE_WINNER"
            confidence = f"lo_CI={lo_pct*100:+.1f}%, best_strat=${(best_total or 0):+.0f}"
        elif n >= args.probable_min_n and lo_pct > 0 and wr >= 40 and (best_total or 0) > 0:
            verdict = "PROBABLE_WINNER"
            confidence = f"lo_CI={lo_pct*100:+.1f}%, best_strat=${(best_total or 0):+.0f}"

        summary.append({
            "kol_group": kol,
            "chain": chain,
            "n": n,
            "tokens": len(b["tokens"]),
            "active_days": days,
            "wr": round(wr, 1),
            "avg_pct": round(mean_pct * 100, 2),
            "ci_low_pct": round(lo_pct * 100, 2),
            "ci_high_pct": round(hi_pct * 100, 2),
            "sign_test_p": round(sign_p, 4),
            "total_usd": round(total, 2),
            "dpd_usd": round(dpd, 2),
            "n_strats_pos": n_pos_strat,
            "n_strats_neg": n_neg_strat,
            "best_strat": best_strat_name,
            "best_strat_total": round(best_total, 2) if best_total is not None else None,
            "verdict": verdict,
            "confidence_note": confidence,
        })

    # Sort by verdict tier
    tier_order = {"RELIABLE_BLACKLIST": 0, "PROBABLE_BLACKLIST": 1, "INCONCLUSIVE": 2,
                  "PROBABLE_WINNER": 3, "RELIABLE_WINNER": 4}
    summary.sort(key=lambda r: (tier_order.get(r["verdict"], 99), r["total_usd"]))

    # Save
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    sum_path = os.path.join(out_dir, f"kol_reliable_{ts}.csv")
    strat_path = os.path.join(out_dir, f"kol_reliable_strats_{ts}.csv")
    with open(sum_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    with open(strat_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(strat_rows[0].keys()))
        w.writeheader()
        w.writerows(strat_rows)
    print(f"Saved -> {sum_path}")
    print(f"Saved -> {strat_path}\n")

    # Console — bucketed
    by_tier = defaultdict(list)
    for r in summary:
        by_tier[r["verdict"]].append(r)

    def print_tier(name, rows):
        print("=" * 130)
        print(f"{name}  ({len(rows)} cells)")
        print("=" * 130)
        if not rows:
            print("  (none)\n"); return
        for r in rows:
            print(f"  {r['kol_group']:<26} {r['chain']:<9} N={r['n']:>4}  WR={r['wr']:>5.1f}%  "
                  f"avg={r['avg_pct']:>+6.1f}%  95%CI=[{r['ci_low_pct']:>+6.1f}%, {r['ci_high_pct']:>+6.1f}%]  "
                  f"sign_p={r['sign_test_p']:.3f}  $/d={r['dpd_usd']:>+8.0f}  "
                  f"strats±={r['n_strats_pos']}/{r['n_strats_neg']}  best={r['best_strat'] or '-'} "
                  f"(${r['best_strat_total'] if r['best_strat_total'] is not None else '?'})")
        print()

    print_tier("RELIABLE_BLACKLIST  (N>=30, upper CI<0, NO winning strat)", by_tier["RELIABLE_BLACKLIST"])
    print_tier("PROBABLE_BLACKLIST  (N 15-29, upper CI<0, NO winning strat)", by_tier["PROBABLE_BLACKLIST"])
    print_tier("RELIABLE_WINNER  (N>=30, lower CI>0, WR>=40%, has winning strat)", by_tier["RELIABLE_WINNER"])
    print_tier("PROBABLE_WINNER  (N 15-29)", by_tier["PROBABLE_WINNER"])

    # Inconclusive: only show those with N>=10 and big total magnitudes (interesting borderlines)
    interesting = [r for r in by_tier["INCONCLUSIVE"] if r["n"] >= 10]
    interesting.sort(key=lambda r: r["total_usd"])
    print_tier("INCONCLUSIVE  (N>=10, displayed) — keep allowed, more data needed", interesting)

    # Chain-split
    by_kol = defaultdict(dict)
    for r in summary:
        by_kol[r["kol_group"]][r["chain"]] = r
    chain_splits = []
    for kol, chs in by_kol.items():
        if len(chs) < 2:
            continue
        sol = chs.get("solana", {})
        eth = chs.get("ethereum", {})
        if not sol or not eth:
            continue
        a, b = sol.get("verdict"), eth.get("verdict")
        if a != b and "BLACKLIST" in (a, b):
            chain_splits.append((kol, sol, eth))

    print("=" * 130)
    print(f"CHAIN-SPLIT  ({len(chain_splits)} KOLs differ across chains)")
    print("=" * 130)
    for kol, sol, eth in chain_splits:
        print(f"  {kol:<26} SOL: {sol['verdict']:<22} N={sol['n']:>3} 95%CI=[{sol['ci_low_pct']:>+6.1f}%, {sol['ci_high_pct']:>+6.1f}%] tot={sol['total_usd']:>+8.0f}")
        print(f"  {' '*26} ETH: {eth['verdict']:<22} N={eth['n']:>3} 95%CI=[{eth['ci_low_pct']:>+6.1f}%, {eth['ci_high_pct']:>+6.1f}%] tot={eth['total_usd']:>+8.0f}")
    print()

    # Aggregate impact
    bl_savings = -sum(r["total_usd"] for r in by_tier["RELIABLE_BLACKLIST"])
    bl_prob_savings = -sum(r["total_usd"] for r in by_tier["PROBABLE_BLACKLIST"])
    win_keep = sum(r["total_usd"] for r in by_tier["RELIABLE_WINNER"])
    print("=" * 70)
    print("AGGREGATE IMPACT  (recover by removing each tier)")
    print("=" * 70)
    print(f"  Remove RELIABLE_BLACKLIST only:    +${bl_savings:>9,.0f}  / {args.days}d  =  +${bl_savings/args.days:>6,.0f}/day")
    print(f"  Also remove PROBABLE_BLACKLIST:    +${bl_savings + bl_prob_savings:>9,.0f}  / {args.days}d  =  +${(bl_savings+bl_prob_savings)/args.days:>6,.0f}/day")
    print(f"  RELIABLE_WINNER current contrib:   +${win_keep:>9,.0f}  / {args.days}d  =  +${win_keep/args.days:>6,.0f}/day")


if __name__ == "__main__":
    main()
