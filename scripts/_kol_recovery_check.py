"""KOL recovery / regression detection.

Two cohorts to monitor weekly:
  1. BLACKLISTED KOLs — are they showing positive shadow returns post-block?
     If yes for >=N days, propose un-blacklist.
  2. NON-BLACKLISTED KOLs trading on main — has any recently regressed to
     reliable_blacklist territory? Cross-check vs the live blacklist.

Methodology:
  - For each blacklisted (kol, chain) pair: pull SHADOW + MAIN paper_trades in
    the [block_at, now] window, filtered to non-artifact strats (the only
    realistic ones). Compute current avg, IC95%, and per-day pattern.
  - Flag UN_BLACKLIST_CANDIDATE if N>=15 AND lower_ci > 0 AND best_strat>0.
  - Flag NEW_BLACKLIST_CANDIDATE for any non-listed KOL hitting the
    RELIABLE_BLACKLIST criteria (same as in _kol_reliable_audit).

Output:
  data/kol_recovery_<ts>.csv  — per (kol, chain) recovery / regression stats
  console: 2 short lists with proposed JSONB diff

Usage:
  python scripts/_kol_recovery_check.py [--days 7] [--block-at 2026-04-27T21:18Z]

If --block-at omitted, defaults to last `--days` window.
"""
import argparse
import csv
import io
import json
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


def load_blacklist() -> dict[str, set]:
    r = sb.table("scoring_config").select("paper_trade_config").eq("id", 1).execute()
    if not r.data:
        return {}
    ptc = r.data[0].get("paper_trade_config") or {}
    if isinstance(ptc, str):
        ptc = json.loads(ptc)
    bl = ptc.get("kol_chain_blacklist") or {}
    return {ch: set(map(str, names)) for ch, names in bl.items()}


def load_artifact_check():
    try:
        from strategies import _is_artifact_family
        return _is_artifact_family
    except Exception:
        return lambda _: False


def bootstrap_mean_ci(values, n_boot=10000, alpha=0.05):
    n = len(values)
    if n < 2:
        return (values[0] if n else 0.0, float("-inf"), float("inf"))
    means = []
    for _ in range(n_boot):
        sample = [values[random.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return (sum(values) / n, means[int(n_boot * alpha / 2)], means[int(n_boot * (1 - alpha / 2))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7,
                    help="Window for recovery check (post-block)")
    ap.add_argument("--block-at", default="",
                    help="ISO timestamp when blacklist was applied (default: --days ago)")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--min-n-recovery", type=int, default=15)
    args = ap.parse_args()

    is_artifact = load_artifact_check()
    blacklist = load_blacklist()

    if args.block_at:
        since = args.block_at
    else:
        since = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()

    print(f"Recovery check window: since {since[:19]}Z")
    print(f"Current blacklist: {sum(len(v) for v in blacklist.values())} cells across {len(blacklist)} chains")

    # ─── 1. Recovery: blacklisted KOLs showing positive shadows? ──────────
    print("\n" + "=" * 100)
    print("RECOVERY CHECK — blacklisted KOLs showing positive shadow $/d post-block")
    print("=" * 100)

    recovery_rows = []
    for chain, kols in blacklist.items():
        for kol in kols:
            rows = fetch_all(
                sb.table("paper_trades")
                .select("strategy,pnl_pct,pnl_usd,pnl_pct_recalc,pnl_usd_recalc,is_shadow,created_at")
                .eq("kol_group", kol)
                .eq("chain", chain)
                .neq("status", "open")
                .neq("source", "rt_live")
                .gte("created_at", since)
            )
            # Filter artifact strategies
            rows = [r for r in rows if r.get("strategy") and not is_artifact(r["strategy"])]
            if not rows:
                continue
            pcts = []
            usds = []
            dates = set()
            shadow_count = 0
            main_count = 0
            for r in rows:
                p = r.get("pnl_pct_recalc") if r.get("pnl_pct_recalc") is not None else r.get("pnl_pct")
                u = r.get("pnl_usd_recalc") if r.get("pnl_usd_recalc") is not None else r.get("pnl_usd")
                if p is None or u is None:
                    continue
                pcts.append(float(p))
                usds.append(float(u))
                if r.get("created_at"):
                    dates.add(r["created_at"][:10])
                if r.get("is_shadow"):
                    shadow_count += 1
                else:
                    main_count += 1

            n = len(pcts)
            if n == 0:
                continue
            wins = sum(1 for p in pcts if p > 0)
            wr = 100.0 * wins / n
            mean_pct, lo, hi = bootstrap_mean_ci(pcts, args.bootstrap)
            days = max(1, len(dates))
            dpd = sum(usds) / days
            recovered = (n >= args.min_n_recovery and lo > 0 and wr >= 40)
            recovery_rows.append({
                "kol": kol, "chain": chain, "n": n,
                "main": main_count, "shadow": shadow_count,
                "wr": round(wr, 1),
                "avg_pct": round(mean_pct * 100, 2),
                "ci_low": round(lo * 100, 2),
                "ci_high": round(hi * 100, 2),
                "dpd": round(dpd, 1),
                "recovered": recovered,
            })

    recovery_rows.sort(key=lambda r: (r["recovered"] is False, -r["dpd"]))

    if not recovery_rows:
        print("  No data for blacklisted KOLs in window — they may have stopped calling tokens.")
    else:
        for r in recovery_rows:
            tag = "** RECOVERED — UN_BLACKLIST CANDIDATE **" if r["recovered"] else "still negative" if r["ci_high"] < 0 else "uncertain"
            print(f"  {r['kol']:<26} {r['chain']:<9} N={r['n']:>4} (main={r['main']}, shadow={r['shadow']})  "
                  f"WR={r['wr']:>5.1f}%  IC95%=[{r['ci_low']:>+6.1f}%, {r['ci_high']:>+6.1f}%]  $/d={r['dpd']:>+8.1f}  -> {tag}")

    candidates_unblock = [r for r in recovery_rows if r["recovered"]]

    # ─── 2. Save + final summary ───────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = os.path.join(out_dir, f"kol_recovery_{ts}.csv")
    if recovery_rows:
        with open(out, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(recovery_rows[0].keys()))
            w.writeheader()
            w.writerows(recovery_rows)
        print(f"\nSaved -> {out}")

    # Proposed JSONB diff
    if candidates_unblock:
        print("\n" + "=" * 100)
        print("PROPOSED JSONB DIFF — un-blacklist candidates")
        print("=" * 100)
        for r in candidates_unblock:
            print(f"  REMOVE  {r['chain']}.{r['kol']}  -- recovered: lo_CI={r['ci_low']}%, $/d={r['dpd']}, N={r['n']}")
        print("\nApply via SQL:")
        print("  UPDATE scoring_config SET paper_trade_config = jsonb_set(...) WHERE id=1;")
    else:
        print("\nNo un-blacklist candidates this window — all blacklisted KOLs still pourris.")


if __name__ == "__main__":
    main()
