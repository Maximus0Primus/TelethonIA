"""Mega-sweep sim vs realized calibration report.

v14e.34 — for each historical mega_sweep_runs row, compares the predicted
$/day against the realized $/day from paper_trades over the post-run window.
Flags persistent over/under-estimation per strategy and per family.

Methodology
-----------
For each row in `mega_sweep_runs`:
  1. predicted_dpd = pnl_per_day at run_at
  2. realized_dpd = SUM(pnl_usd_recalc OR pnl_usd) / days
                    on paper_trades for that strategy
                    where created_at IN [run_at, run_at + window_days]
  3. drift_dpd    = realized - predicted
  4. drift_ratio  = realized / predicted   (only when predicted!=0)

Aggregations:
  - per (strategy, run_id) — single-run drift
  - per strategy across runs — chronic over/under?
  - per family (FAST/BE/HIGHSCORE/SCALP/...) — systematic family bias?

Output
------
data/sim_calibration_<ts>.csv  (per-(strategy, run_at) row)
+ console summary: top 10 chronic over-estimators, top 10 under-estimators

Usage
-----
  python scripts/_mega_sweep_calibration.py [--window-days 7] [--min-runs 1]
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

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])


def fetch_all(table, select, **filters):
    out, step, off = [], 1000, 0
    while True:
        q = sb.table(table).select(select)
        for k, v in filters.items():
            if k.startswith("gte_"):
                q = q.gte(k[4:], v)
            elif k.startswith("lte_"):
                q = q.lte(k[4:], v)
            elif k.startswith("eq_"):
                q = q.eq(k[3:], v)
        r = q.range(off, off + step - 1).execute()
        if not r.data:
            break
        out.extend(r.data)
        if len(r.data) < step:
            break
        off += step
    return out


def family_of(strategy: str) -> str:
    s = strategy.upper()
    for fam in ("FAST", "SCALP", "DECAY", "DTRAIL", "TRAIL", "DIP", "SPLIT",
                "BE25_LOCK", "BE50_LOCK", "BE15", "BE25", "BE30", "BE50",
                "HIGHSCORE", "NOZEROLIQ", "BOND_FAST", "AGE24", "AGE48", "AGE72",
                "SLOW4H", "SLOW6H", "MOONBAG", "WIDE_RUNNER"):
        if s.startswith(fam):
            return fam
    return "OTHER"


def realized_dpd(strategy: str, since_iso: str, until_iso: str) -> tuple[float, int, int]:
    """Sum pnl_usd_recalc when present (post-v14e.34 honest slip), else pnl_usd.
    Returns (dpd, n_trades, days_observed)."""
    rows = fetch_all(
        "paper_trades",
        "pnl_usd,pnl_usd_recalc,created_at",
        eq_strategy=strategy,
        eq_chain="solana",
        gte_created_at=since_iso,
        lte_created_at=until_iso,
    )
    if not rows:
        return 0.0, 0, 0
    total = 0.0
    dates = set()
    for r in rows:
        total += float(r.get("pnl_usd_recalc") if r.get("pnl_usd_recalc") is not None else (r.get("pnl_usd") or 0))
        if r.get("created_at"):
            dates.add(r["created_at"][:10])
    days = max(1, len(dates))
    return total / days, len(rows), days


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-days", type=int, default=7,
                    help="How many days post-run to measure realized P&L")
    ap.add_argument("--min-runs", type=int, default=1,
                    help="Minimum runs per strategy to include in chronic-bias report")
    ap.add_argument("--out", default="",
                    help="Output CSV (default: data/sim_calibration_<ts>.csv)")
    args = ap.parse_args()

    print(f"Window: {args.window_days}d post-run")

    runs = fetch_all(
        "mega_sweep_runs",
        "id,run_at,run_id,strategy,filter_name,pnl_per_day,avg_pnl_pct,n_sim_trades,family_realism,is_top_robust",
    )
    if not runs:
        print("No mega_sweep_runs records found — nothing to calibrate yet.")
        print("Hint: GH workflow must have run with --persist at least once.")
        return

    print(f"Loaded {len(runs)} mega_sweep_runs rows across {len({r['run_id'] for r in runs if r.get('run_id')})} GH runs")

    rows_out = []
    by_strat = defaultdict(list)

    for r in runs:
        run_at = r["run_at"]
        # Don't measure runs whose window is still open
        run_dt = datetime.fromisoformat(run_at.replace("Z", "+00:00"))
        until = run_dt + timedelta(days=args.window_days)
        if until > datetime.now(timezone.utc):
            continue
        until_iso = until.isoformat()

        realized, n_trades, days = realized_dpd(r["strategy"], run_at, until_iso)
        predicted = float(r.get("pnl_per_day") or 0)
        drift_dpd = realized - predicted
        drift_ratio = realized / predicted if predicted != 0 else None
        rec = {
            "run_at": run_at,
            "run_id": r.get("run_id"),
            "strategy": r["strategy"],
            "filter_name": r.get("filter_name"),
            "family": family_of(r["strategy"]),
            "is_top_robust": r.get("is_top_robust"),
            "predicted_dpd": round(predicted, 2),
            "realized_dpd": round(realized, 2),
            "drift_dpd": round(drift_dpd, 2),
            "drift_ratio": round(drift_ratio, 3) if drift_ratio is not None else None,
            "realized_n_trades": n_trades,
            "realized_days": days,
            "family_realism": r.get("family_realism"),
        }
        rows_out.append(rec)
        by_strat[r["strategy"]].append(rec)

    if not rows_out:
        print("All mega_sweep_runs are still within their measurement window — wait a few days.")
        return

    out = args.out or os.path.join(
        os.path.dirname(__file__), "..", "data",
        f"sim_calibration_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nSaved -> {out}")

    # --- Chronic per-strategy bias --------------------------------------
    chronic = []
    for s, recs in by_strat.items():
        finished = [r for r in recs if r["realized_n_trades"] > 0]
        if len(finished) < args.min_runs:
            continue
        avg_pred = sum(r["predicted_dpd"] for r in finished) / len(finished)
        avg_real = sum(r["realized_dpd"] for r in finished) / len(finished)
        chronic.append({
            "strategy": s,
            "family": family_of(s),
            "n_runs": len(finished),
            "avg_predicted_dpd": round(avg_pred, 2),
            "avg_realized_dpd": round(avg_real, 2),
            "drift_dpd": round(avg_real - avg_pred, 2),
        })

    chronic.sort(key=lambda x: x["drift_dpd"])

    print("\n" + "=" * 90)
    print("TOP 10 CHRONIC SIM OVER-ESTIMATORS  (predicted >> realized — sim too optimistic)")
    print("=" * 90)
    for r in chronic[:10]:
        print(f"  {r['strategy']:<32}  fam={r['family']:<14}  runs={r['n_runs']}  "
              f"pred={r['avg_predicted_dpd']:+.1f}  real={r['avg_realized_dpd']:+.1f}  Δ={r['drift_dpd']:+.1f}")

    print("\n" + "=" * 90)
    print("TOP 10 CHRONIC SIM UNDER-ESTIMATORS  (realized >> predicted — sim too pessimistic)")
    print("=" * 90)
    for r in chronic[-10:][::-1]:
        print(f"  {r['strategy']:<32}  fam={r['family']:<14}  runs={r['n_runs']}  "
              f"pred={r['avg_predicted_dpd']:+.1f}  real={r['avg_realized_dpd']:+.1f}  Δ={r['drift_dpd']:+.1f}")

    # --- Per-family bias -------------------------------------------------
    fam_drift = defaultdict(lambda: {"n": 0, "pred": 0.0, "real": 0.0})
    for r in rows_out:
        if r["realized_n_trades"] == 0:
            continue
        f = r["family"]
        fam_drift[f]["n"] += 1
        fam_drift[f]["pred"] += r["predicted_dpd"]
        fam_drift[f]["real"] += r["realized_dpd"]

    print("\n" + "=" * 70)
    print("PER-FAMILY DRIFT  (over runs)")
    print("=" * 70)
    fam_sorted = sorted(fam_drift.items(),
                        key=lambda kv: (kv[1]["real"] - kv[1]["pred"]) / max(1, kv[1]["n"]))
    for f, d in fam_sorted:
        avg_pred = d["pred"] / max(1, d["n"])
        avg_real = d["real"] / max(1, d["n"])
        print(f"  {f:<14}  N={d['n']:>4}  pred_avg={avg_pred:+.2f}  real_avg={avg_real:+.2f}  "
              f"Δ={(avg_real - avg_pred):+.2f}")


if __name__ == "__main__":
    main()
