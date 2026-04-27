"""One-shot backfill of mega_sweep_runs from GH artifact CSVs.

Mirrors the persistence logic in analyze_mega_sweep.py --persist, but reads
historical artifacts already on disk (via `gh run download`) and tags rows
with the ORIGINAL workflow run timestamp so the calibration script can
measure realized P&L over the correct post-run window.

For each artifact directory:
  - Reads `_mega_sweep_extended_annotated.csv`
  - Picks top-N robust (avg_pnl_pct>0, family_realism>=0.5, cross_regime if present)
  - Picks N runners-up by avg_pnl_pct (family_realism>=0.5)
  - INSERTs into mega_sweep_runs with run_source='backfill' + run_id=<gh_run_id>

Usage:
  python scripts/_backfill_mega_sweep_runs.py \
      --root data/_backfill \
      --top 30 --extra 50

Each subdir of --root must be named `run_<gh_run_id>` and contain the
annotated CSV. Run timestamps are fetched from the GH API.
"""
import argparse
import io
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

REPO = "Maximus0Primus/TelethonIA"


def fetch_run_meta(run_id: str) -> dict:
    """Get run created_at + head_sha via gh api."""
    r = subprocess.run(
        ["gh", "api", f"repos/{REPO}/actions/runs/{run_id}",
         "--jq", "{created_at, run_started_at, head_sha, head_branch}"],
        capture_output=True, text=True, check=False,
    )
    if r.returncode != 0:
        return {}
    try:
        return json.loads(r.stdout)
    except Exception:
        return {}


def row_to_record(rr, rank: int, is_top: bool, run_at: str, run_id: str) -> dict:
    g = lambda k: rr[k] if k in rr.index and pd.notna(rr[k]) else None
    return {
        "run_at": run_at,
        "run_source": "backfill",
        "run_id": str(run_id),
        "chain": "solana",
        "strategy": str(g("strategy") or ""),
        "filter_name": str(g("filter") or "") or None,
        "rank_at_run": int(rank),
        "n_sim_trades": int(g("n")) if g("n") is not None else None,
        "avg_pnl_pct": float(g("avg_pnl_pct")) if g("avg_pnl_pct") is not None else None,
        "median_pnl_pct": float(g("median_pnl_pct")) if g("median_pnl_pct") is not None else None,
        "win_rate": float(g("wr_pct")) / 100.0 if g("wr_pct") is not None else None,
        "pnl_per_day": float(g("dollars_per_day")) if g("dollars_per_day") is not None else None,
        "family_realism": float(g("family_realism")) if g("family_realism") is not None else None,
        "fdr_q": float(g("fdr_q")) if g("fdr_q") is not None else None,
        "p_corrected": float(g("p_corrected")) if g("p_corrected") is not None else None,
        "bootstrap_rank_pct": float(g("bootstrap_rank_pct")) if g("bootstrap_rank_pct") is not None else None,
        "rank_stability": float(g("rank_stability")) if g("rank_stability") is not None else None,
        "cross_regime_robust": bool(g("cross_regime_robust")) if g("cross_regime_robust") is not None else None,
        "is_top_robust": bool(is_top),
        "metadata": {
            "smoothing": str(g("smoothing") or "") or None,
            "polling_mode": str(g("polling_mode") or "") or None,
            "source": str(g("source") or "") or None,
            "wf_consistent": bool(g("wf_consistent")) if g("wf_consistent") is not None else None,
            "pnl_active_pct": float(g("pnl_active_pct")) if g("pnl_active_pct") is not None else None,
            "pnl_quiet_pct": float(g("pnl_quiet_pct")) if g("pnl_quiet_pct") is not None else None,
            "head_sha": None,  # set below
        },
    }


def process_run(run_dir: Path, top_n: int, extra_n: int) -> int:
    name = run_dir.name
    if not name.startswith("run_"):
        return 0
    run_id = name[len("run_"):]
    annotated = run_dir / "_mega_sweep_extended_annotated.csv"
    if not annotated.exists():
        # Sometimes gh extracts inside `scraper/` subdir
        candidates = list(run_dir.rglob("_mega_sweep_extended_annotated.csv"))
        if not candidates:
            print(f"[{run_id}] no annotated CSV, skip")
            return 0
        annotated = candidates[0]

    meta = fetch_run_meta(run_id)
    run_at = meta.get("run_started_at") or meta.get("created_at")
    if not run_at:
        run_at = datetime.now(timezone.utc).isoformat()
    head_sha = meta.get("head_sha")

    # Idempotency: skip if this run_id already in DB
    existing = sb.table("mega_sweep_runs").select("id", count="exact").eq("run_id", str(run_id)).limit(1).execute()
    if existing.count and existing.count > 0:
        print(f"[{run_id}] already backfilled ({existing.count} rows), skip")
        return 0

    print(f"[{run_id}] reading {annotated} (run_at={run_at}, sha={head_sha[:8] if head_sha else '?'})")
    df = pd.read_csv(annotated, low_memory=False)
    print(f"  loaded {len(df):,} rows")

    # Same gating as analyze_mega_sweep.py top_robust logic
    base = df[(df["avg_pnl_pct"] > 0) & (df["family_realism"].fillna(0) >= 0.5)]
    if "cross_regime_robust" in df.columns:
        # Robust gate first; if too few, fall back to base
        gated = base[base["cross_regime_robust"].fillna(False) == True]
        if len(gated) >= 1:
            base = gated
    base = base.sort_values("avg_pnl_pct", ascending=False)
    robust = base.head(top_n)
    runners = base.iloc[top_n:top_n + extra_n] if len(base) > top_n else pd.DataFrame()

    records = []
    for i, (_, rr) in enumerate(robust.iterrows(), start=1):
        rec = row_to_record(rr, i, True, run_at, run_id)
        rec["metadata"]["head_sha"] = head_sha
        records.append(rec)
    for j, (_, rr) in enumerate(runners.iterrows(), start=len(records) + 1):
        rec = row_to_record(rr, j, False, run_at, run_id)
        rec["metadata"]["head_sha"] = head_sha
        records.append(rec)

    if not records:
        print(f"  [{run_id}] no qualifying rows")
        return 0

    inserted = 0
    for k in range(0, len(records), 100):
        try:
            sb.table("mega_sweep_runs").insert(records[k:k + 100]).execute()
            inserted += len(records[k:k + 100])
        except Exception as e:
            print(f"  [{run_id}] insert chunk {k} failed: {e}")
    print(f"  [{run_id}] inserted {inserted} rows (top_robust={len(robust)}, runners_up={len(runners)})")
    return inserted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/_backfill",
                    help="Directory containing run_<id>/ subdirs from `gh run download`")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--extra", type=int, default=50)
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Root not found: {root}")
        sys.exit(1)

    total = 0
    for sub in sorted(root.iterdir()):
        if sub.is_dir() and sub.name.startswith("run_"):
            total += process_run(sub, args.top, args.extra)
    print(f"\nTotal inserted: {total}")


if __name__ == "__main__":
    main()
