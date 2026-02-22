"""
Backtest pipeline — validate scoring algorithm against historical token_snapshots.

Standalone CLI tool. Read-only analysis, no DB writes.

Usage:
    python backtest.py                    # Basic hit-rate report
    python backtest.py --threshold 70     # Custom score threshold
    python backtest.py --sensitivity      # Parameter sensitivity analysis
    python backtest.py --walk-forward     # Walk-forward validation
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("ERROR: pandas and numpy are required. Install with: pip install pandas numpy")
    sys.exit(1)

try:
    from supabase import create_client
except ImportError:
    print("ERROR: supabase-py is required. Install with: pip install supabase")
    sys.exit(1)


# Mirror of pipeline.py scoring weights
DEFAULT_WEIGHTS = {
    "consensus": 0.25,
    "sentiment": 0.05,
    "conviction": 0.15,
    "breadth": 0.15,
    "price_action": 0.40,
}

TOTAL_KOLS = 59  # approximate KOL count


def _get_client():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        print("ERROR: Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables")
        sys.exit(1)
    return create_client(url, key)


def fetch_snapshots(client, min_snapshots: int = 0) -> pd.DataFrame:
    """Fetch all snapshots that have outcome labels (did_2x_24h)."""
    result = (
        client.table("token_snapshots")
        .select(
            "symbol, snapshot_at, mentions, sentiment, breadth, avg_conviction, "
            "recency_score, unique_kols, price_action_score, did_2x_24h, did_2x_12h, "
            "price_at_snapshot, price_after_24h, max_price_24h, volume_24h, liquidity_usd"
        )
        .not_.is_("did_2x_24h", "null")
        .gte("snapshot_at", "2026-02-14T00:00:00Z")  # skip pre-v34 poisoned data
        .order("snapshot_at", desc=True)
        .limit(1000)
        .execute()
    )

    if not result.data:
        return pd.DataFrame()

    df = pd.DataFrame(result.data)
    df["snapshot_at"] = pd.to_datetime(df["snapshot_at"])

    if len(df) < min_snapshots:
        logger.warning(
            "Only %d labeled snapshots (need %d for statistical significance)",
            len(df), min_snapshots,
        )

    return df


def fetch_all_snapshots(client) -> pd.DataFrame:
    """Fetch ALL snapshots (labeled + unlabeled) for general stats."""
    result = (
        client.table("token_snapshots")
        .select(
            "symbol, snapshot_at, mentions, sentiment, breadth, avg_conviction, "
            "recency_score, price_action_score, price_at_snapshot, "
            "volume_24h, liquidity_usd, did_2x_24h"
        )
        .gte("snapshot_at", "2026-02-14T00:00:00Z")  # skip pre-v34 poisoned data
        .order("snapshot_at", desc=True)
        .limit(2000)
        .execute()
    )

    if not result.data:
        return pd.DataFrame()

    df = pd.DataFrame(result.data)
    df["snapshot_at"] = pd.to_datetime(df["snapshot_at"])
    return df


def compute_score_from_snapshot(row: pd.Series, weights: dict) -> int:
    """Recompute balanced score from snapshot features using given weights."""
    # Consensus: unique_kols / (total_kols * 0.15)
    uk = row.get("unique_kols")
    if uk is not None and not pd.isna(uk):
        consensus = min(1.0, float(uk) / (TOTAL_KOLS * 0.15))
    else:
        consensus = None

    # Sentiment: (sentiment + 1) / 2
    s = row.get("sentiment")
    if s is not None and not pd.isna(s):
        sentiment_val = (float(s) + 1) / 2
    else:
        sentiment_val = None

    # Conviction: (avg_conviction - 5) / 5
    ac = row.get("avg_conviction")
    if ac is not None and not pd.isna(ac):
        conviction_val = max(0, min(1, (float(ac) - 5) / 5))
    else:
        conviction_val = None

    # Breadth: direct from snapshot
    b = row.get("breadth")
    if b is not None and not pd.isna(b):
        breadth_val = float(b)
    else:
        breadth_val = None

    # Price action: direct or None
    pa = row.get("price_action_score")
    if pa is not None and not pd.isna(pa):
        pa_val = float(pa)
    else:
        pa_val = None

    # Renormalization: only use available components
    components = {
        "consensus": consensus,
        "sentiment": sentiment_val,
        "conviction": conviction_val,
        "breadth": breadth_val,
        "price_action": pa_val,
    }

    available = {}
    for comp, val in components.items():
        if val is not None:
            available[comp] = (val, weights[comp])

    if not available:
        return 0

    total_w = sum(w for _, w in available.values())
    raw = sum(v * (w / total_w) for v, w in available.values())
    return min(100, max(0, int(raw * 100)))


def evaluate_weights(df: pd.DataFrame, weights: dict, score_threshold: int = 60) -> dict:
    """For tokens scoring >= threshold, compute hit_rate, avg_return, sample_size."""
    if df.empty:
        return {"hit_rate": 0, "sample_size": 0, "avg_return": 0}

    df = df.copy()
    df["recomputed_score"] = df.apply(lambda r: compute_score_from_snapshot(r, weights), axis=1)

    above = df[df["recomputed_score"] >= score_threshold]
    if above.empty:
        return {"hit_rate": 0, "sample_size": 0, "avg_return": 0}

    hits = above["did_2x_24h"].sum()
    total = len(above)
    hit_rate = hits / total if total > 0 else 0

    # Average return: max_price_24h / price_at_snapshot
    returns = []
    for _, row in above.iterrows():
        p0 = row.get("price_at_snapshot")
        pmax = row.get("max_price_24h")
        if p0 and pmax and float(p0) > 0:
            returns.append(float(pmax) / float(p0))

    avg_return = np.mean(returns) if returns else 0

    return {
        "hit_rate": round(float(hit_rate), 4),
        "sample_size": int(total),
        "avg_return": round(float(avg_return), 4),
        "hits": int(hits),
    }


def parameter_sensitivity(
    df: pd.DataFrame,
    base_weights: dict,
    threshold: int = 60,
    variation: float = 0.25,
    steps: int = 5,
) -> pd.DataFrame:
    """
    Vary each weight +/-25% in steps. Output: weight_config -> hit_rate.
    Checks if we're on a plateau (stable) or spike (overfitting).
    """
    results = []

    for comp in base_weights:
        base_val = base_weights[comp]
        for step in range(-steps, steps + 1):
            delta = base_val * variation * (step / steps)
            test_weights = base_weights.copy()
            test_weights[comp] = max(0.01, base_val + delta)

            # Renormalize to sum to 1.0
            total = sum(test_weights.values())
            test_weights = {k: v / total for k, v in test_weights.items()}

            eval_result = evaluate_weights(df, test_weights, threshold)
            results.append({
                "component": comp,
                "delta_pct": round(delta / max(0.01, base_val) * 100, 1),
                "weight_value": round(test_weights[comp], 4),
                "hit_rate": eval_result["hit_rate"],
                "sample_size": eval_result["sample_size"],
            })

    return pd.DataFrame(results)


def walk_forward(
    df: pd.DataFrame,
    weights: dict,
    threshold: int = 60,
    train_weeks: int = 4,
    test_weeks: int = 1,
) -> list[dict]:
    """
    Walk-forward validation: train on N weeks, test on next week, roll forward.
    Returns per-fold: train_hit_rate, test_hit_rate, gap (overfit indicator).
    """
    if df.empty or "snapshot_at" not in df.columns:
        return []

    df = df.sort_values("snapshot_at")
    min_date = df["snapshot_at"].min()
    max_date = df["snapshot_at"].max()

    total_days = (max_date - min_date).days
    train_days = train_weeks * 7
    test_days = test_weeks * 7

    if total_days < train_days + test_days:
        logger.warning(
            "Not enough data for walk-forward: %d days available, need %d+%d",
            total_days, train_days, test_days,
        )
        return []

    folds = []
    fold_start = min_date

    while fold_start + pd.Timedelta(days=train_days + test_days) <= max_date:
        train_end = fold_start + pd.Timedelta(days=train_days)
        test_end = train_end + pd.Timedelta(days=test_days)

        train_df = df[(df["snapshot_at"] >= fold_start) & (df["snapshot_at"] < train_end)]
        test_df = df[(df["snapshot_at"] >= train_end) & (df["snapshot_at"] < test_end)]

        if len(train_df) < 5 or len(test_df) < 2:
            fold_start += pd.Timedelta(days=test_days)
            continue

        train_eval = evaluate_weights(train_df, weights, threshold)
        test_eval = evaluate_weights(test_df, weights, threshold)

        gap = abs(train_eval["hit_rate"] - test_eval["hit_rate"])
        folds.append({
            "fold_start": fold_start.isoformat(),
            "train_size": len(train_df),
            "test_size": len(test_df),
            "train_hit_rate": train_eval["hit_rate"],
            "test_hit_rate": test_eval["hit_rate"],
            "gap": round(gap, 4),
            "overfit_risk": "HIGH" if gap > 0.2 else "LOW" if gap < 0.1 else "MEDIUM",
        })

        fold_start += pd.Timedelta(days=test_days)

    return folds


def print_report(
    all_df: pd.DataFrame,
    labeled_df: pd.DataFrame,
    eval_result: dict,
    threshold: int,
    sensitivity_df: pd.DataFrame | None = None,
    walk_folds: list[dict] | None = None,
):
    """Print formatted backtest report to stdout."""
    print("\n" + "=" * 60)
    print("  BACKTEST REPORT")
    print("=" * 60)

    print(f"\nSnapshots: {len(all_df)} total, {len(labeled_df)} with outcome labels")
    print(f"Score threshold: >= {threshold}")

    if eval_result["sample_size"] > 0:
        print(f"\nHit rate (2x in 24h): {eval_result['hit_rate']*100:.1f}% "
              f"({eval_result.get('hits', 0)}/{eval_result['sample_size']})")
        print(f"Avg max return (24h): {eval_result['avg_return']:.2f}x")
    else:
        print("\nNo tokens scored above threshold with outcome labels.")

    if eval_result["sample_size"] < 30:
        print(f"\n** WARNING: need 30+ labeled snapshots for statistical significance "
              f"(currently {eval_result['sample_size']})")

    # Score distribution
    if not labeled_df.empty:
        labeled_df = labeled_df.copy()
        labeled_df["score"] = labeled_df.apply(
            lambda r: compute_score_from_snapshot(r, DEFAULT_WEIGHTS), axis=1
        )
        print(f"\nScore distribution (labeled):")
        print(f"  Mean: {labeled_df['score'].mean():.1f}")
        print(f"  Median: {labeled_df['score'].median():.1f}")
        print(f"  Min: {labeled_df['score'].min()}, Max: {labeled_df['score'].max()}")

    # Sensitivity
    if sensitivity_df is not None and not sensitivity_df.empty:
        print(f"\n{'='*60}")
        print("  PARAMETER SENSITIVITY")
        print(f"{'='*60}")
        for comp in DEFAULT_WEIGHTS:
            comp_data = sensitivity_df[sensitivity_df["component"] == comp]
            if not comp_data.empty:
                hr_range = comp_data["hit_rate"].max() - comp_data["hit_rate"].min()
                stability = "STABLE" if hr_range < 0.1 else "SENSITIVE" if hr_range < 0.2 else "UNSTABLE"
                print(f"\n  {comp} (current={DEFAULT_WEIGHTS[comp]:.2f}):")
                print(f"    Hit rate range: {comp_data['hit_rate'].min()*100:.1f}% - "
                      f"{comp_data['hit_rate'].max()*100:.1f}% [{stability}]")

    # Walk-forward
    if walk_folds is not None:
        print(f"\n{'='*60}")
        print("  WALK-FORWARD VALIDATION")
        print(f"{'='*60}")
        if not walk_folds:
            print("\n  Not enough data for walk-forward (need 5+ weeks of labeled snapshots)")
        else:
            for i, fold in enumerate(walk_folds):
                print(f"\n  Fold {i+1} (start: {fold['fold_start'][:10]}):")
                print(f"    Train: {fold['train_hit_rate']*100:.1f}% "
                      f"(n={fold['train_size']})")
                print(f"    Test:  {fold['test_hit_rate']*100:.1f}% "
                      f"(n={fold['test_size']})")
                print(f"    Gap:   {fold['gap']*100:.1f}% [{fold['overfit_risk']}]")

    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Backtest memecoin scoring algorithm")
    parser.add_argument("--threshold", type=int, default=60,
                        help="Score threshold for hit-rate analysis (default: 60)")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Run parameter sensitivity analysis")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Run walk-forward validation")
    parser.add_argument("--output", type=str, default="backtest_results.json",
                        help="Output JSON file (default: backtest_results.json)")
    args = parser.parse_args()

    client = _get_client()

    # Fetch data
    print("Fetching snapshots from Supabase...")
    all_df = fetch_all_snapshots(client)
    labeled_df = fetch_snapshots(client)

    print(f"  Total snapshots: {len(all_df)}")
    print(f"  Labeled (with outcomes): {len(labeled_df)}")

    # Basic evaluation
    eval_result = evaluate_weights(labeled_df, DEFAULT_WEIGHTS, args.threshold)

    # Optional: sensitivity
    sensitivity_df = None
    if args.sensitivity and not labeled_df.empty:
        print("Running parameter sensitivity analysis...")
        sensitivity_df = parameter_sensitivity(labeled_df, DEFAULT_WEIGHTS, args.threshold)

    # Optional: walk-forward
    walk_folds = None
    if args.walk_forward and not labeled_df.empty:
        print("Running walk-forward validation...")
        walk_folds = walk_forward(labeled_df, DEFAULT_WEIGHTS, args.threshold)

    # Print report
    print_report(all_df, labeled_df, eval_result, args.threshold, sensitivity_df, walk_folds)

    # Save results
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_snapshots": len(all_df),
        "labeled_snapshots": len(labeled_df),
        "threshold": args.threshold,
        "weights": DEFAULT_WEIGHTS,
        "evaluation": eval_result,
    }
    if sensitivity_df is not None:
        output["sensitivity"] = sensitivity_df.to_dict(orient="records")
    if walk_folds is not None:
        output["walk_forward"] = walk_folds

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
