"""
Automated backtesting & algorithm diagnosis engine.

Runs after fill_outcomes() in each scrape cycle. Produces a structured report
that progressively enables more analyses as labeled data accumulates.

Thresholds:
  < 30 labeled: skip (waiting for data)
  30-49:        score calibration + return distribution
  50-99:        + feature correlation + gate/false-positive autopsy + v8 signals
  100-199:      + weight sensitivity + KOL accuracy + optimal threshold
  200+:         + walk-forward validation

Can also be run standalone: python auto_backtest.py
"""

import os
import sys
import json
import math
import logging
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    import numpy as np
    import pandas as pd
except ImportError:
    logger.error("pandas and numpy are required: pip install pandas numpy")
    sys.exit(1)

from supabase import create_client

# --- Constants (mirror pipeline.py v13) ---

BALANCED_WEIGHTS = {
    "consensus": 0.30,
    "conviction": 0.15,
    "breadth": 0.25,
    "price_action": 0.30,
}

TOTAL_KOLS = 60

SCORE_BANDS = [
    ("band_0_30", 0, 30),
    ("band_30_50", 30, 50),
    ("band_50_70", 50, 70),
    ("band_70_100", 70, 100),
]

# Features to correlate with 2x outcome
NUMERIC_FEATURES = [
    "price_action_score", "volume_24h", "volume_1h", "volume_6h",
    "liquidity_usd", "market_cap", "sentiment", "breadth", "avg_conviction",
    "mentions", "txn_count_24h", "buy_sell_ratio_24h", "buy_sell_ratio_1h",
    "short_term_heat", "txn_velocity", "ultra_short_heat", "volume_acceleration",
    "top10_holder_pct", "insider_pct", "risk_count", "holder_count",
    "safety_penalty", "onchain_multiplier", "social_velocity",
    "whale_total_pct", "whale_count", "wash_trading_score",
    "volatility_proxy", "whale_dominance", "already_pumped_penalty",
    "squeeze_score", "trend_strength", "data_confidence",
    "rsi_14", "macd_histogram", "bb_width", "obv_slope_norm",
    # v9/v10 features
    "freshest_mention_hours", "death_penalty",
    "boosts_active", "has_twitter", "has_telegram", "has_website", "social_count",
    # v12/v13 features
    "entry_premium", "entry_premium_mult",
    "lp_locked_pct", "unique_wallet_24h_change", "whale_new_entries",
]

# Columns to fetch from token_snapshots
SNAPSHOT_COLUMNS = (
    "id, symbol, snapshot_at, mentions, sentiment, breadth, avg_conviction, "
    "recency_score, price_action_score, volume_24h, volume_1h, volume_6h, "
    "liquidity_usd, market_cap, txn_count_24h, buy_sell_ratio_24h, "
    "buy_sell_ratio_1h, short_term_heat, txn_velocity, ultra_short_heat, "
    "volume_acceleration, top10_holder_pct, insider_pct, risk_count, "
    "holder_count, safety_penalty, onchain_multiplier, social_velocity, "
    "whale_total_pct, whale_count, wash_trading_score, volatility_proxy, "
    "whale_dominance, already_pumped_penalty, squeeze_score, squeeze_state, "
    "trend_strength, confirmation_pillars, data_confidence, "
    "rsi_14, macd_histogram, bb_width, obv_slope_norm, "
    "price_at_snapshot, price_after_1h, price_after_6h, price_after_12h, price_after_24h, "
    "max_price_1h, max_price_6h, max_price_12h, max_price_24h, "
    "did_2x_1h, did_2x_6h, did_2x_12h, did_2x_24h, mentions, "
    "freshest_mention_hours, death_penalty, lifecycle_phase, "
    "boosts_active, has_twitter, has_telegram, has_website, social_count, "
    "entry_premium, entry_premium_mult, lp_locked_pct, "
    "unique_wallet_24h_change, whale_new_entries, "
    "consensus_val, conviction_val, breadth_val, price_action_val, "
    "pump_bonus, wash_pen, pvp_pen, pump_pen, activity_mult, breadth_pen, crash_pen, stale_pen"
)


def _get_client():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        logger.error("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY")
        return None
    return create_client(url, key)


def _fetch_snapshots(client) -> pd.DataFrame:
    """Fetch all snapshots (up to 2000) with outcome labels and features."""
    result = (
        client.table("token_snapshots")
        .select(SNAPSHOT_COLUMNS)
        .order("snapshot_at", desc=True)
        .limit(2000)
        .execute()
    )
    if not result.data:
        return pd.DataFrame()

    df = pd.DataFrame(result.data)
    df["snapshot_at"] = pd.to_datetime(df["snapshot_at"])
    return df


def _safe_mult(row: pd.Series, col: str, default: float = 1.0) -> float:
    """Read a multiplier from a snapshot row, defaulting if absent/null."""
    val = row.get(col)
    return float(val) if pd.notna(val) else default


def _compute_score(row: pd.Series, weights: dict) -> int:
    """
    Recompute production-equivalent score from snapshot features.
    v14: no sentiment, (ac-6)/4 conviction, all multipliers in chain, confirmation gate 0.8.
    Prefers stored component values (consensus_val, etc.) when available.
    """
    # Prefer stored component values from pipeline (v14+), fallback to recomputing
    consensus = _safe_mult(row, "consensus_val", default=0)
    if not pd.notna(row.get("consensus_val")):
        b_raw = row.get("breadth")
        consensus = min(1.0, float(b_raw) / 0.15) if pd.notna(b_raw) else None

    conviction_val = _safe_mult(row, "conviction_val", default=0)
    if not pd.notna(row.get("conviction_val")):
        ac = row.get("avg_conviction")
        conviction_val = max(0, min(1, (float(ac) - 6) / 4)) if pd.notna(ac) else None

    breadth_val = _safe_mult(row, "breadth_val", default=0)
    if not pd.notna(row.get("breadth_val")):
        b = row.get("breadth")
        breadth_val = float(b) if pd.notna(b) else None

    pa_val = _safe_mult(row, "price_action_val", default=0.5)
    if not pd.notna(row.get("price_action_val")):
        pa = row.get("price_action_score")
        pa_val = float(pa) if pd.notna(pa) else None

    components = {
        "consensus": consensus,
        "conviction": conviction_val,
        "breadth": breadth_val,
        "price_action": pa_val,
    }

    available = {k: (v, weights[k]) for k, v in components.items() if v is not None and k in weights}
    if not available:
        return 0

    total_w = sum(w for _, w in available.values())
    raw = sum(v * (w / total_w) for v, w in available.values())
    base_score = raw * 100

    # Prefer stored crash_pen (already min of lifecycle, death, entry_premium)
    crash_pen = _safe_mult(row, "crash_pen")
    if not pd.notna(row.get("crash_pen")):
        lifecycle_pen = _safe_mult(row, "already_pumped_penalty")
        death_pen = _safe_mult(row, "death_penalty")
        entry_pen = _safe_mult(row, "entry_premium_mult")
        crash_pen = min(lifecycle_pen, death_pen, entry_pen)

    # All multipliers from stored snapshot values (v14 full chain)
    safety = _safe_mult(row, "safety_penalty")
    onchain = _safe_mult(row, "onchain_multiplier")
    pump_bonus = _safe_mult(row, "pump_bonus")
    wash_pen = _safe_mult(row, "wash_pen")
    pvp_pen = _safe_mult(row, "pvp_pen")
    pump_pen = _safe_mult(row, "pump_pen")
    activity_mult = _safe_mult(row, "activity_mult")
    breadth_pen = _safe_mult(row, "breadth_pen")
    stale_pen = _safe_mult(row, "stale_pen")

    # Squeeze + trend from v8
    sq_score = float(row.get("squeeze_score") or 0) if pd.notna(row.get("squeeze_score")) else 0
    sq_state = row.get("squeeze_state") or "none"
    squeeze_mult = 1.0
    if sq_state == "firing":
        squeeze_mult = 1.0 + sq_score * 0.2
    elif sq_state == "squeezing":
        squeeze_mult = 1.05

    ts = float(row.get("trend_strength") or 0) if pd.notna(row.get("trend_strength")) else 0
    trend_mult = 1.0
    if ts >= 0.7:
        trend_mult = 1.15
    elif ts >= 0.4:
        trend_mult = 1.07

    combined = (onchain * safety * crash_pen * squeeze_mult * trend_mult
                * pump_bonus * wash_pen * pvp_pen * pump_pen
                * activity_mult * breadth_pen * stale_pen)
    score = base_score * combined

    # v13: Confirmation gate softened to 0.8
    pillars = int(row.get("confirmation_pillars") or 3) if pd.notna(row.get("confirmation_pillars")) else 3
    if pillars < 2:
        score *= 0.8

    return min(100, max(0, int(score)))


def _compute_return(row: pd.Series, horizon: str = "12h") -> float | None:
    """Compute max return ratio for a given horizon."""
    p0 = row.get("price_at_snapshot")
    max_col = f"max_price_{horizon}"
    pmax = row.get(max_col)

    if pd.notna(p0) and pd.notna(pmax) and float(p0) > 0:
        return float(pmax) / float(p0)
    return None


def _top1_hit_rate(df: pd.DataFrame) -> dict:
    """
    THE key metric: does the #1 ranked token 2x?
    Target = 100% hit rate for the top-scored token in each scrape cycle.
    Groups snapshots by snapshot_at (same cycle = same timestamp batch),
    picks the highest-scoring token, checks if it 2x'd.
    """
    df = df.copy()
    df["score"] = df.apply(lambda r: _compute_score(r, BALANCED_WEIGHTS), axis=1)

    # Group snapshots into cycles (same minute = same cycle)
    df["cycle"] = df["snapshot_at"].dt.floor("15min")

    results = {"1h": {}, "6h": {}, "12h": {}, "24h": {}}

    for horizon in ["1h", "6h", "12h", "24h"]:
        flag_col = f"did_2x_{horizon}"
        if flag_col not in df.columns:
            continue

        cycles_tested = 0
        cycles_hit = 0
        top1_details = []

        for cycle_ts, group in df.groupby("cycle"):
            labeled = group[group[flag_col].notna()]
            if labeled.empty:
                continue

            # #1 token = highest score in this cycle
            top1 = labeled.loc[labeled["score"].idxmax()]
            did_2x = bool(top1[flag_col])
            ret = _compute_return(top1, horizon)

            cycles_tested += 1
            if did_2x:
                cycles_hit += 1

            top1_details.append({
                "cycle": str(cycle_ts),
                "symbol": top1["symbol"],
                "score": int(top1["score"]),
                "did_2x": did_2x,
                "max_return": round(ret, 2) if ret else None,
            })

        if cycles_tested > 0:
            results[horizon] = {
                "cycles_tested": cycles_tested,
                "cycles_hit": cycles_hit,
                "hit_rate": round(cycles_hit / cycles_tested, 4),
                "target": 1.0,
                "details": top1_details,
            }

    return results


# ---- Analysis functions ----

def _score_calibration(df: pd.DataFrame, horizon_col: str = "did_2x_12h") -> dict:
    """Hit rate and return stats by score band."""
    df = df.copy()
    df["score"] = df.apply(lambda r: _compute_score(r, BALANCED_WEIGHTS), axis=1)

    horizon_suffix = horizon_col.replace("did_2x_", "")
    df["max_return"] = df.apply(lambda r: _compute_return(r, horizon_suffix), axis=1)

    result = {}
    for band_name, lo, hi in SCORE_BANDS:
        band = df[(df["score"] >= lo) & (df["score"] < hi)]
        if band.empty:
            result[band_name] = {"count": 0, "hits": 0, "hit_rate": 0, "avg_return": None, "median_return": None}
            continue

        labeled = band[band[horizon_col].notna()]
        hits = int(labeled[horizon_col].sum()) if not labeled.empty else 0
        hit_rate = hits / len(labeled) if len(labeled) > 0 else 0

        returns = band["max_return"].dropna()
        result[band_name] = {
            "count": int(len(labeled)),
            "hits": hits,
            "hit_rate": round(hit_rate, 4),
            "avg_return": round(float(returns.mean()), 4) if len(returns) > 0 else None,
            "median_return": round(float(returns.median()), 4) if len(returns) > 0 else None,
            "p25_return": round(float(returns.quantile(0.25)), 4) if len(returns) > 0 else None,
            "p75_return": round(float(returns.quantile(0.75)), 4) if len(returns) > 0 else None,
        }

    return result


def _feature_correlation(df: pd.DataFrame, horizon_col: str = "did_2x_12h") -> dict:
    """Pearson correlation of each numeric feature with the 2x outcome."""
    labeled = df[df[horizon_col].notna()].copy()
    if labeled.empty:
        return {}

    target = labeled[horizon_col].astype(float)
    correlations = {}

    for feat in NUMERIC_FEATURES:
        if feat not in labeled.columns:
            continue
        col = pd.to_numeric(labeled[feat], errors="coerce")
        valid = col.notna() & target.notna()
        if valid.sum() < 10:
            continue
        try:
            r = float(np.corrcoef(col[valid], target[valid])[0, 1])
            if not math.isnan(r):
                correlations[feat] = round(r, 4)
        except Exception:
            pass

    # Sort by absolute correlation descending
    return dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))


def _gate_autopsy(df: pd.DataFrame, horizon_col: str = "did_2x_12h") -> dict:
    """For tokens that DID 2x: identify what penalties/gates reduced their score."""
    winners = df[(df[horizon_col] == True)].copy()
    if winners.empty:
        return {"missed_winners": [], "total_winners": 0}

    winners["score"] = winners.apply(lambda r: _compute_score(r, BALANCED_WEIGHTS), axis=1)

    missed = []
    for _, row in winners.iterrows():
        issues = []
        score = row["score"]

        # Check confirmation pillars
        cp = row.get("confirmation_pillars")
        if pd.notna(cp) and int(cp) < 2:
            issues.append(f"confirmation_pillars={int(cp)} (0.8x penalty)")

        # Check already-pumped penalty
        app = row.get("already_pumped_penalty")
        if pd.notna(app) and float(app) < 0.8:
            issues.append(f"already_pumped_penalty={float(app):.2f}")

        # Check safety penalty
        sp = row.get("safety_penalty")
        if pd.notna(sp) and float(sp) < 0.7:
            issues.append(f"safety_penalty={float(sp):.2f}")

        # Check data confidence
        dc = row.get("data_confidence")
        if pd.notna(dc) and float(dc) < 0.5:
            issues.append(f"data_confidence={float(dc):.2f}")

        # Low price action despite 2x
        pa = row.get("price_action_score")
        if pd.notna(pa) and float(pa) < 0.3:
            issues.append(f"price_action_score={float(pa):.2f} (low despite 2x)")

        if issues and score < 60:
            missed.append({
                "symbol": row["symbol"],
                "score": int(score),
                "issues": issues,
            })

    return {
        "missed_winners": sorted(missed, key=lambda x: x["score"]),
        "total_winners": len(winners),
        "missed_count": len(missed),
    }


def _false_positive_autopsy(df: pd.DataFrame, horizon_col: str = "did_2x_12h") -> dict:
    """For high-scoring tokens that did NOT 2x: what components were misleading?"""
    df = df.copy()
    df["score"] = df.apply(lambda r: _compute_score(r, BALANCED_WEIGHTS), axis=1)

    labeled = df[df[horizon_col].notna()]
    high_score_losers = labeled[(labeled["score"] >= 60) & (labeled[horizon_col] == False)]

    if high_score_losers.empty:
        return {"false_positives": [], "total_high_score": 0}

    fp_details = []
    for _, row in high_score_losers.iterrows():
        misleading = []

        # Which components pushed the score up?
        pa = row.get("price_action_score")
        if pd.notna(pa) and float(pa) > 0.6:
            ret = _compute_return(row, horizon_col.replace("did_2x_", ""))
            actual = f"{ret:.2f}x" if ret else "unknown"
            misleading.append(f"price_action={float(pa):.2f} but actual return={actual}")

        sent = row.get("sentiment")
        if pd.notna(sent) and float(sent) > 0.5:
            misleading.append(f"sentiment={float(sent):.2f} (high positive)")

        sv = row.get("social_velocity")
        if pd.notna(sv) and float(sv) > 2:
            misleading.append(f"social_velocity={float(sv):.1f} (high hype)")

        if misleading:
            fp_details.append({
                "symbol": row["symbol"],
                "score": int(row["score"]),
                "misleading": misleading,
            })

    total_high = int(len(labeled[labeled["score"] >= 60]))
    return {
        "false_positives": sorted(fp_details, key=lambda x: x["score"], reverse=True)[:10],
        "total_high_score": total_high,
        "false_positive_count": len(high_score_losers),
        "false_positive_rate": round(len(high_score_losers) / max(1, total_high), 4),
    }


def _weight_sensitivity(df: pd.DataFrame, horizon_col: str = "did_2x_12h") -> dict:
    """Vary each BALANCED_WEIGHTS component +/-25%, measure hit rate delta."""
    labeled = df[df[horizon_col].notna()].copy()
    if labeled.empty:
        return {}

    # Baseline hit rate at threshold 50 (lower threshold for more signal)
    threshold = 50
    labeled["base_score"] = labeled.apply(lambda r: _compute_score(r, BALANCED_WEIGHTS), axis=1)
    above = labeled[labeled["base_score"] >= threshold]
    base_hit_rate = float(above[horizon_col].sum()) / max(1, len(above))

    results = {}
    for comp in BALANCED_WEIGHTS:
        for direction in ["+25%", "-25%"]:
            mult = 1.25 if direction == "+25%" else 0.75
            test_weights = BALANCED_WEIGHTS.copy()
            test_weights[comp] = test_weights[comp] * mult

            # Renormalize to sum to 1.0
            total = sum(test_weights.values())
            test_weights = {k: v / total for k, v in test_weights.items()}

            labeled["test_score"] = labeled.apply(lambda r: _compute_score(r, test_weights), axis=1)
            above_test = labeled[labeled["test_score"] >= threshold]
            test_hr = float(above_test[horizon_col].sum()) / max(1, len(above_test))

            key = f"{comp}_{direction}"
            results[key] = {
                "hit_rate": round(test_hr, 4),
                "hit_rate_delta": round(test_hr - base_hit_rate, 4),
                "sample_size": len(above_test),
            }

    results["baseline"] = {
        "hit_rate": round(base_hit_rate, 4),
        "sample_size": len(above),
        "threshold": threshold,
    }

    return results


def _v8_signal_validation(df: pd.DataFrame, horizon_col: str = "did_2x_12h") -> dict:
    """Do v8 signals (squeeze, trend, confirmation) correlate with 2x?"""
    labeled = df[df[horizon_col].notna()].copy()
    if labeled.empty:
        return {}

    result = {}

    # Squeeze state
    if "squeeze_state" in labeled.columns:
        for state in ["firing", "squeezing", "none"]:
            subset = labeled[labeled["squeeze_state"] == state]
            if len(subset) >= 3:
                hr = float(subset[horizon_col].sum()) / len(subset)
                result[f"squeeze_{state}_hit_rate"] = round(hr, 4)
                result[f"squeeze_{state}_count"] = len(subset)

    # Trend strength bins
    if "trend_strength" in labeled.columns:
        ts = pd.to_numeric(labeled["trend_strength"], errors="coerce")
        for label, lo, hi in [("low", 0, 0.3), ("medium", 0.3, 0.6), ("high", 0.6, 1.01)]:
            subset = labeled[(ts >= lo) & (ts < hi)]
            if len(subset) >= 3:
                hr = float(subset[horizon_col].sum()) / len(subset)
                result[f"trend_{label}_hit_rate"] = round(hr, 4)
                result[f"trend_{label}_count"] = len(subset)

    # Confirmation pillars
    if "confirmation_pillars" in labeled.columns:
        cp = pd.to_numeric(labeled["confirmation_pillars"], errors="coerce")
        for n_pillars in [0, 1, 2, 3]:
            subset = labeled[cp == n_pillars]
            if len(subset) >= 3:
                hr = float(subset[horizon_col].sum()) / len(subset)
                result[f"pillars_{n_pillars}_hit_rate"] = round(hr, 4)
                result[f"pillars_{n_pillars}_count"] = len(subset)

    return result


def _optimal_threshold(df: pd.DataFrame, horizon_col: str = "did_2x_12h") -> dict:
    """Find score threshold that maximizes F1 for 2x prediction."""
    labeled = df[df[horizon_col].notna()].copy()
    if labeled.empty:
        return {}

    labeled["score"] = labeled.apply(lambda r: _compute_score(r, BALANCED_WEIGHTS), axis=1)

    best_f1 = 0
    best_threshold = 50
    threshold_results = []

    for threshold in range(20, 90, 5):
        predicted = labeled["score"] >= threshold
        actual = labeled[horizon_col].astype(bool)

        tp = int((predicted & actual).sum())
        fp = int((predicted & ~actual).sum())
        fn = int((~predicted & actual).sum())

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(0.001, precision + recall)

        entry = {
            "threshold": threshold,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "predicted_positive": int(predicted.sum()),
            "true_positive": tp,
        }
        threshold_results.append(entry)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return {
        "best_threshold": best_threshold,
        "best_f1": round(best_f1, 4),
        "all_thresholds": threshold_results,
    }


def _walk_forward(df: pd.DataFrame, horizon_col: str = "did_2x_12h") -> list[dict]:
    """Time-based train/test split validation."""
    labeled = df[df[horizon_col].notna()].copy()
    if labeled.empty:
        return []

    labeled = labeled.sort_values("snapshot_at")
    labeled["score"] = labeled.apply(lambda r: _compute_score(r, BALANCED_WEIGHTS), axis=1)

    min_date = labeled["snapshot_at"].min()
    max_date = labeled["snapshot_at"].max()
    total_days = (max_date - min_date).days

    if total_days < 7:
        return []

    # Split into roughly 3-day train / 1-day test windows
    train_days = max(3, total_days * 3 // 4)
    test_days = max(1, total_days - train_days)

    folds = []
    fold_start = min_date

    while fold_start + pd.Timedelta(days=train_days + test_days) <= max_date + pd.Timedelta(hours=12):
        train_end = fold_start + pd.Timedelta(days=train_days)
        test_end = train_end + pd.Timedelta(days=test_days)

        train_df = labeled[(labeled["snapshot_at"] >= fold_start) & (labeled["snapshot_at"] < train_end)]
        test_df = labeled[(labeled["snapshot_at"] >= train_end) & (labeled["snapshot_at"] < test_end)]

        if len(train_df) < 5 or len(test_df) < 2:
            fold_start += pd.Timedelta(days=test_days)
            continue

        threshold = 50
        train_above = train_df[train_df["score"] >= threshold]
        test_above = test_df[test_df["score"] >= threshold]

        train_hr = float(train_above[horizon_col].sum()) / max(1, len(train_above))
        test_hr = float(test_above[horizon_col].sum()) / max(1, len(test_above))
        gap = abs(train_hr - test_hr)

        folds.append({
            "fold_start": fold_start.isoformat(),
            "train_size": len(train_df),
            "test_size": len(test_df),
            "train_hit_rate": round(train_hr, 4),
            "test_hit_rate": round(test_hr, 4),
            "gap": round(gap, 4),
            "overfit_risk": "HIGH" if gap > 0.2 else "LOW" if gap < 0.1 else "MEDIUM",
        })

        fold_start += pd.Timedelta(days=test_days)

    return folds


def _generate_recommendations(report: dict) -> list[str]:
    """Generate actionable text recommendations from analysis results."""
    recs = []

    # Score calibration insights
    cal = report.get("score_calibration", {})
    high_band = cal.get("band_70_100", {})
    low_band = cal.get("band_0_30", {})
    if high_band.get("hit_rate", 0) > 0 and low_band.get("hit_rate", 0) > 0:
        ratio = high_band["hit_rate"] / max(0.001, low_band["hit_rate"])
        if ratio > 2:
            recs.append(f"GOOD: High scores ({high_band['hit_rate']*100:.0f}% hit rate) "
                       f"outperform low scores ({low_band['hit_rate']*100:.0f}%) by {ratio:.1f}x")
        else:
            recs.append(f"WARNING: Score discrimination is weak ({ratio:.1f}x ratio). "
                       "Algorithm needs major recalibration.")
    elif high_band.get("count", 0) == 0:
        recs.append("NOTE: No tokens scored 70+. Consider lowering analysis threshold.")

    # Feature correlation insights
    corr = report.get("feature_correlation", {})
    if corr:
        top_features = list(corr.items())[:3]
        bottom_features = [(k, v) for k, v in corr.items() if abs(v) < 0.05]

        for feat, r in top_features:
            direction = "positive" if r > 0 else "negative"
            recs.append(f"TOP SIGNAL: {feat} has {direction} correlation ({r:.3f}) with 2x")

        if bottom_features:
            names = [f[0] for f in bottom_features[:3]]
            recs.append(f"WEAK SIGNALS: {', '.join(names)} show near-zero correlation with 2x")

    # Weight sensitivity insights
    ws = report.get("weight_sensitivity", {})
    if ws and "baseline" in ws:
        baseline_hr = ws["baseline"]["hit_rate"]
        best_change = None
        best_delta = 0
        for key, val in ws.items():
            if key == "baseline":
                continue
            delta = val.get("hit_rate_delta", 0)
            if delta > best_delta:
                best_delta = delta
                best_change = key

        if best_change and best_delta > 0.02:
            recs.append(f"INCREASE: {best_change} improves hit rate by "
                       f"{best_delta*100:.1f}pp (from {baseline_hr*100:.1f}%)")

        # Find worst change
        worst_change = None
        worst_delta = 0
        for key, val in ws.items():
            if key == "baseline":
                continue
            delta = val.get("hit_rate_delta", 0)
            if delta < worst_delta:
                worst_delta = delta
                worst_change = key

        if worst_change and worst_delta < -0.02:
            recs.append(f"AVOID: {worst_change} reduces hit rate by "
                       f"{abs(worst_delta)*100:.1f}pp")

    # v8 signal insights
    v8 = report.get("v8_signals", {})
    squeeze_firing = v8.get("squeeze_firing_hit_rate")
    squeeze_none = v8.get("squeeze_none_hit_rate")
    if squeeze_firing is not None and squeeze_none is not None:
        if squeeze_firing > squeeze_none * 1.5:
            recs.append(f"KEEP squeeze signal: firing={squeeze_firing*100:.0f}% vs "
                       f"none={squeeze_none*100:.0f}% hit rate")
        elif squeeze_firing <= squeeze_none:
            recs.append("REMOVE squeeze signal: firing tokens don't outperform baseline")

    pillars_3 = v8.get("pillars_3_hit_rate")
    pillars_0 = v8.get("pillars_0_hit_rate")
    if pillars_3 is not None and pillars_0 is not None:
        if pillars_3 > pillars_0 * 1.5:
            recs.append(f"KEEP confirmation gate: 3-pillar={pillars_3*100:.0f}% vs "
                       f"0-pillar={pillars_0*100:.0f}% hit rate")

    # Gate autopsy insights
    gate = report.get("gate_autopsy", {})
    missed = gate.get("missed_count", 0)
    total_winners = gate.get("total_winners", 0)
    if total_winners > 0 and missed > 0:
        miss_rate = missed / total_winners
        if miss_rate > 0.3:
            recs.append(f"FALSE NEGATIVES: {missed}/{total_winners} winners scored <60. "
                       "Gates may be too aggressive.")

    # False positive insights
    fp = report.get("false_positive_autopsy", {})
    fp_rate = fp.get("false_positive_rate", 0)
    if fp_rate > 0.8:
        recs.append(f"HIGH FALSE POSITIVE RATE: {fp_rate*100:.0f}% of high-scoring tokens "
                   "didn't 2x. Algorithm is overconfident.")

    # === PRIMARY TARGET: #1 token must always 2x ===
    top1 = report.get("top1_hit_rate", {})
    for horizon in ["12h", "6h", "24h"]:
        t1 = top1.get(horizon, {})
        if t1.get("cycles_tested", 0) > 0:
            hr = t1["hit_rate"]
            tested = t1["cycles_tested"]
            hits = t1["cycles_hit"]
            if hr >= 1.0:
                recs.append(f"TARGET MET ({horizon}): #1 token 2x'd in {hits}/{tested} cycles (100%)")
            elif hr >= 0.5:
                recs.append(f"CLOSE ({horizon}): #1 token 2x'd in {hits}/{tested} cycles ({hr*100:.0f}%). Target: 100%")
            else:
                recs.append(f"NEEDS WORK ({horizon}): #1 token 2x'd in {hits}/{tested} cycles ({hr*100:.0f}%). Target: 100%")
            # Show which #1 tokens failed
            for d in t1.get("details", []):
                if not d["did_2x"]:
                    ret = d.get("max_return")
                    ret_str = f"{ret}x" if ret else "?"
                    recs.append(f"  MISS: {d['symbol']} score={d['score']} return={ret_str}")
            break  # Only show primary horizon

    # Overall hit rate
    data = report.get("data_summary", {})
    hr = data.get("hit_rate_12h", 0)
    if hr > 0:
        recs.append(f"Overall 12h 2x rate: {hr*100:.0f}% across all tokens")

    if not recs:
        recs.append("Insufficient data for recommendations. Continue collecting snapshots.")

    return recs


def run_auto_backtest() -> dict | None:
    """
    Main entry point. Fetches snapshots, runs progressive analyses,
    returns structured report dict. Returns None if not enough data.
    """
    client = _get_client()
    if not client:
        return None

    df = _fetch_snapshots(client)
    if df.empty:
        logger.info("Auto-backtest: no snapshots found")
        return None

    total = len(df)

    # Count labeled snapshots per horizon
    labeled_12h = df[df["did_2x_12h"].notna()]
    labeled_24h = df[df["did_2x_24h"].notna()]
    n_labeled_12h = len(labeled_12h)
    n_labeled_24h = len(labeled_24h)

    # Use 12h as primary horizon (faster feedback loop)
    n_labeled = n_labeled_12h
    horizon_col = "did_2x_12h"

    if n_labeled < 30:
        logger.info("Auto-backtest: waiting for data (%d/30 labeled snapshots)", n_labeled)
        return None

    # Data summary
    date_range = (df["snapshot_at"].max() - df["snapshot_at"].min()).days
    hit_rate_12h = float(labeled_12h[horizon_col].sum()) / max(1, n_labeled_12h)
    hit_rate_24h = float(labeled_24h["did_2x_24h"].sum()) / max(1, n_labeled_24h) if n_labeled_24h > 0 else None

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_summary": {
            "total_snapshots": total,
            "labeled_12h": n_labeled_12h,
            "labeled_24h": n_labeled_24h,
            "hit_rate_12h": round(hit_rate_12h, 4),
            "hit_rate_24h": round(hit_rate_24h, 4) if hit_rate_24h is not None else None,
            "date_range_days": date_range,
        },
    }

    # === TOP 1 TOKEN HIT RATE (the primary target) ===
    logger.info("Auto-backtest: computing #1 token hit rate")
    report["top1_hit_rate"] = _top1_hit_rate(df)

    # Progressive analyses based on available data

    # Tier 1: 30+ labeled
    logger.info("Auto-backtest: running score calibration (n=%d)", n_labeled)
    report["score_calibration"] = _score_calibration(df, horizon_col)

    # Tier 2: 50+ labeled
    if n_labeled >= 50:
        logger.info("Auto-backtest: running feature correlation + autopsies")
        report["feature_correlation"] = _feature_correlation(df, horizon_col)
        report["gate_autopsy"] = _gate_autopsy(df, horizon_col)
        report["false_positive_autopsy"] = _false_positive_autopsy(df, horizon_col)
        report["v8_signals"] = _v8_signal_validation(df, horizon_col)

    # Tier 3: 100+ labeled
    if n_labeled >= 100:
        logger.info("Auto-backtest: running weight sensitivity + optimal threshold")
        report["weight_sensitivity"] = _weight_sensitivity(df, horizon_col)
        report["optimal_threshold"] = _optimal_threshold(df, horizon_col)

    # Tier 4: 200+ labeled
    if n_labeled >= 200:
        logger.info("Auto-backtest: running walk-forward validation")
        report["walk_forward"] = _walk_forward(df, horizon_col)

    # Generate recommendations
    report["recommendations"] = _generate_recommendations(report)

    # Push to Supabase
    try:
        _insert_backtest_report(client, report, total, n_labeled, hit_rate_12h)
    except Exception as e:
        logger.error("Failed to insert backtest report to Supabase: %s", e, exc_info=True)

    return report


def _insert_backtest_report(
    client, report: dict, snapshots_analyzed: int,
    labeled_count: int, overall_hit_rate: float,
) -> None:
    """Insert report into backtest_reports table."""
    # Serialize report to JSON-safe dict (handles numpy, datetime, etc.)
    report_json = json.loads(json.dumps(report, default=str))
    recs = [str(r) for r in report.get("recommendations", [])]

    row = {
        "snapshots_analyzed": int(snapshots_analyzed),
        "labeled_count": int(labeled_count),
        "overall_hit_rate": round(float(overall_hit_rate), 4),
        "report": report_json,
        "recommendations": recs,
    }
    result = client.table("backtest_reports").insert(row).execute()
    logger.info("Backtest report saved to Supabase (labeled=%d, hr=%.1f%%, id=%s)",
                labeled_count, overall_hit_rate * 100,
                result.data[0]["id"] if result.data else "?")


# --- CLI ---

def main():
    """Run auto-backtest standalone and print report."""
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")

    report = run_auto_backtest()
    if not report:
        print("No report generated (not enough labeled data).")
        return

    # Pretty print
    print("\n" + "=" * 60)
    print("  AUTO-BACKTEST REPORT")
    print("=" * 60)

    ds = report["data_summary"]
    print(f"\nData: {ds['total_snapshots']} snapshots, "
          f"{ds['labeled_12h']} labeled (12h), "
          f"{ds['date_range_days']} days")
    print(f"Hit rate: 12h={ds['hit_rate_12h']*100:.1f}%"
          + (f", 24h={ds['hit_rate_24h']*100:.1f}%" if ds.get('hit_rate_24h') else ""))

    # #1 Token hit rate (PRIMARY TARGET)
    top1 = report.get("top1_hit_rate", {})
    if top1:
        print(f"\n#1 Token Hit Rate (TARGET = 100%):")
        for horizon in ["1h", "6h", "12h", "24h"]:
            t1 = top1.get(horizon, {})
            if t1.get("cycles_tested", 0) > 0:
                hr = t1["hit_rate"]
                marker = "OK" if hr >= 1.0 else "MISS"
                print(f"  {horizon}: {t1['cycles_hit']}/{t1['cycles_tested']} = "
                      f"{hr*100:.0f}% [{marker}]")
                for d in t1.get("details", []):
                    status = "2x" if d["did_2x"] else "FAIL"
                    ret = f"{d['max_return']}x" if d.get("max_return") else "?"
                    print(f"    {d['symbol']:15s} score={d['score']:3d} "
                          f"return={ret:>6s} [{status}]")

    # Score calibration
    cal = report.get("score_calibration", {})
    if cal:
        print(f"\nScore Calibration:")
        for band_name, lo, hi in SCORE_BANDS:
            b = cal.get(band_name, {})
            if b.get("count", 0) > 0:
                print(f"  {lo}-{hi}: {b['hit_rate']*100:.1f}% hit rate "
                      f"({b['hits']}/{b['count']}), "
                      f"avg_return={b.get('avg_return', 'N/A')}x")

    # Feature correlation
    corr = report.get("feature_correlation", {})
    if corr:
        print(f"\nTop Feature Correlations:")
        for feat, r in list(corr.items())[:8]:
            bar = "+" * max(0, int(abs(r) * 20)) if r > 0 else "-" * max(0, int(abs(r) * 20))
            print(f"  {feat:30s} {r:+.3f} {bar}")

    # v8 signals
    v8 = report.get("v8_signals", {})
    if v8:
        print(f"\nv8 Signals:")
        for key, val in v8.items():
            print(f"  {key}: {val}")

    # Weight sensitivity
    ws = report.get("weight_sensitivity", {})
    if ws:
        print(f"\nWeight Sensitivity:")
        baseline = ws.get("baseline", {})
        print(f"  Baseline: {baseline.get('hit_rate', 0)*100:.1f}% "
              f"(threshold={baseline.get('threshold', 50)}, n={baseline.get('sample_size', 0)})")
        for key, val in ws.items():
            if key == "baseline":
                continue
            delta = val.get("hit_rate_delta", 0)
            arrow = "^" if delta > 0 else "v" if delta < 0 else "="
            print(f"  {key:30s} {arrow} {delta*100:+.1f}pp (n={val.get('sample_size', 0)})")

    # Recommendations
    recs = report.get("recommendations", [])
    if recs:
        print(f"\nRecommendations:")
        for i, rec in enumerate(recs, 1):
            print(f"  {i}. {rec}")

    print(f"\n{'=' * 60}\n")

    # Save to file
    output_file = Path(__file__).parent / "backtest_report.json"
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Full report saved to {output_file}")


if __name__ == "__main__":
    main()
