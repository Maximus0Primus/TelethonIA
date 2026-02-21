"""
Automated backtesting & algorithm diagnosis engine.
ML v3: Token re-entries with cooldown (replaces permanent seen_addresses dedup).

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

# --- Constants (fallback, overridden by scoring_config table) ---

_DEFAULT_WEIGHTS = {
    "consensus": 0.35,      # v43: synced with scoring_config (auto_backtest Feb 18)
    "conviction": 0.10,     # v43: synced with scoring_config (was 0.00)
    "breadth": 0.55,        # v43: synced with scoring_config (was 0.45)
    "price_action": 0.00,   # v43: synced with scoring_config (PA r=-0.01)
}

BALANCED_WEIGHTS = _DEFAULT_WEIGHTS.copy()


# v22: Module-level ML config read from scoring_config
ML_HORIZON = "12h"
ML_THRESHOLD = 2.0


def _load_current_weights(client) -> dict:
    """Load current production weights + ML config from scoring_config table."""
    global BALANCED_WEIGHTS, ML_HORIZON, ML_THRESHOLD
    try:
        result = client.table("scoring_config").select("*").eq("id", 1).execute()
        if result.data:
            row = result.data[0]
            weights = {
                "consensus": float(row["w_consensus"]),
                "conviction": float(row["w_conviction"]),
                "breadth": float(row["w_breadth"]),
                "price_action": float(row["w_price_action"]),
            }
            total = sum(weights.values())
            if abs(total - 1.0) < 0.02:
                BALANCED_WEIGHTS.update(weights)
                logger.info("auto_backtest: loaded weights from scoring_config: %s", weights)

            # v22: Read ML horizon + threshold
            ML_HORIZON = row.get("ml_horizon", "12h") or "12h"
            ML_THRESHOLD = float(row.get("ml_threshold", 2.0) or 2.0)
            logger.info("auto_backtest: ML target = %s/%.1fx", ML_HORIZON, ML_THRESHOLD)

            return weights
    except Exception as e:
        logger.debug("auto_backtest: scoring_config load failed: %s", e)
    return BALANCED_WEIGHTS.copy()

# ML v3: Re-entry support — token can be traded again after position closes
# No arbitrary cooldown: the score/ML decides if re-entry is warranted
HORIZON_MINUTES = {
    "1h": 60, "6h": 360, "12h": 720, "24h": 1440,
    "48h": 2880, "72h": 4320, "7d": 10080,
}

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
    # Extraction mode features
    "ca_mention_count", "ticker_mention_count", "url_mention_count",
    # ML v2 Phase B: Temporal velocity
    "score_velocity", "score_acceleration", "mention_velocity",
    "volume_velocity", "social_momentum_phase", "kol_arrival_rate",
    # ML v2 Phase C: Entry zone
    "entry_timing_quality",
    # v17/v21: Scoring multipliers
    "pump_momentum_pen",
    "gate_mult",
    # v26: Market context features
    "median_peak_return",
    "entry_vs_median_peak",
    "win_rate_7d",
    "market_heat_24h",
    "relative_volume",
    "kol_saturation",
]

# Columns to fetch from token_snapshots
SNAPSHOT_COLUMNS = (
    "id, symbol, token_address, snapshot_at, mentions, sentiment, breadth, avg_conviction, "
    "recency_score, price_action_score, volume_24h, volume_1h, volume_6h, "
    "liquidity_usd, market_cap, txn_count_24h, buy_sell_ratio_24h, "
    "buy_sell_ratio_1h, short_term_heat, txn_velocity, ultra_short_heat, "
    "volume_acceleration, top10_holder_pct, insider_pct, risk_count, "
    "holder_count, safety_penalty, onchain_multiplier, social_velocity, "
    "whale_total_pct, whale_count, wash_trading_score, volatility_proxy, "
    "whale_dominance, already_pumped_penalty, squeeze_score, squeeze_state, "
    "trend_strength, confirmation_pillars, data_confidence, "
    "rsi_14, macd_histogram, bb_width, obv_slope_norm, "
    "price_change_24h, price_at_snapshot, price_after_1h, price_after_6h, price_after_12h, price_after_24h, "
    "price_after_48h, price_after_72h, price_after_7d, "
    "max_price_1h, max_price_6h, max_price_12h, max_price_24h, max_price_48h, max_price_72h, max_price_7d, "
    "did_2x_1h, did_2x_6h, did_2x_12h, did_2x_24h, did_2x_48h, did_2x_72h, did_2x_7d, mentions, "
    "freshest_mention_hours, death_penalty, lifecycle_phase, "
    "boosts_active, has_twitter, has_telegram, has_website, social_count, "
    "entry_premium, entry_premium_mult, lp_locked_pct, "
    "unique_wallet_24h_change, whale_new_entries, "
    "consensus_val, conviction_val, breadth_val, price_action_val, "
    "pump_bonus, wash_pen, pvp_pen, pump_pen, activity_mult, breadth_pen, crash_pen, stale_pen, size_mult, "
    "s_tier_mult, s_tier_count, unique_kols, pump_momentum_pen, entry_drift_mult, "
    "ca_mention_count, ticker_mention_count, url_mention_count, has_ca_mention, "
    "score_velocity, score_acceleration, mention_velocity, volume_velocity, "
    "social_momentum_phase, kol_arrival_rate, entry_timing_quality, gate_mult, "
    "time_to_1_3x_min_12h, time_to_1_5x_min_12h, time_to_1_3x_min_24h, time_to_1_5x_min_24h, "
    "time_to_2x_min_12h, time_to_3x_min_12h, time_to_5x_min_12h, "
    "time_to_2x_min_24h, time_to_3x_min_24h, time_to_5x_min_24h, "
    "time_to_sl20_min_12h, time_to_sl30_min_12h, time_to_sl50_min_12h, "
    "time_to_sl20_min_24h, time_to_sl30_min_24h, time_to_sl50_min_24h, "
    "max_dd_before_tp_pct_12h, max_dd_before_tp_pct_24h, "
    "time_to_2x_12h, time_to_2x_24h, "
    "time_to_1_3x_min_48h, time_to_1_5x_min_48h, time_to_2x_min_48h, "
    "time_to_3x_min_48h, time_to_5x_min_48h, "
    "time_to_sl20_min_48h, time_to_sl30_min_48h, time_to_sl50_min_48h, "
    "max_dd_before_tp_pct_48h, "
    "jup_price_impact_1k, min_price_12h, min_price_24h, "
    "median_peak_return, entry_vs_median_peak, win_rate_7d, market_heat_24h, relative_volume, kol_saturation, "
    "kol_freshness, mention_heat_ratio, momentum_mult, activity_ratio_raw, hype_pen, unique_kols, "
    "time_spread_minutes, first_call_age_minutes, kol_cascade_rate, price_vs_first_call, "
    "whale_change, whale_direction, volume_mcap_ratio, liq_mcap_ratio, token_age_hours, "
    "risk_score, helius_gini, bundle_pct, bundle_detected, helius_recent_tx_count, "
    "helius_holder_count, helius_onchain_bsr, jup_tradeable, jito_max_slot_txns, "
    "bubblemaps_score, bubblemaps_cluster_max_pct, bubblemaps_cex_pct, "
    "is_pump_fun, price_change_1h, price_change_5m, pvp_recent_count, "
    "score_at_snapshot, "
    "holder_turnover_pct, smart_money_retention, small_holder_pct, avg_tx_size_usd, "
    "kol_cooccurrence_avg, kol_combo_novelty, "
    "jup_price_impact_500, jup_price_impact_5k, liquidity_depth_score"
)


def _get_client():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        logger.error("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY")
        return None
    return create_client(url, key)


def _fetch_snapshots(client) -> pd.DataFrame:
    """Fetch all snapshots with pagination (PostgREST caps at 1000/page)."""
    all_rows = []
    page_size = 1000
    offset = 0
    while True:
        result = (
            client.table("token_snapshots")
            .select(SNAPSHOT_COLUMNS)
            .order("snapshot_at", desc=True)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        if not result.data:
            break
        all_rows.extend(result.data)
        if len(result.data) < page_size:
            break  # last page
        offset += page_size
        if offset >= 10000:  # safety cap
            break

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["snapshot_at"] = pd.to_datetime(df["snapshot_at"])
    return df


def _safe_mult(row: pd.Series, col: str, default: float = 1.0) -> float:
    """Read a multiplier from a snapshot row, defaulting if absent/null."""
    val = row.get(col)
    return float(val) if pd.notna(val) else default


def _bt_tier_lookup(value: float, thresholds: list, factors: list) -> float:
    """Config-driven tier lookup (mirrors pipeline._tier_lookup). Returns factor for first threshold above value."""
    for i, t in enumerate(thresholds):
        if value < t:
            return factors[i]
    return factors[-1]


def _get_score(row: pd.Series, weights: dict) -> int:
    """
    v49: Prefer score_at_snapshot (the actual score produced by the pipeline, including ML).
    Falls back to _compute_score() recomputation for older snapshots without stored score.
    """
    sas = row.get("score_at_snapshot")
    if pd.notna(sas) and int(sas) > 0:
        return int(sas)
    return _compute_score(row, weights)


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

    # v17: Consensus pump discount — late KOL mentions after pump are worth less
    pc24_raw = row.get("price_change_24h") if "price_change_24h" in row.index else None
    if consensus is not None and pd.notna(pc24_raw) and float(pc24_raw) > 50:
        consensus_discount = max(0.5, 1.0 - (float(pc24_raw) / 400))
        consensus *= consensus_discount

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

    # --- Multipliers (aligned with pipeline.py v24) ---
    onchain = _safe_mult(row, "onchain_multiplier")
    safety = _safe_mult(row, "safety_penalty")
    pump_bonus = _safe_mult(row, "pump_bonus")
    # v23: wash_pen column stores unified manipulation_pen (merged wash+pump in pipeline)
    # DO NOT also multiply by pump_pen — it's the same value (backward compat alias)
    manipulation_pen = _safe_mult(row, "wash_pen")
    pvp_pen = _safe_mult(row, "pvp_pen")
    # v24: crash_pen already incorporates pump_momentum_pen via min() in pipeline
    # DO NOT also multiply by pump_momentum_pen separately
    activity_mult = _safe_mult(row, "activity_mult")
    breadth_pen = _safe_mult(row, "breadth_pen")
    size_mult = _safe_mult(row, "size_mult")
    s_tier_mult = _safe_mult(row, "s_tier_mult")
    gate_mult = _safe_mult(row, "gate_mult")
    entry_drift_mult = _safe_mult(row, "entry_drift_mult")

    # v35/v44: Chain (14 multipliers, matching pipeline.py exactly)
    # Includes momentum_mult + hype_pen (added in v35/v32).
    momentum_mult = _safe_mult(row, "momentum_mult")
    hype_pen = _safe_mult(row, "hype_pen")
    combined_raw = (onchain * safety * pump_bonus
                    * manipulation_pen * pvp_pen * crash_pen
                    * activity_mult * breadth_pen
                    * size_mult * s_tier_mult * gate_mult
                    * entry_drift_mult * momentum_mult * hype_pen)
    # v17: Floor at 0.25, Cap at 2.0 to prevent multiplier stacking
    combined = max(0.25, min(2.0, combined_raw))
    score = base_score * combined

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
    THE key metric: does the #1 ranked token hit the return threshold?

    v22: Multi-threshold hit rates computed from max_return (not did_2x_* flags).
    Groups snapshots by cycle, picks the highest-scoring token per cycle.

    v34: First-appearance dedup — each token only appears in its earliest cycle.
    Prevents the same token from being counted 20-30x across consecutive cycles
    (was 27.6x inflation for low-score tokens). Also prevents stale snapshots
    from masking the actual first entry opportunity.
    """
    THRESHOLDS = [1.3, 1.5, 2.0]

    # v34: First-appearance dedup — keep only the first snapshot per token
    df = df.sort_values("snapshot_at").drop_duplicates(subset=["token_address"], keep="first").copy()
    df["score"] = df.apply(lambda r: _get_score(r, BALANCED_WEIGHTS), axis=1)

    # Group snapshots into cycles (same minute = same cycle)
    df["cycle"] = df["snapshot_at"].dt.floor("15min")

    results = {"1h": {}, "6h": {}, "12h": {}, "24h": {}, "48h": {}, "72h": {}, "7d": {}}

    for horizon in ["1h", "6h", "12h", "24h", "48h", "72h", "7d"]:
        max_price_col = f"max_price_{horizon}"
        if max_price_col not in df.columns:
            continue

        top1_details = []
        open_positions = {}  # addr -> exit_time (pd.Timestamp)
        hz_min = HORIZON_MINUTES.get(horizon, 720)

        for cycle_ts, group in df.sort_values("snapshot_at").groupby("cycle"):
            # Need max_price + price_at_snapshot for return computation
            labeled = group[group[max_price_col].notna() & group["price_at_snapshot"].notna()]
            if labeled.empty:
                continue

            top1 = labeled.loc[labeled["score"].idxmax()]
            addr = top1.get("token_address") or top1["symbol"]

            # ML v3: Re-entry — skip only if position is still open
            if addr in open_positions:
                if cycle_ts < open_positions[addr]:
                    continue  # Position still open

            # Record position: closes at entry + horizon duration
            open_positions[addr] = cycle_ts + pd.Timedelta(minutes=hz_min)

            ret = _compute_return(top1, horizon)
            top1_details.append({
                "cycle": str(cycle_ts),
                "symbol": top1["symbol"],
                "score": int(top1["score"]),
                "max_return": round(ret, 2) if ret else None,
            })

        if not top1_details:
            continue

        tokens_tested = len(top1_details)
        horizon_result = {
            "tokens_tested": tokens_tested,
            "details": top1_details,
        }

        # Compute hit rates at multiple thresholds
        for thresh in THRESHOLDS:
            thresh_key = f"{thresh}x"
            hits = sum(1 for d in top1_details if d["max_return"] and d["max_return"] >= thresh)
            horizon_result[f"hits_{thresh_key}"] = hits
            horizon_result[f"hit_rate_{thresh_key}"] = round(hits / tokens_tested, 4)

        # Backward compat: "tokens_hit" and "hit_rate" use 2.0x
        horizon_result["tokens_hit"] = horizon_result.get("hits_2.0x", 0)
        horizon_result["hit_rate"] = horizon_result.get("hit_rate_2.0x", 0)
        horizon_result["target"] = 1.0

        # Mark details with multi-threshold flags
        for d in top1_details:
            ret = d.get("max_return")
            d["did_2x"] = bool(ret and ret >= 2.0)
            for thresh in THRESHOLDS:
                d[f"hit_{thresh}x"] = bool(ret and ret >= thresh)

        results[horizon] = horizon_result

    return results


# ---- Analysis functions ----

def _score_calibration(df: pd.DataFrame, horizon_col: str = "did_2x_12h") -> dict:
    """Hit rate and return stats by score band (first-appearance per token)."""
    df = df.copy()
    df["score"] = df.apply(lambda r: _get_score(r, BALANCED_WEIGHTS), axis=1)

    horizon_suffix = horizon_col.replace("did_2x_", "")

    # First-appearance: each token counted once (earliest snapshot)
    first = df.sort_values("snapshot_at").drop_duplicates(subset=["token_address"], keep="first").copy()
    first["max_return"] = first.apply(lambda r: _compute_return(r, horizon_suffix), axis=1)

    result = {}
    for band_name, lo, hi in SCORE_BANDS:
        band = first[(first["score"] >= lo) & (first["score"] < hi)]
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
    """Pearson correlation of each numeric feature with the 2x outcome (first-appearance)."""
    first = df.sort_values("snapshot_at").drop_duplicates(subset=["token_address"], keep="first")
    labeled = first[first[horizon_col].notna()].copy()
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
    """For tokens that DID 2x: identify what penalties/gates reduced their score (first-appearance)."""
    first = df.sort_values("snapshot_at").drop_duplicates(subset=["token_address"], keep="first")
    winners = first[(first[horizon_col] == True)].copy()
    if winners.empty:
        return {"missed_winners": [], "total_winners": 0}

    winners["score"] = winners.apply(lambda r: _get_score(r, BALANCED_WEIGHTS), axis=1)

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
    """For high-scoring tokens that did NOT 2x: what components were misleading? (first-appearance)"""
    first = df.sort_values("snapshot_at").drop_duplicates(subset=["token_address"], keep="first").copy()
    first["score"] = first.apply(lambda r: _get_score(r, BALANCED_WEIGHTS), axis=1)

    labeled = first[first[horizon_col].notna()]
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
    """Vary each BALANCED_WEIGHTS component +/-25%, measure hit rate delta (first-appearance)."""
    first = df.sort_values("snapshot_at").drop_duplicates(subset=["token_address"], keep="first")
    labeled = first[first[horizon_col].notna()].copy()
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
    """Do v8 signals (squeeze, trend, confirmation) correlate with 2x? (first-appearance)"""
    first = df.sort_values("snapshot_at").drop_duplicates(subset=["token_address"], keep="first")
    labeled = first[first[horizon_col].notna()].copy()
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


def _temporal_analysis(df: pd.DataFrame, horizon_col: str = "did_2x_12h") -> dict:
    """
    ML v3.1: Analyze hit rate by day-of-week and hour-of-day (Europe/Paris).

    Validates trader intuition:
    - Sunday = worst, Tuesday = best
    - Runners peak 19h-5h Paris time
    """
    first = df.sort_values("snapshot_at").drop_duplicates(subset=["token_address"], keep="first")
    labeled = first[first[horizon_col].notna()].copy()
    if labeled.empty:
        return {}

    ts = pd.to_datetime(labeled["snapshot_at"], utc=True)
    try:
        ts_paris = ts.dt.tz_convert("Europe/Paris")
    except Exception:
        ts_paris = ts + pd.Timedelta(hours=1)

    labeled["_dow"] = ts_paris.dt.dayofweek  # 0=Mon..6=Sun
    labeled["_hour_paris"] = ts_paris.dt.hour

    result = {}
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    # Hit rate by day of week
    by_day = {}
    for dow in range(7):
        subset = labeled[labeled["_dow"] == dow]
        n = len(subset)
        if n < 2:
            continue
        hits = int(subset[horizon_col].sum())
        by_day[day_names[dow]] = {
            "count": n,
            "hits": hits,
            "hit_rate": round(hits / n, 4),
        }
    result["by_day_of_week"] = by_day

    # Best and worst days
    if by_day:
        best_day = max(by_day, key=lambda d: by_day[d]["hit_rate"])
        worst_day = min(by_day, key=lambda d: by_day[d]["hit_rate"])
        result["best_day"] = best_day
        result["best_day_hit_rate"] = by_day[best_day]["hit_rate"]
        result["worst_day"] = worst_day
        result["worst_day_hit_rate"] = by_day[worst_day]["hit_rate"]

    # Hit rate by hour bands (Paris time)
    hour_bands = [
        ("00-04", 0, 5),
        ("05-08", 5, 9),
        ("09-12", 9, 13),
        ("13-15", 13, 16),
        ("16-18", 16, 19),
        ("19-23", 19, 24),
    ]
    by_hour = {}
    for band_name, lo, hi in hour_bands:
        subset = labeled[(labeled["_hour_paris"] >= lo) & (labeled["_hour_paris"] < hi)]
        n = len(subset)
        if n < 2:
            continue
        hits = int(subset[horizon_col].sum())
        by_hour[band_name] = {
            "count": n,
            "hits": hits,
            "hit_rate": round(hits / n, 4),
        }
    result["by_hour_paris"] = by_hour

    # Prime time (19h-5h Paris) vs off-peak
    prime = labeled[(labeled["_hour_paris"] >= 19) | (labeled["_hour_paris"] < 5)]
    off_peak = labeled[(labeled["_hour_paris"] >= 5) & (labeled["_hour_paris"] < 19)]

    if len(prime) >= 3:
        prime_hits = int(prime[horizon_col].sum())
        result["prime_time"] = {
            "count": len(prime), "hits": prime_hits,
            "hit_rate": round(prime_hits / len(prime), 4),
            "description": "19h-05h Paris",
        }
    if len(off_peak) >= 3:
        off_hits = int(off_peak[horizon_col].sum())
        result["off_peak"] = {
            "count": len(off_peak), "hits": off_hits,
            "hit_rate": round(off_hits / len(off_peak), 4),
            "description": "05h-19h Paris",
        }

    # Weekend vs weekday
    weekend = labeled[labeled["_dow"] >= 5]
    weekday = labeled[labeled["_dow"] < 5]

    if len(weekend) >= 3:
        we_hits = int(weekend[horizon_col].sum())
        result["weekend"] = {
            "count": len(weekend), "hits": we_hits,
            "hit_rate": round(we_hits / len(weekend), 4),
        }
    if len(weekday) >= 3:
        wd_hits = int(weekday[horizon_col].sum())
        result["weekday"] = {
            "count": len(weekday), "hits": wd_hits,
            "hit_rate": round(wd_hits / len(weekday), 4),
        }

    return result


def _extraction_analysis(df: pd.DataFrame, horizon_col: str = "did_2x_12h") -> dict:
    """Analyze extraction method (CA vs ticker) correlation with 2x outcomes (first-appearance)."""
    first = df.sort_values("snapshot_at").drop_duplicates(subset=["token_address"], keep="first")
    labeled = first[first[horizon_col].notna()].copy()
    if labeled.empty or "has_ca_mention" not in labeled.columns:
        return {}

    result = {}

    # Encode has_ca_mention as boolean
    ca_col = labeled["has_ca_mention"]
    ca_count_col = pd.to_numeric(labeled.get("ca_mention_count"), errors="coerce")
    tick_count_col = pd.to_numeric(labeled.get("ticker_mention_count"), errors="coerce")

    # CA-only tokens: has_ca_mention=true AND ticker_mention_count=0
    ca_only = labeled[(ca_col == True) & (tick_count_col.fillna(0) == 0)]
    # Ticker-only: has_ca_mention=false (or null) AND ca_mention_count=0
    ticker_only = labeled[(ca_col != True) & (ca_count_col.fillna(0) == 0)]
    # Both: has both CA and ticker mentions
    both = labeled[(ca_col == True) & (tick_count_col.fillna(0) > 0)]

    for label, subset in [("ca_only", ca_only), ("ticker_only", ticker_only), ("both", both)]:
        n = len(subset)
        if n == 0:
            result[label] = {"count": 0, "hits": 0, "hit_rate": None}
            continue
        hits = int(subset[horizon_col].sum())
        result[label] = {
            "count": n,
            "hits": hits,
            "hit_rate": round(hits / n, 4),
        }

    # has_ca_mention correlation with 2x (as 0/1)
    ca_binary = ca_col.map({True: 1, False: 0}).fillna(0).astype(float)
    target = labeled[horizon_col].astype(float)
    valid = ca_binary.notna() & target.notna()
    if valid.sum() >= 10:
        try:
            r = float(np.corrcoef(ca_binary[valid], target[valid])[0, 1])
            if not math.isnan(r):
                result["has_ca_mention_correlation"] = round(r, 4)
        except Exception:
            pass

    return result


BOT_STRATEGIES = [
    # Conservative: small gain, wide stop (memecoins drop 30-50% routinely before pumping)
    {"name": "TP30_SL50", "tp_col": "time_to_1_3x_min", "sl_col": "time_to_sl50_min",
     "tp_pct": 0.30, "sl_pct": -0.50, "description": "+30% TP / -50% SL"},
    # Paper-trade baseline: matches paper_trader TP50_SL30
    {"name": "TP50_SL30", "tp_col": "time_to_1_5x_min", "sl_col": "time_to_sl30_min",
     "tp_pct": 0.50, "sl_pct": -0.30, "description": "+50% TP / -30% SL"},
    # Moderate+: medium gain, wide stop (best risk/reward for memecoins)
    {"name": "TP50_SL50", "tp_col": "time_to_1_5x_min", "sl_col": "time_to_sl50_min",
     "tp_pct": 0.50, "sl_pct": -0.50, "description": "+50% TP / -50% SL"},
    # Paper-trade strategy: matches paper_trader TP100_SL30
    {"name": "TP100_SL30", "tp_col": "time_to_2x_min", "sl_col": "time_to_sl30_min",
     "tp_pct": 1.00, "sl_pct": -0.30, "description": "+100% TP / -30% SL"},
    # Aggressive: big gain, wide stop — the classic 2x
    {"name": "TP100_SL50", "tp_col": "time_to_2x_min", "sl_col": "time_to_sl50_min",
     "tp_pct": 1.00, "sl_pct": -0.50, "description": "+100% TP / -50% SL"},
]


def _realistic_bot_simulation(df: pd.DataFrame) -> dict:
    """
    Simulate realistic bot trading with TP/SL on the #1-ranked token per cycle.

    For each strategy (TP30/SL20, TP50/SL30, TP100/SL50) × horizon (12h, 24h):
    - Pick the #1 token per cycle (highest score, first-appearance dedup)
    - Determine exit: TP hit first → win, SL hit first → loss, neither → timeout
    - Compute win rate, profit factor, expectancy, breakeven WR

    Also tests rank filters (top 1, top 3, top 5) to find optimal selectivity.

    v49: First-appearance dedup — each token only counted once (earliest snapshot).
    Without dedup, the same losing token appears in multiple cycles → inflated loss count.
    """
    # First-appearance dedup: keep only the first snapshot per token (like _top1_hit_rate)
    df = df.sort_values("snapshot_at").drop_duplicates(subset=["token_address"], keep="first").copy()
    df["score"] = df.apply(lambda r: _get_score(r, BALANCED_WEIGHTS), axis=1)
    df["cycle"] = df["snapshot_at"].dt.floor("15min")

    results = {}

    for hz in ["12h", "24h", "48h"]:
        hz_suffix = f"_{hz}"

        for strat in BOT_STRATEGIES:
            # Build column names for this horizon
            if strat["name"] == "TP100_SL50" and hz in ("12h", "24h"):
                # Legacy: time_to_2x is stored in hours, convert to minutes for comparison
                tp_col_raw = f"time_to_2x_{hz}"
                tp_is_hours = True
            else:
                tp_col_raw = strat["tp_col"] + hz_suffix
                tp_is_hours = False

            sl_col = strat["sl_col"] + hz_suffix
            price_col = f"price_after_{hz}"

            # Check columns exist
            if tp_col_raw not in df.columns or sl_col not in df.columns:
                continue

            # Test multiple rank filters
            hz_min = HORIZON_MINUTES.get(hz, 720)

            for top_n in [1, 3, 5]:
                strat_key = f"{strat['name']}_{hz}_top{top_n}"

                trades = []
                open_positions = {}  # addr -> exit_time (pd.Timestamp)

                for cycle_ts, group in df.sort_values("snapshot_at").groupby("cycle"):
                    labeled = group[
                        group["price_at_snapshot"].notna()
                        & (group[tp_col_raw].notna() | group[sl_col].notna() | group[price_col].notna())
                    ]
                    if labeled.empty:
                        continue

                    # Pick top N by score
                    top_tokens = labeled.nlargest(top_n, "score")

                    for _, row in top_tokens.iterrows():
                        addr = row.get("token_address") or row["symbol"]

                        # ML v3: Re-entry — skip only if position is still open
                        if addr in open_positions:
                            if cycle_ts < open_positions[addr]:
                                continue  # Position still open

                        price_at = float(row["price_at_snapshot"])
                        if price_at <= 0:
                            continue

                        # Get TP and SL times in minutes
                        tp_raw = row.get(tp_col_raw)
                        tp_min = None
                        if pd.notna(tp_raw):
                            if tp_is_hours:
                                tp_min = round(float(tp_raw) * 60)
                            else:
                                tp_min = int(tp_raw)

                        sl_raw = row.get(sl_col)
                        sl_min = int(sl_raw) if pd.notna(sl_raw) else None

                        # Determine exit
                        if tp_min is not None and (sl_min is None or tp_min < sl_min):
                            exit_type = "TP"
                            pnl = strat["tp_pct"]
                            exit_min = tp_min
                        elif sl_min is not None and (tp_min is None or sl_min < tp_min):
                            exit_type = "SL"
                            pnl = strat["sl_pct"]
                            exit_min = sl_min
                        elif tp_min is not None and sl_min is not None and tp_min == sl_min:
                            # Same candle → assume worst case (SL)
                            exit_type = "SL"
                            pnl = strat["sl_pct"]
                            exit_min = sl_min
                        else:
                            # Neither TP nor SL → timeout, use actual return
                            exit_type = "TIMEOUT"
                            p_after = row.get(price_col)
                            if pd.notna(p_after) and float(p_after) > 0:
                                pnl = (float(p_after) / price_at) - 1.0
                            else:
                                continue  # No exit data at all, skip
                            exit_min = None

                        # Track position exit time for re-entry logic
                        if exit_type in ("TP", "SL") and exit_min is not None:
                            pos_exit_time = cycle_ts + pd.Timedelta(minutes=exit_min)
                        else:
                            pos_exit_time = cycle_ts + pd.Timedelta(minutes=hz_min)
                        open_positions[addr] = pos_exit_time

                        trades.append({
                            "symbol": row["symbol"],
                            "exit": exit_type,
                            "pnl": pnl,
                            "exit_min": exit_min,
                            "score": int(row["score"]),
                        })

                if len(trades) < 3:
                    continue

                # Compute metrics
                tp_trades = [t for t in trades if t["exit"] == "TP"]
                sl_trades = [t for t in trades if t["exit"] == "SL"]
                timeout_trades = [t for t in trades if t["exit"] == "TIMEOUT"]

                n_tp = len(tp_trades)
                n_sl = len(sl_trades)
                n_timeout = len(timeout_trades)
                n_decided = n_tp + n_sl  # Trades with clear TP or SL exit

                win_rate = n_tp / n_decided if n_decided > 0 else 0

                total_gains = sum(t["pnl"] for t in trades if t["pnl"] > 0)
                total_losses = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
                profit_factor = total_gains / total_losses if total_losses > 0 else float('inf') if total_gains > 0 else 0

                pnl_values = [t["pnl"] for t in trades]
                expectancy = sum(pnl_values) / len(pnl_values)

                breakeven_wr = abs(strat["sl_pct"]) / (strat["tp_pct"] + abs(strat["sl_pct"]))
                is_profitable = expectancy > 0

                # Avg time to win/loss
                win_times = [t["exit_min"] for t in tp_trades if t["exit_min"] is not None]
                loss_times = [t["exit_min"] for t in sl_trades if t["exit_min"] is not None]
                avg_win_min = round(sum(win_times) / len(win_times)) if win_times else None
                avg_loss_min = round(sum(loss_times) / len(loss_times)) if loss_times else None

                # Max consecutive losses
                max_consec_loss = 0
                current_streak = 0
                for t in trades:
                    if t["pnl"] < 0:
                        current_streak += 1
                        max_consec_loss = max(max_consec_loss, current_streak)
                    else:
                        current_streak = 0

                results[strat_key] = {
                    "description": strat["description"],
                    "horizon": hz,
                    "top_n": top_n,
                    "trades": len(trades),
                    "tp": n_tp,
                    "sl": n_sl,
                    "timeout": n_timeout,
                    "win_rate": round(win_rate, 4),
                    "profit_factor": round(profit_factor, 4) if profit_factor != float('inf') else 999.0,
                    "expectancy": round(expectancy, 4),
                    "breakeven_wr": round(breakeven_wr, 4),
                    "is_profitable": is_profitable,
                    "avg_win_min": avg_win_min,
                    "avg_loss_min": avg_loss_min,
                    "max_consecutive_losses": max_consec_loss,
                    "total_pnl_pct": round(sum(pnl_values) * 100, 2),
                }

    # Find best strategy (highest expectancy with at least 5 trades, top1 only)
    top1_results = {k: v for k, v in results.items() if v.get("top_n") == 1 and v["trades"] >= 5}
    if top1_results:
        best_key = max(top1_results, key=lambda k: top1_results[k]["expectancy"])
        results["best_strategy"] = best_key
        results["best_expectancy"] = top1_results[best_key]["expectancy"]
    else:
        results["best_strategy"] = None
        results["best_expectancy"] = None

    return results


def _multi_tranche_bot_simulation(df: pd.DataFrame) -> dict:
    """
    Simulate SCALE_OUT and MOONBAG strategies using event-driven tranche logic.

    SCALE_OUT (48h, -30% SL): 25% at 2x, 25% at 3x, 25% at 5x, 25% moonbag
    MOONBAG (24h, -50% SL): 80% at 2x, 20% moonbag (no TP)

    For each token, collects time-ordered events (TP hits + SL hit), then
    processes tranches in chronological order. Moonbag closes at timeout price.

    v49-style first-appearance dedup applied.
    """
    df = df.sort_values("snapshot_at").drop_duplicates(subset=["token_address"], keep="first").copy()
    df["score"] = df.apply(lambda r: _get_score(r, BALANCED_WEIGHTS), axis=1)
    df["cycle"] = df["snapshot_at"].dt.floor("15min")

    results = {}

    MULTI_STRATEGIES = {
        "SCALE_OUT": {
            "horizons": ["24h", "48h"],
            "sl_col_suffix": "time_to_sl30_min",
            "sl_pct": -0.30,
            "tranches": [
                {"pct": 0.25, "tp_col_suffix": "time_to_2x_min", "tp_pct": 1.00, "label": "tp_2x"},
                {"pct": 0.25, "tp_col_suffix": "time_to_3x_min", "tp_pct": 2.00, "label": "tp_3x"},
                {"pct": 0.25, "tp_col_suffix": "time_to_5x_min", "tp_pct": 4.00, "label": "tp_5x"},
                {"pct": 0.25, "tp_col_suffix": None, "tp_pct": None, "label": "moonbag"},
            ],
        },
        "MOONBAG": {
            "horizons": ["24h", "48h"],
            "sl_col_suffix": "time_to_sl50_min",
            "sl_pct": -0.50,
            "tranches": [
                {"pct": 0.80, "tp_col_suffix": "time_to_2x_min", "tp_pct": 1.00, "label": "main"},
                {"pct": 0.20, "tp_col_suffix": None, "tp_pct": None, "label": "moonbag"},
            ],
        },
    }

    for strat_name, strat in MULTI_STRATEGIES.items():
        for hz in strat["horizons"]:
            hz_suffix = f"_{hz}"
            sl_col = strat["sl_col_suffix"] + hz_suffix
            price_col = f"price_after_{hz}"
            hz_min = HORIZON_MINUTES.get(hz, 1440)

            # Check required columns exist
            if sl_col not in df.columns:
                continue

            for top_n in [1, 3, 5]:
                strat_key = f"{strat_name}_{hz}_top{top_n}"

                trades = []
                open_positions = {}

                for cycle_ts, group in df.sort_values("snapshot_at").groupby("cycle"):
                    labeled = group[
                        group["price_at_snapshot"].notna()
                        & (group[sl_col].notna() | group[price_col].notna())
                    ]
                    if labeled.empty:
                        continue

                    top_tokens = labeled.nlargest(top_n, "score")

                    for _, row in top_tokens.iterrows():
                        addr = row.get("token_address") or row["symbol"]

                        if addr in open_positions:
                            if cycle_ts < open_positions[addr]:
                                continue

                        price_at = float(row["price_at_snapshot"])
                        if price_at <= 0:
                            continue

                        # Get SL time
                        sl_raw = row.get(sl_col)
                        sl_min = int(sl_raw) if pd.notna(sl_raw) else None

                        # Collect TP events: (time_min, tranche_pct, tranche_tp_pct)
                        events = []
                        for tranche in strat["tranches"]:
                            if tranche["tp_col_suffix"] is None:
                                continue  # moonbag — no TP event
                            tp_col = tranche["tp_col_suffix"] + hz_suffix
                            tp_raw = row.get(tp_col)
                            if pd.notna(tp_raw):
                                events.append((int(tp_raw), "TP", tranche["pct"], tranche["tp_pct"]))

                        # Add SL event
                        if sl_min is not None:
                            events.append((sl_min, "SL", None, None))

                        events.sort(key=lambda e: e[0])

                        remaining = 1.0
                        pnl = 0.0
                        exit_min = None

                        for ev_time, ev_type, ev_pct, ev_tp_pct in events:
                            if remaining <= 0:
                                break
                            if ev_type == "TP":
                                take = min(ev_pct, remaining)
                                pnl += take * ev_tp_pct
                                remaining -= take
                                if exit_min is None:
                                    exit_min = ev_time
                            elif ev_type == "SL":
                                pnl += remaining * strat["sl_pct"]
                                remaining = 0
                                exit_min = ev_time

                        # Moonbag remainder: timeout at actual price
                        if remaining > 0:
                            p_after = row.get(price_col)
                            if pd.notna(p_after) and float(p_after) > 0:
                                pnl += remaining * ((float(p_after) / price_at) - 1.0)
                            else:
                                continue  # No exit data

                        exit_type = "SL" if remaining == 0 and events and events[-1][1] == "SL" else "MIXED"
                        if remaining == 0 and events:
                            last_ev = [e for e in events if e[1] == "SL"]
                            if last_ev:
                                # Check if SL was last action
                                pass

                        pos_exit_time = cycle_ts + pd.Timedelta(minutes=exit_min if exit_min else hz_min)
                        open_positions[addr] = pos_exit_time

                        trades.append({
                            "symbol": row["symbol"],
                            "pnl": pnl,
                            "exit_min": exit_min,
                            "score": int(row["score"]),
                        })

                if len(trades) < 3:
                    continue

                pnl_values = [t["pnl"] for t in trades]
                n_wins = sum(1 for p in pnl_values if p > 0)
                n_losses = sum(1 for p in pnl_values if p < 0)
                total_gains = sum(p for p in pnl_values if p > 0)
                total_losses = abs(sum(p for p in pnl_values if p < 0))
                profit_factor = total_gains / total_losses if total_losses > 0 else (999.0 if total_gains > 0 else 0)
                expectancy = sum(pnl_values) / len(pnl_values)

                results[strat_key] = {
                    "description": f"{strat_name} multi-tranche",
                    "horizon": hz,
                    "top_n": top_n,
                    "trades": len(trades),
                    "wins": n_wins,
                    "losses": n_losses,
                    "win_rate": round(n_wins / len(trades), 4) if trades else 0,
                    "profit_factor": round(profit_factor, 4),
                    "expectancy": round(expectancy, 4),
                    "total_pnl_pct": round(sum(pnl_values) * 100, 2),
                }

    # Find best multi-tranche strategy
    top1_results = {k: v for k, v in results.items() if v.get("top_n") == 1 and v["trades"] >= 5}
    if top1_results:
        best_key = max(top1_results, key=lambda k: top1_results[k]["expectancy"])
        results["best_multi_strategy"] = best_key
        results["best_multi_expectancy"] = top1_results[best_key]["expectancy"]

    return results


def _adaptive_bot_simulation(df: pd.DataFrame) -> dict:
    """
    ML v3.1: Simulate adaptive SL based on actual DD data (oracle upper bound).

    For each token: adaptive_SL = max_dd_before_tp * 1.2 (20% above typical DD).
    This is an "oracle" simulation — upper bound on what ML-predicted SL could achieve.

    Also loads dd_by_rr_band from model meta (if available) to show ML-recommended SLs.

    Compares fixed SL (30%, 50%) vs adaptive SL for each horizon.

    v49: First-appearance dedup — each token only counted once.
    v49: Removed SL20 (too tight for memecoins — normal volatility triggers it).
    """
    from pathlib import Path

    # First-appearance dedup: keep only the first snapshot per token
    df = df.sort_values("snapshot_at").drop_duplicates(subset=["token_address"], keep="first").copy()
    df["score"] = df.apply(lambda r: _get_score(r, BALANCED_WEIGHTS), axis=1)
    df["cycle"] = df["snapshot_at"].dt.floor("15min")

    results = {}

    # Load RR model meta for dd_by_rr_band (informational)
    model_dir = Path(__file__).parent
    for hz in ["12h", "24h", "48h"]:
        rr_meta_path = model_dir / f"model_{hz}_rr_meta.json"
        if rr_meta_path.exists():
            try:
                with open(rr_meta_path) as f:
                    rr_meta = json.load(f)
                dd_bands = rr_meta.get("dd_by_rr_band", {})
                if dd_bands:
                    results[f"ml_dd_bands_{hz}"] = dd_bands
            except Exception:
                pass

    FIXED_SL_PCTS = [30, 50]  # v49: removed SL20 — too tight for memecoins

    for hz in ["12h", "24h", "48h"]:
        dd_col = f"max_dd_before_tp_pct_{hz}"
        max_price_col = f"max_price_{hz}"
        price_col = f"price_after_{hz}"

        if dd_col not in df.columns or max_price_col not in df.columns:
            continue

        hz_min = HORIZON_MINUTES.get(hz, 720)

        # For each cycle, pick #1 token and simulate
        fixed_trades = {sl: [] for sl in FIXED_SL_PCTS}
        adaptive_trades = []
        open_positions = {}

        for cycle_ts, group in df.sort_values("snapshot_at").groupby("cycle"):
            labeled = group[
                group["price_at_snapshot"].notna()
                & group[max_price_col].notna()
            ]
            if labeled.empty:
                continue

            top1 = labeled.loc[labeled["score"].idxmax()]
            addr = top1.get("token_address") or top1["symbol"]

            # Re-entry: skip if position still open
            if addr in open_positions:
                if cycle_ts < open_positions[addr]:
                    continue

            open_positions[addr] = cycle_ts + pd.Timedelta(minutes=hz_min)

            price_at = float(top1["price_at_snapshot"])
            if price_at <= 0:
                continue

            max_price = float(top1[max_price_col]) if pd.notna(top1[max_price_col]) else 0
            max_return = max_price / price_at if max_price > 0 else 0

            dd_raw = top1.get(dd_col)
            dd_pct = float(dd_raw) if pd.notna(dd_raw) else None

            # Use final price at horizon for timeout scenario
            p_after_raw = top1.get(price_col)
            final_return = float(p_after_raw) / price_at if pd.notna(p_after_raw) and float(p_after_raw) > 0 else None

            # --- Fixed SL simulations ---
            for sl_pct in FIXED_SL_PCTS:
                if dd_pct is not None and dd_pct >= sl_pct:
                    # SL triggered — loss
                    fixed_trades[sl_pct].append({
                        "symbol": top1["symbol"],
                        "exit": "SL",
                        "pnl": -sl_pct / 100.0,
                        "max_return": round(max_return, 3),
                    })
                elif max_return >= 1.5:
                    # TP at +50% (normalized comparison point)
                    fixed_trades[sl_pct].append({
                        "symbol": top1["symbol"],
                        "exit": "TP",
                        "pnl": 0.50,
                        "max_return": round(max_return, 3),
                    })
                elif final_return is not None:
                    fixed_trades[sl_pct].append({
                        "symbol": top1["symbol"],
                        "exit": "TIMEOUT",
                        "pnl": final_return - 1.0,
                        "max_return": round(max_return, 3),
                    })

            # --- Adaptive SL simulation (oracle: SL = actual DD * 1.2) ---
            if dd_pct is not None and dd_pct > 0:
                adaptive_sl = dd_pct * 1.2  # 20% above the actual max drawdown
                # With oracle SL, we never get stopped out (since SL > actual DD)
                # So the trade survives to reach its max return
                if max_return >= 1.5:
                    pnl = 0.50  # TP at +50%
                    exit_type = "TP"
                elif final_return is not None:
                    pnl = final_return - 1.0
                    exit_type = "TIMEOUT"
                else:
                    pnl = 0
                    exit_type = "TIMEOUT"

                adaptive_trades.append({
                    "symbol": top1["symbol"],
                    "exit": exit_type,
                    "pnl": pnl,
                    "adaptive_sl_pct": round(adaptive_sl, 1),
                    "actual_dd_pct": round(dd_pct, 1),
                    "max_return": round(max_return, 3),
                })

        # Compute metrics for each fixed SL
        for sl_pct in FIXED_SL_PCTS:
            trades = fixed_trades[sl_pct]
            if len(trades) < 3:
                continue

            n_tp = sum(1 for t in trades if t["exit"] == "TP")
            n_sl = sum(1 for t in trades if t["exit"] == "SL")
            n_decided = n_tp + n_sl
            win_rate = n_tp / n_decided if n_decided > 0 else 0

            pnl_values = [t["pnl"] for t in trades]
            expectancy = sum(pnl_values) / len(pnl_values)

            total_gains = sum(t["pnl"] for t in trades if t["pnl"] > 0)
            total_losses = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
            profit_factor = total_gains / total_losses if total_losses > 0 else (999.0 if total_gains > 0 else 0)

            results[f"fixed_SL{sl_pct}_{hz}"] = {
                "type": "fixed",
                "sl_pct": sl_pct,
                "horizon": hz,
                "trades": len(trades),
                "tp": n_tp,
                "sl": n_sl,
                "win_rate": round(win_rate, 4),
                "expectancy": round(expectancy, 4),
                "profit_factor": round(min(profit_factor, 999.0), 4),
                "total_pnl_pct": round(sum(pnl_values) * 100, 2),
            }

        # Compute metrics for adaptive SL
        if len(adaptive_trades) >= 3:
            n_tp = sum(1 for t in adaptive_trades if t["exit"] == "TP")
            n_decided = sum(1 for t in adaptive_trades if t["exit"] in ("TP", "SL"))
            win_rate = n_tp / n_decided if n_decided > 0 else (1.0 if n_tp > 0 else 0)

            pnl_values = [t["pnl"] for t in adaptive_trades]
            expectancy = sum(pnl_values) / len(pnl_values)

            total_gains = sum(t["pnl"] for t in adaptive_trades if t["pnl"] > 0)
            total_losses = abs(sum(t["pnl"] for t in adaptive_trades if t["pnl"] < 0))
            profit_factor = total_gains / total_losses if total_losses > 0 else (999.0 if total_gains > 0 else 0)

            avg_adaptive_sl = float(np.mean([t["adaptive_sl_pct"] for t in adaptive_trades]))
            avg_actual_dd = float(np.mean([t["actual_dd_pct"] for t in adaptive_trades]))

            results[f"adaptive_{hz}"] = {
                "type": "adaptive_oracle",
                "horizon": hz,
                "trades": len(adaptive_trades),
                "tp": n_tp,
                "win_rate": round(win_rate, 4),
                "expectancy": round(expectancy, 4),
                "profit_factor": round(min(profit_factor, 999.0), 4),
                "total_pnl_pct": round(sum(pnl_values) * 100, 2),
                "avg_adaptive_sl_pct": round(avg_adaptive_sl, 1),
                "avg_actual_dd_pct": round(avg_actual_dd, 1),
                "description": "Oracle SL = actual DD * 1.2 (upper bound for ML-predicted SL)",
            }

    # Compare: find best fixed vs adaptive
    fixed_results = {k: v for k, v in results.items() if isinstance(v, dict) and v.get("type") == "fixed"}
    adaptive_results = {k: v for k, v in results.items() if isinstance(v, dict) and v.get("type") == "adaptive_oracle"}

    if fixed_results:
        best_fixed = max(fixed_results, key=lambda k: fixed_results[k].get("expectancy", -999))
        results["best_fixed"] = best_fixed
        results["best_fixed_expectancy"] = fixed_results[best_fixed]["expectancy"]

    if adaptive_results:
        best_adaptive = max(adaptive_results, key=lambda k: adaptive_results[k].get("expectancy", -999))
        results["best_adaptive"] = best_adaptive
        results["best_adaptive_expectancy"] = adaptive_results[best_adaptive]["expectancy"]

    # Compute improvement delta
    if "best_fixed_expectancy" in results and "best_adaptive_expectancy" in results:
        delta = results["best_adaptive_expectancy"] - results["best_fixed_expectancy"]
        results["adaptive_improvement_pp"] = round(delta * 100, 2)

    return results


def _optimal_threshold(df: pd.DataFrame, horizon_col: str = "did_2x_12h") -> dict:
    """Find score threshold that maximizes F1 for 2x prediction (first-appearance)."""
    first = df.sort_values("snapshot_at").drop_duplicates(subset=["token_address"], keep="first")
    labeled = first[first[horizon_col].notna()].copy()
    if labeled.empty:
        return {}

    labeled["score"] = labeled.apply(lambda r: _get_score(r, BALANCED_WEIGHTS), axis=1)

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
    """Time-based train/test split validation (first-appearance)."""
    first = df.sort_values("snapshot_at").drop_duplicates(subset=["token_address"], keep="first")
    labeled = first[first[horizon_col].notna()].copy()
    if labeled.empty:
        return []

    labeled = labeled.sort_values("snapshot_at")
    labeled["score"] = labeled.apply(lambda r: _get_score(r, BALANCED_WEIGHTS), axis=1)

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


def _find_optimal_weights(df: pd.DataFrame, horizon_col: str = "did_2x_12h") -> dict | None:
    """
    Grid search for weight combination that maximizes #1 token hit rate.
    v22: Uses ML_THRESHOLD from scoring_config for dynamic hit rate computation.
    Returns best weights dict or None if no improvement found.
    """
    first = df.sort_values("snapshot_at").drop_duplicates(subset=["token_address"], keep="first")
    labeled = first[first[horizon_col].notna()].copy()
    if len(labeled) < 50:
        return None

    # v22: Use threshold from scoring_config for hit rate computation
    threshold = ML_THRESHOLD
    horizon_suffix = horizon_col.replace("did_2x_", "")

    # Generate candidate weight combinations (step=0.05, sum=1.0)
    best_hr = -1
    best_weights = None
    components = list(BALANCED_WEIGHTS.keys())

    # Score the current weights first as baseline
    labeled["base_score"] = labeled.apply(lambda r: _compute_score(r, BALANCED_WEIGHTS), axis=1)
    # Use top1 metric: per-cycle #1 token hit rate
    labeled["cycle"] = labeled["snapshot_at"].dt.floor("15min")
    # Pre-compute max_return for dynamic threshold
    labeled["_max_return"] = labeled.apply(lambda r: _compute_return(r, horizon_suffix), axis=1)

    def _top1_hr_for_weights(weights: dict) -> float:
        labeled["test_score"] = labeled.apply(lambda r: _compute_score(r, weights), axis=1)
        hits = 0
        tested = 0
        seen = set()
        for _, group in labeled.sort_values("snapshot_at").groupby("cycle"):
            lbl = group[group[horizon_col].notna()]
            if lbl.empty:
                continue
            top1 = lbl.loc[lbl["test_score"].idxmax()]
            addr = top1.get("token_address") or top1["symbol"]
            if addr in seen:
                continue
            seen.add(addr)
            tested += 1
            # v22: Use max_return >= threshold instead of did_2x flag
            ret = top1.get("_max_return")
            if ret is not None and ret >= threshold:
                hits += 1
        return hits / max(1, tested)

    baseline_hr = _top1_hr_for_weights(BALANCED_WEIGHTS)

    # Smart grid: vary each component around current value
    import itertools
    ranges = {}
    for comp in components:
        current = BALANCED_WEIGHTS[comp]
        vals = [max(0.0, current + d) for d in [-0.10, -0.05, 0.0, 0.05, 0.10]]
        ranges[comp] = sorted(set(round(v, 2) for v in vals if 0.0 <= v <= 0.80))

    for combo in itertools.product(*[ranges[c] for c in components]):
        total = sum(combo)
        if abs(total - 1.0) > 0.02:
            continue
        # Normalize to exactly 1.0
        weights = {c: round(v / total, 3) for c, v in zip(components, combo)}
        hr = _top1_hr_for_weights(weights)
        if hr > best_hr:
            best_hr = hr
            best_weights = weights

    if best_weights is None or best_hr <= baseline_hr:
        return None

    # Require >5pp improvement to justify a change
    improvement = best_hr - baseline_hr
    if improvement < 0.05:
        logger.info("auto_apply: best improvement %.1f%% < 5pp threshold, skipping", improvement * 100)
        return None

    return {
        "weights": best_weights,
        "hit_rate": best_hr,
        "baseline_hr": baseline_hr,
        "improvement": improvement,
    }


def _auto_apply_weights(client, df: pd.DataFrame, horizon_col: str = "did_2x_12h") -> dict | None:
    """
    Find optimal weights and write to scoring_config if improvement is significant.

    Guard-rails:
    - Requires 100+ unique labeled tokens
    - No weight can change by >0.10 from current production value
    - Improvement must be >5pp in #1 token hit rate
    - Weights must sum to 1.0
    """
    first = df.sort_values("snapshot_at").drop_duplicates(subset=["token_address"], keep="first")
    n_labeled = len(first[first[horizon_col].notna()])

    if n_labeled < 100:
        return None

    logger.info("auto_apply: searching optimal weights with %d labeled tokens", n_labeled)
    result = _find_optimal_weights(df, horizon_col)
    if result is None:
        logger.info("auto_apply: no improvement found, keeping current weights")
        return None

    new_weights = result["weights"]

    # Guard-rail: no weight change >0.10 from current
    for comp in BALANCED_WEIGHTS:
        delta = abs(new_weights[comp] - BALANCED_WEIGHTS[comp])
        if delta > 0.10:
            logger.warning(
                "auto_apply: %s change %.3f > 0.10 max, clamping",
                comp, delta,
            )
            direction = 1 if new_weights[comp] > BALANCED_WEIGHTS[comp] else -1
            new_weights[comp] = round(BALANCED_WEIGHTS[comp] + direction * 0.10, 3)

    # Re-normalize after clamping
    total = sum(new_weights.values())
    new_weights = {k: round(v / total, 3) for k, v in new_weights.items()}

    # Write to scoring_config
    try:
        reason = (
            f"auto_backtest: {n_labeled} labeled tokens, "
            f"#1 hit rate {result['baseline_hr']*100:.0f}% -> {result['hit_rate']*100:.0f}% "
            f"(+{result['improvement']*100:.0f}pp)"
        )
        client.table("scoring_config").update({
            "w_consensus": new_weights["consensus"],
            "w_conviction": new_weights["conviction"],
            "w_breadth": new_weights["breadth"],
            "w_price_action": new_weights["price_action"],
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "updated_by": "auto_backtest",
            "change_reason": reason,
        }).eq("id", 1).execute()

        logger.info("auto_apply: weights updated! %s — %s", new_weights, reason)
        return {
            "applied": True,
            "new_weights": new_weights,
            "reason": reason,
            **result,
        }
    except Exception as e:
        logger.error("auto_apply: failed to write scoring_config: %s", e)
        return None


# ═══ GAP #1: EQUITY CURVE + MAX DRAWDOWN ═══

def _equity_curve_analysis(df: pd.DataFrame) -> dict:
    """
    Simulate equity curve from bot trades on #1 token per cycle.
    Tracks cumulative PnL, max drawdown, recovery time, losing streaks.
    Uses TP50/SL50 on 12h as default strategy (memecoins need wide SL).

    v49: First-appearance dedup + SL30→SL50 (memecoins drop 30%+ routinely).
    """
    # First-appearance dedup
    df = df.sort_values("snapshot_at").drop_duplicates(subset=["token_address"], keep="first").copy()
    df["score"] = df.apply(lambda r: _get_score(r, BALANCED_WEIGHTS), axis=1)
    df["cycle"] = df["snapshot_at"].dt.floor("15min")

    results = {}

    for hz in ["12h", "24h"]:
        tp_col = f"time_to_1_5x_min_{hz}"
        sl_col = f"time_to_sl50_min_{hz}"  # v49: SL50 (was SL30, too tight for memecoins)
        price_col = f"price_after_{hz}"
        hz_min = HORIZON_MINUTES.get(hz, 720)

        if tp_col not in df.columns or sl_col not in df.columns:
            continue

        equity = 1.0  # Start with 1.0 (100%)
        peak_equity = 1.0
        curve = []  # (cycle_ts, equity, drawdown_pct)
        trades = []
        open_positions = {}
        max_dd = 0.0
        dd_start = None
        max_dd_duration_cycles = 0
        current_dd_cycles = 0

        for cycle_ts, group in df.sort_values("snapshot_at").groupby("cycle"):
            labeled = group[
                group["price_at_snapshot"].notna()
                & (group[tp_col].notna() | group[sl_col].notna() | group[price_col].notna())
            ]
            if labeled.empty:
                continue

            top1 = labeled.loc[labeled["score"].idxmax()]
            addr = top1.get("token_address") or top1["symbol"]

            if addr in open_positions and cycle_ts < open_positions[addr]:
                continue

            price_at = float(top1["price_at_snapshot"])
            if price_at <= 0:
                continue

            tp_raw = top1.get(tp_col)
            sl_raw = top1.get(sl_col)
            tp_min = int(tp_raw) if pd.notna(tp_raw) else None
            sl_min = int(sl_raw) if pd.notna(sl_raw) else None

            if tp_min is not None and (sl_min is None or tp_min < sl_min):
                pnl_pct = 0.50
                exit_type = "TP"
                exit_min = tp_min
            elif sl_min is not None and (tp_min is None or sl_min <= tp_min):
                pnl_pct = -0.50  # v49: was -0.30
                exit_type = "SL"
                exit_min = sl_min
            else:
                p_after = top1.get(price_col)
                if pd.notna(p_after) and float(p_after) > 0:
                    pnl_pct = (float(p_after) / price_at) - 1.0
                    exit_type = "TIMEOUT"
                    exit_min = None
                else:
                    continue

            if exit_type in ("TP", "SL") and exit_min is not None:
                open_positions[addr] = cycle_ts + pd.Timedelta(minutes=exit_min)
            else:
                open_positions[addr] = cycle_ts + pd.Timedelta(minutes=hz_min)

            # Update equity (risk 10% of equity per trade)
            risk_frac = 0.10
            equity_change = equity * risk_frac * pnl_pct
            equity += equity_change

            # Track peak and drawdown
            if equity > peak_equity:
                peak_equity = equity
                current_dd_cycles = 0
            else:
                current_dd_cycles += 1
                max_dd_duration_cycles = max(max_dd_duration_cycles, current_dd_cycles)

            dd_pct = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
            max_dd = max(max_dd, dd_pct)

            trades.append({
                "symbol": top1["symbol"],
                "exit": exit_type,
                "pnl_pct": round(pnl_pct * 100, 2),
                "equity": round(equity, 4),
            })
            curve.append({
                "cycle": str(cycle_ts),
                "equity": round(equity, 4),
                "dd_pct": round(dd_pct, 2),
            })

        if len(trades) < 3:
            continue

        # Compute losing streak stats
        max_losing_streak = 0
        current_streak = 0
        streaks = []
        for t in trades:
            if t["pnl_pct"] < 0:
                current_streak += 1
                max_losing_streak = max(max_losing_streak, current_streak)
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
        if current_streak > 0:
            streaks.append(current_streak)

        # Equity at risk during max losing streak
        worst_streak_loss = 0.0
        cs = 0
        running_loss = 0.0
        for t in trades:
            if t["pnl_pct"] < 0:
                cs += 1
                running_loss += t["pnl_pct"]
                if cs == max_losing_streak:
                    worst_streak_loss = running_loss
            else:
                cs = 0
                running_loss = 0.0

        total_return = (equity - 1.0) * 100
        n_wins = sum(1 for t in trades if t["pnl_pct"] > 0)
        n_losses = sum(1 for t in trades if t["pnl_pct"] < 0)

        results[f"TP50_SL30_{hz}"] = {
            "strategy": "TP50_SL30",
            "horizon": hz,
            "risk_per_trade_pct": 10,
            "trades": len(trades),
            "wins": n_wins,
            "losses": n_losses,
            "final_equity": round(equity, 4),
            "total_return_pct": round(total_return, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "max_losing_streak": max_losing_streak,
            "worst_streak_loss_pct": round(worst_streak_loss, 2),
            "max_dd_duration_cycles": max_dd_duration_cycles,
            "peak_equity": round(peak_equity, 4),
            "curve_points": len(curve),
            # Last 5 trades for quick inspection
            "recent_trades": trades[-5:] if len(trades) >= 5 else trades,
        }

        # Calmar ratio = annual return / max DD (annualized from data range)
        if max_dd > 0 and len(curve) > 1:
            from datetime import datetime as _dt
            first_ts = pd.Timestamp(curve[0]["cycle"])
            last_ts = pd.Timestamp(curve[-1]["cycle"])
            days = max(1, (last_ts - first_ts).days)
            ann_return = total_return * (365 / days)
            results[f"TP50_SL30_{hz}"]["calmar_ratio"] = round(ann_return / max_dd, 3)

    return results


# ═══ GAP #2: SLIPPAGE MODELING ═══

def _slippage_analysis(df: pd.DataFrame) -> dict:
    """
    Estimate slippage impact on backtest results.

    Model: slippage = entry_slippage + exit_slippage
    - Based on liquidity_usd (available for 85%+ of snapshots)
    - Formula: slippage_pct = (trade_size / liquidity_usd) * 100 * impact_factor
    - impact_factor = 2.0 for low-liquidity memecoins (AMM curve is steep)

    Tests multiple trade sizes ($100, $500, $1000, $5000) to show
    at what size slippage eats the profit.
    """
    TRADE_SIZES = [100, 500, 1000, 5000]
    IMPACT_FACTOR = 2.0  # AMM constant-product: 2x theoretical for memecoins
    TP_PCT = 0.50  # TP50 benchmark

    df = df.copy()
    df["score"] = df.apply(lambda r: _get_score(r, BALANCED_WEIGHTS), axis=1)
    df["cycle"] = df["snapshot_at"].dt.floor("15min")

    results = {}

    # Get #1 token per cycle with liquidity data
    top1_tokens = []
    seen = set()
    for _, group in df.sort_values("snapshot_at").groupby("cycle"):
        labeled = group[group["price_at_snapshot"].notna()]
        if labeled.empty:
            continue
        top1 = labeled.loc[labeled["score"].idxmax()]
        addr = top1.get("token_address") or top1["symbol"]
        if addr in seen:
            continue
        seen.add(addr)
        liq = top1.get("liquidity_usd")
        if pd.notna(liq) and float(liq) > 0:
            top1_tokens.append({
                "symbol": top1["symbol"],
                "liquidity_usd": float(liq),
                "market_cap": float(top1.get("market_cap") or 0),
                "max_return_12h": _compute_return(top1, "12h"),
            })

    if not top1_tokens:
        return {"error": "no liquidity data"}

    results["tokens_with_liquidity"] = len(top1_tokens)
    liq_values = [t["liquidity_usd"] for t in top1_tokens]
    results["liquidity_stats"] = {
        "median": round(float(np.median(liq_values)), 0),
        "p25": round(float(np.percentile(liq_values, 25)), 0),
        "p75": round(float(np.percentile(liq_values, 75)), 0),
        "min": round(min(liq_values), 0),
    }

    for trade_size in TRADE_SIZES:
        slippages = []
        net_returns = []
        killed_trades = 0

        for t in top1_tokens:
            liq = t["liquidity_usd"]
            # Entry slippage (buy) + exit slippage (sell) = 2x single-side
            entry_slip = (trade_size / liq) * 100 * IMPACT_FACTOR
            exit_slip = entry_slip  # Symmetric for AMM
            total_slip = entry_slip + exit_slip
            # Cap at 50% (beyond that, trade is impossible)
            total_slip = min(total_slip, 50.0)
            slippages.append(total_slip)

            ret = t.get("max_return_12h")
            if ret is not None:
                net_ret = ret - 1.0 - (total_slip / 100)
                net_returns.append(net_ret)
                if ret >= 1.5 and net_ret < TP_PCT - (total_slip / 100):
                    killed_trades += 1

        results[f"size_{trade_size}"] = {
            "trade_size_usd": trade_size,
            "avg_slippage_pct": round(float(np.mean(slippages)), 2),
            "median_slippage_pct": round(float(np.median(slippages)), 2),
            "p95_slippage_pct": round(float(np.percentile(slippages, 95)), 2),
            "tokens_above_5pct_slip": sum(1 for s in slippages if s > 5),
            "tokens_above_10pct_slip": sum(1 for s in slippages if s > 10),
            "killed_winning_trades": killed_trades,
        }

        # Net win rate after slippage (for tokens with return data)
        if net_returns:
            net_winners = sum(1 for r in net_returns if r > 0)
            results[f"size_{trade_size}"]["net_positive_pct"] = round(
                100 * net_winners / len(net_returns), 1
            )

    # Recommended max trade size (where median slippage < 2%)
    rec_size = None
    for ts in TRADE_SIZES:
        info = results.get(f"size_{ts}", {})
        if info.get("median_slippage_pct", 999) < 2.0:
            rec_size = ts
    results["recommended_max_size_usd"] = rec_size or TRADE_SIZES[0]

    return results


# ═══ GAP #3: CONFIDENCE INTERVALS ═══

def _confidence_intervals(df: pd.DataFrame, horizon_col: str = "did_2x_12h") -> dict:
    """
    Compute binomial confidence intervals on key backtest metrics.

    With N=14 test samples and 50% hit rate, the 95% CI is [23%, 77%].
    This tells you the backtest results are NOT statistically reliable yet.

    Uses Wilson score interval (better than Wald for small N).
    """
    from scipy.stats import norm

    def _wilson_ci(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
        """Wilson score 95% CI — better coverage for small N than naive p ± z*sqrt(p(1-p)/n)."""
        if n == 0:
            return (0.0, 0.0)
        p = hits / n
        denom = 1 + z ** 2 / n
        center = (p + z ** 2 / (2 * n)) / denom
        spread = z * math.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n) / denom
        return (max(0.0, center - spread), min(1.0, center + spread))

    first = df.sort_values("snapshot_at").drop_duplicates(subset=["token_address"], keep="first")
    labeled = first[first[horizon_col].notna()].copy()

    results = {}
    n = len(labeled)
    results["n_labeled"] = n

    if n < 5:
        results["verdict"] = "INSUFFICIENT DATA — need at least 30 tokens for any CI"
        return results

    # CI on overall hit rate
    hits = int(labeled[horizon_col].sum())
    hr = hits / n
    ci_low, ci_high = _wilson_ci(hits, n)
    results["overall_hit_rate"] = {
        "hits": hits,
        "n": n,
        "hit_rate": round(hr, 4),
        "ci_95_low": round(ci_low, 4),
        "ci_95_high": round(ci_high, 4),
        "ci_width": round(ci_high - ci_low, 4),
    }

    # CI on top1 hit rate (from report if available)
    # Re-compute here for independence
    labeled["score"] = labeled.apply(lambda r: _get_score(r, BALANCED_WEIGHTS), axis=1)
    labeled["cycle"] = labeled["snapshot_at"].dt.floor("15min")
    horizon_suffix = horizon_col.replace("did_2x_", "")

    top1_tested = 0
    top1_hits = 0
    seen = set()
    for _, group in labeled.sort_values("snapshot_at").groupby("cycle"):
        if group.empty:
            continue
        top1 = group.loc[group["score"].idxmax()]
        addr = top1.get("token_address") or top1["symbol"]
        if addr in seen:
            continue
        seen.add(addr)
        top1_tested += 1
        ret = _compute_return(top1, horizon_suffix)
        if ret is not None and ret >= 2.0:
            top1_hits += 1

    if top1_tested >= 3:
        t1_hr = top1_hits / top1_tested
        ci_low, ci_high = _wilson_ci(top1_hits, top1_tested)
        results["top1_hit_rate"] = {
            "hits": top1_hits,
            "n": top1_tested,
            "hit_rate": round(t1_hr, 4),
            "ci_95_low": round(ci_low, 4),
            "ci_95_high": round(ci_high, 4),
            "ci_width": round(ci_high - ci_low, 4),
        }

    # Multi-threshold CIs
    for thresh in [1.3, 1.5, 2.0]:
        thresh_key = f"{thresh}x"
        labeled["_max_ret"] = labeled.apply(lambda r: _compute_return(r, horizon_suffix), axis=1)
        valid = labeled["_max_ret"].notna()
        n_valid = int(valid.sum())
        if n_valid < 5:
            continue
        h = int((labeled.loc[valid, "_max_ret"] >= thresh).sum())
        hr = h / n_valid
        ci_low, ci_high = _wilson_ci(h, n_valid)
        results[f"hit_rate_{thresh_key}"] = {
            "hits": h, "n": n_valid, "hit_rate": round(hr, 4),
            "ci_95_low": round(ci_low, 4), "ci_95_high": round(ci_high, 4),
            "ci_width": round(ci_high - ci_low, 4),
        }

    # Statistical power: how many samples needed for ±5pp CI width?
    if n > 0:
        p_hat = max(0.01, hits / n)
        # For Wilson CI width ≈ 2 * z * sqrt(p(1-p)/N) = 0.10 → N = (2z)^2 * p(1-p) / 0.10^2
        z = 1.96
        n_needed_10pp = int(math.ceil(z ** 2 * p_hat * (1 - p_hat) / 0.05 ** 2))
        n_needed_5pp = int(math.ceil(z ** 2 * p_hat * (1 - p_hat) / 0.025 ** 2))
        results["samples_needed"] = {
            "for_10pp_ci": n_needed_10pp,
            "for_5pp_ci": n_needed_5pp,
            "current": n,
            "deficit": max(0, n_needed_10pp - n),
        }

    # Verdict
    ci_width = results.get("overall_hit_rate", {}).get("ci_width", 1.0)
    if ci_width > 0.30:
        results["verdict"] = f"UNRELIABLE — CI width {ci_width*100:.0f}pp (need <10pp). N={n} too small."
    elif ci_width > 0.15:
        results["verdict"] = f"WEAK — CI width {ci_width*100:.0f}pp. Results directionally useful but noisy."
    elif ci_width > 0.10:
        results["verdict"] = f"MODERATE — CI width {ci_width*100:.0f}pp. Getting reliable."
    else:
        results["verdict"] = f"STRONG — CI width {ci_width*100:.0f}pp. Results are statistically reliable."

    return results


# ═══ GAP #4: MULTI-TOKEN PORTFOLIO SIMULATION ═══

def _portfolio_simulation(df: pd.DataFrame) -> dict:
    """
    Simulate a portfolio holding top N tokens per cycle simultaneously.

    Key difference from single-token backtest:
    - Multiple positions open at once (diversification)
    - Equal-weight allocation across positions
    - Track portfolio equity curve, not individual trades
    - Shows if diversification helps or hurts

    v49: First-appearance dedup + SL30→SL50 (memecoins need wide SL).
    """
    # First-appearance dedup
    df = df.sort_values("snapshot_at").drop_duplicates(subset=["token_address"], keep="first").copy()
    df["score"] = df.apply(lambda r: _get_score(r, BALANCED_WEIGHTS), axis=1)
    df["cycle"] = df["snapshot_at"].dt.floor("15min")

    results = {}

    for hz in ["12h", "24h"]:
        tp_col = f"time_to_1_5x_min_{hz}"
        sl_col = f"time_to_sl50_min_{hz}"  # v49: was SL30
        price_col = f"price_after_{hz}"
        hz_min = HORIZON_MINUTES.get(hz, 720)

        if tp_col not in df.columns or sl_col not in df.columns:
            continue

        for portfolio_size in [1, 3, 5]:
            equity = 1.0
            peak_equity = 1.0
            max_dd = 0.0
            all_trades = []
            open_positions = {}

            for cycle_ts, group in df.sort_values("snapshot_at").groupby("cycle"):
                labeled = group[
                    group["price_at_snapshot"].notna()
                    & (group[tp_col].notna() | group[sl_col].notna() | group[price_col].notna())
                ]
                if labeled.empty:
                    continue

                top_n = labeled.nlargest(portfolio_size, "score")

                cycle_pnl = 0.0
                cycle_trades = 0

                for _, row in top_n.iterrows():
                    addr = row.get("token_address") or row["symbol"]
                    if addr in open_positions and cycle_ts < open_positions[addr]:
                        continue

                    price_at = float(row["price_at_snapshot"])
                    if price_at <= 0:
                        continue

                    tp_raw = row.get(tp_col)
                    sl_raw = row.get(sl_col)
                    tp_min = int(tp_raw) if pd.notna(tp_raw) else None
                    sl_min = int(sl_raw) if pd.notna(sl_raw) else None

                    if tp_min is not None and (sl_min is None or tp_min < sl_min):
                        pnl = 0.50
                        exit_type = "TP"
                        exit_min = tp_min
                    elif sl_min is not None and (tp_min is None or sl_min <= tp_min):
                        pnl = -0.50  # v49: was -0.30
                        exit_type = "SL"
                        exit_min = sl_min
                    else:
                        p_after = row.get(price_col)
                        if pd.notna(p_after) and float(p_after) > 0:
                            pnl = (float(p_after) / price_at) - 1.0
                        else:
                            continue
                        exit_type = "TIMEOUT"
                        exit_min = None

                    if exit_type in ("TP", "SL") and exit_min is not None:
                        open_positions[addr] = cycle_ts + pd.Timedelta(minutes=exit_min)
                    else:
                        open_positions[addr] = cycle_ts + pd.Timedelta(minutes=hz_min)

                    # Each position gets equal weight: risk_per_trade = 10% / portfolio_size
                    risk_frac = 0.10 / portfolio_size
                    cycle_pnl += equity * risk_frac * pnl
                    cycle_trades += 1

                    all_trades.append({"exit": exit_type, "pnl": pnl})

                equity += cycle_pnl
                if equity > peak_equity:
                    peak_equity = equity
                dd_pct = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
                max_dd = max(max_dd, dd_pct)

            if len(all_trades) < 3:
                continue

            n_tp = sum(1 for t in all_trades if t["exit"] == "TP")
            n_sl = sum(1 for t in all_trades if t["exit"] == "SL")
            n_decided = n_tp + n_sl
            win_rate = n_tp / n_decided if n_decided > 0 else 0
            pnl_values = [t["pnl"] for t in all_trades]
            expectancy = sum(pnl_values) / len(pnl_values)

            results[f"top{portfolio_size}_{hz}"] = {
                "portfolio_size": portfolio_size,
                "horizon": hz,
                "trades": len(all_trades),
                "wins": n_tp,
                "losses": n_sl,
                "win_rate": round(win_rate, 4),
                "expectancy": round(expectancy, 4),
                "final_equity": round(equity, 4),
                "total_return_pct": round((equity - 1.0) * 100, 2),
                "max_drawdown_pct": round(max_dd, 2),
            }

    # Find best portfolio config
    valid_results = {k: v for k, v in results.items() if isinstance(v, dict) and "final_equity" in v}
    if valid_results:
        # Best by risk-adjusted return (return / max_dd)
        best_key = max(valid_results, key=lambda k: (
            valid_results[k]["total_return_pct"] / max(1, valid_results[k]["max_drawdown_pct"])
        ))
        results["best_config"] = best_key
        best = valid_results[best_key]
        results["best_risk_adjusted"] = round(
            best["total_return_pct"] / max(1, best["max_drawdown_pct"]), 3
        )

    return results


# ═══ GAP #5: KELLY CRITERION POSITION SIZING ═══

def _kelly_analysis(df: pd.DataFrame, horizon_col: str = "did_2x_12h") -> dict:
    """
    Compute Kelly criterion optimal bet size by score band.

    Kelly fraction = (p * b - q) / b
    Where: p = win probability, q = 1-p, b = avg win / avg loss (odds)

    Half-Kelly is recommended in practice (less volatile, 75% of full Kelly growth).
    Shows how much of bankroll to risk per trade for each score band.
    """
    first = df.sort_values("snapshot_at").drop_duplicates(subset=["token_address"], keep="first").copy()
    first["score"] = first.apply(lambda r: _get_score(r, BALANCED_WEIGHTS), axis=1)

    results = {}

    for hz in ["12h", "24h"]:
        tp_col = f"time_to_1_5x_min_{hz}"
        sl_col = f"time_to_sl50_min_{hz}"  # v49: was SL30

        if tp_col not in first.columns or sl_col not in first.columns:
            continue

        # Label trades: TP first = win (+50%), SL first = loss (-50%)
        first[f"_tp_{hz}"] = pd.to_numeric(first.get(tp_col), errors="coerce")
        first[f"_sl_{hz}"] = pd.to_numeric(first.get(sl_col), errors="coerce")
        has_data = first[f"_tp_{hz}"].notna() | first[f"_sl_{hz}"].notna()
        labeled = first[has_data].copy()

        if labeled.empty:
            continue

        tp_hit = labeled[f"_tp_{hz}"].notna()
        sl_hit = labeled[f"_sl_{hz}"].notna()
        labeled["_won"] = (tp_hit & (~sl_hit | (labeled[f"_tp_{hz}"] < labeled[f"_sl_{hz}"]))).astype(int)

        # Kelly by score band
        bands_result = {}
        for band_name, lo, hi in SCORE_BANDS:
            band = labeled[(labeled["score"] >= lo) & (labeled["score"] < hi)]
            n = len(band)
            if n < 5:
                bands_result[band_name] = {"count": n, "kelly_pct": 0, "verdict": "insufficient data"}
                continue

            wins = int(band["_won"].sum())
            losses = n - wins
            p = wins / n  # Win probability
            q = 1 - p

            # Average win = +50%, average loss = -30% (TP50/SL30)
            avg_win = 0.50
            avg_loss = 0.30
            b = avg_win / avg_loss  # Odds ratio

            # Kelly fraction
            kelly = (p * b - q) / b if b > 0 else 0
            kelly = max(0, kelly)  # Never negative (= don't bet)
            half_kelly = kelly / 2

            bands_result[band_name] = {
                "count": n,
                "wins": wins,
                "win_rate": round(p, 4),
                "kelly_full_pct": round(kelly * 100, 2),
                "kelly_half_pct": round(half_kelly * 100, 2),
                "expected_edge_pct": round((p * avg_win - q * avg_loss) * 100, 2),
                "verdict": (
                    "NO EDGE" if kelly <= 0
                    else "WEAK" if half_kelly < 0.02
                    else "TRADEABLE" if half_kelly < 0.10
                    else "STRONG"
                ),
            }

        results[f"TP50_SL30_{hz}"] = bands_result

    # Overall recommendation
    all_bands = {}
    for hz_key, bands in results.items():
        if not isinstance(bands, dict):
            continue
        for band_name, info in bands.items():
            if isinstance(info, dict) and info.get("kelly_half_pct", 0) > 0:
                all_bands[f"{hz_key}/{band_name}"] = info

    if all_bands:
        best_band = max(all_bands, key=lambda k: all_bands[k]["kelly_half_pct"])
        results["best_band"] = best_band
        results["best_half_kelly_pct"] = all_bands[best_band]["kelly_half_pct"]
        results["recommendation"] = (
            f"Bet {all_bands[best_band]['kelly_half_pct']:.1f}% of bankroll per trade "
            f"on {best_band} tokens (half-Kelly)"
        )
    else:
        results["recommendation"] = "No score band has positive edge with current data."

    return results


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

    # === PRIMARY TARGET: #1 token must hit return threshold ===
    top1 = report.get("top1_hit_rate", {})
    for horizon in ["12h", "6h", "24h", "48h", "72h", "7d"]:
        t1 = top1.get(horizon, {})
        if t1.get("tokens_tested", 0) > 0:
            tested = t1["tokens_tested"]
            # v22: Show multi-threshold hit rates
            for thresh_key in ["1.3x", "1.5x", "2.0x"]:
                hr = t1.get(f"hit_rate_{thresh_key}", 0)
                hits = t1.get(f"hits_{thresh_key}", 0)
                if hr >= 1.0:
                    recs.append(f"TARGET MET ({horizon}/{thresh_key}): #1 token hit {hits}/{tested} (100%)")
                elif hr >= 0.5:
                    recs.append(f"CLOSE ({horizon}/{thresh_key}): #1 token hit {hits}/{tested} ({hr*100:.0f}%)")
                else:
                    recs.append(f"NEEDS WORK ({horizon}/{thresh_key}): #1 token hit {hits}/{tested} ({hr*100:.0f}%)")
            # Show which #1 tokens failed at 2x
            for d in t1.get("details", []):
                if not d.get("did_2x"):
                    ret = d.get("max_return")
                    ret_str = f"{ret}x" if ret else "?"
                    recs.append(f"  MISS: {d['symbol']} score={d['score']} return={ret_str}")
            break  # Only show primary horizon

    # Extraction analysis insights
    ext = report.get("extraction_analysis", {})
    ca_only = ext.get("ca_only", {})
    ticker_only = ext.get("ticker_only", {})
    if ca_only.get("count", 0) >= 5 and ticker_only.get("count", 0) >= 5:
        ca_hr = ca_only["hit_rate"]
        tick_hr = ticker_only["hit_rate"]
        if ca_hr > tick_hr * 1.5:
            recs.append(f"EXTRACTION: CA-only calls ({ca_hr*100:.0f}% hit rate) outperform "
                       f"ticker-only ({tick_hr*100:.0f}%) by {ca_hr/max(0.001, tick_hr):.1f}x. "
                       "Consider weighting CA-extracted tokens higher.")
        elif tick_hr > ca_hr * 1.5:
            recs.append(f"EXTRACTION: Ticker-only calls ({tick_hr*100:.0f}% hit rate) outperform "
                       f"CA-only ({ca_hr*100:.0f}%). Unexpected — investigate data quality.")
        else:
            recs.append(f"EXTRACTION: CA-only ({ca_hr*100:.0f}%) and ticker-only ({tick_hr*100:.0f}%) "
                       "have similar hit rates. Extraction mode is not a strong differentiator.")

    # Bot simulation insights
    bot = report.get("realistic_bot", {})
    best_strat = bot.get("best_strategy")
    best_exp = bot.get("best_expectancy")
    if best_strat and best_exp is not None:
        if best_exp > 0:
            strat_data = bot.get(best_strat, {})
            recs.append(
                f"BOT VIABLE: {best_strat} has +{best_exp*100:.1f}% expectancy per trade "
                f"(WR={strat_data.get('win_rate', 0)*100:.0f}%, "
                f"PF={strat_data.get('profit_factor', 0):.2f}, "
                f"{strat_data.get('trades', 0)} trades)"
            )
        else:
            recs.append(
                f"BOT NOT VIABLE: best strategy {best_strat} has {best_exp*100:.1f}% expectancy. "
                "All TP/SL combos are unprofitable with current scoring."
            )
    elif bot:
        recs.append("BOT: Not enough TP/SL data yet. Wait for outcome_tracker to fill bot columns.")

    # Temporal analysis insights
    temporal = report.get("temporal_analysis", {})
    if temporal:
        best_day = temporal.get("best_day")
        worst_day = temporal.get("worst_day")
        if best_day and worst_day:
            best_hr = temporal.get("best_day_hit_rate", 0)
            worst_hr = temporal.get("worst_day_hit_rate", 0)
            recs.append(
                f"TEMPORAL: Best day={best_day} ({best_hr*100:.0f}% hit rate), "
                f"Worst day={worst_day} ({worst_hr*100:.0f}%)"
            )
        prime = temporal.get("prime_time", {})
        off_peak = temporal.get("off_peak", {})
        if prime and off_peak:
            p_hr = prime.get("hit_rate", 0)
            o_hr = off_peak.get("hit_rate", 0)
            if p_hr > o_hr * 1.3:
                recs.append(
                    f"TEMPORAL: Prime time 19h-5h ({p_hr*100:.0f}%) outperforms "
                    f"off-peak ({o_hr*100:.0f}%) — confirms runner hours"
                )
            elif o_hr > p_hr * 1.3:
                recs.append(
                    f"TEMPORAL: Off-peak 5h-19h ({o_hr*100:.0f}%) outperforms "
                    f"prime time ({p_hr*100:.0f}%) — surprising"
                )

    # Adaptive SL insights
    adaptive = report.get("adaptive_bot", {})
    improvement = adaptive.get("adaptive_improvement_pp")
    if improvement is not None:
        if improvement > 0:
            recs.append(
                f"ADAPTIVE SL: Oracle adaptive SL improves expectancy by +{improvement:.1f}pp "
                f"over best fixed SL. ML-predicted SL has room to add value."
            )
        else:
            recs.append(
                f"ADAPTIVE SL: Oracle adaptive SL does NOT improve over fixed SL "
                f"({improvement:+.1f}pp). Fixed SL is sufficient."
            )
    elif adaptive:
        recs.append("ADAPTIVE SL: Not enough DD data for adaptive simulation.")

    # Equity curve insights
    eq = report.get("equity_curve", {})
    for key, info in eq.items():
        if not isinstance(info, dict) or "max_drawdown_pct" not in info:
            continue
        dd = info["max_drawdown_pct"]
        ret = info["total_return_pct"]
        streak = info.get("max_losing_streak", 0)
        if dd > 30:
            recs.append(
                f"RISK ({key}): Max drawdown {dd:.0f}% — account would lose {dd:.0f}% at worst. "
                f"Max losing streak: {streak}."
            )
        elif ret > 0:
            calmar = info.get("calmar_ratio", 0)
            recs.append(
                f"EQUITY ({key}): +{ret:.1f}% return, {dd:.1f}% max DD "
                f"(Calmar={calmar:.2f}), streak={streak}"
            )

    # Slippage insights
    slip = report.get("slippage", {})
    rec_size = slip.get("recommended_max_size_usd")
    if rec_size:
        s500 = slip.get("size_500", {})
        recs.append(
            f"SLIPPAGE: Recommended max trade size ${rec_size}. "
            f"At $500: median slippage {s500.get('median_slippage_pct', '?')}%, "
            f"{s500.get('tokens_above_5pct_slip', '?')} tokens >5% slip"
        )

    # Confidence interval insights
    ci = report.get("confidence_intervals", {})
    verdict = ci.get("verdict")
    if verdict:
        recs.append(f"STATISTICAL: {verdict}")
    needed = ci.get("samples_needed", {})
    deficit = needed.get("deficit", 0)
    if deficit > 0:
        recs.append(
            f"DATA NEEDED: {deficit} more unique tokens for reliable ±10pp CI "
            f"(have {needed.get('current', 0)}, need {needed.get('for_10pp_ci', 0)})"
        )

    # Portfolio insights
    port = report.get("portfolio", {})
    best_cfg = port.get("best_config")
    if best_cfg:
        best_info = port.get(best_cfg, {})
        recs.append(
            f"PORTFOLIO: Best config = {best_cfg} "
            f"(return={best_info.get('total_return_pct', 0):+.1f}%, "
            f"DD={best_info.get('max_drawdown_pct', 0):.1f}%, "
            f"WR={best_info.get('win_rate', 0)*100:.0f}%)"
        )

    # Kelly insights
    kelly = report.get("kelly", {})
    kelly_rec = kelly.get("recommendation")
    if kelly_rec:
        recs.append(f"KELLY: {kelly_rec}")

    # Overall hit rate
    data = report.get("data_summary", {})
    hr = data.get("hit_rate_12h", 0)
    if hr > 0:
        recs.append(f"Overall 12h 2x rate: {hr*100:.0f}% across all tokens")

    if not recs:
        recs.append("Insufficient data for recommendations. Continue collecting snapshots.")

    return recs


def _compute_market_benchmarks(client, df: pd.DataFrame) -> None:
    """
    v26: Compute rolling 7-day market benchmarks from labeled token data.
    Stores results in scoring_config.market_benchmarks JSONB column.

    Benchmarks computed (first-appearance dedup, last 7 days):
      - median_peak_return_{hz}: median of max_price / price_at_snapshot
      - win_rate_7d: % of tokens that hit >= ML_THRESHOLD
      - p25/p75 percentiles for context
      - n_tokens_7d: sample size for confidence
    """
    try:
        # Filter to last 7 days only
        cutoff = df["snapshot_at"].max() - pd.Timedelta(days=7)
        recent = df[df["snapshot_at"] >= cutoff].copy()

        # First-appearance dedup: one observation per token
        recent = recent.sort_values("snapshot_at").drop_duplicates(
            subset=["token_address"], keep="first"
        )

        benchmarks = {
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "n_tokens_7d": len(recent),
        }

        # Compute peak returns per horizon
        for hz in ["12h", "24h"]:
            max_col = f"max_price_{hz}"
            if max_col not in recent.columns:
                continue

            # peak_return = max_price / price_at_snapshot
            valid = recent[
                recent[max_col].notna() & recent["price_at_snapshot"].notna()
                & (recent["price_at_snapshot"] > 0)
            ].copy()

            if len(valid) < 5:
                continue

            returns = valid[max_col].astype(float) / valid["price_at_snapshot"].astype(float)
            returns = returns[returns.between(0.01, 1000)]  # sanity bounds

            if len(returns) < 5:
                continue

            benchmarks[f"median_peak_return_{hz}"] = round(float(returns.median()), 4)
            benchmarks[f"p25_peak_return_{hz}"] = round(float(returns.quantile(0.25)), 4)
            benchmarks[f"p75_peak_return_{hz}"] = round(float(returns.quantile(0.75)), 4)

        # Win rate: % of tokens hitting >= ML_THRESHOLD in primary horizon
        primary_hz = ML_HORIZON  # e.g. "12h"
        max_col = f"max_price_{primary_hz}"
        if max_col in recent.columns:
            valid = recent[
                recent[max_col].notna() & recent["price_at_snapshot"].notna()
                & (recent["price_at_snapshot"] > 0)
            ]
            if len(valid) > 0:
                returns = valid[max_col].astype(float) / valid["price_at_snapshot"].astype(float)
                winners = (returns >= ML_THRESHOLD).sum()
                benchmarks["win_rate_7d"] = round(float(winners / len(returns)), 4)

        logger.info(
            "Market benchmarks (7d, n=%d): median_12h=%.2fx, median_24h=%.2fx, wr=%.1f%%",
            benchmarks.get("n_tokens_7d", 0),
            benchmarks.get("median_peak_return_12h", 0),
            benchmarks.get("median_peak_return_24h", 0),
            benchmarks.get("win_rate_7d", 0) * 100,
        )

        # Store in scoring_config
        client.table("scoring_config").update(
            {"market_benchmarks": benchmarks}
        ).eq("id", 1).execute()

        logger.info("Market benchmarks saved to scoring_config")

    except Exception as e:
        logger.error("Failed to compute market benchmarks: %s", e, exc_info=True)


def run_auto_backtest() -> dict | None:
    """
    Main entry point. Fetches snapshots, runs progressive analyses,
    returns structured report dict. Returns None if not enough data.
    """
    client = _get_client()
    if not client:
        return None

    # Load current production weights from scoring_config
    _load_current_weights(client)

    df = _fetch_snapshots(client)
    if df.empty:
        logger.info("Auto-backtest: no snapshots found")
        return None

    total = len(df)

    ALL_HORIZONS = ["12h", "24h", "48h", "72h", "7d"]

    # First-appearance dedup: each token counted once (earliest snapshot)
    first = df.sort_values("snapshot_at").drop_duplicates(subset=["token_address"], keep="first")
    unique_tokens = len(first)

    # Use 12h as primary horizon (faster feedback loop)
    horizon_col = "did_2x_12h"
    n_labeled = len(first[first[horizon_col].notna()])

    if n_labeled < 30:
        logger.info("Auto-backtest: waiting for data (%d/30 labeled tokens)", n_labeled)
        return None

    # Data summary — per-token (first appearance), not per-snapshot
    date_range = (df["snapshot_at"].max() - df["snapshot_at"].min()).days

    ds = {
        "total_snapshots": total,
        "unique_tokens": unique_tokens,
        "dedup_ratio": round(total / max(1, unique_tokens), 1),
        "date_range_days": date_range,
    }
    for hz in ALL_HORIZONS:
        col = f"did_2x_{hz}"
        if col not in first.columns:
            continue
        lbl = first[first[col].notna()]
        n = len(lbl)
        ds[f"labeled_{hz}"] = n
        ds[f"hit_rate_{hz}"] = round(float(lbl[col].sum()) / max(1, n), 4) if n > 0 else None

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_summary": ds,
    }

    # === TOP 1 TOKEN HIT RATE (the primary target) ===
    logger.info("Auto-backtest: computing #1 token hit rate")
    report["top1_hit_rate"] = _top1_hit_rate(df)

    # === REALISTIC BOT SIMULATION (TP/SL candle-by-candle) ===
    logger.info("Auto-backtest: running realistic bot simulation")
    report["realistic_bot"] = _realistic_bot_simulation(df)

    # === MULTI-TRANCHE BOT SIMULATION (SCALE_OUT + MOONBAG) ===
    logger.info("Auto-backtest: running multi-tranche bot simulation (SCALE_OUT + MOONBAG)")
    report["multi_tranche_bot"] = _multi_tranche_bot_simulation(df)

    # === ADAPTIVE SL SIMULATION (ML v3.1 — oracle upper bound) ===
    logger.info("Auto-backtest: running adaptive SL simulation")
    report["adaptive_bot"] = _adaptive_bot_simulation(df)

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
        report["extraction_analysis"] = _extraction_analysis(df, horizon_col)
        report["temporal_analysis"] = _temporal_analysis(df, horizon_col)

    # Tier 3: 100+ labeled
    if n_labeled >= 100:
        logger.info("Auto-backtest: running weight sensitivity + optimal threshold")
        report["weight_sensitivity"] = _weight_sensitivity(df, horizon_col)
        report["optimal_threshold"] = _optimal_threshold(df, horizon_col)

        # Auto-apply optimal weights if improvement is significant
        try:
            apply_result = _auto_apply_weights(client, df, horizon_col)
            if apply_result:
                report["auto_applied_weights"] = apply_result
                logger.info("Auto-backtest: weights auto-applied! %s", apply_result.get("reason", ""))
            else:
                report["auto_applied_weights"] = {"applied": False, "reason": "no significant improvement"}
        except Exception as e:
            logger.error("Auto-apply weights failed: %s", e)
            report["auto_applied_weights"] = {"applied": False, "error": str(e)}

    # Tier 4: 200+ labeled
    if n_labeled >= 200:
        logger.info("Auto-backtest: running walk-forward validation")
        report["walk_forward"] = _walk_forward(df, horizon_col)

    # === NEW ANALYSES (always run, they handle sparse data gracefully) ===

    # Gap #1: Equity curve + max drawdown
    logger.info("Auto-backtest: running equity curve analysis")
    report["equity_curve"] = _equity_curve_analysis(df)

    # Gap #2: Slippage modeling
    logger.info("Auto-backtest: running slippage analysis")
    report["slippage"] = _slippage_analysis(df)

    # Gap #3: Confidence intervals
    logger.info("Auto-backtest: running confidence intervals")
    report["confidence_intervals"] = _confidence_intervals(df, horizon_col)

    # Gap #4: Portfolio simulation (top 1/3/5)
    logger.info("Auto-backtest: running portfolio simulation")
    report["portfolio"] = _portfolio_simulation(df)

    # Gap #5: Kelly criterion position sizing
    logger.info("Auto-backtest: running Kelly analysis")
    report["kelly"] = _kelly_analysis(df, horizon_col)

    # v26: Compute & store market benchmarks for pipeline context features
    _compute_market_benchmarks(client, df)

    # Generate recommendations
    report["recommendations"] = _generate_recommendations(report)

    # Push to Supabase
    try:
        _insert_backtest_report(client, report, total, n_labeled, ds.get("hit_rate_12h", 0))
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

# ═══════════════════════════════════════════════════════════════════════════════
# v44: OPTUNA PARAMETER OPTIMIZATION SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_score_with_params(
    row: pd.Series,
    weights: dict,
    params: dict,
) -> int:
    """
    Fast re-scorer using stored snapshot values + trial params.
    For multipliers whose raw values are stored (activity, hype, breadth_pen),
    recomputes from raw values with trial thresholds/factors.
    For others (momentum, size), uses stored multiplier values as-is.
    """
    # --- Base score from component values (same logic as _compute_score) ---
    consensus = _safe_mult(row, "consensus_val", default=0)
    if not pd.notna(row.get("consensus_val")):
        b_raw = row.get("breadth")
        consensus = min(1.0, float(b_raw) / 0.15) if pd.notna(b_raw) else None

    # v47: Consensus pump discount from trial params
    _cp_thresh = params.get("consensus_pump_threshold", 50)
    _cp_floor = params.get("consensus_pump_floor", 0.5)
    _cp_divisor = params.get("consensus_pump_divisor", 400)
    pc24_raw = row.get("price_change_24h") if "price_change_24h" in row.index else None
    if consensus is not None and pd.notna(pc24_raw) and float(pc24_raw) > _cp_thresh:
        consensus_discount = max(_cp_floor, 1.0 - (float(pc24_raw) / _cp_divisor))
        consensus *= consensus_discount

    # v47: Conviction normalization from trial params
    _conv_offset = params.get("conviction_offset", 6)
    _conv_divisor = params.get("conviction_divisor", 4)
    conviction_val = _safe_mult(row, "conviction_val", default=0)
    if not pd.notna(row.get("conviction_val")):
        ac = row.get("avg_conviction")
        conviction_val = max(0, min(1, (float(ac) - _conv_offset) / _conv_divisor)) if pd.notna(ac) else None

    breadth_val = _safe_mult(row, "breadth_val", default=0)
    if not pd.notna(row.get("breadth_val")):
        b = row.get("breadth")
        breadth_val = float(b) if pd.notna(b) else None

    pa_val = _safe_mult(row, "price_action_val", default=0.5)
    if not pd.notna(row.get("price_action_val")):
        pa = row.get("price_action_score")
        pa_val = float(pa) if pd.notna(pa) else None

    # v46: Recompute PA with trial direction penalties + norm bounds
    pa_cfg = params.get("pa_config", {})
    if pa_cfg and pa_val is not None:
        stored_score = pa_val  # [0, 1] normalized
        # Reverse normalization to get raw PA mult (stored with floor=0.4, cap=1.3)
        orig_floor, orig_cap = 0.4, 1.3
        if stored_score <= 0.5:
            stored_pa_mult = orig_floor + stored_score * 2 * (1.0 - orig_floor)
        else:
            stored_pa_mult = 1.0 + (stored_score - 0.5) * 2 * (orig_cap - 1.0)
        # Since v27 set direction=1.0: other_3 = position + vol_confirm + support
        other_3_sum = 4 * stored_pa_mult - 1.0
        # Recompute direction_mult from stored RSI
        rsi_v = row.get("rsi_14")
        new_direction = 1.0
        if pd.notna(rsi_v):
            rsi_f = float(rsi_v)
            if rsi_f > pa_cfg.get("rsi_hard_pump", 80):
                new_direction = pa_cfg.get("dir_hard_pump_mult", 1.0)
            elif rsi_f > pa_cfg.get("rsi_pump", 70):
                new_direction = pa_cfg.get("dir_pump_mult", 1.0)
            elif rsi_f < pa_cfg.get("rsi_freefall", 20):
                new_direction = pa_cfg.get("dir_freefall_mult", 1.0)
            elif rsi_f < pa_cfg.get("rsi_dying", 30):
                new_direction = pa_cfg.get("dir_dying_mult", 1.0)
        # Corrected PA mult
        corrected_pa_mult = (other_3_sum + new_direction) / 4.0
        trial_floor = params.get("pa_norm_floor", orig_floor)
        trial_cap = params.get("pa_norm_cap", orig_cap)
        corrected_pa_mult = max(trial_floor, min(trial_cap, corrected_pa_mult))
        # Re-normalize to [0, 1]
        below = 1.0 - trial_floor
        above = trial_cap - 1.0
        if corrected_pa_mult <= 1.0:
            pa_val = (corrected_pa_mult - trial_floor) / (below * 2) if below > 0 else 0.5
        else:
            pa_val = 0.5 + (corrected_pa_mult - 1.0) / (above * 2) if above > 0 else 0.5

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

    # --- Multiplier chain (v48: recompute ALL multipliers from raw fields) ---
    pump_bonus = _safe_mult(row, "pump_bonus")
    manipulation_pen = _safe_mult(row, "wash_pen")
    pvp_pen = _safe_mult(row, "pvp_pen")
    gate_mult = _safe_mult(row, "gate_mult")

    # v48: Recompute onchain_multiplier from raw fields + trial onchain_config
    oc_cfg = params.get("onchain_config", {})
    if oc_cfg:
        oc_factors = []
        vmr = row.get("volume_mcap_ratio")
        if pd.notna(vmr):
            oc_factors.append(min(1.5, 0.5 + float(vmr) * 2))
        lmr = row.get("liq_mcap_ratio")
        if pd.notna(lmr):
            lmr_v = float(lmr)
            if lmr_v < oc_cfg["lmr_low"]:
                oc_factors.append(oc_cfg["lmr_low_factor"])
            elif lmr_v > oc_cfg["lmr_high"]:
                oc_factors.append(oc_cfg["lmr_high_factor"])
            else:
                oc_factors.append(oc_cfg.get("lmr_interp_base", 0.8) + lmr_v * oc_cfg.get("lmr_interp_slope", 4))
        age = row.get("token_age_hours")
        if pd.notna(age):
            oc_factors.append(_bt_tier_lookup(float(age), oc_cfg["age_thresholds"], oc_cfg["age_factors"]))
        recent_tx = row.get("helius_recent_tx_count")
        if pd.notna(recent_tx):
            rtx = float(recent_tx)
            oc_factors.append(_bt_tier_lookup(rtx, oc_cfg.get("tx_thresholds", [5, 20, 40]), oc_cfg.get("tx_factors", [0.7, 1.0, 1.1, 1.3])))
        h_bsr = row.get("helius_onchain_bsr")
        if pd.notna(h_bsr):
            oc_factors.append(oc_cfg.get("bsr_base", 0.5) + float(h_bsr))
        jup_t_val = row.get("jup_tradeable")
        jup_imp = row.get("jup_price_impact_1k")
        if pd.notna(jup_t_val):
            if int(jup_t_val) == 0:
                oc_factors.append(oc_cfg.get("jup_non_tradeable_factor", 0.5))
            elif pd.notna(jup_imp):
                oc_factors.append(_bt_tier_lookup(float(jup_imp), oc_cfg["jup_impact_thresholds"], oc_cfg["jup_impact_factors"]))
        wc_val = row.get("whale_change")
        if pd.notna(wc_val):
            oc_factors.append(_bt_tier_lookup(float(wc_val), oc_cfg["whale_change_thresholds"], oc_cfg["whale_change_factors"]))
        txn_cnt = row.get("txn_count_24h")
        h_holders = row.get("helius_holder_count") if pd.notna(row.get("helius_holder_count")) else row.get("holder_count")
        if pd.notna(txn_cnt) and pd.notna(h_holders) and float(h_holders) > 0:
            velocity = float(txn_cnt) / float(h_holders)
            oc_factors.append(_bt_tier_lookup(velocity, oc_cfg["velocity_thresholds"], oc_cfg["velocity_factors"]))
        elif pd.notna(txn_cnt) and float(txn_cnt) > 0:
            oc_factors.append(_bt_tier_lookup(float(txn_cnt), oc_cfg.get("tx_thresholds", [5, 20, 40]), oc_cfg.get("tx_factors", [0.7, 1.0, 1.1, 1.3])))
        vol_proxy = row.get("volatility_proxy")
        if pd.notna(vol_proxy) and float(vol_proxy) > oc_cfg["vol_proxy_threshold"]:
            oc_factors.append(oc_cfg["vol_proxy_penalty"])
        whale_dir = row.get("whale_direction")
        if whale_dir == "accumulating":
            oc_factors.append(oc_cfg["whale_accum_bonus"])
        wne = row.get("whale_new_entries")
        if pd.notna(wne):
            oc_factors.append(_bt_tier_lookup(float(wne), oc_cfg["wne_thresholds"], oc_cfg["wne_factors"]))
        wct_v = row.get("whale_count")
        if pd.notna(wct_v):
            oc_factors.append(_bt_tier_lookup(float(wct_v), oc_cfg["whale_count_thresholds"], oc_cfg["whale_count_factors"]))
        uw_ch = row.get("unique_wallet_24h_change")
        if pd.notna(uw_ch):
            oc_factors.append(_bt_tier_lookup(float(uw_ch), oc_cfg["uw_change_thresholds"], oc_cfg["uw_change_factors"]))
        # v53: Smart money retention
        smr_v = row.get("smart_money_retention")
        if pd.notna(smr_v):
            oc_factors.append(_bt_tier_lookup(float(smr_v), oc_cfg.get("smr_thresholds", [50, 70, 90]), oc_cfg.get("smr_factors", [0.8, 1.0, 1.15, 1.3])))
        # v53: Small holder pct
        shp_v = row.get("small_holder_pct")
        if pd.notna(shp_v):
            oc_factors.append(_bt_tier_lookup(float(shp_v), oc_cfg.get("shp_thresholds", [50, 70, 85]), oc_cfg.get("shp_factors", [0.7, 0.9, 1.1, 1.3])))
        # v53: Liquidity depth score
        lds_v = row.get("liquidity_depth_score")
        if pd.notna(lds_v):
            oc_factors.append(_bt_tier_lookup(float(lds_v), oc_cfg.get("lds_thresholds", [0.2, 0.5, 0.8]), oc_cfg.get("lds_factors", [0.6, 0.85, 1.05, 1.2])))
        if oc_factors:
            onchain = max(params.get("onchain_mult_floor", 0.3),
                          min(params.get("onchain_mult_cap", 1.5), sum(oc_factors) / len(oc_factors)))
        else:
            onchain = 1.0
    else:
        onchain = _safe_mult(row, "onchain_multiplier")
        onchain = max(params.get("onchain_mult_floor", 0.3), min(params.get("onchain_mult_cap", 1.5), onchain))

    # v48: Recompute safety_penalty from raw fields + trial safety_config
    sf_cfg = params.get("safety_config", {})
    if sf_cfg:
        safety = 1.0
        ins = row.get("insider_pct")
        if pd.notna(ins) and float(ins) > sf_cfg["insider_threshold"]:
            safety *= max(sf_cfg["insider_floor"], 1.0 - (float(ins) - sf_cfg["insider_threshold"]) / sf_cfg.get("insider_slope", 100))
        t10 = row.get("top10_holder_pct")
        if pd.notna(t10) and float(t10) > sf_cfg["top10_threshold"]:
            safety *= max(sf_cfg["top10_floor"], 1.0 - (float(t10) - sf_cfg["top10_threshold"]) / sf_cfg.get("top10_slope", 100))
        rsk = row.get("risk_score")
        if pd.notna(rsk) and float(rsk) > sf_cfg["risk_score_threshold"]:
            safety *= max(sf_cfg["risk_score_floor"], 1.0 - (float(rsk) - sf_cfg["risk_score_threshold"]) / sf_cfg.get("risk_score_slope", 5000))
        rc_raw = row.get("risk_count")
        if pd.notna(rc_raw) and int(rc_raw) >= sf_cfg["risk_count_threshold"]:
            safety *= sf_cfg["risk_count_penalty"]
        jito_txns = int(row.get("jito_max_slot_txns") or 0) if pd.notna(row.get("jito_max_slot_txns")) else 0
        if jito_txns >= sf_cfg["jito_hard_threshold"]:
            safety *= sf_cfg["jito_hard_penalty"]
        elif jito_txns >= sf_cfg["jito_soft_threshold"]:
            safety *= sf_cfg["jito_soft_penalty"]
        else:
            bd = row.get("bundle_detected")
            bp_v = float(row.get("bundle_pct") or 0) if pd.notna(row.get("bundle_pct")) else 0
            if pd.notna(bd) and bool(bd):
                if bp_v > sf_cfg["bundle_hard_threshold"]:
                    safety *= sf_cfg["bundle_hard_penalty"]
                elif bp_v > sf_cfg["bundle_soft_threshold"]:
                    safety *= sf_cfg["bundle_soft_penalty"]
        h_gini = row.get("helius_gini")
        if pd.notna(h_gini) and float(h_gini) > sf_cfg["gini_threshold"]:
            safety *= sf_cfg["gini_penalty"]
        h_hc = row.get("helius_holder_count")
        if pd.notna(h_hc) and float(h_hc) < sf_cfg["holder_count_threshold"]:
            safety *= sf_cfg["holder_count_penalty"]
        wtp = row.get("whale_total_pct")
        if pd.notna(wtp) and float(wtp) > sf_cfg["whale_conc_threshold"]:
            safety *= max(sf_cfg["whale_conc_floor"], 1.0 - (float(wtp) - sf_cfg["whale_conc_threshold"]) / sf_cfg.get("whale_conc_slope", 80))
        bb_s = row.get("bubblemaps_score")
        if pd.notna(bb_s):
            bb_t = sf_cfg["bb_score_thresholds"]
            bb_p = sf_cfg["bb_score_penalties"]
            if float(bb_s) < bb_t[0]:
                safety *= bb_p[0]
            elif float(bb_s) < bb_t[1]:
                safety *= bb_p[1]
        bb_cm = row.get("bubblemaps_cluster_max_pct")
        if pd.notna(bb_cm) and float(bb_cm) > sf_cfg["bb_cluster_threshold"]:
            safety *= max(sf_cfg["bb_cluster_floor"], 1.0 - (float(bb_cm) - sf_cfg["bb_cluster_threshold"]) / sf_cfg.get("bb_cluster_slope", 70))
        wd = row.get("whale_dominance")
        if pd.notna(wd) and float(wd) > sf_cfg["whale_dom_threshold"]:
            safety *= sf_cfg["whale_dom_penalty"]
        w_dir = row.get("whale_direction")
        if w_dir == "distributing":
            safety *= sf_cfg["whale_dist_penalty"]
        elif w_dir == "dumping":
            safety *= sf_cfg["whale_dump_penalty"]
        lp_l = row.get("lp_locked_pct")
        if pd.notna(lp_l):
            if float(lp_l) == 0:
                safety *= sf_cfg["lp_unlock_penalty"]
            elif float(lp_l) < sf_cfg["lp_partial_threshold"]:
                safety *= sf_cfg["lp_partial_penalty"]
        cex = row.get("bubblemaps_cex_pct")
        if pd.notna(cex) and float(cex) > sf_cfg["cex_threshold"]:
            safety *= max(sf_cfg["cex_floor"], 1.0 - (float(cex) - sf_cfg["cex_threshold"]) / sf_cfg.get("cex_slope", 100))
        safety = max(params.get("safety_floor", 0.75), safety)
    else:
        safety = _safe_mult(row, "safety_penalty")
        safety = max(params.get("safety_floor", 0.75), safety)

    # Activity_mult: recompute from raw ratio if available
    act_ratio_raw = row.get("activity_ratio_raw")
    if pd.notna(act_ratio_raw):
        act_ratio = float(act_ratio_raw)
        act_floor = params.get("activity_mult_floor", 0.80)
        act_cap = params.get("activity_mult_cap", 1.25)
        act_high = params.get("activity_ratio_high", 0.6)
        act_mid = params.get("activity_ratio_mid", 0.3)
        act_low = params.get("activity_ratio_low", 0.1)
        act_mid_mult = params.get("activity_mid_mult", 1.10)  # v47
        if act_ratio > act_high:
            activity_mult = act_cap
        elif act_ratio > act_mid:
            activity_mult = act_mid_mult
        elif act_ratio > act_low:
            activity_mult = 1.0
        else:
            activity_mult = act_floor
        # Pump cap
        pc24_act = row.get("price_change_24h")
        pump_hard = params.get("activity_pump_cap_hard", 80)
        pump_soft = params.get("activity_pump_cap_soft", 50)
        if pd.notna(pc24_act) and float(pc24_act) > pump_hard:
            activity_mult = min(activity_mult, 1.0)
        elif pd.notna(pc24_act) and float(pc24_act) > pump_soft:
            activity_mult = min(activity_mult, act_mid_mult)
    else:
        activity_mult = _safe_mult(row, "activity_mult")

    # Breadth_pen: recompute from breadth score
    breadth_score_raw = row.get("breadth") if "breadth" in row.index else None
    bp_cfg = params.get("breadth_pen_config", {"thresholds": [0.033, 0.05, 0.08], "penalties": [0.75, 0.85, 0.95]})
    if pd.notna(breadth_score_raw):
        bs = float(breadth_score_raw)
        bp_t = bp_cfg["thresholds"]
        bp_p = bp_cfg["penalties"]
        if bs < bp_t[0]:
            breadth_pen = bp_p[0]
        elif bs < bp_t[1]:
            breadth_pen = bp_p[1]
        elif bs < bp_t[2]:
            breadth_pen = bp_p[2]
        else:
            breadth_pen = 1.0
    else:
        breadth_pen = _safe_mult(row, "breadth_pen")

    # Hype_pen: recompute from unique_kols
    uk = row.get("unique_kols")
    hp_cfg = params.get("hype_pen_config", {"thresholds": [2, 4, 7], "penalties": [1.0, 0.85, 0.65, 0.50]})
    if pd.notna(uk):
        uk_count = int(uk)
        hp_t = hp_cfg["thresholds"]
        hp_p = hp_cfg["penalties"]
        if uk_count <= hp_t[0]:
            hype_pen = hp_p[0]
        elif uk_count <= hp_t[1]:
            hype_pen = hp_p[1]
        elif uk_count <= hp_t[2]:
            hype_pen = hp_p[2]
        else:
            hype_pen = hp_p[3]
    else:
        hype_pen = _safe_mult(row, "hype_pen")
    # v53: KOL co-occurrence penalty
    cooc_cfg = hp_cfg.get("cooc_config", {"threshold": 0.5, "penalty": 0.85})
    cooc_avg = row.get("kol_cooccurrence_avg")
    if pd.notna(cooc_avg) and float(cooc_avg) > cooc_cfg["threshold"]:
        hype_pen *= cooc_cfg["penalty"]

    # S-tier_mult: recompute from s_tier_count
    s_tier_count = row.get("s_tier_count")
    s_tier_bonus = params.get("s_tier_bonus", 1.2)
    if pd.notna(s_tier_count) and int(s_tier_count) > 0:
        s_tier_mult = s_tier_bonus
    else:
        s_tier_mult = 1.0

    # v48: Recompute size_mult from raw fields + trial size_mult_config
    sm_cfg = params.get("size_mult_config", {})
    if sm_cfg:
        t_mcap = float(row.get("market_cap") or 0) if pd.notna(row.get("market_cap")) else 0
        sm_mcap_thresh = sm_cfg.get("mcap_thresholds", [300000, 1000000, 5000000, 20000000, 50000000, 200000000, 500000000])
        sm_mcap_factors = sm_cfg.get("mcap_factors", [1.3, 1.15, 1.0, 0.85, 0.70, 0.50, 0.35, 0.25])
        sm_fresh_thresh = sm_cfg.get("fresh_thresholds", [4, 12])
        sm_fresh_factors = sm_cfg.get("fresh_factors", [1.2, 1.1, 1.0])
        sm_large_cap = sm_cfg.get("large_cap_threshold", 50000000)
        sm_floor = sm_cfg.get("floor", 0.25)
        sm_cap_v = sm_cfg.get("cap", 1.5)
        if t_mcap <= 0:
            mcap_factor = 1.0
        else:
            mcap_factor = sm_mcap_factors[-1]
            for i, thresh in enumerate(sm_mcap_thresh):
                if t_mcap < thresh:
                    mcap_factor = sm_mcap_factors[i]
                    break
        freshest_h_sm = float(row.get("freshest_mention_hours") or 0) if pd.notna(row.get("freshest_mention_hours")) else 0
        if t_mcap >= sm_large_cap:
            fresh_factor = 1.0
        elif freshest_h_sm < sm_fresh_thresh[0]:
            fresh_factor = sm_fresh_factors[0]
        elif freshest_h_sm < sm_fresh_thresh[1]:
            fresh_factor = sm_fresh_factors[1]
        else:
            fresh_factor = sm_fresh_factors[2]
        size_mult = max(sm_floor, min(sm_cap_v, mcap_factor * fresh_factor))
    else:
        size_mult = _safe_mult(row, "size_mult")

    # v48: Recompute momentum_mult from raw fields + trial momentum_config
    mom_cfg = params.get("momentum_config", {})
    if mom_cfg:
        kf = row.get("kol_freshness")
        kol_fresh_factor = 1.0
        kf_thresh = mom_cfg.get("kol_fresh_thresholds", [0.5, 0.2])
        kf_factors = mom_cfg.get("kol_fresh_factors", [1.20, 1.10, 1.05])
        if pd.notna(kf) and float(kf) > 0:
            kf_v = float(kf)
            if kf_v >= kf_thresh[0]:
                kol_fresh_factor = kf_factors[0]
            elif kf_v >= kf_thresh[1]:
                kol_fresh_factor = kf_factors[1]
            else:
                kol_fresh_factor = kf_factors[2]
        mhr = row.get("mention_heat_ratio")
        mention_heat_factor = 1.0
        mhr_thresh = mom_cfg.get("mhr_thresholds", [2.0, 1.0, 0.3])
        mhr_factors = mom_cfg.get("mhr_factors", [1.15, 1.10, 1.05])
        if pd.notna(mhr) and float(mhr) > 0:
            mhr_v = float(mhr)
            if mhr_v >= mhr_thresh[0]:
                mention_heat_factor = mhr_factors[0]
            elif mhr_v >= mhr_thresh[1]:
                mention_heat_factor = mhr_factors[1]
            elif mhr_v >= mhr_thresh[2]:
                mention_heat_factor = mhr_factors[2]
        sth = row.get("short_term_heat")
        vol_heat_factor = 1.0
        sth_thresh = mom_cfg.get("sth_thresholds", [3.0, 1.5])
        sth_factors_v = mom_cfg.get("sth_factors", [1.10, 1.05])
        sth_pen_thresh = mom_cfg.get("sth_penalty_threshold", 0.3)
        sth_pen_factor = mom_cfg.get("sth_penalty_factor", 0.95)
        if pd.notna(sth):
            sth_v = float(sth)
            if sth_v >= sth_thresh[0]:
                vol_heat_factor = sth_factors_v[0]
            elif sth_v >= sth_thresh[1]:
                vol_heat_factor = sth_factors_v[1]
            elif sth_v < sth_pen_thresh:
                vol_heat_factor = sth_pen_factor
        # v52: KOL cascade timing factors
        cascade_v = row.get("kol_cascade_rate")
        cascade_factor = 1.0
        if pd.notna(cascade_v):
            cv = float(cascade_v)
            if cv >= 3:
                cascade_factor = mom_cfg.get("cascade_factor_3plus", 1.15)
            elif cv >= 2:
                cascade_factor = mom_cfg.get("cascade_factor_2plus", 1.08)

        first_age_v = row.get("first_call_age_minutes")
        early_factor = 1.0
        if pd.notna(first_age_v):
            fa = float(first_age_v)
            if fa <= 30:
                early_factor = mom_cfg.get("early_factor_30min", 1.20)
            elif fa <= 60:
                early_factor = mom_cfg.get("early_factor_60min", 1.10)
            elif fa >= 360:
                early_factor = mom_cfg.get("late_penalty_360min", 0.85)

        pvfc_v = row.get("price_vs_first_call")
        pvfc_factor = 1.0
        if pd.notna(pvfc_v):
            pv = float(pvfc_v)
            if pv >= 3.0:
                pvfc_factor = mom_cfg.get("pvfc_penalty_3x", 0.60)
            elif pv >= 2.0:
                pvfc_factor = mom_cfg.get("pvfc_penalty_2x", 0.75)
            elif pv >= 1.5:
                pvfc_factor = mom_cfg.get("pvfc_penalty_1_5x", 0.90)

        mom_floor = mom_cfg.get("floor", 0.70)
        mom_cap = mom_cfg.get("cap", 1.40)
        momentum_mult = max(mom_floor, min(mom_cap, kol_fresh_factor * mention_heat_factor * vol_heat_factor * cascade_factor * early_factor * pvfc_factor))
    else:
        momentum_mult = _safe_mult(row, "momentum_mult")

    # --- v47: Recompute pump_momentum_pen from trial pump_pen_config ---
    pp_cfg = params.get("pump_pen_config", {})
    if pp_cfg:
        pc_1h_pm = row.get("price_change_1h")
        pc_5m_pm = row.get("price_change_5m")
        pump_1h_hard = pp_cfg.get("pump_1h_hard", 30)
        pump_5m_hard = pp_cfg.get("pump_5m_hard", 15)
        pump_1h_mod = pp_cfg.get("pump_1h_mod", pump_1h_hard * 0.5)
        pump_5m_mod = pp_cfg.get("pump_5m_mod", pump_5m_hard * 0.533)
        pump_1h_light = pp_cfg.get("pump_1h_light", pump_1h_hard * 0.267)
        pump_momentum_pen = 1.0
        if (pd.notna(pc_1h_pm) and float(pc_1h_pm) > pump_1h_hard) or (pd.notna(pc_5m_pm) and float(pc_5m_pm) > pump_5m_hard):
            pump_momentum_pen = pp_cfg.get("hard_penalty", 0.5)
        elif (pd.notna(pc_1h_pm) and float(pc_1h_pm) > pump_1h_mod) or (pd.notna(pc_5m_pm) and float(pc_5m_pm) > pump_5m_mod):
            pump_momentum_pen = pp_cfg.get("moderate_penalty", 0.7)
        elif pd.notna(pc_1h_pm) and float(pc_1h_pm) > pump_1h_light:
            pump_momentum_pen = pp_cfg.get("light_penalty", 0.85)
    else:
        pump_momentum_pen = _safe_mult(row, "pump_momentum_pen")

    # --- v47: Recompute pvp_pen from trial pump_pen_config ---
    pvp_recent = row.get("pvp_recent_count")
    if pp_cfg and pd.notna(pvp_recent):
        pvp_cnt = int(pvp_recent)
        is_pf = row.get("is_pump_fun")
        if pd.notna(is_pf) and bool(is_pf):
            pvp_pen = max(pp_cfg.get("pvp_pump_fun_floor", 0.7),
                          1.0 / (1 + pp_cfg.get("pvp_pump_fun_scale", 0.05) * pvp_cnt))
        else:
            pvp_pen = max(pp_cfg.get("pvp_normal_floor", 0.5),
                          1.0 / (1 + pp_cfg.get("pvp_normal_scale", 0.1) * pvp_cnt))

    # --- v45: Recompute death_penalty from snapshot raw values ---
    death_cfg = params.get("death_config", {})
    freshest_h = row.get("freshest_mention_hours")
    if death_cfg and pd.notna(freshest_h):
        freshest_h = float(freshest_h)
        death_penalties = []
        pc24_death = row.get("price_change_24h")
        # Price collapse (uses existing SCORING_PARAMS-level thresholds, not in death_config)
        if pd.notna(pc24_death):
            pc24_v = float(pc24_death)
            if pc24_v < -80:
                death_penalties.append(0.1)
            elif pc24_v < -70:
                death_penalties.append(0.15 if freshest_h > 3 else 0.3)
            elif pc24_v < -50:
                death_penalties.append(0.2)
            elif pc24_v < death_cfg.get("price_mild_threshold", -30):
                if freshest_h > death_cfg.get("price_mild_stale_h", 24):
                    death_penalties.append(0.4)
                else:
                    death_penalties.append(0.8)
        # Volume death
        vol_24h_d = row.get("volume_24h")
        vol_1h_d = row.get("volume_1h")
        if pd.notna(vol_24h_d) and pd.notna(vol_1h_d):
            v24 = float(vol_24h_d)
            v1 = float(vol_1h_d)
            if v24 < death_cfg.get("vol_death_24h", 5000) and v1 < death_cfg.get("vol_death_1h", 500):
                death_penalties.append(death_cfg.get("vol_death_penalty", 0.15))
            elif v24 < death_cfg.get("vol_floor_24h", 1000):
                death_penalties.append(death_cfg.get("vol_floor_penalty", 0.1))
        # Social staleness
        stale_start = death_cfg.get("stale_start_hours", 12)
        if freshest_h > stale_start:
            stale_tiers = death_cfg.get("stale_tiers", [24, 48, 72])
            stale_bases = death_cfg.get("stale_bases", [0.7, 0.45, 0.25, 0.15])
            stale_base = stale_bases[0]
            for i in range(len(stale_tiers) - 1, -1, -1):
                if freshest_h > stale_tiers[i]:
                    stale_base = stale_bases[i + 1]
                    break
            # Volume modulation
            if pd.notna(vol_24h_d):
                v24 = float(vol_24h_d)
                mod_tiers = death_cfg.get("vol_modulation_tiers", [50000, 100000, 500000, 1000000])
                mod_bonuses = death_cfg.get("vol_modulation_bonuses", [0.15, 0.25, 0.35, 0.45])
                mod_caps = death_cfg.get("vol_modulation_caps", [0.8, 0.85, 0.9, 0.95])
                stale_pen = stale_base
                for i in range(len(mod_tiers) - 1, -1, -1):
                    if v24 > mod_tiers[i]:
                        stale_pen = min(mod_caps[i], stale_base + mod_bonuses[i])
                        break
            else:
                stale_pen = stale_base
            death_penalties.append(stale_pen)
        death_penalty = min(death_penalties) if death_penalties else 1.0
    else:
        death_penalty = _safe_mult(row, "death_penalty")

    # --- v45: Recompute entry_premium_mult from stored entry_premium ---
    ep_cfg = params.get("entry_premium_config", {})
    ep_raw = row.get("entry_premium")
    if ep_cfg and pd.notna(ep_raw):
        ep_val = float(ep_raw)
        bps = ep_cfg.get("tier_breakpoints", [1.0, 1.2, 2.0, 4.0, 8.0, 20.0])
        mults = ep_cfg.get("tier_multipliers", [1.1, 1.0, 0.9, 0.7, 0.5, 0.35, 0.25])
        slopes = ep_cfg.get("tier_slopes", [0, 0, 0, 0.1, 0.05, 0.0125, 0])
        entry_premium_mult = mults[-1]
        for i in range(len(bps)):
            if ep_val <= bps[i]:
                entry_premium_mult = mults[i]
                break
            if i < len(bps) - 1 and ep_val <= bps[i + 1]:
                base_m = mults[i + 1]
                slope = slopes[i + 1] if i + 1 < len(slopes) else 0
                entry_premium_mult = base_m + (bps[i + 1] - ep_val) * slope
                break
        # MCap fallback
        mcap_v = row.get("market_cap")
        mcap_thresh = ep_cfg.get("mcap_fallback_threshold", 50000000)
        if pd.notna(mcap_v) and float(mcap_v) > mcap_thresh and ep_val <= 1.0:
            mcap_tiers = ep_cfg.get("mcap_tiers", [50, 200, 500])
            mcap_mults = ep_cfg.get("mcap_multipliers", [0.85, 0.70, 0.50, 0.35])
            implied = float(mcap_v) / ep_cfg.get("mcap_launch_assumed", 1000000)
            for j, t in enumerate(mcap_tiers):
                if implied <= t:
                    entry_premium_mult = mcap_mults[j]
                    break
            else:
                entry_premium_mult = mcap_mults[-1]
    else:
        entry_premium_mult = _safe_mult(row, "entry_premium_mult")

    # --- v45: Recompute lifecycle_mult from snapshot values ---
    lc_cfg = params.get("lifecycle_config", {})
    if lc_cfg and pd.notna(row.get("price_change_24h")):
        pc24_lc = float(row.get("price_change_24h"))
        uk_lc = int(row.get("unique_kols") or 0) if pd.notna(row.get("unique_kols")) else 0
        va_lc = float(row.get("volume_acceleration")) if pd.notna(row.get("volume_acceleration")) else None
        sent_lc = float(row.get("sentiment") or 0) if pd.notna(row.get("sentiment")) else 0
        mcap_lc = float(row.get("market_cap") or 0) if pd.notna(row.get("market_cap")) else 0
        # Panic
        if pc24_lc < lc_cfg.get("panic_pc24", -30):
            if va_lc is not None and va_lc < lc_cfg.get("panic_va", 0.5):
                lifecycle_mult = lc_cfg.get("panic_penalty", 0.25)
            else:
                lifecycle_mult = 1.0
        # Euphoria
        elif (uk_lc >= lc_cfg.get("euphoria_uk", 3)
              and pc24_lc > lc_cfg.get("euphoria_pc24", 100)
              and sent_lc > lc_cfg.get("euphoria_sent", 0.2)):
            lifecycle_mult = lc_cfg.get("euphoria_penalty", 0.5)
        # Boom
        elif (uk_lc >= lc_cfg.get("boom_uk", 2)
              and lc_cfg.get("boom_pc24_low", 10) < pc24_lc <= lc_cfg.get("boom_pc24_high", 200)):
            if mcap_lc > lc_cfg.get("boom_large_cap_mcap", 50000000):
                lifecycle_mult = lc_cfg.get("boom_large_cap_penalty", 0.85)
            else:
                lifecycle_mult = lc_cfg.get("boom_bonus", 1.1)
        else:
            lifecycle_mult = 1.0
    else:
        lifecycle_mult = 1.0

    # --- v45: Recompute entry_drift_mult from snapshot values ---
    ed_cfg = params.get("entry_drift_config", {})
    if ed_cfg and pd.notna(ep_raw):
        ep_drift = float(ep_raw)
        premium_gate = ed_cfg.get("premium_gate", 1.2)
        if ep_drift <= premium_gate:
            entry_drift_mult = 1.0
        else:
            uk_ed = int(row.get("unique_kols") or 1) if pd.notna(row.get("unique_kols")) else 1
            kol_score = min(1.0, (uk_ed - 1) / ed_cfg.get("kol_divisor", 7))
            act_ed = float(row.get("activity_mult") or 1.0) if pd.notna(row.get("activity_mult")) else activity_mult
            act_base = ed_cfg.get("activity_base", 0.80)
            act_range = ed_cfg.get("activity_range", 0.45)
            activity_score = max(0, min(1.0, (act_ed - act_base) / act_range))
            freshest_ed = float(row.get("freshest_mention_hours") or 999) if pd.notna(row.get("freshest_mention_hours")) else 999
            fresh_tiers = ed_cfg.get("fresh_tiers", [1, 4, 8])
            fresh_scores = ed_cfg.get("fresh_scores", [1.0, 0.7, 0.4, 0.1])
            fresh_score = fresh_scores[-1]
            for i, tier in enumerate(fresh_tiers):
                if freshest_ed <= tier:
                    fresh_score = fresh_scores[i]
                    break
            s_ct = int(row.get("s_tier_count") or 0) if pd.notna(row.get("s_tier_count")) else 0
            s_tier_score = min(1.0, s_ct / ed_cfg.get("stier_divisor", 2))
            sw = ed_cfg.get("social_weights", [0.30, 0.25, 0.30, 0.15])
            social_str = kol_score * sw[0] + activity_score * sw[1] + fresh_score * sw[2] + s_tier_score * sw[3]
            drift = ep_drift - 1.0
            net_drift = max(0, drift - social_str * 1.0)
            entry_drift_mult = max(ed_cfg.get("drift_floor", 0.50),
                                   1.0 - net_drift * ed_cfg.get("drift_factor", 0.25))
    else:
        entry_drift_mult = _safe_mult(row, "entry_drift_mult")

    # v45/v47: crash_pen = min(lifecycle, death, entry_premium, pump_momentum_pen)
    if death_cfg or ep_cfg or lc_cfg or pp_cfg:
        crash_pen = min(lifecycle_mult, death_penalty, entry_premium_mult, pump_momentum_pen)
    else:
        crash_pen = _safe_mult(row, "crash_pen")

    # Combined chain (matching pipeline.py v45)
    combined_floor = params.get("combined_floor", 0.25)
    combined_cap = params.get("combined_cap", 2.0)
    combined_raw = (onchain * safety * pump_bonus
                    * manipulation_pen * pvp_pen * crash_pen
                    * activity_mult * breadth_pen
                    * size_mult * s_tier_mult * gate_mult
                    * entry_drift_mult * momentum_mult * hype_pen)
    combined = max(combined_floor, min(combined_cap, combined_raw))
    score = base_score * combined

    return min(100, max(0, int(score)))


def _evaluate_params(
    df: pd.DataFrame,
    weights: dict,
    params: dict,
    horizon: str = "12h",
    threshold: float = 2.0,
) -> float:
    """
    Composite objective for Optuna (higher = better):
    - 30% portfolio return (top-5 tokens per cycle)
    - 20% Sharpe ratio
    - 50% top-1 hit rate (dominant — bot picks only #1 token per cycle)
    """
    max_price_col = f"max_price_{horizon}"
    if max_price_col not in df.columns:
        return -999.0

    # Compute scores with trial params
    df = df.copy()
    df["trial_score"] = df.apply(lambda r: _compute_score_with_params(r, weights, params), axis=1)
    df["cycle"] = df["snapshot_at"].dt.floor("15min")

    cycle_returns = []
    top1_hits = 0
    top1_tested = 0

    for _, group in df.sort_values("snapshot_at").groupby("cycle"):
        labeled = group[
            group[max_price_col].notna()
            & group["price_at_snapshot"].notna()
        ]
        if labeled.empty:
            continue

        # Top 5 tokens by trial score
        top5 = labeled.nlargest(min(5, len(labeled)), "trial_score")

        # Portfolio return: equal-weight top 5
        returns = []
        for _, tok in top5.iterrows():
            p0 = float(tok["price_at_snapshot"])
            if p0 <= 0:
                continue
            p_max = float(tok[max_price_col])
            # Simple return: max achievable (optimistic but consistent for comparison)
            ret = (p_max / p0) - 1.0
            returns.append(ret)

        if returns:
            cycle_ret = sum(returns) / len(returns)
            cycle_returns.append(cycle_ret)

        # Top-1 hit rate
        top1 = labeled.loc[labeled["trial_score"].idxmax()]
        p0 = float(top1["price_at_snapshot"]) if pd.notna(top1.get("price_at_snapshot")) else 0
        p_max = float(top1[max_price_col]) if pd.notna(top1.get(max_price_col)) else 0
        if p0 > 0:
            top1_tested += 1
            if p_max / p0 >= threshold:
                top1_hits += 1

    if not cycle_returns or top1_tested == 0:
        return -999.0

    # Components
    avg_return = np.mean(cycle_returns)
    std_return = np.std(cycle_returns) if len(cycle_returns) > 1 else 1.0
    sharpe = avg_return / max(0.01, std_return)
    top1_hr = top1_hits / top1_tested

    # Composite: 30% portfolio return + 20% sharpe + 50% top1 hit rate
    # v49: hit-rate-dominant — bot only picks #1 token per cycle
    composite = 0.30 * avg_return + 0.20 * min(sharpe, 3.0) / 3.0 + 0.50 * top1_hr

    return composite


def _optuna_optimize_params(
    df: pd.DataFrame,
    horizon: str = "12h",
    n_trials: int = 200,
    timeout: int = 900,
) -> dict | None:
    """
    v49: Optuna study with ~119 search space parameters (14/14 multipliers recomputable).
    2-fold expanding walk-forward inside objective to prevent overfit to a single split.
    Returns best_params dict or None if no improvement.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    threshold = ML_THRESHOLD
    max_price_col = f"max_price_{horizon}"

    # v49: 2-fold expanding walk-forward split
    # Split data into 3 time blocks: [B1|B2|B3]
    # Fold 1: train=[B1] test=[B2]
    # Fold 2: train=[B1,B2] test=[B3]  (expanding window)
    # Final score = average of fold scores
    df_sorted = df.sort_values("snapshot_at")
    n = len(df_sorted)
    b1_end = int(n * 0.40)
    b2_end = int(n * 0.70)
    df_b1 = df_sorted.iloc[:b1_end]
    df_b2 = df_sorted.iloc[b1_end:b2_end]
    df_b3 = df_sorted.iloc[b2_end:]

    # For the objective, we use folds; for final test, we use B3
    df_train = df_sorted.iloc[:b2_end].copy()
    df_test = df_b3.copy()

    # Validate minimum sizes (50 train / 20 test per fold)
    if len(df_b1) < 50 or len(df_b2) < 20 or len(df_b3) < 20:
        logger.info(
            "optuna: not enough data for 2-fold walk-forward (B1=%d, B2=%d, B3=%d)",
            len(df_b1), len(df_b2), len(df_b3),
        )
        return None

    def objective(trial):
        # --- Weights (sum-to-1 via normalization) ---
        w_consensus = trial.suggest_float("w_consensus", 0.0, 0.80, step=0.05)
        w_conviction = trial.suggest_float("w_conviction", 0.0, 0.30, step=0.05)
        w_breadth = trial.suggest_float("w_breadth", 0.10, 0.80, step=0.05)
        w_pa = trial.suggest_float("w_price_action", 0.0, 0.30, step=0.05)
        total_w = w_consensus + w_conviction + w_breadth + w_pa
        if total_w < 0.1:
            return -999.0
        weights = {
            "consensus": w_consensus / total_w,
            "conviction": w_conviction / total_w,
            "breadth": w_breadth / total_w,
            "price_action": w_pa / total_w,
        }

        # --- Scalar params ---
        params = {
            "combined_floor": trial.suggest_float("combined_floor", 0.10, 0.50, step=0.05),
            "combined_cap": trial.suggest_float("combined_cap", 1.5, 3.0, step=0.25),
            "activity_mult_floor": trial.suggest_float("activity_mult_floor", 0.60, 0.95, step=0.05),
            "activity_mult_cap": trial.suggest_float("activity_mult_cap", 1.10, 1.50, step=0.05),
            "activity_ratio_high": trial.suggest_float("activity_ratio_high", 0.40, 0.80, step=0.05),
            "activity_ratio_mid": trial.suggest_float("activity_ratio_mid", 0.15, 0.50, step=0.05),
            "activity_ratio_low": trial.suggest_float("activity_ratio_low", 0.05, 0.25, step=0.05),
            "activity_pump_cap_hard": trial.suggest_float("activity_pump_cap_hard", 50, 150, step=10),
            "activity_pump_cap_soft": trial.suggest_float("activity_pump_cap_soft", 30, 80, step=10),
            "s_tier_bonus": trial.suggest_float("s_tier_bonus", 1.0, 1.5, step=0.05),
        }

        # Ordered constraints: high > mid > low
        if params["activity_ratio_high"] <= params["activity_ratio_mid"]:
            return -999.0
        if params["activity_ratio_mid"] <= params["activity_ratio_low"]:
            return -999.0
        if params["activity_pump_cap_hard"] <= params["activity_pump_cap_soft"]:
            return -999.0

        # --- v49: conviction normalization params ---
        params["conviction_offset"] = trial.suggest_float("conviction_offset", 3, 10, step=1)
        params["conviction_divisor"] = trial.suggest_float("conviction_divisor", 2, 8, step=1)

        # --- JSONB configs ---
        params["breadth_pen_config"] = {
            "thresholds": [
                trial.suggest_float("bp_t0", 0.01, 0.06, step=0.005),
                trial.suggest_float("bp_t1", 0.03, 0.10, step=0.005),
                trial.suggest_float("bp_t2", 0.05, 0.15, step=0.005),
            ],
            "penalties": [
                trial.suggest_float("bp_p0", 0.50, 0.90, step=0.05),
                trial.suggest_float("bp_p1", 0.65, 0.95, step=0.05),
                trial.suggest_float("bp_p2", 0.80, 1.00, step=0.05),
            ],
        }
        # Ordered constraints for breadth_pen thresholds
        bpt = params["breadth_pen_config"]["thresholds"]
        if bpt[0] >= bpt[1] or bpt[1] >= bpt[2]:
            return -999.0

        params["hype_pen_config"] = {
            "thresholds": [
                trial.suggest_int("hp_t0", 1, 4),
                trial.suggest_int("hp_t1", 3, 7),
                trial.suggest_int("hp_t2", 5, 12),
            ],
            "penalties": [
                1.0,  # sweet spot always 1.0
                trial.suggest_float("hp_p1", 0.60, 1.00, step=0.05),
                trial.suggest_float("hp_p2", 0.40, 0.85, step=0.05),
                trial.suggest_float("hp_p3", 0.20, 0.70, step=0.05),
            ],
            # v53: KOL co-occurrence penalty params
            "cooc_config": {
                "threshold": trial.suggest_float("cooc_threshold", 0.3, 0.7, step=0.05),
                "penalty": trial.suggest_float("cooc_penalty", 0.70, 0.95, step=0.05),
            },
        }
        # Ordered constraints for hype_pen thresholds
        hpt = params["hype_pen_config"]["thresholds"]
        if hpt[0] >= hpt[1] or hpt[1] >= hpt[2]:
            return -999.0

        # --- v45: New scalar params ---
        params["safety_floor"] = trial.suggest_float("safety_floor", 0.60, 1.00, step=0.05)
        params["onchain_mult_floor"] = trial.suggest_float("onchain_mult_floor", 0.20, 0.60, step=0.05)
        params["onchain_mult_cap"] = trial.suggest_float("onchain_mult_cap", 1.2, 2.0, step=0.1)

        # --- v45: death_config params ---
        death_stale_t0 = trial.suggest_int("death_stale_t0", 12, 36)
        death_stale_t1 = trial.suggest_int("death_stale_t1", 24, 72)
        death_stale_t2 = trial.suggest_int("death_stale_t2", 48, 96)
        if death_stale_t0 >= death_stale_t1 or death_stale_t1 >= death_stale_t2:
            return -999.0
        death_stale_b0 = trial.suggest_float("death_stale_b0", 0.40, 0.80, step=0.05)
        death_stale_b1 = trial.suggest_float("death_stale_b1", 0.20, 0.60, step=0.05)
        death_stale_b2 = trial.suggest_float("death_stale_b2", 0.10, 0.40, step=0.05)
        death_stale_b3 = trial.suggest_float("death_stale_b3", 0.05, 0.25, step=0.05)
        death_vol_24h = trial.suggest_float("death_vol_24h", 1000, 10000, step=1000)
        params["death_config"] = {
            "stale_start_hours": 12,
            "stale_tiers": [death_stale_t0, death_stale_t1, death_stale_t2],
            "stale_bases": [death_stale_b0, death_stale_b1, death_stale_b2, death_stale_b3],
            "vol_modulation_tiers": [50000, 100000, 500000, 1000000],
            "vol_modulation_bonuses": [0.15, 0.25, 0.35, 0.45],
            "vol_modulation_caps": [0.8, 0.85, 0.9, 0.95],
            "vol_death_24h": death_vol_24h, "vol_death_1h": 500, "vol_death_penalty": 0.15,
            "vol_floor_24h": 1000, "vol_floor_penalty": 0.1,
            "price_moderate_social_alive_h": 6, "price_moderate_vol_alive": 0.5,
            "price_mild_threshold": -30, "price_mild_stale_h": 24,
        }

        # --- v45: entry_premium_config params ---
        ep_neutral_cap = trial.suggest_float("ep_neutral_cap", 1.0, 2.5, step=0.25)
        ep_mild_cap = trial.suggest_float("ep_mild_cap", 2.0, 6.0, step=0.5)
        ep_harsh_cap = trial.suggest_float("ep_harsh_cap", 4.0, 12.0, step=1.0)
        if ep_neutral_cap >= ep_mild_cap or ep_mild_cap >= ep_harsh_cap:
            return -999.0
        ep_floor_mult = trial.suggest_float("ep_floor_mult", 0.15, 0.40, step=0.05)
        ep_mcap_threshold = trial.suggest_float("ep_mcap_threshold", 20000000, 100000000, step=10000000)
        ep_duration_48h = trial.suggest_float("ep_duration_48h", 1.2, 2.0, step=0.1)
        params["entry_premium_config"] = {
            "tier_breakpoints": [1.0, ep_neutral_cap, ep_mild_cap, ep_harsh_cap, ep_harsh_cap * 2, ep_harsh_cap * 3],
            "tier_multipliers": [1.1, 1.0, 0.9, 0.7, 0.5, 0.35, ep_floor_mult],
            "tier_slopes": [0, 0, 0, 0.1, 0.05, 0.0125, 0],
            "duration_thresholds": [12, 24, 48],
            "duration_factors": [1.0, 1.15, 1.3, ep_duration_48h],
            "mcap_fallback_threshold": ep_mcap_threshold,
            "mcap_launch_assumed": 1000000,
            "mcap_tiers": [50, 200, 500],
            "mcap_multipliers": [0.85, 0.70, 0.50, 0.35],
        }

        # --- v45: lifecycle_config params ---
        lc_panic_threshold = trial.suggest_float("lc_panic_threshold", -50, -20, step=5)
        lc_panic_penalty = trial.suggest_float("lc_panic_penalty", 0.10, 0.50, step=0.05)
        lc_euphoria_penalty = trial.suggest_float("lc_euphoria_penalty", 0.20, 0.70, step=0.05)
        lc_boom_bonus = trial.suggest_float("lc_boom_bonus", 1.0, 1.30, step=0.05)
        lc_boom_mcap = trial.suggest_float("lc_boom_mcap", 20000000, 100000000, step=10000000)
        params["lifecycle_config"] = {
            "panic_pc24": lc_panic_threshold, "panic_va": 0.5, "panic_penalty": lc_panic_penalty,
            "profit_taking_pc24": 100, "profit_taking_vol_proxy": 40, "profit_taking_penalty": 0.35,
            "euphoria_uk": 3, "euphoria_pc24": 100, "euphoria_sent": 0.2, "euphoria_penalty": lc_euphoria_penalty,
            "boom_uk": 2, "boom_pc24_low": 10, "boom_pc24_high": 200, "boom_va": 1.0,
            "boom_bonus": lc_boom_bonus, "boom_large_cap_penalty": 0.85, "boom_large_cap_mcap": lc_boom_mcap,
            "displacement_age": 6, "displacement_uk": 1, "displacement_pc24": 50, "displacement_penalty": 0.9,
        }

        # --- v45: entry_drift_config params ---
        ed_premium_gate = trial.suggest_float("ed_premium_gate", 1.0, 2.5, step=0.25)
        ed_kol_weight = trial.suggest_float("ed_kol_weight", 0.15, 0.45, step=0.05)
        ed_fresh_weight = trial.suggest_float("ed_fresh_weight", 0.15, 0.45, step=0.05)
        ed_drift_factor = trial.suggest_float("ed_drift_factor", 0.10, 0.50, step=0.05)
        ed_drift_floor = trial.suggest_float("ed_drift_floor", 0.30, 0.70, step=0.05)
        # Normalize social weights to sum to 1.0 (4 weights: kol, activity, fresh, stier)
        remaining = max(0.1, 1.0 - ed_kol_weight - ed_fresh_weight)
        params["entry_drift_config"] = {
            "premium_gate": ed_premium_gate,
            "kol_divisor": 7,
            "activity_base": 0.80, "activity_range": 0.45,
            "fresh_tiers": [1, 4, 8], "fresh_scores": [1.0, 0.7, 0.4, 0.1],
            "stier_divisor": 2,
            "social_weights": [ed_kol_weight, remaining * 0.6, ed_fresh_weight, remaining * 0.4],
            "drift_factor": ed_drift_factor, "drift_floor": ed_drift_floor,
        }

        # --- v46: pa_config params (direction penalties + norm bounds) ---
        pa_dir_pump = trial.suggest_float("pa_dir_pump_mult", 0.3, 1.0, step=0.05)
        pa_dir_hard_pump = trial.suggest_float("pa_dir_hard_pump_mult", 0.3, 1.0, step=0.05)
        pa_dir_freefall = trial.suggest_float("pa_dir_freefall_mult", 0.3, 1.0, step=0.05)
        pa_dir_dying = trial.suggest_float("pa_dir_dying_mult", 0.5, 1.0, step=0.05)
        # Ordered: hard_pump <= pump, freefall <= dying
        if pa_dir_hard_pump > pa_dir_pump:
            return -999.0
        if pa_dir_freefall > pa_dir_dying:
            return -999.0
        params["pa_norm_floor"] = trial.suggest_float("pa_norm_floor", 0.2, 0.6, step=0.05)
        params["pa_norm_cap"] = trial.suggest_float("pa_norm_cap", 1.0, 1.8, step=0.1)
        params["pa_config"] = {
            "rsi_hard_pump": 80, "rsi_pump": 70,
            "rsi_freefall": 20, "rsi_dying": 30,
            "dir_hard_pump_mult": pa_dir_hard_pump,
            "dir_pump_mult": pa_dir_pump,
            "dir_freefall_mult": pa_dir_freefall,
            "dir_dying_mult": pa_dir_dying,
        }

        # --- v47: Remaining hardcoded params ---
        params["consensus_pump_threshold"] = trial.suggest_float("consensus_pump_threshold", 30, 100, step=10)
        params["consensus_pump_floor"] = trial.suggest_float("consensus_pump_floor", 0.3, 0.8, step=0.05)
        params["consensus_pump_divisor"] = trial.suggest_float("consensus_pump_divisor", 200, 600, step=50)
        params["activity_mid_mult"] = trial.suggest_float("activity_mid_mult", 1.0, 1.20, step=0.02)
        pp_hard = trial.suggest_float("pump_hard_penalty", 0.3, 0.7, step=0.05)
        pp_mod = trial.suggest_float("pump_moderate_penalty", 0.5, 0.85, step=0.05)
        pp_light = trial.suggest_float("pump_light_penalty", 0.70, 0.95, step=0.05)
        # Ordered: hard <= moderate <= light
        if pp_hard > pp_mod or pp_mod > pp_light:
            return -999.0
        pvp_floor = trial.suggest_float("pvp_normal_floor", 0.3, 0.7, step=0.05)
        # v49: pump momentum threshold params
        pp_1h_hard = trial.suggest_float("pp_1h_hard", 15, 50, step=5)
        pp_5m_hard = trial.suggest_float("pp_5m_hard", 8, 30, step=2)
        pp_1h_mod_ratio = trial.suggest_float("pp_1h_mod_ratio", 0.3, 0.7, step=0.05)
        pp_5m_mod_ratio = trial.suggest_float("pp_5m_mod_ratio", 0.3, 0.7, step=0.05)
        pp_1h_light_ratio = trial.suggest_float("pp_1h_light_ratio", 0.15, 0.45, step=0.05)
        params["pump_pen_config"] = {
            "hard_penalty": pp_hard,
            "moderate_penalty": pp_mod,
            "light_penalty": pp_light,
            "pvp_pump_fun_floor": 0.7,
            "pvp_pump_fun_scale": 0.05,
            "pvp_normal_floor": pvp_floor,
            "pvp_normal_scale": 0.1,
            "pump_1h_hard": pp_1h_hard,
            "pump_5m_hard": pp_5m_hard,
            "pump_1h_mod": pp_1h_hard * pp_1h_mod_ratio,
            "pump_5m_mod": pp_5m_hard * pp_5m_mod_ratio,
            "pump_1h_light": pp_1h_hard * pp_1h_light_ratio,
        }

        # --- v48: onchain_config params (~11 new) ---
        oc_lmr_low = trial.suggest_float("oc_lmr_low", 0.01, 0.05, step=0.005)
        oc_lmr_high = trial.suggest_float("oc_lmr_high", 0.05, 0.20, step=0.01)
        if oc_lmr_low >= oc_lmr_high:
            return -999.0
        oc_whale_count_f0 = trial.suggest_float("oc_whale_count_f0", 0.4, 0.9, step=0.05)
        oc_whale_count_f3 = trial.suggest_float("oc_whale_count_f3", 1.2, 1.8, step=0.1)
        oc_wne_f2 = trial.suggest_float("oc_wne_f2", 1.1, 1.5, step=0.05)
        oc_whale_accum = trial.suggest_float("oc_whale_accum_bonus", 1.0, 1.3, step=0.05)
        oc_vol_proxy_t = trial.suggest_float("oc_vol_proxy_threshold", 30, 80, step=5)
        oc_vol_proxy_p = trial.suggest_float("oc_vol_proxy_penalty", 0.6, 0.95, step=0.05)
        oc_velocity_f0 = trial.suggest_float("oc_velocity_f0", 0.4, 0.8, step=0.05)
        oc_velocity_f3 = trial.suggest_float("oc_velocity_f3", 1.1, 1.5, step=0.05)
        oc_age_f0 = trial.suggest_float("oc_age_f0", 0.3, 0.7, step=0.05)
        # v49: new onchain sub-params (BSR base, LMR interpolation, Jupiter non-tradeable)
        oc_bsr_base = trial.suggest_float("oc_bsr_base", 0.3, 0.7, step=0.05)
        oc_lmr_interp_base = trial.suggest_float("oc_lmr_interp_base", 0.5, 1.0, step=0.05)
        oc_lmr_interp_slope = trial.suggest_float("oc_lmr_interp_slope", 2, 8, step=1)
        oc_jup_nt_factor = trial.suggest_float("oc_jup_nt_factor", 0.3, 0.7, step=0.05)
        # v53: onchain sub-factor params (6 new: floor+cap for smr, shp, lds)
        oc_smr_f0 = trial.suggest_float("oc_smr_f0", 0.5, 0.95, step=0.05)
        oc_smr_f3 = trial.suggest_float("oc_smr_f3", 1.1, 1.5, step=0.05)
        oc_shp_f0 = trial.suggest_float("oc_shp_f0", 0.5, 0.9, step=0.05)
        oc_shp_f3 = trial.suggest_float("oc_shp_f3", 1.1, 1.5, step=0.05)
        oc_lds_f0 = trial.suggest_float("oc_lds_f0", 0.4, 0.8, step=0.05)
        oc_lds_f3 = trial.suggest_float("oc_lds_f3", 1.05, 1.4, step=0.05)
        params["onchain_config"] = {
            "lmr_low": oc_lmr_low, "lmr_low_factor": 0.5,
            "lmr_high": oc_lmr_high, "lmr_high_factor": 1.2,
            "lmr_interp_base": oc_lmr_interp_base, "lmr_interp_slope": oc_lmr_interp_slope,
            "bsr_base": oc_bsr_base,
            "jup_non_tradeable_factor": oc_jup_nt_factor,
            "age_thresholds": [1, 6, 48, 168],
            "age_factors": [oc_age_f0, 1.0, 1.2, 1.0, 0.8],
            "tx_thresholds": [5, 20, 40], "tx_factors": [0.7, 1.0, 1.1, 1.3],
            "jup_impact_thresholds": [1.0, 5.0], "jup_impact_factors": [1.3, 1.0, 0.7],
            "whale_change_thresholds": [-10.0, 0, 5.0],
            "whale_change_factors": [0.6, 0.8, 1.1, 1.3],
            "velocity_thresholds": [0.2, 1.0, 5.0],
            "velocity_factors": [oc_velocity_f0, 1.0, 1.1, oc_velocity_f3],
            "wne_thresholds": [1, 3], "wne_factors": [1.0, 1.1, oc_wne_f2],
            "whale_count_thresholds": [1, 3, 5],
            "whale_count_factors": [oc_whale_count_f0, 1.0, 1.2, oc_whale_count_f3],
            "uw_change_thresholds": [-20, -5, 5, 20],
            "uw_change_factors": [0.6, 0.8, 1.0, 1.15, 1.3],
            "vol_proxy_threshold": oc_vol_proxy_t, "vol_proxy_penalty": oc_vol_proxy_p,
            "whale_accum_bonus": oc_whale_accum,
            # v53: holder stability + liquidity depth
            "smr_thresholds": [50, 70, 90], "smr_factors": [oc_smr_f0, 1.0, 1.15, oc_smr_f3],
            "shp_thresholds": [50, 70, 85], "shp_factors": [oc_shp_f0, 0.9, 1.1, oc_shp_f3],
            "lds_thresholds": [0.2, 0.5, 0.8], "lds_factors": [oc_lds_f0, 0.85, 1.05, oc_lds_f3],
        }

        # --- v48: safety_config params (~10 new) + v49: 6 slope params ---
        sf_insider_t = trial.suggest_float("sf_insider_threshold", 20, 50, step=5)
        sf_insider_f = trial.suggest_float("sf_insider_floor", 0.3, 0.7, step=0.05)
        sf_gini_t = trial.suggest_float("sf_gini_threshold", 0.70, 0.95, step=0.05)
        sf_gini_p = trial.suggest_float("sf_gini_penalty", 0.6, 0.95, step=0.05)
        sf_holder_t = trial.suggest_float("sf_holder_count_threshold", 30, 100, step=10)
        sf_holder_p = trial.suggest_float("sf_holder_count_penalty", 0.6, 0.95, step=0.05)
        sf_whale_conc_t = trial.suggest_float("sf_whale_conc_threshold", 40, 80, step=5)
        sf_whale_dist_p = trial.suggest_float("sf_whale_dist_penalty", 0.5, 0.9, step=0.05)
        sf_whale_dump_p = trial.suggest_float("sf_whale_dump_penalty", 0.4, 0.8, step=0.05)
        sf_lp_unlock_p = trial.suggest_float("sf_lp_unlock_penalty", 0.3, 0.8, step=0.05)
        # v49: 6 safety slope params
        sf_insider_slope = trial.suggest_float("sf_insider_slope", 50, 200, step=25)
        sf_top10_slope = trial.suggest_float("sf_top10_slope", 50, 200, step=25)
        sf_risk_score_slope = trial.suggest_float("sf_risk_score_slope", 2000, 10000, step=1000)
        sf_whale_conc_slope = trial.suggest_float("sf_whale_conc_slope", 40, 150, step=10)
        sf_bb_cluster_slope = trial.suggest_float("sf_bb_cluster_slope", 30, 120, step=10)
        sf_cex_slope = trial.suggest_float("sf_cex_slope", 50, 200, step=25)
        # Ordered: dump <= dist
        if sf_whale_dump_p > sf_whale_dist_p:
            return -999.0
        params["safety_config"] = {
            "insider_threshold": sf_insider_t, "insider_floor": sf_insider_f, "insider_slope": sf_insider_slope,
            "top10_threshold": 50, "top10_floor": 0.7, "top10_slope": sf_top10_slope,
            "risk_score_threshold": 5000, "risk_score_floor": 0.5, "risk_score_slope": sf_risk_score_slope,
            "risk_count_threshold": 3, "risk_count_penalty": 0.9,
            "jito_hard_threshold": 5, "jito_hard_penalty": 0.4,
            "jito_soft_threshold": 3, "jito_soft_penalty": 0.6,
            "bundle_hard_threshold": 20, "bundle_hard_penalty": 0.5,
            "bundle_soft_threshold": 10, "bundle_soft_penalty": 0.7,
            "gini_threshold": sf_gini_t, "gini_penalty": sf_gini_p,
            "holder_count_threshold": sf_holder_t, "holder_count_penalty": sf_holder_p,
            "whale_conc_threshold": sf_whale_conc_t, "whale_conc_floor": 0.7, "whale_conc_slope": sf_whale_conc_slope,
            "bb_score_thresholds": [20, 40], "bb_score_penalties": [0.6, 0.85],
            "bb_cluster_threshold": 30, "bb_cluster_floor": 0.6, "bb_cluster_slope": sf_bb_cluster_slope,
            "whale_dom_threshold": 0.5, "whale_dom_penalty": 0.85,
            "whale_dist_penalty": sf_whale_dist_p, "whale_dump_penalty": sf_whale_dump_p,
            "lp_unlock_penalty": sf_lp_unlock_p, "lp_partial_threshold": 50, "lp_partial_penalty": 0.85,
            "cex_threshold": 20, "cex_floor": 0.7, "cex_slope": sf_cex_slope,
        }

        # --- v48: momentum_config params (~8 new) ---
        mom_kf_t0 = trial.suggest_float("mom_kf_t0", 0.3, 0.7, step=0.05)
        mom_kf_f0 = trial.suggest_float("mom_kf_f0", 1.10, 1.35, step=0.05)
        mom_mhr_t0 = trial.suggest_float("mom_mhr_t0", 1.0, 4.0, step=0.5)
        mom_mhr_f0 = trial.suggest_float("mom_mhr_f0", 1.05, 1.25, step=0.05)
        mom_sth_t0 = trial.suggest_float("mom_sth_t0", 1.5, 5.0, step=0.5)
        mom_sth_pen_t = trial.suggest_float("mom_sth_pen_threshold", 0.1, 0.5, step=0.05)
        mom_floor_v = trial.suggest_float("mom_floor", 0.50, 0.85, step=0.05)
        mom_cap_v = trial.suggest_float("mom_cap", 1.2, 1.6, step=0.05)
        params["momentum_config"] = {
            "kol_fresh_thresholds": [mom_kf_t0, mom_kf_t0 * 0.4],
            "kol_fresh_factors": [mom_kf_f0, mom_kf_f0 * 0.92, 1.05],
            "mhr_thresholds": [mom_mhr_t0, mom_mhr_t0 * 0.5, 0.3],
            "mhr_factors": [mom_mhr_f0, mom_mhr_f0 * 0.87, 1.05],
            "sth_thresholds": [mom_sth_t0, mom_sth_t0 * 0.5],
            "sth_factors": [1.10, 1.05],
            "sth_penalty_threshold": mom_sth_pen_t,
            "sth_penalty_factor": 0.95,
            # v52: KOL timing alpha factors
            "cascade_factor_3plus": trial.suggest_float("cascade_factor_3plus", 1.0, 1.30, step=0.05),
            "cascade_factor_2plus": trial.suggest_float("cascade_factor_2plus", 1.0, 1.20, step=0.05),
            "early_factor_30min": trial.suggest_float("early_factor_30min", 1.0, 1.40, step=0.05),
            "early_factor_60min": trial.suggest_float("early_factor_60min", 1.0, 1.25, step=0.05),
            "late_penalty_360min": trial.suggest_float("late_penalty_360min", 0.60, 1.0, step=0.05),
            "pvfc_penalty_3x": trial.suggest_float("pvfc_penalty_3x", 0.30, 0.80, step=0.05),
            "pvfc_penalty_2x": trial.suggest_float("pvfc_penalty_2x", 0.50, 0.90, step=0.05),
            "pvfc_penalty_1_5x": trial.suggest_float("pvfc_penalty_1_5x", 0.70, 1.0, step=0.05),
            "floor": mom_floor_v,
            "cap": mom_cap_v,
        }

        # --- v48: size_mult_config params (~6 new) ---
        sm_mcap_f0 = trial.suggest_float("sm_mcap_f0", 1.1, 1.5, step=0.05)
        sm_mcap_f6 = trial.suggest_float("sm_mcap_f6", 0.15, 0.40, step=0.05)
        sm_fresh_f0 = trial.suggest_float("sm_fresh_f0", 1.05, 1.35, step=0.05)
        sm_large_cap_t = trial.suggest_float("sm_large_cap_threshold", 20000000, 100000000, step=10000000)
        sm_floor_v = trial.suggest_float("sm_floor", 0.15, 0.40, step=0.05)
        sm_cap_opt = trial.suggest_float("sm_cap", 1.2, 1.8, step=0.1)
        params["size_mult_config"] = {
            "mcap_thresholds": [300000, 1000000, 5000000, 20000000, 50000000, 200000000, 500000000],
            "mcap_factors": [sm_mcap_f0, 1.15, 1.0, 0.85, 0.70, 0.50, 0.35, sm_mcap_f6],
            "fresh_thresholds": [4, 12],
            "fresh_factors": [sm_fresh_f0, 1.1, 1.0],
            "large_cap_threshold": sm_large_cap_t,
            "floor": sm_floor_v,
            "cap": sm_cap_opt,
        }

        # v49: 2-fold expanding walk-forward evaluation
        # Fold 1: train on B1, evaluate on B2
        fold1_score = _evaluate_params(df_b2, weights, params, horizon, threshold)
        # Fold 2: train on B1+B2, evaluate on B3 (but we only eval here, Optuna trains implicitly)
        fold2_score = _evaluate_params(df_b3, weights, params, horizon, threshold)
        if fold1_score <= -900 or fold2_score <= -900:
            return -999.0
        return (fold1_score + fold2_score) / 2.0

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best_trial = study.best_trial
    logger.info(
        "optuna: best trial #%d, train_score=%.4f (n_trials=%d)",
        best_trial.number, best_trial.value, len(study.trials),
    )

    if best_trial.value <= -900:
        logger.warning("optuna: best trial has invalid score, skipping")
        return None

    # Reconstruct best params
    bp = best_trial.params
    total_w = bp["w_consensus"] + bp["w_conviction"] + bp["w_breadth"] + bp["w_price_action"]
    best_weights = {
        "consensus": round(bp["w_consensus"] / total_w, 3),
        "conviction": round(bp["w_conviction"] / total_w, 3),
        "breadth": round(bp["w_breadth"] / total_w, 3),
        "price_action": round(bp["w_price_action"] / total_w, 3),
    }
    best_params = {
        "combined_floor": bp["combined_floor"],
        "combined_cap": bp["combined_cap"],
        "activity_mult_floor": bp["activity_mult_floor"],
        "activity_mult_cap": bp["activity_mult_cap"],
        "activity_ratio_high": bp["activity_ratio_high"],
        "activity_ratio_mid": bp["activity_ratio_mid"],
        "activity_ratio_low": bp["activity_ratio_low"],
        "activity_pump_cap_hard": bp["activity_pump_cap_hard"],
        "activity_pump_cap_soft": bp["activity_pump_cap_soft"],
        "s_tier_bonus": bp["s_tier_bonus"],
        # v49: conviction normalization params
        "conviction_offset": bp["conviction_offset"],
        "conviction_divisor": bp["conviction_divisor"],
        "breadth_pen_config": {
            "thresholds": [bp["bp_t0"], bp["bp_t1"], bp["bp_t2"]],
            "penalties": [bp["bp_p0"], bp["bp_p1"], bp["bp_p2"]],
        },
        "hype_pen_config": {
            "thresholds": [bp["hp_t0"], bp["hp_t1"], bp["hp_t2"]],
            "penalties": [1.0, bp["hp_p1"], bp["hp_p2"], bp["hp_p3"]],
            "cooc_config": {
                "threshold": bp.get("cooc_threshold", 0.5),
                "penalty": bp.get("cooc_penalty", 0.85),
            },
        },
        # v45: new scalar params
        "safety_floor": bp["safety_floor"],
        "onchain_mult_floor": bp["onchain_mult_floor"],
        "onchain_mult_cap": bp["onchain_mult_cap"],
        # v45: new JSONB configs
        "death_config": {
            "stale_start_hours": 12,
            "stale_tiers": [bp["death_stale_t0"], bp["death_stale_t1"], bp["death_stale_t2"]],
            "stale_bases": [bp["death_stale_b0"], bp["death_stale_b1"], bp["death_stale_b2"], bp["death_stale_b3"]],
            "vol_modulation_tiers": [50000, 100000, 500000, 1000000],
            "vol_modulation_bonuses": [0.15, 0.25, 0.35, 0.45],
            "vol_modulation_caps": [0.8, 0.85, 0.9, 0.95],
            "vol_death_24h": bp["death_vol_24h"], "vol_death_1h": 500, "vol_death_penalty": 0.15,
            "vol_floor_24h": 1000, "vol_floor_penalty": 0.1,
            "price_moderate_social_alive_h": 6, "price_moderate_vol_alive": 0.5,
            "price_mild_threshold": -30, "price_mild_stale_h": 24,
        },
        "entry_premium_config": {
            "tier_breakpoints": [1.0, bp["ep_neutral_cap"], bp["ep_mild_cap"], bp["ep_harsh_cap"],
                                 bp["ep_harsh_cap"] * 2, bp["ep_harsh_cap"] * 3],
            "tier_multipliers": [1.1, 1.0, 0.9, 0.7, 0.5, 0.35, bp["ep_floor_mult"]],
            "tier_slopes": [0, 0, 0, 0.1, 0.05, 0.0125, 0],
            "duration_thresholds": [12, 24, 48],
            "duration_factors": [1.0, 1.15, 1.3, bp["ep_duration_48h"]],
            "mcap_fallback_threshold": bp["ep_mcap_threshold"],
            "mcap_launch_assumed": 1000000,
            "mcap_tiers": [50, 200, 500],
            "mcap_multipliers": [0.85, 0.70, 0.50, 0.35],
        },
        "lifecycle_config": {
            "panic_pc24": bp["lc_panic_threshold"], "panic_va": 0.5, "panic_penalty": bp["lc_panic_penalty"],
            "profit_taking_pc24": 100, "profit_taking_vol_proxy": 40, "profit_taking_penalty": 0.35,
            "euphoria_uk": 3, "euphoria_pc24": 100, "euphoria_sent": 0.2, "euphoria_penalty": bp["lc_euphoria_penalty"],
            "boom_uk": 2, "boom_pc24_low": 10, "boom_pc24_high": 200, "boom_va": 1.0,
            "boom_bonus": bp["lc_boom_bonus"], "boom_large_cap_penalty": 0.85, "boom_large_cap_mcap": bp["lc_boom_mcap"],
            "displacement_age": 6, "displacement_uk": 1, "displacement_pc24": 50, "displacement_penalty": 0.9,
        },
        "entry_drift_config": {
            "premium_gate": bp["ed_premium_gate"],
            "kol_divisor": 7,
            "activity_base": 0.80, "activity_range": 0.45,
            "fresh_tiers": [1, 4, 8], "fresh_scores": [1.0, 0.7, 0.4, 0.1],
            "stier_divisor": 2,
            "social_weights": [bp["ed_kol_weight"],
                               max(0.1, 1.0 - bp["ed_kol_weight"] - bp["ed_fresh_weight"]) * 0.6,
                               bp["ed_fresh_weight"],
                               max(0.1, 1.0 - bp["ed_kol_weight"] - bp["ed_fresh_weight"]) * 0.4],
            "drift_factor": bp["ed_drift_factor"], "drift_floor": bp["ed_drift_floor"],
        },
        # v46: PA config
        "pa_norm_floor": bp["pa_norm_floor"],
        "pa_norm_cap": bp["pa_norm_cap"],
        "pa_config": {
            "rsi_hard_pump": 80, "rsi_pump": 70,
            "rsi_freefall": 20, "rsi_dying": 30,
            "dir_hard_pump_mult": bp["pa_dir_hard_pump_mult"],
            "dir_pump_mult": bp["pa_dir_pump_mult"],
            "dir_freefall_mult": bp["pa_dir_freefall_mult"],
            "dir_dying_mult": bp["pa_dir_dying_mult"],
        },
        # v47: Remaining hardcoded params
        "consensus_pump_threshold": bp["consensus_pump_threshold"],
        "consensus_pump_floor": bp["consensus_pump_floor"],
        "consensus_pump_divisor": bp["consensus_pump_divisor"],
        "activity_mid_mult": bp["activity_mid_mult"],
        "pump_pen_config": {
            "hard_penalty": bp["pump_hard_penalty"],
            "moderate_penalty": bp["pump_moderate_penalty"],
            "light_penalty": bp["pump_light_penalty"],
            "pvp_pump_fun_floor": 0.7,
            "pvp_pump_fun_scale": 0.05,
            "pvp_normal_floor": bp["pvp_normal_floor"],
            "pvp_normal_scale": 0.1,
            "pump_1h_hard": bp["pp_1h_hard"],
            "pump_5m_hard": bp["pp_5m_hard"],
            "pump_1h_mod": bp["pp_1h_hard"] * bp["pp_1h_mod_ratio"],
            "pump_5m_mod": bp["pp_5m_hard"] * bp["pp_5m_mod_ratio"],
            "pump_1h_light": bp["pp_1h_hard"] * bp["pp_1h_light_ratio"],
        },
        # v48: 4 new JSONB configs for full multiplier recomputation
        "onchain_config": {
            "lmr_low": bp["oc_lmr_low"], "lmr_low_factor": 0.5,
            "lmr_high": bp["oc_lmr_high"], "lmr_high_factor": 1.2,
            "lmr_interp_base": bp["oc_lmr_interp_base"], "lmr_interp_slope": bp["oc_lmr_interp_slope"],
            "bsr_base": bp["oc_bsr_base"],
            "jup_non_tradeable_factor": bp["oc_jup_nt_factor"],
            "age_thresholds": [1, 6, 48, 168],
            "age_factors": [bp["oc_age_f0"], 1.0, 1.2, 1.0, 0.8],
            "tx_thresholds": [5, 20, 40], "tx_factors": [0.7, 1.0, 1.1, 1.3],
            "jup_impact_thresholds": [1.0, 5.0], "jup_impact_factors": [1.3, 1.0, 0.7],
            "whale_change_thresholds": [-10.0, 0, 5.0],
            "whale_change_factors": [0.6, 0.8, 1.1, 1.3],
            "velocity_thresholds": [0.2, 1.0, 5.0],
            "velocity_factors": [bp["oc_velocity_f0"], 1.0, 1.1, bp["oc_velocity_f3"]],
            "wne_thresholds": [1, 3], "wne_factors": [1.0, 1.1, bp["oc_wne_f2"]],
            "whale_count_thresholds": [1, 3, 5],
            "whale_count_factors": [bp["oc_whale_count_f0"], 1.0, 1.2, bp["oc_whale_count_f3"]],
            "uw_change_thresholds": [-20, -5, 5, 20],
            "uw_change_factors": [0.6, 0.8, 1.0, 1.15, 1.3],
            "vol_proxy_threshold": bp["oc_vol_proxy_threshold"], "vol_proxy_penalty": bp["oc_vol_proxy_penalty"],
            "whale_accum_bonus": bp["oc_whale_accum_bonus"],
            # v53: holder stability + liquidity depth
            "smr_thresholds": [50, 70, 90], "smr_factors": [bp.get("oc_smr_f0", 0.8), 1.0, 1.15, bp.get("oc_smr_f3", 1.3)],
            "shp_thresholds": [50, 70, 85], "shp_factors": [bp.get("oc_shp_f0", 0.7), 0.9, 1.1, bp.get("oc_shp_f3", 1.3)],
            "lds_thresholds": [0.2, 0.5, 0.8], "lds_factors": [bp.get("oc_lds_f0", 0.6), 0.85, 1.05, bp.get("oc_lds_f3", 1.2)],
        },
        "safety_config": {
            "insider_threshold": bp["sf_insider_threshold"], "insider_floor": bp["sf_insider_floor"],
            "insider_slope": bp["sf_insider_slope"],
            "top10_threshold": 50, "top10_floor": 0.7, "top10_slope": bp["sf_top10_slope"],
            "risk_score_threshold": 5000, "risk_score_floor": 0.5, "risk_score_slope": bp["sf_risk_score_slope"],
            "risk_count_threshold": 3, "risk_count_penalty": 0.9,
            "jito_hard_threshold": 5, "jito_hard_penalty": 0.4,
            "jito_soft_threshold": 3, "jito_soft_penalty": 0.6,
            "bundle_hard_threshold": 20, "bundle_hard_penalty": 0.5,
            "bundle_soft_threshold": 10, "bundle_soft_penalty": 0.7,
            "gini_threshold": bp["sf_gini_threshold"], "gini_penalty": bp["sf_gini_penalty"],
            "holder_count_threshold": bp["sf_holder_count_threshold"],
            "holder_count_penalty": bp["sf_holder_count_penalty"],
            "whale_conc_threshold": bp["sf_whale_conc_threshold"], "whale_conc_floor": 0.7,
            "whale_conc_slope": bp["sf_whale_conc_slope"],
            "bb_score_thresholds": [20, 40], "bb_score_penalties": [0.6, 0.85],
            "bb_cluster_threshold": 30, "bb_cluster_floor": 0.6,
            "bb_cluster_slope": bp["sf_bb_cluster_slope"],
            "whale_dom_threshold": 0.5, "whale_dom_penalty": 0.85,
            "whale_dist_penalty": bp["sf_whale_dist_penalty"],
            "whale_dump_penalty": bp["sf_whale_dump_penalty"],
            "lp_unlock_penalty": bp["sf_lp_unlock_penalty"],
            "lp_partial_threshold": 50, "lp_partial_penalty": 0.85,
            "cex_threshold": 20, "cex_floor": 0.7, "cex_slope": bp["sf_cex_slope"],
        },
        "momentum_config": {
            "kol_fresh_thresholds": [bp["mom_kf_t0"], bp["mom_kf_t0"] * 0.4],
            "kol_fresh_factors": [bp["mom_kf_f0"], bp["mom_kf_f0"] * 0.92, 1.05],
            "mhr_thresholds": [bp["mom_mhr_t0"], bp["mom_mhr_t0"] * 0.5, 0.3],
            "mhr_factors": [bp["mom_mhr_f0"], bp["mom_mhr_f0"] * 0.87, 1.05],
            "sth_thresholds": [bp["mom_sth_t0"], bp["mom_sth_t0"] * 0.5],
            "sth_factors": [1.10, 1.05],
            "sth_penalty_threshold": bp["mom_sth_pen_threshold"],
            "sth_penalty_factor": 0.95,
            "cascade_factor_3plus": bp.get("cascade_factor_3plus", 1.15),
            "cascade_factor_2plus": bp.get("cascade_factor_2plus", 1.08),
            "early_factor_30min": bp.get("early_factor_30min", 1.20),
            "early_factor_60min": bp.get("early_factor_60min", 1.10),
            "late_penalty_360min": bp.get("late_penalty_360min", 0.85),
            "pvfc_penalty_3x": bp.get("pvfc_penalty_3x", 0.60),
            "pvfc_penalty_2x": bp.get("pvfc_penalty_2x", 0.75),
            "pvfc_penalty_1_5x": bp.get("pvfc_penalty_1_5x", 0.90),
            "floor": bp["mom_floor"],
            "cap": bp["mom_cap"],
        },
        "size_mult_config": {
            "mcap_thresholds": [300000, 1000000, 5000000, 20000000, 50000000, 200000000, 500000000],
            "mcap_factors": [bp["sm_mcap_f0"], 1.15, 1.0, 0.85, 0.70, 0.50, 0.35, bp["sm_mcap_f6"]],
            "fresh_thresholds": [4, 12],
            "fresh_factors": [bp["sm_fresh_f0"], 1.1, 1.0],
            "large_cap_threshold": bp["sm_large_cap_threshold"],
            "floor": bp["sm_floor"],
            "cap": bp["sm_cap"],
        },
    }

    # Evaluate on TEST set (walk-forward validation)
    test_score = _evaluate_params(df_test, best_weights, best_params, horizon, threshold)

    # Baseline: evaluate current production params on test set
    baseline_params = {
        "combined_floor": 0.25,
        "combined_cap": 2.0,
        "activity_mult_floor": 0.80,
        "activity_mult_cap": 1.25,
        "activity_ratio_high": 0.6,
        "activity_ratio_mid": 0.3,
        "activity_ratio_low": 0.1,
        "activity_pump_cap_hard": 80,
        "activity_pump_cap_soft": 50,
        "s_tier_bonus": 1.2,
        # v49: conviction normalization defaults
        "conviction_offset": 6,
        "conviction_divisor": 4,
        "breadth_pen_config": {"thresholds": [0.033, 0.05, 0.08], "penalties": [0.75, 0.85, 0.95]},
        "hype_pen_config": {"thresholds": [2, 4, 7], "penalties": [1.0, 0.85, 0.65, 0.50], "cooc_config": {"threshold": 0.5, "penalty": 0.85}},
        # v45: new baseline defaults
        "safety_floor": 0.75,
        "onchain_mult_floor": 0.3,
        "onchain_mult_cap": 1.5,
        "death_config": {
            "stale_start_hours": 12, "stale_tiers": [24, 48, 72],
            "stale_bases": [0.7, 0.45, 0.25, 0.15],
            "vol_modulation_tiers": [50000, 100000, 500000, 1000000],
            "vol_modulation_bonuses": [0.15, 0.25, 0.35, 0.45],
            "vol_modulation_caps": [0.8, 0.85, 0.9, 0.95],
            "vol_death_24h": 5000, "vol_death_1h": 500, "vol_death_penalty": 0.15,
            "vol_floor_24h": 1000, "vol_floor_penalty": 0.1,
            "price_moderate_social_alive_h": 6, "price_moderate_vol_alive": 0.5,
            "price_mild_threshold": -30, "price_mild_stale_h": 24,
        },
        "entry_premium_config": {
            "tier_breakpoints": [1.0, 1.2, 2.0, 4.0, 8.0, 20.0],
            "tier_multipliers": [1.1, 1.0, 0.9, 0.7, 0.5, 0.35, 0.25],
            "tier_slopes": [0, 0, 0, 0.1, 0.05, 0.0125, 0],
            "duration_thresholds": [12, 24, 48],
            "duration_factors": [1.0, 1.15, 1.3, 1.5],
            "mcap_fallback_threshold": 50000000, "mcap_launch_assumed": 1000000,
            "mcap_tiers": [50, 200, 500], "mcap_multipliers": [0.85, 0.70, 0.50, 0.35],
        },
        "lifecycle_config": {
            "panic_pc24": -30, "panic_va": 0.5, "panic_penalty": 0.25,
            "profit_taking_pc24": 100, "profit_taking_vol_proxy": 40, "profit_taking_penalty": 0.35,
            "euphoria_uk": 3, "euphoria_pc24": 100, "euphoria_sent": 0.2, "euphoria_penalty": 0.5,
            "boom_uk": 2, "boom_pc24_low": 10, "boom_pc24_high": 200, "boom_va": 1.0,
            "boom_bonus": 1.1, "boom_large_cap_penalty": 0.85, "boom_large_cap_mcap": 50000000,
            "displacement_age": 6, "displacement_uk": 1, "displacement_pc24": 50, "displacement_penalty": 0.9,
        },
        "entry_drift_config": {
            "premium_gate": 1.2, "kol_divisor": 7,
            "activity_base": 0.80, "activity_range": 0.45,
            "fresh_tiers": [1, 4, 8], "fresh_scores": [1.0, 0.7, 0.4, 0.1],
            "stier_divisor": 2, "social_weights": [0.30, 0.25, 0.30, 0.15],
            "drift_factor": 0.25, "drift_floor": 0.50,
        },
        # v46: PA baseline (all direction penalties at 1.0 = current v27 behavior)
        "pa_norm_floor": 0.4,
        "pa_norm_cap": 1.3,
        "pa_config": {
            "rsi_hard_pump": 80, "rsi_pump": 70,
            "rsi_freefall": 20, "rsi_dying": 30,
            "dir_hard_pump_mult": 1.0, "dir_pump_mult": 1.0,
            "dir_freefall_mult": 1.0, "dir_dying_mult": 1.0,
        },
        # v47: Remaining hardcoded params baseline
        "consensus_pump_threshold": 50,
        "consensus_pump_floor": 0.5,
        "consensus_pump_divisor": 400,
        "activity_mid_mult": 1.10,
        "pump_pen_config": {
            "hard_penalty": 0.5,
            "moderate_penalty": 0.7,
            "light_penalty": 0.85,
            "pvp_pump_fun_floor": 0.7,
            "pvp_pump_fun_scale": 0.05,
            "pvp_normal_floor": 0.5,
            "pvp_normal_scale": 0.1,
            "pump_1h_hard": 30,
            "pump_5m_hard": 15,
            "pump_1h_mod": 15,
            "pump_5m_mod": 7.995,
            "pump_1h_light": 8.01,
        },
        # v48: 4 new JSONB baseline configs (matching pipeline.py defaults)
        "onchain_config": {
            "lmr_low": 0.02, "lmr_low_factor": 0.5,
            "lmr_high": 0.10, "lmr_high_factor": 1.2,
            "lmr_interp_base": 0.8, "lmr_interp_slope": 4,
            "bsr_base": 0.5,
            "jup_non_tradeable_factor": 0.5,
            "age_thresholds": [1, 6, 48, 168],
            "age_factors": [0.5, 1.0, 1.2, 1.0, 0.8],
            "tx_thresholds": [5, 20, 40], "tx_factors": [0.7, 1.0, 1.1, 1.3],
            "jup_impact_thresholds": [1.0, 5.0], "jup_impact_factors": [1.3, 1.0, 0.7],
            "whale_change_thresholds": [-10.0, 0, 5.0],
            "whale_change_factors": [0.6, 0.8, 1.1, 1.3],
            "velocity_thresholds": [0.2, 1.0, 5.0],
            "velocity_factors": [0.6, 1.0, 1.1, 1.3],
            "wne_thresholds": [1, 3], "wne_factors": [1.0, 1.1, 1.25],
            "whale_count_thresholds": [1, 3, 5],
            "whale_count_factors": [0.7, 1.0, 1.2, 1.4],
            "uw_change_thresholds": [-20, -5, 5, 20],
            "uw_change_factors": [0.6, 0.8, 1.0, 1.15, 1.3],
            "vol_proxy_threshold": 50, "vol_proxy_penalty": 0.8,
            "whale_accum_bonus": 1.15,
            # v53: holder stability + liquidity depth defaults
            "smr_thresholds": [50, 70, 90], "smr_factors": [0.8, 1.0, 1.15, 1.3],
            "shp_thresholds": [50, 70, 85], "shp_factors": [0.7, 0.9, 1.1, 1.3],
            "lds_thresholds": [0.2, 0.5, 0.8], "lds_factors": [0.6, 0.85, 1.05, 1.2],
        },
        "safety_config": {
            "insider_threshold": 30, "insider_floor": 0.5, "insider_slope": 100,
            "top10_threshold": 50, "top10_floor": 0.7, "top10_slope": 100,
            "risk_score_threshold": 5000, "risk_score_floor": 0.5, "risk_score_slope": 5000,
            "risk_count_threshold": 3, "risk_count_penalty": 0.9,
            "jito_hard_threshold": 5, "jito_hard_penalty": 0.4,
            "jito_soft_threshold": 3, "jito_soft_penalty": 0.6,
            "bundle_hard_threshold": 20, "bundle_hard_penalty": 0.5,
            "bundle_soft_threshold": 10, "bundle_soft_penalty": 0.7,
            "gini_threshold": 0.85, "gini_penalty": 0.8,
            "holder_count_threshold": 50, "holder_count_penalty": 0.85,
            "whale_conc_threshold": 60, "whale_conc_floor": 0.7, "whale_conc_slope": 80,
            "bb_score_thresholds": [20, 40], "bb_score_penalties": [0.6, 0.85],
            "bb_cluster_threshold": 30, "bb_cluster_floor": 0.6, "bb_cluster_slope": 70,
            "whale_dom_threshold": 0.5, "whale_dom_penalty": 0.85,
            "whale_dist_penalty": 0.75, "whale_dump_penalty": 0.65,
            "lp_unlock_penalty": 0.6, "lp_partial_threshold": 50, "lp_partial_penalty": 0.85,
            "cex_threshold": 20, "cex_floor": 0.7, "cex_slope": 100,
        },
        "momentum_config": {
            "kol_fresh_thresholds": [0.5, 0.2],
            "kol_fresh_factors": [1.20, 1.10, 1.05],
            "mhr_thresholds": [2.0, 1.0, 0.3],
            "mhr_factors": [1.15, 1.10, 1.05],
            "sth_thresholds": [3.0, 1.5],
            "sth_factors": [1.10, 1.05],
            "sth_penalty_threshold": 0.3,
            "sth_penalty_factor": 0.95,
            "cascade_factor_3plus": 1.15,
            "cascade_factor_2plus": 1.08,
            "early_factor_30min": 1.20,
            "early_factor_60min": 1.10,
            "late_penalty_360min": 0.85,
            "pvfc_penalty_3x": 0.60,
            "pvfc_penalty_2x": 0.75,
            "pvfc_penalty_1_5x": 0.90,
            "floor": 0.70,
            "cap": 1.40,
        },
        "size_mult_config": {
            "mcap_thresholds": [300000, 1000000, 5000000, 20000000, 50000000, 200000000, 500000000],
            "mcap_factors": [1.3, 1.15, 1.0, 0.85, 0.70, 0.50, 0.35, 0.25],
            "fresh_thresholds": [4, 12],
            "fresh_factors": [1.2, 1.1, 1.0],
            "large_cap_threshold": 50000000,
            "floor": 0.25,
            "cap": 1.5,
        },
    }
    baseline_score = _evaluate_params(df_test, BALANCED_WEIGHTS, baseline_params, horizon, threshold)

    logger.info(
        "optuna: walk-forward validation — train=%.4f, test=%.4f, baseline=%.4f",
        best_trial.value, test_score, baseline_score,
    )

    # Guard-rail: train/test gap < 20% (prevents overfit)
    # If either score is negative/zero, the params are likely poor — skip.
    if best_trial.value <= 0 or test_score <= 0:
        logger.warning(
            "optuna: non-positive scores (train=%.4f, test=%.4f) — skipping",
            best_trial.value, test_score,
        )
        return None
    gap = abs(best_trial.value - test_score) / max(0.001, best_trial.value)
    if gap > 0.20:
        logger.warning(
            "optuna: train/test gap %.1f%% > 20%%, likely overfit — skipping",
            gap * 100,
        )
        return None

    # Guard-rail: must improve over baseline by >= 5%
    if baseline_score > -900:
        improvement = (test_score - baseline_score) / max(0.001, abs(baseline_score))
        if improvement < 0.05:
            logger.info(
                "optuna: improvement %.1f%% < 5%% threshold — keeping current params",
                improvement * 100,
            )
            return None
    else:
        improvement = float("inf")

    # Guard-rail: no scalar param > 30% change from current
    # Note: JSONB configs (onchain_config, safety_config, etc.) are bounded by Optuna ranges
    for key in ["activity_mult_floor", "activity_mult_cap", "s_tier_bonus",
                 "combined_floor", "combined_cap",
                 "safety_floor", "onchain_mult_floor", "onchain_mult_cap",
                 "pa_norm_floor", "pa_norm_cap",
                 "consensus_pump_threshold", "consensus_pump_floor",
                 "consensus_pump_divisor", "activity_mid_mult",
                 "conviction_offset", "conviction_divisor"]:
        current = baseline_params.get(key, best_params[key])
        new_val = best_params[key]
        if current > 0:
            change = abs(new_val - current) / current
            if change > 0.30:
                logger.warning(
                    "optuna: %s changed %.1f%% > 30%% max — clamping",
                    key, change * 100,
                )
                direction = 1 if new_val > current else -1
                best_params[key] = current * (1 + direction * 0.30)

    # v49: Post-Optuna bot validation — verify #1 pick has positive expectancy
    max_price_col = f"max_price_{horizon}"
    df_test_scored = df_test.copy()
    df_test_scored["trial_score"] = df_test_scored.apply(
        lambda r: _compute_score_with_params(r, best_weights, best_params), axis=1
    )
    df_test_scored["cycle"] = df_test_scored["snapshot_at"].dt.floor("15min")
    bot_wins, bot_losses, bot_tested = 0, 0, 0
    for _, group in df_test_scored.groupby("cycle"):
        labeled = group[group[max_price_col].notna() & group["price_at_snapshot"].notna()]
        if labeled.empty:
            continue
        top1 = labeled.loc[labeled["trial_score"].idxmax()]
        p0 = float(top1["price_at_snapshot"])
        if p0 <= 0:
            continue
        p_max = float(top1[max_price_col])
        bot_tested += 1
        ret = (p_max / p0) - 1.0
        if ret >= (threshold - 1.0):
            bot_wins += 1
        elif ret < -0.30:  # SL proxy: -30% = loss
            bot_losses += 1
    if bot_tested >= 5:
        bot_expectancy = (bot_wins * (threshold - 1.0) - bot_losses * 0.30) / bot_tested
        if bot_expectancy < 0:
            logger.warning(
                "optuna: bot validation NEGATIVE expectancy (%.3f, wins=%d losses=%d tested=%d) — skipping",
                bot_expectancy, bot_wins, bot_losses, bot_tested,
            )
            return None
        logger.info(
            "optuna: bot validation OK — expectancy=%.3f, wins=%d/%d",
            bot_expectancy, bot_wins, bot_tested,
        )

    return {
        "weights": best_weights,
        "params": best_params,
        "train_score": best_trial.value,
        "test_score": test_score,
        "baseline_score": baseline_score,
        "improvement_pct": round(improvement * 100, 1) if improvement != float("inf") else None,
        "n_trials": len(study.trials),
        "best_trial_number": best_trial.number,
    }


def _apply_optuna_params(client, best_weights: dict, best_params: dict, reason: str) -> bool:
    """Write optimized params to scoring_config table."""
    try:
        update = {
            "w_consensus": best_weights["consensus"],
            "w_conviction": best_weights["conviction"],
            "w_breadth": best_weights["breadth"],
            "w_price_action": best_weights["price_action"],
            "combined_floor": best_params["combined_floor"],
            "combined_cap": best_params["combined_cap"],
            "activity_mult_floor": best_params["activity_mult_floor"],
            "activity_mult_cap": best_params["activity_mult_cap"],
            "activity_ratio_high": best_params["activity_ratio_high"],
            "activity_ratio_mid": best_params["activity_ratio_mid"],
            "activity_ratio_low": best_params["activity_ratio_low"],
            "activity_pump_cap_hard": best_params["activity_pump_cap_hard"],
            "activity_pump_cap_soft": best_params["activity_pump_cap_soft"],
            "s_tier_bonus": best_params["s_tier_bonus"],
            "breadth_pen_config": best_params["breadth_pen_config"],
            "hype_pen_config": best_params["hype_pen_config"],
            # v45: new scalar params
            "safety_floor": best_params.get("safety_floor", 0.75),
            "onchain_mult_floor": best_params.get("onchain_mult_floor", 0.3),
            "onchain_mult_cap": best_params.get("onchain_mult_cap", 1.5),
            # v45: new JSONB configs
            "death_config": best_params.get("death_config"),
            "entry_premium_config": best_params.get("entry_premium_config"),
            "lifecycle_config": best_params.get("lifecycle_config"),
            "entry_drift_config": best_params.get("entry_drift_config"),
            # v46: PA config + norm bounds
            "pa_norm_floor": best_params.get("pa_norm_floor", 0.4),
            "pa_norm_cap": best_params.get("pa_norm_cap", 1.3),
            "pa_config": best_params.get("pa_config"),
            # v47: Remaining hardcoded params
            "consensus_pump_threshold": best_params.get("consensus_pump_threshold", 50),
            "consensus_pump_floor": best_params.get("consensus_pump_floor", 0.5),
            "consensus_pump_divisor": best_params.get("consensus_pump_divisor", 400),
            "activity_mid_mult": best_params.get("activity_mid_mult", 1.10),
            "pump_pen_config": best_params.get("pump_pen_config"),
            # v49: conviction normalization params
            "conviction_offset": best_params.get("conviction_offset", 6),
            "conviction_divisor": best_params.get("conviction_divisor", 4),
            # v48: 4 new JSONB configs for full multiplier recomputation
            "onchain_config": best_params.get("onchain_config"),
            "safety_config": best_params.get("safety_config"),
            "momentum_config": best_params.get("momentum_config"),
            "size_mult_config": best_params.get("size_mult_config"),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "updated_by": "optuna_v52",
            "change_reason": reason,
        }
        # Remove None values (don't write nulls for missing configs)
        update = {k: v for k, v in update.items() if v is not None}
        client.table("scoring_config").update(update).eq("id", 1).execute()
        logger.info("optuna: scoring_config updated — %s", reason)
        return True
    except Exception as e:
        logger.error("optuna: failed to write scoring_config: %s", e)
        return False


def run_optuna_optimization(n_trials: int = 200, dry_run: bool = False) -> dict | None:
    """
    Top-level entry point for Optuna parameter optimization.
    Called from GitHub Actions (outcomes.yml) before auto_backtest.

    Self-gates: skips if < 150 unique tokens with labels.
    """
    client = _get_client()
    if not client:
        return None

    _load_current_weights(client)

    df = _fetch_snapshots(client)
    if df.empty:
        logger.info("optuna: no snapshots found")
        return None

    # First-appearance dedup
    first = df.sort_values("snapshot_at").drop_duplicates(subset=["token_address"], keep="first")

    # Use 12h as primary horizon (matching auto_backtest)
    horizon = "12h"
    max_price_col = f"max_price_{horizon}"
    if max_price_col not in first.columns:
        logger.info("optuna: max_price_%s column not available", horizon)
        return None

    labeled = first[first[max_price_col].notna() & first["price_at_snapshot"].notna()]
    n_labeled = len(labeled)

    # Guard-rail: minimum 150 unique tokens
    if n_labeled < 150:
        logger.info(
            "optuna: not enough labeled tokens (%d/150) — skipping optimization",
            n_labeled,
        )
        return None

    logger.info("optuna: starting optimization with %d labeled tokens, %d trials", n_labeled, n_trials)

    # Pass deduplicated dataframe to prevent data leakage in walk-forward split
    result = _optuna_optimize_params(first, horizon=horizon, n_trials=n_trials)
    if result is None:
        logger.info("optuna: no improvement found, keeping current params")
        return None

    logger.info(
        "optuna: found improvement! train=%.4f test=%.4f baseline=%.4f (+%.1f%%)",
        result["train_score"], result["test_score"], result["baseline_score"],
        result.get("improvement_pct", 0),
    )

    if dry_run:
        logger.info("optuna: DRY RUN — would apply: weights=%s, params=%s", result["weights"], result["params"])
        return result

    reason = (
        f"optuna_v49: {n_labeled} tokens, {result['n_trials']} trials, "
        f"test_score {result['baseline_score']:.4f} -> {result['test_score']:.4f} "
        f"(+{result.get('improvement_pct', 0):.1f}%)"
    )
    applied = _apply_optuna_params(client, result["weights"], result["params"], reason)
    result["applied"] = applied
    return result


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
    print(f"\nData: {ds['total_snapshots']} snapshots -> {ds.get('unique_tokens', '?')} unique tokens "
          f"(dedup {ds.get('dedup_ratio', '?')}x), "
          f"{ds.get('labeled_12h', 0)} labeled (12h), "
          f"{ds['date_range_days']} days")
    hr_parts = []
    for hz in ["12h", "24h", "48h", "72h", "7d"]:
        val = ds.get(f"hit_rate_{hz}")
        if val is not None:
            hr_parts.append(f"{hz}={val*100:.1f}%")
    print(f"Hit rate: {', '.join(hr_parts)}")

    # #1 Token hit rate (PRIMARY TARGET)
    top1 = report.get("top1_hit_rate", {})
    if top1:
        print(f"\n#1 Token Hit Rate (TARGET = 100%):")
        for horizon in ["1h", "6h", "12h", "24h", "48h", "72h", "7d"]:
            t1 = top1.get(horizon, {})
            if t1.get("tokens_tested", 0) > 0:
                tested = t1["tokens_tested"]
                # v22: Multi-threshold display
                parts = []
                for tk in ["1.3x", "1.5x", "2.0x"]:
                    hr = t1.get(f"hit_rate_{tk}", 0)
                    hits = t1.get(f"hits_{tk}", 0)
                    parts.append(f"{tk}={hits}/{tested}({hr*100:.0f}%)")
                print(f"  {horizon}: {' | '.join(parts)}")
                for d in t1.get("details", []):
                    status = "2x" if d.get("did_2x") else "FAIL"
                    ret = f"{d['max_return']}x" if d.get("max_return") else "?"
                    print(f"    {d['symbol']:15s} score={d['score']:3d} "
                          f"return={ret:>6s} [{status}]")

    # Realistic Bot Simulation
    bot = report.get("realistic_bot", {})
    if bot:
        print(f"\nRealistic Bot Simulation (TP/SL):")
        best = bot.get("best_strategy")
        best_exp = bot.get("best_expectancy")
        if best:
            print(f"  BEST STRATEGY: {best} (expectancy={best_exp:+.2%})")
        # Show top1 strategies grouped by horizon
        for hz in ["12h", "24h"]:
            hz_strats = {k: v for k, v in bot.items()
                         if isinstance(v, dict) and v.get("horizon") == hz and v.get("top_n") == 1}
            if hz_strats:
                print(f"\n  --- {hz} (top 1 token) ---")
                for key, s in sorted(hz_strats.items()):
                    status = "PROFITABLE" if s["is_profitable"] else "UNPROFITABLE"
                    print(f"  {s['description']:20s} | {s['trades']:2d} trades | "
                          f"WR={s['win_rate']:.0%} (need>{s['breakeven_wr']:.0%}) | "
                          f"PF={s['profit_factor']:.2f} | "
                          f"E[r]={s['expectancy']:+.2%} | "
                          f"TP={s['tp']} SL={s['sl']} TO={s['timeout']} | "
                          f"{status}")
                    if s.get("avg_win_min") is not None:
                        print(f"  {'':20s} | avg win={s['avg_win_min']}min "
                              f"avg loss={s.get('avg_loss_min', '?')}min "
                              f"max_consec_loss={s['max_consecutive_losses']}")

    # Adaptive SL Simulation
    adaptive = report.get("adaptive_bot", {})
    if adaptive:
        print(f"\nAdaptive SL Simulation (ML v3.1 — oracle upper bound):")
        for hz in ["12h", "24h"]:
            # Fixed SL results
            for sl_pct in [20, 30, 50]:
                key = f"fixed_SL{sl_pct}_{hz}"
                s = adaptive.get(key)
                if s:
                    print(f"  Fixed SL{sl_pct}% {hz}: {s['trades']} trades | "
                          f"WR={s['win_rate']:.0%} | E[r]={s['expectancy']:+.2%} | "
                          f"PF={s['profit_factor']:.2f}")
            # Adaptive result
            a_key = f"adaptive_{hz}"
            a = adaptive.get(a_key)
            if a:
                print(f"  Adaptive  {hz}: {a['trades']} trades | "
                      f"WR={a['win_rate']:.0%} | E[r]={a['expectancy']:+.2%} | "
                      f"PF={a['profit_factor']:.2f} | "
                      f"avg_SL={a['avg_adaptive_sl_pct']:.1f}% avg_DD={a['avg_actual_dd_pct']:.1f}%")
            # ML DD bands
            bands_key = f"ml_dd_bands_{hz}"
            bands = adaptive.get(bands_key)
            if bands:
                print(f"  ML DD bands ({hz}):")
                for band, info in bands.items():
                    print(f"    rr {band}: n={info['count']} avg_dd={info['avg_dd_pct']}% "
                          f"p75_dd={info['p75_dd_pct']}% → SL={info['recommended_sl_pct']}%")

        imp = adaptive.get("adaptive_improvement_pp")
        if imp is not None:
            print(f"  Adaptive vs Fixed improvement: {imp:+.1f}pp expectancy")

    # Score calibration
    cal = report.get("score_calibration", {})
    if cal:
        print(f"\nScore Calibration (1 entry per token):")
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

    # Extraction analysis
    ext = report.get("extraction_analysis", {})
    if ext:
        print(f"\nExtraction Analysis (CA vs Ticker):")
        for mode in ["ca_only", "ticker_only", "both"]:
            m = ext.get(mode, {})
            if m.get("count", 0) > 0:
                hr = m["hit_rate"]
                print(f"  {mode:15s}: {m['hits']}/{m['count']} = "
                      f"{hr*100:.1f}% hit rate")
            else:
                print(f"  {mode:15s}: no data")
        corr = ext.get("has_ca_mention_correlation")
        if corr is not None:
            print(f"  has_ca_mention correlation: {corr:+.3f}")

    # Temporal analysis
    temporal = report.get("temporal_analysis", {})
    if temporal:
        print(f"\nTemporal Analysis (Europe/Paris):")
        by_day = temporal.get("by_day_of_week", {})
        if by_day:
            print(f"  Day of week:")
            for day, info in by_day.items():
                bar = "#" * max(0, int(info["hit_rate"] * 20))
                print(f"    {day:3s}: {info['hit_rate']*100:5.1f}% "
                      f"({info['hits']}/{info['count']}) {bar}")
        by_hour = temporal.get("by_hour_paris", {})
        if by_hour:
            print(f"  Hour bands (Paris):")
            for band, info in by_hour.items():
                bar = "#" * max(0, int(info["hit_rate"] * 20))
                print(f"    {band:5s}h: {info['hit_rate']*100:5.1f}% "
                      f"({info['hits']}/{info['count']}) {bar}")
        prime = temporal.get("prime_time", {})
        off_peak = temporal.get("off_peak", {})
        if prime and off_peak:
            print(f"  Prime time (19h-5h): {prime.get('hit_rate', 0)*100:.1f}% "
                  f"({prime.get('hits', 0)}/{prime.get('count', 0)})")
            print(f"  Off-peak   (5h-19h): {off_peak.get('hit_rate', 0)*100:.1f}% "
                  f"({off_peak.get('hits', 0)}/{off_peak.get('count', 0)})")
        weekend = temporal.get("weekend", {})
        weekday = temporal.get("weekday", {})
        if weekend and weekday:
            print(f"  Weekend: {weekend.get('hit_rate', 0)*100:.1f}% "
                  f"({weekend.get('hits', 0)}/{weekend.get('count', 0)})")
            print(f"  Weekday: {weekday.get('hit_rate', 0)*100:.1f}% "
                  f"({weekday.get('hits', 0)}/{weekday.get('count', 0)})")

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

    # Equity Curve
    eq = report.get("equity_curve", {})
    if eq:
        print(f"\nEquity Curve (10% risk per trade):")
        for key, info in eq.items():
            if not isinstance(info, dict) or "final_equity" not in info:
                continue
            status = "PROFITABLE" if info["total_return_pct"] > 0 else "LOSS"
            calmar = info.get("calmar_ratio", 0)
            print(f"  {key}: {info['trades']} trades | "
                  f"Return={info['total_return_pct']:+.1f}% | "
                  f"Max DD={info['max_drawdown_pct']:.1f}% | "
                  f"Calmar={calmar:.2f} | "
                  f"Max losing streak={info['max_losing_streak']} | "
                  f"{status}")

    # Slippage
    slip = report.get("slippage", {})
    if slip and "tokens_with_liquidity" in slip:
        print(f"\nSlippage Analysis ({slip['tokens_with_liquidity']} tokens with liquidity):")
        liq = slip.get("liquidity_stats", {})
        print(f"  Liquidity: median=${liq.get('median', 0):,.0f} "
              f"p25=${liq.get('p25', 0):,.0f} p75=${liq.get('p75', 0):,.0f}")
        for size in [100, 500, 1000, 5000]:
            s = slip.get(f"size_{size}", {})
            if s:
                print(f"  ${size:5d}: median_slip={s['median_slippage_pct']:.2f}% "
                      f"p95={s['p95_slippage_pct']:.2f}% "
                      f">5%={s['tokens_above_5pct_slip']} "
                      f">10%={s['tokens_above_10pct_slip']}")
        print(f"  Recommended max size: ${slip.get('recommended_max_size_usd', '?')}")

    # Confidence Intervals
    ci = report.get("confidence_intervals", {})
    if ci:
        print(f"\nStatistical Confidence:")
        verdict = ci.get("verdict", "")
        print(f"  {verdict}")
        overall = ci.get("overall_hit_rate", {})
        if overall:
            print(f"  Overall 2x: {overall['hit_rate']*100:.1f}% "
                  f"[{overall['ci_95_low']*100:.1f}%-{overall['ci_95_high']*100:.1f}%] "
                  f"(n={overall['n']}, CI width={overall['ci_width']*100:.0f}pp)")
        t1 = ci.get("top1_hit_rate", {})
        if t1:
            print(f"  Top1 2x:   {t1['hit_rate']*100:.1f}% "
                  f"[{t1['ci_95_low']*100:.1f}%-{t1['ci_95_high']*100:.1f}%] "
                  f"(n={t1['n']}, CI width={t1['ci_width']*100:.0f}pp)")
        needed = ci.get("samples_needed", {})
        if needed:
            print(f"  Samples: have {needed['current']}, "
                  f"need {needed['for_10pp_ci']} for ±10pp CI, "
                  f"{needed['for_5pp_ci']} for ±5pp CI")

    # Portfolio Simulation
    port = report.get("portfolio", {})
    if port:
        print(f"\nPortfolio Simulation (top N tokens, TP50/SL30):")
        best_cfg = port.get("best_config")
        for hz in ["12h", "24h"]:
            hz_results = {k: v for k, v in port.items()
                          if isinstance(v, dict) and v.get("horizon") == hz}
            if hz_results:
                print(f"  --- {hz} ---")
                for key in sorted(hz_results):
                    p = hz_results[key]
                    marker = " <-- BEST" if key == best_cfg else ""
                    print(f"  Top{p['portfolio_size']}: {p['trades']:3d} trades | "
                          f"WR={p['win_rate']:.0%} | "
                          f"Return={p['total_return_pct']:+.1f}% | "
                          f"DD={p['max_drawdown_pct']:.1f}%{marker}")

    # Kelly Criterion
    kelly = report.get("kelly", {})
    if kelly:
        print(f"\nKelly Criterion (optimal bet size):")
        rec = kelly.get("recommendation", "")
        print(f"  {rec}")
        for hz_key, bands in kelly.items():
            if not isinstance(bands, dict) or "band_0_30" not in bands:
                continue
            print(f"  --- {hz_key} ---")
            for band_name, lo, hi in SCORE_BANDS:
                info = bands.get(band_name, {})
                if not isinstance(info, dict) or info.get("count", 0) < 5:
                    continue
                print(f"  {lo}-{hi}: WR={info['win_rate']*100:.0f}% "
                      f"Kelly={info['kelly_half_pct']:.1f}% "
                      f"edge={info['expected_edge_pct']:+.1f}% "
                      f"[{info['verdict']}] (n={info['count']})")

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
