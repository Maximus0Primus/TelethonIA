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
    "consensus": 0.30,
    "conviction": 0.05,
    "breadth": 0.10,
    "price_action": 0.55,
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
    "s_tier_mult, s_tier_count, unique_kols, pump_momentum_pen, "
    "ca_mention_count, ticker_mention_count, url_mention_count, has_ca_mention, "
    "score_velocity, score_acceleration, mention_velocity, volume_velocity, "
    "social_momentum_phase, kol_arrival_rate, entry_timing_quality, gate_mult, "
    "time_to_1_3x_min_12h, time_to_1_5x_min_12h, time_to_1_3x_min_24h, time_to_1_5x_min_24h, "
    "time_to_sl20_min_12h, time_to_sl30_min_12h, time_to_sl50_min_12h, "
    "time_to_sl20_min_24h, time_to_sl30_min_24h, time_to_sl50_min_24h, "
    "max_dd_before_tp_pct_12h, max_dd_before_tp_pct_24h, "
    "time_to_2x_12h, time_to_2x_24h, "
    "jup_price_impact_1k, min_price_12h, min_price_24h"
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

    # Squeeze + trend: disabled in v16 (anti-predictive in memecoins)
    squeeze_mult = 1.0
    trend_mult = 1.0

    # v19: Size opportunity multiplier — progressive large-cap penalty
    size_mult = _safe_mult(row, "size_mult")
    if not pd.notna(row.get("size_mult")):
        t_mcap = float(row.get("market_cap") or 0) if pd.notna(row.get("market_cap")) else 0
        fmh = float(row.get("freshest_mention_hours") or 999) if pd.notna(row.get("freshest_mention_hours")) else 999
        if t_mcap <= 0:
            mcap_f = 1.0
        elif t_mcap < 300_000:
            mcap_f = 1.3
        elif t_mcap < 1_000_000:
            mcap_f = 1.15
        elif t_mcap < 5_000_000:
            mcap_f = 1.0
        elif t_mcap < 20_000_000:
            mcap_f = 0.85
        elif t_mcap < 50_000_000:
            mcap_f = 0.70
        elif t_mcap < 200_000_000:
            mcap_f = 0.50
        elif t_mcap < 500_000_000:
            mcap_f = 0.35
        else:
            mcap_f = 0.25
        # v19: No freshness boost for large caps
        if t_mcap >= 50_000_000:
            fresh_f = 1.0
        elif fmh < 4:
            fresh_f = 1.2
        elif fmh < 12:
            fresh_f = 1.1
        else:
            fresh_f = 1.0
        size_mult = max(0.25, min(1.5, mcap_f * fresh_f))

    # v15.3: S-tier bonus
    s_tier_mult = _safe_mult(row, "s_tier_mult")
    if not pd.notna(row.get("s_tier_mult")):
        s_tier_count = int(row.get("s_tier_count") or 0) if pd.notna(row.get("s_tier_count")) else 0
        s_tier_mult = 1.2 if s_tier_count > 0 else 1.0

    # v17: Pump momentum penalty
    pump_momentum_pen = _safe_mult(row, "pump_momentum_pen")

    # v21: Soft gate penalty (replaces hard gate ejection)
    gate_mult = _safe_mult(row, "gate_mult")

    # v17: stale_pen REMOVED from chain — death_penalty Signal 3 already
    # handles staleness with volume modulation. Keeping both was double-penalizing.
    combined_raw = (onchain * safety * crash_pen * squeeze_mult * trend_mult
                    * pump_bonus * wash_pen * pvp_pen * pump_pen
                    * activity_mult * breadth_pen * size_mult
                    * s_tier_mult * pump_momentum_pen * gate_mult)
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
    Deduplicates: if the same token is #1 across consecutive cycles, it's
    counted only ONCE (first occurrence).
    """
    THRESHOLDS = [1.3, 1.5, 2.0]

    df = df.copy()
    df["score"] = df.apply(lambda r: _compute_score(r, BALANCED_WEIGHTS), axis=1)

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
    df["score"] = df.apply(lambda r: _compute_score(r, BALANCED_WEIGHTS), axis=1)

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
    """For high-scoring tokens that did NOT 2x: what components were misleading? (first-appearance)"""
    first = df.sort_values("snapshot_at").drop_duplicates(subset=["token_address"], keep="first").copy()
    first["score"] = first.apply(lambda r: _compute_score(r, BALANCED_WEIGHTS), axis=1)

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
    # Conservative: small gain, tight stop
    {"name": "TP30_SL20", "tp_col": "time_to_1_3x_min", "sl_col": "time_to_sl20_min",
     "tp_pct": 0.30, "sl_pct": -0.20, "description": "+30% TP / -20% SL"},
    # Moderate: medium gain, medium risk
    {"name": "TP50_SL30", "tp_col": "time_to_1_5x_min", "sl_col": "time_to_sl30_min",
     "tp_pct": 0.50, "sl_pct": -0.30, "description": "+50% TP / -30% SL"},
    # Aggressive: big gain, big risk — the classic 2x
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
    """
    df = df.copy()
    df["score"] = df.apply(lambda r: _compute_score(r, BALANCED_WEIGHTS), axis=1)
    df["cycle"] = df["snapshot_at"].dt.floor("15min")

    results = {}

    for hz in ["12h", "24h"]:
        hz_suffix = f"_{hz}"

        for strat in BOT_STRATEGIES:
            # Build column names for this horizon
            if strat["name"] == "TP100_SL50":
                # time_to_2x is stored in hours, convert to minutes for comparison
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


def _adaptive_bot_simulation(df: pd.DataFrame) -> dict:
    """
    ML v3.1: Simulate adaptive SL based on actual DD data (oracle upper bound).

    For each token: adaptive_SL = max_dd_before_tp * 1.2 (20% above typical DD).
    This is an "oracle" simulation — upper bound on what ML-predicted SL could achieve.

    Also loads dd_by_rr_band from model meta (if available) to show ML-recommended SLs.

    Compares fixed SL (20%, 30%, 50%) vs adaptive SL for each horizon.
    """
    from pathlib import Path

    df = df.copy()
    df["score"] = df.apply(lambda r: _compute_score(r, BALANCED_WEIGHTS), axis=1)
    df["cycle"] = df["snapshot_at"].dt.floor("15min")

    results = {}

    # Load RR model meta for dd_by_rr_band (informational)
    model_dir = Path(__file__).parent
    for hz in ["12h", "24h"]:
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

    FIXED_SL_PCTS = [20, 30, 50]

    for hz in ["12h", "24h"]:
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
    """Time-based train/test split validation (first-appearance)."""
    first = df.sort_values("snapshot_at").drop_duplicates(subset=["token_address"], keep="first")
    labeled = first[first[horizon_col].notna()].copy()
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
    Uses TP50/SL30 on 12h as default strategy (most data).
    """
    df = df.copy()
    df["score"] = df.apply(lambda r: _compute_score(r, BALANCED_WEIGHTS), axis=1)
    df["cycle"] = df["snapshot_at"].dt.floor("15min")

    results = {}

    for hz in ["12h", "24h"]:
        tp_col = f"time_to_1_5x_min_{hz}"
        sl_col = f"time_to_sl30_min_{hz}"
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
                pnl_pct = -0.30
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
    df["score"] = df.apply(lambda r: _compute_score(r, BALANCED_WEIGHTS), axis=1)
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
    labeled["score"] = labeled.apply(lambda r: _compute_score(r, BALANCED_WEIGHTS), axis=1)
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
    """
    df = df.copy()
    df["score"] = df.apply(lambda r: _compute_score(r, BALANCED_WEIGHTS), axis=1)
    df["cycle"] = df["snapshot_at"].dt.floor("15min")

    results = {}

    for hz in ["12h", "24h"]:
        tp_col = f"time_to_1_5x_min_{hz}"
        sl_col = f"time_to_sl30_min_{hz}"
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
                        pnl = -0.30
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
    first["score"] = first.apply(lambda r: _compute_score(r, BALANCED_WEIGHTS), axis=1)

    results = {}

    for hz in ["12h", "24h"]:
        tp_col = f"time_to_1_5x_min_{hz}"
        sl_col = f"time_to_sl30_min_{hz}"

        if tp_col not in first.columns or sl_col not in first.columns:
            continue

        # Label trades: TP first = win (+50%), SL first = loss (-30%)
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
