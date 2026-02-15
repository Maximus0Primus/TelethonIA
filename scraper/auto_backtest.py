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

# --- Constants (fallback, overridden by scoring_config table) ---

_DEFAULT_WEIGHTS = {
    "consensus": 0.30,
    "conviction": 0.05,
    "breadth": 0.10,
    "price_action": 0.55,
}

BALANCED_WEIGHTS = _DEFAULT_WEIGHTS.copy()


def _load_current_weights(client) -> dict:
    """Load current production weights from scoring_config table."""
    global BALANCED_WEIGHTS
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
                return weights
    except Exception as e:
        logger.debug("auto_backtest: scoring_config load failed: %s", e)
    return BALANCED_WEIGHTS.copy()

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
    "social_momentum_phase, kol_arrival_rate, entry_timing_quality, gate_mult"
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
    THE key metric: does the #1 ranked token 2x?

    Groups snapshots by cycle, picks the highest-scoring token per cycle.
    Deduplicates: if the same token is #1 across consecutive cycles, it's
    counted only ONCE (first occurrence). This prevents $LARRY being #1 for
    4 cycles from inflating the test count.
    """
    df = df.copy()
    df["score"] = df.apply(lambda r: _compute_score(r, BALANCED_WEIGHTS), axis=1)

    # Group snapshots into cycles (same minute = same cycle)
    df["cycle"] = df["snapshot_at"].dt.floor("15min")

    results = {"1h": {}, "6h": {}, "12h": {}, "24h": {}, "48h": {}, "72h": {}, "7d": {}}

    for horizon in ["1h", "6h", "12h", "24h", "48h", "72h", "7d"]:
        flag_col = f"did_2x_{horizon}"
        if flag_col not in df.columns:
            continue

        tokens_tested = 0
        tokens_hit = 0
        top1_details = []
        seen_addresses = set()  # dedup: same token as #1 counted once

        for cycle_ts, group in df.sort_values("snapshot_at").groupby("cycle"):
            labeled = group[group[flag_col].notna()]
            if labeled.empty:
                continue

            # #1 token = highest score in this cycle
            top1 = labeled.loc[labeled["score"].idxmax()]
            addr = top1.get("token_address") or top1["symbol"]
            did_2x = bool(top1[flag_col])
            ret = _compute_return(top1, horizon)

            # Skip if this token was already #1 in a previous cycle
            if addr in seen_addresses:
                continue
            seen_addresses.add(addr)

            tokens_tested += 1
            if did_2x:
                tokens_hit += 1

            top1_details.append({
                "cycle": str(cycle_ts),
                "symbol": top1["symbol"],
                "score": int(top1["score"]),
                "did_2x": did_2x,
                "max_return": round(ret, 2) if ret else None,
            })

        if tokens_tested > 0:
            results[horizon] = {
                "tokens_tested": tokens_tested,
                "tokens_hit": tokens_hit,
                "hit_rate": round(tokens_hit / tokens_tested, 4),
                "target": 1.0,
                "details": top1_details,
            }

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
    Returns best weights dict or None if no improvement found.
    """
    first = df.sort_values("snapshot_at").drop_duplicates(subset=["token_address"], keep="first")
    labeled = first[first[horizon_col].notna()].copy()
    if len(labeled) < 50:
        return None

    # Generate candidate weight combinations (step=0.05, sum=1.0)
    best_hr = -1
    best_weights = None
    step = 0.05
    components = list(BALANCED_WEIGHTS.keys())

    # Score the current weights first as baseline
    labeled["base_score"] = labeled.apply(lambda r: _compute_score(r, BALANCED_WEIGHTS), axis=1)
    # Use top1 metric: per-cycle #1 token hit rate
    labeled["cycle"] = labeled["snapshot_at"].dt.floor("15min")

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
            if bool(top1[horizon_col]):
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
    for horizon in ["12h", "6h", "24h", "48h", "72h", "7d"]:
        t1 = top1.get(horizon, {})
        if t1.get("tokens_tested", 0) > 0:
            hr = t1["hit_rate"]
            tested = t1["tokens_tested"]
            hits = t1["tokens_hit"]
            if hr >= 1.0:
                recs.append(f"TARGET MET ({horizon}): #1 token 2x'd {hits}/{tested} unique tokens (100%)")
            elif hr >= 0.5:
                recs.append(f"CLOSE ({horizon}): #1 token 2x'd {hits}/{tested} unique tokens ({hr*100:.0f}%). Target: 100%")
            else:
                recs.append(f"NEEDS WORK ({horizon}): #1 token 2x'd {hits}/{tested} unique tokens ({hr*100:.0f}%). Target: 100%")
            # Show which #1 tokens failed
            for d in t1.get("details", []):
                if not d["did_2x"]:
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
                hr = t1["hit_rate"]
                marker = "OK" if hr >= 1.0 else "MISS"
                print(f"  {horizon}: {t1['tokens_hit']}/{t1['tokens_tested']} = "
                      f"{hr*100:.0f}% [{marker}]")
                for d in t1.get("details", []):
                    status = "2x" if d["did_2x"] else "FAIL"
                    ret = f"{d['max_return']}x" if d.get("max_return") else "?"
                    print(f"    {d['symbol']:15s} score={d['score']:3d} "
                          f"return={ret:>6s} [{status}]")

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
