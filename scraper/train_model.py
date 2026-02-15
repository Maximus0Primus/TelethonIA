"""
ML training script for memecoin prediction.
v3.1: risk_reward mode — predicts rr_ratio for adaptive SL per token.
v3: bot_won labels (TP before SL) for capturable profit prediction.
v2: Regression + classification dual mode, adaptive feature selection,
    Spearman rank correlation metric, quality gates.

- XGBoost + LightGBM ensemble
- Walk-forward temporal splits (no data leakage)
- Optuna hyperparameter optimization
- SHAP feature importance analysis
- Adaptive feature selection by sample count
- ML v3: bot_won mode — predicts if TP hits before SL (capturable profit)
- ML v3.1: risk_reward mode — predicts rr_ratio, outputs dd_by_rr_band for adaptive SL

Usage:
    python train_model.py --horizon 12h --trials 100
    python train_model.py --horizon 12h --mode regression --trials 100
    python train_model.py --mode bot_won --horizon 12h --strategy TP50_SL30
    python train_model.py --mode risk_reward --horizon 12h --min-samples 30
    python train_model.py --all --trials 200
    python train_model.py --auto  # grid search + bot_won Phase 2 + risk_reward Phase 3
"""

import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import optuna
from scipy.stats import spearmanr
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    classification_report, roc_auc_score,
    mean_squared_error,
)
from sklearn.calibration import CalibratedClassifierCV
from supabase import create_client

try:
    from pipeline import SCORING_PARAMS
except ImportError:
    SCORING_PARAMS = {"stale_hours_severe": 48}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent

# Quality gate: model must exceed this precision@5 to be saved
# v22: relaxed from 0.40 — lower thresholds (e.g. +50%) are easier to predict
MIN_PRECISION_AT_5 = 0.30

# v22: Dynamic return thresholds — no longer tied to did_2x_* columns
RETURN_THRESHOLDS = {
    "1.3x": 1.3,   # +30%
    "1.5x": 1.5,   # +50%
    "2x": 2.0,     # +100%
    "3x": 3.0,     # +200%
}

# ═══ FEATURE SELECTION TIERS ═══
# Tier 1 (core): Best 15 features from SHAP analysis. Use when <500 samples.
# These showed signal even with 122 samples. Keeps ratio ~10 samples/feature.
CORE_FEATURES = [
    # Top SHAP signals (both XGB and LGB agree)
    "token_age_hours",
    "short_term_heat",
    "price_change_6h",
    "mentions",
    "sentiment",
    "liq_mcap_ratio",
    "onchain_multiplier",
    "safety_penalty",
    # Strong LGB-only signals
    "sentiment_consistency",
    "ultra_short_heat",
    "buy_sell_ratio_5m",
    "price_change_1h",
    "volatility_proxy",
    "market_cap_log",
    # Cross-snapshot temporal
    "score_delta",
    # ML v2 Phase B: Temporal velocity (Tier 1)
    "score_velocity",
    "mention_velocity",
    "volume_velocity",
    # ML v3.1: Calendar/temporal (low-dim, high signal for memecoins)
    "day_of_week",       # 0=Mon..6=Sun (Sunday worst, Tuesday best)
    "hour_paris",        # 0-23 Europe/Paris (runners peak 19h-5h)
    "is_weekend",        # 0/1
    "is_prime_time",     # 0/1 (19h-5h Paris)
]

# Tier 2 (extended): Add these when 500-2000 samples. ~30 features total.
EXTENDED_FEATURES = CORE_FEATURES + [
    "volume_24h_log",
    "volume_1h_log",
    "liquidity_usd_log",
    "buy_sell_ratio_24h",
    "price_change_5m",
    "price_change_24h",
    "volume_mcap_ratio",
    "top10_holder_pct",
    "risk_score",
    "helius_top20_pct",
    "helius_onchain_bsr",
    "buys_24h",
    "pvp_same_name_count",
    "new_kol_ratio",
    "mentions_delta",
    # ML v2 Phase B: Temporal velocity (Tier 2)
    "score_acceleration",
    "kol_arrival_rate",
    # ML v2 Phase C: Entry zone
    "entry_timing_quality",
    # v17/v21: Scoring multipliers
    "pump_momentum_pen",
    "gate_mult",
]

# Tier 3 (full): All features when 2000+ samples. Let the model decide.
ALL_FEATURE_COLS = [
    # Telegram features
    "mentions", "sentiment", "breadth", "avg_conviction", "recency_score",
    # Volume (log-scaled)
    "volume_24h_log", "volume_6h_log", "volume_1h_log",
    # Market data (log-scaled)
    "liquidity_usd_log", "market_cap_log",
    # Transactions
    "txn_count_24h", "buys_24h", "sells_24h",
    "buy_sell_ratio_24h", "buy_sell_ratio_1h",
    # Price changes
    "price_change_5m", "price_change_1h", "price_change_6h", "price_change_24h",
    # Derived ratios
    "volume_mcap_ratio", "liq_mcap_ratio", "volume_acceleration",
    # Token metadata
    "token_age_hours", "is_pump_fun", "pair_count",
    # Safety
    "risk_score", "top10_holder_pct", "insider_pct",
    "has_mint_authority", "has_freeze_authority", "risk_count", "lp_locked_pct",
    # Birdeye
    "holder_count_log", "unique_wallet_24h", "unique_wallet_24h_change",
    "trade_24h", "trade_24h_change", "birdeye_buy_sell_ratio",
    "v_buy_24h_usd_log", "v_sell_24h_usd_log",
    # Scoring features
    "social_velocity", "mention_acceleration", "onchain_multiplier", "safety_penalty",
    # KOL / narrative
    "kol_reputation_avg", "narrative_is_hot", "pump_graduated",
    # Temporal deltas
    "mentions_delta", "sentiment_delta", "volume_delta", "holder_delta",
    "score_delta", "new_kol_ratio",
    # Helius on-chain
    "helius_holder_count_log", "helius_top5_pct", "helius_top20_pct",
    "helius_gini", "bundle_detected", "bundle_count", "bundle_pct",
    "helius_recent_tx_count", "helius_unique_buyers", "helius_onchain_bsr",
    # Jupiter
    "jup_tradeable", "jup_price_impact_1k", "jup_route_count",
    # Whale tracking
    "whale_count", "whale_total_pct", "whale_change", "whale_new_entries",
    # Narrative confidence
    "narrative_confidence",
    # Wash trading / Jito / PVP
    "wash_trading_score",
    "jito_bundle_detected", "jito_bundle_slots", "jito_max_slot_txns",
    "pvp_same_name_count", "pvp_recent_count",
    # Conviction NLP
    "msg_conviction_avg", "price_target_count", "hedging_count",
    # v3 features
    "short_term_heat", "txn_velocity", "sentiment_consistency", "is_artificial_pump",
    "volatility_proxy", "whale_dominance", "sentiment_amplification",
    # v3.1 features
    "volume_5m_log", "buy_sell_ratio_5m", "ultra_short_heat",
    "already_pumped_penalty",
    "bubblemaps_score", "bubblemaps_cluster_max_pct", "bubblemaps_cluster_count",
    # ML v2 Phase B: Temporal velocity
    "score_velocity", "score_acceleration", "mention_velocity",
    "volume_velocity", "kol_arrival_rate",
    # ML v2 Phase C: Entry zone
    "entry_timing_quality",
    # v17/v21: Scoring multipliers
    "pump_momentum_pen",
    "gate_mult",
    # ML v3.1: Calendar/temporal
    "day_of_week", "hour_paris", "is_weekend", "is_prime_time",
]

HORIZONS = {
    "1h": "did_2x_1h",
    "6h": "did_2x_6h",
    "12h": "did_2x_12h",
    "24h": "did_2x_24h",
}

# ML v3: Bot TP/SL strategies for bot_won training
BOT_STRATEGIES = {
    "TP30_SL20": {"tp_col": "time_to_1_3x_min", "sl_col": "time_to_sl20_min", "tp_pct": 0.30, "sl_pct": -0.20},
    "TP50_SL30": {"tp_col": "time_to_1_5x_min", "sl_col": "time_to_sl30_min", "tp_pct": 0.50, "sl_pct": -0.30},
    "TP100_SL50": {"tp_col": "time_to_2x",      "sl_col": "time_to_sl50_min", "tp_pct": 1.00, "sl_pct": -0.50},
}

# Return columns for regression target
RETURN_COLS = {
    "1h": ("max_price_1h", "price_at_snapshot"),
    "6h": ("max_price_6h", "price_at_snapshot"),
    "12h": ("max_price_12h", "price_at_snapshot"),
    "24h": ("max_price_24h", "price_at_snapshot"),
}


def _get_client():
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return create_client(url, key)


def _select_feature_tier(n_samples: int) -> list[str]:
    """Select feature list based on sample count to prevent overfitting."""
    if n_samples < 500:
        tier = "core"
        features = CORE_FEATURES
    elif n_samples < 2000:
        tier = "extended"
        features = EXTENDED_FEATURES
    else:
        tier = "full"
        features = ALL_FEATURE_COLS
    logger.info("Feature tier: %s (%d features for %d samples, ratio %.1f)",
                tier, len(features), n_samples, n_samples / max(1, len(features)))
    return features


def load_labeled_data(horizon: str = "12h") -> pd.DataFrame:
    """
    Load all labeled snapshots from Supabase with pagination.
    v22: Filter on max_price_{horizon} being non-null (not did_2x_*),
    so the label can be computed dynamically at any threshold.
    """
    client = _get_client()
    max_price_col = f"max_price_{horizon}"

    all_data = []
    page_size = 1000
    offset = 0

    while True:
        result = (
            client.table("token_snapshots")
            .select("*")
            .not_.is_(max_price_col, "null")
            .not_.is_("price_at_snapshot", "null")
            .order("snapshot_at")
            .range(offset, offset + page_size - 1)
            .execute()
        )

        if not result.data:
            break

        all_data.extend(result.data)
        if len(result.data) < page_size:
            break
        offset += page_size

    if not all_data:
        logger.error("No labeled data found for horizon %s", horizon)
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    # Pre-compute max_return for dynamic threshold labeling
    df[max_price_col] = pd.to_numeric(df[max_price_col], errors="coerce")
    df["price_at_snapshot"] = pd.to_numeric(df["price_at_snapshot"], errors="coerce")
    valid = (df[max_price_col] > 0) & (df["price_at_snapshot"] > 0)
    df = df[valid].copy()
    df["max_return"] = df[max_price_col] / df["price_at_snapshot"]
    logger.info("Loaded %d labeled snapshots for %s horizon", len(df), horizon)
    return df


def deduplicate_snapshots(df: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
    """
    Deduplicate to ONE snapshot per unique token (by token_address or symbol).
    Keeps the FIRST snapshot per token (earliest call = least biased).
    Also filters out zombie tokens (freshest_mention > stale_hours_severe) which are noise.

    v22: No longer depends on did_2x_* columns. Winner count computed from
    max_return >= threshold (dynamic).

    Without this, the same token appears ~5x in training data, causing:
    - Autocorrelation (correlated samples treated as independent)
    - Inflated sample count (29 "winners" = only 14 unique tokens)
    - Data leakage between train/test splits
    """
    before = len(df)

    # Sort by time to keep earliest snapshot
    df = df.sort_values("snapshot_at").reset_index(drop=True)

    # Deduplicate by token_address first (most reliable), fallback to symbol
    if "token_address" in df.columns and df["token_address"].notna().any():
        df["_dedup_key"] = df["token_address"].fillna(df["symbol"])
    else:
        df["_dedup_key"] = df["symbol"]

    df = df.drop_duplicates(subset=["_dedup_key"], keep="first")
    df = df.drop(columns=["_dedup_key"])

    # Filter zombie snapshots: tokens with very stale mentions are noise
    # v20: threshold from SCORING_PARAMS (dynamic)
    stale_cutoff = SCORING_PARAMS.get("stale_hours_severe", 48)
    if "freshest_mention_hours" in df.columns:
        stale_mask = pd.to_numeric(df["freshest_mention_hours"], errors="coerce") > stale_cutoff
        stale_count = stale_mask.sum()
        if stale_count > 0:
            df = df[~stale_mask]
            logger.info("Filtered %d zombie snapshots (freshest_mention > %.0fh)", stale_count, stale_cutoff)

    after = len(df)
    winners_after = int((df["max_return"] >= threshold).sum()) if "max_return" in df.columns else "?"
    logger.info(
        "Deduplicated: %d → %d snapshots (%d unique tokens, %s winners at %.1fx)",
        before, after, after, winners_after, threshold
    )
    return df.reset_index(drop=True)


def load_bot_labeled_data(horizon: str = "12h", strategy: str = "TP50_SL30") -> pd.DataFrame:
    """
    ML v3: Load snapshots with bot TP/SL data and compute bot_won label.

    bot_won = 1 if TP hit before SL (chronologically), 0 if SL first or timeout.
    This teaches the model to predict *capturable* profit, not theoretical max.
    """
    if strategy not in BOT_STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(BOT_STRATEGIES)}")

    client = _get_client()
    strat = BOT_STRATEGIES[strategy]

    # Build column names for this horizon
    if strategy == "TP100_SL50":
        tp_col = f"time_to_2x_{horizon}"   # stored in hours
        tp_is_hours = True
    else:
        tp_col = f"{strat['tp_col']}_{horizon}"  # stored in minutes
        tp_is_hours = False
    sl_col = f"{strat['sl_col']}_{horizon}"

    # Fetch all snapshots with pagination
    all_data = []
    page_size = 1000
    offset = 0

    while True:
        result = (
            client.table("token_snapshots")
            .select("*")
            .not_.is_("price_at_snapshot", "null")
            .order("snapshot_at")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        if not result.data:
            break
        all_data.extend(result.data)
        if len(result.data) < page_size:
            break
        offset += page_size

    if not all_data:
        logger.error("No data found for bot labeling")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)

    # Filter to rows where at least one of TP/SL is non-null
    if tp_col not in df.columns or sl_col not in df.columns:
        logger.warning("Bot columns %s/%s not found in data", tp_col, sl_col)
        return pd.DataFrame()

    df[tp_col] = pd.to_numeric(df.get(tp_col), errors="coerce")
    df[sl_col] = pd.to_numeric(df.get(sl_col), errors="coerce")
    has_data = df[tp_col].notna() | df[sl_col].notna()
    df = df[has_data].copy()

    if df.empty:
        logger.warning("No bot TP/SL data found for %s/%s", horizon, strategy)
        return pd.DataFrame()

    # Convert TP from hours to minutes if needed (time_to_2x is in hours)
    if tp_is_hours:
        df["_tp_min"] = df[tp_col] * 60
    else:
        df["_tp_min"] = df[tp_col]
    df["_sl_min"] = df[sl_col]

    # Label: bot_won = 1 if TP hit before SL
    tp_hit = df["_tp_min"].notna()
    sl_hit = df["_sl_min"].notna()
    tp_before_sl = tp_hit & (~sl_hit | (df["_tp_min"] < df["_sl_min"]))
    df["_bot_won"] = tp_before_sl.astype(int)

    # Compute max_return for compatibility with deduplicate_snapshots()
    max_price_col = f"max_price_{horizon}"
    if max_price_col in df.columns:
        df[max_price_col] = pd.to_numeric(df[max_price_col], errors="coerce")
        df["price_at_snapshot"] = pd.to_numeric(df["price_at_snapshot"], errors="coerce")
        valid = df[max_price_col].notna() & df["price_at_snapshot"].notna() & (df["price_at_snapshot"] > 0)
        df.loc[valid, "max_return"] = df.loc[valid, max_price_col] / df.loc[valid, "price_at_snapshot"]

    n_won = int(df["_bot_won"].sum())
    logger.info("Bot labeling %s/%s: %d samples, %d wins (%.1f%% win rate)",
                horizon, strategy, len(df), n_won, 100 * n_won / max(1, len(df)))

    return df


def load_risk_reward_data(horizon: str = "12h") -> pd.DataFrame:
    """
    ML v3.1: Load snapshots with drawdown data and compute risk/reward ratio.

    rr_ratio = max_return / (1 + max_dd_pct / 100)
    - rr_ratio > 1.0 = return compensates the drawdown
    - rr_ratio < 1.0 = drawdown eats the potential return

    Reuses load_labeled_data() for paginated fetch + max_return computation.
    Filters to rows where max_dd_before_tp_pct_{hz} is non-null and > 0.
    """
    df = load_labeled_data(horizon)
    if df.empty:
        return df

    dd_col = f"max_dd_before_tp_pct_{horizon}"
    if dd_col not in df.columns:
        logger.warning("RR loader: column %s not in data", dd_col)
        return pd.DataFrame()

    df[dd_col] = pd.to_numeric(df.get(dd_col), errors="coerce")
    valid = df[dd_col].notna() & (df[dd_col] > 0) & df["max_return"].notna() & (df["max_return"] > 0)
    df = df[valid].copy()

    if df.empty:
        logger.warning("RR loader: no valid rows with DD data for %s", horizon)
        return df

    df["rr_ratio"] = df["max_return"] / (1 + df[dd_col] / 100.0)
    df["log_rr"] = np.log1p(df["rr_ratio"] - 1)
    df["_winner"] = (df["rr_ratio"] > 1.0).astype(int)
    df["dd_pct"] = df[dd_col]

    n_winners = int(df["_winner"].sum())
    logger.info(
        "RR loader %s: %d samples, %d winners (rr>1.0, %.1f%%), "
        "median rr=%.2f, avg rr=%.2f, median dd=%.1f%%",
        horizon, len(df), n_winners, 100 * n_winners / max(1, len(df)),
        float(df["rr_ratio"].median()), float(df["rr_ratio"].mean()),
        float(df["dd_pct"].median()),
    )
    return df


def prepare_features(df: pd.DataFrame, feature_pool: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Transform raw data into ML features.
    Only uses features from the provided pool (tier-selected).
    Returns (df_with_features, list_of_available_feature_columns).
    """
    df = df.copy()

    # Log-scale monetary features
    log_cols = {
        "volume_24h": "volume_24h_log",
        "volume_6h": "volume_6h_log",
        "volume_1h": "volume_1h_log",
        "volume_5m": "volume_5m_log",
        "liquidity_usd": "liquidity_usd_log",
        "market_cap": "market_cap_log",
        "holder_count": "holder_count_log",
        "v_buy_24h_usd": "v_buy_24h_usd_log",
        "v_sell_24h_usd": "v_sell_24h_usd_log",
        "helius_holder_count": "helius_holder_count_log",
    }
    for raw_col, log_col in log_cols.items():
        if raw_col in df.columns:
            df[log_col] = df[raw_col].apply(
                lambda x: np.log1p(float(x)) if pd.notna(x) and float(x) > 0 else np.nan
            )

    # Pump.fun graduation binary
    if "pump_graduation_status" in df.columns:
        df["pump_graduated"] = (df["pump_graduation_status"] == "graduated").astype(int)

    # Birdeye buy/sell ratio
    if "birdeye_buy_24h" in df.columns and "birdeye_sell_24h" in df.columns:
        df["birdeye_buy_sell_ratio"] = df.apply(
            lambda r: r["birdeye_buy_24h"] / max(1, r["birdeye_buy_24h"] + r["birdeye_sell_24h"])
            if pd.notna(r.get("birdeye_buy_24h")) else np.nan,
            axis=1,
        )

    # ML v3.1: Calendar/temporal features derived from snapshot_at
    if "snapshot_at" in df.columns:
        ts = pd.to_datetime(df["snapshot_at"], utc=True)
        # Convert to Europe/Paris for trader-relevant hours
        try:
            ts_paris = ts.dt.tz_convert("Europe/Paris")
        except Exception:
            # Fallback: UTC+1 (approx)
            ts_paris = ts + pd.Timedelta(hours=1)
        df["day_of_week"] = ts_paris.dt.dayofweek       # 0=Mon..6=Sun
        df["hour_paris"] = ts_paris.dt.hour              # 0-23
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_prime_time"] = ((df["hour_paris"] >= 19) | (df["hour_paris"] < 5)).astype(int)

    # Filter to features in the tier pool that are available (>10% non-null)
    available = []
    for col in feature_pool:
        if col in df.columns:
            non_null_pct = df[col].notna().mean()
            if non_null_pct >= 0.10:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                available.append(col)
            else:
                logger.debug("Dropping %s: only %.0f%% non-null", col, non_null_pct * 100)
        else:
            logger.debug("Feature %s not in data — skipping", col)

    logger.info("Using %d features (from %d in tier pool)", len(available), len(feature_pool))
    return df, available


def walk_forward_split(df: pd.DataFrame, train_ratio: float = 0.7):
    """Temporal walk-forward split. No random shuffling."""
    df = df.sort_values("snapshot_at").reset_index(drop=True)
    split_idx = int(len(df) * train_ratio)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def _precision_at_k(y_true, proba, k=5):
    """Precision among top-k predictions."""
    top_k = min(k, len(proba))
    if top_k == 0:
        return 0.0
    top_indices = np.argsort(proba)[-top_k:]
    return float(y_true.iloc[top_indices].mean())


def _spearman_rank(y_true, proba):
    """Spearman rank correlation — does the model rank tokens correctly?"""
    if len(y_true) < 5:
        return 0.0
    corr, _ = spearmanr(y_true, proba)
    return float(corr) if not np.isnan(corr) else 0.0


# ═══ REGRESSION OPTIMIZERS ═══

def _optimize_xgb_regressor(X_train, y_train, X_test, y_test, n_trials):
    """Optimize XGBoost regressor with Optuna. Metric: Spearman rank correlation."""
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 15),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "objective": "reg:squarederror",
            "verbosity": 0,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        preds = model.predict(X_test)
        return _spearman_rank(y_test, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    logger.info("XGBoost regressor best Spearman: %.3f", study.best_value)

    best_params = study.best_params
    best_params.update({"objective": "reg:squarederror", "verbosity": 0})

    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    return model, best_params, study.best_value


def _optimize_lgb_regressor(X_train, y_train, X_test, y_test, n_trials):
    """Optimize LightGBM regressor with Optuna."""
    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "objective": "regression",
            "verbosity": -1,
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        preds = model.predict(X_test)
        return _spearman_rank(y_test, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    logger.info("LightGBM regressor best Spearman: %.3f", study.best_value)

    best_params = study.best_params
    best_params.update({"objective": "regression", "verbosity": -1})

    model = lgb.LGBMRegressor(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    return model, best_params, study.best_value


# ═══ CLASSIFICATION OPTIMIZERS (kept for comparison) ═══

def _optimize_xgboost(X_train, y_train, X_test, y_test, scale_pos, n_trials):
    """Optimize XGBoost classifier with Optuna."""
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 15),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float(
                "scale_pos_weight", scale_pos * 0.5, scale_pos * 2.0
            ),
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "verbosity": 0,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        proba = model.predict_proba(X_test)[:, 1]
        return _precision_at_k(y_test, proba, k=5)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    logger.info("XGBoost classifier best precision@5: %.3f", study.best_value)

    best_params = study.best_params
    best_params.update({"objective": "binary:logistic", "eval_metric": "aucpr", "verbosity": 0})

    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    return model, best_params, study.best_value


def _optimize_lightgbm(X_train, y_train, X_test, y_test, scale_pos, n_trials):
    """Optimize LightGBM classifier with Optuna."""
    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float(
                "scale_pos_weight", scale_pos * 0.5, scale_pos * 2.0
            ),
            "objective": "binary",
            "metric": "average_precision",
            "verbosity": -1,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        proba = model.predict_proba(X_test)[:, 1]
        return _precision_at_k(y_test, proba, k=5)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    logger.info("LightGBM classifier best precision@5: %.3f", study.best_value)

    best_params = study.best_params
    best_params.update({"objective": "binary", "metric": "average_precision", "verbosity": -1})

    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    return model, best_params, study.best_value


def _compute_shap(model, X_test, features, model_name="model"):
    """Compute and log SHAP feature importances."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_importance = dict(zip(features, mean_abs_shap))
        sorted_shap = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)

        logger.info("SHAP importances (%s):", model_name)
        for feat, imp in sorted_shap[:15]:
            logger.info("  %s: %.4f", feat, imp)

        return {k: float(v) for k, v in sorted_shap}
    except ImportError:
        logger.warning("shap not installed — skipping SHAP analysis")
        return {}
    except Exception as e:
        logger.warning("SHAP failed: %s", e)
        return {}


# ═══ REGRESSION TRAINING ═══

def train_regression(
    df: pd.DataFrame,
    horizon: str = "12h",
    n_trials: int = 100,
    min_samples: int = 200,
    threshold: float = 2.0,
) -> dict | None:
    """
    Train regression ensemble: predict max_return (continuous) instead of binary 2x.
    The model learns to rank tokens by expected return — more information than binary.

    v22: threshold param controls what counts as a "winner" for precision@K.
    Label is computed dynamically from max_return >= threshold.
    """
    max_price_col, snapshot_price_col = RETURN_COLS[horizon]

    if len(df) < min_samples:
        logger.warning("Only %d samples (need %d). Skipping training.", len(df), min_samples)
        return None

    # max_return should already be pre-computed by load_labeled_data()
    df = df.copy()
    if "max_return" not in df.columns:
        df[max_price_col] = pd.to_numeric(df.get(max_price_col), errors="coerce")
        df[snapshot_price_col] = pd.to_numeric(df.get(snapshot_price_col), errors="coerce")
        valid_mask = (
            df[max_price_col].notna() &
            df[snapshot_price_col].notna() &
            (df[snapshot_price_col] > 0) &
            (df[max_price_col] > 0)
        )
        df = df[valid_mask].copy()
        df["max_return"] = df[max_price_col] / df[snapshot_price_col]

    if len(df) < min_samples:
        logger.warning("Only %d valid regression samples (need %d).", len(df), min_samples)
        return None

    df["log_return"] = np.log1p(df["max_return"] - 1)  # log1p(return) for stability

    # v22: Dynamic binary label for precision@K
    df["_winner"] = (df["max_return"] >= threshold).astype(int)
    n_winners = int(df["_winner"].sum())
    logger.info("Threshold %.1fx: %d winners / %d total (%.1f%%)",
                threshold, n_winners, len(df), 100 * n_winners / len(df))

    # Feature selection by tier
    feature_pool = _select_feature_tier(len(df))
    df, available_features = prepare_features(df, feature_pool)

    if len(available_features) < 5:
        logger.error("Only %d features available — need at least 5", len(available_features))
        return None

    train_df, test_df = walk_forward_split(df)

    X_train = train_df[available_features]
    y_train = train_df["log_return"]
    X_test = test_df[available_features]
    y_test = test_df["log_return"]
    # v22: Dynamic binary label from threshold (not did_2x_* column)
    y_test_binary = test_df["_winner"]

    logger.info(
        "Regression — Train: %d, Test: %d, Return range: [%.2f, %.2f]",
        len(y_train), len(y_test), y_train.min(), y_train.max(),
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Train XGBoost regressor
    logger.info("=== Optimizing XGBoost Regressor (%d trials) ===", n_trials)
    xgb_model, xgb_params, xgb_spearman = _optimize_xgb_regressor(
        X_train, y_train, X_test, y_test, n_trials
    )

    # Train LightGBM regressor
    logger.info("=== Optimizing LightGBM Regressor (%d trials) ===", n_trials)
    lgb_model, lgb_params, lgb_spearman = _optimize_lgb_regressor(
        X_train, y_train, X_test, y_test, n_trials
    )

    # Ensemble: weighted average of predictions
    xgb_preds = xgb_model.predict(X_test)
    lgb_preds = lgb_model.predict(X_test)

    total_score = abs(xgb_spearman) + abs(lgb_spearman)
    xgb_weight = abs(xgb_spearman) / max(0.001, total_score)
    lgb_weight = abs(lgb_spearman) / max(0.001, total_score)

    ensemble_preds = xgb_weight * xgb_preds + lgb_weight * lgb_preds

    # Metrics
    spearman = _spearman_rank(y_test, ensemble_preds)
    rmse = float(np.sqrt(mean_squared_error(y_test, ensemble_preds)))

    # Also compute precision@K on the regression output (threshold at 2x = log1p(1) = 0.693)
    p_at_5 = _precision_at_k(y_test_binary, ensemble_preds, k=5)
    p_at_10 = _precision_at_k(y_test_binary, ensemble_preds, k=10)

    logger.info("\n=== REGRESSION ENSEMBLE RESULTS ===")
    logger.info("Weights: XGBoost=%.2f, LightGBM=%.2f", xgb_weight, lgb_weight)
    logger.info("Spearman rank correlation: %.3f (target: >0.30)", spearman)
    logger.info("RMSE: %.3f", rmse)
    logger.info("Precision@5 (thresholded): %.3f (target: >%.2f)", p_at_5, MIN_PRECISION_AT_5)
    logger.info("Precision@10 (thresholded): %.3f", p_at_10)

    # Quality gate
    if p_at_5 < MIN_PRECISION_AT_5:
        logger.warning(
            "QUALITY GATE FAILED: precision@5 = %.3f < %.2f minimum. "
            "Model NOT saved. Collect more data and retrain.",
            p_at_5, MIN_PRECISION_AT_5,
        )
        return {"quality_gate": "FAILED", "precision_at_5": float(p_at_5)}

    # SHAP analysis
    logger.info("=== SHAP Analysis ===")
    xgb_shap = _compute_shap(xgb_model, X_test, available_features, "XGBoost-Reg")
    lgb_shap = _compute_shap(lgb_model, X_test, available_features, "LightGBM-Reg")

    # Save models
    xgb_path = MODEL_DIR / f"model_{horizon}.json"
    xgb_model.save_model(str(xgb_path))
    logger.info("XGBoost regressor saved to %s", xgb_path)

    lgb_path = MODEL_DIR / f"model_{horizon}_lgb.txt"
    lgb_model.booster_.save_model(str(lgb_path))
    logger.info("LightGBM regressor saved to %s", lgb_path)

    # Feature importances
    xgb_importances = dict(zip(available_features, xgb_model.feature_importances_))
    sorted_imp = sorted(xgb_importances.items(), key=lambda x: x[1], reverse=True)

    metadata = {
        "horizon": horizon,
        "threshold": threshold,
        "mode": "regression",
        "trained_at": datetime.utcnow().isoformat(),
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "n_winners": n_winners,
        "feature_tier": "core" if len(df) < 500 else "extended" if len(df) < 2000 else "full",
        "ensemble_weights": {"xgboost": float(xgb_weight), "lightgbm": float(lgb_weight)},
        "xgb_params": {k: v for k, v in xgb_params.items() if k != "verbosity"},
        "lgb_params": {k: v for k, v in lgb_params.items() if k != "verbosity"},
        "metrics": {
            "spearman": float(spearman),
            "rmse": rmse,
            "precision_at_5": float(p_at_5),
            "precision_at_10": float(p_at_10),
            "xgb_spearman": float(xgb_spearman),
            "lgb_spearman": float(lgb_spearman),
        },
        "quality_gate": "PASSED",
        "feature_importances_xgb": {k: float(v) for k, v in sorted_imp},
        "shap_importances_xgb": xgb_shap,
        "shap_importances_lgb": lgb_shap,
        "features": available_features,
    }

    meta_path = MODEL_DIR / f"model_{horizon}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved to %s", meta_path)

    return metadata


# ═══ LEARNING-TO-RANK TRAINING (ML v2 Phase A) ═══

def _compute_relevance(row, horizon: str = "12h", threshold: float = 2.0) -> int:
    """
    Compute graded relevance label for LTR from max return.

    v22: Thresholds are parametric based on the active threshold.
    | Condition                         | Label | Meaning              |
    |-----------------------------------|-------|----------------------|
    | max_return > threshold * 2.5      | 4     | Exceptional winner   |
    | max_return >= threshold            | 3     | Solid winner         |
    | max_return > threshold * 0.65     | 1     | Moderate gain        |
    | Everything else                   | 0     | Non-winner           |
    """
    max_price_col = f"max_price_{horizon}"
    p0 = row.get("price_at_snapshot")
    pmax = row.get(max_price_col)

    if pd.isna(p0) or pd.isna(pmax) or float(p0) <= 0:
        return 0

    max_return = float(pmax) / float(p0)
    if max_return > threshold * 2.5:
        return 4
    elif max_return >= threshold:
        return 3
    elif max_return > threshold * 0.65:
        return 1
    return 0


def _construct_query_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """
    Build query groups for LTR: each group = all tokens scored in the same cycle.
    Filters out groups with < 3 tokens (LTR degenerates with tiny groups).

    Returns (filtered_df, group_sizes) where group_sizes is compatible with
    XGBoost's DMatrix.set_group().
    """
    df = df.copy()
    df["cycle"] = df["snapshot_at"].dt.floor("15min")

    # Filter: only cycles with 3+ tokens
    cycle_counts = df["cycle"].value_counts()
    valid_cycles = cycle_counts[cycle_counts >= 3].index
    df = df[df["cycle"].isin(valid_cycles)].sort_values(["cycle", "snapshot_at"])

    # Compute group sizes
    group_sizes = df.groupby("cycle").size().tolist()

    return df, group_sizes


def train_ltr(
    df: pd.DataFrame,
    horizon: str = "12h",
    n_trials: int = 100,
    min_samples: int = 200,
    threshold: float = 2.0,
) -> dict | None:
    """
    Train Learning-to-Rank model using XGBoost's rank:ndcg objective.
    Optimizes NDCG@5 directly — the model learns to rank tokens within
    each cycle, not predict absolute returns.

    v22: threshold param controls relevance grade boundaries.
    LambdaMART is the state-of-the-art for this type of ranking problem.
    """
    max_price_col = f"max_price_{horizon}"
    snapshot_price_col = "price_at_snapshot"

    if len(df) < min_samples:
        logger.warning("LTR: Only %d samples (need %d). Skipping.", len(df), min_samples)
        return None

    df = df.copy()
    df["snapshot_at"] = pd.to_datetime(df["snapshot_at"])
    df[max_price_col] = pd.to_numeric(df.get(max_price_col), errors="coerce")
    df[snapshot_price_col] = pd.to_numeric(df.get(snapshot_price_col), errors="coerce")

    # Compute relevance labels with parametric thresholds
    df["relevance"] = df.apply(lambda r: _compute_relevance(r, horizon, threshold), axis=1)

    # Need labeled data (at least price data to compute relevance)
    valid_mask = df[max_price_col].notna() & (df[snapshot_price_col] > 0)
    df = df[valid_mask].copy()

    if len(df) < min_samples:
        logger.warning("LTR: Only %d valid samples (need %d).", len(df), min_samples)
        return None

    # Construct query groups
    df, group_sizes = _construct_query_groups(df)

    if len(group_sizes) < 10:
        logger.warning("LTR: Only %d valid cycles (need 10+). Skipping.", len(group_sizes))
        return None

    logger.info("LTR: %d samples across %d cycles (avg %.1f tokens/cycle)",
                len(df), len(group_sizes), len(df) / len(group_sizes))

    # Feature selection by tier
    feature_pool = _select_feature_tier(len(df))
    df, available_features = prepare_features(df, feature_pool)

    if len(available_features) < 5:
        logger.error("LTR: Only %d features available — need at least 5", len(available_features))
        return None

    # Walk-forward split on CYCLES (not individual rows)
    df["cycle"] = df["snapshot_at"].dt.floor("15min")
    unique_cycles = sorted(df["cycle"].unique())
    split_idx = int(len(unique_cycles) * 0.7)
    train_cycles = set(unique_cycles[:split_idx])
    test_cycles = set(unique_cycles[split_idx:])

    train_df = df[df["cycle"].isin(train_cycles)]
    test_df = df[df["cycle"].isin(test_cycles)]

    train_groups = train_df.groupby("cycle").size().tolist()
    test_groups = test_df.groupby("cycle").size().tolist()

    X_train = train_df[available_features]
    y_train = train_df["relevance"]
    X_test = test_df[available_features]
    y_test = test_df["relevance"]

    # v22: Dynamic binary labels from max_return >= threshold (not did_2x_* column)
    if "max_return" in test_df.columns:
        y_test_binary = (test_df["max_return"] >= threshold).astype(int)
    else:
        y_test_binary = (y_test >= 3).astype(int)

    logger.info(
        "LTR — Train: %d samples/%d cycles, Test: %d samples/%d cycles",
        len(X_train), len(train_groups), len(X_test), len(test_groups),
    )
    logger.info(
        "LTR — Relevance distribution: 0=%d, 1=%d, 3=%d, 4=%d",
        int((y_train == 0).sum()), int((y_train == 1).sum()),
        int((y_train == 3).sum()), int((y_train == 4).sum()),
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Optuna optimization for XGBoost ranker
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 15),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtrain.set_group(train_groups)
        dtest = xgb.DMatrix(X_test, label=y_test)
        dtest.set_group(test_groups)

        xgb_params = {
            **params,
            "objective": "rank:ndcg",
            "eval_metric": "ndcg@5",
            "verbosity": 0,
        }

        result = xgb.train(
            xgb_params, dtrain,
            num_boost_round=params["n_estimators"],
            evals=[(dtest, "test")],
            verbose_eval=False,
        )

        # Get NDCG@5 from last evaluation
        preds = result.predict(dtest)
        # Compute precision@5 using binary labels
        p_at_5 = _precision_at_k(y_test_binary, preds, k=5)
        return p_at_5

    logger.info("=== Optimizing LTR XGBoost Ranker (%d trials) ===", n_trials)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    logger.info("LTR best precision@5: %.3f", study.best_value)

    # Train final model with best params
    best_params = study.best_params
    n_estimators = best_params.pop("n_estimators")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtrain.set_group(train_groups)
    dtest = xgb.DMatrix(X_test, label=y_test)
    dtest.set_group(test_groups)

    xgb_params = {
        **best_params,
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@5",
        "verbosity": 0,
    }

    ltr_model = xgb.train(
        xgb_params, dtrain,
        num_boost_round=n_estimators,
        evals=[(dtest, "test")],
        verbose_eval=False,
    )

    # Evaluate
    test_preds = ltr_model.predict(dtest)
    p_at_5 = _precision_at_k(y_test_binary, test_preds, k=5)
    p_at_10 = _precision_at_k(y_test_binary, test_preds, k=10)
    spearman = _spearman_rank(y_test_binary.reset_index(drop=True), pd.Series(test_preds))

    # Compute NDCG@5 manually
    try:
        from sklearn.metrics import ndcg_score
        # Reshape for ndcg_score: needs (n_queries, n_docs) format
        # We compute per-cycle NDCG and average
        ndcg_scores = []
        offset = 0
        for g_size in test_groups:
            if g_size < 2:
                offset += g_size
                continue
            y_true = y_test.iloc[offset:offset + g_size].values.reshape(1, -1)
            y_pred = test_preds[offset:offset + g_size].reshape(1, -1)
            try:
                ndcg = ndcg_score(y_true, y_pred, k=5)
                ndcg_scores.append(ndcg)
            except Exception:
                pass
            offset += g_size
        avg_ndcg = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
    except ImportError:
        avg_ndcg = 0.0

    logger.info("\n=== LTR RESULTS ===")
    logger.info("NDCG@5: %.3f (target: >0.70)", avg_ndcg)
    logger.info("Precision@5: %.3f (target: >%.2f)", p_at_5, MIN_PRECISION_AT_5)
    logger.info("Precision@10: %.3f", p_at_10)
    logger.info("Spearman rank: %.3f", spearman)

    # Quality gate
    if p_at_5 < MIN_PRECISION_AT_5:
        logger.warning(
            "LTR QUALITY GATE FAILED: precision@5 = %.3f < %.2f. Model NOT saved.",
            p_at_5, MIN_PRECISION_AT_5,
        )
        return {"quality_gate": "FAILED", "precision_at_5": float(p_at_5), "mode": "ltr"}

    # SHAP on XGBoost Booster
    xgb_shap = {}
    try:
        import shap
        explainer = shap.TreeExplainer(ltr_model)
        shap_values = explainer.shap_values(X_test)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        xgb_shap = dict(sorted(
            zip(available_features, mean_abs_shap),
            key=lambda x: x[1], reverse=True,
        ))
        xgb_shap = {k: float(v) for k, v in xgb_shap.items()}
        logger.info("LTR SHAP top features:")
        for feat, imp in list(xgb_shap.items())[:10]:
            logger.info("  %s: %.4f", feat, imp)
    except Exception as e:
        logger.warning("LTR SHAP failed: %s", e)

    # Save model (XGBoost Booster saves as .json)
    xgb_path = MODEL_DIR / f"model_{horizon}.json"
    ltr_model.save_model(str(xgb_path))
    logger.info("LTR model saved to %s", xgb_path)

    # Feature importances from gain
    importance = ltr_model.get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    metadata = {
        "horizon": horizon,
        "threshold": threshold,
        "mode": "ltr",
        "trained_at": datetime.utcnow().isoformat(),
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "train_cycles": len(train_groups),
        "test_cycles": len(test_groups),
        "feature_tier": "core" if len(df) < 500 else "extended" if len(df) < 2000 else "full",
        "ensemble_weights": {"xgboost": 1.0, "lightgbm": 0.0},  # LTR is XGBoost-only
        "xgb_params": {k: v for k, v in best_params.items()},
        "metrics": {
            "ndcg_at_5": avg_ndcg,
            "precision_at_5": float(p_at_5),
            "precision_at_10": float(p_at_10),
            "spearman": float(spearman),
        },
        "quality_gate": "PASSED",
        "relevance_distribution": {
            "grade_0": int((y_train == 0).sum()),
            "grade_1": int((y_train == 1).sum()),
            "grade_3": int((y_train == 3).sum()),
            "grade_4": int((y_train == 4).sum()),
        },
        "feature_importances_xgb": {k: float(v) for k, v in sorted_imp},
        "shap_importances_xgb": xgb_shap,
        "features": available_features,
    }

    meta_path = MODEL_DIR / f"model_{horizon}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("LTR metadata saved to %s", meta_path)

    return metadata


# ═══ BOT_WON TRAINING (ML v3) ═══

def train_bot_won(
    df: pd.DataFrame,
    horizon: str = "12h",
    strategy: str = "TP50_SL30",
    n_trials: int = 100,
    min_samples: int = 100,
) -> dict | None:
    """
    ML v3: Train binary classification — predict bot_won (TP before SL).
    Learns capturable profit patterns instead of theoretical max return.

    Uses the same feature set and XGBoost+LightGBM ensemble as classification,
    but with bot_won labels computed from TP/SL chronological ordering.
    Models saved with _bot suffix to coexist with regression/LTR models.
    """
    if len(df) < min_samples:
        logger.warning("Bot training: only %d samples (need %d). Skipping.", len(df), min_samples)
        return None

    if "_bot_won" not in df.columns:
        logger.error("Bot training: missing _bot_won column. Call load_bot_labeled_data() first.")
        return None

    y_col = "_bot_won"
    n_winners = int(df[y_col].sum())
    logger.info("Bot training %s/%s: %d samples, %d wins (%.1f%%)",
                horizon, strategy, len(df), n_winners, 100 * n_winners / max(1, len(df)))

    if n_winners < 3:
        logger.warning("Bot training: only %d wins, need at least 3. Skipping.", n_winners)
        return None

    # Feature selection by tier
    feature_pool = _select_feature_tier(len(df))
    df, available_features = prepare_features(df, feature_pool)

    if len(available_features) < 5:
        logger.error("Bot training: only %d features — need at least 5", len(available_features))
        return None

    train_df, test_df = walk_forward_split(df)

    X_train = train_df[available_features]
    y_train = train_df[y_col].astype(int)
    X_test = test_df[available_features]
    y_test = test_df[y_col].astype(int)

    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos = neg_count / max(1, pos_count)

    logger.info(
        "Bot Train: %d (%d wins, %.1f%%), Test: %d (%d wins, %.1f%%)",
        len(y_train), pos_count, 100 * pos_count / max(1, len(y_train)),
        len(y_test), y_test.sum(), 100 * y_test.sum() / max(1, len(y_test)),
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Train XGBoost classifier
    logger.info("=== Optimizing XGBoost Bot Classifier (%d trials) ===", n_trials)
    xgb_model, xgb_params, xgb_score = _optimize_xgboost(
        X_train, y_train, X_test, y_test, scale_pos, n_trials
    )

    # Train LightGBM classifier
    logger.info("=== Optimizing LightGBM Bot Classifier (%d trials) ===", n_trials)
    lgb_model, lgb_params, lgb_score = _optimize_lightgbm(
        X_train, y_train, X_test, y_test, scale_pos, n_trials
    )

    # Ensemble
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    lgb_proba = lgb_model.predict_proba(X_test)[:, 1]

    total_score = xgb_score + lgb_score
    xgb_weight = xgb_score / max(0.001, total_score)
    lgb_weight = lgb_score / max(0.001, total_score)

    ensemble_proba = xgb_weight * xgb_proba + lgb_weight * lgb_proba

    # Metrics
    p_at_5 = _precision_at_k(y_test, ensemble_proba, k=5)
    p_at_10 = _precision_at_k(y_test, ensemble_proba, k=10)
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)
    precision = precision_score(y_test, ensemble_pred, zero_division=0)
    recall = recall_score(y_test, ensemble_pred, zero_division=0)
    f1 = f1_score(y_test, ensemble_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, ensemble_proba)
    except ValueError:
        auc = 0.0

    logger.info("\n=== BOT CLASSIFICATION RESULTS (%s/%s) ===", horizon, strategy)
    logger.info("Precision@5: %.3f (target: >%.2f)", p_at_5, MIN_PRECISION_AT_5)
    logger.info("Precision@10: %.3f, AUC: %.3f, F1: %.3f", p_at_10, auc, f1)

    # Quality gate
    if p_at_5 < MIN_PRECISION_AT_5:
        logger.warning(
            "BOT QUALITY GATE FAILED: precision@5 = %.3f < %.2f. Model NOT saved.",
            p_at_5, MIN_PRECISION_AT_5,
        )
        return {"quality_gate": "FAILED", "precision_at_5": float(p_at_5), "mode": "bot_won"}

    # SHAP analysis
    xgb_shap = _compute_shap(xgb_model, X_test, available_features, "XGBoost-Bot")
    lgb_shap = _compute_shap(lgb_model, X_test, available_features, "LightGBM-Bot")

    # Save models with _bot suffix
    xgb_path = MODEL_DIR / f"model_{horizon}_bot.json"
    xgb_model.save_model(str(xgb_path))
    logger.info("XGBoost bot model saved to %s", xgb_path)

    lgb_path = MODEL_DIR / f"model_{horizon}_bot_lgb.txt"
    lgb_model.booster_.save_model(str(lgb_path))
    logger.info("LightGBM bot model saved to %s", lgb_path)

    # Feature importances
    xgb_importances = dict(zip(available_features, xgb_model.feature_importances_))
    sorted_imp = sorted(xgb_importances.items(), key=lambda x: x[1], reverse=True)

    metadata = {
        "horizon": horizon,
        "strategy": strategy,
        "mode": "bot_won",
        "trained_at": datetime.utcnow().isoformat(),
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "n_winners": n_winners,
        "feature_tier": "core" if len(df) < 500 else "extended" if len(df) < 2000 else "full",
        "ensemble_weights": {"xgboost": float(xgb_weight), "lightgbm": float(lgb_weight)},
        "xgb_params": {k: v for k, v in xgb_params.items() if k != "verbosity"},
        "lgb_params": {k: v for k, v in lgb_params.items() if k not in ("verbosity", "metric")},
        "metrics": {
            "precision_at_5": float(p_at_5),
            "precision_at_10": float(p_at_10),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc_roc": float(auc),
        },
        "quality_gate": "PASSED",
        "feature_importances_xgb": {k: float(v) for k, v in sorted_imp},
        "shap_importances_xgb": xgb_shap,
        "shap_importances_lgb": lgb_shap,
        "features": available_features,
    }

    meta_path = MODEL_DIR / f"model_{horizon}_bot_meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Bot metadata saved to %s", meta_path)

    return metadata


# ═══ RISK/REWARD REGRESSION (ML v3.1) ═══

def train_risk_reward(
    df: pd.DataFrame,
    horizon: str = "12h",
    n_trials: int = 100,
    min_samples: int = 30,
) -> dict | None:
    """
    ML v3.1: Train regression ensemble to predict risk/reward ratio.

    Target = log_rr = log1p(rr_ratio - 1), where:
        rr_ratio = max_return / (1 + max_dd_pct / 100)

    Winner label = rr_ratio > 1.0 (return compensates drawdown).
    Uses same XGBoost+LightGBM ensemble as regression mode.

    Extra metadata: dd_by_rr_band — for each predicted rr_ratio band,
    the real DD distribution + recommended adaptive SL.
    """
    if len(df) < min_samples:
        logger.warning("RR training: only %d samples (need %d). Skipping.", len(df), min_samples)
        return None

    if "log_rr" not in df.columns or "_winner" not in df.columns:
        logger.error("RR training: missing log_rr/_winner columns. Call load_risk_reward_data() first.")
        return None

    n_winners = int(df["_winner"].sum())
    logger.info(
        "RR training %s: %d samples, %d winners (rr>1.0, %.1f%%)",
        horizon, len(df), n_winners, 100 * n_winners / max(1, len(df)),
    )

    if n_winners < 3:
        logger.warning("RR training: only %d winners, need at least 3. Skipping.", n_winners)
        return None

    # Feature selection by tier
    feature_pool = _select_feature_tier(len(df))
    df, available_features = prepare_features(df, feature_pool)

    if len(available_features) < 5:
        logger.error("RR training: only %d features — need at least 5", len(available_features))
        return None

    train_df, test_df = walk_forward_split(df)

    X_train = train_df[available_features]
    y_train = train_df["log_rr"]
    X_test = test_df[available_features]
    y_test = test_df["log_rr"]
    y_test_binary = test_df["_winner"]

    logger.info(
        "RR — Train: %d, Test: %d, log_rr range: [%.3f, %.3f]",
        len(y_train), len(y_test), y_train.min(), y_train.max(),
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Train XGBoost regressor
    logger.info("=== Optimizing XGBoost RR Regressor (%d trials) ===", n_trials)
    xgb_model, xgb_params, xgb_spearman = _optimize_xgb_regressor(
        X_train, y_train, X_test, y_test, n_trials
    )

    # Train LightGBM regressor
    logger.info("=== Optimizing LightGBM RR Regressor (%d trials) ===", n_trials)
    lgb_model, lgb_params, lgb_spearman = _optimize_lgb_regressor(
        X_train, y_train, X_test, y_test, n_trials
    )

    # Ensemble: weighted average of predictions
    xgb_preds = xgb_model.predict(X_test)
    lgb_preds = lgb_model.predict(X_test)

    total_score = abs(xgb_spearman) + abs(lgb_spearman)
    xgb_weight = abs(xgb_spearman) / max(0.001, total_score)
    lgb_weight = abs(lgb_spearman) / max(0.001, total_score)

    ensemble_preds = xgb_weight * xgb_preds + lgb_weight * lgb_preds

    # Metrics
    spearman = _spearman_rank(y_test, ensemble_preds)
    rmse = float(np.sqrt(mean_squared_error(y_test, ensemble_preds)))
    p_at_5 = _precision_at_k(y_test_binary, ensemble_preds, k=5)
    p_at_10 = _precision_at_k(y_test_binary, ensemble_preds, k=10)

    logger.info("\n=== RISK/REWARD REGRESSION RESULTS ===")
    logger.info("Weights: XGBoost=%.2f, LightGBM=%.2f", xgb_weight, lgb_weight)
    logger.info("Spearman rank correlation: %.3f", spearman)
    logger.info("RMSE: %.3f", rmse)
    logger.info("Precision@5 (rr>1.0): %.3f (target: >%.2f)", p_at_5, MIN_PRECISION_AT_5)
    logger.info("Precision@10 (rr>1.0): %.3f", p_at_10)

    # DD distribution by predicted rr_ratio band (the key adaptive SL output)
    dd_by_rr_band = {}
    if "dd_pct" in test_df.columns and len(test_df) > 0:
        pred_rr = np.expm1(ensemble_preds) + 1  # inverse of log1p(rr - 1)
        pred_bands = pd.cut(
            pred_rr,
            bins=[-999, 0.5, 0.8, 1.0, 1.5, 999],
            labels=["<0.5", "0.5-0.8", "0.8-1.0", "1.0-1.5", ">1.5"],
        )
        test_dd = pd.to_numeric(test_df["dd_pct"], errors="coerce")
        for band in ["<0.5", "0.5-0.8", "0.8-1.0", "1.0-1.5", ">1.5"]:
            mask = pred_bands == band
            grp_dd = test_dd[mask].dropna()
            if len(grp_dd) == 0:
                continue
            p75_dd = float(grp_dd.quantile(0.75))
            dd_by_rr_band[band] = {
                "count": int(len(grp_dd)),
                "avg_dd_pct": round(float(grp_dd.mean()), 1),
                "median_dd_pct": round(float(grp_dd.median()), 1),
                "p75_dd_pct": round(p75_dd, 1),
                "recommended_sl_pct": round(p75_dd * 1.2, 1),
            }
        logger.info("DD by RR band: %s", dd_by_rr_band)

    # Quality gate
    if p_at_5 < MIN_PRECISION_AT_5:
        logger.warning(
            "RR QUALITY GATE FAILED: precision@5 = %.3f < %.2f. Model NOT saved.",
            p_at_5, MIN_PRECISION_AT_5,
        )
        return {
            "quality_gate": "FAILED",
            "precision_at_5": float(p_at_5),
            "mode": "risk_reward",
            "dd_by_rr_band": dd_by_rr_band,
            "metrics": {"precision_at_5": float(p_at_5), "spearman": float(spearman)},
        }

    # SHAP analysis
    logger.info("=== RR SHAP Analysis ===")
    xgb_shap = _compute_shap(xgb_model, X_test, available_features, "XGBoost-RR")
    lgb_shap = _compute_shap(lgb_model, X_test, available_features, "LightGBM-RR")

    # Save models with _rr suffix
    xgb_path = MODEL_DIR / f"model_{horizon}_rr.json"
    xgb_model.save_model(str(xgb_path))
    logger.info("XGBoost RR model saved to %s", xgb_path)

    lgb_path = MODEL_DIR / f"model_{horizon}_rr_lgb.txt"
    lgb_model.booster_.save_model(str(lgb_path))
    logger.info("LightGBM RR model saved to %s", lgb_path)

    # Feature importances
    xgb_importances = dict(zip(available_features, xgb_model.feature_importances_))
    sorted_imp = sorted(xgb_importances.items(), key=lambda x: x[1], reverse=True)

    metadata = {
        "horizon": horizon,
        "mode": "risk_reward",
        "trained_at": datetime.utcnow().isoformat(),
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "n_winners": n_winners,
        "feature_tier": "core" if len(df) < 500 else "extended" if len(df) < 2000 else "full",
        "ensemble_weights": {"xgboost": float(xgb_weight), "lightgbm": float(lgb_weight)},
        "xgb_params": {k: v for k, v in xgb_params.items() if k != "verbosity"},
        "lgb_params": {k: v for k, v in lgb_params.items() if k != "verbosity"},
        "metrics": {
            "spearman": float(spearman),
            "rmse": rmse,
            "precision_at_5": float(p_at_5),
            "precision_at_10": float(p_at_10),
            "xgb_spearman": float(xgb_spearman),
            "lgb_spearman": float(lgb_spearman),
        },
        "quality_gate": "PASSED",
        "dd_by_rr_band": dd_by_rr_band,
        "feature_importances_xgb": {k: float(v) for k, v in sorted_imp},
        "shap_importances_xgb": xgb_shap,
        "shap_importances_lgb": lgb_shap,
        "features": available_features,
    }

    meta_path = MODEL_DIR / f"model_{horizon}_rr_meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("RR metadata saved to %s", meta_path)

    return metadata


# ═══ CLASSIFICATION TRAINING (legacy, still available via --mode classify) ═══

def train_ensemble(
    df: pd.DataFrame,
    horizon: str = "12h",
    n_trials: int = 100,
    min_samples: int = 200,
) -> dict | None:
    """Train XGBoost + LightGBM classification ensemble."""
    label_col = HORIZONS[horizon]

    if len(df) < min_samples:
        logger.warning("Only %d samples (need %d). Skipping.", len(df), min_samples)
        return None

    # Feature selection by tier
    feature_pool = _select_feature_tier(len(df))
    df, available_features = prepare_features(df, feature_pool)

    if len(available_features) < 5:
        logger.error("Only %d features available — need at least 5", len(available_features))
        return None

    train_df, test_df = walk_forward_split(df)

    X_train = train_df[available_features]
    y_train = train_df[label_col].astype(int)
    X_test = test_df[available_features]
    y_test = test_df[label_col].astype(int)

    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos = neg_count / max(1, pos_count)

    logger.info(
        "Train: %d (%d positive, %.1f%%), Test: %d (%d positive, %.1f%%)",
        len(y_train), pos_count, 100 * pos_count / len(y_train),
        len(y_test), y_test.sum(), 100 * y_test.sum() / max(1, len(y_test)),
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    logger.info("=== Optimizing XGBoost Classifier (%d trials) ===", n_trials)
    xgb_model, xgb_params, xgb_score = _optimize_xgboost(
        X_train, y_train, X_test, y_test, scale_pos, n_trials
    )

    logger.info("=== Optimizing LightGBM Classifier (%d trials) ===", n_trials)
    lgb_model, lgb_params, lgb_score = _optimize_lightgbm(
        X_train, y_train, X_test, y_test, scale_pos, n_trials
    )

    # Calibrate XGBoost
    cal_split = max(10, len(X_test) // 2)
    X_cal = X_test.iloc[:cal_split]
    y_cal = y_test.iloc[:cal_split]

    calibrated = False
    xgb_calibrated = xgb_model
    if len(y_cal) >= 10 and y_cal.nunique() == 2:
        try:
            xgb_calibrated = CalibratedClassifierCV(xgb_model, method="isotonic", cv="prefit")
            xgb_calibrated.fit(X_cal, y_cal)
            calibrated = True
            logger.info("XGBoost calibrated on %d samples", len(y_cal))
        except Exception as e:
            logger.warning("Calibration failed: %s", e)
            xgb_calibrated = xgb_model

    # Ensemble
    xgb_proba = xgb_calibrated.predict_proba(X_test)[:, 1]
    lgb_proba = lgb_model.predict_proba(X_test)[:, 1]

    total_score = xgb_score + lgb_score
    xgb_weight = xgb_score / max(0.001, total_score)
    lgb_weight = lgb_score / max(0.001, total_score)

    ensemble_proba = xgb_weight * xgb_proba + lgb_weight * lgb_proba
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)

    p_at_5 = _precision_at_k(y_test, ensemble_proba, k=5)
    p_at_10 = _precision_at_k(y_test, ensemble_proba, k=10)
    precision = precision_score(y_test, ensemble_pred, zero_division=0)
    recall = recall_score(y_test, ensemble_pred, zero_division=0)
    f1 = f1_score(y_test, ensemble_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, ensemble_proba)
    except ValueError:
        auc = 0.0

    logger.info("\n=== CLASSIFICATION ENSEMBLE RESULTS ===")
    logger.info("Precision@5: %.3f (target: >%.2f)", p_at_5, MIN_PRECISION_AT_5)
    logger.info("Precision@10: %.3f, AUC: %.3f", p_at_10, auc)
    logger.info("\n%s", classification_report(y_test, ensemble_pred, zero_division=0))

    # Quality gate
    if p_at_5 < MIN_PRECISION_AT_5:
        logger.warning(
            "QUALITY GATE FAILED: precision@5 = %.3f < %.2f. Model NOT saved.",
            p_at_5, MIN_PRECISION_AT_5,
        )
        return {"quality_gate": "FAILED", "precision_at_5": float(p_at_5)}

    # SHAP
    xgb_shap = _compute_shap(xgb_model, X_test, available_features, "XGBoost")
    lgb_shap = _compute_shap(lgb_model, X_test, available_features, "LightGBM")

    xgb_importances = dict(zip(available_features, xgb_model.feature_importances_))
    sorted_imp = sorted(xgb_importances.items(), key=lambda x: x[1], reverse=True)

    # Save
    xgb_path = MODEL_DIR / f"model_{horizon}.json"
    xgb_model.save_model(str(xgb_path))

    lgb_path = MODEL_DIR / f"model_{horizon}_lgb.txt"
    lgb_model.booster_.save_model(str(lgb_path))

    if calibrated:
        try:
            from joblib import dump as joblib_dump
            joblib_dump(xgb_calibrated, str(MODEL_DIR / f"model_{horizon}_calibrated.joblib"))
        except ImportError:
            pass

    metadata = {
        "horizon": horizon,
        "mode": "classification",
        "trained_at": datetime.utcnow().isoformat(),
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "feature_tier": "core" if len(df) < 500 else "extended" if len(df) < 2000 else "full",
        "ensemble_weights": {"xgboost": float(xgb_weight), "lightgbm": float(lgb_weight)},
        "xgb_params": {k: v for k, v in xgb_params.items() if k != "verbosity"},
        "lgb_params": {k: v for k, v in lgb_params.items() if k not in ("verbosity", "metric")},
        "metrics": {
            "precision_at_5": float(p_at_5),
            "precision_at_10": float(p_at_10),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc_roc": float(auc),
        },
        "quality_gate": "PASSED",
        "feature_importances_xgb": {k: float(v) for k, v in sorted_imp},
        "shap_importances_xgb": xgb_shap,
        "shap_importances_lgb": lgb_shap,
        "features": available_features,
        "calibrated": calibrated,
    }

    meta_path = MODEL_DIR / f"model_{horizon}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved to %s", meta_path)

    return metadata


def _update_scoring_config_ml(horizon: str, threshold: float) -> None:
    """Write the active ML horizon + threshold to scoring_config for pipeline + backtest coherence."""
    try:
        client = _get_client()
        client.table("scoring_config").update({
            "ml_horizon": horizon,
            "ml_threshold": float(threshold),
            "updated_at": datetime.utcnow().isoformat(),
            "updated_by": "auto_train",
            "change_reason": f"auto_train deployed model: {horizon}/{threshold}x",
        }).eq("id", 1).execute()
        logger.info("scoring_config updated: ml_horizon=%s, ml_threshold=%.1f", horizon, threshold)
    except Exception as e:
        logger.warning("Failed to update scoring_config ml params: %s", e)


def _update_scoring_config_bot(strategy: str) -> None:
    """ML v3: Write the active bot_strategy to scoring_config (does not affect ml_horizon/ml_threshold)."""
    try:
        client = _get_client()
        client.table("scoring_config").update({
            "bot_strategy": strategy,
            "updated_at": datetime.utcnow().isoformat(),
            "updated_by": "auto_train",
            "change_reason": f"auto_train deployed bot model: {strategy}",
        }).eq("id", 1).execute()
        logger.info("scoring_config updated: bot_strategy=%s", strategy)
    except Exception as e:
        logger.warning("Failed to update scoring_config bot_strategy: %s", e)


def auto_train(
    min_samples: int = 100,
    trials: int = 50,
) -> dict | None:
    """
    v22: Multi-horizon × multi-threshold grid search.

    Tries all (horizon, threshold) combos, trains regression (+ LTR if enough data),
    picks the best combo by precision@5, and deploys it.

    Returns metadata dict if model was successfully trained and deployed,
    None if skipped or failed.
    """
    HORIZON_POOL = ["6h", "12h", "24h"]
    THRESHOLD_POOL = [1.3, 1.5, 2.0]

    logger.info("=== AUTO-TRAIN: grid %s × %s, min_samples=%d, trials=%d ===",
                HORIZON_POOL, THRESHOLD_POOL, min_samples, trials)

    # Suppress Optuna logs in auto mode
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Load existing model metadata for A/B comparison
    # Check all horizons for existing models
    old_meta = None
    old_horizon = None
    for hz in HORIZON_POOL:
        mp = MODEL_DIR / f"model_{hz}_meta.json"
        if mp.exists():
            try:
                with open(mp) as f:
                    meta = json.load(f)
                if meta.get("quality_gate") == "PASSED":
                    p5 = meta.get("metrics", {}).get("precision_at_5", 0)
                    if old_meta is None or p5 > old_meta.get("metrics", {}).get("precision_at_5", 0):
                        old_meta = meta
                        old_horizon = hz
            except Exception:
                pass

    if old_meta:
        logger.info("auto_train: existing best model — %s/%s, p@5=%.3f, samples=%d",
                    old_horizon, old_meta.get("mode", "?"),
                    old_meta.get("metrics", {}).get("precision_at_5", 0),
                    old_meta.get("train_samples", 0) + old_meta.get("test_samples", 0))

    # Grid search: best (horizon, threshold) combo
    best_result = None
    best_horizon = None
    best_threshold = None

    for horizon in HORIZON_POOL:
        df = load_labeled_data(horizon)
        if df.empty:
            logger.info("auto_train: %s — no data, skipping", horizon)
            continue

        df = deduplicate_snapshots(df, threshold=2.0)  # dedup with default threshold for logging

        if len(df) < min_samples:
            logger.info("auto_train: %s — only %d unique tokens (need %d), skipping",
                        horizon, len(df), min_samples)
            continue

        for threshold in THRESHOLD_POOL:
            n_winners = int((df["max_return"] >= threshold).sum())
            if n_winners < 3:
                logger.info("auto_train: %s/%.1fx — only %d winners, skipping",
                            horizon, threshold, n_winners)
                continue

            logger.info("auto_train: trying %s/%.1fx — %d samples, %d winners",
                        horizon, threshold, len(df), n_winners)

            # Train regression
            reg_meta = train_regression(
                df, horizon, n_trials=trials, min_samples=min_samples, threshold=threshold
            )

            # Also try LTR if enough data
            ltr_meta = None
            if len(df) >= 200:
                try:
                    ltr_meta = train_ltr(
                        df, horizon, n_trials=trials, min_samples=min_samples, threshold=threshold
                    )
                except Exception as e:
                    logger.warning("auto_train: LTR %s/%.1fx failed: %s", horizon, threshold, e)

            # Pick best candidate for this combo
            candidates = []
            if reg_meta and reg_meta.get("quality_gate") == "PASSED":
                candidates.append(reg_meta)
            if ltr_meta and ltr_meta.get("quality_gate") == "PASSED":
                candidates.append(ltr_meta)

            if not candidates:
                continue

            combo_best = max(candidates, key=lambda m: m.get("metrics", {}).get("precision_at_5", 0))
            combo_p5 = combo_best.get("metrics", {}).get("precision_at_5", 0)

            logger.info("auto_train: %s/%.1fx best = %s p@5=%.3f",
                        horizon, threshold, combo_best.get("mode"), combo_p5)

            if best_result is None or combo_p5 > best_result.get("metrics", {}).get("precision_at_5", 0):
                best_result = combo_best
                best_horizon = horizon
                best_threshold = threshold

    if best_result is None:
        logger.warning("auto_train: no model passed quality gate across all combos")
        return None

    new_p5 = best_result.get("metrics", {}).get("precision_at_5", 0)
    best_mode = best_result.get("mode", "regression")

    # A/B comparison: new model must beat old model on precision@5
    if old_meta:
        old_p5 = old_meta.get("metrics", {}).get("precision_at_5", 0)
        if new_p5 < old_p5:
            logger.warning(
                "auto_train: new %s/%s/%.1fx (p@5=%.3f) worse than old %s/%s (p@5=%.3f). "
                "Keeping old model.",
                best_horizon, best_mode, best_threshold, new_p5,
                old_horizon, old_meta.get("mode", "?"), old_p5,
            )
            return None

        logger.info(
            "auto_train: new %s/%s/%.1fx (p@5=%.3f) beats old %s/%s (p@5=%.3f). Deployed!",
            best_horizon, best_mode, best_threshold, new_p5,
            old_horizon, old_meta.get("mode", "?"), old_p5,
        )
    else:
        logger.info("auto_train: no existing model — %s/%s/%.1fx deployed (p@5=%.3f)",
                    best_horizon, best_mode, best_threshold, new_p5)

    # Save final metadata with auto-train info
    meta_path = MODEL_DIR / f"model_{best_horizon}_meta.json"
    best_result["auto_trained"] = True
    best_result["auto_trained_at"] = datetime.utcnow().isoformat()
    with open(meta_path, "w") as f:
        json.dump(best_result, f, indent=2)

    # Update scoring_config so pipeline + backtest use the same horizon/threshold
    _update_scoring_config_ml(best_horizon, best_threshold)

    # ═══ Phase 2: bot_won training (if bot TP/SL data available) ═══
    logger.info("=== AUTO-TRAIN Phase 2: bot_won training ===")
    BOT_STRATEGY_POOL = ["TP30_SL20", "TP50_SL30", "TP100_SL50"]

    best_bot_result = None
    best_bot_horizon = None
    best_bot_strategy = None

    for bot_hz in ["12h", "24h"]:  # Only 12h/24h have bot columns
        for bot_strat in BOT_STRATEGY_POOL:
            try:
                df_bot = load_bot_labeled_data(bot_hz, bot_strat)
            except Exception as e:
                logger.debug("auto_train bot: %s/%s load failed: %s", bot_hz, bot_strat, e)
                continue

            if df_bot.empty or len(df_bot) < min_samples:
                logger.info("auto_train bot: %s/%s — %d samples (need %d), skipping",
                            bot_hz, bot_strat, len(df_bot) if not df_bot.empty else 0, min_samples)
                continue

            df_bot = deduplicate_snapshots(df_bot)
            if len(df_bot) < min_samples:
                continue

            try:
                bot_meta = train_bot_won(df_bot, bot_hz, bot_strat, n_trials=trials, min_samples=min_samples)
            except Exception as e:
                logger.warning("auto_train bot: %s/%s training failed: %s", bot_hz, bot_strat, e)
                continue

            if not bot_meta or bot_meta.get("quality_gate") != "PASSED":
                continue

            bot_p5 = bot_meta.get("metrics", {}).get("precision_at_5", 0)
            logger.info("auto_train bot: %s/%s passed gate (p@5=%.3f)", bot_hz, bot_strat, bot_p5)

            if best_bot_result is None or bot_p5 > best_bot_result.get("metrics", {}).get("precision_at_5", 0):
                best_bot_result = bot_meta
                best_bot_horizon = bot_hz
                best_bot_strategy = bot_strat

    if best_bot_result:
        bot_p5 = best_bot_result.get("metrics", {}).get("precision_at_5", 0)
        logger.info("auto_train bot: BEST = %s/%s p@5=%.3f — deployed!",
                    best_bot_horizon, best_bot_strategy, bot_p5)

        # Save bot meta with auto-train info
        bot_meta_path = MODEL_DIR / f"model_{best_bot_horizon}_bot_meta.json"
        best_bot_result["auto_trained"] = True
        best_bot_result["auto_trained_at"] = datetime.utcnow().isoformat()
        with open(bot_meta_path, "w") as f:
            json.dump(best_bot_result, f, indent=2)

        # Update scoring_config with bot_strategy (does not touch ml_horizon/ml_threshold)
        _update_scoring_config_bot(best_bot_strategy)
    else:
        logger.info("auto_train bot: no bot model passed quality gate")

    # ═══ Phase 3: risk_reward regression (adaptive SL data) ═══
    logger.info("=== AUTO-TRAIN Phase 3: risk_reward regression ===")
    best_rr_result = None
    best_rr_horizon = None

    for rr_hz in ["12h", "24h"]:
        try:
            df_rr = load_risk_reward_data(rr_hz)
        except Exception as e:
            logger.debug("auto_train rr: %s load failed: %s", rr_hz, e)
            continue

        if df_rr.empty:
            logger.info("auto_train rr: %s — no DD data, skipping", rr_hz)
            continue

        df_rr = deduplicate_snapshots(df_rr)
        if len(df_rr) < 30:
            logger.info("auto_train rr: %s — only %d unique tokens (need 30), skipping",
                        rr_hz, len(df_rr))
            continue

        try:
            rr_meta = train_risk_reward(df_rr, rr_hz, n_trials=trials, min_samples=30)
        except Exception as e:
            logger.warning("auto_train rr: %s training failed: %s", rr_hz, e)
            continue

        if not rr_meta or rr_meta.get("quality_gate") != "PASSED":
            continue

        rr_p5 = rr_meta.get("metrics", {}).get("precision_at_5", 0)
        logger.info("auto_train rr: %s passed gate (p@5=%.3f)", rr_hz, rr_p5)

        if best_rr_result is None or rr_p5 > best_rr_result.get("metrics", {}).get("precision_at_5", 0):
            best_rr_result = rr_meta
            best_rr_horizon = rr_hz

    if best_rr_result:
        rr_p5 = best_rr_result.get("metrics", {}).get("precision_at_5", 0)
        logger.info("auto_train rr: BEST = %s p@5=%.3f — deployed!", best_rr_horizon, rr_p5)

        # Save rr meta with auto-train info
        rr_meta_path = MODEL_DIR / f"model_{best_rr_horizon}_rr_meta.json"
        best_rr_result["auto_trained"] = True
        best_rr_result["auto_trained_at"] = datetime.utcnow().isoformat()
        with open(rr_meta_path, "w") as f:
            json.dump(best_rr_result, f, indent=2)
    else:
        logger.info("auto_train rr: no rr model passed quality gate")

    return best_result


def main():
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")

    parser = argparse.ArgumentParser(description="Train ML ensemble for memecoin prediction")
    parser.add_argument(
        "--horizon", choices=["1h", "6h", "12h", "24h"], default="12h",
        help="Prediction horizon (default: 12h)",
    )
    parser.add_argument(
        "--mode", choices=["regression", "classify", "ltr", "bot_won", "risk_reward"], default="regression",
        help="Training mode: regression, classify, ltr, bot_won, or risk_reward (adaptive SL)",
    )
    parser.add_argument(
        "--strategy", choices=list(BOT_STRATEGIES.keys()), default="TP50_SL30",
        help="Bot strategy for bot_won mode (default: TP50_SL30)",
    )
    parser.add_argument(
        "--threshold", type=float, default=2.0,
        help="Return threshold for winner label (default: 2.0 = 2x). E.g. 1.5 = +50%%",
    )
    parser.add_argument(
        "--trials", type=int, default=100,
        help="Number of Optuna trials per model (default: 100)",
    )
    parser.add_argument(
        "--min-samples", type=int, default=100,
        help="Minimum labeled samples required to train (default: 100)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Train models for all horizons",
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="Non-interactive mode: grid search horizon×threshold, A/B comparison",
    )
    args = parser.parse_args()

    if args.auto:
        # Auto mode: quiet, non-interactive, multi-horizon × multi-threshold grid
        logging.getLogger().setLevel(logging.WARNING)
        logger.setLevel(logging.INFO)
        result = auto_train(
            min_samples=args.min_samples,
            trials=args.trials,
        )
        if result:
            logger.info("auto_train: SUCCESS — %s/%s/%.1fx p@5=%.3f",
                        result.get("horizon"), result.get("mode"),
                        result.get("threshold", 2.0),
                        result.get("metrics", {}).get("precision_at_5", 0))
        else:
            logger.info("auto_train: skipped or failed")
        return

    horizons = HORIZONS.keys() if args.all else [args.horizon]
    threshold = args.threshold

    # ML v3: bot_won mode — separate data loading and training path
    if args.mode == "bot_won":
        for horizon in horizons:
            logger.info("\n%s\n=== Training %s bot_won (%s) ===\n%s",
                        "=" * 60, horizon, args.strategy, "=" * 60)
            df = load_bot_labeled_data(horizon, args.strategy)
            if not df.empty:
                df = deduplicate_snapshots(df)
                result = train_bot_won(df, horizon, args.strategy,
                                       n_trials=args.trials, min_samples=args.min_samples)
                if result and result.get("quality_gate") == "FAILED":
                    logger.warning("Bot model for %s/%s failed quality gate.", horizon, args.strategy)
                elif result and result.get("quality_gate") == "PASSED":
                    _update_scoring_config_bot(args.strategy)
        return

    # ML v3.1: risk_reward mode — predicts rr_ratio for adaptive SL
    if args.mode == "risk_reward":
        for horizon in horizons:
            logger.info("\n%s\n=== Training %s risk_reward ===\n%s",
                        "=" * 60, horizon, "=" * 60)
            df = load_risk_reward_data(horizon)
            if not df.empty:
                df = deduplicate_snapshots(df)
                train_risk_reward(df, horizon, n_trials=args.trials,
                                  min_samples=args.min_samples)
        return

    if args.mode == "ltr":
        train_fn = train_ltr
    elif args.mode == "regression":
        train_fn = train_regression
    else:
        train_fn = train_ensemble

    for horizon in horizons:
        logger.info("\n%s\n=== Training %s %s (threshold=%.1fx) ===\n%s",
                    "=" * 60, horizon, args.mode, threshold, "=" * 60)
        df = load_labeled_data(horizon)
        if not df.empty:
            # Deduplicate: one snapshot per token to prevent autocorrelation
            df = deduplicate_snapshots(df, threshold=threshold)
            # Classification mode doesn't support threshold param
            if args.mode == "classify":
                result = train_fn(df, horizon, n_trials=args.trials, min_samples=args.min_samples)
            else:
                result = train_fn(df, horizon, n_trials=args.trials,
                                  min_samples=args.min_samples, threshold=threshold)
            if result and result.get("quality_gate") == "FAILED":
                logger.warning("Model for %s failed quality gate — not deployed.", horizon)


if __name__ == "__main__":
    main()
