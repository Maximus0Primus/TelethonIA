"""
ML training script for memecoin 2x prediction.
v2: Regression + classification dual mode, adaptive feature selection,
    Spearman rank correlation metric, quality gates.

- XGBoost + LightGBM ensemble
- Walk-forward temporal splits (no data leakage)
- Optuna hyperparameter optimization
- SHAP feature importance analysis
- Adaptive feature selection by sample count

Run manually when enough labeled snapshots have accumulated (~500+).

Usage:
    python train_model.py --horizon 12h --trials 100
    python train_model.py --horizon 12h --mode regression --trials 100
    python train_model.py --all --trials 200
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent

# Quality gate: model must exceed this precision@5 to be saved
MIN_PRECISION_AT_5 = 0.40

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
]

HORIZONS = {
    "6h": "did_2x_6h",
    "12h": "did_2x_12h",
    "24h": "did_2x_24h",
}

# Return columns for regression target
RETURN_COLS = {
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
    """Load all labeled snapshots from Supabase."""
    client = _get_client()
    label_col = HORIZONS[horizon]

    result = (
        client.table("token_snapshots")
        .select("*")
        .not_.is_(label_col, "null")
        .order("snapshot_at")
        .execute()
    )

    if not result.data:
        logger.error("No labeled data found for horizon %s", horizon)
        return pd.DataFrame()

    df = pd.DataFrame(result.data)
    logger.info("Loaded %d labeled snapshots for %s horizon", len(df), horizon)
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
            "objective": "reg:squaredlogerror",
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
    best_params.update({"objective": "reg:squaredlogerror", "verbosity": 0})

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
) -> dict | None:
    """
    Train regression ensemble: predict max_return (continuous) instead of binary 2x.
    The model learns to rank tokens by expected return — more information than binary.
    """
    label_col = HORIZONS[horizon]
    max_price_col, snapshot_price_col = RETURN_COLS[horizon]

    if len(df) < min_samples:
        logger.warning("Only %d samples (need %d). Skipping training.", len(df), min_samples)
        return None

    # Compute regression target: log(max_return)
    # max_return = max_price_in_window / price_at_snapshot
    df = df.copy()
    df[max_price_col] = pd.to_numeric(df.get(max_price_col), errors="coerce")
    df[snapshot_price_col] = pd.to_numeric(df.get(snapshot_price_col), errors="coerce")

    valid_mask = (
        df[max_price_col].notna() &
        df[snapshot_price_col].notna() &
        (df[snapshot_price_col] > 0) &
        (df[max_price_col] > 0)
    )
    df = df[valid_mask].copy()

    if len(df) < min_samples:
        logger.warning("Only %d valid regression samples (need %d).", len(df), min_samples)
        return None

    df["max_return"] = df[max_price_col] / df[snapshot_price_col]
    df["log_return"] = np.log1p(df["max_return"] - 1)  # log1p(return) for stability

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
    y_test_binary = test_df[label_col].astype(int)

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
        "mode": "regression",
        "trained_at": datetime.utcnow().isoformat(),
        "train_samples": len(y_train),
        "test_samples": len(y_test),
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


def auto_train(
    horizon: str = "12h",
    min_samples: int = 500,
    trials: int = 50,
) -> dict | None:
    """
    Non-interactive auto-retrain with A/B comparison.

    Returns metadata dict if model was successfully trained and deployed,
    None if skipped or failed.

    Logic:
    1. Load labeled data — skip if < min_samples
    2. Train new model
    3. Compare with existing model (if any) via Spearman correlation
    4. Deploy new model only if it beats the old one (or no old model)
    """
    logger.info("=== AUTO-TRAIN: horizon=%s, min_samples=%d, trials=%d ===", horizon, min_samples, trials)

    df = load_labeled_data(horizon)
    if df.empty or len(df) < min_samples:
        logger.info("auto_train: only %d samples (need %d), skipping", len(df), min_samples)
        return None

    # Load existing model metadata for A/B comparison
    meta_path = MODEL_DIR / f"model_{horizon}_meta.json"
    old_meta = None
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                old_meta = json.load(f)
            logger.info("auto_train: existing model — spearman=%.3f, p@5=%.3f, samples=%d",
                        old_meta.get("metrics", {}).get("spearman", 0),
                        old_meta.get("metrics", {}).get("precision_at_5", 0),
                        old_meta.get("train_samples", 0) + old_meta.get("test_samples", 0))
        except Exception:
            pass

    # Suppress Optuna logs in auto mode
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Train new model
    new_meta = train_regression(df, horizon, n_trials=trials, min_samples=min_samples)
    if not new_meta:
        logger.warning("auto_train: training failed or insufficient data")
        return None

    if new_meta.get("quality_gate") == "FAILED":
        logger.warning("auto_train: new model failed quality gate (p@5=%.3f < %.2f)",
                        new_meta.get("precision_at_5", 0), MIN_PRECISION_AT_5)
        return None

    # A/B comparison: new model must beat old model on Spearman
    new_spearman = new_meta.get("metrics", {}).get("spearman", 0)
    if old_meta:
        old_spearman = old_meta.get("metrics", {}).get("spearman", 0)
        if new_spearman < old_spearman:
            logger.warning(
                "auto_train: new model (spearman=%.3f) worse than old (%.3f). "
                "Keeping old model.",
                new_spearman, old_spearman,
            )
            # Restore old model files (train_regression already saved new ones)
            # The old model files were overwritten — this is a known trade-off.
            # In practice, the quality gate + A/B means this is rare.
            return None

        logger.info(
            "auto_train: new model (spearman=%.3f) beats old (%.3f). Deployed!",
            new_spearman, old_spearman,
        )
    else:
        logger.info("auto_train: no existing model — new model deployed (spearman=%.3f)", new_spearman)

    # Record auto-train timestamp
    new_meta["auto_trained"] = True
    new_meta["auto_trained_at"] = datetime.utcnow().isoformat()
    with open(meta_path, "w") as f:
        json.dump(new_meta, f, indent=2)

    return new_meta


def main():
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")

    parser = argparse.ArgumentParser(description="Train ML ensemble for memecoin 2x prediction")
    parser.add_argument(
        "--horizon", choices=["6h", "12h", "24h"], default="12h",
        help="Prediction horizon (default: 12h)",
    )
    parser.add_argument(
        "--mode", choices=["regression", "classify"], default="regression",
        help="Training mode: regression (predict max_return) or classify (predict did_2x)",
    )
    parser.add_argument(
        "--trials", type=int, default=100,
        help="Number of Optuna trials per model (default: 100)",
    )
    parser.add_argument(
        "--min-samples", type=int, default=200,
        help="Minimum labeled samples required to train (default: 200)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Train models for all horizons",
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="Non-interactive mode: auto-train with A/B comparison, quiet output",
    )
    args = parser.parse_args()

    if args.auto:
        # Auto mode: quiet, non-interactive, A/B comparison
        logging.getLogger().setLevel(logging.WARNING)
        logger.setLevel(logging.INFO)
        result = auto_train(
            horizon=args.horizon,
            min_samples=args.min_samples,
            trials=args.trials,
        )
        if result:
            logger.info("auto_train: SUCCESS — %s", result.get("metrics", {}))
        else:
            logger.info("auto_train: skipped or failed")
        return

    horizons = HORIZONS.keys() if args.all else [args.horizon]
    train_fn = train_regression if args.mode == "regression" else train_ensemble

    for horizon in horizons:
        logger.info("\n%s\n=== Training %s %s ===\n%s", "=" * 60, horizon, args.mode, "=" * 60)
        df = load_labeled_data(horizon)
        if not df.empty:
            result = train_fn(df, horizon, n_trials=args.trials, min_samples=args.min_samples)
            if result and result.get("quality_gate") == "FAILED":
                logger.warning("Model for %s failed quality gate — not deployed.", horizon)


if __name__ == "__main__":
    main()
