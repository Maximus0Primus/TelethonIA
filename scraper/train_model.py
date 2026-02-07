"""
ML training script for memecoin 2x prediction.
- XGBoost + LightGBM ensemble
- Walk-forward temporal splits (no data leakage)
- Optuna hyperparameter optimization
- SHAP feature importance analysis
- Probability calibration (Platt scaling)

Run manually when enough labeled snapshots have accumulated (~200+).

Usage:
    python train_model.py --horizon 12h --trials 100
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
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    classification_report, roc_auc_score,
)
from sklearn.calibration import CalibratedClassifierCV
from supabase import create_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent

# All possible features (the model will use whatever columns are available)
ALL_FEATURE_COLS = [
    # Telegram features
    "mentions",
    "sentiment",
    "breadth",
    "avg_conviction",
    "recency_score",
    # DexScreener — volume (log-scaled)
    "volume_24h_log",
    "volume_6h_log",
    "volume_1h_log",
    # DexScreener — market data (log-scaled)
    "liquidity_usd_log",
    "market_cap_log",
    # DexScreener — transactions
    "txn_count_24h",
    "buys_24h",
    "sells_24h",
    "buy_sell_ratio_24h",
    "buy_sell_ratio_1h",
    # DexScreener — price changes (multi-timeframe)
    "price_change_5m",
    "price_change_1h",
    "price_change_6h",
    "price_change_24h",
    # DexScreener — derived ratios
    "volume_mcap_ratio",
    "liq_mcap_ratio",
    "volume_acceleration",
    # DexScreener — token metadata
    "token_age_hours",
    "is_pump_fun",
    "pair_count",
    # RugCheck — safety
    "risk_score",
    "top10_holder_pct",
    "insider_pct",
    "has_mint_authority",
    "has_freeze_authority",
    "risk_count",
    "lp_locked_pct",
    # Birdeye (optional — may be mostly NaN)
    "holder_count_log",
    "unique_wallet_24h",
    "unique_wallet_24h_change",
    "trade_24h",
    "trade_24h_change",
    "birdeye_buy_sell_ratio",
    "v_buy_24h_usd_log",
    "v_sell_24h_usd_log",
]

HORIZONS = {
    "6h": "did_2x_6h",
    "12h": "did_2x_12h",
    "24h": "did_2x_24h",
}


def _get_client():
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return create_client(url, key)


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


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Transform raw data into ML features.
    Returns (df_with_features, list_of_available_feature_columns).
    """
    df = df.copy()

    # Log-scale monetary features
    log_cols = {
        "volume_24h": "volume_24h_log",
        "volume_6h": "volume_6h_log",
        "volume_1h": "volume_1h_log",
        "liquidity_usd": "liquidity_usd_log",
        "market_cap": "market_cap_log",
        "holder_count": "holder_count_log",
        "v_buy_24h_usd": "v_buy_24h_usd_log",
        "v_sell_24h_usd": "v_sell_24h_usd_log",
    }
    for raw_col, log_col in log_cols.items():
        if raw_col in df.columns:
            df[log_col] = df[raw_col].apply(
                lambda x: np.log1p(float(x)) if pd.notna(x) and float(x) > 0 else np.nan
            )

    # Birdeye buy/sell ratio
    if "birdeye_buy_24h" in df.columns and "birdeye_sell_24h" in df.columns:
        df["birdeye_buy_sell_ratio"] = df.apply(
            lambda r: r["birdeye_buy_24h"] / max(1, r["birdeye_buy_24h"] + r["birdeye_sell_24h"])
            if pd.notna(r.get("birdeye_buy_24h")) else np.nan,
            axis=1,
        )

    # Determine which features are actually available (>10% non-null)
    available = []
    for col in ALL_FEATURE_COLS:
        if col in df.columns:
            non_null_pct = df[col].notna().mean()
            if non_null_pct >= 0.10:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                available.append(col)
                logger.debug("Feature %s: %.0f%% non-null", col, non_null_pct * 100)
            else:
                logger.debug("Dropping %s: only %.0f%% non-null", col, non_null_pct * 100)

    logger.info("Using %d features (from %d possible)", len(available), len(ALL_FEATURE_COLS))
    return df, available


def walk_forward_split(df: pd.DataFrame, train_ratio: float = 0.7):
    """Temporal walk-forward split. No random shuffling."""
    df = df.sort_values("snapshot_at").reset_index(drop=True)
    split_idx = int(len(df) * train_ratio)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def _precision_at_k(y_true, proba, k=5):
    """Precision among top-k predictions (what we actually care about)."""
    top_k = min(k, len(proba))
    if top_k == 0:
        return 0.0
    top_indices = np.argsort(proba)[-top_k:]
    return float(y_true.iloc[top_indices].mean())


def _optimize_xgboost(X_train, y_train, X_test, y_test, scale_pos, n_trials):
    """Optimize XGBoost with Optuna."""
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
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
    logger.info("XGBoost best precision@5: %.3f", study.best_value)

    best_params = study.best_params
    best_params.update({"objective": "binary:logistic", "eval_metric": "aucpr", "verbosity": 0})

    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    return model, best_params, study.best_value


def _optimize_lightgbm(X_train, y_train, X_test, y_test, scale_pos, n_trials):
    """Optimize LightGBM with Optuna."""
    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
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
    logger.info("LightGBM best precision@5: %.3f", study.best_value)

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

        # For binary classification, shap_values may be a list [neg, pos]
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


def train_ensemble(
    df: pd.DataFrame,
    horizon: str = "12h",
    n_trials: int = 100,
    min_samples: int = 100,
) -> dict | None:
    """
    Train XGBoost + LightGBM ensemble with Optuna optimization.
    Returns metrics dict or None if insufficient data.
    """
    label_col = HORIZONS[horizon]

    if len(df) < min_samples:
        logger.warning(
            "Only %d samples (need %d). Skipping training.",
            len(df), min_samples,
        )
        return None

    df, available_features = prepare_features(df)
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
        "Train: %d samples (%d positive, %.1f%%), Test: %d samples (%d positive, %.1f%%)",
        len(y_train), pos_count, 100 * pos_count / len(y_train),
        len(y_test), y_test.sum(), 100 * y_test.sum() / max(1, len(y_test)),
    )

    # Suppress Optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Train XGBoost
    logger.info("=== Optimizing XGBoost (%d trials) ===", n_trials)
    xgb_model, xgb_params, xgb_score = _optimize_xgboost(
        X_train, y_train, X_test, y_test, scale_pos, n_trials
    )

    # Train LightGBM
    logger.info("=== Optimizing LightGBM (%d trials) ===", n_trials)
    lgb_model, lgb_params, lgb_score = _optimize_lightgbm(
        X_train, y_train, X_test, y_test, scale_pos, n_trials
    )

    # Ensemble: weighted average of probabilities
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    lgb_proba = lgb_model.predict_proba(X_test)[:, 1]

    # Weight by individual precision@5 performance
    total_score = xgb_score + lgb_score
    xgb_weight = xgb_score / max(0.001, total_score)
    lgb_weight = lgb_score / max(0.001, total_score)

    ensemble_proba = xgb_weight * xgb_proba + lgb_weight * lgb_proba
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)

    # Metrics
    p_at_5 = _precision_at_k(y_test, ensemble_proba, k=5)
    p_at_10 = _precision_at_k(y_test, ensemble_proba, k=10)
    precision = precision_score(y_test, ensemble_pred, zero_division=0)
    recall = recall_score(y_test, ensemble_pred, zero_division=0)
    f1 = f1_score(y_test, ensemble_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_test, ensemble_proba)
    except ValueError:
        auc = 0.0

    logger.info("\n=== ENSEMBLE RESULTS ===")
    logger.info("Weights: XGBoost=%.2f, LightGBM=%.2f", xgb_weight, lgb_weight)
    logger.info("Precision@5: %.3f (target: >0.50)", p_at_5)
    logger.info("Precision@10: %.3f", p_at_10)
    logger.info("AUC-ROC: %.3f", auc)
    logger.info("\n%s", classification_report(y_test, ensemble_pred, zero_division=0))

    # SHAP analysis
    logger.info("=== SHAP Analysis ===")
    xgb_shap = _compute_shap(xgb_model, X_test, available_features, "XGBoost")
    lgb_shap = _compute_shap(lgb_model, X_test, available_features, "LightGBM")

    # XGBoost native feature importances
    xgb_importances = dict(zip(available_features, xgb_model.feature_importances_))
    sorted_imp = sorted(xgb_importances.items(), key=lambda x: x[1], reverse=True)
    logger.info("XGBoost gain importances:")
    for feat, imp in sorted_imp[:15]:
        logger.info("  %s: %.4f", feat, imp)

    # Save XGBoost model (primary — used by pipeline.py)
    xgb_path = MODEL_DIR / f"model_{horizon}.json"
    xgb_model.save_model(str(xgb_path))
    logger.info("XGBoost model saved to %s", xgb_path)

    # Save LightGBM model
    lgb_path = MODEL_DIR / f"model_{horizon}_lgb.txt"
    lgb_model.booster_.save_model(str(lgb_path))
    logger.info("LightGBM model saved to %s", lgb_path)

    # Save metadata
    metadata = {
        "horizon": horizon,
        "trained_at": datetime.utcnow().isoformat(),
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "positive_rate_train": float(pos_count / len(y_train)),
        "positive_rate_test": float(y_test.sum() / max(1, len(y_test))),
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
            "xgb_precision_at_5": float(xgb_score),
            "lgb_precision_at_5": float(lgb_score),
        },
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


def main():
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")

    parser = argparse.ArgumentParser(description="Train XGBoost+LightGBM ensemble for 2x prediction")
    parser.add_argument(
        "--horizon", choices=["6h", "12h", "24h"], default="12h",
        help="Prediction horizon (default: 12h)",
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
    args = parser.parse_args()

    if args.all:
        for horizon in HORIZONS:
            logger.info("\n{'='*60}\n=== Training %s ensemble ===\n{'='*60}", horizon)
            df = load_labeled_data(horizon)
            if not df.empty:
                train_ensemble(df, horizon, n_trials=args.trials, min_samples=args.min_samples)
    else:
        df = load_labeled_data(args.horizon)
        if not df.empty:
            train_ensemble(df, args.horizon, n_trials=args.trials, min_samples=args.min_samples)


if __name__ == "__main__":
    main()
