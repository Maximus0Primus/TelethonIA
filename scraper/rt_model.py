"""
v66: Unified RT Strategy ML Model — The brain of the real-time trading system.

Learns:  f(kol_features, token_features, strategy) → expected_pnl
Decides: which strategies to open, how much to bet

Architecture:
  VPS (safe_scraper)     →  opens RT trades (all strategies, exploration)
  paper_trader           →  closes trades, records pnl_pct
  GH Actions (outcomes)  →  trains this model on closed trades
  scoring_config         →  stores model centrally (Supabase)
  VPS loads model        →  predicts per-strategy PnL → smart decisions

Phases:
  N < 100 closed RT trades:  No model, pure exploration (all strategies)
  N >= 100:                  Model trained, guides strategy selection + sizing
  Ongoing:                   Retrained every outcomes.yml run, gets smarter

Model: LightGBM regressor, target = pnl_pct per trade row.
       At inference: predict PnL for each of 7 strategies → pick positive ones.
"""

import os
import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Features available at RT trade-open time (stored in paper_trades)
RT_FEATURES = [
    "kol_score",
    "kol_win_rate",
    "kol_tier_encoded",       # S=1, A=0
    "rt_liquidity_usd",
    "rt_volume_24h",
    "rt_buy_sell_ratio",
    "rt_token_age_hours",
    "rt_is_pump_fun",
    "entry_mcap",
    "rt_score",
    "n_confirmations",        # how many other KOLs traded same token within 1h
    "strategy_encoded",       # 0-6 categorical
    "hour_of_day",            # 0-23, market regime proxy
]

# Strategy encoding (deterministic order)
STRATEGY_MAP = {
    "TP30_SL50": 0,
    "TP50_SL30": 1,
    "TP100_SL30": 2,
    "SCALE_OUT": 3,
    "MOONBAG": 4,
    "FRESH_MICRO": 5,
    "QUICK_SCALP": 6,
    "WIDE_RUNNER": 7,
}
STRATEGY_NAMES = {v: k for k, v in STRATEGY_MAP.items()}

MIN_TRAINING_SAMPLES = 100


def _get_client():
    from supabase import create_client
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


def _fetch_closed_rt_trades(client) -> list[dict]:
    """Fetch all closed RT trades with features needed for training."""
    try:
        result = (
            client.table("paper_trades")
            .select(
                "id, kol_score, kol_win_rate, kol_tier, kol_group, "
                "rt_score, rt_liquidity_usd, rt_volume_24h, rt_buy_sell_ratio, "
                "rt_token_age_hours, rt_is_pump_fun, entry_mcap, "
                "pnl_pct, pnl_usd, position_usd, strategy, "
                "token_address, created_at, status"
            )
            .eq("source", "rt")
            .neq("status", "open")
            .execute()
        )
        return result.data or []
    except Exception as e:
        logger.error("rt_model: fetch failed: %s", e)
        return []


def _compute_confirmations(trades: list[dict]) -> dict[int, int]:
    """
    For each trade, count how many OTHER KOLs traded the same token within 1 hour.
    Returns {trade_id: n_confirmations}.
    """
    from datetime import datetime as dt, timedelta

    # Group trades by token_address
    by_token: dict[str, list[dict]] = {}
    for t in trades:
        addr = t.get("token_address", "")
        if addr:
            by_token.setdefault(addr, []).append(t)

    confirmations = {}
    for addr, token_trades in by_token.items():
        # Parse timestamps
        for t in token_trades:
            try:
                t["_created_dt"] = dt.fromisoformat(
                    t["created_at"].replace("Z", "+00:00")
                )
            except Exception:
                t["_created_dt"] = None

        # For each trade, count distinct KOLs within 1h window
        for t in token_trades:
            if not t.get("_created_dt"):
                confirmations[t["id"]] = 0
                continue
            other_kols = set()
            for other in token_trades:
                if other["id"] == t["id"]:
                    continue
                if not other.get("_created_dt"):
                    continue
                if other.get("kol_group") == t.get("kol_group"):
                    continue  # same KOL doesn't count
                delta = abs((other["_created_dt"] - t["_created_dt"]).total_seconds())
                if delta <= 3600:
                    other_kols.add(other.get("kol_group", ""))
            confirmations[t["id"]] = len(other_kols)

    return confirmations


def _build_feature_matrix(trades: list[dict], confirmations: dict[int, int]):
    """Build feature matrix + target vector from closed RT trades."""
    import numpy as np

    X_rows = []
    y = []
    valid_trades = []

    for t in trades:
        strat = t.get("strategy", "")
        if strat not in STRATEGY_MAP:
            continue

        pnl = t.get("pnl_pct")
        if pnl is None:
            continue

        # Parse hour of day
        hour = 12  # default
        try:
            created = datetime.fromisoformat(t["created_at"].replace("Z", "+00:00"))
            hour = created.hour
        except Exception:
            pass

        row = [
            float(t.get("kol_score") or 0),
            float(t.get("kol_win_rate") or 0),
            1 if t.get("kol_tier") == "S" else 0,
            float(t.get("rt_liquidity_usd") or 0),
            float(t.get("rt_volume_24h") or 0),
            float(t.get("rt_buy_sell_ratio") or 0.5),
            float(t.get("rt_token_age_hours") or 0),
            int(t.get("rt_is_pump_fun") or 0),
            float(t.get("entry_mcap") or 0),
            float(t.get("rt_score") or 0),
            confirmations.get(t["id"], 0),
            STRATEGY_MAP[strat],
            hour,
        ]

        X_rows.append(row)
        y.append(float(pnl))
        valid_trades.append(t)

    if not X_rows:
        return None, None, None

    return np.array(X_rows), np.array(y), valid_trades


def train_rt_model(client=None, min_samples: int = MIN_TRAINING_SAMPLES) -> dict | None:
    """
    Train LightGBM on closed RT trades. Stores model in scoring_config.

    Returns dict with metrics or None if not enough data.
    """
    try:
        import lightgbm as lgb
        import numpy as np
    except ImportError:
        logger.error("rt_model: lightgbm/numpy not installed")
        return None

    if client is None:
        client = _get_client()
    if not client:
        return None

    trades = _fetch_closed_rt_trades(client)
    if len(trades) < min_samples:
        logger.info("rt_model: %d closed RT trades (need %d), skipping training",
                     len(trades), min_samples)
        return None

    logger.info("rt_model: training on %d closed RT trades", len(trades))

    # Compute confirmation features
    confirmations = _compute_confirmations(trades)

    # Build feature matrix
    X, y, valid_trades = _build_feature_matrix(trades, confirmations)
    if X is None or len(X) < min_samples:
        logger.info("rt_model: insufficient valid samples after filtering")
        return None

    # Walk-forward split: train on first 70%, test on last 30% (temporal)
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    if len(X_test) < 20:
        logger.info("rt_model: test set too small (%d), skipping", len(X_test))
        return None

    # Train LightGBM regressor
    train_data = lgb.Dataset(
        X_train, y_train,
        feature_name=RT_FEATURES,
        categorical_feature=["strategy_encoded"],
    )
    valid_data = lgb.Dataset(X_test, y_test, reference=train_data)

    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 10,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }

    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    model = lgb.train(
        params, train_data,
        num_boost_round=500,
        valid_sets=[valid_data],
        callbacks=callbacks,
    )

    # Evaluate
    y_pred_test = model.predict(X_test)
    mae_test = float(np.mean(np.abs(y_test - y_pred_test)))
    # Directional accuracy: did we predict the right sign?
    direction_correct = float(np.mean((y_pred_test > 0) == (y_test > 0)))

    # Per-strategy analysis on test set
    strat_metrics = {}
    for strat_name, strat_code in STRATEGY_MAP.items():
        mask = X_test[:, RT_FEATURES.index("strategy_encoded")] == strat_code
        if mask.sum() < 5:
            continue
        s_pred = y_pred_test[mask]
        s_actual = y_test[mask]
        # If model says "take this trade" (pred > 0), what's the actual avg PnL?
        take_mask = s_pred > 0
        skip_mask = s_pred <= 0
        take_pnl = float(np.mean(s_actual[take_mask])) if take_mask.sum() > 0 else 0
        skip_pnl = float(np.mean(s_actual[skip_mask])) if skip_mask.sum() > 0 else 0
        strat_metrics[strat_name] = {
            "n_test": int(mask.sum()),
            "n_take": int(take_mask.sum()),
            "take_avg_pnl": round(take_pnl, 4),
            "skip_avg_pnl": round(skip_pnl, 4),
            "edge": round(take_pnl - skip_pnl, 4),  # positive = model adds value
        }

    # Feature importance
    importance = dict(zip(RT_FEATURES, [int(x) for x in model.feature_importance()]))

    # Overall: if we only took trades where model predicts PnL > 0
    take_mask_all = y_pred_test > 0
    if take_mask_all.sum() > 0:
        selective_pnl = float(np.mean(y_test[take_mask_all]))
        baseline_pnl = float(np.mean(y_test))
        model_edge = selective_pnl - baseline_pnl
    else:
        selective_pnl = 0
        baseline_pnl = float(np.mean(y_test))
        model_edge = 0

    metrics = {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "mae_test": round(mae_test, 4),
        "direction_accuracy": round(direction_correct, 3),
        "baseline_avg_pnl": round(baseline_pnl, 4),
        "selective_avg_pnl": round(selective_pnl, 4),
        "model_edge": round(model_edge, 4),
        "strategy_metrics": strat_metrics,
        "feature_importance": importance,
    }

    logger.info(
        "rt_model: TRAINED — %d train, %d test | MAE=%.4f, dir_acc=%.1f%% | "
        "baseline=%.2f%%, selective=%.2f%%, edge=%.2f%%",
        len(X_train), len(X_test), mae_test, direction_correct * 100,
        baseline_pnl * 100, selective_pnl * 100, model_edge * 100,
    )
    for s, m in strat_metrics.items():
        logger.info("  %s: take=%d(avg=%.2f%%) skip=%d(avg=%.2f%%) edge=%.2f%%",
                     s, m["n_take"], m["take_avg_pnl"] * 100,
                     m["n_test"] - m["n_take"], m["skip_avg_pnl"] * 100,
                     m["edge"] * 100)

    # Only deploy if model actually adds value (edge > 0)
    if model_edge <= 0:
        logger.info("rt_model: edge=%.4f <= 0, NOT deploying (model doesn't help yet)", model_edge)
        meta = {
            "status": "trained_not_deployed",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            **metrics,
        }
        try:
            client.table("scoring_config").update({
                "rt_ml_meta": json.dumps(meta),
            }).eq("id", 1).execute()
        except Exception:
            pass
        return metrics

    # Store model in Supabase
    model_text = model.model_to_string()
    meta = {
        "status": "deployed",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "version": "v66",
        **metrics,
    }

    try:
        client.table("scoring_config").update({
            "rt_ml_model": model_text,
            "rt_ml_meta": json.dumps(meta),
            "updated_by": "rt_model_v66",
        }).eq("id", 1).execute()
        logger.info("rt_model: DEPLOYED to scoring_config (edge=+%.2f%%)", model_edge * 100)
    except Exception as e:
        logger.error("rt_model: failed to store model: %s", e)

    return metrics


# ---- Inference (called from VPS safe_scraper.py) ----

_cached_model = None
_cached_model_loaded_at: float = 0.0
_MODEL_CACHE_TTL = 300  # 5 minutes


def load_rt_model(client=None):
    """Load model from Supabase. Returns LightGBM Booster or None."""
    global _cached_model, _cached_model_loaded_at
    import time
    now = time.time()
    if _cached_model is not None and now - _cached_model_loaded_at < _MODEL_CACHE_TTL:
        return _cached_model

    try:
        import lightgbm as lgb
    except ImportError:
        return None

    if client is None:
        client = _get_client()
    if not client:
        return None

    try:
        result = (
            client.table("scoring_config")
            .select("rt_ml_model, rt_ml_meta")
            .eq("id", 1)
            .execute()
        )
        if not result.data:
            return None
        row = result.data[0]
        model_text = row.get("rt_ml_model")
        meta = row.get("rt_ml_meta")

        if not model_text:
            return None

        # Check if model is deployed (not just trained)
        if meta:
            if isinstance(meta, str):
                meta = json.loads(meta)
            if meta.get("status") != "deployed":
                logger.debug("rt_model: model exists but status=%s, not using", meta.get("status"))
                return None

        model = lgb.Booster(model_str=model_text)
        _cached_model = model
        _cached_model_loaded_at = now
        logger.info("rt_model: loaded from Supabase (trained=%s, edge=%.2f%%)",
                     meta.get("trained_at", "?")[:16] if meta else "?",
                     meta.get("model_edge", 0) * 100 if meta else 0)
        return model
    except Exception as e:
        logger.debug("rt_model: failed to load: %s", e)
        return None


def predict_strategy_pnl(
    model,
    kol_info: dict,
    token_info: dict,
    tier: str,
    rt_score: float,
    n_confirmations: int,
    hour: int = 12,
) -> dict[str, float]:
    """
    Predict expected PnL for each of the 7 strategies.
    Returns {strategy_name: predicted_pnl_pct}.
    """
    import numpy as np

    # Base feature vector (without strategy)
    base = [
        float(kol_info.get("score", 0)),
        float(kol_info.get("win_rate", 0)),
        1 if tier == "S" else 0,
        float(token_info.get("liquidity_usd", 0)),
        float(token_info.get("volume_24h", 0)),
        float(token_info.get("buy_sell_ratio", 0.5)),
        float(token_info.get("token_age_hours", 0)),
        int(token_info.get("is_pump_fun", 0)),
        float(token_info.get("mcap", 0)),
        float(rt_score),
        int(n_confirmations),
        0,  # placeholder for strategy_encoded
        int(hour),
    ]

    predictions = {}
    for strat_name, strat_code in STRATEGY_MAP.items():
        row = base.copy()
        row[11] = strat_code  # strategy_encoded position
        pred = model.predict([row])[0]
        predictions[strat_name] = float(pred)

    return predictions


def select_strategies(
    predictions: dict[str, float],
    config: dict,
) -> list[tuple[str, float]]:
    """
    Select which strategies to trade and compute position multiplier for each.
    Returns [(strategy_name, size_multiplier), ...] sorted by predicted PnL.

    - Strategies with predicted PnL > 0 → trade, size proportional to PnL
    - Strategies with predicted PnL <= 0 → skip (or micro-explore)
    - Optuna strategy_multipliers applied on top if available
    """
    strat_mults = config.get("strategy_multipliers", {})
    min_explore_pct = 0.1  # 10% of position for "bad" strategies (keep exploring)

    selected = []
    for strat, pred_pnl in sorted(predictions.items(), key=lambda x: -x[1]):
        optuna_mult = float(strat_mults.get(strat, 1.0))

        if pred_pnl > 0:
            # Positive prediction → trade with confidence-weighted size
            # Normalize: 5% predicted PnL → 1.0x, 20%+ → 2.0x cap
            size_mult = min(2.0, max(0.3, pred_pnl / 0.05))
            size_mult *= optuna_mult
            selected.append((strat, round(max(0.1, size_mult), 2)))
        else:
            # Negative prediction → micro-explore (keep learning)
            explore_mult = min_explore_pct * optuna_mult
            if explore_mult > 0.05:  # don't bother with dust
                selected.append((strat, round(explore_mult, 2)))

    return selected
