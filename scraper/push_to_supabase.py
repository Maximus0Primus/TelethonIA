"""
Supabase upsert module for the scraper pipeline.
Uses service_role key to bypass RLS.
"""

import os
import json
import math
import logging
from datetime import datetime, timezone

from supabase import create_client, Client

from pipeline import SCORING_PARAMS

logger = logging.getLogger(__name__)


def _sanitize_value(v):
    """
    Convert a value to a JSON-safe Python type.
    Handles numpy types, NaN, Infinity, sets, and other non-serializable objects.
    """
    if v is None:
        return None

    # Handle numpy types (np.float64, np.int64, np.bool_, etc.)
    type_name = type(v).__module__
    if type_name == "numpy":
        # numpy scalar → Python native
        if hasattr(v, "item"):
            v = v.item()

    # Handle float NaN/Infinity → None
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v

    # Handle bool (must check before int since bool is subclass of int)
    if isinstance(v, bool):
        return v

    # Handle int, str — pass through
    if isinstance(v, (int, str)):
        return v

    # Handle sets → lists
    if isinstance(v, set):
        return list(v)

    # Handle lists/dicts — pass through (json.dumps handles nested structures)
    if isinstance(v, (list, dict)):
        return v

    # Fallback: convert to string
    return str(v)


# Max values for bounded numeric(p,s) columns to prevent DB overflow
NUMERIC_LIMITS = {
    "volatility_proxy": 999.999,
    "short_term_heat": 99.999,
    "ultra_short_heat": 99.999,
    "txn_velocity": 99.999,
    "volume_acceleration": 99999.999,
    "sentiment": 1.0,
    "breadth": 1.0,
    "recency_score": 9.999,
    "wash_trading_score": 9.999,
    "buy_sell_ratio_1h": 99.999,
    "buy_sell_ratio_24h": 99.999,
    "buy_sell_ratio_5m": 9.999,
    "already_pumped_penalty": 9.999,
    "sentiment_consistency": 9.999,
    "rsi_14": 100.0,
    "macd_histogram": 99999.999,
    "bb_width": 99.999,
    "bb_pct_b": 9.999,
    "obv_slope_norm": 999.999,
    "data_confidence": 1.0,
    "weakest_component_value": 1.0,
    "squeeze_score": 1.0,
    "trend_strength": 1.0,
    "death_penalty": 1.0,
    "freshest_mention_hours": 999.99,
    "oldest_mention_hours": 999999.99,
    "sol_price_at_snapshot": 99999999.99,
    "entry_premium": 9999.999,
    "entry_premium_mult": 1.1,
    "new_kol_ratio": 1.0,
    "pump_bonus": 1.1,
    "wash_pen": 1.0,
    "pvp_pen": 1.0,
    "pump_pen": 1.0,
    "breadth_pen": 1.0,
    "size_mult": 1.5,
    "s_tier_mult": 1.5,
    # ML v2: Temporal velocity features
    "score_velocity": 9999.999,
    "score_acceleration": 9999.999,
    "mention_velocity": 9999.999,
    "volume_velocity": 99.999,
    "kol_arrival_rate": 99.999,
    "entry_timing_quality": 1.0,
    "gate_mult": 1.0,
    "hype_pen": 1.0,
    "kol_freshness": 1.0,
    "mention_heat_ratio": 999.999,
    "momentum_mult": 1.5,
    "entry_drift_mult": 1.0,
    "price_velocity": 9999.999,
    "price_drift_from_first_seen": 9999999.999,
    # v25: Message-level text features
    "call_type_score": 999.999,
    "avg_msg_length": 9999999.9,
    "ca_mention_ratio": 1.0,
    "caps_ratio": 1.0,
    "emoji_density": 99.9999,
    "multi_token_ratio": 1.0,
    "question_ratio": 1.0,
    "link_ratio": 1.0,
    # v26: Market context features
    "median_peak_return": 9999.9999,
    "entry_vs_median_peak": 99.9999,
    "win_rate_7d": 1.0,
    "market_heat_24h": 99999.99,
    "relative_volume": 9999.9999,
}

# v24: Ordinal encoding for categorical phase features → ML numeric
# lifecycle: panic(worst) → boom(best entry)
_LIFECYCLE_PHASE_MAP = {
    "panic": 0,
    "profit_taking": 1,
    "euphoria": 2,
    "unknown": 3,
    "displacement": 4,
    "boom": 5,
}
# social momentum: declining(worst) → building(best)
_MOMENTUM_PHASE_MAP = {
    "declining": 0,
    "plateau": 1,
    "building": 2,
}


# Integer columns in token_snapshots — must receive int, not float
_INT_COLS = {
    "birdeye_buy_24h", "birdeye_sell_24h", "boosts_active", "score_at_snapshot", "score_delta",
    "bubblemaps_cluster_count", "bundle_count", "buys_24h",
    "ca_mention_count", "confirmation_pillars", "hedging_count", "helius_holder_count",
    "helius_recent_tx_count", "helius_unique_buyers", "holder_count",
    "jito_bundle_detected", "jito_bundle_slots", "jito_max_slot_txns",
    "mentions", "mentions_delta", "pair_count", "price_target_count",
    "pvp_recent_count", "pvp_same_name_count", "risk_count", "risk_score",
    "sells_24h", "ticker_mention_count", "trade_24h", "txn_count_24h",
    "unique_wallet_24h", "url_mention_count",
    "lifecycle_phase_num", "social_momentum_num",
    "whale_count", "whale_new_entries",
    "kol_saturation",
}


def _sanitize_row(row: dict) -> dict:
    """Sanitize all values in a row dict for JSON serialization + numeric clamping."""
    sanitized = {k: _sanitize_value(v) for k, v in row.items()}
    # Clamp bounded numeric fields to prevent DB overflow
    for field, limit in NUMERIC_LIMITS.items():
        val = sanitized.get(field)
        if val is not None and isinstance(val, (int, float)):
            if val > limit:
                sanitized[field] = limit
            elif val < -limit:
                sanitized[field] = -limit
    # Cast float→int for integer DB columns (PostgREST rejects 0.0 for int4)
    for field in _INT_COLS:
        val = sanitized.get(field)
        if val is not None and isinstance(val, float):
            sanitized[field] = int(val)
    return sanitized


def _get_client() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return create_client(url, key)


def _purge_absent_tokens(client: Client, time_window: str, current_symbols: set[str]) -> int:
    """
    v11: Delete tokens not present in the current scrape cycle.
    Prevents zombie tokens from accumulating forever via UPSERT.
    """
    try:
        existing = client.table("tokens").select("symbol").eq("time_window", time_window).execute()
        existing_symbols = {r["symbol"] for r in (existing.data or [])}
    except Exception as e:
        logger.warning("Failed to fetch existing tokens for purge: %s", e)
        return 0

    to_delete = existing_symbols - current_symbols
    if not to_delete:
        return 0

    deleted = 0
    for sym in to_delete:
        try:
            client.table("tokens").delete().eq("symbol", sym).eq("time_window", time_window).execute()
            deleted += 1
        except Exception as e:
            logger.warning("Failed to purge token %s/%s: %s", sym, time_window, e)

    logger.info("v11 purge: removed %d stale tokens from window %s", deleted, time_window)
    return deleted


def upsert_tokens(
    ranking_by_window: dict[str, list[dict]],
    stats: dict,
) -> None:
    """
    Upsert token rankings for each time window into the `tokens` table.
    Also updates the `scrape_metadata` singleton.

    Parameters
    ----------
    ranking_by_window : { "24h": [ { symbol, score, mentions, ... }, ... ], ... }
    stats : { totalTokens, totalMentions, avgSentiment, totalKols }
    """
    client = _get_client()

    for time_window, tokens in ranking_by_window.items():
        if not tokens:
            continue

        # v11: Purge tokens not in current batch (removes zombies)
        current_symbols = {t["symbol"] for t in tokens}
        _purge_absent_tokens(client, time_window, current_symbols)

        rows = [
            _sanitize_row({
                "symbol": t["symbol"],
                "score": t["score"],
                "score_conviction": t.get("score_conviction", t["score"]),
                "score_momentum": t.get("score_momentum", t["score"]),
                "mentions": t["mentions"],
                "unique_kols": t["unique_kols"],
                "sentiment": t["sentiment"],
                "trend": t["trend"],
                "time_window": time_window,
                "change_24h": t.get("change_24h", 0.0),
                # Pipeline-computed features
                "conviction_weighted": round(t.get("avg_conviction", 0), 2),
                "momentum": round(t.get("recency_score", 0), 3),
                "breadth": round(t.get("breadth_score", 0), 3),
                # v4: base scores for micro-refresh recalculation
                "base_score": t["score"],
                "base_score_conviction": t.get("score_conviction", t["score"]),
                "base_score_momentum": t.get("score_momentum", t["score"]),
                # v7: scoring improvements
                "weakest_component": t.get("weakest_component"),
                "score_interpretation": t.get("score_interpretation"),
                "data_confidence": t.get("data_confidence"),
                # v11: freshest_mention_hours for price_refresh social decay
                "freshest_mention_hours": t.get("freshest_mention_hours"),
                # v21: token_address for frontend DexScreener links + price_refresh simplification
                "token_address": t.get("token_address"),
                # v27: market_cap for frontend display
                "market_cap": t.get("market_cap"),
                # v40: Track CA provenance (kol=from KOL message, dexscreener=from search)
                "ca_source": "kol" if t.get("kol_resolved_ca") else "dexscreener",
            })
            for t in tokens
        ]

        # Upsert on (symbol, time_window) unique constraint
        result = (
            client.table("tokens")
            .upsert(rows, on_conflict="symbol,time_window")
            .execute()
        )
        logger.info(
            "Upserted %d tokens for window %s", len(rows), time_window
        )

    # Update scrape_metadata singleton
    client.table("scrape_metadata").upsert(
        {
            "id": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "stats": stats,
        },
        on_conflict="id",
    ).execute()

    logger.info("Updated scrape_metadata with stats: %s", stats)


def _fetch_previous_snapshots(client: Client, symbols: list[str]) -> dict[str, dict]:
    """
    For each symbol, fetch the most recent snapshot to compute temporal deltas.
    Returns { "symbol": { mentions, sentiment, volume_24h, holder_count, score_at_snapshot, top_kols, ... } }
    """
    if not symbols:
        return {}

    previous: dict[str, dict] = {}

    # Batch query: get the latest snapshot per symbol using a single query
    # Supabase doesn't support DISTINCT ON, so we fetch recent snapshots and deduplicate
    try:
        result = (
            client.table("token_snapshots")
            .select("symbol, mentions, sentiment, volume_24h, holder_count, score_at_snapshot, top_kols, snapshot_at, score_velocity, unique_kols, first_seen_price, price_at_snapshot")
            .in_("symbol", symbols)
            .order("snapshot_at", desc=True)
            .limit(len(symbols) * 3)  # enough to get at least 1 per symbol
            .execute()
        )

        for row in (result.data or []):
            sym = row.get("symbol")
            if sym and sym not in previous:
                previous[sym] = row

    except Exception as e:
        logger.warning("Failed to fetch previous snapshots: %s", e)

    return previous


def _safe_delta(current, previous) -> float | None:
    """Compute delta between two values, handling None."""
    if current is None or previous is None:
        return None
    try:
        return float(current) - float(previous)
    except (ValueError, TypeError):
        return None


def _compute_temporal_features(current: dict, previous: dict | None) -> dict:
    """
    Compute deltas and velocity features between current and previous snapshot.
    ML v2 Phase B: adds score_velocity, score_acceleration, mention_velocity,
    volume_velocity, kol_arrival_rate.
    """
    result = {
        "mentions_delta": None,
        "sentiment_delta": None,
        "volume_delta": None,
        "holder_delta": None,
        "score_at_snapshot": current.get("score"),
        "score_delta": None,
        "new_kol_ratio": None,
        # ML v2: Velocity features
        "score_velocity": None,
        "score_acceleration": None,
        "mention_velocity": None,
        "volume_velocity": None,
        "kol_arrival_rate": None,
        "price_velocity": None,
    }

    if previous is None:
        return result

    result["mentions_delta"] = _safe_delta(current.get("mentions"), previous.get("mentions"))
    sent_delta = _safe_delta(current.get("sentiment"), previous.get("sentiment"))
    result["sentiment_delta"] = round(sent_delta, 3) if sent_delta is not None else None
    result["volume_delta"] = _safe_delta(current.get("volume_24h"), previous.get("volume_24h"))
    result["holder_delta"] = _safe_delta(current.get("holder_count"), previous.get("holder_count"))

    # score_delta: current score minus previous score
    prev_score = previous.get("score_at_snapshot")
    curr_score = current.get("score")
    if prev_score is not None and curr_score is not None:
        result["score_delta"] = int(curr_score) - int(prev_score)

    # --- ML v2: Compute hours_between from snapshot_at timestamps ---
    # v27: Use current snapshot_at (not wall clock) so velocities are correct
    # for historical data and backtesting, not just live scraping.
    hours_between = None
    prev_snapshot_at = previous.get("snapshot_at")
    curr_snapshot_at = current.get("snapshot_at")
    if prev_snapshot_at:
        try:
            from datetime import datetime as _dt
            if isinstance(prev_snapshot_at, str):
                prev_ts = _dt.fromisoformat(prev_snapshot_at.replace("Z", "+00:00"))
            else:
                prev_ts = prev_snapshot_at
            # Use current snapshot_at if available, else fall back to now()
            if curr_snapshot_at:
                if isinstance(curr_snapshot_at, str):
                    curr_ts = _dt.fromisoformat(curr_snapshot_at.replace("Z", "+00:00"))
                else:
                    curr_ts = curr_snapshot_at
            else:
                curr_ts = _dt.now(prev_ts.tzinfo) if prev_ts.tzinfo else _dt.utcnow()
            hours_between = max(0.1, (curr_ts - prev_ts).total_seconds() / 3600)
        except Exception:
            hours_between = None

    # --- ML v2: Score velocity = score_delta / hours_between ---
    if result["score_delta"] is not None and hours_between is not None:
        result["score_velocity"] = round(result["score_delta"] / hours_between, 3)

    # --- ML v2: Score acceleration = current_velocity - previous_velocity ---
    prev_velocity = previous.get("score_velocity")
    if result["score_velocity"] is not None and prev_velocity is not None:
        try:
            result["score_acceleration"] = round(result["score_velocity"] - float(prev_velocity), 3)
        except (ValueError, TypeError):
            pass

    # --- ML v2: Mention velocity = mentions_delta / hours_between ---
    if result["mentions_delta"] is not None and hours_between is not None:
        result["mention_velocity"] = round(result["mentions_delta"] / hours_between, 3)

    # --- ML v2: Volume velocity = (log(vol_now) - log(vol_prev)) / hours_between ---
    curr_vol = current.get("volume_24h")
    prev_vol = previous.get("volume_24h")
    if curr_vol is not None and prev_vol is not None and hours_between is not None:
        try:
            cv = float(curr_vol)
            pv = float(prev_vol)
            if cv > 0 and pv > 0:
                result["volume_velocity"] = round((math.log(cv) - math.log(pv)) / hours_between, 3)
        except (ValueError, TypeError):
            pass

    # --- v24: Price velocity = log-ratio of price between cycles ---
    curr_price = current.get("price_usd")
    prev_price = previous.get("price_at_snapshot")
    if curr_price is not None and prev_price is not None and hours_between is not None:
        try:
            cp = float(curr_price)
            pp = float(prev_price)
            if cp > 0 and pp > 0:
                result["price_velocity"] = round((math.log(cp) - math.log(pp)) / hours_between, 3)
        except (ValueError, TypeError):
            pass

    # new_kol_ratio: what % of current KOLs are NEW (not in previous snapshot)
    current_kols = current.get("top_kols") or []
    prev_kols_raw = previous.get("top_kols")
    new_kol_count = 0
    if current_kols:
        if prev_kols_raw:
            # top_kols is stored as JSON string in DB, parse if needed
            if isinstance(prev_kols_raw, str):
                try:
                    prev_kols = json.loads(prev_kols_raw)
                except (json.JSONDecodeError, TypeError):
                    prev_kols = []
            else:
                prev_kols = prev_kols_raw
            prev_set = set(prev_kols) if prev_kols else set()
            new_kol_count = sum(1 for k in current_kols if k not in prev_set)
            result["new_kol_ratio"] = round(new_kol_count / len(current_kols), 3)
        else:
            new_kol_count = len(current_kols)
            result["new_kol_ratio"] = 1.0  # All KOLs are "new" if no previous data

    # --- ML v2: KOL arrival rate = new_kol_ratio / hours_between ---
    if result["new_kol_ratio"] is not None and hours_between is not None:
        result["kol_arrival_rate"] = round(new_kol_count / hours_between, 3)

    return result


def insert_snapshots(ranking: list[dict]) -> None:
    """
    Insert a snapshot row for each token in the ranking into token_snapshots.
    Used to collect ML training data (features at time of observation).
    Outcome labels are filled later by outcome_tracker.
    """
    if not ranking:
        return

    client = _get_client()

    # Fetch previous snapshots to compute temporal deltas
    symbols = [t["symbol"] for t in ranking if t.get("symbol")]
    prev_snapshots = _fetch_previous_snapshots(client, symbols)

    rows = []
    skipped_stale = 0
    for t in ranking:
        # v19: Skip zombie tokens — stale mentions produce noise snapshots
        # v20: threshold from SCORING_PARAMS (dynamic)
        freshest_h = t.get("freshest_mention_hours") or 0
        if freshest_h > SCORING_PARAMS["stale_hours_severe"]:
            skipped_stale += 1
            continue

        # Compute temporal deltas vs previous snapshot
        prev = prev_snapshots.get(t["symbol"])
        deltas = _compute_temporal_features(t, prev)

        row = {
            "symbol": t["symbol"],
            # Telegram features
            "mentions": t.get("mentions"),
            "sentiment": t.get("sentiment"),
            "breadth": round(t.get("unique_kols", 0) / max(1, t.get("_total_kols", 50)), 3)
                if t.get("unique_kols") is not None else None,
            "avg_conviction": t.get("avg_conviction"),
            "recency_score": t.get("recency_score"),
            # DexScreener — volume & liquidity
            "volume_24h": t.get("volume_24h"),
            "volume_6h": t.get("volume_6h"),
            "volume_1h": t.get("volume_1h"),
            "liquidity_usd": t.get("liquidity_usd"),
            "market_cap": t.get("market_cap"),
            # DexScreener — transactions
            "txn_count_24h": t.get("txn_count_24h"),
            "buys_24h": t.get("buys_24h"),
            "sells_24h": t.get("sells_24h"),
            "buy_sell_ratio_24h": t.get("buy_sell_ratio_24h"),
            "buy_sell_ratio_1h": t.get("buy_sell_ratio_1h"),
            # DexScreener — price changes
            "price_change_5m": t.get("price_change_5m"),
            "price_change_1h": t.get("price_change_1h"),
            "price_change_6h": t.get("price_change_6h"),
            "price_change_24h": t.get("price_change_24h"),
            # DexScreener — derived ratios
            "volume_mcap_ratio": t.get("volume_mcap_ratio"),
            "liq_mcap_ratio": t.get("liq_mcap_ratio"),
            "volume_acceleration": t.get("volume_acceleration"),
            # DexScreener — token metadata
            "token_age_hours": t.get("token_age_hours"),
            "is_pump_fun": t.get("is_pump_fun"),
            "pair_count": t.get("pair_count"),
            "dex_id": t.get("dex_id"),
            # RugCheck — safety
            "risk_score": t.get("risk_score"),
            "top10_holder_pct": t.get("top10_holder_pct"),
            "insider_pct": t.get("insider_pct"),
            "has_mint_authority": t.get("has_mint_authority"),
            "has_freeze_authority": t.get("has_freeze_authority"),
            "risk_count": t.get("risk_count"),
            "lp_locked_pct": t.get("lp_locked_pct"),
            # Birdeye (optional — may be None)
            "holder_count": t.get("holder_count"),
            "unique_wallet_24h": t.get("unique_wallet_24h"),
            "unique_wallet_24h_change": t.get("unique_wallet_24h_change"),
            "trade_24h": t.get("trade_24h"),
            "trade_24h_change": t.get("trade_24h_change"),
            "birdeye_buy_24h": t.get("buy_24h"),
            "birdeye_sell_24h": t.get("sell_24h"),
            "v_buy_24h_usd": t.get("v_buy_24h_usd"),
            "v_sell_24h_usd": t.get("v_sell_24h_usd"),
            # Price + address
            "price_at_snapshot": t.get("price_usd"),
            "token_address": t.get("token_address"),
            "pair_address": t.get("pair_address"),
            # Phase 1 scoring features
            "social_velocity": t.get("social_velocity"),
            "mention_acceleration": t.get("mention_acceleration"),
            "onchain_multiplier": t.get("onchain_multiplier"),
            "safety_penalty": t.get("safety_penalty"),
            # Phase 2: KOL reputation
            "top_kols": t.get("top_kols") if t.get("top_kols") else None,  # pass list directly, supabase-py handles jsonb
            "kol_reputation_avg": t.get("kol_reputation_avg"),
            # Phase 2: Narrative
            "narrative": t.get("narrative"),
            "narrative_is_hot": t.get("narrative_is_hot"),
            # Phase 2: Pump.fun graduation
            "pump_graduation_status": t.get("pump_graduation_status"),
            # Phase 2: Temporal deltas
            "mentions_delta": deltas.get("mentions_delta"),
            "sentiment_delta": deltas.get("sentiment_delta"),
            "volume_delta": deltas.get("volume_delta"),
            "holder_delta": deltas.get("holder_delta"),
            # Phase 3: Helius on-chain intelligence
            "helius_holder_count": t.get("helius_holder_count"),
            "helius_top5_pct": t.get("helius_top5_pct"),
            "helius_top20_pct": t.get("helius_top20_pct"),
            "helius_gini": t.get("helius_gini"),
            "bundle_detected": t.get("bundle_detected"),
            "bundle_count": t.get("bundle_count"),
            "bundle_pct": t.get("bundle_pct"),
            "helius_recent_tx_count": t.get("helius_recent_tx_count"),
            "helius_unique_buyers": t.get("helius_unique_buyers"),
            "helius_onchain_bsr": t.get("helius_onchain_bsr"),
            # Phase 3B: Jupiter swap data
            "jup_tradeable": t.get("jup_tradeable"),
            "jup_price_impact_1k": t.get("jup_price_impact_1k"),
            "jup_route_count": t.get("jup_route_count"),
            "jup_price_usd": t.get("jup_price_usd"),
            # Phase 3B: Whale tracking
            "whale_count": t.get("whale_count"),
            "whale_total_pct": t.get("whale_total_pct"),
            "whale_change": t.get("whale_change"),
            "whale_new_entries": t.get("whale_new_entries"),
            # Phase 3B: Narrative confidence
            "narrative_confidence": t.get("narrative_confidence"),
            # Algorithm v2: Wash trading detection
            "wash_trading_score": t.get("wash_trading_score"),
            # Algorithm v2: Jito bundle detection
            "jito_bundle_detected": t.get("jito_bundle_detected"),
            "jito_bundle_slots": t.get("jito_bundle_slots"),
            "jito_max_slot_txns": t.get("jito_max_slot_txns"),
            # Algorithm v2: PVP detection
            "pvp_same_name_count": t.get("pvp_same_name_count"),
            "pvp_recent_count": t.get("pvp_recent_count"),
            # Algorithm v2: Per-message conviction NLP
            "msg_conviction_avg": t.get("msg_conviction_avg"),
            "price_target_count": t.get("price_target_count"),
            "hedging_count": t.get("hedging_count"),
            # Algorithm v3 Sprint A: New features
            "short_term_heat": t.get("short_term_heat"),
            "txn_velocity": t.get("txn_velocity"),
            "sentiment_consistency": t.get("sentiment_consistency"),
            # Algorithm v3 Sprint C: ME2F-inspired ML features
            "volatility_proxy": t.get("volatility_proxy"),
            "whale_dominance": t.get("whale_dominance"),
            "sentiment_amplification": t.get("sentiment_amplification"),
            # Algorithm v3.1: 5m granularity + Bubblemaps
            "volume_5m": t.get("volume_5m"),
            "buy_sell_ratio_5m": t.get("buy_sell_ratio_5m"),
            "ultra_short_heat": t.get("ultra_short_heat"),
            "already_pumped_penalty": t.get("already_pumped_penalty"),
            "bubblemaps_score": t.get("bubblemaps_score"),
            "bubblemaps_cluster_max_pct": t.get("bubblemaps_cluster_max_pct"),
            "bubblemaps_cluster_count": t.get("bubblemaps_cluster_count"),
            # Algorithm v4 Sprint 1: Price action analysis
            "ath_24h": t.get("ath_24h"),
            "ath_ratio": t.get("ath_ratio"),
            "price_action_score": t.get("price_action_score"),
            "momentum_direction": t.get("momentum_direction"),
            # Algorithm v4 Sprint 4: Whale direction
            "whale_direction": t.get("whale_direction"),
            # Algorithm v7: Scoring improvements
            "weakest_component": t.get("weakest_component"),
            "weakest_component_value": t.get("weakest_component_value"),
            "score_interpretation": t.get("score_interpretation"),
            "data_confidence": t.get("data_confidence"),
            # Algorithm v9: Death detection + recency
            "freshest_mention_hours": t.get("freshest_mention_hours"),
            "death_penalty": t.get("death_penalty"),
            # v24: lifecycle phase + numeric encoding for ML
            "lifecycle_phase": t.get("lifecycle_phase"),
            "lifecycle_phase_num": _LIFECYCLE_PHASE_MAP.get(t.get("lifecycle_phase"), 3),
            "social_momentum_num": _MOMENTUM_PHASE_MAP.get(t.get("social_momentum_phase"), 1),
            # v10: DexScreener social/boost signals
            "boosts_active": t.get("boosts_active"),
            "has_twitter": t.get("has_twitter"),
            "has_telegram": t.get("has_telegram"),
            "has_website": t.get("has_website"),
            "social_count": t.get("social_count"),
            # v12: KOL entry premium
            "entry_premium": t.get("entry_premium"),
            "entry_premium_mult": t.get("entry_premium_mult"),
            # Tuning platform: normalized component values for client-side re-scoring
            "consensus_val": t.get("_consensus_val"),
            "sentiment_val": t.get("_sentiment_val"),
            "conviction_val": t.get("_conviction_val"),
            "breadth_val": t.get("_breadth_val"),
            "price_action_val": t.get("_price_action_val"),
            "pump_bonus": t.get("pump_bonus"),
            "wash_pen": t.get("wash_pen"),
            "pvp_pen": t.get("pvp_pen"),
            "pump_pen": t.get("pump_pen"),
            "breadth_pen": t.get("breadth_pen"),
            "activity_mult": t.get("activity_mult"),
            "crash_pen": t.get("crash_pen"),
            "pump_momentum_pen": t.get("pump_momentum_pen"),
            # stale_pen removed in v23 (dead code, redundant with death_penalty)
            "size_mult": t.get("size_mult"),
            "s_tier_mult": t.get("s_tier_mult"),
            # v15: KOL counts for conviction dampening
            "unique_kols": t.get("unique_kols"),
            "s_tier_count": sum(1 for tier in t.get("kol_tiers", {}).values() if tier == "S"),
            # v16: Extraction source counts
            "ca_mention_count": t.get("ca_mention_count", 0),
            "ticker_mention_count": t.get("ticker_mention_count", 0),
            "url_mention_count": t.get("url_mention_count", 0),
            "has_ca_mention": (t.get("ca_mention_count", 0) or 0) > 0 or (t.get("url_mention_count", 0) or 0) > 0,
            # v16: Gate reason (NULL = passed all gates, else reason for ejection)
            # v21: gate_mult — soft penalty value (1.0 = no penalty)
            "gate_reason": t.get("gate_reason"),
            "gate_mult": t.get("gate_mult", 1.0),
            "hype_pen": t.get("hype_pen", 1.0),
            "entry_drift_mult": t.get("entry_drift_mult"),
            # v16: Backtesting features
            "sol_price_at_snapshot": t.get("sol_price_at_snapshot"),
            "oldest_mention_hours": t.get("oldest_mention_hours"),
            # ML temporal features: cross-snapshot deltas
            "score_at_snapshot": deltas.get("score_at_snapshot"),
            "score_delta": deltas.get("score_delta"),
            "new_kol_ratio": deltas.get("new_kol_ratio"),
            # ML v2 Phase B: Temporal velocity features
            "score_velocity": deltas.get("score_velocity"),
            "score_acceleration": deltas.get("score_acceleration"),
            "mention_velocity": deltas.get("mention_velocity"),
            "volume_velocity": deltas.get("volume_velocity"),
            "price_velocity": deltas.get("price_velocity"),
            "social_momentum_phase": t.get("social_momentum_phase"),
            "kol_arrival_rate": deltas.get("kol_arrival_rate"),
            # ML v2 Phase C: Entry zone detection
            "entry_timing_quality": t.get("entry_timing_quality"),
            # v35: Proxy signals for top predictors
            "kol_freshness": t.get("kol_freshness"),
            "mention_heat_ratio": t.get("mention_heat_ratio"),
            "momentum_mult": t.get("momentum_mult"),
            # v25: Message-level text features
            "call_type_score": t.get("call_type_score"),
            "avg_msg_length": t.get("avg_msg_length"),
            "ca_mention_ratio": t.get("ca_mention_ratio"),
            "caps_ratio": t.get("caps_ratio"),
            "emoji_density": t.get("emoji_density"),
            "multi_token_ratio": t.get("multi_token_ratio"),
            "question_ratio": t.get("question_ratio"),
            "link_ratio": t.get("link_ratio"),
            # v26: Market context features
            "median_peak_return": t.get("median_peak_return"),
            "entry_vs_median_peak": t.get("entry_vs_median_peak"),
            "win_rate_7d": t.get("win_rate_7d"),
            "market_heat_24h": t.get("market_heat_24h"),
            "relative_volume": t.get("relative_volume"),
            "kol_saturation": t.get("kol_saturation"),
            # v23: First-seen price — carry forward from earliest snapshot
            # v27: Fixed operator precedence — parentheses around the or-chain
            "first_seen_price": (
                (prev.get("first_seen_price") or prev.get("price_at_snapshot"))
                if prev else None
            ) or t.get("price_usd"),
        }
        # v24: Compute price drift from first-seen price
        fsp = row.get("first_seen_price")
        cur_p = row.get("price_at_snapshot")
        if fsp and cur_p:
            try:
                fsp_f, cur_f = float(fsp), float(cur_p)
                if fsp_f > 0:
                    row["price_drift_from_first_seen"] = round(cur_f / fsp_f, 3)
            except (ValueError, TypeError):
                pass
        rows.append(_sanitize_row(row))

    if skipped_stale:
        logger.info("Skipped %d zombie snapshots (freshest_mention > 48h)", skipped_stale)

    if not rows:
        logger.info("No snapshots to insert (all filtered)")
        return

    # Batch insert with row-by-row fallback on failure
    try:
        client.table("token_snapshots").insert(rows).execute()
        logger.info("Inserted %d token snapshots", len(rows))
    except Exception as e:
        logger.error("Batch snapshot insert failed (%s: %s) — trying row-by-row", type(e).__name__, e)
        inserted = 0
        for row in rows:
            try:
                client.table("token_snapshots").insert(row).execute()
                inserted += 1
            except Exception as row_err:
                logger.error(
                    "Snapshot insert failed for %s (%s: %s)",
                    row.get("symbol", "?"), type(row_err).__name__, row_err,
                )
        logger.info("Row-by-row fallback: inserted %d/%d snapshots", inserted, len(rows))


def insert_kol_mentions(mentions: list[dict]) -> None:
    """
    v10: Bulk insert raw KOL mention texts into kol_mentions table.
    Deduplicates by (kol_group, message_date, symbol) to avoid re-inserting
    the same mention across scrape cycles.
    """
    if not mentions:
        return

    client = _get_client()

    # Deduplicate within batch (same KOL + same timestamp + same symbol = same mention)
    seen = set()
    unique_rows = []
    for m in mentions:
        key = (m["kol_group"], m["message_date"], m["symbol"])
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(_sanitize_row({
            "symbol": m["symbol"],
            "kol_group": m["kol_group"],
            "message_text": m["message_text"],
            "message_date": m["message_date"],
            "sentiment": m.get("sentiment"),
            "msg_conviction_score": m.get("msg_conviction_score"),
            "hours_ago": m.get("hours_ago"),
            "is_positive": m.get("is_positive", True),
            "narrative": m.get("narrative"),
            "tokens_in_message": m.get("tokens_in_message"),
            # v16: Extraction audit fields
            "extraction_method": m.get("extraction_method"),
            "extracted_cas": m.get("extracted_cas"),
            # v40: Resolved CA from extraction (CA/URL sources)
            "resolved_ca": m.get("resolved_ca"),
        }))

    # Upsert in chunks of 500 — ON CONFLICT (kol_group, message_date, symbol) DO NOTHING
    # This prevents duplicates across scrape cycles (same 7d messages re-processed)
    CHUNK = 500
    inserted = 0
    for i in range(0, len(unique_rows), CHUNK):
        chunk = unique_rows[i:i + CHUNK]
        try:
            client.table("kol_mentions").upsert(
                chunk, on_conflict="kol_group,message_date,symbol", ignore_duplicates=True
            ).execute()
            inserted += len(chunk)
        except Exception as e:
            logger.warning("kol_mentions upsert failed for chunk %d: %s", i // CHUNK, e)

    logger.info("Inserted %d KOL mentions (%d deduplicated from %d raw)",
                inserted, len(mentions) - len(unique_rows), len(mentions))
