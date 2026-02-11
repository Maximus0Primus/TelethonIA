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
    return sanitized


def _get_client() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return create_client(url, key)


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
    Returns { "symbol": { mentions, sentiment, volume_24h, holder_count, ... } }
    """
    if not symbols:
        return {}

    previous: dict[str, dict] = {}

    # Batch query: get the latest snapshot per symbol using a single query
    # Supabase doesn't support DISTINCT ON, so we fetch recent snapshots and deduplicate
    try:
        result = (
            client.table("token_snapshots")
            .select("symbol, mentions, sentiment, volume_24h, holder_count, snapshot_at")
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
    """Compute deltas between current and previous snapshot."""
    if previous is None:
        return {
            "mentions_delta": None,
            "sentiment_delta": None,
            "volume_delta": None,
            "holder_delta": None,
        }

    return {
        "mentions_delta": _safe_delta(current.get("mentions"), previous.get("mentions")),
        "sentiment_delta": round(
            _safe_delta(current.get("sentiment"), previous.get("sentiment")) or 0, 3
        ) if _safe_delta(current.get("sentiment"), previous.get("sentiment")) is not None else None,
        "volume_delta": _safe_delta(current.get("volume_24h"), previous.get("volume_24h")),
        "holder_delta": _safe_delta(current.get("holder_count"), previous.get("holder_count")),
    }


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
    for t in ranking:
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
            "top_kols": json.dumps(t.get("top_kols")) if t.get("top_kols") else None,
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
            "is_artificial_pump": t.get("is_artificial_pump"),
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
            "support_level": t.get("support_level"),
            # Algorithm v4 Sprint 4: Whale direction
            "whale_direction": t.get("whale_direction"),
            # Algorithm v6: pandas-ta technical indicators
            "rsi_14": t.get("rsi_14"),
            "macd_histogram": t.get("macd_histogram"),
            "bb_width": t.get("bb_width"),
            "bb_pct_b": t.get("bb_pct_b"),
            "obv_slope_norm": t.get("obv_slope_norm"),
            # Algorithm v7: Scoring improvements
            "lifecycle_phase": t.get("lifecycle_phase"),
            "weakest_component": t.get("weakest_component"),
            "weakest_component_value": t.get("weakest_component_value"),
            "score_interpretation": t.get("score_interpretation"),
            "data_confidence": t.get("data_confidence"),
            # Algorithm v8: Harvard adaptations (squeeze, trend, confirmation)
            "squeeze_state": t.get("squeeze_state"),
            "squeeze_score": t.get("squeeze_score"),
            "trend_strength": t.get("trend_strength"),
            "confirmation_pillars": t.get("confirmation_pillars"),
        }
        rows.append(_sanitize_row(row))

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
