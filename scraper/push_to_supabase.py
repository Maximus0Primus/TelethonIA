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
    "death_penalty": 1.0,
    "freshest_mention_hours": 999.99,
    "entry_premium": 9999.999,
    "entry_premium_mult": 1.1,
    "new_kol_ratio": 1.0,
    "pump_bonus": 1.1,
    "wash_pen": 1.0,
    "pvp_pen": 1.0,
    "pump_pen": 1.0,
    "breadth_pen": 1.0,
    "stale_pen": 1.0,
}


# Integer columns in token_snapshots — must receive int, not float
_INT_COLS = {
    "birdeye_buy_24h", "birdeye_sell_24h", "boosts_active", "score_at_snapshot", "score_delta",
    "bubblemaps_cluster_count", "bundle_count", "buys_24h",
    "confirmation_pillars", "hedging_count", "helius_holder_count",
    "helius_recent_tx_count", "helius_unique_buyers", "holder_count",
    "jito_bundle_detected", "jito_bundle_slots", "jito_max_slot_txns",
    "mentions", "mentions_delta", "pair_count", "price_target_count",
    "pvp_recent_count", "pvp_same_name_count", "risk_count", "risk_score",
    "sells_24h", "trade_24h", "txn_count_24h", "unique_wallet_24h",
    "whale_count", "whale_new_entries",
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
            .select("symbol, mentions, sentiment, volume_24h, holder_count, score_at_snapshot, top_kols, snapshot_at")
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
    result = {
        "mentions_delta": None,
        "sentiment_delta": None,
        "volume_delta": None,
        "holder_delta": None,
        "score_at_snapshot": current.get("score"),
        "score_delta": None,
        "new_kol_ratio": None,
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

    # new_kol_ratio: what % of current KOLs are NEW (not in previous snapshot)
    current_kols = current.get("top_kols") or []
    prev_kols_raw = previous.get("top_kols")
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
            new_count = sum(1 for k in current_kols if k not in prev_set)
            result["new_kol_ratio"] = round(new_count / len(current_kols), 3)
        else:
            result["new_kol_ratio"] = 1.0  # All KOLs are "new" if no previous data

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
            # Algorithm v9: Death detection + recency
            "freshest_mention_hours": t.get("freshest_mention_hours"),
            "death_penalty": t.get("death_penalty"),
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
            "stale_pen": t.get("stale_pen"),
            # ML temporal features: cross-snapshot deltas
            "score_at_snapshot": deltas.get("score_at_snapshot"),
            "score_delta": deltas.get("score_delta"),
            "new_kol_ratio": deltas.get("new_kol_ratio"),
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
