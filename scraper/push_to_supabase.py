"""
Supabase upsert module for the scraper pipeline.
Uses service_role key to bypass RLS.
"""

import os
import logging
from datetime import datetime, timezone

from supabase import create_client, Client

logger = logging.getLogger(__name__)


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
            {
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
            }
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


def insert_snapshots(ranking: list[dict]) -> None:
    """
    Insert a snapshot row for each token in the ranking into token_snapshots.
    Used to collect ML training data (features at time of observation).
    Outcome labels are filled later by outcome_tracker.
    """
    if not ranking:
        return

    client = _get_client()

    rows = []
    for t in ranking:
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
        }
        rows.append(row)

    try:
        client.table("token_snapshots").insert(rows).execute()
        logger.info("Inserted %d token snapshots", len(rows))
    except Exception as e:
        logger.error("Failed to insert snapshots: %s", e)
