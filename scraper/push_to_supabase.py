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
