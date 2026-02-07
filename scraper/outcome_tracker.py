"""
Outcome tracker: fills price labels in token_snapshots after sufficient time has passed.
Checks 3 horizons (6h, 12h, 24h) and marks whether each token did a 2x.
"""

import os
import time
import logging
from datetime import datetime, timezone, timedelta

import requests
from supabase import create_client, Client

logger = logging.getLogger(__name__)

DEXSCREENER_SEARCH_URL = "https://api.dexscreener.com/latest/dex/search"
GECKOTERMINAL_OHLCV_URL = "https://api.geckoterminal.com/api/v2/networks/solana/pools/{pool}/ohlcv/minute"

HORIZONS = [
    {"hours": 6, "price_col": "price_after_6h", "flag_col": "did_2x_6h"},
    {"hours": 12, "price_col": "price_after_12h", "flag_col": "did_2x_12h"},
    {"hours": 24, "price_col": "price_after_24h", "flag_col": "did_2x_24h"},
]


def _get_client() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return create_client(url, key)


def _get_current_price(symbol: str) -> float | None:
    """Fetch current price from DexScreener search."""
    raw = symbol.lstrip("$")
    try:
        resp = requests.get(
            DEXSCREENER_SEARCH_URL,
            params={"q": raw},
            timeout=10,
        )
        if resp.status_code != 200:
            return None

        pairs = resp.json().get("pairs") or []
        # Find best Solana pair with exact match
        sol_pairs = [
            p for p in pairs
            if p.get("baseToken", {}).get("symbol", "").upper() == raw
            and p.get("chainId") == "solana"
        ]
        if not sol_pairs:
            sol_pairs = [
                p for p in pairs
                if p.get("baseToken", {}).get("symbol", "").upper() == raw
            ]
        if not sol_pairs:
            return None

        best = max(sol_pairs, key=lambda p: float(p.get("volume", {}).get("h24", 0) or 0))
        price = float(best.get("priceUsd", 0) or 0)
        return price if price > 0 else None

    except requests.RequestException as e:
        logger.warning("DexScreener price fetch failed for %s: %s", symbol, e)
        return None


def fill_outcomes() -> None:
    """
    For each horizon, find snapshots old enough to label but not yet labeled.
    Fetch current price and compute whether a 2x occurred.
    """
    client = _get_client()
    now = datetime.now(timezone.utc)
    total_updated = 0

    for horizon in HORIZONS:
        hours = horizon["hours"]
        price_col = horizon["price_col"]
        flag_col = horizon["flag_col"]

        # Find snapshots where this horizon's flag is still NULL
        # and the snapshot is old enough (snapshot_at < now - hours)
        cutoff = (now - timedelta(hours=hours)).isoformat()

        try:
            result = (
                client.table("token_snapshots")
                .select("id, symbol, price_at_snapshot")
                .is_(flag_col, "null")
                .lt("snapshot_at", cutoff)
                .limit(50)  # Process in batches to avoid timeout
                .execute()
            )
        except Exception as e:
            logger.error("Failed to query pending snapshots for %dh: %s", hours, e)
            continue

        snapshots = result.data or []
        if not snapshots:
            logger.debug("No pending snapshots for %dh horizon", hours)
            continue

        logger.info("Processing %d snapshots for %dh horizon", len(snapshots), hours)

        for snap in snapshots:
            snap_id = snap["id"]
            symbol = snap["symbol"]
            price_at = snap.get("price_at_snapshot")

            if not price_at or float(price_at) <= 0:
                # No baseline price — mark as False (can't determine 2x)
                try:
                    client.table("token_snapshots").update({
                        price_col: None,
                        flag_col: False,
                    }).eq("id", snap_id).execute()
                except Exception as e:
                    logger.error("Failed to update snapshot %d: %s", snap_id, e)
                continue

            price_at = float(price_at)
            current_price = _get_current_price(symbol)
            time.sleep(0.3)  # Rate limit

            if current_price is None:
                # Token may be dead/delisted — mark as no 2x
                try:
                    client.table("token_snapshots").update({
                        price_col: 0,
                        flag_col: False,
                    }).eq("id", snap_id).execute()
                except Exception as e:
                    logger.error("Failed to update snapshot %d: %s", snap_id, e)
                continue

            did_2x = current_price >= (price_at * 2.0)

            update_data = {
                price_col: current_price,
                flag_col: did_2x,
            }

            # For 24h horizon, also record max_price_24h (using current as approximation)
            # A future improvement could use GeckoTerminal OHLCV for true max
            if hours == 24:
                update_data["max_price_24h"] = current_price

            try:
                client.table("token_snapshots").update(update_data).eq("id", snap_id).execute()
                total_updated += 1
                if did_2x:
                    logger.info(
                        "2x CONFIRMED: %s at %dh (%.10f -> %.10f = %.1fx)",
                        symbol, hours, price_at, current_price, current_price / price_at,
                    )
            except Exception as e:
                logger.error("Failed to update snapshot %d: %s", snap_id, e)

    if total_updated > 0:
        logger.info("Outcome tracker: updated %d snapshot labels", total_updated)
    else:
        logger.debug("Outcome tracker: no snapshots to update")
