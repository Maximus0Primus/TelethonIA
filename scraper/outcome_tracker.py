"""
Outcome tracker: fills price labels in token_snapshots after sufficient time has passed.
Checks 3 horizons (6h, 12h, 24h) and marks whether each token did a 2x.

Uses GeckoTerminal OHLCV to find the MAX PRICE during the window, not just the
price at a single point in time. This prevents false negatives where a token
pumps to 3x then dumps back before we check.

GeckoTerminal free tier: 30 req/min, no auth required.
Fallback: DexScreener current price if OHLCV fails (original behavior).
"""

import os
import json
import time
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
from supabase import create_client, Client

logger = logging.getLogger(__name__)

DEXSCREENER_SEARCH_URL = "https://api.dexscreener.com/latest/dex/search"
GECKOTERMINAL_POOLS_URL = "https://api.geckoterminal.com/api/v2/networks/solana/tokens/{token_address}/pools"
GECKOTERMINAL_OHLCV_URL = "https://api.geckoterminal.com/api/v2/networks/solana/pools/{pool}/ohlcv/minute"

# Pool address cache (token_address → pool_address, stable mapping)
_POOL_CACHE_FILE = Path(__file__).parent / "pool_address_cache.json"
_POOL_CACHE_TTL = 7 * 24 * 3600  # 7 days — pool addresses are very stable

HORIZONS = [
    {"hours": 6, "price_col": "price_after_6h", "flag_col": "did_2x_6h", "max_col": "max_price_6h"},
    {"hours": 12, "price_col": "price_after_12h", "flag_col": "did_2x_12h", "max_col": "max_price_12h"},
    {"hours": 24, "price_col": "price_after_24h", "flag_col": "did_2x_24h", "max_col": "max_price_24h"},
]


def _get_client() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return create_client(url, key)


# === Pool Address Cache ===

def _load_pool_cache() -> dict:
    if _POOL_CACHE_FILE.exists():
        try:
            with open(_POOL_CACHE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_pool_cache(cache: dict) -> None:
    with open(_POOL_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


# === GeckoTerminal API ===

def _get_pool_address(token_address: str, pool_cache: dict) -> str | None:
    """
    Get the top Solana pool address for a token via GeckoTerminal.
    Caches result for 7 days (pool addresses are stable).
    """
    if not token_address:
        return None

    # Check cache
    cached = pool_cache.get(token_address)
    if cached and (time.time() - cached.get("_ts", 0)) < _POOL_CACHE_TTL:
        return cached.get("pool")

    try:
        resp = requests.get(
            GECKOTERMINAL_POOLS_URL.format(token_address=token_address),
            timeout=10,
        )
        if resp.status_code == 429:
            logger.warning("GeckoTerminal rate limited on pool lookup")
            return None
        if resp.status_code != 200:
            return None

        pools = resp.json().get("data") or []
        if not pools:
            return None

        # First pool is highest volume/liquidity
        pool_address = pools[0].get("attributes", {}).get("address")
        if pool_address:
            pool_cache[token_address] = {"pool": pool_address, "_ts": time.time()}

        return pool_address

    except requests.RequestException as e:
        logger.debug("GeckoTerminal pool lookup failed for %s: %s", token_address[:8], e)
        return None


def _get_max_price_in_window(
    pool_address: str,
    snapshot_ts: float,
    window_hours: int,
) -> float | None:
    """
    Get the maximum price (high) during a time window using GeckoTerminal OHLCV.
    Uses 5-min candles: 6h=72, 12h=144, 24h=288 candles — all fit in 1 request.

    Returns the max high price in USD, or None if API fails.
    """
    if not pool_address:
        return None

    # Window: from snapshot_ts to snapshot_ts + window_hours
    window_end_ts = int(snapshot_ts + window_hours * 3600)
    num_candles = (window_hours * 60) // 5 + 10  # 5-min candles + buffer

    try:
        resp = requests.get(
            GECKOTERMINAL_OHLCV_URL.format(pool=pool_address),
            params={
                "aggregate": 5,  # 5-minute candles
                "before_timestamp": window_end_ts,
                "limit": min(1000, num_candles),
                "currency": "usd",
                "token": "base",
            },
            timeout=15,
        )
        if resp.status_code == 429:
            logger.warning("GeckoTerminal rate limited on OHLCV")
            return None
        if resp.status_code != 200:
            logger.debug("GeckoTerminal OHLCV %d for pool %s", resp.status_code, pool_address[:8])
            return None

        ohlcv_list = resp.json().get("data", {}).get("attributes", {}).get("ohlcv_list") or []
        if not ohlcv_list:
            return None

        # Filter candles within our window [snapshot_ts, snapshot_ts + window_hours*3600]
        window_start_ts = int(snapshot_ts)
        max_high = 0.0
        candles_in_window = 0

        for candle in ohlcv_list:
            # candle = [timestamp, open, high, low, close, volume]
            candle_ts = candle[0]
            if window_start_ts <= candle_ts <= window_end_ts:
                high = float(candle[2])
                if high > max_high:
                    max_high = high
                candles_in_window += 1

        if candles_in_window == 0:
            logger.debug("No candles in window for pool %s", pool_address[:8])
            return None

        logger.debug(
            "OHLCV: %d candles in %dh window, max_high=%.10f",
            candles_in_window, window_hours, max_high,
        )
        return max_high if max_high > 0 else None

    except requests.RequestException as e:
        logger.debug("GeckoTerminal OHLCV failed for pool %s: %s", pool_address[:8], e)
        return None


# === DexScreener Fallback ===

def _get_current_price(symbol: str) -> float | None:
    """Fallback: fetch current price from DexScreener search."""
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


# === Main ===

def fill_outcomes() -> None:
    """
    For each horizon, find snapshots old enough to label but not yet labeled.
    Uses GeckoTerminal OHLCV to find the MAX price during the window.
    Falls back to DexScreener current price if OHLCV is unavailable.
    """
    client = _get_client()
    now = datetime.now(timezone.utc)
    pool_cache = _load_pool_cache()
    total_updated = 0
    ohlcv_used = 0
    fallback_used = 0

    for horizon in HORIZONS:
        hours = horizon["hours"]
        price_col = horizon["price_col"]
        flag_col = horizon["flag_col"]
        max_col = horizon["max_col"]

        # Find snapshots where this horizon's flag is still NULL
        # and the snapshot is old enough (snapshot_at < now - hours)
        cutoff = (now - timedelta(hours=hours)).isoformat()

        try:
            result = (
                client.table("token_snapshots")
                .select("id, symbol, price_at_snapshot, snapshot_at, token_address, pair_address")
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
                # No baseline price — mark as False
                try:
                    client.table("token_snapshots").update({
                        price_col: None,
                        flag_col: False,
                    }).eq("id", snap_id).execute()
                except Exception as e:
                    logger.error("Failed to update snapshot %d: %s", snap_id, e)
                continue

            price_at = float(price_at)

            # Parse snapshot timestamp
            snap_at_str = snap.get("snapshot_at", "")
            try:
                if snap_at_str.endswith("Z"):
                    snap_dt = datetime.fromisoformat(snap_at_str.replace("Z", "+00:00"))
                elif "+" in snap_at_str:
                    snap_dt = datetime.fromisoformat(snap_at_str)
                else:
                    snap_dt = datetime.fromisoformat(snap_at_str).replace(tzinfo=timezone.utc)
                snapshot_ts = snap_dt.timestamp()
            except (ValueError, AttributeError):
                snapshot_ts = None

            # Strategy: try OHLCV max price first, fallback to current price
            max_price = None
            used_ohlcv = False

            if snapshot_ts:
                # Get pool address (from snapshot or cache/API)
                pool_addr = snap.get("pair_address")
                token_addr = snap.get("token_address")

                if not pool_addr and token_addr:
                    pool_addr = _get_pool_address(token_addr, pool_cache)
                    time.sleep(2.1)  # GeckoTerminal: 30 req/min = 1 every 2s

                if pool_addr:
                    max_price = _get_max_price_in_window(pool_addr, snapshot_ts, hours)
                    time.sleep(2.1)  # Rate limit between OHLCV calls
                    if max_price is not None:
                        used_ohlcv = True

            # Fallback: current price (original behavior)
            if max_price is None:
                max_price = _get_current_price(symbol)
                time.sleep(0.3)

            if max_price is None:
                # Token may be dead/delisted — mark as no 2x
                try:
                    client.table("token_snapshots").update({
                        price_col: 0,
                        flag_col: False,
                    }).eq("id", snap_id).execute()
                except Exception as e:
                    logger.error("Failed to update snapshot %d: %s", snap_id, e)
                continue

            did_2x = max_price >= (price_at * 2.0)

            update_data = {
                price_col: max_price,
                flag_col: did_2x,
                max_col: max_price,
            }

            try:
                client.table("token_snapshots").update(update_data).eq("id", snap_id).execute()
                total_updated += 1
                if used_ohlcv:
                    ohlcv_used += 1
                else:
                    fallback_used += 1
                if did_2x:
                    method = "OHLCV max" if used_ohlcv else "current price"
                    logger.info(
                        "2x CONFIRMED (%s): %s at %dh (%.10f -> %.10f = %.1fx)",
                        method, symbol, hours, price_at, max_price, max_price / price_at,
                    )
            except Exception as e:
                logger.error("Failed to update snapshot %d: %s", snap_id, e)

    _save_pool_cache(pool_cache)

    if total_updated > 0:
        logger.info(
            "Outcome tracker: updated %d labels (%d via OHLCV max, %d via price fallback)",
            total_updated, ohlcv_used, fallback_used,
        )
    else:
        logger.debug("Outcome tracker: no snapshots to update")
