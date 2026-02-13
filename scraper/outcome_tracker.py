"""
Outcome tracker: fills price labels in token_snapshots after sufficient time has passed.
Checks 4 horizons (1h, 6h, 12h, 24h) and marks whether each token did a 2x.

Uses OHLCV candle data to find the MAX PRICE during the window, not just the
price at a single point in time. This prevents false negatives where a token
pumps to 3x then dumps back before we check.

Fallback chain: GeckoTerminal OHLCV -> DexPaprika OHLCV -> SKIP (retry next cycle).
DexScreener current price is NOT used as fallback because it gives false negatives
on pump-and-dump patterns.

Consistency enforcement: if did_2x in a shorter horizon, all longer horizons
inherit that result (e.g. if 6h did 2x, then 12h and 24h must also be >= 2x).

GeckoTerminal free tier: 30 req/min, no auth required.
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

GECKOTERMINAL_POOLS_URL = "https://api.geckoterminal.com/api/v2/networks/solana/tokens/{token_address}/pools"
GECKOTERMINAL_OHLCV_URL = "https://api.geckoterminal.com/api/v2/networks/solana/pools/{pool}/ohlcv/minute"
DEXPAPRIKA_BASE = "https://api.dexpaprika.com"

# Pool address cache (token_address -> pool_address, stable mapping)
_POOL_CACHE_FILE = Path(__file__).parent / "pool_address_cache.json"
_POOL_CACHE_TTL = 7 * 24 * 3600  # 7 days -- pool addresses are very stable

# Process horizons from shortest to longest — shorter windows are nested in longer ones
# so we can enforce consistency (if 1h did 2x, all longer horizons must too)
HORIZONS = [
    {"hours": 1, "price_col": "price_after_1h", "flag_col": "did_2x_1h", "max_col": "max_price_1h"},
    {"hours": 6, "price_col": "price_after_6h", "flag_col": "did_2x_6h", "max_col": "max_price_6h"},
    {"hours": 12, "price_col": "price_after_12h", "flag_col": "did_2x_12h", "max_col": "max_price_12h"},
    {"hours": 24, "price_col": "price_after_24h", "flag_col": "did_2x_24h", "max_col": "max_price_24h"},
]

# Max snapshots to process per cycle (generous — we need to catch up)
BATCH_LIMIT = 150

# Time budget in seconds — exit gracefully before GH Action timeout
TIME_BUDGET_SECONDS = 10 * 60  # 10 minutes


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
    try:
        with open(_POOL_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except OSError as e:
        logger.warning("Failed to save pool cache: %s", e)


# === GeckoTerminal API ===

def _get_pool_address(token_address: str, pool_cache: dict) -> str | None:
    """Get the top Solana pool address for a token via GeckoTerminal."""
    if not token_address:
        return None

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

        pool_address = pools[0].get("attributes", {}).get("address")
        if pool_address:
            pool_cache[token_address] = {"pool": pool_address, "_ts": time.time()}

        return pool_address

    except requests.RequestException as e:
        logger.debug("GeckoTerminal pool lookup failed for %s: %s", token_address[:8], e)
        return None


def _get_max_price_gecko(
    pool_address: str,
    snapshot_ts: float,
    window_hours: int,
) -> tuple[float | None, float | None]:
    """
    Get the max high price AND the close of the last candle during a time window
    using GeckoTerminal OHLCV (5-min candles).

    Returns (max_high, last_close) or (None, None) if API fails.
    """
    if not pool_address:
        return None, None

    window_start_ts = int(snapshot_ts)
    window_end_ts = int(snapshot_ts + window_hours * 3600)
    num_candles = (window_hours * 60) // 5 + 10

    try:
        resp = requests.get(
            GECKOTERMINAL_OHLCV_URL.format(pool=pool_address),
            params={
                "aggregate": 5,
                "before_timestamp": window_end_ts,
                "limit": min(1000, num_candles),
                "currency": "usd",
                "token": "base",
            },
            timeout=15,
        )
        if resp.status_code == 429:
            logger.warning("GeckoTerminal rate limited on OHLCV")
            return None, None
        if resp.status_code != 200:
            return None, None

        ohlcv_list = resp.json().get("data", {}).get("attributes", {}).get("ohlcv_list") or []
        if not ohlcv_list:
            return None, None

        max_high = 0.0
        last_close = None
        last_ts = 0
        candles_in_window = 0

        for candle in ohlcv_list:
            candle_ts = candle[0]
            if window_start_ts <= candle_ts <= window_end_ts:
                high = float(candle[2])
                close = float(candle[4])
                if high > max_high:
                    max_high = high
                if candle_ts > last_ts:
                    last_ts = candle_ts
                    last_close = close
                candles_in_window += 1

        if candles_in_window == 0:
            return None, None

        logger.debug(
            "GeckoOHLCV: %d candles in %dh window, max=%.10f, close=%.10f",
            candles_in_window, window_hours, max_high, last_close or 0,
        )
        return (max_high if max_high > 0 else None, last_close)

    except requests.RequestException as e:
        logger.debug("GeckoTerminal OHLCV failed: %s", e)
        return None, None


# === DexPaprika Fallback (free, 10K req/day) ===

def _get_max_price_dexpaprika(
    pool_address: str,
    snapshot_ts: float,
    window_hours: int,
) -> tuple[float | None, float | None]:
    """
    Fallback: get max high + last close via DexPaprika OHLCV (free, no auth, 15-min candles).
    Returns (max_high, last_close) or (None, None).
    """
    if not pool_address:
        return None, None

    start_dt = datetime.fromtimestamp(snapshot_ts, tz=timezone.utc)
    end_dt = start_dt + timedelta(hours=window_hours)

    try:
        resp = requests.get(
            f"{DEXPAPRIKA_BASE}/networks/solana/pools/{pool_address}/ohlcv",
            params={
                "start": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end": end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "limit": (window_hours * 4) + 10,
                "interval": "15m",
            },
            timeout=15,
        )
        if resp.status_code != 200:
            return None, None

        items = resp.json()
        if not isinstance(items, list) or not items:
            return None, None

        max_high = 0.0
        last_close = None
        for item in items:
            h = float(item.get("high", 0) or 0)
            c = float(item.get("close", 0) or 0)
            if h > max_high:
                max_high = h
            last_close = c  # items are chronological, last one is the most recent

        return (max_high if max_high > 0 else None, last_close)

    except requests.RequestException:
        return None, None


# === Main ===

def fill_outcomes() -> None:
    """
    For each horizon, find snapshots old enough to label but not yet labeled.
    Uses OHLCV candle data to find the MAX price during the window.
    NO DexScreener current price fallback — that causes false negatives.
    If OHLCV fails, snapshot stays unlabeled and will be retried next cycle.

    Consistency enforcement: a longer horizon's max_price is always >= shorter horizon's.
    """
    client = _get_client()
    now = datetime.now(timezone.utc)
    pool_cache = _load_pool_cache()
    stats = {"updated": 0, "ohlcv": 0, "skipped": 0, "no_price": 0, "consistent": 0}
    start_time = time.time()
    budget_exceeded = False

    for horizon in HORIZONS:
        if budget_exceeded:
            break
        hours = horizon["hours"]
        price_col = horizon["price_col"]
        flag_col = horizon["flag_col"]
        max_col = horizon["max_col"]

        cutoff = (now - timedelta(hours=hours)).isoformat()

        try:
            result = (
                client.table("token_snapshots")
                .select("id, symbol, price_at_snapshot, snapshot_at, token_address, pair_address, "
                        "max_price_1h, max_price_6h, max_price_12h, max_price_24h, "
                        "did_2x_1h, did_2x_6h, did_2x_12h, did_2x_24h")
                .is_(flag_col, "null")
                .lt("snapshot_at", cutoff)
                .limit(BATCH_LIMIT)
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
            # Check time budget before each snapshot
            if time.time() - start_time > TIME_BUDGET_SECONDS:
                logger.warning("Outcome tracker: time budget exceeded (%.0fs), stopping gracefully",
                               time.time() - start_time)
                budget_exceeded = True
                break

            snap_id = snap["id"]
            symbol = snap["symbol"]
            price_at = snap.get("price_at_snapshot")

            if not price_at or float(price_at) <= 0:
                try:
                    client.table("token_snapshots").update({
                        price_col: None,
                        flag_col: False,
                    }).eq("id", snap_id).execute()
                    stats["no_price"] += 1
                except Exception as e:
                    logger.error("Failed to update %d: %s", snap_id, e)
                continue

            price_at = float(price_at)

            # Consistency check: if a shorter horizon already found 2x,
            # the longer horizon's max must be >= that shorter max.
            # E.g. if max_price_6h = 0.001 and we're computing 12h,
            # then max_price_12h must be >= 0.001.
            shorter_max = 0.0
            shorter_did_2x = False
            for sh in HORIZONS:
                if sh["hours"] >= hours:
                    break
                sh_max = snap.get(sh["max_col"])
                sh_did = snap.get(sh["flag_col"])
                if sh_max is not None and float(sh_max) > shorter_max:
                    shorter_max = float(sh_max)
                if sh_did is True:
                    shorter_did_2x = True

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

            max_price = None
            last_close = None
            source = "none"

            if snapshot_ts:
                pool_addr = snap.get("pair_address")
                token_addr = snap.get("token_address")

                if not pool_addr and token_addr:
                    pool_addr = _get_pool_address(token_addr, pool_cache)
                    time.sleep(2.1)

                if pool_addr:
                    # Try GeckoTerminal OHLCV first
                    max_price, last_close = _get_max_price_gecko(pool_addr, snapshot_ts, hours)
                    time.sleep(2.1)
                    if max_price is not None:
                        source = "gecko_ohlcv"

                    # Fallback: DexPaprika OHLCV
                    if max_price is None:
                        max_price, last_close = _get_max_price_dexpaprika(pool_addr, snapshot_ts, hours)
                        time.sleep(0.3)
                        if max_price is not None:
                            source = "dexpaprika_ohlcv"

            # If OHLCV failed but shorter horizon has data, use consistency rule
            if max_price is None and shorter_max > 0:
                max_price = shorter_max
                last_close = None
                source = "consistency_inherited"
                stats["consistent"] += 1

            if max_price is None:
                # No OHLCV data — skip and retry next cycle (DON'T use current price)
                stats["skipped"] += 1
                continue

            # Apply consistency floor: max_price can't be less than shorter horizon's max
            if shorter_max > 0 and max_price < shorter_max:
                logger.debug(
                    "Consistency fix %s/%dh: %.10f -> %.10f (from shorter window)",
                    symbol, hours, max_price, shorter_max,
                )
                max_price = shorter_max

            did_2x = max_price >= (price_at * 2.0)

            # If shorter horizon did 2x, this horizon must also be 2x
            if shorter_did_2x and not did_2x:
                did_2x = True
                logger.debug(
                    "Consistency override %s/%dh: shorter horizon was 2x, forcing did_2x=True",
                    symbol, hours,
                )

            update_data = {
                price_col: last_close if last_close else max_price,
                flag_col: did_2x,
                max_col: max_price,
            }

            try:
                client.table("token_snapshots").update(update_data).eq("id", snap_id).execute()
                stats["updated"] += 1
                stats["ohlcv"] += 1
                if did_2x:
                    logger.info(
                        "2x CONFIRMED (%s): %s at %dh (%.10f -> %.10f = %.1fx)",
                        source, symbol, hours, price_at, max_price, max_price / price_at,
                    )
            except Exception as e:
                logger.error("Failed to update %d (%s/%dh): %s", snap_id, symbol, hours, e)

    _save_pool_cache(pool_cache)

    logger.info(
        "Outcome tracker: updated=%d, ohlcv=%d, skipped=%d, no_price=%d, consistency_inherited=%d",
        stats["updated"], stats["ohlcv"], stats["skipped"], stats["no_price"], stats["consistent"],
    )

    # Second pass: fix existing inconsistencies in already-labeled data
    _fix_existing_inconsistencies(client)


def _fix_existing_inconsistencies(client: Client) -> None:
    """
    Fix historical data where max_6h > max_12h (physically impossible).
    If a shorter horizon has a higher max, propagate it to all longer horizons.
    """
    try:
        result = (
            client.table("token_snapshots")
            .select("id, symbol, price_at_snapshot, "
                    "max_price_1h, max_price_6h, max_price_12h, max_price_24h, "
                    "did_2x_1h, did_2x_6h, did_2x_12h, did_2x_24h")
            .not_.is_("max_price_6h", "null")
            .not_.is_("max_price_12h", "null")
            .limit(500)
            .execute()
        )
    except Exception as e:
        logger.warning("Failed to query for inconsistency fix: %s", e)
        return

    fixed = 0
    for snap in (result.data or []):
        price_at = float(snap.get("price_at_snapshot") or 0)
        if price_at <= 0:
            continue

        # Collect all known max prices
        maxes = {
            1: float(snap["max_price_1h"]) if snap.get("max_price_1h") else None,
            6: float(snap["max_price_6h"]) if snap.get("max_price_6h") else None,
            12: float(snap["max_price_12h"]) if snap.get("max_price_12h") else None,
            24: float(snap["max_price_24h"]) if snap.get("max_price_24h") else None,
        }

        # Compute running max (longer window must be >= shorter window)
        running_max = 0.0
        updates = {}
        horizon_map = {1: HORIZONS[0], 6: HORIZONS[1], 12: HORIZONS[2], 24: HORIZONS[3]}

        for h in [1, 6, 12, 24]:
            val = maxes[h]
            if val is not None:
                if val < running_max:
                    # Bug: shorter window had higher max -> fix this
                    hz = horizon_map[h]
                    new_did_2x = running_max >= (price_at * 2.0)
                    updates[hz["max_col"]] = running_max
                    updates[hz["price_col"]] = running_max  # also fix price_after
                    updates[hz["flag_col"]] = new_did_2x
                else:
                    running_max = val
            # Also propagate running_max forward even if this horizon is NULL

        if updates:
            try:
                client.table("token_snapshots").update(updates).eq("id", snap["id"]).execute()
                fixed += 1
            except Exception as e:
                logger.warning("Failed to fix inconsistency for %s: %s", snap["symbol"], e)

    if fixed > 0:
        logger.info("Fixed %d snapshots with window inconsistencies", fixed)
