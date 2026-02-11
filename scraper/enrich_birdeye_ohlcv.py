"""
Birdeye OHLCV enrichment: fetch 24h of 15-minute candles for price action analysis.

Endpoint: GET https://public-api.birdeye.so/defi/ohlcv
Free Standard tier: 30K CUs/month. OHLCV = 30 CUs per call.
Only called for top 15 tokens per cycle to conserve CUs.
"""

import os
import json
import time
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

BIRDEYE_OHLCV_URL = "https://public-api.birdeye.so/defi/ohlcv"
OHLCV_TOP_N = 5  # Max tokens — Birdeye free tier = 30K CUs/month, 30 CUs/call
OHLCV_CACHE_FILE = Path(__file__).parent / "ohlcv_cache.json"
OHLCV_CACHE_TTL = 60 * 60  # 1 hour — candles don't change retroactively


def _load_ohlcv_cache() -> dict:
    if OHLCV_CACHE_FILE.exists():
        try:
            with open(OHLCV_CACHE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_ohlcv_cache(cache: dict) -> None:
    with open(OHLCV_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def _get_api_key() -> str | None:
    return os.environ.get("BIRDEYE_API_KEY")


def fetch_birdeye_ohlcv(token_address: str, api_key: str | None = None) -> dict | None:
    """
    Fetch 24h of 15-minute OHLCV candles from Birdeye for a single token.

    Returns dict with:
        candle_data: list of candle dicts
        ath_24h: highest high in 24h
        atl_24h: lowest low in 24h
        current_price: close of most recent candle
        ath_ratio: current_price / ath_24h
        support_level: detected support price (or None)
        resistance_level: detected resistance price (or None)

    Returns None on failure.
    """
    key = api_key or _get_api_key()
    if not key or not token_address:
        return None

    now = int(datetime.now(timezone.utc).timestamp())
    time_from = now - 24 * 3600  # 24h ago

    headers = {
        "X-API-KEY": key,
        "x-chain": "solana",
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.get(
                BIRDEYE_OHLCV_URL,
                params={
                    "address": token_address,
                    "type": "15m",
                    "time_from": time_from,
                    "time_to": now,
                },
                headers=headers,
                timeout=15,
            )

            if resp.status_code == 429:
                wait = 2 ** attempt + 1  # 2s, 3s, 5s
                logger.debug("Birdeye 429 for %s — retry %d in %ds", token_address[:8], attempt + 1, wait)
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                logger.warning("Birdeye OHLCV %d for %s", resp.status_code, token_address[:8])
                return None

            data = resp.json().get("data", {})
            items = data.get("items", [])

            if not items:
                logger.debug("Birdeye OHLCV: no candles for %s", token_address[:8])
                return None

            # Extract OHLCV data
            candles = []
            highs = []
            lows = []
            for item in items:
                h = item.get("h", 0)
                l = item.get("l", 0)
                o = item.get("o", 0)
                c = item.get("c", 0)
                v = item.get("v", 0)
                t = item.get("unixTime", 0)

                if h > 0 and l > 0:
                    highs.append(h)
                    lows.append(l)
                    candles.append({
                        "open": o,
                        "high": h,
                        "low": l,
                        "close": c,
                        "volume": v,
                        "timestamp": t,
                    })

            if not highs:
                return None

            ath_24h = max(highs)
            atl_24h = min(lows)
            current_price = candles[-1]["close"] if candles else 0

            ath_ratio = current_price / ath_24h if ath_24h > 0 else 0

            # Detect support/resistance levels
            support_level = _detect_support(candles)
            resistance_level = _detect_resistance(candles)

            return {
                "candle_data": candles,
                "ath_24h": ath_24h,
                "atl_24h": atl_24h,
                "current_price": current_price,
                "ath_ratio": round(ath_ratio, 4),
                "support_level": support_level,
                "resistance_level": resistance_level,
            }

        except requests.RequestException as e:
            logger.warning("Birdeye OHLCV error for %s: %s", token_address[:8], e)
            return None

    # All retries exhausted (429s)
    logger.warning("Birdeye OHLCV 429 exhausted retries for %s", token_address[:8])
    return None


def _detect_support(candles: list[dict], tolerance: float = 0.03) -> float | None:
    """
    Detect support level: a price zone where the low bounced 2+ times.
    Uses candle lows within ±tolerance of each other.
    """
    if len(candles) < 4:
        return None

    lows = [c["low"] for c in candles if c["low"] > 0]
    if not lows:
        return None

    # Cluster lows that are within tolerance of each other
    sorted_lows = sorted(lows)
    best_level = None
    best_count = 0

    for i, pivot in enumerate(sorted_lows):
        count = sum(1 for l in sorted_lows if abs(l - pivot) / pivot <= tolerance)
        if count > best_count:
            best_count = count
            best_level = pivot

    if best_count >= 2:
        return round(best_level, 8)
    return None


def _detect_resistance(candles: list[dict], tolerance: float = 0.03) -> float | None:
    """
    Detect resistance level: a price zone where the high was rejected 2+ times.
    """
    if len(candles) < 4:
        return None

    highs = [c["high"] for c in candles if c["high"] > 0]
    if not highs:
        return None

    sorted_highs = sorted(highs, reverse=True)
    best_level = None
    best_count = 0

    for pivot in sorted_highs:
        count = sum(1 for h in sorted_highs if abs(h - pivot) / pivot <= tolerance)
        if count > best_count:
            best_count = count
            best_level = pivot

    if best_count >= 2:
        return round(best_level, 8)
    return None


def count_support_touches(candles: list[dict], tolerance: float = 0.03) -> int:
    """
    Count how many times price touched the detected support level.
    Used by price_action.py for support strength scoring.
    """
    support = _detect_support(candles, tolerance)
    if support is None or support == 0:
        return 0

    touches = 0
    for c in candles:
        low = c.get("low", 0)
        if low > 0 and abs(low - support) / support <= tolerance:
            touches += 1

    return touches


def enrich_tokens_ohlcv(ranking: list[dict]) -> list[dict]:
    """
    Fetch Birdeye OHLCV for the top OHLCV_TOP_N tokens.
    Adds candle_data, ath_24h, ath_ratio, support_level, resistance_level to token dicts.
    Uses 1h cache to avoid redundant API calls.
    Modifies tokens in-place.
    """
    api_key = _get_api_key()
    if not api_key:
        logger.info("BIRDEYE_API_KEY not set — skipping OHLCV enrichment")
        return ranking

    if not ranking:
        return ranking

    cache = _load_ohlcv_cache()
    enriched = 0
    api_calls = 0

    for i, token in enumerate(ranking):
        if i >= OHLCV_TOP_N:
            break

        mint = token.get("token_address")
        if not mint:
            continue

        # Check cache first
        cached = cache.get(mint)
        if cached and (time.time() - cached.get("_cached_at", 0)) < OHLCV_CACHE_TTL:
            token["candle_data"] = cached.get("candle_data")
            token["ath_24h"] = cached.get("ath_24h")
            token["atl_24h"] = cached.get("atl_24h")
            token["ath_ratio"] = cached.get("ath_ratio")
            token["support_level"] = cached.get("support_level")
            token["resistance_level"] = cached.get("resistance_level")
            enriched += 1
            continue

        ohlcv = fetch_birdeye_ohlcv(mint, api_key)
        api_calls += 1
        if ohlcv:
            token["candle_data"] = ohlcv["candle_data"]
            token["ath_24h"] = ohlcv["ath_24h"]
            token["atl_24h"] = ohlcv["atl_24h"]
            token["ath_ratio"] = ohlcv["ath_ratio"]
            token["support_level"] = ohlcv["support_level"]
            token["resistance_level"] = ohlcv["resistance_level"]
            enriched += 1

            # Cache (without candle_data to keep file small)
            cache[mint] = {
                "ath_24h": ohlcv["ath_24h"],
                "atl_24h": ohlcv["atl_24h"],
                "ath_ratio": ohlcv["ath_ratio"],
                "support_level": ohlcv["support_level"],
                "resistance_level": ohlcv["resistance_level"],
                "candle_data": ohlcv["candle_data"][-10:],  # Keep last 10 candles only
                "_cached_at": time.time(),
            }

        time.sleep(0.5)

    _save_ohlcv_cache(cache)
    logger.info(
        "Birdeye OHLCV: %d enriched (top %d), %d API calls (%d cached)",
        enriched, OHLCV_TOP_N, api_calls, enriched - api_calls,
    )
    return ranking
