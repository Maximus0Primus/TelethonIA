"""
DexPaprika OHLCV enrichment: fetch 24h of 15-minute candles for price action analysis.

Replaces Birdeye OHLCV — DexPaprika is free (no API key, 10K req/day).
Endpoint: GET https://api.dexpaprika.com/networks/solana/pools/{pool}/ohlcv

Falls back to Birdeye if DexPaprika returns no data and BIRDEYE_API_KEY is set.
"""

import json
import time
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

from enrich_birdeye_ohlcv import (
    fetch_birdeye_ohlcv,
    _detect_support,
    _detect_resistance,
    count_support_touches,
    _get_api_key as _get_birdeye_key,
)

logger = logging.getLogger(__name__)

DEXPAPRIKA_BASE = "https://api.dexpaprika.com"
OHLCV_TOP_N = 100      # DexPaprika is free (10K req/day) — enrich all viable tokens
OHLCV_CACHE_FILE = Path(__file__).parent / "ohlcv_cache.json"
OHLCV_CACHE_TTL = 30 * 60  # 30 min — survives across consecutive 15-min cron runs


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


def _find_pool_address(token_address: str) -> str | None:
    """
    Look up the highest-volume Solana pool for a token via DexPaprika.
    Used when pair_address is not available from DexScreener enrichment.
    """
    url = f"{DEXPAPRIKA_BASE}/networks/solana/tokens/{token_address}/pools"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            logger.debug("DexPaprika pool lookup %d for %s", resp.status_code, token_address[:8])
            return None
        pools = resp.json()
        if not isinstance(pools, list) or not pools:
            return None
        # Pick pool with highest 24h volume
        best = max(pools, key=lambda p: float(p.get("volume_usd_24h", 0) or 0))
        return best.get("id")
    except (requests.RequestException, ValueError, KeyError) as e:
        logger.debug("DexPaprika pool lookup error for %s: %s", token_address[:8], e)
        return None


def fetch_dexpaprika_ohlcv(pool_address: str) -> dict | None:
    """
    Fetch 24h of 15-minute OHLCV candles from DexPaprika for a single pool.

    Returns dict with:
        candle_data: list of candle dicts
        ath_24h, atl_24h, current_price, ath_ratio
        support_level, resistance_level

    Returns None on failure.
    """
    if not pool_address:
        return None

    start = (datetime.now(timezone.utc) - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")
    url = f"{DEXPAPRIKA_BASE}/networks/solana/pools/{pool_address}/ohlcv"

    try:
        resp = requests.get(
            url,
            params={"start": start, "limit": 96, "interval": "15m"},
            timeout=15,
        )

        if resp.status_code == 429:
            logger.debug("DexPaprika 429 for pool %s", pool_address[:8])
            return None

        if resp.status_code != 200:
            logger.debug("DexPaprika OHLCV %d for pool %s", resp.status_code, pool_address[:8])
            return None

        items = resp.json()
        if not isinstance(items, list) or not items:
            logger.debug("DexPaprika OHLCV: no candles for pool %s", pool_address[:8])
            return None

        candles = []
        highs = []
        lows = []
        for item in items:
            o = float(item.get("open", 0) or 0)
            h = float(item.get("high", 0) or 0)
            l = float(item.get("low", 0) or 0)
            c = float(item.get("close", 0) or 0)
            v = float(item.get("volume", 0) or 0)
            # Parse time_close ISO → unix timestamp
            tc = item.get("time_close", "")
            try:
                ts = int(datetime.fromisoformat(tc.replace("Z", "+00:00")).timestamp())
            except (ValueError, AttributeError):
                ts = 0

            if h > 0 and l > 0:
                highs.append(h)
                lows.append(l)
                candles.append({
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": v,
                    "timestamp": ts,
                })

        if not highs:
            return None

        # v57: Detect SOL base token leak — Pump.fun pools with SOL as base return ~$85
        # Memecoins are always < $1. If median close > $50, data is SOL price, not token price.
        closes = [c["close"] for c in candles if c["close"] > 0]
        if closes:
            median_close = sorted(closes)[len(closes) // 2]
            if median_close > 50.0:
                logger.warning(
                    "DexPaprika SOL price leak for pool %s — median_close=%.2f, rejecting → Birdeye fallback",
                    pool_address[:12], median_close,
                )
                return None

        ath_24h = max(highs)
        atl_24h = min(lows)
        current_price = candles[-1]["close"] if candles else 0
        ath_ratio = current_price / ath_24h if ath_24h > 0 else 0

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
        logger.warning("DexPaprika OHLCV error for pool %s: %s", pool_address[:8], e)
        return None


def enrich_tokens_ohlcv(ranking: list[dict]) -> list[dict]:
    """
    Fetch OHLCV for the top OHLCV_TOP_N tokens using DexPaprika (free).
    Falls back to Birdeye if DexPaprika returns no data and BIRDEYE_API_KEY is set.
    Adds candle_data, ath_24h, ath_ratio, support_level, resistance_level to token dicts.
    Uses 1h cache. Modifies tokens in-place.
    """
    if not ranking:
        return ranking

    cache = _load_ohlcv_cache()
    birdeye_key = _get_birdeye_key()
    enriched = 0
    dexpaprika_calls = 0
    birdeye_calls = 0

    for i, token in enumerate(ranking):
        if i >= OHLCV_TOP_N:
            break

        pool = token.get("pair_address")
        mint = token.get("token_address")
        cache_key = pool or mint
        if not cache_key:
            continue

        # Check cache first
        cached = cache.get(cache_key)
        if cached and (time.time() - cached.get("_cached_at", 0)) < OHLCV_CACHE_TTL:
            token["candle_data"] = cached.get("candle_data")
            token["ath_24h"] = cached.get("ath_24h")
            token["atl_24h"] = cached.get("atl_24h")
            token["ath_ratio"] = cached.get("ath_ratio")
            token["support_level"] = cached.get("support_level")
            token["resistance_level"] = cached.get("resistance_level")
            enriched += 1
            continue

        # Resolve pool address if not available from DexScreener
        if not pool and mint:
            pool = _find_pool_address(mint)
            time.sleep(0.3)

        # Try DexPaprika first
        ohlcv = None
        if pool:
            ohlcv = fetch_dexpaprika_ohlcv(pool)
            dexpaprika_calls += 1

        # Fallback to Birdeye if DexPaprika failed and API key is available
        if ohlcv is None and birdeye_key and mint:
            ohlcv = fetch_birdeye_ohlcv(mint, birdeye_key)
            birdeye_calls += 1

        if ohlcv:
            token["candle_data"] = ohlcv["candle_data"]
            token["ath_24h"] = ohlcv["ath_24h"]
            token["atl_24h"] = ohlcv["atl_24h"]
            token["ath_ratio"] = ohlcv["ath_ratio"]
            token["support_level"] = ohlcv["support_level"]
            token["resistance_level"] = ohlcv["resistance_level"]
            enriched += 1

            # Cache (keep last 50 candles — RSI needs 15+, MACD 35+, BBands 20+)
            cache[cache_key] = {
                "ath_24h": ohlcv["ath_24h"],
                "atl_24h": ohlcv["atl_24h"],
                "ath_ratio": ohlcv["ath_ratio"],
                "support_level": ohlcv["support_level"],
                "resistance_level": ohlcv["resistance_level"],
                "candle_data": ohlcv["candle_data"][-50:],
                "_cached_at": time.time(),
            }

        time.sleep(0.3)  # Rate limit

    _save_ohlcv_cache(cache)
    logger.info(
        "OHLCV enrichment: %d enriched (top %d), %d DexPaprika + %d Birdeye fallback calls",
        enriched, OHLCV_TOP_N, dexpaprika_calls, birdeye_calls,
    )
    return ranking
