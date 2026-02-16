"""
Outcome tracker: fills price labels in token_snapshots after sufficient time has passed.
Checks 7 horizons (1h, 6h, 12h, 24h, 48h, 72h, 7d) and marks whether each token did a 2x.

Uses OHLCV candle data to find the MAX PRICE during the window, not just the
price at a single point in time. This prevents false negatives where a token
pumps to 3x then dumps back before we check.

Fallback chain: GeckoTerminal OHLCV -> DexPaprika OHLCV -> Birdeye OHLCV -> SKIP.
Birdeye uses the token MINT address (not pool), so it can find tokens that were
deindexed by GeckoTerminal/DexPaprika/DexScreener but still exist on-chain.
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


class _RateLimiter:
    """Simple rate limiter: sleeps only the remaining time to maintain req/min target."""

    def __init__(self, requests_per_minute: int = 28):
        self._interval = 60.0 / requests_per_minute  # seconds between requests
        self._last_request = 0.0

    def wait(self):
        """Sleep just enough to maintain rate limit (accounts for processing time)."""
        now = time.monotonic()
        elapsed = now - self._last_request
        if elapsed < self._interval:
            time.sleep(self._interval - elapsed)
        self._last_request = time.monotonic()


# GeckoTerminal: 30 req/min, use 28 to leave headroom
_gecko_limiter = _RateLimiter(28)
# Birdeye: free tier is very limited (~30K CUs/month), use 10 req/min
_birdeye_limiter = _RateLimiter(10)


GECKOTERMINAL_POOLS_URL = "https://api.geckoterminal.com/api/v2/networks/solana/tokens/{token_address}/pools"
GECKOTERMINAL_OHLCV_URL = "https://api.geckoterminal.com/api/v2/networks/solana/pools/{pool}/ohlcv/minute"
DEXPAPRIKA_BASE = "https://api.dexpaprika.com"
BIRDEYE_OHLCV_URL = "https://public-api.birdeye.so/defi/ohlcv"

# Pool address cache (token_address -> pool_address, stable mapping)
_POOL_CACHE_FILE = Path(__file__).parent / "pool_address_cache.json"
_POOL_CACHE_TTL = 7 * 24 * 3600  # 7 days -- pool addresses are very stable

# Process horizons from shortest to longest — shorter windows are nested in longer ones
# so we can enforce consistency (if 1h did 2x, all longer horizons must too)
HORIZONS = [
    {"hours": 1, "price_col": "price_after_1h", "flag_col": "did_2x_1h", "max_col": "max_price_1h", "peak_col": "peak_hour_1h", "min_col": "min_price_1h", "t2x_col": "time_to_2x_1h"},
    {"hours": 6, "price_col": "price_after_6h", "flag_col": "did_2x_6h", "max_col": "max_price_6h", "peak_col": "peak_hour_6h", "min_col": "min_price_6h", "t2x_col": "time_to_2x_6h"},
    {"hours": 12, "price_col": "price_after_12h", "flag_col": "did_2x_12h", "max_col": "max_price_12h", "peak_col": "peak_hour_12h", "min_col": "min_price_12h", "t2x_col": "time_to_2x_12h"},
    {"hours": 24, "price_col": "price_after_24h", "flag_col": "did_2x_24h", "max_col": "max_price_24h", "peak_col": "peak_hour_24h", "min_col": "min_price_24h", "t2x_col": "time_to_2x_24h"},
    {"hours": 48, "price_col": "price_after_48h", "flag_col": "did_2x_48h", "max_col": "max_price_48h", "peak_col": "peak_hour_48h", "min_col": "min_price_48h", "t2x_col": "time_to_2x_48h"},
    {"hours": 72, "price_col": "price_after_72h", "flag_col": "did_2x_72h", "max_col": "max_price_72h", "peak_col": "peak_hour_72h", "min_col": "min_price_72h", "t2x_col": "time_to_2x_72h"},
    {"hours": 168, "price_col": "price_after_7d", "flag_col": "did_2x_7d", "max_col": "max_price_7d", "peak_col": "peak_hour_7d", "min_col": "min_price_7d", "t2x_col": "time_to_2x_7d"},
]

# Max snapshots to process per cycle
# v21: increased from 150 to 500 — 93% of snapshots were unlabeled, need to catch up
BATCH_LIMIT = 500

# Time budget in seconds — exit gracefully before GH Action timeout
# v23: 18 min (was 25). Must leave room for _fix_inconsistencies, _fill_first_call,
# backfill_bot_data, and auto_backtest which run after the main loop.
TIME_BUDGET_SECONDS = 18 * 60  # 18 minutes

# Sanity check: max plausible price ratio per horizon.
# If OHLCV returns max_price/price_at > this, the data is likely wrong
# (e.g. GeckoTerminal returning SOL price ~$85 instead of token price).
# Even for memecoins, 200x in 24h is extraordinary — anything above is a data bug.
MAX_PLAUSIBLE_RATIO = {1: 50, 6: 100, 12: 200, 24: 500, 48: 1000, 72: 2000, 168: 5000}

# v32: OHLCV-level sanity check threshold.
# If median candle price is >500x price_at_snapshot, the entire dataset is rejected.
# Catches SOL quote price leak where GeckoTerminal returns ~$85 SOL instead of token price.
_CANDLE_SANITY_RATIO = 500


def _check_candle_sanity(candles: list, price_at: float, symbol: str, source: str) -> bool:
    """Return True if candles look reasonable vs price_at_snapshot."""
    if not candles or price_at <= 0:
        return True  # can't check, let per-horizon check handle it
    median_high = candles[len(candles) // 2][2]  # high price of median candle
    if median_high > 0 and median_high / price_at > _CANDLE_SANITY_RATIO:
        logger.warning(
            "OHLCV price mismatch for %s (%s): candle=%.6f vs snapshot=%.6f (%.0fx) — rejecting",
            symbol, source, median_high, price_at, median_high / price_at,
        )
        return False
    return True


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


def _ts_to_minutes(ts: float | None, snapshot_ts: float) -> int | None:
    """Convert a Unix timestamp to minutes after snapshot. None if ts is None."""
    return round((ts - snapshot_ts) / 60) if ts else None


# TP/SL thresholds for bot simulation
TP_MULTS = [1.3, 1.5]   # +30%, +50%  (2.0x already tracked as time_to_2x)
SL_MULTS = [0.8, 0.7, 0.5]  # -20%, -30%, -50%

logger.info(
    "Outcome tracker: TP levels=%s, SL levels=%s",
    [f"+{int((t-1)*100)}%" for t in TP_MULTS],
    [f"-{int((1-s)*100)}%" for s in SL_MULTS],
)


def _get_max_price_gecko(
    pool_address: str,
    snapshot_ts: float,
    window_hours: int,
    price_at: float | None = None,
) -> tuple[float | None, float | None, float | None, float | None, float | None, dict | None]:
    """
    Get the max high price, min low price, and the close of the last candle
    during a time window using GeckoTerminal OHLCV (5-min candles).

    Returns (max_high, min_low, last_close, peak_ts, time_to_2x_hours, bot_data) or all Nones.
    peak_ts is the Unix timestamp of the candle with the highest price.
    time_to_2x_hours is hours after snapshot when price first reached 2x.
    bot_data contains TP/SL timing in minutes for bot simulation (only for 12h/24h).
    """
    if not pool_address:
        return None, None, None, None, None, None

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
            return None, None, None, None, None, None
        if resp.status_code != 200:
            return None, None, None, None, None, None

        ohlcv_list = resp.json().get("data", {}).get("attributes", {}).get("ohlcv_list") or []
        if not ohlcv_list:
            return None, None, None, None, None, None

        max_high = 0.0
        min_low = float('inf')
        peak_ts = None
        time_to_2x_ts = None
        target_2x = price_at * 2.0 if price_at and price_at > 0 else None
        last_close = None
        last_ts = 0
        candles_in_window = 0

        # Bot simulation: TP/SL tracking
        tp_timestamps = {m: None for m in TP_MULTS}
        sl_timestamps = {m: None for m in SL_MULTS}
        tp_prices = {m: price_at * m for m in TP_MULTS} if price_at and price_at > 0 else {}
        sl_prices = {m: price_at * m for m in SL_MULTS} if price_at and price_at > 0 else {}
        max_dd_before_tp30 = 0.0

        # Sort chronologically for time_to_2x (first candle that hits 2x)
        sorted_candles = sorted(ohlcv_list, key=lambda c: c[0])
        for candle in sorted_candles:
            candle_ts = candle[0]
            if window_start_ts <= candle_ts <= window_end_ts:
                high = float(candle[2])
                low = float(candle[3])
                close = float(candle[4])
                if high > max_high:
                    max_high = high
                    peak_ts = candle_ts
                if low < min_low:
                    min_low = low
                if target_2x and time_to_2x_ts is None and high >= target_2x:
                    time_to_2x_ts = candle_ts
                if candle_ts > last_ts:
                    last_ts = candle_ts
                    last_close = close
                candles_in_window += 1

                # Bot: TP detection (first candle where high >= target)
                for mult, target in tp_prices.items():
                    if tp_timestamps[mult] is None and high >= target:
                        tp_timestamps[mult] = candle_ts
                # Bot: SL detection (first candle where low <= target)
                for mult, target in sl_prices.items():
                    if sl_timestamps[mult] is None and low <= target:
                        sl_timestamps[mult] = candle_ts
                # Bot: Drawdown before TP30% (max dip from entry before first +30%)
                if price_at and price_at > 0 and tp_timestamps.get(1.3) is None:
                    dd = (price_at - low) / price_at
                    if dd > max_dd_before_tp30:
                        max_dd_before_tp30 = dd

        if candles_in_window == 0:
            return None, None, None, None, None, None

        time_to_2x_hours = None
        if time_to_2x_ts:
            time_to_2x_hours = round((time_to_2x_ts - snapshot_ts) / 3600, 2)
            time_to_2x_hours = max(0, min(window_hours, time_to_2x_hours))

        # Build bot_data dict (only meaningful for 12h/24h horizons)
        bot_data = None
        if window_hours in (12, 24) and price_at and price_at > 0:
            bot_data = {
                "t_1_3x": _ts_to_minutes(tp_timestamps.get(1.3), snapshot_ts),
                "t_1_5x": _ts_to_minutes(tp_timestamps.get(1.5), snapshot_ts),
                "t_sl20": _ts_to_minutes(sl_timestamps.get(0.8), snapshot_ts),
                "t_sl30": _ts_to_minutes(sl_timestamps.get(0.7), snapshot_ts),
                "t_sl50": _ts_to_minutes(sl_timestamps.get(0.5), snapshot_ts),
                "max_dd_pct": round(max_dd_before_tp30 * 100, 2) if max_dd_before_tp30 > 0 else 0,
            }

        logger.debug(
            "GeckoOHLCV: %d candles in %dh window, max=%.10f, min=%.10f, close=%.10f",
            candles_in_window, window_hours, max_high, min_low if min_low < float('inf') else 0, last_close or 0,
        )
        return (
            max_high if max_high > 0 else None,
            min_low if min_low < float('inf') else None,
            last_close,
            peak_ts,
            time_to_2x_hours,
            bot_data,
        )

    except requests.RequestException as e:
        logger.debug("GeckoTerminal OHLCV failed: %s", e)
        return None, None, None, None, None, None


# === DexPaprika Fallback (free, 10K req/day) ===

def _get_max_price_dexpaprika(
    pool_address: str,
    snapshot_ts: float,
    window_hours: int,
    price_at: float | None = None,
) -> tuple[float | None, float | None, float | None, float | None, float | None, dict | None]:
    """
    Fallback: get max high, min low, last close via DexPaprika OHLCV (free, no auth, 15-min candles).
    Returns (max_high, min_low, last_close, peak_ts, time_to_2x_hours, bot_data) or all Nones.
    """
    if not pool_address:
        return None, None, None, None, None, None

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
            return None, None, None, None, None, None

        items = resp.json()
        if not isinstance(items, list) or not items:
            return None, None, None, None, None, None

        max_high = 0.0
        min_low = float('inf')
        peak_ts = None
        time_to_2x_ts = None
        target_2x = price_at * 2.0 if price_at and price_at > 0 else None
        last_close = None

        # Bot simulation: TP/SL tracking
        tp_timestamps = {m: None for m in TP_MULTS}
        sl_timestamps = {m: None for m in SL_MULTS}
        tp_prices = {m: price_at * m for m in TP_MULTS} if price_at and price_at > 0 else {}
        sl_prices = {m: price_at * m for m in SL_MULTS} if price_at and price_at > 0 else {}
        max_dd_before_tp30 = 0.0

        for item in items:
            h = float(item.get("high", 0) or 0)
            lo = float(item.get("low", 0) or 0)
            c = float(item.get("close", 0) or 0)

            # Parse candle timestamp
            candle_ts = None
            ts_str = item.get("time_open") or item.get("time_close") or ""
            if ts_str:
                try:
                    candle_ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
                except (ValueError, AttributeError):
                    pass

            if h > max_high:
                max_high = h
                peak_ts = candle_ts
            if lo > 0 and lo < min_low:
                min_low = lo
            if target_2x and time_to_2x_ts is None and h >= target_2x and candle_ts:
                time_to_2x_ts = candle_ts
            last_close = c  # items are chronological, last one is the most recent

            # Bot: TP detection
            if candle_ts:
                for mult, target in tp_prices.items():
                    if tp_timestamps[mult] is None and h >= target:
                        tp_timestamps[mult] = candle_ts
                # Bot: SL detection
                for mult, target in sl_prices.items():
                    if sl_timestamps[mult] is None and lo > 0 and lo <= target:
                        sl_timestamps[mult] = candle_ts
                # Bot: Drawdown before TP30%
                if price_at and price_at > 0 and lo > 0 and tp_timestamps.get(1.3) is None:
                    dd = (price_at - lo) / price_at
                    if dd > max_dd_before_tp30:
                        max_dd_before_tp30 = dd

        time_to_2x_hours = None
        if time_to_2x_ts:
            time_to_2x_hours = round((time_to_2x_ts - snapshot_ts) / 3600, 2)
            time_to_2x_hours = max(0, min(window_hours, time_to_2x_hours))

        # Build bot_data dict (only meaningful for 12h/24h horizons)
        bot_data = None
        if window_hours in (12, 24) and price_at and price_at > 0:
            bot_data = {
                "t_1_3x": _ts_to_minutes(tp_timestamps.get(1.3), snapshot_ts),
                "t_1_5x": _ts_to_minutes(tp_timestamps.get(1.5), snapshot_ts),
                "t_sl20": _ts_to_minutes(sl_timestamps.get(0.8), snapshot_ts),
                "t_sl30": _ts_to_minutes(sl_timestamps.get(0.7), snapshot_ts),
                "t_sl50": _ts_to_minutes(sl_timestamps.get(0.5), snapshot_ts),
                "max_dd_pct": round(max_dd_before_tp30 * 100, 2) if max_dd_before_tp30 > 0 else 0,
            }

        return (
            max_high if max_high > 0 else None,
            min_low if min_low < float('inf') else None,
            last_close,
            peak_ts,
            time_to_2x_hours,
            bot_data,
        )

    except requests.RequestException:
        return None, None, None, None, None, None


# === Main ===

def _parse_snapshot_ts(snap_at_str: str) -> float | None:
    """Parse snapshot_at string to Unix timestamp."""
    try:
        if snap_at_str.endswith("Z"):
            snap_dt = datetime.fromisoformat(snap_at_str.replace("Z", "+00:00"))
        elif "+" in snap_at_str:
            snap_dt = datetime.fromisoformat(snap_at_str)
        else:
            snap_dt = datetime.fromisoformat(snap_at_str).replace(tzinfo=timezone.utc)
        return snap_dt.timestamp()
    except (ValueError, AttributeError):
        return None


def _extract_horizons_from_candles(
    sorted_candles: list,
    snapshot_ts: float,
    price_at: float,
    horizons_to_fill: list[dict],
) -> dict[int, dict]:
    """
    From a single set of chronologically sorted candles, extract max_price, min_price,
    last_close, peak_ts, time_to_2x, and bot_data for ALL requested horizons at once.

    This is the key optimization: 1 OHLCV fetch → label all pending horizons.
    """
    results = {}
    target_2x = price_at * 2.0 if price_at > 0 else None

    for hz in horizons_to_fill:
        hours = hz["hours"]
        window_end_ts = snapshot_ts + hours * 3600

        max_high = 0.0
        min_low = float('inf')
        peak_ts = None
        time_to_2x_ts = None
        last_close = None
        last_ts = 0
        candles_in_window = 0

        # Bot simulation: TP/SL tracking (only for 12h/24h)
        tp_timestamps = {m: None for m in TP_MULTS}
        sl_timestamps = {m: None for m in SL_MULTS}
        tp_prices = {m: price_at * m for m in TP_MULTS} if price_at > 0 else {}
        sl_prices = {m: price_at * m for m in SL_MULTS} if price_at > 0 else {}
        max_dd_before_tp30 = 0.0

        for candle in sorted_candles:
            candle_ts = candle[0]
            if candle_ts < snapshot_ts or candle_ts > window_end_ts:
                continue
            high = float(candle[2])
            low = float(candle[3])
            close = float(candle[4])

            if high > max_high:
                max_high = high
                peak_ts = candle_ts
            if low < min_low:
                min_low = low
            if target_2x and time_to_2x_ts is None and high >= target_2x:
                time_to_2x_ts = candle_ts
            if candle_ts > last_ts:
                last_ts = candle_ts
                last_close = close
            candles_in_window += 1

            # Bot: TP/SL detection
            if hours in (12, 24):
                for mult, target in tp_prices.items():
                    if tp_timestamps[mult] is None and high >= target:
                        tp_timestamps[mult] = candle_ts
                for mult, target in sl_prices.items():
                    if sl_timestamps[mult] is None and low <= target:
                        sl_timestamps[mult] = candle_ts
                if price_at > 0 and tp_timestamps.get(1.3) is None:
                    dd = (price_at - low) / price_at
                    if dd > max_dd_before_tp30:
                        max_dd_before_tp30 = dd

        if candles_in_window == 0:
            results[hours] = None
            continue

        time_to_2x_hours = None
        if time_to_2x_ts:
            time_to_2x_hours = round((time_to_2x_ts - snapshot_ts) / 3600, 2)
            time_to_2x_hours = max(0, min(hours, time_to_2x_hours))

        bot_data = None
        if hours in (12, 24) and price_at > 0:
            bot_data = {
                "t_1_3x": _ts_to_minutes(tp_timestamps.get(1.3), snapshot_ts),
                "t_1_5x": _ts_to_minutes(tp_timestamps.get(1.5), snapshot_ts),
                "t_sl20": _ts_to_minutes(sl_timestamps.get(0.8), snapshot_ts),
                "t_sl30": _ts_to_minutes(sl_timestamps.get(0.7), snapshot_ts),
                "t_sl50": _ts_to_minutes(sl_timestamps.get(0.5), snapshot_ts),
                "max_dd_pct": round(max_dd_before_tp30 * 100, 2) if max_dd_before_tp30 > 0 else 0,
            }

        results[hours] = {
            "max_price": max_high if max_high > 0 else None,
            "min_price": min_low if min_low < float('inf') else None,
            "last_close": last_close,
            "peak_ts": peak_ts,
            "time_to_2x_hours": time_to_2x_hours,
            "bot_data": bot_data,
        }

    return results


def fill_outcomes() -> None:
    """
    Batch-optimized outcome labeling: 1 OHLCV call per snapshot labels ALL pending horizons.

    Old approach: loop per horizon × per snapshot = 7 API calls per token.
    New approach: find snapshots with ANY unlabeled horizon, fetch longest needed window once,
    extract all shorter horizons from the same candle data.

    GeckoTerminal rate limit: 30 req/min. Old: ~100 tokens/run. New: ~700 tokens/run (7x faster).
    """
    client = _get_client()
    now = datetime.now(timezone.utc)
    pool_cache = _load_pool_cache()
    stats = {"updated": 0, "api_calls": 0, "skipped": 0, "no_price": 0, "consistent": 0}
    start_time = time.time()

    # Find snapshots with fillable unlabeled horizons.
    # v29 fix: each horizon is only included when the snapshot is old enough to fill it.
    # Without this, snapshots with only did_2x_7d=NULL (but <7 days old) clog the batch
    # and block newer snapshots from being labeled — the root cause of the labeling backlog.
    # v30: Order by score DESC so high-score tokens get labeled first (most useful for backtesting).
    or_parts = []
    for hz in HORIZONS:
        cutoff = (now - timedelta(hours=hz["hours"])).strftime("%Y-%m-%dT%H:%M:%SZ")
        or_parts.append(f'and({hz["flag_col"]}.is.null,snapshot_at.lt.{cutoff})')
    filter_str = ",".join(or_parts)

    try:
        result = (
            client.table("token_snapshots")
            .select("id, symbol, price_at_snapshot, snapshot_at, token_address, pair_address, "
                    "max_price_1h, max_price_6h, max_price_12h, max_price_24h, max_price_48h, max_price_72h, max_price_7d, "
                    "did_2x_1h, did_2x_6h, did_2x_12h, did_2x_24h, did_2x_48h, did_2x_72h, did_2x_7d")
            .or_(filter_str)
            # v33: Skip phantom scores (pre-v28 hard-gated tokens with no components).
            # They have inflated scores (54-69) and jump to front of queue, blocking real tokens.
            .not_.is_("consensus_val", "null")
            .order("score_at_snapshot", desc=True, nullsfirst=False)
            .order("snapshot_at", desc=False)
            .limit(BATCH_LIMIT)
            .execute()
        )
    except Exception as e:
        logger.error("Failed to query pending snapshots: %s", e)
        return

    snapshots = result.data or []
    if not snapshots:
        logger.info("No pending snapshots to label")
        _save_pool_cache(pool_cache)
        return

    logger.info("Processing %d snapshots (batch-optimized, all horizons per token)", len(snapshots))

    for snap in snapshots:
        # Check time budget
        if time.time() - start_time > TIME_BUDGET_SECONDS:
            logger.warning("Time budget exceeded (%.0fs), stopping gracefully", time.time() - start_time)
            break

        snap_id = snap["id"]
        symbol = snap["symbol"]
        price_at = snap.get("price_at_snapshot")

        if not price_at or float(price_at) <= 0:
            # Mark all unlabeled horizons as False (no price = can't evaluate)
            update = {}
            for hz in HORIZONS:
                if snap.get(hz["flag_col"]) is None:
                    update[hz["price_col"]] = None
                    update[hz["flag_col"]] = False
            if update:
                try:
                    client.table("token_snapshots").update(update).eq("id", snap_id).execute()
                    stats["no_price"] += 1
                except Exception as e:
                    logger.error("Failed to update %d: %s", snap_id, e)
            continue

        price_at = float(price_at)
        snapshot_ts = _parse_snapshot_ts(snap.get("snapshot_at", ""))
        if not snapshot_ts:
            stats["skipped"] += 1
            continue

        # Determine which horizons need filling and are old enough
        age_hours = (now.timestamp() - snapshot_ts) / 3600
        horizons_to_fill = []
        for hz in HORIZONS:
            if snap.get(hz["flag_col"]) is None and age_hours >= hz["hours"]:
                horizons_to_fill.append(hz)

        if not horizons_to_fill:
            continue

        # Resolve pool address (1 API call, cached for 7 days)
        pool_addr = snap.get("pair_address")
        token_addr = snap.get("token_address")
        if not pool_addr and token_addr:
            _gecko_limiter.wait()
            pool_addr = _get_pool_address(token_addr, pool_cache)
            stats["api_calls"] += 1

        if not pool_addr:
            # v27: No pool address = dead/unresolvable token. Mark due horizons as did_2x=false
            # instead of skipping (which left them as NULL forever, skewing ML training).
            update = {}
            for hz in horizons_to_fill:
                if snap.get(hz["flag_col"]) is None:
                    update[hz["flag_col"]] = False
                    update[hz["price_col"]] = None
                    update[hz["max_col"]] = None
            if update:
                try:
                    client.table("token_snapshots").update(update).eq("id", snap_id).execute()
                    stats["no_price"] += 1
                except Exception as e:
                    logger.error("Failed to update dead pool %d (%s): %s", snap_id, symbol, e)
            else:
                stats["skipped"] += 1
            continue

        # ONE OHLCV call for the LONGEST horizon needed → extract all shorter ones
        longest_hours = max(hz["hours"] for hz in horizons_to_fill)
        window_end_ts = int(snapshot_ts + longest_hours * 3600)
        num_candles = (longest_hours * 60) // 5 + 10

        sorted_candles = None
        source = "none"

        # Try GeckoTerminal
        try:
            _gecko_limiter.wait()
            resp = requests.get(
                GECKOTERMINAL_OHLCV_URL.format(pool=pool_addr),
                params={
                    "aggregate": 5,
                    "before_timestamp": window_end_ts,
                    "limit": min(1000, num_candles),
                    "currency": "usd",
                    "token": "base",
                },
                timeout=15,
            )
            stats["api_calls"] += 1

            if resp.status_code == 200:
                ohlcv_list = resp.json().get("data", {}).get("attributes", {}).get("ohlcv_list") or []
                if ohlcv_list:
                    sorted_candles = sorted(ohlcv_list, key=lambda c: c[0])
                    source = "gecko_ohlcv"
            elif resp.status_code == 429:
                logger.warning("GeckoTerminal rate limited — falling back to DexPaprika")
                # Don't sleep 10s — immediately try DexPaprika fallback below
        except requests.RequestException as e:
            logger.debug("GeckoTerminal OHLCV failed for %s: %s", symbol, e)

        # v32: reject SOL quote price leak before fallback chain short-circuits
        if sorted_candles and not _check_candle_sanity(sorted_candles, price_at, symbol, source):
            sorted_candles = None

        # Fallback: DexPaprika
        if sorted_candles is None:
            start_dt = datetime.fromtimestamp(snapshot_ts, tz=timezone.utc)
            end_dt = start_dt + timedelta(hours=longest_hours)
            try:
                resp = requests.get(
                    f"{DEXPAPRIKA_BASE}/networks/solana/pools/{pool_addr}/ohlcv",
                    params={
                        "start": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "end": end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "limit": (longest_hours * 4) + 10,
                        "interval": "15m",
                    },
                    timeout=15,
                )
                time.sleep(0.3)
                stats["api_calls"] += 1

                if resp.status_code == 200:
                    candles_raw = resp.json()
                    if isinstance(candles_raw, list) and candles_raw:
                        # DexPaprika format: list of dicts with open/high/low/close/time
                        sorted_candles = []
                        for c in candles_raw:
                            try:
                                ts_str = c.get("time") or c.get("timestamp", "")
                                if ts_str.endswith("Z"):
                                    ts_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                                else:
                                    ts_dt = datetime.fromisoformat(ts_str)
                                sorted_candles.append([
                                    int(ts_dt.timestamp()),
                                    float(c.get("open", 0)),
                                    float(c.get("high", 0)),
                                    float(c.get("low", 0)),
                                    float(c.get("close", 0)),
                                    float(c.get("volume", 0)),
                                ])
                            except (ValueError, TypeError):
                                continue
                        sorted_candles.sort(key=lambda x: x[0])
                        if sorted_candles:
                            source = "dexpaprika_ohlcv"
                        else:
                            sorted_candles = None
            except requests.RequestException as e:
                logger.debug("DexPaprika OHLCV failed for %s: %s", symbol, e)

        # v32: reject SOL quote price leak before Birdeye fallback
        if sorted_candles and not _check_candle_sanity(sorted_candles, price_at, symbol, source):
            sorted_candles = None

        # Fallback 3: Birdeye OHLCV (uses token MINT address, not pool)
        # v31: Recovers tokens deindexed by GeckoTerminal/DexPaprika but still on-chain
        if sorted_candles is None and token_addr:
            birdeye_key = os.environ.get("BIRDEYE_API_KEY", "")
            if birdeye_key:
                try:
                    _birdeye_limiter.wait()
                    resp = requests.get(
                        BIRDEYE_OHLCV_URL,
                        params={
                            "address": token_addr,
                            "type": "15m",
                            "time_from": int(snapshot_ts),
                            "time_to": int(snapshot_ts + longest_hours * 3600),
                        },
                        headers={"X-API-KEY": birdeye_key, "x-chain": "solana"},
                        timeout=15,
                    )
                    stats["api_calls"] += 1

                    if resp.status_code == 200:
                        items = resp.json().get("data", {}).get("items", [])
                        if items:
                            sorted_candles = []
                            for c in items:
                                sorted_candles.append([
                                    int(c.get("unixTime", 0)),
                                    float(c.get("o", 0)),
                                    float(c.get("h", 0)),
                                    float(c.get("l", 0)),
                                    float(c.get("c", 0)),
                                    float(c.get("v", 0)),
                                ])
                            sorted_candles.sort(key=lambda x: x[0])
                            if sorted_candles:
                                source = "birdeye_ohlcv"
                                logger.info("Birdeye recovered %d candles for %s", len(sorted_candles), symbol)
                            else:
                                sorted_candles = None
                    elif resp.status_code == 429:
                        logger.warning("Birdeye rate limited for %s", symbol)
                except requests.RequestException as e:
                    logger.debug("Birdeye OHLCV failed for %s: %s", symbol, e)

        # v32: final sanity check on Birdeye candles too
        if sorted_candles and not _check_candle_sanity(sorted_candles, price_at, symbol, source):
            sorted_candles = None

        if sorted_candles is None:
            # v27: All OHLCV sources failed (or all rejected by sanity check).
            # If snapshot is old enough (>48h), mark due horizons as did_2x=false.
            if age_hours > 48:
                update = {}
                for hz in horizons_to_fill:
                    if snap.get(hz["flag_col"]) is None:
                        update[hz["flag_col"]] = False
                        update[hz["price_col"]] = None
                        update[hz["max_col"]] = None
                if update:
                    try:
                        client.table("token_snapshots").update(update).eq("id", snap_id).execute()
                        stats["no_price"] += 1
                    except Exception as e:
                        logger.error("Failed to update dead OHLCV %d (%s): %s", snap_id, symbol, e)
                    continue
            stats["skipped"] += 1
            continue

        # Extract all horizons from the single candle dataset
        hz_results = _extract_horizons_from_candles(
            sorted_candles, snapshot_ts, price_at, horizons_to_fill,
        )

        # Build the update for all horizons at once
        update_data = {}
        running_max = 0.0
        running_did_2x = False

        # Also check already-labeled shorter horizons for consistency
        for hz in HORIZONS:
            existing_max = snap.get(hz["max_col"])
            existing_did = snap.get(hz["flag_col"])
            if existing_max is not None and float(existing_max) > running_max:
                running_max = float(existing_max)
            if existing_did is True:
                running_did_2x = True

            if hz not in horizons_to_fill:
                continue

            hours = hz["hours"]
            result_data = hz_results.get(hours)

            max_price = result_data["max_price"] if result_data else None
            min_price = result_data["min_price"] if result_data else None
            last_close = result_data["last_close"] if result_data else None
            peak_ts_val = result_data["peak_ts"] if result_data else None
            t2x_hours = result_data["time_to_2x_hours"] if result_data else None
            bot_data = result_data["bot_data"] if result_data else None

            # Consistency: inherit from shorter horizon if no OHLCV data
            if max_price is None and running_max > 0:
                max_price = running_max
                last_close = None
                stats["consistent"] += 1

            if max_price is None:
                continue  # Skip this horizon, retry next cycle

            # Sanity check
            ratio = max_price / price_at if price_at > 0 else 0
            max_ratio = MAX_PLAUSIBLE_RATIO.get(hours, 500)
            if ratio > max_ratio:
                logger.warning(
                    "Implausible OHLCV for %s/%dh: %.6f -> %.6f (%.0fx) — skipping",
                    symbol, hours, price_at, max_price, ratio,
                )
                continue

            # Consistency floor
            if running_max > 0 and max_price < running_max:
                max_price = running_max

            did_2x = max_price >= (price_at * 2.0)
            if running_did_2x and not did_2x:
                did_2x = True

            # Validate last_close
            safe_close = last_close
            if safe_close and price_at > 0 and (safe_close / price_at) > max_ratio:
                safe_close = None

            # Peak hour
            peak_hour = None
            if peak_ts_val and snapshot_ts:
                peak_hour = round((peak_ts_val - snapshot_ts) / 3600, 2)
                peak_hour = max(0, min(hours, peak_hour))

            update_data[hz["price_col"]] = safe_close if safe_close else max_price
            update_data[hz["flag_col"]] = did_2x
            update_data[hz["max_col"]] = max_price
            update_data[hz["peak_col"]] = peak_hour
            update_data[hz["min_col"]] = min_price
            update_data[hz["t2x_col"]] = t2x_hours

            # Bot simulation columns
            if bot_data and hours in (12, 24):
                hz_suffix = f"_{hours}h"
                update_data[f"time_to_1_3x_min{hz_suffix}"] = bot_data.get("t_1_3x")
                update_data[f"time_to_1_5x_min{hz_suffix}"] = bot_data.get("t_1_5x")
                update_data[f"time_to_sl20_min{hz_suffix}"] = bot_data.get("t_sl20")
                update_data[f"time_to_sl30_min{hz_suffix}"] = bot_data.get("t_sl30")
                update_data[f"time_to_sl50_min{hz_suffix}"] = bot_data.get("t_sl50")
                update_data[f"max_dd_before_tp_pct{hz_suffix}"] = bot_data.get("max_dd_pct")

            # Update running max for next horizons
            if max_price > running_max:
                running_max = max_price
            if did_2x:
                running_did_2x = True

            if did_2x:
                logger.info(
                    "2x CONFIRMED (%s): %s at %dh (%.10f -> %.10f = %.1fx)",
                    source, symbol, hours, price_at, max_price, max_price / price_at,
                )

        # Single DB update for ALL horizons of this snapshot
        if update_data:
            try:
                client.table("token_snapshots").update(update_data).eq("id", snap_id).execute()
                stats["updated"] += 1
            except Exception as e:
                logger.error("Failed to update %d (%s): %s", snap_id, symbol, e)

    _save_pool_cache(pool_cache)

    elapsed = time.time() - start_time
    throughput = stats["updated"] / max(1, elapsed) * 60  # tokens/minute
    logger.info(
        "Outcome tracker: updated=%d, api_calls=%d, skipped=%d, no_price=%d, consistency=%d "
        "(%.0fs elapsed, %.1f tokens/min)",
        stats["updated"], stats["api_calls"], stats["skipped"], stats["no_price"], stats["consistent"],
        elapsed, throughput,
    )

    # Second pass: fix existing inconsistencies (skip if <3 min left for other steps)
    remaining = 24 * 60 - (time.time() - start_time)  # 24 min total budget for fill_outcomes
    if remaining > 120:
        _fix_existing_inconsistencies(client)
    else:
        logger.warning("Skipping consistency fix (%.0fs remaining)", remaining)

    # Third pass: fill price_at_first_call (skip if tight on time)
    remaining = 24 * 60 - (time.time() - start_time)
    if remaining > 60:
        _fill_first_call_prices(client, pool_cache)
    else:
        logger.warning("Skipping first_call_prices (%.0fs remaining)", remaining)


def _fix_existing_inconsistencies(client: Client) -> None:
    """
    Fix historical data where max_6h > max_12h (physically impossible).
    If a shorter horizon has a higher max, propagate it to all longer horizons.
    """
    try:
        result = (
            client.table("token_snapshots")
            .select("id, symbol, price_at_snapshot, "
                    "max_price_1h, max_price_6h, max_price_12h, max_price_24h, max_price_48h, max_price_72h, max_price_7d, "
                    "did_2x_1h, did_2x_6h, did_2x_12h, did_2x_24h, did_2x_48h, did_2x_72h, did_2x_7d")
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
        maxes = {h["hours"]: float(snap[h["max_col"]]) if snap.get(h["max_col"]) else None for h in HORIZONS}

        # Compute running max (longer window must be >= shorter window)
        running_max = 0.0
        updates = {}
        horizon_map = {h["hours"]: h for h in HORIZONS}

        for h in [hz["hours"] for hz in HORIZONS]:
            val = maxes[h]
            if val is not None:
                # Sanity check: reject implausible values (SOL price leak)
                max_ratio = MAX_PLAUSIBLE_RATIO.get(h, 500)
                if price_at > 0 and val / price_at > max_ratio:
                    logger.warning(
                        "Implausible consistency val for %s/%dh: %.6f -> %.6f (%.0fx), nullifying",
                        snap["symbol"], h, price_at, val, val / price_at,
                    )
                    hz = horizon_map[h]
                    updates[hz["max_col"]] = None
                    updates[hz["price_col"]] = None
                    updates[hz["flag_col"]] = None
                    continue
                if val < running_max:
                    hz = horizon_map[h]
                    new_did_2x = running_max >= (price_at * 2.0)
                    updates[hz["max_col"]] = running_max
                    updates[hz["price_col"]] = running_max
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


def _fill_first_call_prices(client: Client, pool_cache: dict) -> None:
    """
    For snapshots with oldest_mention_hours but no price_at_first_call,
    look up OHLCV candle data to find the price when the first KOL called.
    Uses DexPaprika (free) to fetch candles around the first-call timestamp.
    Limited to 20 per cycle to stay within rate limits.
    """
    try:
        result = (
            client.table("token_snapshots")
            .select("id, symbol, snapshot_at, oldest_mention_hours, pair_address, token_address")
            .is_("price_at_first_call", "null")
            .not_.is_("oldest_mention_hours", "null")
            .not_.is_("pair_address", "null")
            .limit(20)
            .execute()
        )
    except Exception as e:
        logger.warning("Failed to query for first_call_prices: %s", e)
        return

    snapshots = result.data or []
    if not snapshots:
        return

    filled = 0
    for snap in snapshots:
        snap_at_str = snap.get("snapshot_at", "")
        oldest_h = float(snap.get("oldest_mention_hours") or 0)
        pool_addr = snap.get("pair_address")

        if not pool_addr or oldest_h <= 0:
            continue

        try:
            if snap_at_str.endswith("Z"):
                snap_dt = datetime.fromisoformat(snap_at_str.replace("Z", "+00:00"))
            elif "+" in snap_at_str:
                snap_dt = datetime.fromisoformat(snap_at_str)
            else:
                snap_dt = datetime.fromisoformat(snap_at_str).replace(tzinfo=timezone.utc)
            snapshot_ts = snap_dt.timestamp()
        except (ValueError, AttributeError):
            continue

        # First call happened at snapshot_ts - oldest_mention_hours * 3600
        first_call_ts = snapshot_ts - oldest_h * 3600
        # Fetch a small window of candles around the first call time
        start_dt = datetime.fromtimestamp(first_call_ts - 900, tz=timezone.utc)  # 15min before
        end_dt = datetime.fromtimestamp(first_call_ts + 900, tz=timezone.utc)    # 15min after

        try:
            resp = requests.get(
                f"{DEXPAPRIKA_BASE}/networks/solana/pools/{pool_addr}/ohlcv",
                params={
                    "start": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "end": end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "limit": 4,
                    "interval": "15m",
                },
                timeout=10,
            )
            time.sleep(0.3)
            if resp.status_code != 200:
                continue

            items = resp.json()
            if not isinstance(items, list) or not items:
                continue

            # Use close of the candle closest to first_call_ts
            best_candle = None
            best_diff = float('inf')
            for item in items:
                ts_str = item.get("time_open") or item.get("time_close") or ""
                if not ts_str:
                    continue
                try:
                    candle_ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
                except (ValueError, AttributeError):
                    continue
                diff = abs(candle_ts - first_call_ts)
                if diff < best_diff:
                    best_diff = diff
                    best_candle = item

            if best_candle:
                close_price = float(best_candle.get("close", 0) or 0)
                if close_price > 0:
                    client.table("token_snapshots").update({
                        "price_at_first_call": close_price,
                    }).eq("id", snap["id"]).execute()
                    filled += 1

        except Exception as e:
            logger.debug("first_call_price failed for %s: %s", snap["symbol"], e)

    if filled > 0:
        logger.info("Filled price_at_first_call for %d snapshots", filled)


def backfill_bot_data(batch_limit: int = 50) -> None:
    """
    Backfill TP/SL timing columns for snapshots that already have OHLCV labels
    but are missing bot simulation data. Targets 12h and 24h horizons.

    This is needed because outcome_tracker skips already-labeled snapshots
    (did_2x_12h IS NOT NULL), so they never get the new bot columns filled.

    Rate-limited: processes batch_limit snapshots per call, ~2.1s per OHLCV fetch.
    """
    client = _get_client()
    pool_cache = _load_pool_cache()
    stats = {"filled": 0, "skipped": 0, "errors": 0}

    for hours in [12, 24]:
        hz = f"_{hours}h"
        bot_col = f"time_to_1_3x_min{hz}"  # sentinel: if this is NULL, bot data is missing
        max_col = f"max_price_{hours}h"

        try:
            result = (
                client.table("token_snapshots")
                .select("id, symbol, price_at_snapshot, snapshot_at, token_address, pair_address")
                .is_(bot_col, "null")
                .not_.is_(max_col, "null")  # has OHLCV data
                .not_.is_("price_at_snapshot", "null")
                .limit(batch_limit)
                .execute()
            )
        except Exception as e:
            logger.error("backfill_bot_data: query failed for %dh: %s", hours, e)
            continue

        snapshots = result.data or []
        if not snapshots:
            logger.info("backfill_bot_data: no pending snapshots for %dh", hours)
            continue

        logger.info("backfill_bot_data: processing %d snapshots for %dh", len(snapshots), hours)

        for snap in snapshots:
            snap_id = snap["id"]
            symbol = snap["symbol"]
            price_at = float(snap.get("price_at_snapshot") or 0)
            if price_at <= 0:
                stats["skipped"] += 1
                continue

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
                stats["skipped"] += 1
                continue

            pool_addr = snap.get("pair_address")
            token_addr = snap.get("token_address")

            if not pool_addr and token_addr:
                _gecko_limiter.wait()
                pool_addr = _get_pool_address(token_addr, pool_cache)

            if not pool_addr:
                stats["skipped"] += 1
                continue

            # Fetch OHLCV — try Gecko then DexPaprika
            _gecko_limiter.wait()
            _, _, _, _, _, bot_data = _get_max_price_gecko(
                pool_addr, snapshot_ts, hours, price_at=price_at,
            )

            if bot_data is None:
                _, _, _, _, _, bot_data = _get_max_price_dexpaprika(
                    pool_addr, snapshot_ts, hours, price_at=price_at,
                )
                time.sleep(0.3)

            if bot_data is None:
                stats["skipped"] += 1
                continue

            # Write bot columns
            update_data = {
                f"time_to_1_3x_min{hz}": bot_data.get("t_1_3x"),
                f"time_to_1_5x_min{hz}": bot_data.get("t_1_5x"),
                f"time_to_sl20_min{hz}": bot_data.get("t_sl20"),
                f"time_to_sl30_min{hz}": bot_data.get("t_sl30"),
                f"time_to_sl50_min{hz}": bot_data.get("t_sl50"),
                f"max_dd_before_tp_pct{hz}": bot_data.get("max_dd_pct"),
            }

            try:
                client.table("token_snapshots").update(update_data).eq("id", snap_id).execute()
                stats["filled"] += 1
                logger.debug("backfill_bot_data: filled %s/%dh — TP30=%s SL20=%s",
                             symbol, hours, bot_data.get("t_1_3x"), bot_data.get("t_sl20"))
            except Exception as e:
                logger.error("backfill_bot_data: update failed %d: %s", snap_id, e)
                stats["errors"] += 1

    _save_pool_cache(pool_cache)
    logger.info("backfill_bot_data: filled=%d, skipped=%d, errors=%d",
                stats["filled"], stats["skipped"], stats["errors"])
