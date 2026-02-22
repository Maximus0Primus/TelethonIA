"""
Outcome tracker: fills price labels in token_snapshots after sufficient time has passed.
Checks 7 horizons (1h, 6h, 12h, 24h, 48h, 72h, 7d) and marks whether each token did a 2x.

Uses OHLCV candle data to find the MAX PRICE during the window, not just the
price at a single point in time. This prevents false negatives where a token
pumps to 3x then dumps back before we check.

Fallback chain: DexPaprika OHLCV -> GeckoTerminal OHLCV -> Birdeye OHLCV -> SKIP.
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
import threading
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from supabase import create_client, Client

logger = logging.getLogger(__name__)


class _RateLimiter:
    """Thread-safe rate limiter: sleeps only the remaining time to maintain req/min target."""

    def __init__(self, requests_per_minute: int = 28):
        self._interval = 60.0 / requests_per_minute  # seconds between requests
        self._last_request = 0.0
        self._lock = threading.Lock()

    def wait(self):
        """Sleep just enough to maintain rate limit (accounts for processing time)."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request
            if elapsed < self._interval:
                time.sleep(self._interval - elapsed)
            self._last_request = time.monotonic()


# GeckoTerminal: 30 req/min, use 28 to leave headroom
_gecko_limiter = _RateLimiter(28)
# Birdeye: free tier is very limited (~30K CUs/month), use 10 req/min
_birdeye_limiter = _RateLimiter(10)
# v54: DexPaprika rate limiter (replaces hardcoded time.sleep(0.3))
_dexpaprika_limiter = _RateLimiter(requests_per_minute=180)  # ~3 req/s

# v34: Adaptive GeckoTerminal circuit breaker.
# After N consecutive 429s, skip Gecko for the rest of the run.
# Logs show 179 rate limits per 18min run — each wastes 2.14s for nothing.
_GECKO_429_THRESHOLD = 3
_gecko_consecutive_429s = 0
_gecko_disabled = False
_gecko_lock = threading.Lock()  # v54: Thread safety for gecko circuit breaker


GECKOTERMINAL_POOLS_URL = "https://api.geckoterminal.com/api/v2/networks/solana/tokens/{token_address}/pools"
GECKOTERMINAL_OHLCV_URL = "https://api.geckoterminal.com/api/v2/networks/solana/pools/{pool}/ohlcv/minute"
DEXPAPRIKA_BASE = "https://api.dexpaprika.com"
BIRDEYE_OHLCV_URL = "https://public-api.birdeye.so/defi/ohlcv"

# v57: SOL mint address — used to detect inverted Pump.fun pools where SOL is the base token
_SOL_MINT = "So11111111111111111111111111111111111111112"

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
# v34: increased to 2000 — token-grouping means ~1 API call per unique token, not per snapshot
BATCH_LIMIT = 2000

# Time budget in seconds — exit gracefully before GH Action timeout
# v23: 18 min (was 25). Must leave room for _fix_inconsistencies, _fill_first_call,
# backfill_bot_data, and auto_backtest which run after the main loop.
# v34: 30 min (was 18). outcomes.yml timeout increased to 45min to clear labeling backlog.
# Feb 17 had only 39% of Feb 15 snapshots labeled (should be 100% after 48h).
TIME_BUDGET_SECONDS = 30 * 60  # 30 minutes

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


def _is_sol_base_pool(pool_addr: str, pool_cache: dict) -> bool:
    """Check if pool has SOL as base token (tokens[0]).

    v57: DexPaprika OHLCV returns the base token's price. Some Pump.fun pools
    have SOL as base instead of the memecoin, so OHLCV returns ~$85 SOL price.
    Result cached in pool_cache to avoid repeated API calls.
    """
    if not pool_addr:
        return False
    cache_key = f"_sol_base_{pool_addr}"
    cached = pool_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        resp = requests.get(
            f"{DEXPAPRIKA_BASE}/networks/solana/pools/{pool_addr}",
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            tokens = data.get("tokens", [])
            if tokens:
                base_addr = tokens[0].get("address", "")
                is_sol = base_addr == _SOL_MINT
                pool_cache[cache_key] = is_sol
                if is_sol:
                    logger.info(
                        "Pool %s has SOL as base token — DexPaprika OHLCV would return SOL price, skipping",
                        pool_addr[:12],
                    )
                return is_sol
    except requests.RequestException:
        pass
    return False  # assume normal on failure — _check_candle_sanity is fallback


# === GeckoTerminal API ===

def _get_pool_address(token_address: str, pool_cache: dict) -> str | None:
    """Get the top Solana pool address for a token. Falls back: GeckoTerminal → DexPaprika."""
    global _gecko_consecutive_429s, _gecko_disabled
    if not token_address:
        return None

    cached = pool_cache.get(token_address)
    if cached and (time.time() - cached.get("_ts", 0)) < _POOL_CACHE_TTL:
        return cached.get("pool")

    pool_address = None

    # 1. Try GeckoTerminal (unless circuit-breaker disabled)
    with _gecko_lock:
        gecko_ok = not _gecko_disabled
    if gecko_ok:
        try:
            resp = requests.get(
                GECKOTERMINAL_POOLS_URL.format(token_address=token_address),
                timeout=10,
            )
            if resp.status_code == 429:
                with _gecko_lock:
                    _gecko_consecutive_429s += 1
                    if _gecko_consecutive_429s >= _GECKO_429_THRESHOLD:
                        _gecko_disabled = True
                        logger.warning("GeckoTerminal disabled (pool lookup 429s)")
            elif resp.status_code == 200:
                pools = resp.json().get("data") or []
                if pools:
                    pool_address = pools[0].get("attributes", {}).get("address")
        except requests.RequestException as e:
            logger.debug("GeckoTerminal pool lookup failed for %s: %s", token_address[:8], e)

    # 2. v34 fallback: DexPaprika token→pool lookup (no API key needed, 10K req/day)
    if not pool_address:
        try:
            resp = requests.get(
                f"{DEXPAPRIKA_BASE}/search?query={token_address}&network=solana",
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                pools = data.get("pools") or []
                for p in pools:
                    addr = p.get("id", "")
                    # DexPaprika pool id format: "solana_<pool_address>"
                    if addr.startswith("solana_"):
                        pool_address = addr.replace("solana_", "", 1)
                        break
        except requests.RequestException as e:
            logger.debug("DexPaprika pool lookup failed for %s: %s", token_address[:8], e)

    if pool_address:
        pool_cache[token_address] = {"pool": pool_address, "_ts": time.time()}

    return pool_address


def _ts_to_minutes(ts: float | None, snapshot_ts: float) -> int | None:
    """Convert a Unix timestamp to minutes after snapshot. None if ts is None."""
    return round((ts - snapshot_ts) / 60) if ts else None


# TP/SL thresholds for bot simulation
TP_MULTS = [1.3, 1.5, 2.0, 3.0, 5.0]   # +30%, +50%, +100%, +200%, +400%
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
            if window_start_ts <= candle_ts < window_end_ts:
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

        # Build bot_data dict (for 12h/24h/48h horizons)
        bot_data = None
        if window_hours in (12, 24, 48) and price_at and price_at > 0:
            bot_data = {
                "t_1_3x": _ts_to_minutes(tp_timestamps.get(1.3), snapshot_ts),
                "t_1_5x": _ts_to_minutes(tp_timestamps.get(1.5), snapshot_ts),
                "t_2x":   _ts_to_minutes(tp_timestamps.get(2.0), snapshot_ts),
                "t_3x":   _ts_to_minutes(tp_timestamps.get(3.0), snapshot_ts),
                "t_5x":   _ts_to_minutes(tp_timestamps.get(5.0), snapshot_ts),
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

        # Build bot_data dict (for 12h/24h/48h horizons)
        bot_data = None
        if window_hours in (12, 24, 48) and price_at and price_at > 0:
            bot_data = {
                "t_1_3x": _ts_to_minutes(tp_timestamps.get(1.3), snapshot_ts),
                "t_1_5x": _ts_to_minutes(tp_timestamps.get(1.5), snapshot_ts),
                "t_2x":   _ts_to_minutes(tp_timestamps.get(2.0), snapshot_ts),
                "t_3x":   _ts_to_minutes(tp_timestamps.get(3.0), snapshot_ts),
                "t_5x":   _ts_to_minutes(tp_timestamps.get(5.0), snapshot_ts),
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

    # v41: Entry price freshness correction. price_at_snapshot can be 5-15min stale
    # (from DexScreener enrichment). Use the first OHLCV candle's open price if it
    # diverges significantly — the candle is time-aligned to the snapshot window.
    if sorted_candles and price_at > 0:
        first_candle = None
        for c in sorted_candles:
            if c[0] >= snapshot_ts:
                first_candle = c
                break
        if first_candle:
            candle_open = float(first_candle[1])
            if candle_open > 0:
                drift = abs(candle_open - price_at) / price_at
                if drift > 0.30:
                    logger.debug(
                        "Entry price corrected: snapshot=%.6f candle_open=%.6f (drift=%.0f%%)",
                        price_at, candle_open, drift * 100,
                    )
                    price_at = candle_open

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
            if candle_ts < snapshot_ts or candle_ts >= window_end_ts:
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
            if hours in (12, 24, 48):
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
        if hours in (12, 24, 48) and price_at > 0:
            bot_data = {
                "t_1_3x": _ts_to_minutes(tp_timestamps.get(1.3), snapshot_ts),
                "t_1_5x": _ts_to_minutes(tp_timestamps.get(1.5), snapshot_ts),
                "t_2x":   _ts_to_minutes(tp_timestamps.get(2.0), snapshot_ts),
                "t_3x":   _ts_to_minutes(tp_timestamps.get(3.0), snapshot_ts),
                "t_5x":   _ts_to_minutes(tp_timestamps.get(5.0), snapshot_ts),
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


def _fetch_ohlcv_candles(
    pool_addr: str,
    token_addr: str | None,
    start_ts: float,
    end_ts: float,
    price_at: float,
    symbol: str,
    stats: dict,
    pool_cache: dict | None = None,
) -> tuple[list | None, str]:
    """
    Fetch OHLCV candles for a pool covering [start_ts, end_ts].
    v39: Reordered to DexPaprika -> GeckoTerminal -> Birdeye.
    v57: Skip DexPaprika for pools with SOL as base token (returns SOL price ~$85).
    Returns (sorted_candles, source) where candles are [ts, o, h, l, c, v].
    """
    sorted_candles = None
    source = "none"

    window_seconds = end_ts - start_ts

    # v57: Skip DexPaprika if pool has SOL as base token (returns SOL price instead of memecoin)
    sol_base = pool_cache is not None and _is_sol_base_pool(pool_addr, pool_cache)

    # 1. DexPaprika first (10K req/day) — skip for inverted pools (v57: SOL as base)
    if not sol_base:
        start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)
        num_candles_15m = int(window_seconds / 900) + 10
        try:
            _dexpaprika_limiter.wait()
            resp = requests.get(
                f"{DEXPAPRIKA_BASE}/networks/solana/pools/{pool_addr}/ohlcv",
                params={
                    "start": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "end": end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "limit": min(1000, num_candles_15m),
                    "interval": "15m",
                },
                timeout=15,
            )
            stats["api_calls"] += 1

            if resp.status_code == 200:
                candles_raw = resp.json()
                if isinstance(candles_raw, list) and candles_raw:
                    sorted_candles = []
                    for c in candles_raw:
                        try:
                            # v36-fix: DexPaprika uses time_open/time_close, NOT time/timestamp
                            ts_str = c.get("time_open") or c.get("time_close") or c.get("time") or c.get("timestamp", "")
                            if not ts_str:
                                continue
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

        # v32: reject SOL quote price leak
        if sorted_candles and not _check_candle_sanity(sorted_candles, price_at, symbol, source):
            sorted_candles = None

    # 2. GeckoTerminal fallback (30 req/min — rate-limits fast, skip if circuit breaker tripped)
    global _gecko_consecutive_429s, _gecko_disabled
    with _gecko_lock:
        gecko_ok = not _gecko_disabled
    if sorted_candles is None and gecko_ok:
        num_candles_5m = int(window_seconds / 300) + 10
        try:
            _gecko_limiter.wait()
            resp = requests.get(
                GECKOTERMINAL_OHLCV_URL.format(pool=pool_addr),
                params={
                    "aggregate": 5,
                    "before_timestamp": int(end_ts),
                    "limit": min(1000, num_candles_5m),
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
                with _gecko_lock:
                    _gecko_consecutive_429s = 0  # Reset on success
            elif resp.status_code == 429:
                with _gecko_lock:
                    _gecko_consecutive_429s += 1
                    if _gecko_consecutive_429s >= _GECKO_429_THRESHOLD:
                        _gecko_disabled = True
                        logger.warning(
                            "GeckoTerminal disabled for this run (%d consecutive 429s) — using Birdeye",
                            _gecko_consecutive_429s,
                        )
                    else:
                        logger.warning("GeckoTerminal rate limited — falling back to Birdeye")
        except requests.RequestException as e:
            logger.debug("GeckoTerminal OHLCV failed for %s: %s", symbol, e)

    # v32: reject SOL quote price leak before Birdeye fallback
    if sorted_candles and not _check_candle_sanity(sorted_candles, price_at, symbol, source):
        sorted_candles = None

    # 3. Birdeye OHLCV (uses token MINT address, not pool)
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
                        "time_from": int(start_ts),
                        "time_to": int(end_ts),
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

    return sorted_candles, source


def _mark_no_price(client: "Client", snap: dict, stats: dict) -> None:
    """Snapshot has no valid price_at_snapshot — can never be labeled.

    v36-fix: Set max_price sentinels (0) so it exits the batch query.
    Without this, no-price snapshots are zombies that re-enter every run.
    """
    update_data = {}
    for hz in HORIZONS:
        if snap.get(hz["flag_col"]) is None:
            update_data[hz["max_col"]] = 0  # sentinel
    if update_data:
        try:
            client.table("token_snapshots").update(update_data).eq("id", snap["id"]).execute()
        except Exception as e:
            logger.debug("Failed to mark no_price for snap %d: %s", snap["id"], e)
    stats["no_price"] += 1


def _mark_dead_pool(client: "Client", snap: dict, now_ts: float, stats: dict) -> None:
    """Token has no resolvable pool address.

    v36-fix: Set max_price sentinels (0) so the snapshot exits the batch query.
    did_2x stays NULL (genuinely unknown). Training filters max_price > 0.
    """
    snapshot_ts = _parse_snapshot_ts(snap.get("snapshot_at"))
    age_hours = (now_ts - snapshot_ts) / 3600 if snapshot_ts else 999

    # Only mark as dead if old enough (>36h) — give pool resolution time to work
    if age_hours > 36:
        update_data = {}
        for hz in HORIZONS:
            if snap.get(hz["flag_col"]) is None:
                update_data[hz["max_col"]] = 0  # sentinel
        if update_data:
            try:
                client.table("token_snapshots").update(update_data).eq("id", snap["id"]).execute()
            except Exception as e:
                logger.debug("Failed to mark dead_pool for snap %d: %s", snap["id"], e)
    stats["no_price"] += 1


def _mark_ohlcv_failed(client: "Client", snap: dict, horizons_to_fill: list[dict], stats: dict) -> None:
    """Mark snapshot as permanently unfillable after exhausting all OHLCV sources.

    v34 fix: Never set did_2x=False without real price data (phantom labels).
    v36 fix: But leaving NULL caused zombie snapshots to re-enter the queue forever,
    wasting API budget on 4,274+ dead tokens every run. Solution: set max_price=0
    as a sentinel meaning 'checked all sources, no OHLCV available'. Training code
    must filter max_price_24h > 0 to exclude these.
    did_2x stays NULL (truly unknown), but the max_price sentinel removes them from
    the batch query because _extract_horizons_from_candles won't be called.
    """
    update_data = {}
    for hz in horizons_to_fill:
        # Set max_price = 0 as sentinel (real prices are always > 0)
        # This fills the column so the OR filter (flag.is.null AND max.is.null) won't match
        update_data[hz["max_col"]] = 0
        # Leave did_2x as NULL — we genuinely don't know, but we can't get the data
    if update_data:
        try:
            client.table("token_snapshots").update(update_data).eq("id", snap["id"]).execute()
        except Exception as e:
            logger.debug("Failed to mark ohlcv_failed for snap %d: %s", snap["id"], e)
    stats["no_price"] += 1


def _label_snapshot(
    client: "Client",
    snap: dict,
    snapshot_ts: float,
    price_at: float,
    horizons_to_fill: list[dict],
    sorted_candles: list,
    source: str,
    stats: dict,
) -> None:
    """Label a single snapshot using shared candle data."""
    hz_results = _extract_horizons_from_candles(
        sorted_candles, snapshot_ts, price_at, horizons_to_fill,
    )

    update_data = {}
    symbol = snap["symbol"]
    snap_id = snap["id"]

    for hz in horizons_to_fill:
        hours = hz["hours"]
        result_data = hz_results.get(hours)

        max_price = result_data["max_price"] if result_data else None
        min_price = result_data["min_price"] if result_data else None
        last_close = result_data["last_close"] if result_data else None
        peak_ts_val = result_data["peak_ts"] if result_data else None
        t2x_hours = result_data["time_to_2x_hours"] if result_data else None
        bot_data = result_data["bot_data"] if result_data else None

        if max_price is None:
            # v39: If snapshot is old enough (2x horizon), mark with sentinel to prevent
            # infinite retry. Candles exist but don't cover this horizon = permanently unfillable.
            age_h = (time.time() - snapshot_ts) / 3600
            if age_h > hours * 2:
                update_data[hz["max_col"]] = 0  # sentinel: checked, no data
            continue

        # Sanity check
        ratio = max_price / price_at if price_at > 0 else 0
        max_ratio = MAX_PLAUSIBLE_RATIO.get(hours, 500)
        if ratio > max_ratio:
            logger.warning(
                "Implausible OHLCV for %s/%dh: %.6f -> %.6f (%.0fx) — skipping",
                symbol, hours, price_at, max_price, ratio,
            )
            continue

        # v41: Removed consistency floor. Each horizon computes its own did_2x
        # from its own candles. In memecoins, a 5min 2x spike that crashes by 24h
        # should NOT label 24h as did_2x=True. Realistic labels > mathematical consistency.
        did_2x = max_price >= (price_at * 2.0)

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
        if bot_data and hours in (12, 24, 48):
            hz_suffix = f"_{hours}h"
            update_data[f"time_to_1_3x_min{hz_suffix}"] = bot_data.get("t_1_3x")
            update_data[f"time_to_1_5x_min{hz_suffix}"] = bot_data.get("t_1_5x")
            update_data[f"time_to_2x_min{hz_suffix}"] = bot_data.get("t_2x")
            update_data[f"time_to_3x_min{hz_suffix}"] = bot_data.get("t_3x")
            update_data[f"time_to_5x_min{hz_suffix}"] = bot_data.get("t_5x")
            update_data[f"time_to_sl20_min{hz_suffix}"] = bot_data.get("t_sl20")
            update_data[f"time_to_sl30_min{hz_suffix}"] = bot_data.get("t_sl30")
            update_data[f"time_to_sl50_min{hz_suffix}"] = bot_data.get("t_sl50")
            update_data[f"max_dd_before_tp_pct{hz_suffix}"] = bot_data.get("max_dd_pct")

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


def fill_outcomes() -> None:
    """
    Batch-optimized outcome labeling with token grouping.

    Groups snapshots by token_address and makes 1 OHLCV API call per unique token
    (covering all snapshots' time ranges), then labels all snapshots from shared candles.

    Old approach: 1 API call per snapshot -> ~250 snapshots/run.
    New approach: 1 API call per unique token -> ~2000+ snapshots/run (5-10x faster).
    """
    client = _get_client()
    now = datetime.now(timezone.utc)
    now_ts = now.timestamp()
    pool_cache = _load_pool_cache()
    stats = {"updated": 0, "api_calls": 0, "skipped": 0, "no_price": 0, "consistent": 0, "tokens_processed": 0}
    start_time = time.time()

    # Find snapshots with fillable unlabeled horizons.
    # v29 fix: each horizon is only included when the snapshot is old enough to fill it.
    # Without this, snapshots with only did_2x_7d=NULL (but <7 days old) clog the batch
    # and block newer snapshots from being labeled — the root cause of the labeling backlog.
    # v30: Order by score DESC so high-score tokens get labeled first (most useful for backtesting).
    # v36-fix: Add max_price.is.null to filter — zombies with max_price=0 (sentinel for
    # "checked all OHLCV sources, no data") are excluded. Without this, _mark_ohlcv_failed
    # snapshots re-enter the batch forever, wasting 4,274+ API calls per run.
    or_parts = []
    for hz in HORIZONS:
        cutoff = (now - timedelta(hours=hz["hours"])).strftime("%Y-%m-%dT%H:%M:%SZ")
        or_parts.append(f'and({hz["flag_col"]}.is.null,{hz["max_col"]}.is.null,snapshot_at.lt.{cutoff})')
    filter_str = ",".join(or_parts)

    try:
        result = (
            client.table("token_snapshots")
            .select("id, symbol, price_at_snapshot, snapshot_at, token_address, pair_address, "
                    "max_price_1h, max_price_6h, max_price_12h, max_price_24h, max_price_48h, max_price_72h, max_price_7d, "
                    "did_2x_1h, did_2x_6h, did_2x_12h, did_2x_24h, did_2x_48h, did_2x_72h, did_2x_7d")
            .or_(filter_str)
            # v36-fix: FIFO ordering (oldest first) instead of score-DESC.
            # Score-DESC caused zombies to starve fresh snapshots.
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

    # v34: Group snapshots by token_address — 1 OHLCV fetch per unique token
    token_groups = defaultdict(list)
    for snap in snapshots:
        key = snap.get("token_address") or str(snap["id"])
        token_groups[key].append(snap)

    logger.info(
        "Processing %d snapshots across %d unique tokens (token-grouped)",
        len(snapshots), len(token_groups),
    )

    def _process_token_group(token_key, group, _client, _now_ts, _pool_cache, _stats, _stats_lock):
        """Process one token group — thread-safe. v54."""
        # 1. Pre-filter: handle no-price snapshots immediately (no API needed)
        valid_snaps = []
        for snap in group:
            if not snap.get("price_at_snapshot") or float(snap["price_at_snapshot"]) <= 0:
                _mark_no_price(_client, snap, _stats)
            else:
                valid_snaps.append(snap)
        if not valid_snaps:
            return

        # 2. Resolve pool address ONCE per token
        pool_addr = None
        for snap in valid_snaps:
            if snap.get("pair_address"):
                pool_addr = snap["pair_address"]
                break

        actual_token_addr = valid_snaps[0].get("token_address")
        if not pool_addr and actual_token_addr:
            _gecko_limiter.wait()
            pool_addr = _get_pool_address(actual_token_addr, _pool_cache)
            with _stats_lock:
                _stats["api_calls"] += 1

        if not pool_addr:
            for snap in valid_snaps:
                _mark_dead_pool(_client, snap, _now_ts, _stats)
            return

        # 3. Parse timestamps and determine horizons for each snapshot
        snap_data = []  # [(snap, snapshot_ts, price_at, horizons_to_fill)]
        for snap in valid_snaps:
            snapshot_ts = _parse_snapshot_ts(snap.get("snapshot_at", ""))
            if not snapshot_ts:
                with _stats_lock:
                    _stats["skipped"] += 1
                continue
            price_at = float(snap["price_at_snapshot"])
            age_hours = (_now_ts - snapshot_ts) / 3600
            horizons_to_fill = [hz for hz in HORIZONS if snap.get(hz["flag_col"]) is None and age_hours >= hz["hours"]]
            if horizons_to_fill:
                snap_data.append((snap, snapshot_ts, price_at, horizons_to_fill))

        if not snap_data:
            return

        # 4. Compute the FULL time range covering ALL snapshots + their longest horizon
        earliest_ts = min(sd[1] for sd in snap_data)
        latest_end_ts = max(sd[1] + max(hz["hours"] for hz in sd[3]) * 3600 for sd in snap_data)

        # Use first snap's price_at for sanity check (all same token, close enough)
        ref_price = snap_data[0][2]
        ref_symbol = snap_data[0][0]["symbol"]

        # 5. Fetch OHLCV ONCE for the full range
        sorted_candles, source = _fetch_ohlcv_candles(
            pool_addr, actual_token_addr,
            earliest_ts, latest_end_ts,
            ref_price, ref_symbol, _stats, _pool_cache,
        )

        if sorted_candles is None:
            # Mark as failed if old enough, otherwise skip for retry
            for snap, snapshot_ts, price_at, horizons_to_fill in snap_data:
                age = (_now_ts - snapshot_ts) / 3600
                if age > 36:  # v36-fix: was 48h; 36h is enough — OHLCV APIs rarely have data beyond 24-48h
                    _mark_ohlcv_failed(_client, snap, horizons_to_fill, _stats)
                else:
                    with _stats_lock:
                        _stats["skipped"] += 1
            return

        # 6. Process each snapshot using SHARED candle data
        for snap, snapshot_ts, price_at, horizons_to_fill in snap_data:
            _label_snapshot(_client, snap, snapshot_ts, price_at, horizons_to_fill,
                            sorted_candles, source, _stats)

        with _stats_lock:
            _stats["tokens_processed"] += 1

    # v54: Parallel processing with ThreadPoolExecutor (4 workers, rate-limiter bounded)
    stats_lock = threading.Lock()
    _MAX_WORKERS = 4

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        futures = {}
        for token_key, group in token_groups.items():
            if time.time() - start_time > TIME_BUDGET_SECONDS:
                logger.warning("Time budget exceeded (%.0fs), stopping submission", time.time() - start_time)
                break
            fut = executor.submit(
                _process_token_group, token_key, group, client, now_ts,
                pool_cache, stats, stats_lock,
            )
            futures[fut] = token_key

        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                logger.error("Token group %s failed: %s", futures[fut], e)

    _save_pool_cache(pool_cache)

    elapsed = time.time() - start_time
    throughput = stats["updated"] / max(1, elapsed) * 60  # snapshots/minute
    logger.info(
        "Outcome tracker: updated=%d, tokens=%d, api_calls=%d, skipped=%d, no_price=%d, consistency=%d "
        "(%.0fs elapsed, %.1f snapshots/min)",
        stats["updated"], stats["tokens_processed"], stats["api_calls"],
        stats["skipped"], stats["no_price"], stats["consistent"],
        elapsed, throughput,
    )

    # Second pass: fix existing inconsistencies (skip if <3 min left for other steps)
    remaining = 38 * 60 - (time.time() - start_time)  # 38 min total budget for fill_outcomes
    if remaining > 120:
        _fix_existing_inconsistencies(client)
    else:
        logger.warning("Skipping consistency fix (%.0fs remaining)", remaining)

    # Third pass: fill price_at_first_call (skip if tight on time)
    remaining = 38 * 60 - (time.time() - start_time)
    if remaining > 60:
        _fill_first_call_prices(client, pool_cache)
    else:
        logger.warning("Skipping first_call_prices (%.0fs remaining)", remaining)


def _fix_existing_inconsistencies(client: Client) -> None:
    """
    v41: Only fix implausible values (SOL price leaks). No longer propagates
    shorter-horizon max_price to longer horizons — each horizon stands on its own.
    In memecoins, 1h can spike 2x while 24h doesn't. That's reality, not inconsistency.
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

        updates = {}
        horizon_map = {h["hours"]: h for h in HORIZONS}

        for h in [hz["hours"] for hz in HORIZONS]:
            hz = horizon_map[h]
            val = float(snap[hz["max_col"]]) if snap.get(hz["max_col"]) else None
            if val is not None:
                # Sanity check: reject implausible values (SOL price leak)
                max_ratio = MAX_PLAUSIBLE_RATIO.get(h, 500)
                if price_at > 0 and val / price_at > max_ratio:
                    logger.warning(
                        "Implausible val for %s/%dh: %.6f -> %.6f (%.0fx), nullifying",
                        snap["symbol"], h, price_at, val, val / price_at,
                    )
                    updates[hz["max_col"]] = None
                    updates[hz["price_col"]] = None
                    updates[hz["flag_col"]] = None

        if updates:
            try:
                client.table("token_snapshots").update(updates).eq("id", snap["id"]).execute()
                fixed += 1
            except Exception as e:
                logger.warning("Failed to fix inconsistency for %s: %s", snap["symbol"], e)

    if fixed > 0:
        logger.info("Fixed %d snapshots with implausible values", fixed)


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

    for hours in [12, 24, 48]:
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
                f"time_to_2x_min{hz}": bot_data.get("t_2x"),
                f"time_to_3x_min{hz}": bot_data.get("t_3x"),
                f"time_to_5x_min{hz}": bot_data.get("t_5x"),
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


# =============================================================================
# KOL Call Outcomes — accurate winrate tracking per KOL
# =============================================================================
# Tracks each KOL's FIRST call per token with:
#   - Exact entry price (OHLCV close at call_timestamp, not delayed scraper price)
#   - ATH after call (continuously updated, no fixed time window)
#   - did_2x = TRUE if token EVER doubled from entry price
# =============================================================================

KCO_BATCH_LIMIT = 500      # v36: was 200, batch-per-token means fewer API calls
KCO_TIME_BUDGET = 20 * 60  # v36: 20 minutes (was 8). DexPaprika-first + batch = much faster


def fill_kol_outcomes() -> None:
    """
    Three-phase pipeline for KOL call outcome tracking.

    Phase A: Sync new (kol_group, token_address) pairs from kol_mentions → kol_call_outcomes.
    Phase B: Fill entry_price for rows missing it (OHLCV at exact call_timestamp).
    Phase C: Update ath_after_call for rows that have entry_price.
    """
    client = _get_client()
    pool_cache = _load_pool_cache()
    start_time = time.time()
    stats = {"synced": 0, "entry_filled": 0, "ath_updated": 0, "skipped": 0, "api_calls": 0}

    # --- Phase A: Sync mentions → kol_call_outcomes ---
    _kco_phase_a_sync(client, stats)

    # --- Phase B: Fill entry prices ---
    if time.time() - start_time < KCO_TIME_BUDGET:
        _kco_phase_b_entry_prices(client, pool_cache, stats, start_time)

    # --- Phase C: Update ATH ---
    if time.time() - start_time < KCO_TIME_BUDGET:
        _kco_phase_c_update_ath(client, pool_cache, stats, start_time)

    _save_pool_cache(pool_cache)

    elapsed = time.time() - start_time
    logger.info(
        "fill_kol_outcomes: synced=%d, entry_filled=%d, ath_updated=%d, skipped=%d, api_calls=%d (%.0fs)",
        stats["synced"], stats["entry_filled"], stats["ath_updated"],
        stats["skipped"], stats["api_calls"], elapsed,
    )


def _kco_paginate_query(client: Client, table: str, select: str, order_col: str = None,
                         filters: list[tuple] = None, page_size: int = 1000) -> list:
    """Paginate a supabase-py query to avoid the default 1000-row limit."""
    all_rows = []
    offset = 0
    while True:
        q = client.table(table).select(select)
        if filters:
            for method, args in filters:
                if method == "not_is_null":
                    q = q.not_.is_(args, "null")
        if order_col:
            q = q.order(order_col, desc=False)
        q = q.range(offset, offset + page_size - 1)
        result = q.execute()
        rows = result.data or []
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        offset += page_size
    return all_rows


def _kco_phase_a_sync(client: Client, stats: dict) -> None:
    """
    Sync first call per (kol_group, token_address) from kol_mentions into kol_call_outcomes.
    Paginates kol_mentions to avoid supabase-py's 1000-row default limit.
    """
    logger.info("KCO Phase A: syncing mentions → kol_call_outcomes")

    # Step 1: Get all tokens with addresses (symbol → token_address mapping)
    try:
        tokens_result = (
            client.table("tokens")
            .select("symbol, token_address")
            .not_.is_("token_address", "null")
            .execute()
        )
    except Exception as e:
        logger.error("KCO Phase A: failed to query tokens: %s", e)
        return

    token_map = {}  # UPPER(symbol) → token_address
    token_addrs = set()
    for t in (tokens_result.data or []):
        sym = (t.get("symbol") or "").upper().strip()
        addr = (t.get("token_address") or "").strip()
        if sym and addr:
            token_map[sym] = addr
            token_addrs.add(addr)

    if not token_map:
        logger.warning("KCO Phase A: no tokens with addresses found")
        return

    # Step 1b: Build token_address → pair_address lookup from token_snapshots
    # v34: Without pair_address, Phase B can't fetch OHLCV → entry_price never fills
    pair_map = {}  # token_address → pair_address
    try:
        pair_result = (
            client.table("token_snapshots")
            .select("token_address, pair_address")
            .not_.is_("pair_address", "null")
            .not_.is_("token_address", "null")
            .order("snapshot_at", desc=True)
            .limit(1000)
            .execute()
        )
        for row in (pair_result.data or []):
            ta = row.get("token_address")
            pa = row.get("pair_address")
            if ta and pa and ta not in pair_map:
                pair_map[ta] = pa
        logger.info("KCO Phase A: loaded %d pair_address mappings", len(pair_map))
    except Exception as e:
        logger.warning("KCO Phase A: failed to load pair_addresses: %s", e)

    # Step 2: Get ALL kol_mentions (paginated)
    try:
        mentions = _kco_paginate_query(
            client, "kol_mentions",
            "id, symbol, kol_group, message_date, extracted_cas, resolved_ca",
            order_col="message_date",
        )
    except Exception as e:
        logger.error("KCO Phase A: failed to query kol_mentions: %s", e)
        return

    if not mentions:
        logger.info("KCO Phase A: no mentions found")
        return

    # Step 3: Find first mention per (kol_group, token_address)
    first_calls = {}
    for m in mentions:
        sym = (m.get("symbol") or "").upper().strip()
        kol = m.get("kol_group") or ""
        msg_date = m.get("message_date")
        mention_id = m.get("id")

        if not sym or not kol or not msg_date:
            continue

        # v40: Prefer resolved_ca (exact CA from KOL's message), then symbol match, then extracted_cas
        token_addr = None
        resolved_ca = (m.get("resolved_ca") or "").strip()
        if resolved_ca:
            token_addr = resolved_ca
        if not token_addr:
            token_addr = token_map.get(sym)
        if not token_addr:
            cas = m.get("extracted_cas") or []
            for ca in cas:
                ca_clean = (ca or "").strip()
                if ca_clean in token_addrs:
                    token_addr = ca_clean
                    break
        if not token_addr:
            continue

        key = (kol, token_addr)
        if key not in first_calls:
            first_calls[key] = {
                "mention_id": mention_id,
                "call_timestamp": msg_date,
                "symbol": sym,
                "token_address": token_addr,
                "kol_group": kol,
            }
        # Already sorted by message_date ASC, so first occurrence wins

    logger.info("KCO Phase A: found %d unique (kol, token) pairs from %d mentions",
                len(first_calls), len(mentions))

    # Step 4: Get existing kol_call_outcomes to skip duplicates
    try:
        existing = _kco_paginate_query(
            client, "kol_call_outcomes", "kol_group, token_address",
        )
    except Exception as e:
        logger.error("KCO Phase A: failed to query existing outcomes: %s", e)
        return

    existing_keys = {(r["kol_group"], r["token_address"]) for r in existing}

    # Step 5: Insert new rows (with pair_address from snapshots)
    new_rows = []
    for key, call in first_calls.items():
        if key in existing_keys:
            continue
        row = {
            "mention_id": call["mention_id"],
            "token_address": call["token_address"],
            "symbol": call["symbol"],
            "kol_group": call["kol_group"],
            "call_timestamp": call["call_timestamp"],
        }
        pa = pair_map.get(call["token_address"])
        if pa:
            row["pair_address"] = pa
        new_rows.append(row)

    if not new_rows:
        logger.info("KCO Phase A: all pairs already synced")
        return

    # Batch insert (upsert with ON CONFLICT skip on mention_id)
    batch_size = 100
    for i in range(0, len(new_rows), batch_size):
        batch = new_rows[i:i + batch_size]
        try:
            client.table("kol_call_outcomes").upsert(
                batch, on_conflict="mention_id"
            ).execute()
            stats["synced"] += len(batch)
        except Exception as e:
            logger.error("KCO Phase A: insert batch failed: %s", e)

    logger.info("KCO Phase A: inserted %d new call outcomes", stats["synced"])


def _fetch_ohlcv_candles_kco(
    pool_addr: str | None,
    token_addr: str,
    start_ts: float,
    end_ts: float,
    symbol: str,
    stats: dict,
    ref_price: float = 0.0,
    pool_cache: dict | None = None,
) -> tuple[list | None, str]:
    """
    Fetch OHLCV candles for KCO Phase B/C. Order: DexPaprika → Birdeye → GeckoTerminal.

    v36: DexPaprika is primary because GeckoTerminal rate limit is usually exhausted
    by fill_outcomes which runs first in the workflow. Birdeye uses token MINT address
    (no pool resolution needed) and recovers deindexed tokens.
    v57: Skip DexPaprika for pools with SOL as base token (returns SOL price ~$85).
    """
    sorted_candles = None
    source = "none"
    window_seconds = end_ts - start_ts

    # v57: Skip DexPaprika if pool has SOL as base token
    sol_base = pool_addr and pool_cache is not None and _is_sol_base_pool(pool_addr, pool_cache)

    # 1. DexPaprika first (pool-based, 15min candles) — skip for inverted pools (v57)
    if pool_addr and not sol_base:
        start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)
        num_candles_15m = int(window_seconds / 900) + 10
        try:
            resp = requests.get(
                f"{DEXPAPRIKA_BASE}/networks/solana/pools/{pool_addr}/ohlcv",
                params={
                    "start": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "end": end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "limit": min(1000, num_candles_15m),
                    "interval": "15m",
                },
                timeout=15,
            )
            time.sleep(1.0)  # v37: 0.3→1.0s — DexPaprika rate-limits aggressively
            stats["api_calls"] += 1

            if resp.status_code == 200:
                candles_raw = resp.json()
                if isinstance(candles_raw, list) and candles_raw:
                    sorted_candles = []
                    for c in candles_raw:
                        try:
                            # v36-fix: DexPaprika uses time_open/time_close, NOT time/timestamp
                            ts_str = c.get("time_open") or c.get("time_close") or c.get("time") or c.get("timestamp", "")
                            if not ts_str:
                                continue
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
                else:
                    logger.debug("KCO DexPaprika empty candles for %s", symbol)
            elif resp.status_code == 429:
                logger.warning("KCO DexPaprika rate limited for %s (429)", symbol)
            else:
                logger.debug("KCO DexPaprika %d for %s", resp.status_code, symbol)
        except requests.RequestException as e:
            logger.debug("KCO DexPaprika OHLCV failed for %s: %s", symbol, e)

    # v39: reject SOL quote price leak (same as _fetch_ohlcv_candles)
    if sorted_candles and ref_price > 0 and not _check_candle_sanity(sorted_candles, ref_price, symbol, source):
        sorted_candles = None

    # 2. Birdeye fallback (uses token MINT address — no pool needed, recovers deindexed tokens)
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
                        "time_from": int(start_ts),
                        "time_to": int(end_ts),
                    },
                    headers={"X-API-KEY": birdeye_key, "x-chain": "solana"},
                    timeout=15,
                )
                stats["api_calls"] += 1

                if resp.status_code == 200:
                    items = resp.json().get("data", {}).get("items", [])
                    if items:
                        sorted_candles = [
                            [int(c.get("unixTime", 0)), float(c.get("o", 0)),
                             float(c.get("h", 0)), float(c.get("l", 0)),
                             float(c.get("c", 0)), float(c.get("v", 0))]
                            for c in items
                        ]
                        sorted_candles.sort(key=lambda x: x[0])
                        if sorted_candles:
                            source = "birdeye_ohlcv"
                        else:
                            sorted_candles = None
                elif resp.status_code == 429:
                    logger.warning("KCO Birdeye rate limited for %s", symbol)
            except requests.RequestException as e:
                logger.debug("KCO Birdeye OHLCV failed for %s: %s", symbol, e)

    # v39: reject SOL quote price leak after Birdeye
    if sorted_candles and ref_price > 0 and not _check_candle_sanity(sorted_candles, ref_price, symbol, source):
        sorted_candles = None

    # 3. GeckoTerminal last (usually exhausted by main fill_outcomes)
    global _gecko_consecutive_429s, _gecko_disabled
    with _gecko_lock:
        gecko_ok = not _gecko_disabled
    if sorted_candles is None and pool_addr and gecko_ok:
        num_candles_5m = int(window_seconds / 300) + 10
        try:
            _gecko_limiter.wait()
            resp = requests.get(
                GECKOTERMINAL_OHLCV_URL.format(pool=pool_addr),
                params={
                    "aggregate": 5,
                    "before_timestamp": int(end_ts),
                    "limit": min(1000, num_candles_5m),
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
                with _gecko_lock:
                    _gecko_consecutive_429s = 0
            elif resp.status_code == 429:
                with _gecko_lock:
                    _gecko_consecutive_429s += 1
                    if _gecko_consecutive_429s >= _GECKO_429_THRESHOLD:
                        _gecko_disabled = True
                        logger.warning("GeckoTerminal disabled (KCO Phase B)")
        except requests.RequestException as e:
            logger.debug("KCO GeckoTerminal OHLCV failed for %s: %s", symbol, e)

    # v39: final sanity check on GeckoTerminal candles too
    if sorted_candles and ref_price > 0 and not _check_candle_sanity(sorted_candles, ref_price, symbol, source):
        sorted_candles = None

    return sorted_candles, source


def _kco_phase_b_entry_prices(client: Client, pool_cache: dict, stats: dict, start_time: float) -> None:
    """Fill entry_price for rows where it's NULL using OHLCV at exact call_timestamp.

    v36 optimization: Batch by token — fetch OHLCV once per unique token covering all KOL
    call timestamps, then extract individual entry prices from the same candle data.
    Uses DexPaprika as primary (10K/day), Birdeye by mint as fallback (deindexed tokens).
    Previously: 1 API call per row (588 calls). Now: 1 API call per token (~296 calls).
    """
    logger.info("KCO Phase B: filling entry prices")

    try:
        result = (
            client.table("kol_call_outcomes")
            .select("id, token_address, symbol, call_timestamp, pair_address")
            .is_("entry_price", "null")
            .is_("outcome_status", "null")          # v42: skip already-dead rows
            .order("call_timestamp", desc=False)     # v42: FIFO — oldest first
            .limit(KCO_BATCH_LIMIT)
            .execute()
        )
    except Exception as e:
        logger.error("KCO Phase B: query failed: %s", e)
        return

    rows = result.data or []
    if not rows:
        logger.info("KCO Phase B: no rows need entry prices")
        return

    # Group by token_address — 1 OHLCV fetch covers all KOL calls for the same token
    token_groups = defaultdict(list)
    for row in rows:
        token_groups[row["token_address"]].append(row)

    logger.info("KCO Phase B: %d rows across %d unique tokens", len(rows), len(token_groups))

    filled = 0
    for token_addr, group in token_groups.items():
        if time.time() - start_time > KCO_TIME_BUDGET:
            logger.warning("KCO Phase B: time budget exceeded after %d tokens", filled)
            break

        # Resolve pool address from any row in the group
        pool_addr = None
        for row in group:
            if row.get("pair_address"):
                pool_addr = row["pair_address"]
                break
        if not pool_addr:
            pool_addr = _get_pool_address(token_addr, pool_cache)
            stats["api_calls"] += 1

        # Parse all call timestamps for this token
        call_entries = []
        for row in group:
            call_ts = _parse_snapshot_ts(row["call_timestamp"])
            if call_ts:
                call_entries.append((row, call_ts))
            else:
                stats["skipped"] += 1

        if not call_entries:
            continue

        # Compute time window covering ALL KOL calls for this token
        min_ts = min(ts for _, ts in call_entries)
        max_ts = max(ts for _, ts in call_entries)
        window_start = min_ts - 30 * 60   # 30min before earliest call
        window_end = max_ts + 30 * 60     # 30min after latest call

        # Single OHLCV fetch for all KOL calls on this token
        candles, source = _fetch_ohlcv_candles_kco(
            pool_addr, token_addr, window_start, window_end,
            group[0]["symbol"], stats, pool_cache=pool_cache,
        )

        if not candles:
            # v42: Mark old calls as dead instead of retrying forever.
            # Calls older than 5 days with no OHLCV data will never resolve.
            age_cutoff = time.time() - 5 * 24 * 3600
            old_ids = [row["id"] for row, ts in call_entries if ts < age_cutoff]
            if old_ids:
                try:
                    client.table("kol_call_outcomes").update({
                        "outcome_status": "dead_no_ohlcv",
                        "last_checked_at": datetime.now(timezone.utc).isoformat(),
                    }).in_("id", old_ids).execute()
                    stats["dead_marked"] = stats.get("dead_marked", 0) + len(old_ids)
                    logger.info("KCO Phase B: marked %d old calls dead for %s (no OHLCV after 5d)",
                                len(old_ids), group[0]["symbol"])
                except Exception as e:
                    logger.error("KCO Phase B: failed to mark dead for %s: %s", token_addr, e)
            stats["skipped"] += len(call_entries) - len(old_ids)
            continue

        # Extract entry price for each KOL call from the shared candle data
        for row, call_ts in call_entries:
            best_candle = min(candles, key=lambda c: abs(c[0] - call_ts))
            entry_price = best_candle[4]  # close price

            if entry_price <= 0:
                stats["skipped"] += 1
                continue

            # v39: SOL price leak filter — SOL trades at ~$80-200.
            # Memecoins are < $1, but large-caps like $TRUMP/$HYPE can be $1-$50.
            # Threshold at $50 catches SOL leaks while allowing legitimate large-caps.
            if entry_price > 50.0:
                logger.warning("KCO Phase B: SOL price leak for %s — entry=%.2f (%s), skipping",
                               row.get("symbol", "?"), entry_price, source)
                stats["skipped"] += 1
                continue

            update_data = {
                "entry_price": float(entry_price),
                "price_source": source,
                "last_checked_at": datetime.now(timezone.utc).isoformat(),
            }
            if not row.get("pair_address") and pool_addr:
                update_data["pair_address"] = pool_addr

            try:
                client.table("kol_call_outcomes").update(update_data).eq("id", row["id"]).execute()
                stats["entry_filled"] += 1
                logger.debug("KCO Phase B: %s [%s] entry=%.10f (%s)",
                             row["symbol"], row.get("kol_group", "?")[:12], entry_price, source)
            except Exception as e:
                logger.error("KCO Phase B: update failed for id=%d: %s", row["id"], e)

        filled += 1


def _kco_phase_c_update_ath(client: Client, pool_cache: dict, stats: dict, start_time: float) -> None:
    """Update ath_after_call for rows that have entry_price. Fetches OHLCV from call to now."""
    logger.info("KCO Phase C: updating ATH")

    try:
        result = (
            client.table("kol_call_outcomes")
            .select("id, token_address, symbol, call_timestamp, pair_address, entry_price, ath_after_call")
            .not_.is_("entry_price", "null")
            # v51: include dead_no_ohlcv — Phase B may have found entry_price
            # but Phase C never ran. Give them a chance to compute ATH.
            .or_("outcome_status.is.null,outcome_status.eq.active,outcome_status.eq.dead_no_ohlcv")
            .order("last_checked_at", desc=False, nullsfirst=True)
            .limit(KCO_BATCH_LIMIT)
            .execute()
        )
    except Exception as e:
        logger.error("KCO Phase C: query failed: %s", e)
        return

    rows = result.data or []
    if not rows:
        logger.info("KCO Phase C: no rows to update ATH")
        return

    logger.info("KCO Phase C: %d rows to check ATH", len(rows))

    # Group by token_address — same token's candles cover multiple KOL calls
    token_groups = defaultdict(list)
    for row in rows:
        token_groups[row["token_address"]].append(row)

    now_ts = time.time()

    for token_addr, group in token_groups.items():
        if time.time() - start_time > KCO_TIME_BUDGET:
            logger.warning("KCO Phase C: time budget exceeded")
            break

        # Resolve pool address
        pool_addr = None
        for row in group:
            if row.get("pair_address"):
                pool_addr = row["pair_address"]
                break
        if not pool_addr:
            _gecko_limiter.wait()
            pool_addr = _get_pool_address(token_addr, pool_cache)
            stats["api_calls"] += 1

        if not pool_addr:
            # v37: Mark old calls as dead if no pool can be found
            # v39: Use outcome_status instead of fake ath sentinel
            for row in group:
                ts = _parse_snapshot_ts(row["call_timestamp"])
                if ts and (now_ts - ts) / 3600 > 72:
                    try:
                        client.table("kol_call_outcomes").update({
                            "outcome_status": "dead_no_ohlcv",
                            "last_checked_at": datetime.now(timezone.utc).isoformat(),
                        }).eq("id", row["id"]).execute()
                    except Exception:
                        pass
            stats["skipped"] += len(group)
            continue

        # Find the earliest call_timestamp in this group to cover all calls
        call_timestamps = []
        for row in group:
            ts = _parse_snapshot_ts(row["call_timestamp"])
            if ts:
                call_timestamps.append((row, ts))

        if not call_timestamps:
            continue

        earliest_ts = min(ts for _, ts in call_timestamps)
        ref_price = float(call_timestamps[0][0]["entry_price"])
        ref_symbol = group[0]["symbol"]

        # v36: Use DexPaprika-first (GeckoTerminal usually exhausted by fill_outcomes)
        # v57: Skip DexPaprika for inverted pools (SOL as base token)
        candles, source = _fetch_ohlcv_candles_kco(
            pool_addr, token_addr,
            earliest_ts, now_ts,
            ref_symbol, stats,
            ref_price=ref_price, pool_cache=pool_cache,
        )

        if not candles:
            # v37-fix: DON'T update last_checked_at when candles fail —
            # let the row retry next cycle. Only mark as dead if call is old enough.
            for row, call_ts_val in call_timestamps:
                call_age_h = (now_ts - call_ts_val) / 3600
                if call_age_h > 72:
                    # Token is 3+ days old with no OHLCV data — mark as dead
                    # v39: Use outcome_status instead of fake ath sentinel
                    try:
                        client.table("kol_call_outcomes").update({
                            "outcome_status": "dead_no_ohlcv",
                            "last_checked_at": datetime.now(timezone.utc).isoformat(),
                        }).eq("id", row["id"]).execute()
                        stats["ath_updated"] += 1
                        logger.debug("KCO Phase C: %s marked dead (no candles, %.0fh old)",
                                     row["symbol"], call_age_h)
                    except Exception:
                        pass
                else:
                    # Fresh call — skip without updating last_checked_at (will retry)
                    stats["skipped"] += 1
                    logger.debug("KCO Phase C: %s no candles yet (%.0fh old, will retry)",
                                 row["symbol"], call_age_h)
            continue

        # For each row, find max high AFTER its call_timestamp
        for row, call_ts in call_timestamps:
            current_ath = float(row.get("ath_after_call") or 0)

            # Filter candles to only those AFTER this row's call_timestamp
            max_high = 0.0
            ath_ts = None
            for candle in candles:
                if candle[0] >= call_ts:  # candle timestamp >= call time
                    if candle[2] > max_high:  # candle high
                        max_high = candle[2]
                        ath_ts = candle[0]

            # v40: SOL price leak filter — same as Phase B ($50 threshold)
            # Memecoins are < $1, large-caps like $TRUMP/$HYPE can be $1-$50
            entry = float(row["entry_price"])
            if max_high > 50.0 and entry < 1.0:
                logger.warning("KCO Phase C: SOL price leak for %s — entry=%.8f, ath=%.2f (%s), skipping",
                               row["symbol"], entry, max_high, source)
                stats["skipped"] += 1
                continue

            # v39: cross-validate denomination — reject insane ratios
            if max_high > 0 and entry > 0:
                ratio = max_high / entry
                if ratio > 10000 or ratio < 0.0001:
                    logger.warning("KCO Phase C: denomination mismatch %s — entry=%.10f, ath=%.10f (%.0fx), skipping",
                                   row["symbol"], entry, max_high, ratio)
                    stats["skipped"] += 1
                    continue

            if max_high <= 0:
                # v37-fix: same retry logic — don't mark checked if no data
                call_age_h = (now_ts - call_ts) / 3600
                if call_age_h > 72:
                    try:
                        client.table("kol_call_outcomes").update({
                            "outcome_status": "dead_no_ohlcv",
                            "last_checked_at": datetime.now(timezone.utc).isoformat(),
                        }).eq("id", row["id"]).execute()
                        stats["ath_updated"] += 1
                    except Exception:
                        pass
                continue

            # Update ATH and max_return (did_2x is a GENERATED column — never set it!)
            update_data = {"last_checked_at": datetime.now(timezone.utc).isoformat()}
            entry = float(row["entry_price"])

            if max_high > current_ath:
                update_data["ath_after_call"] = float(max_high)
                update_data["ath_price_source"] = source  # v39: track ATH data source
                # v51: resurrect dead rows — Phase C found ATH for previously-dead rows
                update_data["outcome_status"] = "active"
                if ath_ts:
                    update_data["ath_timestamp"] = datetime.fromtimestamp(ath_ts, tz=timezone.utc).isoformat()
                if not row.get("pair_address") and pool_addr:
                    update_data["pair_address"] = pool_addr

            # max_return + did_2x are GENERATED ALWAYS columns — auto-computed from ath_after_call / entry_price
            best_ath = max(max_high, current_ath)
            max_return = best_ath / entry if entry > 0 else 0

            try:
                client.table("kol_call_outcomes").update(update_data).eq("id", row["id"]).execute()
                if max_high > current_ath:
                    stats["ath_updated"] += 1
                    logger.debug("KCO Phase C: %s ATH=%.10f (%.1fx entry)",
                                 row["symbol"], max_high, max_return if entry > 0 else 0)
            except Exception as e:
                logger.error("KCO Phase C: update failed for id=%d: %s", row["id"], e)
