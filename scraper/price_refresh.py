"""
Price Micro-Refresh: update price/volume + recalculate multipliers every 5 minutes.

Between 30-minute full scrape cycles, this module:
1. Fetches top N tokens from Supabase (by score DESC)
2. For each: fetches fresh DexScreener data (price, volume, txns)
3. Recalculates on-chain multiplier + price_action_mult + already_pumped_penalty
4. Updates the score in Supabase (base_score * new_combined_multiplier)
5. Bumps scrape_metadata.updated_at to trigger frontend refresh

The base Telegram score (social signals) does NOT change — only market
multipliers are recalculated with fresh price data.
"""

import os
import math
import time
import logging
from datetime import datetime, timezone

import requests
from supabase import create_client

from pipeline import SCORING_PARAMS

logger = logging.getLogger(__name__)

# v67: Monitoring — conditional import
try:
    from monitor import estimate_egress as _estimate_egress
    _monitoring = True
except ImportError:
    _monitoring = False

def _two_phase_decay(hours_ago: float) -> float:
    """
    Single-phase exponential decay (mirrors pipeline._two_phase_decay).
    v20: lambda from SCORING_PARAMS["decay_lambda"] (shared with pipeline).
    Default lambda=0.12, half-life ~5.8h.
    """
    return math.exp(-SCORING_PARAMS["decay_lambda"] * hours_ago)


DEXSCREENER_TOKEN_URL = "https://api.dexscreener.com/tokens/v1/solana/{address}"
DEXSCREENER_BATCH_URL = "https://api.dexscreener.com/tokens/v1/solana/{addresses}"
BATCH_SIZE = 30  # DexScreener max per batch call
REFRESH_TOP_N = 20
REFRESH_INTERVAL_SECONDS = 3 * 60  # 3 minutes


def _get_supabase():
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return create_client(url, key)


def _safe_float(val, default=None) -> float | None:
    if val is None:
        return default
    try:
        result = float(val)
        return result if result == result else default
    except (ValueError, TypeError):
        return default


def _fetch_dexscreener_batch(addresses: list[str]) -> dict[str, dict]:
    """
    Batch fetch market data for multiple tokens in 1 API call.
    DexScreener supports up to 30 comma-separated addresses.
    Returns { address: market_data_dict } for each address found.
    """
    if not addresses:
        return {}

    result = {}
    # Split into chunks of BATCH_SIZE
    for i in range(0, len(addresses), BATCH_SIZE):
        chunk = addresses[i:i + BATCH_SIZE]
        addr_str = ",".join(chunk)
        try:
            resp = requests.get(
                DEXSCREENER_BATCH_URL.format(addresses=addr_str),
                timeout=15,
            )
            if resp.status_code != 200:
                logger.warning("DexScreener batch %d for %d tokens", resp.status_code, len(chunk))
                continue

            pairs = resp.json() if isinstance(resp.json(), list) else resp.json().get("pairs", [])
            if not isinstance(pairs, list):
                continue

            # Group pairs by base token address, pick highest-volume pair per token
            by_address: dict[str, list] = {}
            for p in pairs:
                addr = p.get("baseToken", {}).get("address", "")
                if addr:
                    by_address.setdefault(addr, []).append(p)

            for addr, token_pairs in by_address.items():
                best = max(token_pairs, key=lambda p: float(p.get("volume", {}).get("h24", 0) or 0))
                price_changes = best.get("priceChange", {})
                volumes = best.get("volume", {})
                txns_h1 = best.get("txns", {}).get("h1", {})
                txns_m5 = best.get("txns", {}).get("m5", {})

                result[addr] = {
                    "price_usd": _safe_float(best.get("priceUsd"), 0),
                    "price_change_5m": _safe_float(price_changes.get("m5")),
                    "price_change_1h": _safe_float(price_changes.get("h1")),
                    "price_change_6h": _safe_float(price_changes.get("h6")),
                    "price_change_24h": _safe_float(price_changes.get("h24")),
                    "volume_24h": _safe_float(volumes.get("h24"), 0),
                    "volume_6h": _safe_float(volumes.get("h6"), 0),
                    "volume_1h": _safe_float(volumes.get("h1"), 0),
                    "volume_5m": _safe_float(volumes.get("m5"), 0),
                    "liquidity_usd": _safe_float(best.get("liquidity", {}).get("usd"), 0),
                    "market_cap": _safe_float(best.get("marketCap"), 0) or _safe_float(best.get("fdv"), 0),
                    "buy_sell_ratio_1h": int(txns_h1.get("buys", 0) or 0) / max(1, int(txns_h1.get("buys", 0) or 0) + int(txns_h1.get("sells", 0) or 0)),
                    "buy_sell_ratio_5m": int(txns_m5.get("buys", 0) or 0) / max(1, int(txns_m5.get("buys", 0) or 0) + int(txns_m5.get("sells", 0) or 0)),
                }

        except requests.RequestException as e:
            logger.warning("DexScreener batch error: %s", e)

    return result


def _fetch_dexscreener_by_address(address: str) -> dict | None:
    """Fetch fresh market data for a token by its Solana address."""
    try:
        resp = requests.get(
            DEXSCREENER_TOKEN_URL.format(address=address),
            timeout=10,
        )
        if resp.status_code != 200:
            return None

        pairs = resp.json() if isinstance(resp.json(), list) else resp.json().get("pairs", [])
        if not isinstance(pairs, list) or not pairs:
            return None

        best = max(pairs, key=lambda p: float(p.get("volume", {}).get("h24", 0) or 0))

        price_changes = best.get("priceChange", {})
        volumes = best.get("volume", {})
        txns_h1 = best.get("txns", {}).get("h1", {})
        txns_m5 = best.get("txns", {}).get("m5", {})

        return {
            "price_usd": _safe_float(best.get("priceUsd"), 0),
            "price_change_5m": _safe_float(price_changes.get("m5")),
            "price_change_1h": _safe_float(price_changes.get("h1")),
            "price_change_6h": _safe_float(price_changes.get("h6")),
            "price_change_24h": _safe_float(price_changes.get("h24")),
            "volume_24h": _safe_float(volumes.get("h24"), 0),
            "volume_6h": _safe_float(volumes.get("h6"), 0),
            "volume_1h": _safe_float(volumes.get("h1"), 0),
            "volume_5m": _safe_float(volumes.get("m5"), 0),
            "liquidity_usd": _safe_float(best.get("liquidity", {}).get("usd"), 0),
            "market_cap": _safe_float(best.get("marketCap"), 0) or _safe_float(best.get("fdv"), 0),
            "buy_sell_ratio_1h": int(txns_h1.get("buys", 0) or 0) / max(1, int(txns_h1.get("buys", 0) or 0) + int(txns_h1.get("sells", 0) or 0)),
            "buy_sell_ratio_5m": int(txns_m5.get("buys", 0) or 0) / max(1, int(txns_m5.get("buys", 0) or 0) + int(txns_m5.get("sells", 0) or 0)),
        }

    except requests.RequestException as e:
        logger.debug("DexScreener refresh failed for %s: %s", address[:8], e)
        return None


def _compute_refresh_multiplier(market_data: dict, stored_lifecycle_pen: float = 1.0) -> float:
    """
    Market-condition multiplier using fresh DexScreener data.
    Does NOT apply pa_mult (price_action already at 30% in base_score).
    Applies: death detection + already-pumped + buy/sell pressure + mcap penalty.
    Returns combined multiplier [0.2, 1.5].

    v17: Uses stored lifecycle penalty from last full pipeline cycle as baseline
    for already_pumped. Only overrides with fresh price-based penalty if it's
    WORSE (lower). This preserves the lifecycle model's context-aware detection
    (euphoria, boom, profit_taking) between full cycles.
    """
    pc24 = market_data.get("price_change_24h")

    # v9: Death/rug detection (mirrors pipeline._detect_death_penalty logic)
    # v20: thresholds from SCORING_PARAMS (shared single source of truth)
    death_severe = SCORING_PARAMS["death_pc24_severe"]    # default -80
    death_moderate = SCORING_PARAMS["death_pc24_moderate"]  # default -50
    death_pen = 1.0
    if pc24 is not None:
        if pc24 < death_severe:
            death_pen = 0.1
        elif pc24 < (death_severe + 10):  # -70 at default
            death_pen = 0.2
        elif pc24 < death_moderate:
            death_pen = 0.4
        elif pc24 < -30:
            death_pen = 0.7

    # v17: Already pumped — use stored lifecycle penalty from pipeline as base.
    # Only compute fresh price-based penalty and take the WORSE of the two.
    # This preserves euphoria/boom/profit_taking context from full cycle.
    fresh_pump_pen = 1.0
    if pc24 is not None and pc24 > 100:
        if pc24 > 700:
            fresh_pump_pen = 0.2
        elif pc24 > 400:
            fresh_pump_pen = 0.35
        elif pc24 > 200:
            fresh_pump_pen = 0.6
        else:
            fresh_pump_pen = 0.85
    already_pumped = min(stored_lifecycle_pen, fresh_pump_pen)

    # Buy/sell ratio pressure
    bsr_mult = 1.0
    bsr = market_data.get("buy_sell_ratio_1h")
    if bsr is not None:
        if bsr > 0.7:
            bsr_mult = 1.1
        elif bsr < 0.3:
            bsr_mult = 0.8

    # v17: Pump momentum penalty — penalize tokens actively pumping RIGHT NOW
    # v20: thresholds from SCORING_PARAMS (shared with pipeline)
    pump_1h_hard = SCORING_PARAMS["pump_pc1h_hard"]  # default 30
    pump_5m_hard = SCORING_PARAMS["pump_pc5m_hard"]  # default 15
    pump_1h_mod = pump_1h_hard * 0.5   # 15 at default
    pump_5m_mod = pump_5m_hard * 0.533  # ~8 at default
    pump_1h_light = pump_1h_hard * 0.267  # ~8 at default
    pc_1h = market_data.get("price_change_1h")
    pc_5m = market_data.get("price_change_5m")
    pump_momentum_pen = 1.0
    if (pc_1h is not None and pc_1h > pump_1h_hard) or (pc_5m is not None and pc_5m > pump_5m_hard):
        pump_momentum_pen = 0.5
    elif (pc_1h is not None and pc_1h > pump_1h_mod) or (pc_5m is not None and pc_5m > pump_5m_mod):
        pump_momentum_pen = 0.7
    elif pc_1h is not None and pc_1h > pump_1h_light:
        pump_momentum_pen = 0.85

    # v19: Market cap size penalty — large caps can't 2x easily.
    # Mirrors pipeline.py size_mult thresholds (without freshness boost).
    t_mcap = market_data.get("market_cap") or 0
    if t_mcap <= 0:
        mcap_pen = 1.0
    elif t_mcap < 5_000_000:
        mcap_pen = 1.0   # small/mid cap — no penalty
    elif t_mcap < 20_000_000:
        mcap_pen = 0.85
    elif t_mcap < 50_000_000:
        mcap_pen = 0.70
    elif t_mcap < 200_000_000:
        mcap_pen = 0.50
    elif t_mcap < 500_000_000:
        mcap_pen = 0.35
    else:
        mcap_pen = 0.25

    combined = death_pen * already_pumped * bsr_mult * pump_momentum_pen * mcap_pen
    return max(0.2, min(1.5, combined))


def refresh_top_tokens(n: int = REFRESH_TOP_N) -> int:
    """
    Mini-cycle: fetch fresh DexScreener data for top N tokens,
    recalculate market multipliers, update scores in Supabase.

    Returns number of tokens updated.
    """
    client = _get_supabase()

    # 1. Fetch top N tokens from Supabase (7d window, by score DESC)
    #    v21: token_address now on tokens table — build address_map directly
    result = (
        client.table("tokens")
        .select("symbol, score, base_score, base_score_conviction, base_score_momentum, change_24h, freshest_mention_hours, token_address")
        .eq("time_window", "7d")
        .order("score", desc=True)
        .limit(n)
        .execute()
    )

    tokens = result.data or []
    if _monitoring:
        _estimate_egress("price_refresh", "tokens", len(tokens))
    if not tokens:
        logger.info("Price refresh: no tokens to update")
        return 0

    # Build address map from tokens table directly (no snapshot JOIN needed)
    symbols = [t["symbol"] for t in tokens]
    address_map: dict[str, str] = {}
    for t in tokens:
        addr = t.get("token_address")
        if addr:
            address_map[t["symbol"]] = addr

    # Fetch stored lifecycle penalty from latest snapshots (still needed for already_pumped)
    stored_lifecycle_pen: dict[str, float] = {}
    syms_with_addr = [s for s in symbols if s in address_map]
    if syms_with_addr:
        snap_result = (
            client.table("token_snapshots")
            .select("symbol, already_pumped_penalty")
            .in_("symbol", syms_with_addr)
            .order("snapshot_at", desc=True)
            .limit(n * 3)
            .execute()
        )
        snap_rows = snap_result.data or []
        if _monitoring:
            _estimate_egress("price_refresh", "token_snapshots", len(snap_rows))
        for row in snap_rows:
            sym = row.get("symbol")
            if sym and sym not in stored_lifecycle_pen:
                stored_lifecycle_pen[sym] = _safe_float(row.get("already_pumped_penalty"), 1.0)

    # Batch fetch all token addresses in 1 DexScreener call (max 30)
    all_addresses = [addr for addr in address_map.values()]
    batch_data = _fetch_dexscreener_batch(all_addresses)
    logger.info("Price refresh: batch fetched %d/%d tokens in 1 call", len(batch_data), len(all_addresses))

    updated = 0
    update_rows = []

    # All time windows to refresh (same DexScreener data, just more upserts)
    ALL_WINDOWS = ["3h", "6h", "12h", "24h", "48h", "7d"]

    # v11: Compute elapsed time since last full scrape for social decay
    # Use scrape_metadata to find when the last full cycle ran
    try:
        meta_result = client.table("scrape_metadata").select("updated_at").eq("id", 1).execute()
        last_scrape_at = meta_result.data[0]["updated_at"] if meta_result.data else None
    except Exception:
        last_scrape_at = None

    minutes_since_scrape = 3  # default: assume 1 refresh interval
    if last_scrape_at:
        try:
            last_dt = datetime.fromisoformat(last_scrape_at.replace("Z", "+00:00"))
            delta = datetime.now(timezone.utc) - last_dt
            minutes_since_scrape = max(0, delta.total_seconds() / 60)
        except Exception:
            pass

    # Pre-fetch which symbols exist in which windows (avoid upserting to windows that don't have the token)
    existing_by_window: dict[str, set[str]] = {}
    for window in ALL_WINDOWS:
        try:
            wr = (
                client.table("tokens")
                .select("symbol")
                .eq("time_window", window)
                .in_("symbol", symbols)
                .execute()
            )
            existing_by_window[window] = {r["symbol"] for r in (wr.data or [])}
        except Exception:
            existing_by_window[window] = set()

    for token in tokens:
        symbol = token["symbol"]
        base_score = token.get("base_score") or token["score"]
        base_conv = token.get("base_score_conviction") or token["score"]
        base_mom = token.get("base_score_momentum") or token["score"]

        address = address_map.get(symbol)
        if not address:
            continue

        market_data = batch_data.get(address)
        if not market_data:
            continue

        lifecycle_pen = stored_lifecycle_pen.get(symbol, 1.0)
        refresh_mult = _compute_refresh_multiplier(market_data, lifecycle_pen)

        # v11: Social decay — scores decay between full scrape cycles
        # for tokens with no new mentions
        freshest_hours = _safe_float(token.get("freshest_mention_hours"), 0)
        elapsed_hours = minutes_since_scrape / 60
        effective_freshest = freshest_hours + elapsed_hours
        decay_at_scrape = _two_phase_decay(freshest_hours)
        decay_now = _two_phase_decay(effective_freshest)
        social_decay = decay_now / max(0.01, decay_at_scrape) if decay_at_scrape > 0.01 else 1.0
        # Clamp: never boost (>1.0), floor at 0.3 to avoid total zeroing
        social_decay = max(0.3, min(1.0, social_decay))

        new_score = min(100, max(0, int(base_score * refresh_mult * social_decay)))
        new_conv = min(100, max(0, int(base_conv * refresh_mult * social_decay)))
        new_mom = min(100, max(0, int(base_mom * refresh_mult * social_decay)))
        new_change = market_data.get("price_change_24h") or token.get("change_24h", 0)

        # Update ALL time windows that have this symbol (not just 7d)
        for window in ALL_WINDOWS:
            if symbol in existing_by_window.get(window, set()):
                update_rows.append({
                    "symbol": symbol,
                    "time_window": window,
                    "score": new_score,
                    "score_conviction": new_conv,
                    "score_momentum": new_mom,
                    "change_24h": round(new_change, 2) if new_change else 0,
                })

        updated += 1

    # Batch upsert updates
    if update_rows:
        client.table("tokens").upsert(
            update_rows, on_conflict="symbol,time_window"
        ).execute()

        # Bump scrape_metadata.updated_at to trigger frontend refresh
        client.table("scrape_metadata").upsert(
            {
                "id": 1,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            on_conflict="id",
        ).execute()

    logger.info("Price refresh: updated %d/%d tokens", updated, len(tokens))
    return updated
