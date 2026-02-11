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
import time
import logging
from datetime import datetime, timezone

import requests
from supabase import create_client
from price_action import compute_price_action_score

logger = logging.getLogger(__name__)

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


def _compute_refresh_multiplier(market_data: dict) -> float:
    """
    Full price-action + market multiplier using fresh DexScreener data.
    Runs compute_price_action_score for chart analysis, then layers market checks.
    Returns combined multiplier [0.2, 1.5].
    """
    # Approximate ath_ratio from 24h price change for price_action scoring
    # If pc24=-60% → price is at 40% of recent high → ath_ratio ≈ 0.40
    pc24 = market_data.get("price_change_24h")
    if pc24 is not None and pc24 < 0:
        market_data["ath_ratio"] = max(0.01, 1.0 + pc24 / 100)
    elif pc24 is not None and pc24 >= 0:
        market_data["ath_ratio"] = 1.0  # At or near 24h high

    # Run full price action scoring (position, direction, volume confirm, support)
    pa = compute_price_action_score(market_data)
    pa_mult = pa.get("price_action_mult", 1.0)

    # Already pumped penalty (aggressive tiers, matches pipeline v5)
    already_pumped = 1.0
    if pc24 is not None and pc24 > 100:
        if pc24 > 700:
            already_pumped = 0.2
        elif pc24 > 400:
            already_pumped = 0.35
        elif pc24 > 200:
            already_pumped = 0.6
        else:
            already_pumped = 0.85

    # Buy/sell ratio pressure
    bsr_mult = 1.0
    bsr = market_data.get("buy_sell_ratio_1h")
    if bsr is not None:
        if bsr > 0.7:
            bsr_mult = 1.1
        elif bsr < 0.3:
            bsr_mult = 0.8

    combined = pa_mult * already_pumped * bsr_mult
    return max(0.2, min(1.5, combined))


def refresh_top_tokens(n: int = REFRESH_TOP_N) -> int:
    """
    Mini-cycle: fetch fresh DexScreener data for top N tokens,
    recalculate market multipliers, update scores in Supabase.

    Returns number of tokens updated.
    """
    client = _get_supabase()

    # 1. Fetch top N tokens from Supabase (7d window, by score DESC)
    result = (
        client.table("tokens")
        .select("symbol, score, base_score, base_score_conviction, base_score_momentum, change_24h")
        .eq("time_window", "7d")
        .order("score", desc=True)
        .limit(n)
        .execute()
    )

    tokens = result.data or []
    if not tokens:
        logger.info("Price refresh: no tokens to update")
        return 0

    # We need token addresses. Fetch from the most recent snapshots.
    symbols = [t["symbol"] for t in tokens]
    snap_result = (
        client.table("token_snapshots")
        .select("symbol, token_address")
        .in_("symbol", symbols)
        .not_.is_("token_address", "null")
        .order("snapshot_at", desc=True)
        .limit(n * 3)
        .execute()
    )

    # Deduplicate: latest snapshot per symbol
    address_map: dict[str, str] = {}
    for row in (snap_result.data or []):
        sym = row.get("symbol")
        addr = row.get("token_address")
        if sym and addr and sym not in address_map:
            address_map[sym] = addr

    # Batch fetch all token addresses in 1 DexScreener call (max 30)
    all_addresses = [addr for addr in address_map.values()]
    batch_data = _fetch_dexscreener_batch(all_addresses)
    logger.info("Price refresh: batch fetched %d/%d tokens in 1 call", len(batch_data), len(all_addresses))

    updated = 0
    update_rows = []

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

        refresh_mult = _compute_refresh_multiplier(market_data)

        new_score = min(100, max(0, int(base_score * refresh_mult)))
        new_conv = min(100, max(0, int(base_conv * refresh_mult)))
        new_mom = min(100, max(0, int(base_mom * refresh_mult)))
        new_change = market_data.get("price_change_24h") or token.get("change_24h", 0)

        update_rows.append({
            "symbol": symbol,
            "time_window": "7d",
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
