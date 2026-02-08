"""
On-chain data enrichment via DexScreener + RugCheck + Birdeye (optional).
Extracts maximum features from each API for ML training.

APIs:
- DexScreener: FREE, no auth, 300 req/min — volume, liquidity, mcap, txns, price changes, token age, DEX platform
- RugCheck: FREE, no auth, ~60 req/min — risk score, holder distribution, mint/freeze authority
- Birdeye: FREE Standard tier (30K CUs/month), needs API key — holder count, unique wallets, trade volume
  → Optional: only called if BIRDEYE_API_KEY is in env, only for top N tokens to conserve CUs
"""

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)

CACHE_FILE = Path(__file__).parent / "enrich_cache.json"
CACHE_TTL_SECONDS = 30 * 60  # 30 minutes (same as scraper cycle)

DEXSCREENER_SEARCH_URL = "https://api.dexscreener.com/latest/dex/search"
RUGCHECK_REPORT_URL = "https://api.rugcheck.xyz/v1/tokens/{mint}/report"
BIRDEYE_TOKEN_OVERVIEW_URL = "https://public-api.birdeye.so/defi/token_overview"
BIRDEYE_TOKEN_SECURITY_URL = "https://public-api.birdeye.so/defi/token_security"

# Max tokens to enrich via Birdeye per cycle (conserve CUs)
BIRDEYE_TOP_N = 10


def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_cache(cache: dict) -> None:
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def _is_cache_fresh(entry: dict) -> bool:
    ts = entry.get("_cached_at", 0)
    return (time.time() - ts) < CACHE_TTL_SECONDS


def _safe_float(val, default=None) -> float | None:
    """Safely convert a value to float, returning default on failure."""
    if val is None:
        return default
    try:
        result = float(val)
        return result if result == result else default  # NaN check
    except (ValueError, TypeError):
        return default


def _fetch_dexscreener(symbol: str) -> dict | None:
    """
    Search DexScreener for a token symbol.
    Extracts ALL available fields for maximum ML feature coverage.
    """
    raw = symbol.lstrip("$")
    try:
        resp = requests.get(
            DEXSCREENER_SEARCH_URL,
            params={"q": raw},
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning("DexScreener %d for %s", resp.status_code, symbol)
            return None

        pairs = resp.json().get("pairs") or []
        if not pairs:
            return None

        # Prefer Solana pairs with exact symbol match
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

        # Pick highest-volume pair
        best = max(sol_pairs, key=lambda p: float(p.get("volume", {}).get("h24", 0) or 0))

        # Count total active pairs for this token (more pairs = more liquidity venues)
        pair_count = len(sol_pairs)

        # Transaction data
        txns_h24 = best.get("txns", {}).get("h24", {})
        buys_24h = int(txns_h24.get("buys", 0) or 0)
        sells_24h = int(txns_h24.get("sells", 0) or 0)
        total_txns = buys_24h + sells_24h

        txns_h6 = best.get("txns", {}).get("h6", {})
        buys_6h = int(txns_h6.get("buys", 0) or 0)
        sells_6h = int(txns_h6.get("sells", 0) or 0)

        txns_h1 = best.get("txns", {}).get("h1", {})
        buys_1h = int(txns_h1.get("buys", 0) or 0)
        sells_1h = int(txns_h1.get("sells", 0) or 0)

        txns_m5 = best.get("txns", {}).get("m5", {})
        buys_5m = int(txns_m5.get("buys", 0) or 0)
        sells_5m = int(txns_m5.get("sells", 0) or 0)

        # Buy/sell ratio (0 to 1, 0.5 = balanced, >0.5 = more buys)
        buy_sell_ratio_24h = buys_24h / max(1, total_txns)
        buy_sell_ratio_1h = buys_1h / max(1, buys_1h + sells_1h)

        # Price changes at multiple timeframes
        price_changes = best.get("priceChange", {})
        price_change_5m = _safe_float(price_changes.get("m5"))
        price_change_1h = _safe_float(price_changes.get("h1"))
        price_change_6h = _safe_float(price_changes.get("h6"))
        price_change_24h = _safe_float(price_changes.get("h24"))

        # Volume data
        volumes = best.get("volume", {})
        volume_24h = _safe_float(volumes.get("h24"), 0)
        volume_6h = _safe_float(volumes.get("h6"), 0)
        volume_1h = _safe_float(volumes.get("h1"), 0)

        # Market data
        liquidity_usd = _safe_float(best.get("liquidity", {}).get("usd"), 0)
        market_cap = _safe_float(best.get("marketCap"), 0) or _safe_float(best.get("fdv"), 0)
        price_usd = _safe_float(best.get("priceUsd"), 0)

        # Derived ratios
        volume_mcap_ratio = volume_24h / max(1, market_cap) if market_cap else None
        liq_mcap_ratio = liquidity_usd / max(1, market_cap) if market_cap else None

        # Volume acceleration (6h volume relative to 24h — is volume increasing?)
        volume_acceleration = (volume_6h * 4) / max(1, volume_24h) if volume_24h else None

        # Token age (hours since pair creation)
        token_age_hours = None
        created_at = best.get("pairCreatedAt")
        if created_at:
            try:
                created_ts = int(created_at) / 1000  # ms to seconds
                age_seconds = time.time() - created_ts
                token_age_hours = round(max(0, age_seconds / 3600), 1)
            except (ValueError, TypeError):
                pass

        # DEX platform (pump.fun vs raydium vs others)
        dex_id = best.get("dexId", "").lower()
        is_pump_fun = 1 if "pump" in dex_id else 0

        return {
            "token_address": best.get("baseToken", {}).get("address", ""),
            "price_usd": price_usd,
            # Volume features
            "volume_24h": volume_24h,
            "volume_6h": volume_6h,
            "volume_1h": volume_1h,
            # Liquidity & market cap
            "liquidity_usd": liquidity_usd,
            "market_cap": market_cap,
            # Transaction counts
            "txn_count_24h": total_txns,
            "buys_24h": buys_24h,
            "sells_24h": sells_24h,
            # Buy/sell ratios
            "buy_sell_ratio_24h": round(buy_sell_ratio_24h, 3),
            "buy_sell_ratio_1h": round(buy_sell_ratio_1h, 3),
            # Price changes (multiple timeframes)
            "price_change_5m": price_change_5m,
            "price_change_1h": price_change_1h,
            "price_change_6h": price_change_6h,
            "price_change_24h": price_change_24h,
            # Derived ratios
            "volume_mcap_ratio": round(volume_mcap_ratio, 6) if volume_mcap_ratio is not None else None,
            "liq_mcap_ratio": round(liq_mcap_ratio, 6) if liq_mcap_ratio is not None else None,
            "volume_acceleration": round(volume_acceleration, 3) if volume_acceleration is not None else None,
            # Token metadata
            "token_age_hours": token_age_hours,
            "is_pump_fun": is_pump_fun,
            "pair_count": pair_count,
            "dex_id": dex_id,
        }

    except requests.RequestException as e:
        logger.warning("DexScreener error for %s: %s", symbol, e)
        return None


def _fetch_rugcheck(mint: str) -> dict | None:
    """
    Fetch RugCheck full report for a token mint address.
    Extracts risk score, holder distribution, and security flags.
    """
    if not mint:
        return None

    try:
        resp = requests.get(
            RUGCHECK_REPORT_URL.format(mint=mint),
            timeout=15,
        )
        if resp.status_code != 200:
            logger.warning("RugCheck %d for %s", resp.status_code, mint)
            return None

        data = resp.json()

        # Risk score
        risk_score = data.get("score", None)
        if risk_score is not None:
            risk_score = min(10000, int(risk_score))

        # Top holders analysis
        top_holders = data.get("topHolders") or []
        top10 = top_holders[:10] if top_holders else []
        top10_pct = sum(h.get("pct", 0) for h in top10)
        insider_pct = sum(h.get("pct", 0) for h in top_holders if h.get("insider", False))
        holder_count = len(top_holders)

        # Security flags from risks array
        risks = data.get("risks") or []
        risk_names = [r.get("name", "") for r in risks]
        has_mint_authority = 1 if any("mint" in r.lower() for r in risk_names) else 0
        has_freeze_authority = 1 if any("freeze" in r.lower() for r in risk_names) else 0
        risk_count = len(risks)

        # LP locked info
        lp_locked_pct = None
        for h in top_holders:
            if h.get("owner", "").lower() in ("raydium", "orca", "meteora"):
                lp_locked_pct = h.get("pct", 0)
                break

        return {
            "risk_score": risk_score,
            "top10_holder_pct": round(top10_pct, 2),
            "insider_pct": round(insider_pct, 2),
            "has_mint_authority": has_mint_authority,
            "has_freeze_authority": has_freeze_authority,
            "risk_count": risk_count,
            "lp_locked_pct": round(lp_locked_pct, 2) if lp_locked_pct is not None else None,
        }

    except requests.RequestException as e:
        logger.warning("RugCheck error for %s: %s", mint, e)
        return None


def _fetch_birdeye(mint: str, api_key: str) -> dict | None:
    """
    Fetch Birdeye token overview for holder count and trade data.
    Requires API key (free Standard tier: 30K CUs/month).
    Token overview = 30 CUs per call.
    """
    if not mint or not api_key:
        return None

    headers = {
        "X-API-KEY": api_key,
        "x-chain": "solana",
    }

    try:
        resp = requests.get(
            BIRDEYE_TOKEN_OVERVIEW_URL,
            params={"address": mint},
            headers=headers,
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning("Birdeye %d for %s", resp.status_code, mint)
            return None

        data = resp.json().get("data", {})
        if not data:
            return None

        return {
            "holder_count": data.get("holder"),
            "unique_wallet_24h": data.get("uniqueWallet24h"),
            "unique_wallet_24h_change": _safe_float(data.get("uniqueWallet24hChangePercent")),
            "trade_24h": data.get("trade24h"),
            "trade_24h_change": _safe_float(data.get("trade24hChangePercent")),
            "buy_24h": data.get("buy24h"),
            "sell_24h": data.get("sell24h"),
            "v_buy_24h_usd": _safe_float(data.get("vBuy24hUSD")),
            "v_sell_24h_usd": _safe_float(data.get("vSell24hUSD")),
        }

    except requests.RequestException as e:
        logger.warning("Birdeye error for %s: %s", mint, e)
        return None


def _empty_result() -> dict:
    """Return a dict with all enrichment fields set to None."""
    return {
        "token_address": None,
        "price_usd": None,
        # DexScreener features
        "volume_24h": None,
        "volume_6h": None,
        "volume_1h": None,
        "liquidity_usd": None,
        "market_cap": None,
        "txn_count_24h": None,
        "buys_24h": None,
        "sells_24h": None,
        "buy_sell_ratio_24h": None,
        "buy_sell_ratio_1h": None,
        "price_change_5m": None,
        "price_change_1h": None,
        "price_change_6h": None,
        "price_change_24h": None,
        "volume_mcap_ratio": None,
        "liq_mcap_ratio": None,
        "volume_acceleration": None,
        "token_age_hours": None,
        "is_pump_fun": None,
        "pair_count": None,
        "dex_id": None,
        # RugCheck features
        "risk_score": None,
        "top10_holder_pct": None,
        "insider_pct": None,
        "has_mint_authority": None,
        "has_freeze_authority": None,
        "risk_count": None,
        "lp_locked_pct": None,
        # Birdeye features (optional)
        "holder_count": None,
        "unique_wallet_24h": None,
        "unique_wallet_24h_change": None,
        "trade_24h": None,
        "trade_24h_change": None,
        "buy_24h": None,
        "sell_24h": None,
        "v_buy_24h_usd": None,
        "v_sell_24h_usd": None,
    }


def enrich_token(symbol: str, cache: dict, birdeye_key: str | None = None) -> dict:
    """
    Enrich a single token with on-chain data from DexScreener + RugCheck.
    Birdeye is called only if birdeye_key is provided.
    """
    raw = symbol.lstrip("$")

    # Check cache
    if raw in cache and _is_cache_fresh(cache[raw]):
        logger.debug("Cache hit for %s", symbol)
        cached = dict(cache[raw])
        cached.pop("_cached_at", None)
        return cached

    result = _empty_result()

    # DexScreener (free, 300/min)
    dex_data = _fetch_dexscreener(symbol)
    if dex_data:
        result.update(dex_data)
        time.sleep(0.2)

        mint = dex_data.get("token_address")
        if mint:
            # RugCheck (free, ~60/min)
            rug_data = _fetch_rugcheck(mint)
            if rug_data:
                result.update(rug_data)
            time.sleep(1.0)

            # Birdeye (optional, 30 CUs per call)
            if birdeye_key:
                bird_data = _fetch_birdeye(mint, birdeye_key)
                if bird_data:
                    result.update(bird_data)
                time.sleep(1.0)
    else:
        time.sleep(0.2)

    # Update cache
    cache_entry = dict(result)
    cache_entry["_cached_at"] = time.time()
    cache[raw] = cache_entry

    return result


def enrich_tokens(ranking: list[dict]) -> list[dict]:
    """
    Enrich all tokens in a ranking list with on-chain data.
    Birdeye is only called for top BIRDEYE_TOP_N tokens to conserve free CUs.
    Modifies tokens in-place and returns the list.
    """
    if not ranking:
        return ranking

    cache = _load_cache()
    birdeye_key = os.environ.get("BIRDEYE_API_KEY")
    enriched_count = 0
    birdeye_count = 0

    for i, token in enumerate(ranking):
        symbol = token.get("symbol", "")
        if not symbol:
            continue

        # Only use Birdeye for top N tokens
        use_birdeye = birdeye_key if i < BIRDEYE_TOP_N else None

        data = enrich_token(symbol, cache, birdeye_key=use_birdeye)
        token.update(data)

        if data.get("volume_24h") is not None:
            enriched_count += 1
        if data.get("holder_count") is not None:
            birdeye_count += 1

    _save_cache(cache)
    logger.info(
        "Enriched %d/%d tokens (DexScreener+RugCheck), %d with Birdeye",
        enriched_count, len(ranking), birdeye_count,
    )
    return ranking
