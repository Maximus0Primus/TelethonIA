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

# v67: Monitoring — conditional import
try:
    from monitor import track_api_call as _track_api_call
    _monitoring = True
except ImportError:
    _monitoring = False

CACHE_FILE = Path(__file__).parent / "enrich_cache.json"
# Per-source TTLs: only re-fetch when data is actually stale
# v58: These are defaults — overridden by scoring_config.pipeline_config.enrichment
TTL_DEXSCREENER = 5 * 60     # 5 min — prices change fast
TTL_RUGCHECK = 2 * 60 * 60   # 2 hours — risk flags rarely change
TTL_BIRDEYE = 60 * 60        # 1 hour — holder counts change slowly

DEXSCREENER_SEARCH_URL = "https://api.dexscreener.com/latest/dex/search"
RUGCHECK_REPORT_URL = "https://api.rugcheck.xyz/v1/tokens/{mint}/report"
BIRDEYE_TOKEN_OVERVIEW_URL = "https://public-api.birdeye.so/defi/token_overview"
BIRDEYE_TOKEN_SECURITY_URL = "https://public-api.birdeye.so/defi/token_security"

# Max tokens to enrich via Birdeye per cycle (free tier = 30K CUs/month)
# Raised from 5 → 20: ~600 CU/cycle × 96 cycles/day = ~18K CU/month (within 30K limit)
BIRDEYE_TOP_N = 20


def load_enrichment_config(client) -> None:
    """v58: Load enrichment config from scoring_config.pipeline_config.enrichment.
    Overrides module-level TTL and TOP_N constants. No-op if DB unreachable."""
    global TTL_DEXSCREENER, TTL_RUGCHECK, TTL_BIRDEYE, BIRDEYE_TOP_N
    try:
        result = client.table("scoring_config").select("pipeline_config").eq("id", 1).execute()
        if not result.data or not result.data[0].get("pipeline_config"):
            return
        cfg = result.data[0]["pipeline_config"].get("enrichment", {})
        if cfg:
            TTL_DEXSCREENER = int(cfg.get("ttl_dexscreener", TTL_DEXSCREENER))
            TTL_RUGCHECK = int(cfg.get("ttl_rugcheck", TTL_RUGCHECK))
            TTL_BIRDEYE = int(cfg.get("ttl_birdeye", TTL_BIRDEYE))
            BIRDEYE_TOP_N = int(cfg.get("birdeye_top_n", BIRDEYE_TOP_N))
            logger.info("v58: Loaded enrichment config: dex_ttl=%ds, rug_ttl=%ds, bird_ttl=%ds, bird_n=%d",
                        TTL_DEXSCREENER, TTL_RUGCHECK, TTL_BIRDEYE, BIRDEYE_TOP_N)
    except Exception as e:
        logger.warning("v58: Failed to load enrichment config: %s (using defaults)", e)


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
        if _monitoring:
            with _track_api_call("dexscreener", "/search") as _t:
                resp = requests.get(
                    DEXSCREENER_SEARCH_URL,
                    params={"q": raw},
                    timeout=10,
                )
                _t.set_response(resp)
        else:
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

        # Solana-only: all downstream APIs (RugCheck, Helius, Jupiter, Birdeye) are Solana-specific
        sol_pairs = [
            p for p in pairs
            if p.get("baseToken", {}).get("symbol", "").upper() == raw
            and p.get("chainId") == "solana"
        ]
        if not sol_pairs:
            return None

        # Pick highest-volume pair
        best = max(sol_pairs, key=lambda p: float(p.get("volume", {}).get("h24", 0) or 0))

        # Count total active pairs for this token (more pairs = more liquidity venues)
        pair_count = len(sol_pairs)

        # PVP detection: count distinct token addresses with same symbol
        best_address = best.get("baseToken", {}).get("address", "")
        distinct_addresses = set()
        for p in sol_pairs:
            addr = p.get("baseToken", {}).get("address", "")
            if addr:
                distinct_addresses.add(addr)
        pvp_same_name_count = len(distinct_addresses)

        # Count competing tokens created within 4h of our best pair
        pvp_recent_count = 0
        best_created = best.get("pairCreatedAt")
        if best_created and pvp_same_name_count > 1:
            for p in sol_pairs:
                p_addr = p.get("baseToken", {}).get("address", "")
                p_created = p.get("pairCreatedAt")
                if p_addr != best_address and p_created:
                    try:
                        diff_h = abs(int(best_created) - int(p_created)) / (1000 * 3600)
                        if diff_h <= 4:
                            pvp_recent_count += 1
                    except (ValueError, TypeError):
                        pass

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
        buy_sell_ratio_5m = buys_5m / max(1, buys_5m + sells_5m)

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
        volume_5m = _safe_float(volumes.get("m5"), 0)

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

        # Pump.fun graduation detection
        has_pump_pair = any("pump" in p.get("dexId", "").lower() for p in sol_pairs)
        has_raydium_pair = any("raydium" in p.get("dexId", "").lower() for p in sol_pairs)

        if has_pump_pair and has_raydium_pair:
            pump_graduation_status = "graduated"
        elif has_pump_pair:
            pump_graduation_status = "bonding"
        else:
            pump_graduation_status = None

        # v10: Extract boosts (paid promotion), socials, websites
        boosts_active = 0
        boosts_data = best.get("boosts") or {}
        if isinstance(boosts_data, dict):
            boosts_active = int(boosts_data.get("active", 0) or 0)
        elif isinstance(boosts_data, int):
            boosts_active = boosts_data

        info = best.get("info") or {}
        socials = info.get("socials") or []
        websites = info.get("websites") or []
        has_twitter = 1 if any(s.get("type") == "twitter" for s in socials if isinstance(s, dict)) else 0
        has_telegram = 1 if any(s.get("type") == "telegram" for s in socials if isinstance(s, dict)) else 0
        has_website = 1 if len(websites) > 0 else 0
        social_count = len(socials)

        return {
            "token_address": best.get("baseToken", {}).get("address", ""),
            "pair_address": best.get("pairAddress", ""),  # Pool address for OHLCV lookups
            "price_usd": price_usd,
            # Volume features
            "volume_24h": volume_24h,
            "volume_6h": volume_6h,
            "volume_1h": volume_1h,
            "volume_5m": volume_5m,
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
            "buy_sell_ratio_5m": round(buy_sell_ratio_5m, 3),
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
            "pump_graduation_status": pump_graduation_status,
            # PVP detection
            "pvp_same_name_count": pvp_same_name_count,
            "pvp_recent_count": pvp_recent_count,
            # v10: Boosts & social signals (ML features)
            "boosts_active": boosts_active,
            "has_twitter": has_twitter,
            "has_telegram": has_telegram,
            "has_website": has_website,
            "social_count": social_count,
        }

    except requests.RequestException as e:
        logger.warning("DexScreener error for %s: %s", symbol, e)
        return None


def _fetch_rugcheck(mint: str) -> dict | None:
    """
    Fetch RugCheck full report for a Solana token mint address.
    Extracts risk score, holder distribution, and security flags.
    """
    if not mint:
        return None

    # RugCheck is Solana-only — skip Ethereum/EVM addresses (0x prefix)
    if mint.startswith("0x"):
        logger.debug("Skipping RugCheck for non-Solana address %s", mint)
        return None

    try:
        if _monitoring:
            with _track_api_call("rugcheck", "/report") as _t:
                resp = requests.get(
                    RUGCHECK_REPORT_URL.format(mint=mint),
                    timeout=15,
                )
                _t.set_response(resp)
        else:
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

        # LP locked info — check multiple sources in the RugCheck response
        lp_locked_pct = None

        # Source 1: top-level lpLockedPct (present in some response versions)
        if data.get("lpLockedPct") is not None:
            lp_locked_pct = float(data["lpLockedPct"])

        # Source 2: markets array — each market has lp.lpLockedPct
        if lp_locked_pct is None:
            markets = data.get("markets") or []
            for m in markets:
                lp = m.get("lp") or {}
                pct = lp.get("lpLockedPct")
                if pct is not None and pct > 0:
                    lp_locked_pct = float(pct)
                    break  # use first market with locked LP

        # Source 3: topHolders flagged as LP (isLpToken or owner label)
        if lp_locked_pct is None:
            for h in top_holders:
                if h.get("isLpToken") or h.get("is_lp"):
                    lp_locked_pct = h.get("pct", 0)
                    break

        if lp_locked_pct is None:
            # Debug: log available keys so we can find the right field
            avail_keys = [k for k in data.keys() if "lp" in k.lower() or "lock" in k.lower() or "market" in k.lower()]
            if avail_keys:
                logger.debug("RugCheck LP debug for %s: relevant keys=%s", mint[:8], avail_keys)

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
        if _monitoring:
            with _track_api_call("birdeye", "/overview") as _t:
                resp = requests.get(
                    BIRDEYE_TOKEN_OVERVIEW_URL,
                    params={"address": mint},
                    headers=headers,
                    timeout=10,
                )
                _t.set_response(resp)
        else:
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
        "pair_address": None,
        "price_usd": None,
        # DexScreener features
        "volume_24h": None,
        "volume_6h": None,
        "volume_1h": None,
        "volume_5m": None,
        "liquidity_usd": None,
        "market_cap": None,
        "txn_count_24h": None,
        "buys_24h": None,
        "sells_24h": None,
        "buy_sell_ratio_24h": None,
        "buy_sell_ratio_1h": None,
        "buy_sell_ratio_5m": None,
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
        "pump_graduation_status": None,
        # PVP detection
        "pvp_same_name_count": None,
        "pvp_recent_count": None,
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


def _fetch_dexscreener_by_address(address: str) -> dict | None:
    """
    v40: Fetch token data by exact contract address (not by symbol search).
    Prevents symbol collision where DexScreener search picks the wrong CA.
    Uses the same /tokens/v1/solana/{address} endpoint as CA resolution.
    """
    try:
        if _monitoring:
            with _track_api_call("dexscreener", "/by-address") as _t:
                resp = requests.get(
                    f"https://api.dexscreener.com/tokens/v1/solana/{address}",
                    timeout=10,
                )
                _t.set_response(resp)
        else:
            resp = requests.get(
                f"https://api.dexscreener.com/tokens/v1/solana/{address}",
                timeout=10,
            )
        if resp.status_code != 200:
            logger.warning("DexScreener by-address %d for %s…", resp.status_code, address[:8])
            return None

        pairs = resp.json() if isinstance(resp.json(), list) else resp.json().get("pairs", [])
        if not pairs or not isinstance(pairs, list):
            return None

        # Filter to pairs where baseToken.address matches our target
        target_pairs = [
            p for p in pairs
            if p.get("baseToken", {}).get("address", "") == address
            and p.get("chainId") == "solana"
        ]
        if not target_pairs:
            # Fallback: use all pairs (API should only return pairs for this token)
            target_pairs = [p for p in pairs if p.get("chainId") == "solana"]
        if not target_pairs:
            return None

        # Pick highest-volume pair (same logic as _fetch_dexscreener)
        best = max(target_pairs, key=lambda p: float(p.get("volume", {}).get("h24", 0) or 0))

        pair_count = len(target_pairs)
        best_address = best.get("baseToken", {}).get("address", "")
        distinct_addresses = {p.get("baseToken", {}).get("address", "") for p in target_pairs if p.get("baseToken", {}).get("address")}
        pvp_same_name_count = len(distinct_addresses)

        pvp_recent_count = 0
        best_created = best.get("pairCreatedAt")
        if best_created and pvp_same_name_count > 1:
            for p in target_pairs:
                p_addr = p.get("baseToken", {}).get("address", "")
                p_created = p.get("pairCreatedAt")
                if p_addr != best_address and p_created:
                    try:
                        diff_h = abs(int(best_created) - int(p_created)) / (1000 * 3600)
                        if diff_h <= 4:
                            pvp_recent_count += 1
                    except (ValueError, TypeError):
                        pass

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

        buy_sell_ratio_24h = buys_24h / max(1, total_txns)
        buy_sell_ratio_1h = buys_1h / max(1, buys_1h + sells_1h)
        buy_sell_ratio_5m = buys_5m / max(1, buys_5m + sells_5m)

        price_changes = best.get("priceChange", {})
        price_change_5m = _safe_float(price_changes.get("m5"))
        price_change_1h = _safe_float(price_changes.get("h1"))
        price_change_6h = _safe_float(price_changes.get("h6"))
        price_change_24h = _safe_float(price_changes.get("h24"))

        volumes = best.get("volume", {})
        volume_24h = _safe_float(volumes.get("h24"), 0)
        volume_6h = _safe_float(volumes.get("h6"), 0)
        volume_1h = _safe_float(volumes.get("h1"), 0)
        volume_5m = _safe_float(volumes.get("m5"), 0)

        liquidity_usd = _safe_float(best.get("liquidity", {}).get("usd"), 0)
        market_cap = _safe_float(best.get("marketCap"), 0) or _safe_float(best.get("fdv"), 0)
        price_usd = _safe_float(best.get("priceUsd"), 0)

        volume_mcap_ratio = volume_24h / max(1, market_cap) if market_cap else None
        liq_mcap_ratio = liquidity_usd / max(1, market_cap) if market_cap else None
        volume_acceleration = (volume_6h * 4) / max(1, volume_24h) if volume_24h else None

        token_age_hours = None
        created_at = best.get("pairCreatedAt")
        if created_at:
            try:
                created_ts = int(created_at) / 1000
                age_seconds = time.time() - created_ts
                token_age_hours = round(max(0, age_seconds / 3600), 1)
            except (ValueError, TypeError):
                pass

        dex_id = best.get("dexId", "").lower()
        is_pump_fun = 1 if "pump" in dex_id else 0

        has_pump_pair = any("pump" in p.get("dexId", "").lower() for p in target_pairs)
        has_raydium_pair = any("raydium" in p.get("dexId", "").lower() for p in target_pairs)
        if has_pump_pair and has_raydium_pair:
            pump_graduation_status = "graduated"
        elif has_pump_pair:
            pump_graduation_status = "bonding"
        else:
            pump_graduation_status = None

        boosts_active = 0
        boosts_data = best.get("boosts") or {}
        if isinstance(boosts_data, dict):
            boosts_active = int(boosts_data.get("active", 0) or 0)
        elif isinstance(boosts_data, int):
            boosts_active = boosts_data

        info = best.get("info") or {}
        socials = info.get("socials") or []
        websites = info.get("websites") or []
        has_twitter = 1 if any(s.get("type") == "twitter" for s in socials if isinstance(s, dict)) else 0
        has_telegram = 1 if any(s.get("type") == "telegram" for s in socials if isinstance(s, dict)) else 0
        has_website = 1 if len(websites) > 0 else 0
        social_count = len(socials)

        return {
            "token_address": best.get("baseToken", {}).get("address", ""),
            "pair_address": best.get("pairAddress", ""),
            "price_usd": price_usd,
            "volume_24h": volume_24h,
            "volume_6h": volume_6h,
            "volume_1h": volume_1h,
            "volume_5m": volume_5m,
            "liquidity_usd": liquidity_usd,
            "market_cap": market_cap,
            "txn_count_24h": total_txns,
            "buys_24h": buys_24h,
            "sells_24h": sells_24h,
            "buy_sell_ratio_24h": round(buy_sell_ratio_24h, 3),
            "buy_sell_ratio_1h": round(buy_sell_ratio_1h, 3),
            "buy_sell_ratio_5m": round(buy_sell_ratio_5m, 3),
            "price_change_5m": price_change_5m,
            "price_change_1h": price_change_1h,
            "price_change_6h": price_change_6h,
            "price_change_24h": price_change_24h,
            "volume_mcap_ratio": round(volume_mcap_ratio, 6) if volume_mcap_ratio is not None else None,
            "liq_mcap_ratio": round(liq_mcap_ratio, 6) if liq_mcap_ratio is not None else None,
            "volume_acceleration": round(volume_acceleration, 3) if volume_acceleration is not None else None,
            "token_age_hours": token_age_hours,
            "is_pump_fun": is_pump_fun,
            "pair_count": pair_count,
            "dex_id": dex_id,
            "pump_graduation_status": pump_graduation_status,
            "pvp_same_name_count": pvp_same_name_count,
            "pvp_recent_count": pvp_recent_count,
            "boosts_active": boosts_active,
            "has_twitter": has_twitter,
            "has_telegram": has_telegram,
            "has_website": has_website,
            "social_count": social_count,
        }

    except requests.RequestException as e:
        logger.warning("DexScreener by-address error for %s…: %s", address[:8], e)
        return None


def enrich_token(symbol: str, cache: dict, birdeye_key: str | None = None, known_ca: str | None = None) -> dict:
    """
    Enrich a single token with on-chain data from DexScreener + RugCheck + Birdeye.
    Uses per-source TTLs: only re-fetches data that is actually stale.

    v40: When known_ca is provided, fetches by exact contract address instead of
    symbol search, preventing symbol collision where multiple CAs share the same name.
    """
    raw = symbol.lstrip("$")
    now = time.time()

    entry = cache.get(raw, {})
    result = _empty_result()

    # Start with any cached data
    for k, v in entry.items():
        if not k.startswith("_") and k in result:
            result[k] = v

    # --- DexScreener (5 min TTL, free, 300/min) ---
    # v40+v50: Force re-fetch if known_ca differs from cached token_address (CA changed)
    # v50: Also re-fetch when known_ca is provided but cache has no CA yet
    cached_ca = entry.get("token_address")
    ca_changed = known_ca and (not cached_ca or known_ca != cached_ca)
    if now - entry.get("_dex_at", 0) > TTL_DEXSCREENER or ca_changed:
        # v40: Use exact CA lookup when available to prevent symbol collision
        if known_ca:
            dex_data = _fetch_dexscreener_by_address(known_ca)
        else:
            dex_data = _fetch_dexscreener(symbol)
        if dex_data:
            result.update(dex_data)
            entry["_dex_at"] = now
        time.sleep(0.2)
    else:
        logger.debug("DexScreener cache hit for %s", symbol)

    mint = result.get("token_address")

    if mint:
        # --- RugCheck (2h TTL, free, ~60/min) ---
        if now - entry.get("_rug_at", 0) > TTL_RUGCHECK:
            rug_data = _fetch_rugcheck(mint)
            if rug_data:
                result.update(rug_data)
                entry["_rug_at"] = now
            time.sleep(0.5)  # v34: 1.0→0.5s (RugCheck ~60/min, 0.5s = safe)
        else:
            logger.debug("RugCheck cache hit for %s", symbol)

        # --- Birdeye (1h TTL, optional, 30 CUs per call) ---
        if birdeye_key and now - entry.get("_bird_at", 0) > TTL_BIRDEYE:
            bird_data = _fetch_birdeye(mint, birdeye_key)
            if bird_data:
                result.update(bird_data)
                entry["_bird_at"] = now
            time.sleep(0.5)  # v34: 1.0→0.5s (Birdeye 30K CUs/month, 0.5s = safe)

    # Update cache with all data + per-source timestamps
    cache_entry = dict(result)
    cache_entry["_dex_at"] = entry.get("_dex_at", 0)
    cache_entry["_rug_at"] = entry.get("_rug_at", 0)
    cache_entry["_bird_at"] = entry.get("_bird_at", 0)
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
    helius_key = os.environ.get("HELIUS_API_KEY")

    if not birdeye_key:
        logger.warning("BIRDEYE_API_KEY not set — Birdeye enrichment (holder_count, wallets) will be skipped")
    if not helius_key:
        logger.warning("HELIUS_API_KEY not set — Helius enrichment (gini, whale, jito) will be skipped")

    enriched_count = 0
    birdeye_count = 0

    for i, token in enumerate(ranking):
        symbol = token.get("symbol", "")
        if not symbol:
            continue

        # Only use Birdeye for top N tokens
        use_birdeye = birdeye_key if i < BIRDEYE_TOP_N else None

        # v40: Pass known_ca to fetch by exact address, preventing symbol collision
        data = enrich_token(symbol, cache, birdeye_key=use_birdeye, known_ca=token.get("kol_resolved_ca"))
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
