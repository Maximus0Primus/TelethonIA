"""
Jupiter API enrichment: swap routing + price data.

Endpoints used:
- GET /swap/v1/quote: Check if token is tradeable, get price impact + route info (requires API key)
- GET /price/v2: Batch price lookup

Get API key at https://portal.jup.ag
"""

import os
import json
import time
import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# v67: Monitoring — conditional import
try:
    from monitor import track_api_call as _track_api_call
    _monitoring = True
except ImportError:
    _monitoring = False

CACHE_FILE = Path(__file__).parent / "jupiter_cache.json"
CACHE_TTL_SECONDS = 2 * 3600  # 2 hours — tradeability rarely changes

# v36: 30 → 200. Jupiter API is free, 200 × 0.2s = 40s — fits in 15min cycle.
JUPITER_TOP_N = 200  # How many tokens to enrich per cycle
JUPITER_SLEEP = 0.2  # seconds between calls (under 10 RPS)

# ~$1000 worth of SOL in lamports (SOL ~$150, so ~6.67 SOL = 6670000000 lamports)
WSOL_MINT = "So11111111111111111111111111111111111111112"
QUOTE_AMOUNT_LAMPORTS = 6_670_000_000  # ~$1000 in SOL
# v53: Liquidity depth profile — test at $500 and $5K
QUOTE_AMOUNT_500 = 3_335_000_000      # ~$500 SOL
QUOTE_AMOUNT_5K = 33_350_000_000      # ~$5K SOL


# === Cache ===

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


# === Jupiter API calls ===

def _get_api_key() -> str | None:
    return os.environ.get("JUPITER_API_KEY")


def _fetch_jupiter_quote(mint: str, amount_lamports: int = QUOTE_AMOUNT_LAMPORTS) -> dict | None:
    """
    Get a swap quote from Jupiter: WSOL -> token for given SOL amount.
    Returns quote data or None if the token is not routable.
    A 400 response means the token is not tradeable on Jupiter.
    """
    headers = {}
    api_key = _get_api_key()
    if api_key:
        headers["x-api-key"] = api_key

    try:
        if _monitoring:
            with _track_api_call("jupiter", "/quote") as _t:
                resp = requests.get(
                    "https://api.jup.ag/swap/v1/quote",
                    params={
                        "inputMint": WSOL_MINT,
                        "outputMint": mint,
                        "amount": str(amount_lamports),
                        "slippageBps": "50",
                    },
                    headers=headers,
                    timeout=10,
                )
                _t.set_response(resp)
        else:
            resp = requests.get(
                "https://api.jup.ag/swap/v1/quote",
                params={
                    "inputMint": WSOL_MINT,
                    "outputMint": mint,
                    "amount": str(amount_lamports),
                    "slippageBps": "50",
                },
                headers=headers,
                timeout=10,
            )

        if resp.status_code == 400:
            # Token not routable on Jupiter
            return {"tradeable": False}

        if resp.status_code != 200:
            logger.warning("Jupiter quote %d for %s", resp.status_code, mint[:8])
            return None

        data = resp.json()

        # Extract price impact
        price_impact_pct = None
        raw_impact = data.get("priceImpactPct")
        if raw_impact is not None:
            try:
                price_impact_pct = abs(float(raw_impact)) * 100  # Convert to percentage
            except (ValueError, TypeError):
                pass

        # Count route steps
        route_plan = data.get("routePlan") or []
        route_count = len(route_plan)

        return {
            "tradeable": True,
            "price_impact_pct": price_impact_pct,
            "route_count": route_count,
        }

    except requests.RequestException as e:
        logger.warning("Jupiter quote error for %s: %s", mint[:8], e)
        return None


def _fetch_jupiter_prices(mints: list[str]) -> dict[str, float]:
    """
    Batch price lookup via Jupiter Price API v3.
    Returns { mint: price_usd } for tokens that have prices.
    """
    if not mints:
        return {}

    headers = {}
    api_key = _get_api_key()
    if api_key:
        headers["x-api-key"] = api_key

    try:
        if _monitoring:
            with _track_api_call("jupiter", "/price") as _t:
                resp = requests.get(
                    "https://api.jup.ag/price/v3/price",
                    params={"ids": ",".join(mints)},
                    headers=headers,
                    timeout=10,
                )
                _t.set_response(resp)
        else:
            resp = requests.get(
                "https://api.jup.ag/price/v3/price",
                params={"ids": ",".join(mints)},
                headers=headers,
                timeout=10,
            )

        if resp.status_code != 200:
            logger.warning("Jupiter price API %d", resp.status_code)
            return {}

        # v3 response: { mint: { usdPrice, liquidity, ... } } — no "data" wrapper
        data = resp.json()
        prices = {}
        for mint, info in data.items():
            if not isinstance(info, dict):
                continue
            price = info.get("usdPrice")
            if price is not None:
                try:
                    prices[mint] = float(price)
                except (ValueError, TypeError):
                    pass

        return prices

    except requests.RequestException as e:
        logger.warning("Jupiter price error: %s", e)
        return {}


# === Public API ===

def _empty_jupiter_result() -> dict:
    """Return a dict with all Jupiter fields set to None."""
    return {
        "jup_tradeable": None,
        "jup_price_impact_1k": None,
        "jup_route_count": None,
        "jup_price_usd": None,
        # v53: Liquidity depth profile
        "jup_price_impact_500": None,
        "jup_price_impact_5k": None,
        "liquidity_depth_score": None,
    }


def enrich_tokens_jupiter(ranking: list[dict]) -> list[dict]:
    """
    Enrich top tokens with Jupiter swap data (tradeability, price impact, routes, price).

    Modifies tokens in-place and returns the list.
    """
    if not ranking:
        return ranking

    cache = _load_cache()
    enriched_count = 0

    # Collect mints for batch price lookup
    mints_to_price: list[str] = []
    mint_to_index: dict[str, int] = {}

    for i, token in enumerate(ranking):
        if i >= JUPITER_TOP_N:
            break

        mint = token.get("token_address")
        if not mint:
            continue

        # Check cache
        cache_key = f"jup_{mint}"
        if cache_key in cache and _is_cache_fresh(cache[cache_key]):
            cached = dict(cache[cache_key])
            cached.pop("_cached_at", None)
            token.update(cached)
            if cached.get("jup_tradeable") is not None:
                enriched_count += 1
            logger.debug("Jupiter cache hit for %s", mint[:8])
            continue

        # Fetch quote
        quote = _fetch_jupiter_quote(mint)
        time.sleep(JUPITER_SLEEP)

        result = _empty_jupiter_result()

        if quote is not None:
            if not quote.get("tradeable", False):
                result["jup_tradeable"] = 0
            else:
                result["jup_tradeable"] = 1
                result["jup_price_impact_1k"] = quote.get("price_impact_pct")
                result["jup_route_count"] = quote.get("route_count")
                mints_to_price.append(mint)
                mint_to_index[mint] = i

                # v53: Liquidity depth profile — $500 and $5K quotes
                quote_500 = _fetch_jupiter_quote(mint, QUOTE_AMOUNT_500)
                time.sleep(JUPITER_SLEEP)
                if quote_500 and quote_500.get("tradeable"):
                    result["jup_price_impact_500"] = quote_500.get("price_impact_pct")

                quote_5k = _fetch_jupiter_quote(mint, QUOTE_AMOUNT_5K)
                time.sleep(JUPITER_SLEEP)
                if quote_5k and quote_5k.get("tradeable"):
                    result["jup_price_impact_5k"] = quote_5k.get("price_impact_pct")

                # Compute liquidity depth score
                impact_500 = result.get("jup_price_impact_500")
                impact_5k = result.get("jup_price_impact_5k")
                if impact_500 is not None and impact_5k is not None and impact_500 > 0:
                    ratio = impact_5k / max(0.001, impact_500)
                    result["liquidity_depth_score"] = round(min(1.0, 10.0 / max(1.0, ratio)), 4)

        # Cache result (price filled later)
        cache_entry = dict(result)
        cache_entry["_cached_at"] = time.time()
        cache[cache_key] = cache_entry

        token.update(result)
        if result.get("jup_tradeable") is not None:
            enriched_count += 1

    # Batch price lookup for tradeable tokens
    if mints_to_price:
        prices = _fetch_jupiter_prices(mints_to_price)
        for mint, price in prices.items():
            idx = mint_to_index.get(mint)
            if idx is not None and idx < len(ranking):
                ranking[idx]["jup_price_usd"] = price
                # Update cache with price
                cache_key = f"jup_{mint}"
                if cache_key in cache:
                    cache[cache_key]["jup_price_usd"] = price

    _save_cache(cache)
    logger.info(
        "Jupiter enriched %d/%d tokens (top %d analyzed)",
        enriched_count, len(ranking), JUPITER_TOP_N,
    )
    return ranking
