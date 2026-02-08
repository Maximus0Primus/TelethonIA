"""
Jupiter API enrichment: swap routing + price data.

Endpoints used (all public, no auth):
- GET /quote: Check if token is tradeable, get price impact + route info
- GET /price/v2: Batch price lookup

Free tier: 10 req/s, 25M requests/month. Estimated usage: ~500/month (<0.01%).
"""

import json
import time
import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

CACHE_FILE = Path(__file__).parent / "jupiter_cache.json"
CACHE_TTL_SECONDS = 30 * 60  # 30 minutes

JUPITER_TOP_N = 10  # How many tokens to enrich per cycle
JUPITER_SLEEP = 0.2  # seconds between calls (under 10 RPS)

# ~$1000 worth of SOL in lamports (SOL ~$150, so ~6.67 SOL = 6670000000 lamports)
WSOL_MINT = "So11111111111111111111111111111111111111112"
QUOTE_AMOUNT_LAMPORTS = 6_670_000_000  # ~$1000 in SOL


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

def _fetch_jupiter_quote(mint: str) -> dict | None:
    """
    Get a swap quote from Jupiter: WSOL -> token for ~$1000.
    Returns quote data or None if the token is not routable.
    A 400 response means the token is not tradeable on Jupiter.
    """
    try:
        resp = requests.get(
            "https://api.jup.ag/quote",
            params={
                "inputMint": WSOL_MINT,
                "outputMint": mint,
                "amount": str(QUOTE_AMOUNT_LAMPORTS),
                "slippageBps": "50",
            },
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
    Batch price lookup via Jupiter Price API v2.
    Returns { mint: price_usd } for tokens that have prices.
    """
    if not mints:
        return {}

    try:
        resp = requests.get(
            "https://api.jup.ag/price/v2",
            params={"ids": ",".join(mints)},
            timeout=10,
        )

        if resp.status_code != 200:
            logger.warning("Jupiter price API %d", resp.status_code)
            return {}

        data = resp.json().get("data", {})
        prices = {}
        for mint, info in data.items():
            price = info.get("price")
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
