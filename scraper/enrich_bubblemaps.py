"""
Bubblemaps API enrichment module for wallet clustering analysis.

API: https://api.bubblemaps.io (beta, invite-only)
- Decentralization score (0-100)
- Wallet clusters (connected wallets holding same token)
- Holder labels (CEX, DEX, contract identification)

Requires BUBBLEMAPS_API_KEY in environment.
Gracefully skips if key not set or API unavailable.

Rate limiting: daily query-seconds quota (varies per partner).
Cache: 30min (same as scraper cycle).
"""

import os
import json
import time
import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

BUBBLEMAPS_API_URL = "https://api.bubblemaps.io/maps/solana/{address}"
CACHE_FILE = Path(__file__).parent / "bubblemaps_cache.json"
CACHE_TTL_SECONDS = 4 * 3600  # 4 hours — wallet clusters are very stable

# Only enrich top N tokens per cycle (conserve quota)
BUBBLEMAPS_TOP_N = 10


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


def _empty_bubblemaps_result() -> dict:
    return {
        "bubblemaps_score": None,
        "bubblemaps_cluster_count": None,
        "bubblemaps_cluster_max_pct": None,
        "bubblemaps_cex_pct": None,
        "bubblemaps_dex_pct": None,
    }


def _fetch_bubblemaps(address: str, api_key: str) -> dict | None:
    """
    Fetch Bubblemaps map data for a Solana token.
    Returns decentralization score + cluster analysis.
    """
    try:
        resp = requests.get(
            BUBBLEMAPS_API_URL.format(address=address),
            headers={"X-ApiKey": api_key},
            params={
                "return_clusters": "true",
                "return_decentralization_score": "true",
                "return_nodes": "false",  # save quota — we only need clusters + score
                "return_relationships": "false",
            },
            timeout=20,  # Bubblemaps can be slow (up to 15s for uncached tokens)
        )

        if resp.status_code == 429:
            logger.warning("Bubblemaps rate limited — stopping enrichment for this cycle")
            return None
        if resp.status_code != 200:
            logger.warning("Bubblemaps %d for %s", resp.status_code, address[:8])
            return None

        data = resp.json()

        # Decentralization score (0-100, higher = more decentralized = healthier)
        decentralization_score = data.get("decentralization_score")

        # Cluster analysis
        clusters = data.get("clusters") or []
        cluster_count = len(clusters)

        # Largest cluster share (highest concentration of connected wallets)
        cluster_max_pct = 0.0
        if clusters:
            cluster_max_pct = max(c.get("share", 0) for c in clusters) * 100  # decimal → %

        # Identified supply breakdown
        identified = data.get("metadata", {}).get("identified_supply", {})
        cex_pct = (identified.get("share_in_cexs") or 0) * 100
        dex_pct = (identified.get("share_in_dexs") or 0) * 100

        return {
            "bubblemaps_score": round(decentralization_score, 1) if decentralization_score is not None else None,
            "bubblemaps_cluster_count": cluster_count,
            "bubblemaps_cluster_max_pct": round(cluster_max_pct, 2),
            "bubblemaps_cex_pct": round(cex_pct, 2),
            "bubblemaps_dex_pct": round(dex_pct, 2),
        }

    except requests.RequestException as e:
        logger.warning("Bubblemaps error for %s: %s", address[:8], e)
        return None


def enrich_tokens_bubblemaps(ranking: list[dict]) -> None:
    """
    Enrich top N tokens with Bubblemaps wallet clustering data.
    Modifies tokens in-place. Skips gracefully if no API key.
    """
    api_key = os.environ.get("BUBBLEMAPS_API_KEY")
    if not api_key:
        logger.debug("BUBBLEMAPS_API_KEY not set — skipping Bubblemaps enrichment")
        return

    cache = _load_cache()
    enriched = 0
    rate_limited = False

    for i, token in enumerate(ranking):
        if i >= BUBBLEMAPS_TOP_N:
            break
        if rate_limited:
            break

        address = token.get("token_address")
        if not address:
            continue

        # Check cache
        if address in cache and _is_cache_fresh(cache[address]):
            cached = dict(cache[address])
            cached.pop("_cached_at", None)
            token.update(cached)
            if cached.get("bubblemaps_score") is not None:
                enriched += 1
            continue

        data = _fetch_bubblemaps(address, api_key)
        if data is None:
            # Could be rate limit — stop trying for this cycle
            if i > 0:  # first failure after some success = likely rate limit
                rate_limited = True
            continue

        token.update(data)
        enriched += 1

        # Cache result
        cache_entry = dict(data)
        cache_entry["_cached_at"] = time.time()
        cache[address] = cache_entry

        time.sleep(1.0)  # Respect API — 1 req/s conservative

    _save_cache(cache)
    if enriched:
        logger.info("Bubblemaps enriched %d/%d tokens (top %d)", enriched, len(ranking), BUBBLEMAPS_TOP_N)
