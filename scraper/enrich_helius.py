"""
Helius API enrichment: bundle detection + holder quality + transaction analysis.

Endpoints used (all JSON-RPC on mainnet.helius-rpc.com):
- getTokenAccounts (DAS): 10 credits/page, paginated up to 5 pages (5000 holders)
- getSignaturesForAddress: 10 credits/call, 50 recent signatures

Free tier budget: 1M credits/month. Estimated usage: ~22K/month (2.2%).

Bundle detection algorithm:
  1. Fetch all token holders via getTokenAccounts
  2. Sort by balance descending, filter dust
  3. Cluster consecutive balances within 5% tolerance
  4. Flag clusters of 3+ wallets as suspected bundles
"""

import os
import json
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)

CACHE_FILE = Path(__file__).parent / "helius_cache.json"
CACHE_TTL_SECONDS = 2 * 3600  # 2 hours (on-chain data changes slower than social)

# How many tokens to enrich per cycle
# Raised from 20 → 50: still under 1M CU/month Helius free tier
HELIUS_TOP_N = 50          # getTokenAccounts (holder analysis + bundles)
HELIUS_SMART_MONEY_N = 5   # getSignaturesForAddress (transaction analysis)

# Rate limiting
HELIUS_SLEEP = 0.15  # seconds between calls (under 10 RPS)
HELIUS_MAX_PAGES = 5  # max pagination pages (5000 holders max)

# v34: Thread-safe rate limiter for parallel Helius processing
_helius_lock = threading.Lock()
_helius_last_call = 0.0


def _helius_rate_limit():
    """Thread-safe rate limiter: ensures minimum 0.12s between Helius API calls (~8 RPS)."""
    global _helius_last_call
    with _helius_lock:
        now = time.time()
        elapsed = now - _helius_last_call
        if elapsed < 0.12:
            time.sleep(0.12 - elapsed)
        _helius_last_call = time.time()


def _get_api_key() -> str | None:
    return os.environ.get("HELIUS_API_KEY")


def _rpc_url(api_key: str) -> str:
    return f"https://mainnet.helius-rpc.com/?api-key={api_key}"


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


# === Helius API calls ===

def _fetch_token_accounts(mint: str, api_key: str) -> list[dict] | None:
    """
    Fetch all token holder accounts via Helius DAS getTokenAccounts.
    Paginates up to HELIUS_MAX_PAGES pages (1000 accounts each).
    Returns list of { owner, amount } dicts, or None on failure.
    """
    url = _rpc_url(api_key)
    all_accounts = []
    cursor = None

    for page in range(HELIUS_MAX_PAGES):
        payload = {
            "jsonrpc": "2.0",
            "id": f"helius-holders-{page}",
            "method": "getTokenAccounts",
            "params": {
                "mint": mint,
                "limit": 1000,
                "options": {
                    "showZeroBalance": False,
                },
            },
        }
        if cursor:
            payload["params"]["cursor"] = cursor

        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = requests.post(url, json=payload, timeout=15)

                if resp.status_code == 429:
                    wait = 2 ** attempt + 1
                    logger.debug("Helius 429 for %s page %d — retry %d in %ds", mint[:8], page, attempt + 1, wait)
                    time.sleep(wait)
                    continue

                if resp.status_code != 200:
                    logger.warning("Helius getTokenAccounts %d for %s (page %d)", resp.status_code, mint[:8], page)
                    break

                data = resp.json()
                if "error" in data:
                    logger.warning("Helius RPC error for %s: %s", mint[:8], data["error"])
                    break

                result = data.get("result", {})
                accounts = result.get("token_accounts", [])
                if not accounts:
                    break

                for acc in accounts:
                    amount = acc.get("amount")
                    owner = acc.get("owner")
                    if amount is not None and owner:
                        try:
                            amt = int(amount)
                            if amt > 0:
                                all_accounts.append({"owner": owner, "amount": amt})
                        except (ValueError, TypeError):
                            pass

                cursor = result.get("cursor")
                if not cursor:
                    break

                _helius_rate_limit()
                break  # Success — exit retry loop

            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                logger.warning("Helius getTokenAccounts error for %s: %s", mint[:8], e)
                break
        else:
            # All retries exhausted (429s)
            logger.warning("Helius 429 exhausted retries for %s page %d", mint[:8], page)
            break

    if all_accounts:
        logger.debug("Fetched %d holder accounts for %s", len(all_accounts), mint[:8])
    return all_accounts if all_accounts else None


def _fetch_recent_signatures(mint: str, api_key: str, limit: int = 50) -> list[dict] | None:
    """
    Fetch recent transaction signatures for a token mint address.
    Uses getSignaturesForAddress JSON-RPC method.
    Returns list of signature info dicts, or None on failure.
    """
    url = _rpc_url(api_key)
    payload = {
        "jsonrpc": "2.0",
        "id": "helius-sigs",
        "method": "getSignaturesForAddress",
        "params": [
            mint,
            {"limit": limit},
        ],
    }

    try:
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code != 200:
            logger.warning("Helius getSignaturesForAddress %d for %s", resp.status_code, mint[:8])
            return None

        data = resp.json()
        if "error" in data:
            logger.warning("Helius sigs error for %s: %s", mint[:8], data["error"])
            return None

        result = data.get("result", [])
        return result if result else None

    except requests.RequestException as e:
        logger.warning("Helius sigs error for %s: %s", mint[:8], e)
        return None


# === Analysis functions ===

def _compute_gini_coefficient(amounts: list[int]) -> float:
    """
    Compute Gini coefficient from holder balances.
    Returns float in [0, 1]. 0 = perfect equality, 1 = one holder owns all.
    """
    if not amounts or len(amounts) < 2:
        return 0.0

    sorted_amounts = sorted(amounts)
    n = len(sorted_amounts)
    total = sum(sorted_amounts)
    if total == 0:
        return 0.0

    # Gini formula: G = (2 * sum(i * x_i) / (n * total)) - (n + 1) / n
    cumulative = sum((i + 1) * x for i, x in enumerate(sorted_amounts))
    gini = (2 * cumulative) / (n * total) - (n + 1) / n
    return max(0.0, min(1.0, gini))


def _detect_bundles(accounts: list[dict], total_supply: int | None = None) -> dict:
    """
    Detect bundled wallets: groups of 3+ wallets with similar balances.

    Algorithm:
    1. Sort holders by balance descending
    2. Walk through, cluster consecutive amounts within 5% tolerance
    3. Flag clusters of 3+ wallets as suspected bundles

    Returns dict with bundle_detected, bundle_count, bundle_wallets, bundle_pct.
    """
    if not accounts or len(accounts) < 3:
        return {
            "bundle_detected": 0,
            "bundle_count": 0,
            "bundle_wallets": 0,
            "bundle_pct": 0.0,
        }

    # Sort by amount descending
    sorted_accs = sorted(accounts, key=lambda x: x["amount"], reverse=True)

    # Filter out dust (bottom 1% by balance)
    if len(sorted_accs) > 10:
        threshold = sorted_accs[0]["amount"] * 0.0001  # 0.01% of largest
        sorted_accs = [a for a in sorted_accs if a["amount"] > threshold]

    if len(sorted_accs) < 3:
        return {
            "bundle_detected": 0,
            "bundle_count": 0,
            "bundle_wallets": 0,
            "bundle_pct": 0.0,
        }

    # Cluster consecutive similar balances (5% tolerance)
    tolerance = 0.05
    clusters = []
    current_cluster = [sorted_accs[0]]

    for i in range(1, len(sorted_accs)):
        prev_amt = current_cluster[-1]["amount"]
        curr_amt = sorted_accs[i]["amount"]

        if prev_amt > 0 and abs(curr_amt - prev_amt) / prev_amt <= tolerance:
            current_cluster.append(sorted_accs[i])
        else:
            if len(current_cluster) >= 3:
                clusters.append(current_cluster)
            current_cluster = [sorted_accs[i]]

    # Don't forget last cluster
    if len(current_cluster) >= 3:
        clusters.append(current_cluster)

    if not clusters:
        return {
            "bundle_detected": 0,
            "bundle_count": 0,
            "bundle_wallets": 0,
            "bundle_pct": 0.0,
        }

    bundle_wallets = sum(len(c) for c in clusters)
    bundle_amount = sum(a["amount"] for c in clusters for a in c)

    # Compute bundle_pct as % of total supply held by bundle wallets
    if total_supply and total_supply > 0:
        bundle_pct = (bundle_amount / total_supply) * 100
    else:
        # Estimate from all known holders
        total_known = sum(a["amount"] for a in accounts)
        bundle_pct = (bundle_amount / max(1, total_known)) * 100

    return {
        "bundle_detected": 1,
        "bundle_count": len(clusters),
        "bundle_wallets": bundle_wallets,
        "bundle_pct": round(min(100.0, bundle_pct), 2),
    }


def _analyze_holder_quality(accounts: list[dict]) -> dict:
    """
    Analyze holder distribution quality from token accounts.
    Returns helius_holder_count, helius_top5_pct, helius_top20_pct, helius_gini.
    """
    if not accounts:
        return {
            "helius_holder_count": None,
            "helius_top5_pct": None,
            "helius_top20_pct": None,
            "helius_gini": None,
        }

    amounts = [a["amount"] for a in accounts]
    total = sum(amounts)
    if total == 0:
        return {
            "helius_holder_count": len(accounts),
            "helius_top5_pct": None,
            "helius_top20_pct": None,
            "helius_gini": None,
        }

    # Sort descending for top-N calculations
    sorted_amounts = sorted(amounts, reverse=True)

    top5_sum = sum(sorted_amounts[:5])
    top20_sum = sum(sorted_amounts[:20])

    return {
        "helius_holder_count": len(accounts),
        "helius_top5_pct": round((top5_sum / total) * 100, 2),
        "helius_top20_pct": round((top20_sum / total) * 100, 2),
        "helius_gini": round(_compute_gini_coefficient(amounts), 4),
    }


def _analyze_transactions(signatures: list[dict]) -> dict:
    """
    Analyze recent transaction signatures for activity metrics.
    Returns helius_recent_tx_count, helius_unique_buyers.
    """
    if not signatures:
        return {
            "helius_recent_tx_count": None,
            "helius_unique_buyers": None,
        }

    tx_count = len(signatures)

    # Count unique signers (approximation of unique participants)
    # getSignaturesForAddress doesn't return signer details directly,
    # but we can count distinct signatures as a proxy for activity
    unique_sigs = set()
    for sig in signatures:
        s = sig.get("signature")
        if s:
            unique_sigs.add(s)

    return {
        "helius_recent_tx_count": tx_count,
        "helius_unique_buyers": len(unique_sigs),
    }


def _detect_jito_bundles(signatures: list[dict]) -> dict:
    """
    Detect Jito bundles: multiple transactions in the same Solana slot.
    Jito bundles execute atomically — 5+ txns in one slot = definite bundle.
    3-4 txns = suspicious. This is far more reliable than balance clustering.
    """
    if not signatures:
        return {
            "jito_bundle_detected": 0,
            "jito_bundle_slots": 0,
            "jito_max_slot_txns": 0,
        }

    slot_counts: dict[int, int] = {}
    for sig in signatures:
        slot = sig.get("slot")
        if slot is not None:
            slot_counts[slot] = slot_counts.get(slot, 0) + 1

    if not slot_counts:
        return {
            "jito_bundle_detected": 0,
            "jito_bundle_slots": 0,
            "jito_max_slot_txns": 0,
        }

    # Slots with 3+ transactions are potential bundles
    bundle_slots = {s: c for s, c in slot_counts.items() if c >= 3}
    max_slot_txns = max(slot_counts.values())

    return {
        "jito_bundle_detected": 1 if max_slot_txns >= 5 else 0,
        "jito_bundle_slots": len(bundle_slots),
        "jito_max_slot_txns": max_slot_txns,
    }


def _compute_onchain_bsr(accounts: list[dict], signatures: list[dict] | None) -> float | None:
    """
    Compute a simple on-chain buy/sell ratio proxy.
    Based on holder distribution shape: more small holders = more organic buying.
    Returns float in [0, 1] or None.
    """
    if not accounts or len(accounts) < 10:
        return None

    amounts = sorted([a["amount"] for a in accounts], reverse=True)
    total = sum(amounts)
    if total == 0:
        return None

    # Ratio of holders with < 1% of supply (retail buyers)
    one_pct = total * 0.01
    retail_count = sum(1 for a in amounts if a < one_pct)
    retail_ratio = retail_count / len(amounts)

    # More retail = healthier buy pressure (0.3 to 0.9 range)
    bsr = 0.3 + retail_ratio * 0.6
    return round(min(1.0, max(0.0, bsr)), 3)


# === Public API ===

def _analyze_whales(accounts: list[dict], cache: dict, mint: str) -> dict:
    """
    Identify whale holders (>1% of supply) and track changes between cycles.

    Uses the cache to compare current whale holdings vs previous cycle.
    Stores whale data under 'whales_{mint}' key in cache for next comparison.

    Returns whale_count, whale_total_pct, whale_change, whale_new_entries.
    """
    if not accounts:
        return {
            "whale_count": None,
            "whale_total_pct": None,
            "whale_change": None,
            "whale_new_entries": None,
        }

    total_supply = sum(a["amount"] for a in accounts)
    if total_supply == 0:
        return {
            "whale_count": 0,
            "whale_total_pct": 0.0,
            "whale_change": None,
            "whale_new_entries": None,
        }

    one_pct = total_supply * 0.01

    # Current whales: holders with >1% of supply
    current_whales: dict[str, float] = {}
    for acc in accounts:
        if acc["amount"] >= one_pct:
            pct = (acc["amount"] / total_supply) * 100
            current_whales[acc["owner"]] = round(pct, 2)

    whale_count = len(current_whales)
    whale_total_pct = round(sum(current_whales.values()), 2)

    # Compare with previous cycle
    whale_cache_key = f"whales_{mint}"
    prev_entry = cache.get(whale_cache_key)
    whale_change = None
    whale_new_entries = None

    if prev_entry and isinstance(prev_entry, dict):
        prev_whales = prev_entry.get("whales", {})
        prev_total = prev_entry.get("total_pct", 0)

        if prev_total is not None:
            whale_change = round(whale_total_pct - prev_total, 2)

        # Count new whale addresses (not in previous set)
        prev_addrs = set(prev_whales.keys())
        curr_addrs = set(current_whales.keys())
        whale_new_entries = len(curr_addrs - prev_addrs)

    # Algorithm v4: Track whale_change history (last 3 cycles) for direction trend
    whale_change_history = prev_entry.get("whale_change_history", []) if prev_entry else []
    if whale_change is not None:
        whale_change_history.append(whale_change)
    # Keep only the last 3 entries
    whale_change_history = whale_change_history[-3:]

    # Compute whale direction trend from history
    whale_direction = "unknown"
    if len(whale_change_history) >= 2:
        if all(wc > 0 for wc in whale_change_history):
            whale_direction = "accumulating"    # Consistently buying — very bullish
        elif whale_change_history[-1] < 0 and whale_change_history[0] > 0:
            whale_direction = "distributing"    # Was buying, now selling — DANGER
        elif all(wc < 0 for wc in whale_change_history):
            whale_direction = "dumping"         # Consistent selling — bearish
        elif all(abs(wc) < 2 for wc in whale_change_history):
            whale_direction = "holding"         # Stable — neutral
        else:
            whale_direction = "mixed"

    # Store for next cycle (no TTL — we want to compare across cycles)
    cache[whale_cache_key] = {
        "whales": current_whales,
        "total_pct": whale_total_pct,
        "whale_change_history": whale_change_history,
        "_cached_at": time.time(),
    }

    return {
        "whale_count": whale_count,
        "whale_total_pct": whale_total_pct,
        "whale_change": whale_change,
        "whale_new_entries": whale_new_entries,
        "whale_direction": whale_direction,
    }


def _empty_helius_result() -> dict:
    """Return a dict with all Helius fields set to None."""
    return {
        "helius_holder_count": None,
        "helius_top5_pct": None,
        "helius_top20_pct": None,
        "helius_gini": None,
        "bundle_detected": None,
        "bundle_count": None,
        "bundle_pct": None,
        "helius_recent_tx_count": None,
        "helius_unique_buyers": None,
        "helius_onchain_bsr": None,
        # Jito slot-based bundle detection
        "jito_bundle_detected": None,
        "jito_bundle_slots": None,
        "jito_max_slot_txns": None,
        "whale_count": None,
        "whale_total_pct": None,
        "whale_change": None,
        "whale_new_entries": None,
        # Algorithm v4: Whale direction tracking
        "whale_direction": None,
    }


def enrich_token_helius(
    mint: str,
    api_key: str,
    cache: dict,
    fetch_signatures: bool = False,
) -> dict:
    """
    Enrich a single token with Helius on-chain data.

    Parameters
    ----------
    mint : Solana token mint address
    api_key : Helius API key
    cache : shared cache dict (mutated in-place)
    fetch_signatures : whether to also fetch recent signatures (extra 10 credits)
    """
    if not mint or mint.startswith("0x") or len(mint) < 32 or len(mint) > 44:
        return _empty_helius_result()

    # Check cache
    cache_key = f"helius_{mint}"
    if cache_key in cache and _is_cache_fresh(cache[cache_key]):
        cached = dict(cache[cache_key])
        cached.pop("_cached_at", None)
        logger.debug("Helius cache hit for %s", mint[:8])
        return cached

    result = _empty_helius_result()

    # Fetch token accounts (holder analysis)
    _helius_rate_limit()
    accounts = _fetch_token_accounts(mint, api_key)

    if accounts:
        # Holder quality metrics
        quality = _analyze_holder_quality(accounts)
        result.update(quality)

        # Bundle detection
        bundles = _detect_bundles(accounts)
        result.update(bundles)

        # Whale tracking (reuses holder data, no extra API calls)
        whale_data = _analyze_whales(accounts, cache, mint)
        result.update(whale_data)

        # On-chain BSR
        # Always fetch signatures for Jito bundle detection (10 credits, negligible)
        _helius_rate_limit()
        signatures = _fetch_recent_signatures(mint, api_key)

        if signatures:
            tx_metrics = _analyze_transactions(signatures)
            result.update(tx_metrics)

            # Slot-based Jito bundle detection (uses slot field from signatures)
            jito_data = _detect_jito_bundles(signatures)
            result.update(jito_data)

        bsr = _compute_onchain_bsr(accounts, signatures)
        if bsr is not None:
            result["helius_onchain_bsr"] = bsr

    # Only cache successful results — don't poison cache with transient failures
    if result.get("helius_holder_count") is not None:
        cache_entry = dict(result)
        cache_entry["_cached_at"] = time.time()
        cache[cache_key] = cache_entry

    return result


def enrich_tokens_helius(ranking: list[dict]) -> list[dict]:
    """
    Enrich top tokens with Helius on-chain data (bundle detection, holder quality).

    - Top HELIUS_TOP_N tokens: full holder analysis + bundles
    - Top HELIUS_SMART_MONEY_N tokens: also get transaction signatures

    v34: Parallel processing with ThreadPoolExecutor (3 workers).
    Rate limited via _helius_rate_limit() to stay under 10 RPS.
    Modifies tokens in-place and returns the list.
    """
    api_key = _get_api_key()
    if not api_key:
        logger.info("HELIUS_API_KEY not set — skipping Helius enrichment")
        return ranking

    if not ranking:
        return ranking

    cache = _load_cache()
    enriched_count = 0

    # Build work items: (index, mint, fetch_sigs)
    work_items = []
    for i, token in enumerate(ranking):
        if i >= HELIUS_TOP_N:
            break
        mint = token.get("token_address")
        if not mint:
            continue
        fetch_sigs = i < HELIUS_SMART_MONEY_N
        work_items.append((i, mint, fetch_sigs))

    def _process_token(item):
        idx, mint, fetch_sigs = item
        return idx, enrich_token_helius(mint, api_key, cache, fetch_signatures=fetch_sigs)

    # v34: 3 workers × rate limiter (0.12s min between calls) ≈ 8 RPS, under Helius 10 RPS limit
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="helius") as pool:
        futures = {pool.submit(_process_token, item): item for item in work_items}
        for fut in as_completed(futures):
            try:
                idx, data = fut.result()
                ranking[idx].update(data)
                if data.get("helius_holder_count") is not None:
                    enriched_count += 1
            except Exception as e:
                item = futures[fut]
                logger.error("Helius enrichment failed for %s: %s", item[1][:8], e)

    _save_cache(cache)
    logger.info(
        "Helius enriched %d/%d tokens (top %d analyzed, top %d with signatures)",
        enriched_count, len(ranking), HELIUS_TOP_N, HELIUS_SMART_MONEY_N,
    )
    return ranking
