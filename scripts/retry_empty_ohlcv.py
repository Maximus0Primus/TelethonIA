"""
Retry empty OHLCV cache files using Birdeye (token mint fallback).

Scans ohlcv_cache/ for empty [] files, resolves pool→token via the pair cache,
and retries fetching via Birdeye which uses token mint (not pool address).

Usage:
    python scripts/retry_empty_ohlcv.py                # dry run (show counts)
    python scripts/retry_empty_ohlcv.py --run           # actually fetch
    python scripts/retry_empty_ohlcv.py --run --limit 200  # limit API calls
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

SCRAPER_DIR = Path(__file__).resolve().parent.parent / "scraper"
CACHE_DIR = SCRAPER_DIR / "ohlcv_cache"
PAIR_CACHE_FILE = CACHE_DIR / "_pair_address_cache.json"

load_dotenv(SCRAPER_DIR / ".env")
BIRDEYE_API_KEY = os.environ.get("BIRDEYE_API_KEY")

MAX_WINDOW_MIN = 365


def load_pair_cache() -> dict[str, str | None]:
    if PAIR_CACHE_FILE.exists():
        try:
            return json.loads(PAIR_CACHE_FILE.read_text())
        except (json.JSONDecodeError, KeyError):
            pass
    return {}


def invert_pair_cache(pair_cache: dict[str, str | None]) -> dict[str, str]:
    """pool_address → token_address (first match)."""
    inv = {}
    for token, pool in pair_cache.items():
        if pool and pool not in inv:
            inv[pool] = token
    return inv


def find_empty_cache_files() -> list[dict]:
    """Find all cache files that contain [] (empty candles)."""
    empty = []
    for f in CACHE_DIR.glob("*.json"):
        if f.name.startswith("_"):
            continue
        try:
            data = json.loads(f.read_text())
            if isinstance(data, list) and len(data) == 0:
                empty.append({"path": f, "name": f.name})
        except (json.JSONDecodeError, OSError):
            continue
    return empty


def parse_cache_filename(name: str) -> dict | None:
    """Extract pool_prefix and start_ts from cache filename."""
    # Format: {pool[:12]}_{start_ts}_{window}_{hash}.json
    parts = name.replace(".json", "").split("_")
    if len(parts) < 4:
        return None
    pool_prefix = parts[0]
    try:
        start_ts = int(parts[1])
        window = int(parts[2])
    except (ValueError, IndexError):
        return None
    return {"pool_prefix": pool_prefix, "start_ts": start_ts, "window": window}


def fetch_birdeye(token_mint: str, start_ts: int, end_ts: int) -> list[dict] | None:
    if not BIRDEYE_API_KEY:
        return None
    try:
        r = requests.get(
            "https://public-api.birdeye.so/defi/ohlcv",
            params={"address": token_mint, "type": "15m",
                    "time_from": start_ts, "time_to": end_ts},
            headers={"X-API-KEY": BIRDEYE_API_KEY, "x-chain": "solana"},
            timeout=15,
        )
        if r.status_code == 429:
            time.sleep(5)
            r = requests.get(
                "https://public-api.birdeye.so/defi/ohlcv",
                params={"address": token_mint, "type": "15m",
                        "time_from": start_ts, "time_to": end_ts},
                headers={"X-API-KEY": BIRDEYE_API_KEY, "x-chain": "solana"},
                timeout=15,
            )
            if r.status_code == 429:
                return None
        if r.status_code != 200:
            return None
        items = r.json().get("data", {}).get("items", [])
        if not items:
            return None
        candles = []
        for c in items:
            h, l, o, cl, v = c.get("h", 0), c.get("l", 0), c.get("o", 0), c.get("c", 0), c.get("v", 0)
            ts = int(c.get("unixTime", 0))
            if h > 0 and l > 0 and ts >= start_ts - 60:
                candles.append({"timestamp": ts, "open": float(o), "high": float(h),
                                "low": float(l), "close": float(cl), "volume": float(v)})
        candles.sort(key=lambda x: x["timestamp"])
        return candles if len(candles) >= 3 else None
    except Exception as e:
        print(f"  Birdeye error for {token_mint[:12]}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Retry empty OHLCV cache files via Birdeye")
    parser.add_argument("--run", action="store_true", help="Actually fetch (default: dry run)")
    parser.add_argument("--limit", type=int, default=500, help="Max API calls")
    args = parser.parse_args()

    if not BIRDEYE_API_KEY:
        print("ERROR: BIRDEYE_API_KEY not set in environment")
        sys.exit(1)

    # Load pair cache and build reverse mapping
    pair_cache = load_pair_cache()
    pool_to_token = invert_pair_cache(pair_cache)
    print(f"Pair cache: {len(pair_cache)} entries, {len(pool_to_token)} reverse mappings")

    # Find empty cache files
    empties = find_empty_cache_files()
    print(f"Empty cache files: {len(empties)}")

    # Match empty files to tokens
    retryable = []
    unresolvable = 0
    for entry in empties:
        parsed = parse_cache_filename(entry["name"])
        if not parsed:
            continue
        # Find the full pool address that starts with this prefix
        matching_pool = None
        matching_token = None
        for pool, token in pool_to_token.items():
            if pool[:12] == parsed["pool_prefix"]:
                matching_pool = pool
                matching_token = token
                break
        if not matching_token:
            unresolvable += 1
            continue
        retryable.append({
            **entry, **parsed,
            "pool": matching_pool,
            "token": matching_token,
        })

    print(f"Retryable (have token mint): {len(retryable)}")
    print(f"Unresolvable (no reverse mapping): {unresolvable}")

    if not args.run:
        print("\n[DRY RUN] Use --run to actually fetch. Exiting.")
        return

    # Retry via Birdeye
    recovered = 0
    still_empty = 0
    api_calls = 0

    for i, entry in enumerate(retryable):
        if api_calls >= args.limit:
            print(f"\nReached limit of {args.limit} API calls. Stopping.")
            break

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(retryable)}] recovered={recovered}, still_empty={still_empty}, api={api_calls}")

        start_ts = entry["start_ts"]
        end_ts = start_ts + entry["window"] * 60

        candles = fetch_birdeye(entry["token"], start_ts, end_ts)
        api_calls += 1
        time.sleep(0.5)  # conservative rate limit

        if candles:
            # Overwrite the empty cache file with real data
            entry["path"].write_text(json.dumps(candles))
            recovered += 1
        else:
            still_empty += 1

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"API calls:    {api_calls}")
    print(f"Recovered:    {recovered}")
    print(f"Still empty:  {still_empty}")
    print(f"Birdeye CUs:  ~{api_calls * 30} / 30,000 monthly")


if __name__ == "__main__":
    main()
