"""
KOL Reputation Scorer — compute each KOL's historical 2x hit rate.

Queries token_snapshots (which now stores top_kols) to measure which KOLs
actually call winners. Results are cached to kol_scores.json and refreshed
once per day (or per training cycle).

Usage:
    python kol_scorer.py              # compute & save scores
    python kol_scorer.py --verbose    # show per-KOL breakdown
"""

import os
import json
import time
import logging
import argparse
from pathlib import Path

from supabase import create_client

logger = logging.getLogger(__name__)

CACHE_FILE = Path(__file__).parent / "kol_scores.json"
CACHE_TTL = 24 * 3600  # Refresh once per day


def _get_client():
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return create_client(url, key)


def load_cached_scores() -> dict[str, float]:
    """Load cached KOL scores from disk. Returns { "kol_username": hit_rate }."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r") as f:
                data = json.load(f)
            computed_at = data.get("_computed_at", 0)
            if time.time() - computed_at < CACHE_TTL:
                # Return only actual scores (skip metadata keys)
                return {k: v for k, v in data.items() if not k.startswith("_")}
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def compute_kol_scores(min_calls: int = 10) -> dict[str, float]:
    """
    Compute each KOL's historical 2x hit rate from token_snapshots.

    Only snapshots with:
    - did_2x_12h not null (has outcome)
    - top_kols not null (has KOL attribution)

    Parameters
    ----------
    min_calls : minimum tokens a KOL must have called to get a score
                (avoids noisy 1/1 = 100% from a single call)

    Returns
    -------
    { "kol_username": hit_rate_0_to_1 }
    """
    client = _get_client()

    # Fetch snapshots that have both outcomes and KOL data
    # Paginate in case of large dataset (Supabase default limit = 1000)
    all_rows = []
    page_size = 1000
    offset = 0

    while True:
        result = (
            client.table("token_snapshots")
            .select("top_kols, did_2x_12h")
            .not_.is_("did_2x_12h", "null")
            .not_.is_("top_kols", "null")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        rows = result.data or []
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        offset += page_size

    if not all_rows:
        logger.warning("No snapshots with outcomes + KOL data found")
        return {}

    logger.info("Processing %d snapshots with KOL data", len(all_rows))

    # Count hits and total calls per KOL
    kol_calls: dict[str, int] = {}
    kol_hits: dict[str, int] = {}

    for row in all_rows:
        kols = row.get("top_kols")
        did_2x = bool(row.get("did_2x_12h"))

        # Handle JSON string from DB (top_kols stored via json.dumps)
        if isinstance(kols, str):
            try:
                kols = json.loads(kols)
            except (json.JSONDecodeError, TypeError):
                continue

        if not isinstance(kols, list):
            continue

        for kol in kols:
            if not isinstance(kol, str):
                continue
            kol_calls[kol] = kol_calls.get(kol, 0) + 1
            if did_2x:
                kol_hits[kol] = kol_hits.get(kol, 0) + 1

    # Compute hit rates (only for KOLs with enough calls)
    scores = {}
    for kol, total in kol_calls.items():
        if total >= min_calls:
            hits = kol_hits.get(kol, 0)
            scores[kol] = round(hits / total, 3)

    logger.info(
        "Computed scores for %d KOLs (from %d with <%d calls filtered)",
        len(scores), len(kol_calls), min_calls,
    )

    # Save to cache
    cache_data = dict(scores)
    cache_data["_computed_at"] = time.time()
    cache_data["_total_snapshots"] = len(all_rows)
    cache_data["_total_kols"] = len(kol_calls)
    cache_data["_scored_kols"] = len(scores)

    with open(CACHE_FILE, "w") as f:
        json.dump(cache_data, f, indent=2)

    logger.info("KOL scores saved to %s", CACHE_FILE)
    return scores


def get_kol_scores() -> dict[str, float]:
    """Get KOL scores (from cache if fresh, recompute otherwise)."""
    scores = load_cached_scores()
    if scores:
        return scores
    try:
        return compute_kol_scores()
    except Exception as e:
        logger.warning("Failed to compute KOL scores: %s — using empty scores", e)
        return {}


def main():
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Compute KOL reputation scores")
    parser.add_argument("--verbose", action="store_true", help="Show per-KOL breakdown")
    parser.add_argument("--min-calls", type=int, default=3, help="Minimum calls to score a KOL")
    args = parser.parse_args()

    scores = compute_kol_scores(min_calls=args.min_calls)

    if args.verbose:
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        print(f"\nKOL Reputation Scores ({len(sorted_scores)} KOLs):")
        print("-" * 45)
        for kol, rate in sorted_scores:
            print(f"  {kol:30s} {rate:.1%}")
    else:
        print(f"Computed scores for {len(scores)} KOLs → {CACHE_FILE}")


if __name__ == "__main__":
    main()
