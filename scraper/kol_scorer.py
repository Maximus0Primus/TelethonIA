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
import math
import time
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path

from supabase import create_client

logger = logging.getLogger(__name__)

CACHE_FILE = Path(__file__).parent / "kol_scores.json"
CACHE_TTL = 6 * 3600  # Refresh every 6h — faster iteration on algo changes


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


def compute_kol_scores(min_calls: int = 3) -> dict[str, float]:
    """
    Compute each KOL's historical 2x hit rate from token_snapshots.

    Uses first-call-per-token methodology: for each (KOL, token) pair,
    only the earliest snapshot counts. A hit = 2x within ANY tracked
    horizon (12h, 24h, 48h, 72h, 7d). A KOL shouldn't be penalized
    because their call took longer to pump.

    Parameters
    ----------
    min_calls : minimum tokens a KOL must have called to get a score
                (avoids noisy 1/1 = 100% from a single call)

    Returns
    -------
    { "kol_username": normalized_score }  (1.0 = average, >1.0 = above avg)
    """
    client = _get_client()

    # Fetch snapshots with outcomes and KOL data
    # v17: symbol for dedup, did_2x_24h for multi-horizon scoring
    all_rows = []
    page_size = 1000
    offset = 0

    while True:
        result = (
            client.table("token_snapshots")
            .select("top_kols, did_2x_12h, did_2x_24h, did_2x_48h, did_2x_72h, did_2x_7d, snapshot_at, symbol, token_address, has_ca_mention")
            .not_.is_("top_kols", "null")
            # Accept rows where ANY horizon has outcome data
            .or_("did_2x_12h.not.is.null,did_2x_24h.not.is.null,did_2x_48h.not.is.null,did_2x_72h.not.is.null,did_2x_7d.not.is.null")
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

    now = datetime.now(timezone.utc)

    # Exponential decay — half-life of 10 days
    HALF_LIFE_DAYS = 10.0

    def _parse_snapshot_dt(snap_at: str) -> datetime | None:
        """Parse snapshot_at string to timezone-aware datetime."""
        try:
            if snap_at.endswith("Z"):
                return datetime.fromisoformat(snap_at.replace("Z", "+00:00"))
            elif "+" in snap_at:
                return datetime.fromisoformat(snap_at)
            else:
                return datetime.fromisoformat(snap_at).replace(tzinfo=timezone.utc)
        except (ValueError, AttributeError):
            return None

    # --- Step 1: For each (kol, token_address) pair, keep ONLY the earliest snapshot ---
    # A token ranked for 12h creates ~24 snapshots; we count it once.
    # Dedup by token_address (not symbol) to avoid merging different contracts
    # sharing the same ticker (e.g. 3 different $LUNA contracts).
    first_calls: dict[tuple[str, str], dict] = {}  # (kol, token_address) -> {dt, did_2x_*}

    for row in all_rows:
        kols = row.get("top_kols")
        did_2x_12h = bool(row.get("did_2x_12h")) if row.get("did_2x_12h") is not None else None
        did_2x_24h = bool(row.get("did_2x_24h")) if row.get("did_2x_24h") is not None else None
        did_2x_48h = bool(row.get("did_2x_48h")) if row.get("did_2x_48h") is not None else None
        did_2x_72h = bool(row.get("did_2x_72h")) if row.get("did_2x_72h") is not None else None
        did_2x_7d = bool(row.get("did_2x_7d")) if row.get("did_2x_7d") is not None else None
        symbol = row.get("symbol", "")
        token_addr = row.get("token_address") or symbol  # fallback to symbol
        has_ca = bool(row.get("has_ca_mention"))

        if isinstance(kols, str):
            try:
                kols = json.loads(kols)
            except (json.JSONDecodeError, TypeError):
                continue

        if not isinstance(kols, list) or not token_addr:
            continue

        snap_at = row.get("snapshot_at", "")
        snap_dt = _parse_snapshot_dt(snap_at) if snap_at else None

        for kol in kols:
            if not isinstance(kol, str):
                continue
            key = (kol, token_addr)
            if key not in first_calls or (snap_dt and snap_dt < first_calls[key]["dt"]):
                first_calls[key] = {
                    "dt": snap_dt or datetime.min.replace(tzinfo=timezone.utc),
                    "symbol": symbol,
                    "did_2x_12h": did_2x_12h,
                    "did_2x_24h": did_2x_24h,
                    "did_2x_48h": did_2x_48h,
                    "did_2x_72h": did_2x_72h,
                    "did_2x_7d": did_2x_7d,
                    "has_ca": has_ca,
                }

    logger.info("Deduplicated to %d first-calls from %d snapshot rows",
                len(first_calls), len(all_rows))

    # --- Step 2: Count from first_calls — hit = 2x in 12h OR 24h ---
    kol_calls: dict[str, float] = {}
    kol_hits: dict[str, float] = {}
    # CA-only tracking
    kol_calls_ca: dict[str, float] = {}
    kol_hits_ca: dict[str, float] = {}
    # Per-horizon stats for verbose output
    # Per-horizon hit tracking
    TRACKED_HORIZONS = ["12h", "24h", "48h", "72h", "7d"]
    kol_hits_by_hz: dict[str, dict[str, float]] = {hz: {} for hz in TRACKED_HORIZONS}
    kol_tokens: dict[str, list[tuple[str, bool]]] = {}  # kol -> [(symbol, hit)]

    for (kol, _token_addr), call in first_calls.items():
        # Skip if no outcome data at all
        if all(call.get(f"did_2x_{hz}") is None for hz in TRACKED_HORIZONS):
            continue

        days_ago = (now - call["dt"]).total_seconds() / 86400
        weight = math.exp(-0.693 * days_ago / HALF_LIFE_DAYS)

        hits = {hz: call.get(f"did_2x_{hz}") is True for hz in TRACKED_HORIZONS}
        hit_any = any(hits.values())
        has_ca = call.get("has_ca", False)
        display_symbol = call.get("symbol", _token_addr)

        kol_calls[kol] = kol_calls.get(kol, 0) + weight
        if hit_any:
            kol_hits[kol] = kol_hits.get(kol, 0) + weight
        for hz in TRACKED_HORIZONS:
            if hits[hz]:
                kol_hits_by_hz[hz][kol] = kol_hits_by_hz[hz].get(kol, 0) + weight

        # CA-only tracking
        if has_ca:
            kol_calls_ca[kol] = kol_calls_ca.get(kol, 0) + weight
            if hit_any:
                kol_hits_ca[kol] = kol_hits_ca.get(kol, 0) + weight

        kol_tokens.setdefault(kol, []).append((display_symbol, hit_any))

    # Compute hit rates (only for KOLs with enough weighted calls)
    raw_rates = {}
    for kol, total in kol_calls.items():
        if total >= min_calls:
            hits = kol_hits.get(kol, 0)
            raw_rates[kol] = hits / total

    # Normalize: score = hit_rate / baseline so that:
    #   1.0 = average performance (unscored KOL default)
    #   >1.0 = better than average, <1.0 = worse
    # This ensures kol_scores.get(kol, 1.0) correctly weights breadth.
    total_hits = sum(kol_hits.values())
    total_calls = sum(kol_calls.values())
    baseline = total_hits / total_calls if total_calls > 0 else 0.1
    if baseline <= 0:
        baseline = 0.1  # safety floor

    scores = {}
    for kol, rate in raw_rates.items():
        normalized = rate / baseline
        scores[kol] = round(max(0.1, min(3.0, normalized)), 3)  # cap [0.1, 3.0]

    logger.info(
        "Computed scores for %d KOLs (from %d with <%d calls filtered, baseline=%.1f%%)",
        len(scores), len(kol_calls), min_calls, baseline * 100,
    )

    # Build per-KOL detail for cache (useful for dashboard / debugging)
    kol_details = {}
    for kol in scores:
        total_w = kol_calls.get(kol, 0)
        hits_w = kol_hits.get(kol, 0)
        tokens = kol_tokens.get(kol, [])
        ca_w = kol_calls_ca.get(kol, 0)
        ca_hits_w = kol_hits_ca.get(kol, 0)
        detail = {
            "score": scores[kol],
            "raw_hit_rate": round(raw_rates.get(kol, 0), 4),
            "tokens_called": len(tokens),
            "tokens_hit": sum(1 for _, h in tokens if h),
            "ca_calls": round(ca_w, 2),
            "ca_hits": round(ca_hits_w, 2),
            "hit_rate_ca_only": round(ca_hits_w / ca_w, 4) if ca_w >= 1 else None,
        }
        for hz in TRACKED_HORIZONS:
            detail[f"hit_rate_{hz}"] = round(kol_hits_by_hz[hz].get(kol, 0) / total_w, 4) if total_w > 0 else 0
        kol_details[kol] = detail

    # Save to cache
    cache_data = dict(scores)
    cache_data["_computed_at"] = time.time()
    cache_data["_total_snapshots"] = len(all_rows)
    cache_data["_total_first_calls"] = len(first_calls)
    cache_data["_total_kols"] = len(kol_calls)
    cache_data["_scored_kols"] = len(scores)
    cache_data["_baseline"] = round(baseline, 4)
    cache_data["_kol_details"] = kol_details

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
    parser.add_argument("--force", action="store_true", help="Force recompute (ignore cache)")
    args = parser.parse_args()

    if args.force and CACHE_FILE.exists():
        CACHE_FILE.unlink()
        logger.info("Cache cleared — forcing recompute")

    scores = compute_kol_scores(min_calls=args.min_calls)

    if args.verbose:
        # Load details from cache for rich output
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
        details = cache.get("_kol_details", {})
        baseline = cache.get("_baseline", 0)
        total_snaps = cache.get("_total_snapshots", 0)
        total_fc = cache.get("_total_first_calls", 0)

        print(f"\n{'=' * 80}")
        print(f"  KOL REPUTATION SCORES -- First-Call-Per-Token (12h to 7d)")
        print(f"{'=' * 80}")
        print(f"  Snapshots: {total_snaps}  ->  First-calls: {total_fc}  "
              f"(dedup {total_snaps/max(1,total_fc):.1f}x)")
        print(f"  Baseline hit rate: {baseline*100:.1f}%  |  1.0 = average\n")

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        hz_labels = ["12h", "24h", "48h", "72h", "7d"]
        hz_header = "  ".join(f"{h+'%':>5s}" for h in hz_labels)
        hz_dashes = "  ".join(f"{'-'*5}" for _ in hz_labels)
        print(f"  {'KOL':30s} {'Score':>6s}  {'Tkns':>4s}  {'Hits':>4s}  "
              f"{'HR%':>5s}  {hz_header}  {'CA%':>5s}")
        print(f"  {'-'*30} {'-'*6}  {'-'*4}  {'-'*4}  {'-'*5}  {hz_dashes}  {'-'*5}")

        for kol, score in sorted_scores:
            d = details.get(kol, {})
            tokens = d.get("tokens_called", "?")
            hits = d.get("tokens_hit", "?")
            hr = d.get("raw_hit_rate", 0)
            hz_vals = "  ".join(
                f"{d.get(f'hit_rate_{h}', 0)*100:4.1f}%" for h in hz_labels
            )
            ca_hr = d.get("hit_rate_ca_only")
            ca_str = f"{ca_hr*100:4.1f}%" if ca_hr is not None else "  N/A"
            print(f"  {kol:30s} {score:5.2f}x  {tokens:>4}  {hits:>4}  "
                  f"{hr*100:4.1f}%  {hz_vals}  {ca_str}")

        print(f"\n{'=' * 80}\n")
    else:
        print(f"Computed scores for {len(scores)} KOLs → {CACHE_FILE}")


if __name__ == "__main__":
    main()
