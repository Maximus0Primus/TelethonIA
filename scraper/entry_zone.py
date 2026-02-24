"""
ML v2 Phase C: Entry Zone Detection — Winner Profile Analysis.

Analyzes tokens that achieved 2x to identify temporal profiles
at their snapshot moment. Used to calibrate entry_timing_quality
thresholds in pipeline.py.

Can be run standalone for analysis:
    python entry_zone.py

Produces a JSON report of winner clusters (cached 24h).
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CACHE_PATH = Path(__file__).parent / "winner_profiles.json"
CACHE_TTL_HOURS = 24


def _get_client():
    from supabase import create_client
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return create_client(url, key)


def _load_cached_profiles() -> dict | None:
    """Load cached profiles if fresh enough."""
    if not CACHE_PATH.exists():
        return None
    try:
        with open(CACHE_PATH) as f:
            data = json.load(f)
        cached_at = datetime.fromisoformat(data.get("generated_at", "2000-01-01"))
        if datetime.now(timezone.utc) - cached_at.replace(tzinfo=timezone.utc) < timedelta(hours=CACHE_TTL_HOURS):
            return data
    except Exception:
        pass
    return None


def analyze_winner_profiles(client=None) -> dict:
    """
    Cluster analysis on winners (did_2x_12h == True) to identify
    common temporal profiles at snapshot time.

    Returns dict with:
    - profiles: list of cluster descriptions
    - thresholds: recommended entry_timing_quality calibration
    - stats: winner vs loser feature means
    """
    cached = _load_cached_profiles()
    if cached:
        logger.info("Using cached winner profiles (age < %dh)", CACHE_TTL_HOURS)
        return cached

    if client is None:
        client = _get_client()

    # Fetch winners and losers
    cols = (
        "symbol, snapshot_at, did_2x_12h, "
        "freshest_mention_hours, price_change_24h, activity_mult, "
        "score_velocity, mention_velocity, volume_velocity, "
        "market_cap, price_at_snapshot, max_price_12h, "
        "score_at_snapshot, social_momentum_phase, entry_timing_quality"
    )

    all_rows = []
    offset = 0
    page_size = 1000
    while True:
        result = (
            client.table("token_snapshots")
            .select(cols)
            .not_.is_("did_2x_12h", "null")
            .gte("snapshot_at", "2026-02-14T00:00:00Z")  # skip pre-v34 poisoned data
            .order("snapshot_at", desc=True)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        if not result.data:
            break
        all_rows.extend(result.data)
        if len(result.data) < page_size:
            break
        offset += page_size
        if offset >= 50000:
            break

    if not all_rows:
        logger.warning("No labeled data for winner analysis")
        return {"profiles": [], "stats": {}}

    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        logger.error("pandas/numpy required for winner analysis")
        return {"profiles": [], "stats": {}}

    df = pd.DataFrame(all_rows)
    df["snapshot_at"] = pd.to_datetime(df["snapshot_at"])

    # First-appearance per token (dedup by token_address, fallback to symbol)
    dedup_col = "token_address" if "token_address" in df.columns and df["token_address"].notna().any() else "symbol"
    first = df.sort_values("snapshot_at").drop_duplicates(subset=[dedup_col], keep="first")
    winners = first[first["did_2x_12h"] == True].copy()
    losers = first[first["did_2x_12h"] == False].copy()

    logger.info("Winner analysis: %d winners, %d losers", len(winners), len(losers))

    # Feature comparison: winners vs losers
    feature_cols = [
        "freshest_mention_hours", "price_change_24h", "activity_mult",
        "score_velocity", "mention_velocity", "volume_velocity", "market_cap",
    ]

    stats = {}
    for col in feature_cols:
        w_vals = pd.to_numeric(winners.get(col), errors="coerce").dropna()
        l_vals = pd.to_numeric(losers.get(col), errors="coerce").dropna()
        if len(w_vals) >= 3 and len(l_vals) >= 3:
            stats[col] = {
                "winner_mean": round(float(w_vals.mean()), 3),
                "winner_median": round(float(w_vals.median()), 3),
                "loser_mean": round(float(l_vals.mean()), 3),
                "loser_median": round(float(l_vals.median()), 3),
                "separation": round(float(w_vals.mean()) - float(l_vals.mean()), 3),
            }

    # Entry timing quality comparison (if available)
    etq_stats = {}
    if "entry_timing_quality" in first.columns:
        w_etq = pd.to_numeric(winners.get("entry_timing_quality"), errors="coerce").dropna()
        l_etq = pd.to_numeric(losers.get("entry_timing_quality"), errors="coerce").dropna()
        if len(w_etq) >= 3 and len(l_etq) >= 3:
            etq_stats = {
                "winner_mean": round(float(w_etq.mean()), 3),
                "loser_mean": round(float(l_etq.mean()), 3),
                "correlation_positive": float(w_etq.mean()) > float(l_etq.mean()),
            }

    # Profile classification based on temporal patterns
    profiles = []

    # Profile 1: Flash Pump — fresh mentions + high velocity
    flash = winners[
        (pd.to_numeric(winners["freshest_mention_hours"], errors="coerce") < 4) &
        (pd.to_numeric(winners["price_change_24h"], errors="coerce") < 100)
    ]
    if len(flash) >= 2:
        profiles.append({
            "name": "flash_pump",
            "description": "Fresh KOL calls (<4h) + pre-pump stage (<100% 24h change)",
            "count": len(flash),
            "pct_of_winners": round(len(flash) / len(winners), 3),
            "avg_freshest_hours": round(float(pd.to_numeric(flash["freshest_mention_hours"], errors="coerce").mean()), 1),
            "avg_pc24": round(float(pd.to_numeric(flash["price_change_24h"], errors="coerce").mean()), 1),
        })

    # Profile 2: Slow Build — older mentions + building momentum
    slow = winners[
        (pd.to_numeric(winners["freshest_mention_hours"], errors="coerce") >= 4) &
        (pd.to_numeric(winners["freshest_mention_hours"], errors="coerce") < 24) &
        (pd.to_numeric(winners.get("activity_mult", 1.0), errors="coerce") >= 1.0)
    ]
    if len(slow) >= 2:
        profiles.append({
            "name": "slow_build",
            "description": "Older calls (4-24h) but sustained activity (activity_mult >= 1.0)",
            "count": len(slow),
            "pct_of_winners": round(len(slow) / len(winners), 3),
        })

    # Profile 3: Breakout — volume velocity spike
    vol_vel_col = pd.to_numeric(winners.get("volume_velocity"), errors="coerce")
    breakout = winners[vol_vel_col > 0.5] if vol_vel_col.notna().sum() >= 3 else pd.DataFrame()
    if len(breakout) >= 2:
        profiles.append({
            "name": "volume_breakout",
            "description": "Volume accelerating (volume_velocity > 0.5)",
            "count": len(breakout),
            "pct_of_winners": round(len(breakout) / len(winners), 3),
        })

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_winners": len(winners),
        "total_losers": len(losers),
        "profiles": profiles,
        "feature_stats": stats,
        "entry_timing_quality": etq_stats,
    }

    # Cache
    try:
        with open(CACHE_PATH, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Winner profiles cached to %s", CACHE_PATH)
    except Exception as e:
        logger.warning("Failed to cache winner profiles: %s", e)

    return report


def main():
    """Run standalone winner profile analysis."""
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")

    report = analyze_winner_profiles()

    print("\n" + "=" * 60)
    print("  WINNER PROFILE ANALYSIS")
    print("=" * 60)
    print(f"\nWinners: {report['total_winners']}, Losers: {report['total_losers']}")

    if report["profiles"]:
        print("\nProfiles:")
        for p in report["profiles"]:
            print(f"  {p['name']}: {p['count']} winners ({p['pct_of_winners']*100:.0f}%)")
            print(f"    {p['description']}")

    stats = report.get("feature_stats", {})
    if stats:
        print("\nFeature Comparison (winner vs loser mean):")
        for feat, s in stats.items():
            sep = s["separation"]
            arrow = "+" if sep > 0 else ""
            print(f"  {feat:30s} W={s['winner_mean']:>10.3f}  L={s['loser_mean']:>10.3f}  "
                  f"delta={arrow}{sep:.3f}")

    etq = report.get("entry_timing_quality", {})
    if etq:
        print(f"\nEntry Timing Quality:")
        print(f"  Winner mean: {etq['winner_mean']}")
        print(f"  Loser mean:  {etq['loser_mean']}")
        print(f"  Positive correlation: {etq['correlation_positive']}")

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
