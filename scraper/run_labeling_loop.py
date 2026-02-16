"""
Labeling loop: runs fill_outcomes() repeatedly to clear the backlog.
Usage: python run_labeling_loop.py [--runs N] [--pause SECONDS]

Default: 25 runs with 30s pause between runs (~8 hours, clears ~10K snapshots).
"""
import argparse
import logging
import time
import os
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run outcome labeling loop")
    parser.add_argument("--runs", type=int, default=25, help="Number of runs (default: 25)")
    parser.add_argument("--pause", type=int, default=30, help="Seconds between runs (default: 30)")
    args = parser.parse_args()

    from outcome_tracker import fill_outcomes
    from supabase import create_client

    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

    for i in range(1, args.runs + 1):
        # Check remaining backlog
        try:
            result = client.table("token_snapshots").select("id", count="exact").is_("did_2x_24h", "null").lt(
                "snapshot_at",
                (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 24 * 3600))),
            ).execute()
            backlog = result.count or 0
        except Exception:
            backlog = "?"

        logger.info("=== Run %d/%d — backlog_24h=%s ===", i, args.runs, backlog)

        if backlog == 0:
            logger.info("Backlog cleared! Stopping.")
            break

        start = time.time()
        try:
            fill_outcomes()
        except Exception as e:
            logger.error("fill_outcomes() failed: %s", e)

        elapsed = time.time() - start
        logger.info("=== Run %d done in %.0fs ===", i, elapsed)

        if i < args.runs:
            logger.info("Pausing %ds before next run...", args.pause)
            time.sleep(args.pause)

    # Final status
    try:
        result = client.table("token_snapshots").select("id", count="exact").is_("did_2x_24h", "null").lt(
            "snapshot_at",
            (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 24 * 3600))),
        ).execute()
        logger.info("Final backlog_24h: %s", result.count)
    except Exception:
        pass


if __name__ == "__main__":
    main()
