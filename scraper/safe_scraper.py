"""
Safe Telegram scraper with:
- Credentials from .env
- FloodWaitError handling with jitter
- Random group order per cycle
- 30-minute loop: scrape → process → push Supabase
"""

import os
import sys
import json
import time
import random
import asyncio
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.sessions import StringSession
from telethon.tl.functions.messages import GetHistoryRequest
from telethon.tl.types import PeerChannel
from telethon.errors import FloodWaitError

from pipeline import aggregate_ranking
from push_to_supabase import upsert_tokens, insert_snapshots
from outcome_tracker import fill_outcomes

# Load .env from the scraper directory
load_dotenv(Path(__file__).parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# === CONFIGURATION ===

TELEGRAM_API_ID = int(os.environ["TELEGRAM_API_ID"])
TELEGRAM_API_HASH = os.environ["TELEGRAM_API_HASH"]
MESSAGES_PER_GROUP = 200


def _get_session():
    """Return Telethon session: StringSession (CI) or file-based (local)."""
    string_session = os.environ.get("TELEGRAM_STRING_SESSION")
    if string_session:
        logger.info("Using StringSession from environment variable")
        return StringSession(string_session)
    session_name = os.environ.get("TELEGRAM_SESSION_NAME", "scraper_session")
    logger.info("Using file-based session: %s", session_name)
    return session_name
CACHE_FILE = Path(__file__).parent.parent / "group_cache.json"
CYCLE_INTERVAL_SECONDS = 30 * 60  # 30 minutes

# Time windows to compute
TIME_WINDOWS = {"3h": 3, "6h": 6, "12h": 12, "24h": 24, "48h": 48, "7d": 168}

# === KOL GROUPS (conviction scores) ===

GROUPS_DATA = {
    "missorplays": {"conviction": 7},
    "slingdeez": {"conviction": 8},
    "overdose_gems_calls": {"conviction": 10},
    "marcellcooks": {"conviction": 9},
    "shahlito": {"conviction": 7},
    "sadcatgamble": {"conviction": 7},
    "ghastlygems": {"conviction": 8},
    "archercallz": {"conviction": 8},
    "LevisAlpha": {"conviction": 8},
    "MarkDegens": {"conviction": 7},
    "darkocalls": {"conviction": 8},
    "kweensjournal": {"conviction": 8},
    "explorer_gems": {"conviction": 7},
    "ArcaneGems": {"conviction": 8},
    "veigarcalls": {"conviction": 7},
    "watisdes": {"conviction": 7},
    "Luca_Apes": {"conviction": 7},
    "wuziemakesmoney": {"conviction": 7},
    "BatmanSafuCalls": {"conviction": 7},
    "chiggajogambles": {"conviction": 7},
    "dylansdegens": {"conviction": 8},
    "AnimeGems": {"conviction": 7},
    "robogems": {"conviction": 7},
    "ALSTEIN_GEMCLUB": {"conviction": 8},
    "PoseidonTAA": {"conviction": 9},
    "canisprintoooors": {"conviction": 7},
    "jsdao": {"conviction": 8},
    "MaybachCalls": {"conviction": 8},
    "slingTA": {"conviction": 7},
    "MaybachGambleCalls": {"conviction": 7},
    "cryptorugmuncher": {"conviction": 10},
    "inside_calls": {"conviction": 8},
    "BossmanCallsOfficial": {"conviction": 8},
    "bounty_journal": {"conviction": 8},
    "cryptotalkwithfrog": {"conviction": 7},
    "StereoCalls": {"conviction": 8},
    "CarnagecallsGambles": {"conviction": 7},
    "PowsGemCalls": {"conviction": 8},
    "houseofdegeneracy": {"conviction": 6},
    "CatfishcallsbyPoe": {"conviction": 8},
    "spidersjournal": {"conviction": 8},
    "KittysKasino": {"conviction": 7},
    "cryptolyxecalls": {"conviction": 8},
    "izzycooks": {"conviction": 8},
    "cryptowhalecalls7": {"conviction": 7},
    "Carnagecalls": {"conviction": 9},
    "wulfcryptocalls": {"conviction": 8},
    "waldosalpha": {"conviction": 7},
    "thetonymoontana": {"conviction": 10},
    "OnyxxGems": {"conviction": 8},
    "SaviourCALLS": {"conviction": 7},
    "eunicalls": {"conviction": 8},
    "lollycalls": {"conviction": 7},
    "leoclub168c": {"conviction": 7},
    "TheCabalCalls": {"conviction": 8},
    "sugarydick": {"conviction": 8},
    "leoclub168g": {"conviction": 7},
    "jadendegens": {"conviction": 7},
    "certifiedprintor": {"conviction": 8},
    "MarkGems": {"conviction": 9},
    "LittleMustachoCalls": {"conviction": 8},
}

GROUPS_CONVICTION = {k: v["conviction"] for k, v in GROUPS_DATA.items()}


def load_group_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


async def _fetch_messages(client: TelegramClient, peer, count: int) -> list[dict]:
    """
    Fetch `count` messages from a peer with pagination.
    Telegram returns max 100 per GetHistoryRequest, so we paginate.
    """
    all_msgs = []
    offset_id = 0

    while len(all_msgs) < count:
        batch_limit = min(100, count - len(all_msgs))
        history = await client(GetHistoryRequest(
            peer=peer,
            limit=batch_limit,
            offset_date=None,
            offset_id=offset_id,
            max_id=0,
            min_id=0,
            add_offset=0,
            hash=0,
        ))

        if not history.messages:
            break

        for message in history.messages:
            if not message.message:
                continue
            all_msgs.append({
                "text": message.message.strip(),
                "date": message.date.isoformat(),
            })

        offset_id = history.messages[-1].id
        if len(history.messages) < batch_limit:
            break  # No more messages available

    return all_msgs


async def scrape_groups(client: TelegramClient) -> dict[str, list[dict]]:
    """
    Fetch the latest messages from all groups.
    Returns { "group_username": [ { text, date, ... }, ... ] }
    """
    group_cache = load_group_cache()
    messages_data: dict[str, list[dict]] = {}
    skipped = []

    group_list = list(GROUPS_DATA.keys())
    random.shuffle(group_list)

    for username in group_list:
        if username not in group_cache:
            logger.warning("SKIP %s — not in group_cache.json", username)
            skipped.append(username)
            continue

        try:
            peer = PeerChannel(group_cache[username])
            group_msgs = await _fetch_messages(client, peer, MESSAGES_PER_GROUP)
            messages_data[username] = group_msgs
            logger.info("Fetched %d msgs from %s", len(group_msgs), username)

        except FloodWaitError as e:
            wait = e.seconds + random.uniform(5, 15)
            logger.warning("FloodWait %ds for %s — sleeping %.0fs", e.seconds, username, wait)
            await asyncio.sleep(wait)
            # Retry once after wait
            try:
                peer = PeerChannel(group_cache[username])
                group_msgs = await _fetch_messages(client, peer, MESSAGES_PER_GROUP)
                messages_data[username] = group_msgs
                logger.info("Retry OK: %d msgs from %s", len(group_msgs), username)
            except Exception as retry_err:
                logger.error("Retry failed for %s: %s", username, retry_err)

        except Exception as e:
            logger.error("Error fetching %s: %s", username, e)

        # Jitter between groups: 3-7s
        jitter = random.uniform(3.0, 5.0) + random.uniform(0, 2.0)
        await asyncio.sleep(jitter)

    if skipped:
        logger.warning("Skipped groups (not in cache): %s", skipped)

    return messages_data


def process_and_push(messages_data: dict[str, list[dict]]) -> None:
    """Run the pipeline and push results to Supabase."""
    ranking_by_window: dict[str, list[dict]] = {}

    for window_name, hours in TIME_WINDOWS.items():
        ranking = aggregate_ranking(messages_data, GROUPS_CONVICTION, hours)
        ranking_by_window[window_name] = ranking
        logger.info("Window %s: %d tokens ranked", window_name, len(ranking))

    # Compute stats from 24h window
    data_24h = ranking_by_window.get("24h", [])
    total_mentions = sum(t["mentions"] for t in data_24h)
    avg_sentiment = (
        round(sum(t["sentiment"] for t in data_24h) / max(1, len(data_24h)) * 100, 1)
        if data_24h
        else 0
    )

    stats = {
        "totalTokens": len(data_24h),
        "totalMentions": total_mentions,
        "avgSentiment": avg_sentiment,
        "totalKols": len(GROUPS_CONVICTION),
    }

    upsert_tokens(ranking_by_window, stats)
    logger.info("Pushed to Supabase: %d tokens (24h)", len(data_24h))

    # Insert snapshots for ML training data (use 24h window — most balanced)
    if data_24h:
        insert_snapshots(data_24h)

    # Fill outcome labels for old snapshots
    try:
        fill_outcomes()
    except Exception as e:
        logger.error("Outcome tracker failed: %s", e)


async def run_one_cycle(client: TelegramClient) -> None:
    """Execute a single scrape-process-push cycle."""
    logger.info("=== Scrape cycle starting ===")
    messages_data = await scrape_groups(client)

    total_msgs = sum(len(v) for v in messages_data.values())
    logger.info("Scraped %d messages from %d groups", total_msgs, len(messages_data))

    if total_msgs > 0:
        process_and_push(messages_data)
    else:
        logger.warning("No messages scraped — skipping push")


async def main():
    parser = argparse.ArgumentParser(description="Telegram KOL scraper")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single scrape cycle then exit (for CI/cron)",
    )
    args = parser.parse_args()

    session = _get_session()
    client = TelegramClient(session, TELEGRAM_API_ID, TELEGRAM_API_HASH)
    await client.start()

    if args.once:
        logger.info("Running single cycle (--once mode).")
        try:
            await run_one_cycle(client)
        finally:
            await client.disconnect()
        logger.info("Single cycle complete. Exiting.")
        return

    # Default: infinite loop (backward compatible)
    logger.info("Entering 30-min loop.")
    while True:
        cycle_start = time.time()
        try:
            await run_one_cycle(client)
        except Exception as e:
            logger.error("Cycle failed: %s", e, exc_info=True)

        elapsed = time.time() - cycle_start
        remaining = max(0, CYCLE_INTERVAL_SECONDS - elapsed)
        logger.info("Cycle done in %.0fs. Sleeping %.0fs until next cycle.", elapsed, remaining)
        await asyncio.sleep(remaining)


if __name__ == "__main__":
    asyncio.run(main())
