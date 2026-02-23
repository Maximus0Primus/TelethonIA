"""
Safe Telegram scraper with:
- Credentials from .env
- FloodWaitError handling with jitter
- Random group order per cycle
- access_hash caching: after first run, zero get_entity API calls
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
from datetime import datetime, timezone, timedelta
from pathlib import Path

from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.sessions import StringSession
from telethon.tl.functions.messages import GetHistoryRequest
from telethon.tl.types import PeerChannel, InputPeerChannel
from telethon.errors import FloodWaitError

from pipeline import aggregate_ranking
from push_to_supabase import upsert_tokens, insert_snapshots, insert_kol_mentions, _get_client as _get_supabase
# fill_outcomes is handled by dedicated outcomes.yml workflow (every 2h)
from price_refresh import refresh_top_tokens
from debug_dump import dump_debug_data

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
MAX_MESSAGE_AGE_HOURS = 7 * 24  # 7 days


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
PRICE_REFRESH_INTERVAL = 3 * 60   # 3 minutes

# Time windows to compute
TIME_WINDOWS = {"3h": 3, "6h": 6, "12h": 12, "24h": 24, "48h": 48, "7d": 168}

# === KOL GROUPS (conviction scores + tier) ===
# Tier S (weight 2.0): elite callers — each mention counts double
# Tier A (weight 1.0): good callers — baseline weight
# All groups default to tier "A". Update tiers as you identify elite KOLs.

TIER_WEIGHTS = {"S": 2.0, "A": 1.0}

GROUPS_DATA = {
    # === S-TIER (weight 2.0, conviction 10) — elite callers ===
    "archercallz": {"conviction": 10, "tier": "S"},
    "MoonsCallz": {"conviction": 10, "tier": "S"},
    "Luca_Apes": {"conviction": 10, "tier": "S"},
    "donniesdegen": {"conviction": 10, "tier": "S"},
    "legerlegends": {"conviction": 10, "tier": "S"},
    "ghastlygems": {"conviction": 10, "tier": "S"},
    "certifiedprintor": {"conviction": 10, "tier": "S"},
    "bounty_journal": {"conviction": 10, "tier": "S"},
    "degenncabal": {"conviction": 10, "tier": "S"},
    "eveesL": {"conviction": 10, "tier": "S"},
    "MaybachGambleCalls": {"conviction": 10, "tier": "S"},
    "MaybachCalls": {"conviction": 10, "tier": "S"},
    "darkocalls": {"conviction": 10, "tier": "S"},
    # === A-TIER (weight 1.0, conviction 7) — good callers ===
    "BrodyCalls": {"conviction": 7, "tier": "A"},
    "explorer_gems": {"conviction": 7, "tier": "A"},
    "missorplays": {"conviction": 7, "tier": "A"},
    "ramcalls": {"conviction": 7, "tier": "A"},
    "snoopsalpha": {"conviction": 7, "tier": "A"},
    "slingoorioyaps": {"conviction": 7, "tier": "A"},
    "ALSTEIN_GEMCLUB": {"conviction": 7, "tier": "A"},
    "wuziemakesmoney": {"conviction": 7, "tier": "A"},
    "letswinallgems": {"conviction": 7, "tier": "A"},
    "dylansdegens": {"conviction": 7, "tier": "A"},
    "BossmanCallsOfficial": {"conviction": 7, "tier": "A"},
    "menacedegendungeon": {"conviction": 7, "tier": "A"},
    "arcanegems": {"conviction": 7, "tier": "A"},
    "PumpItCabal": {"conviction": 7, "tier": "A"},
    "MarcellsFightclub": {"conviction": 7, "tier": "A"},
    "x666calls": {"conviction": 7, "tier": "A"},
    "dylansdirtydiary": {"conviction": 7, "tier": "A"},
    "invacooksclub": {"conviction": 7, "tier": "A"},
    "leoclub168c": {"conviction": 7, "tier": "A"},
    "caniscooks": {"conviction": 7, "tier": "A"},
    "maritocalls": {"conviction": 7, "tier": "A"},
    "PowsGemCalls": {"conviction": 7, "tier": "A"},
    "waldosalpha": {"conviction": 7, "tier": "A"},
    "LevisAlpha": {"conviction": 7, "tier": "A"},
    "eunicalls": {"conviction": 7, "tier": "A"},
    "spidersjournal": {"conviction": 7, "tier": "A"},
    "marcellcooks": {"conviction": 7, "tier": "A"},
    "DegenSeals": {"conviction": 7, "tier": "A"},
    "KittysKasino": {"conviction": 7, "tier": "A"},
    "Archerrgambles": {"conviction": 7, "tier": "A"},
    "fakepumpsbynumer0": {"conviction": 7, "tier": "A"},
    "robogems": {"conviction": 7, "tier": "A"},
    "shahlito": {"conviction": 7, "tier": "A"},
    "CryptoChefCooks": {"conviction": 7, "tier": "A"},
    "LittleMustachoCalls": {"conviction": 7, "tier": "A"},
    "OnyxxGems": {"conviction": 7, "tier": "A"},
    "pantherjournal": {"conviction": 7, "tier": "A"},
    "CSCalls": {"conviction": 7, "tier": "A"},
    "kweensjournal": {"conviction": 7, "tier": "A"},
    "lollycalls": {"conviction": 7, "tier": "A"},
    "CatfishcallsbyPoe": {"conviction": 7, "tier": "A"},
    "CarnagecallsGambles": {"conviction": 7, "tier": "A"},
    "ChairmanDN1": {"conviction": 7, "tier": "A"},
    "NisoksChadHouse": {"conviction": 7, "tier": "A"},
    "sadcatgamble": {"conviction": 7, "tier": "A"},
    "shmooscasino": {"conviction": 7, "tier": "A"},
    "AnimeGems": {"conviction": 7, "tier": "A"},
    "veigarcalls": {"conviction": 7, "tier": "A"},
    "papicall": {"conviction": 7, "tier": "A"},
}

GROUPS_CONVICTION = {k: v["conviction"] for k, v in GROUPS_DATA.items()}
GROUPS_TIER = {k: v.get("tier", "A") for k, v in GROUPS_DATA.items()}


def load_group_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


async def _fetch_messages(client: TelegramClient, peer, count: int) -> list[dict]:
    """
    Fetch up to `count` messages from a peer, stopping early if messages
    are older than MAX_MESSAGE_AGE_HOURS. Paginates in batches of 100.
    """
    all_msgs = []
    offset_id = 0
    age_cutoff = datetime.now(timezone.utc) - timedelta(hours=MAX_MESSAGE_AGE_HOURS)
    hit_age_limit = False

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
            # Stop if we've gone past the age cutoff
            if message.date < age_cutoff:
                hit_age_limit = True
                break
            if not message.message:
                continue

            # Extract URLs hidden in Telegram entities (hyperlinked text, buttons)
            text = message.message.strip()
            if message.entities:
                entity_urls = []
                for entity in message.entities:
                    if hasattr(entity, 'url') and entity.url:
                        entity_urls.append(entity.url)
                if entity_urls:
                    text += "\n" + "\n".join(entity_urls)

            all_msgs.append({
                "text": text,
                "date": message.date.isoformat(),
                "is_forwarded": message.fwd_from is not None,
                "is_reply": message.reply_to is not None,
            })

        if hit_age_limit:
            break

        offset_id = history.messages[-1].id
        if len(history.messages) < batch_limit:
            break  # No more messages available

    return all_msgs


def save_group_cache(cache: dict) -> None:
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


async def _resolve_peer(client: TelegramClient, username: str, cached_entry, group_cache: dict):
    """
    Resolve a Telegram group to a usable peer.
    Strategy:
    1. If access_hash cached → InputPeerChannel directly (ZERO API calls)
    2. Else → get_entity(username) + cache the access_hash for future runs
    3. Fallback → PeerChannel(cached_id)
    """
    # Backward compat: old cache is int, new cache is {id, access_hash}
    if isinstance(cached_entry, dict):
        cached_id = cached_entry["id"]
        access_hash = cached_entry.get("access_hash")
    else:
        cached_id = cached_entry
        access_hash = None

    # Fast path: access_hash cached → skip get_entity entirely
    if access_hash is not None:
        return InputPeerChannel(channel_id=cached_id, access_hash=access_hash)

    # Slow path: resolve via API (contacts.resolveUsername) + cache access_hash
    try:
        entity = await client.get_entity(username)
        if hasattr(entity, "id") and hasattr(entity, "access_hash"):
            group_cache[username] = {"id": entity.id, "access_hash": entity.access_hash}
            save_group_cache(group_cache)
            logger.info("Cached access_hash for %s (id=%d)", username, entity.id)
        return entity
    except Exception as e:
        logger.debug("get_entity(%s) failed: %s — trying PeerChannel", username, e)

    return PeerChannel(cached_id)


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
            # New group: resolve via get_entity + cache access_hash
            try:
                entity = await client.get_entity(username)
                if hasattr(entity, "id") and hasattr(entity, "access_hash"):
                    group_cache[username] = {"id": entity.id, "access_hash": entity.access_hash}
                elif hasattr(entity, "id"):
                    group_cache[username] = {"id": entity.id}
                save_group_cache(group_cache)
                logger.info("Resolved new group %s → ID %d", username, entity.id)
                peer = entity
            except Exception as e:
                logger.warning("SKIP %s — not in cache and resolution failed: %s", username, e)
                skipped.append(username)
                jitter = random.uniform(2.0, 4.0)
                await asyncio.sleep(jitter)
                continue
        else:
            peer = await _resolve_peer(client, username, group_cache[username], group_cache)

        try:
            group_msgs = await _fetch_messages(client, peer, MESSAGES_PER_GROUP)
            messages_data[username] = group_msgs
            logger.info("Fetched %d msgs from %s", len(group_msgs), username)

        except FloodWaitError as e:
            wait = e.seconds + random.uniform(5, 15)
            logger.warning("FloodWait %ds for %s — sleeping %.0fs", e.seconds, username, wait)
            await asyncio.sleep(wait)
            # Retry once after wait
            try:
                peer = await _resolve_peer(client, username, group_cache[username], group_cache)
                group_msgs = await _fetch_messages(client, peer, MESSAGES_PER_GROUP)
                messages_data[username] = group_msgs
                logger.info("Retry OK: %d msgs from %s", len(group_msgs), username)
            except Exception as retry_err:
                logger.error("Retry failed for %s: %s", username, retry_err)

        except Exception as e:
            # Stale access_hash? Invalidate and retry with get_entity
            cached = group_cache.get(username)
            if isinstance(cached, dict) and cached.get("access_hash"):
                logger.warning("Fetch error for %s (%s) — retrying with fresh get_entity", username, e)
                try:
                    entity = await client.get_entity(username)
                    if hasattr(entity, "id") and hasattr(entity, "access_hash"):
                        group_cache[username] = {"id": entity.id, "access_hash": entity.access_hash}
                        save_group_cache(group_cache)
                    group_msgs = await _fetch_messages(client, entity, MESSAGES_PER_GROUP)
                    messages_data[username] = group_msgs
                    logger.info("Fresh resolve OK: %d msgs from %s", len(group_msgs), username)
                except Exception as retry_err:
                    logger.error("Fresh resolve also failed for %s: %s", username, retry_err)
            else:
                logger.error("Error fetching %s: %s", username, e)

        # Jitter between groups: 2-4s (safe — mostly GetHistoryRequest, no get_entity)
        jitter = random.uniform(2.0, 3.0) + random.uniform(0, 1.0)
        await asyncio.sleep(jitter)

    if skipped:
        logger.warning("Skipped groups (not in cache): %s", skipped)

    return messages_data


def _fetch_sol_price() -> float | None:
    """Fetch current SOL/USD price from CoinGecko (free, no auth). Returns None on failure."""
    import requests as _req
    try:
        resp = _req.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": "solana", "vs_currencies": "usd"},
            timeout=5,
        )
        if resp.status_code == 200:
            price = resp.json().get("solana", {}).get("usd")
            if price and float(price) > 0:
                logger.info("SOL price: $%.2f", float(price))
                return round(float(price), 2)
    except Exception as e:
        logger.warning("Failed to fetch SOL price: %s", e)
    return None


def process_and_push(messages_data: dict[str, list[dict]], dump: bool = False) -> None:
    """Run the pipeline and push results to Supabase."""
    _cycle_start = time.time()
    # Load dynamic scoring weights from Supabase (auto-learning loop)
    try:
        from pipeline import load_scoring_config
        load_scoring_config()
    except Exception as e:
        logger.warning("Failed to load scoring config: %s (using defaults)", e)

    ranking_by_window: dict[str, list[dict]] = {}
    all_enriched_by_window: dict[str, list[dict]] = {}
    all_raw_mentions: list[dict] = []

    for window_name, hours in TIME_WINDOWS.items():
        ranking, raw_mentions, all_enriched = aggregate_ranking(
            messages_data, GROUPS_CONVICTION, hours,
            groups_tier=GROUPS_TIER, tier_weights=TIER_WEIGHTS,
        )
        ranking_by_window[window_name] = ranking
        all_enriched_by_window[window_name] = all_enriched
        # v10: Keep mentions from the largest window only (7d) to avoid duplicates
        if window_name == "7d":
            all_raw_mentions = raw_mentions
        penalized = sum(1 for t in all_enriched if t.get("gate_mult", 1.0) < 1.0)
        logger.info("Window %s: %d tokens (%d penalized)", window_name, len(ranking), penalized)

    # Debug dump (before push, so we capture pre-Supabase state)
    if dump:
        dump_debug_data(messages_data, ranking_by_window)

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

    # v16: Fetch SOL price once per cycle for market context in backtesting
    sol_price = _fetch_sol_price()

    # v33: all_enriched == ranking (no hard gates), all tokens fully scored.
    all_24h = all_enriched_by_window.get("24h", [])
    if sol_price and all_24h:
        for t in all_24h:
            t["sol_price_at_snapshot"] = sol_price
    if all_24h:
        penalized_24h = sum(1 for t in all_24h if t.get("gate_mult", 1.0) < 1.0)
        insert_snapshots(all_24h)
        logger.info("Inserted %d snapshots (24h): %d penalized",
                     len(all_24h), penalized_24h)

    # Also insert snapshots for 7d window (covers tokens not in 24h)
    all_7d = all_enriched_by_window.get("7d", [])
    if all_7d:
        symbols_24h = {t["symbol"] for t in all_24h}
        extra_7d = [t for t in all_7d if t["symbol"] not in symbols_24h]
        if extra_7d:
            if sol_price:
                for t in extra_7d:
                    t["sol_price_at_snapshot"] = sol_price
            insert_snapshots(extra_7d)
            logger.info("Inserted %d extra snapshots from 7d window", len(extra_7d))

    # Paper trading: open new positions (config-driven)
    try:
        from paper_trader import open_paper_trades, _load_paper_trade_config
        if data_24h:
            sb_pt = _get_supabase()
            pt_config = _load_paper_trade_config(sb_pt)
            open_paper_trades(sb_pt, data_24h, cycle_ts=datetime.now(timezone.utc), config=pt_config)
    except Exception as e:
        logger.error("Paper trading (open) failed: %s", e)

    # v10: Store raw KOL mention texts for NLP analysis
    if all_raw_mentions:
        try:
            insert_kol_mentions(all_raw_mentions)
        except Exception as e:
            logger.error("KOL mentions insert failed: %s", e)

    # Outcome labels are filled by the dedicated outcomes.yml workflow (every 2h).
    # Running fill_outcomes() here was causing GeckoTerminal rate-limit stalls
    # that pushed the scraper past its 25-min GitHub Actions timeout.

    # Run automated backtest diagnosis (only if enough labeled data)
    try:
        from auto_backtest import run_auto_backtest
        report = run_auto_backtest()
        if report:
            ds = report.get("data_summary", {})
            recs = report.get("recommendations", [])
            logger.info(
                "Auto-backtest: %d snapshots, hit_rate_12h=%.1f%%, %d recommendations",
                ds.get("labeled_12h", 0),
                ds.get("hit_rate_12h", 0) * 100,
                len(recs),
            )
    except Exception as e:
        logger.error("Auto-backtest failed: %s", e)

    # Auto-retrain ML model when enough labeled data accumulated
    # v22: auto_train does multi-horizon × multi-threshold grid search
    # Conditions: 100+ labeled samples AND >7 days since last training
    # v34: Skip if cycle already consumed >20min (auto_train grid search takes 2-5min,
    # pushing GH Actions past its 25-min timeout on cold-cache runs)
    _elapsed_so_far = time.time() - _cycle_start
    if _elapsed_so_far > 20 * 60:
        logger.warning("Skipping auto_train: cycle already at %.0fs (>20min)", _elapsed_so_far)
    else:
        try:
            from train_model import auto_train, MODEL_DIR
            should_train = False

            # Check any existing model meta across all horizons
            meta_files = list(MODEL_DIR.glob("model_*_meta.json"))
            if not meta_files:
                should_train = True
            else:
                import json as _json
                latest_train = None
                for mf in meta_files:
                    try:
                        with open(mf) as _f:
                            meta = _json.load(_f)
                        trained_at = meta.get("auto_trained_at") or meta.get("trained_at", "")
                        if trained_at:
                            from datetime import datetime as _dt
                            last_dt = _dt.fromisoformat(trained_at)
                            if latest_train is None or last_dt > latest_train:
                                latest_train = last_dt
                    except Exception:
                        pass

                if latest_train:
                    days_since = (datetime.now(timezone.utc).replace(tzinfo=None) - latest_train).days
                    if days_since >= 7:
                        should_train = True
                        logger.info("Auto-retrain: %d days since last train, triggering", days_since)
                else:
                    should_train = True

            if should_train:
                result = auto_train(min_samples=100, trials=50)
                if result:
                    logger.info(
                        "Auto-retrain: SUCCESS — %s/%s/%.1fx p@5=%.3f",
                        result.get("horizon"), result.get("mode"),
                        result.get("threshold", 2.0),
                        result.get("metrics", {}).get("precision_at_5", 0),
                    )
                else:
                    logger.info("Auto-retrain: skipped (not enough data or no improvement)")
        except Exception as e:
            logger.error("Auto-retrain failed: %s", e)

    # v10: Cleanup old data to prevent unbounded growth
    try:
        sb = _get_supabase()
        # Existing: snapshots > 90d, kol_mentions > 90d
        result = sb.rpc("cleanup_old_snapshots").execute()
        deleted = result.data if result.data else 0
        if deleted and deleted > 0:
            logger.info("Retention: cleaned %d snapshots (>90d)", deleted)
        result2 = sb.rpc("cleanup_old_kol_mentions").execute()
        deleted2 = result2.data if result2.data else 0
        if deleted2 and deleted2 > 0:
            logger.info("Retention: cleaned %d kol_mentions (>90d)", deleted2)
        # v51: closed paper_trades > 30d, backtest_reports > 60d, scoring_history > 90d
        for rpc_name, label in [
            ("cleanup_closed_paper_trades", "closed paper_trades (>30d)"),
            ("cleanup_old_backtest_reports", "backtest_reports (>60d)"),
            ("cleanup_old_scoring_history", "scoring_config_history (>90d)"),
        ]:
            try:
                r = sb.rpc(rpc_name).execute()
                d = r.data if r.data else 0
                if d and d > 0:
                    logger.info("Retention: cleaned %d %s", d, label)
            except Exception as e2:
                logger.debug("Retention %s skipped: %s", rpc_name, e2)
    except Exception as e:
        logger.debug("Retention cleanup skipped: %s", e)


async def run_one_cycle(client: TelegramClient, dump: bool = False) -> None:
    """Execute a single scrape-process-push cycle."""
    logger.info("=== Scrape cycle starting ===")
    messages_data = await scrape_groups(client)

    total_msgs = sum(len(v) for v in messages_data.values())
    logger.info("Scraped %d messages from %d groups", total_msgs, len(messages_data))

    if total_msgs > 0:
        process_and_push(messages_data, dump=dump)

        # C1 fix: Run price refresh at end of cycle so --once mode
        # (GH Action) also gets fresh DexScreener prices before exiting.
        try:
            updated = refresh_top_tokens()
            logger.info("Post-cycle price refresh: %d tokens updated", updated)
        except Exception as e:
            logger.error("Post-cycle price refresh failed: %s", e)

        # Paper trading: check open positions + log summary
        try:
            from paper_trader import check_paper_trades, paper_trade_summary
            sb_pt = _get_supabase()
            check_paper_trades(sb_pt)
            summary = paper_trade_summary(sb_pt)
            if summary:
                logger.info("paper_trader: %s", summary)
        except Exception as e:
            logger.error("Paper trading (cycle-end) failed: %s", e)
    else:
        logger.warning("No messages scraped — skipping push")


async def main():
    parser = argparse.ArgumentParser(description="Telegram KOL scraper")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single scrape cycle then exit (for CI/cron)",
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Save debug dump of all messages and pipeline output to scraper/debug_dumps/",
    )
    args = parser.parse_args()

    session = _get_session()
    client = TelegramClient(session, TELEGRAM_API_ID, TELEGRAM_API_HASH)
    await client.start()

    if args.once:
        logger.info("Running single cycle (--once mode).")
        try:
            await run_one_cycle(client, dump=args.dump)
        finally:
            await client.disconnect()
        logger.info("Single cycle complete. Exiting.")
        return

    # Default: infinite loop with parallel price refresh
    logger.info("Entering 30-min loop with 3-min price refresh.")

    async def price_refresh_loop():
        """Mini-cycle: refresh top token prices every 5 minutes."""
        while True:
            await asyncio.sleep(PRICE_REFRESH_INTERVAL)
            try:
                # refresh_top_tokens is sync (uses requests), run in thread
                updated = await asyncio.get_event_loop().run_in_executor(
                    None, refresh_top_tokens
                )
                logger.info("Price refresh: %d tokens updated", updated)
                # Paper trading: check open positions every 3 min
                try:
                    from paper_trader import check_paper_trades
                    sb_pt = _get_supabase()
                    check_paper_trades(sb_pt)
                except Exception as e:
                    logger.error("Paper trading (check) failed: %s", e)
            except Exception as e:
                logger.error("Price refresh failed: %s", e)

    # Start the price refresh loop as a background task
    refresh_task = asyncio.create_task(price_refresh_loop())

    try:
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
    finally:
        refresh_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
