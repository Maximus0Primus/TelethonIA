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
from telethon import TelegramClient, events
from telethon.sessions import StringSession
from telethon.tl.functions.messages import GetHistoryRequest
from telethon.tl.types import PeerChannel, InputPeerChannel
from telethon.errors import FloodWaitError

from pipeline import aggregate_ranking
from push_to_supabase import upsert_tokens, insert_snapshots, insert_kol_mentions, _get_client as _get_supabase
# fill_outcomes is handled by dedicated outcomes.yml workflow (every 2h)
from price_refresh import refresh_top_tokens
from debug_dump import dump_debug_data

# v67: Monitoring — conditional import (fail-safe)
try:
    from monitor import metrics as _metrics, track_api_call, estimate_egress
    from alerter import (
        send_startup_message, alert_cycle_failure, alert_rt_listener_down,
        alert_api_errors, alert_egress_warning, send_daily_summary,
        reset_alert,
    )
    _monitoring = True
except ImportError:
    _monitoring = False

# Load .env from the scraper directory
load_dotenv(Path(__file__).parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# === CONFIGURATION ===
# v58: These defaults are overridden by scoring_config.pipeline_config.scraper
# from Supabase when available. Loaded at startup via _load_pipeline_config().

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

# v65: Throttle expensive windows on VPS continuous mode to reduce egress.
# 48h runs every 2nd cycle, 7d every 6th. Saves ~60% of enrichment API calls + Supabase writes.
WINDOW_CYCLE_FREQ = {"3h": 1, "6h": 1, "12h": 1, "24h": 1, "48h": 2, "7d": 6}
_cycle_counter: int = 0  # incremented in main loop

# === KOL GROUPS (conviction scores + tier) ===
# Tier S (weight 2.0): elite callers — each mention counts double
# Tier A (weight 1.0): good callers — baseline weight
# All groups default to tier "A". Update tiers as you identify elite KOLs.

TIER_WEIGHTS = {"S": 2.0, "A": 1.0}


def _load_pipeline_config():
    """v58: Load scraper config from scoring_config.pipeline_config.scraper.
    Overrides module-level constants. No-op if DB is unreachable."""
    global MESSAGES_PER_GROUP, MAX_MESSAGE_AGE_HOURS, CYCLE_INTERVAL_SECONDS
    global PRICE_REFRESH_INTERVAL, TIER_WEIGHTS
    try:
        sb = _get_supabase()
        if not sb:
            return
        result = sb.table("scoring_config").select("pipeline_config").eq("id", 1).execute()
        if not result.data or not result.data[0].get("pipeline_config"):
            return
        cfg = result.data[0]["pipeline_config"]
        scraper_cfg = cfg.get("scraper", {})
        if scraper_cfg:
            MESSAGES_PER_GROUP = int(scraper_cfg.get("messages_per_group", MESSAGES_PER_GROUP))
            MAX_MESSAGE_AGE_HOURS = int(scraper_cfg.get("max_message_age_hours", MAX_MESSAGE_AGE_HOURS))
            CYCLE_INTERVAL_SECONDS = int(scraper_cfg.get("cycle_interval_seconds", CYCLE_INTERVAL_SECONDS))
            PRICE_REFRESH_INTERVAL = int(scraper_cfg.get("price_refresh_interval", PRICE_REFRESH_INTERVAL))
            TIER_WEIGHTS = scraper_cfg.get("tier_weights", TIER_WEIGHTS)
            logger.info("v58: Loaded scraper config from DB: msgs=%d, cycle=%ds, refresh=%ds",
                        MESSAGES_PER_GROUP, CYCLE_INTERVAL_SECONDS, PRICE_REFRESH_INTERVAL)
    except Exception as e:
        logger.warning("v58: Failed to load pipeline_config: %s (using defaults)", e)

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


def process_and_push(messages_data: dict[str, list[dict]], dump: bool = False,
                     cycle_num: int = 0, once_mode: bool = False) -> None:
    """Run the pipeline and push results to Supabase.

    Args:
        cycle_num: current cycle counter (0-based). Used to throttle expensive windows.
        once_mode: if True, run ALL windows regardless of throttle (--once / GH Actions).
    """
    _cycle_start = time.time()
    if _monitoring:
        _metrics.cycle_started(cycle_num)
    # Load dynamic scoring weights from Supabase (auto-learning loop)
    try:
        from pipeline import load_scoring_config
        load_scoring_config()
    except Exception as e:
        logger.warning("Failed to load scoring config: %s (using defaults)", e)

    ranking_by_window: dict[str, list[dict]] = {}
    all_enriched_by_window: dict[str, list[dict]] = {}
    all_raw_mentions: list[dict] = []

    # v65: Decide which windows to run this cycle
    for window_name, hours in TIME_WINDOWS.items():
        freq = WINDOW_CYCLE_FREQ.get(window_name, 1)
        if not once_mode and freq > 1 and cycle_num % freq != 0:
            logger.info("Window %s: SKIPPED (runs every %d cycles, next in %d)",
                        window_name, freq, freq - (cycle_num % freq))
            continue
        ranking, raw_mentions, all_enriched = aggregate_ranking(
            messages_data, GROUPS_CONVICTION, hours,
            groups_tier=GROUPS_TIER, tier_weights=TIER_WEIGHTS,
        )
        ranking_by_window[window_name] = ranking
        all_enriched_by_window[window_name] = all_enriched
        # v10: Keep mentions from the widest window processed (avoids duplicates)
        # v65: When 7d is skipped, the widest processed window provides mentions
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

    # v66: Batch confirmation — update open RT trades with batch score
    try:
        sb_rt = _get_supabase()
        if sb_rt and data_24h:
            rt_open = (
                sb_rt.table("paper_trades")
                .select("id, token_address, symbol")
                .eq("source", "rt")
                .eq("status", "open")
                .execute()
            )
            rt_trades = rt_open.data or []
            if rt_trades:
                # Build lookup: token_address → batch score
                batch_scores = {}
                for t in data_24h:
                    addr = t.get("token_address")
                    if addr:
                        batch_scores[addr] = int(t.get("score", 0))

                confirmed = 0
                for rt in rt_trades:
                    addr = rt["token_address"]
                    if addr in batch_scores:
                        try:
                            sb_rt.table("paper_trades").update({
                                "batch_confirmed_at": datetime.now(timezone.utc).isoformat(),
                                "batch_score": batch_scores[addr],
                            }).eq("id", rt["id"]).execute()
                            confirmed += 1
                            logger.info("RT trade %s confirmed by batch (score=%d)",
                                        rt["symbol"], batch_scores[addr])
                        except Exception as e2:
                            logger.debug("RT confirm update failed for %s: %s", rt["symbol"], e2)
                if confirmed:
                    logger.info("v66: %d/%d open RT trades confirmed by batch", confirmed, len(rt_trades))
    except Exception as e:
        logger.debug("v66: batch confirmation skipped: %s", e)

    # v10: Store raw KOL mention texts for NLP analysis
    if all_raw_mentions:
        try:
            insert_kol_mentions(all_raw_mentions)
        except Exception as e:
            logger.error("KOL mentions insert failed: %s", e)

    # Outcome labels are filled by the dedicated outcomes.yml workflow (every 2h).
    # Running fill_outcomes() here was causing GeckoTerminal rate-limit stalls
    # that pushed the scraper past its 25-min GitHub Actions timeout.

    # v63: Auto-backtest REMOVED from 15-min scrape cycle to fix egress.
    # Was doing full table scan (34k rows × 170 cols = ~400MB) every 15min = 10+ GB/day.
    # Now runs ONLY in outcomes.yml (1x/hour) — 75% fewer API calls.

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

    # v67: Record cycle completion
    if _monitoring:
        _metrics.cycle_completed(
            scored=len(data_24h),
            pushed=len(ranking_by_window.get("24h", [])),
            windows=list(ranking_by_window.keys()),
        )


# ---------------------------------------------------------------------------
# v66: Real-time KOL listener — EXPLORATION MODE
# Philosophy: Never block a trade. Score everything, size by confidence.
#   - Cooldown per (KOL × token), not per token — 3 KOLs same token = 3 signals
#   - All 7 strategies active — let Optuna/ML learn which work per-context
#   - Score drives sizing, not entry — low score = $1 micro position (learn), high = $30
#   - Only hard block: no price (can't compute PnL)
#   - When ML model exists: predicts per-strategy PnL → smart strategy selection
# ---------------------------------------------------------------------------
_rt_ca_cache: dict = {}
_rt_ml_model = None  # LightGBM Booster, loaded from Supabase
_rt_trade_cooldown: dict[str, float] = {}  # (kol, ca) -> last_trade_timestamp
_rt_in_flight: set = set()  # (kol, ca) tuples currently being processed
_rt_group_id_to_username: dict[int, str] = {}
_rt_kol_scores: dict = {}
_rt_kol_scores_loaded_at: float = 0.0
_rt_config: dict = {}
_rt_config_loaded_at: float = 0.0
RT_COOLDOWN_SECONDS = 1800
RT_KOL_SCORES_TTL = 3600  # 1 hour
RT_CONFIG_TTL = 300  # 5 minutes

# v71: KOL WR whitelist + bankroll management
_rt_kol_whitelist: dict = {}           # {kol_group: {wr, total, hits, approved}}
_rt_kol_whitelist_loaded_at: float = 0.0
_rt_bankroll: dict = {}                # single row from rt_bankroll table
_rt_bankroll_loaded_at: float = 0.0
RT_KOL_WHITELIST_TTL = 3600  # 1h
RT_BANKROLL_TTL = 60          # 1min


def _rt_load_kol_whitelist(config: dict) -> dict:
    """v71: Load KOL whitelist based on historical WR from kol_call_outcomes. Cached 1h."""
    global _rt_kol_whitelist, _rt_kol_whitelist_loaded_at
    now = time.time()
    if _rt_kol_whitelist and now - _rt_kol_whitelist_loaded_at < RT_KOL_WHITELIST_TTL:
        return _rt_kol_whitelist

    kf = config.get("kol_filter", {})
    wr_threshold = float(kf.get("wr_threshold", 0.60))
    return_threshold = float(kf.get("return_threshold", 1.5))
    min_calls = int(kf.get("min_calls", 5))

    try:
        sb = _get_supabase()
        if not sb:
            return _rt_kol_whitelist
        # Fetch all KCO rows with a max_return value
        result = sb.table("kol_call_outcomes").select(
            "kol_group, max_return"
        ).filter("max_return", "not.is", "null").execute()
        rows = result.data or []
        if not rows:
            logger.warning("RT KOL whitelist: no KCO rows with max_return")
            return _rt_kol_whitelist

        # Aggregate by kol_group
        from collections import defaultdict
        agg = defaultdict(lambda: {"total": 0, "hits": 0})
        for r in rows:
            kol = r["kol_group"]
            agg[kol]["total"] += 1
            if float(r["max_return"] or 0) >= return_threshold:
                agg[kol]["hits"] += 1

        whitelist = {}
        approved_count = 0
        for kol, stats in agg.items():
            wr = stats["hits"] / stats["total"] if stats["total"] > 0 else 0
            approved = wr >= wr_threshold and stats["total"] >= min_calls
            whitelist[kol] = {
                "wr": round(wr, 4),
                "total": stats["total"],
                "hits": stats["hits"],
                "approved": approved,
            }
            if approved:
                approved_count += 1

        _rt_kol_whitelist = whitelist
        _rt_kol_whitelist_loaded_at = now
        logger.info("RT KOL whitelist: %d/%d approved (wr>=%.0f%%, calls>=%d, ret>=%.1fx)",
                     approved_count, len(whitelist), wr_threshold * 100, min_calls, return_threshold)
    except Exception as e:
        logger.warning("RT KOL whitelist load failed: %s", e)
    return _rt_kol_whitelist


def _rt_load_bankroll() -> dict:
    """v71: Load bankroll state from rt_bankroll table. Cached 60s."""
    global _rt_bankroll, _rt_bankroll_loaded_at
    now = time.time()
    if _rt_bankroll and now - _rt_bankroll_loaded_at < RT_BANKROLL_TTL:
        return _rt_bankroll
    try:
        sb = _get_supabase()
        if not sb:
            return _rt_bankroll or {"current_balance": 100.0}
        result = sb.table("rt_bankroll").select("*").limit(1).execute()
        if result.data:
            _rt_bankroll = result.data[0]
            _rt_bankroll_loaded_at = now
            logger.debug("RT bankroll loaded: $%.2f (peak $%.2f, dd %.1f%%)",
                         float(_rt_bankroll["current_balance"]),
                         float(_rt_bankroll["peak_balance"]),
                         float(_rt_bankroll["max_drawdown_pct"]))
        else:
            _rt_bankroll = {"current_balance": 100.0, "peak_balance": 100.0,
                            "total_pnl": 0.0, "total_trades": 0, "max_drawdown_pct": 0.0}
    except Exception as e:
        logger.warning("RT bankroll load failed: %s", e)
        if not _rt_bankroll:
            _rt_bankroll = {"current_balance": 100.0, "peak_balance": 100.0,
                            "total_pnl": 0.0, "total_trades": 0, "max_drawdown_pct": 0.0}
    return _rt_bankroll


def _rt_update_bankroll(pnl_usd: float, n_trades: int) -> None:
    """v71: Update bankroll after trades close. Read-modify-write with peak/drawdown tracking."""
    global _rt_bankroll, _rt_bankroll_loaded_at
    if pnl_usd == 0 and n_trades == 0:
        return
    try:
        sb = _get_supabase()
        if not sb:
            return
        # Fresh read to avoid stale data
        result = sb.table("rt_bankroll").select("*").limit(1).execute()
        if not result.data:
            return
        row = result.data[0]
        old_balance = float(row["current_balance"])
        new_balance = round(old_balance + pnl_usd, 2)
        new_total_pnl = round(float(row["total_pnl"]) + pnl_usd, 2)
        new_total_trades = int(row["total_trades"]) + n_trades
        new_peak = max(float(row["peak_balance"]), new_balance)
        new_dd = round((1 - new_balance / new_peak) * 100, 2) if new_peak > 0 else 0
        new_dd = max(float(row["max_drawdown_pct"]), new_dd)

        sb.table("rt_bankroll").update({
            "current_balance": new_balance,
            "total_pnl": new_total_pnl,
            "total_trades": new_total_trades,
            "peak_balance": new_peak,
            "max_drawdown_pct": new_dd,
            "last_updated_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", row["id"]).execute()

        # Invalidate cache so next read gets fresh data
        _rt_bankroll_loaded_at = 0
        logger.info("RT bankroll: $%.2f → $%.2f (pnl=$%+.2f, %d trades, peak=$%.2f, dd=%.1f%%)",
                     old_balance, new_balance, pnl_usd, n_trades, new_peak, new_dd)
    except Exception as e:
        logger.error("RT bankroll update failed: %s", e)


def _rt_load_kol_scores() -> dict:
    """Load KOL scores from kol_scores.json. Refreshed every hour."""
    global _rt_kol_scores, _rt_kol_scores_loaded_at
    now = time.time()
    if _rt_kol_scores and now - _rt_kol_scores_loaded_at < RT_KOL_SCORES_TTL:
        return _rt_kol_scores
    try:
        kol_path = Path(__file__).parent / "kol_scores.json"
        if kol_path.exists():
            with open(kol_path, "r") as f:
                data = json.load(f)
            details = data.get("_kol_details", {})
            scores = {}
            for kol, info in details.items():
                if kol.startswith("_"):
                    continue
                scores[kol] = {
                    "score": float(info.get("score", 0)),
                    "win_rate": float(info.get("raw_hit_rate", 0)),
                    "total_calls": int(info.get("tokens_called", 0)),
                }
            for k, v in data.items():
                if not k.startswith("_") and k not in scores and isinstance(v, (int, float)):
                    scores[k] = {"score": float(v), "win_rate": 0.0, "total_calls": 0}
            _rt_kol_scores = scores
            _rt_kol_scores_loaded_at = now
            logger.info("RT: KOL scores loaded (%d KOLs)", len(scores))
    except Exception as e:
        logger.warning("RT: failed to load kol_scores.json: %s", e)
    return _rt_kol_scores


def _rt_load_config() -> dict:
    """Load RT trade config from scoring_config.rt_trade_config. 5min cache."""
    global _rt_config, _rt_config_loaded_at, RT_COOLDOWN_SECONDS
    now = time.time()
    if _rt_config and now - _rt_config_loaded_at < RT_CONFIG_TTL:
        return _rt_config
    defaults = {
        "enabled": True,
        "cooldown_seconds": 1800,
        "base_budget_usd": 20,
        "min_position_usd": 1.0,
        "max_position_usd": 30.0,
        "rt_strategies": "all",  # "all" = all 7 strategies
        "score_weights": {
            "kol_quality": 0.35,
            "token_safety": 0.30,
            "market_momentum": 0.20,
            "confirmation": 0.15,
        },
        # Sizing multiplier curves (Optuna-tunable)
        "sizing": {
            "kol_score_mult_cap": 1.8,     # best KOL → 1.8× budget
            "kol_score_mult_floor": 0.3,    # unknown KOL → 0.3× budget
            "tier_s_bonus": 1.3,            # S-tier KOL → extra 1.3×
            "safety_mult_floor": 0.2,       # very risky token → 0.2× budget
            "safety_mult_cap": 1.5,         # very safe token → 1.5× budget
            "momentum_mult_floor": 0.5,     # negative momentum → 0.5×
            "momentum_mult_cap": 1.5,       # strong momentum → 1.5×
        },
    }
    try:
        sb = _get_supabase()
        if sb:
            result = sb.table("scoring_config").select("rt_trade_config").eq("id", 1).execute()
            if result.data and result.data[0].get("rt_trade_config"):
                raw = result.data[0]["rt_trade_config"]
                if isinstance(raw, str):
                    raw = json.loads(raw)
                # Deep merge for nested dicts
                for k, v in raw.items():
                    if isinstance(v, dict) and isinstance(defaults.get(k), dict):
                        defaults[k].update(v)
                    else:
                        defaults[k] = v
                logger.info("RT: config loaded from DB (enabled=%s, budget=$%.0f, strategies=%s)",
                            defaults["enabled"], defaults["base_budget_usd"], defaults["rt_strategies"])
    except Exception as e:
        logger.warning("RT: failed to load rt_trade_config: %s (using defaults)", e)
    _rt_config = defaults
    _rt_config_loaded_at = now
    RT_COOLDOWN_SECONDS = int(defaults.get("cooldown_seconds", 1800))
    return _rt_config


def _rt_should_trade(kol: str, ca: str) -> bool:
    """Cooldown per (KOL × token). Does NOT consume — call _rt_mark_traded() on success."""
    now = time.time()
    key = (kol, ca)
    last = _rt_trade_cooldown.get(key, 0)
    return now - last >= RT_COOLDOWN_SECONDS


def _rt_mark_traded(kol: str, ca: str) -> None:
    """Consume cooldown slot for this (KOL × token) pair."""
    _rt_trade_cooldown[(kol, ca)] = time.time()


def _rt_compute_kol_quality(kol_info: dict, tier: str) -> float:
    """KOL quality sub-score (0-100). No gates — unknown KOL = low score, not blocked."""
    kol_score_raw = min(kol_info.get("score", 0), 3.0)
    kol_wr = min(kol_info.get("win_rate", 0), 0.5)
    kol_calls = kol_info.get("total_calls", 0)

    # Score component (0-40): based on kol_score from kol_scores.json
    score_part = (kol_score_raw / 3.0) * 40

    # Win rate component (0-40): scaled to 50% cap
    wr_part = (kol_wr / 0.5) * 40

    # Tier bonus (0-15): S-tier gets flat boost
    tier_part = 15 if tier == "S" else 0

    # Experience bonus (0-5): more calls = more data = more trust
    exp_part = min(5, kol_calls / 10 * 5) if kol_calls > 0 else 0

    return min(100, max(0, score_part + wr_part + tier_part + exp_part))


def _rt_compute_token_safety(token_info: dict) -> float:
    """Token safety sub-score (0-100). Everything is a gradient, nothing blocked."""
    liq = token_info.get("liquidity_usd", 0)
    mcap = max(token_info.get("mcap", 1), 1)
    vol = token_info.get("volume_24h", 0)
    bsr = token_info.get("buy_sell_ratio", 0.5)
    age_h = token_info.get("token_age_hours", 0)
    is_pf = token_info.get("is_pump_fun", 0)
    liq_mcap = liq / mcap

    # Liquidity (0-30): $0→0, $5K→10, $50K→20, $500K+→30
    if liq < 5000:
        liq_score = (liq / 5000) * 10
    elif liq < 50000:
        liq_score = 10 + (liq - 5000) / 45000 * 10
    else:
        liq_score = min(30, 20 + (liq - 50000) / 450000 * 10)

    # Volume (0-20): piecewise
    if vol < 10_000:
        vol_score = (vol / 10_000) * 6
    elif vol < 100_000:
        vol_score = 6 + (vol - 10_000) / 90_000 * 8
    else:
        vol_score = min(20, 14 + (vol - 100_000) / 900_000 * 6)

    # BSR (0-15): below 0.3 = very bearish but still scored
    bsr_score = min(15, max(0, bsr / 2.0 * 15))

    # Liq/MCap ratio (0-15): healthy ratio = higher safety
    liq_mcap_score = min(15, max(0, liq_mcap / 0.2 * 15))

    # Age (0-10): new = 0, mature = 10
    age_score = min(10, max(0, age_h / 168 * 10))

    # Pump.fun penalty (0 or -10): bonding curve = extra risky but NOT blocked
    pf_penalty = -10 if (is_pf and liq < 10000) else 0

    return max(0, min(100, liq_score + vol_score + bsr_score + liq_mcap_score + age_score + pf_penalty))


def _rt_compute_momentum(token_info: dict) -> float:
    """Market momentum sub-score (0-100). Negative momentum = low score, not blocked."""
    pc_1h = token_info.get("price_change_1h", 0)
    pc_5m = token_info.get("price_change_5m", 0)
    vol = token_info.get("volume_24h", 0)

    # Price momentum 1h (0-35): -50%→0, 0%→17, +25%→35
    m_1h = min(35, max(0, 17.5 + pc_1h * 0.7))

    # Price momentum 5m (0-35): -20%→0, 0%→17, +10%→35
    m_5m = min(35, max(0, 17.5 + pc_5m * 1.75))

    # Volume (0-30): piecewise
    if vol < 10_000:
        v_score = (vol / 10_000) * 9
    elif vol < 100_000:
        v_score = 9 + (vol - 10_000) / 90_000 * 12
    else:
        v_score = min(30, 21 + (vol - 100_000) / 900_000 * 9)

    return max(0, min(100, m_1h + m_5m + v_score))


def _rt_compute_confirmation(kol: str, ca: str, token_info: dict) -> float:
    """
    Confirmation sub-score (0-100): is this token being called by multiple KOLs?
    Check if other KOLs have recently traded the same CA (from cooldown dict).
    """
    now = time.time()
    recent_kols = set()
    for (k, c), ts in _rt_trade_cooldown.items():
        if c == ca and k != kol and (now - ts) < 3600:  # within last hour
            recent_kols.add(k)

    n_other_kols = len(recent_kols)
    # 0 others → 0, 1 other → 40, 2 others → 70, 3+ → 100
    if n_other_kols == 0:
        return 0
    elif n_other_kols == 1:
        return 40
    elif n_other_kols == 2:
        return 70
    else:
        return 100


def _rt_compute_score(kol_username: str, ca: str, kol_info: dict,
                      token_info: dict, tier: str, config: dict) -> float:
    """
    Compute RT score (0-100). Drives position sizing, NOT entry/exit.
    4 components: KOL quality, token safety, momentum, confirmation.
    """
    weights = config.get("score_weights", {
        "kol_quality": 0.35, "token_safety": 0.30,
        "market_momentum": 0.20, "confirmation": 0.15,
    })

    kol_q = _rt_compute_kol_quality(kol_info, tier)
    safety = _rt_compute_token_safety(token_info)
    momentum = _rt_compute_momentum(token_info)
    confirmation = _rt_compute_confirmation(kol_username, ca, token_info)

    w_kol = float(weights.get("kol_quality", 0.35))
    w_safety = float(weights.get("token_safety", 0.30))
    w_momentum = float(weights.get("market_momentum", 0.20))
    w_confirm = float(weights.get("confirmation", 0.15))

    rt_score = kol_q * w_kol + safety * w_safety + momentum * w_momentum + confirmation * w_confirm
    return round(max(0, min(100, rt_score)), 1)


def _rt_position_size(rt_score: float, kol_info: dict, token_info: dict,
                      tier: str, config: dict) -> float:
    """
    v71: Bankroll-based position sizing with Kelly fraction.
    base_bet = bankroll × kelly_fraction (7%)
    Modulated by KOL WR and RT score. Floor $1, cap $200.
    Falls back to old fixed-budget sizing if sizing.mode != "bankroll".
    """
    sizing = config.get("sizing", {})
    mode = sizing.get("mode", "fixed")

    if mode == "bankroll":
        bankroll = _rt_load_bankroll()
        balance = float(bankroll.get("current_balance", 100))
        kelly = float(sizing.get("kelly_fraction", 0.07))
        base_bet = balance * kelly

        # WR multiplier: higher WR KOLs get bigger size
        wr = float(kol_info.get("win_rate", 0))
        if wr >= 0.80:
            wr_mult = 1.5
        elif wr >= 0.70:
            wr_mult = 1.2
        else:
            wr_mult = 1.0  # 60% threshold already passed by WR gate

        # Score multiplier: RT score 0→0.7x, 100→1.3x
        score_frac = max(0, min(1, rt_score / 100))
        score_mult = 0.7 + score_frac * 0.6

        size = base_bet * wr_mult * score_mult

        min_pos = float(config.get("min_position_usd", sizing.get("min_position_usd", 1.0)))
        max_pos = float(config.get("max_position_usd", sizing.get("max_position_usd", 200.0)))
        return round(max(min_pos, min(max_pos, size)), 2)

    # --- Legacy fixed-budget mode (fallback) ---
    budget = float(config.get("base_budget_usd", 20))

    # KOL multiplier: [floor, cap] based on kol_score
    kol_score = kol_info.get("score", 0)
    kol_floor = float(sizing.get("kol_score_mult_floor", 0.3))
    kol_cap = float(sizing.get("kol_score_mult_cap", 1.8))
    kol_frac = min(1.0, kol_score / 3.0) if kol_score > 0 else 0
    kol_mult = kol_floor + kol_frac * (kol_cap - kol_floor)

    if tier == "S":
        kol_mult *= float(sizing.get("tier_s_bonus", 1.3))

    safety_sub = _rt_compute_token_safety(token_info)
    safety_floor = float(sizing.get("safety_mult_floor", 0.2))
    safety_cap = float(sizing.get("safety_mult_cap", 1.5))
    safety_mult = safety_floor + (safety_sub / 100) * (safety_cap - safety_floor)

    momentum_sub = _rt_compute_momentum(token_info)
    mom_floor = float(sizing.get("momentum_mult_floor", 0.5))
    mom_cap = float(sizing.get("momentum_mult_cap", 1.5))
    mom_mult = mom_floor + (momentum_sub / 100) * (mom_cap - mom_floor)

    combined = (kol_mult * safety_mult * mom_mult) ** (1 / 3)
    size = budget * combined

    min_pos = float(config.get("min_position_usd", 1.0))
    max_pos = float(config.get("max_position_usd", 30.0))
    return round(max(min_pos, min(max_pos, size)), 2)


def _rt_count_confirmations(kol: str, ca: str) -> int:
    """Count distinct other KOLs that traded the same CA within the last hour."""
    now = time.time()
    other_kols = set()
    for (k, c), ts in _rt_trade_cooldown.items():
        if c == ca and k != kol and (now - ts) < 3600:
            other_kols.add(k)
    return len(other_kols)


def _rt_open_trades(ca: str, symbol: str, price: float, mcap: float,
                    kol_username: str, tier: str, rt_score: float, pos_size: float,
                    kol_info: dict, token_info: dict, config: dict):
    """
    v71: Open paper trades with hybrid strategy allocation.
    When hybrid_strategy.enabled: split pos_size across configured strategies (e.g. 60/40).
    Fallback to exploration mode if disabled.
    """
    from paper_trader import open_paper_trades, _load_paper_trade_config, STRATEGIES
    sb = _get_supabase()
    if not sb:
        return 0

    pt_config = _load_paper_trade_config(sb)

    # Build base token entry with RT metadata
    token_entry = {
        "symbol": symbol,
        "token_address": ca,
        "price_usd": price,
        "market_cap": mcap,
        "score": int(rt_score),
        "ca_mention_count": 1,
        "url_mention_count": 0,
        "unique_kols": 1,
        "kol_freshness": 1.0,
        "momentum_mult": 1.0,
        "whale_new_entries": None,
        "_rt_source": "rt",
        "_rt_kol_group": kol_username,
        "_rt_kol_tier": tier,
        "_rt_kol_score": kol_info.get("score"),
        "_rt_kol_win_rate": kol_info.get("win_rate"),
        "_rt_score": rt_score,
        "_rt_liquidity_usd": token_info.get("liquidity_usd"),
        "_rt_volume_24h": token_info.get("volume_24h"),
        "_rt_buy_sell_ratio": token_info.get("buy_sell_ratio"),
        "_rt_token_age_hours": token_info.get("token_age_hours"),
        "_rt_is_pump_fun": token_info.get("is_pump_fun"),
    }

    now = datetime.now(timezone.utc)
    total_opened = 0

    # v71: Hybrid strategy — split position across configured allocations
    hybrid_cfg = config.get("hybrid_strategy", {})
    if hybrid_cfg.get("enabled", False):
        allocations = hybrid_cfg.get("allocations", {"TP50_SL30": 0.60, "TP100_SL30": 0.40})
        for strat_name, alloc_pct in allocations.items():
            if strat_name not in STRATEGIES:
                continue
            strat_pos = round(pos_size * float(alloc_pct), 2)
            if strat_pos < 1.0:
                strat_pos = 1.0

            strat_config = dict(pt_config)
            strat_config["budget_usd"] = strat_pos
            strat_config["active_strategies"] = [strat_name]
            strat_config["top_n"] = 1
            strat_config["ca_filter"] = False

            opened = open_paper_trades(sb, [token_entry], cycle_ts=now, config=strat_config)
            total_opened += opened

        if total_opened > 0:
            alloc_str = " + ".join(f"{s}={p:.0%}" for s, p in allocations.items())
            logger.info(
                "RT TRADE [HYBRID]: %s @ $%.6f | %s | KOL: %s (%s, wr=%.0f%%) "
                "rt_score=%.0f pos=$%.2f liq=$%.0fK",
                symbol, price, alloc_str, kol_username, tier,
                kol_info.get("win_rate", 0) * 100,
                rt_score, pos_size,
                token_info.get("liquidity_usd", 0) / 1000,
            )

        # v72: Live trading — mirror paper trades with real execution
        live_cfg = config.get("live_trading", {})
        if live_cfg.get("enabled", False) and total_opened > 0:
            try:
                from live_trader import open_live_trade
                live_allocs = live_cfg.get("allocations", allocations)
                for strat_name, alloc_pct in live_allocs.items():
                    live_pos = round(pos_size * float(alloc_pct), 2)
                    open_live_trade(sb, token_entry, strat_name, live_pos, live_cfg)
            except Exception as e:
                logger.error("Live trading (open) failed: %s", e)

        return total_opened

    # --- Fallback: exploration mode (all active strategies, equal sizing) ---
    # v72: Warn if live trading is enabled but hybrid mode is off (live trades only fire in hybrid mode)
    live_cfg = config.get("live_trading", {})
    if live_cfg.get("enabled", False):
        logger.warning("live_trading.enabled=True but hybrid_strategy.enabled=False — "
                        "live trades only fire in hybrid mode. Enable hybrid_strategy or disable live_trading.")

    db_active = [s for s in pt_config.get("active_strategies", list(STRATEGIES.keys())) if s in STRATEGIES]
    rt_strats = config.get("rt_strategies", "all")
    if rt_strats == "all":
        valid_strategies = db_active
    else:
        valid_strategies = [s for s in rt_strats if s in STRATEGIES]
        if not valid_strategies:
            valid_strategies = db_active

    explore_config = dict(pt_config)
    explore_config["budget_usd"] = pos_size
    explore_config["active_strategies"] = valid_strategies
    explore_config["top_n"] = 1
    explore_config["ca_filter"] = False

    total_opened = open_paper_trades(sb, [token_entry], cycle_ts=now, config=explore_config)

    if total_opened > 0:
        logger.info(
            "RT TRADE [EXPLORE]: %s @ $%.6f | %d strats | KOL: %s (%s, wr=%.0f%%) "
            "rt_score=%.0f pos=$%.2f liq=$%.0fK",
            symbol, price, len(valid_strategies), kol_username, tier,
            kol_info.get("win_rate", 0) * 100,
            rt_score, pos_size,
            token_info.get("liquidity_usd", 0) / 1000,
        )
    return total_opened


def _rt_extract_token_info(raw: dict) -> dict:
    """Parse DexScreener response into normalized token_info dict."""
    return {
        "liquidity_usd": float(raw.get("liquidity_usd") or 0),
        "volume_24h": float(raw.get("volume_24h") or 0),
        "buy_sell_ratio": float(raw.get("buy_sell_ratio_24h") or raw.get("bsr_24h") or 0.5),
        "token_age_hours": float(raw.get("token_age_hours") or 0),
        "is_pump_fun": int(raw.get("is_pump_fun") or 0),
        "mcap": float(raw.get("mcap") or 0),
        "price_change_1h": float(raw.get("price_change_1h") or 0),
        "price_change_5m": float(raw.get("price_change_5m") or 0),
    }


_rt_debug_counter = 0  # Log first N events for debugging


async def _rt_on_new_message(event: events.NewMessage.Event):
    """v66 Exploration Mode: Score everything, size by confidence, learn from all trades."""
    global _rt_ca_cache, _rt_debug_counter

    chat_id = event.chat_id
    username = _rt_group_id_to_username.get(chat_id)
    # v69: Also try unmarked positive ID (Telethon channels use -100{id})
    if not username and chat_id < 0:
        raw_id = int(str(chat_id).replace("-100", "", 1)) if str(chat_id).startswith("-100") else abs(chat_id)
        username = _rt_group_id_to_username.get(raw_id)
        if username:
            # Cache the marked ID for future lookups
            _rt_group_id_to_username[chat_id] = username
    if not username:
        if _rt_debug_counter < 20:
            _rt_debug_counter += 1
            logger.info("RT: unmatched chat_id=%s (event %d)", chat_id, _rt_debug_counter)
        return

    msg = event.message
    if not msg or not msg.message:
        return

    text = msg.message.strip()
    if msg.entities:
        entity_urls = []
        for entity in msg.entities:
            if hasattr(entity, "url") and entity.url:
                entity_urls.append(entity.url)
        if entity_urls:
            text += "\n" + "\n".join(entity_urls)

    from pipeline import extract_tokens, _load_ca_cache, _save_ca_cache
    if not _rt_ca_cache:
        _rt_ca_cache = _load_ca_cache()

    tokens = extract_tokens(text, ca_cache=_rt_ca_cache)
    if not tokens:
        return

    config = _rt_load_config()
    if not config.get("enabled", True):
        return
    kol_scores = _rt_load_kol_scores()

    tier = GROUPS_TIER.get(username, "A")
    kol_info = kol_scores.get(username, {"score": 0.0, "win_rate": 0.0, "total_calls": 0})

    # v71: KOL WR gate — only trade KOLs with proven historical win rate
    kol_filter_cfg = config.get("kol_filter", {})
    if kol_filter_cfg.get("enabled", False):
        whitelist = _rt_load_kol_whitelist(config)
        if whitelist:  # only apply gate when whitelist is actually populated
            kol_wl = whitelist.get(username)
            if kol_wl and kol_wl.get("approved"):
                # Enrich kol_info with real KCO win rate (replaces kol_scores.json WR)
                kol_info = dict(kol_info)  # don't mutate cached dict
                kol_info["win_rate"] = kol_wl["wr"]
                kol_info["total_calls"] = kol_wl["total"]
            else:
                wr_str = f" wr={kol_wl['wr']:.0%}/{kol_wl['total']}calls" if kol_wl else " (no KCO data)"
                logger.info("RT SKIP (KOL WR): %s%s", username, wr_str)
                return

    for symbol, source, ca in tokens:
        if not ca:
            continue

        # Cooldown per (KOL × token) — different KOL on same token = new signal
        if not _rt_should_trade(username, ca):
            continue

        flight_key = (username, ca)
        if flight_key in _rt_in_flight:
            continue
        _rt_in_flight.add(flight_key)

        try:
            logger.info("RT detect: %s (CA: %s...) from %s (%s-tier)", symbol, ca[:8], username, tier)

            # DexScreener fetch — only hard requirement: price must exist
            from enrich import _fetch_dexscreener_by_address
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(None, _fetch_dexscreener_by_address, ca)
            if not raw or not raw.get("price_usd"):
                logger.info("RT SKIP: %s — no price (only hard block)", ca[:8])
                continue

            try:
                price = float(raw["price_usd"])
            except (ValueError, TypeError):
                continue
            if price <= 0:
                continue

            mcap = float(raw.get("mcap") or 0)
            token_info = _rt_extract_token_info(raw)

            # Compute RT score (drives sizing, never blocks)
            rt_score = _rt_compute_score(username, ca, kol_info, token_info, tier, config)

            # Position sizing: confidence → dollars
            pos_size = _rt_position_size(rt_score, kol_info, token_info, tier, config)

            # v70: KCO model prediction — augment RT score with ML-predicted return
            kco_predicted_return = None
            try:
                from pipeline import predict_kco_score
                kco_predicted_return = predict_kco_score(username, tier, token_info)
                if kco_predicted_return is not None:
                    # Adjust position size based on KCO prediction
                    # >1.5x predicted → boost, <1.0x → reduce
                    if kco_predicted_return >= 1.5:
                        pos_size = min(pos_size * 1.3, config.get("max_position_usd", 30))
                    elif kco_predicted_return < 1.0:
                        pos_size = max(pos_size * 0.5, config.get("min_position_usd", 1))
            except Exception as e:
                logger.debug("KCO prediction unavailable: %s", e)

            # Store KCO prediction in token_info for paper trade metadata
            if kco_predicted_return is not None:
                token_info["kco_predicted_return"] = kco_predicted_return

            # v74: RT ML model — adjust position size based on predicted PnL
            ml_tag = ""
            if _rt_ml_model is not None:
                try:
                    from rt_model import predict_strategy_pnl
                    hour = datetime.now(timezone.utc).hour
                    preds = predict_strategy_pnl(
                        _rt_ml_model, kol_info, token_info, tier, rt_score,
                        n_confirmations=0, hour=hour,
                    )
                    # Average predicted PnL across strategies
                    avg_pred = sum(preds.values()) / len(preds) if preds else 0
                    if avg_pred > 0.02:
                        pos_size = min(pos_size * 1.5, config.get("max_position_usd", 30))
                        ml_tag = f" ml_pred=+{avg_pred:.1%}(boost)"
                    elif avg_pred < -0.02:
                        pos_size = max(pos_size * 0.5, config.get("min_position_usd", 1))
                        ml_tag = f" ml_pred={avg_pred:.1%}(reduce)"
                    else:
                        ml_tag = f" ml_pred={avg_pred:.1%}"
                    token_info["_rt_ml_pred"] = avg_pred
                except Exception as e:
                    logger.debug("RT ML prediction failed: %s", e)

            logger.info(
                "RT score: %s rt_score=%.0f → pos=$%.2f | kol=%s(%.2f/%.0f%%) liq=$%.0fK bsr=%.2f%s%s",
                symbol, rt_score, pos_size, username,
                kol_info.get("score", 0), kol_info.get("win_rate", 0) * 100,
                token_info.get("liquidity_usd", 0) / 1000,
                token_info.get("buy_sell_ratio", 0),
                f" kco={kco_predicted_return:.2f}x" if kco_predicted_return else "",
                ml_tag,
            )

            # Open trades across all strategies
            opened = await loop.run_in_executor(
                None, _rt_open_trades, ca, symbol, price, mcap,
                username, tier, rt_score, pos_size, kol_info, token_info, config,
            )

            if opened and opened > 0:
                _rt_mark_traded(username, ca)
                # v67: Record RT event for monitoring
                if _monitoring:
                    _metrics.record_rt_event(symbol, username, ca, rt_score, pos_size, opened)
        finally:
            _rt_in_flight.discard(flight_key)


async def setup_realtime_listener(client: TelegramClient):
    """Register event handlers for all KOL groups. Returns list of group IDs."""
    global _rt_group_id_to_username
    cache = load_group_cache()
    group_ids = []

    # v69: First, join all groups to ensure we receive RT updates.
    # Telegram only pushes real-time updates for channels you're a member of.
    # Strategy: use cached IDs first (no API call), only resolve uncached groups.
    from telethon.tl.functions.channels import JoinChannelRequest
    import asyncio as _aio

    joined = 0
    already = 0
    failed = 0
    for username in GROUPS_DATA:
        # Try cache first to avoid get_entity flood waits
        cached = cache.get(username)
        gid = None
        if isinstance(cached, dict) and "id" in cached:
            gid = cached["id"]
        elif isinstance(cached, int):
            gid = cached

        try:
            if gid:
                # Use cached ID — try to join via InputChannel (no username resolve)
                from telethon.tl.types import InputChannel
                access_hash = cached.get("access_hash", 0) if isinstance(cached, dict) else 0
                input_ch = InputChannel(gid, access_hash or 0)
                try:
                    await client(JoinChannelRequest(input_ch))
                    joined += 1
                except Exception as je:
                    je_str = str(je)
                    if "ALREADY" in je_str.upper() or "already" in je_str.lower():
                        already += 1
                    else:
                        # access_hash may be stale — fall back to username resolve
                        entity = await client.get_entity(username)
                        await client(JoinChannelRequest(entity))
                        gid = entity.id
                        cache[username] = {
                            "id": gid,
                            "access_hash": getattr(entity, "access_hash", None),
                        }
                        save_group_cache(cache)
                        joined += 1
                        await _aio.sleep(1)  # rate-limit between resolves
            else:
                # No cache — must resolve username (costs 1 API call)
                entity = await client.get_entity(username)
                await client(JoinChannelRequest(entity))
                gid = entity.id
                cache[username] = {
                    "id": gid,
                    "access_hash": getattr(entity, "access_hash", None),
                }
                save_group_cache(cache)
                joined += 1
                await _aio.sleep(1)  # rate-limit between resolves

            # Map both raw and marked IDs
            group_ids.append(gid)
            _rt_group_id_to_username[gid] = username
            marked_id = int(f"-100{gid}")
            _rt_group_id_to_username[marked_id] = username
        except Exception as e:
            failed += 1
            # Still use cached ID for the mapping if available
            if gid:
                group_ids.append(gid)
                _rt_group_id_to_username[gid] = username
                marked_id = int(f"-100{gid}")
                _rt_group_id_to_username[marked_id] = username
            else:
                logger.warning("RT: could not join/resolve %s: %s", username, e)

    logger.info("RT: joined %d, already %d, failed %d (total mapped: %d)",
                joined, already, failed, len(group_ids))

    if group_ids:
        # v69: Register WITHOUT chats filter — filter in handler instead.
        # Telethon's chats filter can silently drop events when IDs don't
        # match the internal "marked" format. Handler filters via the mapping.
        client.add_event_handler(
            _rt_on_new_message,
            events.NewMessage(),
        )

        # v69: Debug — log first 10 raw updates to confirm Telethon receives events
        _raw_count = {"n": 0}

        async def _rt_debug_raw(event):
            if _raw_count["n"] < 10:
                _raw_count["n"] += 1
                logger.info(
                    "RT RAW EVENT #%d: type=%s chat_id=%s",
                    _raw_count["n"], type(event).__name__,
                    getattr(event, "chat_id", "?"),
                )

        client.add_event_handler(_rt_debug_raw, events.NewMessage())
        def _cache_gid(entry):
            if isinstance(entry, dict):
                return entry.get("id")
            elif isinstance(entry, int):
                return entry
            return None
        s_count = sum(1 for u, d in GROUPS_DATA.items()
                      if d.get("tier") == "S" and _cache_gid(cache.get(u)) in group_ids)

        # v66: Pre-load KOL scores + RT config + ML model at startup
        global _rt_ml_model
        kol_scores = _rt_load_kol_scores()
        rt_config = _rt_load_config()

        # Try loading ML model from Supabase
        ml_status = "no model"
        try:
            from rt_model import load_rt_model
            _rt_ml_model = load_rt_model()
            if _rt_ml_model:
                ml_status = "ML model loaded"
        except Exception as e:
            logger.debug("RT: ML model load failed: %s", e)

        logger.info(
            "RT v66: %d groups (%d S-tier) | %d KOLs | %s | "
            "budget=$%.0f, strategies=%s, cooldown=%ds",
            len(group_ids), s_count, len(kol_scores), ml_status,
            rt_config.get("base_budget_usd", 20),
            rt_config.get("rt_strategies", "all"),
            rt_config.get("cooldown_seconds", 1800),
        )
    return group_ids


async def run_one_cycle(client: TelegramClient, dump: bool = False,
                        once_mode: bool = False) -> None:
    """Execute a single scrape-process-push cycle."""
    global _cycle_counter
    logger.info("=== Scrape cycle starting (cycle #%d) ===", _cycle_counter)

    # v74: Replay any buffered failed writes from previous cycles
    try:
        from push_to_supabase import retry_failed_writes
        replay = retry_failed_writes()
        if replay.get("replayed", 0) > 0:
            logger.info("Write buffer replay: %d/%d succeeded",
                        replay["succeeded"], replay["replayed"])
    except Exception as e:
        logger.debug("Write buffer replay skipped: %s", e)

    messages_data = await scrape_groups(client)

    total_msgs = sum(len(v) for v in messages_data.values())
    logger.info("Scraped %d messages from %d groups", total_msgs, len(messages_data))

    if total_msgs > 0:
        process_and_push(messages_data, dump=dump,
                         cycle_num=_cycle_counter, once_mode=once_mode)

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
            pt_result = check_paper_trades(sb_pt)
            # v71: Update bankroll with RT PnL from cycle-end check
            rt_pnl = pt_result.get("rt_pnl_usd", 0)
            rt_closed = pt_result.get("rt_closed", 0)
            if rt_pnl != 0 or rt_closed > 0:
                _rt_update_bankroll(rt_pnl, rt_closed)
            summary = paper_trade_summary(sb_pt)
            if summary:
                logger.info("paper_trader: %s", summary)
            # v74: KOL attribution — aggregate paper trade PnL by KOL
            from paper_trader import kol_attribution
            kol_attribution(sb_pt, days=7)
        except Exception as e:
            logger.error("Paper trading (cycle-end) failed: %s", e)

        # v72 note: live trades monitored by dedicated live_trade_monitor_loop (10s)
    else:
        logger.warning("No messages scraped — skipping push")


async def main():
    global _cycle_counter

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

    # v58: Load dynamic config from DB before starting
    _load_pipeline_config()

    session = _get_session()
    client = TelegramClient(session, TELEGRAM_API_ID, TELEGRAM_API_HASH)
    await client.start()

    if args.once:
        logger.info("Running single cycle (--once mode).")
        try:
            await run_one_cycle(client, dump=args.dump, once_mode=True)
        finally:
            await client.disconnect()
        logger.info("Single cycle complete. Exiting.")
        return

    # Default: infinite loop with parallel price refresh + real-time listener
    logger.info("Entering continuous mode: RT listener + 30-min batch + 3-min price refresh.")

    # v64: Register real-time event handlers for instant paper trading
    try:
        rt_groups = await setup_realtime_listener(client)
        logger.info("Real-time listener registered for %d groups", len(rt_groups))
        # v69: Force Telethon to subscribe to channel updates.
        # StringSession loses update state on restart — get_dialogs() forces
        # Telegram to push updates for all joined channels/groups.
        try:
            dialogs = await client.get_dialogs(limit=200)
            dialog_ids = {d.id for d in dialogs}
            matched = sum(1 for gid in rt_groups if gid in dialog_ids)
            logger.info(
                "RT: get_dialogs() fetched %d dialogs (%d/%d KOL groups matched)",
                len(dialogs), matched, len(rt_groups),
            )
            await client.catch_up()
            logger.info("RT: catch_up() completed — update state synced")
        except Exception as e:
            logger.warning("RT: dialog/catch_up failed (non-fatal): %s", e)
    except Exception as e:
        logger.error("Failed to setup RT listener (batch mode continues): %s", e)

    # v74: Position reconciliation on startup — verify DB vs on-chain balance
    try:
        from live_trader import reconcile_positions
        sb_recon = _get_supabase()
        if sb_recon:
            recon = await asyncio.get_event_loop().run_in_executor(
                None, reconcile_positions, sb_recon
            )
            if recon["mismatches"] > 0:
                logger.warning("Startup reconciliation: %d mismatches found", recon["mismatches"])
    except Exception as e:
        logger.debug("Startup reconciliation skipped: %s", e)

    # v67: Send startup alert + launch monitor loop
    if _monitoring:
        try:
            send_startup_message(len(GROUPS_DATA), len(rt_groups) if 'rt_groups' in dir() else 0)
        except Exception:
            pass

    async def monitor_loop():
        """v67: Health check loop — every 5 minutes, evaluate scraper health."""
        _last_daily_hour = -1
        while True:
            await asyncio.sleep(300)  # 5 minutes
            try:
                # Check cycle health
                cs = _metrics.get_cycle_stats(5)
                recent_errors = cs.get("recent_errors", [])
                if recent_errors:
                    last_err = recent_errors[-1]
                    if time.time() - last_err["ts"] < 600:
                        alert_cycle_failure(
                            last_err.get("num", 0),
                            last_err.get("msg", "unknown"),
                            0,
                        )

                # Check RT listener health
                rt_stats = _metrics.get_rt_stats(2.0)
                last_age = rt_stats.get("last_event_age_s")
                if last_age is not None and last_age > 7200:
                    alert_rt_listener_down(last_age / 60)
                elif last_age is not None and last_age <= 7200:
                    # RT is healthy again — reset backoff so next outage re-alerts
                    reset_alert("rt_listener_down")

                # Check API error rates
                api_stats = _metrics.get_api_stats(1.0)
                for api, stats in api_stats.items():
                    if stats["calls"] >= 5 and stats["error_rate"] > 0.30:
                        alert_api_errors(api, stats["error_rate"], stats["errors"], stats["calls"])

                # Check egress
                egress = _metrics.get_egress_estimate()
                total_mb = egress.get("total_mb", 0)
                if total_mb >= 500:
                    alert_egress_warning(total_mb, egress.get("by_module", {}), 500)

                # Daily summary at 8h UTC
                now_utc = datetime.now(timezone.utc)
                if now_utc.hour == 8 and _last_daily_hour != 8:
                    _last_daily_hour = 8
                    snapshot = _metrics.get_full_snapshot()
                    send_daily_summary(snapshot)
                elif now_utc.hour != 8:
                    _last_daily_hour = now_utc.hour

            except Exception as e:
                logger.warning("Monitor loop error: %s", e, exc_info=True)

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
                    pt_result = check_paper_trades(sb_pt)
                    # v71: Update bankroll with RT PnL
                    rt_pnl = pt_result.get("rt_pnl_usd", 0)
                    rt_closed = pt_result.get("rt_closed", 0)
                    if rt_pnl != 0 or rt_closed > 0:
                        _rt_update_bankroll(rt_pnl, rt_closed)
                except Exception as e:
                    logger.error("Paper trading (check) failed: %s", e)
                # v72 note: live trades monitored by dedicated live_trade_monitor_loop (10s)
            except Exception as e:
                logger.error("Price refresh failed: %s", e)

    # v72: Dedicated fast-poll loop for live trades (10s interval).
    # Single owner of check_live_trades() — no calls from price_refresh or cycle-end.
    LIVE_TRADE_POLL_INTERVAL = 10  # seconds

    async def live_trade_monitor_loop():
        """Fast monitoring loop for live trades only. 10s polling."""
        _consecutive_empty = 0
        while True:
            await asyncio.sleep(LIVE_TRADE_POLL_INTERVAL)
            try:
                # Check if live trading is enabled (re-read config periodically)
                config = _rt_load_config()
                live_cfg = config.get("live_trading", {})
                if not live_cfg.get("enabled", False):
                    _consecutive_empty = 0
                    continue

                from live_trader import check_live_trades
                sb_lt = _get_supabase()
                if not sb_lt:
                    continue

                live_result = await asyncio.get_event_loop().run_in_executor(
                    None, check_live_trades, sb_lt
                )
                checked = live_result.get("checked", 0)
                closed = live_result.get("closed", 0)

                if checked == 0:
                    _consecutive_empty += 1
                    # Reduce logging noise: only log every 30th empty check (~5 min)
                    if _consecutive_empty % 30 == 1:
                        logger.debug("live_trade_monitor: no open live trades")
                    continue

                _consecutive_empty = 0
                if closed > 0:
                    logger.info(
                        "LIVE MONITOR: closed %d/%d trades (TP=%d SL=%d TO=%d) pnl=$%+.2f",
                        closed, checked,
                        live_result.get("tp", 0), live_result.get("sl", 0),
                        live_result.get("timeout", 0), live_result.get("pnl_usd", 0),
                    )
                    # Update bankroll with live PnL
                    live_pnl = live_result.get("rt_pnl_usd", 0)
                    if live_pnl != 0 or closed > 0:
                        _rt_update_bankroll(live_pnl, closed)

            except Exception as e:
                logger.error("live_trade_monitor error: %s", e)

    # Start the price refresh loop as a background task
    refresh_task = asyncio.create_task(price_refresh_loop())
    # v67: Start monitor loop
    monitor_task = asyncio.create_task(monitor_loop()) if _monitoring else None
    # v72: Start live trade fast-poll loop
    live_monitor_task = asyncio.create_task(live_trade_monitor_loop())

    try:
        while True:
            cycle_start = time.time()
            try:
                await run_one_cycle(client)
            except Exception as e:
                logger.error("Cycle failed: %s", e, exc_info=True)
                if _monitoring:
                    _metrics.cycle_error(str(e))
                    _metrics.cycle_completed(0, 0, [])
                    alert_cycle_failure(_cycle_counter, str(e), time.time() - cycle_start)

            _cycle_counter += 1
            elapsed = time.time() - cycle_start
            remaining = max(0, CYCLE_INTERVAL_SECONDS - elapsed)
            logger.info("Cycle #%d done in %.0fs. Sleeping %.0fs until next cycle.",
                        _cycle_counter, elapsed, remaining)
            await asyncio.sleep(remaining)
    finally:
        refresh_task.cancel()
        live_monitor_task.cancel()
        if monitor_task:
            monitor_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
