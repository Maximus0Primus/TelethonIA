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
    # === v107: new groups (A-tier default, let data prove themselves) ===
    "BatmanSafuCalls": {"conviction": 7, "tier": "A"},
    "ChadleyGambles123": {"conviction": 7, "tier": "A"},
    "DoxxedChannel": {"conviction": 7, "tier": "A"},
    "DoxxedGuy": {"conviction": 7, "tier": "A"},
    "FaithCalls": {"conviction": 7, "tier": "A"},
    "FrenzGems": {"conviction": 7, "tier": "A"},
    "NeoCallss": {"conviction": 7, "tier": "A"},
    "TheReaperGems": {"conviction": 7, "tier": "A"},
    "ZAGREBIAN": {"conviction": 7, "tier": "A"},
    "aliensalphacalls": {"conviction": 7, "tier": "A"},
    # "bat_gamble" removed v108 — -11.18% avg PnL, -248K% cumul on 22K shadow trades
    "chiggajogambles": {"conviction": 7, "tier": "A"},
    "gubbinscalls": {"conviction": 7, "tier": "A"},
    "ihzanswhaleschool": {"conviction": 7, "tier": "A"},
    "lucyalphas": {"conviction": 7, "tier": "A"},
    "maythouscalls": {"conviction": 7, "tier": "A"},
    "olympeqg": {"conviction": 7, "tier": "A"},
    "robo_gambles": {"conviction": 7, "tier": "A"},
    "trishhhxy": {"conviction": 7, "tier": "A"},
    "watisdes": {"conviction": 7, "tier": "A"},
    "zcallz": {"conviction": 7, "tier": "A"},
    "Carnagecalls": {"conviction": 7, "tier": "A"},
    "DegensCabal": {"conviction": 7, "tier": "A"},
    "OnyxxGambles": {"conviction": 7, "tier": "A"},
    "cryptowhalecalls7": {"conviction": 7, "tier": "A"},
    "dddegens": {"conviction": 7, "tier": "A"},
    "jadendegens": {"conviction": 7, "tier": "A"},
    "maythousdegens": {"conviction": 7, "tier": "A"},
    # === v144.18 additions ===
    "leoclub69": {"conviction": 7, "tier": "A"},
    "markdegens": {"conviction": 7, "tier": "A"},
    # === v14e.14 additions (Apr 24) — paper+shadow test only, NOT live ===
    # Force-excluded from live via rt_trade_config.live_trading.kol_blacklist.
    # Objective: build N>=30 paper trades per KOL to detect if any of them
    # (or a combo) yields a profitable strategy before allowing live exposure.
    # bat_gamble re-added after v108 removal — user wants to re-test since the
    # scoring engine evolved heavily since then (v108->v144). TheReaperGems
    # was already in the registry (capitalized there) so not re-added.
    "batman_gem": {"conviction": 7, "tier": "A"},
    "venom_gambles": {"conviction": 7, "tier": "A"},
    "ryoshigamble": {"conviction": 7, "tier": "A"},
    # v14e.14c: ryoshikushama dropped (Telegram user, not a channel).
    # v14e.14d: ryoshikdegen added — Channel "Ryoshi Degen" (broadcast=1).
    # v14e.29 (Apr 26): ryoshikdegen REMOVED from scraping. Apr 25-26 paper
    # stats: N=350 paper trades, avg -12.7%, WR 9.1%, sum -$2,419 (paper).
    # Confirmed loser pattern — pumps already exhausted before he calls.
    # Stays in live blacklist as defense-in-depth in case he's ever re-added.
    "mad_apes_gambles": {"conviction": 7, "tier": "A"},
    # v14e.28 (Apr 26): MaestrosDegen REMOVED from scraping. Apr 1-26 stats:
    # SOL N=1000 paper trades, avg -49.8%, sum -$23,985 (paper). ETH N=12,
    # avg -21.5%. Pump-and-dump pattern: he calls AT the top, every strategy
    # ends WR ~0%. No strategy variant rescues his calls. Zero signal value
    # — strictly noise that distorts paper ranking.
    # v14e.14e (kept for git blame): maestrodegen was wrong handle (fetched
    # 0 msgs every cycle); MaestrosDegen is the correct title that resolved.
    # v14e.17: bat_gamble restricted to ETH paper trades only. Solana paper
    # trades bled -$11,386 across 9 strats (727 closes); ETH paper trades
    # earned +$1,627 across 6 strats (23 closes). Solana history wiped, refund
    # credited back to SOL bankrolls. Per-KOL chain whitelist enforced in the
    # RT trade gate below.
    "bat_gamble": {"conviction": 7, "tier": "A", "chains": ["ethereum"]},
    "reapergamble": {"conviction": 7, "tier": "A"},
    "bagcalls": {"conviction": 7, "tier": "A"},
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
                "message_id": message.id,  # v109: track for dedup
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


def _check_api_health(tokens: list) -> None:
    """v80: Check fill rates for all enrichment APIs after each cycle. Alert if degraded."""
    if not tokens:
        return
    total = len(tokens)

    fill_rates = {
        "dexscreener": round(sum(1 for t in tokens if t.get("price_usd")           is not None) / total * 100),
        "helius":      round(sum(1 for t in tokens if t.get("whale_count")          is not None) / total * 100),
        "jupiter":     round(sum(1 for t in tokens if t.get("jup_tradeable")        is not None) / total * 100),
    }

    # Birdeye: removed from health alerts (noisy, not actionable)

    from alerter import alert_api_health
    alert_api_health(fill_rates, total)


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
        "duration_seconds": int(time.time() - _cycle_start),
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
        # v144.19: API health Telegram alerts disabled (user: too noisy).
        # Fill-rate computation still available via _check_api_health() if needed.

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
    # v104: batch_trading_enabled flag — when False, batch cycles only collect data (snapshots/scoring)
    # and skip opening paper trades. RT trades are unaffected.
    try:
        from paper_trader import open_paper_trades, _load_paper_trade_config
        if data_24h:
            sb_pt = _get_supabase()
            pt_config = _load_paper_trade_config(sb_pt)
            batch_enabled = pt_config.get("batch_trading_enabled", False)  # v110: default=False
            if batch_enabled:
                open_paper_trades(sb_pt, data_24h, cycle_ts=datetime.now(timezone.utc), config=pt_config)
            else:
                logger.info("Batch paper trading disabled (batch_trading_enabled=false) — data collection only")
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
# v14e: 0x → chain cache. Populated by resolve_evm_chain on first sight so
# repeated KOL mentions of the same ETH/BSC/Base CA don't re-hit DexScreener.
# Keyed by lowercased address. Never expired in-process — the CA→chain
# mapping doesn't change within a token's lifetime.
_rt_evm_chain_cache: dict[str, str] = {}
RT_KOL_WHITELIST_TTL = 3600  # 1h
RT_BANKROLL_TTL = 60          # 1min


def _rt_load_kol_whitelist(config: dict) -> dict:
    """v82: Load KOL whitelist from scoring_config.kol_rt_whitelist (cached by auto_backtest).
    Falls back to live paper_trades query if cache is empty.
    Cached in-memory for 1h (RT_KOL_WHITELIST_TTL)."""
    global _rt_kol_whitelist, _rt_kol_whitelist_loaded_at
    now = time.time()
    if _rt_kol_whitelist and now - _rt_kol_whitelist_loaded_at < RT_KOL_WHITELIST_TTL:
        return _rt_kol_whitelist

    try:
        sb = _get_supabase()
        if not sb:
            return _rt_kol_whitelist

        # Primary: read cached whitelist from scoring_config (updated by auto_backtest every 2h)
        result = sb.table("scoring_config").select(
            "kol_rt_whitelist,kol_manual_overrides"
        ).eq("id", 1).execute()
        cached = None
        overrides = None
        if result.data:
            raw_wl = result.data[0].get("kol_rt_whitelist")
            raw_ov = result.data[0].get("kol_manual_overrides")
            if raw_wl:
                cached = json.loads(raw_wl) if isinstance(raw_wl, str) else raw_wl
            if raw_ov:
                overrides = json.loads(raw_ov) if isinstance(raw_ov, str) else raw_ov

        if cached and len(cached) > 0:
            # v83: Apply manual overrides — force approve/block regardless of stats
            # Format: {"KolName": {"approved": true/false, "reason": "manual"}, ...}
            if overrides:
                for kol_name, ov in overrides.items():
                    if kol_name in cached:
                        cached[kol_name]["approved"] = ov.get("approved", False)
                        cached[kol_name]["manual_override"] = True
                    else:
                        # KOL not in auto-whitelist — add as override-only entry
                        cached[kol_name] = {
                            "wr": 0, "total": 0, "hits": 0, "pnl": 0,
                            "approved": ov.get("approved", False),
                            "manual_override": True,
                        }
                n_overrides = len(overrides)
                if n_overrides:
                    logger.info("RT KOL whitelist: %d manual overrides applied", n_overrides)

            _rt_kol_whitelist = cached
            _rt_kol_whitelist_loaded_at = now
            approved_count = sum(1 for v in cached.values() if v.get("approved"))
            logger.info("RT KOL whitelist (cached): %d/%d approved", approved_count, len(cached))
            return _rt_kol_whitelist

        # Fallback: live query paper_trades (first boot before auto_backtest runs)
        logger.info("RT KOL whitelist: no cache in scoring_config, querying paper_trades live")
        return _rt_load_kol_whitelist_live(config)

    except Exception as e:
        logger.warning("RT KOL whitelist load failed: %s", e)
    return _rt_kol_whitelist


def _rt_load_kol_whitelist_live(config: dict) -> dict:
    """Fallback: compute whitelist from paper_trades directly (used before auto_backtest populates cache)."""
    global _rt_kol_whitelist, _rt_kol_whitelist_loaded_at

    kf = config.get("kol_filter", {})
    wr_threshold = float(kf.get("wr_threshold", 0.40))
    min_calls = int(kf.get("min_calls", 3))
    lookback_days = int(kf.get("lookback_days", 7))
    # v91: fetch active strategies to filter to current config only
    active_strats = None
    ptc = None
    try:
        sb = _get_supabase()
        if sb:
            cfg_r = sb.table("scoring_config").select("paper_trade_config").eq("id", 1).execute()
            if cfg_r.data and cfg_r.data[0].get("paper_trade_config"):
                ptc = cfg_r.data[0]["paper_trade_config"]
                if isinstance(ptc, str):
                    import json as _json
                    ptc = _json.loads(ptc)
                active_strats = set(ptc.get("active_strategies", []))
    except Exception:
        pass

    # v92: dynamic deprecated from DB config
    deprecated = set(ptc.get("deprecated_strategies", [])) if ptc else set()
    if not deprecated:
        deprecated = {"MOONBAG", "WIDE_RUNNER", "SCALE_OUT", "TP100_SL30"}

    try:
        sb = _get_supabase()
        if not sb:
            return _rt_kol_whitelist
        cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()

        # v91: include shadow trades, add pnl_pct for win condition
        result = sb.table("paper_trades").select(
            "kol_group, pnl_usd, pnl_pct, strategy, is_shadow"
        ).neq("status", "open").filter(
            "kol_group", "not.is", "null"
        ).gte("exit_at", cutoff).execute()
        rows = result.data or []
        if not rows:
            logger.warning("RT KOL whitelist live: no closed RT paper trades in last %dd", lookback_days)
            return _rt_kol_whitelist

        rows = [r for r in rows if r.get("strategy") not in deprecated]
        if active_strats:
            rows = [r for r in rows if r.get("strategy") in active_strats]
        if not rows:
            return _rt_kol_whitelist

        from collections import defaultdict
        # v91: win condition = pnl_pct > 0 (works for real + shadow)
        agg = defaultdict(lambda: {"total": 0, "wins": 0, "pnl": 0.0})
        for r in rows:
            kol = r["kol_group"]
            pnl_pct = float(r.get("pnl_pct") or 0)
            pnl_usd = float(r.get("pnl_usd") or 0)
            agg[kol]["total"] += 1
            agg[kol]["pnl"] += pnl_usd
            if pnl_pct > 0:
                agg[kol]["wins"] += 1

        whitelist = {}
        approved_count = 0
        for kol, stats in agg.items():
            wr = stats["wins"] / stats["total"] if stats["total"] > 0 else 0
            has_enough = stats["total"] >= min_calls
            # v92: PnL-based approval — profitable KOLs with low WR still approved
            # WR path requires pnl > -20 to block false positives (e.g. PowsGemCalls: 60% WR but -$83)
            approved = has_enough and (stats["pnl"] > 0 or (wr >= 0.50 and stats["pnl"] > -20))
            whitelist[kol] = {
                "wr": round(wr, 4),
                "total": stats["total"],
                "hits": stats["wins"],
                "pnl": round(stats["pnl"], 2),
                "approved": approved,
            }
            if approved:
                approved_count += 1

        # Fallback: relax if < 3 KOLs pass — accept mildly negative PnL
        if approved_count < 3:
            for kol, stats in agg.items():
                if not whitelist[kol]["approved"] and stats["pnl"] > -5.0 and stats["total"] >= min_calls:
                    whitelist[kol]["approved"] = True
                    approved_count += 1

        _rt_kol_whitelist = whitelist
        _rt_kol_whitelist_loaded_at = time.time()
        logger.info("RT KOL whitelist (live): %d/%d approved (pnl>0 or wr>=50%%, trades>=%d, last %dd)",
                     approved_count, len(whitelist), min_calls, lookback_days)
    except Exception as e:
        logger.warning("RT KOL whitelist live query failed: %s", e)
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


def _rt_strategy_bankrolls_for_chain(bankroll_row: dict, chain: str) -> dict:
    """v14e: Return the per-chain strategy bankroll dict for one chain, with
    read-only fallback to the legacy flat dict if the new column isn't yet
    populated for this chain. Callers must NOT mutate the returned dict —
    use _rt_update_bankroll(chain=...) for writes.
    """
    if not bankroll_row:
        return {}
    per_chain = bankroll_row.get("strategy_bankrolls_per_chain") or {}
    if isinstance(per_chain, str):
        try:
            per_chain = json.loads(per_chain)
        except Exception:
            per_chain = {}
    c = (chain or "solana").lower()
    got = per_chain.get(c)
    if got:
        return got
    # Legacy fallback: flat dict. Only used if the per-chain column wasn't
    # populated (pre-v14e migration). Post-migration all reads go through the
    # nested column.
    flat = bankroll_row.get("strategy_bankrolls") or {}
    if not flat:
        return {}
    # Only return the subset that maps to this chain by naming heuristic.
    prefix_map = {"ethereum": "ETH_", "bsc": "BSC_", "base": "BASE_"}
    prefix = prefix_map.get(c)
    if prefix is None:  # solana = everything not prefixed ETH_/BSC_/BASE_
        return {k: v for k, v in flat.items()
                if not (k.startswith("ETH_") or k.startswith("BSC_") or k.startswith("BASE_"))}
    return {k: v for k, v in flat.items() if k.startswith(prefix)}


def _rt_update_bankroll(pnl_usd: float, n_trades: int, strategy: str = "",
                        chain: str = "solana") -> None:
    """v71: Update bankroll after trades close. Read-modify-write with peak/drawdown tracking.
    v115: Also tracks per-strategy bankroll in strategy_bankrolls JSONB.
    v14e: Per-chain bankroll in strategy_bankrolls_per_chain — each chain has
    its own isolated pot. Legacy flat strategy_bankrolls kept in sync for one
    release cycle (read-only mirror, easy rollback)."""
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

        update_data = {
            "current_balance": new_balance,
            "total_pnl": new_total_pnl,
            "total_trades": new_total_trades,
            "peak_balance": new_peak,
            "max_drawdown_pct": new_dd,
            "last_updated_at": datetime.now(timezone.utc).isoformat(),
        }

        # v14e: write to per-chain nested dict. Strategy scoped to its chain.
        if strategy:
            c = (chain or "solana").lower()
            per_chain = row.get("strategy_bankrolls_per_chain") or {}
            if isinstance(per_chain, str):
                try:
                    per_chain = json.loads(per_chain)
                except Exception:
                    per_chain = {}
            chain_bucket = dict(per_chain.get(c) or {})
            entry = dict(chain_bucket.get(strategy) or {"balance": 500, "trades": 0, "pnl": 0})
            entry["balance"] = round(float(entry.get("balance", 500)) + pnl_usd, 2)
            entry["pnl"] = round(float(entry.get("pnl", 0)) + pnl_usd, 2)
            entry["trades"] = int(entry.get("trades", 0)) + n_trades
            chain_bucket[strategy] = entry
            per_chain[c] = chain_bucket
            update_data["strategy_bankrolls_per_chain"] = per_chain

            # Legacy mirror — keep the flat dict in sync so pre-v14e readers
            # still see fresh numbers during rollout. Remove next release.
            # v14e.32: defensive .get() with defaults — Apr 26 PM bug: a manual
            # bankroll reset put 'pnl_usd'/'total_trades' field names instead
            # of the expected 'pnl'/'trades'. This branch was raising KeyError
            # on every ETH trade close ("RT bankroll update failed: 'pnl'")
            # which silently swallowed all bankroll updates for ~3h. 110 trades
            # had to be credited manually after the fact.
            strat_bankrolls = row.get("strategy_bankrolls") or {}
            legacy_entry = dict(strat_bankrolls.get(strategy) or {"balance": 500, "trades": 0, "pnl": 0})
            legacy_entry["balance"] = round(float(legacy_entry.get("balance", 500)) + pnl_usd, 2)
            legacy_entry["pnl"] = round(float(legacy_entry.get("pnl", 0)) + pnl_usd, 2)
            legacy_entry["trades"] = int(legacy_entry.get("trades", 0)) + n_trades
            strat_bankrolls[strategy] = legacy_entry
            update_data["strategy_bankrolls"] = strat_bankrolls

        sb.table("rt_bankroll").update(update_data).eq("id", row["id"]).execute()

        # Invalidate cache so next read gets fresh data
        _rt_bankroll_loaded_at = 0
        strat_str = f" [{chain}/{strategy}]" if strategy else ""
        logger.info("RT bankroll: $%.2f → $%.2f (pnl=$%+.2f, %d trades, peak=$%.2f, dd=%.1f%%)%s",
                     old_balance, new_balance, pnl_usd, n_trades, new_peak, new_dd, strat_str)
    except Exception as e:
        # v14e.32: include strategy + chain context so the next silent failure
        # is debuggable without grepping. Apr 26 PM bug went undetected for 3h
        # because logs only said "RT bankroll update failed: 'pnl'" with no
        # hint which strat/chain was failing.
        logger.error("RT bankroll update failed for [%s/%s] pnl=$%+.2f: %r",
                     chain, strategy, pnl_usd, e)


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
        "min_position_usd": 10.0,
        "max_position_usd": 100.0,
        "rt_strategies": "all",  # "all" = all 7 strategies
        "score_weights": {
            "kol_quality": 0.35,
            "token_safety": 0.30,
            "market_momentum": 0.20,
            "confirmation": 0.15,
        },
        # v110: Kelly-based sizing — bankroll × kelly_fraction. No multipliers.
        "sizing": {
            "kelly_fraction": 0.127,  # Half-Kelly 12.7%
        },
    }
    try:
        sb = _get_supabase()
        if sb:
            result = sb.table("scoring_config").select("rt_trade_config,rt_ml_weights").eq("id", 1).execute()
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
            # Also load rt_ml_weights (blend weights for KCO + RT ML)
            if result.data and result.data[0].get("rt_ml_weights"):
                ml_w = result.data[0]["rt_ml_weights"]
                if isinstance(ml_w, str):
                    ml_w = json.loads(ml_w)
                defaults["rt_ml_weights"] = ml_w
    except Exception as e:
        logger.warning("RT: failed to load rt_trade_config: %s (using defaults)", e)
    _rt_config = defaults
    _rt_config_loaded_at = now
    RT_COOLDOWN_SECONDS = int(defaults.get("cooldown_seconds", 1800))
    return _rt_config


def _rt_should_trade(kol: str, ca: str) -> bool:
    """Cooldown per (KOL × token). Does NOT consume — call _rt_mark_traded() on success.
    v94: DB fallback — survives restarts.
    v96: Also checks batch trades to prevent RT+batch double-trading same token."""
    now = time.time()
    key = (kol, ca)
    last = _rt_trade_cooldown.get(key, 0)
    if now - last < RT_COOLDOWN_SECONDS:
        return False
    # DB fallback: check if THIS KOL already traded this CA recently (any source)
    try:
        sb = _get_supabase()
        if sb:
            cutoff = (datetime.now(timezone.utc) - timedelta(seconds=RT_COOLDOWN_SECONDS)).isoformat()
            # v96: Check RT trades (same kol+ca) — different KOL = new signal, so keep per-KOL
            res = sb.table("paper_trades").select("id", count="exact") \
                .eq("kol_group", kol).eq("token_address", ca) \
                .eq("is_shadow", False) \
                .gte("created_at", cutoff).execute()
            if res.count and res.count > 0:
                _rt_trade_cooldown[key] = now  # warm cache
                return False
    except Exception:
        pass
    return True


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

    v141: additive bonuses data-driven (backfill audit on N=74 closed trades):
      +8 if 0.3h < age < 1.3h (sweet spot: Q2/Q3 age = +18% avg; <0.3h = -18% rug)
      +8 if bsr > 0.7 (strong buy pressure: Q4 bsr = +17.87% avg)
      -5 if liq = 0 (pump.fun pre-grad toxic: measured -$19.82/jour drag)
    Measured corr +0.207 → +0.236 (+14%). Filter≥30 avg +12.45% → +19.01% (+53%).
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

    # v141 additive bonuses — data-driven, tested via _rt_score_v2_audit.py
    age_h = float(token_info.get("token_age_hours") or 0)
    bsr = float(token_info.get("buy_sell_ratio") or 0.5)
    liq = float(token_info.get("liquidity_usd") or 0)
    if 0.3 < age_h < 1.3:
        rt_score += 8  # sweet spot: fresh but past rug window
    if bsr > 0.7:
        rt_score += 8  # strong buy pressure
    if liq <= 0:
        rt_score -= 5  # pump.fun pre-grad penalty

    return round(max(0, min(100, rt_score)), 1)


def _rt_compute_score_v2(kol_info: dict, token_info: dict) -> float:
    """v14e.43 — alternative score formula derived from reverse-engineering
    (`scripts/_score_reverse_engineer.py`) on 65K closed paper trades.

    Findings: AUC 0.54 SOL / 0.50 ETH on rt_score actuel — mediocre. The 4
    strongest single predictors empirically are:
      - kol_win_rate (LogReg coef +0.451 on SOL)
      - rt_buy_sell_ratio  (cross-strat universal lift +5-8pp WR at thr>=0.52)
      - log(rt_volume_24h)
      - log(entry_mcap)

    Formula (clipped 0-100):
      30*kol_win_rate + 5*min(bsr, 5) + 8*log10(volume+1) + 8*log10(mcap+1)

    v14e.55 (May 2) audit on N=37K paired trades: V2 AUC = 0.486 (anti-predictive).
    All 4 features are INDIVIDUALLY anti-predictive in current regime:
    kol_win_rate AUC 0.45, BSR 0.43, vol 0.50, mcap 0.50 — V2 just scaled them.
    Kept here for backward compatibility (rt_score_v2 column still populated)
    but superseded by V3 (binned bonuses on V1 base, AUC 0.562). See _rt_compute_score_v3.
    """
    import math
    win_rate = float(kol_info.get("win_rate") or 0)
    bsr = float(token_info.get("buy_sell_ratio") or 0)
    vol = float(token_info.get("volume_24h") or 0)
    mcap = float(token_info.get("market_cap") or 0)
    score_v2 = (
        30.0 * win_rate
        + 5.0 * min(bsr, 5.0)
        + 8.0 * math.log10(max(1.0, vol))
        + 8.0 * math.log10(max(1.0, mcap))
    )
    return round(max(0.0, min(100.0, score_v2)), 1)


def _rt_compute_score_v3(rt_score: float, token_info: dict) -> float:
    """v14e.55 (May 2) — V1 base + binned bonuses on the strongest non-monotonic
    signals discovered via 7d SOL audit (N=70,740 closed trades):

      MCAP sweet spot 50-100k:  WR 55.3% vs base 42% (+13pp) — biggest signal
      AGE sweet spot 1-3h:      WR 52.1% vs base 42% (+10pp)
      AGE sweet spot 24-72h:    WR 46.7% vs base 42% (+5pp)
      MCAP >= 1M:               WR 33.8% — dump zone
      AGE 3-6h:                 WR 31.8% — known dead zone
      LIQ in (0, 10k):          WR 32.9% avg PnL -0.18 — micro-rug zone

    Performance vs V1 (paired SOL data, multiple windows):
      AUC: V1 0.534 -> V3 0.562 (+0.028)
      Filter >=35: V1 WR 43.5%, +$0.010 -> V3 WR 45.8%, +$0.021 (2x PnL)
      Filter >=35 trades: +6-10% throughput (bonuses lift more rows)

    Computed AND PERSISTED IN PARALLEL with rt_score and rt_score_v2 for
    shadow A/B walk-forward validation. NOT used for filtering yet — collect
    7-14 days then re-audit on out-of-sample data before switching strat
    filters from rt_score to rt_score_v3.
    """
    age_h = float(token_info.get("token_age_hours") or 0)
    liq = float(token_info.get("liquidity_usd") or 0)
    mcap = float(token_info.get("market_cap") or 0)

    score_v3 = float(rt_score)
    if 50_000 <= mcap < 100_000: score_v3 += 15
    elif mcap >= 1_000_000:      score_v3 -= 10
    if 1.0 <= age_h < 3.0:       score_v3 += 10
    elif 24.0 <= age_h < 72.0:   score_v3 += 8
    elif 3.0 <= age_h < 6.0:     score_v3 -= 8
    if 0 < liq < 10_000:         score_v3 -= 10

    return round(max(0.0, min(100.0, score_v3)), 1)


def _rt_position_size(rt_score: float, kol_info: dict, token_info: dict,
                      tier: str, config: dict, strategy: str = "") -> float:
    """
    v111: Kelly-based position sizing — bankroll × kelly_fraction, capped at max_position_usd.
    v116: Per-strategy bankroll — each strategy sizes from its own balance (parallel mode).
    """
    sizing = config.get("sizing", {})
    kelly = float(sizing.get("kelly_fraction", 0.127))
    max_pos = float(config.get("max_position_usd", 120))
    bankroll = _rt_load_bankroll()

    # v116: Use strategy-specific bankroll if available
    if strategy:
        strat_bankrolls = bankroll.get("strategy_bankrolls") or {}
        strat_entry = strat_bankrolls.get(strategy, {})
        balance = float(strat_entry.get("balance", 500))
    else:
        balance = float(bankroll.get("current_balance", 500))

    raw = balance * kelly
    return round(min(raw, max_pos), 2)


def _rt_ml_position_mult(
    username: str, tier: str,
    kol_info: dict, token_info: dict,
    rt_score: float, config: dict,
) -> tuple[float, str]:
    """v79: Unified ML position multiplier. Weighted blend of KCO (primary) + RT ML (secondary).

    KCO has N=1,035 labeled calls → more reliable signal.
    RT ML has N=~90 paper trades → smaller but strategy-specific.
    Weights auto-updated every 2h by auto_backtest._compute_rt_ml_weights()
    based on direction accuracy from paper_trades A/B data.

    Returns (mult: float in [floor, cap], label: str for logging).
    """
    floor = float(config.get("rt_ml_mult_floor", 0.4))
    cap   = float(config.get("rt_ml_mult_cap",   1.8))

    # Load blend weights (auto-updated by auto_backtest every 2h)
    w = config.get("rt_ml_weights") or {}
    kco_w  = float(w.get("kco_w",  0.7))
    rtml_w = float(w.get("rtml_w", 0.3))

    # --- KCO signal [-1, +1] ---
    kco_signal = 0.0
    kco_label  = ""
    try:
        from pipeline import predict_kco_score
        kco_pred = predict_kco_score(username, tier, token_info)
        if kco_pred is not None:
            # 1.5x = neutral(0), 2.0x = +1, 1.0x = -1
            kco_signal = max(-1.0, min(1.0, (kco_pred - 1.5) / 0.5))
            kco_label  = f"kco={kco_pred:.2f}x"
            token_info["_rt_kol_ml_pred"] = kco_pred  # → paper_trades.kol_ml_pred
    except Exception as e:
        logger.debug("KCO prediction unavailable: %s", e)
        kco_w = 0.0  # fallback: all weight to RT ML

    # --- RT ML signal [-1, +1] ---
    rtml_signal = 0.0
    rtml_label  = ""
    if _rt_ml_model is not None:
        try:
            from rt_model import predict_strategy_pnl
            hour  = datetime.now(timezone.utc).hour
            preds = predict_strategy_pnl(
                _rt_ml_model, kol_info, token_info, tier,
                rt_score, n_confirmations=0, hour=hour,
            )
            avg_pred = sum(preds.values()) / len(preds) if preds else 0.0
            # ±30% PnL maps to ±1 signal
            rtml_signal = max(-1.0, min(1.0, avg_pred / 0.30))
            rtml_label  = f"rtml={avg_pred:.1%}"
            token_info["_rt_ml_pred"] = avg_pred
        except Exception as e:
            logger.debug("RT ML prediction failed: %s", e)
            rtml_w = 0.0  # fallback: all weight to KCO

    # Normalize weights (graceful degradation when one model unavailable)
    total_w = kco_w + rtml_w
    if total_w <= 0:
        return 1.0, ""  # both unavailable → neutral

    kco_w  /= total_w
    rtml_w /= total_w

    # Weighted blend → mult in [floor, cap]
    signal = kco_w * kco_signal + rtml_w * rtml_signal
    mult   = 1.0 + (cap - floor) / 2.0 * signal
    mult   = round(max(floor, min(cap, mult)), 3)

    label_parts = [p for p in [kco_label, rtml_label] if p]
    label = " ".join(label_parts) + (f" blend={mult:.2f}x" if label_parts else "")
    return mult, label


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
                    kol_info: dict, token_info: dict, config: dict, msg=None):
    """
    v71: Open paper trades with hybrid strategy allocation.
    When hybrid_strategy.enabled: split pos_size across configured strategies (e.g. 60/40).
    Fallback to exploration mode if disabled.
    """
    from paper_trader import open_paper_trades, _load_paper_trade_config, STRATEGIES
    sb = _get_supabase()
    if not sb:
        return 0

    # v14e.28: normalize EVM addresses to lowercase BEFORE dedup queries.
    # Bug surfaced Apr 26: $HENRY (0x1c798B...) opened twice by Luca_Apes 4min
    # apart because the first row stored mixed-case (0x1c798BFDf4...) and the
    # second came in lowercase. Postgres .eq() is case-sensitive on text, so
    # the dedup didn't match → two positions on the same token.
    try:
        from chain_detect import normalize_address
        ca = normalize_address(ca)
    except Exception:
        pass

    # v94: Max positions per token (concentration limit)
    max_per_token = int(config.get("max_positions_per_token", 2))
    try:
        open_on_token = sb.table("paper_trades").select("id", count="exact") \
            .eq("token_address", ca).eq("status", "open") \
            .eq("is_shadow", False).execute()
        if open_on_token.count and open_on_token.count >= max_per_token:
            logger.info("RT SKIP: %s — %d open positions (max %d)", symbol, open_on_token.count, max_per_token)
            return 0
    except Exception as e:
        logger.warning("RT: token concentration check failed: %s", e)

    pt_config = _load_paper_trade_config(sb)

    # v14e: resolve chain once, propagate on token_entry so downstream (paper
    # fee model, strategy filter, alerter, live gate) all branch consistently.
    # Falls back to solana if detection fails (base58 shape already guaranteed
    # upstream by pipeline.extract_tokens / RT handler).
    try:
        from chain_detect import detect_chain
        token_chain = token_info.get("chain") or detect_chain(ca) or "solana"
    except Exception:
        token_chain = token_info.get("chain") or "solana"

    # v14e.17: per-KOL chain whitelist. If GROUPS_DATA[kol]["chains"] is set,
    # the KOL only opens paper/live trades on tokens whose chain is in that
    # list. Default (no `chains` field) = all chains allowed.
    # Used for bat_gamble (ETH-only): SOL calls bleed, ETH calls profit.
    kol_chain_whitelist = GROUPS_DATA.get(kol_username, {}).get("chains")
    if kol_chain_whitelist and token_chain not in kol_chain_whitelist:
        logger.info(
            "RT SKIP (kol chain filter v14e.17): %s on %s — KOL allowed only on %s",
            kol_username, token_chain, kol_chain_whitelist,
        )
        return 0

    # v14e.40 / v14e.41: RECALL detection. Query kol_call_outcomes for the FIRST
    # mention of this CA + the post-1st-call ATH (peak before this call).
    # Used by RECALL_* shadow strategies to test the "INVA pattern" — re-entering
    # on a dipped recall after another KOL already called the token.
    #
    # Two drift fields:
    #   recall_drift_pct      = (now / first_call_price) - 1   (vs entry of 1st)
    #   recall_drift_vs_peak  = (now / peak_after_first) - 1   (vs ATH between)
    # The peak-based drift catches pump-then-dump patterns where price > 1st-call
    # price for a while, then dumps below pic but above 1st-call entry.
    # Example $PARANOID 27 Apr: 1st @ $0.000112, pic $0.000193, recall @ $0.0000884.
    # drift_vs_first = -21% (passes DIP10 filter) but drift_vs_peak = -54% (passes
    # DIP30/DIP50). Both exposed; strats decide which to gate on.
    #
    # v14e.41: lower gate to 600s (10 min). Pump-then-dump cycles often resolve
    # within 15-30min — the prior 1800s threshold missed real INVA-style recalls.
    is_recall = False
    recall_drift_pct = None
    recall_drift_vs_peak = None
    hours_since_first_call = None
    first_call_price = None
    peak_after_first = None
    try:
        _first = (
            sb.table("kol_call_outcomes")
            .select("entry_price, call_timestamp, ath_after_call")
            .eq("token_address", ca)
            .order("call_timestamp", desc=False)
            .limit(1)
            .execute()
        )
        if _first.data and _first.data[0].get("entry_price"):
            _row = _first.data[0]
            _fp = float(_row["entry_price"])
            _ft = _row.get("call_timestamp")
            _ath = _row.get("ath_after_call")
            from datetime import datetime as _dt
            _ft_dt = _dt.fromisoformat(str(_ft).replace("Z", "+00:00")) if _ft else None
            now_utc = datetime.now(timezone.utc)
            if _fp > 0 and _ft_dt and (now_utc - _ft_dt).total_seconds() > 600:
                first_call_price = _fp
                recall_drift_pct = (price / _fp) - 1.0 if price > 0 else None
                hours_since_first_call = (now_utc - _ft_dt).total_seconds() / 3600.0
                if _ath and float(_ath) > 0 and price > 0:
                    peak_after_first = float(_ath)
                    recall_drift_vs_peak = (price / peak_after_first) - 1.0
                is_recall = True
                logger.info(
                    "RT RECALL: %s — drift_vs_1st=%.1f%% drift_vs_peak=%s @ $%.6g (1st %.1fh ago, peak $%.6g)",
                    symbol,
                    (recall_drift_pct or 0) * 100,
                    f"{recall_drift_vs_peak*100:.1f}%" if recall_drift_vs_peak is not None else "n/a",
                    _fp, hours_since_first_call, peak_after_first or 0,
                )
    except Exception as e:
        logger.debug("recall detection failed for %s: %s", ca, e)

    # Build base token entry with RT metadata
    token_entry = {
        "symbol": symbol,
        "token_address": ca,
        "chain": token_chain,
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
        "_rt_score_v2": _rt_compute_score_v2(kol_info, token_info),  # v14e.43 shadow A/B (data collection only)
        "_rt_score_v3": _rt_compute_score_v3(rt_score, token_info),  # v14e.55 shadow A/B (binned bonuses, V1 base)
        "_rt_liquidity_usd": token_info.get("liquidity_usd"),
        "_rt_volume_24h": token_info.get("volume_24h"),
        "_rt_buy_sell_ratio": token_info.get("buy_sell_ratio"),
        "_rt_token_age_hours": token_info.get("token_age_hours"),
        "_rt_is_pump_fun": token_info.get("is_pump_fun"),
        "_rt_pair_address": token_info.get("pair_address"),  # v109: track pool for migration debug
        "_rt_ml_pred": token_info.get("_rt_ml_pred"),        # v77: RT ML pred for A/B
        "_rt_kol_ml_pred": token_info.get("_rt_kol_ml_pred"),  # v78: KCO score (fix: was reading wrong key)
        # v121: Call reaction speed — message timestamp + price at message time
        "_rt_message_ts": msg.date.isoformat() if hasattr(msg, "date") and msg.date else None,
        "_rt_price_at_message": price,  # DexScreener price fetched right after msg (token_info lacks "price_usd" key)
        # Latency profiling — propagated to live_trader for final timing breakdown
        "_rt_t_msg": token_info.get("_rt_t_msg"),
        "_rt_t_ds_done": token_info.get("_rt_t_ds_done"),
        # v14e.40 / v14e.41: recall fields consumed by RECALL_* filters.
        # drift_pct = vs 1st-call entry, drift_vs_peak = vs post-1st-call ATH.
        "_rt_is_recall": is_recall,
        "_rt_recall_drift_pct": recall_drift_pct,
        "_rt_recall_drift_vs_peak": recall_drift_vs_peak,
        "_rt_hours_since_first_call": hours_since_first_call,
        "_rt_first_call_price": first_call_price,
        "_rt_peak_after_first": peak_after_first,
    }

    now = datetime.now(timezone.utc)
    total_opened = 0

    # v110: KOL whitelist gate — disabled by default (whitelist_enabled=false in rt_trade_config).
    # Dynamic trail strategy is profitable across all KOLs, whitelist not needed.
    whitelist_enabled = config.get("whitelist_enabled", False)
    if whitelist_enabled:
        kol_wl = _rt_load_kol_whitelist(config)
        kol_entry = kol_wl.get(kol_username, {})
        kol_approved = kol_entry.get("approved", False)
        if not kol_approved:
            logger.info("RT KOL REJECTED (shadow-only): %s — opening shadows but no main trades", kol_username)
            shadow_config = dict(pt_config)
            shadow_config["budget_usd"] = pos_size
            shadow_config["active_strategies"] = []
            shadow_config["top_n"] = 1
            shadow_config["ca_filter"] = False
            shadow_config["shadow_enabled"] = True
            token_entry["_rt_experiment_id"] = pt_config.get("experiment_id")
            token_entry["_rt_variant_id"] = "shadow_only_rejected_kol"
            open_paper_trades(sb, [token_entry], cycle_ts=now, config=shadow_config)
            return 0

    # v96: Hybrid strategy — split position across configured allocations
    # v116: Each strategy sizes independently from its own bankroll (parallel mode)
    # v142 E: LIVE trades open FIRST so paper can shadow-sync to the actual
    # Jupiter execution_price (eliminates P1 inversion + P4 ±9% entry divergence).
    hybrid_cfg = config.get("hybrid_strategy", {})
    if hybrid_cfg.get("enabled", False):
        allocations = dict(hybrid_cfg.get("allocations", {"TP50_SL30": 0.50, "TP30_SL50": 0.30, "TP50_SL50": 0.20}))
        all_hybrid_strats = [s for s in allocations if s in STRATEGIES]

        # v142 E — Phase 1: open LIVE trades first, collect their Jupiter fill
        # prices so paper can reuse them. If live disabled/fails, dict stays
        # empty and paper falls back to its own Ultra quote (prior behavior).
        live_cfg = config.get("live_trading", {})
        live_fill_prices: dict[str, float] = {}  # strategy -> execution_price
        # v14e: skip live branch entirely for non-Solana (Phase 2 ETH not yet
        # approved; BSC/Base have no live adapter). Paper still runs via the
        # loop below — only the Jupiter-backed live path is short-circuited.
        # v14e.14: per-KOL live opt-out. KOLs in live_trading.kol_blacklist
        # execute paper + shadow normally but never open live trades. Used
        # to test unproven KOLs (spammers, re-added blacklisted ones) without
        # putting capital at risk while N<30 paper sample builds.
        # v14e.37: also honor paper_trade_config.kol_chain_blacklist per-chain
        # (statistical reliable_blacklist, shared with paper main gate).
        # v14e.57: NEW live_trading.kol_chain_blacklist (per-chain, mirrors
        # paper). Lets us split a KOL like venom_gambles (SOL toxic / ETH
        # profitable) without giving up the all-chain ban or copying every
        # paper-banned KOL into the live list.
        live_kol_blacklist = set(live_cfg.get("kol_blacklist", []))
        live_chain_bl_raw = live_cfg.get("kol_chain_blacklist", {}) or {}
        live_chain_bl = set(live_chain_bl_raw.get(token_chain, []) or [])
        try:
            from paper_trader import _load_paper_trade_config
            _ptc = _load_paper_trade_config(_get_supabase()) or {}
            _chain_bl_raw = _ptc.get("kol_chain_blacklist", {}) or {}
            _chain_bl = set(_chain_bl_raw.get(token_chain, []) or [])
        except Exception:
            _chain_bl = set()
        _is_chain_banned = kol_username in _chain_bl
        _is_live_chain_banned = kol_username in live_chain_bl
        if (
            kol_username in live_kol_blacklist
            or _is_chain_banned
            or _is_live_chain_banned
        ):
            if _is_live_chain_banned:
                _reason = f"v14e.57 live chain blacklist ({token_chain})"
            elif _is_chain_banned:
                _reason = "v14e.37 chain blacklist"
            else:
                _reason = "v14e.14 flat blacklist"
            logger.info(
                "RT LIVE SKIP (%s): %s on %s — paper/shadow telemetry only",
                _reason, kol_username, token_chain,
            )
        elif live_cfg.get("enabled", False) and token_chain == "solana":
            try:
                from live_trader import open_live_trade
                from paper_trader import _passes_strategy_filter
                live_allocs = live_cfg.get("allocations", allocations)
                for strat_name, alloc_pct in live_allocs.items():
                    # v144.16: mirror paper gate — STRATEGY_FILTERS applies to live too
                    # (was paper-only → live BOND_FAST bought non-bonding tokens, e.g. $OOO @ liq $125K).
                    if not _passes_strategy_filter(token_entry, strat_name):
                        logger.info("RT LIVE SKIP (filter): %s/%s liq=$%.0f rt_score=%.1f",
                                    symbol, strat_name,
                                    float(token_entry.get("_rt_liquidity_usd") or 0),
                                    float(token_entry.get("_rt_score") or 0))
                        continue
                    live_pos = round(pos_size * float(alloc_pct), 2)
                    live_res = open_live_trade(sb, token_entry, strat_name, live_pos, live_cfg)
                    if isinstance(live_res, dict) and live_res.get("success"):
                        exe_p = live_res.get("execution_price")
                        if exe_p and float(exe_p) > 0:
                            live_fill_prices[strat_name] = float(exe_p)
            except Exception as e:
                logger.error("Live trading (open) failed: %s", e)
        # v14e.28: ETH live dispatch via Uniswap V3 + Flashbots Protect.
        # Gated behind a SEPARATE flag (eth_live_enabled) so flipping
        # live_trading.enabled for SOL doesn't accidentally open ETH positions.
        #
        # Close-side: live_trader_eth.check_live_trades_eth is wired in the
        # same live monitor loop below (~line 2722), so ETH positions auto-close
        # via Uniswap V3 + Flashbots Protect when eth_live_enabled=True. Validated
        # in production (v14e.28+: $MUSK +$10.67 auto-recovered via retry-sell).
        elif (live_cfg.get("enabled", False)
              and token_chain == "ethereum"
              and live_cfg.get("eth_live_enabled", False)):
            try:
                from live_trader_eth import open_live_trade as open_live_trade_eth
                from paper_trader import _passes_strategy_filter
                eth_live_allocs = live_cfg.get("eth_allocations", live_cfg.get("allocations", allocations))
                for strat_name, alloc_pct in eth_live_allocs.items():
                    if not _passes_strategy_filter(token_entry, strat_name):
                        logger.info("RT ETH LIVE SKIP (filter): %s/%s liq=$%.0f rt_score=%.1f",
                                    symbol, strat_name,
                                    float(token_entry.get("_rt_liquidity_usd") or 0),
                                    float(token_entry.get("_rt_score") or 0))
                        continue
                    live_pos = round(pos_size * float(alloc_pct), 2)
                    live_res = open_live_trade_eth(sb, token_entry, strat_name, live_pos, live_cfg)
                    if isinstance(live_res, dict) and live_res.get("success"):
                        exe_p = live_res.get("execution_price")
                        if exe_p and float(exe_p) > 0:
                            live_fill_prices[strat_name] = float(exe_p)
                        logger.info("RT ETH LIVE OPENED: %s/%s pos=$%.2f tx=%s gas=$%.2f",
                                    symbol, strat_name, live_pos,
                                    live_res.get("tx_hash", "?")[:14],
                                    float(live_res.get("gas_usd") or 0))
                    elif isinstance(live_res, dict):
                        logger.warning("RT ETH LIVE FAIL: %s/%s err=%s",
                                       symbol, strat_name, live_res.get("error"))
            except Exception as e:
                logger.error("ETH live trading (open) failed: %s", e)

        # v142 E — Phase 2: open PAPER per-strat, injecting live's execution_price
        # when available so entry_price matches bit-for-bit.
        for i, (strat_name, alloc_pct) in enumerate(allocations.items()):
            if strat_name not in STRATEGIES:
                continue
            # v116: Each strategy computes its own pos_size from its own bankroll
            strat_pos = _rt_position_size(rt_score, kol_info, token_info, tier, config, strategy=strat_name)
            strat_pos = round(strat_pos * float(alloc_pct), 2)
            if strat_pos < 1.0:
                strat_pos = 1.0

            strat_config = dict(pt_config)
            strat_config["budget_usd"] = strat_pos
            strat_config["active_strategies"] = [strat_name]
            # Shadow knows all hybrid strategies to avoid blocking sibling real trades
            strat_config["all_real_strategies"] = all_hybrid_strats
            strat_config["top_n"] = 1
            strat_config["ca_filter"] = False
            # Only first call opens shadows (others would dedup anyway)
            if i > 0:
                strat_config["shadow_enabled"] = False

            # v142 E: shadow-sync entry_price from live's actual Jupiter fill
            # (per-strategy — each live strat has its own fill price).
            strat_token = dict(token_entry)
            strat_token["_rt_experiment_id"] = pt_config.get("experiment_id")
            strat_token["_rt_variant_id"] = "hybrid_default"
            if strat_name in live_fill_prices:
                strat_token["_rt_force_entry_price"] = live_fill_prices[strat_name]

            opened = open_paper_trades(sb, [strat_token], cycle_ts=now, config=strat_config)
            total_opened += opened

        if total_opened > 0:
            alloc_str = " + ".join(f"{s}={p:.0%}" for s, p in allocations.items())
            sync_str = f" [sync={len(live_fill_prices)}]" if live_fill_prices else ""
            logger.info(
                "RT TRADE [HYBRID]: %s @ $%.6f | %s%s | KOL: %s (%s, wr=%.0f%%) "
                "rt_score=%.0f pos=$%.2f liq=$%.0fK",
                symbol, price, alloc_str, sync_str, kol_username, tier,
                kol_info.get("win_rate", 0) * 100,
                rt_score, pos_size,
                token_info.get("liquidity_usd", 0) / 1000,
            )

        return total_opened

    # --- Fallback: exploration mode (all active strategies, equal sizing) ---
    # v144: shadow-sync entry also fires in exploration path. If live_trading is
    # enabled without hybrid_strategy, open live trades first (per alloc) so paper
    # rows can reuse the Jupiter fill price via `_rt_force_entry_price`. Prevents
    # sync=False outliers on any future non-hybrid config.
    live_cfg = config.get("live_trading", {})
    live_fill_prices: dict[str, float] = {}
    # v14e: same chain gate as hybrid branch — no live on non-Solana.
    if live_cfg.get("enabled", False) and token_chain == "solana":
        try:
            from live_trader import open_live_trade
            from paper_trader import _passes_strategy_filter
            live_allocs = live_cfg.get("allocations") or {}
            for strat_name, alloc_pct in live_allocs.items():
                if strat_name not in STRATEGIES:
                    continue
                # v144.16: mirror paper gate on live (see hybrid branch).
                if not _passes_strategy_filter(token_entry, strat_name):
                    logger.info("RT LIVE SKIP (filter): %s/%s liq=$%.0f rt_score=%.1f",
                                symbol, strat_name,
                                float(token_entry.get("_rt_liquidity_usd") or 0),
                                float(token_entry.get("_rt_score") or 0))
                    continue
                live_pos = round(pos_size * float(alloc_pct), 2)
                live_res = open_live_trade(sb, token_entry, strat_name, live_pos, live_cfg)
                if isinstance(live_res, dict) and live_res.get("success"):
                    exe_p = live_res.get("execution_price")
                    if exe_p and float(exe_p) > 0:
                        live_fill_prices[strat_name] = float(exe_p)
            if not live_allocs:
                logger.warning("live_trading.enabled=True but no allocations defined — no live trades opened in exploration mode.")
        except Exception as e:
            logger.error("Exploration-mode live trading (open) failed: %s", e)

    db_active = [s for s in pt_config.get("active_strategies", list(STRATEGIES.keys())) if s in STRATEGIES]
    rt_strats = config.get("rt_strategies", "all")
    if rt_strats == "all":
        valid_strategies = db_active
    else:
        valid_strategies = [s for s in rt_strats if s in STRATEGIES]
        if not valid_strategies:
            valid_strategies = db_active

    # v144: per-strategy paper open so we can inject live fill price when present
    # (mirrors hybrid branch). Pre-v144 used single batch call; still falls back
    # to batch when no live fills to preserve behavior.
    total_opened = 0
    if live_fill_prices:
        for strat_name in valid_strategies:
            strat_config = dict(pt_config)
            strat_config["budget_usd"] = pos_size
            strat_config["active_strategies"] = [strat_name]
            strat_config["top_n"] = 1
            strat_config["ca_filter"] = False
            strat_token = dict(token_entry)
            if strat_name in live_fill_prices:
                strat_token["_rt_force_entry_price"] = live_fill_prices[strat_name]
            total_opened += open_paper_trades(sb, [strat_token], cycle_ts=now, config=strat_config)
    else:
        explore_config = dict(pt_config)
        explore_config["budget_usd"] = pos_size
        explore_config["active_strategies"] = valid_strategies
        explore_config["top_n"] = 1
        explore_config["ca_filter"] = False
        total_opened = open_paper_trades(sb, [token_entry], cycle_ts=now, config=explore_config)

    if total_opened > 0:
        sync_str = f" [sync={len(live_fill_prices)}]" if live_fill_prices else ""
        logger.info(
            "RT TRADE [EXPLORE]: %s @ $%.6f | %d strats%s | KOL: %s (%s, wr=%.0f%%) "
            "rt_score=%.0f pos=$%.2f liq=$%.0fK",
            symbol, price, len(valid_strategies), sync_str, kol_username, tier,
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
        "mcap": float(raw.get("market_cap") or raw.get("mcap") or 0),  # v109: fix key name
        "price_change_1h": float(raw.get("price_change_1h") or 0),
        "price_change_5m": float(raw.get("price_change_5m") or 0),
        # v109: bonding curve support — needed for liq gate bypass
        "pump_graduation_status": raw.get("pump_graduation_status"),
        "dex_id": raw.get("dex_id"),
        "pair_address": raw.get("pair_address"),
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

    # v109+: Filter out flex/recap/PnL-report messages — these are NOT new calls.
    # KOLscope bot posts "MULTIPLIER DETECTED: 5x+", "CALL ALERT", "DIP MODE" etc.
    # KOLs post recaps like "100k → 700k (7x)" or "last few goated calls".
    # These contain CAs but are NOT entry signals — they refer to past gains.
    import re as _re
    # Strip Solana + Ethereum addresses from text before flex matching to avoid
    # false positives. CAs like "Dq3to3Yw..." contain substrings that match PnL
    # patterns ("3to3"). v14: ETH 0x addresses also stripped so hex chars don't
    # trigger the \d+[xX] regex family.
    _SOLANA_ADDR = _re.compile(r'[1-9A-HJ-NP-Za-km-z]{32,44}')
    _ETH_ADDR_STRIP = _re.compile(r'0x[a-fA-F0-9]{40}')
    _text_no_ca = _ETH_ADDR_STRIP.sub('', _SOLANA_ADDR.sub('', text))
    _FLEX_PATTERNS = [
        # --- Bot-generated messages ---
        _re.compile(r'KOLscope|KOLscopeBot', _re.IGNORECASE),
        _re.compile(r'MULTIPLIER DETECTED', _re.IGNORECASE),
        _re.compile(r'CALL ALERT:.*✈', _re.IGNORECASE),
        _re.compile(r'DIP MODE', _re.IGNORECASE),
        _re.compile(r'STATUS UNLOCKED', _re.IGNORECASE),
        _re.compile(r'Achievement Unlocked', _re.IGNORECASE),
        _re.compile(r'reached\s+[\d.]+[xX]\s+after', _re.IGNORECASE),
        _re.compile(r'Buy!\n[🟢🐳🔵🟣🔴]{20,}', _re.IGNORECASE),
        # --- Gain celebrations ---
        _re.compile(r'(made|hit|scored|bagged|caught)\s+\d+x', _re.IGNORECASE),
        _re.compile(r'\d+x\s*(🔥|🔫|💰|💎|🚀|from\s+(call|bottom|dip)|since\s+(call|entry)|\+)', _re.IGNORECASE),
        _re.compile(r'^\d+[xX]\s*\n', _re.MULTILINE),  # standalone "8X\n"
        # --- Follow-up/update flex ("up 2x from our entry", "we're now up over 3x") ---
        _re.compile(r'(up|gained|profits?)\s+(over\s+)?\d+[xX%]?\s*(from|since)\s+(our|my|the)\s+(entry|call|dip)', _re.IGNORECASE),
        _re.compile(r"we.re\s+(now\s+)?up\s+(over\s+)?\d+[xX]", _re.IGNORECASE),
        _re.compile(r'already\s+(went|gone|did)\s+\d+x', _re.IGNORECASE),
        _re.compile(r'(new|another)\s+ATH', _re.IGNORECASE),
        # --- Cabal/update brags ("10X since cabal", "3X since my update", "another 10X in cabal") ---
        _re.compile(r'\d+[xX]\s+since\s+(cabal|update|call|entry|my)', _re.IGNORECASE),
        _re.compile(r'(another|yet another)\s+\d+[xX]\s+(in|from|for)', _re.IGNORECASE),
        _re.compile(r'since\s+(cabal\s+)?chat\s+call', _re.IGNORECASE),
        _re.compile(r'for\s+cabal\s+chads', _re.IGNORECASE),
        # --- PnL reports ---
        _re.compile(r'\d+[km]?\s*(->|→|⮕|to)\s*\d+[km]?', _re.IGNORECASE),
        # --- Recaps ---
        _re.compile(r'last (few|couple|recent)\s+.*calls', _re.IGNORECASE),
        _re.compile(r'(very )?recent shills', _re.IGNORECASE),
        _re.compile(r'keep\s+print', _re.IGNORECASE),
        # --- Twitter/X reposts (forwarded content, not original calls) ---
        _re.compile(r'^.{0,5}by\s+@\w+\s+\(\d+', _re.IGNORECASE),
        # --- Guest posts in KOL channels ---
        _re.compile(r'^Yo its @\w+', _re.IGNORECASE),
    ]
    for pat in _FLEX_PATTERNS:
        if pat.search(_text_no_ca):
            logger.debug("RT FLEX SKIP: %s — matched %s", username, pat.pattern[:40])
            return

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

    # v96: No KOL gates — paper trade ALL calls to build unbiased leaderboard.
    # Gates (whitelist, probation, blacklist) removed. Only hard blocks: CA + liquidity.
    # KOL filtering should happen AFTER leaderboard data is collected, not before.

    for symbol, source, ca in tokens:
        if not ca:
            continue

        # Cooldown per (KOL × token) — different KOL on same token = new signal
        if not _rt_should_trade(username, ca):
            continue

        # v108: Lock on CA only (not per-KOL) to prevent double-inserts when
        # two KOLs post the same token within seconds. The second KOL waits until
        # the first finishes inserting, so the dedup check in open_paper_trades sees
        # the already-open trades and skips.
        flight_key = ca
        if flight_key in _rt_in_flight:
            logger.debug("RT SKIP: %s — already in-flight from another KOL", ca[:8])
            continue
        _rt_in_flight.add(flight_key)

        try:
            _t_event = time.time()
            _t_msg = msg.date.timestamp() if (hasattr(msg, "date") and msg.date) else _t_event
            logger.info("RT detect: %s (CA: %s...) from %s (%s-tier) | msg→detect=%.2fs",
                        symbol, ca[:8], username, tier, _t_event - _t_msg)

            # DexScreener fetch — only hard requirement: price must exist
            # v14/v14e: Chain detection. Base58 → solana immediately. 0x requires
            # disambiguation across ETH/BSC/Base (same shape) — use DexScreener's
            # multi-chain tokens endpoint once per CA, cached for this session.
            from enrich import _fetch_dexscreener_by_address
            from chain_detect import detect_chain, resolve_evm_chain
            # v14e.56: hard skip on unrecognized CA shape. chain_detect.py docstring
            # explicitly warns callers to treat None as "skip this token — never
            # default to solana silently". Previous `or "solana"` fallback could
            # silently route a malformed/garbage CA (broken extraction, ticker→CA
            # hijack residue, truncated copy-paste) through a SOL Jupiter quote.
            # The downstream DexScreener call rejects such CAs anyway (returns None
            # at enrich.py:540-543), so this just surfaces the skip earlier with a
            # clearer log — zero behavioral change for valid Solana/EVM addresses.
            shape_chain = detect_chain(ca)
            if shape_chain is None:
                logger.info("RT SKIP: %s — unrecognized CA shape: %s", symbol, ca[:16])
                continue
            if shape_chain == "ethereum":  # ambiguous — could be ETH/BSC/Base
                cached = _rt_evm_chain_cache.get(ca.lower())
                if cached:
                    ca_chain = cached
                else:
                    ca_chain = await asyncio.get_event_loop().run_in_executor(
                        None, resolve_evm_chain, ca,
                    ) or "ethereum"  # default to ETH if DS lookup fails
                    _rt_evm_chain_cache[ca.lower()] = ca_chain
            else:
                ca_chain = shape_chain
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(
                None, _fetch_dexscreener_by_address, ca, ca_chain,
            )
            _t_ds_done = time.time()
            if not raw or not raw.get("price_usd"):
                logger.info("RT SKIP: %s — no price (only hard block)", ca[:8])
                continue

            try:
                price = float(raw["price_usd"])
            except (ValueError, TypeError):
                continue
            if price <= 0:
                continue

            mcap = float(raw.get("market_cap") or raw.get("mcap") or 0)  # v109: fix key
            token_info = _rt_extract_token_info(raw)

            # v109: Bonding curve tokens are tradeable on pump.fun even with liq=$0
            # in DexScreener. Use mcap as proxy — bonding curve fills at ~$60-70K mcap.
            is_bonding = token_info.get("pump_graduation_status") == "bonding"
            dex_id = token_info.get("dex_id") or ""
            is_pump_dex = "pump" in dex_id.lower()

            # v94+v109: Hard liquidity gate (RT only) — bypass for bonding curve
            min_liq = float(config.get("min_liquidity_rt", 5000))
            liq_usd = token_info.get("liquidity_usd", 0)
            if liq_usd < min_liq:
                if is_bonding or is_pump_dex:
                    # Bonding curve: DexScreener reports liq=0 but token IS tradeable.
                    # Use mcap as safety check — skip if mcap too low (< $5K).
                    min_mcap_bonding = float(config.get("min_mcap_bonding_rt", 5000))
                    if mcap < min_mcap_bonding:
                        logger.info("RT SKIP: %s — bonding mcap $%.0f < $%.0f min",
                                    symbol, mcap, min_mcap_bonding)
                        continue
                    logger.info("RT BONDING: %s — liq $%.0f but bonding curve (mcap=$%.0fK, dex=%s), allowing",
                                symbol, liq_usd, mcap / 1000, dex_id)
                else:
                    logger.info("RT SKIP: %s — liq $%.0f < $%.0f min", symbol, liq_usd, min_liq)
                    continue

            # v109: Skip old tokens — real memecoin calls are on tokens < 24h old.
            # Tokens with age > 24h are likely re-calls, recaps, or flex posts that
            # slipped past the text filter (e.g. "WHAT A REVERSE FROM 3M BOTTOM").
            # v14e.16 (Apr 24): default relaxed 12h → 72h so tokens in the 12-72h
            # band reach the strategy-level filter. Existing mains + shadows keep
            # max_age_hours=12 via STRATEGY_FILTERS default (zero-regression). New
            # AGE24_* / AGE48_* / AGE72_* shadows declare their own disjoint
            # [min_age, max_age] windows to A/B-test relaxing the gate without
            # touching baseline.
            max_age_hours = float(config.get("max_token_age_hours_rt", 72))
            token_age = token_info.get("token_age_hours", 0)
            if token_age > max_age_hours:
                logger.info("RT SKIP: %s — token too old (%.0fh > %.0fh max)",
                            symbol, token_age, max_age_hours)
                continue

            # Compute RT score (drives sizing, never blocks)
            rt_score = _rt_compute_score(username, ca, kol_info, token_info, tier, config)

            # Position sizing: confidence → dollars
            pos_size = _rt_position_size(rt_score, kol_info, token_info, tier, config)

            # v80: Multi-KOL confirmation — track count but no position boost (v110: flat sizing)
            n_confs = _rt_count_confirmations(username, ca)
            if n_confs > 0:
                token_info["_rt_n_kol_confirmations"] = n_confs

            # v79: Unified ML position multiplier (KCO primary + RT ML secondary)
            # v109: ML disabled — skip entirely. Was anti-predictive, wasted API calls.
            ml_mult = 1.0
            ml_label = ""

            logger.info(
                "RT score: %s rt_score=%.0f → pos=$%.2f | kol=%s(%.2f/%.0f%%) liq=$%.0fK bsr=%.2f%s%s",
                symbol, rt_score, pos_size, username,
                kol_info.get("score", 0), kol_info.get("win_rate", 0) * 100,
                token_info.get("liquidity_usd", 0) / 1000,
                token_info.get("buy_sell_ratio", 0),
                f" confs={n_confs}" if n_confs else "",
                f" {ml_label}" if ml_label else "",
            )

            # v14: propagate chain tag downstream so paper_trader dispatches
            # the right fee model + strategy filter chain gate. Solana is the
            # explicit default (not None) so legacy consumers don't have to
            # care about falsy checks.
            token_info["chain"] = ca_chain
            # Propagate timing for downstream live_trader log
            token_info["_rt_t_msg"] = _t_msg
            token_info["_rt_t_ds_done"] = _t_ds_done
            _t_pre_open = time.time()
            logger.info("RT timing: %s msg→detect=%.2fs detect→ds=%.2fs ds→open=%.2fs",
                        symbol, _t_event - _t_msg, _t_ds_done - _t_event, _t_pre_open - _t_ds_done)

            # Open trades across all strategies
            opened = await loop.run_in_executor(
                None, _rt_open_trades, ca, symbol, price, mcap,
                username, tier, rt_score, pos_size, kol_info, token_info, config, msg,
            )

            if opened and opened > 0:
                _rt_mark_traded(username, ca)
                # v67: Record RT event for monitoring
                if _monitoring:
                    _metrics.record_rt_event(symbol, username, ca, rt_score, pos_size, opened)
                # v109: Alert on KOL trade (for live monitoring)
                try:
                    from alerter import alert_kol_trade
                    from paper_trader import get_open_portfolio, _passes_strategy_filter
                    try:
                        from paper_trader import STRATEGIES as _STRATS
                        _br = _rt_load_bankroll()
                        bal = float(_br.get("current_balance", 0))
                        # v14e: scope per-strat positions to THIS chain only.
                        # Strats of other chains would fail _passes_strategy_filter
                        # anyway; listing them in the alert was pure noise.
                        _strat_bals = _rt_strategy_bankrolls_for_chain(_br, ca_chain)
                        _hybrid_cfg = config.get("hybrid_strategy", {})
                        strat_positions = None
                        if _hybrid_cfg.get("enabled"):
                            _allocs = _hybrid_cfg.get("allocations", {})
                            strat_positions = {}
                            total_pos = 0
                            # token_entry built above in _rt_open_trades carries
                            # chain + rt_liquidity; rebuild a minimal dict here
                            # for the filter check (we don't have the full one).
                            _tok_for_filter = {
                                "chain": ca_chain,
                                "_rt_liquidity_usd": liq_usd,
                                "_rt_score": rt_score,
                            }
                            for s, apct in _allocs.items():
                                if s not in _STRATS:
                                    continue
                                # Filter: same chain + min_liq/rt_score pass.
                                if not _passes_strategy_filter(_tok_for_filter, s):
                                    continue
                                s_pos = _rt_position_size(rt_score, kol_info, token_info, tier, config, strategy=s)
                                s_pos = round(s_pos * float(apct), 2)
                                s_bal = float((_strat_bals.get(s) or {}).get("balance", 500))
                                strat_positions[s] = {"pos": s_pos, "balance": s_bal}
                                total_pos += s_pos
                        else:
                            total_pos = pos_size
                    except Exception:
                        bal = 0
                        total_pos = pos_size
                        strat_positions = None
                    sb_alert = _get_supabase()
                    portfolio = get_open_portfolio(sb_alert) if sb_alert else {"open_count": 0, "deployed_usd": 0}
                    # v112: Only tag as bonding if actually on bonding curve (liq < min_liq)
                    # Previously is_pump_dex tagged graduated tokens as bonding too
                    _actually_bonding = (is_bonding or is_pump_dex) and liq_usd < min_liq
                    alert_kol_trade(
                        symbol, username, price, total_pos, rt_score,
                        liq_usd, is_bonding=_actually_bonding,
                        ca=ca, mcap=mcap, tier=tier, bankroll=bal,
                        deployed_usd=portfolio["deployed_usd"],
                        open_count=portfolio["open_count"],
                        strategy_positions=strat_positions,
                        chain=ca_chain,  # v14/v14e: solana | ethereum | bsc | base
                    )
                except Exception as e:
                    logger.warning("KOL trade alert failed: %s", e)
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

        # v109: ML model disabled — anti-predictive, quality gate fails, wastes API calls.
        # Skip loading entirely. ml_threshold=99 and ml_gate_mode=disabled already block usage.
        ml_status = "ML disabled (v109)"

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
    # v14e.34: every sync block in this cycle is now off-loaded via
    # asyncio.to_thread. Reason: prior to this commit, process_and_push,
    # refresh_top_tokens, and check_paper_trades all ran on the main event
    # loop, blocking it for 5–27 minutes per cycle. During that window the
    # telethon RT listener could not drain incoming Telegram updates → KOL
    # calls were buffered and processed in burst at end-of-cycle (15-min
    # detect lag observed on $ALIENPEPE et al). Threading the heavy work
    # frees the loop so the RT listener stays sub-second.
    try:
        from push_to_supabase import retry_failed_writes
        replay = await asyncio.to_thread(retry_failed_writes)
        if replay.get("replayed", 0) > 0:
            logger.info("Write buffer replay: %d/%d succeeded",
                        replay["succeeded"], replay["replayed"])
    except Exception as e:
        logger.debug("Write buffer replay skipped: %s", e)

    messages_data = await scrape_groups(client)

    total_msgs = sum(len(v) for v in messages_data.values())
    logger.info("Scraped %d messages from %d groups", total_msgs, len(messages_data))

    if total_msgs > 0:
        await asyncio.to_thread(
            process_and_push, messages_data, dump,
            _cycle_counter, once_mode,
        )

        # C1 fix: Run price refresh at end of cycle so --once mode
        # (GH Action) also gets fresh DexScreener prices before exiting.
        try:
            updated = await asyncio.to_thread(refresh_top_tokens)
            logger.info("Post-cycle price refresh: %d tokens updated", updated)
        except Exception as e:
            logger.error("Post-cycle price refresh failed: %s", e)

        # Paper trading: check open positions + log summary
        try:
            from paper_trader import (
                check_paper_trades, paper_trade_summary,
                correct_closed_prices, kol_attribution,
            )
            sb_pt = _get_supabase()
            await asyncio.to_thread(check_paper_trades, sb_pt)
            # v113: bankroll now updated per-trade inside paper_trader (before alerts)
            summary = await asyncio.to_thread(paper_trade_summary, sb_pt)
            if summary:
                logger.info("paper_trader: %s", summary)
            # v107: Fix high_price_seen for recently closed trades (pool migration bug)
            await asyncio.to_thread(correct_closed_prices, sb_pt)
            # v74: KOL attribution — aggregate paper trade PnL by KOL
            await asyncio.to_thread(kol_attribution, sb_pt, 7)
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
        _monitor_iteration = 0
        _last_avg_score = 0.0
        while True:
            await asyncio.sleep(300)  # 5 minutes
            _monitor_iteration += 1
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

                # v144.17: API error alerts disabled (user: too noisy).
                # Logs still capture error rates — re-enable here if needed.

                # Check egress
                egress = _metrics.get_egress_estimate()
                total_mb = egress.get("total_mb", 0)
                if total_mb >= 500:
                    alert_egress_warning(total_mb, egress.get("by_module", {}), 500)

                # v105: Wallet balance check (every 30min = 6 iterations)
                if _monitor_iteration % 6 == 0:
                    try:
                        from live_trader import get_wallet_balance
                        from alerter import alert_wallet_low
                        bal = await asyncio.get_event_loop().run_in_executor(
                            None, get_wallet_balance
                        )
                        if bal and bal["sol_balance"] < 0.1:
                            alert_wallet_low(bal["sol_balance"])
                    except Exception:
                        pass

                # v105→v112: KOL silence check DISABLED — too noisy, whitelist rotates often
                # Was: alert_kol_silence() every 30min for whitelisted KOLs silent >48h

                # v111: Daily summary disabled — replaced by /today bot command
                # Old: send_daily_summary() at 8h UTC + daily-summary.yml GH Action
                # Now: user triggers /today when needed, no spam
                now_utc = datetime.now(timezone.utc)
                if now_utc.hour != 8:
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
                    # v113: bankroll now updated per-trade inside paper_trader (before alerts)
                except Exception as e:
                    logger.error("Paper trading (check) failed: %s", e)
                # v72 note: live trades monitored by dedicated live_trade_monitor_loop (10s)
            except Exception as e:
                logger.error("Price refresh failed: %s", e)

    # v72→v121: Live trade monitor — aligned to 30s (same as paper_fast).
    # Strategies were optimized on 30s paper data; live must match to avoid divergence.
    LIVE_TRADE_POLL_INTERVAL = 30  # seconds (was 10s, caused paper/live strategy mismatch)

    async def live_trade_monitor_loop():
        """Monitoring loop for live trades. 30s polling (aligned with paper_fast)."""
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
                # v120: Log when live trades exist but none closed (every 6th = ~1min)
                if closed == 0 and not hasattr(live_trade_monitor_loop, '_open_log_ctr'):
                    live_trade_monitor_loop._open_log_ctr = 0
                if closed == 0:
                    live_trade_monitor_loop._open_log_ctr = getattr(live_trade_monitor_loop, '_open_log_ctr', 0) + 1
                    if live_trade_monitor_loop._open_log_ctr % 6 == 1:
                        logger.info("LIVE MONITOR: %d open trades, no exit yet", checked)
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

    # v124: Unified check loop — paper + live in same iteration, same price fetch.
    # Before v124: two separate loops (paper_fast 30s, live_monitor 30s) ran independently
    # with 0-30s clock skew → paper and live saw different prices → PnL divergence.
    UNIFIED_CHECK_INTERVAL = 30  # seconds

    async def unified_check_loop():
        """v124: Combined paper + live check loop. Same iteration = same prices."""
        _consecutive_empty_live = 0
        while True:
            await asyncio.sleep(UNIFIED_CHECK_INTERVAL)

            # --- Paper fast check ---
            try:
                from paper_trader import check_paper_trades_fast
                sb_pf = _get_supabase()
                if sb_pf:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, check_paper_trades_fast, sb_pf
                    )
                    closed = result.get("closed", 0)
                    if closed > 0:
                        logger.info("paper_fast_check: closed %d/%d recent trades",
                                    closed, result.get("checked", 0))
            except Exception as e:
                logger.error("paper_fast_check error: %s", e)

            # --- Live trade check (immediately after, same Jupiter cache) ---
            try:
                config = _rt_load_config()
                live_cfg = config.get("live_trading", {})
                if not live_cfg.get("enabled", False):
                    _consecutive_empty_live = 0
                else:
                    from live_trader import check_live_trades
                    sb_lt = _get_supabase()
                    if sb_lt:
                        live_result = await asyncio.get_event_loop().run_in_executor(
                            None, check_live_trades, sb_lt
                        )
                        checked = live_result.get("checked", 0)
                        closed = live_result.get("closed", 0)

                        if checked == 0:
                            _consecutive_empty_live += 1
                            if _consecutive_empty_live % 30 == 1:
                                logger.debug("live_trade_monitor: no open live trades")
                        else:
                            _consecutive_empty_live = 0
                            if closed == 0:
                                if not hasattr(unified_check_loop, '_open_log_ctr'):
                                    unified_check_loop._open_log_ctr = 0
                                unified_check_loop._open_log_ctr += 1
                                if unified_check_loop._open_log_ctr % 6 == 1:
                                    logger.info("LIVE MONITOR: %d open trades, no exit yet", checked)
                            if closed > 0:
                                logger.info(
                                    "LIVE MONITOR: closed %d/%d trades (TP=%d SL=%d TO=%d) pnl=$%+.2f",
                                    closed, checked,
                                    live_result.get("tp", 0), live_result.get("sl", 0),
                                    live_result.get("timeout", 0), live_result.get("pnl_usd", 0),
                                )
                                live_pnl = live_result.get("rt_pnl_usd", 0)
                                if live_pnl != 0 or closed > 0:
                                    _rt_update_bankroll(live_pnl, closed)

                        # v14e.28: ETH close-side. Runs in same iteration so ETH and
                        # SOL prices are checked at the same wall-clock — keeps cross-
                        # chain divergence measurement consistent. Gated separately so
                        # SOL live can run without loading web3/RPC if ETH disabled.
                        if live_cfg.get("eth_live_enabled", False):
                            try:
                                from live_trader_eth import check_live_trades_eth
                                eth_result = await asyncio.get_event_loop().run_in_executor(
                                    None, check_live_trades_eth, sb_lt
                                )
                                eth_closed = eth_result.get("closed", 0)
                                if eth_closed > 0:
                                    logger.info(
                                        "ETH LIVE MONITOR: closed %d/%d trades (TP=%d SL=%d TO=%d) pnl=$%+.2f",
                                        eth_closed, eth_result.get("checked", 0),
                                        eth_result.get("tp", 0), eth_result.get("sl", 0),
                                        eth_result.get("timeout", 0), eth_result.get("pnl_usd", 0),
                                    )
                                    eth_pnl = eth_result.get("rt_pnl_usd", 0)
                                    if eth_pnl != 0 or eth_closed > 0:
                                        _rt_update_bankroll(eth_pnl, eth_closed)
                            except Exception as e:
                                logger.error("ETH live close error: %s", e)
            except Exception as e:
                logger.error("live_trade_monitor error: %s", e)

    # v111: Telegram bot command listener (polls every 5s)
    BOT_CMD_INTERVAL = 5  # seconds

    async def bot_command_loop():
        """Poll Telegram for /commands and respond."""
        await asyncio.sleep(10)  # let scraper initialize first
        while True:
            try:
                from bot_commands import process_updates
                sb_cmd = _get_supabase()
                if sb_cmd:
                    await asyncio.get_event_loop().run_in_executor(
                        None, process_updates, sb_cmd
                    )
            except Exception as e:
                logger.debug("bot_command_loop error: %s", e)
            await asyncio.sleep(BOT_CMD_INTERVAL)

    # Start the price refresh loop as a background task
    refresh_task = asyncio.create_task(price_refresh_loop())
    # v67: Start monitor loop
    monitor_task = asyncio.create_task(monitor_loop()) if _monitoring else None
    # v124: Unified paper+live check loop (replaces separate live_monitor + paper_fast)
    unified_task = asyncio.create_task(unified_check_loop())
    # v111: Start bot command listener
    bot_cmd_task = asyncio.create_task(bot_command_loop())

    try:
        _recon_last_run = 0  # v105: track last reconciliation time
        while True:
            cycle_start = time.time()

            # v105: Run reconciliation every 4 hours (not just on startup)
            if time.time() - _recon_last_run > 14400:  # 4h
                try:
                    from live_trader import reconcile_positions
                    sb_recon2 = _get_supabase()
                    if sb_recon2:
                        recon2 = await asyncio.get_event_loop().run_in_executor(
                            None, reconcile_positions, sb_recon2
                        )
                        if recon2["mismatches"] > 0:
                            logger.warning("Periodic reconciliation: %d mismatches", recon2["mismatches"])
                    _recon_last_run = time.time()
                except Exception as e:
                    logger.debug("Periodic reconciliation skipped: %s", e)

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
        bot_cmd_task.cancel()
        if monitor_task:
            monitor_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
