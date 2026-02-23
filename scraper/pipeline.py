"""
Pipeline module: extract tokens, calculate sentiment, aggregate ranking.
Reusable functions called by the scraper's main loop.
"""

import re
import json
import math
import time
import logging
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from collections import Counter, defaultdict
from typing import TypedDict
from pathlib import Path

import os
import numpy as np
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from enrich import enrich_tokens
from enrich_helius import enrich_tokens_helius
from enrich_jupiter import enrich_tokens_jupiter
from enrich_bubblemaps import enrich_tokens_bubblemaps
from enrich_dexpaprika_ohlcv import enrich_tokens_ohlcv
from price_action import compute_price_action_score
from kol_scorer import get_kol_scores

logger = logging.getLogger(__name__)

# === TOKEN EXTRACTION ===

TOKEN_REGEX = re.compile(r"(?<![A-Za-z0-9_])\$([A-Za-z][A-Za-z0-9]{0,14})\b")
BARE_TOKEN_REGEX = re.compile(r"\b([A-Z][A-Z0-9]{2,14})\b")
# Solana base58 addresses (32-44 chars, no 0/O/I/l)
CA_REGEX = re.compile(r"\b([1-9A-HJ-NP-Za-km-z]{32,44})\b")
# URL-based token discovery (DexScreener links + pump.fun links)
DEXSCREENER_URL_REGEX = re.compile(r"dexscreener\.com/(\w+)/([a-zA-Z0-9]{20,70})")
PUMP_FUN_URL_REGEX = re.compile(r"pump\.fun/(?:coin/)?([a-zA-Z0-9]{20,50})")
# GMGN URLs: gmgn.ai/sol/token/papi_<CA> or gmgn.ai/sol/token/<prefix>_<CA>
GMGN_URL_REGEX = re.compile(r"gmgn\.ai/sol/token/\w+_([a-zA-Z0-9]{32,50})")
# Photon-sol URLs: photon-sol.tinyastro.io/en/lp/<pair_address>
PHOTON_URL_REGEX = re.compile(r"photon-sol\.\w+\.io/en/lp/([a-zA-Z0-9]{32,50})")

# Well-known Solana program addresses — never actual tokens
KNOWN_PROGRAM_ADDRESSES = {
    "11111111111111111111111111111111",           # System Program
    "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",  # Token Program
    "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb",  # Token-2022
    "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL",  # ATA Program
    "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",  # Raydium AMM V4
    "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc",   # Orca Whirlpool
    "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",   # Jupiter V6
    "metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s",   # Metaplex Metadata
    "So11111111111111111111111111111111111111112",      # Wrapped SOL
    "ComputeBudget111111111111111111111111111111",      # Compute Budget
    "CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK",   # Raydium CLMM
    "srmqPvymJeFKQ4zGQed1GFppgkRHL9kaELCbyksJtPX",    # Serum/OpenBook
}

EXCLUDED_TOKENS = {
    # Stablecoins (never memecoins)
    "USD", "USDT", "USDC", "BUSD", "DAI", "TUSD",
    # L1/L2 blue chips (not memecoins — these have their own markets)
    "SOL", "ETH", "BTC", "BNB", "XRP", "ADA", "DOT", "AVAX", "MATIC",
    # Crypto abbreviations that are NEVER token names
    # (kept minimal — real tokens like $DEGEN, $COPE, $PUMP are allowed)
    "CA", "LP", "MC", "ATH", "ATL", "FDV", "TVL",
    "PNL", "ROI", "APY", "APR", "DEX", "CEX",
    "ICO", "IDO", "IEO", "TGE", "KOL", "NFA", "DYOR", "DCA",
    "URL", "API", "GMT", "UTC", "EST", "PST",
    "DM", "RT", "TG", "CT",
    "GG", "GM", "GN",
    # L1/L2 tokens sometimes mentioned as context, not as calls
    "SOLANA",
    # TradFi tickers that KOLs discuss but aren't memecoins
    "SPX", "TSLA", "SNP",
    # Known paid shills / casino platforms
    "METAWIN",
}

# Common English words / crypto slang that bare ALLCAPS regex often matches.
# These are ONLY blocked when found as bare ALLCAPS (without $ prefix).
# If a KOL writes "$WAR" (with $), it's treated as a real token call.
# If a KOL writes "WAR" (bare), it needs confirmation from another $ or CA mention.
# This set is used as an EXTRA gate for bare ALLCAPS to avoid obvious noise
# even when the symbol got "confirmed" by one KOL writing e.g. "$NFA" (Not Financial Advice).
BARE_WORD_SUSPECTS = {
    # 2-3 letter English words that are almost never tokens
    "THE", "AND", "FOR", "NOT", "YOU", "ALL", "CAN", "HER", "WAS", "ONE",
    "OUR", "OUT", "ARE", "HAS", "BUT", "GET", "HIS", "HOW", "ITS", "LET",
    "MAY", "NEW", "NOW", "OLD", "SEE", "WAY", "WHO", "DID", "GOT", "HIM",
    "MAN", "USE", "SAY", "SHE", "TOO", "BIG", "END", "TRY", "ASK",
    "MEN", "RUN", "RAN", "SAT", "YES", "YET", "BAD",
    "LOT", "SET", "SIT", "CUT", "PUT",
    # 4+ letter common English words
    "JUST", "LIKE", "WITH", "THIS", "THAT", "FROM",
    "HAVE", "WILL", "BEEN", "WHAT", "WHEN", "YOUR", "THEM", "THAN",
    "EACH", "MAKE", "VERY", "SOME", "BACK", "ONLY", "COME", "MADE",
    "AFTER", "ALSO", "INTO", "OVER", "SUCH", "TAKE", "MOST", "GOOD",
    "KNOW", "TIME", "LOOK", "NEXT", "MUCH", "MORE", "LAST", "STILL",
    "GONNA", "GOING", "ABOUT", "THINK", "THESE", "RIGHT",
    "CHECK", "LOOKS", "BUYING", "TODAY", "WATCH", "WAITING", "EARLY",
    "UNTIL", "BONUS", "HIGHER", "BETTER", "NEVER", "FALLS", "FIRST",
    "STRONG", "CRAZY", "LEVEL", "COST", "HERE", "SOON", "EVERY",
    "ANYMORE", "NOBODY", "PERFECT", "BREATHE", "CAMP", "WORTH",
    "RANGE", "MOVE", "TRUST", "HOLY", "PAIN",
}

# === CRYPTO LEXICON ===

CRYPTO_LEXICON = {
    "bullish": 0.5, "moon": 0.4, "mooning": 0.45,
    "pump": 0.3, "pumping": 0.35, "gem": 0.35,
    "alpha": 0.3, "lfg": 0.4, "send it": 0.35,
    "printing": 0.4, "cook": 0.35, "cooking": 0.4,
    "listing": 0.55, "partnership": 0.4, "airdrop": 0.25,
    "conviction": 0.4, "early": 0.3, "buy": 0.2,
    "long": 0.25, "100x": 0.5, "10x": 0.4,
    "runner": 0.35, "winner": 0.35, "insane": 0.3, "massive": 0.25,
    # Bearish
    "rug": -0.8, "rugged": -0.85, "scam": -0.75, "scammer": -0.8,
    "dump": -0.6, "dumping": -0.65, "honeypot": -0.8,
    "avoid": -0.5, "warning": -0.4, "careful": -0.3,
    "sell": -0.2, "short": -0.25, "dead": -0.6,
    "rekt": -0.5, "exit": -0.4,
}

# === PER-MESSAGE CONVICTION NLP ===

# === UPDATE/BRAG MESSAGE DETECTION ===
# These patterns detect retrospective messages (KOL bragging, scorecards, updates)
# that should NOT count as fresh calls. Matches get 0.3x weight in consensus/breadth.
_UPDATE_BRAG_PATTERNS = [
    re.compile(r"(?:called\s+at|entry\s+was|we\s+gave|we\s+caught)", re.I),
    re.compile(r"(?:now\s+at|currently\s+at|sitting\s+at|still\s+holding)", re.I),
    re.compile(r"(?:took\s+profit|closed|cashed\s+out|already\s+up)", re.I),
    re.compile(r"(?:\d+x\s+done|\d+x\s+hit|hit\s+\d+x|made\s+a?\s*x?\d+)", re.I),
    re.compile(r"\d+x\s*[-–—]\s*\$", re.I),  # scorecard: "9x - $TOILET"
    # v32: "4x from the call" / "4x from my play" — bragging about past call performance
    re.compile(r"\d+x\s+from\s+(?:the\s+|my\s+)?(?:call|play|entry|pick)", re.I),
    # v32: "warned you" / "told you" / "I said" — retrospective self-reference
    re.compile(r"(?:warned\s+you|told\s+you|i\s+said|i\s+called)", re.I),
    # v32: "5M -> 20M" / "went from 5M to 20M" — market cap growth brag
    re.compile(r"\d+[km]\s*(?:->|→|to)\s*\d+[km]", re.I),
]


def _is_update_or_brag(text: str) -> bool:
    """Detect if a message is an update/brag rather than a fresh call."""
    return any(p.search(text) for p in _UPDATE_BRAG_PATTERNS)


# === TRACKING BOT DETECTION (v35b) ===
# KOLscope, SpyDefi, and similar bots that post automatic "multiplier detected" brags.
# These inflate unique_kols and trigger hype penalties — zero scoring weight.
_TRACKING_BOT_PATTERNS = [
    re.compile(r"\U0001f7ea\s*(?:MULTIPLIER\s+DETECTED|DIP\s+MODE)", re.I),  # KOLscope purple square
    re.compile(r"Achievement\s+Unlocked:\s*x?\d+", re.I),                     # SpyDefi achievement
    re.compile(r"\U0001f3c6.*?\U0001f550.*?#GROUPATH", re.I),                 # SpyDefi trophy+clock
    re.compile(r"@\w+\s+made\s+(?:a\s+)?x?\d+x?\+?\s+(?:on|call)", re.I),    # "@user made x15+ on"
    re.compile(r"(?:\u27a1\ufe0f?|\u2b95|\u279c|\u27a5)\s*\$[\d,.]+[KkMm]", re.I),  # "➡️ $381K ⮕ $5.98M"
]


def _is_tracking_bot_message(text: str) -> bool:
    """Detect tracking bot brag messages (KOLscope, SpyDefi, etc.)."""
    return any(p.search(text) for p in _TRACKING_BOT_PATTERNS)


# === ENTRY MARKET CAP EXTRACTION ===
# Captures KOL-stated entry prices like "at 500k", "aped at 1.2m", "called at 3m"
_ENTRY_MCAP_REGEX = re.compile(
    r"(?:at|@|called\s+at|aped\s+at|bought\s+at|entry)\s*\$?"
    r"(\d+(?:\.\d+)?)\s*(k|m|mil|million)",
    re.I,
)


def _extract_entry_mcap(text: str) -> float | None:
    """Extract entry market cap from message text. Returns value in USD or None."""
    match = _ENTRY_MCAP_REGEX.search(text)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "k":
        return value * 1_000
    elif unit in ("m", "mil", "million"):
        return value * 1_000_000
    return None


_CONVICTION_PATTERNS = {
    # Price targets (strong conviction)
    "price_target": re.compile(
        r"(?:\d+x|\$\d|target\s*\d|going\s*to\s*\d|reach\s*\d|to\s*\$)", re.I
    ),
    # Personal stake (strong conviction)
    "personal_stake": re.compile(
        r"(?:loaded|i\s*bought|my\s*bag|my\s*position|i\s*aped|"
        r"buying\s*more|added\s*more|my\s*top|my\s*biggest|i\'m\s*in)", re.I
    ),
    # Urgency (medium-strong conviction)
    "urgency": re.compile(
        r"(?:right\s*now|don.t\s*miss|last\s*chance|this\s*is\s*it|"
        r"floor\s*is\s*in|before\s*it)", re.I
    ),
    # Hedging (reduces conviction)
    "hedging": re.compile(
        r"(?:might|could|maybe|nfa|dyor|risky|careful|"
        r"small\s*bag|just\s*watching|be\s*cautious)", re.I
    ),
    # Strong positive language
    "strong_positive": re.compile(
        r"(?:life\s*chang|generational|100x|1000x|biggest\s*play|"
        r"printing|insane\s*entry|never\s*seen)", re.I
    ),
    # v25: Alpha framing — KOL signals this is their quality pick
    "alpha_framing": re.compile(
        r"(?:alpha\s*(?:call|play|leak|info|pick)|"
        r"high\s*conviction|my\s*(?:best|comfiest)|top\s*pick|"
        r"don.t\s*sleep\s*on|sleeper|undervalued|"
        r"slow\s*cook)", re.I
    ),
    # v25: Gamble framing — KOL admits low conviction
    "gamble_framing": re.compile(
        r"(?:gamble|degen\s*(?:play|bet|gamble)|just\s*a\s*punt|"
        r"lottery|lotto\s*ticket|fun\s*bet|risky\s*play|"
        r"throwing\s*a\s*small|micro\s*bet|for\s*fun)", re.I
    ),
    # v25: Technical analysis — KOL did chart work (medium-high conviction)
    "has_chart_analysis": re.compile(
        r"(?:chart\s*looks|clean\s*chart|reversal|"
        r"support\s*(?:level|zone|at)|resistance|"
        r"consolidat|accumulation\s*zone|"
        r"find\s*your\s*entr|good\s*entry|nice\s*entry)", re.I
    ),
}


def _compute_message_conviction(text: str) -> dict:
    """
    Analyze a single message for conviction signals via regex patterns.
    Returns dict with msg_conviction_score (0.5-2.5) and boolean flags.
    """
    score = 1.0  # baseline: neutral mention

    has_price_target = bool(_CONVICTION_PATTERNS["price_target"].search(text))
    has_personal_stake = bool(_CONVICTION_PATTERNS["personal_stake"].search(text))
    has_urgency = bool(_CONVICTION_PATTERNS["urgency"].search(text))
    has_hedging = bool(_CONVICTION_PATTERNS["hedging"].search(text))
    has_strong_positive = bool(_CONVICTION_PATTERNS["strong_positive"].search(text))
    # v25: Call framing patterns
    has_alpha_framing = bool(_CONVICTION_PATTERNS["alpha_framing"].search(text))
    has_gamble_framing = bool(_CONVICTION_PATTERNS["gamble_framing"].search(text))
    has_chart_analysis = bool(_CONVICTION_PATTERNS["has_chart_analysis"].search(text))

    if has_price_target:
        score += 0.5
    if has_personal_stake:
        score += 0.5
    if has_urgency:
        score += 0.3
    if has_strong_positive:
        score += 0.4
    if has_hedging:
        score -= 0.5
    # v25: Call framing scoring
    if has_alpha_framing:
        score += 0.6
    if has_gamble_framing:
        score -= 0.4
    if has_chart_analysis:
        score += 0.3

    score = max(0.5, min(2.5, score))

    return {
        "msg_conviction_score": round(score, 2),
        "has_price_target": has_price_target,
        "has_personal_stake": has_personal_stake,
        "has_urgency": has_urgency,
        "has_hedging": has_hedging,
        # v25: Call framing flags
        "has_alpha_framing": has_alpha_framing,
        "has_gamble_framing": has_gamble_framing,
        "has_chart_analysis": has_chart_analysis,
    }


# === CRYPTOBERT SENTIMENT (OPTIONAL) ===

_cryptobert_pipe = None
_USE_CRYPTOBERT = None


def _get_cryptobert():
    """Lazy-load CryptoBERT pipeline. Returns None if transformers not installed."""
    global _cryptobert_pipe, _USE_CRYPTOBERT
    if _USE_CRYPTOBERT is not None:
        return _cryptobert_pipe
    try:
        from transformers import pipeline as hf_pipeline
        _cryptobert_pipe = hf_pipeline(
            "text-classification",
            model="ElKulako/cryptobert",
            max_length=128,
            truncation=True,
        )
        _USE_CRYPTOBERT = True
        logger.info("CryptoBERT loaded — using transformer sentiment")
    except ImportError:
        _USE_CRYPTOBERT = False
        logger.info("transformers not installed — using VADER fallback")
    except Exception as e:
        _USE_CRYPTOBERT = False
        logger.warning("CryptoBERT failed to load: %s — using VADER fallback", e)
    return _cryptobert_pipe


def _cryptobert_sentiment(text: str) -> float | None:
    """
    Get sentiment from CryptoBERT. Returns float in [-1, 1] or None if unavailable.
    Labels: LABEL_0=Bearish, LABEL_1=Neutral, LABEL_2=Bullish
    """
    pipe = _get_cryptobert()
    if pipe is None:
        return None
    try:
        result = pipe(text[:512])[0]
        label = result["label"]
        score = result["score"]
        if label == "LABEL_0":  # Bearish
            return -score
        elif label == "LABEL_2":  # Bullish
            return score
        else:  # Neutral
            return 0.0
    except Exception:
        return None


# === TYPES ===

class Message(TypedDict):
    group: str
    conviction: int
    text: str
    date: datetime


class TokenStats(TypedDict):
    mentions: int
    sentiments: list[float]
    groups: set[str]
    convictions: list[int]
    hours_ago: list[float]


class TokenRanking(TypedDict):
    symbol: str
    score: int
    score_conviction: int
    score_momentum: int
    mentions: int
    unique_kols: int
    sentiment: float
    trend: str
    change_24h: float
    top_kols: list[str]
    # ML features (added for snapshot collection)
    avg_conviction: float
    recency_score: float
    _total_kols: int


# === VADER ANALYZER (module-level singleton) ===

_vader = SentimentIntensityAnalyzer()

# === DEXSCREENER TOKEN VERIFICATION ===

_DEXSCREENER_CACHE_FILE = Path(__file__).parent / "token_cache.json"
_DEXSCREENER_URL = "https://api.dexscreener.com/latest/dex/search"
_DEXSCREENER_TOKEN_URL = "https://api.dexscreener.com/tokens/v1/solana/{address}"

# === CA → SYMBOL CACHE ===

_CA_CACHE_FILE = Path(__file__).parent / "ca_cache.json"
_CA_CACHE_TTL = 24 * 3600  # 24h — symbols don't change


def _load_kol_win_rates() -> dict[str, dict]:
    """
    v56: Load per-KOL win rates from kol_call_outcomes (last 14 days).
    Returns {kol_group: {"wr": float, "total": int}}.
    """
    try:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            return {}
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
        resp = requests.get(
            f"{url}/rest/v1/kol_call_outcomes",
            headers={"apikey": key, "Authorization": f"Bearer {key}"},
            params={
                "select": "kol_group,did_2x",
                "entry_price": "not.is.null",
                "max_return": "not.is.null",
                "call_timestamp": f"gte.{cutoff}",
            },
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning("kol_win_rates: HTTP %d", resp.status_code)
            return {}
        rows = resp.json()
        if not rows:
            return {}
        from collections import defaultdict
        stats: dict[str, dict] = defaultdict(lambda: {"wins": 0, "total": 0})
        for r in rows:
            g = r.get("kol_group")
            if not g:
                continue
            stats[g]["total"] += 1
            if r.get("did_2x"):
                stats[g]["wins"] += 1
        result = {}
        for g, s in stats.items():
            if s["total"] > 0:
                result[g] = {"wr": round(s["wins"] / s["total"], 3), "total": s["total"]}
        logger.info("kol_win_rates: loaded %d KOLs from %d outcomes", len(result), len(rows))
        return result
    except Exception as e:
        logger.warning("kol_win_rates: failed (%s)", e)
        return {}


def _load_ca_cache() -> dict[str, dict]:
    """Load persistent CA→symbol cache from disk."""
    if _CA_CACHE_FILE.exists():
        try:
            with open(_CA_CACHE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_ca_cache(cache: dict[str, dict]) -> None:
    with open(_CA_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def _resolve_ca_to_symbol(address: str, ca_cache: dict[str, dict]) -> str | None:
    """
    Resolve a Solana contract address to its token symbol via DexScreener.
    Returns the symbol (e.g. "POPCAT") or None if unresolvable.
    """
    # Check cache first
    if address in ca_cache:
        entry = ca_cache[address]
        if time.time() - entry.get("resolved_at", 0) < _CA_CACHE_TTL:
            sym = entry.get("symbol")
            return sym if sym else None

    try:
        resp = requests.get(
            _DEXSCREENER_TOKEN_URL.format(address=address),
            timeout=10,
        )
        if resp.status_code == 200:
            pairs = resp.json() if isinstance(resp.json(), list) else resp.json().get("pairs", [])
            # The /tokens/v1/ endpoint returns a list of pairs directly
            if isinstance(pairs, list) and pairs:
                base_token = pairs[0].get("baseToken", {})
                # Pick highest-volume Solana pair
                symbol = base_token.get("symbol", "").upper().strip()
                # v21: fallback to name if symbol is non-ASCII (e.g. emoji tokens like 🌹)
                if not symbol or not symbol.isascii():
                    symbol = base_token.get("name", "").upper().strip()
                    # Remove spaces/special chars from name to make a clean ticker
                    symbol = re.sub(r"[^A-Z0-9]", "", symbol)
                if symbol and len(symbol) <= 15 and symbol.isascii():
                    ca_cache[address] = {"symbol": symbol, "resolved_at": time.time()}
                    logger.info("Resolved CA %s… → $%s", address[:8], symbol)
                    return symbol
        # Not found or no pairs
        ca_cache[address] = {"symbol": None, "resolved_at": time.time()}
        return None
    except requests.RequestException as e:
        logger.debug("CA resolution failed for %s…: %s", address[:8], e)
        return None
    finally:
        time.sleep(0.3)  # Rate limit: ~3 req/s


_DEXSCREENER_PAIRS_URL = "https://api.dexscreener.com/latest/dex/pairs/{chain}/{pair_address}"


def _resolve_pair_to_symbol(chain: str, pair_address: str, ca_cache: dict[str, dict]) -> str | None:
    """
    Resolve a DexScreener pair address to its base token symbol.
    Uses the pairs API endpoint (different from the tokens endpoint).
    """
    cache_key = f"pair:{chain}:{pair_address}"
    if cache_key in ca_cache:
        entry = ca_cache[cache_key]
        if time.time() - entry.get("resolved_at", 0) < _CA_CACHE_TTL:
            return entry.get("symbol")

    try:
        resp = requests.get(
            _DEXSCREENER_PAIRS_URL.format(chain=chain, pair_address=pair_address),
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            pairs = data.get("pairs") or data if isinstance(data, list) else data.get("pairs", [])
            if isinstance(pairs, list) and pairs:
                base_token = pairs[0].get("baseToken", {})
                symbol = base_token.get("symbol", "").upper().strip()
                # v21: fallback to name if symbol is non-ASCII (e.g. emoji tokens)
                if not symbol or not symbol.isascii():
                    symbol = base_token.get("name", "").upper().strip()
                    symbol = re.sub(r"[^A-Z0-9]", "", symbol)
                if symbol and len(symbol) <= 15 and symbol.isascii():
                    ca_cache[cache_key] = {"symbol": symbol, "resolved_at": time.time()}
                    logger.info("Resolved pair %s/%s… → $%s", chain, pair_address[:8], symbol)
                    return symbol
        ca_cache[cache_key] = {"symbol": None, "resolved_at": time.time()}
        return None
    except requests.RequestException as e:
        logger.debug("Pair resolution failed for %s/%s…: %s", chain, pair_address[:8], e)
        return None
    finally:
        time.sleep(0.3)


def _resolve_pair_to_symbol_and_ca(
    chain: str, pair_address: str, ca_cache: dict[str, dict]
) -> tuple[str | None, str | None]:
    """
    Resolve a DexScreener pair address to (symbol, token_ca).
    Returns (symbol, baseToken address) or (None, None).
    v40: needed to propagate the actual token CA from URL sources.
    """
    cache_key = f"pair:{chain}:{pair_address}"
    if cache_key in ca_cache:
        entry = ca_cache[cache_key]
        if time.time() - entry.get("resolved_at", 0) < _CA_CACHE_TTL:
            return entry.get("symbol"), entry.get("token_ca")

    try:
        resp = requests.get(
            _DEXSCREENER_PAIRS_URL.format(chain=chain, pair_address=pair_address),
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            pairs = data.get("pairs") or data if isinstance(data, list) else data.get("pairs", [])
            if isinstance(pairs, list) and pairs:
                base_token = pairs[0].get("baseToken", {})
                symbol = base_token.get("symbol", "").upper().strip()
                if not symbol or not symbol.isascii():
                    symbol = base_token.get("name", "").upper().strip()
                    symbol = re.sub(r"[^A-Z0-9]", "", symbol)
                token_ca = base_token.get("address", "").strip() or None
                if symbol and len(symbol) <= 15 and symbol.isascii():
                    ca_cache[cache_key] = {
                        "symbol": symbol, "token_ca": token_ca,
                        "resolved_at": time.time(),
                    }
                    return symbol, token_ca
        ca_cache[cache_key] = {"symbol": None, "token_ca": None, "resolved_at": time.time()}
        return None, None
    except requests.RequestException as e:
        logger.debug("Pair+CA resolution failed for %s/%s…: %s", chain, pair_address[:8], e)
        return None, None
    finally:
        time.sleep(0.3)


_CACHE_FALSE_TTL = 4 * 3600  # v28: re-check rejected tokens after 4h


def _load_token_cache() -> dict[str, dict]:
    """Load cached token verification results. v28: { "SYM": {"v": bool, "t": ts} }"""
    import json
    if _DEXSCREENER_CACHE_FILE.exists():
        try:
            with open(_DEXSCREENER_CACHE_FILE, "r") as f:
                raw = json.load(f)
            # Migrate old format (bare bool) → new format (dict with timestamp)
            now = time.time()
            cache = {}
            for k, v in raw.items():
                if isinstance(v, dict):
                    cache[k] = v
                else:
                    cache[k] = {"v": v, "t": now}
            return cache
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_token_cache(cache: dict[str, dict]) -> None:
    import json
    with open(_DEXSCREENER_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def _is_active_token(pairs: list[dict], symbol_raw: str) -> bool:
    """
    v28: Check if any Solana trading pair exists for this symbol.
    Volume/liquidity thresholds moved to soft penalties (gate_mult) so
    outcome_tracker can label these tokens and backtest can validate.
    """
    for p in pairs:
        if p.get("chainId") != "solana":
            continue
        if p.get("baseToken", {}).get("symbol", "").upper() != symbol_raw:
            continue
        return True  # Pair exists on Solana = token is real

    return False


def verify_tokens_exist(symbols: list[str]) -> set[str]:
    """
    Check which token symbols have a Solana pair on DexScreener.
    v28: Only checks pair existence (no volume threshold). Cache entries for
    rejected tokens expire after _CACHE_FALSE_TTL (4h) so early tokens
    get re-checked instead of being permanently blacklisted.
    """
    cache = _load_token_cache()
    verified: set[str] = set()
    to_check: list[str] = []
    now = time.time()

    for sym in symbols:
        raw = sym.lstrip("$")
        if raw in cache:
            entry = cache[raw]
            if entry["v"]:
                verified.add(sym)
            elif now - entry.get("t", 0) > _CACHE_FALSE_TTL:
                # v28: false entry expired — re-check
                to_check.append(sym)
            # else: still cached as false and within TTL, skip
        else:
            to_check.append(sym)

    for sym in to_check:
        raw = sym.lstrip("$")
        try:
            resp = requests.get(
                _DEXSCREENER_URL,
                params={"q": raw},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                pairs = data.get("pairs") or []
                found = _is_active_token(pairs, raw)
                cache[raw] = {"v": found, "t": now}
                if found:
                    verified.add(sym)
                    logger.info("Token %s verified (pair exists)", sym)
                else:
                    logger.info("Token %s filtered out (no Solana pair)", sym)
            elif resp.status_code >= 500:
                # Server error: not our fault, keep the token to retry next cycle
                logger.warning("DexScreener API %d for %s — keeping token (server error)", resp.status_code, sym)
                verified.add(sym)
            else:
                # 4xx = bad request / not found — token is invalid, cache with TTL
                cache[raw] = {"v": False, "t": now}
                logger.info("Token %s rejected (DexScreener %d)", sym, resp.status_code)
        except requests.RequestException as e:
            # Network error: keep the token to retry next cycle
            logger.warning("DexScreener failed for %s: %s — keeping token (network error)", sym, e)
            verified.add(sym)

        time.sleep(0.5)

    _save_token_cache(cache)
    return verified


# === ML MODEL (lazy-loaded) ===

_ML_MODEL_DIR = Path(__file__).parent
_ml_model = None          # XGBoost model (regressor or classifier)
_ml_lgb_model = None      # LightGBM model (regressor or classifier)
_ml_features = None       # Feature list
_ml_meta = None           # Full metadata dict (mode, weights, quality gate)
_ml_calibrator = None     # v54: Isotonic calibrator (optional)
_ml_loaded = False         # Prevent repeated load attempts

# Quality gate: refuse to load models below this threshold
# v22: relaxed from 0.40 — lower return thresholds (e.g. +50%) are easier to predict
_MIN_PRECISION_AT_5 = 0.30


def _load_ml_model(horizon: str = "12h"):
    """
    Lazy-load ML ensemble (XGBoost + LightGBM) if available and quality gate passes.
    Supports both regression and classification models based on metadata 'mode' field.
    v54: Also loads isotonic calibrator if available.
    Returns (xgb_model, lgb_model, features, meta, calibrator) or (None, None, None, None, None).
    """
    global _ml_model, _ml_lgb_model, _ml_features, _ml_meta, _ml_calibrator, _ml_loaded
    if _ml_loaded:
        return _ml_model, _ml_lgb_model, _ml_features, _ml_meta, _ml_calibrator

    _ml_loaded = True  # Don't retry on failure

    xgb_path = _ML_MODEL_DIR / f"model_{horizon}.json"
    lgb_path = _ML_MODEL_DIR / f"model_{horizon}_lgb.txt"
    meta_path = _ML_MODEL_DIR / f"model_{horizon}_meta.json"

    if not xgb_path.exists():
        return None, None, None, None, None

    # Load metadata first — check quality gate BEFORE loading models
    if not meta_path.exists():
        logger.warning("ML metadata not found at %s — refusing to load model without quality gate", meta_path)
        return None, None, None, None, None

    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except Exception as e:
        logger.warning("Failed to read ML metadata: %s", e)
        return None, None, None, None, None

    # Quality gate check
    p_at_5 = meta.get("metrics", {}).get("precision_at_5", 0)
    n_test = meta.get("metrics", {}).get("n_test", meta.get("n_test", 0))
    if meta.get("quality_gate") != "PASSED" or p_at_5 < _MIN_PRECISION_AT_5:
        logger.warning(
            "ML quality gate REJECTED: precision@5=%.3f (need >=%.2f), gate=%s. Using manual scores only.",
            p_at_5, _MIN_PRECISION_AT_5, meta.get("quality_gate", "UNKNOWN"),
        )
        return None, None, None, None, None

    # Refuse models trained on tiny test sets — statistically meaningless
    if n_test < 200:
        logger.warning(
            "ML DISABLED: model trained on only %d test samples (need >=200). "
            "Collect more data before trusting ML scores.",
            n_test,
        )
        return None, None, None, None, None

    mode = meta.get("mode", "classification")
    features = meta.get("features", [])
    if not features:
        logger.warning("ML metadata has empty feature list — refusing to load")
        return None, None, None, None, None

    try:
        import xgboost as xgb
        if mode == "ltr":
            # LTR uses xgb.Booster directly (not sklearn wrapper)
            _ml_model = xgb.Booster()
            _ml_model.load_model(str(xgb_path))
        elif mode == "regression":
            _ml_model = xgb.XGBRegressor()
            _ml_model.load_model(str(xgb_path))
        else:
            _ml_model = xgb.XGBClassifier()
            _ml_model.load_model(str(xgb_path))
        logger.info("XGBoost %s loaded from %s", mode, xgb_path)
    except Exception as e:
        logger.warning("Failed to load XGBoost model: %s", e)
        _ml_model = None

    # Load LightGBM if available
    if lgb_path.exists():
        try:
            import lightgbm as lgb_lib
            _ml_lgb_model = lgb_lib.Booster(model_file=str(lgb_path))
            logger.info("LightGBM %s loaded from %s", mode, lgb_path)
        except Exception as e:
            logger.warning("Failed to load LightGBM model: %s — using XGBoost only", e)
            _ml_lgb_model = None

    if _ml_model is None and _ml_lgb_model is None:
        return None, None, None, None, None

    # v54: Load isotonic calibrator if available
    cal_path = _ML_MODEL_DIR / f"model_{horizon}_calibrator.joblib"
    if cal_path.exists():
        try:
            from joblib import load as joblib_load
            _ml_calibrator = joblib_load(cal_path)
            logger.info("Isotonic calibrator loaded from %s", cal_path)
        except Exception as e:
            logger.warning("Failed to load calibrator: %s — proceeding without", e)
            _ml_calibrator = None

    _ml_features = features
    _ml_meta = meta
    logger.info(
        "ML ensemble loaded: mode=%s, %d features, precision@5=%.3f, ensemble_weights=%s",
        mode, len(features), p_at_5, meta.get("ensemble_weights", {}),
    )
    return _ml_model, _ml_lgb_model, _ml_features, _ml_meta, _ml_calibrator


def _build_feature_row(token: dict, features: list[str]) -> dict:
    """Build a single feature row for ML prediction from a token dict."""
    # Features that need log-scaling
    log_map = {
        "volume_24h_log": "volume_24h",
        "volume_6h_log": "volume_6h",
        "volume_1h_log": "volume_1h",
        "volume_5m_log": "volume_5m",
        "liquidity_usd_log": "liquidity_usd",
        "market_cap_log": "market_cap",
        "holder_count_log": "holder_count",
        "v_buy_24h_usd_log": "v_buy_24h_usd",
        "v_sell_24h_usd_log": "v_sell_24h_usd",
        "helius_holder_count_log": "helius_holder_count",
    }
    # v44: Component values stored with underscore prefix in token dict
    # but ML features use unprefixed names. Map ML name → token dict key.
    component_val_map = {
        "consensus_val": "_consensus_val",
        "conviction_val": "_conviction_val",
        "breadth_val": "_breadth_val",
        "price_action_val": "_price_action_val",
        "sentiment_val": "_sentiment_val",
    }

    row = {}
    for feat in features:
        if feat in log_map:
            raw = token.get(log_map[feat])
            row[feat] = np.log1p(float(raw)) if raw is not None and float(raw) > 0 else np.nan
        elif feat == "breadth":
            uk = token.get("unique_kols", 0)
            tk = token.get("_total_kols", 50)
            row[feat] = uk / max(1, tk)
        elif feat == "pump_graduated":
            status = token.get("pump_graduation_status")
            row[feat] = 1.0 if status == "graduated" else 0.0
        elif feat == "birdeye_buy_sell_ratio":
            b = token.get("buy_24h")
            s = token.get("sell_24h")
            if b is not None and s is not None:
                row[feat] = float(b) / max(1, float(b) + float(s))
            else:
                row[feat] = np.nan
        elif feat in ("day_of_week", "hour_paris", "is_weekend", "is_prime_time"):
            # ML v3.1: Calendar/temporal features — compute from current time
            # At inference time, use NOW (Europe/Paris) since we're scoring live tokens
            from datetime import datetime, timezone, timedelta
            try:
                from zoneinfo import ZoneInfo
                now_paris = datetime.now(ZoneInfo("Europe/Paris"))
            except Exception:
                now_paris = datetime.now(timezone(timedelta(hours=1)))
            if feat == "day_of_week":
                row[feat] = float(now_paris.weekday())       # 0=Mon..6=Sun
            elif feat == "hour_paris":
                row[feat] = float(now_paris.hour)
            elif feat == "is_weekend":
                row[feat] = 1.0 if now_paris.weekday() >= 5 else 0.0
            elif feat == "is_prime_time":
                row[feat] = 1.0 if now_paris.hour >= 19 or now_paris.hour < 5 else 0.0
        elif feat in component_val_map:
            # v44: Map ML feature name to underscore-prefixed token dict key
            # At inference: pipeline dict uses "_consensus_val" (underscore prefix)
            # At training: DB row uses "consensus_val" (no prefix)
            val = token.get(component_val_map[feat])
            if val is None:
                val = token.get(feat)  # fallback to unprefixed (DB row)
            row[feat] = float(val) if val is not None else np.nan
        else:
            val = token.get(feat)
            row[feat] = float(val) if val is not None else np.nan
    return row


def _apply_ml_scores(ranking: list[dict]) -> None:
    """
    ML as MULTIPLIER: predictions scale the manual score within [0.5, 1.5].
    This preserves manual penalties (safety, death, crash) while letting ML boost/dampen.

    Supports 3 modes:
    - regression: raw prediction → percentile rank → scale to [0.5, 1.5]
    - classification: predict_proba → scale to [0.5, 1.5]
    - ltr (Learning-to-Rank): relevance score → percentile rank → [0.5, 1.5]
    """
    # v22: Read ML horizon from scoring_config (dynamic)
    ml_horizon = SCORING_PARAMS.get("ml_horizon", "12h")
    xgb_model, lgb_model, features, meta, calibrator = _load_ml_model(horizon=ml_horizon)
    if (xgb_model is None and lgb_model is None) or not features or not meta:
        return

    mode = meta.get("mode", "classification")
    weights = meta.get("ensemble_weights", {"xgboost": 0.5, "lightgbm": 0.5})
    xgb_w = weights.get("xgboost", 0.5)
    lgb_w = weights.get("lightgbm", 0.5)

    # Build feature matrix
    rows = [_build_feature_row(token, features) for token in ranking]

    try:
        import pandas as pd
        X = pd.DataFrame(rows, columns=features)

        # Get raw predictions from each model
        preds = None

        if mode in ("regression", "ltr"):
            # Both regression and LTR output raw scores → percentile-based scaling
            # LTR: XGBoost rank:ndcg outputs relevance scores (higher = better ranking)
            if mode == "ltr" and xgb_model is not None:
                # LTR uses Booster.predict(DMatrix), not sklearn .predict(DataFrame)
                import xgboost as _xgb
                dmat = _xgb.DMatrix(X)
                xgb_preds = xgb_model.predict(dmat)
            else:
                xgb_preds = xgb_model.predict(X) if xgb_model is not None else None
            lgb_preds = lgb_model.predict(X.values) if lgb_model is not None else None

            if xgb_preds is not None and lgb_preds is not None:
                preds = xgb_w * xgb_preds + lgb_w * lgb_preds
            elif xgb_preds is not None:
                preds = xgb_preds
            else:
                preds = lgb_preds

            # Convert to percentile ranks (0-1), then scale to [floor, cap]
            # v44: dynamic bounds from scoring_config (widened from [0.5, 1.5])
            ml_floor = SCORING_PARAMS["ml_mult_floor"]  # default 0.3
            ml_cap = SCORING_PARAMS["ml_mult_cap"]      # default 2.0
            from scipy.stats import rankdata
            ranks = rankdata(preds) / len(preds)  # 0..1 percentile
            ml_multipliers = ml_floor + ranks * (ml_cap - ml_floor)

        else:
            # Classification: predict_proba → [0.5, 1.5]
            xgb_proba = xgb_model.predict_proba(X)[:, 1] if xgb_model is not None else None
            lgb_proba = None
            if lgb_model is not None:
                # LightGBM Booster.predict returns raw probabilities directly
                lgb_proba = lgb_model.predict(X.values)

            if xgb_proba is not None and lgb_proba is not None:
                proba = xgb_w * xgb_proba + lgb_w * lgb_proba
            elif xgb_proba is not None:
                proba = xgb_proba
            else:
                proba = lgb_proba

            # Scale probability to multiplier: p=0 → floor, p=1 → cap
            # v44: dynamic bounds from scoring_config
            ml_floor = SCORING_PARAMS["ml_mult_floor"]  # default 0.3
            ml_cap = SCORING_PARAMS["ml_mult_cap"]      # default 2.0
            ml_multipliers = ml_floor + np.clip(proba, 0, 1) * (ml_cap - ml_floor)

        # v54: Apply calibrated win probabilities if calibrator available
        if calibrator is not None and preds is not None:
            try:
                calibrated = calibrator.predict(preds)
                for token, cal_prob in zip(ranking, calibrated):
                    token["ml_win_probability"] = round(float(cal_prob), 4)
                logger.info("Calibrated win probabilities applied [%.3f - %.3f]",
                            float(calibrated.min()), float(calibrated.max()))
            except Exception as cal_err:
                logger.warning("Calibration failed at inference: %s", cal_err)

        # Apply multiplier to all score variants
        # v44: dynamic bounds from scoring_config
        ml_floor = SCORING_PARAMS["ml_mult_floor"]
        ml_cap = SCORING_PARAMS["ml_mult_cap"]
        for token, ml_mult in zip(ranking, ml_multipliers):
            ml_mult = float(np.clip(ml_mult, ml_floor, ml_cap))
            token["ml_multiplier"] = round(ml_mult, 3)
            token["score"] = int(token["score"] * ml_mult)
            token["score_conviction"] = int(token["score_conviction"] * ml_mult)
            token["score_momentum"] = int(token["score_momentum"] * ml_mult)

        logger.info(
            "ML multiplier applied (%s mode) for %d tokens [%.2f - %.2f]",
            mode, len(ranking), float(ml_multipliers.min()), float(ml_multipliers.max()),
        )
    except Exception as e:
        logger.warning("ML scoring failed: %s — keeping manual scores", e)


# === PUBLIC API ===

def extract_tokens(
    text: str,
    ca_cache: dict[str, dict] | None = None,
    confirmed_symbols: set[str] | None = None,
) -> list[tuple[str, str, str | None]]:
    """
    Extract token symbols from text via three methods:
    1) $TOKEN prefix — high confidence, always extracted
    2) Bare ALL-CAPS words (3+ chars) — only if symbol is confirmed by $ or CA elsewhere
    3) Solana contract addresses → resolved to symbols via DexScreener

    Returns list of (symbol, source, ca_or_none) tuples.
    v40: 3rd element is the token contract address when available (from CA/URL sources).
    This prevents symbol collisions where multiple CAs share the same symbol.

    Parameters
    ----------
    confirmed_symbols : set of uppercase symbol names (without $) that have been
        confirmed via $-prefix or CA resolution elsewhere in this scrape cycle.
        When provided, bare ALLCAPS words are only extracted if they appear in this set
        AND are not in BARE_WORD_SUSPECTS. When None (backward compat), bare ALLCAPS
        extraction is unrestricted (legacy behavior).
    """
    tokens: list[tuple[str, str, str | None]] = []
    seen: set[str] = set()

    # 1) $-prefixed tokens: always extract (KOL intentionally naming a token)
    for match in TOKEN_REGEX.findall(text):
        upper = match.upper()
        if not upper.isascii():
            continue
        symbol = f"${upper}"
        if upper not in EXCLUDED_TOKENS and symbol not in seen:
            tokens.append((symbol, "ticker", None))
            seen.add(symbol)

    # v15: Bare ALLCAPS detection REMOVED.
    # When a KOL writes $DOG, "DOG" becomes confirmed, then every bare "DOG"
    # in other chats counts as a mention → massive false positives for common
    # English words (DOG, CAT, MOON, ROCK, SHOT, TOKEN, COW, JUICE...).
    # Real KOL calls always use $ prefix or post CA/URLs. Bare words = noise.

    # 3) Solana contract addresses → resolve to symbol (definitive proof)
    #    Skip addresses that are part of DexScreener/pump.fun/GMGN/Photon URLs (handled in step 4)
    url_addresses = set()
    for _, pair_addr in DEXSCREENER_URL_REGEX.findall(text):
        url_addresses.add(pair_addr)
    for pump_addr in PUMP_FUN_URL_REGEX.findall(text):
        url_addresses.add(pump_addr)
    for gmgn_addr in GMGN_URL_REGEX.findall(text):
        url_addresses.add(gmgn_addr)
    for photon_addr in PHOTON_URL_REGEX.findall(text):
        url_addresses.add(photon_addr)

    if ca_cache is not None:
        for match in CA_REGEX.findall(text):
            if match in KNOWN_PROGRAM_ADDRESSES:
                continue
            if match in url_addresses:
                continue  # Will be handled by URL extraction below
            resolved = _resolve_ca_to_symbol(match, ca_cache)
            if resolved:
                symbol = f"${resolved}"
                if resolved not in EXCLUDED_TOKENS and symbol not in seen:
                    tokens.append((symbol, "ca", match))  # v40: CA is the match itself
                    seen.add(symbol)
                elif symbol in seen:
                    # v56: Symbol already extracted (e.g. via $ticker) — backfill CA
                    # so ca_by_symbol gets populated and prevents cross-contamination
                    for i, (s, src, ca) in enumerate(tokens):
                        if s == symbol and ca is None:
                            tokens[i] = (s, src, match)
                            break

    # 4) DexScreener/pump.fun URLs → resolve pair/token address to symbol
    # v56: Helper to backfill CA on already-seen ticker-only symbols
    def _backfill_ca(sym: str, ca_val: str | None) -> None:
        if sym in seen and ca_val:
            for i, (s, src, ca) in enumerate(tokens):
                if s == sym and ca is None:
                    tokens[i] = (s, src, ca_val)
                    break

    if ca_cache is not None:
        # DexScreener: pair address → resolve via pairs API (v40: also get token CA)
        for chain, pair_addr in DEXSCREENER_URL_REGEX.findall(text):
            resolved, token_ca = _resolve_pair_to_symbol_and_ca(chain, pair_addr, ca_cache)
            if resolved:
                symbol = f"${resolved}"
                if resolved not in EXCLUDED_TOKENS and symbol not in seen:
                    tokens.append((symbol, "url", token_ca))
                    seen.add(symbol)
                else:
                    _backfill_ca(symbol, token_ca)

        # pump.fun: token CA directly → resolve via tokens API
        for pump_addr in PUMP_FUN_URL_REGEX.findall(text):
            resolved = _resolve_ca_to_symbol(pump_addr, ca_cache)
            if resolved:
                symbol = f"${resolved}"
                if resolved not in EXCLUDED_TOKENS and symbol not in seen:
                    tokens.append((symbol, "url", pump_addr))  # v40: URL contains the CA
                    seen.add(symbol)
                else:
                    _backfill_ca(symbol, pump_addr)

        # GMGN: token CA directly (same as pump.fun — CA in URL)
        for gmgn_addr in GMGN_URL_REGEX.findall(text):
            resolved = _resolve_ca_to_symbol(gmgn_addr, ca_cache)
            if resolved:
                symbol = f"${resolved}"
                if resolved not in EXCLUDED_TOKENS and symbol not in seen:
                    tokens.append((symbol, "url", gmgn_addr))  # v40: URL contains the CA
                    seen.add(symbol)
                else:
                    _backfill_ca(symbol, gmgn_addr)

        # Photon-sol: LP pair address → resolve via pairs API (v40: also get token CA)
        for photon_addr in PHOTON_URL_REGEX.findall(text):
            resolved, token_ca = _resolve_pair_to_symbol_and_ca("solana", photon_addr, ca_cache)
            if resolved:
                symbol = f"${resolved}"
                if resolved not in EXCLUDED_TOKENS and symbol not in seen:
                    tokens.append((symbol, "url", token_ca))
                    seen.add(symbol)
                else:
                    _backfill_ca(symbol, token_ca)

    return tokens


def calculate_sentiment(text: str) -> float:
    """
    Sentiment blend, clamped to [-1, 1].
    With CryptoBERT: 0.6 * CryptoBERT + 0.2 * VADER + 0.2 * lexicon
    Without:         0.7 * VADER + 0.3 * lexicon
    """
    vader_score = _vader.polarity_scores(text)["compound"]

    text_lower = text.lower()
    lexicon_boost = 0.0
    matches = 0
    for word, weight in CRYPTO_LEXICON.items():
        if word in text_lower:
            lexicon_boost += weight
            matches += 1

    if matches > 0:
        lexicon_boost = max(-1, min(1, lexicon_boost / max(1, matches * 0.5)))

    bert_score = _cryptobert_sentiment(text)
    if bert_score is not None:
        final = 0.6 * bert_score + 0.2 * vader_score + 0.2 * lexicon_boost
    else:
        final = 0.7 * vader_score + 0.3 * lexicon_boost

    return max(-1.0, min(1.0, final))


def _two_phase_decay(hours_ago: float) -> float:
    """
    Single-phase exponential decay for recency scoring.
    Default lambda=0.12, half-life ~5.8h.
    12h-old mention retains ~24% weight (was 2% with old two-phase).
    v20: lambda read from SCORING_PARAMS["decay_lambda"] (dynamic).
    """
    return math.exp(-SCORING_PARAMS["decay_lambda"] * hours_ago)


def _compute_wash_trading_score(token: dict) -> float:
    """
    Estimate probability of wash trading [0.0=clean, 1.0=fully washed].
    Uses volume/liquidity mismatch and volume/mcap extremes from DexScreener data.
    """
    scores = []

    # 1. Volume/Liquidity mismatch (>50x = almost certainly washed)
    vol = token.get("volume_24h")
    liq = token.get("liquidity_usd")
    if vol is not None and liq is not None and liq > 0:
        ratio = vol / liq
        if ratio > 100:
            scores.append(1.0)
        elif ratio > 50:
            scores.append(0.8)
        elif ratio > 20:
            scores.append(0.5)
        else:
            scores.append(0.0)

    # 2. Volume/MCap extreme (>5x on any token = artificial)
    vmcr = token.get("volume_mcap_ratio")
    if vmcr is not None:
        if vmcr > 5.0:
            scores.append(1.0)
        elif vmcr > 2.0:
            scores.append(0.6)
        else:
            scores.append(0.0)

    # 3. Algorithm v3 A6: Volume spike with flat price = wash trading
    pc24 = token.get("price_change_24h")
    va = token.get("volume_acceleration")
    if pc24 is not None and va is not None:
        if va > 3.0 and abs(pc24) < 5.0:
            scores.append(0.7)
        elif va > 2.0 and abs(pc24) < 3.0:
            scores.append(0.5)

    if not scores:
        return 0.0
    # v27: Use max() — one severe wash signal is enough to flag the token.
    # mean() was diluting obvious wash trading when other signals were clean.
    return max(0.0, min(1.0, max(scores)))


def _get_price_from_candles(token: dict, hours: float) -> float | None:
    """
    Get the REAL historical price from OHLCV candle data.
    Uses only actual chart data — no interpolation, no estimation.
    Returns the close price of the nearest candle, or None if no data.

    Candles are 15-min intervals from DexPaprika (~24h of history).
    Only matches if a candle exists within 15 min of the target time.
    """
    candles = token.get("candle_data")
    if not candles or len(candles) < 2:
        return None

    target_ts = time.time() - hours * 3600

    # Find closest candle by timestamp
    best_candle = None
    best_diff = float("inf")
    for c in candles:
        diff = abs(c["timestamp"] - target_ts)
        if diff < best_diff:
            best_diff = diff
            best_candle = c

    # Only accept if within 15 min (one candle interval) — no guessing
    if best_candle and best_diff <= 900:
        close = best_candle.get("close", 0)
        if close > 0:
            return close

    return None


def _compute_manipulation_penalty(token: dict) -> float:
    """
    Manipulation detection: artificial pump only.
    Returns penalty multiplier in [0.3, 1.0].

    v34: REMOVED wash_pen from this function. Wash trading data (38.1% hit rate)
    shows it's anti-predictive as a penalty. Kept as ML feature only.
    """
    # Artificial pump signal (from _detect_artificial_pump)
    pump_pen = 1.0
    if token.get("is_artificial_pump"):
        pc24 = token.get("price_change_24h") or 0
        if pc24 > 400:
            pump_pen = 0.3
        elif pc24 > 200:
            pump_pen = 0.5
        else:
            pump_pen = 0.7

    return pump_pen


def _compute_entry_timing_quality(token: dict) -> float:
    """
    ML v2 Phase C: Compute entry timing quality (0-1).

    Measures how close the token is to an optimal buy window by combining:
    1. Freshness sweet spot (winners peak 1-8h after fresh KOL calls)
    2. Price position (not too pumped, not crashing)
    3. Social momentum (building > declining)
    4. Score velocity (rising score = good timing)

    Returns float 0-1 (higher = better entry timing).
    ML-only feature initially. May become soft multiplier if correlation > 0.05.
    """
    signals = []

    # 1. Freshness sweet spot (winners peak at 1-8h after fresh calls)
    freshest = token.get("freshest_mention_hours", 999)
    if freshest < 1:
        signals.append(0.7)       # very fresh, unproven
    elif freshest < 4:
        signals.append(1.0)       # sweet spot
    elif freshest < 8:
        signals.append(0.8)
    elif freshest < 16:
        signals.append(0.5)
    else:
        signals.append(0.2)

    # 2. Price momentum (v32: inverted from original — momentum IS the signal)
    # Data shows winners are already pumping +100-300% at snapshot time
    # and continue to 2x. "Too late" is a myth in memecoins.
    pc24 = token.get("price_change_24h")
    if pc24 is not None:
        if pc24 < -30:
            signals.append(0.2)       # crashing hard
        elif pc24 < 0:
            signals.append(0.4)       # declining
        elif pc24 < 50:
            signals.append(0.5)       # flat/slow — unproven
        elif 50 <= pc24 < 200:
            signals.append(0.9)       # momentum zone (winners live here)
        elif 200 <= pc24 < 500:
            signals.append(0.7)       # strong pump, might continue
        else:
            signals.append(0.4)       # extreme overextension (>500%)
    else:
        signals.append(0.5)           # neutral when no data

    # 3. Social momentum (building > declining)
    activity = token.get("activity_mult", 1.0)
    if activity > 1.15:
        signals.append(0.9)
    elif activity > 1.0:
        signals.append(0.7)
    else:
        signals.append(0.3)

    # 4. Score velocity (Phase B) — rising score = good timing
    vel = token.get("score_velocity")
    if vel is not None:
        if vel > 5:
            signals.append(0.9)
        elif vel > 0:
            signals.append(0.7)
        elif vel < -5:
            signals.append(0.2)
        else:
            signals.append(0.5)

    return round(sum(signals) / len(signals), 3) if signals else 0.5


def _entry_premium_tier_mult(premium: float, cfg: dict) -> float:
    """v45: Compute entry premium multiplier from config tier breakpoints."""
    bps = cfg["tier_breakpoints"]
    mults = cfg["tier_multipliers"]
    slopes = cfg["tier_slopes"]
    # Walk breakpoints: return flat multiplier if below breakpoint,
    # or interpolate with slope between breakpoints.
    for i in range(len(bps)):
        if premium <= bps[i]:
            return mults[i]
        if i < len(bps) - 1 and premium <= bps[i + 1]:
            base_mult = mults[i + 1]
            slope = slopes[i + 1] if i + 1 < len(slopes) else 0
            return base_mult + (bps[i + 1] - premium) * slope
    return mults[-1]


def _compute_kol_entry_premium(token: dict) -> tuple[float, float]:
    """
    Compute how much the price has moved since KOLs called the token.
    v45: Tier breakpoints + duration + mcap fallback read from SCORING_PARAMS["entry_premium_config"].
    """
    cfg = SCORING_PARAMS["entry_premium_config"]
    current_price = token.get("price_usd")
    current_mcap = token.get("market_cap") or token.get("fdv")
    hours_ago_list = token.get("_hours_ago", [])

    # Primary source — KOL-stated entry mcap
    stated_mcaps = token.get("kol_stated_entry_mcaps", [])
    if stated_mcaps and current_mcap and current_mcap > 0:
        avg_stated = sum(stated_mcaps) / len(stated_mcaps)
        if avg_stated > 0:
            entry_premium = current_mcap / avg_stated
            mult = _entry_premium_tier_mult(entry_premium, cfg)
            return round(entry_premium, 3), round(mult, 3)

    # Fallback 2: OHLCV candle data
    candle_result = None
    if current_price and current_price > 0 and hours_ago_list and token.get("candle_data"):
        valid_hours = [h for h in hours_ago_list if h > 5 / 60]
        if valid_hours:
            weighted_sum = 0.0
            weight_sum = 0.0
            matched_count = 0
            for h in valid_hours:
                real_price = _get_price_from_candles(token, h)
                if real_price is not None and real_price > 0:
                    decay = _two_phase_decay(h)
                    weighted_sum += real_price * decay
                    weight_sum += decay
                    matched_count += 1

            if matched_count > 0 and weight_sum > 0:
                avg_entry = weighted_sum / weight_sum
                entry_premium = current_price / avg_entry

                # Duration scaling from config
                dur_t = cfg["duration_thresholds"]
                dur_f = cfg["duration_factors"]
                oldest_mention = max(valid_hours)
                duration_factor = dur_f[0]  # default for shortest
                for i in range(len(dur_t) - 1, -1, -1):
                    if oldest_mention > dur_t[i]:
                        duration_factor = dur_f[i + 1]
                        break

                effective_premium = entry_premium ** duration_factor
                mult = _entry_premium_tier_mult(effective_premium, cfg)
                candle_result = (round(entry_premium, 3), round(mult, 3))

    if candle_result is not None:
        return candle_result

    # Fallback 3: Market cap magnitude
    mcap_threshold = cfg["mcap_fallback_threshold"]
    mcap_launch = cfg["mcap_launch_assumed"]
    if current_mcap and current_mcap > mcap_threshold:
        implied_premium = current_mcap / mcap_launch
        mcap_tiers = cfg["mcap_tiers"]
        mcap_mults = cfg["mcap_multipliers"]
        mult = mcap_mults[-1]
        for i, tier in enumerate(mcap_tiers):
            if implied_premium <= tier:
                mult = mcap_mults[i]
                break
        return round(implied_premium, 3), round(mult, 3)

    return 1.0, 1.0


def _compute_entry_drift_mult(token: dict) -> float:
    """
    Penalizes when price outpaced social growth since first KOL calls.
    v45: All weights/thresholds read from SCORING_PARAMS["entry_drift_config"].
    """
    cfg = SCORING_PARAMS["entry_drift_config"]
    entry_premium = token.get("entry_premium") or 1.0
    if entry_premium <= cfg["premium_gate"]:
        return 1.0

    # Social strength [0, 1]
    unique_kols = token.get("unique_kols") or 1
    kol_score = min(1.0, (unique_kols - 1) / cfg["kol_divisor"])

    activity_mult = token.get("activity_mult") or 1.0
    activity_score = max(0, min(1.0, (activity_mult - cfg["activity_base"]) / cfg["activity_range"]))

    freshest_h = token.get("freshest_mention_hours") or 999
    fresh_tiers = cfg["fresh_tiers"]
    fresh_scores = cfg["fresh_scores"]
    fresh_score = fresh_scores[-1]
    for i, tier in enumerate(fresh_tiers):
        if freshest_h <= tier:
            fresh_score = fresh_scores[i]
            break

    s_count = sum(1 for t in token.get("kol_tiers", {}).values() if t == "S")
    s_tier_score = min(1.0, s_count / cfg["stier_divisor"])

    sw = cfg["social_weights"]
    social_strength = (kol_score * sw[0] + activity_score * sw[1]
                       + fresh_score * sw[2] + s_tier_score * sw[3])

    drift = entry_premium - 1.0
    net_drift = max(0, drift - social_strength * 1.0)
    mult = max(cfg["drift_floor"], 1.0 - net_drift * cfg["drift_factor"])
    return round(mult, 3)


def _detect_artificial_pump(token: dict) -> bool:
    """
    Algorithm v3 A5: Flag tokens with price pump but no organic growth behind it.
    Catches LPI (Liquidity Pool-based Price Inflation) manipulation.

    v3.1 fix: Whale accumulation during pump = supply control (bullish),
    not manipulation. Only flag if whales are NOT accumulating.
    """
    pc24 = token.get("price_change_24h")
    bsr = token.get("buy_sell_ratio_24h")
    holders = token.get("helius_holder_count") or token.get("holder_count")
    vol = token.get("volume_24h") or 0
    liq = token.get("liquidity_usd") or 1
    whale_change = token.get("whale_change")

    # v3.1: If whales are accumulating during the pump, it's supply control — not a rug
    if whale_change is not None and whale_change > 0:
        return False

    if pc24 is not None and pc24 > 100:  # >100% pump in 24h
        if bsr is not None and bsr > 0.85:  # almost all buys, no real selling
            if holders is not None and holders < 100:  # very few holders
                return True
        if vol / max(1, liq) > 50:  # volume dwarfs liquidity (wash trading pump)
            return True
    return False


def _detect_death_penalty(token: dict, freshest_mention_hours: float) -> float:
    """
    Detect dead/rugged tokens. Returns penalty multiplier [0.1, 1.0].
    v45: All thresholds read from SCORING_PARAMS["death_config"].
    """
    cfg = SCORING_PARAMS["death_config"]
    penalties = []

    # --- Signal 1: Price collapse ---
    death_severe = SCORING_PARAMS["death_pc24_severe"]
    death_moderate = SCORING_PARAMS["death_pc24_moderate"]
    pc24 = token.get("price_change_24h")
    if pc24 is not None:
        pc24 = float(pc24)

        if pc24 < death_severe:
            penalties.append(0.1)
        elif pc24 < (death_severe + 10):
            penalties.append(0.15 if freshest_mention_hours > 3 else 0.3)
        elif pc24 < death_moderate:
            va = token.get("volume_acceleration")
            volume_alive = va is not None and float(va) > cfg["price_moderate_vol_alive"]
            social_alive = freshest_mention_hours < cfg["price_moderate_social_alive_h"]
            if volume_alive and social_alive:
                penalties.append(0.6)
            elif volume_alive or social_alive:
                penalties.append(0.4)
            else:
                penalties.append(0.2)
        elif pc24 < cfg["price_mild_threshold"]:
            if freshest_mention_hours > cfg["price_mild_stale_h"]:
                penalties.append(0.4)
            else:
                penalties.append(0.8)

    # --- Signal 2: Volume death ---
    vol_24h = token.get("volume_24h")
    vol_1h = token.get("volume_1h")
    if vol_24h is not None and vol_1h is not None:
        if vol_24h < cfg["vol_death_24h"] and vol_1h < cfg["vol_death_1h"]:
            penalties.append(cfg["vol_death_penalty"])
        elif vol_24h < cfg["vol_floor_24h"]:
            penalties.append(cfg["vol_floor_penalty"])

    # --- Signal 3: Social staleness — volume-modulated ---
    if freshest_mention_hours > cfg["stale_start_hours"]:
        stale_tiers = cfg["stale_tiers"]
        stale_bases = cfg["stale_bases"]
        # Walk tiers in reverse (highest first) to find the matching base
        stale_base = stale_bases[0]  # default: stale_start_hours to first tier
        for i in range(len(stale_tiers) - 1, -1, -1):
            if freshest_mention_hours > stale_tiers[i]:
                stale_base = stale_bases[i + 1]
                break

        # Volume modulator
        vol_24h = token.get("volume_24h")
        if vol_24h is None:
            stale_pen = stale_base
        else:
            mod_tiers = cfg["vol_modulation_tiers"]
            mod_bonuses = cfg["vol_modulation_bonuses"]
            mod_caps = cfg["vol_modulation_caps"]
            stale_pen = stale_base
            for i in range(len(mod_tiers) - 1, -1, -1):
                if vol_24h > mod_tiers[i]:
                    stale_pen = min(mod_caps[i], stale_base + mod_bonuses[i])
                    break

        penalties.append(stale_pen)

    if not penalties:
        return 1.0
    return min(penalties)


def _apply_hard_gates(ranking: list[dict]) -> list[dict]:
    """
    v33: ALL gates are soft penalties now — no more hard removal.
    Every token stays in the pipeline, gets scored, enriched, and labeled
    by outcome_tracker so we can empirically validate all gates.

    v34: REMOVED anti-predictive gates based on N=1630 audit data:
      - high_risk_score (46.7% hit rate vs 9.9% scored) — REMOVED
      - low_liquidity (17.4% hit rate vs 9.9% scored) — REMOVED
      - wash_trading (38.1% hit rate vs 9.9% scored) — REMOVED (separate function)

    v58 AUDIT: ALL gate penalties DISABLED. Data at N=29,345 shows gates are
    anti-predictive:
      - Killed tokens (gate=0): 18.4% hit rate (BEST)
      - No penalty: 15.0%
      - Penalized: 13.5% (WORST)
    Gate reasons are still RECORDED for ML features, but gate_mult is always 1.0.
    """
    gate_top10 = SCORING_PARAMS["gate_top10_pct"]
    gate_holders = int(SCORING_PARAMS["gate_min_holders"])

    tagged = 0
    for token in ranking:
        # v58: gate_mult ALWAYS 1.0 — gates are anti-predictive, destroy alpha.
        # We still tag gate_reason so ML can use it as a feature.
        gate_reasons = []

        if token.get("has_mint_authority"):
            gate_reasons.append("mint_authority")
        if token.get("has_freeze_authority"):
            gate_reasons.append("freeze_authority")

        top10 = token.get("top10_holder_pct")
        if top10 is not None and top10 > gate_top10:
            gate_reasons.append("top10_concentration")

        hcount = token.get("helius_holder_count") or token.get("holder_count")
        if hcount is not None and hcount < gate_holders:
            gate_reasons.append("low_holders")

        token["gate_mult"] = 1.0  # v58: no penalty, data proves gates anti-predictive
        if gate_reasons:
            token["gate_reason"] = gate_reasons[0]
            tagged += 1

    if tagged:
        logger.info("Gate reasons tagged on %d tokens (NO penalty — v58 audit: gates anti-predictive)", tagged)
    return ranking


# v58 AUDIT: Removed _detect_volume_squeeze() and _compute_trend_strength()
# Both were dead code: computed but never used in scoring formula or multiplier chain.
# squeeze_score correlation ~0, trend_strength explicitly marked dead in v23.


# v53: KOL co-occurrence matrix — cached per cycle (15min TTL)
_cooc_cache: dict = {"matrix": None, "token_kols": None, "kol_tokens": None, "total_tokens": 0, "_ts": 0}
_COOC_TTL = 15 * 60  # 15 min


def _build_cooccurrence_matrix() -> tuple[dict, dict, dict, int]:
    """
    v53: Build KOL co-occurrence matrix from kol_mentions (last 7 days).
    Returns (token_kols, kol_tokens, cooccurrence_matrix, total_tokens).
    Cached for 15min (same as scrape cycle).
    """
    now = time.time()
    if _cooc_cache["matrix"] is not None and (now - _cooc_cache["_ts"]) < _COOC_TTL:
        return _cooc_cache["token_kols"], _cooc_cache["kol_tokens"], _cooc_cache["matrix"], _cooc_cache["total_tokens"]

    token_kols: dict[str, set] = defaultdict(set)
    kol_tokens: dict[str, set] = defaultdict(set)

    try:
        from supabase import create_client
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            return {}, {}, {}, 0

        client = create_client(url, key)
        # Paginated fetch: max 10K rows from last 7 days
        cutoff_iso = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        offset = 0
        page_size = 1000
        while offset < 10000:
            resp = client.table("kol_mentions").select("symbol, kol_group").gte("message_date", cutoff_iso).range(offset, offset + page_size - 1).execute()
            if not resp.data:
                break
            for row in resp.data:
                sym = row.get("symbol")
                kol = row.get("kol_group")
                if sym and kol:
                    token_kols[sym].add(kol)
                    kol_tokens[kol].add(sym)
            if len(resp.data) < page_size:
                break
            offset += page_size

    except Exception as e:
        logger.warning("v53: co-occurrence matrix build failed: %s", e)
        return {}, {}, {}, 0

    # Build pairwise co-occurrence: for each pair of KOLs, count shared tokens
    matrix: dict[tuple, int] = {}
    kol_list = list(kol_tokens.keys())
    for i in range(len(kol_list)):
        for j in range(i + 1, len(kol_list)):
            shared = len(kol_tokens[kol_list[i]] & kol_tokens[kol_list[j]])
            if shared > 0:
                matrix[(kol_list[i], kol_list[j])] = shared
                matrix[(kol_list[j], kol_list[i])] = shared

    total_tokens = len(token_kols)
    _cooc_cache.update({"matrix": matrix, "token_kols": dict(token_kols), "kol_tokens": dict(kol_tokens), "total_tokens": total_tokens, "_ts": now})
    logger.info("v53: Co-occurrence matrix built: %d tokens, %d KOLs, %d pairs", total_tokens, len(kol_list), len(matrix) // 2)
    return dict(token_kols), dict(kol_tokens), matrix, total_tokens


def _compute_kol_cooccurrence(symbol: str, token_kol_set: set, kol_tokens: dict, matrix: dict, total_tokens: int) -> tuple[float | None, float | None]:
    """
    v53: Compute KOL co-occurrence metrics for a token.
    Returns (kol_cooccurrence_avg, kol_combo_novelty).
    """
    if not token_kol_set or len(token_kol_set) < 2 or not matrix:
        return None, None

    kol_list = list(token_kol_set)
    pairwise_scores = []
    for i in range(len(kol_list)):
        for j in range(i + 1, len(kol_list)):
            shared = matrix.get((kol_list[i], kol_list[j]), 0)
            min_tokens = min(len(kol_tokens.get(kol_list[i], set())), len(kol_tokens.get(kol_list[j], set())))
            if min_tokens > 0:
                pairwise_scores.append(shared / min_tokens)

    if not pairwise_scores:
        return None, None

    cooc_avg = round(sum(pairwise_scores) / len(pairwise_scores), 4)

    # Novelty: 1 - (observed overlap / expected random overlap)
    # Expected: if KOL_i mentions k_i tokens and KOL_j mentions k_j tokens out of N total,
    # expected shared = k_i * k_j / N
    novelty_scores = []
    for i in range(len(kol_list)):
        for j in range(i + 1, len(kol_list)):
            k_i = len(kol_tokens.get(kol_list[i], set()))
            k_j = len(kol_tokens.get(kol_list[j], set()))
            shared = matrix.get((kol_list[i], kol_list[j]), 0)
            expected = (k_i * k_j) / max(1, total_tokens)
            if expected > 0:
                novelty_scores.append(1 - min(1, shared / expected))

    combo_novelty = round(sum(novelty_scores) / len(novelty_scores), 4) if novelty_scores else None

    return cooc_avg, combo_novelty


def _tier_lookup(value: float, thresholds: list, factors: list) -> float:
    """v45: Generic tier-based lookup. thresholds=[t0,t1,...], factors=[f0,f1,...,f_last].
    Returns factors[i] for the first threshold[i] where value < threshold[i].
    If value >= all thresholds, returns factors[-1]."""
    for i, t in enumerate(thresholds):
        if value < t:
            return factors[i]
    return factors[-1]


def _compute_onchain_multiplier(token: dict) -> float:
    """
    Returns a multiplier [0.3, 1.5] based on on-chain health.
    v45: Tier thresholds read from SCORING_PARAMS["onchain_config"].
    """
    cfg = SCORING_PARAMS["onchain_config"]
    factors = []

    # 1. Volume/MCap ratio (healthy > 0.1, great > 0.5)
    vmr = token.get("volume_mcap_ratio")
    if vmr is not None:
        factors.append(min(1.5, 0.5 + vmr * 2))

    # 3. Liquidity adequacy
    lmr = token.get("liq_mcap_ratio")
    if lmr is not None:
        if lmr < cfg["lmr_low"]:
            factors.append(cfg["lmr_low_factor"])
        elif lmr > cfg["lmr_high"]:
            factors.append(cfg["lmr_high_factor"])
        else:
            factors.append(0.8 + lmr * 4)

    # 5. Token age
    age = token.get("token_age_hours")
    if age is not None:
        factors.append(_tier_lookup(age, cfg["age_thresholds"], cfg["age_factors"]))

    # 6. Helius recent transaction activity
    recent_tx = token.get("helius_recent_tx_count")
    if recent_tx is not None:
        if recent_tx > 40:
            factors.append(1.3)
        elif recent_tx > 20:
            factors.append(1.1)
        elif recent_tx < 5:
            factors.append(0.7)

    # 7. Helius on-chain buy/sell ratio
    helius_bsr = token.get("helius_onchain_bsr")
    if helius_bsr is not None:
        factors.append(0.5 + helius_bsr)

    # 8. Jupiter tradeability + liquidity depth
    jup_tradeable = token.get("jup_tradeable")
    jup_impact = token.get("jup_price_impact_1k")
    if jup_tradeable is not None:
        if jup_tradeable == 0:
            factors.append(0.5)
        elif jup_impact is not None:
            jup_t = cfg["jup_impact_thresholds"]
            jup_f = cfg["jup_impact_factors"]
            factors.append(_tier_lookup(jup_impact, jup_t, jup_f))

    # 9. Whale accumulation signal
    whale_change = token.get("whale_change")
    if whale_change is not None:
        wc_t = cfg["whale_change_thresholds"]
        wc_f = cfg["whale_change_factors"]
        factors.append(_tier_lookup(whale_change, wc_t, wc_f))

    # 10. Short-term volume heat — compute for data but NOT in factors
    vol_1h = token.get("volume_1h")
    vol_6h = token.get("volume_6h")
    if vol_1h and vol_6h and vol_6h > 0:
        short_heat = (vol_1h * 6) / vol_6h
        token["short_term_heat"] = round(short_heat, 3)

    vol_5m = token.get("volume_5m")
    if vol_5m and vol_1h and vol_1h > 0:
        ultra_heat = (vol_5m * 12) / vol_1h
        token["ultra_short_heat"] = round(ultra_heat, 3)

    # 11. Transaction velocity
    txn_count = token.get("txn_count_24h")
    holders = token.get("helius_holder_count") or token.get("holder_count")
    if txn_count and holders and holders > 0:
        velocity = txn_count / holders
        token["txn_velocity"] = round(velocity, 3)
        vel_t = cfg["velocity_thresholds"]
        vel_f = cfg["velocity_factors"]
        factors.append(_tier_lookup(velocity, vel_t, vel_f))
    elif txn_count and txn_count > 0:
        token["txn_velocity"] = None
        factors.append(_tier_lookup(txn_count, cfg["tx_thresholds"], cfg["tx_factors"]))

    # Volatility proxy penalty
    vol_proxy = token.get("volatility_proxy")
    if vol_proxy is not None and vol_proxy > cfg["vol_proxy_threshold"]:
        factors.append(cfg["vol_proxy_penalty"])

    # Whale direction accumulation bonus
    whale_dir = token.get("whale_direction")
    if whale_dir == "accumulating":
        factors.append(cfg["whale_accum_bonus"])

    # New whale entries
    wne = token.get("whale_new_entries")
    if wne is not None:
        wne_t = cfg["wne_thresholds"]
        wne_f = cfg["wne_factors"]
        factors.append(_tier_lookup(wne, wne_t, wne_f))

    # Whale count — #1 predictor (r=+0.47)
    whale_count = token.get("whale_count")
    if whale_count is not None:
        wct = cfg["whale_count_thresholds"]
        wcf = cfg["whale_count_factors"]
        factors.append(_tier_lookup(whale_count, wct, wcf))

    # Unique wallet growth
    uw_change = token.get("unique_wallet_24h_change")
    if uw_change is not None:
        uwt = cfg["uw_change_thresholds"]
        uwf = cfg["uw_change_factors"]
        factors.append(_tier_lookup(uw_change, uwt, uwf))

    # v53: Smart money retention — high retention = bullish
    smr = token.get("smart_money_retention")
    if smr is not None:
        smr_t = cfg.get("smr_thresholds", [50, 70, 90])
        smr_f = cfg.get("smr_factors", [0.8, 1.0, 1.15, 1.3])
        factors.append(_tier_lookup(smr, smr_t, smr_f))

    # v53: Small holder pct — more retail = organic
    shp = token.get("small_holder_pct")
    if shp is not None:
        shp_t = cfg.get("shp_thresholds", [50, 70, 85])
        shp_f = cfg.get("shp_factors", [0.7, 0.9, 1.1, 1.3])
        factors.append(_tier_lookup(shp, shp_t, shp_f))

    # v53: Liquidity depth score — deep liquidity = safer
    lds = token.get("liquidity_depth_score")
    if lds is not None:
        lds_t = cfg.get("lds_thresholds", [0.2, 0.5, 0.8])
        lds_f = cfg.get("lds_factors", [0.6, 0.85, 1.05, 1.2])
        factors.append(_tier_lookup(lds, lds_t, lds_f))

    if not factors:
        return 1.0

    return max(SCORING_PARAMS["onchain_mult_floor"],
               min(SCORING_PARAMS["onchain_mult_cap"], sum(factors) / len(factors)))


def _compute_safety_penalty(token: dict) -> float:
    """
    Returns a penalty multiplier [0.3, 1.0].
    1.0 = safe, 0.3 = floor (never destroy a token completely).
    v45: All thresholds read from SCORING_PARAMS["safety_config"].
    """
    cfg = SCORING_PARAMS["safety_config"]
    penalty = 1.0

    # High insider concentration (bundled)
    insider_pct = token.get("insider_pct")
    if insider_pct is not None and insider_pct > cfg["insider_threshold"]:
        penalty *= max(cfg["insider_floor"], 1.0 - (insider_pct - cfg["insider_threshold"]) / 100)

    # Top 10 holders own too much
    top10 = token.get("top10_holder_pct")
    if top10 is not None and top10 > cfg["top10_threshold"]:
        penalty *= max(cfg["top10_floor"], 1.0 - (top10 - cfg["top10_threshold"]) / 100)

    # RugCheck risk score (0=safe, 10000=dangerous)
    risk = token.get("risk_score")
    if risk is not None and risk > cfg["risk_score_threshold"]:
        penalty *= max(cfg["risk_score_floor"], 1.0 - (risk - cfg["risk_score_threshold"]) / 5000)

    # Multiple risk flags
    risk_count = token.get("risk_count", 0) or 0
    if risk_count >= cfg["risk_count_threshold"]:
        penalty *= cfg["risk_count_penalty"]

    # Jito slot-based bundle detection
    jito_max_txns = token.get("jito_max_slot_txns") or 0
    if jito_max_txns >= cfg["jito_hard_threshold"]:
        penalty *= cfg["jito_hard_penalty"]
    elif jito_max_txns >= cfg["jito_soft_threshold"]:
        penalty *= cfg["jito_soft_penalty"]
    else:
        # Fallback: balance-based bundle detection
        bundle_detected = token.get("bundle_detected")
        bundle_pct = token.get("bundle_pct") or 0
        if bundle_detected:
            if bundle_pct > cfg["bundle_hard_threshold"]:
                penalty *= cfg["bundle_hard_penalty"]
            elif bundle_pct > cfg["bundle_soft_threshold"]:
                penalty *= cfg["bundle_soft_penalty"]

    # High Gini = wealth too concentrated
    helius_gini = token.get("helius_gini")
    if helius_gini is not None and helius_gini > cfg["gini_threshold"]:
        penalty *= cfg["gini_penalty"]

    # Too few holders
    helius_holder_count = token.get("helius_holder_count")
    if helius_holder_count is not None and helius_holder_count < cfg["holder_count_threshold"]:
        penalty *= cfg["holder_count_penalty"]

    # Whale concentration too high
    whale_total_pct = token.get("whale_total_pct")
    if whale_total_pct is not None and whale_total_pct > cfg["whale_conc_threshold"]:
        penalty *= max(cfg["whale_conc_floor"], 1.0 - (whale_total_pct - cfg["whale_conc_threshold"]) / 80)

    # Bubblemaps — low decentralization = clustered supply
    bb_score = token.get("bubblemaps_score")
    if bb_score is not None:
        bb_thresholds = cfg["bb_score_thresholds"]
        bb_penalties = cfg["bb_score_penalties"]
        if bb_score < bb_thresholds[0]:
            penalty *= bb_penalties[0]
        elif bb_score < bb_thresholds[1]:
            penalty *= bb_penalties[1]

    # Bubblemaps — largest wallet cluster holds too much
    bb_cluster_max = token.get("bubblemaps_cluster_max_pct")
    if bb_cluster_max is not None and bb_cluster_max > cfg["bb_cluster_threshold"]:
        penalty *= max(cfg["bb_cluster_floor"], 1.0 - (bb_cluster_max - cfg["bb_cluster_threshold"]) / 70)

    # Whale dominance (concentration * inequality)
    whale_dom = token.get("whale_dominance")
    if whale_dom is not None and whale_dom > cfg["whale_dom_threshold"]:
        penalty *= cfg["whale_dom_penalty"]

    # Whale direction tracking
    whale_dir = token.get("whale_direction")
    if whale_dir == "distributing":
        penalty *= cfg["whale_dist_penalty"]
    elif whale_dir == "dumping":
        penalty *= cfg["whale_dump_penalty"]

    # LP lock — unlocked LP = rug vector
    lp_locked = token.get("lp_locked_pct")
    if lp_locked is not None:
        if lp_locked == 0:
            penalty *= cfg["lp_unlock_penalty"]
        elif lp_locked < cfg["lp_partial_threshold"]:
            penalty *= cfg["lp_partial_penalty"]

    # CEX supply pressure
    cex_pct = token.get("bubblemaps_cex_pct")
    if cex_pct is not None and cex_pct > cfg["cex_threshold"]:
        penalty *= max(cfg["cex_floor"], 1.0 - (cex_pct - cfg["cex_threshold"]) / 100)

    return max(SCORING_PARAMS["safety_floor"], penalty)


# === SCORE COMPUTATION WEIGHTS ===
# Hardcoded fallback — overridden by scoring_config table when available
_DEFAULT_WEIGHTS = {
    "consensus": 0.35,      # v43: synced with scoring_config (auto_backtest Feb 18)
    "conviction": 0.10,     # v43: synced with scoring_config (was 0.00)
    "breadth": 0.55,        # v43: synced with scoring_config (was 0.45)
    "price_action": 0.00,   # v43: synced with scoring_config (PA r=-0.01)
}

_DEFAULT_SCORING_PARAMS = {
    "combined_floor": 0.25,
    "combined_cap": 2.0,
    "safety_floor": 0.75,
    # v20: 15 critical constants made dynamic via scoring_config table
    "decay_lambda": 0.12,
    "activity_mult_floor": 0.80,
    "activity_mult_cap": 1.25,
    "pa_norm_floor": 0.4,
    "pa_norm_cap": 1.3,
    "onchain_mult_floor": 0.3,
    "onchain_mult_cap": 1.5,
    "death_pc24_severe": -80,
    "death_pc24_moderate": -50,
    "pump_pc1h_hard": 30,
    "pump_pc5m_hard": 15,
    "stale_hours_severe": 48,
    "gate_top10_pct": 70,
    "gate_min_liquidity": 10000,
    "gate_min_holders": 30,
    # v22: ML model horizon + threshold (dynamic)
    "ml_horizon": "12h",
    "ml_threshold": 2.0,
    # ML v3: Bot strategy for capturable profit prediction
    "bot_strategy": "TP50_SL30",
    # v26: Market benchmarks (populated by auto_backtest)
    "market_benchmarks": {},
    # v44: Dynamic parameter optimization — activity bucketing
    "activity_ratio_high": 0.6,
    "activity_ratio_mid": 0.3,
    "activity_ratio_low": 0.1,
    "activity_pump_cap_hard": 80,
    "activity_pump_cap_soft": 50,
    # v44: S-tier bonus
    "s_tier_bonus": 1.2,
    # v44: JSONB configs for grouped breakpoints
    "momentum_config": {
        "kol_fresh_thresholds": [0.5, 0.2],
        "kol_fresh_factors": [1.20, 1.10, 1.05],
        "mhr_thresholds": [2.0, 1.0, 0.3],
        "mhr_factors": [1.15, 1.10, 1.05],
        "sth_thresholds": [3.0, 1.5],
        "sth_factors": [1.10, 1.05],
        "sth_penalty_threshold": 0.3,
        "sth_penalty_factor": 0.95,
        # v52: KOL timing alpha factors
        "cascade_factor_3plus": 1.15,
        "cascade_factor_2plus": 1.08,
        "early_factor_30min": 1.20,
        "early_factor_60min": 1.10,
        "late_penalty_360min": 0.85,
        "pvfc_penalty_3x": 0.60,
        "pvfc_penalty_2x": 0.75,
        "pvfc_penalty_1_5x": 0.90,
        "floor": 0.70,
        "cap": 1.40,
    },
    "breadth_pen_config": {
        "thresholds": [0.033, 0.05, 0.08],
        "penalties": [0.75, 0.85, 0.95],
    },
    "hype_pen_config": {
        "thresholds": [2, 4, 7],
        "penalties": [1.0, 0.85, 0.65, 0.50],
        # v53: KOL co-occurrence penalty — shill rings get penalized
        "cooc_config": {"threshold": 0.5, "penalty": 0.85},
    },
    "size_mult_config": {
        "mcap_thresholds": [300_000, 1_000_000, 5_000_000, 20_000_000, 50_000_000, 200_000_000, 500_000_000],
        "mcap_factors": [1.3, 1.15, 1.0, 0.85, 0.70, 0.50, 0.35, 0.25],
        "fresh_thresholds": [4, 12],
        "fresh_factors": [1.2, 1.1, 1.0],
        "large_cap_threshold": 50_000_000,
        "floor": 0.25,
        "cap": 1.5,
    },
    # v44: ML multiplier bounds (widened from [0.5,1.5])
    "ml_mult_floor": 0.3,
    "ml_mult_cap": 2.0,
    # v44: Scoring mode — formula (default), hybrid, ml_primary
    "scoring_mode": "formula",
    # v45: Full multiplier chain optimization — 6 new JSONB configs
    "safety_config": {
        "insider_threshold": 30, "insider_floor": 0.5,
        "top10_threshold": 50, "top10_floor": 0.7,
        "risk_score_threshold": 5000, "risk_score_floor": 0.5,
        "risk_count_threshold": 3, "risk_count_penalty": 0.9,
        "jito_hard_threshold": 5, "jito_hard_penalty": 0.4,
        "jito_soft_threshold": 3, "jito_soft_penalty": 0.6,
        "bundle_hard_threshold": 20, "bundle_hard_penalty": 0.5,
        "bundle_soft_threshold": 10, "bundle_soft_penalty": 0.7,
        "gini_threshold": 0.85, "gini_penalty": 0.8,
        "holder_count_threshold": 50, "holder_count_penalty": 0.85,
        "whale_conc_threshold": 60, "whale_conc_floor": 0.7,
        "bb_score_thresholds": [20, 40], "bb_score_penalties": [0.6, 0.85],
        "bb_cluster_threshold": 30, "bb_cluster_floor": 0.6,
        "whale_dom_threshold": 0.5, "whale_dom_penalty": 0.85,
        "whale_dist_penalty": 0.75, "whale_dump_penalty": 0.65,
        "lp_unlock_penalty": 0.6, "lp_partial_threshold": 50, "lp_partial_penalty": 0.85,
        "cex_threshold": 20, "cex_floor": 0.7,
    },
    "onchain_config": {
        "lmr_low": 0.02, "lmr_low_factor": 0.5,
        "lmr_high": 0.10, "lmr_high_factor": 1.2,
        "age_thresholds": [1, 6, 48, 168],
        "age_factors": [0.5, 1.0, 1.2, 1.0, 0.8],
        "tx_thresholds": [5, 20, 40], "tx_factors": [0.7, 1.0, 1.1, 1.3],
        "jup_impact_thresholds": [1.0, 5.0], "jup_impact_factors": [1.3, 1.0, 0.7],
        "whale_change_thresholds": [-10.0, 0, 5.0],
        "whale_change_factors": [0.6, 0.8, 1.1, 1.3],
        "velocity_thresholds": [0.2, 1.0, 5.0], "velocity_factors": [0.6, 1.0, 1.1, 1.3],
        "wne_thresholds": [1, 3], "wne_factors": [1.0, 1.1, 1.25],
        "whale_count_thresholds": [1, 3, 5],
        "whale_count_factors": [0.7, 1.0, 1.2, 1.4],
        "uw_change_thresholds": [-20, -5, 5, 20],
        "uw_change_factors": [0.6, 0.8, 1.0, 1.15, 1.3],
        "vol_proxy_threshold": 50, "vol_proxy_penalty": 0.8,
        "whale_accum_bonus": 1.15,
        # v53: Holder stability + liquidity depth sub-factors
        "smr_thresholds": [50, 70, 90], "smr_factors": [0.8, 1.0, 1.15, 1.3],
        "shp_thresholds": [50, 70, 85], "shp_factors": [0.7, 0.9, 1.1, 1.3],
        "lds_thresholds": [0.2, 0.5, 0.8], "lds_factors": [0.6, 0.85, 1.05, 1.2],
    },
    "death_config": {
        "stale_start_hours": 12,
        "stale_tiers": [24, 48, 72],
        "stale_bases": [0.7, 0.45, 0.25, 0.15],
        "vol_modulation_tiers": [50000, 100000, 500000, 1000000],
        "vol_modulation_bonuses": [0.15, 0.25, 0.35, 0.45],
        "vol_modulation_caps": [0.8, 0.85, 0.9, 0.95],
        "vol_death_24h": 5000, "vol_death_1h": 500, "vol_death_penalty": 0.15,
        "vol_floor_24h": 1000, "vol_floor_penalty": 0.1,
        "price_moderate_social_alive_h": 6, "price_moderate_vol_alive": 0.5,
        "price_mild_threshold": -30, "price_mild_stale_h": 24,
    },
    "entry_premium_config": {
        "tier_breakpoints": [1.0, 1.2, 2.0, 4.0, 8.0, 20.0],
        "tier_multipliers": [1.1, 1.0, 0.9, 0.7, 0.5, 0.35, 0.25],
        "tier_slopes": [0, 0, 0, 0.1, 0.05, 0.0125, 0],
        "duration_thresholds": [12, 24, 48],
        "duration_factors": [1.0, 1.15, 1.3, 1.5],
        "mcap_fallback_threshold": 50000000,
        "mcap_launch_assumed": 1000000,
        "mcap_tiers": [50, 200, 500],
        "mcap_multipliers": [0.85, 0.70, 0.50, 0.35],
    },
    "lifecycle_config": {
        "panic_pc24": -30, "panic_va": 0.5, "panic_penalty": 0.25,
        "profit_taking_pc24": 100, "profit_taking_vol_proxy": 40, "profit_taking_penalty": 0.35,
        "euphoria_uk": 3, "euphoria_pc24": 100, "euphoria_sent": 0.2, "euphoria_penalty": 0.5,
        "boom_uk": 2, "boom_pc24_low": 10, "boom_pc24_high": 200, "boom_va": 1.0,
        "boom_bonus": 1.1, "boom_large_cap_penalty": 0.85, "boom_large_cap_mcap": 50000000,
        "displacement_age": 6, "displacement_uk": 1, "displacement_pc24": 50, "displacement_penalty": 0.9,
    },
    "entry_drift_config": {
        "premium_gate": 1.2,
        "kol_divisor": 7,
        "activity_base": 0.80, "activity_range": 0.45,
        "fresh_tiers": [1, 4, 8], "fresh_scores": [1.0, 0.7, 0.4, 0.1],
        "stier_divisor": 2,
        "social_weights": [0.30, 0.25, 0.30, 0.15],
        "drift_factor": 0.25, "drift_floor": 0.50,
    },
    # v46: Price Action config — all PA thresholds/factors configurable
    "pa_config": {
        "pos_thresholds": [0.10, 0.20, 0.50, 0.70, 0.90],
        "pos_factors": [0.4, 0.8, 1.3, 0.9, 0.6, 0.4],
        "rsi_hard_pump": 80, "rsi_pump": 70,
        "rsi_freefall": 20, "rsi_dying": 30,
        "rsi_bounce_upper": 40, "rsi_plateau_lower": 40, "rsi_plateau_upper": 60,
        "rsi_strong_bounce_slope": 3, "rsi_plateau_slope_max": 3,
        "dir_hard_pump_mult": 1.0, "dir_pump_mult": 1.0,
        "dir_freefall_mult": 1.0, "dir_dying_mult": 1.0,
        "dir_strong_bounce_mult": 1.4, "dir_bounce_mult": 1.3,
        "dir_plateau_mult": 1.1, "dir_macd_bonus": 0.1, "dir_cap": 1.5,
        "fb_hard_pump_1h": 50, "fb_hard_pump_5m": 25, "fb_hard_pump_mult": 1.0,
        "fb_pump_1h": 20, "fb_pump_5m": 10, "fb_pump_mult": 1.0,
        "fb_dying_1h": -10, "fb_dying_6h": -30, "fb_dying_mult": 1.0,
        "fb_freefall_1h": -20, "fb_freefall_5m": -10, "fb_freefall_mult": 1.0,
        "fb_plateau_1h": 5, "fb_plateau_6h": 10, "fb_plateau_mult": 1.1,
        "fb_bounce_6h": -15, "fb_bounce_1h": 5, "fb_bounce_mult": 1.3,
        "fb_strong_bounce_6h": -30, "fb_strong_bounce_1h": 10, "fb_strong_bounce_mult": 1.4,
        "fb_bsr_bounce_bonus": 0.1, "fb_bsr_bounce_threshold": 0.6,
        "fb_bsr_dead_cat_penalty": 0.2, "fb_bsr_dead_cat_threshold": 0.4,
        "pc24_freefall": -60, "pc24_freefall_mult": 1.0,
        "pc24_dying": -40, "pc24_dying_mult": 1.0,
        "pc24_bleeding": -20, "pc24_bleeding_mult": 1.0,
        "pc24_pumping_hard": 100, "pc24_pumping_hard_mult": 1.0,
        "pc24_pumping": 50, "pc24_pumping_mult": 1.0,
        "pc24_pos_crash70_mult": 0.4, "pc24_pos_crash50_mult": 0.6, "pc24_pos_crash30_mult": 0.8,
        "pc24_pos_pump200_mult": 0.5, "pc24_pos_pump100_mult": 0.6,
        "obv_strong_accum": 2.0, "obv_strong_mult": 1.3,
        "obv_mod_accum": 0.5, "obv_mod_mult": 1.2,
        "obv_dead_cat": -0.5, "obv_dead_cat_mult": 0.6,
        "obv_pump_exhaust": -0.5, "obv_pump_exhaust_mult": 0.7,
        "obv_bottom_accum": 1.0, "obv_bottom_mult": 1.2,
        "ush_high": 2.0, "ush_high_mult": 1.3,
        "ush_mid": 1.2, "ush_mid_mult": 1.1,
        "ush_low": 0.3, "ush_low_mult": 0.6,
        "vol_conc_high": 0.3, "vol_conc_high_mult": 1.2,
        "vol_conc_mid": 0.1, "vol_conc_mid_mult": 1.05,
        "vol_conc_dead": 0.02, "vol_conc_dead_mult": 0.6,
        "bb_strong_pctb": 0.1, "bb_strong_mult": 1.2,
        "bb_near_pctb": 0.2, "bb_near_mult": 1.15,
        "candle_strong_count": 3, "candle_strong_mult": 1.15,
        "candle_bounce_count": 2, "candle_bounce_mult": 1.2,
        "support_tolerance": 0.03,
    },
    # v47: Message mention weighting — type weights, length thresholds, call type adjustments
    "mention_weight_config": {
        "bot_weight": 0.0,
        "forward_weight": 0.2,
        "update_weight": 0.3,
        "reply_weight": 0.4,
        "original_weight": 1.0,
        "long_length_threshold": 150,
        "long_length_weight": 1.2,
        "short_length_threshold": 20,
        "short_length_weight": 0.5,
        "alpha_framing_score": 1.0,
        "gamble_framing_score": -1.0,
        "chart_analysis_score": 0.5,
        "positive_mention_threshold": -0.3,
    },
    # v47: Conviction normalization + dampening + data confidence
    "conviction_config": {
        "norm_offset": 6,
        "norm_divisor": 4,
        "kol_dampening": 2,
        "confidence_w_component": 0.4,
        "confidence_w_kol": 0.3,
        "confidence_w_breadth": 0.2,
        "confidence_w_enrichment": 0.1,
        "kol_conf_max": 3,
        "breadth_conf_min": 0.15,
        "breadth_conf_default": 0.3,
    },
    # v47: Pump momentum penalties + PVP penalty scaling
    "pump_pen_config": {
        "hard_penalty": 0.5,
        "moderate_penalty": 0.7,
        "light_penalty": 0.85,
        "pvp_pump_fun_floor": 0.7,
        "pvp_pump_fun_scale": 0.05,
        "pvp_normal_floor": 0.5,
        "pvp_normal_scale": 0.1,
    },
}

# Module-level cache: refreshed once per scrape cycle via load_scoring_config()
BALANCED_WEIGHTS = _DEFAULT_WEIGHTS.copy()
SCORING_PARAMS = _DEFAULT_SCORING_PARAMS.copy()


def load_scoring_config() -> None:
    """
    Load scoring weights + params from Supabase scoring_config table.
    Called once at the start of each scrape cycle.
    Falls back to hardcoded defaults on any error.
    """
    global BALANCED_WEIGHTS, SCORING_PARAMS
    try:
        from supabase import create_client
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            logger.debug("scoring_config: no Supabase credentials, using defaults")
            return

        client = create_client(url, key)
        result = client.table("scoring_config").select("*").eq("id", 1).execute()
        if not result.data:
            logger.debug("scoring_config: table empty, using defaults")
            return

        row = result.data[0]
        new_weights = {
            "consensus": float(row["w_consensus"]),
            "conviction": float(row["w_conviction"]),
            "breadth": float(row["w_breadth"]),
            "price_action": float(row["w_price_action"]),
        }
        # Validate sum ≈ 1.0
        total = sum(new_weights.values())
        if abs(total - 1.0) > 0.02:
            logger.warning("scoring_config: weights sum %.3f != 1.0, using defaults", total)
            return

        BALANCED_WEIGHTS.update(new_weights)
        SCORING_PARAMS["combined_floor"] = float(row.get("combined_floor", 0.25))
        SCORING_PARAMS["combined_cap"] = float(row.get("combined_cap", 2.0))
        SCORING_PARAMS["safety_floor"] = float(row.get("safety_floor", 0.75))

        # v20: Load 15 dynamic scoring constants (safe fallback per key)
        _DYNAMIC_KEYS = {
            "decay_lambda": 0.12,
            "activity_mult_floor": 0.80,
            "activity_mult_cap": 1.25,
            "pa_norm_floor": 0.4,
            "pa_norm_cap": 1.3,
            "onchain_mult_floor": 0.3,
            "onchain_mult_cap": 1.5,
            "death_pc24_severe": -80,
            "death_pc24_moderate": -50,
            "pump_pc1h_hard": 30,
            "pump_pc5m_hard": 15,
            "stale_hours_severe": 48,
            "gate_top10_pct": 70,
            "gate_min_liquidity": 10000,
            "gate_min_holders": 30,
            # v41: Feature computation constants (were hardcoded)
            "consensus_norm_divisor": 0.10,
            "breadth_norm_divisor": 10.0,
            "mention_heat_epsilon": 1.0,
            "mention_heat_cap": 10.0,
        }
        for key, default in _DYNAMIC_KEYS.items():
            SCORING_PARAMS[key] = float(row.get(key, default))

        # v22: ML model horizon + threshold
        SCORING_PARAMS["ml_horizon"] = row.get("ml_horizon", "12h") or "12h"
        SCORING_PARAMS["ml_threshold"] = float(row.get("ml_threshold", 2.0) or 2.0)

        # ML v3: Bot strategy for capturable profit prediction
        SCORING_PARAMS["bot_strategy"] = row.get("bot_strategy", "TP50_SL30") or "TP50_SL30"

        # v26: Market benchmarks (computed by auto_backtest, stored as JSONB)
        SCORING_PARAMS["market_benchmarks"] = row.get("market_benchmarks") or {}

        # v44: Dynamic parameter optimization — new scalar params
        _V44_SCALAR_KEYS = {
            "activity_ratio_high": 0.6,
            "activity_ratio_mid": 0.3,
            "activity_ratio_low": 0.1,
            "activity_pump_cap_hard": 80,
            "activity_pump_cap_soft": 50,
            "s_tier_bonus": 1.2,
            "ml_mult_floor": 0.3,
            "ml_mult_cap": 2.0,
            # v47: Remaining hardcoded scalar params
            "consensus_pump_threshold": 50,
            "consensus_pump_floor": 0.5,
            "consensus_pump_divisor": 400,
            "activity_mid_mult": 1.10,
            "mention_heat_divisor": 4,
        }
        for key, default in _V44_SCALAR_KEYS.items():
            SCORING_PARAMS[key] = float(row.get(key, default))

        # v44: JSONB configs (grouped breakpoints for Optuna optimization)
        _V44_JSONB_KEYS = [
            "momentum_config", "breadth_pen_config",
            "hype_pen_config", "size_mult_config",
            # v45: Full multiplier chain optimization
            "safety_config", "onchain_config", "death_config",
            "entry_premium_config", "lifecycle_config", "entry_drift_config",
            # v46: Price Action config
            "pa_config",
            # v47: Remaining hardcoded params
            "mention_weight_config", "conviction_config", "pump_pen_config",
            # v56: KOL win rate multiplier config
            "kol_wr_config",
        ]
        for key in _V44_JSONB_KEYS:
            val = row.get(key)
            if val and isinstance(val, dict):
                SCORING_PARAMS[key] = val

        # v44: Scoring mode
        SCORING_PARAMS["scoring_mode"] = row.get("scoring_mode", "formula") or "formula"

        logger.info(
            "scoring_config loaded: consensus=%.2f conviction=%.2f breadth=%.2f PA=%.2f "
            "decay=%.3f activity=[%.2f,%.2f] ml=%s/%.1fx bot=%s (updated_by=%s, %s)",
            new_weights["consensus"], new_weights["conviction"],
            new_weights["breadth"], new_weights["price_action"],
            SCORING_PARAMS["decay_lambda"],
            SCORING_PARAMS["activity_mult_floor"], SCORING_PARAMS["activity_mult_cap"],
            SCORING_PARAMS["ml_horizon"], SCORING_PARAMS["ml_threshold"],
            SCORING_PARAMS["bot_strategy"],
            row.get("updated_by", "?"), row.get("change_reason", ""),
        )
    except Exception as e:
        logger.warning("scoring_config: failed to load (%s), using defaults", e)


def _get_component_value(token: dict, component: str) -> float | None:
    """
    Extract a normalized [0, 1] component value from a token dict.
    Returns None if data is genuinely missing.
    """
    if component == "consensus":
        # v9: Use pre-computed recency-decayed consensus (avoids recomputing without decay)
        decayed = token.get("_decayed_consensus")
        if decayed is not None:
            return decayed
        # Fallback for tokens without decay data (e.g. price_refresh path)
        uk = token.get("unique_kols")
        if uk is None:
            return None
        tk = token.get("_total_kols", 50)
        kol_tiers = token.get("kol_tiers", {})
        if kol_tiers:
            tw_func = token.get("_tw_func")
            if tw_func:
                weighted = sum(tw_func(g) for g in kol_tiers)
            else:
                weighted = uk
            return min(1.0, weighted / (tk * SCORING_PARAMS["consensus_norm_divisor"]))
        return min(1.0, uk / (tk * SCORING_PARAMS["consensus_norm_divisor"]))
    elif component == "sentiment":
        s = token.get("sentiment")
        if s is None:
            return None
        return (s + 1) / 2
    elif component == "conviction":
        ac = token.get("avg_conviction")
        if ac is None:
            return None
        # v14: Compressed range — 7→0.25, 8→0.5, 10→1.0
        # v47: offset/divisor from conviction_config
        ccfg = SCORING_PARAMS["conviction_config"]
        return max(0, min(1, (ac - ccfg["norm_offset"]) / ccfg["norm_divisor"]))
    elif component == "breadth":
        bs = token.get("breadth_score")
        if bs is not None:
            return bs
        m = token.get("mentions")
        if m is None:
            return None
        # v14/v41: Recalibrated — now uses breadth_norm_divisor from config
        return min(1.0, m / SCORING_PARAMS["breadth_norm_divisor"])
    elif component == "price_action":
        pa = token.get("price_action_score")
        return pa  # None if no OHLCV data
    return None


def _compute_score_with_renormalization(token: dict) -> tuple[float, float]:
    """
    Compute balanced score with weight renormalization for missing components.
    Returns (raw_score_0_to_1, data_confidence).

    v14: data_confidence now reflects actual data quality, not just component presence.
    Factors: component availability, unique KOL count, breadth value, enrichment data.
    """
    available = {}
    for comp, weight in BALANCED_WEIGHTS.items():
        value = _get_component_value(token, comp)
        if value is not None:
            available[comp] = (value, weight)

    # v17: Consensus discount when token already pumped — late KOL mentions
    # (arriving after the pump) are worth less than early calls.
    # v47: thresholds from scoring_config (Optuna-tunable).
    _cp_thresh = SCORING_PARAMS["consensus_pump_threshold"]   # default 50
    _cp_floor = SCORING_PARAMS["consensus_pump_floor"]        # default 0.5
    _cp_divisor = SCORING_PARAMS["consensus_pump_divisor"]    # default 400
    pc24 = token.get("price_change_24h")
    if "consensus" in available and pc24 is not None and pc24 > _cp_thresh:
        consensus_discount = max(_cp_floor, 1.0 - (pc24 / _cp_divisor))
        old_val, old_weight = available["consensus"]
        available["consensus"] = (old_val * consensus_discount, old_weight)
        token["_consensus_pump_discount"] = round(consensus_discount, 3)
    else:
        token["_consensus_pump_discount"] = 1.0

    total_available_weight = sum(w for _, w in available.values())
    if total_available_weight == 0:
        return 0.0, 0.0

    renormalized_score = sum(v * (w / total_available_weight) for v, w in available.values())

    # v14: Multi-factor data confidence
    # v47: All weights/thresholds from conviction_config (Optuna-tunable).
    ccfg = SCORING_PARAMS["conviction_config"]
    component_conf = total_available_weight / sum(BALANCED_WEIGHTS.values())
    # KOL coverage: 1 KOL = low confidence, 3+ KOLs = full
    uk = token.get("unique_kols", 1)
    kol_conf = min(1.0, uk / ccfg["kol_conf_max"])
    # Breadth quality: near-zero breadth = low confidence
    breadth = token.get("breadth_score", 0) or 0
    breadth_conf = min(1.0, breadth / ccfg["breadth_conf_min"]) if breadth > 0 else ccfg["breadth_conf_default"]
    # Enrichment: has on-chain data?
    enrichment_conf = 1.0 if token.get("token_address") else 0.5

    data_confidence = (component_conf * ccfg["confidence_w_component"]
                       + kol_conf * ccfg["confidence_w_kol"]
                       + breadth_conf * ccfg["confidence_w_breadth"]
                       + enrichment_conf * ccfg["confidence_w_enrichment"])

    return renormalized_score, round(data_confidence, 3)


def _classify_lifecycle_phase(token: dict) -> tuple[str, float]:
    """
    Classify token into Minsky lifecycle phase.
    v45: All thresholds read from SCORING_PARAMS["lifecycle_config"].
    """
    cfg = SCORING_PARAMS["lifecycle_config"]
    pc24 = token.get("price_change_24h")
    uk = token.get("unique_kols", 0)
    va = token.get("volume_acceleration")
    sv = token.get("social_velocity", 0)
    ma = token.get("mention_acceleration", 0)
    sent = token.get("sentiment", 0)
    vol_proxy = token.get("volatility_proxy")
    whale_dir = token.get("whale_direction")
    age = token.get("token_age_hours")

    # Panic: dump in progress
    if pc24 is not None and pc24 < cfg["panic_pc24"]:
        if (va is not None and va < cfg["panic_va"]) or whale_dir == "dumping":
            return "panic", cfg["panic_penalty"]

    # Profit-taking: smart money exiting
    if pc24 is not None and pc24 > cfg["profit_taking_pc24"]:
        vol_spike = vol_proxy is not None and vol_proxy > cfg["profit_taking_vol_proxy"]
        whale_exit = whale_dir in ("distributing", "dumping")
        if (vol_spike or whale_exit) and ma < 0:
            return "profit_taking", cfg["profit_taking_penalty"]

    # Euphoria: over-saturated
    if uk >= cfg["euphoria_uk"] and pc24 is not None and pc24 > cfg["euphoria_pc24"] and sent > cfg["euphoria_sent"]:
        return "euphoria", cfg["euphoria_penalty"]

    # Boom: sweet spot
    t_mcap = token.get("market_cap") or 0
    if uk >= cfg["boom_uk"] and pc24 is not None and cfg["boom_pc24_low"] < pc24 <= cfg["boom_pc24_high"]:
        has_vol = va is not None and va > cfg["boom_va"]
        has_social = sv > 0.3 or ma > 0.2
        if has_vol and has_social:
            if t_mcap > cfg["boom_large_cap_mcap"]:
                return "boom", cfg["boom_large_cap_penalty"]
            return "boom", cfg["boom_bonus"]

    # Displacement: fresh, unproven
    if age is not None and age < cfg["displacement_age"] and uk <= cfg["displacement_uk"]:
        if pc24 is None or pc24 < cfg["displacement_pc24"]:
            return "displacement", cfg["displacement_penalty"]

    return "unknown", 1.0


def _identify_weakest_component(token: dict) -> tuple[str, float]:
    """
    Identify the weakest scoring component and return (name, value).
    """
    components = {}
    for comp in BALANCED_WEIGHTS:
        val = _get_component_value(token, comp)
        if val is not None:
            components[comp] = val

    if not components:
        return "unknown", 0.0

    weakest = min(components, key=components.get)
    return weakest, round(components[weakest], 3)


def _interpret_score(score: int) -> str:
    """Return human-readable interpretation band."""
    if score >= 80:
        return "strong_signal"
    elif score >= 65:
        return "good_signal"
    elif score >= 50:
        return "moderate_signal"
    elif score >= 35:
        return "weak_signal"
    else:
        return "low_conviction"


def aggregate_ranking(
    messages_dict: dict[str, list[dict]],
    groups_conviction: dict[str, int],
    hours: int,
    groups_tier: dict[str, str] | None = None,
    tier_weights: dict[str, float] | None = None,
) -> list[TokenRanking]:
    """
    Given raw messages grouped by channel username, compute ranked token list
    for the specified time window.

    Parameters
    ----------
    messages_dict : { "group_username": [ { "text": ..., "date": ISO str, ... }, ... ] }
    groups_conviction : { "group_username": conviction_int }
    hours : time window in hours
    groups_tier : { "group_username": "S" | "A" } — KOL tier classification
    tier_weights : { "S": 2.0, "A": 1.0 } — weight multiplier per tier

    Returns
    -------
    (ranking, raw_kol_mentions, all_enriched) where:
    - ranking: Sorted list of surviving TokenRanking dicts (highest score first)
    - raw_kol_mentions: List of raw KOL mention dicts for NLP storage
    - all_enriched: ALL tokens after DexScreener enrichment (including gated ones
      with gate_reason set). Used for backtesting snapshots.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours)
    total_kols = len(groups_conviction)

    # Tier weight helper: returns the tier multiplier for a KOL group
    def tw(group: str) -> float:
        if tier_weights and groups_tier:
            return tier_weights.get(groups_tier.get(group, "A"), 1.0)
        return 1.0

    # Load KOL reputation scores
    kol_scores = get_kol_scores()
    if kol_scores:
        logger.info("Loaded KOL reputation scores for %d KOLs", len(kol_scores))

    # Load CA cache once for the entire aggregation cycle
    ca_cache = _load_ca_cache()

    # v56: Load KOL win rates once per cycle
    kol_win_rates = _load_kol_win_rates()

    # === Phase 1: Build confirmed symbols from $-prefix and CA across ALL messages ===
    # A symbol is "confirmed" if at least one KOL explicitly named it with $ or posted its CA.
    # Bare ALLCAPS words (like "SHARK") only count as mentions if confirmed here.
    confirmed_symbols: set[str] = set()

    for group_name, msgs in messages_dict.items():
        for msg in msgs:
            text = msg.get("text", "")
            if not text or len(text) < 3:
                continue
            # $-prefixed tokens → confirmed
            for match in TOKEN_REGEX.findall(text):
                upper = match.upper()
                if upper.isascii() and upper not in EXCLUDED_TOKENS:
                    confirmed_symbols.add(upper)
            # CA-resolved tokens → confirmed
            for match in CA_REGEX.findall(text):
                if match in KNOWN_PROGRAM_ADDRESSES:
                    continue
                resolved = _resolve_ca_to_symbol(match, ca_cache)
                if resolved and resolved not in EXCLUDED_TOKENS:
                    confirmed_symbols.add(resolved)
            # DexScreener URL pair addresses → confirmed (v40: use _and_ca to warm cache with token_ca)
            for chain, pair_addr in DEXSCREENER_URL_REGEX.findall(text):
                resolved, _ = _resolve_pair_to_symbol_and_ca(chain, pair_addr, ca_cache)
                if resolved and resolved not in EXCLUDED_TOKENS:
                    confirmed_symbols.add(resolved)
            # pump.fun URLs → confirmed
            for pump_addr in PUMP_FUN_URL_REGEX.findall(text):
                resolved = _resolve_ca_to_symbol(pump_addr, ca_cache)
                if resolved and resolved not in EXCLUDED_TOKENS:
                    confirmed_symbols.add(resolved)
            # GMGN URLs → confirmed
            for gmgn_addr in GMGN_URL_REGEX.findall(text):
                resolved = _resolve_ca_to_symbol(gmgn_addr, ca_cache)
                if resolved and resolved not in EXCLUDED_TOKENS:
                    confirmed_symbols.add(resolved)
            # Photon-sol URLs → confirmed (v40: use _and_ca to warm cache with token_ca)
            for photon_addr in PHOTON_URL_REGEX.findall(text):
                resolved, _ = _resolve_pair_to_symbol_and_ca("solana", photon_addr, ca_cache)
                if resolved and resolved not in EXCLUDED_TOKENS:
                    confirmed_symbols.add(resolved)

    logger.info(
        "Phase 1: %d confirmed symbols from $-prefix, CA, and URL resolution",
        len(confirmed_symbols),
    )

    # === Phase 2: Full message processing with confirmation gate ===

    token_data: dict[str, TokenStats] = defaultdict(lambda: {
        "mentions": 0,
        "sentiments": [],
        "groups": set(),
        "convictions": [],
        "hours_ago": [],
        "msg_conviction_scores": [],  # per-message NLP conviction scores
        "price_target_count": 0,      # messages with price targets
        "hedging_count": 0,           # messages with hedging language
        # v25: Message-level text feature accumulators
        "call_type_scores": [],       # +1 alpha, -1 gamble, +0.5 chart_analysis, 0 neutral
        "msg_lengths": [],            # len(text) per message
        "msg_has_ca": [],             # bool: message contains a CA
        "msg_tokens_count": [],       # nb tokens extracted from the message
        "msg_texts_raw": [],          # raw text for caps/emoji/question/link analysis
        "kol_mention_counts": {},     # per-KOL mention count for quality weighting
        "hours_ago_by_group": defaultdict(list),  # v9: group_name → [hours_ago, ...] for recency decay
        "kol_stated_entry_mcaps": [],  # v14: entry mcap stated by KOLs in message text
        # v16: Extraction source counts
        "ca_mention_count": 0,
        "ticker_mention_count": 0,
        "url_mention_count": 0,
        # v40: Track contract addresses from CA/URL sources
        "known_cas": [],
    })

    # v10: Collect raw KOL mentions for NLP storage
    raw_kol_mentions: list[dict] = []

    for group_name, msgs in messages_dict.items():
        conviction = groups_conviction.get(group_name, 7)

        for msg in msgs:
            # Parse date
            date_str = msg.get("date", "")
            try:
                if date_str.endswith("Z"):
                    date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                elif "+" in date_str or date_str.endswith("00:00"):
                    date = datetime.fromisoformat(date_str)
                else:
                    date = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
            except (ValueError, AttributeError):
                continue

            if date < cutoff:
                continue

            text = msg.get("text", "")
            if not text or len(text) < 3:
                continue

            # v14: Cap tickers per message — scorecard/DCA-list posts
            # inflate many tickers at once. Keep first 3 only.
            token_tuples = extract_tokens(text, ca_cache=ca_cache, confirmed_symbols=confirmed_symbols)
            if not token_tuples:
                continue
            if len(token_tuples) > 3:
                token_tuples = token_tuples[:3]

            # Flat list of symbols for backward-compat uses (raw_kol_mentions etc.)
            tokens = [t[0] for t in token_tuples]
            # v16: Map symbol → extraction_method for per-mention audit
            source_by_symbol = {t[0]: t[1] for t in token_tuples}
            # v40: Map symbol → resolved CA for per-mention tracking
            ca_by_symbol = {t[0]: t[2] for t in token_tuples if t[2]}
            # v16: Extract ALL CAs from message text (for extraction audit)
            msg_cas = [m for m in CA_REGEX.findall(text) if m not in KNOWN_PROGRAM_ADDRESSES]

            sentiment = calculate_sentiment(text)
            hours_ago = (now - date).total_seconds() / 3600

            # v35b: Detect tracking bot brags (KOLscope, SpyDefi) — zero weight
            is_bot_msg = _is_tracking_bot_message(text)
            # v14: Detect update/brag messages — weight reduction
            is_update = _is_update_or_brag(text)
            # Forward/reply detection — forwarded content inflates breadth
            is_forwarded = msg.get("is_forwarded", False)
            is_reply = msg.get("is_reply", False)
            # v14: Message length as confidence signal
            # v47: All type weights + length thresholds from mention_weight_config.
            mwcfg = SCORING_PARAMS["mention_weight_config"]
            msg_len = len(text)
            if msg_len > mwcfg["long_length_threshold"]:
                length_weight = mwcfg["long_length_weight"]
            elif msg_len < mwcfg["short_length_threshold"]:
                length_weight = mwcfg["short_length_weight"]
            else:
                length_weight = 1.0
            # Combined mention weight: bots 0x, forwards 0.2x, updates 0.3x, replies 0.4x
            if is_bot_msg:
                mention_weight = mwcfg["bot_weight"]
            elif is_forwarded:
                mention_weight = mwcfg["forward_weight"] * length_weight
            elif is_update:
                mention_weight = mwcfg["update_weight"] * length_weight
            elif is_reply:
                mention_weight = mwcfg["reply_weight"] * length_weight
            else:
                mention_weight = mwcfg["original_weight"] * length_weight

            # Per-message conviction NLP (Sprint 6)
            msg_conv = _compute_message_conviction(text)

            # v14: Extract entry market cap from message text
            stated_mcap = _extract_entry_mcap(text)

            # v50: Deduplicate msg_cas for fallback resolution
            unique_msg_cas = list(dict.fromkeys(msg_cas)) if msg_cas else []

            # v56: CAs already claimed by a token via direct extraction should not
            # be assigned to other tokens. Build set of "owned" CAs.
            owned_cas = set(ca_by_symbol.values())
            # Unowned CAs = CAs in msg not claimed by any token
            unowned_cas = [c for c in unique_msg_cas if c not in owned_cas]

            # v10: Store raw mention for each token in this message
            for token in tokens:
                # v50: Resolve CA — prefer direct extraction, fallback to msg CAs
                # v56: Only fallback to UNOWNED CAs to prevent cross-contamination
                resolved = ca_by_symbol.get(token)
                if not resolved and unowned_cas:
                    if len(tokens) == 1:
                        # Single token in message — all CAs belong to it
                        resolved = unowned_cas[0]
                    elif len(unowned_cas) == 1 and len(tokens) - len(ca_by_symbol) == 1:
                        # Multiple tokens, 1 unowned CA, and only 1 token without CA — safe to assign
                        resolved = unowned_cas[0]

                raw_kol_mentions.append({
                    "symbol": token,
                    "kol_group": group_name,
                    "message_text": text[:2000],  # cap at 2000 chars
                    "message_date": date.isoformat(),
                    "sentiment": round(sentiment, 3),
                    "msg_conviction_score": round(msg_conv["msg_conviction_score"], 3),
                    "hours_ago": round(hours_ago, 2),
                    "is_positive": sentiment >= mwcfg["positive_mention_threshold"],
                    "narrative": None,
                    "tokens_in_message": list(tokens),
                    # v16: Extraction audit fields
                    "extraction_method": source_by_symbol.get(token, "unknown"),
                    "extracted_cas": msg_cas if msg_cas else None,
                    # v40+v50: Resolved CA — direct extraction or msg fallback
                    "resolved_ca": resolved,
                })

            for token, source, ca in token_tuples:
                # v40+v50+v56: Track known CAs — direct extraction OR unowned msg fallback
                if ca:
                    token_data[token]["known_cas"].append(ca)
                elif unowned_cas and (len(tokens) == 1 or (len(unowned_cas) == 1 and len(tokens) - len(ca_by_symbol) == 1)):
                    # v56: Same fallback logic as resolved_ca above — only unowned CAs
                    token_data[token]["known_cas"].append(unowned_cas[0])

                # v16: Track extraction source counts
                if source == "ca":
                    token_data[token]["ca_mention_count"] += 1
                elif source == "ticker":
                    token_data[token]["ticker_mention_count"] += 1
                elif source == "url":
                    token_data[token]["url_mention_count"] += 1

                # Always track sentiment (negative views are useful signal)
                token_data[token]["sentiments"].append(sentiment)
                token_data[token]["hours_ago"].append(hours_ago)
                token_data[token]["msg_conviction_scores"].append(msg_conv["msg_conviction_score"])
                # v25: Message-level text features
                # v47: call type weights from mention_weight_config
                ct = 0
                if msg_conv.get("has_alpha_framing"): ct += mwcfg["alpha_framing_score"]
                if msg_conv.get("has_gamble_framing"): ct += mwcfg["gamble_framing_score"]
                if msg_conv.get("has_chart_analysis"): ct += mwcfg["chart_analysis_score"]
                token_data[token]["call_type_scores"].append(ct)
                token_data[token]["msg_lengths"].append(msg_len)
                token_data[token]["msg_has_ca"].append(bool(msg_cas))
                token_data[token]["msg_tokens_count"].append(len(token_tuples))
                token_data[token]["msg_texts_raw"].append(text)

                # Negative mentions = KOL is warning, not calling
                # Don't count toward mentions/breadth/conviction — prevents false positives
                # v47: threshold from mention_weight_config
                is_positive_mention = sentiment >= mwcfg["positive_mention_threshold"]

                if is_positive_mention:
                    # v14: Weighted mentions — updates/brags count 0.3x, short msgs 0.5x
                    token_data[token]["mentions"] += mention_weight
                    # v35b: Bot messages excluded from breadth/conviction/kol_counts
                    if not is_forwarded and not is_bot_msg:
                        token_data[token]["groups"].add(group_name)
                    if not is_bot_msg:
                        token_data[token]["convictions"].append(conviction)
                        # Track per-KOL mention counts (weighted) for quality-weighted breadth
                        kol_counts = token_data[token]["kol_mention_counts"]
                        kol_counts[group_name] = kol_counts.get(group_name, 0) + mention_weight
                        # v9: Track hours_ago per group for recency-weighted consensus/breadth
                        token_data[token]["hours_ago_by_group"][group_name].append(hours_ago)
                        if msg_conv["has_price_target"]:
                            token_data[token]["price_target_count"] += 1
                        # v14: Store KOL-stated entry mcap
                        if stated_mcap is not None:
                            token_data[token]["kol_stated_entry_mcaps"].append(stated_mcap)

                if msg_conv["has_hedging"]:
                    token_data[token]["hedging_count"] += 1
    # Persist CA cache after processing all messages
    _save_ca_cache(ca_cache)

    # v58 AUDIT: Backfill resolved_ca on ticker-only mentions.
    # Problem: 90% of mentions had resolved_ca=NULL because ticker extractions
    # ($POPCAT) don't carry CAs. But the SAME token may have been mentioned in
    # other messages WITH a CA (via direct CA extraction or URL).
    # Fix: Use the token-level known_cas (majority vote) to backfill NULL resolved_ca.
    _backfill_count = 0
    for m in raw_kol_mentions:
        if m["resolved_ca"] is None:
            sym = m["symbol"]
            known = token_data.get(sym, {}).get("known_cas", [])
            if known:
                # Use most common CA for this symbol (same logic as kol_resolved_ca)
                m["resolved_ca"] = Counter(known).most_common(1)[0][0]
                _backfill_count += 1
    if _backfill_count:
        logger.info("v58: Backfilled resolved_ca on %d/%d ticker-only mentions using token-level known_cas",
                     _backfill_count, len(raw_kol_mentions))

    # Score & rank
    ranking: list[TokenRanking] = []

    for symbol, data in token_data.items():
        if data["mentions"] == 0:
            continue

        unique_kols = len(data["groups"])
        # v9: Recency-weighted consensus — fresh mentions matter, stale ones decay
        recency_weighted_unique = 0
        for g in data["groups"]:
            group_hours = data["hours_ago_by_group"].get(g, [])
            freshest = min(group_hours) if group_hours else 999
            decay = _two_phase_decay(freshest)
            recency_weighted_unique += tw(g) * decay
        # v41: Was 0.05 (=2.95 with 59 KOLs → 2 S-tier maxed consensus). 0.10 (=5.9)
        # requires real agreement: ~4 S-tier or ~12 A-tier to reach 1.0.
        kol_consensus = min(1.0, recency_weighted_unique / (total_kols * SCORING_PARAMS["consensus_norm_divisor"]))

        avg_sentiment = sum(data["sentiments"]) / len(data["sentiments"])
        sentiment_score = (avg_sentiment + 1) / 2

        # --- Dynamic conviction: hit_rate replaces static conviction ---
        # v11: Recency-weighted conviction — recent KOL mentions weigh more
        group_list = list(data["groups"])

        weighted_conv_sum = 0.0
        conv_weight_sum = 0.0
        kol_rep_values = []
        for g in group_list:
            # Temporal decay per KOL based on their freshest mention
            group_hours = data["hours_ago_by_group"].get(g, [])
            freshest = min(group_hours) if group_hours else 48
            decay = _two_phase_decay(freshest)

            kol_score = kol_scores.get(g)  # None if < min_calls data
            if kol_score is not None:
                # v15.3: kol_score is normalized (1.0 = avg). Map to conviction:
                # 0.1→5.2, 1.0→7.0 (matches unscored default), 3.0→10.0
                base_conv = 5.0 + min(5.0, kol_score * 2.0)
                kol_rep_values.append(min(1.0, kol_score / 3.0))
            else:
                # Fallback: static conviction until enough data
                base_conv = groups_conviction.get(g, 7)
                kol_rep_values.append(0.5)

            weighted_conv_sum += base_conv * decay
            conv_weight_sum += decay

        kol_reputation_avg = sum(kol_rep_values) / max(1, len(kol_rep_values))
        effective_conviction = weighted_conv_sum / max(0.01, conv_weight_sum)

        # Per-message NLP conviction amplifier/dampener (Sprint 6)
        msg_convictions = data.get("msg_conviction_scores", [])
        avg_msg_conv = sum(msg_convictions) / max(1, len(msg_convictions)) if msg_convictions else 1.0
        effective_conviction *= avg_msg_conv  # amplifies (>1.0) or dampens (<1.0)

        avg_conviction = sum(data["convictions"]) / len(data["convictions"])
        # v14: Compressed range — 7→0.25, 8→0.5, 10→1.0 (was 5-10→0-1, clustering at 0.4-1.0)
        # v47: offset/divisor/dampening from conviction_config (Optuna-tunable).
        _ccfg = SCORING_PARAMS["conviction_config"]
        conviction_score = max(0, min(1, (effective_conviction - _ccfg["norm_offset"]) / _ccfg["norm_divisor"]))

        # v15: Conviction dampening — single KOL shouldn't max conviction
        kol_count_factor = min(1.0, unique_kols / _ccfg["kol_dampening"])
        conviction_score *= kol_count_factor

        # v9: Recency-weighted breadth (KOL reputation * tier * count * freshness decay)
        # v41: Unified normalization (/10) for both paths. Was /8 primary, /12 fallback
        # — inconsistency penalized tokens without kol_mention_counts unfairly.
        kol_mention_counts = data.get("kol_mention_counts", {})
        if kol_mention_counts and kol_scores:
            weighted_mentions = 0
            for kol, count in kol_mention_counts.items():
                group_hours = data["hours_ago_by_group"].get(kol, [])
                freshest = min(group_hours) if group_hours else 24
                decay = _two_phase_decay(freshest)
                weighted_mentions += kol_scores.get(kol, 1.0) * tw(kol) * count * decay
            breadth_score = min(1.0, weighted_mentions / SCORING_PARAMS["breadth_norm_divisor"])
        else:
            # Fallback: decay total mentions by freshest overall mention
            freshest_overall = min(data["hours_ago"]) if data["hours_ago"] else 999
            decay = _two_phase_decay(freshest_overall)
            breadth_score = min(1.0, (data["mentions"] * decay) / SCORING_PARAMS["breadth_norm_divisor"])

        # --- Algorithm v3 A3: Sentiment consistency (low std = consensus) ---
        sentiment_consistency = 1.0
        if len(data["sentiments"]) >= 2:
            sentiment_std = float(np.std(data["sentiments"]))
            sentiment_consistency = max(0.1, 1.0 - sentiment_std)

        # --- Social velocity: how concentrated are mentions in recent time? ---
        recent_2h = sum(1 for h in data["hours_ago"] if h <= 2)
        recent_12h = sum(1 for h in data["hours_ago"] if h <= 12)
        social_velocity = recent_2h / max(1, recent_12h)  # 0-1, higher = heating up NOW

        # Mention acceleration: are mentions speeding up or dying?
        mid = hours / 2
        first_half = sum(1 for h in data["hours_ago"] if h > mid)
        second_half = sum(1 for h in data["hours_ago"] if h <= mid)
        mention_acceleration = (second_half - first_half) / max(1, first_half + second_half)
        # -1 (dying) to +1 (accelerating)

        # --- v35: Proxy signals for top predictors (kol_arrival_rate, mention_velocity) ---
        # kol_freshness: fraction of KOLs with their freshest mention <=2h (proxy for kol_arrival_rate)
        hours_by_group = data.get("hours_ago_by_group", {})
        if hours_by_group and unique_kols > 0:
            fresh_kols = sum(1 for g, hrs in hours_by_group.items() if hrs and min(hrs) <= 2)
            kol_freshness = fresh_kols / unique_kols
        else:
            kol_freshness = None

        # mention_heat_ratio: recent mention density vs older (proxy for mention_velocity)
        # v41: epsilon 0.1→1.0 (was exploding to 10*mentions when 2-6h=0), cap at 10.0
        mentions_1h = sum(1 for h in data["hours_ago"] if h <= 1)
        mentions_2h_6h = sum(1 for h in data["hours_ago"] if 2 < h <= 6)
        _mh_eps = SCORING_PARAMS["mention_heat_epsilon"]
        _mh_cap = SCORING_PARAMS["mention_heat_cap"]
        _mh_div = SCORING_PARAMS["mention_heat_divisor"]  # v47: was hardcoded 4
        mention_heat_ratio = min(_mh_cap, mentions_1h / (mentions_2h_6h / _mh_div + _mh_eps)) if data["hours_ago"] else None

        # --- v52: KOL timing alpha signals ---
        # Collect freshest mention per KOL group (= per KOL)
        all_freshest = []
        for g, hrs in hours_by_group.items():
            if hrs:
                all_freshest.append(min(hrs))  # freshest mention per KOL

        # 1. time_spread_minutes: gap between first and last KOL call
        if len(all_freshest) >= 2:
            time_spread_minutes = round((max(all_freshest) - min(all_freshest)) * 60, 1)
        else:
            time_spread_minutes = None

        # 2. first_call_age_minutes: age of the very first KOL call
        if all_freshest:
            first_call_age_minutes = round(max(all_freshest) * 60, 1)
        else:
            first_call_age_minutes = None

        # 3. kol_cascade_rate: max KOLs in any 30-min window
        if all_freshest and len(all_freshest) >= 2:
            min_h = min(all_freshest)
            buckets = defaultdict(int)
            for h in all_freshest:
                bucket_idx = int((h - min_h) * 60 / 30)
                buckets[bucket_idx] += 1
            kol_cascade_rate = max(buckets.values()) if buckets else 1
        else:
            kol_cascade_rate = 1 if all_freshest else None

        # 4. price_vs_first_call: current_price / price_at_first_call
        price_first = data.get("price_at_first_call")
        price_now = data.get("price_usd")
        if price_first and price_now and float(price_first) > 0:
            price_vs_first_call = round(float(price_now) / float(price_first), 3)
        else:
            price_vs_first_call = None

        # --- ML v2 Phase B: Social momentum phase classification ---
        if mention_acceleration > 0.2 and social_velocity > 0.3:
            social_momentum_phase = "building"
        elif mention_acceleration < -0.2 or social_velocity < 0.1:
            social_momentum_phase = "declining"
        else:
            social_momentum_phase = "plateau"

        # --- Base Telegram scores (3 modes) ---

        # Algorithm v7: Balanced score with weight renormalization
        # price_action will be None until OHLCV enrichment — renormalization handles this
        # by distributing its 40% weight across available components
        # Store _tw_func for renormalization to use
        _initial_token = {
            "unique_kols": unique_kols,
            "_total_kols": total_kols,
            "kol_tiers": {g: groups_tier.get(g, "A") for g in data["groups"]} if groups_tier else {},
            "_tw_func": tw,
            "sentiment": avg_sentiment,
            "avg_conviction": avg_conviction,
            "breadth_score": breadth_score,
            "price_action_score": None,  # not yet enriched
        }
        raw_score, data_conf = _compute_score_with_renormalization(_initial_token)
        score = min(100, max(0, int(raw_score * 100)))

        # Conviction mode: sustained discussion across many KOLs
        raw_conviction = (
            0.35 * kol_consensus
            + 0.30 * conviction_score
            + 0.10 * sentiment_score
            + 0.25 * breadth_score
        )
        score_conviction = min(100, max(0, int(raw_conviction * 100)))

        # v11: Recency score — max + sustained hybrid (freshest mention dominates)
        # Fixes: 20 old decayed mentions no longer beat 2 fresh ones
        recency_weights = [_two_phase_decay(h) for h in data["hours_ago"]]
        if recency_weights:
            max_recency = max(recency_weights)
            avg_top3 = sum(sorted(recency_weights, reverse=True)[:3]) / min(3, len(recency_weights))
            recency_score = 0.6 * max_recency + 0.4 * avg_top3
        else:
            recency_score = 0.0

        # Momentum mode: what's trending NOW — driven by recency + KOL quality
        raw_momentum = (
            0.50 * recency_score
            + 0.10 * sentiment_score
            + 0.30 * kol_consensus
            + 0.10 * breadth_score
        )
        score_momentum = min(100, max(0, int(raw_momentum * 100)))

        # v9: Freshest mention age for death detection
        freshest_mention_hours = min(data["hours_ago"]) if data["hours_ago"] else 999
        # v16: Oldest mention age for backtesting (when did the FIRST KOL call this token?)
        oldest_mention_hours = max(data["hours_ago"]) if data["hours_ago"] else 0

        trend = "up" if avg_sentiment > 0.15 else ("down" if avg_sentiment < -0.15 else "stable")

        groups_with_conv = sorted(
            [(g, groups_conviction.get(g, 7)) for g in data["groups"]],
            key=lambda x: x[1],
            reverse=True,
        )
        top_kols_list = [g[0] for g in groups_with_conv]  # ALL KOLs, not just top 5

        # v25: Aggregate message-level text features
        cts = data["call_type_scores"]
        call_type_score = sum(cts) / max(1, len(cts)) if cts else 0.0
        raw_texts = data["msg_texts_raw"]
        n_msgs = len(raw_texts)
        if n_msgs > 0:
            avg_msg_length = sum(data["msg_lengths"]) / n_msgs
            ca_mention_ratio = sum(data["msg_has_ca"]) / n_msgs
            multi_token_ratio = sum(1 for c in data["msg_tokens_count"] if c >= 2) / n_msgs
            total_caps = sum(sum(1 for c in t if c.isupper()) for t in raw_texts)
            total_chars = sum(len(t) for t in raw_texts)
            caps_ratio = total_caps / max(1, total_chars)
            total_emojis = sum(len(re.findall(r'[\U0001F300-\U0001FAFF]', t)) for t in raw_texts)
            emoji_density = total_emojis / max(1, total_chars)
            question_ratio = sum(1 for t in raw_texts if '?' in t) / n_msgs
            link_ratio = sum(1 for t in raw_texts if 'http' in t.lower()) / n_msgs
        else:
            avg_msg_length = 0.0
            ca_mention_ratio = 0.0
            multi_token_ratio = 0.0
            caps_ratio = 0.0
            emoji_density = 0.0
            question_ratio = 0.0
            link_ratio = 0.0

        ranking.append({
            "symbol": symbol,
            "score": score,
            "score_conviction": score_conviction,
            "score_momentum": score_momentum,
            "mentions": data["mentions"],
            "unique_kols": unique_kols,
            "sentiment": round(avg_sentiment, 3),
            "trend": trend,
            "change_24h": 0.0,
            "top_kols": top_kols_list,
            "avg_conviction": round(avg_conviction, 2),
            "recency_score": round(recency_score, 3),
            "social_velocity": round(social_velocity, 3),
            "mention_acceleration": round(mention_acceleration, 3),
            "_total_kols": total_kols,
            # Phase 2: KOL reputation
            "kol_reputation_avg": round(kol_reputation_avg, 3),
            "narrative": None,
            "narrative_is_hot": 0,
            "narrative_confidence": 0.0,
            # Algorithm v2: Per-message conviction NLP (Sprint 6)
            "msg_conviction_avg": round(avg_msg_conv, 2),
            "price_target_count": data.get("price_target_count", 0),
            "hedging_count": data.get("hedging_count", 0),
            # Algorithm v3 A3: Sentiment consistency
            "sentiment_consistency": round(sentiment_consistency, 3),
            # ML v2 Phase B: Social momentum phase
            "social_momentum_phase": social_momentum_phase,
            # v24: numeric encoding for ML (declining=0, plateau=1, building=2)
            "social_momentum_num": {"declining": 0, "plateau": 1, "building": 2}.get(social_momentum_phase, 1),
            # v9: Freshest mention age for death detection
            "freshest_mention_hours": round(freshest_mention_hours, 2),
            # v16: Oldest mention age for backtesting
            "oldest_mention_hours": round(oldest_mention_hours, 2),
            # Breadth score for upsert and price action recalculation
            "breadth_score": round(breadth_score, 3),
            # v9: Store recency-decayed consensus for _get_component_value
            "_decayed_consensus": round(kol_consensus, 4),
            # KOL tier info (for debugging/dashboard)
            "kol_tiers": {g: groups_tier.get(g, "A") for g in data["groups"]} if groups_tier else {},
            # v56: Best KOL win rate for this token
            "best_kol_win_rate": None,
            "best_kol_total_calls": None,
            # Algorithm v7: Weight renormalization
            "_tw_func": tw,
            "data_confidence": data_conf,
            # v11: Raw hours_ago list for activity ratio multiplier
            "_hours_ago": list(data["hours_ago"]),
            # v35: Proxy signals for top predictors
            "kol_freshness": round(kol_freshness, 3) if kol_freshness is not None else None,
            "mention_heat_ratio": round(mention_heat_ratio, 3) if mention_heat_ratio is not None else None,
            # v52: KOL timing alpha signals
            "time_spread_minutes": time_spread_minutes,
            "first_call_age_minutes": first_call_age_minutes,
            "kol_cascade_rate": kol_cascade_rate,
            "price_vs_first_call": price_vs_first_call,
            # v14: KOL-stated entry mcaps for entry premium calculation
            "kol_stated_entry_mcaps": data.get("kol_stated_entry_mcaps", []),
            # v16: Extraction source counts
            "ca_mention_count": data.get("ca_mention_count", 0),
            "ticker_mention_count": data.get("ticker_mention_count", 0),
            "url_mention_count": data.get("url_mention_count", 0),
            # v25: Message-level text features
            "call_type_score": round(call_type_score, 3),
            "avg_msg_length": round(avg_msg_length, 1),
            "ca_mention_ratio": round(ca_mention_ratio, 3),
            "caps_ratio": round(caps_ratio, 3),
            "emoji_density": round(emoji_density, 4),
            "multi_token_ratio": round(multi_token_ratio, 3),
            "question_ratio": round(question_ratio, 3),
            "link_ratio": round(link_ratio, 3),
            # v40: Best known CA from KOL mentions (most frequently referenced)
            "kol_resolved_ca": Counter(data["known_cas"]).most_common(1)[0][0] if data["known_cas"] else None,
        })

    # v56: Compute best KOL win rate per token from kol_win_rates
    if kol_win_rates:
        wr_cfg = SCORING_PARAMS.get("kol_wr_config", {"thresholds": [0.30, 0.40, 0.50], "factors": [1.0, 1.10, 1.25, 1.40], "min_calls": 3})
        wr_min_calls = wr_cfg.get("min_calls", 3)
        for t in ranking:
            kol_groups = t.get("kol_tiers", {})
            best_wr, best_calls = 0.0, 0
            for g in kol_groups:
                kwr = kol_win_rates.get(g)
                if kwr and kwr["total"] >= wr_min_calls and kwr["wr"] > best_wr:
                    best_wr = kwr["wr"]
                    best_calls = kwr["total"]
            if best_calls > 0:
                t["best_kol_win_rate"] = best_wr
                t["best_kol_total_calls"] = best_calls

    # Verify tokens exist on-chain via DexScreener (filters false positives)
    if ranking:
        all_symbols = [r["symbol"] for r in ranking]
        verified = verify_tokens_exist(all_symbols)
        before_count = len(ranking)
        ranking = [r for r in ranking if r["symbol"] in verified]
        filtered = before_count - len(ranking)
        if filtered > 0:
            logger.info("DexScreener filter removed %d fake tokens", filtered)

    # Enrich with on-chain data (DexScreener + RugCheck) — needed for gates
    enrich_tokens(ranking)

    # v33: all_enriched == ranking (no hard gates remove tokens anymore).
    # Kept as alias for backward compatibility with safe_scraper.py.
    all_enriched = ranking

    # === Quality gates BEFORE expensive enrichment ===
    # v58 AUDIT: ALL pre-enrichment gate penalties DISABLED — data proves gates
    # are anti-predictive (killed tokens = 18.4% hit rate vs penalized = 13.5%).
    # Gate reasons still TAGGED for ML features but gate_mult stays 1.0.
    addr_tagged = 0
    for t in ranking:
        if not t.get("token_address"):
            t["gate_reason"] = t.get("gate_reason") or "no_address"
            addr_tagged += 1
    if addr_tagged:
        logger.info("No-address tagged on %d tokens (no penalty — v58)", addr_tagged)

    data_tagged = 0
    for t in ranking:
        if (t.get("volume_24h") or 0) <= 0 and (t.get("liquidity_usd") or 0) <= 0:
            t["gate_reason"] = t.get("gate_reason") or "no_data"
            data_tagged += 1
    if data_tagged:
        logger.info("No-data tagged on %d tokens (no penalty — v58)", data_tagged)

    if hours >= 48:
        single_a_count = 0
        for t in ranking:
            tiers = t.get("kol_tiers", {})
            has_s_tier = any(tier == "S" for tier in tiers.values())
            if not has_s_tier and t.get("unique_kols", 0) < 2:
                t["gate_reason"] = t.get("gate_reason") or "single_a_tier"
                single_a_count += 1
        if single_a_count:
            logger.info("Single-A-tier tagged on %d tokens (window=%dh, no penalty — v58)", single_a_count, hours)

    # === Hard gates (uses RugCheck + DexScreener data) ===
    # v33: _apply_hard_gates is now all-soft (returns same list, no removals)
    ranking = _apply_hard_gates(ranking)

    # === Wash trading — compute score for ML features only, NO penalty ===
    # v34: REMOVED wash_trading gate penalty. Data shows 38.1% hit rate vs 9.9% for scored tokens.
    # Wash trading score kept as feature for ML model but no longer penalizes score.
    for token in ranking:
        token["wash_trading_score"] = round(_compute_wash_trading_score(token), 3)

    # v33: All gates are soft now — no more hard removal, ranking == all_enriched
    soft_penalized = sum(1 for t in ranking if t.get("gate_mult", 1.0) < 1.0)
    logger.info("Post-gate: %d tokens (%d soft-penalized) — now running expensive enrichment",
                len(ranking), soft_penalized)

    # === Expensive enrichment on ALL tokens (v33: no hard gates) ===
    # v34: Run all 4 enrichments in parallel — each writes different keys, no conflicts.
    # Helius → helius_*, Jupiter → jup_*, Bubblemaps → bubblemaps_*, OHLCV → candle_data/ath_*
    # Python GIL makes dict key assignment atomic; no shared keys between enrichers.
    logger.info("Starting parallel enrichment (Helius + Jupiter + Bubblemaps + OHLCV)")
    _enrich_start = time.time()
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="enrich") as pool:
        futures = {
            pool.submit(enrich_tokens_helius, ranking): "Helius",
            pool.submit(enrich_tokens_jupiter, ranking): "Jupiter",
            pool.submit(enrich_tokens_bubblemaps, ranking): "Bubblemaps",
            pool.submit(enrich_tokens_ohlcv, ranking): "OHLCV",
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                fut.result()
            except Exception as e:
                logger.error("%s enrichment failed: %s", name, e, exc_info=True)
    logger.info("Parallel enrichment completed in %.1fs", time.time() - _enrich_start)

    # === Algorithm v4: Price Action scoring ===
    for token in ranking:
        pa = compute_price_action_score(
            token,
            pa_norm_floor=SCORING_PARAMS["pa_norm_floor"],
            pa_norm_cap=SCORING_PARAMS["pa_norm_cap"],
            pa_config=SCORING_PARAMS.get("pa_config"),
        )
        token.update(pa)

        # v58 AUDIT: Removed _detect_volume_squeeze() and _compute_trend_strength()
        # Both computed but never used in scoring or ML. Correlation ~0 at N=29k.

        # Algorithm v7: Recalculate balanced score with renormalization
        # Now price_action_score is available (or still None if no OHLCV)
        new_raw, data_conf = _compute_score_with_renormalization(token)
        token["score"] = min(100, max(0, int(new_raw * 100)))
        token["data_confidence"] = data_conf

        # Algorithm v7: Weakest component + interpretation
        wk_name, wk_val = _identify_weakest_component(token)
        token["weakest_component"] = wk_name
        token["weakest_component_value"] = wk_val
        token["score_interpretation"] = _interpret_score(token["score"])

    # === Algorithm v3 C: ME2F-inspired ML features (computed from existing data) ===
    for token in ranking:
        # C1: Volatility proxy (multi-timeframe price dispersion from VDS concept)
        changes = [
            token.get("price_change_5m"),
            token.get("price_change_1h"),
            token.get("price_change_6h"),
            token.get("price_change_24h"),
        ]
        valid = [c for c in changes if c is not None]
        if len(valid) >= 2:
            token["volatility_proxy"] = round(float(np.std(valid)), 3)

        # C2: Whale dominance (concentration * inequality from WDS concept)
        top10 = token.get("top10_holder_pct")
        gini = token.get("helius_gini")
        if top10 is not None and gini is not None:
            token["whale_dominance"] = round((top10 / 100) * gini, 4)

        # C3: Sentiment amplification (sentiment volatility * price reaction from SAS concept)
        sent_std = 1.0 - token.get("sentiment_consistency", 1.0)
        pc24 = abs(token.get("price_change_24h") or 0)
        if sent_std > 0 and pc24 > 0:
            token["sentiment_amplification"] = round(sent_std * (pc24 / 100), 4)

    # v53: Compute avg_tx_size_usd (derived from existing DexScreener data)
    for token in ranking:
        vol_24h = token.get("volume_24h")
        txn_24h = token.get("txn_count_24h")
        if vol_24h and txn_24h and txn_24h > 0:
            token["avg_tx_size_usd"] = round(float(vol_24h) / int(txn_24h), 2)
        else:
            token["avg_tx_size_usd"] = None

    # v53: KOL co-occurrence detection
    token_kols, kol_tokens_map, cooc_matrix, total_tokens_cooc = _build_cooccurrence_matrix()
    for token in ranking:
        sym = token.get("symbol")
        kol_set = token_kols.get(sym, set()) if sym else set()
        cooc_avg, combo_novelty = _compute_kol_cooccurrence(sym or "", kol_set, kol_tokens_map, cooc_matrix, total_tokens_cooc)
        token["kol_cooccurrence_avg"] = cooc_avg
        token["kol_combo_novelty"] = combo_novelty

    # Algorithm v3 A5: Flag artificial pumps
    for token in ranking:
        token["is_artificial_pump"] = _detect_artificial_pump(token)
    pump_count = sum(1 for t in ranking if t["is_artificial_pump"])
    if pump_count:
        logger.info("Flagged %d tokens as artificial pumps", pump_count)

    # Apply on-chain multiplier + safety penalty to all scores
    for token in ranking:
        onchain_mult = _compute_onchain_multiplier(token)
        safety_pen = _compute_safety_penalty(token)
        token["onchain_multiplier"] = round(onchain_mult, 3)
        token["safety_penalty"] = round(safety_pen, 3)

        # Pump.fun graduation bonus: survived bonding curve = proven demand
        pump_bonus = 1.0
        if token.get("pump_graduation_status") == "graduated":
            pump_bonus = 1.1
        token["pump_graduated"] = 1 if token.get("pump_graduation_status") == "graduated" else 0

        # v9: Death/rug penalty (price collapse + social silence + volume death)
        death_pen = _detect_death_penalty(token, token.get("freshest_mention_hours", 999))
        token["death_penalty"] = round(death_pen, 3)

        # v23: Unified manipulation penalty (merges wash_pen + pump_pen)
        manipulation_pen = _compute_manipulation_penalty(token)
        token["wash_pen"] = round(manipulation_pen, 3)  # backward compat column name
        token["pump_pen"] = manipulation_pen  # backward compat

        # PVP penalty: same-name tokens — softer on pump.fun where copycats are inevitable
        # v47: floors/scales from pump_pen_config (Optuna-tunable).
        _ppcfg = SCORING_PARAMS["pump_pen_config"]
        pvp_recent = token.get("pvp_recent_count") or 0
        if token.get("is_pump_fun"):
            # pump.fun: copycats are normal, scraper already resolves to highest-volume pair
            pvp_pen = max(_ppcfg["pvp_pump_fun_floor"], 1.0 / (1 + _ppcfg["pvp_pump_fun_scale"] * pvp_recent))
        else:
            pvp_pen = max(_ppcfg["pvp_normal_floor"], 1.0 / (1 + _ppcfg["pvp_normal_scale"] * pvp_recent))

        # Algorithm v7: Minsky lifecycle phase classification
        # Replaces flat already_pumped_penalty with 5-phase model
        phase, phase_pen = _classify_lifecycle_phase(token)
        token["lifecycle_phase"] = phase
        # v24: numeric encoding for ML (panic=0 → boom=5)
        _LIFECYCLE_NUM = {"panic": 0, "profit_taking": 1, "euphoria": 2, "unknown": 3, "displacement": 4, "boom": 5}
        token["lifecycle_phase_num"] = _LIFECYCLE_NUM.get(phase, 3)
        token["already_pumped_penalty"] = round(phase_pen, 3)  # backward compat field name
        already_pumped_pen = phase_pen

        # v9: pa_mult REMOVED from multiplier chain — price_action already has 40% weight
        # in base score via renormalization. Applying pa_mult here double-counted it,
        # making price_action control ~60% instead of the intended 40%.

        # v16: Activity ratio — STRONGEST predictor (+0.212 correlation with 2x).
        # v20: bounds from SCORING_PARAMS (dynamic).
        # v44: thresholds from scoring_config (Optuna-tunable).
        act_floor = SCORING_PARAMS["activity_mult_floor"]  # default 0.80
        act_cap = SCORING_PARAMS["activity_mult_cap"]      # default 1.25
        act_high = SCORING_PARAMS["activity_ratio_high"]   # default 0.6
        act_mid = SCORING_PARAMS["activity_ratio_mid"]     # default 0.3
        act_low = SCORING_PARAMS["activity_ratio_low"]     # default 0.1
        recent_6h_count = sum(1 for h in token.get("_hours_ago", []) if h <= 6)
        total_mention_count = len(token.get("_hours_ago", []))
        if total_mention_count > 0:
            activity_ratio = recent_6h_count / total_mention_count
            # v44: Store raw ratio for Optuna re-scoring
            token["activity_ratio_raw"] = round(activity_ratio, 4)
            if activity_ratio > act_high:
                activity_mult = act_cap   # most mentions are fresh
            elif activity_ratio > act_mid:
                activity_mult = SCORING_PARAMS["activity_mid_mult"]  # v47: was hardcoded 1.10
            elif activity_ratio > act_low:
                activity_mult = 1.0       # neutral
            else:
                activity_mult = act_floor  # dead social activity
        else:
            activity_mult = 1.0
            token["activity_ratio_raw"] = None

        # v17: Activity_mult cap when token already pumped — buzz peaks during
        # pumps but that's exactly when 2x potential is LOWEST.
        # v44: thresholds from scoring_config.
        pump_cap_hard = SCORING_PARAMS["activity_pump_cap_hard"]  # default 80
        pump_cap_soft = SCORING_PARAMS["activity_pump_cap_soft"]  # default 50
        pc24_act = token.get("price_change_24h")
        if pc24_act is not None and pc24_act > pump_cap_hard:
            activity_mult = min(activity_mult, 1.0)   # No bonus if already pumped hard
        elif pc24_act is not None and pc24_act > pump_cap_soft:
            activity_mult = min(activity_mult, 1.10)  # Reduced bonus

        # v35/v42: Momentum multiplier — combines top 3 proxy signals.
        # v42 fix: zero = neutral (1.0), only boost tokens with genuine momentum.
        # v44: All breakpoints/factors from scoring_config (Optuna-tunable).
        mcfg = SCORING_PARAMS["momentum_config"]
        kf_thresh = mcfg.get("kol_fresh_thresholds", [0.5, 0.2])
        kf_factors = mcfg.get("kol_fresh_factors", [1.20, 1.10, 1.05])
        mhr_thresh = mcfg.get("mhr_thresholds", [2.0, 1.0, 0.3])
        mhr_factors = mcfg.get("mhr_factors", [1.15, 1.10, 1.05])
        sth_thresh = mcfg.get("sth_thresholds", [3.0, 1.5])
        sth_factors = mcfg.get("sth_factors", [1.10, 1.05])
        sth_pen_thresh = mcfg.get("sth_penalty_threshold", 0.3)
        sth_pen_factor = mcfg.get("sth_penalty_factor", 0.95)
        mom_floor = mcfg.get("floor", 0.70)
        mom_cap = mcfg.get("cap", 1.40)

        kol_fresh = token.get("kol_freshness")
        kol_fresh_factor = 1.0
        if kol_fresh is not None and kol_fresh > 0:
            if kol_fresh >= kf_thresh[0]:
                kol_fresh_factor = kf_factors[0]
            elif kol_fresh >= kf_thresh[1]:
                kol_fresh_factor = kf_factors[1]
            else:
                kol_fresh_factor = kf_factors[2]

        mhr = token.get("mention_heat_ratio")
        mention_heat_factor = 1.0
        if mhr is not None and mhr > 0:
            if mhr >= mhr_thresh[0]:
                mention_heat_factor = mhr_factors[0]
            elif mhr >= mhr_thresh[1]:
                mention_heat_factor = mhr_factors[1]
            elif mhr >= mhr_thresh[2]:
                mention_heat_factor = mhr_factors[2]

        sth = token.get("short_term_heat")
        vol_heat_factor = 1.0
        if sth is not None:
            if sth >= sth_thresh[0]:
                vol_heat_factor = sth_factors[0]
            elif sth >= sth_thresh[1]:
                vol_heat_factor = sth_factors[1]
            elif sth < sth_pen_thresh:
                vol_heat_factor = sth_pen_factor

        # v52: KOL cascade timing boost
        cascade = token.get("kol_cascade_rate")
        cascade_factor = 1.0
        if cascade is not None and cascade >= 3:
            cascade_factor = mcfg.get("cascade_factor_3plus", 1.15)
        elif cascade is not None and cascade >= 2:
            cascade_factor = mcfg.get("cascade_factor_2plus", 1.08)

        # v52: Early entry boost (first call age)
        first_age = token.get("first_call_age_minutes")
        early_factor = 1.0
        if first_age is not None:
            if first_age <= 30:
                early_factor = mcfg.get("early_factor_30min", 1.20)
            elif first_age <= 60:
                early_factor = mcfg.get("early_factor_60min", 1.10)
            elif first_age >= 360:
                early_factor = mcfg.get("late_penalty_360min", 0.85)

        # v52: Price already moved penalty
        pvfc = token.get("price_vs_first_call")
        pvfc_factor = 1.0
        if pvfc is not None:
            if pvfc >= 3.0:
                pvfc_factor = mcfg.get("pvfc_penalty_3x", 0.60)
            elif pvfc >= 2.0:
                pvfc_factor = mcfg.get("pvfc_penalty_2x", 0.75)
            elif pvfc >= 1.5:
                pvfc_factor = mcfg.get("pvfc_penalty_1_5x", 0.90)

        momentum_mult = max(mom_floor, min(mom_cap, kol_fresh_factor * mention_heat_factor * vol_heat_factor * cascade_factor * early_factor * pvfc_factor))
        token["momentum_mult"] = momentum_mult

        # v15: Breadth floor — low KOL count is a mild flag, not a death sentence
        # v44: thresholds/penalties from scoring_config (Optuna-tunable).
        bpcfg = SCORING_PARAMS["breadth_pen_config"]
        bp_thresh = bpcfg.get("thresholds", [0.033, 0.05, 0.08])
        bp_pens = bpcfg.get("penalties", [0.75, 0.85, 0.95])
        breadth_raw = float(token.get("breadth_score", 0) or 0)
        if breadth_raw < bp_thresh[0]:
            breadth_pen = bp_pens[0]
        elif breadth_raw < bp_thresh[1]:
            breadth_pen = bp_pens[1]
        elif breadth_raw < bp_thresh[2]:
            breadth_pen = bp_pens[2]
        else:
            breadth_pen = 1.0

        # v17: Pump momentum penalty — penalize tokens actively pumping RIGHT NOW
        # This is NOT a duplicate of price_action (which averages 4 sub-components).
        # This multiplier acts on the FINAL score to directly penalize active pumps.
        # v20: thresholds from SCORING_PARAMS (dynamic).
        # v47: penalty values from pump_pen_config (Optuna-tunable).
        pump_1h_hard = SCORING_PARAMS["pump_pc1h_hard"]  # default 30
        pump_5m_hard = SCORING_PARAMS["pump_pc5m_hard"]  # default 15
        pump_1h_mod = pump_1h_hard * 0.5   # 15 at default
        pump_5m_mod = pump_5m_hard * 0.533  # ~8 at default
        pump_1h_light = pump_1h_hard * 0.267  # ~8 at default
        pc_1h = token.get("price_change_1h")
        pc_5m = token.get("price_change_5m")
        pump_momentum_pen = 1.0
        if (pc_1h is not None and pc_1h > pump_1h_hard) or (pc_5m is not None and pc_5m > pump_5m_hard):
            pump_momentum_pen = _ppcfg["hard_penalty"]   # active pump
        elif (pc_1h is not None and pc_1h > pump_1h_mod) or (pc_5m is not None and pc_5m > pump_5m_mod):
            pump_momentum_pen = _ppcfg["moderate_penalty"]  # moderate pump
        elif pc_1h is not None and pc_1h > pump_1h_light:
            pump_momentum_pen = _ppcfg["light_penalty"]  # light pump
        token["pump_momentum_pen"] = pump_momentum_pen

        # v12: KOL entry premium — penalize tokens that pumped far above KOL call prices
        entry_premium, entry_premium_mult = _compute_kol_entry_premium(token)
        token["entry_premium"] = entry_premium
        token["entry_premium_mult"] = entry_premium_mult

        # v24: Entry drift multiplier — penalizes when price outpaced social growth
        entry_drift_mult = _compute_entry_drift_mult(token)
        token["entry_drift_mult"] = entry_drift_mult

        # Freshest mention hours (used by size_mult freshness tier)
        freshest_h = token.get("freshest_mention_hours", 0)

        # v15.2/v19: Size opportunity multiplier — smaller tokens need less $ to 2x.
        # v44: All thresholds/factors from scoring_config (Optuna-tunable).
        smcfg = SCORING_PARAMS["size_mult_config"]
        sm_mcap_thresh = smcfg.get("mcap_thresholds", [300_000, 1_000_000, 5_000_000, 20_000_000, 50_000_000, 200_000_000, 500_000_000])
        sm_mcap_factors = smcfg.get("mcap_factors", [1.3, 1.15, 1.0, 0.85, 0.70, 0.50, 0.35, 0.25])
        sm_fresh_thresh = smcfg.get("fresh_thresholds", [4, 12])
        sm_fresh_factors = smcfg.get("fresh_factors", [1.2, 1.1, 1.0])
        sm_large_cap = smcfg.get("large_cap_threshold", 50_000_000)
        sm_floor = smcfg.get("floor", 0.25)
        sm_cap = smcfg.get("cap", 1.5)
        t_mcap = token.get("market_cap") or 0
        if t_mcap <= 0:
            mcap_factor = 1.0  # no data = neutral
        else:
            mcap_factor = sm_mcap_factors[-1]  # fallback: largest tier
            for i, thresh in enumerate(sm_mcap_thresh):
                if t_mcap < thresh:
                    mcap_factor = sm_mcap_factors[i]
                    break
        # freshness tier: KOL just called = buy pressure incoming
        # No freshness bonus for large caps.
        if t_mcap >= sm_large_cap:
            fresh_factor = 1.0
        elif freshest_h < sm_fresh_thresh[0]:
            fresh_factor = sm_fresh_factors[0]
        elif freshest_h < sm_fresh_thresh[1]:
            fresh_factor = sm_fresh_factors[1]
        else:
            fresh_factor = sm_fresh_factors[2]
        size_mult = max(sm_floor, min(sm_cap, mcap_factor * fresh_factor))
        token["size_mult"] = round(size_mult, 3)

        # v15.3: S-tier KOL bonus — S-tier callers have proven track records
        # v44: bonus value from scoring_config (Optuna-tunable).
        kol_tiers = token.get("kol_tiers", {})
        s_tier_count = sum(1 for tier in kol_tiers.values() if tier == "S")
        s_tier_bonus = SCORING_PARAMS["s_tier_bonus"]  # default 1.2
        s_tier_mult = s_tier_bonus if s_tier_count > 0 else 1.0
        token["s_tier_mult"] = s_tier_mult

        # v55: CA mention boost — tokens with explicit CA have 25% HR vs 8.9% ticker-only
        ca_bonus = SCORING_PARAMS.get("ca_mention_bonus", 1.15)
        ca_mult = ca_bonus if (token.get("ca_mention_count", 0) or 0) > 0 or (token.get("url_mention_count", 0) or 0) > 0 else 1.0
        token["ca_mult"] = ca_mult

        # v56: KOL win rate multiplier — graduated boost for tokens called by high-WR KOLs
        wr_cfg = SCORING_PARAMS.get("kol_wr_config", {"thresholds": [0.30, 0.40, 0.50], "factors": [1.0, 1.10, 1.25, 1.40], "min_calls": 3})
        best_wr = token.get("best_kol_win_rate", 0) or 0
        best_calls = token.get("best_kol_total_calls", 0) or 0
        if best_calls >= wr_cfg.get("min_calls", 3):
            kol_wr_mult = _tier_lookup(best_wr, wr_cfg["thresholds"], wr_cfg["factors"])
        else:
            kol_wr_mult = 1.0
        token["kol_wr_mult"] = round(kol_wr_mult, 3)

        # v9+v12+v23: Use min(lifecycle, death, entry_premium, pump_momentum)
        # — no double-penalizing pump signals. pump_momentum_pen folded in here
        # instead of being a separate chain multiplier.
        crash_pen = min(already_pumped_pen, death_pen, entry_premium_mult, pump_momentum_pen)

        # Tuning platform: store ALL multiplier values for client-side re-scoring
        token["pump_bonus"] = pump_bonus
        token["pvp_pen"] = round(pvp_pen, 3)
        token["breadth_pen"] = breadth_pen
        token["activity_mult"] = activity_mult
        token["crash_pen"] = crash_pen
        token["_consensus_val"] = _get_component_value(token, "consensus")
        token["_sentiment_val"] = (token.get("sentiment", 0) + 1) / 2 if token.get("sentiment") is not None else None
        # v14: Use same compressed range as _get_component_value
        _ccfg2 = SCORING_PARAMS["conviction_config"]
        token["_conviction_val"] = max(0, min(1, (token.get("avg_conviction", _ccfg2["norm_offset"]) - _ccfg2["norm_offset"]) / _ccfg2["norm_divisor"])) if token.get("avg_conviction") is not None else None
        token["_breadth_val"] = _get_component_value(token, "breadth")
        token["_price_action_val"] = _get_component_value(token, "price_action")

        # v32: Hype penalty — more KOLs = WORSE outcomes (corr -0.132, N=1630)
        # v44: thresholds/penalties from scoring_config (Optuna-tunable).
        hpcfg = SCORING_PARAMS["hype_pen_config"]
        hp_thresh = hpcfg.get("thresholds", [2, 4, 7])
        hp_pens = hpcfg.get("penalties", [1.0, 0.85, 0.65, 0.50])
        uk_count = token.get("unique_kols") or 1
        if uk_count <= hp_thresh[0]:
            hype_pen = hp_pens[0]     # sweet spot — early discovery
        elif uk_count <= hp_thresh[1]:
            hype_pen = hp_pens[1]     # moderate hype
        elif uk_count <= hp_thresh[2]:
            hype_pen = hp_pens[2]     # crowded trade
        else:
            hype_pen = hp_pens[3]     # everyone knows — probably too late
        # v53: KOL co-occurrence penalty — shill ring detection
        cooc_cfg = hpcfg.get("cooc_config", {"threshold": 0.5, "penalty": 0.85})
        cooc = token.get("kol_cooccurrence_avg")
        if cooc is not None and cooc > cooc_cfg["threshold"]:
            hype_pen *= cooc_cfg["penalty"]
        token["hype_pen"] = hype_pen

        # v21: gate_mult — soft safety penalties (top10, risk, liquidity, holders, single_a_tier)
        # v27: Explicit default — tokens that bypass _apply_hard_gates get 1.0
        gate_mult = float(token.get("gate_mult", 1.0) or 1.0)

        # v56: Chain (16 multipliers). Added kol_wr_mult.
        # v23 removals: squeeze_mult (dead), trend_mult (dead),
        # wash_pen+pump_pen (merged → manipulation_pen),
        # pump_momentum_pen (folded into crash_pen min()).
        combined_raw = (onchain_mult * safety_pen * pump_bonus
                        * manipulation_pen * pvp_pen * crash_pen
                        * activity_mult * breadth_pen
                        * size_mult * s_tier_mult * ca_mult * kol_wr_mult * gate_mult
                        * entry_drift_mult * hype_pen * momentum_mult)
        # v16: Floor at 0.25 decompresses the 0-14 band where 97% of tokens stuck.
        # v17: Cap at 2.0 prevents multiplier stacking (activity*s_tier*size)
        # from inflating mediocre base scores beyond 100.
        combined = max(SCORING_PARAMS["combined_floor"], min(SCORING_PARAMS["combined_cap"], combined_raw))

        # Apply to all three scoring modes
        token["score"] = min(100, max(0, int(token["score"] * combined)))
        token["score_conviction"] = min(100, max(0, int(token["score_conviction"] * combined)))
        token["score_momentum"] = min(100, max(0, int(token["score_momentum"] * combined)))

        # Multi-indicator confirmation pillars (data only, no score penalty)
        # v58 AUDIT: Removed confirmation_pillars computation (dead since v15.2).
        # Gate was removed after data showed 0 pillars = 28.6% hit vs 3 pillars = 0%.
        # Value was only used for logging, never in scoring or ML.

        # Update interpretation band after final score
        token["score_interpretation"] = _interpret_score(token["score"])

        # ML v2 Phase C: Entry zone timing quality (0-1)
        # Measures how close the token is to an optimal buy window.
        # ML-only initially; may become a soft multiplier if correlation > 0.05.
        token["entry_timing_quality"] = _compute_entry_timing_quality(token)

    logger.info("Applied on-chain multipliers & safety penalties to %d tokens", len(ranking))

    # v44: Apply ML scoring based on scoring_mode from scoring_config
    scoring_mode = SCORING_PARAMS.get("scoring_mode", "formula")
    if scoring_mode in ("formula", "hybrid", "ml_primary"):
        _apply_ml_scores(ranking)
    # In "hybrid" mode, formula score is blended with ML score (handled in _apply_ml_scores)
    # In "ml_primary" mode, ML provides ranking, formula provides safety guardrails only
    # Note: "formula" mode = current behavior (ML as multiplier on formula score)

    # === v26: Market context features (regime + relative positioning) ===
    # Computed AFTER all individual scoring, BEFORE final sort.
    # Category A: Historical benchmarks from auto_backtest (via scoring_config)
    market_benchmarks = SCORING_PARAMS.get("market_benchmarks", {})
    ml_horizon = SCORING_PARAMS.get("ml_horizon", "12h")

    # Category B: Cycle-level aggregates (computed from current ranking)
    all_volumes = [t.get("volume_24h", 0) for t in ranking if t.get("volume_24h")]
    all_price_changes = [t.get("price_change_24h", 0) for t in ranking
                         if t.get("price_change_24h") is not None]
    cycle_median_volume = statistics.median(all_volumes) if all_volumes else 1
    cycle_median_price_change = statistics.median(all_price_changes) if all_price_changes else 0
    n_tokens_cycle = len(ranking)

    # Inject into each token
    median_peak = market_benchmarks.get(f"median_peak_return_{ml_horizon}", 1.5)
    win_rate_7d = market_benchmarks.get("win_rate_7d", 0.2)

    for token in ranking:
        # Cat A: Historical benchmarks
        pc24 = token.get("price_change_24h")
        current_return = (pc24 / 100 + 1) if pc24 is not None else 1.0
        entry_vs_peak = current_return / median_peak if median_peak > 1 else 0.5

        token["median_peak_return"] = round(median_peak, 4)
        token["entry_vs_median_peak"] = round(min(entry_vs_peak, 5.0), 4)  # cap at 5x
        token["win_rate_7d"] = round(win_rate_7d, 4)

        # Cat B: Cycle-level features
        token["market_heat_24h"] = round(cycle_median_price_change, 2)
        vol = token.get("volume_24h") or 0
        token["relative_volume"] = round(vol / max(1, cycle_median_volume), 4)
        token["kol_saturation"] = n_tokens_cycle

    logger.info(
        "v26 market context: median_peak=%.2fx, wr_7d=%.1f%%, heat=%.1f%%, "
        "med_vol=$%.0f, n_tokens=%d",
        median_peak, win_rate_7d * 100, cycle_median_price_change,
        cycle_median_volume, n_tokens_cycle,
    )

    ranking.sort(key=lambda x: (x["score"], x["mentions"]), reverse=True)
    return ranking, raw_kol_mentions, all_enriched
