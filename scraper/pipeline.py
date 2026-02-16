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
from datetime import datetime, timezone, timedelta
from collections import defaultdict
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
_ml_loaded = False         # Prevent repeated load attempts

# Quality gate: refuse to load models below this threshold
# v22: relaxed from 0.40 — lower return thresholds (e.g. +50%) are easier to predict
_MIN_PRECISION_AT_5 = 0.30


def _load_ml_model(horizon: str = "12h"):
    """
    Lazy-load ML ensemble (XGBoost + LightGBM) if available and quality gate passes.
    Supports both regression and classification models based on metadata 'mode' field.
    Returns (xgb_model, lgb_model, features, meta) or (None, None, None, None).
    """
    global _ml_model, _ml_lgb_model, _ml_features, _ml_meta, _ml_loaded
    if _ml_loaded:
        return _ml_model, _ml_lgb_model, _ml_features, _ml_meta

    _ml_loaded = True  # Don't retry on failure

    xgb_path = _ML_MODEL_DIR / f"model_{horizon}.json"
    lgb_path = _ML_MODEL_DIR / f"model_{horizon}_lgb.txt"
    meta_path = _ML_MODEL_DIR / f"model_{horizon}_meta.json"

    if not xgb_path.exists():
        return None, None, None, None

    # Load metadata first — check quality gate BEFORE loading models
    if not meta_path.exists():
        logger.warning("ML metadata not found at %s — refusing to load model without quality gate", meta_path)
        return None, None, None, None

    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except Exception as e:
        logger.warning("Failed to read ML metadata: %s", e)
        return None, None, None, None

    # Quality gate check
    p_at_5 = meta.get("metrics", {}).get("precision_at_5", 0)
    n_test = meta.get("metrics", {}).get("n_test", meta.get("n_test", 0))
    if meta.get("quality_gate") != "PASSED" or p_at_5 < _MIN_PRECISION_AT_5:
        logger.warning(
            "ML quality gate REJECTED: precision@5=%.3f (need >=%.2f), gate=%s. Using manual scores only.",
            p_at_5, _MIN_PRECISION_AT_5, meta.get("quality_gate", "UNKNOWN"),
        )
        return None, None, None, None

    # Refuse models trained on tiny test sets — statistically meaningless
    if n_test < 200:
        logger.warning(
            "ML DISABLED: model trained on only %d test samples (need >=200). "
            "Collect more data before trusting ML scores.",
            n_test,
        )
        return None, None, None, None

    mode = meta.get("mode", "classification")
    features = meta.get("features", [])
    if not features:
        logger.warning("ML metadata has empty feature list — refusing to load")
        return None, None, None, None

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
        return None, None, None, None

    _ml_features = features
    _ml_meta = meta
    logger.info(
        "ML ensemble loaded: mode=%s, %d features, precision@5=%.3f, ensemble_weights=%s",
        mode, len(features), p_at_5, meta.get("ensemble_weights", {}),
    )
    return _ml_model, _ml_lgb_model, _ml_features, _ml_meta


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
    xgb_model, lgb_model, features, meta = _load_ml_model(horizon=ml_horizon)
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

            # Convert to percentile ranks (0-1), then scale to [0.5, 1.5]
            from scipy.stats import rankdata
            ranks = rankdata(preds) / len(preds)  # 0..1 percentile
            ml_multipliers = 0.5 + ranks  # [0.5, 1.5]

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

            # Scale probability to multiplier: p=0 → 0.5x, p=0.5 → 1.0x, p=1 → 1.5x
            ml_multipliers = 0.5 + np.clip(proba, 0, 1)

        # Apply multiplier to all score variants
        for token, ml_mult in zip(ranking, ml_multipliers):
            ml_mult = float(np.clip(ml_mult, 0.5, 1.5))
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
) -> list[tuple[str, str]]:
    """
    Extract token symbols from text via three methods:
    1) $TOKEN prefix — high confidence, always extracted
    2) Bare ALL-CAPS words (3+ chars) — only if symbol is confirmed by $ or CA elsewhere
    3) Solana contract addresses → resolved to symbols via DexScreener

    Returns list of (symbol, source) tuples where source is "ticker", "ca", or "url".

    Parameters
    ----------
    confirmed_symbols : set of uppercase symbol names (without $) that have been
        confirmed via $-prefix or CA resolution elsewhere in this scrape cycle.
        When provided, bare ALLCAPS words are only extracted if they appear in this set
        AND are not in BARE_WORD_SUSPECTS. When None (backward compat), bare ALLCAPS
        extraction is unrestricted (legacy behavior).
    """
    tokens: list[tuple[str, str]] = []
    seen: set[str] = set()

    # 1) $-prefixed tokens: always extract (KOL intentionally naming a token)
    for match in TOKEN_REGEX.findall(text):
        upper = match.upper()
        if not upper.isascii():
            continue
        symbol = f"${upper}"
        if upper not in EXCLUDED_TOKENS and symbol not in seen:
            tokens.append((symbol, "ticker"))
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
                    tokens.append((symbol, "ca"))
                    seen.add(symbol)

    # 4) DexScreener/pump.fun URLs → resolve pair/token address to symbol
    if ca_cache is not None:
        # DexScreener: pair address → resolve via pairs API
        for chain, pair_addr in DEXSCREENER_URL_REGEX.findall(text):
            resolved = _resolve_pair_to_symbol(chain, pair_addr, ca_cache)
            if resolved:
                symbol = f"${resolved}"
                if resolved not in EXCLUDED_TOKENS and symbol not in seen:
                    tokens.append((symbol, "url"))
                    seen.add(symbol)

        # pump.fun: token CA directly → resolve via tokens API
        for pump_addr in PUMP_FUN_URL_REGEX.findall(text):
            resolved = _resolve_ca_to_symbol(pump_addr, ca_cache)
            if resolved:
                symbol = f"${resolved}"
                if resolved not in EXCLUDED_TOKENS and symbol not in seen:
                    tokens.append((symbol, "url"))
                    seen.add(symbol)

        # GMGN: token CA directly (same as pump.fun — CA in URL)
        for gmgn_addr in GMGN_URL_REGEX.findall(text):
            resolved = _resolve_ca_to_symbol(gmgn_addr, ca_cache)
            if resolved:
                symbol = f"${resolved}"
                if resolved not in EXCLUDED_TOKENS and symbol not in seen:
                    tokens.append((symbol, "url"))
                    seen.add(symbol)

        # Photon-sol: LP pair address → resolve via pairs API (same as DexScreener)
        for photon_addr in PHOTON_URL_REGEX.findall(text):
            resolved = _resolve_pair_to_symbol("solana", photon_addr, ca_cache)
            if resolved:
                symbol = f"${resolved}"
                if resolved not in EXCLUDED_TOKENS and symbol not in seen:
                    tokens.append((symbol, "url"))
                    seen.add(symbol)

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
    Unified manipulation detection: wash trading + artificial pump.
    Returns penalty multiplier in [0.3, 1.0].
    Takes the HARSHEST of wash trading and artificial pump signals.
    """
    # Wash trading signal (from _compute_wash_trading_score)
    wash_score = token.get("wash_trading_score", 0)
    wash_pen = max(0.3, 1.0 - wash_score)

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

    return min(wash_pen, pump_pen)  # harshest wins


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

    # 2. Price position (not too pumped, not in crash)
    pc24 = token.get("price_change_24h")
    if pc24 is not None:
        if -30 < pc24 < 50:
            signals.append(0.9)       # pre-pump zone
        elif 50 <= pc24 < 100:
            signals.append(0.5)       # moderate pump
        elif pc24 >= 200:
            signals.append(0.1)       # too late
        else:
            signals.append(0.4)       # crashing
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


def _compute_kol_entry_premium(token: dict) -> tuple[float, float]:
    """
    Compute how much the price has moved since KOLs called the token.
    Returns (entry_premium, entry_premium_mult) where:
    - entry_premium = current_price / decay-weighted avg entry price
    - entry_premium_mult = penalty multiplier [0.25, 1.1]

    ONLY uses real OHLCV candle data — no interpolation, no estimation.
    If no candle data exists for a token, returns neutral (1.0, 1.0).
    Scales penalty by how long the pump has been running (duration factor).
    """
    current_price = token.get("price_usd")
    current_mcap = token.get("market_cap") or token.get("fdv")
    hours_ago_list = token.get("_hours_ago", [])

    # v14: Primary source — KOL-stated entry mcap (more accurate than OHLCV interpolation)
    stated_mcaps = token.get("kol_stated_entry_mcaps", [])
    if stated_mcaps and current_mcap and current_mcap > 0:
        avg_stated = sum(stated_mcaps) / len(stated_mcaps)
        if avg_stated > 0:
            entry_premium = current_mcap / avg_stated
            if entry_premium < 1.0:
                mult = 1.1
            elif entry_premium <= 1.2:
                mult = 1.0
            elif entry_premium <= 2.0:
                mult = 0.9
            elif entry_premium <= 4.0:
                mult = 0.9 - (entry_premium - 2.0) * (0.2 / 2.0)
            elif entry_premium <= 8.0:
                mult = 0.7 - (entry_premium - 4.0) * (0.2 / 4.0)
            elif entry_premium <= 20.0:
                mult = 0.5 - (entry_premium - 8.0) * (0.15 / 12.0)
            else:
                mult = 0.25
            return round(entry_premium, 3), round(mult, 3)

    # Fallback 2: OHLCV candle data
    # Need both price and candle data — no candles = try mcap fallback
    candle_result = None
    if current_price and current_price > 0 and hours_ago_list and token.get("candle_data"):
        # Only consider mentions > 5 min old (too fresh = no meaningful price diff)
        valid_hours = [h for h in hours_ago_list if h > 5 / 60]
        if valid_hours:
            # Compute decay-weighted average entry price from REAL candle data
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

                # Duration scaling: longer pumps = more dangerous
                oldest_mention = max(valid_hours)
                if oldest_mention > 48:
                    duration_factor = 1.5
                elif oldest_mention > 24:
                    duration_factor = 1.3
                elif oldest_mention > 12:
                    duration_factor = 1.15
                else:
                    duration_factor = 1.0

                effective_premium = entry_premium ** duration_factor

                # Tier-based multiplier
                if effective_premium < 1.0:
                    mult = 1.1  # dip buy — mild bonus
                elif effective_premium <= 1.2:
                    mult = 1.0  # buying at KOL price — neutral
                elif effective_premium <= 2.0:
                    mult = 0.9  # slight premium
                elif effective_premium <= 4.0:
                    mult = 0.9 - (effective_premium - 2.0) * (0.2 / 2.0)
                elif effective_premium <= 8.0:
                    mult = 0.7 - (effective_premium - 4.0) * (0.2 / 4.0)
                elif effective_premium <= 20.0:
                    mult = 0.5 - (effective_premium - 8.0) * (0.15 / 12.0)
                else:
                    mult = 0.25
                candle_result = (round(entry_premium, 3), round(mult, 3))

    if candle_result is not None:
        return candle_result

    # v19 Fallback 3: Market cap magnitude — for established tokens where
    # KOLs don't state entry mcap AND candles don't show meaningful diff.
    # A $700M memecoin has already pumped 700x+ from launch ($1M typical).
    # Even if KOLs call it today at current price, the 2x potential is near zero.
    if current_mcap and current_mcap > 50_000_000:
        implied_premium = current_mcap / 1_000_000  # vs typical $1M launch
        if implied_premium <= 50:
            mult = 0.85   # $50M — established, limited upside
        elif implied_premium <= 200:
            mult = 0.70   # $200M — very limited
        elif implied_premium <= 500:
            mult = 0.50   # $500M — extremely unlikely to 2x
        else:
            mult = 0.35   # >$500M — near impossible
        return round(implied_premium, 3), round(mult, 3)

    return 1.0, 1.0


def _compute_entry_drift_mult(token: dict) -> float:
    """
    v24: Penalizes when price outpaced social growth since first KOL calls.
    Continuous multiplier in [0.5, 1.0].

    entry_premium = how much price drifted up since KOL calls.
    social_strength = how much new interest justifies it.
    When social keeps pace -> no penalty.
    When price runs ahead -> progressive penalty.
    """
    entry_premium = token.get("entry_premium") or 1.0
    if entry_premium <= 1.2:
        return 1.0

    # Social strength [0, 1]
    unique_kols = token.get("unique_kols") or 1
    kol_score = min(1.0, (unique_kols - 1) / 7)  # 1 KOL=0, 8+=1.0

    activity_mult = token.get("activity_mult") or 1.0
    activity_score = max(0, min(1.0, (activity_mult - 0.80) / 0.45))

    freshest_h = token.get("freshest_mention_hours") or 999
    if freshest_h <= 1:
        fresh_score = 1.0
    elif freshest_h <= 4:
        fresh_score = 0.7
    elif freshest_h <= 8:
        fresh_score = 0.4
    else:
        fresh_score = 0.1

    s_count = sum(1 for t in token.get("kol_tiers", {}).values() if t == "S")
    s_tier_score = min(1.0, s_count / 2)

    social_strength = (kol_score * 0.30 + activity_score * 0.25
                       + fresh_score * 0.30 + s_tier_score * 0.15)

    # Net drift = unjustified price increase
    drift = entry_premium - 1.0
    net_drift = max(0, drift - social_strength * 1.0)
    mult = max(0.50, 1.0 - net_drift * 0.25)
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
    v11: Detect dead/rugged tokens. Returns penalty multiplier [0.1, 1.0].
    Combines price collapse + volume death + social abandonment.
    Takes the MINIMUM (most pessimistic) of all death signals to catch tokens
    rugged 3+ days ago where 24h price change is near 0 but volume is dead.
    """
    penalties = []

    # --- Signal 1: Price collapse (original v9 logic) ---
    # v20: thresholds from SCORING_PARAMS (dynamic)
    death_severe = SCORING_PARAMS["death_pc24_severe"]   # default -80
    death_moderate = SCORING_PARAMS["death_pc24_moderate"]  # default -50
    pc24 = token.get("price_change_24h")
    if pc24 is not None:
        pc24 = float(pc24)

        if pc24 < death_severe:
            penalties.append(0.1)
        elif pc24 < (death_severe + 10):  # -70 when severe=-80
            penalties.append(0.15 if freshest_mention_hours > 3 else 0.3)
        elif pc24 < death_moderate:
            va = token.get("volume_acceleration")
            volume_alive = va is not None and float(va) > 0.5
            social_alive = freshest_mention_hours < 6
            if volume_alive and social_alive:
                penalties.append(0.6)
            elif volume_alive or social_alive:
                penalties.append(0.4)
            else:
                penalties.append(0.2)
        elif pc24 < -30:
            if freshest_mention_hours > 24:
                penalties.append(0.4)
            else:
                penalties.append(0.8)

    # --- Signal 2: Volume death (catches tokens rugged days ago with stable price) ---
    vol_24h = token.get("volume_24h")
    vol_1h = token.get("volume_1h")
    if vol_24h is not None and vol_1h is not None:
        if vol_24h < 5000 and vol_1h < 500:
            penalties.append(0.15)  # practically dead volume
        elif vol_24h < 1000:
            penalties.append(0.1)   # absolute volume floor — no trading happening

    # --- Signal 3 (v15): Social staleness — volume-modulated ---
    # User guidance: "after 12h without mentions, start losing, but mostly look at PA + volume"
    # Time-based decay that's SOFTENED by healthy volume (token still actively traded).
    if freshest_mention_hours > 12:
        if freshest_mention_hours > 72:
            stale_base = 0.15    # 3+ days = very stale
        elif freshest_mention_hours > 48:
            stale_base = 0.25    # 2+ days = stale
        elif freshest_mention_hours > 24:
            stale_base = 0.45    # 1+ day = aging
        else:
            stale_base = 0.7     # 12-24h = mildly stale

        # Volume modulator: healthy trading proves the token is still alive
        # If no volume data, use stale_base without modulation (no false softening)
        vol_24h = token.get("volume_24h")
        if vol_24h is None:
            stale_pen = stale_base
        elif vol_24h > 1_000_000:
            stale_pen = min(0.95, stale_base + 0.45)   # massive volume = barely penalized
        elif vol_24h > 500_000:
            stale_pen = min(0.9, stale_base + 0.35)
        elif vol_24h > 100_000:
            stale_pen = min(0.85, stale_base + 0.25)
        elif vol_24h > 50_000:
            stale_pen = min(0.8, stale_base + 0.15)
        else:
            stale_pen = stale_base

        penalties.append(stale_pen)

    # Return the most pessimistic signal (lowest penalty = harshest)
    if not penalties:
        return 1.0
    return min(penalties)


def _apply_hard_gates(ranking: list[dict]) -> list[dict]:
    """
    v21: Mostly soft penalties — only mint/freeze remain as hard gates.
    All other safety checks (top10, risk, liquidity, holders) are converted to
    gate_mult penalties so tokens stay in the pipeline, get labeled by
    outcome_tracker, and we can empirically validate whether these signals help.

    Hard gates (removed):
      - mint_authority: can inflate supply at will → genuine rug
      - freeze_authority: can freeze your tokens → genuine rug

    Soft penalties (kept, gate_mult applied):
      - top10_concentration > 70%: 0.7x (anti-predictive per v15.3 data)
      - risk_score > 8000: 0.6x
      - low_liquidity < 10K: 0.5x (genuine trading concern)
      - low_holders < 30: 0.7x
    """
    gate_top10 = SCORING_PARAMS["gate_top10_pct"]
    gate_liq = SCORING_PARAMS["gate_min_liquidity"]
    gate_holders = int(SCORING_PARAMS["gate_min_holders"])

    passed = []
    hard_gated = 0
    soft_penalized = 0
    for token in ranking:
        # === HARD GATES (genuinely dangerous — removed from pipeline) ===
        # mint_authority = can inflate supply at will
        if token.get("has_mint_authority"):
            token["gate_reason"] = "mint_authority"
            hard_gated += 1
            continue
        # freeze_authority = can freeze your tokens
        if token.get("has_freeze_authority"):
            token["gate_reason"] = "freeze_authority"
            hard_gated += 1
            continue

        # === SOFT PENALTIES (gate_mult — token stays, score reduced) ===
        gate_mult = 1.0
        gate_reasons = []

        # top10 holders own > gate_top10_pct% = concentration risk
        top10 = token.get("top10_holder_pct")
        if top10 is not None and top10 > gate_top10:
            gate_mult = min(gate_mult, 0.7)
            gate_reasons.append("top10_concentration")

        # RugCheck risk > 8000 (out of 10000)
        risk = token.get("risk_score")
        if risk is not None and risk > 8000:
            gate_mult = min(gate_mult, 0.6)
            gate_reasons.append("high_risk_score")

        # Liquidity floor
        liq = token.get("liquidity_usd")
        if liq is not None and liq < gate_liq:
            gate_mult = min(gate_mult, 0.5)
            gate_reasons.append("low_liquidity")

        # Holder floor (need real organic community)
        hcount = token.get("helius_holder_count") or token.get("holder_count")
        if hcount is not None and hcount < gate_holders:
            gate_mult = min(gate_mult, 0.7)
            gate_reasons.append("low_holders")

        token["gate_mult"] = round(gate_mult, 3)
        if gate_reasons:
            token["gate_reason"] = gate_reasons[0]  # primary reason for diagnostics
            soft_penalized += 1
        passed.append(token)

    if hard_gated:
        logger.info("Hard gates removed %d tokens (mint/freeze only)", hard_gated)
    if soft_penalized:
        logger.info("Soft gate penalties applied to %d tokens (top10>%.0f%%/risk>8000/liq<%.0f/holders<%d)",
                     soft_penalized, gate_top10, gate_liq, gate_holders)
    return passed


def _detect_volume_squeeze(token: dict) -> tuple[str, float]:
    """
    Detect volume squeeze/fire pattern (adapted from Harvard BB Squeeze).
    When volume compresses (6h avg << 24h avg) then explodes (1h >> 6h avg) = breakout.
    Returns (state, squeeze_score) where state is 'squeezing', 'firing', 'none'.
    squeeze_score: 0.0 (no signal) to 1.0 (strong squeeze fire).
    """
    vol_1h = token.get("volume_1h", 0) or 0
    vol_6h = token.get("volume_6h", 0) or 0
    vol_24h = token.get("volume_24h", 0) or 0

    if not vol_6h or not vol_24h:
        return "none", 0.0

    avg_hourly_6h = vol_6h / 6
    avg_hourly_24h = vol_24h / 24

    if avg_hourly_24h <= 0:
        return "none", 0.0

    # Compression: 6h hourly average much lower than 24h hourly average
    compression_ratio = avg_hourly_6h / avg_hourly_24h

    if compression_ratio < 0.5:  # Volume was compressed
        if vol_1h > 0 and avg_hourly_6h > 0:
            expansion = vol_1h / avg_hourly_6h
            if expansion > 2.0:  # Breakout: 1h volume explodes past compressed average
                return "firing", min(1.0, expansion / 5.0)
        return "squeezing", 0.3

    return "none", 0.0


def _compute_trend_strength(token: dict) -> float:
    """
    ADX-like trend strength from available price data (adapted from Harvard ADX).
    Measures directional conviction across timeframes.
    Returns 0.0 (no trend / conflicting signals) to 1.0 (very strong directional move).
    """
    pc1h = token.get("price_change_1h")
    pc6h = token.get("price_change_6h")
    pc24h = token.get("price_change_24h")

    # Only use non-None values for direction analysis
    valid = [(pc, tf) for pc, tf in [(pc1h, "1h"), (pc6h, "6h"), (pc24h, "24h")] if pc is not None]
    if not valid:
        return 0.0

    directions = [1 if pc > 0 else -1 for pc, _ in valid]

    # All available timeframes agree in direction = strong trend
    agreement = len(set(directions)) == 1

    # Magnitude: use longest available timeframe, clamped to 100%
    longest_pc = valid[-1][0]  # last entry = longest timeframe
    magnitude = min(abs(longest_pc) / 100, 1.0)

    if agreement:
        return min(1.0, 0.5 + magnitude * 0.5)
    else:
        return magnitude * 0.3  # Conflicting signals = weak trend


def _compute_onchain_multiplier(token: dict) -> float:
    """
    Returns a multiplier [0.3, 1.5] based on on-chain health.
    Applied to the base Telegram score. Neutral (1.0) if no data available.
    """
    factors = []

    # 1. Volume/MCap ratio (healthy > 0.1, great > 0.5)
    vmr = token.get("volume_mcap_ratio")
    if vmr is not None:
        factors.append(min(1.5, 0.5 + vmr * 2))

    # 2. Buy pressure — REMOVED (v10: already in price_action.py direction_mult + vol_confirm)

    # 3. Liquidity adequacy (liq/mcap > 0.05 = healthy)
    lmr = token.get("liq_mcap_ratio")
    if lmr is not None:
        if lmr < 0.02:
            factors.append(0.5)
        elif lmr > 0.10:
            factors.append(1.2)
        else:
            factors.append(0.8 + lmr * 4)

    # 4. Volume acceleration — REMOVED (v10: already in price_action.py vol_confirm)

    # 5. Token age penalty (v4: 6-48h optimal window)
    age = token.get("token_age_hours")
    if age is not None:
        if age < 1:
            factors.append(0.5)     # Too fresh, no data
        elif age < 6:
            factors.append(1.0)     # Young, unproven
        elif age < 48:
            factors.append(1.2)     # Sweet spot: survived early hours
        elif age < 168:
            factors.append(1.0)     # Established
        else:
            factors.append(0.8)     # Old, less likely to 2x

    # 6. Phase 3: Helius recent transaction activity
    recent_tx = token.get("helius_recent_tx_count")
    if recent_tx is not None:
        if recent_tx > 40:
            factors.append(1.3)
        elif recent_tx > 20:
            factors.append(1.1)
        elif recent_tx < 5:
            factors.append(0.7)

    # 7. Phase 3: Helius on-chain buy/sell ratio (complement DexScreener)
    helius_bsr = token.get("helius_onchain_bsr")
    if helius_bsr is not None:
        factors.append(0.5 + helius_bsr)  # 0.5-1.5

    # 8. Phase 3B: Jupiter tradeability + liquidity depth
    jup_tradeable = token.get("jup_tradeable")
    jup_impact = token.get("jup_price_impact_1k")
    if jup_tradeable is not None:
        if jup_tradeable == 0:
            factors.append(0.5)  # not tradeable = less liquid
        elif jup_impact is not None:
            if jup_impact < 1.0:
                factors.append(1.3)   # deep liquidity
            elif jup_impact < 5.0:
                factors.append(1.0)   # normal
            else:
                factors.append(0.7)   # thin liquidity

    # 9. Phase 3B: Whale accumulation signal
    whale_change = token.get("whale_change")
    if whale_change is not None:
        if whale_change > 5.0:
            factors.append(1.3)
        elif whale_change > 0:
            factors.append(1.1)
        elif whale_change < -10.0:
            factors.append(0.6)
        elif whale_change < 0:
            factors.append(0.8)

    # 10. Short-term volume heat — compute for data but NOT in factors
    # v10: volume already counted in price_action.py vol_confirm + _detect_volume_squeeze
    vol_1h = token.get("volume_1h")
    vol_6h = token.get("volume_6h")
    if vol_1h and vol_6h and vol_6h > 0:
        short_heat = (vol_1h * 6) / vol_6h
        token["short_term_heat"] = round(short_heat, 3)

    vol_5m = token.get("volume_5m")
    if vol_5m and vol_1h and vol_1h > 0:
        ultra_heat = (vol_5m * 12) / vol_1h
        token["ultra_short_heat"] = round(ultra_heat, 3)

    # 11. Algorithm v3 A2: Transaction velocity (txn/holder — activity density)
    txn_count = token.get("txn_count_24h")
    holders = token.get("helius_holder_count") or token.get("holder_count")
    if txn_count and holders and holders > 0:
        velocity = txn_count / holders
        token["txn_velocity"] = round(velocity, 3)
        if velocity > 5.0:
            factors.append(1.3)    # very active
        elif velocity > 1.0:
            factors.append(1.1)    # healthy
        elif velocity < 0.2:
            factors.append(0.6)    # stagnant
    elif txn_count and txn_count > 0:
        # Fallback: absolute txn density (no holder normalization)
        # Thresholds calibrated on DexScreener 24h txn counts for Solana memecoins
        token["txn_velocity"] = None  # Can't compute real velocity without holders
        if txn_count > 5000:
            factors.append(1.2)    # very active trading
        elif txn_count > 1000:
            factors.append(1.05)   # moderate activity
        elif txn_count < 100:
            factors.append(0.7)    # very low activity

    # Algorithm v4: Volatility proxy penalty (high volatility = unstable)
    vol_proxy = token.get("volatility_proxy")
    if vol_proxy is not None and vol_proxy > 50:
        factors.append(0.8)

    # Algorithm v4 Sprint 4: Whale direction accumulation bonus
    whale_dir = token.get("whale_direction")
    if whale_dir == "accumulating":
        factors.append(1.15)  # Whales consistently buying = bullish

    # v13: New whale entries — smart money entering (guide principle #8)
    wne = token.get("whale_new_entries")
    if wne is not None:
        if wne >= 3:
            factors.append(1.25)  # multiple new whales = strong signal
        elif wne >= 1:
            factors.append(1.1)   # at least 1 new whale

    # v13: Unique wallet growth — "who buys after me" (guide principle #5)
    uw_change = token.get("unique_wallet_24h_change")
    if uw_change is not None:
        if uw_change > 20:
            factors.append(1.3)   # >20% wallet growth = strong demand
        elif uw_change > 5:
            factors.append(1.15)
        elif uw_change < -20:
            factors.append(0.6)   # wallets leaving = dying
        elif uw_change < -5:
            factors.append(0.8)

    if not factors:
        return 1.0  # v10: no data = neutral, not a penalty

    # v20: bounds from SCORING_PARAMS (dynamic)
    return max(SCORING_PARAMS["onchain_mult_floor"],
               min(SCORING_PARAMS["onchain_mult_cap"], sum(factors) / len(factors)))


def _compute_safety_penalty(token: dict) -> float:
    """
    Returns a penalty multiplier [0.3, 1.0].
    1.0 = safe, 0.3 = floor (never destroy a token completely).
    v5.1: Softened all factors — memecoin-typical patterns shouldn't obliterate scores.
    """
    penalty = 1.0

    # NOTE: mint_authority and freeze_authority are now handled by _apply_hard_gates()
    # (hard reject, score=0). No soft penalty needed here.

    # High insider concentration (bundled) — softer curve, higher floor
    insider_pct = token.get("insider_pct")
    if insider_pct is not None and insider_pct > 30:
        penalty *= max(0.5, 1.0 - (insider_pct - 30) / 100)

    # Top 10 holders own too much — 50-60% is common on pump.fun
    top10 = token.get("top10_holder_pct")
    if top10 is not None and top10 > 50:
        penalty *= max(0.7, 1.0 - (top10 - 50) / 100)

    # RugCheck risk score (0=safe, 10000=dangerous) — higher floor
    risk = token.get("risk_score")
    if risk is not None and risk > 5000:
        penalty *= max(0.5, 1.0 - (risk - 5000) / 5000)

    # Multiple risk flags — 3 flags is common on pump.fun tokens
    risk_count = token.get("risk_count", 0) or 0
    if risk_count >= 3:
        penalty *= 0.9

    # Jito slot-based bundle detection (high confidence, low false positives)
    jito_max_txns = token.get("jito_max_slot_txns") or 0
    if jito_max_txns >= 5:
        penalty *= 0.4  # definite Jito bundle — still harsh but not near-zero
    elif jito_max_txns >= 3:
        penalty *= 0.6  # suspicious slot clustering
    else:
        # Fallback: balance-based bundle detection (higher false positive rate)
        bundle_detected = token.get("bundle_detected")
        bundle_pct = token.get("bundle_pct") or 0
        if bundle_detected:
            if bundle_pct > 20:
                penalty *= 0.5  # bundling common in memecoins
            elif bundle_pct > 10:
                penalty *= 0.7  # mild bundling

    # Phase 3: High Gini = wealth too concentrated — normal for small tokens
    helius_gini = token.get("helius_gini")
    if helius_gini is not None and helius_gini > 0.85:
        penalty *= 0.8

    # Phase 3: Too few holders — expected for new tokens
    helius_holder_count = token.get("helius_holder_count")
    if helius_holder_count is not None and helius_holder_count < 50:
        penalty *= 0.85

    # Phase 3B: Whale concentration too high — softer curve
    whale_total_pct = token.get("whale_total_pct")
    if whale_total_pct is not None and whale_total_pct > 60:
        penalty *= max(0.7, 1.0 - (whale_total_pct - 60) / 80)

    # Algorithm v3.1: Bubblemaps — low decentralization = clustered supply
    bb_score = token.get("bubblemaps_score")
    if bb_score is not None:
        if bb_score < 20:
            penalty *= 0.6  # very centralized
        elif bb_score < 40:
            penalty *= 0.85  # moderately centralized

    # Algorithm v3.1: Bubblemaps — largest wallet cluster holds too much
    bb_cluster_max = token.get("bubblemaps_cluster_max_pct")
    if bb_cluster_max is not None and bb_cluster_max > 30:
        penalty *= max(0.6, 1.0 - (bb_cluster_max - 30) / 70)

    # Algorithm v4: Whale dominance (concentration * inequality) — mild flag
    whale_dom = token.get("whale_dominance")
    if whale_dom is not None and whale_dom > 0.5:
        penalty *= 0.85

    # Algorithm v4 Sprint 4: Whale direction tracking — softer
    whale_dir = token.get("whale_direction")
    if whale_dir == "distributing":
        penalty *= 0.75   # Whales switching from buy to sell = uncertain
    elif whale_dir == "dumping":
        penalty *= 0.65   # Consistent selling = bearish

    # v13: LP lock — unlocked LP = rug vector (guide principle #7)
    lp_locked = token.get("lp_locked_pct")
    if lp_locked is not None:
        if lp_locked == 0:
            penalty *= 0.6    # LP not locked = rug risk
        elif lp_locked < 50:
            penalty *= 0.85   # partially locked

    # v13: CEX supply pressure (guide principle #13)
    cex_pct = token.get("bubblemaps_cex_pct")
    if cex_pct is not None and cex_pct > 20:
        penalty *= max(0.7, 1.0 - (cex_pct - 20) / 100)

    # FLOOR: v16 raised from 0.6 to 0.75. Safety is still anti-predictive
    # (winners 0.930, losers 0.897 in 131 labeled v15.3 snapshots).
    return max(SCORING_PARAMS["safety_floor"], penalty)


# === SCORE COMPUTATION WEIGHTS ===
# Hardcoded fallback — overridden by scoring_config table when available
_DEFAULT_WEIGHTS = {
    "consensus": 0.30,      # v16: was 0.25; best component (0.072 corr)
    "conviction": 0.05,     # v16: was 0.10; near noise (0.025 corr)
    "breadth": 0.10,        # v16: was 0.20; essentially random (0.003 corr)
    "price_action": 0.55,   # v16: was 0.45; dominant signal (0.062 corr)
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
            return min(1.0, weighted / (tk * 0.05))
        return min(1.0, uk / (tk * 0.05))
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
        return max(0, min(1, (ac - 6) / 4))
    elif component == "breadth":
        bs = token.get("breadth_score")
        if bs is not None:
            return bs
        m = token.get("mentions")
        if m is None:
            return None
        # v14: Recalibrated from /30 → /12
        return min(1.0, m / 12)
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
    pc24 = token.get("price_change_24h")
    if "consensus" in available and pc24 is not None and pc24 > 50:
        consensus_discount = max(0.5, 1.0 - (pc24 / 400))
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
    component_conf = total_available_weight / sum(BALANCED_WEIGHTS.values())
    # KOL coverage: 1 KOL = low confidence, 3+ KOLs = full
    uk = token.get("unique_kols", 1)
    kol_conf = min(1.0, uk / 3)
    # Breadth quality: near-zero breadth = low confidence
    breadth = token.get("breadth_score", 0) or 0
    breadth_conf = min(1.0, breadth / 0.15) if breadth > 0 else 0.3
    # Enrichment: has on-chain data?
    enrichment_conf = 1.0 if token.get("token_address") else 0.5

    data_confidence = component_conf * 0.4 + kol_conf * 0.3 + breadth_conf * 0.2 + enrichment_conf * 0.1

    return renormalized_score, round(data_confidence, 3)


def _classify_lifecycle_phase(token: dict) -> tuple[str, float]:
    """
    Classify token into Minsky lifecycle phase.
    Returns (phase_name, phase_penalty) where penalty is [0.2, 1.1].
    Priority: panic > profit_taking > euphoria > boom > displacement > unknown.
    """
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
    if pc24 is not None and pc24 < -30:
        if (va is not None and va < 0.5) or whale_dir == "dumping":
            return "panic", 0.25

    # Profit-taking: smart money exiting
    if pc24 is not None and pc24 > 100:
        vol_spike = vol_proxy is not None and vol_proxy > 40
        whale_exit = whale_dir in ("distributing", "dumping")
        if (vol_spike or whale_exit) and ma < 0:
            return "profit_taking", 0.35

    # Euphoria: over-saturated, most KOLs already in
    # v17: Lowered thresholds — 4 KOLs at +150% is clearly euphoria, not boom
    if uk >= 3 and pc24 is not None and pc24 > 100 and sent > 0.2:
        return "euphoria", 0.5

    # Boom: sweet spot — growing interest + price confirming
    # v19: No boom bonus for large-cap tokens (>$50M). A $700M token pumping
    # +25% is NOT a "boom" entry — it's an established token getting attention.
    t_mcap = token.get("market_cap") or 0
    if uk >= 2 and pc24 is not None and 10 < pc24 <= 200:
        has_vol = va is not None and va > 1.0
        has_social = sv > 0.3 or ma > 0.2
        if has_vol and has_social:
            if t_mcap > 50_000_000:
                return "boom", 0.85  # large cap "boom" = penalty, not bonus
            return "boom", 1.1

    # Displacement: fresh, unproven
    if age is not None and age < 6 and uk <= 1:
        if pc24 is None or pc24 < 50:
            return "displacement", 0.9

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
            # DexScreener URL pair addresses → confirmed
            for chain, pair_addr in DEXSCREENER_URL_REGEX.findall(text):
                resolved = _resolve_pair_to_symbol(chain, pair_addr, ca_cache)
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
            # Photon-sol URLs → confirmed
            for photon_addr in PHOTON_URL_REGEX.findall(text):
                resolved = _resolve_pair_to_symbol("solana", photon_addr, ca_cache)
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
            # v16: Extract ALL CAs from message text (for extraction audit)
            msg_cas = [m for m in CA_REGEX.findall(text) if m not in KNOWN_PROGRAM_ADDRESSES]

            sentiment = calculate_sentiment(text)
            hours_ago = (now - date).total_seconds() / 3600

            # v14: Detect update/brag messages — weight reduction
            is_update = _is_update_or_brag(text)
            # Forward/reply detection — forwarded content inflates breadth
            is_forwarded = msg.get("is_forwarded", False)
            is_reply = msg.get("is_reply", False)
            # v14: Message length as confidence signal
            msg_len = len(text)
            length_weight = 1.2 if msg_len > 150 else (0.5 if msg_len < 20 else 1.0)
            # Combined mention weight: forwards 0.2x, updates 0.3x, replies 0.4x
            if is_forwarded:
                mention_weight = 0.2 * length_weight
            elif is_update:
                mention_weight = 0.3 * length_weight
            elif is_reply:
                mention_weight = 0.4 * length_weight
            else:
                mention_weight = 1.0 * length_weight

            # Per-message conviction NLP (Sprint 6)
            msg_conv = _compute_message_conviction(text)

            # v14: Extract entry market cap from message text
            stated_mcap = _extract_entry_mcap(text)

            # v10: Store raw mention for each token in this message
            for token in tokens:
                raw_kol_mentions.append({
                    "symbol": token,
                    "kol_group": group_name,
                    "message_text": text[:2000],  # cap at 2000 chars
                    "message_date": date.isoformat(),
                    "sentiment": round(sentiment, 3),
                    "msg_conviction_score": round(msg_conv["msg_conviction_score"], 3),
                    "hours_ago": round(hours_ago, 2),
                    "is_positive": sentiment >= -0.3,
                    "narrative": None,
                    "tokens_in_message": list(tokens),
                    # v16: Extraction audit fields
                    "extraction_method": source_by_symbol.get(token, "unknown"),
                    "extracted_cas": msg_cas if msg_cas else None,
                })

            for token, source in token_tuples:
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
                ct = 0
                if msg_conv.get("has_alpha_framing"): ct += 1
                if msg_conv.get("has_gamble_framing"): ct -= 1
                if msg_conv.get("has_chart_analysis"): ct += 0.5
                token_data[token]["call_type_scores"].append(ct)
                token_data[token]["msg_lengths"].append(msg_len)
                token_data[token]["msg_has_ca"].append(bool(msg_cas))
                token_data[token]["msg_tokens_count"].append(len(token_tuples))
                token_data[token]["msg_texts_raw"].append(text)

                # Negative mentions (sentiment < -0.3) = KOL is warning, not calling
                # Don't count toward mentions/breadth/conviction — prevents false positives
                is_positive_mention = sentiment >= -0.3

                if is_positive_mention:
                    # v14: Weighted mentions — updates/brags count 0.3x, short msgs 0.5x
                    token_data[token]["mentions"] += mention_weight
                    # Forwarded msgs don't count as unique KOL calls for breadth
                    if not is_forwarded:
                        token_data[token]["groups"].add(group_name)
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
        kol_consensus = min(1.0, recency_weighted_unique / (total_kols * 0.05))

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
        conviction_score = max(0, min(1, (effective_conviction - 6) / 4))

        # v15: Conviction dampening — single KOL shouldn't max conviction
        kol_count_factor = min(1.0, unique_kols / 2)  # 1 KOL → 0.5x, 2+ → 1.0x
        conviction_score *= kol_count_factor

        # v9: Recency-weighted breadth (KOL reputation * tier * count * freshness decay)
        kol_mention_counts = data.get("kol_mention_counts", {})
        if kol_mention_counts and kol_scores:
            weighted_mentions = 0
            for kol, count in kol_mention_counts.items():
                group_hours = data["hours_ago_by_group"].get(kol, [])
                freshest = min(group_hours) if group_hours else 24
                decay = _two_phase_decay(freshest)
                weighted_mentions += kol_scores.get(kol, 1.0) * tw(kol) * count * decay
            # v14: Recalibrated from /20 → /8. In memecoin reality, 3-4 KOLs
            # mentioning a token IS strong breadth. Old formula needed 9+ KOLs for 0.5.
            breadth_score = min(1.0, weighted_mentions / 8)
        else:
            # Fallback: decay total mentions by freshest overall mention
            # v14: Recalibrated from /30 → /12
            freshest_overall = min(data["hours_ago"]) if data["hours_ago"] else 999
            decay = _two_phase_decay(freshest_overall)
            breadth_score = min(1.0, (data["mentions"] * decay) / 12)

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
            # Algorithm v7: Weight renormalization
            "_tw_func": tw,
            "data_confidence": data_conf,
            # v11: Raw hours_ago list for activity ratio multiplier
            "_hours_ago": list(data["hours_ago"]),
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
        })

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

    # v16: Save ALL enriched tokens before gates for snapshot backtesting.
    # Gated tokens get gate_reason marked on the dict (shared references).
    all_enriched = list(ranking)

    # === Quality gates BEFORE expensive enrichment ===
    # v28: Converted from hard removal to soft penalties (gate_mult) so
    # outcome_tracker labels these tokens and backtest can validate empirically.
    # Gate 1: No token_address → 0.3x (can't enrich but still scored)
    addr_penalized = 0
    for t in ranking:
        if not t.get("token_address"):
            existing_gate = t.get("gate_mult", 1.0)
            t["gate_mult"] = round(min(existing_gate, 0.3), 3)
            t["gate_reason"] = t.get("gate_reason") or "no_address"
            addr_penalized += 1
    if addr_penalized:
        logger.info("No-address soft penalty (0.3x) applied to %d tokens", addr_penalized)

    # Gate 1b: No volume AND no liquidity → 0.3x
    data_penalized = 0
    for t in ranking:
        if (t.get("volume_24h") or 0) <= 0 and (t.get("liquidity_usd") or 0) <= 0:
            existing_gate = t.get("gate_mult", 1.0)
            t["gate_mult"] = round(min(existing_gate, 0.3), 3)
            t["gate_reason"] = t.get("gate_reason") or "no_data"
            data_penalized += 1
    if data_penalized:
        logger.info("No-data soft penalty (0.3x) applied to %d tokens", data_penalized)

    # Gate 2: For longer windows (48h+), single A-tier KOL = soft penalty (not removal)
    # v21: converted from hard gate to 0.6x penalty — collect outcome data to validate
    if hours >= 48:
        single_a_count = 0
        for t in ranking:
            tiers = t.get("kol_tiers", {})
            has_s_tier = any(tier == "S" for tier in tiers.values())
            if not has_s_tier and t.get("unique_kols", 0) < 2:
                existing_gate = t.get("gate_mult", 1.0)
                t["gate_mult"] = round(min(existing_gate, 0.6), 3)
                t["gate_reason"] = t.get("gate_reason") or "single_a_tier"
                single_a_count += 1
        if single_a_count:
            logger.info("Single-A-tier penalty applied to %d tokens (window=%dh)", single_a_count, hours)

    # === Hard gates (uses RugCheck + DexScreener data) ===
    # v16: _apply_hard_gates now marks gate_reason on ejected tokens
    ranking = _apply_hard_gates(ranking)

    # === Wash trading — soft penalty (already in wash_pen multiplier chain) ===
    # v21: removed hard gate. wash_pen (1.0 - wash_score) already penalizes these tokens.
    # Tokens with score > 0.8 get wash_pen = 0.2x which is severe enough.
    for token in ranking:
        token["wash_trading_score"] = round(_compute_wash_trading_score(token), 3)
        if token.get("wash_trading_score", 0) > 0.8:
            existing_gate = token.get("gate_mult", 1.0)
            token["gate_mult"] = round(min(existing_gate, 0.5), 3)
            token["gate_reason"] = token.get("gate_reason") or "wash_trading"

    # v28: Only mint/freeze are hard-gated now; everything else is soft penalty
    hard_gated = sum(1 for t in all_enriched if t.get("gate_reason") and t not in ranking)
    soft_penalized = sum(1 for t in ranking if t.get("gate_mult", 1.0) < 1.0)
    logger.info("Post-gate: %d tokens (%d soft-penalized, %d hard-gated [mint/freeze only]) — now running expensive enrichment",
                len(ranking), soft_penalized, hard_gated)

    # === Expensive enrichment on SURVIVORS only (v5.1) ===
    # Phase 3: Helius enrichment (bundle detection + holder quality + whale tracking)
    enrich_tokens_helius(ranking)

    # Phase 3B: Jupiter enrichment (tradeability + price impact + routes)
    enrich_tokens_jupiter(ranking)

    # Algorithm v3.1: Bubblemaps enrichment (wallet clustering + decentralization score)
    enrich_tokens_bubblemaps(ranking)

    # === Algorithm v4: Birdeye OHLCV enrichment (top 5 survivors) ===
    enrich_tokens_ohlcv(ranking)

    # === Algorithm v4: Price Action scoring ===
    for token in ranking:
        pa = compute_price_action_score(
            token,
            pa_norm_floor=SCORING_PARAMS["pa_norm_floor"],
            pa_norm_cap=SCORING_PARAMS["pa_norm_cap"],
        )
        token.update(pa)

        # === Harvard adaptations: Volume Squeeze + Trend Strength ===
        squeeze_state, squeeze_score = _detect_volume_squeeze(token)
        token["squeeze_state"] = squeeze_state
        token["squeeze_score"] = round(squeeze_score, 3)

        trend_strength = _compute_trend_strength(token)
        token["trend_strength"] = round(trend_strength, 3)

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
        pvp_recent = token.get("pvp_recent_count") or 0
        if token.get("is_pump_fun"):
            # pump.fun: copycats are normal, scraper already resolves to highest-volume pair
            pvp_pen = max(0.7, 1.0 / (1 + 0.05 * pvp_recent))
        else:
            pvp_pen = max(0.5, 1.0 / (1 + 0.1 * pvp_recent))

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
        act_floor = SCORING_PARAMS["activity_mult_floor"]  # default 0.80
        act_cap = SCORING_PARAMS["activity_mult_cap"]      # default 1.25
        recent_6h_count = sum(1 for h in token.get("_hours_ago", []) if h <= 6)
        total_mention_count = len(token.get("_hours_ago", []))
        if total_mention_count > 0:
            activity_ratio = recent_6h_count / total_mention_count
            if activity_ratio > 0.6:
                activity_mult = act_cap   # most mentions are fresh
            elif activity_ratio > 0.3:
                activity_mult = 1.10
            elif activity_ratio > 0.1:
                activity_mult = 1.0       # neutral
            else:
                activity_mult = act_floor  # dead social activity
        else:
            activity_mult = 1.0

        # v17: Activity_mult cap when token already pumped — buzz peaks during
        # pumps but that's exactly when 2x potential is LOWEST.
        pc24_act = token.get("price_change_24h")
        if pc24_act is not None and pc24_act > 80:
            activity_mult = min(activity_mult, 1.0)   # No bonus if already pumped hard
        elif pc24_act is not None and pc24_act > 50:
            activity_mult = min(activity_mult, 1.10)  # Reduced bonus

        # v15: Breadth floor — low KOL count is a mild flag, not a death sentence
        breadth_raw = float(token.get("breadth_score", 0) or 0)
        if breadth_raw < 0.033:    # ~2 KOLs or fewer
            breadth_pen = 0.75
        elif breadth_raw < 0.05:   # ~3 KOLs
            breadth_pen = 0.85
        elif breadth_raw < 0.08:   # ~5 KOLs
            breadth_pen = 0.95
        else:
            breadth_pen = 1.0

        # v17: Pump momentum penalty — penalize tokens actively pumping RIGHT NOW
        # This is NOT a duplicate of price_action (which averages 4 sub-components).
        # This multiplier acts on the FINAL score to directly penalize active pumps.
        # v20: thresholds from SCORING_PARAMS (dynamic).
        pump_1h_hard = SCORING_PARAMS["pump_pc1h_hard"]  # default 30
        pump_5m_hard = SCORING_PARAMS["pump_pc5m_hard"]  # default 15
        pump_1h_mod = pump_1h_hard * 0.5   # 15 at default
        pump_5m_mod = pump_5m_hard * 0.533  # ~8 at default
        pump_1h_light = pump_1h_hard * 0.267  # ~8 at default
        pc_1h = token.get("price_change_1h")
        pc_5m = token.get("price_change_5m")
        pump_momentum_pen = 1.0
        if (pc_1h is not None and pc_1h > pump_1h_hard) or (pc_5m is not None and pc_5m > pump_5m_hard):
            pump_momentum_pen = 0.5   # active pump
        elif (pc_1h is not None and pc_1h > pump_1h_mod) or (pc_5m is not None and pc_5m > pump_5m_mod):
            pump_momentum_pen = 0.7   # moderate pump
        elif pc_1h is not None and pc_1h > pump_1h_light:
            pump_momentum_pen = 0.85  # light pump
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

        # v15.2: Size opportunity multiplier — backtest-proven signal.
        # Winners avg 4.1M mcap vs losers 21.4M. Fresh (<12h) + small (<500K) = 30% precision.
        # v19: mcap tier — smaller tokens need less $ to 2x.
        # A $700M token needs $700M NEW capital to 2x — nearly impossible for memecoins.
        # Progressive penalty: micro→bonus, mid→neutral, large→heavy penalty.
        t_mcap = token.get("market_cap") or 0
        if t_mcap <= 0:
            mcap_factor = 1.0  # no data = neutral
        elif t_mcap < 300_000:
            mcap_factor = 1.3   # micro cap — room to 100x
        elif t_mcap < 1_000_000:
            mcap_factor = 1.15  # small cap — easy to 2x
        elif t_mcap < 5_000_000:
            mcap_factor = 1.0   # mid cap — neutral
        elif t_mcap < 20_000_000:
            mcap_factor = 0.85  # established — harder to 2x
        elif t_mcap < 50_000_000:
            mcap_factor = 0.70  # large — needs $50M+ new capital
        elif t_mcap < 200_000_000:
            mcap_factor = 0.50  # very large — 2x extremely rare
        elif t_mcap < 500_000_000:
            mcap_factor = 0.35  # mega cap — basically impossible to 2x
        else:
            mcap_factor = 0.25  # >$500M — needs $500M+ to 2x, not happening
        # freshness tier: KOL just called = buy pressure incoming
        # v19: No freshness bonus for large caps — fresh call on $700M token
        # doesn't make it easier to 2x.
        if t_mcap >= 50_000_000:
            fresh_factor = 1.0  # large caps: no freshness boost
        elif freshest_h < 4:
            fresh_factor = 1.2
        elif freshest_h < 12:
            fresh_factor = 1.1
        else:
            fresh_factor = 1.0
        size_mult = max(0.25, min(1.5, mcap_factor * fresh_factor))
        token["size_mult"] = round(size_mult, 3)

        # v15.3: S-tier KOL bonus — S-tier callers have proven track records
        kol_tiers = token.get("kol_tiers", {})
        s_tier_count = sum(1 for tier in kol_tiers.values() if tier == "S")
        s_tier_mult = 1.2 if s_tier_count > 0 else 1.0
        token["s_tier_mult"] = s_tier_mult

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
        token["_conviction_val"] = max(0, min(1, (token.get("avg_conviction", 6) - 6) / 4)) if token.get("avg_conviction") is not None else None
        token["_breadth_val"] = _get_component_value(token, "breadth")
        token["_price_action_val"] = _get_component_value(token, "price_action")

        # v21: gate_mult — soft safety penalties (top10, risk, liquidity, holders, single_a_tier)
        # v27: Explicit default — tokens that bypass _apply_hard_gates get 1.0
        gate_mult = float(token.get("gate_mult", 1.0) or 1.0)

        # v24: Chain (12 multipliers). Added entry_drift_mult (price vs social drift).
        # v23 removals: squeeze_mult (dead), trend_mult (dead),
        # wash_pen+pump_pen (merged → manipulation_pen),
        # pump_momentum_pen (folded into crash_pen min()).
        combined_raw = (onchain_mult * safety_pen * pump_bonus
                        * manipulation_pen * pvp_pen * crash_pen
                        * activity_mult * breadth_pen
                        * size_mult * s_tier_mult * gate_mult
                        * entry_drift_mult)
        # v16: Floor at 0.25 decompresses the 0-14 band where 97% of tokens stuck.
        # v17: Cap at 2.0 prevents multiplier stacking (activity*s_tier*size)
        # from inflating mediocre base scores beyond 100.
        combined = max(SCORING_PARAMS["combined_floor"], min(SCORING_PARAMS["combined_cap"], combined_raw))

        # Apply to all three scoring modes
        token["score"] = min(100, max(0, int(token["score"] * combined)))
        token["score_conviction"] = min(100, max(0, int(token["score_conviction"] * combined)))
        token["score_momentum"] = min(100, max(0, int(token["score_momentum"] * combined)))

        # Multi-indicator confirmation pillars (data only, no score penalty)
        # v15.2: Gate REMOVED — backtest (204 samples) shows 0 pillars = 28.6%
        # hit rate vs 3 pillars = 0%. Gate was penalizing winners.
        pillars = 0
        consensus_val = _get_component_value(token, "consensus")
        pa_val = _get_component_value(token, "price_action")
        breadth_val = _get_component_value(token, "breadth")
        if consensus_val is not None and consensus_val >= 0.2:
            pillars += 1
        if pa_val is not None and pa_val >= 0.35:
            pillars += 1
        if breadth_val is not None and breadth_val >= 0.08:
            pillars += 1
        token["confirmation_pillars"] = pillars

        # Update interpretation band after final score
        token["score_interpretation"] = _interpret_score(token["score"])

        # ML v2 Phase C: Entry zone timing quality (0-1)
        # Measures how close the token is to an optimal buy window.
        # ML-only initially; may become a soft multiplier if correlation > 0.05.
        token["entry_timing_quality"] = _compute_entry_timing_quality(token)

    unconfirmed = sum(1 for t in ranking if t.get("confirmation_pillars", 0) < 2)
    logger.info(
        "Applied on-chain multipliers & safety penalties to %d tokens "
        "(unconfirmed: %d)",
        len(ranking), unconfirmed,
    )

    # Apply ML scoring if model is available (overrides manual score)
    _apply_ml_scores(ranking)

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
