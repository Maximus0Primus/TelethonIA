"""
Pipeline module: extract tokens, calculate sentiment, aggregate ranking.
Reusable functions called by the scraper's main loop.
"""

import re
import time
import logging
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import TypedDict
from pathlib import Path

import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

# === TOKEN EXTRACTION ===

TOKEN_REGEX = re.compile(r"(?<![A-Z0-9_])\$([A-Z][A-Z0-9]{1,14})\b")

EXCLUDED_TOKENS = {
    # Stablecoins & majors (never memecoins)
    "USD", "USDT", "USDC", "BUSD", "DAI", "TUSD",
    "SOL", "ETH", "BTC", "BNB", "XRP", "ADA", "DOT", "AVAX", "MATIC",
    # Crypto jargon (always false positives in KOL messages)
    "CA", "LP", "MC", "ATH", "ATL", "FDV", "TVL",
    "PNL", "ROI", "APY", "APR", "CRYPTO", "DEX", "CEX",
    "ICO", "IDO", "IEO", "TGE", "KOL",
    "CEO", "COO", "CTO", "CFO",
    "URL", "API", "NFT", "DAO", "DEFI",
    "GMT", "UTC", "EST", "PST",
    "DM", "RT", "TG", "CT",
    # User-reported false positives
    "TESLA",
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


# === VADER ANALYZER (module-level singleton) ===

_vader = SentimentIntensityAnalyzer()

# === DEXSCREENER TOKEN VERIFICATION ===

_DEXSCREENER_CACHE_FILE = Path(__file__).parent / "token_cache.json"
_DEXSCREENER_URL = "https://api.dexscreener.com/latest/dex/search"


def _load_token_cache() -> dict[str, bool]:
    """Load cached token verification results. { "SYMBOL": true/false }"""
    import json
    if _DEXSCREENER_CACHE_FILE.exists():
        try:
            with open(_DEXSCREENER_CACHE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_token_cache(cache: dict[str, bool]) -> None:
    import json
    with open(_DEXSCREENER_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def verify_tokens_exist(symbols: list[str]) -> set[str]:
    """
    Check which token symbols have active trading pairs on DexScreener.
    Returns the set of symbols that are verified as real tokens.
    Uses a persistent cache to avoid re-checking known tokens.
    """
    cache = _load_token_cache()
    verified: set[str] = set()
    to_check: list[str] = []

    for sym in symbols:
        raw = sym.lstrip("$")
        if raw in cache:
            if cache[raw]:
                verified.add(sym)
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
                # Check if any pair has this exact symbol with real liquidity
                found = any(
                    p.get("baseToken", {}).get("symbol", "").upper() == raw
                    and float(p.get("liquidity", {}).get("usd", 0) or 0) > 500
                    for p in pairs
                )
                cache[raw] = found
                if found:
                    verified.add(sym)
                    logger.info("Token %s verified on DexScreener", sym)
                else:
                    logger.info("Token %s NOT found on DexScreener — filtered out", sym)
            else:
                logger.warning("DexScreener API %d for %s — skipping filter", resp.status_code, sym)
                # Don't cache errors — let it retry next cycle
                verified.add(sym)
        except requests.RequestException as e:
            logger.warning("DexScreener request failed for %s: %s — keeping token", sym, e)
            verified.add(sym)

        # Rate limit: ~0.5s between requests
        time.sleep(0.5)

    _save_token_cache(cache)
    return verified


# === PUBLIC API ===

def extract_tokens(text: str) -> list[str]:
    """Extract $TOKEN symbols from message text."""
    matches = TOKEN_REGEX.findall(text.upper())
    tokens = []
    seen: set[str] = set()
    for match in matches:
        symbol = f"${match}"
        if match not in EXCLUDED_TOKENS and symbol not in seen:
            tokens.append(symbol)
            seen.add(symbol)
    return tokens


def calculate_sentiment(text: str) -> float:
    """VADER (70%) + crypto lexicon (30%) blend, clamped to [-1, 1]."""
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

    final = 0.7 * vader_score + 0.3 * lexicon_boost
    return max(-1.0, min(1.0, final))


def aggregate_ranking(
    messages_dict: dict[str, list[dict]],
    groups_conviction: dict[str, int],
    hours: int,
) -> list[TokenRanking]:
    """
    Given raw messages grouped by channel username, compute ranked token list
    for the specified time window.

    Parameters
    ----------
    messages_dict : { "group_username": [ { "text": ..., "date": ISO str, ... }, ... ] }
    groups_conviction : { "group_username": conviction_int }
    hours : time window in hours

    Returns
    -------
    Sorted list of TokenRanking dicts (highest score first).
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours)
    total_kols = len(groups_conviction)

    token_data: dict[str, TokenStats] = defaultdict(lambda: {
        "mentions": 0,
        "sentiments": [],
        "groups": set(),
        "convictions": [],
    })

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
            if not text or len(text) < 5:
                continue

            tokens = extract_tokens(text)
            if not tokens:
                continue

            sentiment = calculate_sentiment(text)

            for token in tokens:
                token_data[token]["mentions"] += 1
                token_data[token]["sentiments"].append(sentiment)
                token_data[token]["groups"].add(group_name)
                token_data[token]["convictions"].append(conviction)

    # Score & rank
    ranking: list[TokenRanking] = []

    for symbol, data in token_data.items():
        if data["mentions"] == 0:
            continue

        unique_kols = len(data["groups"])
        kol_consensus = min(1.0, unique_kols / (total_kols * 0.15))

        avg_sentiment = sum(data["sentiments"]) / len(data["sentiments"])
        sentiment_score = (avg_sentiment + 1) / 2

        avg_conviction = sum(data["convictions"]) / len(data["convictions"])
        conviction_score = max(0, min(1, (avg_conviction - 6) / 4))

        breadth_score = min(1.0, data["mentions"] / 30)

        # Balanced score (legacy, kept as default)
        raw_score = (
            0.35 * kol_consensus
            + 0.25 * sentiment_score
            + 0.25 * conviction_score
            + 0.15 * breadth_score
        )
        score = min(100, max(0, int(raw_score * 100)))

        # Conviction mode: long-term holds — trust KOL consensus + conviction
        raw_conviction = (
            0.40 * kol_consensus
            + 0.35 * conviction_score
            + 0.15 * sentiment_score
            + 0.10 * breadth_score
        )
        score_conviction = min(100, max(0, int(raw_conviction * 100)))

        # Momentum mode: quick gains — trust recent buzz + positive sentiment
        raw_momentum = (
            0.35 * sentiment_score
            + 0.30 * breadth_score
            + 0.20 * kol_consensus
            + 0.15 * conviction_score
        )
        score_momentum = min(100, max(0, int(raw_momentum * 100)))

        trend = "up" if avg_sentiment > 0.15 else ("down" if avg_sentiment < -0.15 else "stable")

        groups_with_conv = sorted(
            [(g, groups_conviction.get(g, 7)) for g in data["groups"]],
            key=lambda x: x[1],
            reverse=True,
        )
        top_kols = [g[0] for g in groups_with_conv[:5]]

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
            "top_kols": top_kols,
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

    ranking.sort(key=lambda x: (x["score"], x["mentions"]), reverse=True)
    return ranking
