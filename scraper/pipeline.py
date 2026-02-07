"""
Pipeline module: extract tokens, calculate sentiment, aggregate ranking.
Reusable functions called by the scraper's main loop.
"""

import re
import json
import math
import time
import logging
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import TypedDict
from pathlib import Path

import numpy as np
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from enrich import enrich_tokens

logger = logging.getLogger(__name__)

# === TOKEN EXTRACTION ===

TOKEN_REGEX = re.compile(r"(?<![A-Za-z0-9_])\$([A-Za-z][A-Za-z0-9]{0,14})\b")
BARE_TOKEN_REGEX = re.compile(r"\b([A-Z][A-Z0-9]{2,14})\b")

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
    # Common English words (bare token false positives)
    "THE", "AND", "FOR", "NOT", "YOU", "ALL", "CAN", "HER", "WAS", "ONE",
    "OUR", "OUT", "ARE", "HAS", "BUT", "GET", "HIS", "HOW", "ITS", "LET",
    "MAY", "NEW", "NOW", "OLD", "SEE", "WAY", "WHO", "DID", "GOT", "HIM",
    "MAN", "TOP", "USE", "SAY", "SHE", "TOO", "BIG", "END", "TRY", "ASK",
    "MEN", "RUN", "RAN", "SAT", "WIN", "WON", "YES", "YET", "BET", "BAD",
    "HOT", "LOT", "SET", "SIT", "CUT", "HIT", "PUT", "RIP",
    # Crypto slang (not tokens)
    "GG", "GM", "GN", "WEN", "SER", "FOMO", "HODL", "DYOR", "NGMI", "WAGMI",
    "BASED", "CHAD", "COPE", "SHILL", "ALPHA", "BETA", "GAMMA", "DELTA",
    "LONG", "SHORT", "PUMP", "DUMP", "MOON", "SEND", "CALL", "PLAY",
    "HOLD", "SELL", "JUST", "LIKE", "WITH", "THIS", "THAT", "FROM",
    "HAVE", "WILL", "BEEN", "WHAT", "WHEN", "YOUR", "THEM", "THAN",
    "EACH", "MAKE", "VERY", "SOME", "BACK", "ONLY", "COME", "MADE",
    "AFTER", "ALSO", "INTO", "OVER", "SUCH", "TAKE", "MOST", "GOOD",
    "KNOW", "TIME", "LOOK", "NEXT", "MUCH", "MORE", "LAST", "STILL",
    "HIGH", "GONNA", "GOING", "ABOUT", "THINK", "THESE", "RIGHT",
    "CHECK", "LOOKS", "PRICE", "CHART", "ENTRY", "MARKET", "TOKEN",
    "COINS", "TRADE", "PROFIT", "UPDATE", "BUYING", "TODAY", "WATCH",
    "WAITING", "EARLY", "MASSIVE",
    # Known paid shills / casino platforms (not real memecoins)
    "METAWIN",
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


def _is_active_token(pairs: list[dict], symbol_raw: str) -> bool:
    """
    Check if any trading pair for this symbol shows real recent activity.
    Criteria: exact symbol match + 24h volume > $1000 OR 24h transactions > 10.
    """
    for p in pairs:
        if p.get("baseToken", {}).get("symbol", "").upper() != symbol_raw:
            continue

        vol_24h = float(p.get("volume", {}).get("h24", 0) or 0)
        txns = p.get("txns", {}).get("h24", {})
        buys = int(txns.get("buys", 0) or 0)
        sells = int(txns.get("sells", 0) or 0)
        total_txns = buys + sells

        if vol_24h > 100 or total_txns > 5:
            return True

    return False


def verify_tokens_exist(symbols: list[str]) -> set[str]:
    """
    Check which token symbols have active trading on DexScreener.
    A token is kept only if it has real recent activity (volume or transactions
    in the last 24h), not just leftover liquidity from a dead pair.
    Uses a persistent cache to avoid re-checking known tokens within a cycle.
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
                found = _is_active_token(pairs, raw)
                cache[raw] = found
                if found:
                    verified.add(sym)
                    logger.info("Token %s verified (active trading)", sym)
                else:
                    logger.info("Token %s filtered out (no recent activity)", sym)
            else:
                logger.warning("DexScreener API %d for %s — keeping token", resp.status_code, sym)
                verified.add(sym)
        except requests.RequestException as e:
            logger.warning("DexScreener failed for %s: %s — keeping token", sym, e)
            verified.add(sym)

        time.sleep(0.5)

    _save_token_cache(cache)
    return verified


# === ML MODEL (lazy-loaded) ===

_ML_MODEL_PATH = Path(__file__).parent / "model_12h.json"
_ML_META_PATH = Path(__file__).parent / "model_12h_meta.json"
_ml_model = None
_ml_features = None


def _load_ml_model():
    """Lazy-load XGBoost model if available."""
    global _ml_model, _ml_features
    if _ml_model is not None:
        return _ml_model, _ml_features

    if not _ML_MODEL_PATH.exists():
        return None, None

    try:
        import xgboost as xgb
        _ml_model = xgb.XGBClassifier()
        _ml_model.load_model(str(_ML_MODEL_PATH))

        # Load feature list from metadata
        if _ML_META_PATH.exists():
            with open(_ML_META_PATH, "r") as f:
                meta = json.load(f)
            _ml_features = meta.get("features", [])
        else:
            _ml_features = [
                "mentions", "sentiment", "breadth", "avg_conviction", "recency_score",
                "volume_24h_log", "liquidity_usd_log", "market_cap_log",
                "txn_count_24h", "price_change_1h",
                "risk_score", "top10_holder_pct", "insider_pct",
            ]

        logger.info("ML model loaded from %s (%d features)", _ML_MODEL_PATH, len(_ml_features))
        return _ml_model, _ml_features
    except Exception as e:
        logger.warning("Failed to load ML model: %s — using manual scores", e)
        _ml_model = None
        _ml_features = None
        return None, None


def _apply_ml_scores(ranking: list[dict]) -> None:
    """
    If an XGBoost model is available, compute ML-based scores and override manual scores.
    Falls back silently to manual scores if model is not available.
    """
    model, features = _load_ml_model()
    if model is None or not features:
        return

    # Build feature matrix
    rows = []
    for token in ranking:
        row = {}
        for feat in features:
            if feat == "volume_24h_log":
                v = token.get("volume_24h")
                row[feat] = np.log1p(float(v)) if v is not None and float(v) > 0 else np.nan
            elif feat == "liquidity_usd_log":
                v = token.get("liquidity_usd")
                row[feat] = np.log1p(float(v)) if v is not None and float(v) > 0 else np.nan
            elif feat == "market_cap_log":
                v = token.get("market_cap")
                row[feat] = np.log1p(float(v)) if v is not None and float(v) > 0 else np.nan
            elif feat == "breadth":
                uk = token.get("unique_kols", 0)
                tk = token.get("_total_kols", 50)
                row[feat] = uk / max(1, tk)
            else:
                val = token.get(feat)
                row[feat] = float(val) if val is not None else np.nan
        rows.append(row)

    try:
        import pandas as pd
        X = pd.DataFrame(rows, columns=features)
        probas = model.predict_proba(X)[:, 1]

        for token, prob in zip(ranking, probas):
            token["score_ml"] = int(prob * 100)
            token["score"] = int(prob * 100)  # ML score replaces manual score

        logger.info("ML scores applied to %d tokens", len(ranking))
    except Exception as e:
        logger.warning("ML scoring failed: %s — keeping manual scores", e)


# === PUBLIC API ===

def extract_tokens(text: str) -> list[str]:
    """Extract $TOKEN symbols (case-insensitive) + bare ALL-CAPS tokens from text."""
    matches = TOKEN_REGEX.findall(text)
    tokens = []
    seen: set[str] = set()

    # 1) $-prefixed tokens (case-insensitive match, uppercased)
    for match in matches:
        upper = match.upper()
        symbol = f"${upper}"
        if upper not in EXCLUDED_TOKENS and symbol not in seen:
            tokens.append(symbol)
            seen.add(symbol)

    # 2) Bare ALL-CAPS tokens (3+ chars, no $ prefix)
    bare_matches = BARE_TOKEN_REGEX.findall(text)
    for match in bare_matches:
        if match != match.upper():
            continue  # Only exact ALL-CAPS words
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
        "hours_ago": [],
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
            if not text or len(text) < 3:
                continue

            tokens = extract_tokens(text)
            if not tokens:
                continue

            sentiment = calculate_sentiment(text)
            hours_ago = (now - date).total_seconds() / 3600

            for token in tokens:
                token_data[token]["mentions"] += 1
                token_data[token]["sentiments"].append(sentiment)
                token_data[token]["groups"].add(group_name)
                token_data[token]["convictions"].append(conviction)
                token_data[token]["hours_ago"].append(hours_ago)

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

        # Conviction mode: sustained discussion across many KOLs
        raw_conviction = (
            0.35 * kol_consensus
            + 0.30 * conviction_score
            + 0.10 * sentiment_score
            + 0.25 * breadth_score
        )
        score_conviction = min(100, max(0, int(raw_conviction * 100)))

        # Recency score: exponential decay — recent mentions weigh much more
        # 1h ago ≈ 0.74, 3h ≈ 0.41, 6h ≈ 0.17, 12h ≈ 0.03
        decay_lambda = 0.3
        recency_weights = [math.exp(-decay_lambda * h) for h in data["hours_ago"]]
        recency_score = min(1.0, sum(recency_weights) / 10)

        # Momentum mode: what's trending NOW — driven by recency
        raw_momentum = (
            0.40 * recency_score
            + 0.30 * sentiment_score
            + 0.20 * kol_consensus
            + 0.10 * breadth_score
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
            "avg_conviction": round(avg_conviction, 2),
            "recency_score": round(recency_score, 3),
            "_total_kols": total_kols,
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

    # Enrich with on-chain data (DexScreener + RugCheck)
    enrich_tokens(ranking)

    # Apply ML scoring if model is available (overrides manual score)
    _apply_ml_scores(ranking)

    ranking.sort(key=lambda x: (x["score"], x["mentions"]), reverse=True)
    return ranking
