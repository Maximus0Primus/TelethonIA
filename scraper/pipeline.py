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
from enrich_helius import enrich_tokens_helius
from enrich_jupiter import enrich_tokens_jupiter
from enrich_bubblemaps import enrich_tokens_bubblemaps
from kol_scorer import get_kol_scores

logger = logging.getLogger(__name__)

# === TOKEN EXTRACTION ===

TOKEN_REGEX = re.compile(r"(?<![A-Za-z0-9_])\$([A-Za-z][A-Za-z0-9]{0,14})\b")
BARE_TOKEN_REGEX = re.compile(r"\b([A-Z][A-Z0-9]{2,14})\b")
# Solana base58 addresses (32-44 chars, no 0/O/I/l)
CA_REGEX = re.compile(r"\b([1-9A-HJ-NP-Za-km-z]{32,44})\b")

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

# === NARRATIVE/META CLASSIFICATION ===

NARRATIVE_KEYWORDS = {
    "ai_agent": ["agent", "ai", "gpt", "llm", "neural", "brain", "compute", "gpu", "inference"],
    "animal": ["dog", "cat", "pepe", "frog", "shib", "inu", "doge", "bonk", "wif", "penguin", "hippo"],
    "politics": ["trump", "biden", "maga", "election", "political", "president", "vote", "democrat", "republican"],
    "gaming": ["game", "play", "nft", "metaverse", "virtual", "pixel", "quest", "rpg"],
    "defi": ["swap", "yield", "stake", "lend", "borrow", "vault", "protocol", "liquidity"],
    "celebrity": ["elon", "musk", "kanye", "drake", "celebrity", "famous"],
    "culture": ["meme", "wojak", "chad", "sigma", "based", "viral", "tiktok"],
    "rwa": ["rwa", "real world", "tokenize", "asset", "property", "real estate", "commodity"],
    "social_fi": ["socialfi", "social", "creator", "content", "influence", "follow", "friend.tech"],
    "layer2": ["layer2", "l2", "rollup", "zk", "optimistic", "bridge", "scaling"],
    "privacy": ["privacy", "private", "anonymous", "zero knowledge", "zk-snark", "mixnet"],
    "infra": ["infrastructure", "oracle", "indexer", "rpc", "node", "validator", "middleware"],
}

# Rich descriptions for semantic matching (used by sentence-transformers)
NARRATIVE_DESCRIPTIONS = {
    "ai_agent": "AI agent, artificial intelligence, machine learning, GPT, LLM, neural network, autonomous trading bot, compute, GPU inference",
    "animal": "Animal-themed memecoin, dog, cat, pepe frog, shiba inu, doge, bonk, penguin, hippo, cute animal mascot",
    "politics": "Political memecoin, Trump, Biden, MAGA, election, political candidate, president, voting, democrat, republican",
    "gaming": "Gaming token, play-to-earn, NFT game, metaverse, virtual world, pixel art, quest RPG, GameFi",
    "defi": "DeFi protocol, decentralized finance, swap, yield farming, staking, lending, borrowing, vault, liquidity",
    "celebrity": "Celebrity-themed token, Elon Musk, Kanye, Drake, famous person, influencer coin",
    "culture": "Internet culture memecoin, meme, wojak, chad, sigma, based, viral trend, tiktok, internet humor",
    "rwa": "Real world asset tokenization, RWA, property, real estate, commodities, tokenized assets",
    "social_fi": "Social finance, SocialFi, creator economy, content monetization, influencer platform, friend.tech",
    "layer2": "Layer 2 scaling, L2, rollup, ZK proof, optimistic rollup, bridge, blockchain scaling solution",
    "privacy": "Privacy coin, anonymous transactions, zero knowledge proof, ZK-SNARK, mixnet, private blockchain",
    "infra": "Blockchain infrastructure, oracle, indexer, RPC node, validator, middleware, developer tooling",
}

# Semantic narrative classifier (lazy-loaded)
_narrative_model = None
_narrative_embeddings = None
_USE_SEMANTIC_NARRATIVES = None


def _get_narrative_model():
    """Lazy-load sentence-transformers model for semantic narrative classification."""
    global _narrative_model, _narrative_embeddings, _USE_SEMANTIC_NARRATIVES
    if _USE_SEMANTIC_NARRATIVES is not None:
        return _narrative_model, _narrative_embeddings
    try:
        from sentence_transformers import SentenceTransformer, util as st_util
        _narrative_model = SentenceTransformer("all-MiniLM-L6-v2")
        # Pre-encode narrative descriptions
        labels = list(NARRATIVE_DESCRIPTIONS.keys())
        descriptions = [NARRATIVE_DESCRIPTIONS[k] for k in labels]
        embeddings = _narrative_model.encode(descriptions, convert_to_tensor=True)
        _narrative_embeddings = (labels, embeddings, st_util)
        _USE_SEMANTIC_NARRATIVES = True
        logger.info("Semantic narrative classifier loaded (all-MiniLM-L6-v2, %d narratives)", len(labels))
    except ImportError:
        _USE_SEMANTIC_NARRATIVES = False
        logger.info("sentence-transformers not installed — using keyword fallback for narratives")
    except Exception as e:
        _USE_SEMANTIC_NARRATIVES = False
        logger.warning("Semantic narrative model failed: %s — using keyword fallback", e)
    return _narrative_model, _narrative_embeddings


def _classify_narrative_semantic(text: str) -> tuple[str | None, float]:
    """Classify narrative using sentence-transformers cosine similarity."""
    model, emb_data = _get_narrative_model()
    if model is None or emb_data is None:
        return None, 0.0

    labels, label_embeddings, st_util = emb_data
    try:
        text_embedding = model.encode(text[:256], convert_to_tensor=True)
        similarities = st_util.cos_sim(text_embedding, label_embeddings)[0]
        best_idx = int(similarities.argmax())
        best_score = float(similarities[best_idx])
        if best_score >= 0.3:
            return labels[best_idx], round(best_score, 3)
        return None, round(best_score, 3)
    except Exception:
        return None, 0.0


def _classify_narrative_keywords(text: str, symbol: str) -> tuple[str | None, float]:
    """Keyword-based narrative fallback."""
    text_lower = (text + " " + symbol).lower()
    scores = {}
    for narrative, keywords in NARRATIVE_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text_lower)
        if hits > 0:
            scores[narrative] = hits
    if scores:
        best = max(scores, key=scores.get)
        # Approximate confidence from keyword hit ratio
        confidence = min(1.0, scores[best] / 3.0)
        return best, round(confidence, 3)
    return None, 0.0


def classify_narrative(text: str, symbol: str) -> tuple[str | None, float]:
    """
    Return (dominant_narrative, confidence) or (None, 0.0).
    Uses semantic matching if sentence-transformers is available, else keyword fallback.
    """
    # Try semantic first
    model, _ = _get_narrative_model()
    if model is not None:
        return _classify_narrative_semantic(text + " " + symbol)
    return _classify_narrative_keywords(text, symbol)


# === PER-MESSAGE CONVICTION NLP ===

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

    score = max(0.5, min(2.5, score))

    return {
        "msg_conviction_score": round(score, 2),
        "has_price_target": has_price_target,
        "has_personal_stake": has_personal_stake,
        "has_urgency": has_urgency,
        "has_hedging": has_hedging,
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
                # Pick highest-volume Solana pair
                symbol = pairs[0].get("baseToken", {}).get("symbol", "").upper()
                if symbol and len(symbol) <= 15:
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
            ml_score = int(prob * 100)
            token["score_ml"] = ml_score
            # Algorithm v3 B1: Blend ML (70%) + manual (30%) instead of pure override
            token["score"] = int(0.7 * ml_score + 0.3 * token["score"])
            token["score_conviction"] = int(0.7 * ml_score + 0.3 * token["score_conviction"])
            token["score_momentum"] = int(0.7 * ml_score + 0.3 * token["score_momentum"])

        logger.info("ML scores blended (70/30) for %d tokens", len(ranking))
    except Exception as e:
        logger.warning("ML scoring failed: %s — keeping manual scores", e)


# === PUBLIC API ===

def extract_tokens(text: str, ca_cache: dict[str, dict] | None = None) -> list[str]:
    """
    Extract token symbols from text via three methods:
    1) $TOKEN prefix (case-insensitive)
    2) Bare ALL-CAPS words (3+ chars)
    3) Solana contract addresses → resolved to symbols via DexScreener
    """
    tokens = []
    seen: set[str] = set()

    # 1) $-prefixed tokens (case-insensitive match, uppercased)
    for match in TOKEN_REGEX.findall(text):
        upper = match.upper()
        symbol = f"${upper}"
        if upper not in EXCLUDED_TOKENS and symbol not in seen:
            tokens.append(symbol)
            seen.add(symbol)

    # 2) Bare ALL-CAPS tokens (3+ chars, no $ prefix)
    for match in BARE_TOKEN_REGEX.findall(text):
        if match != match.upper():
            continue
        symbol = f"${match}"
        if match not in EXCLUDED_TOKENS and symbol not in seen:
            tokens.append(symbol)
            seen.add(symbol)

    # 3) Solana contract addresses → resolve to symbol
    if ca_cache is not None:
        for match in CA_REGEX.findall(text):
            if match in KNOWN_PROGRAM_ADDRESSES:
                continue
            resolved = _resolve_ca_to_symbol(match, ca_cache)
            if resolved:
                symbol = f"${resolved}"
                if resolved not in EXCLUDED_TOKENS and symbol not in seen:
                    tokens.append(symbol)
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
    Two-phase exponential decay for recency scoring.
    Phase 1 (0-6h): lambda=0.15, half-life ~4.6h — gives tokens time to prove themselves.
    Phase 2 (6h+):  lambda=0.5, half-life ~1.4h — aggressively discounts stale signals.
    """
    if hours_ago <= 6:
        return math.exp(-0.15 * hours_ago)
    else:
        # Continuity at 6h boundary: exp(-0.9) ≈ 0.407
        return math.exp(-0.9) * math.exp(-0.5 * (hours_ago - 6))


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
    return max(0.0, min(1.0, sum(scores) / len(scores)))


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


def _apply_hard_gates(ranking: list[dict]) -> list[dict]:
    """
    Remove tokens that fail hard safety checks.
    These are non-negotiable: mint authority, freeze authority,
    top10 > 70%, risk > 8000, wash trading > 0.8.
    """
    passed = []
    gated = 0
    for token in ranking:
        # mint_authority = can inflate supply at will
        if token.get("has_mint_authority"):
            gated += 1
            continue
        # freeze_authority = can freeze your tokens
        if token.get("has_freeze_authority"):
            gated += 1
            continue
        # top10 holders own > 70% = extreme concentration
        top10 = token.get("top10_holder_pct")
        if top10 is not None and top10 > 70:
            gated += 1
            continue
        # RugCheck risk > 8000 (out of 10000)
        risk = token.get("risk_score")
        if risk is not None and risk > 8000:
            gated += 1
            continue
        # Algorithm v3 A4: Liquidity floor ($10K minimum — absolute dust)
        liq = token.get("liquidity_usd")
        if liq is not None and liq < 10_000:
            gated += 1
            continue
        # Algorithm v3 A4: Holder floor (need real organic community)
        hcount = token.get("helius_holder_count") or token.get("holder_count")
        if hcount is not None and hcount < 30:
            gated += 1
            continue
        passed.append(token)

    if gated:
        logger.info("Hard gates removed %d tokens (mint/freeze/top10>70%%/risk>8000/liq<10K/holders<30)", gated)
    return passed


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

    # 2. Buy pressure (buy_sell_ratio > 0.6 = bullish)
    bsr = token.get("buy_sell_ratio_1h")
    if bsr is not None:
        factors.append(0.5 + bsr)  # 0.5-1.5

    # 3. Liquidity adequacy (liq/mcap > 0.05 = healthy)
    lmr = token.get("liq_mcap_ratio")
    if lmr is not None:
        if lmr < 0.02:
            factors.append(0.5)
        elif lmr > 0.10:
            factors.append(1.2)
        else:
            factors.append(0.8 + lmr * 4)

    # 4. Volume acceleration (6h vol * 4 vs 24h vol)
    va = token.get("volume_acceleration")
    if va is not None:
        factors.append(min(1.5, 0.5 + va * 0.5))

    # 5. Token age penalty (too new = risky, too old = less likely to 2x)
    age = token.get("token_age_hours")
    if age is not None:
        if age < 1:
            factors.append(0.6)
        elif age < 24:
            factors.append(1.2)
        elif age < 168:
            factors.append(1.0)
        else:
            factors.append(0.8)

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

    # 10. Algorithm v3 A1: Short-term volume heat (1h vs 6h acceleration)
    vol_1h = token.get("volume_1h")
    vol_6h = token.get("volume_6h")
    if vol_1h and vol_6h and vol_6h > 0:
        short_heat = (vol_1h * 6) / vol_6h  # 1.0=uniform, >1=accelerating, <1=dying
        token["short_term_heat"] = round(short_heat, 3)
        factors.append(min(1.5, 0.5 + short_heat * 0.5))

    # 12. Algorithm v3.1: Ultra-short heat (5m vs 1h — real-time momentum)
    vol_5m = token.get("volume_5m")
    if vol_5m and vol_1h and vol_1h > 0:
        ultra_heat = (vol_5m * 12) / vol_1h  # 1.0=stable, >2.0=explosion, <0.5=dying
        token["ultra_short_heat"] = round(ultra_heat, 3)
        factors.append(min(1.5, 0.5 + ultra_heat * 0.5))

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

    if not factors:
        return 0.7  # Algorithm v3 B2: penalize missing data instead of being neutral

    return max(0.3, min(1.5, sum(factors) / len(factors)))


def _compute_safety_penalty(token: dict) -> float:
    """
    Returns a penalty multiplier [0.0, 1.0].
    1.0 = safe, 0.0 = extremely dangerous (score zeroed out).
    """
    penalty = 1.0

    # NOTE: mint_authority and freeze_authority are now handled by _apply_hard_gates()
    # (hard reject, score=0). No soft penalty needed here.

    # High insider concentration (bundled)
    insider_pct = token.get("insider_pct")
    if insider_pct is not None and insider_pct > 30:
        penalty *= max(0.2, 1.0 - (insider_pct - 30) / 70)

    # Top 10 holders own too much
    top10 = token.get("top10_holder_pct")
    if top10 is not None and top10 > 50:
        penalty *= max(0.3, 1.0 - (top10 - 50) / 50)

    # RugCheck risk score (0=safe, 10000=dangerous)
    risk = token.get("risk_score")
    if risk is not None and risk > 5000:
        penalty *= max(0.2, 1.0 - (risk - 5000) / 5000)

    # Multiple risk flags
    risk_count = token.get("risk_count", 0) or 0
    if risk_count >= 3:
        penalty *= 0.7

    # Jito slot-based bundle detection (high confidence, low false positives)
    jito_max_txns = token.get("jito_max_slot_txns") or 0
    if jito_max_txns >= 5:
        penalty *= 0.2  # definite Jito bundle
    elif jito_max_txns >= 3:
        penalty *= 0.4  # suspicious slot clustering
    else:
        # Fallback: balance-based bundle detection (higher false positive rate)
        bundle_detected = token.get("bundle_detected")
        bundle_pct = token.get("bundle_pct") or 0
        if bundle_detected:
            if bundle_pct > 20:
                penalty *= 0.3
            elif bundle_pct > 10:
                penalty *= 0.5

    # Phase 3: High Gini = wealth too concentrated
    helius_gini = token.get("helius_gini")
    if helius_gini is not None and helius_gini > 0.85:
        penalty *= 0.6

    # Phase 3: Too few holders = no organic community
    helius_holder_count = token.get("helius_holder_count")
    if helius_holder_count is not None and helius_holder_count < 50:
        penalty *= 0.7

    # Phase 3B: Whale concentration too high
    whale_total_pct = token.get("whale_total_pct")
    if whale_total_pct is not None and whale_total_pct > 60:
        penalty *= max(0.3, 1.0 - (whale_total_pct - 60) / 40)

    # Algorithm v3.1: Bubblemaps — low decentralization = clustered supply
    bb_score = token.get("bubblemaps_score")
    if bb_score is not None:
        if bb_score < 20:
            penalty *= 0.4  # very centralized
        elif bb_score < 40:
            penalty *= 0.7  # moderately centralized

    # Algorithm v3.1: Bubblemaps — largest wallet cluster holds too much
    bb_cluster_max = token.get("bubblemaps_cluster_max_pct")
    if bb_cluster_max is not None and bb_cluster_max > 30:
        penalty *= max(0.3, 1.0 - (bb_cluster_max - 30) / 70)

    return max(0.0, penalty)


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

    # Load KOL reputation scores
    kol_scores = get_kol_scores()
    if kol_scores:
        logger.info("Loaded KOL reputation scores for %d KOLs", len(kol_scores))

    # Load CA cache once for the entire aggregation cycle
    ca_cache = _load_ca_cache()

    token_data: dict[str, TokenStats] = defaultdict(lambda: {
        "mentions": 0,
        "sentiments": [],
        "groups": set(),
        "convictions": [],
        "hours_ago": [],
        "narratives": [],             # per-message narrative classification
        "narrative_confidences": [],  # per-message confidence scores
        "texts": [],                  # raw texts for narrative aggregation
        "msg_conviction_scores": [],  # per-message NLP conviction scores
        "price_target_count": 0,      # messages with price targets
        "hedging_count": 0,           # messages with hedging language
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

            tokens = extract_tokens(text, ca_cache=ca_cache)
            if not tokens:
                continue

            sentiment = calculate_sentiment(text)
            hours_ago = (now - date).total_seconds() / 3600

            narrative, narr_conf = classify_narrative(text, "")

            # Per-message conviction NLP (Sprint 6)
            msg_conv = _compute_message_conviction(text)

            for token in tokens:
                token_data[token]["mentions"] += 1
                token_data[token]["sentiments"].append(sentiment)
                token_data[token]["groups"].add(group_name)
                token_data[token]["convictions"].append(conviction)
                token_data[token]["hours_ago"].append(hours_ago)
                token_data[token]["msg_conviction_scores"].append(msg_conv["msg_conviction_score"])
                if msg_conv["has_price_target"]:
                    token_data[token]["price_target_count"] += 1
                if msg_conv["has_hedging"]:
                    token_data[token]["hedging_count"] += 1
                if narrative:
                    token_data[token]["narratives"].append(narrative)
                    token_data[token]["narrative_confidences"].append(narr_conf)

    # Persist CA cache after processing all messages
    _save_ca_cache(ca_cache)

    # Score & rank
    ranking: list[TokenRanking] = []

    # Collect narrative counts across all tokens to detect "hot" narratives this cycle
    global_narrative_counts: dict[str, int] = defaultdict(int)
    for data in token_data.values():
        for n in data["narratives"]:
            global_narrative_counts[n] += 1

    # Hot narrative = the narrative with most mentions this cycle (if >= 5 mentions)
    hot_narrative = None
    if global_narrative_counts:
        top_narr = max(global_narrative_counts, key=global_narrative_counts.get)
        if global_narrative_counts[top_narr] >= 5:
            hot_narrative = top_narr
            logger.info("Hot narrative this cycle: %s (%d mentions)", hot_narrative, global_narrative_counts[top_narr])

    for symbol, data in token_data.items():
        if data["mentions"] == 0:
            continue

        unique_kols = len(data["groups"])
        kol_consensus = min(1.0, unique_kols / (total_kols * 0.15))

        avg_sentiment = sum(data["sentiments"]) / len(data["sentiments"])
        sentiment_score = (avg_sentiment + 1) / 2

        # --- Dynamic conviction: hit_rate replaces static conviction ---
        group_list = list(data["groups"])

        # For each KOL: use dynamic hit_rate if available, else static conviction
        kol_conv_values = []
        kol_rep_values = []
        for g in group_list:
            hit_rate = kol_scores.get(g)  # None if < min_calls data
            if hit_rate is not None:
                # Dynamic: hit_rate 0.0→conv 5, hit_rate 0.5→conv 8, hit_rate 1.0→conv 10
                kol_conv_values.append(5.0 + hit_rate * 5.0)
                kol_rep_values.append(hit_rate)
            else:
                # Fallback: static conviction until enough data
                kol_conv_values.append(groups_conviction.get(g, 7))
                kol_rep_values.append(0.5)

        kol_reputation_avg = sum(kol_rep_values) / max(1, len(kol_rep_values))
        effective_conviction = sum(kol_conv_values) / max(1, len(kol_conv_values))

        # Per-message NLP conviction amplifier/dampener (Sprint 6)
        msg_convictions = data.get("msg_conviction_scores", [])
        avg_msg_conv = sum(msg_convictions) / max(1, len(msg_convictions)) if msg_convictions else 1.0
        effective_conviction *= avg_msg_conv  # amplifies (>1.0) or dampens (<1.0)

        avg_conviction = sum(data["convictions"]) / len(data["convictions"])
        conviction_score = max(0, min(1, (effective_conviction - 5) / 5))  # Scale 5-10 → 0-1

        breadth_score = min(1.0, data["mentions"] / 30)

        # --- Narrative classification ---
        token_narrative = None
        narrative_confidence = 0.0
        if data["narratives"]:
            narr_counts = defaultdict(int)
            for n in data["narratives"]:
                narr_counts[n] += 1
            token_narrative = max(narr_counts, key=narr_counts.get)
            # Average confidence across messages with the dominant narrative
            matching_confs = [
                c for n, c in zip(data["narratives"], data["narrative_confidences"])
                if n == token_narrative
            ]
            if matching_confs:
                narrative_confidence = round(sum(matching_confs) / len(matching_confs), 3)

        narrative_is_hot = 1 if (token_narrative and token_narrative == hot_narrative) else 0

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

        # --- Base Telegram scores (3 modes) ---

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

        # Recency score: two-phase decay — gentle first 6h, aggressive after
        recency_weights = [_two_phase_decay(h) for h in data["hours_ago"]]
        recency_score = min(1.0, sum(recency_weights) / 10)

        # Momentum mode: what's trending NOW — driven by recency + KOL quality
        raw_momentum = (
            0.50 * recency_score
            + 0.10 * sentiment_score
            + 0.30 * kol_consensus
            + 0.10 * breadth_score
        )
        score_momentum = min(100, max(0, int(raw_momentum * 100)))

        trend = "up" if avg_sentiment > 0.15 else ("down" if avg_sentiment < -0.15 else "stable")

        groups_with_conv = sorted(
            [(g, groups_conviction.get(g, 7)) for g in data["groups"]],
            key=lambda x: x[1],
            reverse=True,
        )
        top_kols_list = [g[0] for g in groups_with_conv]  # ALL KOLs, not just top 5

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
            # Phase 2: Narrative
            "narrative": token_narrative,
            "narrative_is_hot": narrative_is_hot,
            # Phase 3B: Narrative confidence
            "narrative_confidence": narrative_confidence,
            # Algorithm v2: Per-message conviction NLP (Sprint 6)
            "msg_conviction_avg": round(avg_msg_conv, 2),
            "price_target_count": data.get("price_target_count", 0),
            "hedging_count": data.get("hedging_count", 0),
            # Algorithm v3 A3: Sentiment consistency
            "sentiment_consistency": round(sentiment_consistency, 3),
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

    # Phase 3: Helius enrichment (bundle detection + holder quality + whale tracking)
    enrich_tokens_helius(ranking)

    # Phase 3B: Jupiter enrichment (tradeability + price impact + routes)
    enrich_tokens_jupiter(ranking)

    # Algorithm v3.1: Bubblemaps enrichment (wallet clustering + decentralization score)
    enrich_tokens_bubblemaps(ranking)

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

    # === Algorithm v2: Hard gates (remove dangerous tokens entirely) ===
    ranking = _apply_hard_gates(ranking)

    # === Algorithm v2: Wash trading score (before multipliers) ===
    wash_gated = 0
    for token in ranking:
        token["wash_trading_score"] = round(_compute_wash_trading_score(token), 3)
    # Hard gate: wash_score > 0.8 = almost certainly fake volume
    before_wash = len(ranking)
    ranking = [t for t in ranking if t.get("wash_trading_score", 0) <= 0.8]
    wash_gated = before_wash - len(ranking)
    if wash_gated:
        logger.info("Wash trading gate removed %d tokens (score > 0.8)", wash_gated)

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

        # Hot narrative bonus
        narr_bonus = 1.05 if token.get("narrative_is_hot") else 1.0

        # Wash trading soft penalty (tokens with score 0-0.8 that passed hard gate)
        wash_score = token.get("wash_trading_score", 0)
        wash_pen = max(0.3, 1.0 - wash_score)

        # PVP penalty: same-name tokens launched within 4h = copycat confusion
        pvp_recent = token.get("pvp_recent_count", 0)
        pvp_pen = 1.0 / (1 + 0.2 * pvp_recent)

        # Algorithm v3 A5: Artificial pump severe penalty
        pump_pen = 0.2 if token.get("is_artificial_pump") else 1.0

        # Algorithm v3.1: "Already pumped" penalty — diminishing upside
        # Token that already did +200%: risk/reward for another 2x is worse
        # Degressive: 200% → 1.0, 350% → 0.7, 500%+ → 0.4 (floor)
        already_pumped_pen = 1.0
        pc24 = token.get("price_change_24h")
        if pc24 is not None and pc24 > 200:
            already_pumped_pen = max(0.4, 1.0 - (pc24 - 200) / 500)
        token["already_pumped_penalty"] = round(already_pumped_pen, 3)

        combined = onchain_mult * safety_pen * pump_bonus * narr_bonus * wash_pen * pvp_pen * pump_pen * already_pumped_pen

        # Apply to all three scoring modes
        token["score"] = min(100, max(0, int(token["score"] * combined)))
        token["score_conviction"] = min(100, max(0, int(token["score_conviction"] * combined)))
        token["score_momentum"] = min(100, max(0, int(token["score_momentum"] * combined)))

    logger.info(
        "Applied on-chain multipliers & safety penalties to %d tokens",
        len(ranking),
    )

    # Apply ML scoring if model is available (overrides manual score)
    _apply_ml_scores(ranking)

    ranking.sort(key=lambda x: (x["score"], x["mentions"]), reverse=True)
    return ranking
