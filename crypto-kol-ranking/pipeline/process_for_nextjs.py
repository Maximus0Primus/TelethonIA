"""
Process TelethonIA Telegram exports for Next.js ranking display.

This script:
1. Loads messages from Telegram JSON exports
2. Extracts token mentions ($XXX)
3. Calculates sentiment (VADER + crypto lexicon)
4. Aggregates data per token by time window
5. Calculates super scores
6. Exports ranking_data.json for the Next.js app
"""

import json
import re
import sys
import glob
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from pathlib import Path
from typing import TypedDict, Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# === CONFIGURATION ===

# Regex for extracting tokens like $PEPE, $SOL, $BTC
# Must start with $, then a letter, then 1-14 alphanumeric chars
TOKEN_REGEX = re.compile(r"(?<![A-Z0-9_])\$([A-Z][A-Z0-9]{1,14})\b")

# Crypto-specific lexicon for sentiment boosting
CRYPTO_LEXICON = {
    # Bullish signals
    "bullish": 0.5,
    "moon": 0.4,
    "mooning": 0.45,
    "pump": 0.3,
    "pumping": 0.35,
    "gem": 0.35,
    "alpha": 0.3,
    "lfg": 0.4,
    "send it": 0.35,
    "printing": 0.4,
    "cook": 0.35,
    "cooking": 0.4,
    "listing": 0.55,
    "partnership": 0.4,
    "airdrop": 0.25,
    "conviction": 0.4,
    "early": 0.3,
    "buy": 0.2,
    "long": 0.25,
    "100x": 0.5,
    "10x": 0.4,
    "runner": 0.35,
    "winner": 0.35,
    "insane": 0.3,
    "massive": 0.25,

    # Bearish signals
    "rug": -0.8,
    "rugged": -0.85,
    "scam": -0.75,
    "scammer": -0.8,
    "dump": -0.6,
    "dumping": -0.65,
    "honeypot": -0.8,
    "avoid": -0.5,
    "warning": -0.4,
    "careful": -0.3,
    "sell": -0.2,
    "short": -0.25,
    "dead": -0.6,
    "rekt": -0.5,
    "exit": -0.4,
}

# Groups with their conviction scores (from exportfinaljson.py)
GROUPS_CONVICTION = {
    "missorplays": 7,
    "slingdeez": 8,
    "overdose_gems_calls": 10,
    "marcellcooks": 9,
    "shahlito": 7,
    "sadcatgamble": 7,
    "ghastlygems": 8,
    "archercallz": 8,
    "LevisAlpha": 8,
    "MarkDegens": 7,
    "darkocalls": 8,
    "kweensjournal": 8,
    "explorer_gems": 7,
    "ArcaneGems": 8,
    "veigarcalls": 7,
    "watisdes": 7,
    "Luca_Apes": 7,
    "wuziemakesmoney": 7,
    "BatmanSafuCalls": 7,
    "chiggajogambles": 7,
    "dylansdegens": 8,
    "AnimeGems": 7,
    "robogems": 7,
    "ALSTEIN_GEMCLUB": 8,
    "PoseidonTAA": 9,
    "canisprintoooors": 7,
    "jsdao": 8,
    "MaybachCalls": 8,
    "slingTA": 7,
    "MaybachGambleCalls": 7,
    "cryptorugmuncher": 10,
    "inside_calls": 8,
    "BossmanCallsOfficial": 8,
    "bounty_journal": 8,
    "cryptotalkwithfrog": 7,
    "StereoCalls": 8,
    "CarnagecallsGambles": 7,
    "PowsGemCalls": 8,
    "houseofdegeneracy": 6,
    "CatfishcallsbyPoe": 8,
    "spidersjournal": 8,
    "KittysKasino": 7,
    "cryptolyxecalls": 8,
    "izzycooks": 8,
    "cryptowhalecalls7": 7,
    "Carnagecalls": 9,
    "wulfcryptocalls": 8,
    "waldosalpha": 7,
    "thetonymoontana": 10,
    "OnyxxGems": 8,
    "SaviourCALLS": 7,
    "eunicalls": 8,
    "lollycalls": 7,
    "leoclub168c": 7,
    "TheCabalCalls": 8,
    "sugarydick": 8,
    "leoclub168g": 7,
    "jadendegens": 7,
    "certifiedprintor": 8,
    "MarkGems": 9,
    "LittleMustachoCalls": 8,
}

# Tokens to exclude (common false positives)
EXCLUDED_TOKENS = {
    "USD", "USDT", "USDC", "SOL", "ETH", "BTC", "BNB",  # Major coins (optional, can remove)
    "CA", "LP", "MC", "ATH", "ATL", "FDV", "TVL",  # Crypto abbreviations
    "PNL", "ROI", "APY", "APR",  # Finance terms
    "CEO", "COO", "CTO",  # Titles
    "URL", "API", "NFT", "DAO",  # Tech terms
    "GMT", "UTC", "EST", "PST",  # Timezones
    "DM", "RT", "TG",  # Social media
}


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
    messages: list[dict]


class TokenOutput(TypedDict):
    rank: int
    symbol: str
    score: int
    mentions: int
    uniqueKols: int
    sentiment: float
    trend: str
    change24h: float
    topKols: list[str]


# Initialize VADER analyzer
vader_analyzer = SentimentIntensityAnalyzer()


def load_messages(json_path: str) -> list[Message]:
    """Load messages from Telegram JSON export."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    messages: list[Message] = []

    for group_name, group_messages in data.items():
        # Get conviction from our mapping, default to 7
        conviction = GROUPS_CONVICTION.get(group_name, 7)

        for msg in group_messages:
            # Parse date - handle both ISO format and timezone-aware strings
            date_str = msg.get("date", "")
            try:
                # Try parsing ISO format with timezone
                if date_str.endswith("Z"):
                    date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                elif "+" in date_str or date_str.endswith("00:00"):
                    date = datetime.fromisoformat(date_str)
                else:
                    # Assume UTC if no timezone
                    date = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
            except (ValueError, AttributeError):
                # Skip messages with invalid dates
                continue

            text = msg.get("text", "")
            if not text or len(text) < 5:  # Skip very short messages
                continue

            messages.append({
                "group": group_name,
                "conviction": conviction,
                "text": text,
                "date": date,
            })

    return messages


def extract_tokens(text: str) -> list[str]:
    """Extract token symbols ($XXX) from text."""
    # Find all matches
    matches = TOKEN_REGEX.findall(text.upper())

    # Filter out excluded tokens and duplicates
    tokens = []
    seen = set()
    for match in matches:
        symbol = f"${match}"
        if match not in EXCLUDED_TOKENS and symbol not in seen:
            tokens.append(symbol)
            seen.add(symbol)

    return tokens


def calculate_sentiment(text: str) -> float:
    """Calculate sentiment using VADER + crypto lexicon boost."""
    # VADER sentiment (-1 to 1)
    vader_score = vader_analyzer.polarity_scores(text)["compound"]

    # Crypto lexicon boost
    text_lower = text.lower()
    lexicon_boost = 0.0
    matches = 0

    for word, weight in CRYPTO_LEXICON.items():
        if word in text_lower:
            lexicon_boost += weight
            matches += 1

    # Normalize lexicon boost to -1 to 1 range
    if matches > 0:
        lexicon_boost = max(-1, min(1, lexicon_boost / max(1, matches * 0.5)))

    # Blend: 70% VADER, 30% crypto lexicon
    final = 0.7 * vader_score + 0.3 * lexicon_boost

    return max(-1, min(1, final))


def aggregate_tokens(messages: list[Message], hours: int) -> dict[str, TokenStats]:
    """Aggregate token data for a specific time window."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours)

    token_data: dict[str, TokenStats] = defaultdict(lambda: {
        "mentions": 0,
        "sentiments": [],
        "groups": set(),
        "convictions": [],
        "messages": [],
    })

    for msg in messages:
        # Skip messages outside time window
        if msg["date"] < cutoff:
            continue

        # Extract tokens from message
        tokens = extract_tokens(msg["text"])
        if not tokens:
            continue

        # Calculate sentiment for this message
        sentiment = calculate_sentiment(msg["text"])

        # Add data for each token mentioned
        for token in tokens:
            token_data[token]["mentions"] += 1
            token_data[token]["sentiments"].append(sentiment)
            token_data[token]["groups"].add(msg["group"])
            token_data[token]["convictions"].append(msg["conviction"])
            token_data[token]["messages"].append({
                "text": msg["text"][:200],  # Truncate for storage
                "group": msg["group"],
                "sentiment": round(sentiment, 3),
                "date": msg["date"].isoformat(),
            })

    return token_data


def calculate_score(data: TokenStats, total_kols: int) -> int:
    """
    Calculate super score (0-100) based on:
    - KOL consensus (how many unique KOLs mentioned it)
    - Sentiment (average sentiment of mentions)
    - Conviction (average conviction of mentioning groups)
    - Breadth (total mentions as engagement proxy)
    """
    if data["mentions"] == 0:
        return 0

    # Component 1: KOL Consensus (35%)
    # More unique KOLs = higher score
    unique_kols = len(data["groups"])
    kol_consensus = min(1.0, unique_kols / (total_kols * 0.15))  # Cap at ~9 KOLs

    # Component 2: Sentiment (25%)
    # Convert -1 to 1 range to 0 to 1
    avg_sentiment = sum(data["sentiments"]) / len(data["sentiments"])
    sentiment_score = (avg_sentiment + 1) / 2

    # Component 3: Conviction (25%)
    # Normalize conviction from 6-10 range to 0-1
    avg_conviction = sum(data["convictions"]) / len(data["convictions"])
    conviction_score = (avg_conviction - 6) / 4  # 6->0, 10->1
    conviction_score = max(0, min(1, conviction_score))

    # Component 4: Mention breadth (15%)
    # More mentions = higher engagement (capped)
    breadth_score = min(1.0, data["mentions"] / 30)

    # Weighted combination
    raw_score = (
        0.35 * kol_consensus +
        0.25 * sentiment_score +
        0.25 * conviction_score +
        0.15 * breadth_score
    )

    # Scale to 0-100
    return min(100, max(0, int(raw_score * 100)))


def determine_trend(sentiment: float) -> str:
    """Determine trend based on sentiment."""
    if sentiment > 0.15:
        return "up"
    elif sentiment < -0.15:
        return "down"
    return "stable"


def generate_ranking_json(messages: list[Message], output_path: str) -> None:
    """Generate the final ranking JSON for Next.js."""
    time_windows = {"3h": 3, "6h": 6, "12h": 12, "24h": 24, "48h": 48, "7d": 168}
    total_kols = len(GROUPS_CONVICTION)

    result = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "stats": {},
        "tokens": {},
    }

    for window_name, hours in time_windows.items():
        print(f"  Processing {window_name} window...")
        token_data = aggregate_tokens(messages, hours)

        # Build ranking list
        ranking: list[TokenOutput] = []

        for symbol, data in token_data.items():
            score = calculate_score(data, total_kols)
            avg_sentiment = sum(data["sentiments"]) / len(data["sentiments"])

            # Sort groups by conviction for top KOLs
            groups_with_conviction = [
                (g, GROUPS_CONVICTION.get(g, 7))
                for g in data["groups"]
            ]
            groups_with_conviction.sort(key=lambda x: x[1], reverse=True)
            top_kols = [g[0] for g in groups_with_conviction[:5]]

            ranking.append({
                "rank": 0,  # Will be set after sorting
                "symbol": symbol,
                "score": score,
                "mentions": data["mentions"],
                "uniqueKols": len(data["groups"]),
                "sentiment": round(avg_sentiment, 3),
                "trend": determine_trend(avg_sentiment),
                "change24h": 0.0,  # TODO: Historical comparison
                "topKols": top_kols,
            })

        # Sort by score descending, then by mentions as tiebreaker
        ranking.sort(key=lambda x: (x["score"], x["mentions"]), reverse=True)

        # Assign ranks
        for i, token in enumerate(ranking):
            token["rank"] = i + 1

        result["tokens"][window_name] = ranking
        print(f"    Found {len(ranking)} tokens")

    # Calculate global stats from 24h data
    data_24h = result["tokens"].get("24h", [])
    if data_24h:
        result["stats"] = {
            "totalTokens": len(data_24h),
            "totalMentions": sum(t["mentions"] for t in data_24h),
            "avgSentiment": round(
                sum(t["sentiment"] for t in data_24h) / max(1, len(data_24h)) * 100,
                1
            ),  # Convert to percentage for display
            "totalKols": total_kols,
        }
    else:
        result["stats"] = {
            "totalTokens": 0,
            "totalMentions": 0,
            "avgSentiment": 0,
            "totalKols": total_kols,
        }

    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"Exported to: {output_path}")
    print(f"Total tokens (24h): {result['stats']['totalTokens']}")
    print(f"Total mentions (24h): {result['stats']['totalMentions']}")
    print(f"Avg sentiment: {result['stats']['avgSentiment']}%")


def find_latest_export(base_path: str) -> Optional[str]:
    """Find the most recent messages_export_*.json file."""
    pattern = str(Path(base_path) / "messages_export_*.json")
    files = glob.glob(pattern)

    if not files:
        return None

    # Sort by modification time, newest first
    files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    return files[0]


def main():
    # Determine input file
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        # Handle glob patterns
        if "*" in input_path:
            files = glob.glob(input_path)
            if files:
                files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
                input_path = files[0]
            else:
                print(f"No files found matching: {input_path}")
                sys.exit(1)
    else:
        # Look for export in parent directory
        base_path = Path(__file__).parent.parent.parent
        input_path = find_latest_export(str(base_path))

        if not input_path:
            print("No messages_export_*.json found.")
            print("Usage: python process_for_nextjs.py [path_to_export.json]")
            print("\nOr run exportfinaljson.py first to generate the export.")
            sys.exit(1)

    print(f"Loading messages from: {input_path}")

    # Determine output path
    output_path = Path(__file__).parent.parent / "public" / "data" / "ranking_data.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process
    messages = load_messages(input_path)
    print(f"Loaded {len(messages)} messages from {len(set(m['group'] for m in messages))} groups")

    generate_ranking_json(messages, str(output_path))
    print("\nDone!")


if __name__ == "__main__":
    main()
