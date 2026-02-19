"""
Debug dump module: saves raw messages, extracted tokens, and ranking output
as timestamped JSON files for pipeline diagnosis.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from pipeline import (
    extract_tokens,
    calculate_sentiment,
    _compute_message_conviction,
    TOKEN_REGEX,
    CA_REGEX,
    DEXSCREENER_URL_REGEX,
    PUMP_FUN_URL_REGEX,
    GMGN_URL_REGEX,
    PHOTON_URL_REGEX,
    KNOWN_PROGRAM_ADDRESSES,
    EXCLUDED_TOKENS,
    _resolve_ca_to_symbol,
    _resolve_pair_to_symbol_and_ca,
    _load_ca_cache,
)

logger = logging.getLogger(__name__)


_DEFAULT_DUMP_DIR = Path(__file__).parent / "debug_dumps"


def dump_debug_data(
    messages_data: dict[str, list[dict]],
    ranking_by_window: dict[str, list[dict]],
    output_dir: str | Path | None = None,
) -> Path:
    """
    Save a complete debug dump for diagnosis:
    1. raw_messages.json   - ALL raw messages by group
    2. extracted_tokens.json - Per message: tokens extracted + sentiment + conviction
    3. ranking_output.json  - Final ranking with all fields per window

    Returns the output directory path.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(output_dir) / ts if output_dir else _DEFAULT_DUMP_DIR / ts
    out.mkdir(parents=True, exist_ok=True)

    # 1. Raw messages
    raw_path = out / "raw_messages.json"
    raw_path.write_text(
        json.dumps(messages_data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    total_msgs = sum(len(v) for v in messages_data.values())
    logger.info("Dump: %d raw messages from %d groups -> %s", total_msgs, len(messages_data), raw_path)

    # 2. Build confirmed symbols (same Phase 1 as pipeline) for accurate dump
    ca_cache = _load_ca_cache()
    confirmed_symbols: set[str] = set()
    for group, msgs in messages_data.items():
        for msg in msgs:
            text = msg.get("text", "")
            if not text or len(text) < 3:
                continue
            for match in TOKEN_REGEX.findall(text):
                upper = match.upper()
                if upper.isascii() and upper not in EXCLUDED_TOKENS:
                    confirmed_symbols.add(upper)
            for match in CA_REGEX.findall(text):
                if match in KNOWN_PROGRAM_ADDRESSES:
                    continue
                resolved = _resolve_ca_to_symbol(match, ca_cache)
                if resolved and resolved not in EXCLUDED_TOKENS:
                    confirmed_symbols.add(resolved)
            for chain, pair_addr in DEXSCREENER_URL_REGEX.findall(text):
                resolved, _ca = _resolve_pair_to_symbol_and_ca(chain, pair_addr, ca_cache)
                if resolved and resolved not in EXCLUDED_TOKENS:
                    confirmed_symbols.add(resolved)
            for pump_addr in PUMP_FUN_URL_REGEX.findall(text):
                resolved = _resolve_ca_to_symbol(pump_addr, ca_cache)
                if resolved and resolved not in EXCLUDED_TOKENS:
                    confirmed_symbols.add(resolved)
            for gmgn_addr in GMGN_URL_REGEX.findall(text):
                resolved = _resolve_ca_to_symbol(gmgn_addr, ca_cache)
                if resolved and resolved not in EXCLUDED_TOKENS:
                    confirmed_symbols.add(resolved)
            for photon_addr in PHOTON_URL_REGEX.findall(text):
                resolved, _ca = _resolve_pair_to_symbol_and_ca("solana", photon_addr, ca_cache)
                if resolved and resolved not in EXCLUDED_TOKENS:
                    confirmed_symbols.add(resolved)

    # 3. Extracted tokens per message (with confirmation gate)
    extracted = []
    for group, msgs in messages_data.items():
        for msg in msgs:
            text = msg.get("text", "")
            if not text or len(text) < 3:
                continue
            tokens = extract_tokens(text, ca_cache=ca_cache, confirmed_symbols=confirmed_symbols)
            sentiment = calculate_sentiment(text)
            conv = _compute_message_conviction(text)
            extracted.append({
                "group": group,
                "date": msg.get("date", ""),
                "text": text[:500],  # truncate for readability
                "tokens_extracted": tokens,
                "sentiment": round(sentiment, 3),
                "is_positive_mention": sentiment >= -0.3,
                "msg_conviction": conv["msg_conviction_score"],
                "has_price_target": conv["has_price_target"],
                "has_hedging": conv["has_hedging"],
            })

    # Save confirmed symbols for reference
    meta = {
        "confirmed_symbols": sorted(confirmed_symbols),
        "confirmed_count": len(confirmed_symbols),
        "total_messages": total_msgs,
        "total_groups": len(messages_data),
    }

    tok_path = out / "extracted_tokens.json"
    tok_path.write_text(
        json.dumps({"meta": meta, "messages": extracted}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info(
        "Dump: %d messages analyzed, %d confirmed symbols -> %s",
        len(extracted), len(confirmed_symbols), tok_path,
    )

    # 4. Ranking output (serializable copy)
    ranking_serializable = {}
    for window, tokens in ranking_by_window.items():
        ranking_serializable[window] = [
            {k: v for k, v in t.items() if k != "_total_kols" and not callable(v)}
            for t in tokens
        ]
        # Convert sets to lists for JSON
        for t in ranking_serializable[window]:
            for k, v in t.items():
                if isinstance(v, set):
                    t[k] = list(v)

    rank_path = out / "ranking_output.json"
    rank_path.write_text(
        json.dumps(ranking_serializable, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info("Dump: %d windows -> %s", len(ranking_serializable), rank_path)

    logger.info("Debug dump complete: %s", out)
    return out
