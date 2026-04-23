"""Multi-chain address detection + normalization.

v14 — introduced alongside the ETH L1 shadow pipeline (Sprint #ETH-1).

Chain classification is deterministic from the address shape:
  - 0x + 40 hex chars          -> ethereum
  - base58 32-44 chars          -> solana
  - anything else               -> None (unrecognized)

The Ethereum check is strict on length to avoid accidentally promoting
Solana-style bogus tokens (e.g. "0x1" or "0xDEADBEEF") that would silently
be queried against ETH endpoints and waste rate-limit budget.

Normalization:
  - Ethereum addresses stored lowercase. DB dedup relies on case-insensitive
    matching; Etherscan/DexScreener accept both cases but return lowercase.
  - Solana addresses case-preserved — base58 is case-sensitive.
"""
from __future__ import annotations

import re
from typing import Literal, Optional

# Strict ETH address: exactly 0x + 40 hex chars, word-bounded. This avoids
# matching tx hashes (64 hex) or partial matches inside longer blobs.
ETH_CA_REGEX = re.compile(r"\b(0x[a-fA-F0-9]{40})\b")

# Solana base58: duplicated here from pipeline.py as a read-only reference.
# The authoritative regex for batch extraction still lives in pipeline.py;
# this copy is used only for per-address classification where we must not
# import pipeline (circular deps with enrich modules).
SOLANA_CA_REGEX = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")

Chain = Literal["solana", "ethereum"]


def detect_chain(addr: str) -> Optional[Chain]:
    """Classify an address. Returns None for unrecognized shapes.

    Callers should treat None as "skip this token" — never default to solana
    silently, which would silently route an ETH token to Solana RPCs.
    """
    if not addr or not isinstance(addr, str):
        return None
    addr = addr.strip()
    # ETH: strict length check. 0x alone or short hex not accepted.
    if addr.startswith("0x") and len(addr) == 42:
        if re.fullmatch(r"0x[a-fA-F0-9]{40}", addr):
            return "ethereum"
        return None
    # Solana: base58 32-44 chars, no 0/O/I/l (already enforced by the regex)
    if SOLANA_CA_REGEX.fullmatch(addr):
        return "solana"
    return None


def normalize_address(addr: str, chain: Optional[Chain] = None) -> str:
    """Canonicalize address for storage/comparison.

    ETH -> lowercased (case-insensitive address space).
    Solana -> untouched (base58 case-sensitive).

    If chain isn't provided, it's auto-detected. If auto-detect fails, the
    string is returned unchanged — caller is responsible for validating.
    """
    if not addr:
        return addr
    c = chain or detect_chain(addr)
    if c == "ethereum":
        return addr.lower()
    return addr


def extract_eth_addresses(text: str) -> list[str]:
    """Return all distinct lowercase ETH addresses in text, order-preserving."""
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for m in ETH_CA_REGEX.finditer(text):
        a = m.group(1).lower()
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out
