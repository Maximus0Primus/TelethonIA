"""Multi-chain address detection + normalization.

v14 — introduced alongside the ETH L1 shadow pipeline (Sprint #ETH-1).
v14e — BSC + Base scaffolding. All three EVM chains share the 0x+40-hex
       address shape, so shape alone cannot disambiguate — disambiguation
       happens via DexScreener `chainId` lookup (resolve_evm_chain).

Chain classification is deterministic from the address shape:
  - 0x + 40 hex chars          -> evm (default ethereum, refine via DS lookup)
  - base58 32-44 chars          -> solana
  - anything else               -> None (unrecognized)

The Ethereum check is strict on length to avoid accidentally promoting
Solana-style bogus tokens (e.g. "0x1" or "0xDEADBEEF") that would silently
be queried against ETH endpoints and waste rate-limit budget.

Normalization:
  - EVM addresses stored lowercase. DB dedup relies on case-insensitive
    matching; explorers accept both cases but return lowercase.
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

Chain = Literal["solana", "ethereum", "bsc", "base"]

# EVM chainId -> chain name mapping used when resolving a 0x address via
# DexScreener. DS returns `chainId` strings like "ethereum", "bsc", "base".
_EVM_CHAIN_IDS = {"ethereum", "bsc", "base"}


def detect_chain(addr: str) -> Optional[Chain]:
    """Classify an address by shape alone.

    Callers should treat None as "skip this token" — never default to solana
    silently, which would silently route an ETH token to Solana RPCs.

    IMPORTANT (v14e): a 0x address returns "ethereum" by default for backward
    compat with v14, but it could actually be BSC or Base. When chain really
    matters (price fetch, live trading), call resolve_evm_chain() to refine
    against DexScreener's chainId.
    """
    if not addr or not isinstance(addr, str):
        return None
    addr = addr.strip()
    # EVM: strict length check. 0x alone or short hex not accepted.
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

    EVM -> lowercased (case-insensitive address space).
    Solana -> untouched (base58 case-sensitive).

    If chain isn't provided, it's auto-detected. If auto-detect fails, the
    string is returned unchanged — caller is responsible for validating.
    """
    if not addr:
        return addr
    c = chain or detect_chain(addr)
    if c in _EVM_CHAIN_IDS:
        return addr.lower()
    return addr


def resolve_evm_chain(addr: str, timeout: float = 5.0) -> Optional[Chain]:
    """Disambiguate a 0x address via DexScreener to find its actual EVM chain.

    Returns "ethereum" / "bsc" / "base" based on the first pair's chainId,
    or None on failure. This is the ONLY reliable way to tell these apart —
    all three use the same 0x+40hex shape, and an address CAN exist on more
    than one chain (bridged tokens). We take the first pair returned by DS,
    which ranks by liquidity.

    Kept minimal on purpose: a full multi-chain token resolver would cache
    the result, handle CA existing on multiple chains, and score by volume.
    v14e only needs "is this BSC or Base?" before routing a price fetch.
    """
    if not addr or not isinstance(addr, str) or not addr.startswith("0x"):
        return None
    try:
        import requests
        # DexScreener's generic /latest/dex/tokens/{addr} returns all pairs
        # across chains. We pick the top one by liquidity.
        resp = requests.get(
            f"https://api.dexscreener.com/latest/dex/tokens/{addr.lower()}",
            timeout=timeout,
        )
        if resp.status_code != 200:
            return None
        pairs = (resp.json() or {}).get("pairs") or []
        if not pairs:
            return None
        # Pick chainId with highest liquidity
        top = max(pairs, key=lambda p: float((p.get("liquidity") or {}).get("usd") or 0))
        cid = (top.get("chainId") or "").lower()
        if cid in _EVM_CHAIN_IDS:
            return cid  # type: ignore[return-value]
        return None
    except Exception:
        return None


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
