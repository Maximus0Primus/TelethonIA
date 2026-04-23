"""Tests for chain_detect — address classification + normalization."""
import pytest

from chain_detect import (
    detect_chain,
    normalize_address,
    extract_eth_addresses,
    ETH_CA_REGEX,
)


# Realistic addresses (USDC on each chain)
ETH_USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
ETH_USDC_LOWER = ETH_USDC.lower()
SOL_USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


class TestDetectChain:
    def test_eth_mainnet_address(self):
        assert detect_chain(ETH_USDC) == "ethereum"

    def test_eth_address_lowercase(self):
        assert detect_chain(ETH_USDC_LOWER) == "ethereum"

    def test_eth_address_all_zeros(self):
        # Burn address — still valid ETH format
        assert detect_chain("0x0000000000000000000000000000000000000000") == "ethereum"

    def test_solana_mainnet_address(self):
        assert detect_chain(SOL_USDC) == "solana"

    def test_too_short_eth(self):
        assert detect_chain("0xDEADBEEF") is None

    def test_too_long_eth(self):
        # 0x + 64 hex = tx hash, not an address
        assert detect_chain("0x" + "a" * 64) is None

    def test_eth_with_non_hex_char(self):
        # 'z' is not hex — fails fullmatch
        assert detect_chain("0xz" + "a" * 39) is None

    def test_solana_too_short(self):
        assert detect_chain("abc123") is None

    def test_solana_with_zero(self):
        # base58 excludes '0' — even if length right, '0' means not base58
        bad = "0" + SOL_USDC[1:]
        assert detect_chain(bad) is None

    def test_empty(self):
        assert detect_chain("") is None

    def test_none(self):
        assert detect_chain(None) is None  # type: ignore[arg-type]

    def test_whitespace_stripped(self):
        assert detect_chain(f"  {ETH_USDC}  ") == "ethereum"
        assert detect_chain(f" {SOL_USDC} ") == "solana"

    def test_non_string(self):
        assert detect_chain(12345) is None  # type: ignore[arg-type]


class TestNormalize:
    def test_eth_lowercased(self):
        assert normalize_address(ETH_USDC) == ETH_USDC_LOWER

    def test_eth_already_lower_unchanged(self):
        assert normalize_address(ETH_USDC_LOWER) == ETH_USDC_LOWER

    def test_solana_case_preserved(self):
        # Solana uses case-sensitive base58 — must NOT lowercase
        assert normalize_address(SOL_USDC) == SOL_USDC

    def test_empty(self):
        assert normalize_address("") == ""

    def test_explicit_chain_overrides_detection(self):
        # If caller forces ethereum on a string we can't parse, trust them
        # (the module contract is "caller validates shape; we just case-fold")
        weird = "0xABC"
        # Without hint: detect_chain says None, normalize leaves it
        assert normalize_address(weird) == weird
        # With explicit hint: normalize obeys
        assert normalize_address(weird, chain="ethereum") == "0xabc"


class TestExtractEthAddresses:
    def test_single_address_in_text(self):
        text = f"check this token {ETH_USDC} fire"
        assert extract_eth_addresses(text) == [ETH_USDC_LOWER]

    def test_multiple_distinct(self):
        a1 = "0x" + "a" * 40
        a2 = "0x" + "b" * 40
        text = f"two calls: {a1} and {a2} both pumping"
        assert extract_eth_addresses(text) == [a1, a2]

    def test_dedup(self):
        # Same address mentioned twice, once lowercase once mixed
        mixed = ETH_USDC  # has uppercase
        text = f"{mixed} and again {ETH_USDC_LOWER}"
        result = extract_eth_addresses(text)
        assert result == [ETH_USDC_LOWER]

    def test_empty(self):
        assert extract_eth_addresses("") == []
        assert extract_eth_addresses(None) == []  # type: ignore[arg-type]

    def test_ignores_short_hex(self):
        text = f"gas price 0xdeadbeef high, token {ETH_USDC}"
        assert extract_eth_addresses(text) == [ETH_USDC_LOWER]

    def test_does_not_match_tx_hash(self):
        tx_hash = "0x" + "a" * 64  # tx hash length
        assert extract_eth_addresses(f"see tx {tx_hash}") == []

    def test_ignores_solana_addresses(self):
        text = f"sol {SOL_USDC} and eth {ETH_USDC}"
        assert extract_eth_addresses(text) == [ETH_USDC_LOWER]
