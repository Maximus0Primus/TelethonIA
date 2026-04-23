"""v14: Tests for ETH L1 extraction + chain-gated strategy filtering.

Lives in scraper/tests/ so it runs via the existing pytest.ini under
scraper/, alongside test_pipeline.py and test_paper_trader.py.
"""
from unittest.mock import patch

import pytest


ETH_USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
SOL_USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


class TestExtractTokensMultichain:
    """Verify extract_tokens finds both chains from a single message."""

    def _fake_resolve(self, addr, cache, chain="solana"):
        # Fake resolver: just use addr prefix as symbol. Writes to cache the
        # same way prod does so the caching path gets exercised too.
        from scraper import pipeline as p  # noqa
        key = addr if chain == "solana" else f"{chain}:{addr}"
        cache[key] = {
            "symbol": ("ETHTOK" if chain == "ethereum" else "SOLTOK"),
            "resolved_at": 9_999_999_999,  # far future so cache hit
            "chain": chain,
        }
        return cache[key]["symbol"]

    def test_extracts_eth_and_solana_from_same_text(self):
        from pipeline import extract_tokens

        text = f"pumping hard: {ETH_USDC} and {SOL_USDC}"
        cache: dict = {}
        # Pre-populate cache so no real HTTP happens
        cache[ETH_USDC.lower()] = {
            "symbol": "ETHTOK", "resolved_at": 9_999_999_999, "chain": "ethereum",
        }
        cache[f"ethereum:{ETH_USDC.lower()}"] = {
            "symbol": "ETHTOK", "resolved_at": 9_999_999_999, "chain": "ethereum",
        }
        cache[SOL_USDC] = {
            "symbol": "SOLTOK", "resolved_at": 9_999_999_999, "chain": "solana",
        }
        tokens = extract_tokens(text, ca_cache=cache)
        # Returns list of (symbol, source, ca). Chain is encoded in the ca's
        # shape — we just verify both addresses were found.
        cas = {ca for _, _, ca in tokens if ca}
        assert ETH_USDC.lower() in cas
        assert SOL_USDC in cas

    def test_eth_address_lowercased_in_output(self):
        from pipeline import extract_tokens
        # Mixed-case input
        cache: dict = {
            f"ethereum:{ETH_USDC.lower()}": {
                "symbol": "ETHTOK", "resolved_at": 9_999_999_999, "chain": "ethereum",
            },
        }
        tokens = extract_tokens(f"check {ETH_USDC}", ca_cache=cache)
        cas = [ca for _, _, ca in tokens if ca]
        assert all(c == c.lower() for c in cas if c.startswith("0x"))

    def test_tx_hash_not_extracted(self):
        """64-char hex looks like a tx hash, must not be matched as a token."""
        from pipeline import extract_tokens
        tx = "0x" + "a" * 64
        cache: dict = {}
        tokens = extract_tokens(f"see tx {tx} bro", ca_cache=cache)
        # No 0x-addr-shaped CAs in output
        assert not any(ca and ca.startswith("0x") and len(ca) == 42
                       for _, _, ca in tokens)


class TestStrategyFilterChainGate:
    """_passes_strategy_filter enforces chain opt-in semantics."""

    def test_solana_strategy_rejects_eth_token(self):
        # Solana main strategy (no chain filter) must reject ETH token
        from paper_trader import _passes_strategy_filter
        eth_token = {
            "symbol": "$ETHTOK", "chain": "ethereum",
            "score": 50, "market_cap": 1_000_000,
        }
        # BE25_TP80_SL30 is a Solana production strat with no chain in filter
        assert _passes_strategy_filter(eth_token, "BE25_TP80_SL30") is False

    def test_eth_strategy_rejects_solana_token(self):
        from paper_trader import _passes_strategy_filter
        sol_token = {
            "symbol": "$SOLTOK", "chain": "solana",
            "score": 50, "market_cap": 1_000_000, "_rt_liquidity_usd": 50_000,
        }
        assert _passes_strategy_filter(sol_token, "ETH_TP100_SL50") is False

    def test_eth_strategy_accepts_eth_token_above_liq(self):
        from paper_trader import _passes_strategy_filter
        eth_token = {
            "symbol": "$ETHTOK", "chain": "ethereum",
            "score": 50, "market_cap": 5_000_000, "_rt_liquidity_usd": 50_000,
        }
        assert _passes_strategy_filter(eth_token, "ETH_TP100_SL50") is True

    def test_eth_strategy_accepts_low_liquidity_after_v14e4(self):
        """v14e.4: min_liquidity_usd removed from EVM strats — fee model (gas +
        dynamic slippage) now encodes the cost of shallow pools instead of a
        hard pre-gate. Phase 1 wants the full KOL call distribution."""
        from paper_trader import _passes_strategy_filter
        eth_token = {
            "symbol": "$ETHTOK", "chain": "ethereum",
            "score": 50, "market_cap": 5_000_000, "_rt_liquidity_usd": 10_000,
        }
        assert _passes_strategy_filter(eth_token, "ETH_TP100_SL50") is True

    def test_bond_fast_rejects_eth_token(self):
        """BOND_FAST got chain=solana guard in v14 — must reject ETH tokens
        regardless of liquidity, to prevent the Solana fee model being
        applied to an ETH tx."""
        from paper_trader import _passes_strategy_filter
        eth_low_liq = {
            "symbol": "$ETHMICRO", "chain": "ethereum",
            "score": 50, "_rt_liquidity_usd": 1500,  # < 3k threshold
        }
        assert _passes_strategy_filter(eth_low_liq, "BOND_FAST_TP50_SL20_T20") is False

    def test_legacy_solana_token_without_chain_field_works(self):
        # Backward-compat: a pre-v14 token dict without a 'chain' key must
        # still match a Solana strategy (token_chain defaults to 'solana').
        from paper_trader import _passes_strategy_filter
        legacy = {
            "symbol": "$LEGACY",
            "score": 50, "market_cap": 1_000_000, "_rt_liquidity_usd": 50_000,
        }
        assert _passes_strategy_filter(legacy, "FAST_TP50_SL30") is True


class TestEthFeeModel:
    def test_eth_slip_bps_amortizes_gas_over_position(self):
        from paper_trader import _eth_slip_bps_with_gas
        # $200 position, 200 bps base MEV slip, $7.50 gas:
        # gas_bps = 7.50 / 200 * 10000 = 375; total = 575 bps
        assert _eth_slip_bps_with_gas(200, 200) == 575

    def test_eth_slip_bps_small_position_clamped(self):
        from paper_trader import _eth_slip_bps_with_gas
        # $10 position would give gas_bps = 7500, clamped to 2000
        assert _eth_slip_bps_with_gas(10, 200) == 2000

    def test_eth_slip_bps_large_position_mostly_mev(self):
        from paper_trader import _eth_slip_bps_with_gas
        # $1000 position: gas_bps = 75; total = 275
        assert _eth_slip_bps_with_gas(1000, 200) == 275

    def test_dynamic_sell_slip_uses_eth_model_for_ethereum_trade(self):
        from paper_trader import _dynamic_sell_slip_factor
        trade = {"chain": "ethereum", "position_usd": 200}
        # ETH path is independent of exit_type — any exit_type returns the
        # same factor (gas + MEV). Baseline: 575 bps → factor = 0.9425.
        for exit_type in ("tp_hit", "sl_hit", "trail_stop", "timeout"):
            f = _dynamic_sell_slip_factor(trade, exit_type)
            assert 0.79 < f < 1.0  # clamped, but ETH path doesn't swing wildly
