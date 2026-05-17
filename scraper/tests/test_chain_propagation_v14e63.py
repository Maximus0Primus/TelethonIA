"""v14e.63 — Tests for chain propagation fixes (BSC/Base visibility).

Two bugs were patched:

1. ``push_to_supabase.insert_kol_mentions`` never wrote the ``chain`` column,
   so every row defaulted to ``'solana'`` in the DB — including the 3,484
   mentions with a 0x ``resolved_ca`` observed over 30d (audit 2026-05-17).

2. ``enrich.enrich_token`` defaulted 0x CAs to ``chain='ethereum'`` by shape,
   never calling ``resolve_evm_chain`` to disambiguate ETH/BSC/Base. Any BSC
   or Base CA was thus queried against the DexScreener ethereum endpoint,
   which returned no matching pairs → silent skip of enrichment and snapshot.

These tests pin both fixes in place.
"""
from unittest.mock import MagicMock, patch

import pytest


ETH_USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
BSC_TOKEN = "0xb701c645718588f29417e4119ba5d678357b1110"  # MOONPEPE on BSC
SOL_USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


# ---------------------------------------------------------------------------
# Bug 1 — kol_mentions chain field
# ---------------------------------------------------------------------------

class TestInsertKolMentionsChain:
    """``insert_kol_mentions`` must propagate chain inferred from ``resolved_ca``.

    Shape-only (no network call) — SOL vs EVM granularity is enough for
    analytics partitioning. BSC/Base disambiguation lives downstream in
    ``enrich.enrich_token`` which sets ``tokens.chain`` + ``token_snapshots.chain``.
    """

    def _capture_upsert(self, mentions):
        captured: list[dict] = []
        mock_table = MagicMock()

        def fake_upsert(rows, **kwargs):
            captured.extend(rows)
            return MagicMock(execute=MagicMock())

        mock_table.upsert = fake_upsert
        mock_client = MagicMock()
        mock_client.table = MagicMock(return_value=mock_table)
        with patch("push_to_supabase._get_client", return_value=mock_client):
            from push_to_supabase import insert_kol_mentions
            insert_kol_mentions(mentions)
        return captured

    def _mention(self, **overrides):
        m = {
            "symbol": "$TOK", "kol_group": "kol1",
            "message_text": "test", "message_date": "2026-05-17T00:00:00+00:00",
        }
        m.update(overrides)
        return m

    def test_sol_ca_writes_chain_solana(self):
        rows = self._capture_upsert([self._mention(resolved_ca=SOL_USDC)])
        assert rows[0]["chain"] == "solana"

    def test_eth_ca_writes_chain_ethereum(self):
        rows = self._capture_upsert([self._mention(resolved_ca=ETH_USDC)])
        assert rows[0]["chain"] == "ethereum"

    def test_bsc_shape_ca_writes_chain_ethereum_shape_only(self):
        # Shape-based inference cannot tell BSC from ETH — intentional.
        # The tokens/snapshots tables get the accurate chain via enrich.py.
        rows = self._capture_upsert([self._mention(resolved_ca=BSC_TOKEN)])
        assert rows[0]["chain"] == "ethereum"

    def test_no_resolved_ca_defaults_solana(self):
        rows = self._capture_upsert([self._mention(resolved_ca=None)])
        assert rows[0]["chain"] == "solana"

    def test_unrecognized_ca_shape_defaults_solana(self):
        rows = self._capture_upsert([self._mention(resolved_ca="garbage")])
        assert rows[0]["chain"] == "solana"

    def test_mixed_batch_each_row_gets_its_own_chain(self):
        rows = self._capture_upsert([
            self._mention(symbol="$S", resolved_ca=SOL_USDC),
            self._mention(symbol="$E", resolved_ca=ETH_USDC),
            self._mention(symbol="$N", resolved_ca=None),
        ])
        by_sym = {r["symbol"]: r["chain"] for r in rows}
        assert by_sym == {"$S": "solana", "$E": "ethereum", "$N": "solana"}


# ---------------------------------------------------------------------------
# Bug 2 — enrich.enrich_token EVM chain disambiguation
# ---------------------------------------------------------------------------

class TestEnrichEvmChainResolution:
    """``enrich_token`` must route 0x CAs through ``resolve_evm_chain`` (cached)
    so BSC/Base tokens hit the correct DexScreener chain endpoint."""

    def setup_method(self):
        import enrich
        enrich._EVM_CHAIN_CACHE.clear()

    def test_solana_ca_skips_resolve_evm_chain(self):
        from enrich import enrich_token
        with patch("chain_detect.resolve_evm_chain") as mock_resolve, \
             patch("enrich._fetch_dexscreener_by_address", return_value=None) as mock_ds:
            enrich_token("$SOL", cache={}, known_ca=SOL_USDC)
            mock_resolve.assert_not_called()
            mock_ds.assert_called_once_with(SOL_USDC, chain="solana")

    def test_eth_ca_calls_resolve_evm_chain_once(self):
        from enrich import enrich_token
        with patch("chain_detect.resolve_evm_chain", return_value="ethereum") as mock_resolve, \
             patch("enrich._fetch_dexscreener_by_address", return_value=None) as mock_ds:
            enrich_token("$ETH", cache={}, known_ca=ETH_USDC)
            mock_resolve.assert_called_once_with(ETH_USDC)
            mock_ds.assert_called_once_with(ETH_USDC, chain="ethereum")

    def test_bsc_ca_routes_to_bsc_dexscreener_call(self):
        from enrich import enrich_token
        with patch("chain_detect.resolve_evm_chain", return_value="bsc"), \
             patch("enrich._fetch_dexscreener_by_address", return_value=None) as mock_ds:
            enrich_token("$BSC", cache={}, known_ca=BSC_TOKEN)
            mock_ds.assert_called_once_with(BSC_TOKEN, chain="bsc")

    def test_base_ca_routes_to_base_dexscreener_call(self):
        from enrich import enrich_token
        base_ca = "0x" + "f" * 40
        with patch("chain_detect.resolve_evm_chain", return_value="base"), \
             patch("enrich._fetch_dexscreener_by_address", return_value=None) as mock_ds:
            enrich_token("$BASE", cache={}, known_ca=base_ca)
            mock_ds.assert_called_once_with(base_ca, chain="base")

    def test_evm_chain_cache_avoids_repeat_resolve_call(self):
        from enrich import enrich_token
        with patch("chain_detect.resolve_evm_chain", return_value="bsc") as mock_resolve, \
             patch("enrich._fetch_dexscreener_by_address", return_value=None):
            enrich_token("$T1", cache={}, known_ca=BSC_TOKEN)
            # Second call, fresh per-symbol cache, same CA — hits _EVM_CHAIN_CACHE
            enrich_token("$T2", cache={}, known_ca=BSC_TOKEN)
            assert mock_resolve.call_count == 1

    def test_evm_chain_cache_is_case_insensitive(self):
        from enrich import enrich_token
        upper = BSC_TOKEN.upper().replace("0X", "0x")
        with patch("chain_detect.resolve_evm_chain", return_value="bsc") as mock_resolve, \
             patch("enrich._fetch_dexscreener_by_address", return_value=None):
            enrich_token("$T1", cache={}, known_ca=BSC_TOKEN)
            enrich_token("$T2", cache={}, known_ca=upper)
            assert mock_resolve.call_count == 1

    def test_resolve_evm_chain_failure_falls_back_to_ethereum(self):
        # DS unreachable / no pairs → resolve_evm_chain returns None.
        # Fallback to 'ethereum' preserves v14 behavior for the safe case.
        from enrich import enrich_token
        with patch("chain_detect.resolve_evm_chain", return_value=None), \
             patch("enrich._fetch_dexscreener_by_address", return_value=None) as mock_ds:
            enrich_token("$ETH", cache={}, known_ca=ETH_USDC)
            mock_ds.assert_called_once_with(ETH_USDC, chain="ethereum")
