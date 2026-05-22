"""Tests for live_trader.py — loss limits, execution flow, reconciliation."""

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


class TestLossLimits:
    def test_daily_halt(self, reset_live_trader_globals):
        import live_trader
        live_trader._daily_pnl_sol = -3.0
        live_trader._daily_pnl_reset_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        config = {"daily_loss_limit_sol": 2.0, "weekly_loss_limit_sol": 5.0, "monthly_loss_limit_sol": 10.0}

        with patch("live_trader.alert_loss_limit_hit", create=True):
            result = live_trader._check_loss_limits(config)
        assert result is True
        assert live_trader._daily_halted is True

    def test_weekly_halt(self, reset_live_trader_globals):
        import live_trader
        now = datetime.now(timezone.utc)
        live_trader._daily_pnl_sol = 0.0
        live_trader._daily_pnl_reset_date = now.strftime("%Y-%m-%d")
        live_trader._weekly_pnl_sol = -6.0
        live_trader._weekly_pnl_reset_week = now.strftime("%Y-W%W")
        config = {"daily_loss_limit_sol": 2.0, "weekly_loss_limit_sol": 5.0, "monthly_loss_limit_sol": 10.0}

        with patch("live_trader.alert_loss_limit_hit", create=True):
            result = live_trader._check_loss_limits(config)
        assert result is True

    def test_monthly_halt(self, reset_live_trader_globals):
        import live_trader
        now = datetime.now(timezone.utc)
        live_trader._daily_pnl_sol = 0.0
        live_trader._daily_pnl_reset_date = now.strftime("%Y-%m-%d")
        live_trader._weekly_pnl_sol = 0.0
        live_trader._weekly_pnl_reset_week = now.strftime("%Y-W%W")
        live_trader._monthly_pnl_sol = -11.0
        live_trader._monthly_pnl_reset_month = now.strftime("%Y-%m")
        config = {"daily_loss_limit_sol": 2.0, "weekly_loss_limit_sol": 5.0, "monthly_loss_limit_sol": 10.0}

        with patch("live_trader.alert_loss_limit_hit", create=True):
            result = live_trader._check_loss_limits(config)
        assert result is True

    def test_within_limits(self, reset_live_trader_globals):
        import live_trader
        now = datetime.now(timezone.utc)
        live_trader._daily_pnl_sol = -1.0
        live_trader._daily_pnl_reset_date = now.strftime("%Y-%m-%d")
        live_trader._weekly_pnl_sol = -3.0
        live_trader._weekly_pnl_reset_week = now.strftime("%Y-W%W")
        live_trader._monthly_pnl_sol = -5.0
        live_trader._monthly_pnl_reset_month = now.strftime("%Y-%m")
        config = {"daily_loss_limit_sol": 2.0, "weekly_loss_limit_sol": 5.0, "monthly_loss_limit_sol": 10.0}

        result = live_trader._check_loss_limits(config)
        assert result is False

    def test_daily_reset_on_new_day(self, reset_live_trader_globals):
        import live_trader
        live_trader._daily_pnl_sol = -5.0
        live_trader._daily_pnl_reset_date = "2020-01-01"  # old date
        live_trader._daily_halted = True
        config = {"daily_loss_limit_sol": 2.0, "weekly_loss_limit_sol": 5.0, "monthly_loss_limit_sol": 10.0}

        result = live_trader._check_loss_limits(config)
        # Should have reset daily counter
        assert live_trader._daily_pnl_sol == 0.0
        assert live_trader._daily_halted is False
        assert result is False


class TestTrackPnl:
    def test_accumulates(self, reset_live_trader_globals):
        import live_trader
        live_trader._track_pnl(0.5)
        live_trader._track_pnl(-0.2)
        assert abs(live_trader._daily_pnl_sol - 0.3) < 1e-9
        assert abs(live_trader._weekly_pnl_sol - 0.3) < 1e-9
        assert abs(live_trader._monthly_pnl_sol - 0.3) < 1e-9


# v14e: real Solana base58 mint (32 chars, all legal base58 symbols) used
# throughout these tests. Previous tests used "SomeCA123" which isn't a valid
# base58 shape — the new chain gate rejects it and the tests failed.
VALID_SOL_MINT = "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R"
VALID_ETH_ADDR = "0x65fbda4711f5a4aad6dae92baf9f3a20f5aff111"


class TestExecuteBuy:
    def test_no_client(self, reset_live_trader_globals):
        import live_trader
        result = live_trader.execute_buy(VALID_SOL_MINT, 100_000_000)
        assert result["success"] is False
        assert "not initialized" in result["error"]

    def test_rejects_eth_mint(self, reset_live_trader_globals, monkeypatch):
        """v14e regression: a 0x mint must NEVER reach Jupiter. Previously
        this call produced HTTP 400 storms in the VPS logs."""
        import live_trader
        mock_client = MagicMock()
        monkeypatch.setattr(live_trader, "_ultra_client", mock_client)
        # Order call must NOT happen. Wire it to fail the test if it does.
        def _fail_order(*a, **kw):
            raise AssertionError("Jupiter /ultra/v1/order called with ETH mint")
        monkeypatch.setattr(live_trader, "_order_with_slippage", _fail_order)
        result = live_trader.execute_buy(VALID_ETH_ADDR, 20_000_000, 300)
        assert result["success"] is False
        assert result["error"] == "non-solana-mint"

    def test_success(self, reset_live_trader_globals, monkeypatch):
        import live_trader
        import sys

        mock_client = MagicMock()
        monkeypatch.setattr(live_trader, "_ultra_client", mock_client)

        # Mock jup_python_sdk module so the import inside execute_buy works
        mock_sdk = MagicMock()
        monkeypatch.setitem(sys.modules, "jup_python_sdk", mock_sdk)
        monkeypatch.setitem(sys.modules, "jup_python_sdk.models", mock_sdk)
        monkeypatch.setitem(sys.modules, "jup_python_sdk.models.ultra_api", mock_sdk)
        monkeypatch.setitem(sys.modules, "jup_python_sdk.models.ultra_api.ultra_order_request_model", mock_sdk)
        mock_sdk.UltraOrderRequest = MagicMock

        # Mock the order_with_slippage to return success
        monkeypatch.setattr(
            live_trader, "_order_with_slippage",
            lambda client, order, slip: {
                "status": "Success",
                "signature": "abc123def456",
                "inputAmountResult": "100000000",
                "outputAmountResult": "5000000000",
            }
        )

        result = live_trader.execute_buy(VALID_SOL_MINT, 100_000_000, 300)
        assert result["success"] is True
        assert result["signature"] == "abc123def456"
        assert result["input_amount"] == 100_000_000
        assert result["output_amount"] == 5_000_000_000
        assert result["slippage_bps"] == 300


class TestExecuteSell:
    def test_no_balance(self, reset_live_trader_globals, monkeypatch):
        import live_trader
        mock_client = MagicMock()
        monkeypatch.setattr(live_trader, "_ultra_client", mock_client)
        monkeypatch.setattr(live_trader, "get_wallet_balance", lambda: {
            "sol_balance": 1.0, "token_balances": {}
        })

        result = live_trader.execute_sell(VALID_SOL_MINT)
        assert result["success"] is False
        assert "No balance" in result["error"]

    def test_rejects_eth_mint(self, reset_live_trader_globals, monkeypatch):
        """v14e mirror of execute_buy's chain gate."""
        import live_trader
        mock_client = MagicMock()
        monkeypatch.setattr(live_trader, "_ultra_client", mock_client)
        result = live_trader.execute_sell(VALID_ETH_ADDR)
        assert result["success"] is False
        assert result["error"] == "non-solana-mint"


class TestOpenLiveTradeChainGate:
    """v14e: open_live_trade must skip non-Solana before any state mutation."""

    def test_skips_ethereum_token(self, reset_live_trader_globals, monkeypatch):
        import live_trader
        # If the gate is missing, open_live_trade would fall through to
        # _check_loss_limits (which touches globals) and ultimately execute_buy.
        # We assert the short-circuit by checking no _check_loss_limits call.
        called = {"loss_limits": False}
        def _fake_loss(*a, **kw):
            called["loss_limits"] = True
            return False
        monkeypatch.setattr(live_trader, "_check_loss_limits", _fake_loss)
        token_entry = {
            "token_address": VALID_ETH_ADDR,
            "symbol": "ETHMEME",
            "price_usd": 0.001,
            "chain": "ethereum",
        }
        result = live_trader.open_live_trade(MagicMock(), token_entry, "ETH_TP100_SL50",
                                              10.0, {"max_open_positions": 5})
        assert result["success"] is False
        assert called["loss_limits"] is False, "chain gate must fire BEFORE loss-limit check"


class TestReconciliation:
    def test_mismatch_auto_closes(self, reset_live_trader_globals, mock_supabase, monkeypatch):
        import live_trader

        # Mock wallet with no token balances
        monkeypatch.setattr(live_trader, "get_wallet_balance", lambda: {
            "sol_balance": 1.0, "token_balances": {}
        })

        # Mock DB returning one open live trade
        chain = mock_supabase._make_chain(
            data=[{
                "id": 1, "symbol": "SHIB", "token_address": "SHIBaddr123",
                "entry_price": 0.001, "position_usd": 30, "created_at": "2026-03-14"
            }],
            count=1
        )
        mock_supabase.table.return_value = chain

        result = live_trader.reconcile_positions(mock_supabase)
        assert result["mismatches"] == 1
        assert result["auto_closed"] == 1

    def test_no_mismatches(self, reset_live_trader_globals, mock_supabase, monkeypatch):
        import live_trader
        monkeypatch.setattr(live_trader, "get_wallet_balance", lambda: {
            "sol_balance": 1.0, "token_balances": {
                "SHIBaddr123": {"amount": 1000, "ui_amount": 10.0}
            }
        })
        chain = mock_supabase._make_chain(
            data=[{
                "id": 1, "symbol": "SHIB", "token_address": "SHIBaddr123",
                "entry_price": 0.001, "position_usd": 30, "created_at": "2026-03-14"
            }],
            count=1
        )
        mock_supabase.table.return_value = chain

        result = live_trader.reconcile_positions(mock_supabase)
        assert result["mismatches"] == 0


def _gen_solana_key():
    """Generate a throwaway valid base58 Solana secret key for tests."""
    import base58
    from solders.keypair import Keypair
    return base58.b58encode(bytes(Keypair())).decode()


class TestRentRecoveryV14e68:
    """v14e.68: close_ata leaked rent on EVERY live sell (0/24 closed) because the
    immediate post-sell balance read raced Jupiters fill and saw stale tokens, then
    skipped. These tests pin both the retry-read fix and the catch-all sweep."""

    LEGACY = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
    T2022 = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"

    def test_close_retries_stale_read_then_closes(self, monkeypatch):
        """The bug: read says >0 right after our own sell, then settles to 0.
        Fix: retry the read; once it is 0 we are the last seller and MUST close."""
        import live_trader
        monkeypatch.setenv("SOLANA_PRIVATE_KEY", _gen_solana_key())
        monkeypatch.setattr(live_trader.time, "sleep", lambda *_: None)
        # stale non-zero on first read, settled to 0 on second
        reads = iter([("ATAxx", self.LEGACY, 1234), ("ATAxx", self.LEGACY, 0)])
        monkeypatch.setattr(live_trader, "_find_owned_token_account",
                            lambda owner, mint: next(reads))
        sent = []
        monkeypatch.setattr(live_trader, "_send_close_account",
                            lambda acct, prog, label="": sent.append((acct, prog)) or True)
        assert live_trader._close_token_account("MINTaaaaaaaa") is True
        assert sent == [("ATAxx", self.LEGACY)], "must close once read settles to 0"

    def test_close_skips_when_other_strategy_still_holds(self, monkeypatch):
        """Shared ATA: a non-zero read can be LEGITIMATE (other strategy holds).
        If it stays >0 across retries we must NOT close."""
        import live_trader
        monkeypatch.setenv("SOLANA_PRIVATE_KEY", _gen_solana_key())
        monkeypatch.setattr(live_trader.time, "sleep", lambda *_: None)
        monkeypatch.setattr(live_trader, "_find_owned_token_account",
                            lambda owner, mint: ("ATAxx", self.LEGACY, 9999))
        sent = []
        monkeypatch.setattr(live_trader, "_send_close_account",
                            lambda *a, **k: sent.append(a) or True)
        assert live_trader._close_token_account("MINTaaaaaaaa") is False
        assert sent == [], "must not close while another holder remains"

    def test_sweep_covers_both_programs_and_only_empty(self, monkeypatch):
        """The validation/sweep blind-spot: must enumerate BOTH token programs and
        close only 0-balance accounts."""
        import live_trader
        monkeypatch.setenv("SOLANA_PRIVATE_KEY", _gen_solana_key())

        def _acct(pubkey, prog, mint, amount):
            return {"pubkey": pubkey, "account": {"owner": prog, "lamports": 2039280,
                    "data": {"parsed": {"info": {"mint": mint,
                             "tokenAmount": {"amount": str(amount)}}}}}}

        progs_queried = []

        def fake_post(url, json=None, timeout=None):
            pid = json["params"][1]["programId"]
            progs_queried.append(pid)
            if pid == self.LEGACY:
                value = [_acct("A1", self.LEGACY, "M1", 0),    # empty -> close
                         _acct("A2", self.LEGACY, "M2", 50)]   # holding -> skip
            else:
                value = [_acct("A3", self.T2022, "M3", 0)]     # empty Token-2022 -> close
            resp = MagicMock()
            resp.json.return_value = {"result": {"value": value}}
            return resp

        monkeypatch.setattr("live_trader.requests.post", fake_post)
        closed_calls = []
        monkeypatch.setattr(live_trader, "_send_close_account",
                            lambda acct, prog, label="": closed_calls.append((acct, prog)) or True)

        n, rent = live_trader._sweep_empty_token_accounts(return_rent=True)
        assert self.LEGACY in progs_queried and self.T2022 in progs_queried, "both programs"
        assert closed_calls == [("A1", self.LEGACY), ("A3", self.T2022)], "only empty, both progs"
        assert n == 2
        assert round(rent, 6) == round(2 * 2039280 / 1e9, 6)

    def test_sweep_noop_without_key(self, monkeypatch):
        import live_trader
        monkeypatch.delenv("SOLANA_PRIVATE_KEY", raising=False)
        assert live_trader._sweep_empty_token_accounts() == 0
