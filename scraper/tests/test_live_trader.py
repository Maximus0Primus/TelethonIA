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


class TestExecuteBuy:
    def test_no_client(self, reset_live_trader_globals):
        import live_trader
        result = live_trader.execute_buy("SomeCA", 100_000_000)
        assert result["success"] is False
        assert "not initialized" in result["error"]

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

        result = live_trader.execute_buy("SomeCA123", 100_000_000, 300)
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

        result = live_trader.execute_sell("MissingCA123")
        assert result["success"] is False
        assert "No balance" in result["error"]


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
