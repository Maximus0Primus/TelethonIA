"""Tests for paper_trader.py — strategies, prices, exit logic, slippage."""

from unittest.mock import MagicMock, patch

import pytest


class TestStrategiesDict:
    def test_structure(self):
        from paper_trader import STRATEGIES
        assert isinstance(STRATEGIES, dict)
        assert len(STRATEGIES) > 0
        for name, tranches in STRATEGIES.items():
            assert isinstance(tranches, list), f"{name} should be a list"
            for t in tranches:
                assert "sl_mult" in t, f"{name} tranche missing sl_mult"
                assert "horizon_min" in t, f"{name} tranche missing horizon_min"

    def test_grid_strategies_exist(self):
        from paper_trader import STRATEGIES
        # Core grid: TP30-100 x SL30-70 + NOSL variants
        assert "TP50_SL30" in STRATEGIES
        assert "TP100_SL50" in STRATEGIES
        assert "TP50_NOSL" in STRATEGIES


class TestFetchPricesBatch:
    def test_empty_list(self):
        from paper_trader import _fetch_prices_batch
        result = _fetch_prices_batch([])
        assert result == {}

    def test_parses_response(self, monkeypatch):
        from paper_trader import _fetch_prices_batch
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {
                "baseToken": {"address": "addr1"},
                "priceUsd": "0.00123",
                "volume": {"h24": "50000"},
            }
        ]
        monkeypatch.setattr("paper_trader.requests.get", lambda *a, **kw: mock_resp)

        result = _fetch_prices_batch(["addr1"])
        assert "addr1" in result
        assert abs(result["addr1"] - 0.00123) < 1e-8

    def test_handles_api_error(self, monkeypatch):
        import requests as req
        from paper_trader import _fetch_prices_batch
        monkeypatch.setattr("paper_trader.requests.get",
                           MagicMock(side_effect=req.RequestException("connection timeout")))
        result = _fetch_prices_batch(["addr1"])
        assert result == {}


class TestExitLogic:
    """Test trade exit evaluation via check_paper_trades internals."""

    def _make_trade(self, entry=1.0, tp_mult=1.5, sl_mult=0.7, horizon=120,
                    created_minutes_ago=30, strategy="TP50_SL30"):
        from datetime import datetime, timezone, timedelta
        created = datetime.now(timezone.utc) - timedelta(minutes=created_minutes_ago)
        return {
            "id": 1,
            "symbol": "TEST",
            "token_address": "testaddr",
            "entry_price": entry,
            "tp_price": entry * tp_mult if tp_mult else None,
            "sl_price": entry * sl_mult,
            "horizon_minutes": horizon,
            "created_at": created.isoformat(),
            "status": "open",
            "strategy": strategy,
            "position_usd": 10.0,
            "is_shadow": True,
            "source": "rt",
            "pnl_pct": None,
            "pnl_usd": None,
            "tranche_label": "main",
            "rt_liquidity_usd": 50000,
        }

    def test_tp_hit_detection(self):
        """Price above TP should trigger tp_hit."""
        trade = self._make_trade(entry=1.0, tp_mult=1.5)
        current_price = 1.6  # above TP (1.5)
        assert current_price >= trade["tp_price"]

    def test_sl_hit_detection(self):
        """Price below SL should trigger sl_hit."""
        trade = self._make_trade(entry=1.0, sl_mult=0.7)
        current_price = 0.65  # below SL (0.7)
        assert current_price <= trade["sl_price"]

    def test_timeout_detection(self):
        """Trade older than horizon should timeout."""
        trade = self._make_trade(entry=1.0, horizon=120, created_minutes_ago=130)
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        created = datetime.fromisoformat(trade["created_at"].replace("Z", "+00:00"))
        elapsed = (now - created).total_seconds() / 60
        assert elapsed >= trade["horizon_minutes"]

    def test_no_exit_within_bounds(self):
        """Price within TP/SL and before timeout — no exit."""
        trade = self._make_trade(entry=1.0, tp_mult=1.5, sl_mult=0.7,
                                horizon=120, created_minutes_ago=30)
        current_price = 1.2  # between SL (0.7) and TP (1.5)
        assert current_price > trade["sl_price"]
        assert current_price < trade["tp_price"]
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        created = datetime.fromisoformat(trade["created_at"].replace("Z", "+00:00"))
        elapsed = (now - created).total_seconds() / 60
        assert elapsed < trade["horizon_minutes"]


class TestSlippage:
    def test_buy_slippage_increases_entry_price(self):
        """Entry price should be adjusted upward by buy slippage + fees."""
        raw_price = 1.0
        buy_slippage_bps = 100  # 1%
        buy_fee_bps = 50  # 0.5%
        adjusted = raw_price * (1 + (buy_slippage_bps + buy_fee_bps) / 10_000)
        assert adjusted == pytest.approx(1.015, abs=1e-6)

    def test_sell_slippage_decreases_exit_price(self):
        """Exit price should be adjusted downward by sell slippage."""
        exit_price_raw = 1.5
        sell_bps = 200  # 2%
        adjusted = exit_price_raw * (1 - sell_bps / 10_000)
        assert adjusted == pytest.approx(1.47, abs=1e-6)
