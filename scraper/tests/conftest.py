"""Shared fixtures for TelethonIA scraper tests."""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def reset_alerter():
    """Reset alerter throttling state before each test."""
    import alerter
    alerter._last_alert_times.clear()
    alerter._alert_counts.clear()
    yield


@pytest.fixture
def mock_telegram(monkeypatch):
    """Patch alerter credentials and requests.post to capture Telegram sends."""
    import alerter
    monkeypatch.setattr(alerter, "_BOT_TOKEN", "test-token")
    monkeypatch.setattr(alerter, "_CHAT_ID", "test-chat")
    sent = []

    def fake_post(url, json=None, timeout=None):
        sent.append(json)
        resp = MagicMock()
        resp.status_code = 200
        return resp

    monkeypatch.setattr("alerter.requests.post", fake_post)
    return sent


@pytest.fixture
def mock_supabase():
    """Create a mock Supabase client with chainable query builder."""
    client = MagicMock()

    def make_chain(data=None, count=None):
        chain = MagicMock()
        result = MagicMock()
        result.data = data or []
        result.count = count or 0
        chain.execute.return_value = result
        # Make all query methods return the chain itself
        for method in ("select", "insert", "update", "eq", "neq", "gte",
                       "lte", "in_", "not_", "is_", "order", "range",
                       "limit", "single"):
            getattr(chain, method).return_value = chain
        return chain

    default_chain = make_chain()
    client.table.return_value = default_chain
    client._make_chain = make_chain  # helper for tests to customize
    return client


@pytest.fixture
def reset_live_trader_globals():
    """Reset live_trader module-level globals for isolated tests."""
    import live_trader
    live_trader._daily_pnl_sol = 0.0
    live_trader._daily_pnl_reset_date = ""
    live_trader._daily_halted = False
    live_trader._weekly_pnl_sol = 0.0
    live_trader._weekly_pnl_reset_week = ""
    live_trader._weekly_halted = False
    live_trader._monthly_pnl_sol = 0.0
    live_trader._monthly_pnl_reset_month = ""
    live_trader._monthly_halted = False
    live_trader._ultra_client = None
    live_trader._ultra_client_init_attempted = False
    yield
