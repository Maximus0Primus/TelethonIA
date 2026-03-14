"""Tests for alerter.py — throttling, formatting, and new v105 alert functions."""

import time
from unittest.mock import MagicMock, patch

import alerter


class TestThrottling:
    def test_send_no_credentials(self):
        """_send() returns False when BOT_TOKEN/CHAT_ID unset."""
        with patch.object(alerter, "_BOT_TOKEN", None):
            assert alerter._send("test", "startup") is False

    def test_send_success(self, mock_telegram):
        """_send() posts to Telegram and returns True."""
        assert alerter._send("hello", "startup") is True
        assert len(mock_telegram) == 1
        assert mock_telegram[0]["text"] == "hello"
        assert mock_telegram[0]["parse_mode"] == "HTML"

    def test_send_http_failure(self, monkeypatch):
        """_send() returns False on non-200 status."""
        monkeypatch.setattr(alerter, "_BOT_TOKEN", "tok")
        monkeypatch.setattr(alerter, "_CHAT_ID", "cid")
        resp = MagicMock()
        resp.status_code = 429
        monkeypatch.setattr("alerter.requests.post", lambda *a, **kw: resp)
        assert alerter._send("test", "startup") is False

    def test_throttle_within_cooldown(self, mock_telegram):
        """Second _send() within cooldown is throttled."""
        alerter._send("msg1", "cycle_failure")
        assert len(mock_telegram) == 1
        # Immediate second send should be throttled (cooldown=300s)
        alerter._send("msg2", "cycle_failure")
        assert len(mock_telegram) == 1  # still 1

    def test_throttle_after_cooldown_expired(self, mock_telegram, monkeypatch):
        """_send() succeeds after cooldown passes."""
        alerter._send("msg1", "cycle_failure")
        # Simulate time passing beyond cooldown
        alerter._last_alert_times["cycle_failure"] = time.time() - 600
        alerter._send("msg2", "cycle_failure")
        assert len(mock_telegram) == 2

    def test_exponential_backoff(self, mock_telegram):
        """Cooldown doubles on each consecutive send."""
        # First send at t=0
        alerter._send("msg1", "egress_warning")
        assert len(mock_telegram) == 1

        # After base cooldown (3600s), second send should work
        alerter._last_alert_times["egress_warning"] = time.time() - 3601
        alerter._send("msg2", "egress_warning")
        assert len(mock_telegram) == 2

        # After base cooldown again (3600s) — NOT enough because backoff doubled to 7200s
        alerter._last_alert_times["egress_warning"] = time.time() - 3601
        alerter._send("msg3", "egress_warning")
        assert len(mock_telegram) == 2  # throttled — needs 7200s

    def test_max_alerts_cap(self, mock_telegram):
        """After max_alerts sends, further sends are blocked."""
        for i in range(10):
            alerter._alert_counts["cycle_failure"] = i
            alerter._last_alert_times["cycle_failure"] = 0  # bypass cooldown
            alerter._send(f"msg{i}", "cycle_failure")

        # 11th send should be blocked (max=10)
        alerter._last_alert_times["cycle_failure"] = 0
        alerter._alert_counts["cycle_failure"] = 10
        result = alerter._can_send("cycle_failure")
        assert result is False

    def test_reset_alert(self, mock_telegram):
        """reset_alert() clears count and timestamp."""
        alerter._send("msg", "rt_listener_down")
        assert alerter._alert_counts.get("rt_listener_down", 0) == 1
        alerter.reset_alert("rt_listener_down")
        assert "rt_listener_down" not in alerter._alert_counts
        assert "rt_listener_down" not in alerter._last_alert_times

    def test_no_cooldown_category(self, mock_telegram):
        """live_trade category has 0 cooldown — always sends."""
        for i in range(5):
            alerter._send(f"trade{i}", "live_trade")
        assert len(mock_telegram) == 5


class TestAlertFormatting:
    def test_alert_live_trade_format(self, mock_telegram):
        alerter.alert_live_trade("SHIB", "BUY", 0.15, "abc123signature")
        assert len(mock_telegram) == 1
        text = mock_telegram[0]["text"]
        assert "$SHIB" in text
        assert "BUY" in text
        assert "0.1500 SOL" in text
        assert "solscan.io" in text

    def test_alert_live_trade_failed_format(self, mock_telegram):
        alerter.alert_live_trade_failed("DOGE", "BUY", "timeout after 10s")
        text = mock_telegram[0]["text"]
        assert "FAILED" in text
        assert "$DOGE" in text
        assert "timeout" in text

    def test_alert_wallet_low_format(self, mock_telegram):
        alerter.alert_wallet_low(0.0321)
        text = mock_telegram[0]["text"]
        assert "WALLET LOW" in text
        assert "0.0321" in text

    def test_alert_loss_limit_hit_format(self, mock_telegram):
        alerter.alert_loss_limit_hit("daily", -2.5, 2.0)
        text = mock_telegram[0]["text"]
        assert "LOSS LIMIT" in text
        assert "DAILY" in text
        assert "-2.5" in text
        assert "halted" in text.lower()

    def test_alert_slippage_deviation_format(self, mock_telegram):
        alerter.alert_slippage_deviation("PEPE", 300, 850)
        text = mock_telegram[0]["text"]
        assert "SLIPPAGE" in text
        assert "$PEPE" in text
        assert "300" in text
        assert "850" in text
        assert "+550" in text

    def test_alert_score_anomaly_format(self, mock_telegram):
        alerter.alert_score_anomaly(45.0, 15.0)
        text = mock_telegram[0]["text"]
        assert "SCORE ANOMALY" in text
        assert "67%" in text  # (1 - 15/45) * 100 = 67%

    def test_alert_kol_silence_format(self, mock_telegram):
        alerter.alert_kol_silence("PowsGemCalls", 52.0)
        text = mock_telegram[0]["text"]
        assert "KOL SILENT" in text
        assert "PowsGemCalls" in text
        assert "52" in text


class TestTruncate:
    def test_short_string(self):
        assert alerter._truncate("hello", 10) == "hello"

    def test_exact_length(self):
        assert alerter._truncate("hello", 5) == "hello"

    def test_over_length(self):
        result = alerter._truncate("hello world", 8)
        assert len(result) == 8
        assert result.endswith("...")
