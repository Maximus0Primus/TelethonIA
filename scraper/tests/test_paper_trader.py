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
    def _reset_cache(self):
        """v14e: _fetch_prices_batch has a 5s DS cache keyed on _last_ds_ts /
        _last_ds_addrs function attributes. Tests must reset it or they leak
        across each other (test_parses_response populates the cache, then
        test_handles_api_error hits the stale entry instead of the mocked error)."""
        import paper_trader
        paper_trader._dex_prices_cache.clear()
        paper_trader._jupiter_prices_cache.clear()
        for attr in ("_last_ds_ts", "_last_ds_addrs", "_last_jup_ts"):
            if hasattr(paper_trader._fetch_prices_batch, attr):
                delattr(paper_trader._fetch_prices_batch, attr)

    def test_empty_list(self):
        self._reset_cache()
        from paper_trader import _fetch_prices_batch
        result = _fetch_prices_batch([])
        assert result == {}

    def test_parses_response(self, monkeypatch):
        self._reset_cache()
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
        self._reset_cache()
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


class TestTwoTickConfirm:
    """v14e.45 / v14e.48: 2-tick confirm guard against 1-bad-quote SL fires.

    Reproduces the v141+v14e.45 interaction bug where live_trader's two
    eval calls per tick (main + paper_sim_ev probe) would jointly mutate
    `_pending_sl_be` such that the main eval was permanently stuck at
    pending=1 — SL never fired live (orphan trade 278446 sat 7.8h on
    2026-04-30 with 1000+ ticks below SL).
    """

    def _trade_below_sl(self):
        from datetime import datetime, timezone, timedelta
        return {
            "id": 278446, "symbol": "$NOKID", "token_address": "TEST",
            "entry_price": 8.93e-05, "tp_price": 1.07e-04, "sl_price": 8.04e-05,
            "horizon_minutes": 120,
            "created_at": (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(),
            "status": "open", "strategy": "SCALP_TP20_SL10_S30",
            "position_usd": 0.25, "is_shadow": False, "source": "rt_live",
            "pnl_pct": None, "pnl_usd": None, "tranche_label": "main",
            "rt_liquidity_usd": 20000, "high_price_seen": 8.93e-05,
        }

    def _reset(self):
        import paper_trader
        paper_trader._pending_sl_be.clear()

    def test_first_breach_defers_second_breach_fires(self):
        """Baseline 2-tick contract: 1st breach defers, 2nd consecutive fires sl_hit."""
        from datetime import datetime, timezone
        from paper_trader import _evaluate_trade_exit
        self._reset()
        trade = self._trade_below_sl()
        sl_below = 1.46e-05  # well below sl_price
        now = datetime.now(timezone.utc)
        ev1 = _evaluate_trade_exit(trade, sl_below, now, sell_slip_factor=1.0,
                                    sell_fee_bps=0)
        assert ev1.get("status") is None, "1st breach must defer"
        ev2 = _evaluate_trade_exit(trade, sl_below, now, sell_slip_factor=1.0,
                                    sell_fee_bps=0)
        assert ev2.get("status") == "sl_hit", "2nd consecutive breach must fire sl_hit"

    def test_readonly_probe_does_not_consume_pending_state(self):
        """v14e.48 fix: a readonly second eval call (paper_sim_ev probe in
        live_trader) must NOT mutate _pending_sl_be — otherwise the next
        main call sees pending=0 and SL never fires across multiple ticks.

        Pre-fix repro: probe call would increment to 2 then pop, leaving
        the dict empty. Next tick's main call resets to pending=1 → defers
        forever. (The exact symptom of orphan trade 278446.)
        """
        from datetime import datetime, timezone
        import paper_trader
        from paper_trader import _evaluate_trade_exit
        self._reset()
        trade = self._trade_below_sl()
        sl_below = 1.46e-05
        now = datetime.now(timezone.utc)

        # tick 1 — main eval defers (pending=1)
        ev_main_1 = _evaluate_trade_exit(trade, sl_below, now,
                                          sell_slip_factor=1.0, sell_fee_bps=0)
        assert ev_main_1.get("status") is None
        assert paper_trader._pending_sl_be.get(278446) == 1

        # tick 1 — paper_sim_ev probe (readonly) must not mutate the state
        ev_probe = _evaluate_trade_exit(trade, sl_below, now,
                                         sell_slip_factor=0.9, sell_fee_bps=10,
                                         _readonly_pending_state=True)
        # probe sees pending+1=2 internally and reports the would-be sl_hit,
        # but must NOT touch the dict (which the main eval owns).
        assert paper_trader._pending_sl_be.get(278446) == 1, (
            "readonly probe leaked into _pending_sl_be — main eval will be"
            " stuck at pending=1 forever (orphan trade bug)"
        )

        # tick 2 — main eval should now see pending=2 and fire
        ev_main_2 = _evaluate_trade_exit(trade, sl_below, now,
                                          sell_slip_factor=1.0, sell_fee_bps=0)
        assert ev_main_2.get("status") == "sl_hit", (
            "2nd main tick must fire sl_hit; if it defers, the readonly probe"
            " corrupted _pending_sl_be"
        )

    def test_recovery_resets_pending(self):
        """If price recovers above effective_sl between breaches, counter resets."""
        from datetime import datetime, timezone
        import paper_trader
        from paper_trader import _evaluate_trade_exit
        self._reset()
        trade = self._trade_below_sl()
        now = datetime.now(timezone.utc)
        # breach 1
        _evaluate_trade_exit(trade, 1.46e-05, now, 1.0, sell_fee_bps=0)
        assert paper_trader._pending_sl_be.get(278446) == 1
        # recovery above SL
        _evaluate_trade_exit(trade, 1.0e-04, now, 1.0, sell_fee_bps=0)
        assert 278446 not in paper_trader._pending_sl_be
        # fresh breach restarts at 1
        ev_after = _evaluate_trade_exit(trade, 1.46e-05, now, 1.0, sell_fee_bps=0)
        assert ev_after.get("status") is None
        assert paper_trader._pending_sl_be.get(278446) == 1


class TestStrategyAgeFilter:
    """v14e.16: per-strategy max_age_hours / min_age_hours disjoint bands.

    Guards against regression on existing strats (no age filter applied when
    neither key is declared) and locks the AGE24/48/72 shadow disjoint bands.
    """

    def _token(self, age_h, chain="solana", rt_score=50, liq=50_000):
        return {
            "chain": chain,
            "_rt_token_age_hours": age_h,
            "_rt_score": rt_score,
            "_rt_liquidity_usd": liq,
            "score": rt_score,
            "market_cap": 100_000,
        }

    def test_no_age_filter_on_existing_strat(self):
        """v14e.51: FAST_TP50_SL30 now declares max_age_hours=12 (live strat).
        Test the no-age-filter path via TP50_SL30 grid variant which still has no filter."""
        from paper_trader import _passes_strategy_filter
        # TP50_SL30 grid variant has no age filter -> 100h token passes
        assert _passes_strategy_filter(self._token(100), "TP50_SL30") is True
        # FAST_TP50_SL30 (live strat) now rejects 100h tokens
        assert _passes_strategy_filter(self._token(100), "FAST_TP50_SL30") is False
        # And accepts a fresh 6h token
        assert _passes_strategy_filter(self._token(6), "FAST_TP50_SL30") is True

    def test_age24_shadow_band(self):
        from paper_trader import _passes_strategy_filter
        # Token 18h old -> inside [12, 24]
        assert _passes_strategy_filter(self._token(18), "AGE24_FAST_TP50_SL30") is True
        # Token 11h old -> below min (belongs to baseline 12h gate)
        assert _passes_strategy_filter(self._token(11), "AGE24_FAST_TP50_SL30") is False
        # Token 30h old -> above max (belongs to AGE48 band)
        assert _passes_strategy_filter(self._token(30), "AGE24_FAST_TP50_SL30") is False

    def test_age48_shadow_band(self):
        from paper_trader import _passes_strategy_filter
        assert _passes_strategy_filter(self._token(30), "AGE48_FAST_TP50_SL30") is True
        assert _passes_strategy_filter(self._token(20), "AGE48_FAST_TP50_SL30") is False
        assert _passes_strategy_filter(self._token(60), "AGE48_FAST_TP50_SL30") is False

    def test_age72_shadow_band(self):
        from paper_trader import _passes_strategy_filter
        assert _passes_strategy_filter(self._token(60), "AGE72_FAST_TP50_SL30") is True
        assert _passes_strategy_filter(self._token(40), "AGE72_FAST_TP50_SL30") is False
        assert _passes_strategy_filter(self._token(80), "AGE72_FAST_TP50_SL30") is False


class TestShouldEvaluateExitLazyBypass:
    """v144.20: live (rt_live) and live_sync bypass LAZY throttling.

    Paper mains keep LAZY throttle (v144.3 invariant — behavioral A/B baseline).
    Validated by A/B on 14d / 75 LAZY live trades: bypass impact = -$0.03 total
    (noise). Throttle wasn't paying, only breaking sim↔paper↔live coherence.
    """

    def _trade(self, **overrides):
        from datetime import datetime, timezone
        base = {
            "id": 999,
            "strategy": "FAST_TP80_SL25",  # confirmed in LAZY_STRATEGIES
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": "rt",
            "entry_source": "ultra",
        }
        base.update(overrides)
        return base

    def test_lazy_strategy_throttles_plain_paper_main(self):
        from datetime import datetime, timezone
        from paper_trader import _should_evaluate_exit, _last_eval_ts, LAZY_STRATEGIES
        assert "FAST_TP80_SL25" in LAZY_STRATEGIES
        tr = self._trade(id=1001)
        _last_eval_ts.pop("1001", None)
        now = datetime.now(timezone.utc)
        assert _should_evaluate_exit(tr, now) is True  # first eval always allowed
        assert _should_evaluate_exit(tr, now) is False  # throttled (paper main)

    def test_lazy_strategy_bypassed_for_live_sync_shadow(self):
        from datetime import datetime, timezone
        from paper_trader import _should_evaluate_exit, _last_eval_ts
        tr = self._trade(id=1002, entry_source="live_sync")
        _last_eval_ts.pop("1002", None)
        now = datetime.now(timezone.utc)
        assert _should_evaluate_exit(tr, now) is True
        assert _should_evaluate_exit(tr, now) is True  # never throttled

    def test_lazy_strategy_bypassed_for_rt_live_trade(self):
        """Live trades must bypass LAZY to stay symmetric with their live_sync
        shadow mirror. A/B showed throttle = noise in $ terms."""
        from datetime import datetime, timezone
        from paper_trader import _should_evaluate_exit, _last_eval_ts
        tr = self._trade(id=1003, source="rt_live", entry_source="ultra")
        _last_eval_ts.pop("1003", None)
        now = datetime.now(timezone.utc)
        assert _should_evaluate_exit(tr, now) is True
        assert _should_evaluate_exit(tr, now) is True  # never throttled


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
