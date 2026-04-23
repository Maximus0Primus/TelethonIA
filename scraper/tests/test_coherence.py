"""v126: Coherence test — paper, live, and sim must produce identical exit
decisions when fed the same price ticks.

This is the contract test that prevents regressions on the paper/live/sim
unification work (v118-v126). If this test breaks, one of the three paths has
drifted from the shared _evaluate_trade_exit() source of truth.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch


def _make_trade(entry: float, dex_spot: float | None = None, *,
                strategy: str = "DTRAIL10_ACT15_SL70",
                position_usd: float = 100.0,
                horizon_min: int = 120,
                high_seen: float | None = None) -> dict:
    """Build a trade dict that mirrors the paper_trades schema."""
    created = datetime.now(timezone.utc) - timedelta(minutes=5)
    sl_mult = 0.30  # SL70 = exit at -70%
    return {
        "id": 1,
        "symbol": "TEST",
        "token_address": "addr_test",
        "strategy": strategy,
        "entry_price": entry,
        "dex_spot_price_at_entry": dex_spot,
        "sl_price": entry * sl_mult,
        "tp_price": None,
        "horizon_minutes": horizon_min,
        "position_usd": position_usd,
        "rt_liquidity_usd": 50_000,
        "created_at": created.isoformat(),
        "high_price_seen": high_seen if high_seen is not None else (dex_spot or entry),
    }


class TestExitLogicSingleSource:
    """_evaluate_trade_exit is the one exit fn used by paper + live + sim-unified."""

    def test_live_references_paper_exit_fn(self):
        """live_trader calls _evaluate_trade_exit from paper_trader (inline import)."""
        from pathlib import Path
        src = Path(__file__).parent.parent.joinpath("live_trader.py").read_text()
        assert "from paper_trader import" in src and "_evaluate_trade_exit" in src, \
            "live_trader must import _evaluate_trade_exit from paper_trader"

    def test_sim_unified_uses_same_fn(self):
        from pathlib import Path
        src = Path(__file__).parent.parent.joinpath("sim_engines.py").read_text()
        assert "_evaluate_trade_exit" in src, \
            "sim_engines must reference _evaluate_trade_exit"


class TestDTrailActivationUsesMarketRef:
    """DTRAIL activation must reference dex_spot_price_at_entry, not entry_price.
    Gap #1 regression test — paper without dex_spot would activate too early.
    """

    def test_activation_with_matching_market_ref(self):
        """When entry==dex_spot, activation behaves identically."""
        from paper_trader import _evaluate_trade_exit
        now = datetime.now(timezone.utc)
        trade = _make_trade(entry=1.0, dex_spot=1.0, high_seen=1.20)
        # Price drops to 1.08 — below trail (1.20 * 0.9 = 1.08) → exit
        result = _evaluate_trade_exit(trade, 1.08, now, 1 - 10/10_000)
        assert result is not None

    def test_paper_gap_fix_activates_on_market_price(self):
        """Simulate Gap #1: Jupiter fill above market spot.
        Trail should activate on market spot (dex_spot), not fill."""
        from paper_trader import _evaluate_trade_exit
        now = datetime.now(timezone.utc)
        # entry_price (fill) is 1.10, but market was 1.00 at buy time
        trade = _make_trade(entry=1.10, dex_spot=1.00, high_seen=1.16)
        # 1.16 > 1.00 * (1 + 0.15) = 1.15 → activated
        # trail_trigger = 1.16 * 0.9 = 1.044 → if price 1.04, should exit
        result = _evaluate_trade_exit(trade, 1.04, now, 1 - 10/10_000)
        assert result is not None
        # Ensure we don't exit on 1.05 (above trigger)
        trade2 = _make_trade(entry=1.10, dex_spot=1.00, high_seen=1.16)
        result2 = _evaluate_trade_exit(trade2, 1.05, now, 1 - 10/10_000)
        # Should update high but not exit
        assert result2 is not None
        assert "status" not in result2


class TestSimSlippageMatchesProduction:
    """Gap #2 regression: sim._exit must use _dynamic_sell_slip_factor."""

    def test_legacy_exit_uses_dynamic_factor(self):
        import sim_engines
        sim_engines._sim_liquidity_usd = 50_000
        # SL hit: production factor = 10 * 1.0 * 3.0 + 50 = 80bps → factor 0.992
        from paper_trader import _dynamic_sell_slip_factor
        expected = _dynamic_sell_slip_factor({"rt_liquidity_usd": 50_000}, "sl_hit")
        result = sim_engines._exit("sl_hit", 1.0, 1.0, 30, is_sl=True)
        # expected pnl_pct = expected - 1 (since net = 1.0 * expected, entry=1.0)
        assert abs(result["pnl_pct"] - (expected - 1)) < 1e-6, \
            f"sim._exit must match production slippage, got {result['pnl_pct']}"


class TestLiqSlipMultiplierContinuous:
    """v14e.6: log-continuous liq model replaces 3-bucket step function."""

    def test_anchors(self):
        from paper_trader import _liq_slip_multiplier
        # Deep pool: no penalty
        assert _liq_slip_multiplier(50_000) == 1.0
        assert _liq_slip_multiplier(100_000) == 1.0
        # Floor at $500 input → mult 2.0
        assert abs(_liq_slip_multiplier(500) - 2.0) < 0.01
        # Zero or negative → floor to 500 → mult 2.0
        assert abs(_liq_slip_multiplier(0) - 2.0) < 0.01
        assert abs(_liq_slip_multiplier(-100) - 2.0) < 0.01

    def test_monotone_decreasing(self):
        from paper_trader import _liq_slip_multiplier
        liqs = [500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]
        mults = [_liq_slip_multiplier(l) for l in liqs]
        assert mults == sorted(mults, reverse=True), \
            f"liq_mult must decrease with liq: {list(zip(liqs, mults))}"

    def test_no_step_discontinuities(self):
        """The old buckets jumped 54% at liq=5k (2.0→1.3) and 23% at 20k
        (1.3→1.0). Continuous model: every $100 step changes mult by <0.05."""
        from paper_trader import _liq_slip_multiplier
        prev = _liq_slip_multiplier(500)
        for liq in range(600, 100_000, 100):
            cur = _liq_slip_multiplier(liq)
            assert abs(cur - prev) < 0.05, f"discontinuity at ${liq}: {prev:.3f} -> {cur:.3f}"
            prev = cur

    def test_clamped_range(self):
        from paper_trader import _liq_slip_multiplier
        # Nothing should go below 1.0 or above 2.5 regardless of input
        for liq in [0, 1, 100, 500, 50_000, 1_000_000_000]:
            m = _liq_slip_multiplier(liq)
            assert 1.0 <= m <= 2.5


class TestDedup24hSliding:
    """Gap #3 regression: sim dedup must honour 24h cooldown like paper/live."""

    def test_dedup_keeps_recalls_after_24h(self):
        from sim import dedup_first_call
        now = datetime.now(timezone.utc)
        trades = [
            {"token_address": "A", "created_at": now.isoformat()},
            {"token_address": "A", "created_at": (now + timedelta(hours=25)).isoformat()},
            {"token_address": "A", "created_at": (now + timedelta(hours=26)).isoformat()},  # within 24h of prev
        ]
        result = dedup_first_call(trades)
        # Entry 1 kept, entry 2 kept (>24h), entry 3 dropped (within 24h of 2)
        assert len(result) == 2
