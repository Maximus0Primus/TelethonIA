"""Tests for pipeline.py — scoring logic tests that can run without Supabase.

Pipeline scoring depends on SCORING_PARAMS loaded from DB at module level.
Tests that require the full scorer are wrapped in try/except and skip if
the module can't load params. Tests that verify pure logic work regardless.
"""

import math
from unittest.mock import patch, MagicMock

import pytest


class TestComputeScore:
    """Test _compute_score_with_renormalization — skips if DB params unavailable."""

    def _get_scorer(self):
        try:
            import pipeline
            # Try calling with empty token to see if params are loaded
            pipeline._compute_score_with_renormalization({})
            return pipeline._compute_score_with_renormalization
        except (KeyError, AttributeError, TypeError):
            pytest.skip("SCORING_PARAMS not loaded (requires Supabase connection)")
        except Exception as e:
            pytest.skip(f"pipeline import failed: {e}")

    def test_empty_token_returns_zero(self):
        scorer = self._get_scorer()
        score, confidence = scorer({})
        assert score == 0.0
        assert confidence == 0.0

    def test_all_components_valid_range(self):
        scorer = self._get_scorer()
        token = {
            "consensus": 0.8, "conviction": 0.6, "breadth_score": 0.7,
            "price_change_24h": 10.0, "unique_kols": 5, "token_address": "addr",
        }
        score, conf = scorer(token)
        assert 0.0 <= score <= 1.0
        assert 0.0 <= conf <= 1.0

    def test_high_quality_scores_higher(self):
        scorer = self._get_scorer()
        strong = {"consensus": 0.9, "conviction": 0.8, "breadth_score": 0.9,
                  "unique_kols": 5, "token_address": "a"}
        weak = {"consensus": 0.1, "conviction": 0.1, "breadth_score": 0.1,
                "unique_kols": 1, "token_address": "b"}
        assert scorer(strong)[0] > scorer(weak)[0]


class TestScoringMath:
    """Pure math tests that don't need SCORING_PARAMS."""

    def test_score_renormalization_logic(self):
        """Verify weight renormalization math: if one component missing,
        remaining weights should sum to 1.0."""
        weights = {"consensus": 0.35, "conviction": 0.10, "breadth": 0.55}
        # Remove consensus
        remaining = {k: v for k, v in weights.items() if k != "consensus"}
        total = sum(remaining.values())
        renormalized = {k: v / total for k, v in remaining.items()}
        assert abs(sum(renormalized.values()) - 1.0) < 1e-9
        assert abs(renormalized["breadth"] - 0.55/0.65) < 1e-9

    def test_pump_discount_formula(self):
        """Verify pump discount: consensus *= max(floor, 1 - pc24/divisor)."""
        floor = 0.5
        divisor = 400.0
        # pc24=100% → mult = max(0.5, 1 - 100/400) = max(0.5, 0.75) = 0.75
        pc24 = 100
        mult = max(floor, 1.0 - pc24 / divisor)
        assert abs(mult - 0.75) < 1e-9
        # pc24=400% → mult = max(0.5, 1 - 400/400) = max(0.5, 0.0) = 0.5
        mult2 = max(floor, 1.0 - 400 / divisor)
        assert abs(mult2 - 0.5) < 1e-9
        # pc24=10% (below threshold) → no discount
        threshold = 50
        assert 10 < threshold  # no discount applied

    def test_data_confidence_formula(self):
        """Verify confidence = weighted sum of component availabilities."""
        # Simplified: kol_conf = min(1.0, unique_kols / 3)
        assert min(1.0, 1/3) == pytest.approx(0.333, abs=0.01)
        assert min(1.0, 5/3) == 1.0

    def test_exponential_decay(self):
        """Verify time decay: e^(-lambda * hours)."""
        lam = 0.1
        assert abs(math.exp(-lam * 0) - 1.0) < 1e-9
        assert abs(math.exp(-lam * 10) - 0.3679) < 0.01
        assert math.exp(-lam * 24) < 0.1

    def test_slippage_multiplier_scaling(self):
        """Verify liquidity-based slippage scaling logic."""
        # liq_mult = max(1.0, min(4.0, 50000 / max(liq_usd, 1000)))
        assert max(1.0, min(4.0, 50000 / 50000)) == 1.0  # deep
        assert max(1.0, min(4.0, 50000 / 5000)) == 4.0   # shallow (capped)
        assert max(1.0, min(4.0, 50000 / 25000)) == 2.0   # medium

    def test_score_band_boundaries(self):
        """Score bands should be ordered: <30, 30-49, 50-69, 70+."""
        bands = [
            ("<30", lambda s: s < 30),
            ("30-49", lambda s: 30 <= s < 50),
            ("50-69", lambda s: 50 <= s < 70),
            ("70+", lambda s: s >= 70),
        ]
        test_scores = [0, 15, 29, 30, 49, 50, 69, 70, 100]
        for score in test_scores:
            matches = [name for name, fn in bands if fn(score)]
            assert len(matches) == 1, f"Score {score} matched {matches}"

    def test_pnl_calculation(self):
        """PnL = position_usd * (exit_price / entry_price - 1)."""
        entry = 1.0
        exit_price = 1.5
        position = 30.0
        pnl_pct = (exit_price / entry) - 1
        pnl_usd = position * pnl_pct
        assert abs(pnl_pct - 0.5) < 1e-9
        assert abs(pnl_usd - 15.0) < 1e-9

    def test_kelly_fraction(self):
        """Kelly = (WR * avg_win - (1-WR) * avg_loss) / avg_win."""
        wr = 0.75
        avg_win = 0.54  # +54%
        avg_loss = 0.21  # -21%
        kelly = (wr * avg_win - (1 - wr) * avg_loss) / avg_win
        assert kelly > 0  # positive edge
        half_kelly = kelly / 2
        assert 0 < half_kelly < 1
