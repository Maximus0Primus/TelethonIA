"""Smoke test: shadow rows from open_paper_trades have v144.3 invariants.

Builds a minimal token + config, calls open_paper_trades(), then asserts that
shadow rows (is_shadow=True) have:
  - position_usd > 0
  - entry_source in {ultra, dexscreener, live_sync}
  - dex_spot_price_at_entry == entry_price
  - high_price_seen >= entry_price
  - sol_price_at_entry not None

Catches regression to legacy shadow code (pre-v144.3 with position_usd=0).
"""
import os, sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _MockResp:
    def __init__(self, data=None, count=0):
        self.data = data or []
        self.count = count


class _MockChain:
    """Chainable mock that records inserted rows and returns empty selects."""
    def __init__(self, store):
        self._store = store
        self._payload = None

    def table(self, name):
        self._table = name
        return self

    def select(self, *args, **kwargs):
        return self

    def insert(self, payload):
        self._payload = payload
        return self

    def update(self, payload):
        self._payload = payload
        return self

    def eq(self, k, v): return self
    def neq(self, k, v): return self
    def in_(self, k, v): return self
    def gte(self, k, v): return self
    def lt(self, k, v): return self
    def order(self, *args, **kwargs): return self
    def limit(self, *args, **kwargs): return self
    def range(self, *args, **kwargs): return self

    def execute(self):
        if self._payload is not None:
            payload = self._payload
            self._payload = None
            if isinstance(payload, list):
                self._store.extend(payload)
            elif isinstance(payload, dict):
                self._store.append(payload)
            return _MockResp(data=payload if isinstance(payload, list) else [payload])
        return _MockResp(data=[], count=0)


class ShadowParitySmokeTest(unittest.TestCase):

    def test_shadow_rows_have_v144_3_invariants(self):
        from paper_trader import open_paper_trades, _jupiter_prices_cache

        # Warm Jupiter cache so shadow uses Ultra-quoted entry_price
        addr = "TEST_TOKEN_ADDRESS_111111111111111111111111111"
        _jupiter_prices_cache[addr] = 0.000123  # synthetic price

        token = {
            "token_address": addr,
            "symbol": "TEST",
            "price_usd": 0.000120,
            "score": 50,
            "market_cap": 100_000,
            "_alloc_usd": 50.0,
            "_rt_source": "rt",
            "_rt_kol_group": "test_kol",
            "_rt_score": 50,
            "_rt_liquidity_usd": 30_000,
        }
        store = []
        client = _MockChain(store)
        config = {
            "active_strategies": ["FAST_TP50_SL30"],   # main slot
            "all_real_strategies": ["FAST_TP50_SL30"],
            "shadow_enabled": True,
            "top_n": 1,
            "budget_usd": 50.0,
            "ca_filter": False,
        }
        from datetime import datetime, timezone
        open_paper_trades(client, [token], cycle_ts=datetime.now(timezone.utc), config=config)

        shadow_rows = [r for r in store if r.get("is_shadow") is True]
        self.assertGreater(len(shadow_rows), 0, "no shadow rows inserted")

        for r in shadow_rows:
            with self.subTest(strategy=r.get("strategy")):
                self.assertGreater(r.get("position_usd", 0), 0,
                                   "v144.3 invariant: position_usd > 0")
                self.assertIn(r.get("entry_source"), {"ultra", "dexscreener", "live_sync"},
                              "v144.3 invariant: entry_source tagged")
                self.assertAlmostEqual(r.get("dex_spot_price_at_entry"), r.get("entry_price"), places=10,
                                       msg="v144.3 invariant: dex_spot_price_at_entry == entry_price")
                self.assertGreaterEqual(r.get("high_price_seen", 0), r.get("entry_price", 0) * 0.999,
                                        "v144.3 invariant: high_price_seen >= entry_price")
                self.assertIsNotNone(r.get("sol_price_at_entry"),
                                     "v144.3 invariant: sol_price_at_entry set")

        # Make sure mains also keep their invariants (no regression on main path)
        main_rows = [r for r in store if r.get("is_shadow") in (False, None) and r.get("strategy") == "FAST_TP50_SL30"]
        for r in main_rows:
            self.assertGreater(r.get("position_usd", 0), 0)


if __name__ == "__main__":
    unittest.main()
