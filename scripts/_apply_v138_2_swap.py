"""v138.2 swap — promote mega-sweep winners to paper, optimize live BE25.

Decisions (from --from-trades + 9040-config mega sweep with valid configs only):
  PAPER active_strategies (8):
    KEEP+update: BE25_TP80_SL30 (ds, lazy via code)
                 BE25_TP80_SL30_DS (ds, lazy)
                 FAST_TP100_SL20 (ds, lazy)
                 TP50_SL15 (jupiter, lazy)
                 BE15_TP100_SL50 (ds, fast=poll_sec=30)
    ADD:         FAST_TP80_SL25  (ds, lazy) — kelly 21.29
                 FAST_TP50_SL30  (median_3, lazy) — kelly 19.07
                 FAST_TP40_SL30  (hysteresis, lazy) — kelly 19.04

  LIVE allocations (50/50): BE25 + BE15 — same as v138.1 but now use the
  optimal price_source/polling configurations from the sweep.

  LAZY mode is enabled for the strats in strategies.py:LAZY_STRATEGIES set
  (code change v138.2). For LAZY strats, polling_sec in DB is ignored.

Run: python scripts/_apply_v138_2_swap.py [--apply]
"""
from __future__ import annotations
import os
import sys
import json
import copy

from dotenv import load_dotenv
from supabase import create_client

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SCRAPER = os.path.join(ROOT, "scraper")
load_dotenv(os.path.join(SCRAPER, ".env"))
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

# Optimal config per strategy (from mega sweep, production-valid only)
OPTIMAL = {
    "BE25_TP80_SL30":     {"price_source": "ds",         "polling_sec": 30},  # lazy via code
    "BE25_TP80_SL30_DS":  {"price_source": "ds",         "polling_sec": 30},  # lazy via code
    "FAST_TP100_SL20":    {"price_source": "ds",         "polling_sec": 30},  # lazy via code
    "FAST_TP80_SL25":     {"price_source": "ds",         "polling_sec": 30},  # lazy via code
    "FAST_TP50_SL30":     {"price_source": "median_3",   "polling_sec": 30},  # lazy via code
    "FAST_TP40_SL30":     {"price_source": "hysteresis", "polling_sec": 30},  # lazy via code
    "TP50_SL15":          {"price_source": "jupiter",    "polling_sec": 30},  # lazy via code
    "BE15_TP100_SL50":    {"price_source": "ds",         "polling_sec": 30},  # FAST mode (no lazy)
}

NEW_PAPER_ACTIVE = list(OPTIMAL.keys())

# Live unchanged: BE25 + BE15 (now both with optimal configs)
NEW_LIVE_ALLOCATIONS = {
    "BE25_TP80_SL30": 0.5,
    "BE15_TP100_SL50": 0.5,
}


def _diff(label, old, new):
    if old == new:
        return
    print(f"\n--- {label} ---")
    print(f"OLD: {json.dumps(old, indent=2, sort_keys=True)}")
    print(f"NEW: {json.dumps(new, indent=2, sort_keys=True)}")


def main(apply: bool):
    cur = sb.table("scoring_config").select(
        "paper_trade_config,rt_trade_config").eq("id", 1).execute().data[0]
    ptc = copy.deepcopy(cur["paper_trade_config"])
    rtc = copy.deepcopy(cur["rt_trade_config"])

    # 1. paper.active_strategies
    old_active = list(ptc.get("active_strategies", []))
    ptc["active_strategies"] = NEW_PAPER_ACTIVE

    # Move what was removed -> deprecated
    deprecated = list(ptc.get("deprecated_strategies", []))
    for s in old_active:
        if s not in NEW_PAPER_ACTIVE and s not in deprecated:
            deprecated.append(s)
    deprecated = [s for s in deprecated if s not in NEW_PAPER_ACTIVE]
    ptc["deprecated_strategies"] = sorted(deprecated)

    # 2. live unchanged at top-level (same strats), but config below
    old_live = dict(rtc.get("live_trading", {}).get("allocations", {}))
    rtc.setdefault("live_trading", {})["allocations"] = dict(NEW_LIVE_ALLOCATIONS)

    # 3. hybrid mirrors paper.active
    old_hybrid = dict(rtc.get("hybrid_strategy", {}).get("allocations", {}))
    rtc.setdefault("hybrid_strategy", {})["allocations"] = {s: 1 for s in NEW_PAPER_ACTIVE}

    # 4. strategy_overrides — replace with optimal configs
    old_overrides = dict(rtc.get("strategy_overrides", {}))
    rtc["strategy_overrides"] = dict(OPTIMAL)

    print("=" * 80)
    print("v138.2 SWAP — DRY RUN" if not apply else "v138.2 SWAP — APPLY")
    print("=" * 80)
    _diff("paper.active_strategies", old_active, ptc["active_strategies"])
    _diff("paper.deprecated_strategies",
          cur["paper_trade_config"].get("deprecated_strategies"),
          ptc["deprecated_strategies"])
    _diff("live.allocations", old_live, rtc["live_trading"]["allocations"])
    _diff("hybrid.allocations", old_hybrid, rtc["hybrid_strategy"]["allocations"])
    _diff("strategy_overrides", old_overrides, rtc["strategy_overrides"])

    if not apply:
        print("\nDRY RUN — pass --apply to commit.")
        return

    res = (sb.table("scoring_config")
             .update({"paper_trade_config": ptc, "rt_trade_config": rtc,
                      "updated_by": "v138_2_swap_script",
                      "change_reason": "promote mega-sweep winners + LAZY mode (v138.2)"})
             .eq("id", 1).execute())
    print(f"\nUPDATED: {len(res.data)} row(s) — VPS picks up via 60s config cache TTL")
    print("⚠️  NEEDS DEPLOY: strategies.py code change required for LAZY mode to activate.")


if __name__ == "__main__":
    main("--apply" in sys.argv)
