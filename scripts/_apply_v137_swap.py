"""v137 strategy swap — applies the post-v136 cadence-corrected ranking.

Decisions (from from-trades ground truth + v136 realistic sim):
  PAPER active_strategies:
    REMOVE: DTRAIL3_ACT10_SL70, DTRAIL10_ACT10_SL70, DTRAIL3_ACT20_SL50
    KEEP:   DTRAIL5_ACT10_SL60, BE25_TP80_SL30, BE25_TP80_SL30_DS, FAST_TP100_SL20
    ADD:    TP50_SL15 (#1 ground truth), BE15_TP100_SL50 (#2 ground truth),
            DTRAIL10_ACT5_SL50 (best-of-DTRAIL ground truth)

  LIVE allocations (50/50):
    REMOVE: DTRAIL3_ACT10_SL70 0.5, DTRAIL10_ACT10_SL70 0.5
    ADD:    BE25_TP80_SL30 0.5, FAST_TP100_SL20 0.5

  Hybrid + strategy_overrides + deprecated_strategies kept consistent.

Run: python scripts/_apply_v137_swap.py [--apply]
Without --apply, prints the diff dry-run.
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

REMOVE_PAPER = ["DTRAIL3_ACT10_SL70", "DTRAIL10_ACT10_SL70", "DTRAIL3_ACT20_SL50"]
ADD_PAPER = ["TP50_SL15", "BE15_TP100_SL50", "DTRAIL10_ACT5_SL50"]

NEW_LIVE_ALLOCATIONS = {
    "BE25_TP80_SL30": 0.5,
    "FAST_TP100_SL20": 0.5,
}

# Orchestration for newly added strategies
NEW_OVERRIDES = {
    "TP50_SL15":          {"polling_sec": 60, "price_source": "jupiter"},
    "BE15_TP100_SL50":    {"polling_sec": 120, "price_source": "jupiter"},
    "DTRAIL10_ACT5_SL50": {"polling_sec": 120, "price_source": "jupiter"},
}


def _diff(label: str, old, new):
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

    # 1. paper_trade_config.active_strategies
    old_active = list(ptc.get("active_strategies", []))
    new_active = [s for s in old_active if s not in REMOVE_PAPER]
    for s in ADD_PAPER:
        if s not in new_active:
            new_active.append(s)
    ptc["active_strategies"] = new_active

    # Move removed strats to deprecated; pull added strats out of deprecated
    deprecated = list(ptc.get("deprecated_strategies", []))
    for s in REMOVE_PAPER:
        if s not in deprecated:
            deprecated.append(s)
    deprecated = [s for s in deprecated if s not in ADD_PAPER]
    ptc["deprecated_strategies"] = sorted(deprecated)

    # 2. rt_trade_config.live_trading.allocations
    old_live = dict(rtc.get("live_trading", {}).get("allocations", {}))
    rtc.setdefault("live_trading", {})["allocations"] = dict(NEW_LIVE_ALLOCATIONS)

    # 3. rt_trade_config.hybrid_strategy.allocations — mirror active_strategies
    old_hybrid = dict(rtc.get("hybrid_strategy", {}).get("allocations", {}))
    new_hybrid = {s: 1 for s in new_active}
    rtc.setdefault("hybrid_strategy", {})["allocations"] = new_hybrid

    # 4. rt_trade_config.strategy_overrides
    old_overrides = dict(rtc.get("strategy_overrides", {}))
    new_overrides = {k: v for k, v in old_overrides.items() if k not in REMOVE_PAPER}
    new_overrides.update(NEW_OVERRIDES)
    rtc["strategy_overrides"] = new_overrides

    # 5. drop dead strategy_multipliers entries (only keep entries used by live)
    old_mults = dict(rtc.get("strategy_multipliers", {}))
    # strategy_multipliers is dead-compute per todo. Just leave it; sim doesn't use it.

    # ---- Print diffs ----
    print("=" * 80)
    print("v137 STRATEGY SWAP — DRY RUN" if not apply else "v137 STRATEGY SWAP — APPLY")
    print("=" * 80)
    _diff("paper_trade_config.active_strategies",
          old_active, ptc["active_strategies"])
    _diff("paper_trade_config.deprecated_strategies",
          cur["paper_trade_config"].get("deprecated_strategies"),
          ptc["deprecated_strategies"])
    _diff("rt_trade_config.live_trading.allocations",
          old_live, rtc["live_trading"]["allocations"])
    _diff("rt_trade_config.hybrid_strategy.allocations",
          old_hybrid, rtc["hybrid_strategy"]["allocations"])
    _diff("rt_trade_config.strategy_overrides",
          old_overrides, rtc["strategy_overrides"])

    if not apply:
        print("\nDRY RUN — pass --apply to commit.")
        return

    # ---- Apply ----
    res = (sb.table("scoring_config")
             .update({"paper_trade_config": ptc, "rt_trade_config": rtc,
                      "updated_by": "v137_swap_script",
                      "change_reason": "v136-realistic+from-trades reranking"})
             .eq("id", 1).execute())
    print(f"\nUPDATED: {len(res.data)} row(s)")
    print("Config cache TTL ~60s — VPS will pick up automatically.")


if __name__ == "__main__":
    apply = "--apply" in sys.argv
    main(apply)
