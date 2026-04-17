"""v138 strategy swap — drop losing DTRAIL from paper + upgrade live FAST→BE15.

Decisions (from --from-trades ground truth):
  PAPER active_strategies:
    REMOVE: DTRAIL5_ACT10_SL60 (-1.88%), DTRAIL10_ACT5_SL50 (-2%)
    KEEP:   BE25_TP80_SL30 (+4.2%), BE25_TP80_SL30_DS, FAST_TP100_SL20 (+1.6%),
            TP50_SL15 (+5.5% #1 ground truth), BE15_TP100_SL50 (+6.0% #2)

  LIVE allocations (50/50):
    BE25_TP80_SL30 0.5  (kept — confirmed +4.2% real)
    FAST_TP100_SL20 0.5 -> REPLACED with BE15_TP100_SL50 0.5
    (FAST stays in paper as control; live gets the higher-edge BE15)

Run: python scripts/_apply_v138_swap.py [--apply]
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

REMOVE_PAPER = ["DTRAIL5_ACT10_SL60", "DTRAIL10_ACT5_SL50"]

NEW_LIVE_ALLOCATIONS = {
    "BE25_TP80_SL30": 0.5,
    "BE15_TP100_SL50": 0.5,
}

# BE15_TP100_SL50 already added in v137; just keep its override


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

    # 1. paper.active_strategies — remove the 2 losing DTRAIL
    old_active = list(ptc.get("active_strategies", []))
    new_active = [s for s in old_active if s not in REMOVE_PAPER]
    ptc["active_strategies"] = new_active

    deprecated = list(ptc.get("deprecated_strategies", []))
    for s in REMOVE_PAPER:
        if s not in deprecated:
            deprecated.append(s)
    ptc["deprecated_strategies"] = sorted(deprecated)

    # 2. live_trading.allocations — swap FAST for BE15
    old_live = dict(rtc.get("live_trading", {}).get("allocations", {}))
    rtc.setdefault("live_trading", {})["allocations"] = dict(NEW_LIVE_ALLOCATIONS)

    # 3. hybrid_strategy.allocations — sync to new active list
    old_hybrid = dict(rtc.get("hybrid_strategy", {}).get("allocations", {}))
    rtc.setdefault("hybrid_strategy", {})["allocations"] = {s: 1 for s in new_active}

    # 4. strategy_overrides — drop overrides for removed strats
    old_overrides = dict(rtc.get("strategy_overrides", {}))
    rtc["strategy_overrides"] = {k: v for k, v in old_overrides.items()
                                  if k not in REMOVE_PAPER}

    print("=" * 80)
    print("v138 STRATEGY SWAP — DRY RUN" if not apply else "v138 STRATEGY SWAP — APPLY")
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
                      "updated_by": "v138_swap_script",
                      "change_reason": "drop losing DTRAIL paper + live FAST->BE15 (ground truth)"})
             .eq("id", 1).execute())
    print(f"\nUPDATED: {len(res.data)} row(s) — VPS picks up via 60s config cache TTL")


if __name__ == "__main__":
    main("--apply" in sys.argv)
