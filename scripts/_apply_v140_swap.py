"""v140 — add 8 mega-sweep winner configs as paper strats + reset bankroll $1000 each.

New strategies (all at $1000 fresh bankroll for fair A/B):
  FAST_TP100_SL20_HYST       — top 10 sim +$151
  FAST_TP80_SL25_HYST        — top 10 sim +$140
  BE25_TP80_SL30_HYST        — top 10 sim +$139
  FAST_TP50_SL30_HYST        — top 10 sim +$135
  BE25_TP80_SL30_S30_HYST    — best SCORE30 filter, avg +19.90%
  BE15_TP70_SL50_NZ          — best NOZEROLIQ filter, sim +$87
  BE25_TP80_SL30_NZS30_HYST  — best NOZEROLIQ_SCORE30, avg +25.67% (N=27)
  BE15_TP300_SL50_MCAP       — best MCAP_MID, sim +$85

Existing 10 strats stay active with current configs. User said "ne pas enlever".
Total paper: 18 strats × $1000 = $18,000 bankroll. All reset to $1000 fresh
for clean A/B comparison.

Run: python scripts/_apply_v140_swap.py [--apply]
"""
from __future__ import annotations
import os
import sys
import json
import copy
from datetime import datetime, timezone

from dotenv import load_dotenv
from supabase import create_client

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SCRAPER = os.path.join(ROOT, "scraper")
load_dotenv(os.path.join(SCRAPER, ".env"))
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

# All 18 active strats (10 existing + 8 new v140) with their optimal configs
OPTIMAL = {
    # EXISTING (from v139) — keep their current configs
    "BE25_TP80_SL30":         {"price_source": "median_5",   "polling_sec": 240},
    "BE25_TP80_SL30_DS":      {"price_source": "ds",         "polling_sec": 30},
    "FAST_TP100_SL20":        {"price_source": "ds",         "polling_sec": 30},
    "FAST_TP80_SL25":         {"price_source": "ds",         "polling_sec": 30},
    "FAST_TP50_SL30":         {"price_source": "median_3",   "polling_sec": 30},
    "FAST_TP40_SL30":         {"price_source": "hysteresis", "polling_sec": 30},
    "TP50_SL15":              {"price_source": "jupiter",    "polling_sec": 30},
    "BE15_TP100_SL50":        {"price_source": "ds",         "polling_sec": 30},
    "NOZEROLIQ_TP200_SL40":   {"price_source": "jupiter",    "polling_sec": 120},
    "HIGHSCORE_TP200_SL40":   {"price_source": "jupiter",    "polling_sec": 120},
    # NEW v140 — top 10 (hysteresis + lazy)
    "FAST_TP100_SL20_HYST":   {"price_source": "hysteresis", "polling_sec": 30},
    "FAST_TP80_SL25_HYST":    {"price_source": "hysteresis", "polling_sec": 30},
    "BE25_TP80_SL30_HYST":    {"price_source": "hysteresis", "polling_sec": 30},
    "FAST_TP50_SL30_HYST":    {"price_source": "hysteresis", "polling_sec": 30},
    # NEW v140 — best per filter
    "BE25_TP80_SL30_S30_HYST":    {"price_source": "hysteresis", "polling_sec": 240},
    "BE15_TP70_SL50_NZ":          {"price_source": "jupiter",    "polling_sec": 240},
    "BE25_TP80_SL30_NZS30_HYST":  {"price_source": "hysteresis", "polling_sec": 240},
    "BE15_TP300_SL50_MCAP":       {"price_source": "ds",         "polling_sec": 30},
}

NEW_PAPER_ACTIVE = list(OPTIMAL.keys())
PER_STRAT_BANKROLL = 1000.0


def _diff(label, old, new):
    if old == new: return
    print(f"\n--- {label} ---")
    print(f"OLD: {json.dumps(old, indent=2, sort_keys=True)[:800]}")
    print(f"NEW: {json.dumps(new, indent=2, sort_keys=True)[:800]}")


def main(apply: bool):
    cur = sb.table("scoring_config").select(
        "paper_trade_config,rt_trade_config").eq("id", 1).execute().data[0]
    ptc = copy.deepcopy(cur["paper_trade_config"])
    rtc = copy.deepcopy(cur["rt_trade_config"])

    old_active = list(ptc.get("active_strategies", []))
    ptc["active_strategies"] = NEW_PAPER_ACTIVE

    deprecated = list(ptc.get("deprecated_strategies", []))
    deprecated = [s for s in deprecated if s not in NEW_PAPER_ACTIVE]
    ptc["deprecated_strategies"] = sorted(deprecated)

    old_hybrid = dict(rtc.get("hybrid_strategy", {}).get("allocations", {}))
    rtc.setdefault("hybrid_strategy", {})["allocations"] = {s: 1 for s in NEW_PAPER_ACTIVE}

    old_overrides = dict(rtc.get("strategy_overrides", {}))
    rtc["strategy_overrides"] = dict(OPTIMAL)

    # Bankroll reset — $1000 per active strat
    br = sb.table("rt_bankroll").select("*").eq("id", 1).execute().data[0]
    new_strategy_br = {
        s: {"pnl": 0.0, "trades": 0, "balance": PER_STRAT_BANKROLL,
            "starting_balance": PER_STRAT_BANKROLL}
        for s in NEW_PAPER_ACTIVE
    }
    new_total = PER_STRAT_BANKROLL * len(NEW_PAPER_ACTIVE)
    new_br_row = {
        "starting_capital": new_total,
        "current_balance": new_total,
        "total_trades": 0, "total_pnl": 0.0,
        "peak_balance": new_total, "max_drawdown_pct": 0.0,
        "strategy_bankrolls": new_strategy_br,
        "last_updated_at": datetime.now(timezone.utc).isoformat(),
    }

    print("=" * 80)
    print("v140 SWAP — DRY RUN" if not apply else "v140 SWAP — APPLY")
    print("=" * 80)
    print(f"Active strats: {len(old_active)} → {len(NEW_PAPER_ACTIVE)}")
    print(f"  added: {sorted(set(NEW_PAPER_ACTIVE) - set(old_active))}")
    print(f"  kept:  {sorted(set(NEW_PAPER_ACTIVE) & set(old_active))}")
    print(f"  removed: {sorted(set(old_active) - set(NEW_PAPER_ACTIVE))}")

    _diff("strategy_overrides (added only)",
          {k: v for k, v in old_overrides.items() if k in OPTIMAL},
          {k: v for k, v in OPTIMAL.items()})

    print(f"\n--- rt_bankroll RESET ---")
    print(f"OLD starting=${br['starting_capital']:.0f} current=${br['current_balance']:.0f} pnl=${br['total_pnl']:.0f} trades={br['total_trades']}")
    print(f"NEW starting=${new_total:.0f} current=${new_total:.0f} pnl=$0 trades=0")
    print(f"  {len(NEW_PAPER_ACTIVE)} strats × ${PER_STRAT_BANKROLL:.0f} fresh for A/B")

    if not apply:
        print("\nDRY RUN — pass --apply to commit.")
        return

    sb.table("scoring_config").update({
        "paper_trade_config": ptc, "rt_trade_config": rtc,
        "updated_by": "v140_swap",
        "change_reason": "add 8 mega-sweep v140 hysteresis/filter variants for A/B test",
    }).eq("id", 1).execute()
    sb.table("rt_bankroll").update(new_br_row).eq("id", 1).execute()
    print(f"\n✅ APPLIED. Total bankroll ${new_total:.0f} across {len(NEW_PAPER_ACTIVE)} strats.")
    print("⚠️  Code change in strategies.py needs deploy for new strats to activate.")


if __name__ == "__main__":
    main("--apply" in sys.argv)
