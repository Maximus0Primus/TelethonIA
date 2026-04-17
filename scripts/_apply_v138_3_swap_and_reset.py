"""v138.3 — BE25 config tweak + full bankroll reset to $1000 per strat.

Changes:
  1. BE25_TP80_SL30 strategy_overrides: ds/lazy → median_5/static_240
     (re-rank by avg_pnl_pct showed +0.95pp avg gain)
  2. Reset rt_bankroll: starting_capital=$8000 ($1000 × 8 active strats)
     current_balance=$8000, total_pnl=0, total_trades=0, peak=$8000
     strategy_bankrolls: each active strat → fresh $1000 bankroll
     Removed strats (DTRAIL etc) dropped from strategy_bankrolls.

Run: python scripts/_apply_v138_3_swap_and_reset.py [--apply]
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

# Active strategies post-v138.2 (8 paper)
ACTIVE = ["BE25_TP80_SL30", "BE25_TP80_SL30_DS", "FAST_TP100_SL20",
          "FAST_TP80_SL25", "FAST_TP50_SL30", "FAST_TP40_SL30",
          "TP50_SL15", "BE15_TP100_SL50"]
PER_STRAT_BANKROLL = 1000.0


def _diff(label, old, new):
    if old == new:
        return
    print(f"\n--- {label} ---")
    print(f"OLD: {json.dumps(old, indent=2, sort_keys=True)[:600]}")
    print(f"NEW: {json.dumps(new, indent=2, sort_keys=True)[:600]}")


def main(apply: bool):
    # 1. BE25 config tweak
    cur = sb.table("scoring_config").select(
        "rt_trade_config").eq("id", 1).execute().data[0]
    rtc = copy.deepcopy(cur["rt_trade_config"])
    old_overrides = dict(rtc.get("strategy_overrides", {}))
    new_overrides = dict(old_overrides)
    new_overrides["BE25_TP80_SL30"] = {
        "price_source": "median_5",
        "polling_sec": 240,
    }
    rtc["strategy_overrides"] = new_overrides

    # 2. Bankroll reset
    br = sb.table("rt_bankroll").select("*").eq("id", 1).execute().data[0]
    new_strategy_br = {
        s: {"pnl": 0.0, "trades": 0, "balance": PER_STRAT_BANKROLL,
            "starting_balance": PER_STRAT_BANKROLL}
        for s in ACTIVE
    }
    new_total = PER_STRAT_BANKROLL * len(ACTIVE)
    new_br = {
        "starting_capital": new_total,
        "current_balance": new_total,
        "total_trades": 0,
        "total_pnl": 0.0,
        "peak_balance": new_total,
        "max_drawdown_pct": 0.0,
        "strategy_bankrolls": new_strategy_br,
        "last_updated_at": datetime.now(timezone.utc).isoformat(),
    }

    print("=" * 80)
    print("v138.3 — BE25 config tweak + bankroll reset" +
          (" (DRY RUN)" if not apply else " (APPLY)"))
    print("=" * 80)

    _diff("strategy_overrides.BE25_TP80_SL30",
          old_overrides.get("BE25_TP80_SL30"),
          new_overrides["BE25_TP80_SL30"])

    print("\n--- rt_bankroll reset ---")
    print(f"OLD starting_capital=${br['starting_capital']:.2f} "
          f"current=${br['current_balance']:.2f} pnl=${br['total_pnl']:.2f} "
          f"trades={br['total_trades']} peak=${br['peak_balance']:.2f}")
    print(f"NEW starting_capital=${new_total:.2f} current=${new_total:.2f} "
          f"pnl=$0 trades=0 peak=${new_total:.2f}")
    print(f"\nstrategy_bankrolls (8 strats × ${PER_STRAT_BANKROLL:.0f}):")
    for s in ACTIVE:
        old = br["strategy_bankrolls"].get(s)
        old_str = (f"OLD: balance=${old['balance']:.2f} pnl=${old['pnl']:.2f} "
                   f"trades={old['trades']}") if old else "OLD: (new)"
        print(f"  {s:<22}  {old_str}")
    dropped = [s for s in br["strategy_bankrolls"] if s not in ACTIVE]
    if dropped:
        print(f"\nDropped from strategy_bankrolls (deprecated):")
        for s in dropped:
            old = br["strategy_bankrolls"][s]
            print(f"  {s:<22}  was ${old['balance']:.2f} ({old['pnl']:+.2f})")

    if not apply:
        print("\nDRY RUN — pass --apply to commit.")
        return

    # Apply
    sb.table("scoring_config").update({
        "rt_trade_config": rtc,
        "updated_by": "v138_3_swap",
        "change_reason": "BE25 → median_5/static_240 (avg_pnl_pct rerank)",
    }).eq("id", 1).execute()
    sb.table("rt_bankroll").update(new_br).eq("id", 1).execute()
    print("\n✅ APPLIED. Config + bankroll reset committed.")
    print("⚠️  Code change in strategies.py (BE25 out of LAZY_STRATEGIES) needs deploy.")


if __name__ == "__main__":
    main("--apply" in sys.argv)
