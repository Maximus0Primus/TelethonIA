"""v139 — add NOZEROLIQ_TP200_SL40 + HIGHSCORE_TP200_SL40 to paper.

Tested on 71 post-v132 tokens (shadow data ground truth):
  NOZEROLIQ_TP200_SL40   N=44 WR=48% avg=+14.91% med=-3.77%  → +$83/jour
  HIGHSCORE_TP200_SL40   N=38 WR=50% avg=+14.42% med=-1.97%  → +$69/jour
  vs BASELINE BE25       N=71 WR=37% avg= +4.76% med=-5.65%  → +$43/jour

Both use TP200/SL40 (3x take, 0.6 stop, 4h horizon) with entry filters.
Polling: jupiter/120s static (sweep showed long-poll best for long-horizon).

Run: python scripts/_apply_v139_swap.py [--apply]
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

NEW_STRATS = ["NOZEROLIQ_TP200_SL40", "HIGHSCORE_TP200_SL40"]
NEW_OVERRIDES = {
    "NOZEROLIQ_TP200_SL40": {"price_source": "jupiter", "polling_sec": 120},
    "HIGHSCORE_TP200_SL40": {"price_source": "jupiter", "polling_sec": 120},
}
PER_STRAT_BANKROLL = 1000.0


def _diff(label, old, new):
    if old == new:
        return
    print(f"\n--- {label} ---")
    print(f"OLD: {json.dumps(old, indent=2, sort_keys=True)[:600]}")
    print(f"NEW: {json.dumps(new, indent=2, sort_keys=True)[:600]}")


def main(apply: bool):
    cur = sb.table("scoring_config").select(
        "paper_trade_config,rt_trade_config").eq("id", 1).execute().data[0]
    ptc = copy.deepcopy(cur["paper_trade_config"])
    rtc = copy.deepcopy(cur["rt_trade_config"])

    # 1. Add to paper.active_strategies
    old_active = list(ptc.get("active_strategies", []))
    new_active = list(old_active)
    for s in NEW_STRATS:
        if s not in new_active:
            new_active.append(s)
    ptc["active_strategies"] = new_active

    # Pull from deprecated if present
    deprecated = list(ptc.get("deprecated_strategies", []))
    deprecated = [s for s in deprecated if s not in NEW_STRATS]
    ptc["deprecated_strategies"] = sorted(deprecated)

    # 2. hybrid mirrors paper
    old_hybrid = dict(rtc.get("hybrid_strategy", {}).get("allocations", {}))
    rtc.setdefault("hybrid_strategy", {})["allocations"] = {s: 1 for s in new_active}

    # 3. strategy_overrides — add new entries
    old_overrides = dict(rtc.get("strategy_overrides", {}))
    new_overrides = dict(old_overrides)
    new_overrides.update(NEW_OVERRIDES)
    rtc["strategy_overrides"] = new_overrides

    # 4. Bankroll: add $1000 fresh entries for new strats
    br = sb.table("rt_bankroll").select("*").eq("id", 1).execute().data[0]
    sb_alloc = dict(br.get("strategy_bankrolls") or {})
    for s in NEW_STRATS:
        if s not in sb_alloc:
            sb_alloc[s] = {
                "pnl": 0.0, "trades": 0,
                "balance": PER_STRAT_BANKROLL,
                "starting_balance": PER_STRAT_BANKROLL,
            }
    new_total = float(br["starting_capital"]) + PER_STRAT_BANKROLL * len(NEW_STRATS)
    new_balance = float(br["current_balance"]) + PER_STRAT_BANKROLL * len(NEW_STRATS)
    new_peak = max(float(br["peak_balance"]), new_balance)

    print("=" * 80)
    print("v139 SWAP — DRY RUN" if not apply else "v139 SWAP — APPLY")
    print("=" * 80)
    _diff("paper.active_strategies", old_active, ptc["active_strategies"])
    _diff("hybrid.allocations", old_hybrid, rtc["hybrid_strategy"]["allocations"])
    _diff("strategy_overrides (added)", {}, NEW_OVERRIDES)
    print(f"\n--- rt_bankroll ---")
    print(f"OLD starting=${br['starting_capital']:.0f} current=${br['current_balance']:.0f}")
    print(f"NEW starting=${new_total:.0f} current=${new_balance:.0f} (added 2 × ${PER_STRAT_BANKROLL:.0f})")
    print(f"strategy_bankrolls: added entries for {NEW_STRATS}")

    if not apply:
        print("\nDRY RUN — pass --apply to commit.")
        return

    # Apply config
    sb.table("scoring_config").update({
        "paper_trade_config": ptc,
        "rt_trade_config": rtc,
        "updated_by": "v139_swap",
        "change_reason": "add NOZEROLIQ + HIGHSCORE TP200 strategies (mega test winners)",
    }).eq("id", 1).execute()
    # Apply bankroll
    sb.table("rt_bankroll").update({
        "starting_capital": new_total,
        "current_balance": new_balance,
        "peak_balance": new_peak,
        "strategy_bankrolls": sb_alloc,
    }).eq("id", 1).execute()
    print(f"\n✅ APPLIED. Total bankroll now ${new_total:.0f} ({len(new_active)} active strats).")
    print("⚠️  Code change in strategies.py + paper_trader.py needs deploy.")


if __name__ == "__main__":
    main("--apply" in sys.argv)
