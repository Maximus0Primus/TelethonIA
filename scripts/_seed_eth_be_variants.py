"""One-shot: seed the 3 new ETH BE variants (v14e.13) into live config.

Adds `ETH_BE20_TP100_SL50`, `ETH_BE30_TP100_SL40`, `ETH_BE20_TP80_SL40_T2H`
to:
- scoring_config.rt_trade_config.hybrid_strategy.allocations (alloc=1 each)
- scoring_config.rt_trade_config.strategy_bankrolls_per_chain['ethereum']
  ($1000 seed each, same as Phase 1 baseline)

Idempotent — running twice is safe (skips already-present keys).
"""
import os
import sys
import json
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))

from supabase import create_client

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

NEW_STRATS = [
    "ETH_BE20_TP100_SL50",
    "ETH_BE30_TP100_SL40",
    "ETH_BE20_TP80_SL40_T2H",
]
SEED_BANKROLL = 1000.0


def main():
    # 1) scoring_config.rt_trade_config.hybrid_strategy.allocations
    sc = sb.table("scoring_config").select("rt_trade_config").eq("id", 1).execute()
    if not sc.data:
        print("[err] scoring_config id=1 not found")
        sys.exit(1)
    cfg = sc.data[0].get("rt_trade_config") or {}
    allocs = ((cfg.get("hybrid_strategy") or {}).get("allocations") or {})
    before_alloc = dict(allocs)
    for s in NEW_STRATS:
        if s not in allocs:
            allocs[s] = 1
    if "hybrid_strategy" not in cfg:
        cfg["hybrid_strategy"] = {}
    cfg["hybrid_strategy"]["allocations"] = allocs

    # 2) rt_bankroll.strategy_bankrolls_per_chain['ethereum']
    br = sb.table("rt_bankroll").select("id,strategy_bankrolls_per_chain").limit(1).execute()
    if not br.data:
        print("[err] rt_bankroll has no rows")
        sys.exit(1)
    br_row = br.data[0]
    br_pc = dict(br_row.get("strategy_bankrolls_per_chain") or {})
    eth_br = dict(br_pc.get("ethereum") or {})
    before_br = dict(eth_br)
    for s in NEW_STRATS:
        if s not in eth_br:
            eth_br[s] = {"current_balance": SEED_BANKROLL, "seeded_at": "v14e.13"}
    br_pc["ethereum"] = eth_br

    print("=== Before ===")
    print(f"allocations ({len(before_alloc)}): {sorted(before_alloc.keys())}")
    print(f"eth bankroll keys ({len(before_br)}): {sorted(before_br.keys())}")

    sb.table("scoring_config").update({"rt_trade_config": cfg}).eq("id", 1).execute()
    sb.table("rt_bankroll").update(
        {"strategy_bankrolls_per_chain": br_pc}
    ).eq("id", br_row["id"]).execute()

    print("\n=== After ===")
    print(f"allocations ({len(allocs)}): {sorted(allocs.keys())}")
    print(f"eth bankroll keys ({len(eth_br)}): {sorted(eth_br.keys())}")
    added_alloc = [s for s in NEW_STRATS if s not in before_alloc]
    added_br = [s for s in NEW_STRATS if s not in before_br]
    print(f"\nAdded {len(added_alloc)} to allocations: {added_alloc}")
    print(f"Added {len(added_br)} to eth bankroll: {added_br}")


if __name__ == "__main__":
    main()
