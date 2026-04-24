"""v14e.16 — seed the 3 AGE-window shadows in DB.

Adds AGE24/48/72_FAST_TP50_SL30 to:
- scoring_config.rt_trade_config.hybrid_strategy.allocations (alloc=1 each)
- rt_bankroll.strategy_bankrolls_per_chain.solana (seed $1000 each)

They will run paper-only because they are NOT in live_trading.allocations.
Idempotent.
"""
import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))

from supabase import create_client

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

NEW_STRATS = [
    "AGE24_FAST_TP50_SL30",
    "AGE48_FAST_TP50_SL30",
    "AGE72_FAST_TP50_SL30",
]
SEED_BANKROLL = 1000.0


def main():
    sc = sb.table("scoring_config").select("rt_trade_config").eq("id", 1).execute()
    cfg = sc.data[0].get("rt_trade_config") or {}
    allocs = ((cfg.get("hybrid_strategy") or {}).get("allocations") or {})
    before_alloc = dict(allocs)
    for s in NEW_STRATS:
        if s not in allocs:
            allocs[s] = 1
    if "hybrid_strategy" not in cfg:
        cfg["hybrid_strategy"] = {}
    cfg["hybrid_strategy"]["allocations"] = allocs

    br = sb.table("rt_bankroll").select(
        "id,strategy_bankrolls_per_chain"
    ).limit(1).execute()
    br_row = br.data[0]
    br_pc = dict(br_row.get("strategy_bankrolls_per_chain") or {})
    sol_br = dict(br_pc.get("solana") or {})
    before_br = dict(sol_br)
    for s in NEW_STRATS:
        if s not in sol_br:
            sol_br[s] = {"current_balance": SEED_BANKROLL, "seeded_at": "v14e.16"}
    br_pc["solana"] = sol_br

    print(f"allocations before: {len(before_alloc)}  after: {len(allocs)}")
    print(f"solana bankrolls before: {len(before_br)}  after: {len(sol_br)}")

    sb.table("scoring_config").update({"rt_trade_config": cfg}).eq("id", 1).execute()
    sb.table("rt_bankroll").update(
        {"strategy_bankrolls_per_chain": br_pc}
    ).eq("id", br_row["id"]).execute()

    added = [s for s in NEW_STRATS if s not in before_alloc]
    print(f"\nAdded {len(added)} AGE-window shadows to paper: {added}")


if __name__ == "__main__":
    main()
