"""Promote 3 SOL + 3 ETH shadows to paper main with $1000 bankroll each.

SOL: BE25_LOCK10_TP100_SL30_NZ_S40, BE25_LOCK10_TP100_SL30_S40, FAST45_TP40_SL30_S30
ETH: ETH_SLOW4H_TP100_SL50, ETH_BE50_TP150_SL40_T2H, ETH_TP300_SL50_4H

Backup written to data/rt_bankroll_pre_paper_promote_<ts>.json before writing.
"""
import os, sys, json, copy
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

SOL_NEW = ["BE25_LOCK10_TP100_SL30_NZ_S40",
           "BE25_LOCK10_TP100_SL30_S40",
           "FAST45_TP40_SL30_S30"]
ETH_NEW = ["ETH_SLOW4H_TP100_SL50",
           "ETH_BE50_TP150_SL40_T2H",
           "ETH_TP300_SL50_4H"]
ALL_NEW = SOL_NEW + ETH_NEW
SEED = 1000.0

# 1. Read current config + bankroll
cfg_row = sb.table("scoring_config").select("rt_trade_config").eq("id", 1).execute().data[0]
rt_cfg = cfg_row["rt_trade_config"]
if isinstance(rt_cfg, str):
    rt_cfg = json.loads(rt_cfg)
br = sb.table("rt_bankroll").select("*").limit(1).execute().data[0]
br_id = br["id"]

# 2. Backup
ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
bk_dir = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(bk_dir, exist_ok=True)
bk_path = os.path.join(bk_dir, f"rt_bankroll_pre_paper_promote_{ts}.json")
cfg_path = os.path.join(bk_dir, f"rt_trade_config_pre_paper_promote_{ts}.json")
with open(bk_path, "w") as f:
    json.dump(br, f, indent=2, default=str)
with open(cfg_path, "w") as f:
    json.dump(rt_cfg, f, indent=2, default=str)
print(f"backup bankroll -> {bk_path}")
print(f"backup rt_trade_config -> {cfg_path}")

# 3. Verify shadows exist (sanity)
print("\n[1/4] Verifying shadow data exists for each new strat...")
ok = True
for s in ALL_NEW:
    r = sb.table("paper_trades").select("id", count="exact").eq("strategy", s).eq("source", "rt").limit(1).execute()
    n = r.count or 0
    print(f"  {s:<40}  shadows N={n}")
    if n == 0:
        ok = False
if not ok:
    print("\nERROR: at least one strat has zero shadow rows. Aborting.")
    sys.exit(1)

# 4. Update rt_trade_config.hybrid_strategy.allocations (alloc=1 each)
new_rt = copy.deepcopy(rt_cfg)
allocs = new_rt.setdefault("hybrid_strategy", {}).setdefault("allocations", {})
print("\n[2/4] Adding to rt_trade_config.hybrid_strategy.allocations:")
for s in ALL_NEW:
    if s in allocs:
        print(f"  ALREADY PRESENT  {s}  (alloc={allocs[s]}) — skip")
    else:
        allocs[s] = 1
        print(f"  ADDED            {s}  alloc=1")

# 5. Seed bankrolls per_chain + flat top-level
new_br = copy.deepcopy(br)
seed_entry = lambda: {
    "starting_balance": SEED,
    "current_balance": SEED,
    "total_pnl": 0.0,
    "total_trades": 0,
}
flat = new_br.setdefault("strategy_bankrolls", {})
per_chain = new_br.setdefault("strategy_bankrolls_per_chain", {})
sol_pool = per_chain.setdefault("solana", {})
eth_pool = per_chain.setdefault("ethereum", {})

print("\n[3/4] Seeding $1000 bankroll each:")
seeded = 0
for s in SOL_NEW:
    if s in sol_pool and sol_pool[s].get("current_balance"):
        print(f"  SOL  {s}  ALREADY SEEDED — skip (current={sol_pool[s].get('current_balance')})")
        continue
    sol_pool[s] = seed_entry()
    flat[s] = seed_entry()
    seeded += 1
    print(f"  SOL  {s}  seeded $1000 (per_chain.solana + flat)")
for s in ETH_NEW:
    if s in eth_pool and eth_pool[s].get("current_balance"):
        print(f"  ETH  {s}  ALREADY SEEDED — skip (current={eth_pool[s].get('current_balance')})")
        continue
    eth_pool[s] = seed_entry()
    flat[s] = seed_entry()
    seeded += 1
    print(f"  ETH  {s}  seeded $1000 (per_chain.ethereum + flat)")

added_capital = SEED * seeded
new_br["starting_capital"] = float(br.get("starting_capital") or 0) + added_capital
new_br["current_balance"] = float(br.get("current_balance") or 0) + added_capital
print(f"\n  starting_capital: {br.get('starting_capital')} -> {new_br['starting_capital']}")
print(f"  current_balance:  {br.get('current_balance')} -> {new_br['current_balance']}")

# 6. Write back
print("\n[4/4] Writing to DB...")
sb.table("scoring_config").update({"rt_trade_config": new_rt}).eq("id", 1).execute()
print("  scoring_config.rt_trade_config updated")

br_update = {
    "strategy_bankrolls": new_br["strategy_bankrolls"],
    "strategy_bankrolls_per_chain": new_br["strategy_bankrolls_per_chain"],
    "starting_capital": new_br["starting_capital"],
    "current_balance": new_br["current_balance"],
    "last_updated_at": datetime.now(timezone.utc).isoformat(),
}
sb.table("rt_bankroll").update(br_update).eq("id", br_id).execute()
print("  rt_bankroll updated")

print("\nDONE. Restart VPS (`/deploy`) to reload config.")
