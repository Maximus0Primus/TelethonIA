"""Promote 5 robust shadows to paper main (v14e.54).

Verdict 14d/7d/3d cross-window robustness audit (scripts/_best_strats_robustness.py)
identified these as positives across all 3 windows + N>=20 + WR>=45%:

SOL:
  - BE25_LOCK10_TP80_SL30_S40           N=95, WR 56%, +$18.47/d -> +$36.94/d 7d -> +$62.50/d 3d
  - FAST_TP50_SL30_MCAP_S40             N=97, WR 51%, +$19.90/d -> +$53.22/d 7d -> +$92.55/d 3d
  - BE25_LOCK15_TP200_SL40_4H_NZ_S40    N=80, WR 60%, +$24.29/d (moonshot)
  - BE50_LOCK25_TP200_SL40_4H_MCAP      N=86, WR 48%, +$21.13/d -> +$112.41/d 3d (moonshot)

ETH:
  - ETH_BE25_LOCK15_TP100_SL40_T2H      N=62, WR 69%, +$18.00/d (top WR all chains)
"""
import os
import sys
import json
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

PROMOTE = [
    # (strat_name, chain, alloc)
    ("BE25_LOCK10_TP80_SL30_S40",        "solana",   1),
    ("FAST_TP50_SL30_MCAP_S40",          "solana",   1),
    ("BE25_LOCK15_TP200_SL40_4H_NZ_S40", "solana",   1),
    ("BE50_LOCK25_TP200_SL40_4H_MCAP",   "solana",   1),
    ("ETH_BE25_LOCK15_TP100_SL40_T2H",   "ethereum", 1),
]

ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
backup_dir = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(backup_dir, exist_ok=True)

# === 1. Backup current rt_trade_config + rt_bankroll ===
cfg = sb.table("scoring_config").select("rt_trade_config").eq("id", 1).single().execute().data["rt_trade_config"]
if isinstance(cfg, str):
    cfg = json.loads(cfg)
bkr = sb.table("rt_bankroll").select("*").eq("id", 1).single().execute().data

with open(os.path.join(backup_dir, f"rt_trade_config_pre_promote_{ts}.json"), "w") as f:
    json.dump(cfg, f, indent=2)
with open(os.path.join(backup_dir, f"rt_bankroll_pre_promote_{ts}.json"), "w") as f:
    json.dump(bkr, f, indent=2)
print(f"Backups in {backup_dir}/")

# === 2. Add to hybrid_strategy.allocations ===
allocs = cfg.setdefault("hybrid_strategy", {}).setdefault("allocations", {})
added_to_allocs = []
for strat, chain, alloc in PROMOTE:
    if strat in allocs:
        print(f"  SKIP allocations: {strat} (already present, alloc={allocs[strat]})")
    else:
        allocs[strat] = alloc
        added_to_allocs.append(strat)
        print(f"  ADD allocations: {strat} = {alloc}")

# === 3. Add to strategy_bankrolls_per_chain ===
bkr_per_chain = bkr.setdefault("strategy_bankrolls_per_chain", {})
seeded_count = 0
for strat, chain, alloc in PROMOTE:
    chain_bkr = bkr_per_chain.setdefault(chain, {})
    if strat in chain_bkr:
        print(f"  SKIP bankroll: {strat} on {chain} (existing balance: ${chain_bkr[strat].get('balance', 0):.2f})")
    else:
        chain_bkr[strat] = {
            "pnl": 0,
            "trades": 0,
            "balance": 1000.0,
            "starting_balance": 1000.0,
            "last_updated_at": datetime.now(timezone.utc).isoformat(),
        }
        seeded_count += 1
        print(f"  SEED bankroll: {strat} on {chain} = $1000")

# === 4. Update starting_capital (only count newly seeded) ===
old_starting = float(bkr.get("starting_capital", 0) or 0)
new_starting = old_starting + seeded_count * 1000.0
bkr["starting_capital"] = new_starting
print(f"\nstarting_capital: ${old_starting:.0f} -> ${new_starting:.0f} (+${seeded_count * 1000})")

# === 5. Push ===
sb.table("scoring_config").update({"rt_trade_config": cfg}).eq("id", 1).execute()
sb.table("rt_bankroll").update({
    "strategy_bankrolls_per_chain": bkr_per_chain,
    "starting_capital": new_starting,
}).eq("id", 1).execute()

print(f"\n=== DONE ===")
print(f"  {len(added_to_allocs)} strats added to hybrid_strategy.allocations")
print(f"  {seeded_count} strats seeded $1000 in bankroll")
print(f"  Total paper main allocations: {len(allocs)}")
print(f"\nThe scraper picks up new config on next cycle (5min cache TTL).")
print(f"To force reload: systemctl restart kol-scraper on VPS.")
