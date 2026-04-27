"""Fix the 6 promoted strats bankroll entries.

Bug: _promote_6_paper_mains.py seeded with field names current_balance/total_pnl/
total_trades. paper_trader.py reads/writes balance/pnl/trades. On 1st close,
.get('balance', 500) defaults to $500 (not $1000).

Fix: rewrite the 6 entries in flat + per_chain so they have proper schema:
  {
    "balance": 1000 + observed_pnl_4h,    # corrects $500 phantom
    "pnl":     observed_pnl_4h,            # already correct
    "trades":  observed_trades,            # already correct
    "starting_balance": 1000,
    "last_updated_at": now,
  }
We DROP the wrong keys (current_balance/total_pnl/total_trades) so paper_trader
no longer sees them.
"""
import os, sys, json, copy
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

NEW_SOL = ["BE25_LOCK10_TP100_SL30_NZ_S40", "BE25_LOCK10_TP100_SL30_S40", "FAST45_TP40_SL30_S30"]
NEW_ETH = ["ETH_SLOW4H_TP100_SL50", "ETH_BE50_TP150_SL40_T2H", "ETH_TP300_SL50_4H"]
SEED = 1000.0

br = sb.table("rt_bankroll").select("*").limit(1).execute().data[0]
br_id = br["id"]

# Backup
ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
bk_dir = os.path.join(os.path.dirname(__file__), "..", "data")
bk_path = os.path.join(bk_dir, f"rt_bankroll_pre_promote_fix_{ts}.json")
with open(bk_path, "w") as f:
    json.dump(br, f, indent=2, default=str)
print(f"backup -> {bk_path}\n")

new_flat = copy.deepcopy(br.get("strategy_bankrolls", {}) or {})
new_per_chain = copy.deepcopy(br.get("strategy_bankrolls_per_chain", {}) or {})
sol = new_per_chain.setdefault("solana", {})
eth = new_per_chain.setdefault("ethereum", {})
now_iso = datetime.now(timezone.utc).isoformat()


def rebuild(entry: dict) -> dict:
    """Take an entry that may have wrong-schema keys + new-schema keys, rebuild
    it so it ONLY has new-schema keys (balance/pnl/trades/starting_balance)."""
    pnl = float(entry.get("pnl", 0) or 0)
    trades = int(entry.get("trades", 0) or 0)
    return {
        "balance": round(SEED + pnl, 2),
        "pnl": round(pnl, 2),
        "trades": trades,
        "starting_balance": SEED,
        "last_updated_at": now_iso,
    }


print("Rebuilding entries (drop wrong keys, recompute balance from $1000 base):\n")
print(f"  {'strat':<40}{'old_bal':>10}{'pnl':>9}{'new_bal':>10}")
for s in NEW_SOL:
    old = new_flat.get(s, {})
    fixed = rebuild(old)
    new_flat[s] = fixed
    sol[s] = copy.deepcopy(fixed)
    print(f"  [SOL] {s:<34}{old.get('balance', '—'):>10}{fixed['pnl']:>+9}{fixed['balance']:>10}")
for s in NEW_ETH:
    old = new_flat.get(s, {})
    fixed = rebuild(old)
    new_flat[s] = fixed
    eth[s] = copy.deepcopy(fixed)
    print(f"  [ETH] {s:<34}{old.get('balance', '—'):>10}{fixed['pnl']:>+9}{fixed['balance']:>10}")

sb.table("rt_bankroll").update({
    "strategy_bankrolls": new_flat,
    "strategy_bankrolls_per_chain": new_per_chain,
    "last_updated_at": now_iso,
}).eq("id", br_id).execute()
print("\nrt_bankroll updated. Restart not required (paper_trader re-reads each loop).")
