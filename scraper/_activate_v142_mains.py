"""
v142 (Apr 18) — Activate 3 mega-sweep winners as main paper strats ($1000 each).
STRICT read-modify-write: NEVER replace, only APPEND. Preserves all existing
strategies and bankrolls. One-shot script; safe to re-run (idempotent).

Strategies added:
  - FAST_TP70_SL50              (middle TP/SL gap filler, $100/j sim)
  - BE15_TP200_SL40_4H          (BE long-horizon moonshot, $86/j sim)
  - MCAP_MID_DTRAIL5_ACT25_SL50_2H  (mcap-bucket + trail, $81/j sim)
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

SCRAPER_DIR = Path(__file__).resolve().parent
load_dotenv(SCRAPER_DIR / ".env")

NEW_STRATS = [
    "FAST_TP70_SL50",
    "BE15_TP200_SL40_4H",
    "MCAP_MID_DTRAIL5_ACT25_SL50_2H",
]
BANKROLL_INIT = 1000.0

url = os.environ["SUPABASE_URL"]
key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
c = create_client(url, key)

# ---- STEP 1: active_strategies (scoring_config.paper_trade_config) ----
resp = c.table("scoring_config").select("paper_trade_config").eq("id", 1).execute()
if not resp.data:
    raise SystemExit("scoring_config id=1 not found")

ptc = resp.data[0]["paper_trade_config"]
if isinstance(ptc, str):
    ptc = json.loads(ptc)
existing_active = list(ptc.get("active_strategies") or [])
print(f"[BEFORE] active_strategies: {len(existing_active)} entries")
print(f"         {existing_active}")

to_add_active = [s for s in NEW_STRATS if s not in existing_active]
if not to_add_active:
    print("[SKIP] all 3 strats already active, no change needed.")
else:
    new_active = existing_active + to_add_active
    ptc["active_strategies"] = new_active
    c.table("scoring_config").update({"paper_trade_config": ptc}).eq("id", 1).execute()
    print(f"\n[AFTER]  active_strategies: {len(new_active)} entries (added {to_add_active})")

# ---- STEP 2: rt_bankroll.strategy_bankrolls ($1000 seed per new strat) ----
br_resp = c.table("rt_bankroll").select("*").limit(1).execute()
if not br_resp.data:
    raise SystemExit("rt_bankroll row not found")
row = br_resp.data[0]
row_id = row["id"]
existing_sb = dict(row.get("strategy_bankrolls") or {})
print(f"\n[BEFORE] strategy_bankrolls: {len(existing_sb)} entries")

to_seed = []
for s in NEW_STRATS:
    if s in existing_sb:
        print(f"         {s:<38} EXISTS balance=${float(existing_sb[s].get('balance', 0)):.2f} (skip seed)")
    else:
        to_seed.append(s)

if not to_seed:
    print("[SKIP] all 3 strats already have bankroll entries, no change needed.")
else:
    for s in to_seed:
        existing_sb[s] = {"balance": BANKROLL_INIT, "trades": 0, "pnl": 0}
    c.table("rt_bankroll").update({"strategy_bankrolls": existing_sb}).eq("id", row_id).execute()
    print(f"\n[AFTER]  strategy_bankrolls: {len(existing_sb)} entries (seeded {to_seed} @ ${BANKROLL_INIT})")

# ---- VERIFICATION ----
print("\n=== VERIFICATION ===")
verify_ptc = c.table("scoring_config").select("paper_trade_config").eq("id", 1).execute().data[0]["paper_trade_config"]
if isinstance(verify_ptc, str):
    verify_ptc = json.loads(verify_ptc)
verify_active = verify_ptc.get("active_strategies", [])
for s in NEW_STRATS:
    present_active = s in verify_active
    print(f"  {s:<42} active={present_active}")

verify_br = c.table("rt_bankroll").select("strategy_bankrolls").limit(1).execute().data[0]["strategy_bankrolls"]
for s in NEW_STRATS:
    entry = verify_br.get(s)
    if entry:
        print(f"  {s:<42} bankroll=${float(entry.get('balance', 0)):.2f}")
    else:
        print(f"  {s:<42} bankroll=MISSING")

# Final sanity: active_strategies must still include ALL existing strats
missing = [s for s in existing_active if s not in verify_active]
if missing:
    raise SystemExit(f"CORRUPTION: lost existing strats {missing}")
missing_br = [s for s in row.get("strategy_bankrolls", {}) if s not in verify_br]
if missing_br:
    raise SystemExit(f"CORRUPTION: lost existing bankrolls {missing_br}")
print("\n[OK] Existing strategies + bankrolls preserved. Activation complete.")
