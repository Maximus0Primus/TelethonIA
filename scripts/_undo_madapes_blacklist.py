"""Undo the mad_apes_gambles blacklist applied earlier today.

Reason for undo: paired-test on intersection tokens (correct apples-to-apples
metric per MEMORY paired_test_rule_apr20.md) shows mad_apes drift = -1.17pp on
N=4 (mean) / -1.02pp (median). The +34.44pp on $AMURICA cancels the -33.57pp
on $10%. The earlier blacklist was based on aggregate drift -53.8pp which was
purely selection bias (different sample sets paper vs live).

Restores live_trading.kol_blacklist to the pre-rollback set.
"""
import os, sys, json, copy
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

REMOVE = ["mad_apes_gambles"]

r = sb.table("scoring_config").select("rt_trade_config").eq("id", 1).execute()
cfg = r.data[0]["rt_trade_config"]
if isinstance(cfg, str):
    cfg = json.loads(cfg)
live_cfg = cfg.get("live_trading") or {}
current = list(live_cfg.get("kol_blacklist") or [])
print(f"current (N={len(current)}): {sorted(current)}")
new_bl = sorted(k for k in current if k not in REMOVE)
print(f"removing: {[k for k in REMOVE if k in current]}")
print(f"new (N={len(new_bl)}): {new_bl}")

ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
bk = os.path.join(os.path.dirname(__file__), "..", "data", f"rt_trade_config_pre_undo_madapes_{ts}.json")
with open(bk, "w") as f:
    json.dump(cfg, f, indent=2, default=str)
print(f"backup -> {bk}")

new_cfg = copy.deepcopy(cfg)
new_cfg["live_trading"]["kol_blacklist"] = new_bl
sb.table("scoring_config").update({"rt_trade_config": new_cfg}).eq("id", 1).execute()
print("done. Restart VPS to reload.")
