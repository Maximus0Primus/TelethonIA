"""Pause live trading by flipping rt_trade_config.live_trading.enabled = false in DB.
Backups the previous JSONB to a timestamped table before mutation.

Reason (v14e.53, May 2): live test served its purpose (drift acted non-divergent
on N=175 pairs, slip Jupiter Ultra characterized). Now focus = identify the BEST
strategy via shadow expansion. Live is paused, paper main + shadow continue.
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

# Fetch current rt_trade_config
res = sb.table("scoring_config").select("rt_trade_config").eq("id", 1).single().execute()
cfg = res.data["rt_trade_config"]
if isinstance(cfg, str):
    cfg = json.loads(cfg)

# Backup
ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
backup_path = os.path.join(os.path.dirname(__file__), "..", "data", f"rt_trade_config_pre_pause_{ts}.json")
os.makedirs(os.path.dirname(backup_path), exist_ok=True)
with open(backup_path, "w") as f:
    json.dump(cfg, f, indent=2)
print(f"Backup: {backup_path}")

# Show current state
live_cfg = cfg.get("live_trading", {})
print(f"\nBEFORE:")
print(f"  live_trading.enabled       = {live_cfg.get('enabled')}")
print(f"  live_trading.eth_live_enabled = {live_cfg.get('eth_live_enabled')}")
print(f"  allocations: {live_cfg.get('allocations')}")
print(f"  max_position_sol: {live_cfg.get('max_position_sol')}")

# Mutate: flip enabled to false (keep allocations + position size for fast resume later)
cfg.setdefault("live_trading", {})["enabled"] = False
# Also flip eth (was already false, but be explicit)
cfg["live_trading"]["eth_live_enabled"] = False

# Push
sb.table("scoring_config").update({"rt_trade_config": cfg}).eq("id", 1).execute()
print(f"\nAFTER:")
print(f"  live_trading.enabled       = {cfg['live_trading'].get('enabled')}")
print(f"  live_trading.eth_live_enabled = {cfg['live_trading'].get('eth_live_enabled')}")
print(f"\nLIVE TRADING PAUSED.")
print(f"To resume: set live_trading.enabled=true via bot or this script (modify line)")
