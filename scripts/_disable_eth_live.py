"""Kill switch — flips eth_live_enabled=false.

Use immediately if microtest is bleeding too fast or any infra issue surfaces.
Does NOT close existing open ETH positions (the live monitor will continue to
exit them per their TP/SL/timeout). Just stops new opens.

Usage: python scripts/_disable_eth_live.py --apply
"""
import os, sys, json, copy, argparse
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    cfg = sb.table("scoring_config").select("rt_trade_config").eq("id", 1).execute().data[0]["rt_trade_config"]
    if isinstance(cfg, str):
        cfg = json.loads(cfg)
    live = (cfg.get("live_trading") or {})
    print(f"current eth_live_enabled = {live.get('eth_live_enabled', '<NOT SET>')}")
    if not live.get("eth_live_enabled"):
        print("already disabled — nothing to do")
        return

    if not args.apply:
        print("[DRY RUN] pass --apply to flip eth_live_enabled=false")
        return

    new_cfg = copy.deepcopy(cfg)
    new_cfg["live_trading"]["eth_live_enabled"] = False
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bk = os.path.join(os.path.dirname(__file__), "..", "data",
                      f"rt_trade_config_pre_eth_disable_{ts}.json")
    with open(bk, "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    print(f"backup -> {bk}")
    sb.table("scoring_config").update({"rt_trade_config": new_cfg}).eq("id", 1).execute()
    print("eth_live_enabled = False. Restart VPS to take effect immediately.")
    print("Existing open ETH positions continue to close normally per their TP/SL/timeout.")


if __name__ == "__main__":
    main()
