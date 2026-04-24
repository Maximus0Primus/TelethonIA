"""v14e.14b — Trim live-KOL blacklist: keep only the truly new KOLs.

Per-user request: KOLs that were already in GROUPS_DATA before v14e.14
(TheReaperGems) should NOT be in the blacklist — they were previously
allowed to live-trade implicitly, and the intent is only to gate the
truly new ones.

Removes: TheReaperGems, thereapergems. Keeps the 9 genuinely new ones.
"""
import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))

from supabase import create_client

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

REMOVE = {"TheReaperGems", "thereapergems"}


def main():
    r = sb.table("scoring_config").select("rt_trade_config").eq("id", 1).execute()
    cfg = r.data[0].get("rt_trade_config") or {}
    live_cfg = cfg.get("live_trading") or {}
    current = list(live_cfg.get("kol_blacklist") or [])
    kept = [k for k in current if k not in REMOVE]
    live_cfg["kol_blacklist"] = kept
    cfg["live_trading"] = live_cfg

    print(f"Before ({len(current)}): {current}")
    sb.table("scoring_config").update({"rt_trade_config": cfg}).eq("id", 1).execute()
    print(f"After  ({len(kept)}): {kept}")
    removed = sorted(set(current) - set(kept))
    print(f"Removed: {removed}")


if __name__ == "__main__":
    main()
