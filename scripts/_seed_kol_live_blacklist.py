"""v14e.14 — Seed live-KOL blacklist in scoring_config.rt_trade_config.

Adds the 10 new unproven KOLs to rt_trade_config.live_trading.kol_blacklist.
They stay in scraper.GROUPS_DATA (paper + shadow run normally), but the RT
live branch short-circuits before calling open_live_trade for them.

Idempotent — re-running is safe (set dedup).
"""
import os
import sys
import json
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))

from supabase import create_client

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

NEW_KOLS = [
    "batman_gem",
    "venom_gambles",
    "ryoshigamble",
    "ryoshikushama",
    "mad_apes_gambles",
    # v14e.28 (Apr 26): MaestrosDegen REMOVED from scraping entirely
    # (safe_scraper.GROUPS_DATA). Apr 1-26 paper stats: SOL N=1000 avg -49.8%
    # sum -$23,985. Live blacklist is redundant — he can't reach the live
    # path if not scraped.
    "bat_gamble",
    # v14e.14b: TheReaperGems + thereapergems removed per user — already in
    # GROUPS_DATA before v14e.14, should remain live-eligible. Only the truly
    # new KOLs (8 entries) are blacklisted from live.
    "reapergamble",
    "bagcalls",
]


def main():
    r = sb.table("scoring_config").select("rt_trade_config").eq("id", 1).execute()
    if not r.data:
        print("[err] scoring_config id=1 not found")
        sys.exit(1)
    cfg = r.data[0].get("rt_trade_config") or {}
    live_cfg = cfg.get("live_trading") or {}
    current = list(live_cfg.get("kol_blacklist") or [])
    before = set(current)
    merged = sorted(set(current) | set(NEW_KOLS))
    live_cfg["kol_blacklist"] = merged
    cfg["live_trading"] = live_cfg

    print("=== Before ===")
    print(f"  live_trading.kol_blacklist ({len(current)}): {current}")

    sb.table("scoring_config").update({"rt_trade_config": cfg}).eq("id", 1).execute()

    print("\n=== After ===")
    print(f"  live_trading.kol_blacklist ({len(merged)}): {merged}")
    added = sorted(set(NEW_KOLS) - before)
    print(f"\nAdded {len(added)} KOLs: {added}")


if __name__ == "__main__":
    main()
