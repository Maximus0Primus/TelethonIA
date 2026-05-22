"""Audit: allocated strategies missing from SHADOW_STRATEGIES or the STRATEGIES registry.

Prevents the v14e.60 bug class — a strat allocated in hybrid_strategy.allocations (paper
main) or live_trading.allocations that is NOT in SHADOW_STRATEGIES never gets a shadow
twin, so paired drift (paper↔shadow, live↔shadow A/B) is impossible to measure. A strat
not in STRATEGIES dict can't fire at all.

Exit 0 = clean, exit 1 = gaps found (suitable for a CI gate).
"""
import os
import sys

from dotenv import load_dotenv
from supabase import create_client

load_dotenv("scraper/.env")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from strategies import SHADOW_STRATEGIES, STRATEGIES  # noqa: E402

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
cfg = sb.table("scoring_config").select("rt_trade_config").eq("id", 1).execute().data[0]["rt_trade_config"]
paper = set((cfg.get("hybrid_strategy") or {}).get("allocations", {}).keys())
live = set((cfg.get("live_trading") or {}).get("allocations", {}).keys())
shadow = set(SHADOW_STRATEGIES)
registered = set(STRATEGIES.keys())

print(f"SHADOW_STRATEGIES={len(shadow)} | STRATEGIES={len(registered)} | "
      f"paper allocs={len(paper)} | live allocs={len(live)}")

miss_shadow = sorted((paper | live) - shadow)
miss_reg = sorted((paper | live) - registered)

print(f"\n[allocated but NOT in SHADOW_STRATEGIES] ({len(miss_shadow)}):")
for s in miss_shadow:
    print(f"  - {s}{'  (LIVE!)' if s in live else ''}")

print(f"\n[allocated but NOT in STRATEGIES registry — cannot fire] ({len(miss_reg)}):")
for s in miss_reg:
    print(f"  - {s}{'  (LIVE!)' if s in live else ''}")

if not miss_shadow and not miss_reg:
    print("\nAll allocated strategies have shadow + registry coverage.")
    sys.exit(0)
sys.exit(1)
