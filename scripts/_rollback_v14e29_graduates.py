"""Rollback the v14e.29 KOL graduation (commit 5aa8ee5).

Re-adds mad_apes_gambles, TheReaperGems (a.k.a. reapergamble), bat_gamble to
rt_trade_config.live_trading.kol_blacklist so they STOP trading live but keep
running in paper. Reason: post-graduation drift on BE25_TP80_SL30 widened
from -3.80pp (Apr 22-25) to -16.88pp (Apr 26-27). Removing the 3 graduates
recovers ~9.34pp of that — confirmed by per-KOL window-compare.

DRY RUN by default — pass --apply to write.

Idempotent + writes a backup of the previous rt_trade_config first.
"""
import os, sys, json, copy, argparse
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

# Canonical KOL group names as they appear in paper_trades.kol_group.
# Apr 27 narrowed scope: only mad_apes_gambles re-blacklisted (sole graduate
# with observed live damage — 1 outlier $10% trade -33.57pp post-entry exit
# divergence). bat_gamble has 0 live trades (chains=["ethereum"], not scraped
# SOL); TheReaperGems has 0 live trades since graduation. Both kept whitelisted
# pending more evidence.
GRADUATES = ["mad_apes_gambles"]


def main(apply: bool):
    r = sb.table("scoring_config").select("rt_trade_config").eq("id", 1).execute()
    cfg = r.data[0]["rt_trade_config"] or {}
    if isinstance(cfg, str):
        cfg = json.loads(cfg)

    live_cfg = cfg.get("live_trading") or {}
    current = list(live_cfg.get("kol_blacklist") or [])
    print(f"current live blacklist (N={len(current)}):\n  {sorted(current)}")
    missing = [k for k in GRADUATES if k not in current]
    print(f"\ngraduates to re-add: {missing}")
    if not missing:
        print("nothing to do — all 3 already in blacklist")
        return

    new_cfg = copy.deepcopy(cfg)
    new_live = new_cfg.setdefault("live_trading", {})
    merged = sorted(set(current) | set(GRADUATES))
    new_live["kol_blacklist"] = merged
    print(f"\nnew live blacklist (N={len(merged)}):\n  {merged}")

    if not apply:
        print("\n[DRY RUN] pass --apply to write")
        return

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bk_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(bk_dir, exist_ok=True)
    bk = os.path.join(bk_dir, f"rt_trade_config_pre_v14e29_rollback_{ts}.json")
    with open(bk, "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    print(f"backup -> {bk}")

    sb.table("scoring_config").update({"rt_trade_config": new_cfg}).eq("id", 1).execute()
    print("scoring_config.rt_trade_config updated. Restart VPS (kol-scraper.service) to reload.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    main(ap.parse_args().apply)
