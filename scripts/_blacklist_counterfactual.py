"""Blacklist counterfactual — per-strategy deduped $/d WITH vs WITHOUT the chain
blacklist active. Delta = how much the blacklist amplifies (or costs) each strategy's
edge (sensitivity score). A strat that loses a lot of $/d when the blacklist is removed
depends on the ban to be profitable.

Faithful: rolling-24h dedup per (strategy, token). WITH = blacklisted KOLs excluded;
WITHOUT = all KOLs.

Usage: python scripts/_blacklist_counterfactual.py [--chain solana] [--days 14] [--min-n 20]
"""
import argparse
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from supabase import create_client

load_dotenv("scraper/.env")
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
EXIT = ("tp_hit", "sl_hit", "timeout", "trail_stop", "be_stop")
POS = 50

ap = argparse.ArgumentParser()
ap.add_argument("--chain", default="solana")
ap.add_argument("--days", type=int, default=14)
ap.add_argument("--min-n", type=int, default=20)
args = ap.parse_args()

cfg = sb.table("scoring_config").select("paper_trade_config").eq("id", 1).execute().data[0]
BL = set((cfg["paper_trade_config"].get("kol_chain_blacklist", {}) or {}).get(args.chain, []))
since = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()


def fetch_all():
    rows, off = [], 0
    while True:
        b = (sb.table("paper_trades")
             .select("kol_group,strategy,token_address,pnl_pct,created_at")
             .eq("chain", args.chain).eq("is_shadow", True).eq("source", "rt")
             .in_("status", list(EXIT)).gte("created_at", since)
             .order("created_at").range(off, off + 999).execute().data) or []
        rows += b
        if len(b) < 1000:
            return rows
        off += 1000


def dedup(rows):
    last_seen, kept = {}, []
    for r in sorted(rows, key=lambda r: r["created_at"]):
        ts = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
        key = (r["strategy"], r["token_address"])
        prev = last_seen.get(key)
        if prev and (ts - prev) < timedelta(hours=24):
            continue
        last_seen[key] = ts
        kept.append(r)
    return kept


def agg(rows):
    d = defaultdict(list)
    for r in rows:
        d[r["strategy"]].append(float(r["pnl_pct"]))
    return d


allrows = [r for r in fetch_all() if r.get("kol_group")]
without = agg(dedup(allrows))                                    # all KOLs
with_bl = agg(dedup([r for r in allrows if r["kol_group"] not in BL]))  # banned excluded

print(f"=== Blacklist counterfactual [{args.chain}] {args.days}d | BL={len(BL)} KOLs ===\n")
print(f"{'strategy':<32}{'$/d WITH':>10}{'$/d WO':>10}{'delta':>9}{'N_with':>8}  sensitivity")
out = []
for strat, pnls_with in with_bl.items():
    if len(pnls_with) < args.min_n:
        continue
    dpd_with = sum(pnls_with) * POS / args.days
    pnls_wo = without.get(strat, [])
    dpd_wo = sum(pnls_wo) * POS / args.days if pnls_wo else 0.0
    delta = dpd_with - dpd_wo  # >0 means blacklist HELPS this strat
    out.append((delta, strat, dpd_with, dpd_wo, len(pnls_with)))

for delta, strat, dpd_with, dpd_wo, n in sorted(out, key=lambda x: -x[0]):
    sens = "HIGH (ban-amplified)" if delta > 5 else ("hurts" if delta < -5 else "")
    print(f"{strat:<32}{dpd_with:>10.1f}{dpd_wo:>10.1f}{delta:>+9.1f}{n:>8}  {sens}")
print(f"\n{len(out)} strategies (N_with>={args.min_n}). delta>0 = blacklist amplifies the edge.")
