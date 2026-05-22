"""KOL x strategy breakdown — deduped $/d + WR per (kol, strategy) cell.

Answers "is the optimal blacklist per-strategy-family?" and surfaces KOL-conditioning
(a KOL that bleeds on most strats but prints on one). Faithful: rolling-24h dedup per
(strategy, token), blacklisted KOLs excluded.

Usage: python scripts/_kol_per_strat_breakdown.py [--chain solana] [--days 14] [--min-n 5]
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
ap.add_argument("--min-n", type=int, default=5)
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


rows = sorted(
    [r for r in fetch_all() if r.get("kol_group") and r["kol_group"] not in BL],
    key=lambda r: r["created_at"],
)
last_seen, kept = {}, []
for r in rows:
    ts = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
    key = (r["strategy"], r["token_address"])
    prev = last_seen.get(key)
    if prev and (ts - prev) < timedelta(hours=24):
        continue
    last_seen[key] = ts
    kept.append(r)

cell = defaultdict(list)
for r in kept:
    cell[(r["kol_group"], r["strategy"])].append(float(r["pnl_pct"]))

# rank cells by abs $/d, show the most material
out = []
for (kol, strat), pnls in cell.items():
    if len(pnls) < args.min_n:
        continue
    n = len(pnls)
    dpd = sum(pnls) * POS / args.days
    wr = 100 * sum(1 for p in pnls if p > 0) / n
    out.append((dpd, kol, strat, n, wr, sum(pnls) / n * 100))

print(f"=== KOL x strategy [{args.chain}] {args.days}d (deduped, BL excluded, N>={args.min_n}) ===\n")
print(f"{'KOL':<22}{'strategy':<30}{'N':>5}{'WR%':>6}{'avg%':>8}{'$/d@50':>9}")
print("-- top winners --")
for dpd, kol, strat, n, wr, avg in sorted(out, key=lambda x: -x[0])[:20]:
    print(f"{kol:<22}{strat:<30}{n:>5}{wr:>6.0f}{avg:>8.1f}{dpd:>9.1f}")
print("-- top bleeders --")
for dpd, kol, strat, n, wr, avg in sorted(out, key=lambda x: x[0])[:20]:
    print(f"{kol:<22}{strat:<30}{n:>5}{wr:>6.0f}{avg:>8.1f}{dpd:>9.1f}")
print(f"\n{len(out)} cells with N>={args.min_n}.")
