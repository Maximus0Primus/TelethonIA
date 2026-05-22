"""KOL blacklist audit — paired view of every KOL's deduped shadow performance vs its
blacklist status, flagging mismatches (banned-but-good, allowed-but-bad).

Faithful by construction: rolling-24h dedup per (kol, token), blacklist read from config.
Use weekly to keep the SOL/ETH chain blacklists honest.

Usage: python scripts/_kol_blacklist_audit.py [--chain solana|ethereum] [--days 14]
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
POS = 50  # reference $/trade scale

ap = argparse.ArgumentParser()
ap.add_argument("--chain", default="solana")
ap.add_argument("--days", type=int, default=14)
ap.add_argument("--min-n", type=int, default=30)
args = ap.parse_args()

cfg = sb.table("scoring_config").select("paper_trade_config").eq("id", 1).execute().data[0]
BL = set((cfg["paper_trade_config"].get("kol_chain_blacklist", {}) or {}).get(args.chain, []))
since = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()


def fetch_all():
    rows, off = [], 0
    while True:
        b = (sb.table("paper_trades")
             .select("kol_group,token_address,pnl_pct,created_at")
             .eq("chain", args.chain).eq("is_shadow", True).eq("source", "rt")
             .in_("status", list(EXIT)).gte("created_at", since)
             .order("created_at").range(off, off + 999).execute().data) or []
        rows += b
        if len(b) < 1000:
            return rows
        off += 1000


# rolling-24h dedup per (kol, token)
rows = sorted([r for r in fetch_all() if r.get("kol_group")], key=lambda r: r["created_at"])
last_seen, kept = {}, []
for r in rows:
    ts = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
    key = (r["kol_group"], r["token_address"])
    prev = last_seen.get(key)
    if prev and (ts - prev) < timedelta(hours=24):
        continue
    last_seen[key] = ts
    kept.append(r)

by_kol = defaultdict(list)
for r in kept:
    by_kol[r["kol_group"]].append(float(r["pnl_pct"]))

print(f"=== KOL blacklist audit [{args.chain}] {args.days}d | BL={len(BL)} KOLs | dedup N={len(kept)} ===\n")
print(f"{'KOL':<24}{'N':>5}{'WR%':>7}{'avg%':>8}{'$/d@50':>9}  status / flag")
rowsout = []
for kol, pnls in by_kol.items():
    n = len(pnls)
    wr = 100 * sum(1 for p in pnls if p > 0) / n
    avg = sum(pnls) / n * 100
    dpd = sum(pnls) * POS / args.days
    banned = kol in BL
    flag = ""
    if banned and n >= args.min_n and avg > 0 and wr >= 40:
        flag = "<< UNBAN candidate (banned but good)"
    elif not banned and n >= args.min_n and avg < 0:
        flag = "<< BAN candidate (allowed but bleeding)"
    rowsout.append((dpd, kol, n, wr, avg, banned, flag))

for dpd, kol, n, wr, avg, banned, flag in sorted(rowsout, key=lambda x: x[0]):
    st = "BANNED" if banned else "allowed"
    print(f"{kol:<24}{n:>5}{wr:>7.0f}{avg:>8.2f}{dpd:>9.1f}  {st:<8}{flag}")

flags = [r for r in rowsout if r[6]]
print(f"\n{len(flags)} mismatch(es) to review.")
