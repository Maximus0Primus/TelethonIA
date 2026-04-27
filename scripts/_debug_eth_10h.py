"""Debug ETH 10h discrepancy between ranking and detail."""
import os, sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

NOW = datetime.now(timezone.utc)
SINCE = (NOW - timedelta(hours=10)).isoformat()
print(f"NOW UTC = {NOW.isoformat()}")
print(f"10h ago = {SINCE}")
print()

CLOSED = ("sl_hit", "trail_stop", "tp_hit", "timeout", "be_stop")

# Pull ALL ETH RT shadows in last 10h, no strat filter
rows = []
off, step = 0, 1000
while True:
    r = (sb.table("paper_trades")
         .select("strategy,status,source,pnl_pct,pnl_usd,chain,created_at,symbol,token_address,exit_at")
         .gte("created_at", SINCE)
         .eq("source", "rt")
         .eq("chain", "ethereum")
         .range(off, off+step-1)
         .execute())
    if not r.data: break
    rows.extend(r.data)
    if len(r.data) < step: break
    off += step

print(f"Total ETH RT rows in last 10h (any status): {len(rows)}")

# Unique tokens
tokens = sorted({(r.get("symbol"), r.get("token_address")) for r in rows})
print(f"Unique ETH tokens last 10h: {len(tokens)}")
for sym, addr in tokens:
    print(f"  {sym}  {addr}")

# Per-strategy in 10h, all with N>=1
from collections import defaultdict
by_strat = defaultdict(list)
for r in rows:
    if r.get("status") in CLOSED:
        by_strat[r["strategy"]].append(r)

print()
print("=== ETH strategies, last 10h, ALL N (incl N=1,2) ===")
print(f"{'strat':<40}{'N':>4}{'$ tot':>10}{'avg%':>9}")
ranked = []
for s, xs in by_strat.items():
    n = len(xs)
    usd = sum(float(x.get("pnl_usd") or 0) for x in xs)
    avg = sum(float(x.get("pnl_pct") or 0)*100 for x in xs) / n
    ranked.append((s, n, usd, avg))
ranked.sort(key=lambda r: -r[2])
for s, n, usd, avg in ranked[:20]:
    print(f"{s:<40}{n:>4}{usd:>+9.2f}{avg:>+8.2f}%")

# Specifically check ETH_SLOW6H_TP200_SL40_NZ_S40
print()
print("=== ETH_SLOW6H_TP200_SL40_NZ_S40 — every row in 10h ===")
sub = [r for r in rows if r["strategy"] == "ETH_SLOW6H_TP200_SL40_NZ_S40"]
for r in sub:
    print(f"  created={r['created_at'][:19]}  exit={(r.get('exit_at') or '—')[:19]}  "
          f"sym={r.get('symbol')}  status={r.get('status')}  "
          f"pnl%={float(r.get('pnl_pct') or 0)*100:+.2f}  $={float(r.get('pnl_usd') or 0):+.2f}")
