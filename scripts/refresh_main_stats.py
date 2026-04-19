"""Refresh paper main stats — 7d rolling window, active strategies only.
Sorted by $ gained.
"""
import os, sys, statistics as st
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
from strategies import LAZY_STRATEGIES
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

WINDOW_HOURS = int(os.environ.get("WINDOW_HOURS", "168"))  # 7d default
SINCE = (datetime.now(timezone.utc) - timedelta(hours=WINDOW_HOURS)).isoformat()

def fetch_all(tbl, sel, **f):
    out, step, off = [], 1000, 0
    while True:
        q = sb.table(tbl).select(sel)
        for k, v in f.items():
            if k.startswith("gte_"): q = q.gte(k[4:], v)
            elif k.startswith("eq_"): q = q.eq(k[3:], v)
        r = q.range(off, off+step-1).execute()
        if not r.data: break
        out.extend(r.data)
        if len(r.data) < step: break
        off += step
    return out

print(f"Window: last {WINDOW_HOURS}h (since {SINCE})\n")
rows = fetch_all("paper_trades", "strategy,source,status,pnl_pct,pnl_usd",
                 gte_created_at=SINCE, eq_source="rt")
closed = [r for r in rows if r.get("status") in ("sl_hit","trail_stop","tp_hit","timeout","be_stop")
          and not str(r.get("strategy","")).startswith("DTRAIL")]

by_strat = defaultdict(list)
for r in closed:
    by_strat[r["strategy"]].append(r)

print(f"{'Strategy':<32}{'N':>5}{'WR':>7}{'avg%':>9}{'$':>10}{'LAZY':>6}")
print("-"*72)
results = []
for s, xs in by_strat.items():
    n = len(xs)
    if n < 5: continue
    pnls = [float(x.get("pnl_pct") or 0)*100 for x in xs]
    usd = sum(float(x.get("pnl_usd") or 0) for x in xs)
    wr = 100*sum(1 for p in pnls if p>0)/n
    results.append((s, n, wr, st.mean(pnls), usd, s in LAZY_STRATEGIES))

results.sort(key=lambda x: -x[4])  # by $ desc
for s, n, wr, avg, usd, lazy in results[:25]:
    print(f"{s:<32}{n:>5}{wr:>6.1f}%{avg:>+8.2f}%{usd:>+9.2f}{('Y' if lazy else 'N'):>6}")

print(f"\nTotal trades: {len(closed)}  total $ all: {sum(r[4] for r in results):+.2f}")
print(f"LAZY only: $ {sum(r[4] for r in results if r[5]):+.2f}  |  non-LAZY: $ {sum(r[4] for r in results if not r[5]):+.2f}")
