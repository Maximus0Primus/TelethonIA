"""Paired A/B : main <STRAT> (LAZY) vs shadow <STRAT>_NOLAZY (CURRENT polling).

Both rows share token_address + cycle minute (opened in same RT event). Shadow
has position_usd=0 → _should_poll_trade bypasses LAZY → CURRENT interval.

Run this 24h after deploying v144 shadows to get paired verdict.
"""
import os, sys, statistics as st
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

WINDOW_HOURS = int(os.environ.get("WINDOW_HOURS", "168"))
SINCE = (datetime.now(timezone.utc) - timedelta(hours=WINDOW_HOURS)).isoformat()

PAIRS = [
    ("FAST_TP40_SL30",   "FAST_TP40_SL30_NOLAZY"),
    ("FAST_TP80_SL25",   "FAST_TP80_SL25_NOLAZY"),
    ("FAST_TP50_SL30",   "FAST_TP50_SL30_NOLAZY"),
    ("TP50_SL15",        "TP50_SL15_NOLAZY"),
]

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

all_strats = [s for p in PAIRS for s in p]
print(f"Window: last {WINDOW_HOURS}h\n")
rows = fetch_all("paper_trades",
    "strategy,status,source,pnl_pct,pnl_usd,token_address,created_at,cycle_ts,position_usd",
    gte_created_at=SINCE, eq_source="rt")
rows = [r for r in rows if r["strategy"] in all_strats
        and r.get("status") in ("sl_hit","trail_stop","tp_hit","timeout","be_stop")]
print(f"Matching closed trades: {len(rows)}\n")

def minute_key(r):
    return (r["token_address"], (r.get("cycle_ts") or r["created_at"])[:16])

by_strat = defaultdict(dict)
for r in rows:
    by_strat[r["strategy"]][minute_key(r)] = r

print(f"{'LAZY strat':<22}{'NOLAZY strat':<32}{'N pairs':>9}{'LAZY avg%':>11}{'NOLAZY avg%':>13}{'Δ mean pp':>12}{'wins':>8}")
print("-"*112)
for lazy_s, nolazy_s in PAIRS:
    ml = by_strat.get(lazy_s, {})
    ms = by_strat.get(nolazy_s, {})
    common = set(ml) & set(ms)
    if not common:
        print(f"{lazy_s:<22}{nolazy_s:<32}{'0':>9}  (no paired data yet — wait for shadows to accumulate)")
        continue
    deltas, lp, sp = [], [], []
    for k in common:
        l = float(ml[k].get("pnl_pct") or 0) * 100
        s = float(ms[k].get("pnl_pct") or 0) * 100
        deltas.append(l - s); lp.append(l); sp.append(s)
    n = len(deltas); wins = sum(1 for d in deltas if d > 0)
    print(f"{lazy_s:<22}{nolazy_s:<32}{n:>9}{st.mean(lp):>+10.2f}%{st.mean(sp):>+12.2f}%"
          f"{st.mean(deltas):>+10.2f}pp{wins:>5}/{n}")
