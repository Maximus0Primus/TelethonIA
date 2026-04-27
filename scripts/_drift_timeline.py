"""Timeline of paper vs live drift for BE25 and BOND, day by day, last 14d."""
import os, sys, statistics as st
from collections import defaultdict
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

NOW = datetime.now(timezone.utc)
SINCE = (NOW - timedelta(days=14)).isoformat()
CLOSED = ("sl_hit", "trail_stop", "tp_hit", "timeout", "be_stop")
STRATS = ("BE25_TP80_SL30", "BOND_FAST_TP50_SL20_T20", "FAST_TP50_SL30")


def fetch(src, strat):
    out, off, step = [], 0, 1000
    while True:
        r = sb.table("paper_trades").select(
            "strategy,status,source,pnl_pct,pnl_usd,chain,created_at,is_shadow"
        ).eq("strategy", strat).eq("source", src).eq("chain", "solana").gte("created_at", SINCE).range(off, off+step-1).execute()
        if not r.data:
            break
        out.extend(r.data)
        if len(r.data) < step:
            break
        off += step
    return out


for strat in STRATS:
    print()
    print("=" * 92)
    print(f"DAILY drift  {strat}")
    print("=" * 92)
    live = fetch("rt_live", strat)
    paper = fetch("rt", strat)
    by_day_l = defaultdict(list)
    by_day_p = defaultdict(list)
    for r in live:
        if r.get("status") in CLOSED and not r.get("is_shadow"):
            d = r["created_at"][:10]
            by_day_l[d].append(float(r.get("pnl_pct") or 0) * 100)
    for r in paper:
        if r.get("status") in CLOSED and not r.get("is_shadow"):
            d = r["created_at"][:10]
            by_day_p[d].append(float(r.get("pnl_pct") or 0) * 100)
    days = sorted(set(by_day_l) | set(by_day_p))
    print(f"  {'Date':<12}{'Npaper':>7}{'avg_p':>9}{'Nlive':>7}{'avg_l':>9}{'drift_pp':>11}")
    for d in days:
        p, l = by_day_p.get(d, []), by_day_l.get(d, [])
        ap = st.mean(p) if p else None
        al = st.mean(l) if l else None
        if ap is not None and al is not None:
            drift = f"{al-ap:+.2f}"
        else:
            drift = "—"
        ap_s = f"{ap:+.2f}%" if ap is not None else "—"
        al_s = f"{al:+.2f}%" if al is not None else "—"
        print(f"  {d:<12}{len(p):>7}{ap_s:>9}{len(l):>7}{al_s:>9}{drift:>11}")
