"""What live actually did per chain × period — to compare with the ideal."""
import os, sys, statistics as st
from collections import defaultdict
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

NOW = datetime.now(timezone.utc)
PERIODS = [
    ("ALL", None,                          None),
    ("7d",  NOW - timedelta(days=7),       7.0),
    ("3d",  NOW - timedelta(days=3),       3.0),
    ("10h", NOW - timedelta(hours=10),     10/24),
]
CLOSED = ("sl_hit", "trail_stop", "tp_hit", "timeout", "be_stop")


def fetch_all(tbl, sel, since_iso=None, **filters):
    out, step, off = [], 1000, 0
    while True:
        q = sb.table(tbl).select(sel)
        if since_iso:
            q = q.gte("created_at", since_iso)
        for k, v in filters.items():
            q = q.eq(k, v)
        r = q.range(off, off + step - 1).execute()
        if not r.data:
            break
        out.extend(r.data)
        if len(r.data) < step:
            break
        off += step
    return out


earliest = min((p[1] for p in PERIODS if p[1] is not None), default=None)
recent_iso = earliest.isoformat() if earliest else None
all_rows = fetch_all("paper_trades",
                     "strategy,status,source,pnl_pct,pnl_usd,chain,created_at",
                     since_iso=None, source="rt_live")
print(f"Total rt_live rows fetched (all-time): {len(all_rows)}")

for chain_db, chain_label in (("solana", "SOL"), ("ethereum", "ETH")):
    print()
    print("=" * 92)
    print(f"{chain_label} LIVE — actual real-money result")
    print("=" * 92)
    for label, since_dt, days in PERIODS:
        if since_dt is None:
            sub = [r for r in all_rows if r.get("chain") == chain_db]
        else:
            since_iso = since_dt.isoformat()
            sub = [r for r in all_rows
                   if r.get("chain") == chain_db
                   and r.get("created_at") and r["created_at"] >= since_iso]
        sub = [r for r in sub if r.get("status") in CLOSED]
        by_strat = defaultdict(list)
        for r in sub:
            by_strat[r["strategy"]].append(r)
        if not by_strat:
            print(f"\n[{label}] no live closed trades")
            continue
        total = sum(float(x.get("pnl_usd") or 0) for r in by_strat.values() for x in r)
        n_total = sum(len(r) for r in by_strat.values())
        per_day = f"{total/days:+.2f}" if days else "—"
        print(f"\n[{label}]  N={n_total}  total $={total:+.2f}  $/jour={per_day}")
        print(f"{'  Strategy':<32}{'N':>5}{'WR%':>7}{'avg%':>9}{'$ tot':>10}")
        for s, xs in sorted(by_strat.items(), key=lambda kv: -sum(float(x.get('pnl_usd') or 0) for x in kv[1])):
            n = len(xs)
            pnls = [float(x.get("pnl_pct") or 0) * 100 for x in xs]
            usd = sum(float(x.get("pnl_usd") or 0) for x in xs)
            wr = 100 * sum(1 for p in pnls if p > 0) / n
            print(f"  {s:<30}{n:>5}{wr:>6.1f}%{st.mean(pnls):>+8.2f}%{usd:>+9.2f}")
