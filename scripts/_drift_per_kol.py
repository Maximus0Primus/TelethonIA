"""Per-KOL drift on BE25_TP80_SL30 last 7d — find the culprit causing Apr 26-27 spike."""
import os, sys, statistics as st
from collections import defaultdict
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

NOW = datetime.now(timezone.utc)
SINCE = (NOW - timedelta(days=7)).isoformat()
CLOSED = ("sl_hit", "trail_stop", "tp_hit", "timeout", "be_stop")


def fetch(src, strat):
    out, off, step = [], 0, 1000
    while True:
        r = sb.table("paper_trades").select(
            "strategy,status,source,pnl_pct,pnl_usd,kol_group,token_address,created_at,is_shadow,chain"
        ).eq("strategy", strat).eq("source", src).eq("chain", "solana").gte("created_at", SINCE).range(off, off+step-1).execute()
        if not r.data:
            break
        out.extend(r.data)
        if len(r.data) < step:
            break
        off += step
    return [r for r in out if r.get("status") in CLOSED and not r.get("is_shadow")]


for strat in ("BE25_TP80_SL30", "BOND_FAST_TP50_SL20_T20"):
    print()
    print("=" * 92)
    print(f"PER-KOL drift  {strat}  last 7d (SOL)")
    print("=" * 92)
    live = fetch("rt_live", strat)
    paper = fetch("rt", strat)
    by_kol_l = defaultdict(list)
    by_kol_p = defaultdict(list)
    for r in live:
        by_kol_l[r.get("kol_group") or "—"].append(float(r.get("pnl_pct") or 0) * 100)
    for r in paper:
        by_kol_p[r.get("kol_group") or "—"].append(float(r.get("pnl_pct") or 0) * 100)
    kols = sorted(set(by_kol_l) | set(by_kol_p))
    print(f"  {'KOL':<28}{'Np':>5}{'avgP':>9}{'Nl':>5}{'avgL':>9}{'drift':>9}{'$live':>10}")
    rows = []
    for k in kols:
        p = by_kol_p.get(k, [])
        l = by_kol_l.get(k, [])
        ap = st.mean(p) if p else None
        al = st.mean(l) if l else None
        drift = (al - ap) if (ap is not None and al is not None) else None
        # crude $ live: assume $1.70/trade × pnl_pct/100
        usd = sum(1.70 * x / 100 for x in l) if l else 0
        rows.append((k, len(p), ap, len(l), al, drift, usd))
    # sort by total $ live ascending (worst first)
    rows.sort(key=lambda x: x[6])
    for k, np_, ap, nl, al, drift, usd in rows:
        ap_s = f"{ap:+.2f}%" if ap is not None else "—"
        al_s = f"{al:+.2f}%" if al is not None else "—"
        d_s = f"{drift:+.1f}" if drift is not None else "—"
        print(f"  {k[:27]:<28}{np_:>5}{ap_s:>9}{nl:>5}{al_s:>9}{d_s:>9}{usd:>+9.2f}")
