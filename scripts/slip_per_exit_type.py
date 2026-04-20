"""Per-exit-type slip calibration: bucket pump x exit_type, compute median delta L-P bps.

Outputs a recommended slip offset per cell + N gate. Cells with N<15 flagged 'WAIT'.
"""
import os, sys, statistics as st
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

NOW = datetime.now(timezone.utc)
SINCE = (NOW - timedelta(days=14)).isoformat()

def fetch(source):
    out, off = [], 0
    while True:
        r = sb.table("paper_trades").select(
            "token_address,strategy,status,pnl_pct,rt_is_pump_fun,created_at"
        ).eq("source", source).gte("created_at", SINCE).range(off, off+999).execute().data
        out.extend(r)
        if len(r) < 1000: break
        off += 1000
    return [t for t in out if t.get("status") in ("sl_hit","tp_hit","timeout","be_stop","trail_stop")]

print(f"Loading 14d trades since {SINCE}...")
live = fetch("rt_live")
paper = fetch("rt")
print(f"  live: {len(live)}  paper: {len(paper)}")

# Build (token, strategy) -> live and paper rows
def index(rows):
    out = defaultdict(list)
    for r in rows:
        out[(r["token_address"], r["strategy"])].append(r)
    return out

live_idx = index(live)
paper_idx = index(paper)

# Match pairs
pairs = []
for k, lrows in live_idx.items():
    if k not in paper_idx: continue
    for lr in lrows:
        for pr in paper_idx[k]:
            # require same exit_type to avoid mixing apples/oranges
            if lr["status"] != pr["status"]: continue
            l_pnl = float(lr.get("pnl_pct") or 0) * 100
            p_pnl = float(pr.get("pnl_pct") or 0) * 100
            delta_pp = l_pnl - p_pnl  # in percentage points
            delta_bps = delta_pp * 100
            pairs.append({
                "token": k[0], "strat": k[1], "status": lr["status"],
                "pump": bool(lr.get("rt_is_pump_fun")),
                "live_pnl": l_pnl, "paper_pnl": p_pnl,
                "delta_pp": delta_pp, "delta_bps": delta_bps,
            })

print(f"  matched pairs: {len(pairs)}\n")

# Bucket
buckets = defaultdict(list)
for p in pairs:
    buckets[(p["pump"], p["status"])].append(p["delta_bps"])

print("="*80)
print(f"{'Pump':<6}{'Status':<14}{'N':>5}{'median bps':>14}{'p25':>10}{'p75':>10}{'mean':>10}{'recommendation':>30}")
print("-"*80)
for k in sorted(buckets, key=lambda x: (not x[0], x[1])):
    vals = sorted(buckets[k])
    n = len(vals)
    med = st.median(vals)
    mean = st.mean(vals)
    p25 = vals[max(0, n//4 - 1)]
    p75 = vals[min(n-1, 3*n//4)]
    pump_lbl = "Y" if k[0] else "N"
    if n < 15:
        rec = f"WAIT (need N>=15, have {n})"
    elif abs(med) < 50:
        rec = "OK no change"
    elif med > 0:
        rec = f"DECREASE paper slip by {int(med)} bps"
    else:
        rec = f"INCREASE paper slip by {int(-med)} bps"
    print(f"{pump_lbl:<6}{k[1]:<14}{n:>5}{med:>+14.0f}{p25:>+10.0f}{p75:>+10.0f}{mean:>+10.0f}   {rec}")

# Per-strategy too
print()
print("="*80)
print("Per-strategy x exit_type (pump=Y only, N>=8):")
print("-"*80)
strat_buckets = defaultdict(list)
for p in pairs:
    if not p["pump"]: continue
    strat_buckets[(p["strat"], p["status"])].append(p["delta_bps"])
print(f"{'Strategy':<25}{'Status':<14}{'N':>5}{'median bps':>14}{'mean':>10}")
for k in sorted(strat_buckets):
    vals = strat_buckets[k]
    if len(vals) < 8: continue
    print(f"{k[0]:<25}{k[1]:<14}{len(vals):>5}{st.median(vals):>+14.0f}{st.mean(vals):>+10.0f}")
