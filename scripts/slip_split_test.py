"""Test which variable best explains per-pair (pnl_live - pnl_paper) delta:
is_pump / liquidity_bucket / mcap_bucket.

Criterion: lowest within-bucket std with meaningful between-bucket mean gap.
"""
import os, sys, statistics as st
from collections import defaultdict
from dotenv import load_dotenv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
SINCE = os.environ.get("SINCE", "2026-04-13T00:00:00+00:00")

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

rows = fetch_all("paper_trades",
    "token_address,strategy,status,source,pnl_pct,rt_is_pump_fun,rt_liquidity_usd,entry_mcap",
    gte_created_at=SINCE)
closed = [r for r in rows if r.get("status") in ("sl_hit","trail_stop","tp_hit","timeout","be_stop")
          and not str(r.get("strategy","")).startswith("DTRAIL")]

pairs = defaultdict(dict)
for r in closed:
    src = "live" if r.get("source") == "rt_live" else "paper"
    pairs[(r["token_address"], r["strategy"])][src] = r

matched = [(lv, pp) for v in pairs.values() for lv,pp in [(v.get("live"), v.get("paper"))] if lv and pp]
deltas = []
for lv, pp in matched:
    d = (float(lv.get("pnl_pct") or 0) - float(pp.get("pnl_pct") or 0)) * 10000
    deltas.append({
        "delta_bps": d,
        "pump": bool(lv.get("rt_is_pump_fun")),
        "liq": float(lv.get("rt_liquidity_usd") or 0),
        "mcap": float(lv.get("entry_mcap") or 0),
    })

print(f"N={len(deltas)}  overall  median={st.median(d['delta_bps'] for d in deltas):+.0f}bps  "
      f"mean={st.mean(d['delta_bps'] for d in deltas):+.0f}bps  "
      f"std={st.pstdev(d['delta_bps'] for d in deltas):.0f}bps\n")

def bucketize(deltas, keyfn, labelfn):
    g = defaultdict(list)
    for d in deltas: g[labelfn(keyfn(d))].append(d["delta_bps"])
    return g

def report(name, grp):
    print(f"== split by {name} ==")
    total_within = 0; total_n = 0
    for lbl, xs in sorted(grp.items()):
        if not xs: continue
        n=len(xs); med=st.median(xs); mn=st.mean(xs)
        sd=st.pstdev(xs) if n>1 else 0
        total_within += sd*sd*n; total_n += n
        print(f"  {lbl:<12} N={n:3d}  median={med:+7.0f}  mean={mn:+7.0f}  std={sd:7.0f}")
    pooled_std = (total_within/total_n)**0.5 if total_n else 0
    print(f"  -> pooled within-bucket std = {pooled_std:.0f} bps\n")

report("is_pump", bucketize(deltas, lambda d: d["pump"], lambda x: "pump" if x else "non-pump"))

def liq_b(L):
    if L <= 0: return "bonding"
    if L < 10_000: return "<10K"
    if L < 30_000: return "10-30K"
    if L < 80_000: return "30-80K"
    return ">80K"
report("liquidity", bucketize(deltas, lambda d: d["liq"], liq_b))

def mc_b(M):
    if M <= 0: return "unknown"
    if M < 20_000: return "<20K"
    if M < 60_000: return "20-60K"
    if M < 150_000: return "60-150K"
    return ">150K"
report("mcap", bucketize(deltas, lambda d: d["mcap"], mc_b))

# Combined pump × liq_bucket to see if liq adds info on top of pump
combo = defaultdict(list)
for d in deltas:
    combo[("pump" if d["pump"] else "non", liq_b(d["liq"]))].append(d["delta_bps"])
print("== split by is_pump × liq ==")
for k in sorted(combo):
    xs=combo[k]; n=len(xs)
    if n<3: continue
    print(f"  {k[0]:<4} {k[1]:<8} N={n:3d}  median={st.median(xs):+7.0f}  std={st.pstdev(xs):7.0f}")
