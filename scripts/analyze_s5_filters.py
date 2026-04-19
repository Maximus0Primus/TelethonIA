"""S5 : NOZEROLIQ/HIGHSCORE/SCORE30 filter analysis.

Question: these filters still lose, but todo said wait for N≥50. What do we
actually have, and is the filter killing alpha or just unlucky?

Approach:
  1. Pull closed trades per filtered strategy (N, WR, avg%, $).
  2. Pull same period unfiltered baselines (BE25_TP80_SL30, BE15_TP70_SL50).
  3. Compare to estimate: does the filter improve over what it would have been
     without the filter?
  4. Decompose: unfiltered universe PnL on tokens that PASS vs FAIL the filter.
"""
import os, sys, statistics as st, json
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
        out.extend(r.data);
        if len(r.data) < step: break
        off += step
    return out

rows = fetch_all("paper_trades",
    "strategy,status,source,pnl_pct,pnl_usd,entry_score,rt_liquidity_usd,rt_is_pump_fun",
    gte_created_at=SINCE)
closed = [r for r in rows if r.get("status") in ("sl_hit","trail_stop","tp_hit","timeout","be_stop")]

# Filter strats + their closest base
pairs = [
    ("NOZEROLIQ_TP200_SL40", None, "min_liq=1"),
    ("HIGHSCORE_TP200_SL40", None, "min_score=30"),
    ("BE25_TP80_SL30_S30_HYST", "BE25_TP80_SL30", "SCORE30 + HYST"),
    ("BE15_TP70_SL50_NZ", None, "min_liq=1"),
    ("BE25_TP80_SL30_NZS30_HYST", "BE25_TP80_SL30", "NZ + SCORE30 + HYST"),
]

def summary(name, xs):
    n = len(xs)
    if n == 0: return f"  {name:<30}  N=0"
    pnls = [float(r.get("pnl_pct") or 0)*100 for r in xs]
    usd = sum(float(r.get("pnl_usd") or 0) for r in xs)
    wr = 100 * sum(1 for p in pnls if p > 0) / n
    return (f"  {name:<30}  N={n:4d}  WR={wr:5.1f}%  med={st.median(pnls):+6.2f}%  "
            f"mean={st.mean(pnls):+6.2f}%  $={usd:+.2f}")

print(f"Trades since {SINCE}: {len(closed)} closed\n")
print("=== Filter strategies vs their unfiltered bases ===\n")

for filt, base, rule in pairs:
    fx = [r for r in closed if r["strategy"] == filt]
    print(f"{filt} ({rule})")
    print(summary("  filtered", fx))
    if base:
        bx = [r for r in closed if r["strategy"] == base]
        print(summary(f"  unfiltered base ({base})", bx))
    # Now: from unfiltered base, apply the filter retroactively
    if base and "NZ" in filt:
        bx_pass = [r for r in closed if r["strategy"] == base and float(r.get("rt_liquidity_usd") or 0) >= 1.0]
        bx_fail = [r for r in closed if r["strategy"] == base and not float(r.get("rt_liquidity_usd") or 0) >= 1.0]
        print(summary("  base+NZ retroactive", bx_pass))
        print(summary("  base rejected by NZ", bx_fail))
    if base and "S30" in filt:
        bx_pass = [r for r in closed if r["strategy"] == base and float(r.get("entry_score") or 0) >= 30]
        bx_fail = [r for r in closed if r["strategy"] == base and float(r.get("entry_score") or 0) < 30]
        print(summary("  base+SCORE30 retroactive", bx_pass))
        print(summary("  base rejected by SCORE30", bx_fail))
    print()

# Decompose NOZEROLIQ + HIGHSCORE using BE15 or generic TP200-ish base
print("=== NOZEROLIQ_TP200_SL40 + HIGHSCORE_TP200_SL40 : try to find generic TP200 base for retroactive split ===")
bases_tp200 = sorted({r["strategy"] for r in closed if "TP200" in r["strategy"]})
print(f"Available TP200 strategies: {bases_tp200}")

# Split entire population by liquidity and score bands
print("\n=== Population-wide view (all closed trades) ===")
by_liq = defaultdict(list); by_score = defaultdict(list)
for r in closed:
    L = float(r.get("rt_liquidity_usd") or 0)
    sc = float(r.get("entry_score") or 0)
    p = float(r.get("pnl_pct") or 0) * 100
    if L <= 0: by_liq["bonding(liq=0)"].append(p)
    elif L < 10000: by_liq["<10K"].append(p)
    elif L < 50000: by_liq["10-50K"].append(p)
    else: by_liq[">=50K"].append(p)
    if sc < 20: by_score["<20"].append(p)
    elif sc < 30: by_score["20-30"].append(p)
    elif sc < 40: by_score["30-40"].append(p)
    else: by_score[">=40"].append(p)

def bucket_line(tag, buckets):
    print(f"\n{tag}:")
    for k, xs in sorted(buckets.items()):
        n = len(xs); wr = 100*sum(1 for p in xs if p>0)/n if n else 0
        med = st.median(xs) if xs else 0; mn = st.mean(xs) if xs else 0
        print(f"  {k:<15}  N={n:5d}  WR={wr:5.1f}%  med={med:+6.2f}%  mean={mn:+6.2f}%")

bucket_line("By liquidity bucket", by_liq)
bucket_line("By entry_score bucket", by_score)

out = os.path.join(os.path.dirname(__file__), "..", "data", "s5_filter_analysis.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump({"liquidity_buckets": {k: {"n":len(v),"mean":st.mean(v) if v else 0, "median":st.median(v) if v else 0}
                                      for k,v in by_liq.items()},
               "score_buckets": {k: {"n":len(v),"mean":st.mean(v) if v else 0, "median":st.median(v) if v else 0}
                                 for k,v in by_score.items()}}, f, indent=2)
print(f"\nSaved -> {out}")
