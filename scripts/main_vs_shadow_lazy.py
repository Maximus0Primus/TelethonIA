"""Real A/B : main paper (LAZY throttled) vs shadow (CURRENT interval) of
the same strategy. Paired on same token+created_at.

_should_poll_trade only applies LAZY to position_usd>0 rows; shadows (pos=0) are
the control group. So we have a built-in paired A/B for every LAZY strategy.
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

WINDOW_HOURS = int(os.environ.get("WINDOW_HOURS", "168"))
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

print(f"Window: last {WINDOW_HOURS}h\n")
rows = fetch_all("paper_trades",
    "strategy,status,source,pnl_pct,pnl_usd,token_address,created_at,position_usd,is_shadow",
    gte_created_at=SINCE, eq_source="rt")
closed = [r for r in rows if r.get("status") in ("sl_hit","trail_stop","tp_hit","timeout","be_stop")
          and not str(r.get("strategy","")).startswith("DTRAIL")]
print(f"Closed trades: {len(closed)}\n")

# Build per-strategy maps: main (pos>0) vs shadow (pos=0)
by_strat = defaultdict(lambda: {"main": {}, "shadow": {}})
for r in closed:
    s = r["strategy"]
    pos = float(r.get("position_usd") or 0)
    role = "main" if pos > 0 else "shadow"
    key = (r["token_address"], r["created_at"][:16])  # minute bucket
    by_strat[s][role][key] = r

print(f"{'Strategy':<32}{'pair_N':>7}{'main avg%':>11}{'shadow avg%':>13}{'Δ (main-shad)':>14}{'wins':>7}")
print("-"*84)
results = []
for s, d in sorted(by_strat.items()):
    if s not in LAZY_STRATEGIES: continue
    main_keys = set(d["main"]); shad_keys = set(d["shadow"])
    common = main_keys & shad_keys
    if len(common) < 10: continue

    deltas_pp = []; main_pnls = []; shad_pnls = []
    for k in common:
        mp = float(d["main"][k].get("pnl_pct") or 0) * 100
        sp = float(d["shadow"][k].get("pnl_pct") or 0) * 100
        main_pnls.append(mp); shad_pnls.append(sp)
        deltas_pp.append(mp - sp)
    n = len(deltas_pp)
    wins = sum(1 for d_ in deltas_pp if d_ > 0)
    losses = sum(1 for d_ in deltas_pp if d_ < 0)
    print(f"{s:<32}{n:>7}{st.mean(main_pnls):>+10.2f}%{st.mean(shad_pnls):>+12.2f}%"
          f"{st.mean(deltas_pp):>+12.2f}pp{wins:>4}/{n}")
    results.append({"strategy":s,"n":n,"main_avg":st.mean(main_pnls),
                    "shadow_avg":st.mean(shad_pnls),"delta_mean":st.mean(deltas_pp),
                    "delta_median":st.median(deltas_pp),"wins":wins,"losses":losses})

# Aggregate
if results:
    print()
    total_main = sum(r["main_avg"]*r["n"] for r in results) / sum(r["n"] for r in results)
    total_shad = sum(r["shadow_avg"]*r["n"] for r in results) / sum(r["n"] for r in results)
    total_delta = sum(r["delta_mean"]*r["n"] for r in results) / sum(r["n"] for r in results)
    print(f"Weighted avg  main={total_main:+.2f}%  shadow={total_shad:+.2f}%  Δ={total_delta:+.2f}pp")
    lazy_wins = sum(1 for r in results if r["delta_mean"] > 0)
    print(f"LAZY wins paired: {lazy_wins}/{len(results)} strats")

import json
out = os.path.join(os.path.dirname(__file__), "..", "data", "main_vs_shadow_lazy.json")
with open(out, "w", encoding="utf-8") as f: json.dump(results, f, indent=2)
print(f"\nSaved -> {out}")
