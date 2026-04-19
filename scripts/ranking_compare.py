"""Compare strategy rankings: sim (mega-sweep) vs paper (rt) vs live (rt_live).

If sim predictions correlate with paper/live realized PnL rankings, sim is
predictive even with absolute-level bias. If ranks diverge, sim has systemic
blind spots we need to patch.
"""
import os, sys, pandas as pd, statistics as st
from collections import defaultdict
from dotenv import load_dotenv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

# v142 sweep covers Apr 13-18 universe
SIM_CSV = os.path.join(os.path.dirname(__file__), "..", "scraper", "_mega_sweep_v142.csv")
SINCE = "2026-04-13T20:00:00+00:00"

# -- 1. Load sim mega sweep
df_sim = pd.read_csv(SIM_CSV)
# Collapse variants: pick best (filter, source, smoothing, polling_mode) per strategy
# then take mean avg_pnl_pct across THE production orch for each strat (we don't have
# that mapping here — so use median across configs to get a robust per-strat estimate)
df_sim_agg = (df_sim[df_sim["n"] >= 30]
              .groupby("strategy", as_index=False)
              .agg(sim_avg_pnl=("avg_pnl_pct", "median"),
                   sim_wr=("wr_pct", "median"),
                   sim_n=("n", "max")))

# -- 2. Load paper + live closed trades post Apr 13
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

print(f"Loading trades since {SINCE}...")
rows = fetch_all("paper_trades", "strategy,status,source,pnl_pct", gte_created_at=SINCE)
rows = [r for r in rows if r.get("status") in ("sl_hit","trail_stop","tp_hit","timeout","be_stop")
        and not str(r.get("strategy","")).startswith("DTRAIL")]
print(f"  {len(rows)} closed non-DTRAIL trades\n")

paper_ag = defaultdict(list); live_ag = defaultdict(list)
for r in rows:
    pnl = float(r.get("pnl_pct") or 0) * 100
    if r.get("source") == "rt_live": live_ag[r["strategy"]].append(pnl)
    else: paper_ag[r["strategy"]].append(pnl)

def agg(ag, min_n=10):
    return {k: (len(v), st.mean(v), st.median(v)) for k, v in ag.items() if len(v) >= min_n}

paper_stats = agg(paper_ag, 10)
live_stats = agg(live_ag, 5)

# -- 3. Build ranking table
strats_common = sorted(set(paper_stats) & set(df_sim_agg["strategy"].tolist()))
print(f"{'Strategy':<30}{'sim N':>7}{'sim%':>9}{'paper N':>9}{'paper%':>9}{'live N':>9}{'live%':>9}")
print("-"*82)
rank_data = []
for s in strats_common:
    sim_row = df_sim_agg[df_sim_agg["strategy"] == s].iloc[0]
    pn, pm, _ = paper_stats[s]
    ln, lm, _ = live_stats.get(s, (0, float('nan'), float('nan')))
    rank_data.append((s, sim_row["sim_n"], sim_row["sim_avg_pnl"], pn, pm, ln, lm))
    print(f"{s:<30}{sim_row['sim_n']:>7.0f}{sim_row['sim_avg_pnl']:>+8.2f}%{pn:>9d}{pm:>+8.2f}%{ln:>9d}{lm:>+8.2f}%")

# -- 4. Rank correlation (Spearman)
import math
def spearman(xs, ys):
    n = len(xs)
    if n < 3: return float('nan')
    rx = {v: i for i, v in enumerate(sorted(xs))}
    ry = {v: i for i, v in enumerate(sorted(ys))}
    d2 = sum((rx[x]-ry[y])**2 for x, y in zip(xs, ys))
    return 1 - 6*d2 / (n*(n*n-1))

# Sim vs paper
sp_sim = [r[2] for r in rank_data]
sp_paper = [r[4] for r in rank_data]
print(f"\nSpearman rank corr sim vs paper (N={len(sp_sim)}): rho = {spearman(sp_sim, sp_paper):+.3f}")

rank_live = [(r[2], r[6]) for r in rank_data if r[5] >= 5 and not math.isnan(r[6])]
if rank_live:
    print(f"Spearman rank corr sim vs live  (N={len(rank_live)}): rho = {spearman([x[0] for x in rank_live],[x[1] for x in rank_live]):+.3f}")

rank_pl = [(r[4], r[6]) for r in rank_data if r[5] >= 5 and not math.isnan(r[6])]
if rank_pl:
    print(f"Spearman rank corr paper vs live (N={len(rank_pl)}): rho = {spearman([x[0] for x in rank_pl],[x[1] for x in rank_pl]):+.3f}")

# -- 5. Top-5 overlap
print("\n== Top-5 overlap ==")
top_sim = set([r[0] for r in sorted(rank_data, key=lambda x: -x[2])[:5]])
top_paper = set([r[0] for r in sorted(rank_data, key=lambda x: -x[4])[:5]])
print(f"sim top5   : {top_sim}")
print(f"paper top5 : {top_paper}")
print(f"overlap    : {top_sim & top_paper}")
live_eligible = [r for r in rank_data if r[5] >= 5 and not math.isnan(r[6])]
if live_eligible:
    top_live = set([r[0] for r in sorted(live_eligible, key=lambda x: -x[6])[:5]])
    print(f"live top5  : {top_live}")
    print(f"sim ∩ live : {top_sim & top_live}")
    print(f"paper ∩ live: {top_paper & top_live}")

import json
out = os.path.join(os.path.dirname(__file__), "..", "data", "ranking_compare.json")
with open(out, "w") as f:
    json.dump([{"strategy":r[0],"sim_n":r[1],"sim_pnl":r[2],"paper_n":r[3],"paper_pnl":r[4],"live_n":r[5],"live_pnl":r[6]} for r in rank_data], f, indent=2, default=str)
print(f"\nSaved -> {out}")
