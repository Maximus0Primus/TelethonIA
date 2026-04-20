"""Investigate the Spearman drift sim<->paper.

Hypotheses tested:
  H1: pollution by new v144 shadow strats   -> exclude *_NOLAZY/_LAZY*/_BOTH/_JUPITER/_S30/_S40/_MCAP/_COMBO/_MED3/_DS shadows
  H2: stale sim sweep CSV                    -> compare v142 vs extended (Apr 19)
  H3: low-N paper strategies pollute         -> bump min_n 10 -> 30
  H4: window mismatch                        -> compare 7d vs 14d paper window
"""
import os, sys, math, statistics as st, pandas as pd
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

NOW = datetime.now(timezone.utc)
SCRAPER = os.path.join(os.path.dirname(__file__), "..", "scraper")

V144_SHADOW_SUFFIXES = ("_NOLAZY","_LAZYFAST","_LAZYMED","_LAZYSLOW","_LAZYXSLOW","_LAZY_STD",
                        "_BOTH","_JUPITER","_S30","_S40","_MCAP","_COMBO","_MED3","_DS",
                        "_HYST")

def fetch_all_paper(since_iso):
    out, off = [], 0
    while True:
        q = sb.table("paper_trades").select("strategy,status,source,pnl_pct").gte("created_at", since_iso)
        r = q.range(off, off+999).execute().data
        out.extend(r)
        if len(r) < 1000: break
        off += 1000
    return [r for r in out if r.get("status") in ("sl_hit","trail_stop","tp_hit","timeout","be_stop")
            and not str(r.get("strategy","")).startswith("DTRAIL")]

def spearman(xs, ys):
    n = len(xs)
    if n < 3: return float('nan')
    rx = {v: i for i, v in enumerate(sorted(xs))}
    ry = {v: i for i, v in enumerate(sorted(ys))}
    d2 = sum((rx[x]-ry[y])**2 for x, y in zip(xs, ys))
    return 1 - 6*d2 / (n*(n*n-1))

def agg_paper(rows, min_n):
    bucket = defaultdict(list)
    for r in rows:
        if r.get("source") == "rt_live": continue
        bucket[r["strategy"]].append(float(r.get("pnl_pct") or 0) * 100)
    return {k: st.mean(v) for k, v in bucket.items() if len(v) >= min_n}

def load_sim(csv_path):
    df = pd.read_csv(csv_path)
    return (df[df["n"] >= 30]
            .groupby("strategy", as_index=False)
            .agg(sim=("avg_pnl_pct", "median"))
            .set_index("strategy")["sim"].to_dict())

def correlate(sim_dict, paper_dict, label, exclude_v144_shadows=False):
    common = set(sim_dict) & set(paper_dict)
    if exclude_v144_shadows:
        common = {s for s in common if not any(s.endswith(suf) for suf in V144_SHADOW_SUFFIXES)}
    xs = [sim_dict[s] for s in common]
    ys = [paper_dict[s] for s in common]
    rho = spearman(xs, ys)
    # top-5 overlap
    top_sim = set(sorted(common, key=lambda s: -sim_dict[s])[:5])
    top_pap = set(sorted(common, key=lambda s: -paper_dict[s])[:5])
    print(f"  {label:55s} N={len(common):3d} rho={rho:+.3f}  top5-overlap={len(top_sim & top_pap)}")
    return rho, common

# ------------- main matrix --------------
sim_v142 = load_sim(os.path.join(SCRAPER, "_mega_sweep_v142.csv"))
sim_ext  = load_sim(os.path.join(SCRAPER, "_mega_sweep_extended.csv"))
print(f"sim v142 N strats: {len(sim_v142)}  |  sim extended N strats: {len(sim_ext)}\n")

windows = [(7, "7d"), (14, "14d")]
min_ns = [10, 30]

for days, wlabel in windows:
    since = (NOW - timedelta(days=days)).isoformat()
    rows = fetch_all_paper(since)
    print(f"=== Window {wlabel} | {len(rows)} closed trades ===")
    for min_n in min_ns:
        paper = agg_paper(rows, min_n)
        print(f"  paper N>={min_n:2d}: {len(paper)} strats")
        for sim_name, sim in [("v142", sim_v142), ("extended", sim_ext)]:
            for excl in [False, True]:
                lbl = f"sim={sim_name}, paper N>={min_n}, exclude_v144_shadow={excl}"
                correlate(sim, paper, lbl, exclude_v144_shadows=excl)
    print()
