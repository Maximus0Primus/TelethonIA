"""Paired A/B audit — LAZY vs non-LAZY on matched (token, created_at).

Twins in current strategy set:
  BE25_TP80_SL30_DS (LAZY) vs BE25_TP80_SL30 (non-LAZY)
  BE25_TP80_SL30_HYST (LAZY, also HYST) vs BE25_TP80_SL30 (non-LAZY)  <- confounded
  FAST_TP50_SL30_HYST (LAZY+HYST) vs (no pure-non-LAZY counterpart)

Only _DS is a clean LAZY-vs-base pair. HYST confounds. Still worth reporting both.

We also compare LAZY strat population avg vs non-LAZY strat population avg on
the same token universe (less clean but N is bigger).
"""
import os, sys, statistics as st
from collections import defaultdict
from dotenv import load_dotenv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
from strategies import LAZY_STRATEGIES, STRATEGIES
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
    "strategy,status,source,pnl_pct,pnl_usd,token_address,created_at,entry_score,rt_liquidity_usd",
    gte_created_at=SINCE)
closed = [r for r in rows if r.get("status") in ("sl_hit","trail_stop","tp_hit","timeout","be_stop")
          and not str(r.get("strategy","")).startswith("DTRAIL")]
print(f"Closed trades (DTRAIL excl): {len(closed)}\n")


def summary(tag, xs):
    if not xs: print(f"  {tag}: N=0"); return None
    pnls = [float(r.get("pnl_pct") or 0)*100 for r in xs]
    n = len(pnls); wr = 100 * sum(1 for p in pnls if p>0) / n
    print(f"  {tag:<40}  N={n:5d}  WR={wr:5.1f}%  med={st.median(pnls):+6.2f}%  "
          f"mean={st.mean(pnls):+6.2f}%  std={st.pstdev(pnls):6.2f}%")
    return pnls


# === 1. Global: LAZY vs non-LAZY population ===
print("=== (1) Global population: LAZY vs non-LAZY ===\n")
lazy_rows = [r for r in closed if r["strategy"] in LAZY_STRATEGIES]
non_rows = [r for r in closed if r["strategy"] not in LAZY_STRATEGIES]
summary("LAZY (any strat)", lazy_rows)
summary("non-LAZY (any strat)", non_rows)
print("  → confounded : different strat names, different tokens/dates.\n")


# === 2. Shared-token population (restrict to tokens where BOTH populations appear) ===
lazy_tokens = {r["token_address"] for r in lazy_rows}
non_tokens = {r["token_address"] for r in non_rows}
common = lazy_tokens & non_tokens
print(f"=== (2) Restricted to {len(common)} tokens present in both LAZY and non-LAZY ===\n")
summary("LAZY on common tokens", [r for r in lazy_rows if r["token_address"] in common])
summary("non-LAZY on common tokens", [r for r in non_rows if r["token_address"] in common])
print("  → still confounded by strat-mix composition.\n")


# === 3. Clean paired twin: BE25_TP80_SL30_DS (LAZY) vs BE25_TP80_SL30 (non-LAZY) ===
print("=== (3) Clean paired twin : BE25_TP80_SL30_DS (LAZY) vs BE25_TP80_SL30 (non-LAZY) ===\n")
LAZY_STRAT = "BE25_TP80_SL30_DS"
BASE_STRAT = "BE25_TP80_SL30"
by_key = defaultdict(dict)
for r in closed:
    if r["strategy"] == LAZY_STRAT:
        by_key[r["token_address"]]["lazy"] = r
    elif r["strategy"] == BASE_STRAT:
        by_key[r["token_address"]]["base"] = r

paired = [(v["lazy"], v["base"]) for v in by_key.values() if "lazy" in v and "base" in v]
print(f"Matched pairs: {len(paired)}")
if paired:
    deltas_pp = [(float(lz.get("pnl_pct") or 0) - float(bs.get("pnl_pct") or 0)) * 100 for lz, bs in paired]
    summary("  LAZY (_DS)", [lz for lz, _ in paired])
    summary("  BASE", [bs for _, bs in paired])
    print(f"\n  Delta LAZY−BASE: N={len(deltas_pp)}  median={st.median(deltas_pp):+.2f}pp  "
          f"mean={st.mean(deltas_pp):+.2f}pp  "
          f"wins(Δ>0)={sum(1 for d in deltas_pp if d>0)}/{len(deltas_pp)}")
    # Sign test (simple binomial-ish check)
    wins = sum(1 for d in deltas_pp if d > 0)
    n = len(deltas_pp)
    print(f"  Sign test : {wins}/{n} pairs LAZY wins → "
          f"{'LAZY favored' if wins > n/2 else 'BASE favored' if wins < n/2 else 'tie'}")
print()


# === 4. LAZY+HYST vs base (confounded but informative) ===
print("=== (4) Paired confounded : BE25_TP80_SL30_HYST vs BE25_TP80_SL30 ===\n")
by_key = defaultdict(dict)
for r in closed:
    if r["strategy"] == "BE25_TP80_SL30_HYST":
        by_key[r["token_address"]]["lazy_hyst"] = r
    elif r["strategy"] == BASE_STRAT:
        by_key[r["token_address"]]["base"] = r
paired2 = [(v["lazy_hyst"], v["base"]) for v in by_key.values() if "lazy_hyst" in v and "base" in v]
print(f"Matched pairs: {len(paired2)}")
if paired2:
    deltas2 = [(float(lh.get("pnl_pct") or 0) - float(bs.get("pnl_pct") or 0)) * 100 for lh, bs in paired2]
    summary("  LAZY+HYST", [lh for lh, _ in paired2])
    summary("  BASE", [bs for _, bs in paired2])
    print(f"  Delta LAZY+HYST−BASE: N={len(deltas2)}  median={st.median(deltas2):+.2f}pp  "
          f"mean={st.mean(deltas2):+.2f}pp")
    print("  (confounded with HYST — HYST known losing per todo. Use (3) for LAZY signal.)")
