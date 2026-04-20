"""Calibrate _dynamic_sell_slip_factor from per-pair PnL divergence.

The right metric: for each (token, strategy) pair where we have both a rt_live row
and a paper rt row, slip delta (bps) ≈ (pnl_paper - pnl_live) * 10000.

A positive slip delta means sim was too optimistic (applied less slip than real)
and paper_trader should apply MORE bps.
Break down by (exit_type, is_pump).
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
        out.extend(r.data)
        if len(r.data) < step: break
        off += step
    return out

rows = fetch_all("paper_trades",
    "id,token_address,strategy,status,source,pnl_pct,rt_is_pump_fun,rt_liquidity_usd",
    gte_created_at=SINCE)
closed = [r for r in rows if r.get("status") in ("sl_hit","trail_stop","tp_hit","timeout","be_stop")
          and not str(r.get("strategy","")).startswith("DTRAIL")]

by_key = defaultdict(dict)
for r in closed:
    src = "live" if r.get("source") == "rt_live" else "paper"
    by_key[(r["token_address"], r["strategy"])][src] = r

pairs = [(k, v["live"], v["paper"]) for k, v in by_key.items() if "live" in v and "paper" in v]
print(f"Matched pairs (live, paper same token+strat): {len(pairs)}\n")

# Group by (live exit_type, is_pump)
groups = defaultdict(list)
for (_, strat), lv, pp in pairs:
    et = lv["status"]
    pf = bool(lv.get("rt_is_pump_fun"))
    # bps delta: positive means live PnL > paper PnL -> sim is too pessimistic -> decrease paper type_bps
    pnl_l = float(lv.get("pnl_pct") or 0)
    pnl_p = float(pp.get("pnl_pct") or 0)
    delta_bps = round((pnl_l - pnl_p) * 10000)  # positive = live better than paper
    groups[(et, pf)].append(delta_bps)

print(f"{'exit_type':<14}{'pump':<6}{'N':>5}{'median':>10}{'p25':>8}{'p75':>8}{'mean':>10}")
print("-"*62)
calibration = {}
for (et, pf), xs in sorted(groups.items()):
    xs_s = sorted(xs)
    n = len(xs); med = st.median(xs); mn = st.mean(xs)
    p25 = xs_s[int(0.25*n)] if n >= 4 else med
    p75 = xs_s[int(0.75*n)] if n >= 4 else med
    print(f"{et:<14}{'Y' if pf else 'N':<6}{n:>5}{med:>+10.0f}{p25:>+8.0f}{p75:>+8.0f}{mn:>+10.0f}")
    calibration[f"{et}|pump={pf}"] = {"n":n,"median_bps":round(med),"p25":round(p25),"p75":round(p75)}

out = os.path.join(os.path.dirname(__file__), "..", "data", "slip_calibration_v2.json")
with open(out, "w", encoding="utf-8") as f: json.dump(calibration, f, indent=2)
print(f"\nSaved -> {out}")
print("\nInterpretation:")
print("  median > 0 -> live PnL > paper PnL -> sim too pessimistic -> DECREASE paper type_bps by median")
print("  median < 0 -> paper PnL > live PnL -> sim too optimistic -> INCREASE paper type_bps by |median|")
