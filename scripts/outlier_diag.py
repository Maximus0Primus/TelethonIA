"""Find paper↔live outliers (|pnl_live - pnl_paper| > 10pp) and identify
root cause per pair: entry mismatch, exit status mismatch, or timing skew.
"""
import os, sys, json
from collections import defaultdict
from dotenv import load_dotenv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
SINCE = "2026-04-13T20:00:00+00:00"

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
    "id,symbol,strategy,status,source,pnl_pct,entry_price,exit_price,execution_price,"
    "dex_spot_price_at_entry,high_price_seen,entry_source,created_at,exit_at,"
    "rt_is_pump_fun,rt_liquidity_usd,token_address",
    gte_created_at=SINCE)
closed = [r for r in rows if r.get("status") in ("sl_hit","trail_stop","tp_hit","timeout","be_stop")
          and not str(r.get("strategy","")).startswith("DTRAIL")]

pairs = defaultdict(dict)
for r in closed:
    src = "live" if r.get("source") == "rt_live" else "paper"
    pairs[(r["token_address"], r["strategy"])][src] = r

outliers = []
for (tok, strat), v in pairs.items():
    if "live" not in v or "paper" not in v: continue
    lv, pp = v["live"], v["paper"]
    dpp = (float(lv.get("pnl_pct") or 0) - float(pp.get("pnl_pct") or 0)) * 100
    if abs(dpp) <= 10: continue

    # Root-cause diagnostics
    def _f(x): return float(x) if x is not None else None
    entry_live, entry_paper = _f(lv.get("entry_price")), _f(pp.get("entry_price"))
    entry_div = ((entry_live / entry_paper - 1) * 100) if entry_live and entry_paper else None
    exit_live, exit_paper = _f(lv.get("exit_price")), _f(pp.get("exit_price"))
    exit_div = ((exit_live / exit_paper - 1) * 100) if exit_live and exit_paper else None
    status_match = lv["status"] == pp["status"]
    entry_synced = lv.get("entry_source") == "live_sync" or pp.get("entry_source") == "live_sync"

    outliers.append({
        "symbol": lv["symbol"], "strategy": strat,
        "pnl_live": float(lv.get("pnl_pct") or 0)*100,
        "pnl_paper": float(pp.get("pnl_pct") or 0)*100,
        "delta_pp": dpp,
        "status_live": lv["status"], "status_paper": pp["status"],
        "status_match": status_match,
        "entry_div_pct": entry_div,
        "exit_div_pct": exit_div,
        "entry_source_paper": pp.get("entry_source"),
        "entry_synced": entry_synced,
        "is_pump": lv.get("rt_is_pump_fun"),
    })

outliers.sort(key=lambda x: -abs(x["delta_pp"]))
print(f"Outliers |L-P| > 10pp : {len(outliers)}\n")

# Root-cause buckets
by_cause = defaultdict(list)
for o in outliers:
    if not o["status_match"]:
        cause = "exit_status_mismatch"
    elif o["entry_div_pct"] is not None and abs(o["entry_div_pct"]) > 3:
        cause = "entry_divergence_>3pct"
    elif o["exit_div_pct"] is not None and abs(o["exit_div_pct"]) > 3:
        cause = "exit_divergence_>3pct"
    else:
        cause = "other_timing"
    by_cause[cause].append(o)

print(f"{'root cause':<30}{'N':>4}{'L-P median pp':>16}")
print("-"*55)
import statistics as st
for c, xs in sorted(by_cause.items(), key=lambda x: -len(x[1])):
    med = st.median(o["delta_pp"] for o in xs)
    print(f"{c:<30}{len(xs):>4}{med:>+15.2f}")

print(f"\n{'Top 10 outliers':<40}")
print(f"{'symbol':<12}{'strat':<22}{'L%':>8}{'P%':>8}{'Δpp':>8}{'statL':>10}{'statP':>10}{'entrΔ%':>9}{'exitΔ%':>9}{'sync':>6}")
for o in outliers[:10]:
    print(f"{o['symbol']:<12}{o['strategy']:<22}{o['pnl_live']:>+7.2f}{o['pnl_paper']:>+7.2f}{o['delta_pp']:>+8.1f}"
          f"{o['status_live']:>10}{o['status_paper']:>10}"
          f"{(o['entry_div_pct'] or 0):>+8.2f}{(o['exit_div_pct'] or 0):>+8.2f}{'Y' if o['entry_synced'] else 'N':>6}")

out = os.path.join(os.path.dirname(__file__), "..", "data", "paper_live_outliers.json")
with open(out, "w", encoding="utf-8") as f: json.dump(outliers, f, indent=2, default=str)
print(f"\nSaved -> {out}")
