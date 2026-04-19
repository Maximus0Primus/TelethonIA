"""Benchmark polling modes side-by-side on the SAME trades+ticks universe.

For each live strategy (BE25_TP80_SL30, FAST_TP50_SL30) replay on Apr 13-19
paper trades through every INTERVAL_PROFILE (CURRENT, FAST_15, FAST_30,
LAZY_FAST, LAZY_MED, LAZY_STD, LAZY_SLOW, LAZY_XSLOW).

Answers: is LAZY helping, hurting, or neutral for these specific TP/SL combos?
Same tokens → perfect control for tick noise / strategy mix confounders.
"""
import os, sys, statistics as st, json
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from dotenv import load_dotenv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
import sim as _sim
from sim import _replay_with_intervals, _filter_ticks_by_source, _fetch_ticks_for_tokens
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

SINCE = os.environ.get("SINCE", "2026-04-13T00:00:00+00:00")

INTERVAL_PROFILES = [
    ("CURRENT",    0,   0,   0),
    ("FAST_15",   15,  60,  60),
    ("FAST_30",   30, 120, 120),
    ("LAZY_FAST", 60, 120, 180),
    ("LAZY_MED", 120, 300, 360),
    ("LAZY_STD", 180, 300, 600),
    ("LAZY_SLOW",300, 600, 900),
    ("LAZY_XSLOW",600,900,1200),
]

# (strategy_name, tp_pct_or_None, sl_pct, horizon_min, be_act_pct_or_None)
STRATS = [
    ("BE25_TP80_SL30",   80, 30, 120, 25),
    ("FAST_TP50_SL30",   50, 30, 30,  None),
    ("FAST_TP80_SL25",   80, 25, 30,  None),
    ("BE15_TP100_SL50", 100, 50, 120, 15),
]


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


# Use BE25 trades as the universe (it's the richest shadow'd strat)
print(f"Loading universe since {SINCE}...")
trades = fetch_all("paper_trades",
    "id,token_address,created_at,entry_price,dex_spot_price_at_entry,rt_liquidity_usd,rt_is_pump_fun,strategy",
    gte_created_at=SINCE, eq_strategy="BE25_TP80_SL30", eq_source="rt")
# Dedup per token (first call only, cleaner comparison)
seen = set(); universe = []
for t in sorted(trades, key=lambda x: x["created_at"]):
    if t["token_address"] in seen: continue
    seen.add(t["token_address"]); universe.append(t)
print(f"  universe: {len(universe)} unique tokens")

# Load ticks
print("Loading ticks...")
token_ranges = {}
for t in universe:
    entry = t["created_at"]
    entry_dt = datetime.fromisoformat(entry.replace("Z", "+00:00"))
    end_iso = (entry_dt + timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    token_ranges[t["token_address"]] = (entry, end_iso)
ticks_by_token = _fetch_ticks_for_tokens(token_ranges)
print(f"  ticks for {len(ticks_by_token)}/{len(universe)} tokens")


def run_mode(strat_name, tp, sl, horizon, be_act, mode_label, fs, fw, ss, source="dexscreener"):
    pnls = []
    for t in universe:
        raw = ticks_by_token.get(t["token_address"])
        if not raw: continue
        ticks = _filter_ticks_by_source(raw, source)
        if not ticks: continue
        entry_price = float(t["entry_price"])
        fake = {
            "id": f"{strat_name}_{mode_label}_{t['id']}",
            "entry_price": entry_price,
            "sl_price": entry_price * (1 - sl / 100),
            "tp_price": entry_price * (1 + tp / 100) if tp else None,
            "position_usd": 10.0,
            "strategy": strat_name,
            "tranche_label": "main",
            "horizon_minutes": horizon,
            "created_at": t["created_at"],
            "high_price_seen": entry_price,
            "rt_liquidity_usd": t.get("rt_liquidity_usd"),
            "dex_spot_price_at_entry": float(t.get("dex_spot_price_at_entry") or 0),
            "rt_is_pump_fun": t.get("rt_is_pump_fun"),
        }
        sim = _replay_with_intervals(fake, ticks, fs, fw, ss)
        if sim is None: continue
        pnls.append(float(sim["pnl_pct"]) * 100)
    return pnls


print(f"\n{'Strategy':<22}{'Mode':<12}{'N':>4}{'WR':>7}{'med%':>8}{'mean%':>8}{'$/trade':>9}")
print("-"*72)
all_rows = []
for strat_name, tp, sl, horizon, be_act in STRATS:
    base_pnls = None
    for label, fs, fw, ss in INTERVAL_PROFILES:
        pnls = run_mode(strat_name, tp, sl, horizon, be_act, label, fs, fw, ss)
        if not pnls:
            print(f"{strat_name:<22}{label:<12}  (no data)"); continue
        n = len(pnls); wr = 100*sum(1 for p in pnls if p>0)/n
        med = st.median(pnls); mn = st.mean(pnls)
        dpt = mn * 0.1  # $1 per 10% on $10 notional
        print(f"{strat_name:<22}{label:<12}{n:>4}{wr:>6.1f}%{med:>+7.2f}%{mn:>+7.2f}%{dpt:>+8.2f}")
        all_rows.append({"strategy":strat_name,"mode":label,"n":n,"wr":wr,
                         "median":med,"mean":mn})
        if label == "CURRENT": base_pnls = pnls
        # Paired delta vs CURRENT
        if base_pnls and label != "CURRENT" and len(pnls) == len(base_pnls):
            deltas = [p - b for p, b in zip(pnls, base_pnls)]
            wins = sum(1 for d in deltas if d > 0)
            print(f"{'  → Δ vs CURRENT:':>34}{'':<4}  mean={st.mean(deltas):+5.2f}pp  med={st.median(deltas):+5.2f}pp  wins={wins}/{len(deltas)}")
    print()

out = os.path.join(os.path.dirname(__file__), "..", "data", "bench_poll_modes.json")
with open(out, "w", encoding="utf-8") as f: json.dump(all_rows, f, indent=2)
print(f"Saved -> {out}")
