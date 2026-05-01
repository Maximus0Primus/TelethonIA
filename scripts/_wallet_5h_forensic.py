"""Forensic 5h — pourquoi -$15 entre ~15h45 et 20h45 UTC.
Utilise gas_usd_buy/sell, slippage_actual_bps qui sont en DB."""
import os, sys, statistics as st
from collections import defaultdict, Counter
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

NOW = datetime.now(timezone.utc)
SINCE_5H = (NOW - timedelta(hours=5)).isoformat()
SINCE_24H = (NOW - timedelta(hours=24)).isoformat()
CLOSED = ("sl_hit", "trail_stop", "tp_hit", "timeout", "be_stop")

COLS = ("strategy,status,source,pnl_pct,pnl_usd,kol_group,token_address,symbol,"
        "created_at,exit_at,position_usd,is_shadow,chain,entry_price,exit_price,"
        "execution_price,slippage_actual_bps,buy_slippage_bps,sell_slippage_bps,"
        "gas_usd_buy,gas_usd_sell,buy_attempts,sell_attempts,buy_exec_ms,sell_exec_ms,"
        "paper_sim_pnl_pct")

def fetch(src, since_col, since):
    out, off, step = [], 0, 1000
    while True:
        q = sb.table("paper_trades").select(COLS).eq("source", src).eq("chain", "solana").gte(since_col, since)
        r = q.range(off, off+step-1).execute()
        if not r.data:
            break
        out.extend(r.data)
        if len(r.data) < step:
            break
        off += step
    return [r for r in out if not r.get("is_shadow")]

# trades closed in last 5h (regardless of when opened)
closed5 = [r for r in fetch("rt_live", "exit_at", SINCE_5H) if r["status"] in CLOSED]
closed24 = [r for r in fetch("rt_live", "exit_at", SINCE_24H) if r["status"] in CLOSED]

print("=" * 100)
print(f"5H FORENSIC — trades CLOSED since {SINCE_5H[:19]} UTC")
print("=" * 100)
print(f"\nclosed in last 5h: {len(closed5)}")
print(f"closed in last 24h: {len(closed24)}")

total_pnl_5h = sum(float(r.get("pnl_usd") or 0) for r in closed5)
total_gas_5h = sum((float(r.get("gas_usd_buy") or 0) + float(r.get("gas_usd_sell") or 0)) for r in closed5)
total_pnl_24h = sum(float(r.get("pnl_usd") or 0) for r in closed24)
total_gas_24h = sum((float(r.get("gas_usd_buy") or 0) + float(r.get("gas_usd_sell") or 0)) for r in closed24)

print(f"\n=== 5H ===")
print(f"  PnL realized (already net of slip on exec_price): ${total_pnl_5h:+.2f}")
print(f"  Gas total (buy+sell, from DB):                   ${total_gas_5h:.2f}")
print(f"  TOTAL wallet impact (PnL - gas):                 ${total_pnl_5h - total_gas_5h:+.2f}")

print(f"\n=== 24H ===")
print(f"  PnL realized: ${total_pnl_24h:+.2f}")
print(f"  Gas total:    ${total_gas_24h:.2f}")
print(f"  TOTAL impact: ${total_pnl_24h - total_gas_24h:+.2f}")

# Per-trade gas distribution
gas_per_trade = [(float(r.get("gas_usd_buy") or 0) + float(r.get("gas_usd_sell") or 0)) for r in closed24 if (r.get("gas_usd_buy") or r.get("gas_usd_sell"))]
if gas_per_trade:
    print(f"\n=== GAS PER ROUND-TRIP (N={len(gas_per_trade)} trades with gas data) ===")
    print(f"  median: ${st.median(gas_per_trade):.4f}")
    print(f"  mean:   ${st.mean(gas_per_trade):.4f}")
    print(f"  min:    ${min(gas_per_trade):.4f}")
    print(f"  max:    ${max(gas_per_trade):.4f}")
    print(f"  p90:    ${sorted(gas_per_trade)[int(len(gas_per_trade)*0.9)]:.4f}")

# How many trades have gas data?
no_gas_24h = [r for r in closed24 if not (r.get("gas_usd_buy") or r.get("gas_usd_sell"))]
print(f"\n  trades MISSING gas data 24h: {len(no_gas_24h)}/{len(closed24)}")

# Status distribution 5h
print(f"\n=== STATUS 5H ===")
by_status = Counter(r["status"] for r in closed5)
for s, n in by_status.most_common():
    pnl = sum(float(r.get("pnl_usd") or 0) for r in closed5 if r["status"] == s)
    gas = sum((float(r.get("gas_usd_buy") or 0) + float(r.get("gas_usd_sell") or 0)) for r in closed5 if r["status"] == s)
    print(f"  {s:<15} N={n:<3} pnl=${pnl:+.2f} gas=${gas:.2f} net=${pnl-gas:+.2f}")

# Per-strat 5h
print(f"\n=== PER-STRAT 5H ===")
by_strat = defaultdict(list)
for r in closed5:
    by_strat[r["strategy"]].append(r)
print(f"  {'strat':<34}{'N':>4}{'WR':>6}{'avg%':>9}{'med%':>9}{'$pnl':>9}{'$gas':>8}{'net':>9}")
for s, xs in sorted(by_strat.items(), key=lambda kv: sum(float(x.get('pnl_usd') or 0) - (float(x.get('gas_usd_buy') or 0) + float(x.get('gas_usd_sell') or 0)) for x in kv[1])):
    pcts = [float(r.get("pnl_pct") or 0)*100 for r in xs]
    pnl = sum(float(r.get("pnl_usd") or 0) for r in xs)
    gas = sum((float(r.get("gas_usd_buy") or 0) + float(r.get("gas_usd_sell") or 0)) for r in xs)
    wr = 100*sum(1 for p in pcts if p > 0)/len(xs)
    print(f"  {s:<32}{len(xs):>4}{wr:>5.0f}%{st.mean(pcts):>+8.2f}%{st.median(pcts):>+8.2f}%{pnl:>+8.2f}{gas:>+8.2f}{pnl-gas:>+8.2f}")

# Top 15 worst losses 5h
print(f"\n=== TOP 15 WORST TRADES 5H ===")
print(f"  {'time':<6}{'strat':<32}{'symbol':<10}{'pos':>7}{'pnl':>8}{'gas':>7}{'pct':>8}{'slip_buy':>10}{'slip_act':>10}{'status':<12}{'attempts':<10}")
sorted_pnl = sorted(closed5, key=lambda r: float(r.get("pnl_usd") or 0))
for r in sorted_pnl[:15]:
    pnl = float(r.get("pnl_usd") or 0)
    pct = float(r.get("pnl_pct") or 0) * 100
    pos = float(r.get("position_usd") or 0)
    gas = float(r.get("gas_usd_buy") or 0) + float(r.get("gas_usd_sell") or 0)
    slip_b = r.get("buy_slippage_bps") or 0
    slip_a = r.get("slippage_actual_bps") or 0
    ba = r.get("buy_attempts") or "?"
    sa = r.get("sell_attempts") or "?"
    t = r.get("exit_at","")[11:16]
    print(f"  {t:<6}{r['strategy']:<32}{(r.get('symbol') or '?'):<10}{pos:>6.2f} {pnl:>+7.2f}{gas:>+6.2f}{pct:>+7.1f}%{int(slip_b):>9}b{int(slip_a):>9}b {r['status']:<12} b{ba}/s{sa}")

# Distribution slippage_actual_bps
slip_actuals = [int(r.get("slippage_actual_bps") or 0) for r in closed24 if r.get("slippage_actual_bps")]
if slip_actuals:
    print(f"\n=== SLIPPAGE ACTUAL bps 24h (N={len(slip_actuals)}) ===")
    print(f"  median: {int(st.median(slip_actuals))} bps")
    print(f"  mean:   {int(st.mean(slip_actuals))} bps")
    print(f"  p90:    {int(sorted(slip_actuals)[int(len(slip_actuals)*0.9)])} bps")
    print(f"  max:    {max(slip_actuals)} bps")

# All strats live drift 5h
print(f"\n{'='*100}\nAPPLES-TO-APPLES drift live vs paper 5h\n{'='*100}")
LIVE_STRATS = sorted(set(r["strategy"] for r in closed5))

def fetch_strat(src, since, strat):
    out, off, step = [], 0, 1000
    while True:
        r = sb.table("paper_trades").select(COLS).eq("source", src).eq("chain", "solana").eq("strategy", strat).gte("created_at", since).range(off, off+step-1).execute()
        if not r.data:
            break
        out.extend(r.data)
        if len(r.data) < step:
            break
        off += step
    return [r for r in out if r.get("status") in CLOSED and not r.get("is_shadow")]

def parse_dt(s):
    return datetime.fromisoformat(s.replace("Z","+00:00"))

def pair(live_rows, paper_rows, window_min=60):
    pairs = []
    by_t = defaultdict(list)
    for p in paper_rows:
        by_t[p["token_address"]].append(p)
    for l in live_rows:
        cands = [p for p in by_t.get(l["token_address"], []) if p["strategy"] == l["strategy"]]
        if not cands:
            continue
        l_t = parse_dt(l["created_at"])
        best, best_dt = None, timedelta(minutes=window_min+1)
        for p in cands:
            dt = abs(parse_dt(p["created_at"]) - l_t)
            if dt < best_dt:
                best, best_dt = p, dt
        if best and best_dt <= timedelta(minutes=window_min):
            pairs.append((l, best))
    return pairs

print(f"  {'strat':<34}{'pairs':>7}{'live%':>9}{'paper%':>9}{'drift_avg':>11}{'drift_med':>11}")
for strat in LIVE_STRATS:
    live = [r for r in closed5 if r["strategy"] == strat]
    paper = fetch_strat("rt", SINCE_24H, strat)
    pairs = pair(live, paper, window_min=60)
    if not pairs:
        print(f"  {strat:<32}{0:>7}     no pair")
        continue
    diffs = [(float(l.get("pnl_pct") or 0)*100) - (float(p.get("pnl_pct") or 0)*100) for l, p in pairs]
    live_avg = st.mean([float(l.get("pnl_pct") or 0)*100 for l, _ in pairs])
    twin_avg = st.mean([float(p.get("pnl_pct") or 0)*100 for _, p in pairs])
    print(f"  {strat:<32}{len(pairs):>7}{live_avg:>+8.2f}%{twin_avg:>+8.2f}%{st.mean(diffs):>+10.2f}{st.median(diffs):>+10.2f}")
