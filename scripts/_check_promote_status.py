"""Quick check: where are the 6 newly promoted strats now? + SOL live recap."""
import os, sys, statistics as st
from collections import defaultdict
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

NEW_SOL = ["BE25_LOCK10_TP100_SL30_NZ_S40", "BE25_LOCK10_TP100_SL30_S40", "FAST45_TP40_SL30_S30"]
NEW_ETH = ["ETH_SLOW4H_TP100_SL50", "ETH_BE50_TP150_SL40_T2H", "ETH_TP300_SL50_4H"]
ALL_NEW = NEW_SOL + NEW_ETH
PROMOTE_TS = "2026-04-27T08:26:00+00:00"  # seeding was at ~08:26 UTC
NOW = datetime.now(timezone.utc)
print(f"NOW (UTC) = {NOW.isoformat()}")
print(f"PROMOTE   = {PROMOTE_TS}\n")

# 1. Current bankroll for the 6 new
br = sb.table("rt_bankroll").select("*").limit(1).execute().data[0]
print("=" * 92)
print("1. BANKROLL (per_chain) - 6 new strats")
print("=" * 92)
flat = br.get("strategy_bankrolls", {}) or {}
per_chain = br.get("strategy_bankrolls_per_chain", {}) or {}
for chain_db, chain_label, names in (("solana", "SOL", NEW_SOL), ("ethereum", "ETH", NEW_ETH)):
    pool = per_chain.get(chain_db, {}) or {}
    for s in names:
        e = pool.get(s, {})
        bal = e.get("current_balance", "MISSING")
        pnl = e.get("total_pnl", "—")
        n = e.get("total_trades", "—")
        flat_bal = flat.get(s, {}).get("balance") or flat.get(s, {}).get("current_balance", "—")
        print(f"  [{chain_label}] {s:<38}  bal={bal:>10}  pnl={pnl:>+8}  N={n:<5}  flat={flat_bal}")
print(f"\n  starting_capital={br.get('starting_capital')}  current_balance={br.get('current_balance')}  total_pnl={br.get('total_pnl')}")

# 2. paper_trades since promote — 6 new strats
print()
print("=" * 92)
print("2. PAPER TRADES of 6 new strats SINCE PROMOTE (rt source)")
print("=" * 92)
CLOSED = ("sl_hit", "trail_stop", "tp_hit", "timeout", "be_stop")
for s in ALL_NEW:
    rows = sb.table("paper_trades").select(
        "id,strategy,status,source,pnl_pct,pnl_usd,chain,created_at"
    ).eq("strategy", s).gte("created_at", PROMOTE_TS).execute().data or []
    closed = [r for r in rows if r.get("status") in CLOSED]
    open_ = [r for r in rows if r.get("status") not in CLOSED]
    if not rows:
        print(f"  {s:<38}  no trades since promote")
        continue
    pnls_pct = [float(r.get("pnl_pct") or 0) * 100 for r in closed]
    pnls_usd = [float(r.get("pnl_usd") or 0) for r in closed]
    sum_usd = sum(pnls_usd)
    wr = (100 * sum(1 for p in pnls_pct if p > 0) / len(closed)) if closed else 0
    avg = st.mean(pnls_pct) if pnls_pct else 0
    print(f"  {s:<38}  N_total={len(rows):<3} closed={len(closed):<3} open={len(open_):<3}  WR={wr:.0f}%  avg={avg:+.2f}%  $tot={sum_usd:+.2f}")

# 3. SOL live since 24h
print()
print("=" * 92)
print("3. SOLANA LIVE — last 24h (rt_live)")
print("=" * 92)
since24 = (NOW - timedelta(hours=24)).isoformat()
live_rows = sb.table("paper_trades").select(
    "strategy,status,source,pnl_pct,pnl_usd,chain,created_at"
).eq("source", "rt_live").eq("chain", "solana").gte("created_at", since24).execute().data or []
closed_l = [r for r in live_rows if r.get("status") in CLOSED]
open_l = [r for r in live_rows if r.get("status") not in CLOSED]
print(f"  total live SOL rows last 24h = {len(live_rows)}  closed={len(closed_l)}  open={len(open_l)}")
by_strat = defaultdict(list)
for r in closed_l:
    by_strat[r["strategy"]].append(r)
total_usd = sum(float(x.get("pnl_usd") or 0) for r in by_strat.values() for x in r)
print(f"  total $ closed last 24h = {total_usd:+.2f}\n")
print(f"{'  Strategy':<32}{'N':>5}{'WR%':>7}{'avg%':>9}{'$ tot':>10}")
for s, xs in sorted(by_strat.items(), key=lambda kv: -sum(float(x.get('pnl_usd') or 0) for x in kv[1])):
    n = len(xs)
    pct = [float(x.get("pnl_pct") or 0) * 100 for x in xs]
    usd = sum(float(x.get("pnl_usd") or 0) for x in xs)
    wr = 100 * sum(1 for p in pct if p > 0) / n
    print(f"  {s:<30}{n:>5}{wr:>6.1f}%{st.mean(pct):>+8.2f}%{usd:>+9.2f}")

# 4. SOL live since promote (4h)
print()
print("=" * 92)
print("4. SOLANA LIVE — since promote ~4h ago")
print("=" * 92)
live4 = [r for r in live_rows if r.get("created_at") and r["created_at"] >= PROMOTE_TS]
closed4 = [r for r in live4 if r.get("status") in CLOSED]
open4 = [r for r in live4 if r.get("status") not in CLOSED]
print(f"  total SOL live rows since promote = {len(live4)}  closed={len(closed4)}  open={len(open4)}")
by4 = defaultdict(list)
for r in closed4:
    by4[r["strategy"]].append(r)
total4 = sum(float(x.get("pnl_usd") or 0) for r in by4.values() for x in r)
print(f"  total $ closed since promote = {total4:+.2f}\n")
print(f"{'  Strategy':<32}{'N':>5}{'WR%':>7}{'avg%':>9}{'$ tot':>10}")
for s, xs in sorted(by4.items(), key=lambda kv: -sum(float(x.get('pnl_usd') or 0) for x in kv[1])):
    n = len(xs)
    pct = [float(x.get("pnl_pct") or 0) * 100 for x in xs]
    usd = sum(float(x.get("pnl_usd") or 0) for x in xs)
    wr = 100 * sum(1 for p in pct if p > 0) / n
    print(f"  {s:<30}{n:>5}{wr:>6.1f}%{st.mean(pct):>+8.2f}%{usd:>+9.2f}")
