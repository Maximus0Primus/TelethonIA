"""ETH state — paper performance per strat last 7d/3d, live config, infra readiness."""
import os, sys, json, statistics as st
from collections import defaultdict
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

CLOSED = ("sl_hit", "trail_stop", "tp_hit", "timeout", "be_stop")
NOW = datetime.now(timezone.utc)
PERIODS = [("ALL", None, None), ("7d", NOW - timedelta(days=7), 7),
           ("3d", NOW - timedelta(days=3), 3), ("24h", NOW - timedelta(hours=24), 1)]

# 1. live config — what's currently live for ETH?
cfg_row = sb.table("scoring_config").select("rt_trade_config").eq("id", 1).execute().data[0]
rt_cfg = cfg_row["rt_trade_config"]
if isinstance(rt_cfg, str):
    rt_cfg = json.loads(rt_cfg)
print("=" * 92)
print("ETH LIVE CONFIG")
print("=" * 92)
allocs = (rt_cfg.get("hybrid_strategy") or {}).get("allocations") or {}
eth_allocs = {k: v for k, v in allocs.items() if k.startswith("ETH_")}
print(f"  paper_main allocations (ETH only): {eth_allocs}")
live_chains = (rt_cfg.get("live_trading") or {}).get("chains_enabled") or rt_cfg.get("live_chains") or "—"
print(f"  live_trading.chains_enabled: {live_chains}")
print(f"  live_trading keys: {sorted((rt_cfg.get('live_trading') or {}).keys())}")
# look for any ETH-specific live setting
for k, v in (rt_cfg.get("live_trading") or {}).items():
    if "eth" in k.lower() or "ethereum" in k.lower() or "evm" in k.lower():
        print(f"    {k}: {v}")

# 2. ETH paper trades performance — main strats
print()
print("=" * 92)
print("ETH PAPER MAIN — perf per period (rt source, chain=ethereum)")
print("=" * 92)
eth_strats = sorted([k for k in allocs if k.startswith("ETH_")])
for strat in eth_strats:
    print(f"\n  {strat}")
    print(f"  {'period':<8}{'N':>5}{'WR%':>7}{'avg%':>9}{'$ tot':>10}{'$/d':>9}")
    for label, since, days in PERIODS:
        q = sb.table("paper_trades").select(
            "status,pnl_pct,pnl_usd,created_at"
        ).eq("strategy", strat).eq("source", "rt").eq("chain", "ethereum")
        if since is not None:
            q = q.gte("created_at", since.isoformat())
        rows = []
        off, step = 0, 1000
        while True:
            r = q.range(off, off+step-1).execute().data or []
            rows += r
            if len(r) < step:
                break
            off += step
        rows = [r for r in rows if r.get("status") in CLOSED]
        if not rows:
            print(f"  {label:<8}{0:>5}    —        —         —        —")
            continue
        pcts = [float(r.get("pnl_pct") or 0) * 100 for r in rows]
        usd = sum(float(r.get("pnl_usd") or 0) for r in rows)
        wr = 100 * sum(1 for p in pcts if p > 0) / len(pcts)
        per_d = f"{usd/days:+.2f}" if days else "—"
        print(f"  {label:<8}{len(pcts):>5}{wr:>6.1f}%{st.mean(pcts):>+8.2f}%{usd:>+9.2f}{per_d:>9}")

# 3. ETH live — has it ever traded?
print()
print("=" * 92)
print("ETH LIVE TRADES — has any ETH live trade ever happened?")
print("=" * 92)
all_live = []
off, step = 0, 1000
while True:
    r = sb.table("paper_trades").select(
        "strategy,status,source,pnl_pct,pnl_usd,kol_group,token_address,symbol,created_at,chain"
    ).eq("source", "rt_live").eq("chain", "ethereum").range(off, off+step-1).execute().data or []
    all_live += r
    if len(r) < step:
        break
    off += step
print(f"  total ETH rt_live rows ever: {len(all_live)}")
if all_live:
    closed = [r for r in all_live if r.get("status") in CLOSED]
    print(f"  closed: {len(closed)}")
    if closed:
        usd = sum(float(r.get("pnl_usd") or 0) for r in closed)
        pcts = [float(r.get("pnl_pct") or 0) * 100 for r in closed]
        wr = 100 * sum(1 for p in pcts if p > 0) / len(pcts)
        print(f"  WR={wr:.1f}%  avg={st.mean(pcts):+.2f}%  $tot={usd:+.2f}")
        print(f"  date range: {min(r['created_at'] for r in closed)[:16]} -> {max(r['created_at'] for r in closed)[:16]}")
        for r in sorted(closed, key=lambda x: x["created_at"])[:8]:
            print(f"    {r['created_at'][:16]}  {r['symbol']:<10} {r['strategy']:<26} {float(r.get('pnl_pct') or 0)*100:+.2f}% ${float(r.get('pnl_usd') or 0):+.2f}")
