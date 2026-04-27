"""Confirm seed-schema bug + show live SOL drift since reset."""
import os, sys, json, statistics as st
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

br = sb.table("rt_bankroll").select("*").limit(1).execute().data[0]
flat = br.get("strategy_bankrolls", {}) or {}
per_chain = br.get("strategy_bankrolls_per_chain", {}) or {}

print("=" * 92)
print("FULL DUMP of new strats — both flat + per_chain")
print("=" * 92)
for s in NEW_SOL:
    print(f"\n[SOL] {s}")
    print(f"  flat       : {json.dumps(flat.get(s, {}), default=str)}")
    print(f"  per_chain  : {json.dumps(per_chain.get('solana', {}).get(s, {}), default=str)}")
for s in NEW_ETH:
    print(f"\n[ETH] {s}")
    print(f"  flat       : {json.dumps(flat.get(s, {}), default=str)}")
    print(f"  per_chain  : {json.dumps(per_chain.get('ethereum', {}).get(s, {}), default=str)}")

# Compare live vs paper for BE25/BOND last 7d
print()
print("=" * 92)
print("Paper vs Live drift — BE25_TP80_SL30 + BOND_FAST_TP50_SL20_T20 last 7d")
print("=" * 92)
NOW = datetime.now(timezone.utc)
since7 = (NOW - timedelta(days=7)).isoformat()
CLOSED = ("sl_hit", "trail_stop", "tp_hit", "timeout", "be_stop")
for strat in ("BE25_TP80_SL30", "BOND_FAST_TP50_SL20_T20"):
    for src in ("rt_live", "rt"):
        rows = sb.table("paper_trades").select(
            "strategy,status,source,pnl_pct,pnl_usd,chain,created_at"
        ).eq("strategy", strat).eq("source", src).eq("chain", "solana").gte("created_at", since7).execute().data or []
        closed = [r for r in rows if r.get("status") in CLOSED and not r.get("is_shadow")]
        if not closed:
            print(f"  {strat:<28} src={src:<8}  no closed trades")
            continue
        pct = [float(r.get("pnl_pct") or 0) * 100 for r in closed]
        usd = sum(float(r.get("pnl_usd") or 0) for r in closed)
        wr = 100 * sum(1 for p in pct if p > 0) / len(closed)
        print(f"  {strat:<28} src={src:<8}  N={len(closed):<3}  WR={wr:.0f}%  avg={st.mean(pct):+.2f}%  $tot={usd:+.2f}  $/d={usd/7:+.2f}")
