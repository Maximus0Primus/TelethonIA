import os
from supabase import create_client

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

# Recent closed trades for main 3 strategies
res = (sb.table("paper_trades")
    .select("symbol,strategy,source,is_shadow,status,pnl_pct,pnl_usd,tx_signature,created_at")
    .in_("strategy", ["DTRAIL10_ACT15_SL70", "DTRAIL3_ACT5_SL60", "DIP30_B5_T5_A20_SL70_240m"])
    .eq("is_shadow", False)
    .gte("created_at", "2026-04-10T15:45:00Z")
    .neq("status", "open")
    .order("created_at", desc=True)
    .limit(20)
    .execute())

print(f"{'Symbol':12s} | {'Strategy':25s} | {'Source':8s} | {'Status':10s} | {'PnL%':>8s} | {'Live?':5s} | Created")
print("-" * 110)
for r in res.data:
    pnl = float(r["pnl_pct"] or 0)
    is_live = "YES" if r.get("tx_signature") else "no"
    sym = r["symbol"]
    strat = r["strategy"]
    src = r["source"]
    st = r["status"]
    dt = (r["created_at"] or "")[:19]
    print(f"{sym:12s} | {strat:25s} | {src:8s} | {st:10s} | {pnl*100:+7.1f}% | {is_live:5s} | {dt}")

# Also check: what triggers TG alert?
# Alert is sent when source == "rt" and is_shadow == False
rt_trades = [r for r in res.data if r["source"] == "rt"]
live_trades = [r for r in res.data if r.get("tx_signature")]
print(f"\nRT paper trades (get TG alert): {len(rt_trades)}")
print(f"Live trades (get TG alert): {len(live_trades)}")
print(f"Batch trades (no TG alert): {len(res.data) - len(rt_trades) - len(live_trades)}")
