"""Check paper vs live alignment for trades opened AFTER v123 fix2 deploy (~16:07)."""
import os
from supabase import create_client

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

# All DTRAIL10 trades opened after 16:07 (fix2 deploy)
res = (sb.table("paper_trades")
    .select("symbol,strategy,source,is_shadow,status,pnl_pct,pnl_usd,"
            "entry_price,exit_price,execution_price,high_price_seen,"
            "position_usd,position_sol,tx_signature,"
            "exit_minutes,created_at,exit_at,token_address")
    .eq("strategy", "DTRAIL10_ACT15_SL70")
    .eq("is_shadow", False)
    .gte("created_at", "2026-04-10T16:07:00Z")
    .order("created_at", desc=True)
    .limit(30)
    .execute())

# Group by token_address
from collections import defaultdict
groups = defaultdict(list)
for t in res.data:
    groups[t["token_address"]].append(t)

for addr, trades in groups.items():
    live = [t for t in trades if t.get("tx_signature")]
    paper = [t for t in trades if not t.get("tx_signature")]

    if not live and not paper:
        continue

    sym = trades[0]["symbol"]
    print(f"\n{'='*80}")
    print(f"  ${sym} ({addr[:12]}...)")
    print(f"{'='*80}")

    for label, tlist in [("LIVE", live), ("PAPER", paper)]:
        for t in tlist:
            pnl = float(t["pnl_pct"] or 0)
            entry = t["entry_price"]
            exit_p = t["exit_price"]
            high = t["high_price_seen"]
            pos = t["position_usd"]
            mins = t["exit_minutes"]
            status = t["status"]
            source = t["source"]
            created = (t["created_at"] or "")[:19]
            exit_at = (t["exit_at"] or "")[:19]
            exec_p = t.get("execution_price")

            print(f"  {label:5s} | {status:10s} | pnl={pnl*100:+.2f}% | src={source}")
            print(f"        entry={entry}  exec={exec_p}")
            print(f"        exit={exit_p}  high={high}")
            print(f"        pos=${pos}  {mins}min | {created} -> {exit_at}")

    # Compare if both exist
    if live and paper:
        l = live[0]
        p = paper[0]
        l_entry = float(l["entry_price"] or 0)
        p_entry = float(p["entry_price"] or 0)
        l_pnl = float(l["pnl_pct"] or 0)
        p_pnl = float(p["pnl_pct"] or 0)
        entry_diff = abs(l_entry - p_entry) / l_entry * 100 if l_entry else 0
        print(f"\n  >>> ENTRY DIFF: {entry_diff:.2f}%  |  PNL DIFF: {abs(l_pnl - p_pnl)*100:.2f}pp <<<")

# Also check ALL strategies for post-fix trades
print("\n\n" + "=" * 80)
print("ALL 3 STRATEGIES — trades opened after 16:07")
print("=" * 80)
res2 = (sb.table("paper_trades")
    .select("symbol,strategy,source,is_shadow,status,pnl_pct,entry_price,exit_price,tx_signature,created_at,exit_at,exit_minutes")
    .in_("strategy", ["DTRAIL10_ACT15_SL70", "DTRAIL3_ACT5_SL60", "DIP30_B5_T5_A20_SL70_240m"])
    .eq("is_shadow", False)
    .gte("created_at", "2026-04-10T16:07:00Z")
    .order("created_at", desc=True)
    .limit(30)
    .execute())

for t in res2.data:
    pnl = float(t["pnl_pct"] or 0)
    is_live = "LIVE" if t.get("tx_signature") else "paper"
    sym = t["symbol"]
    strat = t["strategy"]
    src = t["source"]
    status = t["status"]
    entry = t["entry_price"]
    exit_p = t["exit_price"]
    mins = t["exit_minutes"]
    created = (t["created_at"] or "")[:19]
    print(f"  {sym:12s} | {strat:25s} | {is_live:5s} | {status:10s} | pnl={pnl*100:+.1f}% | entry={entry} | {mins}min | {created}")
