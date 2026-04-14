"""Debug paper vs live PnL divergence on same tokens."""
import os
from supabase import create_client

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

# Get ALL DTRAIL10 closed trades since Apr 8 with full details
res = (sb.table("paper_trades")
    .select("id,symbol,strategy,status,source,is_shadow,pnl_pct,pnl_usd,"
            "entry_price,exit_price,execution_price,high_price_seen,"
            "position_usd,position_sol,tx_signature,"
            "buy_slippage_bps,sell_slippage_bps,buy_exec_ms,sell_exec_ms,"
            "sol_price_at_entry,sol_price_at_exit,dex_spot_price_at_entry,"
            "exit_minutes,created_at,exit_at,message_ts")
    .eq("strategy", "DTRAIL10_ACT15_SL70")
    .eq("is_shadow", False)
    .gte("created_at", "2026-04-08T00:00:00Z")
    .in_("status", ["sl_hit", "tp_hit", "timeout", "trail_stop"])
    .order("created_at", desc=True)
    .limit(200)
    .execute())

# Group by (symbol, created_at truncated to minute) to find pairs
from collections import defaultdict
groups = defaultdict(list)
for t in res.data:
    # Group by symbol + minute of entry
    key = (t["symbol"], (t["created_at"] or "")[:16])
    groups[key].append(t)

# Find groups with both live (tx_signature) and paper-only
print("=" * 100)
print("DIVERGENT PAIRS: Same token, same entry time, different PnL%")
print("=" * 100)

divergent_count = 0
for key, trades in sorted(groups.items(), key=lambda x: x[0][1], reverse=True):
    live = [t for t in trades if t.get("tx_signature")]
    paper = [t for t in trades if not t.get("tx_signature")]

    if not live or not paper:
        continue

    l = live[0]
    p = paper[0]
    l_pnl = float(l["pnl_pct"] or 0)
    p_pnl = float(p["pnl_pct"] or 0)
    delta = abs(l_pnl - p_pnl)

    sym = key[0]

    print(f"\n{'='*80}")
    print(f"  {sym} @ {key[1]}")
    print(f"{'='*80}")

    for label, t in [("LIVE ", l), ("PAPER", p)]:
        pnl_pct = float(t["pnl_pct"] or 0)
        entry = t["entry_price"]
        ex_price = t["execution_price"]
        exit_p = t["exit_price"]
        high = t["high_price_seen"]
        pos_usd = t["position_usd"]
        pos_sol = t["position_sol"]
        buy_slip = t["buy_slippage_bps"]
        sell_slip = t["sell_slippage_bps"]
        sol_entry = t["sol_price_at_entry"]
        sol_exit = t["sol_price_at_exit"]
        dex_spot = t["dex_spot_price_at_entry"]
        mins = t["exit_minutes"]
        status = t["status"]
        source = t["source"]
        exit_at = (t["exit_at"] or "")[:19]
        msg_ts = (t["message_ts"] or "")[:19]

        print(f"  {label} | {status:10s} | pnl={pnl_pct:+.2f}% | ${float(t['pnl_usd'] or 0):+.2f}")
        print(f"         entry_price={entry}  execution_price={ex_price}  dex_spot_at_entry={dex_spot}")
        print(f"         exit_price={exit_p}  high_seen={high}")
        print(f"         pos=${pos_usd} ({pos_sol} SOL)  sol@entry={sol_entry}  sol@exit={sol_exit}")
        print(f"         buy_slip={buy_slip}bps  sell_slip={sell_slip}bps")
        print(f"         exit_at={exit_at}  duration={mins}min  source={source}  msg_ts={msg_ts}")

    if delta > 0.01:
        divergent_count += 1
        print(f"  >>> DELTA: {delta:.2f}% <<<")
    else:
        print(f"  (matched)")

print(f"\n\nTotal pairs: {sum(1 for k,v in groups.items() if any(t.get('tx_signature') for t in v) and any(not t.get('tx_signature') for t in v))}")
print(f"Divergent (>0.01% delta): {divergent_count}")
