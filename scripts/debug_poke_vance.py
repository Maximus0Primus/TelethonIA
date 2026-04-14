"""Debug POKE and VANCE paper vs live divergence post-v123."""
import os
from supabase import create_client

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

for sym in ["POKE", "VANCE"]:
    print(f"\n{'='*80}")
    print(f"  ${sym}")
    print(f"{'='*80}")

    res = (sb.table("paper_trades")
        .select("id,symbol,strategy,status,source,is_shadow,pnl_pct,pnl_usd,"
                "entry_price,exit_price,execution_price,high_price_seen,"
                "position_usd,position_sol,tx_signature,"
                "buy_slippage_bps,sell_slippage_bps,"
                "sol_price_at_entry,sol_price_at_exit,dex_spot_price_at_entry,"
                "exit_minutes,created_at,exit_at,message_ts,paper_exit_price,price_divergence_pct")
        .eq("strategy", "DTRAIL10_ACT15_SL70")
        .ilike("symbol", f"%{sym}%")
        .gte("created_at", "2026-04-10T15:00:00Z")
        .order("created_at", desc=True)
        .execute())

    live = [t for t in res.data if t.get("tx_signature")]
    paper = [t for t in res.data if not t.get("tx_signature") and not t.get("is_shadow")]

    for label, trades in [("LIVE", live), ("PAPER (non-shadow)", paper)]:
        print(f"\n  --- {label} ({len(trades)} trades) ---")
        for t in trades:
            pnl_pct = float(t["pnl_pct"] or 0)
            pnl_usd = float(t["pnl_usd"] or 0)
            entry = t["entry_price"]
            exec_p = t["execution_price"]
            exit_p = t["exit_price"]
            high = t["high_price_seen"]
            paper_exit = t.get("paper_exit_price")
            div_pct = t.get("price_divergence_pct")
            pos_usd = t["position_usd"]
            pos_sol = t["position_sol"]
            status = t["status"]
            source = t["source"]
            mins = t["exit_minutes"]
            created = (t["created_at"] or "")[:19]
            exit_at = (t["exit_at"] or "")[:19]
            buy_slip = t["buy_slippage_bps"]
            sell_slip = t["sell_slippage_bps"]
            sol_entry = t["sol_price_at_entry"]
            sol_exit = t["sol_price_at_exit"]
            dex_spot = t["dex_spot_price_at_entry"]

            print(f"  {status:10s} | pnl={pnl_pct*100:+.2f}% ${pnl_usd:+.2f} | source={source}")
            print(f"    entry={entry}  exec={exec_p}  dex_spot={dex_spot}")
            print(f"    exit={exit_p}  paper_exit={paper_exit}  div={div_pct}")
            print(f"    high_seen={high}")
            print(f"    pos=${pos_usd} ({pos_sol} SOL)  sol@entry={sol_entry}  sol@exit={sol_exit}")
            print(f"    buy_slip={buy_slip}bps  sell_slip={sell_slip}bps")
            print(f"    created={created}  exit_at={exit_at}  duration={mins}min")
