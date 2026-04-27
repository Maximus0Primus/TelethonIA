"""Quick view: 5 most recent ETH live trades + key fields."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
r = (sb.table("paper_trades")
     .select("symbol,strategy,status,entry_price,tp_price,sl_price,position_usd,"
             "pnl_pct,pnl_usd,tx_signature,created_at,exit_at,buy_slippage_bps,"
             "gas_usd_buy,quote_slip_bps_buy")
     .eq("source", "rt_live").eq("chain", "ethereum")
     .order("created_at", desc=True).limit(5).execute())

print(f"=== last {len(r.data)} ETH live trades ===\n")
for t in r.data:
    sym = t.get("symbol") or "?"
    strat = t.get("strategy") or "?"
    st = t.get("status") or "?"
    ep = t.get("entry_price") or 0
    tp = t.get("tp_price") or 0
    sl = t.get("sl_price") or 0
    pos = t.get("position_usd") or 0
    pnl_usd = t.get("pnl_usd") or 0
    pnl_pct = (t.get("pnl_pct") or 0) * 100
    gas = t.get("gas_usd_buy") or 0
    qslip = t.get("quote_slip_bps_buy") or 0
    bslip = t.get("buy_slippage_bps") or 0
    tx = (t.get("tx_signature") or "?")[:14]
    print(f"  {sym:<14} {strat:<24} {st:<8}")
    print(f"    entry=${ep:.10f}  tp=${tp:.10f}  sl=${sl:.10f}")
    print(f"    pos=${pos:.2f}  pnl=${pnl_usd:.2f} ({pnl_pct:+.1f}%)")
    print(f"    gas_buy=${gas:.2f}  quote_slip_buy={qslip}bps  ds_slip_buy={bslip}bps")
    print(f"    tx={tx}  created={t.get('created_at')}  exit={t.get('exit_at') or 'still open'}")
    print()
