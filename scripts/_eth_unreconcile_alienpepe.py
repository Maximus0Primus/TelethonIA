"""One-shot: revert the wrongful reconcile-close of ALIENPEPE.

Cause: live_trader.reconcile_positions queries the Solana wallet for all rt_live
open trades, including the ETH-chain ones. ETH addresses obviously don't show
up in Solana wallet balances, so the function auto-closed the live ETH position
as "0 on-chain balance" with pnl=0. We confirmed on-chain that the wallet does
hold 106254.88 ALIENPEPE tokens (decimals=9) — the position is real.

Fix in code: live_trader.py reconcile_positions now filters chain=solana
(commit v14e.34). This script restores the DB state to 'open' so check_live_trades_eth
can monitor TP/SL/timeout.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

TRADE_ID = 251132  # ALIENPEPE

before = sb.table("paper_trades").select(
    "id,symbol,status,exit_at,exit_price,pnl_pct,pnl_usd"
).eq("id", TRADE_ID).execute().data
print("BEFORE:", before)

if not before:
    print("trade not found, abort")
    sys.exit(1)
row = before[0]
if row["symbol"] != "$ALIENPEPE":
    print(f"safety: trade {TRADE_ID} is {row['symbol']} not $ALIENPEPE, abort")
    sys.exit(1)
if row["status"] == "open":
    print("already open, nothing to do")
    sys.exit(0)

upd = {
    "status": "open",
    "exit_at": None,
    "exit_price": None,
    "pnl_pct": None,
    "pnl_usd": None,
    "tx_signature_exit": None,
    "sell_slippage_bps": None,
    "slippage_actual_bps": None,
    "sol_price_at_exit": None,
    "sell_exec_ms": None,
    "sell_output_lamports": None,
    "sell_sol_received": None,
    "exit_minutes": None,
    "paper_exit_price": None,
    "price_divergence_pct": None,
    "paper_sim_pnl_pct": None,
    "gas_usd_sell": None,
    "quote_slip_bps_sell": None,
    "block_number_sell": None,
}
sb.table("paper_trades").update(upd).eq("id", TRADE_ID).execute()
print("\nAFTER:", sb.table("paper_trades").select(
    "id,symbol,status,exit_at,pnl_pct").eq("id", TRADE_ID).execute().data)
print("\nALIENPEPE re-opened. live_trader_eth will resume monitoring on next cycle.")
