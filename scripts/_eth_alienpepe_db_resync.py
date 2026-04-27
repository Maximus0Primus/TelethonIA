"""One-shot: resync paper_trades row 251132 ($ALIENPEPE) with the on-chain sell.

Cause: live_trader_eth.process_open_trades atomically claimed the row
(status open→closing), submitted the Uniswap V2 sell at ~22:18 UTC which mined
in block 24974330 (2026-04-27 22:20:11 UTC, tx 0x0bb165f1...), but the DB update
that follows the sell never landed. Most likely: the VPS process / RPC call
chain crashed between sell mine and the supabase .update(). The retry path in
live_trader_eth lines 1022-1026 keeps re-issuing the sell on every cycle but
the wallet now holds 0 tokens, so each retry call routes to a no-op (or fails
silently), and the row stays pinned in 'closing' forever.

Verified on-chain via publicnode.com eth_getLogs:
- Buy:  block 24973693, tx 0x73675805..., wallet received 106254.88 ALIENPEPE
- Sell: block 24974330, tx 0x0bb165f1..., wallet sent 106254.88 → pool, received 0.0152380707 ETH (15238070748110586 wei)
- Wallet ALIENPEPE balance now: 0
- Chainlink ETH/USD at block 24974330: $2286.6825
- Gas used: 134936 @ 2.688 gwei = 0.00036273 ETH = $0.83

Computed PnL (matches live_trader_eth.py:1140-1186 formula):
- usd_received = 0.0152380707 × 2286.6825 ≈ $34.84
- actual_exit_price = entry_price × (usd_received / position_usd)
- pnl_pct = (usd_received / position_usd) - 1 ≈ +74.21%
- pnl_usd = position_usd × pnl_pct ≈ +$14.84

Status: 'timeout' (high_price_seen 0.0003328 < tp_price 0.000338807 — never hit
TP at +80%; trade ran out the 120min horizon at 22:12 UTC, sell submitted ~22:18,
mined 22:20:11 → exit_minutes ≈ 128).

Idempotent: aborts if the row is no longer in 'closing' or symbol mismatches.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client

TRADE_ID = 251132
TX_EXIT  = "0x0bb165f1515cb6ac4fdd4ae2652df98397300b5ff4f9812b84cbf03d75dcf175"
BLOCK    = 24974330
EXIT_AT  = "2026-04-27T22:20:11+00:00"
ETH_RX_WEI = 15238070748110586          # 0.015238070748110586 ETH
ETH_USD    = 2286.6825                  # Chainlink at block 24974330
GAS_USD    = 0.83                       # 134936 × 2.688 gwei × $2286.68
NEW_STATUS = "timeout"                  # high never hit TP, horizon exceeded

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

row = sb.table("paper_trades").select("*").eq("id", TRADE_ID).execute().data
if not row:
    print(f"trade {TRADE_ID} not found")
    sys.exit(1)
r = row[0]
print(f"BEFORE: status={r['status']} symbol={r['symbol']} exit_price={r.get('exit_price')} pnl_usd={r.get('pnl_usd')}")

if r["symbol"] != "$ALIENPEPE":
    print(f"safety: trade {TRADE_ID} is {r['symbol']} not $ALIENPEPE, abort")
    sys.exit(1)
if r["status"] != "closing":
    print(f"safety: trade is '{r['status']}' not 'closing' — already resolved, abort")
    sys.exit(0)
if r.get("tx_signature_exit"):
    print(f"safety: tx_signature_exit already set ({r['tx_signature_exit']}) — abort")
    sys.exit(0)

entry_price = float(r["entry_price"])
pos_usd     = float(r["position_usd"])
buy_slip    = int(r.get("buy_slippage_bps") or 0)

eth_received = ETH_RX_WEI / 1e18
usd_received = eth_received * ETH_USD
actual_exit_price = entry_price * (usd_received / pos_usd)
pnl_pct = round((actual_exit_price / entry_price) - 1, 4)
pnl_usd = round(pos_usd * pnl_pct, 2)

# elapsed_minutes from cycle_ts → exit_at
from datetime import datetime
ct = datetime.fromisoformat(r["cycle_ts"].replace("Z", "+00:00"))
ex = datetime.fromisoformat(EXIT_AT)
elapsed_minutes = int((ex - ct).total_seconds() / 60)

update = {
    "status": NEW_STATUS,
    "exit_price": actual_exit_price,
    "exit_at": EXIT_AT,
    "pnl_pct": pnl_pct,
    "pnl_usd": pnl_usd,
    "exit_minutes": elapsed_minutes,
    "tx_signature_exit": TX_EXIT,
    "sell_slippage_bps": 0,                    # no quote vs spot data — leave neutral
    "slippage_actual_bps": buy_slip,           # buy slip only (sell unmeasured)
    "sol_price_at_exit": ETH_USD,              # ETH/USD repurposed in this column
    "sell_output_lamports": ETH_RX_WEI,        # wei in this column
    "sell_sol_received": round(eth_received, 6),
    "gas_usd_sell": GAS_USD,
    "block_number_sell": BLOCK,
}

print("\nCOMPUTED:")
for k, v in update.items():
    print(f"  {k}: {v}")

# Strip optional v14e.32+ cols if migration not applied (defensive, mirror live_trader_eth.py:1191-1213)
_OPTIONAL = ("gas_usd_sell", "block_number_sell")
for attempt in range(3):
    try:
        sb.table("paper_trades").update(update).eq("id", TRADE_ID).execute()
        break
    except Exception as e:
        s = str(e)
        stripped = False
        for col in _OPTIONAL:
            if col in s and col in update:
                update.pop(col, None)
                stripped = True
        if not stripped:
            raise

# Update rt_bankroll on the ethereum side (mirror live_trader_eth flow)
from safe_scraper import _rt_update_bankroll
_rt_update_bankroll(pnl_usd, 1, strategy=r["strategy"], chain="ethereum")

after = sb.table("paper_trades").select(
    "status,exit_price,pnl_pct,pnl_usd,exit_at,tx_signature_exit"
).eq("id", TRADE_ID).execute().data[0]
print("\nAFTER:")
for k, v in after.items():
    print(f"  {k}: {v}")
print(f"\n$ALIENPEPE resynced. PnL +{pnl_pct*100:.2f}% (+${pnl_usd}) bankroll updated.")
