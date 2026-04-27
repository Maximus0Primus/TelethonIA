"""Read-only audit of ETH live infra before Phase 1 microtest.

Checks:
  1. ETH_PRIVATE_KEY env var presence
  2. ETH wallet balance (read-only RPC call)
  3. live_trading config keys for ETH (eth_live_enabled, eth_allocations, etc.)
  4. Existing paper_trades column schema (which slip/gas/fee cols exist)
  5. Code paths wired: open + close ETH live
  6. ETH paper trades sample for last 24h — sanity that pipeline produces data
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

print("=" * 92)
print("1. ENV — ETH_PRIVATE_KEY")
print("=" * 92)
pk = os.environ.get("ETH_PRIVATE_KEY")
if not pk:
    print("  MISSING — abort, no ETH wallet configured")
else:
    print(f"  present, len={len(pk)} (first 4 chars: {pk[:4]}...)")

print()
print("=" * 92)
print("2. ETH WALLET BALANCE")
print("=" * 92)
try:
    from live_trader_eth import _client, _eth_usd_price
    w3r, w3w, acct = _client()
    bal_wei = w3r.eth.get_balance(acct.address)
    bal_eth = bal_wei / 1e18
    eth_usd = _eth_usd_price(w3r)
    bal_usd = bal_eth * eth_usd
    print(f"  address: {acct.address}")
    print(f"  balance: {bal_eth:.6f} ETH (≈ ${bal_usd:.2f} @ ${eth_usd:.0f}/ETH)")
    if bal_usd < 100:
        print(f"  WARN: balance below $100 — Phase 1 microtest needs ~$60-80 (1 trade $50 + gas + buffer)")
    if bal_usd < 50:
        print(f"  ABORT: balance below $50 — top up before going live")
except Exception as e:
    print(f"  ERROR reading wallet: {type(e).__name__}: {e}")

print()
print("=" * 92)
print("3. live_trading CONFIG — ETH-specific keys")
print("=" * 92)
cfg = sb.table("scoring_config").select("rt_trade_config").eq("id", 1).execute().data[0]["rt_trade_config"]
if isinstance(cfg, str):
    cfg = json.loads(cfg)
live_cfg = cfg.get("live_trading", {}) or {}
eth_keys = ["eth_live_enabled", "eth_allocations", "eth_max_position_usd",
            "eth_max_open_positions", "eth_buy_slippage_bps", "eth_sell_slippage_bps",
            "eth_daily_loss_limit_usd"]
for k in eth_keys:
    v = live_cfg.get(k, "<NOT SET>")
    print(f"  {k:<32} = {v}")
print(f"\n  eth_live_enabled: {'YES — live ON' if live_cfg.get('eth_live_enabled') else 'NO — gated off (safe default)'}")

print()
print("=" * 92)
print("4. paper_trades SCHEMA — which instrumentation columns exist")
print("=" * 92)
r = sb.table("paper_trades").select("*").eq("chain", "ethereum").limit(1).execute()
if not r.data:
    r = sb.table("paper_trades").select("*").limit(1).execute()
cols = sorted(r.data[0].keys()) if r.data else []
TARGETS = {
    "buy_slippage_bps": "DexScreener mid vs fill",
    "sell_slippage_bps": "DexScreener mid vs fill",
    "slippage_actual_bps": "buy+sell aggregate",
    "buy_fee_bps": "requested slip tolerance",
    "buy_exec_ms": "buy latency",
    "sell_exec_ms": "sell latency",
    "paper_sim_pnl_pct": "paired-test anchor",
    "paper_exit_price": "paired-test anchor",
    "price_divergence_pct": "exit divergence paper-vs-live",
    "gas_usd_buy": "[NEW] real gas paid on buy",
    "gas_usd_sell": "[NEW] real gas paid on sell",
    "quote_slip_bps_buy": "[NEW] Uniswap quote vs fill",
    "quote_slip_bps_sell": "[NEW] Uniswap quote vs fill",
    "block_number_buy": "[NEW] block at submit (latency)",
    "block_number_sell": "[NEW] block at submit (latency)",
}
for c, descr in TARGETS.items():
    mark = "OK    " if c in cols else "MISS  "
    print(f"  [{mark}] {c:<26}  {descr}")

print()
print("=" * 92)
print("5. CODE PATHS — open + close ETH live")
print("=" * 92)
import re
ss = open(os.path.join(os.path.dirname(__file__), "..", "scraper", "safe_scraper.py"),
          encoding="utf-8", errors="ignore").read()
checks = [
    ("ETH OPEN dispatcher (rt path)",
     'open_live_trade as open_live_trade_eth' in ss),
    ("ETH CLOSE dispatcher (live monitor)",
     'check_live_trades_eth' in ss),
    ("ETH config gate (eth_live_enabled)",
     'eth_live_enabled' in ss),
]
for label, ok in checks:
    print(f"  [{'OK  ' if ok else 'MISS'}] {label}")

print()
print("=" * 92)
print("6. ETH PAPER PIPELINE — last 24h smoke")
print("=" * 92)
from datetime import datetime, timedelta, timezone
since = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
r = sb.table("paper_trades").select("strategy,status,source,chain,created_at", count="exact") \
    .eq("chain", "ethereum").eq("source", "rt").gte("created_at", since) \
    .limit(1).execute()
print(f"  last 24h ETH paper main: N={r.count}")
if r.count == 0:
    print("  WARN: no ETH paper trades last 24h — pipeline may be stalled")
print()
