"""ETH live round-trip smoke test — BUY then SELL on Uniswap V3 with full
instrumentation. Uses live_trader_eth.execute_buy/execute_sell (the actual
prod code path). Dumps every metric to data/eth_smoke_<ts>.json so we don't
need to repeat the test.

Usage:
    python scripts/_eth_round_trip_smoke.py --token 0x... --eth-amount 0.003
    python scripts/_eth_round_trip_smoke.py --token 0x... --eth-amount 0.003 --hold-seconds 30

What it captures:
    pre-buy:  block, base_fee_gwei, eth_usd, wallet_eth_balance
    BUY:      tx_hash, exec_ms, gas_used, effective_gas_price_gwei, gas_usd,
              fee_tier_used, expected_tokens, received_tokens, slippage_bps,
              execution_price_usd, decimals
    pre-sell: block, base_fee_gwei, eth_usd, wallet_eth_balance, token_balance
    SELL:     tx_hash, exec_ms, gas_used, effective_gas_price_gwei, gas_usd,
              fee_tier_used, expected_eth_out, received_eth_wei,
              eth_received_human, eth_received_usd
    summary:  net_eth_change, net_usd_change, total_gas_usd, round_trip_pnl_pct,
              elapsed_total_seconds, sandbox/Flashbots mining latency
"""
import os
import sys
import time
import json
import argparse
from datetime import datetime, timezone

# Make scraper modules importable + load env
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))

try:
    from web3 import Web3
except ImportError:
    print("ERROR: web3 not installed")
    sys.exit(1)


def _snapshot(w3, account_addr, token_addr, eth_usd_fn, label: str) -> dict:
    """Capture wallet/chain state at this moment."""
    block = w3.eth.get_block("latest")
    eth_balance_wei = w3.eth.get_balance(account_addr)
    out = {
        "label": label,
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "block_number": block["number"],
        "block_ts": block["timestamp"],
        "base_fee_gwei": block["baseFeePerGas"] / 1e9,
        "eth_usd": eth_usd_fn(),
        "wallet_eth_balance": float(w3.from_wei(eth_balance_wei, "ether")),
        "wallet_eth_balance_wei": int(eth_balance_wei),
    }
    if token_addr:
        ERC20_ABI = [
            {"inputs":[{"name":"who","type":"address"}],"name":"balanceOf","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"},
            {"inputs":[],"name":"decimals","outputs":[{"type":"uint8"}],"stateMutability":"view","type":"function"},
        ]
        try:
            erc20 = w3.eth.contract(
                address=Web3.to_checksum_address(token_addr), abi=ERC20_ABI
            )
            tok_bal_raw = erc20.functions.balanceOf(account_addr).call()
            decimals = erc20.functions.decimals().call()
            out["token_balance_raw"] = int(tok_bal_raw)
            out["token_balance_human"] = tok_bal_raw / (10 ** decimals)
            out["token_decimals"] = decimals
        except Exception as e:
            out["token_balance_error"] = str(e)
    return out


def _enrich_receipt(w3, tx_hash: str, eth_usd: float) -> dict:
    """Pull full receipt + tx for a given hash and compute gas metrics."""
    receipt = w3.eth.get_transaction_receipt(tx_hash)
    tx = w3.eth.get_transaction(tx_hash)
    gas_used = int(receipt["gasUsed"])
    egp_wei = int(receipt["effectiveGasPrice"])
    gas_eth = (gas_used * egp_wei) / 1e18
    return {
        "tx_hash": tx_hash if isinstance(tx_hash, str) else "0x" + tx_hash.hex(),
        "block_number": int(receipt["blockNumber"]),
        "status": int(receipt["status"]),
        "gas_used": gas_used,
        "gas_limit": int(tx.get("gas") or 0),
        "effective_gas_price_wei": egp_wei,
        "effective_gas_price_gwei": egp_wei / 1e9,
        "gas_eth": gas_eth,
        "gas_usd": gas_eth * eth_usd,
        "from": tx["from"],
        "to": tx["to"],
        "logs_count": len(receipt["logs"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", required=True, help="Token address (0x...)")
    ap.add_argument("--eth-amount", type=float, required=True, help="ETH to buy with")
    ap.add_argument("--hold-seconds", type=int, default=15,
                    help="Seconds to hold between BUY and SELL (default 15)")
    ap.add_argument("--buy-slip-bps", type=int, default=300)
    ap.add_argument("--sell-slip-bps", type=int, default=500)
    ap.add_argument("--force-route", choices=("v3", "v2"), default=None,
                    help="Force buy+sell route. Default: auto (whichever quotes higher).")
    args = ap.parse_args()

    # Import the prod modules (the whole point of this test)
    from live_trader_eth import (
        execute_buy, execute_sell, _client, _eth_usd_price,
    )

    w3r, w3w, account = _client()
    addr = account.address
    eth_usd_fn = lambda: _eth_usd_price(w3r)

    record = {
        "test_name": "eth_round_trip_smoke",
        "version": "v14e.28",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "params": {
            "token": args.token,
            "eth_amount": args.eth_amount,
            "hold_seconds": args.hold_seconds,
            "buy_slip_bps": args.buy_slip_bps,
            "sell_slip_bps": args.sell_slip_bps,
        },
        "wallet": addr,
    }

    eth_usd_initial = eth_usd_fn()
    amount_usd = args.eth_amount * eth_usd_initial
    record["amount_usd"] = amount_usd
    print(f"=== ETH round-trip smoke test ===")
    print(f"Wallet:    {addr}")
    print(f"Token:     {args.token}")
    print(f"Amount:    {args.eth_amount} ETH = ${amount_usd:.2f} (ETH @ ${eth_usd_initial:.2f})")
    print(f"Hold:      {args.hold_seconds}s")
    print(f"Route:     {args.force_route or 'auto (best quote)'}")
    print()

    # ─── PRE-BUY snapshot ──
    pre_buy = _snapshot(w3r, addr, args.token, eth_usd_fn, "pre_buy")
    record["pre_buy"] = pre_buy
    print(f"[pre_buy] block={pre_buy['block_number']} base_fee={pre_buy['base_fee_gwei']:.2f}gwei "
          f"eth_bal={pre_buy['wallet_eth_balance']:.6f}")

    if pre_buy["wallet_eth_balance"] < args.eth_amount + 0.005:
        msg = f"insufficient ETH: have {pre_buy['wallet_eth_balance']:.6f}, need {args.eth_amount + 0.005:.4f}"
        print(f"ABORT: {msg}")
        record["aborted"] = msg
        _save(record)
        sys.exit(1)

    # ─── BUY ──
    print()
    print(f"[BUY] submitting Uniswap V3 swap via Flashbots Protect...")
    t_buy_start = time.time()
    buy_result = execute_buy(args.token, amount_usd, slippage_bps=args.buy_slip_bps,
                              force_route=args.force_route)
    t_buy_end = time.time()
    record["buy_call_elapsed_seconds"] = t_buy_end - t_buy_start
    record["buy_result"] = dict(buy_result)
    # Convert non-JSON-safe values (bytes hexes are already strings here)
    if "tokens_received" in record["buy_result"]:
        record["buy_result"]["tokens_received"] = int(buy_result["tokens_received"])
    if "eth_spent_wei" in record["buy_result"]:
        record["buy_result"]["eth_spent_wei"] = int(buy_result["eth_spent_wei"])

    if not buy_result.get("success"):
        print(f"[BUY] FAILED: {buy_result.get('error')}")
        record["aborted"] = "buy_failed"
        _save(record)
        sys.exit(1)

    print(f"[BUY] OK | tx={buy_result['tx_hash'][:14]}... | "
          f"received={buy_result['tokens_received_human']:,.0f} tokens | "
          f"gas=${buy_result['gas_usd']:.2f} | "
          f"slip={buy_result['slippage_actual_bps']}bps | "
          f"fee_tier={buy_result['fee_tier_used']} | "
          f"exec_ms={buy_result['exec_ms']}")

    # Enrich BUY with raw receipt data
    try:
        record["buy_receipt"] = _enrich_receipt(w3r, buy_result["tx_hash"],
                                                  buy_result.get("eth_usd") or eth_usd_initial)
    except Exception as e:
        record["buy_receipt_error"] = str(e)

    # ─── POST-BUY snapshot ──
    post_buy = _snapshot(w3r, addr, args.token, eth_usd_fn, "post_buy")
    record["post_buy"] = post_buy
    print(f"[post_buy] block={post_buy['block_number']} eth_bal={post_buy['wallet_eth_balance']:.6f} "
          f"token_bal={post_buy.get('token_balance_human', 0):,.0f}")

    # ─── HOLD ──
    print()
    print(f"[HOLD] sleeping {args.hold_seconds}s...")
    time.sleep(args.hold_seconds)

    # ─── PRE-SELL snapshot ──
    pre_sell = _snapshot(w3r, addr, args.token, eth_usd_fn, "pre_sell")
    record["pre_sell"] = pre_sell
    print(f"[pre_sell] block={pre_sell['block_number']} base_fee={pre_sell['base_fee_gwei']:.2f}gwei "
          f"eth_bal={pre_sell['wallet_eth_balance']:.6f} "
          f"token_bal={pre_sell.get('token_balance_human', 0):,.0f}")

    # ─── SELL (full balance) ──
    print()
    print(f"[SELL] submitting Uniswap V3 sell via Flashbots Protect...")
    t_sell_start = time.time()
    sell_result = execute_sell(args.token, amount_tokens=None, slippage_bps=args.sell_slip_bps,
                                force_route=args.force_route)
    t_sell_end = time.time()
    record["sell_call_elapsed_seconds"] = t_sell_end - t_sell_start
    record["sell_result"] = dict(sell_result)
    if "eth_received" in record["sell_result"]:
        record["sell_result"]["eth_received"] = int(sell_result["eth_received"])

    if not sell_result.get("success"):
        print(f"[SELL] FAILED: {sell_result.get('error')}")
        record["sell_aborted"] = sell_result.get("error")
        _save(record)
        sys.exit(1)

    print(f"[SELL] OK | tx={sell_result['tx_hash'][:14]}... | "
          f"received={sell_result['eth_received_human']:.6f} ETH (${sell_result['eth_received_usd']:.2f}) | "
          f"gas=${sell_result['gas_usd']:.2f} | "
          f"fee_tier={sell_result['fee_tier_used']} | "
          f"exec_ms={sell_result['exec_ms']}")

    try:
        record["sell_receipt"] = _enrich_receipt(w3r, sell_result["tx_hash"],
                                                  sell_result.get("eth_usd") or eth_usd_initial)
    except Exception as e:
        record["sell_receipt_error"] = str(e)

    # ─── POST-SELL snapshot ──
    post_sell = _snapshot(w3r, addr, args.token, eth_usd_fn, "post_sell")
    record["post_sell"] = post_sell
    print(f"[post_sell] block={post_sell['block_number']} eth_bal={post_sell['wallet_eth_balance']:.6f} "
          f"token_bal={post_sell.get('token_balance_human', 0):,.0f}")

    # ─── SUMMARY ──
    eth_spent = pre_buy["wallet_eth_balance"] - post_buy["wallet_eth_balance"]
    eth_recovered = post_sell["wallet_eth_balance"] - pre_sell["wallet_eth_balance"]
    net_eth = post_sell["wallet_eth_balance"] - pre_buy["wallet_eth_balance"]
    eth_usd_final = post_sell["eth_usd"]
    net_usd = net_eth * eth_usd_final
    total_gas_usd = (record.get("buy_receipt", {}).get("gas_usd", 0)
                     + record.get("sell_receipt", {}).get("gas_usd", 0))
    pnl_pct = (net_eth / args.eth_amount) * 100 if args.eth_amount else 0

    summary = {
        "eth_spent_buy": eth_spent,
        "eth_recovered_sell": eth_recovered,
        "net_eth": net_eth,
        "net_usd_at_final_eth_price": net_usd,
        "total_gas_usd": total_gas_usd,
        "round_trip_pnl_pct": pnl_pct,
        "buy_block": record.get("buy_receipt", {}).get("block_number"),
        "sell_block": record.get("sell_receipt", {}).get("block_number"),
        "blocks_between": (
            record.get("sell_receipt", {}).get("block_number", 0)
            - record.get("buy_receipt", {}).get("block_number", 0)
        ),
        "elapsed_total_seconds": (
            datetime.fromisoformat(post_sell["ts_utc"]).timestamp()
            - datetime.fromisoformat(pre_buy["ts_utc"]).timestamp()
        ),
    }
    record["summary"] = summary

    print()
    print("=== ROUND-TRIP SUMMARY ===")
    print(f"  ETH spent (BUY net):    {eth_spent:.6f} ETH")
    print(f"  ETH recovered (SELL):   {eth_recovered:.6f} ETH")
    print(f"  Net ETH change:         {net_eth:+.6f} ETH = ${net_usd:+.2f}")
    print(f"  Total gas paid:         ${total_gas_usd:.2f}")
    print(f"  Round-trip PnL:         {pnl_pct:+.2f}%")
    print(f"  Blocks between:         {summary['blocks_between']}")
    print(f"  Elapsed total:          {summary['elapsed_total_seconds']:.1f}s")
    print()

    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    out_path = _save(record)
    print(f"Full record saved to: {out_path}")


def _save(record: dict) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"eth_smoke_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)
    return out_path


if __name__ == "__main__":
    main()
