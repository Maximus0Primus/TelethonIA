"""ETH live smoke test — mesure le gas + slip RÉEL sur 1 swap on-chain.

Usage:
    # Step 1 — quote only (no tx, free)
    python scripts/_eth_live_smoke_test.py --token 0x... --eth-amount 0.005

    # Step 2 — execute real swap (irréversible, coûte du gas)
    python scripts/_eth_live_smoke_test.py --token 0x... --eth-amount 0.005 --execute

Required env vars (in scraper/.env):
    ETH_PRIVATE_KEY=0x...           # NEW dedicated wallet, NOT the main Phantom one
    ETH_RPC_URL=https://rpc.flashbots.net   # default if unset (free, MEV-protected)

Reports:
    - gas units used + price + total $ cost
    - expected output (Quoter) vs actual output (logs)
    - slippage % vs quote
    - elapsed time entry-to-mined
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv; load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))

try:
    from web3 import Web3
    from eth_account import Account
except ImportError:
    print("ERROR: web3 + eth-account not installed. Run:")
    print("    pip install web3 eth-account")
    sys.exit(1)

# Uniswap V3 mainnet contracts
SWAP_ROUTER_02 = Web3.to_checksum_address("0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45")
QUOTER_V2 = Web3.to_checksum_address("0x61fFE014bA17989E743c5F6cB21bF9697530B21e")
WETH = Web3.to_checksum_address("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2")

# Minimal ABIs (only the funcs we need)
QUOTER_ABI = [{
    "inputs": [{"components": [
        {"internalType": "address", "name": "tokenIn", "type": "address"},
        {"internalType": "address", "name": "tokenOut", "type": "address"},
        {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
        {"internalType": "uint24", "name": "fee", "type": "uint24"},
        {"internalType": "uint160", "name": "sqrtPriceLimitX96", "type": "uint160"},
    ], "internalType": "struct IQuoterV2.QuoteExactInputSingleParams", "name": "params", "type": "tuple"}],
    "name": "quoteExactInputSingle",
    "outputs": [
        {"internalType": "uint256", "name": "amountOut", "type": "uint256"},
        {"internalType": "uint160", "name": "sqrtPriceX96After", "type": "uint160"},
        {"internalType": "uint32", "name": "initializedTicksCrossed", "type": "uint32"},
        {"internalType": "uint256", "name": "gasEstimate", "type": "uint256"},
    ],
    "stateMutability": "nonpayable", "type": "function"
}]

ROUTER_ABI = [{
    "inputs": [{"components": [
        {"internalType": "address", "name": "tokenIn", "type": "address"},
        {"internalType": "address", "name": "tokenOut", "type": "address"},
        {"internalType": "uint24", "name": "fee", "type": "uint24"},
        {"internalType": "address", "name": "recipient", "type": "address"},
        {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
        {"internalType": "uint256", "name": "amountOutMinimum", "type": "uint256"},
        {"internalType": "uint160", "name": "sqrtPriceLimitX96", "type": "uint160"},
    ], "internalType": "struct ISwapRouter.ExactInputSingleParams", "name": "params", "type": "tuple"}],
    "name": "exactInputSingle",
    "outputs": [{"internalType": "uint256", "name": "amountOut", "type": "uint256"}],
    "stateMutability": "payable", "type": "function"
}]

ERC20_ABI = [
    {"inputs":[{"name":"who","type":"address"}],"name":"balanceOf","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"decimals","outputs":[{"type":"uint8"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"symbol","outputs":[{"type":"string"}],"stateMutability":"view","type":"function"},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", required=True, help="Token address to buy (0x...)")
    ap.add_argument("--eth-amount", type=float, required=True, help="ETH amount to swap (e.g. 0.005)")
    ap.add_argument("--fee-tier", type=int, default=3000, choices=[100, 500, 3000, 10000],
                    help="Uniswap V3 fee tier (default 3000=0.3%%)")
    ap.add_argument("--slippage-bps", type=int, default=200, help="Min-out slippage tolerance, bps (default 200=2%%)")
    ap.add_argument("--execute", action="store_true", help="ACTUALLY send the tx (default: quote-only dry run)")
    args = ap.parse_args()

    pk = os.environ.get("ETH_PRIVATE_KEY")
    if not pk:
        print("ERROR: ETH_PRIVATE_KEY missing from scraper/.env")
        sys.exit(1)
    # Flashbots Protect (rpc.flashbots.net) is for tx submission only — reads return 403.
    # Use ETH_READ_RPC_URL for quotes/balance, ETH_RPC_URL for signed tx submission.
    read_rpc = os.environ.get("ETH_READ_RPC_URL", "https://ethereum-rpc.publicnode.com")
    write_rpc = os.environ.get("ETH_RPC_URL", "https://rpc.flashbots.net")

    w3 = Web3(Web3.HTTPProvider(read_rpc))
    if not w3.is_connected():
        print(f"ERROR: cannot connect to read RPC {read_rpc}")
        sys.exit(1)
    print(f"Read RPC:  {read_rpc}")
    print(f"Write RPC: {write_rpc}")
    print(f"Block: {w3.eth.block_number} | Chain ID: {w3.eth.chain_id}")

    acct = Account.from_key(pk)
    print(f"Wallet: {acct.address}")

    eth_balance = w3.from_wei(w3.eth.get_balance(acct.address), "ether")
    print(f"ETH balance: {eth_balance:.6f}")
    if args.execute and eth_balance < args.eth_amount + 0.005:
        print(f"ERROR: insufficient ETH. Need at least {args.eth_amount + 0.005:.4f} ETH (swap + gas buffer)")
        sys.exit(1)

    token = Web3.to_checksum_address(args.token)
    erc20 = w3.eth.contract(address=token, abi=ERC20_ABI)
    try:
        symbol = erc20.functions.symbol().call()
        decimals = erc20.functions.decimals().call()
    except Exception as e:
        print(f"WARNING: could not read token metadata ({e}), assuming decimals=18")
        symbol = "TOKEN"
        decimals = 18
    print(f"Token: {symbol} ({token}) decimals={decimals}")

    # Step 1 — Quote
    amount_in_wei = w3.to_wei(args.eth_amount, "ether")
    quoter = w3.eth.contract(address=QUOTER_V2, abi=QUOTER_ABI)
    print(f"\n=== QUOTE (no tx) — fee tier {args.fee_tier/10000:.2f}% ===")
    try:
        result = quoter.functions.quoteExactInputSingle({
            "tokenIn": WETH, "tokenOut": token,
            "amountIn": amount_in_wei, "fee": args.fee_tier,
            "sqrtPriceLimitX96": 0,
        }).call()
        amount_out = result[0]
        gas_estimate = result[3]
        amount_out_human = amount_out / (10 ** decimals)
        eth_per_token = args.eth_amount / amount_out_human if amount_out_human else 0
        print(f"  Expected output: {amount_out_human:,.0f} {symbol}")
        print(f"  Implied price:   {eth_per_token:.10f} ETH/{symbol}")
        print(f"  Quoter gas est:  {gas_estimate:,} units")
    except Exception as e:
        print(f"  ERROR Quoter: {e}")
        print(f"  Token may not have a {args.fee_tier/10000:.2f}% pool. Try --fee-tier 500 or 10000.")
        sys.exit(1)

    # Gas price snapshot
    base_fee = w3.eth.get_block("latest")["baseFeePerGas"]
    priority = w3.to_wei(2, "gwei")
    max_fee = base_fee * 2 + priority
    gas_total_wei = (gas_estimate + 50_000) * max_fee  # +50k for token approval/transfer overhead
    gas_total_eth = w3.from_wei(gas_total_wei, "ether")
    eth_usd = _eth_usd_via_chainlink(w3)
    print(f"\n=== GAS ESTIMATE ===")
    print(f"  base_fee:  {w3.from_wei(base_fee, 'gwei'):.2f} gwei")
    print(f"  priority:  2.00 gwei")
    print(f"  max_fee:   {w3.from_wei(max_fee, 'gwei'):.2f} gwei")
    print(f"  Gas units: ~{gas_estimate + 50_000:,}")
    print(f"  Gas total: {gas_total_eth:.6f} ETH = ${float(gas_total_eth)*eth_usd:.2f} (ETH @ ${eth_usd:.2f})")
    print(f"  Cost as % of swap ({args.eth_amount} ETH): {float(gas_total_eth)/args.eth_amount*100:.1f}%")

    if not args.execute:
        print(f"\n[DRY-RUN] Pass --execute to actually send the swap.")
        print(f"\n=== PROJECTED FOR ${args.eth_amount * eth_usd:.0f} TRADE ===")
        gas_usd = float(gas_total_eth) * eth_usd
        position_usd = args.eth_amount * eth_usd
        print(f"  Gas (one-way buy): ${gas_usd:.2f}")
        print(f"  Estimated round-trip gas: ${gas_usd*2:.2f}")
        print(f"  Round-trip gas as % of position: {gas_usd*2/position_usd*100:.1f}%")
        print(f"  vs paper assumption ($15 round-trip): {'CHEAPER' if gas_usd*2 < 15 else 'MORE EXPENSIVE'} by ${abs(gas_usd*2-15):.2f}")
        return

    # Step 2 — Execute
    print(f"\n=== EXECUTING REAL SWAP ===")
    confirm = input(f"Swap {args.eth_amount} ETH (~${args.eth_amount*eth_usd:.2f}) for {symbol}? Type 'yes' to confirm: ")
    if confirm.lower() != "yes":
        print("Aborted.")
        return

    min_out = (amount_out * (10000 - args.slippage_bps)) // 10000
    deadline_padding = 600  # 10min
    router = w3.eth.contract(address=SWAP_ROUTER_02, abi=ROUTER_ABI)
    params = (WETH, token, args.fee_tier, acct.address, amount_in_wei, min_out, 0)

    nonce = w3.eth.get_transaction_count(acct.address)
    tx = router.functions.exactInputSingle(params).build_transaction({
        "from": acct.address,
        "value": amount_in_wei,
        "nonce": nonce,
        "gas": gas_estimate + 80_000,  # buffer
        "maxFeePerGas": max_fee,
        "maxPriorityFeePerGas": priority,
        "chainId": 1,
    })

    pre_token_bal_raw = erc20.functions.balanceOf(acct.address).call()
    # Submit signed tx via Flashbots Protect (write RPC) for MEV protection.
    w3_write = Web3(Web3.HTTPProvider(write_rpc))
    t0 = time.time()
    signed = acct.sign_transaction(tx)
    tx_hash = w3_write.eth.send_raw_transaction(signed.raw_transaction)
    print(f"  tx sent: 0x{tx_hash.hex()}")
    print(f"  etherscan: https://etherscan.io/tx/0x{tx_hash.hex()}")
    print(f"  waiting for receipt (timeout 180s)...")

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
    elapsed = time.time() - t0
    if receipt["status"] != 1:
        print(f"  ❌ TX FAILED — status=0. Likely revert (slippage / no liq).")
        return

    actual_gas = receipt["gasUsed"] * receipt["effectiveGasPrice"]
    actual_gas_eth = w3.from_wei(actual_gas, "ether")
    post_token_bal_raw = erc20.functions.balanceOf(acct.address).call()
    received_raw = post_token_bal_raw - pre_token_bal_raw
    received_human = received_raw / (10 ** decimals)
    expected_human = amount_out / (10 ** decimals)
    slippage_real = (1 - received_human / expected_human) * 100 if expected_human else 0

    print(f"\n=== ACTUAL RESULTS ===")
    print(f"  ✅ TX confirmed in {elapsed:.1f}s | block {receipt['blockNumber']}")
    print(f"  Gas used: {receipt['gasUsed']:,} units (estimate was {gas_estimate:,})")
    print(f"  Gas paid: {actual_gas_eth:.6f} ETH = ${float(actual_gas_eth)*eth_usd:.2f}")
    print(f"  Tokens received: {received_human:,.0f} {symbol}")
    print(f"  vs expected:     {expected_human:,.0f} {symbol}")
    print(f"  Slippage actual: {slippage_real:+.2f}%")
    print(f"\n  Gas as % of swap: {float(actual_gas_eth)/args.eth_amount*100:.2f}%")
    print(f"  Round-trip gas projection: ${float(actual_gas_eth)*eth_usd*2:.2f}")
    print(f"  Round-trip gas as % of ${args.eth_amount*eth_usd:.0f} position: "
          f"{float(actual_gas_eth)*eth_usd*2/(args.eth_amount*eth_usd)*100:.1f}%")


def _eth_usd_via_chainlink(w3):
    """ETH/USD price from Chainlink mainnet feed (no API key needed)."""
    AGGREGATOR = Web3.to_checksum_address("0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419")
    AGG_ABI = [{
        "inputs": [], "name": "latestRoundData",
        "outputs": [
            {"name": "roundId", "type": "uint80"},
            {"name": "answer", "type": "int256"},
            {"name": "startedAt", "type": "uint256"},
            {"name": "updatedAt", "type": "uint256"},
            {"name": "answeredInRound", "type": "uint80"},
        ],
        "stateMutability": "view", "type": "function"
    }]
    agg = w3.eth.contract(address=AGGREGATOR, abi=AGG_ABI)
    answer = agg.functions.latestRoundData().call()[1]
    return answer / 10**8  # Chainlink ETH/USD has 8 decimals


if __name__ == "__main__":
    main()
