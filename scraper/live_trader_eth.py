"""v14e.23 — ETH live trader (Uniswap V3 SwapRouter02 + Flashbots Protect).

Status: SKELETON — implemented but NOT yet wired into safe_scraper. Run the
smoke test first to validate gas/slip on real chain:
    python scripts/_eth_live_smoke_test.py --token 0x... --eth-amount 0.005 --execute

Once empirical numbers confirm the strategy is profitable at the chosen
position size, the dispatcher in safe_scraper._rt_open_trades can be flipped
to call this module via `chain == 'ethereum'` branch.

Architecture:
  - web3.py + eth-account for chain interaction
  - Default RPC = Flashbots Protect (https://rpc.flashbots.net) — sandwich-protected
  - ERC20 approvals cached in-memory per-token (one-time per token)
  - Swap router = Uniswap V3 SwapRouter02 (mainnet 0x68b3...Fc45)
  - Auto fee-tier discovery (try 3000 → 10000 → 500 → 100 in order)

Required env vars (scraper/.env):
  ETH_PRIVATE_KEY=0x...        # NEW dedicated wallet, NOT the main Phantom one
  ETH_RPC_URL=https://...      # default rpc.flashbots.net if unset

Risk model:
  - One compromised key drains ETH side only (Solana key separate, by design).
  - Flashbots Protect blocks sandwich MEV on entry/exit. Doesn't protect
    against pool rugs (token contract pulls liquidity) — fundamental risk.
  - Slippage protection: amountOutMinimum = quote × (1 - slippage_bps/10000).
  - Failed tx (revert): we eat the gas but don't have a position. Logged.
"""
from __future__ import annotations
import os
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Config (mainnet) ──────────────────────────────────────────────────────
SWAP_ROUTER_02 = "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45"
QUOTER_V2 = "0x61fFE014bA17989E743c5F6cB21bF9697530B21e"
WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
DEFAULT_RPC = "https://rpc.flashbots.net"

# Fee tiers to try in order (most liquid first for memecoins)
FEE_TIERS = [3000, 10000, 500, 100]

# ABIs (minimal)
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

ROUTER_ABI = [
    {"inputs": [{"components": [
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
    "stateMutability": "payable", "type": "function"},
]

ERC20_ABI = [
    {"inputs":[{"name":"who","type":"address"}],"name":"balanceOf","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"decimals","outputs":[{"type":"uint8"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"symbol","outputs":[{"type":"string"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"spender","type":"address"},{"name":"amount","type":"uint256"}],"name":"approve","outputs":[{"type":"bool"}],"stateMutability":"nonpayable","type":"function"},
    {"inputs":[{"name":"owner","type":"address"},{"name":"spender","type":"address"}],"name":"allowance","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"},
]


_w3 = None
_account = None
_approval_cache: set[str] = set()


def _client():
    """Lazy web3 client + account init. Imports at call time so the module
    loads even when web3 isn't installed (tests / non-EVM environments)."""
    global _w3, _account
    if _w3 is not None:
        return _w3, _account
    try:
        from web3 import Web3
        from eth_account import Account
    except ImportError:
        raise RuntimeError(
            "web3 + eth-account not installed. Run: pip install web3 eth-account"
        )
    pk = os.environ.get("ETH_PRIVATE_KEY")
    if not pk:
        raise RuntimeError("ETH_PRIVATE_KEY missing from env (scraper/.env)")
    rpc_url = os.environ.get("ETH_RPC_URL", DEFAULT_RPC)
    _w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not _w3.is_connected():
        raise RuntimeError(f"cannot connect to ETH RPC {rpc_url}")
    _account = Account.from_key(pk)
    logger.info(
        "live_trader_eth: connected to %s | wallet %s | block %d",
        rpc_url, _account.address, _w3.eth.block_number,
    )
    return _w3, _account


def _to_checksum(addr: str) -> str:
    from web3 import Web3
    return Web3.to_checksum_address(addr)


def _quote_with_best_fee(w3, amount_in_wei: int, token_out: str):
    """Try fee tiers in order, return (amount_out, fee_tier, gas_estimate) or None."""
    quoter = w3.eth.contract(address=_to_checksum(QUOTER_V2), abi=QUOTER_ABI)
    for fee in FEE_TIERS:
        try:
            result = quoter.functions.quoteExactInputSingle({
                "tokenIn": _to_checksum(WETH),
                "tokenOut": _to_checksum(token_out),
                "amountIn": amount_in_wei,
                "fee": fee,
                "sqrtPriceLimitX96": 0,
            }).call()
            return result[0], fee, result[3]
        except Exception:
            continue
    return None, None, None


def _eth_usd_price(w3) -> float:
    """Chainlink ETH/USD feed."""
    AGG = "0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419"
    ABI = [{"inputs": [], "name": "latestRoundData", "outputs": [
        {"name": "r", "type": "uint80"}, {"name": "answer", "type": "int256"},
        {"name": "s", "type": "uint256"}, {"name": "u", "type": "uint256"},
        {"name": "a", "type": "uint80"},
    ], "stateMutability": "view", "type": "function"}]
    c = w3.eth.contract(address=_to_checksum(AGG), abi=ABI)
    return c.functions.latestRoundData().call()[1] / 1e8


def execute_buy(ca: str, amount_usd: float, slippage_bps: int = 300) -> dict:
    """ETH → memecoin swap via Uniswap V3.

    Returns: {success, tx_hash, execution_price (USD), gas_usd,
              tokens_received, slippage_actual_bps, fee_tier_used}
    """
    w3, acct = _client()
    eth_usd = _eth_usd_price(w3)
    eth_amount = amount_usd / eth_usd
    amount_in_wei = w3.to_wei(eth_amount, "ether")

    # 1. Quote with auto fee-tier discovery
    amount_out, fee_tier, gas_est = _quote_with_best_fee(w3, amount_in_wei, ca)
    if amount_out is None:
        return {"success": False, "error": "no Uniswap V3 pool found across fee tiers"}

    # 2. Min-out with slippage tolerance
    min_out = (amount_out * (10000 - slippage_bps)) // 10000

    # 3. Build + sign tx
    base_fee = w3.eth.get_block("latest")["baseFeePerGas"]
    priority = w3.to_wei(2, "gwei")
    max_fee = base_fee * 2 + priority
    router = w3.eth.contract(address=_to_checksum(SWAP_ROUTER_02), abi=ROUTER_ABI)
    params = (_to_checksum(WETH), _to_checksum(ca), fee_tier, acct.address,
              amount_in_wei, min_out, 0)
    nonce = w3.eth.get_transaction_count(acct.address)
    tx = router.functions.exactInputSingle(params).build_transaction({
        "from": acct.address, "value": amount_in_wei, "nonce": nonce,
        "gas": gas_est + 80_000, "maxFeePerGas": max_fee,
        "maxPriorityFeePerGas": priority, "chainId": 1,
    })

    erc20 = w3.eth.contract(address=_to_checksum(ca), abi=ERC20_ABI)
    pre_bal = erc20.functions.balanceOf(acct.address).call()
    decimals = erc20.functions.decimals().call()

    t0 = time.time()
    signed = acct.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
    elapsed_ms = int((time.time() - t0) * 1000)

    if receipt["status"] != 1:
        gas_paid_eth = w3.from_wei(receipt["gasUsed"] * receipt["effectiveGasPrice"], "ether")
        return {"success": False, "error": "tx reverted",
                "tx_hash": tx_hash.hex(), "gas_usd": float(gas_paid_eth) * eth_usd,
                "exec_ms": elapsed_ms}

    post_bal = erc20.functions.balanceOf(acct.address).call()
    received = post_bal - pre_bal
    received_human = received / (10 ** decimals)
    expected_human = amount_out / (10 ** decimals)
    slippage_real_bps = int((1 - received / amount_out) * 10000) if amount_out else 0

    gas_paid_eth = w3.from_wei(receipt["gasUsed"] * receipt["effectiveGasPrice"], "ether")
    gas_usd = float(gas_paid_eth) * eth_usd
    execution_price_usd = amount_usd / received_human if received_human else 0

    return {
        "success": True,
        "tx_hash": tx_hash.hex(),
        "execution_price": execution_price_usd,
        "gas_usd": gas_usd,
        "tokens_received": received,
        "tokens_received_human": received_human,
        "slippage_actual_bps": slippage_real_bps,
        "fee_tier_used": fee_tier,
        "exec_ms": elapsed_ms,
        "block_number": receipt["blockNumber"],
    }


def _ensure_approval(w3, acct, token_address: str, amount_min: int):
    """Approve SwapRouter02 if not already done. Idempotent + cached."""
    if token_address in _approval_cache:
        return
    erc20 = w3.eth.contract(address=_to_checksum(token_address), abi=ERC20_ABI)
    current = erc20.functions.allowance(acct.address, _to_checksum(SWAP_ROUTER_02)).call()
    if current >= amount_min:
        _approval_cache.add(token_address)
        return
    base_fee = w3.eth.get_block("latest")["baseFeePerGas"]
    priority = w3.to_wei(2, "gwei")
    nonce = w3.eth.get_transaction_count(acct.address)
    MAX_UINT = 2**256 - 1
    tx = erc20.functions.approve(_to_checksum(SWAP_ROUTER_02), MAX_UINT).build_transaction({
        "from": acct.address, "nonce": nonce, "gas": 80_000,
        "maxFeePerGas": base_fee * 2 + priority,
        "maxPriorityFeePerGas": priority, "chainId": 1,
    })
    signed = acct.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
    if receipt["status"] != 1:
        raise RuntimeError(f"ERC20 approve failed for {token_address}: {tx_hash.hex()}")
    _approval_cache.add(token_address)
    logger.info("live_trader_eth: approved router for %s (tx %s)",
                token_address, tx_hash.hex())


def execute_sell(ca: str, amount_tokens: Optional[int] = None,
                 slippage_bps: int = 500) -> dict:
    """Memecoin → ETH swap via Uniswap V3.

    amount_tokens=None → sell entire balance.
    Higher default slippage (500 bps = 5%) on sells because dumping tokens
    on shallow pools moves price more than the entry quote suggests.
    """
    w3, acct = _client()
    eth_usd = _eth_usd_price(w3)
    erc20 = w3.eth.contract(address=_to_checksum(ca), abi=ERC20_ABI)
    decimals = erc20.functions.decimals().call()
    if amount_tokens is None:
        amount_tokens = erc20.functions.balanceOf(acct.address).call()
    if amount_tokens == 0:
        return {"success": False, "error": "zero balance to sell"}

    # 1. Approval (one-time per token)
    _ensure_approval(w3, acct, ca, amount_tokens)

    # 2. Quote token → WETH (try fee tiers)
    quoter = w3.eth.contract(address=_to_checksum(QUOTER_V2), abi=QUOTER_ABI)
    eth_out = None
    fee_tier_used = None
    gas_est = None
    for fee in FEE_TIERS:
        try:
            result = quoter.functions.quoteExactInputSingle({
                "tokenIn": _to_checksum(ca), "tokenOut": _to_checksum(WETH),
                "amountIn": amount_tokens, "fee": fee, "sqrtPriceLimitX96": 0,
            }).call()
            eth_out = result[0]; fee_tier_used = fee; gas_est = result[3]
            break
        except Exception:
            continue
    if eth_out is None:
        return {"success": False, "error": "no liquidity for sell on any fee tier"}

    min_out = (eth_out * (10000 - slippage_bps)) // 10000
    base_fee = w3.eth.get_block("latest")["baseFeePerGas"]
    priority = w3.to_wei(2, "gwei")
    max_fee = base_fee * 2 + priority
    router = w3.eth.contract(address=_to_checksum(SWAP_ROUTER_02), abi=ROUTER_ABI)
    params = (_to_checksum(ca), _to_checksum(WETH), fee_tier_used,
              acct.address, amount_tokens, min_out, 0)
    nonce = w3.eth.get_transaction_count(acct.address)
    tx = router.functions.exactInputSingle(params).build_transaction({
        "from": acct.address, "value": 0, "nonce": nonce,
        "gas": gas_est + 80_000, "maxFeePerGas": max_fee,
        "maxPriorityFeePerGas": priority, "chainId": 1,
    })

    pre_eth = w3.eth.get_balance(acct.address)
    t0 = time.time()
    signed = acct.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
    elapsed_ms = int((time.time() - t0) * 1000)

    if receipt["status"] != 1:
        return {"success": False, "error": "sell tx reverted",
                "tx_hash": tx_hash.hex(), "exec_ms": elapsed_ms}

    post_eth = w3.eth.get_balance(acct.address)
    gas_paid = receipt["gasUsed"] * receipt["effectiveGasPrice"]
    eth_received = post_eth - pre_eth + gas_paid  # net of gas

    return {
        "success": True,
        "tx_hash": tx_hash.hex(),
        "eth_received": eth_received,
        "eth_received_usd": w3.from_wei(eth_received, "ether") * eth_usd if eth_received else 0,
        "gas_usd": float(w3.from_wei(gas_paid, "ether")) * eth_usd,
        "fee_tier_used": fee_tier_used,
        "exec_ms": elapsed_ms,
        "block_number": receipt["blockNumber"],
    }


def open_live_trade(client_sb, token_entry: dict, strategy: str,
                    position_usd: float, config: dict) -> dict:
    """Orchestrator — mirrors live_trader.open_live_trade contract for the
    Solana side. Called by safe_scraper._rt_open_trades when chain=ethereum
    AND eth_live_enabled=True in rt_trade_config.

    Returns: dict with success/error + execution_price/tx_hash for paper sync.
    """
    ca = token_entry.get("token_address") or token_entry.get("_rt_pair_address")
    if not ca:
        return {"success": False, "error": "no token_address in entry"}
    slippage = int((config.get("eth_buy_slippage_bps") or 300))
    result = execute_buy(ca, position_usd, slippage_bps=slippage)
    if not result.get("success"):
        return result
    # Insert into paper_trades / rt_live_trades is deferred — caller (safe_scraper)
    # owns DB writes for parity with the Solana path.
    return result
