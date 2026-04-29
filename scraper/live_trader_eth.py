"""v14e.28 — ETH live trader (Uniswap V3 SwapRouter02 + Flashbots Protect).

Open + close paths now wired. Gated behind live_trading.eth_live_enabled in
rt_trade_config (default False). Flipping that flag opens ETH live positions
via Uniswap V3 mainnet at the size dictated by live_trading.eth_allocations
(or falls back to live_trading.allocations).

Architecture:
  - web3.py + eth-account for chain interaction
  - Read RPC = publicnode (free, accepts eth_call)
  - Write RPC = Flashbots Protect (sandwich-protected tx submission)
  - ERC20 approvals cached in-memory per-token (one-time per token)
  - Swap router = Uniswap V3 SwapRouter02 (mainnet 0x68b3...Fc45)
  - Auto fee-tier discovery (try 3000 → 10000 → 500 → 100 in order)
  - DB schema: paper_trades.chain='ethereum' rows. Solana-specific columns
    (sol_price_at_entry, position_sol, buy_input_lamports) are repurposed
    semantically: lamports = wei, sol = ether. Downstream tooling must
    discriminate by `chain` column. Proper schema migration deferred.

Required env vars (scraper/.env):
  ETH_PRIVATE_KEY=0x...                   # NEW dedicated wallet
  ETH_RPC_URL=https://rpc.flashbots.net   # tx submission (default if unset)
  ETH_READ_RPC_URL=https://...            # quotes/balance reads (default publicnode)

Risk model:
  - One compromised key drains ETH side only (Solana key separate, by design).
  - Flashbots Protect blocks sandwich MEV on entry/exit. Doesn't protect
    against pool rugs (token contract pulls liquidity) — fundamental risk.
  - Failed tx (revert): we eat the gas but don't have a position. Logged.
"""
from __future__ import annotations
import os
import time
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Config (mainnet) ──────────────────────────────────────────────────────
SWAP_ROUTER_02 = "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45"
QUOTER_V2 = "0x61fFE014bA17989E743c5F6cB21bF9697530B21e"
WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
DEFAULT_WRITE_RPC = "https://rpc.flashbots.net"
DEFAULT_READ_RPC = "https://ethereum-rpc.publicnode.com"

# Fee tiers to try in order (most liquid first for memecoins)
FEE_TIERS = [3000, 10000, 500, 100]

# v14e.33: Uniswap V2 fallback. Most ETH memecoins launch on V2 only — without
# this, _quote_with_best_fee returns None on every KOL call and the bot never
# buys. V2 router is the canonical UniswapV2Router02; pools are 0.30% fee fixed.
# Quote via router.getAmountsOut (reverts if no pair) — no factory call needed.
UNISWAP_V2_ROUTER = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"

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
    # multicall for bundling swap + unwrap in one tx
    {"inputs": [{"internalType": "bytes[]", "name": "data", "type": "bytes[]"}],
     "name": "multicall",
     "outputs": [{"internalType": "bytes[]", "name": "results", "type": "bytes[]"}],
     "stateMutability": "payable", "type": "function"},
    # unwrapWETH9: converts router-held WETH to native ETH and forwards to recipient.
    # Needed after a swap where outputToken=WETH so the user receives ETH not WETH.
    {"inputs": [
        {"internalType": "uint256", "name": "amountMinimum", "type": "uint256"},
        {"internalType": "address", "name": "recipient", "type": "address"},
     ],
     "name": "unwrapWETH9",
     "outputs": [],
     "stateMutability": "payable", "type": "function"},
]

# v14e.33: Uniswap V2 Router02 ABI (quote + swap functions only).
# - getAmountsOut: pure quote, reverts if no pair exists for path.
# - swapExactETHForTokensSupportingFeeOnTransferTokens: BUY path (handles
#   fee-on-transfer tokens — common in memecoins; the non-FoT variant reverts
#   on tax tokens).
# - swapExactTokensForETHSupportingFeeOnTransferTokens: SELL path (mirror).
V2_ROUTER_ABI = [
    {"inputs": [
        {"name": "amountIn", "type": "uint256"},
        {"name": "path", "type": "address[]"},
     ], "name": "getAmountsOut",
     "outputs": [{"name": "amounts", "type": "uint256[]"}],
     "stateMutability": "view", "type": "function"},
    {"inputs": [
        {"name": "amountOutMin", "type": "uint256"},
        {"name": "path", "type": "address[]"},
        {"name": "to", "type": "address"},
        {"name": "deadline", "type": "uint256"},
     ], "name": "swapExactETHForTokensSupportingFeeOnTransferTokens",
     "outputs": [], "stateMutability": "payable", "type": "function"},
    {"inputs": [
        {"name": "amountIn", "type": "uint256"},
        {"name": "amountOutMin", "type": "uint256"},
        {"name": "path", "type": "address[]"},
        {"name": "to", "type": "address"},
        {"name": "deadline", "type": "uint256"},
     ], "name": "swapExactTokensForETHSupportingFeeOnTransferTokens",
     "outputs": [], "stateMutability": "nonpayable", "type": "function"},
]

# Uniswap V2 Pair Swap event (different signature than V3):
# event Swap(address indexed sender, uint amount0In, uint amount1In,
#            uint amount0Out, uint amount1Out, address indexed to)
V2_SWAP_EVENT_TOPIC = "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822"

# Uniswap V3 Pool Swap event signature for receipt log parsing.
# event Swap(address sender, address recipient, int256 amount0, int256 amount1,
#            uint160 sqrtPriceX96, uint128 liquidity, int24 tick)
SWAP_EVENT_TOPIC = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"

ERC20_ABI = [
    {"inputs":[{"name":"who","type":"address"}],"name":"balanceOf","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"decimals","outputs":[{"type":"uint8"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"symbol","outputs":[{"type":"string"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"spender","type":"address"},{"name":"amount","type":"uint256"}],"name":"approve","outputs":[{"type":"bool"}],"stateMutability":"nonpayable","type":"function"},
    {"inputs":[{"name":"owner","type":"address"},{"name":"spender","type":"address"}],"name":"allowance","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"},
]


_w3_read = None
_w3_write = None
_account = None
# v14e.33: keyed by (token_address.lower(), router_address.lower()) so V3 and
# V2 approvals are tracked separately — selling on a different DEX than we
# bought on still triggers an approval if needed.
_approval_cache: set[tuple[str, str]] = set()

# v14e.43 — ETH-side daily loss limit (T4 from todo). Mirrors live_trader.py
# SOL impl but in USD (no SOL conversion needed). State resets at UTC midnight.
# Halts open_live_trade buys when cumulative day PnL drops below
# -eth_daily_loss_limit_usd. Existing trades still close normally (only buys
# are gated). Prevents one bad day from compounding into next-day losses.
_eth_daily_pnl_usd: float = 0.0
_eth_daily_pnl_reset_date: str = ""
_eth_daily_halted: bool = False

# v14e.43 — ETH dispatch lock (E5 from todo). open_live_trade is called from
# the RT listener which is async/threaded; without a lock two concurrent
# KOL calls can race past max_open_positions and the dedup check, opening
# 2× the intended exposure. The contention surface is tiny (one buy at a
# time fits the eth_max_open_positions=1 cap perfectly), so a global lock
# is the simplest correct fix. Negligible overhead — buys take seconds.
import threading as _threading
_eth_open_lock = _threading.Lock()


def _check_eth_loss_limit(config: dict) -> bool:
    """T4: returns True if ETH live BUYs should be halted today.

    Reads `eth_daily_loss_limit_usd` from rt_trade_config.live_trading. Default
    $50 (= 2.5x microtest position). Sells always proceed; only new entries
    are blocked once the day loss exceeds the limit.
    """
    global _eth_daily_pnl_usd, _eth_daily_pnl_reset_date, _eth_daily_halted
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if _eth_daily_pnl_reset_date != today:
        _eth_daily_pnl_usd = 0.0
        _eth_daily_pnl_reset_date = today
        _eth_daily_halted = False
    daily_limit = float(config.get("eth_daily_loss_limit_usd", 50.0))
    if daily_limit <= 0:
        return False
    if _eth_daily_pnl_usd < -daily_limit:
        if not _eth_daily_halted:
            logger.warning("ETH LIVE TRADING HALTED: daily loss $%.2f exceeds limit $%.2f",
                           _eth_daily_pnl_usd, daily_limit)
            _eth_daily_halted = True
            try:
                from alerter import alert_loss_limit_hit
                alert_loss_limit_hit("eth_daily", _eth_daily_pnl_usd, daily_limit)
            except Exception:
                pass
        return True
    return False


def _track_eth_pnl(pnl_usd: float):
    """T4: accumulate ETH realized PnL into the daily-rollup counter."""
    global _eth_daily_pnl_usd
    _eth_daily_pnl_usd += pnl_usd

# ERC20 Transfer event topic — used to recover received amount from the
# target token's Transfer-to-recipient log. Works for V2 (where Swap event
# decoding requires knowing token0/token1 ordering) and as a backup for V3.
ERC20_TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"


def _client():
    """Lazy web3 clients + account init. Returns (w3_read, w3_write, account).

    Read calls (quote, balance, receipt) hit publicnode. Write calls
    (send_raw_transaction) hit Flashbots Protect for MEV protection.
    Flashbots Protect returns 403 on eth_call so reads MUST use a separate RPC.
    """
    global _w3_read, _w3_write, _account
    if _w3_read is not None:
        return _w3_read, _w3_write, _account
    try:
        from web3 import Web3
        from eth_account import Account
    except ImportError:
        raise RuntimeError("web3 + eth-account not installed. Run: pip install web3 eth-account")
    pk = os.environ.get("ETH_PRIVATE_KEY")
    if not pk:
        raise RuntimeError("ETH_PRIVATE_KEY missing from env (scraper/.env)")
    read_rpc = os.environ.get("ETH_READ_RPC_URL", DEFAULT_READ_RPC)
    write_rpc = os.environ.get("ETH_RPC_URL", DEFAULT_WRITE_RPC)
    _w3_read = Web3(Web3.HTTPProvider(read_rpc))
    if not _w3_read.is_connected():
        raise RuntimeError(f"cannot connect to read RPC {read_rpc}")
    _w3_write = Web3(Web3.HTTPProvider(write_rpc))
    _account = Account.from_key(pk)
    logger.info(
        "live_trader_eth: read=%s write=%s wallet=%s block=%d",
        read_rpc, write_rpc, _account.address, _w3_read.eth.block_number,
    )
    return _w3_read, _w3_write, _account


def _to_checksum(addr: str) -> str:
    from web3 import Web3
    return Web3.to_checksum_address(addr)


def _quote_with_best_fee(w3, amount_in_wei: int, token_in: str, token_out: str):
    """V3-only quote. Try fee tiers in order, return (amount_out, fee_tier, gas_estimate) or (None, None, None)."""
    quoter = w3.eth.contract(address=_to_checksum(QUOTER_V2), abi=QUOTER_ABI)
    for fee in FEE_TIERS:
        try:
            result = quoter.functions.quoteExactInputSingle({
                "tokenIn": _to_checksum(token_in),
                "tokenOut": _to_checksum(token_out),
                "amountIn": amount_in_wei,
                "fee": fee,
                "sqrtPriceLimitX96": 0,
            }).call()
            return result[0], fee, result[3]
        except Exception:
            continue
    return None, None, None


def _quote_v2(w3, amount_in_wei: int, token_in: str, token_out: str):
    """V2-only quote via UniswapV2Router02.getAmountsOut. Returns amount_out or None.

    getAmountsOut reverts if no pair exists, so the try/except is the
    pair-existence check (no separate factory.getPair call needed).
    """
    router = w3.eth.contract(address=_to_checksum(UNISWAP_V2_ROUTER), abi=V2_ROUTER_ABI)
    try:
        amounts = router.functions.getAmountsOut(
            amount_in_wei,
            [_to_checksum(token_in), _to_checksum(token_out)],
        ).call()
        return amounts[-1] if amounts else None
    except Exception:
        return None


def _quote_best_route(w3, amount_in_wei: int, token_in: str, token_out: str):
    """v14e.33: Unified quote across V3 + V2. Picks the route with the highest
    output (best price for the user).

    Returns (amount_out, route, fee_tier, gas_estimate) where:
      route = "v3" | "v2"
      fee_tier = V3 fee tier (3000/10000/500/100) or 3000 for V2 (constant fee)
      gas_estimate = V3 quoter's gas est, or 200_000 default for V2

    Returns (None, None, None, None) if no liquidity on either DEX.
    """
    v3_out, v3_fee, v3_gas = _quote_with_best_fee(w3, amount_in_wei, token_in, token_out)
    v2_out = _quote_v2(w3, amount_in_wei, token_in, token_out)

    # Both fail
    if not v3_out and not v2_out:
        return None, None, None, None
    # V2 only
    if not v3_out:
        return v2_out, "v2", 3000, 200_000
    # V3 only
    if not v2_out:
        return v3_out, "v3", v3_fee, v3_gas
    # Both — pick higher output (better price). Ties go to V3 (cheaper gas via single-pool).
    if v2_out > v3_out:
        return v2_out, "v2", 3000, 200_000
    return v3_out, "v3", v3_fee, v3_gas


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


def _parse_swap_output_from_logs(receipt, target_token_addr: str) -> int | None:
    """Parse Uniswap V3 Swap event from receipt logs to recover the exact output
    amount, independent of RPC state propagation. Returns raw token amount
    received by the recipient (positive int), or None if no Swap event found.

    Why: balanceOf via publicnode RPC has read-after-write lag — calling it
    immediately after wait_for_transaction_receipt sometimes returns the
    pre-tx state, making (post_bal - pre_bal) = 0 even on successful swaps.
    Receipt logs are authoritative since they're embedded in the block.

    For exactInputSingle WETH→TOKEN: amount0 or amount1 (depending on token
    ordering) is negative for the side the pool sends to recipient. We pick
    the negative side as the output (taking absolute value).
    """
    target = target_token_addr.lower()
    for log in receipt.get("logs", []):
        topics = log.get("topics", [])
        if not topics:
            continue
        topic0 = topics[0]
        topic_hex = topic0.hex() if hasattr(topic0, "hex") else str(topic0)
        if not topic_hex.startswith("0x"):
            topic_hex = "0x" + topic_hex
        if topic_hex.lower() != SWAP_EVENT_TOPIC.lower():
            continue
        # We have a Swap event. Pool address = log["address"].
        # Decode data: amount0 (int256), amount1 (int256), then 3 more fields.
        data = log["data"]
        data_hex = data.hex() if hasattr(data, "hex") else str(data)
        if data_hex.startswith("0x"):
            data_hex = data_hex[2:]
        amount0 = int(data_hex[0:64], 16)
        if amount0 >= 2**255:
            amount0 -= 2**256
        amount1 = int(data_hex[64:128], 16)
        if amount1 >= 2**255:
            amount1 -= 2**256
        # The negative amount is what the pool SENT (= what recipient received).
        # We take absolute value of the negative side.
        if amount0 < 0:
            return -amount0
        if amount1 < 0:
            return -amount1
    return None


def _parse_transfer_to_recipient(receipt, token_addr: str, recipient: str) -> int | None:
    """Sum ERC20 Transfer events of `token_addr` where `to == recipient`.

    Used to recover the exact amount received in a swap regardless of route
    (V2/V3) and resilient to fee-on-transfer tokens (we get the post-fee
    amount that actually landed in the wallet). Returns None if no matching
    Transfer found.
    """
    target_token = token_addr.lower()
    target_to = recipient.lower()
    total = 0
    found = False
    for log in receipt.get("logs", []):
        addr = (log.get("address") or "").lower()
        if addr != target_token:
            continue
        topics = log.get("topics", [])
        if not topics or len(topics) < 3:
            continue
        topic0 = topics[0]
        topic_hex = topic0.hex() if hasattr(topic0, "hex") else str(topic0)
        if not topic_hex.startswith("0x"):
            topic_hex = "0x" + topic_hex
        if topic_hex.lower() != ERC20_TRANSFER_TOPIC.lower():
            continue
        # topics[2] = `to` (indexed). Last 20 bytes = recipient address.
        to_topic = topics[2]
        to_hex = to_topic.hex() if hasattr(to_topic, "hex") else str(to_topic)
        if not to_hex.startswith("0x"):
            to_hex = "0x" + to_hex
        to_addr = "0x" + to_hex[-40:]
        if to_addr.lower() != target_to:
            continue
        data = log.get("data")
        data_hex = data.hex() if hasattr(data, "hex") else str(data)
        if data_hex.startswith("0x"):
            data_hex = data_hex[2:]
        try:
            total += int(data_hex, 16)
            found = True
        except ValueError:
            continue
    return total if found else None


def execute_buy(ca: str, amount_usd: float, slippage_bps: int = 300,
                force_route: str | None = None) -> dict:
    """ETH → memecoin swap via Uniswap V3 or V2 (whichever quotes higher).

    force_route: None (auto-pick best) | "v3" | "v2". Used by the smoke
    test harness to force a specific path even when the other one quotes
    higher; keeps prod auto-routing in execute_sell unchanged.

    Returns: {success, tx_hash, execution_price (USD), gas_usd,
              tokens_received (raw), tokens_received_human, slippage_actual_bps,
              fee_tier_used, route ("v3"|"v2"), exec_ms, block_number,
              eth_spent_wei, eth_usd}
    """
    w3r, w3w, acct = _client()
    eth_usd = _eth_usd_price(w3r)
    eth_amount = amount_usd / eth_usd
    amount_in_wei = w3r.to_wei(eth_amount, "ether")

    # 1. Quote — forced or auto.
    if force_route == "v3":
        v3_out, v3_fee, v3_gas = _quote_with_best_fee(w3r, amount_in_wei, WETH, ca)
        amount_out, route, fee_tier, gas_est = (v3_out, "v3", v3_fee, v3_gas) if v3_out else (None, None, None, None)
    elif force_route == "v2":
        v2_out = _quote_v2(w3r, amount_in_wei, WETH, ca)
        amount_out, route, fee_tier, gas_est = (v2_out, "v2", 3000, 200_000) if v2_out else (None, None, None, None)
    else:
        amount_out, route, fee_tier, gas_est = _quote_best_route(w3r, amount_in_wei, WETH, ca)
    if amount_out is None:
        return {"success": False, "error": f"no Uniswap pool found (route={force_route or 'auto'})"}

    min_out = (amount_out * (10000 - slippage_bps)) // 10000

    # 2. Build + sign tx — branch by route.
    base_fee = w3r.eth.get_block("latest")["baseFeePerGas"]
    priority = w3r.to_wei(2, "gwei")
    max_fee = base_fee * 2 + priority
    nonce = w3r.eth.get_transaction_count(acct.address)
    if route == "v3":
        router = w3r.eth.contract(address=_to_checksum(SWAP_ROUTER_02), abi=ROUTER_ABI)
        params = (_to_checksum(WETH), _to_checksum(ca), fee_tier, acct.address,
                  amount_in_wei, min_out, 0)
        tx = router.functions.exactInputSingle(params).build_transaction({
            "from": acct.address, "value": amount_in_wei, "nonce": nonce,
            "gas": gas_est + 80_000, "maxFeePerGas": max_fee,
            "maxPriorityFeePerGas": priority, "chainId": 1,
        })
    else:  # v2
        # FoT-tolerant variant — non-FoT tokens still work, FoT (tax) tokens
        # don't revert. Deadline = now + 5 min (matches Uniswap UI default).
        router = w3r.eth.contract(address=_to_checksum(UNISWAP_V2_ROUTER), abi=V2_ROUTER_ABI)
        deadline = int(time.time()) + 300
        tx = router.functions.swapExactETHForTokensSupportingFeeOnTransferTokens(
            min_out,
            [_to_checksum(WETH), _to_checksum(ca)],
            acct.address,
            deadline,
        ).build_transaction({
            "from": acct.address, "value": amount_in_wei, "nonce": nonce,
            "gas": gas_est + 50_000, "maxFeePerGas": max_fee,
            "maxPriorityFeePerGas": priority, "chainId": 1,
        })

    erc20 = w3r.eth.contract(address=_to_checksum(ca), abi=ERC20_ABI)
    try:
        decimals = erc20.functions.decimals().call()
    except Exception:
        decimals = 18

    t0 = time.time()
    signed = acct.sign_transaction(tx)
    tx_hash = w3w.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3r.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
    elapsed_ms = int((time.time() - t0) * 1000)

    if receipt["status"] != 1:
        gas_paid_eth = w3r.from_wei(receipt["gasUsed"] * receipt["effectiveGasPrice"], "ether")
        return {"success": False, "error": "tx reverted",
                "tx_hash": tx_hash.hex(), "gas_usd": float(gas_paid_eth) * eth_usd,
                "exec_ms": elapsed_ms}

    # Parse output amount. V3 Swap event is route-deterministic and resolves
    # output by sign of amount0/amount1; V2 Swap event needs token0/token1
    # ordering to disambiguate. We use the target token's ERC20 Transfer-to-
    # recipient log instead — works for BOTH routes and is FoT-aware (returns
    # post-tax amount that actually landed in the wallet).
    received = _parse_transfer_to_recipient(receipt, ca, acct.address)
    if received is None or received <= 0:
        # V3 fallback path (Swap event decode) — kept as a secondary parser
        # in case Transfer parsing misses an edge case.
        received = _parse_swap_output_from_logs(receipt, ca) if route == "v3" else None
    if received is None or received <= 0:
        try:
            time.sleep(2)
            post_bal = erc20.functions.balanceOf(acct.address).call()
            received = max(post_bal, 0)
            logger.warning(
                "live_trader_eth.execute_buy: log parse failed for %s (route=%s), "
                "using balanceOf fallback (received=%d). RPC lag suspected.",
                ca[:10], route, received,
            )
        except Exception:
            received = 0

    received_human = received / (10 ** decimals)
    slippage_real_bps = int((1 - received / amount_out) * 10000) if amount_out else 0

    gas_paid_eth = w3r.from_wei(receipt["gasUsed"] * receipt["effectiveGasPrice"], "ether")
    gas_usd = float(gas_paid_eth) * eth_usd
    execution_price_usd = amount_usd / received_human if received_human else 0

    return {
        "success": True,
        "tx_hash": tx_hash.hex(),
        "execution_price": execution_price_usd,
        "gas_usd": gas_usd,
        "tokens_received": received,
        "tokens_received_human": received_human,
        "decimals": decimals,
        "slippage_actual_bps": slippage_real_bps,
        "fee_tier_used": fee_tier,
        "route": route,
        "exec_ms": elapsed_ms,
        "block_number": receipt["blockNumber"],
        "eth_spent_wei": amount_in_wei,
        "eth_usd": eth_usd,
    }


def _ensure_approval(w3r, w3w, acct, token_address: str, amount_min: int,
                     router_address: str = SWAP_ROUTER_02):
    """Approve `router_address` to spend `token_address`. Idempotent + cached.

    v14e.33: router_address parameterized so V2 sells don't reuse the V3
    approval (different routers, different allowance slots). Cache key is
    (token, router) so each pairing is tracked independently.
    """
    cache_key = (token_address.lower(), router_address.lower())
    if cache_key in _approval_cache:
        return
    erc20 = w3r.eth.contract(address=_to_checksum(token_address), abi=ERC20_ABI)
    current = erc20.functions.allowance(acct.address, _to_checksum(router_address)).call()
    if current >= amount_min:
        _approval_cache.add(cache_key)
        return
    base_fee = w3r.eth.get_block("latest")["baseFeePerGas"]
    priority = w3r.to_wei(2, "gwei")
    nonce = w3r.eth.get_transaction_count(acct.address)
    MAX_UINT = 2**256 - 1
    tx = erc20.functions.approve(_to_checksum(router_address), MAX_UINT).build_transaction({
        "from": acct.address, "nonce": nonce, "gas": 80_000,
        "maxFeePerGas": base_fee * 2 + priority,
        "maxPriorityFeePerGas": priority, "chainId": 1,
    })
    signed = acct.sign_transaction(tx)
    tx_hash = w3w.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3r.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
    if receipt["status"] != 1:
        raise RuntimeError(f"ERC20 approve failed for {token_address} -> {router_address}: {tx_hash.hex()}")
    _approval_cache.add(cache_key)
    logger.info("live_trader_eth: approved %s for %s (tx %s)",
                router_address, token_address, tx_hash.hex())


def execute_sell(ca: str, amount_tokens: Optional[int] = None,
                 slippage_bps: int = 500,
                 force_route: str | None = None) -> dict:
    """Memecoin → ETH swap via Uniswap V3 or V2 (whichever quotes higher).

    amount_tokens=None → sell entire balance. Default slippage 500 bps (5%) on
    sells because dumping shallow-pool tokens moves price more than the entry
    quote suggests.

    V3 path: SwapRouter02.exactInputSingle outputs WETH ERC20, then we bundle
      unwrapWETH9 via multicall so the user receives native ETH atomically.
    V2 path: UniswapV2Router02.swapExactTokensForETHSupportingFeeOnTransferTokens
      outputs native ETH directly — no unwrap needed, FoT-tolerant.

    Returns: {success, tx_hash, eth_received (wei), eth_received_human (eth),
              eth_received_usd, gas_usd, fee_tier_used, route ("v3"|"v2"),
              exec_ms, block_number, eth_usd, slippage_actual_bps}
    """
    w3r, w3w, acct = _client()
    eth_usd = _eth_usd_price(w3r)
    erc20 = w3r.eth.contract(address=_to_checksum(ca), abi=ERC20_ABI)
    try:
        decimals = erc20.functions.decimals().call()
    except Exception:
        decimals = 18
    if amount_tokens is None:
        amount_tokens = erc20.functions.balanceOf(acct.address).call()
    if amount_tokens == 0:
        return {"success": False, "error": "zero balance to sell"}

    if force_route == "v3":
        v3_out, v3_fee, v3_gas = _quote_with_best_fee(w3r, amount_tokens, ca, WETH)
        eth_out, route, fee_tier_used, gas_est = (v3_out, "v3", v3_fee, v3_gas) if v3_out else (None, None, None, None)
    elif force_route == "v2":
        v2_out = _quote_v2(w3r, amount_tokens, ca, WETH)
        eth_out, route, fee_tier_used, gas_est = (v2_out, "v2", 3000, 200_000) if v2_out else (None, None, None, None)
    else:
        eth_out, route, fee_tier_used, gas_est = _quote_best_route(w3r, amount_tokens, ca, WETH)
    if eth_out is None:
        return {"success": False, "error": f"no liquidity for sell (route={force_route or 'auto'})"}

    # Approve only the router we'll actually use.
    target_router = SWAP_ROUTER_02 if route == "v3" else UNISWAP_V2_ROUTER
    _ensure_approval(w3r, w3w, acct, ca, amount_tokens, router_address=target_router)

    min_out = (eth_out * (10000 - slippage_bps)) // 10000
    base_fee = w3r.eth.get_block("latest")["baseFeePerGas"]
    priority = w3r.to_wei(2, "gwei")
    max_fee = base_fee * 2 + priority
    nonce = w3r.eth.get_transaction_count(acct.address)

    if route == "v3":
        router = w3r.eth.contract(address=_to_checksum(SWAP_ROUTER_02), abi=ROUTER_ABI)
        # Step 1: swap with recipient = router (router holds WETH).
        # Step 2: unwrapWETH9 sweeps WETH → native ETH to acct. Atomic via multicall.
        swap_params = (_to_checksum(ca), _to_checksum(WETH), fee_tier_used,
                       _to_checksum(SWAP_ROUTER_02),
                       amount_tokens, min_out, 0)
        swap_calldata = router.encode_abi("exactInputSingle", args=[swap_params])
        unwrap_calldata = router.encode_abi(
            "unwrapWETH9", args=[min_out, acct.address]
        )
        tx = router.functions.multicall([swap_calldata, unwrap_calldata]).build_transaction({
            "from": acct.address, "value": 0, "nonce": nonce,
            "gas": gas_est + 130_000, "maxFeePerGas": max_fee,
            "maxPriorityFeePerGas": priority, "chainId": 1,
        })
    else:  # v2
        router = w3r.eth.contract(address=_to_checksum(UNISWAP_V2_ROUTER), abi=V2_ROUTER_ABI)
        deadline = int(time.time()) + 300
        tx = router.functions.swapExactTokensForETHSupportingFeeOnTransferTokens(
            amount_tokens,
            min_out,
            [_to_checksum(ca), _to_checksum(WETH)],
            acct.address,
            deadline,
        ).build_transaction({
            "from": acct.address, "value": 0, "nonce": nonce,
            "gas": gas_est + 80_000, "maxFeePerGas": max_fee,
            "maxPriorityFeePerGas": priority, "chainId": 1,
        })

    t0 = time.time()
    signed = acct.sign_transaction(tx)
    tx_hash = w3w.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3r.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
    elapsed_ms = int((time.time() - t0) * 1000)

    if receipt["status"] != 1:
        return {"success": False, "error": "sell tx reverted",
                "tx_hash": tx_hash.hex(), "exec_ms": elapsed_ms}

    # Authoritative output amount from receipt logs (RPC-lag-immune).
    # V3: V3 Swap event records pool's amount0/amount1 — the WETH side is what
    #     the router received before unwrapping. Multicall preserves the
    #     amount through unwrapWETH9, so this also equals what acct received.
    # V2: V2 Swap topic differs and amount0/amount1 mapping needs token0
    #     ordering. Easier: sum WETH Transfer events from pair → router; that
    #     amount equals what router unwraps and forwards to acct.
    if route == "v3":
        eth_received_wei = _parse_swap_output_from_logs(receipt, WETH)
    else:
        eth_received_wei = _parse_transfer_to_recipient(receipt, WETH, UNISWAP_V2_ROUTER)
    if eth_received_wei is None or eth_received_wei <= 0:
        # Fallback: native ETH balance delta. Requires archive RPC for the
        # historical balance read at block N-1; publicnode supports this.
        time.sleep(2)
        try:
            post_eth = w3r.eth.get_balance(acct.address)
            gas_paid = receipt["gasUsed"] * receipt["effectiveGasPrice"]
            eth_received_wei = max(
                post_eth + gas_paid - w3r.eth.get_balance(acct.address, receipt["blockNumber"] - 1),
                0,
            )
            logger.warning(
                "live_trader_eth.execute_sell: log parse failed for %s (route=%s), "
                "using balance delta fallback (received=%d wei).",
                ca[:10], route, eth_received_wei,
            )
        except Exception as _e:
            logger.error(
                "live_trader_eth.execute_sell: balance fallback also failed for %s (route=%s): %s",
                ca[:10], route, _e,
            )
            eth_received_wei = 0

    eth_received_human = eth_received_wei / 1e18 if eth_received_wei else 0.0
    eth_received_usd = eth_received_human * eth_usd
    gas_paid_wei = receipt["gasUsed"] * receipt["effectiveGasPrice"]
    slippage_real_bps = int((1 - eth_received_wei / eth_out) * 10000) if eth_out else 0

    return {
        "success": True,
        "tx_hash": tx_hash.hex(),
        "eth_received": eth_received_wei,
        "eth_received_human": eth_received_human,
        "eth_received_usd": eth_received_usd,
        "gas_usd": float(w3r.from_wei(gas_paid_wei, "ether")) * eth_usd,
        "fee_tier_used": fee_tier_used,
        "route": route,
        "exec_ms": elapsed_ms,
        "block_number": receipt["blockNumber"],
        "eth_usd": eth_usd,
        "slippage_actual_bps": slippage_real_bps,
    }


def unwrap_weth_balance() -> dict:
    """Manually unwrap any WETH sitting in the wallet to native ETH.

    Used to recover from pre-multicall versions of execute_sell that left
    WETH on the wallet. Calls WETH.withdraw(amount) directly.
    """
    w3r, w3w, acct = _client()
    eth_usd = _eth_usd_price(w3r)
    weth = w3r.eth.contract(
        address=_to_checksum(WETH),
        abi=ERC20_ABI + [
            {"inputs": [{"name": "amount", "type": "uint256"}],
             "name": "withdraw", "outputs": [],
             "stateMutability": "nonpayable", "type": "function"},
        ],
    )
    bal = weth.functions.balanceOf(acct.address).call()
    if bal == 0:
        return {"success": True, "message": "no WETH to unwrap", "amount_wei": 0}

    base_fee = w3r.eth.get_block("latest")["baseFeePerGas"]
    priority = w3r.to_wei(2, "gwei")
    max_fee = base_fee * 2 + priority
    nonce = w3r.eth.get_transaction_count(acct.address)
    tx = weth.functions.withdraw(bal).build_transaction({
        "from": acct.address, "nonce": nonce, "gas": 80_000,
        "maxFeePerGas": max_fee, "maxPriorityFeePerGas": priority, "chainId": 1,
    })
    signed = acct.sign_transaction(tx)
    tx_hash = w3w.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3r.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
    if receipt["status"] != 1:
        return {"success": False, "error": "withdraw reverted",
                "tx_hash": tx_hash.hex()}
    gas_paid_eth = w3r.from_wei(receipt["gasUsed"] * receipt["effectiveGasPrice"], "ether")
    return {
        "success": True,
        "tx_hash": tx_hash.hex(),
        "amount_wei": bal,
        "amount_eth": bal / 1e18,
        "amount_usd": (bal / 1e18) * eth_usd,
        "gas_usd": float(gas_paid_eth) * eth_usd,
    }


def _calc_message_to_buy(message_ts: str | None) -> int | None:
    """Reaction speed: seconds between Telegram message and buy tx submitted."""
    if not message_ts:
        return None
    try:
        ts = (datetime.fromisoformat(message_ts.replace("Z", "+00:00"))
              if isinstance(message_ts, str) else message_ts)
        return int((datetime.now(timezone.utc) - ts).total_seconds())
    except Exception:
        return None


def open_live_trade(client_sb, token_entry: dict, strategy: str,
                    position_usd: float, config: dict) -> dict:
    """ETH live open + DB insert. Mirrors live_trader.open_live_trade for SOL.

    Inserts paper_trades row with chain='ethereum', source='rt_live'. Solana-
    specific columns repurposed semantically:
      sol_price_at_entry → eth_price_at_entry (USD)
      position_sol       → position_eth (ether amount)
      buy_input_lamports → wei amount sent

    v14e.43: serialized via _eth_open_lock so concurrent KOL calls can't race
    past max_open_positions / dedup. Also gated on _check_eth_loss_limit so a
    bad day stops compounding into the next.

    Returns: {"success": bool, "execution_price": float|None, ...buy result}
    """
    with _eth_open_lock:
        return _open_live_trade_locked(client_sb, token_entry, strategy,
                                        position_usd, config)


def _open_live_trade_locked(client_sb, token_entry: dict, strategy: str,
                             position_usd: float, config: dict) -> dict:
    _FAIL = {"success": False, "execution_price": None}
    ca = token_entry.get("token_address")
    symbol = token_entry.get("symbol", "???")
    # v14e.28: normalize EVM address to lowercase (defensive: also done in
    # safe_scraper._rt_open_trades). EVM is case-insensitive — Postgres .eq()
    # is not — so dedup queries miss when stored case differs from query case.
    if ca:
        ca = ca.lower()

    if not ca:
        logger.warning("live_trader_eth: no CA for %s — skipping", symbol)
        return _FAIL

    token_chain = token_entry.get("chain") or "solana"
    if token_chain != "ethereum":
        logger.info("live_trader_eth: %s/%s chain=%s — skip (not ethereum)",
                    symbol, ca[:8] if ca else "?", token_chain)
        return _FAIL

    entry_price = float(token_entry.get("price_usd", 0))
    if entry_price <= 0:
        logger.error("live_trader_eth: entry_price=0 for %s — aborting", symbol)
        return _FAIL

    # v14e.43 — daily loss limit gate (T4). Halts new buys once
    # cumulative day pnl < -eth_daily_loss_limit_usd. Sells continue to
    # process via check_live_trades_eth so existing positions can close.
    if _check_eth_loss_limit(config):
        return _FAIL

    # v14e.32+: ETH-specific position cap (overrides Kelly-derived size).
    # Without this, _rt_position_size returns Kelly × bankroll capped at the
    # SOL-tuned max_position_usd, which is unsafe for ETH where slippage and
    # gas profile is different. Phase 1 microtest sets eth_max_position_usd=50.
    eth_cap = config.get("eth_max_position_usd")
    if eth_cap is not None:
        eth_cap = float(eth_cap)
        if position_usd > eth_cap:
            logger.info("live_trader_eth: capping position $%.2f -> $%.2f (eth_max_position_usd)",
                        position_usd, eth_cap)
            position_usd = eth_cap

    # v14e.32+: ETH-specific max open. Falls back to shared max_open_positions
    # if eth_max_open_positions is unset (preserves prior behavior).
    max_open = int(config.get("eth_max_open_positions",
                              config.get("max_open_positions", 5)))
    try:
        result = (
            client_sb.table("paper_trades")
            .select("id", count="exact")
            .eq("status", "open")
            .eq("source", "rt_live")
            .eq("chain", "ethereum")
            .execute()
        )
        open_count = result.count or 0
        if open_count >= max_open:
            logger.info("live_trader_eth: max ETH open positions (%d) reached — skipping %s",
                        max_open, symbol)
            return _FAIL
    except Exception as e:
        logger.warning("live_trader_eth: failed to check open positions: %s", e)

    # 24h dedup cooldown — same strategy + same token
    dedup_hours = int(config.get("dedup_cooldown_hours", 24))
    try:
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=dedup_hours)).isoformat()
        dedup_res = (
            client_sb.table("paper_trades")
            .select("id", count="exact")
            .eq("token_address", ca)
            .eq("source", "rt_live")
            .eq("strategy", strategy)
            .gte("created_at", cutoff)
            .execute()
        )
        if dedup_res.count and dedup_res.count > 0:
            logger.info("live_trader_eth: dedup cooldown — %s/%s traded in last %dh",
                        symbol, strategy, dedup_hours)
            return _FAIL
    except Exception as e:
        logger.debug("live_trader_eth: dedup check failed for %s: %s", symbol, e)

    slippage = int(config.get("eth_buy_slippage_bps", 300))
    _t_pre_buy = time.time()
    result = execute_buy(ca, position_usd, slippage_bps=slippage)
    _t_post_buy = time.time()
    if not result.get("success"):
        logger.warning("live_trader_eth: buy failed for %s: %s", symbol, result.get("error"))
        try:
            from alerter import alert_live_trade_failed
            alert_live_trade_failed(symbol, "BUY (ETH)", result.get("error", "unknown"))
        except Exception:
            pass
        return _FAIL

    execution_price = result.get("execution_price") or entry_price
    eth_usd = result.get("eth_usd") or 0
    eth_spent_wei = result.get("eth_spent_wei") or 0
    eth_spent = eth_spent_wei / 1e18 if eth_spent_wei else 0

    # Build TP/SL from strategy
    from paper_trader import STRATEGIES
    tranches = STRATEGIES.get(strategy, [{"tp_mult": 2.0, "sl_mult": 0.70, "horizon_min": 120}])
    tranche = tranches[0]
    tp_price = execution_price * tranche["tp_mult"] if tranche.get("tp_mult") else None
    sl_price = execution_price * tranche["sl_mult"]

    actual_slippage_bps = 0
    if execution_price > 0 and entry_price > 0:
        actual_slippage_bps = round((execution_price / entry_price - 1) * 10000)

    row = {
        "cycle_ts": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "token_address": ca,
        "chain": "ethereum",
        "rank_in_cycle": 0,
        "entry_price": execution_price,
        "entry_score": int(token_entry.get("score", 0)),
        "entry_mcap": float(token_entry["market_cap"]) if token_entry.get("market_cap") else None,
        "status": "open",
        "strategy": strategy,
        "tp_price": tp_price,
        "sl_price": sl_price,
        "horizon_minutes": tranche.get("horizon_min", 120),
        "tranche_pct": 1.0,
        "tranche_label": "main",
        "position_usd": round(position_usd, 2),
        "source": "rt_live",
        "tx_signature": result["tx_hash"],
        "execution_price": execution_price,
        "kol_group": token_entry.get("_rt_kol_group"),
        "kol_tier": token_entry.get("_rt_kol_tier"),
        "kol_score": token_entry.get("_rt_kol_score"),
        "kol_win_rate": token_entry.get("_rt_kol_win_rate"),
        "rt_score": token_entry.get("_rt_score"),
        "rt_liquidity_usd": token_entry.get("_rt_liquidity_usd"),
        "rt_volume_24h": token_entry.get("_rt_volume_24h"),
        "rt_buy_sell_ratio": token_entry.get("_rt_buy_sell_ratio"),
        "rt_token_age_hours": token_entry.get("_rt_token_age_hours"),
        # v14e.42: cast to int (DB col is int2, not bool). A truthy bool used to
        # crash the insert with "invalid input syntax for type integer", which is
        # exactly how $INCOME got orphaned (buy on-chain, no DB row).
        "rt_is_pump_fun": int(bool(token_entry.get("_rt_is_pump_fun"))),
        "message_ts": token_entry.get("_rt_message_ts"),
        "price_at_message": token_entry.get("_rt_price_at_message"),
        "message_to_buy_seconds": _calc_message_to_buy(token_entry.get("_rt_message_ts")),
        "buy_slippage_bps": actual_slippage_bps,
        "buy_fee_bps": slippage,
        # v14e.32+: fine-grained instrumentation for paired-test calibration
        # gas_usd_buy = real gas paid (different from buy_fee_bps which is the
        # SLIPPAGE TOLERANCE asked, not gas). quote_slip_bps_buy = Uniswap quote
        # vs actual fill (router-internal slippage from price impact); distinct
        # from buy_slippage_bps which is DexScreener-mid vs fill (data quality).
        # block_number_buy = chain block at receipt time (latency proxy).
        "gas_usd_buy": float(result.get("gas_usd") or 0),
        "quote_slip_bps_buy": int(result.get("slippage_actual_bps") or 0),
        "block_number_buy": int(result.get("block_number") or 0),
        # Native chain price/amount columns repurposed for ETH (semantics by chain col)
        "sol_price_at_entry": eth_usd,                    # ETH price USD
        "position_sol": round(eth_spent, 6),              # ETH amount
        "buy_exec_ms": result.get("exec_ms"),
        # v14e.42: bigint cap (2^63 ~ 9.22e18). Low-priced ETH tokens with 18
        # decimals routinely exceed this on raw amounts (e.g. $INCOME held 29.13
        # tokens = 2.91e19 raw). Drop the raw and keep a downscaled human value
        # in position_sol if needed. The buy tx hash (tx_signature) preserves the
        # ground truth — raw amounts can be re-derived from chain if ever needed.
        "buy_input_lamports": (eth_spent_wei
                                if eth_spent_wei < 9_000_000_000_000_000_000 else None),
        "buy_output_tokens": (result.get("tokens_received")
                               if (result.get("tokens_received") or 0) < 9_000_000_000_000_000_000
                               else None),
        "dex_spot_price_at_entry": execution_price,
        "high_price_seen": execution_price,
        "entry_source": "uniswap_v3",
        "pair_address": token_entry.get("_rt_pair_address"),
    }

    # v14e.32+ instrumentation cols — drop if DB migration not yet applied.
    _OPTIONAL_BUY = ("gas_usd_buy", "quote_slip_bps_buy", "block_number_buy")
    try:
        client_sb.table("paper_trades").insert(row).execute()
        logger.info(
            "ETH LIVE OPENED: %s %s @ $%.10f | %.6f ETH ($%.2f) | route=%s | tx: %s | gas $%.2f",
            symbol, strategy, execution_price, eth_spent, position_usd,
            result.get("route") or "?",
            result["tx_hash"][:14], result.get("gas_usd") or 0,
        )
    except Exception as e:
        err_str = str(e)
        stripped = [c for c in _OPTIONAL_BUY if c in err_str and c in row]
        if stripped:
            for c in stripped:
                row.pop(c, None)
            try:
                client_sb.table("paper_trades").insert(row).execute()
                logger.warning("ETH LIVE OPENED (instrum cols dropped: %s): %s %s tx=%s",
                               stripped, symbol, strategy, result["tx_hash"][:14])
            except Exception as e2:
                logger.error(
                    "CRITICAL: ETH live trade %s bought (tx=%s) but DB insert failed even after strip: %s",
                    symbol, result.get("tx_hash"), e2)
                return _FAIL
        else:
            logger.error(
                "CRITICAL: ETH live trade %s bought (tx=%s) but DB insert failed: %s. "
                "Position untracked, manual recovery needed.",
                symbol, result.get("tx_hash"), e,
            )
            return _FAIL

    return {
        "success": True,
        "execution_price": execution_price,
        "tx_hash": result["tx_hash"],
        "gas_usd": result.get("gas_usd"),
    }


def _finalize_orphan_eth_sell(client_sb, trade: dict) -> bool:
    """Recover a stuck status='closing' trade whose sell already mined on-chain
    but whose DB row never got patched (process crash between sell-mine and
    .update()). Without this recovery, the retry path in process_open_trades
    re-submits the sell every cycle; the wallet has 0 tokens so each tx reverts,
    burning gas indefinitely (~$0.50-2/cycle until manual intervention).

    Detection rule: status='closing' AND tx_signature_exit IS NULL AND wallet
    on-chain balance == 0. We then scan ERC20 Transfer logs from the wallet
    for this token in the recent block window, find the sell tx, parse the
    ETH received, and finalize the row + bankroll.

    Returns True on successful finalize (caller skips further work for the
    trade), False if the wallet still holds tokens (real retry path applies)
    or if recovery isn't possible (logs missing, decode failure → manual fix).

    Concrete instance: $ALIENPEPE trade 251132, Apr 27 2026 — sell mined block
    24974330 but DB stayed in 'closing' until manually resynced via
    scripts/_eth_alienpepe_db_resync.py.
    """
    if trade.get("status") != "closing" or trade.get("tx_signature_exit"):
        return False
    addr = trade.get("token_address")
    if not addr:
        return False
    try:
        w3r, _, acct = _client()
    except Exception as e:
        logger.debug("orphan-finalize: _client failed for %s: %s",
                     trade.get("symbol"), e)
        return False
    wallet = acct.address

    try:
        erc20 = w3r.eth.contract(address=_to_checksum(addr), abi=ERC20_ABI)
        bal = erc20.functions.balanceOf(wallet).call()
    except Exception as e:
        logger.warning("orphan-finalize: balanceOf failed for %s: %s",
                       trade.get("symbol"), e)
        return False
    if bal > 0:
        return False  # tokens still held → genuine retry path

    # Wallet drained — find the sell tx via Transfer logs from wallet→pool.
    try:
        from_block = max(int(trade.get("block_number_buy") or 0) - 5, 0)
        if from_block == 0:
            from_block = max(w3r.eth.block_number - 2400, 0)  # ~8h ETH window
        addr_topic = "0x" + wallet.lower().replace("0x", "").rjust(64, "0")
        logs = w3r.eth.get_logs({
            "address": _to_checksum(addr),
            "fromBlock": from_block,
            "toBlock": "latest",
            "topics": [ERC20_TRANSFER_TOPIC, addr_topic],
        })
    except Exception as e:
        logger.warning("orphan-finalize: get_logs failed for %s: %s",
                       trade.get("symbol"), e)
        return False
    if not logs:
        logger.error(
            "orphan-finalize: %s wallet 0 tokens but NO Transfer logs from wallet "
            "in window — possible honeypot tax or manual move, manual fix needed.",
            trade.get("symbol"),
        )
        return False

    sell_log = max(logs, key=lambda l: int(l["blockNumber"]) if isinstance(
        l["blockNumber"], int) else int(l["blockNumber"], 16))
    tx_hash = sell_log["transactionHash"]
    if hasattr(tx_hash, "hex"):
        tx_hash = tx_hash.hex()
    if not tx_hash.startswith("0x"):
        tx_hash = "0x" + tx_hash

    try:
        receipt = w3r.eth.get_transaction_receipt(tx_hash)
        block = w3r.eth.get_block(receipt["blockNumber"])
    except Exception as e:
        logger.warning("orphan-finalize: receipt fetch failed for %s tx %s: %s",
                       trade.get("symbol"), tx_hash[:12], e)
        return False

    # Try V3 (multicall: WETH→ETH unwrap to wallet) then V2 (WETH transfer
    # pair→router); fall back to trace_transaction for native ETH internal txs.
    eth_received_wei = _parse_swap_output_from_logs(receipt, WETH)
    if not eth_received_wei:
        eth_received_wei = _parse_transfer_to_recipient(receipt, WETH, UNISWAP_V2_ROUTER)
    if not eth_received_wei:
        try:
            tr = w3r.provider.make_request("trace_transaction", [tx_hash])
            for t in (tr.get("result") or []):
                a = t.get("action") or {}
                if (a.get("to", "").lower() == wallet.lower()
                        and a.get("callType") == "call"):
                    v = a.get("value", "0x0")
                    if isinstance(v, str):
                        v = int(v, 16)
                    eth_received_wei = (eth_received_wei or 0) + v
        except Exception:
            pass
    if not eth_received_wei or eth_received_wei <= 0:
        logger.error(
            "orphan-finalize: %s found sell tx %s but cannot decode ETH received "
            "via V3/V2/trace — manual fix needed.",
            trade.get("symbol"), tx_hash[:12],
        )
        return False

    # Chainlink ETH/USD at the actual sell block (matches what live_trader
    # would have computed in real-time, not the current price).
    eth_usd_at_sell = None
    try:
        AGG = "0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419"
        ABI = [{"inputs": [], "name": "latestRoundData", "outputs": [
            {"name": "r", "type": "uint80"}, {"name": "answer", "type": "int256"},
            {"name": "s", "type": "uint256"}, {"name": "u", "type": "uint256"},
            {"name": "a", "type": "uint80"},
        ], "stateMutability": "view", "type": "function"}]
        c = w3r.eth.contract(address=_to_checksum(AGG), abi=ABI)
        eth_usd_at_sell = c.functions.latestRoundData().call(
            block_identifier=int(receipt["blockNumber"]))[1] / 1e8
    except Exception:
        try:
            eth_usd_at_sell = _eth_usd_price(w3r)
        except Exception:
            return False

    eth_received = eth_received_wei / 1e18
    usd_received = eth_received * eth_usd_at_sell
    pos_usd = float(trade.get("position_usd") or 0)
    entry_price = float(trade.get("entry_price") or 0)
    if pos_usd <= 0 or entry_price <= 0:
        return False
    actual_exit_price = entry_price * (usd_received / pos_usd)
    pnl_pct = round((usd_received / pos_usd) - 1, 4)
    pnl_usd = round(pos_usd * pnl_pct, 2)
    gas_paid_wei = int(receipt["gasUsed"]) * int(receipt["effectiveGasPrice"])
    gas_usd = (gas_paid_wei / 1e18) * eth_usd_at_sell

    exit_at_dt = datetime.fromtimestamp(int(block["timestamp"]), tz=timezone.utc)
    elapsed_minutes = 0
    ct = trade.get("created_at")
    if ct:
        try:
            ctd = (datetime.fromisoformat(ct.replace("Z", "+00:00"))
                   if isinstance(ct, str) else ct)
            elapsed_minutes = int((exit_at_dt - ctd).total_seconds() / 60)
        except Exception:
            pass

    # Best-effort exit-reason classification from price action we DO have.
    high = float(trade.get("high_price_seen") or 0)
    tp_price = float(trade.get("tp_price") or 0)
    sl_price = float(trade.get("sl_price") or 0)
    horizon = int(trade.get("horizon_minutes") or 0)
    if tp_price and high >= tp_price:
        new_status = "tp_hit"
    elif sl_price and actual_exit_price <= sl_price:
        new_status = "sl_hit"
    elif horizon and elapsed_minutes >= horizon:
        new_status = "timeout"
    else:
        new_status = "closing_recovered"

    update = {
        "status": new_status,
        "exit_price": actual_exit_price,
        "exit_at": exit_at_dt.isoformat(),
        "pnl_pct": pnl_pct,
        "pnl_usd": pnl_usd,
        "exit_minutes": elapsed_minutes,
        "tx_signature_exit": tx_hash,
        "sell_slippage_bps": 0,
        "slippage_actual_bps": int(trade.get("buy_slippage_bps") or 0),
        "sol_price_at_exit": eth_usd_at_sell,
        # v14e.42: bigint cap on raw ETH wei (rare for sells, but defensive).
        "sell_output_lamports": (int(eth_received_wei)
                                  if eth_received_wei < 9_000_000_000_000_000_000 else None),
        "sell_sol_received": round(eth_received, 6),
        "gas_usd_sell": gas_usd,
        "block_number_sell": int(receipt["blockNumber"]),
    }

    _OPTIONAL = ("gas_usd_sell", "block_number_sell", "sell_output_lamports")
    for _ in range(3):
        try:
            client_sb.table("paper_trades").update(update).eq("id", trade["id"]).execute()
            break
        except Exception as e:
            s = str(e)
            stripped = False
            for col in _OPTIONAL:
                if col in s and col in update:
                    update.pop(col, None)
                    stripped = True
            if not stripped:
                logger.error("orphan-finalize: DB update failed for %s: %s",
                             trade.get("symbol"), e)
                return False

    try:
        from safe_scraper import _rt_update_bankroll
        _rt_update_bankroll(pnl_usd, 1, strategy=trade.get("strategy", ""),
                            chain="ethereum")
    except Exception as e:
        logger.warning("orphan-finalize: bankroll update failed for %s: %s",
                       trade.get("symbol"), e)

    # v14e.43 (T4): track day PnL so the daily-loss gate trips
    _track_eth_pnl(pnl_usd)

    logger.info(
        "ETH ORPHAN-RECOVERED: %s %s exit=%.10f eth=%.6f usd=$%.2f "
        "pnl=%+.2f%% tx=%s block=%d",
        trade.get("symbol"), new_status, actual_exit_price, eth_received,
        usd_received, pnl_pct * 100, tx_hash[:12], int(receipt["blockNumber"]),
    )
    return True


def check_live_trades_eth(client_sb) -> dict:
    """Close-side mirror of live_trader.check_live_trades for ETH chain.

    Polls open ETH live trades, evaluates exit via paper_trader._evaluate_trade_exit
    (shared logic), executes Uniswap V3 sell on TP/SL/timeout, updates DB.

    Returns {"checked": N, "closed": M, "tp": X, "sl": Y, "timeout": Z, "pnl_usd": ...}.
    """
    from paper_trader import (
        _fetch_prices_batch, _evaluate_trade_exit,
        _strategy_orchestration, _should_poll_trade, _decision_price,
        _record_eval_poll, _flush_eval_history, _log_price_ticks,
    )

    now = datetime.now(timezone.utc)
    result_counts = {
        "checked": 0, "closed": 0, "tp": 0, "sl": 0, "timeout": 0,
        "pnl_usd": 0.0, "rt_pnl_usd": 0.0,
    }

    _rt_cfg_orch = {}
    try:
        from safe_scraper import _rt_load_config as _rt_load
        _rt_cfg_orch = _rt_load() or {}
    except Exception:
        pass

    try:
        result = (
            client_sb.table("paper_trades")
            .select("*")
            # v14e.42: include 'closing_retry'. Pre-fix, a successful retry sell
            # wrote status='closing_retry' (sentinel leak) and the row dropped out
            # of this scan forever, leaving 6 ETH trades silently orphaned even
            # though their sells had landed on-chain.
            .in_("status", ["open", "closing", "closing_retry"])
            .eq("source", "rt_live")
            .eq("chain", "ethereum")
            .execute()
        )
        open_trades = result.data or []
    except Exception as e:
        logger.error("live_trader_eth: failed to fetch open trades: %s", e)
        return result_counts

    if not open_trades:
        return result_counts

    result_counts["checked"] = len(open_trades)

    addresses = list({t["token_address"] for t in open_trades})
    chain_map = {a: "ethereum" for a in addresses}
    prices = _fetch_prices_batch(addresses, chain_by_addr=chain_map)
    _log_price_ticks(client_sb, prices, "live", live_tokens=set(addresses),
                     chain_by_addr=chain_map)

    # Fetch ETH/USD once per cycle
    try:
        w3r, _, _ = _client()
        eth_usd = _eth_usd_price(w3r)
    except Exception as e:
        logger.error("live_trader_eth: cannot fetch ETH price: %s — abort cycle", e)
        return result_counts

    for trade in open_trades:
        addr = trade["token_address"]
        strategy = trade.get("strategy", "")
        trade_id = trade.get("id")
        current_price = prices.get(addr)
        entry_price = float(trade.get("entry_price") or 0)

        # Orphan-sell recovery: detect stuck 'closing' rows whose sell already
        # mined on-chain (wallet drained) and finalize them from receipt logs
        # instead of re-submitting a sell that would revert (no tokens to sell).
        if _finalize_orphan_eth_sell(client_sb, trade):
            result_counts["closed"] += 1
            continue

        _paper_sim_ev = None
        # v14e.42: stuck 'closing' OR 'closing_retry' — same retry path. Pre-fix,
        # a successful retry wrote 'closing_retry' (sentinel leak) and the row
        # dropped out of the scan filter forever. Now both stuck states are
        # handled identically: skip eval, force-sell, derive terminal status
        # post-fill from pnl vs strategy thresholds (see below).
        if trade.get("status") in ("closing", "closing_retry"):
            logger.info("live_trader_eth: retrying sell for stuck '%s' trade %s",
                        trade["status"], trade["symbol"])
            ev = {"status": "force_close",
                  "exit_price": current_price or entry_price}
            decision_price = current_price or entry_price
        else:
            orch = _strategy_orchestration(strategy, _rt_cfg_orch)
            if not _should_poll_trade(trade_id, int(orch.get("polling_sec", 30))):
                continue
            decision_price, exit_ref = _decision_price(addr, strategy, trade_id, orch, trade=trade)
            if exit_ref is not None:
                current_price = exit_ref

            if current_price is None:
                logger.warning("live_trader_eth: no price for %s — exit eval skipped",
                               trade["symbol"])

            _record_eval_poll(trade_id, now, decision_price, current_price,
                              float(trade.get("high_price_seen") or 0))

            ev = _evaluate_trade_exit(trade, current_price, now, sell_slip_factor=1.0,
                                       sell_fee_bps=0, decision_price=decision_price)

            try:
                from paper_trader import (
                    SELL_FEE_BPS as _PT_SELL_FEE_BPS,
                    SELL_SLIPPAGE_BPS as _PT_SELL_SLIP_BPS,
                )
                _paper_slip_factor = 1 - _PT_SELL_SLIP_BPS / 10_000
                _paper_sim_ev = _evaluate_trade_exit(
                    trade, current_price, now,
                    sell_slip_factor=_paper_slip_factor,
                    sell_fee_bps=_PT_SELL_FEE_BPS,
                    decision_price=decision_price,
                )
            except Exception:
                pass

        if ev is None:
            continue

        if ev.get("high_price_seen") is not None:
            new_high = ev["high_price_seen"]
            old_high = float(trade.get("high_price_seen") or 0)
            if new_high > old_high:
                try:
                    client_sb.table("paper_trades").update(
                        {"high_price_seen": new_high}
                    ).eq("id", trade["id"]).execute()
                except Exception:
                    pass

        new_status = ev.get("status")
        if new_status is None:
            continue

        decision_exit_price = ev.get("exit_price")
        paper_exit_price = (
            _paper_sim_ev.get("exit_price")
            if (_paper_sim_ev and _paper_sim_ev.get("exit_price"))
            else decision_exit_price
        )

        elapsed_minutes = ev.get("exit_minutes", 0)
        if not elapsed_minutes:
            created = trade.get("created_at")
            if created:
                try:
                    ct = (datetime.fromisoformat(created.replace("Z", "+00:00"))
                          if isinstance(created, str) else created)
                    elapsed_minutes = int((now - ct).total_seconds() / 60)
                except Exception:
                    elapsed_minutes = 0

        # Atomic claim
        if trade.get("status") not in ("closing", "closing_retry"):
            try:
                claim = (
                    client_sb.table("paper_trades")
                    .update({"status": "closing"})
                    .eq("id", trade["id"])
                    .eq("status", "open")
                    .execute()
                )
                if not claim.data:
                    logger.info("live_trader_eth: trade %s already claimed elsewhere",
                                trade["symbol"])
                    continue
            except Exception as e:
                logger.warning("live_trader_eth: claim failed for %s: %s",
                               trade["symbol"], e)
                continue

        # Execute Uniswap V3 sell
        # v14e.43: read sell tolerance from live_trading.eth_sell_slippage_bps
        # (was hardcoded 500 — ignored JSONB). Default raised to 600 to match
        # the empirical p75 with margin; legit dumping memecoins routinely
        # exceed 500 bps router-internal slip.
        sell_amount = int(trade.get("buy_output_tokens") or 0) or None
        _lt_cfg = _rt_cfg_orch.get("live_trading", {}) if isinstance(_rt_cfg_orch, dict) else {}
        _sell_slip = int(_lt_cfg.get("eth_sell_slippage_bps", 600))
        sell_result = execute_sell(addr, amount_tokens=sell_amount, slippage_bps=_sell_slip)

        if not sell_result.get("success"):
            try:
                cur_attempts = int(trade.get("sell_attempts") or 1)
                client_sb.table("paper_trades").update(
                    {"status": "open", "sell_attempts": cur_attempts + 1}
                ).eq("id", trade["id"]).execute()
            except Exception:
                pass
            logger.warning(
                "live_trader_eth: sell failed for %s — reverted to open: %s",
                trade["symbol"], sell_result.get("error"),
            )
            try:
                from alerter import alert_live_trade_failed
                alert_live_trade_failed(trade["symbol"], "SELL (ETH)",
                                         sell_result.get("error", "unknown"))
            except Exception:
                pass
            continue

        # Compute true PnL from ETH received
        eth_received_wei = sell_result.get("eth_received") or 0
        eth_received = eth_received_wei / 1e18 if eth_received_wei else 0
        usd_received = eth_received * (sell_result.get("eth_usd") or eth_usd)
        pos_usd = float(trade.get("position_usd") or 0)

        if pos_usd > 0 and entry_price > 0:
            actual_exit_price = entry_price * (usd_received / pos_usd)
        else:
            actual_exit_price = decision_exit_price

        pnl_pct = round((actual_exit_price / entry_price) - 1, 4) if (actual_exit_price and entry_price) else 0
        pnl_usd = round(pos_usd * pnl_pct, 2) if pos_usd else 0

        # v14e.42: when force-closing a stuck trade, derive a real terminal status
        # from the actual fill vs strategy thresholds. The original eval status was
        # lost when status flipped to 'closing'/'closing_retry'; reconstruct it
        # from pnl_pct so livestats and outlier monitor pick it up correctly.
        if new_status == "force_close":
            try:
                from paper_trader import STRATEGIES as _STR
                _tr = (_STR.get(strategy) or [{}])[0]
                _tp_mult = _tr.get("tp_mult")
                _sl_mult = _tr.get("sl_mult", 0.0)
                if _tp_mult and pnl_pct >= (_tp_mult - 1):
                    new_status = "tp_hit"
                elif _sl_mult and pnl_pct <= (_sl_mult - 1):
                    new_status = "sl_hit"
                else:
                    new_status = "timeout"
            except Exception:
                new_status = "timeout"
            logger.info("live_trader_eth: force_close %s/%s resolved -> %s (pnl=%+.2f%%)",
                        trade["symbol"], strategy, new_status, pnl_pct * 100)

        sell_slippage_bps = 0
        if actual_exit_price and current_price and current_price > 0:
            sell_slippage_bps = round((actual_exit_price / current_price - 1) * 10000)

        price_divergence_pct = None
        if (paper_exit_price and paper_exit_price > 0
                and actual_exit_price and actual_exit_price > 0):
            price_divergence_pct = round((actual_exit_price / paper_exit_price) - 1, 4)

        update = {
            "status": new_status,
            "exit_price": actual_exit_price,
            "exit_at": now.isoformat(),
            "pnl_pct": pnl_pct,
            "pnl_usd": pnl_usd,
            "exit_minutes": int(elapsed_minutes),
            "tx_signature_exit": sell_result["tx_hash"],
            "sell_slippage_bps": sell_slippage_bps,
            "slippage_actual_bps": int((trade.get("buy_slippage_bps") or 0)) + int(sell_slippage_bps or 0),
            "sol_price_at_exit": sell_result.get("eth_usd") or eth_usd,  # repurposed: ETH price
            "sell_exec_ms": sell_result.get("exec_ms"),
            # v14e.42: bigint cap on raw wei (defensive; same fix as buy-side).
            "sell_output_lamports": (eth_received_wei
                                      if eth_received_wei < 9_000_000_000_000_000_000 else None),
            "sell_sol_received": round(eth_received, 6) if eth_received else None,  # repurposed: ETH amount
            "paper_exit_price": paper_exit_price,
            "price_divergence_pct": price_divergence_pct,
            "paper_sim_pnl_pct": (
                round(float(_paper_sim_ev.get("pnl_pct")), 4)
                if _paper_sim_ev and _paper_sim_ev.get("pnl_pct") is not None else None
            ),
            # v14e.32+: fine-grained instrumentation (mirror of buy-side fields).
            "gas_usd_sell": float(sell_result.get("gas_usd") or 0),
            "quote_slip_bps_sell": int(sell_result.get("slippage_actual_bps") or 0),
            "block_number_sell": int(sell_result.get("block_number") or 0),
        }
        hist = _flush_eval_history(trade["id"])
        if hist:
            update["eval_history"] = hist

        # v14e.32+ instrumentation cols. Strip if migration not yet applied
        # (defensive — keeps insert working even if DB schema is behind code).
        _OPTIONAL_COLS = ("paper_sim_pnl_pct", "gas_usd_sell",
                          "quote_slip_bps_sell", "block_number_sell")
        db_updated = False
        for attempt in range(3):
            try:
                client_sb.table("paper_trades").update(update).eq("id", trade["id"]).execute()
                db_updated = True
                break
            except Exception as e:
                err_str = str(e)
                stripped = False
                for col in _OPTIONAL_COLS:
                    if col in err_str and col in update:
                        update.pop(col, None)
                        stripped = True
                if stripped:
                    try:
                        client_sb.table("paper_trades").update(update).eq("id", trade["id"]).execute()
                        db_updated = True
                        break
                    except Exception as e2:
                        e = e2
                logger.warning("live_trader_eth: DB update attempt %d/3 failed for trade %s: %s",
                               attempt + 1, trade["id"], e)
                if attempt < 2:
                    time.sleep(2 ** attempt)

        if not db_updated:
            logger.error(
                "CRITICAL: ETH live trade %s sold (tx=%s) but DB update failed",
                trade["symbol"], sell_result["tx_hash"],
            )
            continue

        # Paper exit shadow-sync (mirror of v143.5)
        try:
            cycle_ts = trade.get("cycle_ts")
            paper_match_q = (
                client_sb.table("paper_trades")
                .select("id, entry_price, position_usd")
                .eq("source", "rt")
                .eq("token_address", trade["token_address"])
                .eq("strategy", trade["strategy"])
                .eq("status", "open")
            )
            if cycle_ts:
                paper_match_q = paper_match_q.eq("cycle_ts", cycle_ts)
            paper_match = paper_match_q.limit(1).execute().data
            if paper_match:
                pm = paper_match[0]
                pm_entry = float(pm.get("entry_price") or 0) or entry_price
                pm_pos_usd = float(pm.get("position_usd") or 0)
                pm_pnl_pct = (actual_exit_price / pm_entry) - 1 if pm_entry > 0 else 0
                pm_pnl_usd = pm_pos_usd * pm_pnl_pct
                paper_update = {
                    "status": new_status,
                    "exit_price": actual_exit_price,
                    "exit_at": now.isoformat(),
                    "pnl_pct": round(pm_pnl_pct, 4),
                    "pnl_usd": round(pm_pnl_usd, 2),
                    "exit_minutes": int(elapsed_minutes),
                    "sol_price_at_exit": sell_result.get("eth_usd") or eth_usd,
                }
                hist_p = _flush_eval_history(pm["id"])
                if hist_p:
                    paper_update["eval_history"] = hist_p
                client_sb.table("paper_trades").update(paper_update).eq("id", pm["id"]).execute()
        except Exception as _e:
            logger.debug("live_trader_eth: paper_exit_sync skipped for %s: %s",
                         trade.get("symbol"), _e)

        result_counts["closed"] += 1
        result_counts["pnl_usd"] += pnl_usd
        result_counts["rt_pnl_usd"] += pnl_usd
        status_key = new_status.replace("_hit", "").replace("_stop", "")
        if status_key in result_counts:
            result_counts[status_key] += 1
        # v14e.43 (T4): accumulate day PnL so the daily-loss gate trips
        _track_eth_pnl(pnl_usd)

        logger.info(
            "ETH LIVE CLOSED: %s %s @ $%.10f | %.6f ETH ($%.2f) | route=%s | pnl=%+.2f%% (gas=$%.2f)",
            trade["symbol"], new_status, actual_exit_price, eth_received,
            usd_received, sell_result.get("route") or "?",
            pnl_pct * 100, sell_result.get("gas_usd") or 0,
        )

    return result_counts
