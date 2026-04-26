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


_w3_read = None
_w3_write = None
_account = None
_approval_cache: set[str] = set()


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
    """Try fee tiers in order, return (amount_out, fee_tier, gas_estimate) or (None, None, None)."""
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
              tokens_received (raw), tokens_received_human, slippage_actual_bps,
              fee_tier_used, exec_ms, block_number, eth_spent_wei, eth_usd}
    """
    w3r, w3w, acct = _client()
    eth_usd = _eth_usd_price(w3r)
    eth_amount = amount_usd / eth_usd
    amount_in_wei = w3r.to_wei(eth_amount, "ether")

    # 1. Quote with auto fee-tier discovery
    amount_out, fee_tier, gas_est = _quote_with_best_fee(w3r, amount_in_wei, WETH, ca)
    if amount_out is None:
        return {"success": False, "error": "no Uniswap V3 pool found across fee tiers"}

    min_out = (amount_out * (10000 - slippage_bps)) // 10000

    # 2. Build + sign tx
    base_fee = w3r.eth.get_block("latest")["baseFeePerGas"]
    priority = w3r.to_wei(2, "gwei")
    max_fee = base_fee * 2 + priority
    router = w3r.eth.contract(address=_to_checksum(SWAP_ROUTER_02), abi=ROUTER_ABI)
    params = (_to_checksum(WETH), _to_checksum(ca), fee_tier, acct.address,
              amount_in_wei, min_out, 0)
    nonce = w3r.eth.get_transaction_count(acct.address)
    tx = router.functions.exactInputSingle(params).build_transaction({
        "from": acct.address, "value": amount_in_wei, "nonce": nonce,
        "gas": gas_est + 80_000, "maxFeePerGas": max_fee,
        "maxPriorityFeePerGas": priority, "chainId": 1,
    })

    erc20 = w3r.eth.contract(address=_to_checksum(ca), abi=ERC20_ABI)
    pre_bal = erc20.functions.balanceOf(acct.address).call()
    try:
        decimals = erc20.functions.decimals().call()
    except Exception:
        decimals = 18

    t0 = time.time()
    signed = acct.sign_transaction(tx)
    # Submit via Flashbots Protect for MEV protection. Receipt fetched via read RPC.
    tx_hash = w3w.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3r.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
    elapsed_ms = int((time.time() - t0) * 1000)

    if receipt["status"] != 1:
        gas_paid_eth = w3r.from_wei(receipt["gasUsed"] * receipt["effectiveGasPrice"], "ether")
        return {"success": False, "error": "tx reverted",
                "tx_hash": tx_hash.hex(), "gas_usd": float(gas_paid_eth) * eth_usd,
                "exec_ms": elapsed_ms}

    post_bal = erc20.functions.balanceOf(acct.address).call()
    received = post_bal - pre_bal
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
        "exec_ms": elapsed_ms,
        "block_number": receipt["blockNumber"],
        "eth_spent_wei": amount_in_wei,
        "eth_usd": eth_usd,
    }


def _ensure_approval(w3r, w3w, acct, token_address: str, amount_min: int):
    """Approve SwapRouter02 if not already done. Idempotent + cached."""
    if token_address in _approval_cache:
        return
    erc20 = w3r.eth.contract(address=_to_checksum(token_address), abi=ERC20_ABI)
    current = erc20.functions.allowance(acct.address, _to_checksum(SWAP_ROUTER_02)).call()
    if current >= amount_min:
        _approval_cache.add(token_address)
        return
    base_fee = w3r.eth.get_block("latest")["baseFeePerGas"]
    priority = w3r.to_wei(2, "gwei")
    nonce = w3r.eth.get_transaction_count(acct.address)
    MAX_UINT = 2**256 - 1
    tx = erc20.functions.approve(_to_checksum(SWAP_ROUTER_02), MAX_UINT).build_transaction({
        "from": acct.address, "nonce": nonce, "gas": 80_000,
        "maxFeePerGas": base_fee * 2 + priority,
        "maxPriorityFeePerGas": priority, "chainId": 1,
    })
    signed = acct.sign_transaction(tx)
    tx_hash = w3w.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3r.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
    if receipt["status"] != 1:
        raise RuntimeError(f"ERC20 approve failed for {token_address}: {tx_hash.hex()}")
    _approval_cache.add(token_address)
    logger.info("live_trader_eth: approved router for %s (tx %s)",
                token_address, tx_hash.hex())


def execute_sell(ca: str, amount_tokens: Optional[int] = None,
                 slippage_bps: int = 500) -> dict:
    """Memecoin → ETH swap via Uniswap V3.

    amount_tokens=None → sell entire balance. Default slippage 500 bps (5%) on
    sells because dumping shallow-pool tokens moves price more than the entry
    quote suggests.

    Returns: {success, tx_hash, eth_received (wei), eth_received_human (eth),
              eth_received_usd, gas_usd, fee_tier_used, exec_ms, block_number, eth_usd}
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

    _ensure_approval(w3r, w3w, acct, ca, amount_tokens)

    eth_out, fee_tier_used, gas_est = _quote_with_best_fee(w3r, amount_tokens, ca, WETH)
    if eth_out is None:
        return {"success": False, "error": "no liquidity for sell on any fee tier"}

    min_out = (eth_out * (10000 - slippage_bps)) // 10000
    base_fee = w3r.eth.get_block("latest")["baseFeePerGas"]
    priority = w3r.to_wei(2, "gwei")
    max_fee = base_fee * 2 + priority
    router = w3r.eth.contract(address=_to_checksum(SWAP_ROUTER_02), abi=ROUTER_ABI)
    params = (_to_checksum(ca), _to_checksum(WETH), fee_tier_used,
              acct.address, amount_tokens, min_out, 0)
    nonce = w3r.eth.get_transaction_count(acct.address)
    tx = router.functions.exactInputSingle(params).build_transaction({
        "from": acct.address, "value": 0, "nonce": nonce,
        "gas": gas_est + 80_000, "maxFeePerGas": max_fee,
        "maxPriorityFeePerGas": priority, "chainId": 1,
    })

    pre_eth = w3r.eth.get_balance(acct.address)
    t0 = time.time()
    signed = acct.sign_transaction(tx)
    tx_hash = w3w.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3r.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
    elapsed_ms = int((time.time() - t0) * 1000)

    if receipt["status"] != 1:
        return {"success": False, "error": "sell tx reverted",
                "tx_hash": tx_hash.hex(), "exec_ms": elapsed_ms}

    post_eth = w3r.eth.get_balance(acct.address)
    gas_paid = receipt["gasUsed"] * receipt["effectiveGasPrice"]
    eth_received_wei = post_eth - pre_eth + gas_paid  # net of gas
    eth_received_human = eth_received_wei / 1e18 if eth_received_wei else 0.0
    eth_received_usd = eth_received_human * eth_usd

    return {
        "success": True,
        "tx_hash": tx_hash.hex(),
        "eth_received": eth_received_wei,
        "eth_received_human": eth_received_human,
        "eth_received_usd": eth_received_usd,
        "gas_usd": float(w3r.from_wei(gas_paid, "ether")) * eth_usd,
        "fee_tier_used": fee_tier_used,
        "exec_ms": elapsed_ms,
        "block_number": receipt["blockNumber"],
        "eth_usd": eth_usd,
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

    Returns: {"success": bool, "execution_price": float|None, ...buy result}
    """
    _FAIL = {"success": False, "execution_price": None}
    ca = token_entry.get("token_address")
    symbol = token_entry.get("symbol", "???")

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

    # Max open positions check (shared budget across chains, gated by config)
    max_open = int(config.get("max_open_positions", 5))
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
        "rt_is_pump_fun": token_entry.get("_rt_is_pump_fun"),
        "message_ts": token_entry.get("_rt_message_ts"),
        "price_at_message": token_entry.get("_rt_price_at_message"),
        "message_to_buy_seconds": _calc_message_to_buy(token_entry.get("_rt_message_ts")),
        "buy_slippage_bps": actual_slippage_bps,
        "buy_fee_bps": slippage,
        # Native chain price/amount columns repurposed for ETH (semantics by chain col)
        "sol_price_at_entry": eth_usd,                    # ETH price USD
        "position_sol": round(eth_spent, 6),              # ETH amount
        "buy_exec_ms": result.get("exec_ms"),
        "buy_input_lamports": eth_spent_wei,              # wei
        "buy_output_tokens": result.get("tokens_received"),  # raw tokens
        "dex_spot_price_at_entry": execution_price,
        "high_price_seen": execution_price,
        "entry_source": "uniswap_v3",
        "pair_address": token_entry.get("_rt_pair_address"),
    }

    try:
        client_sb.table("paper_trades").insert(row).execute()
        logger.info(
            "ETH LIVE OPENED: %s %s @ $%.10f | %.6f ETH ($%.2f) | tx: %s | gas $%.2f",
            symbol, strategy, execution_price, eth_spent, position_usd,
            result["tx_hash"][:14], result.get("gas_usd") or 0,
        )
    except Exception as e:
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
            .in_("status", ["open", "closing"])
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

        _paper_sim_ev = None
        if trade.get("status") == "closing":
            logger.info("live_trader_eth: retrying sell for stuck 'closing' trade %s",
                        trade["symbol"])
            ev = {"status": "closing_retry",
                  "exit_price": current_price or entry_price}
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
        if trade.get("status") != "closing":
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
        sell_amount = int(trade.get("buy_output_tokens") or 0) or None
        sell_result = execute_sell(addr, amount_tokens=sell_amount, slippage_bps=500)

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
            "sell_output_lamports": eth_received_wei,                    # repurposed: wei
            "sell_sol_received": round(eth_received, 6) if eth_received else None,  # repurposed: ETH amount
            "paper_exit_price": paper_exit_price,
            "price_divergence_pct": price_divergence_pct,
            "paper_sim_pnl_pct": (
                round(float(_paper_sim_ev.get("pnl_pct")), 4)
                if _paper_sim_ev and _paper_sim_ev.get("pnl_pct") is not None else None
            ),
        }
        hist = _flush_eval_history(trade["id"])
        if hist:
            update["eval_history"] = hist

        db_updated = False
        for attempt in range(3):
            try:
                client_sb.table("paper_trades").update(update).eq("id", trade["id"]).execute()
                db_updated = True
                break
            except Exception as e:
                err_str = str(e)
                if "paper_sim_pnl_pct" in err_str and "paper_sim_pnl_pct" in update:
                    update.pop("paper_sim_pnl_pct", None)
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

        logger.info(
            "ETH LIVE CLOSED: %s %s @ $%.10f | %.6f ETH ($%.2f) | pnl=%+.2f%% (gas=$%.2f)",
            trade["symbol"], new_status, actual_exit_price, eth_received,
            usd_received, pnl_pct * 100, sell_result.get("gas_usd") or 0,
        )

    return result_counts
