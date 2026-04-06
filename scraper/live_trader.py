"""
v72: Live Trading Bot — Jupiter Ultra API execution.

Mirrors paper_trader.py logic but executes real swaps on Solana.
Runs in parallel with paper trading (source='rt_live' vs 'rt').
Graceful degradation: if SOLANA_PRIVATE_KEY is not set, all functions no-op.

Safety guards:
- max_position_sol: cap per trade
- max_open_positions: max concurrent live trades
- min_sol_reserve: always keep SOL for fees
- daily_loss_limit_sol: auto-disable buying for the day

MEV/Slippage (v105):
  Jupiter Ultra API uses RFQ (Request for Quote) — market makers give guaranteed
  fill prices, NOT routed through AMMs. This provides inherent MEV protection:
  - No sandwich attack surface (order goes directly to market maker)
  - Fill price is fixed at quote time (no AMM price impact)
  - slippageBps is optional tolerance on quoted price (default 300 buy, 500 sell)
"""

import os
import logging
import time
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)

WSOL_MINT = "So11111111111111111111111111111111111111112"
LAMPORTS_PER_SOL = 1_000_000_000


def _safe_int(val) -> int | None:
    """Convert Jupiter amount result to int, returning None on failure."""
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None

# --- Lazy singleton ---
_ultra_client = None
_ultra_client_init_attempted = False

# --- Loss tracking (v73: daily + weekly + monthly) ---
_daily_pnl_sol: float = 0.0
_daily_pnl_reset_date: str = ""
_daily_halted: bool = False
_weekly_pnl_sol: float = 0.0
_weekly_pnl_reset_week: str = ""
_weekly_halted: bool = False
_monthly_pnl_sol: float = 0.0
_monthly_pnl_reset_month: str = ""
_monthly_halted: bool = False


def _get_ultra_client():
    """Lazy-init Jupiter Ultra API client. Returns None if no private key."""
    global _ultra_client, _ultra_client_init_attempted
    if _ultra_client is not None:
        return _ultra_client
    if _ultra_client_init_attempted:
        return None
    _ultra_client_init_attempted = True

    private_key = os.environ.get("SOLANA_PRIVATE_KEY")
    if not private_key:
        logger.info("live_trader: SOLANA_PRIVATE_KEY not set — live trading disabled")
        return None

    try:
        from jup_python_sdk.clients.ultra_api_client import UltraApiClient

        api_key = os.environ.get("JUPITER_API_KEY")
        # SDK reads PRIVATE_KEY env var by default. Point it to our namespaced var
        # to avoid leaking the key under a generic env var name.
        kwargs = {"private_key_env_var": "SOLANA_PRIVATE_KEY"}
        if api_key:
            kwargs["api_key"] = api_key
        client = UltraApiClient(**kwargs)
        _ultra_client = client
        pubkey = client._get_public_key()
        logger.info("live_trader: Ultra client initialized (wallet: %s)", pubkey)
        return client
    except Exception as e:
        logger.error("live_trader: failed to init Ultra client: %s", e)
        return None


def get_wallet_balance() -> dict | None:
    """
    Fetch wallet SOL + token balances via Jupiter Ultra /holdings endpoint.
    Returns {"sol_balance": float, "token_balances": {mint: {"amount": int, "ui_amount": float}}}
    or None on failure.
    """
    client = _get_ultra_client()
    if not client:
        return None

    pubkey = client._get_public_key()
    api_key = os.environ.get("JUPITER_API_KEY", "")
    base_url = "https://api.jup.ag" if api_key else "https://lite-api.jup.ag"

    try:
        headers = {"x-api-key": api_key} if api_key else {}
        resp = requests.get(
            f"{base_url}/ultra/v1/holdings/{pubkey}",
            headers=headers,
            timeout=15,
        )
        if resp.status_code != 200:
            logger.warning("live_trader: holdings API %d: %s", resp.status_code, resp.text[:200])
            return None

        data = resp.json()
        sol_balance = 0.0
        token_balances = {}

        if isinstance(data, dict):
            # v117: Jupiter Ultra /holdings response format:
            # {amount, uiAmount, uiAmountString, tokens: {mint: {amount, uiAmount}}}
            # SOL balance is at root level, tokens are nested under "tokens" key
            if "uiAmount" in data:
                sol_balance = float(data.get("uiAmount", 0))
            tokens_dict = data.get("tokens", {})
            if isinstance(tokens_dict, dict):
                for mint, info in tokens_dict.items():
                    # v117: tokens can be a dict or a list of account objects
                    if isinstance(info, list):
                        # Sum across all accounts for this mint
                        total_amount = sum(int(acc.get("amount", 0)) for acc in info if isinstance(acc, dict))
                        total_ui = sum(float(acc.get("uiAmount", 0)) for acc in info if isinstance(acc, dict))
                        if total_amount > 0:
                            token_balances[mint] = {"amount": total_amount, "ui_amount": total_ui}
                    elif isinstance(info, dict):
                        ui_amount = float(info.get("uiAmount", 0))
                        amount = int(info.get("amount", 0))
                        if amount > 0:
                            token_balances[mint] = {"amount": amount, "ui_amount": ui_amount}

            # Fallback: old format where SOL is keyed by mint
            if sol_balance == 0:
                for mint, info in data.items():
                    if not isinstance(info, dict):
                        continue
                    if mint == "SOL" or mint == WSOL_MINT:
                        sol_balance = float(info.get("uiAmount", 0))
                    elif int(info.get("amount", 0)) > 0:
                        token_balances[mint] = {
                            "amount": int(info["amount"]),
                            "ui_amount": float(info.get("uiAmount", 0)),
                        }

        return {"sol_balance": sol_balance, "token_balances": token_balances}
    except Exception as e:
        logger.error("live_trader: holdings fetch failed: %s", e)
        return None


def _order_with_slippage(client, order_request, slippage_bps: int):
    """
    v105: Call Jupiter Ultra /order with slippageBps param.
    v117: Sign with solders directly (SDK signing produces invalid signatures).
    """
    import base64
    import base58 as _b58
    from solders.keypair import Keypair as _Keypair
    from solders.transaction import VersionedTransaction as _VTx

    params = order_request.to_dict()
    params["slippageBps"] = slippage_bps

    headers = client._get_headers()
    url = f"{client.base_url}/ultra/v1/order"
    response = client.client.get(url, params=params, headers=headers)
    response.raise_for_status()
    order_response = response.json()

    request_id = order_response["requestId"]
    tx_base64 = order_response["transaction"]

    # Sign with solders (bypasses buggy SDK signing)
    pk_str = os.environ.get("SOLANA_PRIVATE_KEY", "")
    kp = _Keypair.from_bytes(_b58.b58decode(pk_str))
    tx = _VTx.from_bytes(base64.b64decode(tx_base64))
    signed_tx = _VTx(tx.message, [kp])
    signed_b64 = base64.b64encode(bytes(signed_tx)).decode()

    # Execute via raw POST (SDK execute also broken)
    exec_resp = client.client.post(
        f"{client.base_url}/ultra/v1/execute",
        json={"signedTransaction": signed_b64, "requestId": request_id},
        headers={**headers, "Content-Type": "application/json"},
        timeout=30,
    )
    exec_resp.raise_for_status()
    return exec_resp.json()


def execute_buy(ca: str, amount_sol_lamports: int, slippage_bps: int = 300) -> dict:
    """
    Execute a buy swap: SOL → token via Jupiter Ultra.
    v105: Now passes slippageBps to the Ultra /order endpoint for price tolerance.
    Returns {"success": bool, "signature": str, "status": str, "error": str|None,
             "input_amount": int|None, "output_amount": int|None,
             "slippage_bps": int}
    """
    client = _get_ultra_client()
    if not client:
        return {"success": False, "signature": "", "error": "Ultra client not initialized"}

    try:
        from jup_python_sdk.models.ultra_api.ultra_order_request_model import UltraOrderRequest

        order = UltraOrderRequest(
            input_mint=WSOL_MINT,
            output_mint=ca,
            amount=amount_sol_lamports,
            taker=client._get_public_key(),
        )

        t0 = time.time()
        response = _order_with_slippage(client, order, slippage_bps)
        exec_ms = int((time.time() - t0) * 1000)

        status = response.get("status", "Unknown")
        signature = str(response.get("signature", ""))
        success = status == "Success"

        # v74: Extract actual fill amounts from Jupiter response
        input_amount = _safe_int(response.get("inputAmountResult"))
        output_amount = _safe_int(response.get("outputAmountResult"))

        if success:
            logger.info(
                "LIVE BUY: %s | %s SOL → %s tokens | slip=%dbps | %dms (sig: %s...)",
                ca[:12], amount_sol_lamports / LAMPORTS_PER_SOL,
                output_amount or "?", slippage_bps, exec_ms, signature[:16],
            )
        else:
            logger.warning(
                "LIVE BUY FAILED: %s | status=%s code=%s error=%s",
                ca[:12], status, response.get("code", ""), response.get("error", ""),
            )

        return {
            "success": success,
            "signature": signature,
            "status": status,
            "error": response.get("error") if not success else None,
            "input_amount": input_amount,
            "output_amount": output_amount,
            "slippage_bps": slippage_bps,
            "exec_ms": exec_ms,
        }
    except Exception as e:
        logger.error("LIVE BUY ERROR: %s | %s", ca[:12], e)
        return {"success": False, "signature": "", "error": str(e)}


def execute_sell(ca: str, amount_tokens: int | None = None, slippage_bps: int = 500) -> dict:
    """
    Execute a sell swap: token → SOL via Jupiter Ultra.
    If amount_tokens is None, sells entire balance of that token.
    v105: Now passes slippageBps to the Ultra /order endpoint.
    Returns {"success": bool, "signature": str, "status": str, "error": str|None,
             "input_amount": int|None, "output_amount": int|None,
             "slippage_bps": int}
    """
    client = _get_ultra_client()
    if not client:
        return {"success": False, "signature": "", "error": "Ultra client not initialized"}

    try:
        # If no amount specified, fetch full balance
        if amount_tokens is None:
            balances = get_wallet_balance()
            if not balances:
                return {"success": False, "signature": "", "error": "Could not fetch balances"}
            token_info = balances["token_balances"].get(ca)
            if not token_info or token_info["amount"] <= 0:
                return {"success": False, "signature": "", "error": f"No balance for {ca[:12]}"}
            amount_tokens = token_info["amount"]

        from jup_python_sdk.models.ultra_api.ultra_order_request_model import UltraOrderRequest

        order = UltraOrderRequest(
            input_mint=ca,
            output_mint=WSOL_MINT,
            amount=amount_tokens,
            taker=client._get_public_key(),
        )

        t0 = time.time()
        response = _order_with_slippage(client, order, slippage_bps)
        exec_ms = int((time.time() - t0) * 1000)

        status = response.get("status", "Unknown")
        signature = str(response.get("signature", ""))
        success = status == "Success"

        # v74: Extract actual fill amounts
        input_amount = _safe_int(response.get("inputAmountResult"))
        output_amount = _safe_int(response.get("outputAmountResult"))

        if success:
            sol_received = output_amount / LAMPORTS_PER_SOL if output_amount else "?"
            logger.info(
                "LIVE SELL: %s | %d tokens → %s SOL | slip=%dbps | %dms (sig: %s...)",
                ca[:12], amount_tokens, sol_received, slippage_bps, exec_ms, signature[:16],
            )
            # v117: Close ATA to recover ~0.002 SOL rent
            try:
                _close_token_account(ca)
            except Exception:
                pass
        else:
            logger.warning(
                "LIVE SELL FAILED: %s | status=%s code=%s error=%s",
                ca[:12], status, response.get("code", ""), response.get("error", ""),
            )

        return {
            "success": success,
            "signature": signature,
            "status": status,
            "error": response.get("error") if not success else None,
            "input_amount": input_amount,
            "output_amount": output_amount,
            "slippage_bps": slippage_bps,
            "exec_ms": exec_ms,
        }
    except Exception as e:
        logger.error("LIVE SELL ERROR: %s | %s", ca[:12], e)
        return {"success": False, "signature": "", "error": str(e)}


def _close_token_account(ca: str) -> bool:
    """v117: Close the ATA for a token after selling all of it. Recovers ~0.002 SOL rent."""
    client = _get_ultra_client()
    if not client:
        return False
    try:
        import base58 as _b58
        from solders.keypair import Keypair as _Keypair
        from solders.pubkey import Pubkey as _Pubkey
        from solders.transaction import Transaction as _Tx
        from solders.message import Message as _Msg
        from solders.instruction import Instruction as _Ix, AccountMeta as _AM
        from solders.hash import Hash as _Hash

        pk_str = os.environ.get("SOLANA_PRIVATE_KEY", "")
        kp = _Keypair.from_bytes(_b58.b58decode(pk_str))
        owner = kp.pubkey()
        token_mint = _Pubkey.from_string(ca)

        # Derive ATA address
        TOKEN_PROGRAM = _Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
        ATA_PROGRAM = _Pubkey.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")
        ata, _bump = _Pubkey.find_program_address(
            [bytes(owner), bytes(TOKEN_PROGRAM), bytes(token_mint)],
            ATA_PROGRAM,
        )

        # Check if ATA has 0 balance before closing
        balances = get_wallet_balance()
        if balances and ca in balances.get("token_balances", {}):
            remaining = balances["token_balances"][ca].get("amount", 0)
            if remaining > 0:
                logger.debug("close_ata: %s still has %d tokens, skipping", ca[:12], remaining)
                return False

        # Build CloseAccount instruction (SPL Token instruction index 9)
        close_ix = _Ix(
            program_id=TOKEN_PROGRAM,
            accounts=[
                _AM(ata, is_signer=False, is_writable=True),      # account to close
                _AM(owner, is_signer=False, is_writable=True),    # destination for rent
                _AM(owner, is_signer=True, is_writable=False),    # owner/authority
            ],
            data=bytes([9]),  # CloseAccount instruction
        )

        # Get recent blockhash
        rpc_resp = requests.post(
            "https://api.mainnet-beta.solana.com",
            json={"jsonrpc": "2.0", "id": 1, "method": "getLatestBlockhash"},
            timeout=10,
        )
        blockhash_str = rpc_resp.json()["result"]["value"]["blockhash"]
        blockhash = _Hash.from_string(blockhash_str)

        # Build, sign, send
        msg = _Msg.new_with_blockhash([close_ix], owner, blockhash)
        tx = _Tx.new_unsigned(msg)
        tx.sign([kp], blockhash)
        tx_bytes = bytes(tx)

        import base64
        tx_b64 = base64.b64encode(tx_bytes).decode()
        send_resp = requests.post(
            "https://api.mainnet-beta.solana.com",
            json={
                "jsonrpc": "2.0", "id": 1,
                "method": "sendTransaction",
                "params": [tx_b64, {"encoding": "base64", "skipPreflight": True}],
            },
            timeout=15,
        )
        result = send_resp.json()
        if "result" in result:
            logger.info("close_ata: closed %s ATA, recovered rent (tx: %s...)",
                        ca[:12], result["result"][:16])
            return True
        else:
            logger.warning("close_ata: failed for %s: %s", ca[:12], result.get("error", {}).get("message", ""))
            return False
    except Exception as e:
        logger.debug("close_ata: error for %s: %s", ca[:12], e)
        return False


def _get_sol_price_usd() -> float:
    """Fetch current SOL/USD price from DexScreener."""
    try:
        resp = requests.get(
            "https://api.dexscreener.com/tokens/v1/solana/So11111111111111111111111111111111111111112",
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            pairs = data if isinstance(data, list) else data.get("pairs", [])
            if pairs:
                # Pick USDC pair (highest volume)
                best = max(pairs, key=lambda p: float(p.get("volume", {}).get("h24", 0) or 0))
                price = best.get("priceUsd")
                if price:
                    return float(price)
    except Exception as e:
        logger.warning("live_trader: SOL price fetch failed: %s", e)
    # v74: Dynamic fallback — try CoinGecko simple price before static value
    try:
        resp2 = requests.get(
            "https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd",
            timeout=5,
        )
        if resp2.status_code == 200:
            price = resp2.json().get("solana", {}).get("usd")
            if price:
                return float(price)
    except Exception:
        pass
    logger.warning("live_trader: all SOL price sources failed, using last-resort fallback")
    return 170.0  # Last-resort static fallback


def _check_loss_limits(config: dict) -> bool:
    """
    v73: Check daily + weekly + monthly loss limits.
    Returns True if trading should be halted.
    v105: Now sends Telegram alert when limit is hit.
    """
    global _daily_pnl_sol, _daily_pnl_reset_date, _daily_halted
    global _weekly_pnl_sol, _weekly_pnl_reset_week, _weekly_halted
    global _monthly_pnl_sol, _monthly_pnl_reset_month, _monthly_halted

    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    week = now.strftime("%Y-W%W")
    month = now.strftime("%Y-%m")

    # Reset counters on period change
    if _daily_pnl_reset_date != today:
        _daily_pnl_sol = 0.0
        _daily_pnl_reset_date = today
        _daily_halted = False
    if _weekly_pnl_reset_week != week:
        _weekly_pnl_sol = 0.0
        _weekly_pnl_reset_week = week
        _weekly_halted = False
    if _monthly_pnl_reset_month != month:
        _monthly_pnl_sol = 0.0
        _monthly_pnl_reset_month = month
        _monthly_halted = False

    daily_limit = float(config.get("daily_loss_limit_sol", 2.0))
    weekly_limit = float(config.get("weekly_loss_limit_sol", 5.0))
    monthly_limit = float(config.get("monthly_loss_limit_sol", 10.0))

    if _daily_pnl_sol < -daily_limit:
        if not _daily_halted:
            logger.warning("LIVE TRADING HALTED: daily loss %.4f SOL exceeds limit %.1f SOL",
                           _daily_pnl_sol, daily_limit)
            _daily_halted = True
            try:
                from alerter import alert_loss_limit_hit
                alert_loss_limit_hit("daily", _daily_pnl_sol, daily_limit)
            except Exception:
                pass
        return True
    if _weekly_pnl_sol < -weekly_limit:
        if not _weekly_halted:
            logger.warning("LIVE TRADING HALTED: weekly loss %.4f SOL exceeds limit %.1f SOL",
                           _weekly_pnl_sol, weekly_limit)
            _weekly_halted = True
            try:
                from alerter import alert_loss_limit_hit
                alert_loss_limit_hit("weekly", _weekly_pnl_sol, weekly_limit)
            except Exception:
                pass
        return True
    if _monthly_pnl_sol < -monthly_limit:
        if not _monthly_halted:
            logger.warning("LIVE TRADING HALTED: monthly loss %.4f SOL exceeds limit %.1f SOL",
                           _monthly_pnl_sol, monthly_limit)
            _monthly_halted = True
            try:
                from alerter import alert_loss_limit_hit
                alert_loss_limit_hit("monthly", _monthly_pnl_sol, monthly_limit)
            except Exception:
                pass
        return True
    return False


def _track_pnl(pnl_sol: float):
    """v73: Track cumulative PnL across daily/weekly/monthly windows."""
    global _daily_pnl_sol, _weekly_pnl_sol, _monthly_pnl_sol
    _daily_pnl_sol += pnl_sol
    _weekly_pnl_sol += pnl_sol
    _monthly_pnl_sol += pnl_sol


def open_live_trade(client_sb, token_entry: dict, strategy: str,
                    position_usd: float, config: dict) -> bool:
    """
    Open a live trade: convert USD position to SOL lamports and execute buy.
    Inserts row into paper_trades with source='rt_live' on success.
    Returns True on success, False on failure.
    """
    ca = token_entry.get("token_address")
    symbol = token_entry.get("symbol", "???")

    if not ca:
        logger.warning("live_trader: no CA for %s — skipping", symbol)
        return False

    entry_price = float(token_entry.get("price_usd", 0))
    if entry_price <= 0:
        logger.error("live_trader: entry_price=0 for %s — aborting live trade", symbol)
        return False

    # Safety checks
    if _check_loss_limits(config):
        return False

    # Check max open positions
    max_open = int(config.get("max_open_positions", 5))
    try:
        result = (
            client_sb.table("paper_trades")
            .select("id", count="exact")
            .eq("status", "open")
            .eq("source", "rt_live")
            .execute()
        )
        open_count = result.count or 0
        if open_count >= max_open:
            logger.info("live_trader: max open positions (%d) reached — skipping %s", max_open, symbol)
            return False
    except Exception as e:
        logger.warning("live_trader: failed to check open positions: %s", e)

    # Convert USD → SOL → lamports
    sol_price = _get_sol_price_usd()
    position_sol = position_usd / sol_price
    max_sol = float(config.get("max_position_sol", 0.5))
    position_sol = min(position_sol, max_sol)

    # Check minimum SOL reserve
    min_reserve = float(config.get("min_sol_reserve", 0.05))
    balances = get_wallet_balance()
    if balances:
        available_sol = balances["sol_balance"] - min_reserve
        if available_sol <= 0:
            logger.warning("live_trader: insufficient SOL (%.4f, reserve=%.2f) — skipping %s",
                           balances["sol_balance"], min_reserve, symbol)
            return False
        position_sol = min(position_sol, available_sol)

    if position_sol < 0.001:
        logger.info("live_trader: position too small (%.6f SOL) — skipping %s", position_sol, symbol)
        return False

    lamports = int(position_sol * LAMPORTS_PER_SOL)
    slippage = int(config.get("slippage_buy_bps", 300))

    # Execute the buy
    result = execute_buy(ca, lamports, slippage)
    if not result["success"]:
        logger.warning("live_trader: buy failed for %s: %s", symbol, result.get("error"))
        try:
            from alerter import alert_live_trade_failed
            alert_live_trade_failed(symbol, "BUY", result.get("error", "unknown"))
        except Exception:
            pass
        return False

    # v74: Compute actual fill price from Jupiter response
    # execution_price = (SOL spent / tokens received) * SOL price
    execution_price = entry_price  # fallback to estimated price
    input_amt = result.get("input_amount")
    output_amt = result.get("output_amount")
    if input_amt and output_amt and output_amt > 0:
        sol_spent = input_amt / LAMPORTS_PER_SOL
        # We need token decimals to compute price. Use ratio vs estimated:
        # actual_fill_ratio = (sol_spent / position_sol) — how much more/less SOL we spent
        # Adjust entry_price proportionally
        actual_sol_spent = sol_spent
        expected_sol = position_sol
        if expected_sol > 0:
            fill_ratio = actual_sol_spent / expected_sol
            execution_price = entry_price * fill_ratio
            if abs(fill_ratio - 1.0) > 0.01:
                logger.info("live_trader: fill price divergence for %s: %.2f%% (est=$%.8f, fill=$%.8f)",
                            symbol, (fill_ratio - 1) * 100, entry_price, execution_price)

    # Insert into paper_trades with source='rt_live'
    from paper_trader import STRATEGIES
    tranches = STRATEGIES.get(strategy, [{"tp_mult": 2.0, "sl_mult": 0.70, "horizon_min": 120}])
    tranche = tranches[0]  # Live trades always use first tranche

    tp_price = execution_price * tranche["tp_mult"] if tranche.get("tp_mult") else None
    sl_price = execution_price * tranche["sl_mult"]

    # v105: Compute actual slippage from fill vs estimated price
    actual_slippage_bps = 0
    if execution_price > 0 and entry_price > 0:
        actual_slippage_bps = round((execution_price / entry_price - 1) * 10000)

    # v105: Alert on high slippage
    if abs(actual_slippage_bps) > 500:
        try:
            from alerter import alert_slippage_deviation
            alert_slippage_deviation(symbol, slippage, actual_slippage_bps)
        except Exception:
            pass

    # v105: Alert if wallet balance is low after buy
    post_buy_balance = get_wallet_balance()
    if post_buy_balance and post_buy_balance["sol_balance"] < 0.1:
        try:
            from alerter import alert_wallet_low
            alert_wallet_low(post_buy_balance["sol_balance"])
        except Exception:
            pass

    row = {
        "cycle_ts": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "token_address": ca,
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
        "tx_signature": result["signature"],
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
        # v105: Fee tracking — actual execution metrics
        "buy_slippage_bps": actual_slippage_bps,
        "buy_fee_bps": slippage,  # configured tolerance
        "sol_price_at_entry": sol_price,
        "position_sol": round(position_sol, 6),
        # v117: Extended execution tracking
        "buy_exec_ms": result.get("exec_ms"),
        "buy_input_lamports": result.get("input_amount"),  # SOL actually spent
        "buy_output_tokens": result.get("output_amount"),  # tokens received (raw)
    }

    try:
        client_sb.table("paper_trades").insert(row).execute()
        logger.info(
            "LIVE TRADE OPENED: %s %s @ $%.8f | %.4f SOL ($%.2f) | sig: %s",
            symbol, strategy, entry_price, position_sol, position_usd, result["signature"][:16],
        )
        # Alert via Telegram
        try:
            from alerter import alert_live_trade
            alert_live_trade(symbol, "BUY", position_sol, result["signature"])
        except Exception:
            pass
        return True
    except Exception as e:
        logger.error("live_trader: DB insert failed for %s (trade executed but not tracked!): %s",
                     symbol, e)
        return False


def check_live_trades(client_sb) -> dict:
    """
    v113: Check all open live trades using paper_trader's _evaluate_trade_exit().
    Supports ALL strategy types: DTRAIL, TRAIL, BE, DECAY, FIXED, etc.
    For exits: execute sell BEFORE updating DB.
    Returns {"checked": N, "closed": M, "tp": X, "sl": Y, "timeout": Z, "pnl_usd": total}.
    """
    from paper_trader import _fetch_prices_batch, _evaluate_trade_exit
    now = datetime.now(timezone.utc)

    result_counts = {
        "checked": 0, "closed": 0, "tp": 0, "sl": 0, "timeout": 0,
        "pnl_usd": 0.0, "rt_pnl_usd": 0.0,
    }

    try:
        result = (
            client_sb.table("paper_trades")
            .select("*")
            .eq("status", "open")
            .eq("source", "rt_live")
            .execute()
        )
        open_trades = result.data or []
    except Exception as e:
        logger.error("live_trader: failed to fetch open trades: %s", e)
        return result_counts

    if not open_trades:
        return result_counts

    result_counts["checked"] = len(open_trades)

    # Batch fetch current prices
    addresses = list({t["token_address"] for t in open_trades})
    prices = _fetch_prices_batch(addresses)

    for trade in open_trades:
        addr = trade["token_address"]
        current_price = prices.get(addr)

        # v113: Use paper_trader's full evaluation (DTRAIL, TRAIL, BE, DECAY, etc.)
        # sell_slip_factor=1.0 because live uses real Jupiter execution, not simulated slippage
        ev = _evaluate_trade_exit(trade, current_price, now, sell_slip_factor=1.0, sell_fee_bps=0)

        if ev is None:
            continue

        # Always update high_price_seen (even without exit)
        if ev.get("high_price_seen") is not None:
            new_high = ev["high_price_seen"]
            old_high = float(trade.get("high_price_seen") or 0)
            if new_high > old_high:
                try:
                    client_sb.table("paper_trades").update(
                        {"high_price_seen": new_high}
                    ).eq("id", trade["id"]).execute()
                except Exception as e:
                    logger.debug("live_trader: high_price_seen update failed for %s: %s", trade["symbol"], e)

        new_status = ev.get("status")
        if new_status is None:
            continue

        exit_price = ev.get("exit_price")
        entry_price = float(trade.get("entry_price") or 0)
        elapsed_minutes = ev.get("exit_minutes", 0)
        if not elapsed_minutes:
            created = trade.get("created_at")
            if created:
                from datetime import datetime as _dt
                try:
                    ct = _dt.fromisoformat(created.replace("Z", "+00:00")) if isinstance(created, str) else created
                    elapsed_minutes = int((now - ct).total_seconds() / 60)
                except Exception:
                    elapsed_minutes = 0

        # Execute sell BEFORE updating DB
        sell_result = execute_sell(addr)
        if not sell_result["success"]:
            logger.warning(
                "live_trader: sell failed for %s (%s) — keeping trade open (retry next cycle): %s",
                trade["symbol"], new_status, sell_result.get("error"),
            )
            try:
                from alerter import alert_live_trade_failed
                alert_live_trade_failed(trade["symbol"], "SELL", sell_result.get("error", "unknown"))
            except Exception:
                pass
            continue

        # v74: Use actual SOL received from Jupiter to compute real exit price
        # v118: Wrapped in try-except — sell already executed, DB update MUST happen
        sell_output = sell_result.get("output_amount")  # SOL lamports received
        pnl_pct = 0
        pnl_usd = 0
        pos_usd = float(trade.get("position_usd") or 0)
        sell_slippage_bps = 0
        sol_price_at_exit = 0
        sell_sol_received = None

        try:
            if sell_output and sell_output > 0 and entry_price > 0:
                sol_received = sell_output / LAMPORTS_PER_SOL
                sol_price_now = _get_sol_price_usd()
                usd_received = sol_received * sol_price_now
                pos_usd_val = float(trade.get("position_usd") or 0)
                if pos_usd_val > 0:
                    exit_price = entry_price * (usd_received / pos_usd_val)

            pnl_pct = round((exit_price / entry_price) - 1, 4) if exit_price and entry_price else 0
            pnl_usd = round(pos_usd * pnl_pct, 2) if pos_usd else 0

            # Track daily PnL in SOL
            sol_price = _get_sol_price_usd()
            pnl_sol = pnl_usd / sol_price if sol_price > 0 else 0
            _track_pnl(pnl_sol)

            # v105: Compute sell slippage (actual SOL received vs expected)
            sol_price_at_exit = _get_sol_price_usd()
            if sell_output and sell_output > 0 and pos_usd > 0:
                sol_received = sell_output / LAMPORTS_PER_SOL
                usd_received_actual = sol_received * sol_price_at_exit
                expected_usd = pos_usd * (1 + pnl_pct)
                if expected_usd > 0:
                    sell_slippage_bps = round((1 - usd_received_actual / expected_usd) * 10000)

            # v105: Alert on high sell slippage
            if abs(sell_slippage_bps) > 500:
                from alerter import alert_slippage_deviation
                alert_slippage_deviation(trade["symbol"], 500, sell_slippage_bps)

            sell_sol_received = (sell_output / LAMPORTS_PER_SOL) if sell_output else None
        except Exception as e:
            logger.warning("live_trader: PnL calc error for %s (sell OK, sig=%s): %s — DB update proceeds with defaults",
                           trade["symbol"], sell_result.get("signature"), e)
            sell_sol_received = (sell_output / LAMPORTS_PER_SOL) if sell_output else None

        update = {
            "status": new_status,
            "exit_price": exit_price,
            "exit_at": now.isoformat(),
            "pnl_pct": pnl_pct,
            "pnl_usd": pnl_usd,
            "exit_minutes": int(elapsed_minutes),
            "tx_signature_exit": sell_result["signature"],
            # v105: Fee tracking
            "sell_slippage_bps": sell_slippage_bps,
            "sol_price_at_exit": sol_price_at_exit,
            # v117: Extended sell tracking
            "sell_exec_ms": sell_result.get("exec_ms"),
            "sell_output_lamports": sell_output,  # SOL received (lamports)
            "sell_input_tokens": sell_result.get("input_amount"),  # tokens sold (raw)
            "sell_sol_received": round(sell_sol_received, 6) if sell_sol_received else None,
        }

        # DB update with retry — sell already executed, must not leave trade as 'open'
        db_updated = False
        for attempt in range(3):
            try:
                client_sb.table("paper_trades").update(update).eq("id", trade["id"]).execute()
                db_updated = True
                break
            except Exception as e:
                logger.warning("live_trader: DB update attempt %d/3 failed for trade %s: %s",
                               attempt + 1, trade["id"], e)
                if attempt < 2:
                    time.sleep(2 ** attempt)

        if not db_updated:
            logger.error(
                "CRITICAL: live trade %s sold (sig=%s) but DB update failed! "
                "Trade stuck as 'open' with no balance. Manual fix required.",
                trade["symbol"], sell_result["signature"],
            )
            continue

        result_counts["closed"] += 1
        result_counts["pnl_usd"] += pnl_usd
        result_counts["rt_pnl_usd"] += pnl_usd
        status_key = new_status.replace("_hit", "").replace("_stop", "")
        result_counts[status_key] = result_counts.get(status_key, 0) + 1

        logger.info(
            "LIVE TRADE CLOSED: %s %s — %s pnl=%.1f%% $%+.2f | sell sig: %s",
            trade["symbol"], trade["strategy"], new_status,
            pnl_pct * 100, pnl_usd, sell_result["signature"][:16],
        )

        # Alert via Telegram
        try:
            from alerter import alert_live_trade
            alert_live_trade(
                trade["symbol"], "SELL",
                abs(pnl_usd / sol_price) if sol_price else 0,
                sell_result["signature"],
            )
        except Exception:
            pass

    if result_counts["closed"] > 0:
        logger.info(
            "live_trader: checked %d, closed %d (TP=%d SL=%d timeout=%d) pnl=$%+.2f",
            result_counts["checked"], result_counts["closed"],
            result_counts["tp"], result_counts["sl"], result_counts["timeout"],
            result_counts["pnl_usd"],
        )

    return result_counts


def reconcile_positions(client_sb) -> dict:
    """
    v74: Verify on-chain token balances match DB open positions.
    Flags mismatches (DB says open but no on-chain balance, or vice versa).
    Returns {"checked": N, "mismatches": M, "auto_closed": X, "details": [...]}.
    """
    result = {"checked": 0, "mismatches": 0, "auto_closed": 0, "details": []}

    balances = get_wallet_balance()
    if not balances:
        logger.warning("reconcile: cannot fetch wallet balances — skipping")
        return result

    try:
        resp = (
            client_sb.table("paper_trades")
            .select("id, symbol, token_address, entry_price, position_usd, created_at")
            .eq("status", "open")
            .eq("source", "rt_live")
            .execute()
        )
        open_trades = resp.data or []
    except Exception as e:
        logger.error("reconcile: failed to fetch open trades: %s", e)
        return result

    on_chain_mints = set(balances.get("token_balances", {}).keys())
    result["checked"] = len(open_trades)

    for trade in open_trades:
        ca = trade.get("token_address")
        if not ca:
            continue

        if ca not in on_chain_mints:
            # DB says open, but no on-chain balance → position was sold externally or failed
            result["mismatches"] += 1
            detail = {
                "id": trade["id"],
                "symbol": trade["symbol"],
                "ca": ca,
                "issue": "db_open_but_no_balance",
            }
            result["details"].append(detail)
            logger.warning(
                "RECONCILE MISMATCH: %s (%s) open in DB but 0 on-chain balance. "
                "Auto-closing as 'reconciled'.",
                trade["symbol"], ca[:12],
            )
            # Auto-close as reconciled — we can't sell what we don't have
            try:
                client_sb.table("paper_trades").update({
                    "status": "reconciled",
                    "exit_at": datetime.now(timezone.utc).isoformat(),
                    "pnl_pct": -1.0,  # Assume total loss
                    "pnl_usd": -float(trade.get("position_usd") or 0),
                }).eq("id", trade["id"]).execute()
                result["auto_closed"] += 1
            except Exception as e:
                logger.error("reconcile: failed to close trade %s: %s", trade["id"], e)

    # Check reverse: on-chain tokens not tracked in DB (orphaned positions)
    tracked_cas = {t["token_address"] for t in open_trades if t.get("token_address")}
    for mint in on_chain_mints:
        if mint not in tracked_cas and mint != WSOL_MINT:
            bal = balances["token_balances"][mint]
            if bal.get("ui_amount", 0) > 0:
                logger.warning(
                    "RECONCILE: on-chain token %s (%.4f) not tracked in DB — orphaned position",
                    mint[:12], bal["ui_amount"],
                )
                result["mismatches"] += 1
                result["details"].append({
                    "ca": mint,
                    "issue": "on_chain_but_not_in_db",
                    "ui_amount": bal["ui_amount"],
                })

    if result["mismatches"] > 0:
        logger.info(
            "reconcile: %d checked, %d mismatches, %d auto-closed",
            result["checked"], result["mismatches"], result["auto_closed"],
        )
        try:
            from alerter import _send
            _send(
                f"<b>POSITION RECONCILIATION</b>\n"
                f"Checked: {result['checked']}\n"
                f"Mismatches: {result['mismatches']}\n"
                f"Auto-closed: {result['auto_closed']}",
                "cycle_failure",
            )
        except Exception:
            pass

    return result
