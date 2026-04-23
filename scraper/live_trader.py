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


def _is_solana_mint(ca: str) -> bool:
    """Hard gate before any Jupiter call. Rejects 0x (EVM) and any shape
    detect_chain does not classify as solana. Defence in depth — without this,
    ETH addresses leaked into execute_buy and produced 400 Bad Request storms
    on /ultra/v1/order (v14 regression)."""
    if not ca or not isinstance(ca, str):
        return False
    try:
        from chain_detect import detect_chain
        return detect_chain(ca) == "solana"
    except Exception:
        return not ca.startswith("0x")


# --- v121: Trigger event logging ---
def _log_trigger_event(client_sb, *, trade_id=None, token_ca="", token_symbol="",
                       order_id=None, event_type: str, old_sl=None, new_sl=None,
                       current_price=None, high_price=None, fill_sol=None, details=None):
    """Log a trigger order event for later analysis (trigger vs polling)."""
    try:
        row = {
            "trade_id": trade_id,
            "token_ca": token_ca,
            "token_symbol": token_symbol,
            "order_id": order_id,
            "event_type": event_type,
            "old_sl_price": old_sl,
            "new_sl_price": new_sl,
            "current_price": current_price,
            "high_price_seen": high_price,
            "fill_sol_received": fill_sol,
            "details": details,
        }
        client_sb.table("trigger_events").insert(row).execute()
    except Exception as e:
        logger.debug("trigger_events log failed: %s", e)


def _calc_message_to_buy(message_ts_iso: str | None) -> int | None:
    """v121: Seconds between Telegram message and buy execution."""
    if not message_ts_iso:
        return None
    try:
        msg_dt = datetime.fromisoformat(message_ts_iso.replace("Z", "+00:00"))
        delta = (datetime.now(timezone.utc) - msg_dt).total_seconds()
        return int(delta) if delta >= 0 else None
    except Exception:
        return None


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


_WALLET_BALANCE_CACHE: dict = {"ts": 0.0, "value": None}
_WALLET_BALANCE_TTL_SEC = 10.0


def get_wallet_balance(force_refresh: bool = False) -> dict | None:
    """
    Fetch wallet SOL + token balances via Jupiter Ultra /holdings endpoint.
    Returns {"sol_balance": float, "token_balances": {mint: {"amount": int, "ui_amount": float}}}
    or None on failure. Cached 10s to keep buy path fast (Jupiter holdings
    takes ~500-1500ms).
    """
    now = time.time()
    if not force_refresh and _WALLET_BALANCE_CACHE["value"] is not None \
            and (now - _WALLET_BALANCE_CACHE["ts"]) < _WALLET_BALANCE_TTL_SEC:
        return _WALLET_BALANCE_CACHE["value"]

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

        result = {"sol_balance": sol_balance, "token_balances": token_balances}
        _WALLET_BALANCE_CACHE["ts"] = time.time()
        _WALLET_BALANCE_CACHE["value"] = result
        return result
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
    if not _is_solana_mint(ca):
        logger.error("LIVE BUY REJECTED: %s is not a Solana mint — Jupiter Ultra is Solana-only", ca[:12])
        return {"success": False, "signature": "", "error": "non-solana-mint"}
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


def execute_sell(ca: str, amount_tokens: int | None = None, slippage_bps: int = 1000) -> dict:
    """
    Execute a sell swap: token → SOL via Jupiter Ultra.
    If amount_tokens is None, sells entire balance of that token.
    v105: Now passes slippageBps to the Ultra /order endpoint.
    Returns {"success": bool, "signature": str, "status": str, "error": str|None,
             "input_amount": int|None, "output_amount": int|None,
             "slippage_bps": int}
    """
    if not _is_solana_mint(ca):
        logger.error("LIVE SELL REJECTED: %s is not a Solana mint — Jupiter Ultra is Solana-only", ca[:12])
        return {"success": False, "signature": "", "error": "non-solana-mint"}
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


_SOL_PRICE_CACHE: dict = {"ts": 0.0, "value": 0.0}
_SOL_PRICE_TTL_SEC = 5.0


def _get_sol_price_usd() -> float:
    """Fetch current SOL/USD price from DexScreener. Cached 5s — SOL moves
    <1% in 5s so this is safe, and removes a ~300-800ms blocker on the
    critical buy path."""
    now = time.time()
    if _SOL_PRICE_CACHE["value"] > 0 and (now - _SOL_PRICE_CACHE["ts"]) < _SOL_PRICE_TTL_SEC:
        return _SOL_PRICE_CACHE["value"]
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
                    _SOL_PRICE_CACHE["ts"] = time.time()
                    _SOL_PRICE_CACHE["value"] = float(price)
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
                _SOL_PRICE_CACHE["ts"] = time.time()
                _SOL_PRICE_CACHE["value"] = float(price)
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
                    position_usd: float, config: dict):
    """
    Open a live trade: convert USD position to SOL lamports and execute buy.
    Inserts row into paper_trades with source='rt_live' on success.

    v142 E: returns dict {"success": bool, "execution_price": float|None}
    instead of bare bool, so caller (safe_scraper._rt_open_trades) can
    shadow-sync paper's entry_price to live's actual Jupiter fill. On
    failure returns {"success": False, "execution_price": None}.

    Backward-compat: truthiness still works (dict with success=True is truthy).
    """
    _FAIL = {"success": False, "execution_price": None}
    ca = token_entry.get("token_address")
    symbol = token_entry.get("symbol", "???")

    if not ca:
        logger.warning("live_trader: no CA for %s — skipping", symbol)
        return _FAIL

    # v14e: Hard chain gate. live_trader is Solana-only (Jupiter Ultra). ETH/BSC/Base
    # must route through their own live adapter (not yet implemented). Without this,
    # v14b's ETH main-paper promotion caused live_trader to send 0x mints to Jupiter,
    # producing 400 Bad Request storms.
    token_chain = token_entry.get("chain") or "solana"
    if token_chain != "solana" or not _is_solana_mint(ca):
        logger.info("live_trader: %s/%s chain=%s — paper-only (no live adapter for non-Solana)",
                    symbol, ca[:8] if ca else "?", token_chain)
        return _FAIL

    entry_price = float(token_entry.get("price_usd", 0))
    if entry_price <= 0:
        logger.error("live_trader: entry_price=0 for %s — aborting live trade", symbol)
        return _FAIL

    # Safety checks
    if _check_loss_limits(config):
        return _FAIL

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
            return _FAIL
    except Exception as e:
        logger.warning("live_trader: failed to check open positions: %s", e)

    # v121: 24h dedup cooldown — same as paper trades. Prevents re-entering dead tokens.
    dedup_hours = int(config.get("dedup_cooldown_hours", 24))
    try:
        cutoff = (datetime.now(timezone.utc) - __import__("datetime").timedelta(hours=dedup_hours)).isoformat()
        # v133: scope dedup to SAME strategy — otherwise the hybrid allocation loop
        # (FAST then DTRAIL10 for the same token) blocks the 2nd strategy after the
        # 1st row was inserted seconds earlier.
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
            logger.info("live_trader: dedup cooldown — %s/%s traded in last %dh, skipping",
                        symbol, strategy, dedup_hours)
            return _FAIL
    except Exception as e:
        logger.debug("live_trader: dedup check failed for %s: %s", symbol, e)

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
            return _FAIL
        position_sol = min(position_sol, available_sol)

    if position_sol < 0.001:
        logger.info("live_trader: position too small (%.6f SOL) — skipping %s", position_sol, symbol)
        return _FAIL

    # v118: Recalculate position_usd after SOL cap — must match actual SOL spent
    position_usd = round(position_sol * sol_price, 2)

    lamports = int(position_sol * LAMPORTS_PER_SOL)
    slippage = int(config.get("slippage_buy_bps", 1000))

    # Execute the buy
    _t_pre_buy = time.time()
    result = execute_buy(ca, lamports, slippage)
    _t_post_buy = time.time()
    if not result["success"]:
        logger.warning("live_trader: buy failed for %s: %s", symbol, result.get("error"))
        try:
            from alerter import alert_live_trade_failed
            alert_live_trade_failed(symbol, "BUY", result.get("error", "unknown"))
        except Exception:
            pass
        return _FAIL

    # v130: True fill price from outAmount + token decimals.
    # Previously used fill_ratio = sol_spent/expected_sol (≈1.0 for exact-in),
    # which left execution_price ≈ DexScreener spot — defeating the whole point.
    # Now: execution_price = (SOL spent * SOL price USD) / tokens received.
    execution_price = entry_price  # fallback: pre-trade spot
    entry_source = "dexscreener"
    input_amt = result.get("input_amount")
    output_amt = result.get("output_amount")
    if input_amt and output_amt and output_amt > 0:
        try:
            from enrich_jupiter import get_token_decimals
            decimals = get_token_decimals(ca)
        except Exception:
            decimals = None
        if decimals is not None:
            sol_spent = input_amt / LAMPORTS_PER_SOL
            out_tokens = output_amt / (10 ** decimals)
            if out_tokens > 0 and sol_price > 0:
                execution_price = (sol_spent * sol_price) / out_tokens
                entry_source = "ultra"
                divergence_pct = (execution_price / entry_price - 1) * 100 if entry_price > 0 else 0
                if abs(divergence_pct) > 1:
                    logger.info(
                        "live_trader: %s fill divergence %.2f%% (spot=$%.8f, fill=$%.8f)",
                        symbol, divergence_pct, entry_price, execution_price,
                    )
        else:
            logger.warning("live_trader: decimals unavailable for %s — execution_price falls back to spot", ca[:8])

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
        # v121: Call reaction speed
        "message_ts": token_entry.get("_rt_message_ts"),
        "price_at_message": token_entry.get("_rt_price_at_message"),
        "message_to_buy_seconds": _calc_message_to_buy(token_entry.get("_rt_message_ts")),
        # v105: Fee tracking — actual execution metrics
        "buy_slippage_bps": actual_slippage_bps,
        "buy_fee_bps": slippage,  # configured tolerance
        "sol_price_at_entry": sol_price,
        "position_sol": round(position_sol, 6),
        # v117: Extended execution tracking
        "buy_exec_ms": result.get("exec_ms"),
        "buy_input_lamports": result.get("input_amount"),  # SOL actually spent
        "buy_output_tokens": result.get("output_amount"),  # tokens received (raw)
        # v130: Single-source anchor — all refs on execution_price (true Ultra fill).
        # Matches paper_trader.py (v130): entry_price = dex_spot = high = one source.
        # DTRAIL/BE gates evaluate against the same source across paper & live.
        "dex_spot_price_at_entry": execution_price,
        "high_price_seen": execution_price,
        "entry_source": entry_source,
        # v121: pair_address for OHLCV backtesting on live trades
        "pair_address": token_entry.get("_rt_pair_address"),
    }

    try:
        insert_res = client_sb.table("paper_trades").insert(row).execute()
        trade_id = insert_res.data[0]["id"] if insert_res.data else None
        logger.info(
            "LIVE TRADE OPENED: %s %s @ $%.8f | %.4f SOL ($%.2f) | sig: %s",
            symbol, strategy, entry_price, position_sol, position_usd, result["signature"][:16],
        )
        _t_msg = token_entry.get("_rt_t_msg")
        _t_ds_done = token_entry.get("_rt_t_ds_done")
        if _t_msg and _t_ds_done:
            logger.info(
                "LIVE LATENCY: %s total=%.2fs | msg→ds=%.2fs ds→pre_buy=%.2fs buy_exec=%.2fs",
                symbol, _t_post_buy - _t_msg,
                _t_ds_done - _t_msg, _t_pre_buy - _t_ds_done, _t_post_buy - _t_pre_buy,
            )

        # v119: Place Jupiter trigger stop-loss order (on-chain protection)
        _trigger_variant = "polling"  # v122: default, overridden if trigger succeeds
        try:
            _live_cfg = config.get("live_trading", config)
            if _live_cfg.get("trigger_orders_enabled", False):
                token_amount = result.get("output_amount")
                _min_usd = float(_live_cfg.get("trigger_min_usd", 10))
                if token_amount and token_amount > 0 and trade_id and position_usd >= _min_usd:
                    from jupiter_trigger import place_stop_loss
                    _expiry = int(_live_cfg.get("trigger_expiry_seconds", tranche["horizon_min"] * 60))
                    _sl_slip = int(_live_cfg.get("trigger_sl_slippage_bps", 2000))
                    trigger_res = place_stop_loss(
                        ca=ca,
                        amount_tokens=token_amount,
                        sl_price_usd=sl_price,
                        expiry_seconds=_expiry,
                        sl_slippage_bps=_sl_slip,
                    )
                    if trigger_res and "order_id" in trigger_res:
                        _trigger_variant = "trigger"  # v122: keeper will manage SL
                        client_sb.table("paper_trades").update(
                            {"trigger_order_id": trigger_res["order_id"]}
                        ).eq("id", trade_id).execute()
                        logger.info("TRIGGER SL PLACED: %s order=%s sl=$%.8f",
                                    symbol, trigger_res["order_id"][:16], sl_price)
                        _log_trigger_event(
                            client_sb, trade_id=trade_id, token_ca=ca,
                            token_symbol=symbol, order_id=trigger_res["order_id"],
                            event_type="placed", new_sl=sl_price,
                            current_price=execution_price,
                        )
                    else:
                        # v122: Log which step failed for diagnostics
                        fail_step = trigger_res.get("error", "unknown") if trigger_res else "returned_none"
                        logger.warning("TRIGGER SL FAILED: %s step=%s — fallback to polling", symbol, fail_step)
                        _log_trigger_event(
                            client_sb, trade_id=trade_id, token_ca=ca,
                            token_symbol=symbol, event_type="failed",
                            details={"reason": f"failed at step: {fail_step}", **(trigger_res or {})},
                        )
                elif position_usd < _min_usd:
                    logger.info("TRIGGER SL SKIP: %s pos=$%.2f < min $%.0f — polling only",
                                symbol, position_usd, _min_usd)
        except Exception as e:
            logger.warning("live_trader: trigger order failed for %s (fallback to polling): %s",
                           symbol, e)

        # v122: A/B experiment — track trigger vs polling for live trade comparison
        try:
            client_sb.table("paper_trades").update({
                "experiment_id": "trigger_v2_ab",
                "variant_id": _trigger_variant,
            }).eq("id", trade_id).execute()
        except Exception:
            pass

        # Alert via Telegram — v119: detailed buy alert
        # v144.11: include real bankroll + per-strategy breakdown (same shape as paper alerts).
        try:
            from alerter import alert_live_buy
            _br, _strat_bals, _active = {}, {}, None
            try:
                from safe_scraper import _rt_load_bankroll, _rt_strategy_bankrolls_for_chain
                _br = _rt_load_bankroll() or {}
                # v14e: live_trader is Solana-only (chain gate above), scope
                # per-strat display to Solana bucket so ETH/BSC entries don't
                # pollute the live buy alert.
                _strat_bals = _rt_strategy_bankrolls_for_chain(_br, "solana")
                _active = list(_strat_bals.keys()) if _strat_bals else None
            except Exception:
                pass
            alert_live_buy(
                symbol=symbol, strategy=strategy,
                position_sol=position_sol, position_usd=position_usd,
                entry_price=execution_price, score=float(token_entry.get("score", 0)),
                kol=token_entry.get("_rt_kol_group", ""),
                mcap=float(token_entry.get("market_cap", 0)),
                signature=result["signature"],
                slippage_bps=actual_slippage_bps,
                exec_ms=result.get("exec_ms", 0),
                sol_price=sol_price, ca=ca,
                bankroll=float(_br.get("current_balance", 0) or 0),
                strategy_bankrolls=_strat_bals,
                active_strategies=_active,
            )
        except Exception:
            pass
        # v142 E: return execution_price so caller can shadow-sync paper
        # trades to the same fill price (fixes P1+P4 divergence).
        return {"success": True, "execution_price": execution_price}
    except Exception as e:
        logger.error("live_trader: DB insert failed for %s (trade executed but not tracked!): %s",
                     symbol, e)
        return {"success": False, "execution_price": None}


def _handle_trigger_fill(client_sb, trade: dict, order_status: dict, now) -> None:
    """
    v119: Handle a Jupiter trigger order that was filled by keepers.
    Updates DB with PnL, sends alert. No execute_sell needed (already on-chain).
    """
    entry_price = float(trade.get("entry_price") or 0)
    pos_usd = float(trade.get("position_usd") or 0)
    pos_sol = float(trade.get("position_sol") or 0)

    # Extract fill data from order
    fill_event = order_status.get("fill_event") or {}
    output_amount = fill_event.get("outputAmount")

    # Compute PnL from SOL received
    sol_received = 0
    pnl_pct = -1.0  # assume total loss as fallback
    pnl_usd = -pos_usd

    try:
        if output_amount and int(output_amount) > 0:
            sol_received = int(output_amount) / LAMPORTS_PER_SOL
            sol_price = _get_sol_price_usd()
            usd_received = sol_received * sol_price
            if pos_usd > 0:
                pnl_pct = round(usd_received / pos_usd - 1, 4)
                pnl_usd = round(pos_usd * pnl_pct, 2)
                exit_price = entry_price * (usd_received / pos_usd) if pos_usd > 0 else 0
            else:
                exit_price = 0
        else:
            exit_price = float(trade.get("sl_price") or 0)
            sol_price = _get_sol_price_usd()
    except Exception as e:
        logger.warning("_handle_trigger_fill: PnL calc error for %s: %s", trade["symbol"], e)
        exit_price = float(trade.get("sl_price") or 0)
        sol_price = _get_sol_price_usd()

    # Determine exit reason (trigger = SL hit by keeper)
    new_status = "sl_hit"  # trigger orders are SL-only for now

    # Elapsed time
    elapsed_minutes = 0
    created = trade.get("created_at")
    if created:
        try:
            from datetime import datetime as _dt
            ct = _dt.fromisoformat(created.replace("Z", "+00:00")) if isinstance(created, str) else created
            elapsed_minutes = int((now - ct).total_seconds() / 60)
        except Exception:
            pass

    # v122: Paper exit price = SL price with simulated slippage (what paper would get)
    from paper_trader import _dynamic_sell_slip_factor
    paper_exit_price = float(trade.get("sl_price") or 0) * _dynamic_sell_slip_factor(trade, "sl_hit")
    price_divergence_pct = None
    if paper_exit_price and paper_exit_price > 0 and exit_price and exit_price > 0:
        price_divergence_pct = round((exit_price / paper_exit_price) - 1, 4)
    # v125: Structured divergence log
    if price_divergence_pct is not None:
        logger.info(
            "DIVERGENCE trigger_fill: %s %s paper=$%.8f actual=$%.8f delta=%.2f%%",
            trade.get("symbol"), trade.get("strategy"),
            paper_exit_price, exit_price, price_divergence_pct * 100,
        )

    # Update DB
    update = {
        "status": new_status,
        "exit_price": exit_price,
        "exit_at": now.isoformat(),
        "pnl_pct": pnl_pct,
        "pnl_usd": pnl_usd,
        "exit_minutes": elapsed_minutes,
        "trigger_order_id": None,  # consumed
        "sell_sol_received": round(sol_received, 6) if sol_received else None,
        "sol_price_at_exit": sol_price,
        # v122: Paper vs live divergence
        "paper_exit_price": paper_exit_price,
        "price_divergence_pct": price_divergence_pct,
        # v126: Flag keeper-driven exits so analysis can isolate trigger vs polling
        "exit_via_trigger": True,
    }

    try:
        client_sb.table("paper_trades").update(update).eq("id", trade["id"]).execute()
    except Exception as e:
        logger.error("CRITICAL: trigger fill for %s but DB update failed: %s", trade["symbol"], e)
        return

    # Track PnL
    if sol_price > 0:
        _track_pnl(pnl_usd / sol_price)

    _log_trigger_event(
        client_sb, trade_id=trade["id"], token_ca=trade.get("token_address", ""),
        token_symbol=trade["symbol"], order_id=trade.get("trigger_order_id"),
        event_type="filled", new_sl=float(trade.get("sl_price") or 0),
        current_price=exit_price, high_price=float(trade.get("high_price_seen") or 0),
        fill_sol=sol_received,
        details={"pnl_pct": pnl_pct, "pnl_usd": pnl_usd, "elapsed_min": elapsed_minutes},
    )

    logger.info(
        "TRIGGER FILL: %s %s — %s pnl=%.1f%% $%+.2f | keeper executed SL",
        trade["symbol"], trade.get("strategy", ""), new_status,
        pnl_pct * 100, pnl_usd,
    )

    # Alert
    # v144.11: trigger fill path doesn't compute _paper_sim_ev (keeper fills are
    # SL-only on-chain, no polling tick). paper_pnl_pct stays None → per-trade
    # block hides itself. Bankroll + 24h drift are still useful.
    try:
        from alerter import alert_live_sell, _live_paper_strategy_drift_24h
        _br, _strat_bals, _active = {}, {}, None
        try:
            from safe_scraper import _rt_load_bankroll, _rt_strategy_bankrolls_for_chain
            _br = _rt_load_bankroll() or {}
            # v14e: live_trader is Solana-only — scope alert bankroll display.
            _strat_bals = _rt_strategy_bankrolls_for_chain(_br, "solana")
            _active = list(_strat_bals.keys()) if _strat_bals else None
        except Exception:
            pass
        _drift_24h = {}
        try:
            _drift_24h = _live_paper_strategy_drift_24h(client_sb) or {}
        except Exception:
            pass
        alert_live_sell(
            symbol=trade["symbol"], strategy=trade.get("strategy", ""),
            exit_reason=new_status,
            pnl_pct=pnl_pct, pnl_usd=pnl_usd,
            position_usd=pos_usd,
            entry_price=entry_price, exit_price=exit_price,
            sol_received=sol_received,
            minutes=elapsed_minutes,
            kol=trade.get("kol_group", ""),
            signature="trigger-fill",
            sol_price=sol_price,
            ca=trade.get("token_address", ""),
            high_price=float(trade.get("high_price_seen") or 0),
            bankroll=float(_br.get("current_balance", 0) or 0),
            strategy_bankrolls=_strat_bals,
            active_strategies=_active,
            price_divergence_pct=price_divergence_pct,
            strategy_drift_24h=_drift_24h,
        )
    except Exception:
        pass


def check_live_trades(client_sb) -> dict:
    """
    v113: Check all open live trades using paper_trader's _evaluate_trade_exit().
    Supports ALL strategy types: DTRAIL, TRAIL, BE, DECAY, FIXED, etc.
    For exits: execute sell BEFORE updating DB.
    Returns {"checked": N, "closed": M, "tp": X, "sl": Y, "timeout": Z, "pnl_usd": total}.
    """
    from paper_trader import (
        _fetch_prices_batch, _evaluate_trade_exit,
        _strategy_orchestration, _should_poll_trade, _decision_price,
    )
    now = datetime.now(timezone.utc)

    # v132: Load orchestration config once per check
    _rt_cfg_orch = {}
    try:
        from safe_scraper import _rt_load_config as _rt_load
        _rt_cfg_orch = _rt_load() or {}
    except Exception:
        pass

    result_counts = {
        "checked": 0, "closed": 0, "tp": 0, "sl": 0, "timeout": 0,
        "pnl_usd": 0.0, "rt_pnl_usd": 0.0,
    }

    try:
        # v121: Also pick up 'closing' trades (claimed but sell failed/crashed — need retry)
        result = (
            client_sb.table("paper_trades")
            .select("*")
            .in_("status", ["open", "closing"])
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

    # v119: Check trigger order fills FIRST (faster than DexScreener)
    remaining_trades = []
    for trade in open_trades:
        trigger_id = trade.get("trigger_order_id")
        if trigger_id:
            try:
                from jupiter_trigger import check_order_status
                order_status = check_order_status(trigger_id)
                if order_status and order_status.get("state") in ("filled", "completed"):
                    _handle_trigger_fill(client_sb, trade, order_status, now)
                    result_counts["closed"] += 1
                    result_counts["pnl_usd"] += 0  # computed inside _handle_trigger_fill
                    continue  # skip this trade in price check loop
            except Exception as e:
                logger.debug("trigger status check failed for %s: %s", trade["symbol"], e)
        remaining_trades.append(trade)
    open_trades = remaining_trades

    if not open_trades:
        return result_counts

    # Batch fetch current prices
    addresses = list({t["token_address"] for t in open_trades})
    prices = _fetch_prices_batch(addresses)

    # v121: Log price ticks at 15s resolution for live trades
    from paper_trader import _log_price_ticks, _log_cache_snapshot
    _log_price_ticks(client_sb, prices, "live", live_tokens=set(addresses))
    _log_cache_snapshot(client_sb)  # v138 D: snapshot full cache state from live loop too

    for trade in open_trades:
        addr = trade["token_address"]
        strategy = trade.get("strategy", "")
        trade_id = trade.get("id")
        current_price = prices.get(addr)

        # v121: Trade stuck in 'closing' from a previous failed sell — force sell immediately
        _paper_sim_ev = None  # v141: default (closing_retry path has no paper simulation)
        if trade.get("status") == "closing":
            logger.info("live_trader: retrying sell for stuck 'closing' trade %s (%s)", trade["symbol"], trade["id"])
            # Skip evaluation — this trade already decided to exit, just needs the sell
            # Build a minimal ev dict so the sell logic below works
            ev = {"status": "closing_retry", "exit_price": current_price or float(trade.get("entry_price") or 0)}
        else:
            # v132: Per-strategy orchestration (polling + price source) — matches paper_trader
            orch = _strategy_orchestration(strategy, _rt_cfg_orch)
            if not _should_poll_trade(trade_id, int(orch.get("polling_sec", 30))):
                continue

            decision_price, exit_ref = _decision_price(addr, strategy, trade_id, orch, trade=trade)
            if exit_ref is not None:
                current_price = exit_ref

            # v120: Warn if price is None — trade can't exit without a price
            if current_price is None:
                logger.warning("live_trader: no price for %s (%s) — exit eval skipped", trade["symbol"], addr[:8])

            # v138: record this poll BEFORE eval (live trades — captures every decision)
            from paper_trader import _record_eval_poll
            _record_eval_poll(trade_id, now, decision_price, current_price,
                              float(trade.get("high_price_seen") or 0))

            # v113: Use paper_trader's full evaluation (DTRAIL, TRAIL, BE, DECAY, etc.)
            # sell_slip_factor=1.0 because live uses real Jupiter execution, not simulated slippage
            ev = _evaluate_trade_exit(trade, current_price, now, sell_slip_factor=1.0,
                                       sell_fee_bps=0, decision_price=decision_price)

            # v141: Compute a second, simulation-accurate ev for divergence measurement ONLY.
            # Prior to v141, paper_exit_price persisted to DB was the SL trigger level (slip=1,
            # fee=0) — not what paper_trader actually would have booked. On sl_hit with BE
            # activated, this stored entry * ~1.0 instead of entry * slip_factor (~0.36),
            # making DIVERGENCE logs mix 3 effects (paper slip omitted, price drift during
            # gap, live exec slippage) into a single number. This second ev mirrors
            # paper_trader's exact pipeline: dynamic slip + SELL_FEE_BPS + Ultra SELL quote
            # override. Decision ev above is untouched — live still decides on raw price.
            _paper_sim_ev = None
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
                # v144.19: do NOT apply _pt_ultra_override here. Fetching an Ultra
                # SELL quote at live-exit moment contaminates paper_sim_pnl_pct
                # with the same MEV-boosted fill the live sell just got (e.g.
                # $MHGA pump: paper_sim stored +235% instead of the TP80 sim value
                # +87%). Keep this as pure-sim reference: formula exit_price + flat
                # slip. Real Jupiter fill divergence is already captured by
                # pnl_pct vs paper_sim_pnl_pct (execution drift = real edge).
            except Exception as _e:
                logger.debug("paper_sim_ev compute failed for %s: %s", trade.get("symbol"), _e)

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

                # v119: PATCH trigger SL upward when trail activates
                trigger_id = trade.get("trigger_order_id")
                if trigger_id:
                    try:
                        from paper_trader import _get_trail_config
                        trail_pct, activation_pct = _get_trail_config(trade)
                        if trail_pct is not None:
                            entry_p = float(trade.get("entry_price") or 0)
                            activation_price = entry_p * (1 + activation_pct)
                            if new_high >= activation_price:
                                new_sl = new_high * (1 - trail_pct)
                                old_sl = float(trade.get("sl_price") or 0)
                                if new_sl > old_sl:
                                    from jupiter_trigger import update_sl_price
                                    if update_sl_price(trigger_id, new_sl):
                                        client_sb.table("paper_trades").update(
                                            {"sl_price": new_sl}
                                        ).eq("id", trade["id"]).execute()
                                        logger.info(
                                            "TRAIL PATCH: %s SL $%.8f → $%.8f (high=$%.8f)",
                                            trade["symbol"], old_sl, new_sl, new_high,
                                        )
                                        _log_trigger_event(
                                            client_sb, trade_id=trade["id"], token_ca=addr,
                                            token_symbol=trade["symbol"], order_id=trigger_id,
                                            event_type="patched", old_sl=old_sl, new_sl=new_sl,
                                            current_price=current_price, high_price=new_high,
                                        )
                    except Exception as e:
                        logger.debug("trail PATCH failed for %s: %s", trade["symbol"], e)

        new_status = ev.get("status")
        if new_status is None:
            continue

        exit_price = ev.get("exit_price")
        # v141: paper_exit_price reflects what paper_trader ACTUALLY would have booked
        # (dynamic slip + fee + Ultra SELL quote). Falls back to decision-ev exit_price
        # on closing_retry path or if Ultra quote unavailable.
        paper_exit_price = (
            _paper_sim_ev.get("exit_price")
            if (_paper_sim_ev and _paper_sim_ev.get("exit_price"))
            else exit_price
        )
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

        # v121: Claim trade atomically — prevents race condition where another loop
        # (paper_fast, reconcile, restart) closes the same trade concurrently.
        # We set status='closing' so no other loop can pick it up.
        # Skip claim for trades already in 'closing' (retry from previous failed sell).
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
                    logger.info(
                        "live_trader: trade %s (%s) already closed by another loop, skipping sell",
                        trade["symbol"], trade["id"],
                    )
                    continue
            except Exception as e:
                logger.warning("live_trader: claim failed for %s: %s — skipping", trade["symbol"], e)
                continue

        # v119: Cancel trigger order before market selling (avoid double-sell)
        trigger_id = trade.get("trigger_order_id")
        if trigger_id:
            try:
                from jupiter_trigger import cancel_order
                cancel_order(trigger_id)
                _log_trigger_event(
                    client_sb, trade_id=trade["id"], token_ca=addr,
                    token_symbol=trade["symbol"], order_id=trigger_id,
                    event_type="cancelled", current_price=current_price,
                    high_price=float(trade.get("high_price_seen") or 0),
                    details={"reason": new_status, "polling_beat_keeper": True},
                )
            except Exception:
                pass  # Best effort — market sell proceeds regardless

        # v133-D: Sell only THIS strategy's tokens — hybrid allocation (FAST+DTRAIL on same
        # token) shares one ATA. Prior code called execute_sell(addr) without amount → fell
        # through to full wallet balance (live_trader.py:335) → drained both siblings. The
        # winning leg showed ~2× inflated exit_price/pnl; the losing leg got auto-closed as
        # reconciled at −100%. Passing buy_output_tokens caps each sell to that trade's share.
        # Fallback to None (full balance) only for legacy rows missing buy_output_tokens.
        sell_amount = int(trade.get("buy_output_tokens") or 0) or None
        sell_result = execute_sell(addr, amount_tokens=sell_amount)
        if not sell_result["success"]:
            # v121: Revert status from 'closing' back to 'open' so next cycle retries
            # Increment sell_attempts counter for congestion tracking
            try:
                cur_attempts = int(trade.get("sell_attempts") or 1)
                client_sb.table("paper_trades").update(
                    {"status": "open", "sell_attempts": cur_attempts + 1}
                ).eq("id", trade["id"]).execute()
            except Exception:
                pass
            logger.warning(
                "live_trader: sell failed for %s (%s) — reverted to open (retry next cycle): %s",
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
        # v142 (S3 audit fix): track exit_price provenance. When False, exit_price
        # stored in DB is the decision-time DS tick (no real Jupiter fill data) —
        # pnl_pct/pnl_usd derived from it are unreliable. Post-hoc analysis should
        # prefer reconstruction from stored columns:
        #   exit_price_net = entry_price * (sell_sol_received * sol_price_at_exit) / position_usd
        # Only valid when sell_sol_received IS NOT NULL.
        exit_price_from_fill = False

        try:
            if sell_output and sell_output > 0 and entry_price > 0:
                sol_received = sell_output / LAMPORTS_PER_SOL
                sol_price_now = _get_sol_price_usd()
                usd_received = sol_received * sol_price_now
                pos_usd_val = float(trade.get("position_usd") or 0)
                if pos_usd_val > 0:
                    exit_price = entry_price * (usd_received / pos_usd_val)
                    exit_price_from_fill = True

            pnl_pct = round((exit_price / entry_price) - 1, 4) if exit_price and entry_price else 0
            pnl_usd = round(pos_usd * pnl_pct, 2) if pos_usd else 0

            # Track daily PnL in SOL
            sol_price = _get_sol_price_usd()
            pnl_sol = pnl_usd / sol_price if sol_price > 0 else 0
            _track_pnl(pnl_sol)

            # v133-C: Sell slippage = actual fill vs market spot at sell trigger.
            # Prior formula compared usd_received vs pos_usd*(1+pnl_pct), but pnl_pct
            # was itself derived from usd_received → always 0. Now: compare
            # actual_fill (exit_price from Jupiter output) vs current_price (DS/Jupiter
            # spot at decision moment) — the metric that actually measures execution cost.
            sol_price_at_exit = _get_sol_price_usd()
            if exit_price and current_price and current_price > 0:
                # Negative = fill worse than quote (slippage cost). Positive = price moved in our favor.
                sell_slippage_bps = round((exit_price / current_price - 1) * 10000)

            # v105: Alert on high sell slippage
            if abs(sell_slippage_bps) > 1000:
                from alerter import alert_slippage_deviation
                alert_slippage_deviation(trade["symbol"], 1000, sell_slippage_bps)

            sell_sol_received = (sell_output / LAMPORTS_PER_SOL) if sell_output else None
        except Exception as e:
            logger.warning("live_trader: PnL calc error for %s (sell OK, sig=%s): %s — DB update proceeds with defaults",
                           trade["symbol"], sell_result.get("signature"), e)
            sell_sol_received = (sell_output / LAMPORTS_PER_SOL) if sell_output else None

        # v142 (S3): surface the fallback case explicitly so it can be measured.
        # Normal path: sell_output > 0 → exit_price reflects real Jupiter fill.
        # Fallback: exit_price == ev.exit_price (DS tick at decision moment), unreliable.
        if not exit_price_from_fill:
            logger.warning(
                "live_trader: exit_price for %s (%s) is DS-tick fallback, not Jupiter fill "
                "(sell_output=%s, pos_usd=%.2f). pnl_pct/pnl_usd unreliable for this row. "
                "Reconstruct via sell_sol_received if available.",
                trade["symbol"], new_status, sell_output, pos_usd,
            )

        # v122: Compute paper vs live price divergence
        price_divergence_pct = None
        if paper_exit_price and paper_exit_price > 0 and exit_price and exit_price > 0:
            price_divergence_pct = round((exit_price / paper_exit_price) - 1, 4)

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
            # v133-C: Round-trip real slippage (buy + sell), populated so monitoring works.
            "slippage_actual_bps": int((trade.get("buy_slippage_bps") or 0)) + int(sell_slippage_bps or 0),
            "sol_price_at_exit": sol_price_at_exit,
            # v117: Extended sell tracking
            "sell_exec_ms": sell_result.get("exec_ms"),
            "sell_output_lamports": sell_output,  # SOL received (lamports)
            "sell_input_tokens": sell_result.get("input_amount"),  # tokens sold (raw)
            "sell_sol_received": round(sell_sol_received, 6) if sell_sol_received else None,
            # v122: Paper vs live price divergence tracking
            "paper_exit_price": paper_exit_price,
            "price_divergence_pct": price_divergence_pct,
            # v143.6: paper-accurate PnL (same slip/fee pipeline as paper_trader)
            # for direct divergence measurement without re-running sim replay.
            "paper_sim_pnl_pct": (
                round(float(_paper_sim_ev.get("pnl_pct")), 4)
                if _paper_sim_ev and _paper_sim_ev.get("pnl_pct") is not None else None
            ),
        }
        # v138: persist accumulated poll history alongside close fields
        from paper_trader import _flush_eval_history
        hist = _flush_eval_history(trade["id"])
        if hist:
            update["eval_history"] = hist

        # DB update with retry — sell already executed, must not leave trade as 'open'
        db_updated = False
        for attempt in range(3):
            try:
                client_sb.table("paper_trades").update(update).eq("id", trade["id"]).execute()
                db_updated = True
                break
            except Exception as e:
                err_str = str(e)
                # v143.6: tolerate the column not yet existing — migration v13
                # may not have been applied to Supabase yet. Drop the field and
                # retry in the same attempt so we don't burn retry budget.
                if "paper_sim_pnl_pct" in err_str and "paper_sim_pnl_pct" in update:
                    logger.info("live_trader: paper_sim_pnl_pct column missing "
                                "(apply supabase/migrations/v13_paper_sim_pnl.sql); "
                                "retrying update without it")
                    update.pop("paper_sim_pnl_pct", None)
                    try:
                        client_sb.table("paper_trades").update(update).eq("id", trade["id"]).execute()
                        db_updated = True
                        break
                    except Exception as e2:
                        e = e2
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

        # v143.5 — exit shadow-sync : force-close the matching still-open paper
        # trade (same token + strategy + cycle_ts) at live's Jupiter fill price.
        # Symmetric to v142 E (entry sync). Only touches still-open paper rows —
        # paper trades that already exited via their own decision are preserved
        # as independent data points. Keeps median exit divergence <1%.
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
                pm_pnl_pct = (exit_price / pm_entry) - 1 if pm_entry > 0 else 0
                pm_pnl_usd = pm_pos_usd * pm_pnl_pct
                paper_update = {
                    "status": new_status,
                    "exit_price": exit_price,
                    "exit_at": now.isoformat(),
                    "pnl_pct": round(pm_pnl_pct, 4),
                    "pnl_usd": round(pm_pnl_usd, 2),
                    "exit_minutes": int(elapsed_minutes),
                    "sol_price_at_exit": sol_price_at_exit,
                }
                hist_p = _flush_eval_history(pm["id"])
                if hist_p:
                    paper_update["eval_history"] = hist_p
                client_sb.table("paper_trades").update(paper_update).eq("id", pm["id"]).execute()
                logger.info(
                    "paper_exit_sync: %s %s — paper#%s closed at live exit $%.8f (pnl=%.1f%%)",
                    trade["symbol"], trade["strategy"], pm["id"], exit_price, pm_pnl_pct * 100,
                )
        except Exception as _e:
            logger.debug("paper_exit_sync skipped for %s: %s", trade.get("symbol"), _e)

        result_counts["closed"] += 1
        result_counts["pnl_usd"] += pnl_usd
        result_counts["rt_pnl_usd"] += pnl_usd
        status_key = new_status.replace("_hit", "").replace("_stop", "")
        result_counts[status_key] = result_counts.get(status_key, 0) + 1

        div_str = f" div={price_divergence_pct:+.0%}" if price_divergence_pct else ""
        logger.info(
            "LIVE TRADE CLOSED: %s %s — %s pnl=%.1f%% $%+.2f%s | sell sig: %s",
            trade["symbol"], trade["strategy"], new_status,
            pnl_pct * 100, pnl_usd, div_str, sell_result["signature"][:16],
        )
        # v125: Structured divergence log for analysis
        if price_divergence_pct is not None:
            logger.info(
                "DIVERGENCE: %s %s exit_type=%s paper=$%.8f actual=$%.8f delta=%.2f%%",
                trade["symbol"], trade["strategy"], new_status,
                paper_exit_price or 0, exit_price or 0, price_divergence_pct * 100,
            )

        # Alert via Telegram — v119: detailed sell alert
        # v144.11: enriched with real bankroll + per-strategy bankrolls + paper-vs-live drift
        # (both per-trade using paper_sim_pnl_pct from this row, and rolling 24h per strat).
        try:
            from alerter import alert_live_sell, _live_paper_strategy_drift_24h
            _br, _strat_bals, _active = {}, {}, None
            try:
                from safe_scraper import _rt_load_bankroll, _rt_strategy_bankrolls_for_chain
                _br = _rt_load_bankroll() or {}
                # v14e: live_trader is Solana-only — scope alert bankroll display.
                _strat_bals = _rt_strategy_bankrolls_for_chain(_br, "solana")
                _active = list(_strat_bals.keys()) if _strat_bals else None
            except Exception:
                pass
            _paper_sim_pnl = (
                float(_paper_sim_ev.get("pnl_pct"))
                if _paper_sim_ev and _paper_sim_ev.get("pnl_pct") is not None else None
            )
            _drift_24h = {}
            try:
                _drift_24h = _live_paper_strategy_drift_24h(client_sb) or {}
            except Exception:
                pass
            alert_live_sell(
                symbol=trade["symbol"], strategy=trade.get("strategy", ""),
                exit_reason=new_status,
                pnl_pct=pnl_pct, pnl_usd=pnl_usd,
                position_usd=pos_usd,
                entry_price=entry_price, exit_price=exit_price,
                sol_received=sell_sol_received or 0,
                minutes=int(elapsed_minutes),
                kol=trade.get("kol_group", ""),
                signature=sell_result["signature"],
                slippage_bps=sell_slippage_bps,
                exec_ms=sell_result.get("exec_ms", 0),
                sol_price=sol_price_at_exit,
                ca=trade.get("token_address", ""),
                high_price=float(trade.get("high_price_seen") or 0),
                bankroll=float(_br.get("current_balance", 0) or 0),
                strategy_bankrolls=_strat_bals,
                active_strategies=_active,
                paper_pnl_pct=_paper_sim_pnl,
                price_divergence_pct=price_divergence_pct,
                strategy_drift_24h=_drift_24h,
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


def _find_sibling_exit(client_sb, token_ca: str, exclude_trade_id, window_minutes: int = 15) -> dict | None:
    """v133-D: Return the most recent live sibling exit for this token within `window_minutes`.
    Used by reconcile_positions to recover pnl when a hybrid-allocation sibling drained the
    shared ATA. Requires sell_input_tokens + sell_sol_received to compute SOL-per-token.
    """
    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=window_minutes)).isoformat()
    try:
        resp = (
            client_sb.table("paper_trades")
            .select("id, entry_price, position_usd, exit_price, exit_at, "
                    "buy_output_tokens, sell_input_tokens, sell_output_lamports, "
                    "sell_sol_received, sol_price_at_exit, tx_signature_exit, pnl_pct, status")
            .eq("source", "rt_live")
            .eq("token_address", token_ca)
            .neq("id", exclude_trade_id)
            .gte("exit_at", cutoff)
            .not_.is_("tx_signature_exit", "null")
            .order("exit_at", desc=True)
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            return None
        s = rows[0]
        # Need both input tokens + output SOL to derive sol-per-token.
        if not s.get("sell_input_tokens") or not (s.get("sell_output_lamports") or s.get("sell_sol_received")):
            return None
        return s
    except Exception as e:
        logger.debug("_find_sibling_exit failed for %s: %s", token_ca[:12], e)
        return None


def _reconcile_close_payload(trade: dict, sibling: dict | None) -> dict:
    """v133-D: Build DB update for a reconciled trade.
    With sibling: derive exit_price from sibling's SOL-per-token × this trade's
    buy_output_tokens. Without: close neutrally (pnl=0) rather than hardcoding -100%,
    which was poisoning aggregate paper-vs-live analytics.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    base = {"status": "reconciled", "exit_at": now_iso}
    entry_price = float(trade.get("entry_price") or 0)
    pos_usd = float(trade.get("position_usd") or 0)
    our_tokens = int(trade.get("buy_output_tokens") or 0)

    if sibling and entry_price > 0 and pos_usd > 0 and our_tokens > 0:
        sibling_tokens = int(sibling.get("sell_input_tokens") or 0)
        sibling_sol = float(sibling.get("sell_sol_received") or 0)
        if not sibling_sol and sibling.get("sell_output_lamports"):
            sibling_sol = float(sibling["sell_output_lamports"]) / LAMPORTS_PER_SOL
        sol_price = float(sibling.get("sol_price_at_exit") or 0) or _get_sol_price_usd()
        if sibling_tokens > 0 and sibling_sol > 0 and sol_price > 0:
            sol_per_token = sibling_sol / sibling_tokens
            our_sol = our_tokens * sol_per_token
            our_usd = our_sol * sol_price
            exit_price = entry_price * (our_usd / pos_usd)
            pnl_pct = round((exit_price / entry_price) - 1, 4)
            pnl_usd = round(pos_usd * pnl_pct, 2)
            base.update({
                "exit_price": exit_price,
                "pnl_pct": pnl_pct,
                "pnl_usd": pnl_usd,
                "sell_sol_received": round(our_sol, 6),
                "sol_price_at_exit": sol_price,
            })
            return base

    # Fallback: no sibling or missing data. Close neutral rather than phantom -100%.
    base.update({"pnl_pct": 0.0, "pnl_usd": 0.0})
    return base


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
        # v121: Also check 'closing' trades (sell may have succeeded but DB update crashed)
        # v133-D: Include buy_output_tokens to detect sibling-sold case.
        resp = (
            client_sb.table("paper_trades")
            .select("id, symbol, token_address, entry_price, position_usd, "
                    "buy_output_tokens, created_at, status")
            .in_("status", ["open", "closing"])
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
            # DB says open, but no on-chain balance → possibilities:
            # (1) sibling strategy already sold the shared ATA (hybrid allocation pre-v133-D),
            # (2) user sold manually, (3) rug/freeze, (4) genuine failure.
            # v133-D: try to find a sibling exit that explains (1) before booking a phantom -100%.
            result["mismatches"] += 1
            sibling = _find_sibling_exit(client_sb, ca, trade["id"])
            update_row = _reconcile_close_payload(trade, sibling)
            issue = "db_open_but_no_balance_via_sibling" if sibling else "db_open_but_no_balance"
            detail = {
                "id": trade["id"],
                "symbol": trade["symbol"],
                "ca": ca,
                "issue": issue,
                "pnl_pct": update_row.get("pnl_pct"),
                "pnl_usd": update_row.get("pnl_usd"),
            }
            result["details"].append(detail)
            if sibling:
                logger.warning(
                    "RECONCILE SIBLING: %s (%s) drained by prior sell tx=%s — closing at "
                    "pnl=%.2f%% (%.2f$) from sibling's realized SOL-per-token.",
                    trade["symbol"], ca[:12],
                    (sibling.get("tx_signature_exit") or "")[:16],
                    (update_row.get("pnl_pct") or 0) * 100,
                    update_row.get("pnl_usd") or 0,
                )
            else:
                logger.warning(
                    "RECONCILE MISMATCH: %s (%s) open in DB but 0 on-chain balance and no "
                    "sibling exit found — closing neutral (pnl=0) for manual investigation.",
                    trade["symbol"], ca[:12],
                )
            try:
                client_sb.table("paper_trades").update(update_row).eq("id", trade["id"]).execute()
                result["auto_closed"] += 1
                # v136.2: update rt_bankroll on auto-close to keep ledger coherent.
                # Prior behavior: reconcile closed trades silently without touching bankroll
                # → total_pnl drifted from paper_trades ground truth by the reconciled amount.
                try:
                    from safe_scraper import _rt_update_bankroll
                    # v14e: live_trader is Solana-only (enforced by chain gate).
                    _rt_update_bankroll(
                        float(update_row.get("pnl_usd") or 0),
                        1,
                        strategy="",  # reconciled close — no specific strategy bucket
                        chain="solana",
                    )
                except Exception as be:
                    logger.debug("reconcile: bankroll update failed for %s: %s", trade["id"], be)
            except Exception as e:
                logger.error("reconcile: failed to close trade %s: %s", trade["id"], e)

    # Check reverse: on-chain tokens not tracked in DB (orphaned positions)
    # v119: Auto-sell orphaned tokens to recover SOL
    tracked_cas = {t["token_address"] for t in open_trades if t.get("token_address")}
    for mint in on_chain_mints:
        if mint not in tracked_cas and mint != WSOL_MINT:
            bal = balances["token_balances"][mint]
            if bal.get("ui_amount", 0) > 0:
                logger.warning(
                    "RECONCILE: on-chain token %s (%.4f) not tracked in DB — orphaned position, attempting auto-sell",
                    mint[:12], bal["ui_amount"],
                )
                result["mismatches"] += 1
                # Try to sell the orphaned tokens
                sell_res = execute_sell(mint, slippage_bps=1000)  # higher slippage for stale tokens
                if sell_res["success"]:
                    sol_recovered = (sell_res.get("output_amount") or 0) / LAMPORTS_PER_SOL
                    logger.info(
                        "RECONCILE: auto-sold orphan %s — recovered %.4f SOL (sig: %s)",
                        mint[:12], sol_recovered, sell_res.get("signature", "")[:16],
                    )
                    result["auto_closed"] += 1
                    result["details"].append({
                        "ca": mint,
                        "issue": "orphan_auto_sold",
                        "ui_amount": bal["ui_amount"],
                        "sol_recovered": sol_recovered,
                        "signature": sell_res.get("signature"),
                    })
                else:
                    logger.warning(
                        "RECONCILE: failed to sell orphan %s: %s",
                        mint[:12], sell_res.get("error"),
                    )
                    result["details"].append({
                        "ca": mint,
                        "issue": "on_chain_but_not_in_db",
                        "ui_amount": bal["ui_amount"],
                        "sell_error": sell_res.get("error"),
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
