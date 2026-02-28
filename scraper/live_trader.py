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
_monthly_pnl_sol: float = 0.0
_monthly_pnl_reset_month: str = ""


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

        # Holdings response: dict of mint -> {amount, uiAmount, ...}
        if isinstance(data, dict):
            for mint, info in data.items():
                if not isinstance(info, dict):
                    continue
                ui_amount = float(info.get("uiAmount", 0))
                amount = int(info.get("amount", 0))
                if mint == "SOL" or mint == WSOL_MINT:
                    sol_balance = ui_amount
                else:
                    if amount > 0:
                        token_balances[mint] = {"amount": amount, "ui_amount": ui_amount}

        return {"sol_balance": sol_balance, "token_balances": token_balances}
    except Exception as e:
        logger.error("live_trader: holdings fetch failed: %s", e)
        return None


def execute_buy(ca: str, amount_sol_lamports: int, slippage_bps: int = 300) -> dict:
    """
    Execute a buy swap: SOL → token via Jupiter Ultra.
    Returns {"success": bool, "signature": str, "status": str, "error": str|None,
             "input_amount": int|None, "output_amount": int|None}
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

        response = client.order_and_execute(order)
        status = response.get("status", "Unknown")
        signature = str(response.get("signature", ""))
        success = status == "Success"

        # v74: Extract actual fill amounts from Jupiter response
        input_amount = _safe_int(response.get("inputAmountResult"))
        output_amount = _safe_int(response.get("outputAmountResult"))

        if success:
            logger.info(
                "LIVE BUY: %s | %s SOL → %s tokens (sig: %s...)",
                ca[:12], amount_sol_lamports / LAMPORTS_PER_SOL,
                output_amount or "?", signature[:16],
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
        }
    except Exception as e:
        logger.error("LIVE BUY ERROR: %s | %s", ca[:12], e)
        return {"success": False, "signature": "", "error": str(e)}


def execute_sell(ca: str, amount_tokens: int | None = None, slippage_bps: int = 500) -> dict:
    """
    Execute a sell swap: token → SOL via Jupiter Ultra.
    If amount_tokens is None, sells entire balance of that token.
    Returns {"success": bool, "signature": str, "status": str, "error": str|None,
             "input_amount": int|None, "output_amount": int|None}
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

        response = client.order_and_execute(order)
        status = response.get("status", "Unknown")
        signature = str(response.get("signature", ""))
        success = status == "Success"

        # v74: Extract actual fill amounts
        input_amount = _safe_int(response.get("inputAmountResult"))
        output_amount = _safe_int(response.get("outputAmountResult"))

        if success:
            sol_received = output_amount / LAMPORTS_PER_SOL if output_amount else "?"
            logger.info(
                "LIVE SELL: %s | %d tokens → %s SOL (sig: %s...)",
                ca[:12], amount_tokens, sol_received, signature[:16],
            )
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
        }
    except Exception as e:
        logger.error("LIVE SELL ERROR: %s | %s", ca[:12], e)
        return {"success": False, "signature": "", "error": str(e)}


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
    """
    global _daily_pnl_sol, _daily_pnl_reset_date, _daily_halted
    global _weekly_pnl_sol, _weekly_pnl_reset_week
    global _monthly_pnl_sol, _monthly_pnl_reset_month

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
    if _monthly_pnl_reset_month != month:
        _monthly_pnl_sol = 0.0
        _monthly_pnl_reset_month = month

    daily_limit = float(config.get("daily_loss_limit_sol", 2.0))
    weekly_limit = float(config.get("weekly_loss_limit_sol", 5.0))
    monthly_limit = float(config.get("monthly_loss_limit_sol", 10.0))

    if _daily_pnl_sol < -daily_limit:
        if not _daily_halted:
            logger.warning("LIVE TRADING HALTED: daily loss %.4f SOL exceeds limit %.1f SOL",
                           _daily_pnl_sol, daily_limit)
            _daily_halted = True
        return True
    if _weekly_pnl_sol < -weekly_limit:
        logger.warning("LIVE TRADING HALTED: weekly loss %.4f SOL exceeds limit %.1f SOL",
                       _weekly_pnl_sol, weekly_limit)
        return True
    if _monthly_pnl_sol < -monthly_limit:
        logger.warning("LIVE TRADING HALTED: monthly loss %.4f SOL exceeds limit %.1f SOL",
                       _monthly_pnl_sol, monthly_limit)
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
    tranches = STRATEGIES.get(strategy, [{"tp_mult": 2.0, "sl_mult": 0.70, "horizon_min": 1440}])
    tranche = tranches[0]  # Live trades always use first tranche

    tp_price = execution_price * tranche["tp_mult"] if tranche.get("tp_mult") else None
    sl_price = execution_price * tranche["sl_mult"]

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
        "horizon_minutes": tranche.get("horizon_min", 1440),
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
    Check all open live trades against current prices.
    For TP/SL/timeout hits: execute sell BEFORE updating DB.
    Returns {"checked": N, "closed": M, "tp": X, "sl": Y, "timeout": Z, "pnl_usd": total}.
    """
    from paper_trader import _fetch_prices_batch
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
        entry_price = float(trade["entry_price"])
        sl_price = float(trade["sl_price"])
        tp_price = float(trade["tp_price"]) if trade.get("tp_price") is not None else None
        horizon = trade.get("horizon_minutes", 1440)

        created_str = trade["created_at"]
        try:
            created_at = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
        except Exception:
            continue
        elapsed_minutes = (now - created_at).total_seconds() / 60

        new_status = None
        exit_price = None

        if current_price is not None:
            if current_price <= sl_price:
                new_status = "sl_hit"
                exit_price = sl_price
            elif tp_price is not None and current_price >= tp_price:
                new_status = "tp_hit"
                exit_price = tp_price

        if new_status is None and elapsed_minutes >= horizon:
            new_status = "timeout"
            exit_price = current_price if current_price else entry_price

        if new_status is None:
            continue

        # Execute sell BEFORE updating DB
        sell_result = execute_sell(addr)
        if not sell_result["success"]:
            logger.warning(
                "live_trader: sell failed for %s (%s) — keeping trade open (retry next cycle): %s",
                trade["symbol"], new_status, sell_result.get("error"),
            )
            continue

        # v74: Use actual SOL received from Jupiter to compute real exit price
        sell_output = sell_result.get("output_amount")  # SOL lamports received
        if sell_output and sell_output > 0 and entry_price > 0:
            sol_received = sell_output / LAMPORTS_PER_SOL
            sol_price_now = _get_sol_price_usd()
            usd_received = sol_received * sol_price_now
            pos_usd_val = float(trade.get("position_usd") or 0)
            if pos_usd_val > 0:
                exit_price = entry_price * (usd_received / pos_usd_val)

        pnl_pct = round((exit_price / entry_price) - 1, 4) if exit_price and entry_price else 0
        pos_usd = float(trade.get("position_usd") or 0)
        pnl_usd = round(pos_usd * pnl_pct, 2) if pos_usd else 0

        # Track daily PnL in SOL
        sol_price = _get_sol_price_usd()
        pnl_sol = pnl_usd / sol_price if sol_price > 0 else 0
        _track_pnl(pnl_sol)

        update = {
            "status": new_status,
            "exit_price": exit_price,
            "exit_at": now.isoformat(),
            "pnl_pct": pnl_pct,
            "pnl_usd": pnl_usd,
            "exit_minutes": int(elapsed_minutes),
            "tx_signature_exit": sell_result["signature"],
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
        status_key = new_status.replace("_hit", "")
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
