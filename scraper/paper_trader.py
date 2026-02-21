"""
Paper Trading System v3 — Multi-strategy with tranche support + portfolio allocation.

4 strategies run in parallel per token:
- TP50_SL30:  100% at 1.5x, -30% SL, 12h horizon (conservative baseline)
- TP100_SL30: 100% at 2x,   -30% SL, 24h horizon (wait for double)
- SCALE_OUT:  25% tranches at 2x/3x/5x + moonbag, -30% SL, 48h horizon
- MOONBAG:    80% at 2x + 20% moonbag, -50% SL, 7d horizon

Each tranche = 1 row in paper_trades. SL triggers close ALL open tranches
for the same token+strategy. Moonbag tranches (tp_price=NULL) only close
on SL or timeout.

v3: Score-weighted portfolio allocation. $50 budget per cycle split
proportionally by token score. Tracks position_usd and pnl_usd.
"""

import logging
from datetime import datetime, timezone, timedelta

import requests

logger = logging.getLogger(__name__)

DEXSCREENER_BATCH_URL = "https://api.dexscreener.com/tokens/v1/solana/{addresses}"
BATCH_SIZE = 30

TOP_N = 5
PORTFOLIO_BUDGET = 50.0  # USD per cycle, score-weighted across top N

# --- Strategy Definitions ---
# Each strategy has a list of tranches. Moonbag tranches have tp_mult=None.
STRATEGIES = {
    "TP50_SL30": [
        {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.70, "horizon_min": 720, "label": "main"},
    ],
    "TP100_SL30": [
        {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.70, "horizon_min": 1440, "label": "main"},
    ],
    "SCALE_OUT": [
        {"pct": 0.25, "tp_mult": 2.00, "sl_mult": 0.70, "horizon_min": 2880, "label": "tp_2x"},
        {"pct": 0.25, "tp_mult": 3.00, "sl_mult": 0.70, "horizon_min": 2880, "label": "tp_3x"},
        {"pct": 0.25, "tp_mult": 5.00, "sl_mult": 0.70, "horizon_min": 2880, "label": "tp_5x"},
        {"pct": 0.25, "tp_mult": None, "sl_mult": 0.70, "horizon_min": 2880, "label": "moonbag"},
    ],
    "MOONBAG": [
        {"pct": 0.80, "tp_mult": 2.00, "sl_mult": 0.50, "horizon_min": 10080, "label": "main"},
        {"pct": 0.20, "tp_mult": None, "sl_mult": 0.50, "horizon_min": 10080, "label": "moonbag"},
    ],
}


def _fetch_prices_batch(addresses: list[str]) -> dict[str, float]:
    """Batch fetch current USD prices from DexScreener. Returns {address: price}."""
    if not addresses:
        return {}
    prices = {}
    for i in range(0, len(addresses), BATCH_SIZE):
        chunk = addresses[i:i + BATCH_SIZE]
        addr_str = ",".join(chunk)
        try:
            resp = requests.get(
                DEXSCREENER_BATCH_URL.format(addresses=addr_str),
                timeout=15,
            )
            if resp.status_code != 200:
                logger.warning("paper_trader: DexScreener batch %d", resp.status_code)
                continue
            data = resp.json()
            pairs = data if isinstance(data, list) else data.get("pairs", [])
            if not isinstance(pairs, list):
                continue
            # Pick highest-volume pair per base token address
            by_addr: dict[str, list] = {}
            for p in pairs:
                addr = p.get("baseToken", {}).get("address", "")
                if addr:
                    by_addr.setdefault(addr, []).append(p)
            for addr, token_pairs in by_addr.items():
                best = max(token_pairs, key=lambda p: float(p.get("volume", {}).get("h24", 0) or 0))
                price = best.get("priceUsd")
                if price:
                    try:
                        prices[addr] = float(price)
                    except (ValueError, TypeError):
                        pass
        except requests.RequestException as e:
            logger.warning("paper_trader: DexScreener batch error: %s", e)
    return prices


def open_paper_trades(client, ranking: list[dict], cycle_ts: datetime) -> int:
    """
    Open paper trades for top N tokens across all 4 strategies.
    Each strategy may have multiple tranches (e.g. SCALE_OUT has 4 rows per token).
    Dedup: skip if token_address + strategy already has an open trade.
    Returns number of new trade rows opened.
    """
    candidates = [
        t for t in ranking
        if t.get("score", 0) > 0
        and t.get("token_address")
        and t.get("price_usd") and float(t["price_usd"]) > 0
    ][:TOP_N]

    if not candidates:
        return 0

    # Score-weighted portfolio allocation
    scores = [max(t.get("score", 1), 1) for t in candidates]
    total_score = sum(scores)
    for i, token in enumerate(candidates):
        token["_alloc_usd"] = round(PORTFOLIO_BUDGET * scores[i] / total_score, 2)

    # Check which (token_address, strategy) combos already have open trades
    addrs = [t["token_address"] for t in candidates]
    try:
        existing = (
            client.table("paper_trades")
            .select("token_address, strategy")
            .eq("status", "open")
            .in_("token_address", addrs)
            .execute()
        )
        open_combos = {
            (r["token_address"], r["strategy"]) for r in (existing.data or [])
        }
    except Exception as e:
        logger.error("paper_trader: failed to check open trades: %s", e)
        open_combos = set()

    opened = 0
    for rank_idx, token in enumerate(candidates, 1):
        addr = token["token_address"]
        entry_price = float(token["price_usd"])
        alloc_usd = token.get("_alloc_usd", PORTFOLIO_BUDGET / TOP_N)

        # Common fields for all tranches of this token
        base_row = {
            "cycle_ts": cycle_ts.isoformat(),
            "symbol": token.get("symbol", "???"),
            "token_address": addr,
            "rank_in_cycle": rank_idx,
            "entry_price": entry_price,
            "entry_score": int(token.get("score", 0)),
            "entry_mcap": float(token["market_cap"]) if token.get("market_cap") else None,
            "status": "open",
            "unique_kols": token.get("unique_kols"),
            "whale_new_entries": token.get("whale_new_entries"),
            "momentum_mult": float(token["momentum_mult"]) if token.get("momentum_mult") else None,
            "portfolio_budget": PORTFOLIO_BUDGET,
        }
        if token.get("snapshot_id"):
            base_row["snapshot_id"] = int(token["snapshot_id"])

        for strat_name, tranches in STRATEGIES.items():
            if (addr, strat_name) in open_combos:
                continue

            for tranche in tranches:
                tp_price = entry_price * tranche["tp_mult"] if tranche["tp_mult"] else None
                sl_price = entry_price * tranche["sl_mult"]

                row = {
                    **base_row,
                    "strategy": strat_name,
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                    "horizon_minutes": tranche["horizon_min"],
                    "tranche_pct": tranche["pct"],
                    "tranche_label": tranche["label"],
                    "position_usd": round(alloc_usd * tranche["pct"], 2),
                }

                try:
                    client.table("paper_trades").insert(row).execute()
                    opened += 1
                except Exception as e:
                    logger.error(
                        "paper_trader: insert failed for %s/%s/%s: %s",
                        token.get("symbol"), strat_name, tranche["label"], e,
                    )

    allocs = [f"{t.get('symbol','?')}=${t.get('_alloc_usd',0):.1f}" for t in candidates]
    logger.info(
        "paper_trader: opened %d rows, $%.0f budget → %s (%d strategies)",
        opened, PORTFOLIO_BUDGET, ", ".join(allocs), len(STRATEGIES),
    )
    return opened


def check_paper_trades(client) -> dict:
    """
    Check all open paper trades against current prices.
    Closes trades that hit TP, SL, or timeout.

    SL cascade: when SL triggers, ALL open tranches for the same
    (token_address, strategy, cycle_ts) close at -SL%.

    Moonbag tranches (tp_price=NULL) only close on SL or timeout.

    Returns {"checked": N, "closed": M, "tp": X, "sl": Y, "timeout": Z}.
    """
    now = datetime.now(timezone.utc)

    try:
        result = client.table("paper_trades").select("*").eq("status", "open").execute()
        open_trades = result.data or []
    except Exception as e:
        logger.error("paper_trader: failed to fetch open trades: %s", e)
        return {"checked": 0, "closed": 0, "tp": 0, "sl": 0, "timeout": 0}

    if not open_trades:
        return {"checked": 0, "closed": 0, "tp": 0, "sl": 0, "timeout": 0}

    # Batch fetch current prices
    addresses = list({t["token_address"] for t in open_trades})
    prices = _fetch_prices_batch(addresses)

    counts = {"checked": len(open_trades), "closed": 0, "tp": 0, "sl": 0, "timeout": 0}

    # Track SL-triggered groups so we can cascade
    # Key: (token_address, strategy, cycle_ts) -> True if SL was hit
    sl_triggered = set()

    # First pass: detect SL triggers (check non-moonbag trades first for SL detection)
    # Sort so main/tp tranches come before moonbag
    sorted_trades = sorted(open_trades, key=lambda t: (t.get("tranche_label", "") == "moonbag"))

    closed_ids = set()

    for trade in sorted_trades:
        if trade["id"] in closed_ids:
            continue

        addr = trade["token_address"]
        current_price = prices.get(addr)

        created_str = trade["created_at"]
        try:
            created_at = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
        except Exception:
            continue

        elapsed_minutes = (now - created_at).total_seconds() / 60
        horizon = trade.get("horizon_minutes", 720)
        entry_price = float(trade["entry_price"])
        sl_price = float(trade["sl_price"])
        tp_price = float(trade["tp_price"]) if trade.get("tp_price") is not None else None

        group_key = (addr, trade["strategy"], trade["cycle_ts"])

        new_status = None
        exit_price = None

        # Check if this group already had SL triggered by a sibling tranche
        if group_key in sl_triggered:
            new_status = "sl_hit"
            exit_price = current_price if current_price else entry_price * (sl_price / entry_price)

        elif current_price is not None:
            # SL check (applies to all tranches including moonbag)
            if current_price <= sl_price:
                new_status = "sl_hit"
                exit_price = current_price
                sl_triggered.add(group_key)
            # TP check (only for tranches with a TP target)
            elif tp_price is not None and current_price >= tp_price:
                new_status = "tp_hit"
                exit_price = current_price

        # Timeout check
        if new_status is None and elapsed_minutes >= horizon:
            new_status = "timeout"
            exit_price = current_price if current_price else entry_price

        if new_status is None:
            continue

        pnl_pct = round((exit_price / entry_price) - 1, 4) if exit_price and entry_price else 0
        pos_usd = float(trade.get("position_usd") or 0)
        pnl_usd = round(pos_usd * pnl_pct, 2) if pos_usd else None

        update = {
            "status": new_status,
            "exit_price": exit_price,
            "exit_at": now.isoformat(),
            "pnl_pct": pnl_pct,
            "pnl_usd": pnl_usd,
            "exit_minutes": int(elapsed_minutes),
        }

        try:
            client.table("paper_trades").update(update).eq("id", trade["id"]).execute()
            closed_ids.add(trade["id"])
            counts["closed"] += 1
            status_key = new_status.replace("_hit", "")
            counts[status_key] = counts.get(status_key, 0) + 1
            usd_str = f" ${pnl_usd:+.2f}" if pnl_usd is not None else ""
            logger.info(
                "paper_trader: CLOSED %s %s/%s/%s — %s pnl=%.1f%%%s after %dmin",
                trade["symbol"], trade["strategy"], trade.get("tranche_label", "main"),
                addr[:8], new_status, pnl_pct * 100, usd_str, int(elapsed_minutes),
            )
        except Exception as e:
            logger.error("paper_trader: update failed for trade %s: %s", trade["id"], e)

    # Second pass: close remaining open trades in SL-triggered groups
    for trade in open_trades:
        if trade["id"] in closed_ids:
            continue
        group_key = (trade["token_address"], trade["strategy"], trade["cycle_ts"])
        if group_key not in sl_triggered:
            continue

        addr = trade["token_address"]
        current_price = prices.get(addr)
        entry_price = float(trade["entry_price"])
        sl_price = float(trade["sl_price"])

        created_str = trade["created_at"]
        try:
            created_at = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
        except Exception:
            continue
        elapsed_minutes = (now - created_at).total_seconds() / 60

        exit_price = current_price if current_price else entry_price * (sl_price / entry_price)
        pnl_pct = round((exit_price / entry_price) - 1, 4) if exit_price and entry_price else 0
        pos_usd = float(trade.get("position_usd") or 0)
        pnl_usd = round(pos_usd * pnl_pct, 2) if pos_usd else None

        update = {
            "status": "sl_hit",
            "exit_price": exit_price,
            "exit_at": now.isoformat(),
            "pnl_pct": pnl_pct,
            "pnl_usd": pnl_usd,
            "exit_minutes": int(elapsed_minutes),
        }

        try:
            client.table("paper_trades").update(update).eq("id", trade["id"]).execute()
            closed_ids.add(trade["id"])
            counts["closed"] += 1
            counts["sl"] = counts.get("sl", 0) + 1
            usd_str = f" ${pnl_usd:+.2f}" if pnl_usd is not None else ""
            logger.info(
                "paper_trader: CLOSED (SL cascade) %s %s/%s — pnl=%.1f%%%s",
                trade["symbol"], trade["strategy"], trade.get("tranche_label", ""),
                pnl_pct * 100, usd_str,
            )
        except Exception as e:
            logger.error("paper_trader: update failed for trade %s: %s", trade["id"], e)

    if counts["closed"] > 0:
        logger.info(
            "paper_trader: checked %d open, closed %d (TP=%d SL=%d timeout=%d)",
            counts["checked"], counts["closed"], counts["tp"], counts["sl"], counts["timeout"],
        )
    return counts


def paper_trade_summary(client) -> dict | None:
    """
    Compute summary stats for closed paper trades (last 7 days).
    Per-strategy breakdown with weighted PnL for multi-tranche strategies.
    Returns summary dict or None if no trades.
    """
    try:
        result = (
            client.table("paper_trades")
            .select("*")
            .neq("status", "open")
            .gte("created_at", _days_ago_iso(7))
            .execute()
        )
        trades = result.data or []
    except Exception as e:
        logger.error("paper_trader: summary query failed: %s", e)
        return None

    if not trades:
        return None

    # Global stats
    total = len(trades)
    tp_count = sum(1 for t in trades if t["status"] == "tp_hit")
    sl_count = sum(1 for t in trades if t["status"] == "sl_hit")
    timeout_count = sum(1 for t in trades if t["status"] == "timeout")

    pnls = [float(t["pnl_pct"]) for t in trades if t.get("pnl_pct") is not None]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]

    win_rate = len(winners) / total if total else 0
    avg_pnl = sum(pnls) / len(pnls) if pnls else 0

    # Dollar PnL
    total_invested = sum(float(t.get("position_usd") or 0) for t in trades)
    total_pnl_usd = sum(float(t.get("pnl_usd") or 0) for t in trades)
    roi_pct = round(total_pnl_usd / total_invested * 100, 2) if total_invested else 0

    # Per-strategy breakdown
    strategy_stats = {}
    for strat_name in STRATEGIES:
        strat_trades = [t for t in trades if t.get("strategy") == strat_name]
        if not strat_trades:
            continue

        # Group by (token_address, cycle_ts) for weighted PnL across tranches
        positions = {}
        for t in strat_trades:
            key = (t["token_address"], t["cycle_ts"])
            positions.setdefault(key, []).append(t)

        pos_pnls = []
        pos_wins = 0
        for key, pos_trades in positions.items():
            weighted_pnl = sum(
                float(t.get("pnl_pct", 0) or 0) * float(t.get("tranche_pct", 1.0))
                for t in pos_trades
            )
            pos_pnls.append(weighted_pnl)
            if weighted_pnl > 0:
                pos_wins += 1

        n_positions = len(pos_pnls)
        s_tp = sum(1 for t in strat_trades if t["status"] == "tp_hit")
        s_sl = sum(1 for t in strat_trades if t["status"] == "sl_hit")
        s_to = sum(1 for t in strat_trades if t["status"] == "timeout")
        s_avg_pnl = sum(pos_pnls) / n_positions if n_positions else 0

        s_winners = [p for p in pos_pnls if p > 0]
        s_losers = [p for p in pos_pnls if p < 0]
        s_pf = abs(sum(s_winners) / sum(s_losers)) if s_losers and sum(s_losers) != 0 else float("inf")

        s_invested = sum(float(t.get("position_usd") or 0) for t in strat_trades)
        s_pnl_usd = sum(float(t.get("pnl_usd") or 0) for t in strat_trades)

        strategy_stats[strat_name] = {
            "positions": n_positions,
            "trade_rows": len(strat_trades),
            "tp": s_tp,
            "sl": s_sl,
            "timeout": s_to,
            "win_rate": round(pos_wins / n_positions, 3) if n_positions else 0,
            "avg_pnl": round(s_avg_pnl, 4),
            "profit_factor": round(s_pf, 2) if s_pf != float("inf") else "inf",
            "total_pnl_pct": round(sum(pos_pnls) * 100, 2),
            "invested_usd": round(s_invested, 2),
            "pnl_usd": round(s_pnl_usd, 2),
        }

    summary = {
        "total_rows": total,
        "tp": tp_count,
        "sl": sl_count,
        "timeout": timeout_count,
        "win_rate": round(win_rate, 3),
        "avg_pnl": round(avg_pnl, 4),
        "total_invested_usd": round(total_invested, 2),
        "total_pnl_usd": round(total_pnl_usd, 2),
        "roi_pct": roi_pct,
        "strategies": strategy_stats,
    }

    logger.info(
        "paper_trader SUMMARY (7d): %d rows, WR=%.1f%%, avgPnL=%.2f%% | "
        "$%.2f invested, $%+.2f PnL (ROI %.1f%%) | TP=%d SL=%d TO=%d",
        total, win_rate * 100, avg_pnl * 100,
        total_invested, total_pnl_usd, roi_pct,
        tp_count, sl_count, timeout_count,
    )
    for name, s in strategy_stats.items():
        logger.info(
            "  %s: %d pos, WR=%.1f%%, avgPnL=%.2f%%, PF=%s | $%.2f→$%+.2f",
            name, s["positions"], s["win_rate"] * 100, s["avg_pnl"] * 100,
            s["profit_factor"], s["invested_usd"], s["pnl_usd"],
        )
    return summary


def _days_ago_iso(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
