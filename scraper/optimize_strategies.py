"""
Strategy Optimizer — Optuna-powered TP/SL/horizon optimization.

Uses historical OHLCV data from token_snapshots to find optimal TP/SL/horizon
combinations, optionally segmented by token features (mcap, kol_freshness, momentum).

The optimizer simulates trades using max_price/min_price at various horizons
to determine if TP or SL would have hit first, then uses Optuna to find
the combination that maximizes risk-adjusted returns.

Usage:
    python optimize_strategies.py                          # Global optimization
    python optimize_strategies.py --segment mcap           # Segment by market cap
    python optimize_strategies.py --segment freshness      # Segment by KOL freshness
    python optimize_strategies.py --n-trials 500           # More Optuna trials
    python optimize_strategies.py --grid                   # Exhaustive grid search (no Optuna)
"""

import os
import json
import logging
import argparse
import math
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None

from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Slippage (same as paper_trader.py) ---
BUY_SLIPPAGE_BPS = 150   # 1.5%
SELL_SLIPPAGE_BPS = 300   # 3.0%

# --- Supabase ---
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", os.environ.get("SUPABASE_KEY", ""))

# Horizons we can simulate (mapped to column suffixes)
HORIZONS = [1, 6, 12, 24]

# Columns to fetch
FETCH_COLUMNS = ",".join([
    "id", "symbol", "token_address", "price_at_snapshot", "market_cap",
    "kol_freshness", "momentum_mult", "score_at_snapshot", "mention_velocity",
    "volume_velocity", "token_age_hours", "snapshot_at",
    "max_price_1h", "min_price_1h", "price_after_1h",
    "max_price_6h", "min_price_6h", "price_after_6h",
    "max_price_12h", "min_price_12h", "price_after_12h",
    "max_price_24h", "min_price_24h", "price_after_24h",
    "max_dd_before_tp_pct_12h", "max_dd_before_tp_pct_24h",
])


def _build_kol_whitelist(sb, wr_threshold: float = 0.55,
                         return_threshold: float = 1.5,
                         min_calls: int = 5) -> set[str]:
    """
    Build KOL whitelist from kol_call_outcomes — same logic as
    safe_scraper._rt_load_kol_whitelist().

    Returns set of approved kol_group names.
    """
    logger.info(f"Building KOL whitelist (WR>={wr_threshold}, ret>={return_threshold}x, min={min_calls})...")

    all_outcomes = []
    page_size = 1000
    offset = 0
    while True:
        resp = (
            sb.table("kol_call_outcomes")
            .select("kol_group,max_return")
            .not_.is_("max_return", "null")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        rows = resp.data
        if not rows:
            break
        all_outcomes.extend(rows)
        offset += page_size
        if len(rows) < page_size:
            break

    # Group by kol_group and compute WR
    kol_stats = defaultdict(lambda: {"total": 0, "hits": 0})
    for row in all_outcomes:
        g = row["kol_group"]
        kol_stats[g]["total"] += 1
        if float(row["max_return"]) >= return_threshold:
            kol_stats[g]["hits"] += 1

    approved = set()
    for kol, stats in kol_stats.items():
        if stats["total"] >= min_calls:
            wr = stats["hits"] / stats["total"]
            if wr >= wr_threshold:
                approved.add(kol)

    logger.info(f"KOL whitelist: {len(approved)}/{len(kol_stats)} approved "
                f"(WR>={wr_threshold}, {return_threshold}x, min {min_calls} calls)")
    return approved


def _fetch_kol_mention_symbols(sb, approved_kols: set[str]) -> dict[str, set[str]]:
    """
    Fetch kol_mentions for approved KOLs.

    Returns dict: symbol -> set of kol_groups that mentioned it.
    We also track (symbol, snapshot_hour) pairs via snapshot join later.
    """
    logger.info(f"Fetching kol_mentions for {len(approved_kols)} approved KOLs...")

    # Fetch mentions from approved KOLs
    all_mentions = []
    page_size = 1000
    offset = 0

    # supabase-py .in_() can handle the list of approved KOLs
    kol_list = list(approved_kols)

    while True:
        resp = (
            sb.table("kol_mentions")
            .select("symbol,kol_group,message_date")
            .in_("kol_group", kol_list)
            .order("message_date", desc=True)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        rows = resp.data
        if not rows:
            break
        all_mentions.extend(rows)
        offset += page_size
        if len(rows) < page_size:
            break

    logger.info(f"Fetched {len(all_mentions)} mentions from approved KOLs")

    # Build symbol -> set of kol_groups
    symbol_kols = defaultdict(set)
    for m in all_mentions:
        sym = m.get("symbol")
        if sym:
            symbol_kols[sym].add(m["kol_group"])

    return symbol_kols, all_mentions


def fetch_snapshots(top_n_per_cycle: int = 5,
                    use_kol_filter: bool = True,
                    wr_threshold: float = 0.55,
                    return_threshold: float = 1.5,
                    min_calls: int = 5) -> list[dict]:
    """
    Fetch snapshots that simulate paper trader selection with KOL WR filter.

    1. Build KOL whitelist from kol_call_outcomes (same as RT trading)
    2. Fetch kol_mentions from approved KOLs
    3. Filter snapshots to those with >=1 approved KOL mention within 2h before snapshot
    4. Rank by score, pick top N per cycle, 24h dedup
    """
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Step 1: Build KOL whitelist
    approved_kols = set()
    mention_index = []  # (symbol, message_date) tuples for time-window matching
    if use_kol_filter:
        approved_kols = _build_kol_whitelist(sb, wr_threshold, return_threshold, min_calls)
        if not approved_kols:
            logger.warning("No KOLs pass whitelist filter! Falling back to unfiltered.")
            use_kol_filter = False
        else:
            _, mention_index = _fetch_kol_mention_symbols(sb, approved_kols)

    # Step 2: Fetch all scored snapshots with OHLCV data
    logger.info("Fetching scored snapshots...")
    all_rows = []
    page_size = 1000
    offset = 0

    while True:
        resp = (
            sb.table("token_snapshots")
            .select(FETCH_COLUMNS)
            .not_.is_("price_at_snapshot", "null")
            .gt("price_at_snapshot", 0)
            .not_.is_("max_price_1h", "null")
            .not_.is_("min_price_1h", "null")
            .not_.is_("score_at_snapshot", "null")
            .gt("score_at_snapshot", 0)
            .order("snapshot_at", desc=True)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        rows = resp.data
        if not rows:
            break
        all_rows.extend(rows)
        offset += page_size
        if len(rows) < page_size:
            break

    logger.info(f"Fetched {len(all_rows)} scored snapshots")

    # Step 3: Filter by KOL mentions (approved KOL mentioned symbol within 2h before snapshot)
    if use_kol_filter and mention_index:
        # Build index: symbol -> list of (message_date, kol_group) sorted by date
        from datetime import timedelta
        sym_mentions = defaultdict(list)
        for m in mention_index:
            sym = m.get("symbol")
            md = m.get("message_date")
            if sym and md:
                sym_mentions[sym].append(md)
        # Sort each list
        for sym in sym_mentions:
            sym_mentions[sym].sort()

        filtered = []
        for snap in all_rows:
            sym = snap.get("symbol")
            snap_at = snap.get("snapshot_at")
            if not sym or not snap_at:
                continue
            mentions = sym_mentions.get(sym, [])
            if not mentions:
                continue
            # Check if any approved KOL mention is within 2h before snapshot
            # snap_at and message_date are ISO strings — string comparison works for same tz
            # We need: message_date >= snap_at - 2h AND message_date <= snap_at
            # Since we're comparing ISO strings, compute the 2h-before boundary
            try:
                snap_dt = datetime.fromisoformat(snap_at.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                continue
            window_start = (snap_dt - timedelta(hours=2)).isoformat()
            window_end = snap_at

            # Binary search for efficiency (mentions are sorted)
            has_mention = False
            for md in mentions:
                if md >= window_start and md <= window_end:
                    has_mention = True
                    break
                if md > window_end:
                    break
            if has_mention:
                filtered.append(snap)

        logger.info(f"KOL filter: {len(filtered)}/{len(all_rows)} snapshots have approved KOL mentions")
        all_rows = filtered

    # Step 4: Group by hourly cycle and pick top N by score with 24h dedup
    logger.info(f"Selecting top {top_n_per_cycle}/cycle with 24h dedup...")
    cycles = defaultdict(list)
    for row in all_rows:
        ts = row.get("snapshot_at", "")[:13]  # "2026-02-28T12"
        cycles[ts].append(row)

    selected = []
    recent_tokens = {}  # token_address -> cooldown_end (ISO string)

    for cycle_ts in sorted(cycles.keys()):
        cycle_snaps = cycles[cycle_ts]
        cycle_snaps.sort(key=lambda x: float(x.get("score_at_snapshot") or 0), reverse=True)

        picked = 0
        for snap in cycle_snaps:
            if picked >= top_n_per_cycle:
                break
            addr = snap.get("token_address")
            if not addr:
                continue
            # 24h dedup
            cooldown_end = recent_tokens.get(addr)
            if cooldown_end and cycle_ts < cooldown_end:
                continue
            selected.append(snap)
            # Set cooldown 24h from this cycle
            try:
                h = int(cycle_ts[11:13]) if len(cycle_ts) > 11 else 0
                # Simple approach: add 24 to hour part (works for sorting)
                from datetime import timedelta
                base_dt = datetime.fromisoformat(cycle_ts + ":00:00+00:00")
                end_dt = base_dt + timedelta(hours=24)
                recent_tokens[addr] = end_dt.isoformat()[:13]
            except (ValueError, IndexError):
                recent_tokens[addr] = cycle_ts  # fallback: no dedup
            picked += 1

    logger.info(f"Selected {len(selected)} snapshots after top-{top_n_per_cycle} + 24h dedup"
                f"{' + KOL WR filter' if use_kol_filter else ''}")
    return selected


def simulate_trade(snap: dict, tp_pct: float, sl_pct: float, horizon_h: int) -> dict:
    """
    Simulate a single trade on a snapshot.

    Args:
        snap: snapshot dict with price data
        tp_pct: take profit percentage (e.g., 0.30 = +30%)
        sl_pct: stop loss percentage (e.g., 0.15 = -15%)
        horizon_h: maximum hold time in hours (1, 6, 12, or 24)

    Returns:
        dict with keys: result ('tp', 'sl', 'timeout'), pnl_pct, exit_horizon
    """
    entry = float(snap["price_at_snapshot"])
    # Apply buy slippage
    entry_slipped = entry * (1 + BUY_SLIPPAGE_BPS / 10000)

    tp_price = entry_slipped * (1 + tp_pct)
    sl_price = entry_slipped * (1 - sl_pct)

    # Check each horizon progressively
    for h in HORIZONS:
        if h > horizon_h:
            break

        max_p = snap.get(f"max_price_{h}h")
        min_p = snap.get(f"min_price_{h}h")
        if max_p is None or min_p is None:
            continue

        max_p = float(max_p)
        min_p = float(min_p)

        tp_hit = max_p >= tp_price
        sl_hit = min_p <= sl_price

        if tp_hit and not sl_hit:
            # Clean TP hit — apply sell slippage
            exit_price = tp_price * (1 - SELL_SLIPPAGE_BPS / 10000)
            pnl = (exit_price - entry_slipped) / entry_slipped
            return {"result": "tp", "pnl_pct": pnl, "exit_horizon": h}

        elif sl_hit and not tp_hit:
            # Clean SL hit — apply sell slippage
            exit_price = sl_price * (1 - SELL_SLIPPAGE_BPS / 10000)
            pnl = (exit_price - entry_slipped) / entry_slipped
            return {"result": "sl", "pnl_pct": pnl, "exit_horizon": h}

        elif tp_hit and sl_hit:
            # Both hit in same period — use max_dd_before_tp if available
            # to determine order, otherwise assume SL hit first (conservative)
            dd_col = f"max_dd_before_tp_pct_{h}h" if h in (12, 24) else None
            dd_before_tp = snap.get(dd_col) if dd_col else None

            if dd_before_tp is not None:
                dd_before_tp = float(dd_before_tp)
                # If max drawdown before TP was less than SL, TP hit first
                if dd_before_tp > -sl_pct:
                    exit_price = tp_price * (1 - SELL_SLIPPAGE_BPS / 10000)
                    pnl = (exit_price - entry_slipped) / entry_slipped
                    return {"result": "tp", "pnl_pct": pnl, "exit_horizon": h}

            # Conservative: assume SL hit first
            exit_price = sl_price * (1 - SELL_SLIPPAGE_BPS / 10000)
            pnl = (exit_price - entry_slipped) / entry_slipped
            return {"result": "sl", "pnl_pct": pnl, "exit_horizon": h}

    # Timeout — use price_after for the horizon
    price_after = snap.get(f"price_after_{horizon_h}h")
    if price_after and float(price_after) > 0:
        exit_price = float(price_after) * (1 - SELL_SLIPPAGE_BPS / 10000)
        pnl = (exit_price - entry_slipped) / entry_slipped
    else:
        pnl = 0.0

    return {"result": "timeout", "pnl_pct": pnl, "exit_horizon": horizon_h}


def evaluate_strategy(snapshots: list[dict], tp_pct: float, sl_pct: float,
                      horizon_h: int, position_usd: float = 15.0) -> dict:
    """
    Evaluate a TP/SL/horizon combo across all snapshots.

    Returns dict with: total_pnl, avg_pnl, win_rate, n_trades, profit_factor,
                       expectancy, sharpe, tp_hits, sl_hits, timeouts
    """
    pnls = []
    tp_hits = 0
    sl_hits = 0
    timeouts = 0

    for snap in snapshots:
        trade = simulate_trade(snap, tp_pct, sl_pct, horizon_h)
        pnls.append(trade["pnl_pct"])
        if trade["result"] == "tp":
            tp_hits += 1
        elif trade["result"] == "sl":
            sl_hits += 1
        else:
            timeouts += 1

    if not pnls:
        return {"total_pnl": 0, "avg_pnl": 0, "win_rate": 0, "n_trades": 0,
                "profit_factor": 0, "expectancy": 0, "sharpe": 0, "score": -999}

    pnls = np.array(pnls)
    n = len(pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    total_pnl = float(np.sum(pnls))
    avg_pnl = float(np.mean(pnls))
    win_rate = len(wins) / n if n > 0 else 0
    gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0
    gross_loss = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.001
    profit_factor = gross_profit / gross_loss
    std = float(np.std(pnls)) if n > 1 else 1.0
    sharpe = avg_pnl / std if std > 0 else 0

    # Composite score: expectancy × sqrt(n_trades) × sharpe_sign
    # Rewards both edge AND statistical significance
    expectancy = avg_pnl * position_usd
    score = expectancy * math.sqrt(n) * (1 if sharpe > 0 else 0.5)

    return {
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "horizon_h": horizon_h,
        "total_pnl_usd": round(total_pnl * position_usd, 2),
        "avg_pnl_pct": round(avg_pnl * 100, 3),
        "win_rate": round(win_rate * 100, 1),
        "n_trades": n,
        "tp_hits": tp_hits,
        "sl_hits": sl_hits,
        "timeouts": timeouts,
        "profit_factor": round(profit_factor, 3),
        "sharpe": round(sharpe, 4),
        "score": round(score, 2),
    }


def segment_snapshots(snapshots: list[dict], segment_by: str) -> dict[str, list[dict]]:
    """Split snapshots into segments based on a feature."""
    segments = {}

    if segment_by == "mcap":
        segments = {"mcap_<500k": [], "mcap_500k-2M": [], "mcap_2M-10M": [], "mcap_>10M": []}
        for s in snapshots:
            mcap = float(s.get("market_cap") or 0)
            if mcap < 500_000:
                segments["mcap_<500k"].append(s)
            elif mcap < 2_000_000:
                segments["mcap_500k-2M"].append(s)
            elif mcap < 10_000_000:
                segments["mcap_2M-10M"].append(s)
            else:
                segments["mcap_>10M"].append(s)

    elif segment_by == "freshness":
        segments = {"fresh_kol": [], "stale_kol": [], "no_kol": []}
        for s in snapshots:
            kf = float(s.get("kol_freshness") or 0)
            if kf > 0.3:
                segments["fresh_kol"].append(s)
            elif kf > 0:
                segments["stale_kol"].append(s)
            else:
                segments["no_kol"].append(s)

    elif segment_by == "momentum":
        segments = {"high_momentum": [], "normal": [], "low_momentum": []}
        for s in snapshots:
            mm = float(s.get("momentum_mult") or 1.0)
            if mm >= 1.1:
                segments["high_momentum"].append(s)
            elif mm >= 0.9:
                segments["normal"].append(s)
            else:
                segments["low_momentum"].append(s)

    elif segment_by == "age":
        segments = {"<1h": [], "1-6h": [], "6-24h": [], ">24h": []}
        for s in snapshots:
            age = float(s.get("token_age_hours") or 999)
            if age < 1:
                segments["<1h"].append(s)
            elif age < 6:
                segments["1-6h"].append(s)
            elif age < 24:
                segments["6-24h"].append(s)
            else:
                segments[">24h"].append(s)

    else:
        segments = {"all": snapshots}

    return {k: v for k, v in segments.items() if len(v) >= 50}


def optuna_optimize(snapshots: list[dict], n_trials: int = 300,
                    segment_name: str = "all") -> dict:
    """Run Optuna to find optimal TP/SL/horizon."""
    if optuna is None:
        logger.error("optuna not installed. pip install optuna")
        return {}

    def objective(trial):
        tp_pct = trial.suggest_float("tp_pct", 0.08, 1.00, step=0.02)
        sl_pct = trial.suggest_float("sl_pct", 0.05, 0.50, step=0.01)
        horizon_h = trial.suggest_categorical("horizon_h", HORIZONS)

        # Sanity: TP should be > SL for positive expectancy
        if tp_pct < sl_pct * 0.5:
            return -9999

        result = evaluate_strategy(snapshots, tp_pct, sl_pct, horizon_h)

        # Prune if too few trades or negative sharpe
        if result["n_trades"] < 50:
            return -9999

        return result["score"]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=f"strategy_opt_{segment_name}",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    result = evaluate_strategy(snapshots, best["tp_pct"], best["sl_pct"], best["horizon_h"])
    result["segment"] = segment_name

    return result


def grid_search(snapshots: list[dict], segment_name: str = "all") -> list[dict]:
    """Exhaustive grid search over TP/SL/horizon combos."""
    results = []

    tp_range = np.arange(0.10, 1.05, 0.05)    # 10% to 100%, step 5%
    sl_range = np.arange(0.05, 0.55, 0.05)    # 5% to 50%, step 5%

    total = len(tp_range) * len(sl_range) * len(HORIZONS)
    logger.info(f"Grid search: {total} combinations for segment '{segment_name}' "
                f"({len(snapshots)} snapshots)")

    for tp in tp_range:
        for sl in sl_range:
            for h in HORIZONS:
                r = evaluate_strategy(snapshots, round(float(tp), 2),
                                      round(float(sl), 2), h)
                r["segment"] = segment_name
                if r["n_trades"] >= 50 and r["score"] > 0:
                    results.append(r)

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def print_results(results: list[dict], title: str, top_n: int = 15):
    """Pretty-print top results."""
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    print(f"{'TP%':>5} {'SL%':>5} {'H':>3} {'WR%':>6} {'AvgPnL%':>8} {'TotalPnL$':>10} "
          f"{'PF':>6} {'Sharpe':>7} {'N':>6} {'Score':>8} {'Segment':>15}")
    print("-" * 90)

    for r in results[:top_n]:
        print(f"{r['tp_pct']*100:>4.0f}% {r['sl_pct']*100:>4.0f}% {r['horizon_h']:>3}h "
              f"{r['win_rate']:>5.1f}% {r['avg_pnl_pct']:>7.3f}% ${r['total_pnl_usd']:>9.2f} "
              f"{r['profit_factor']:>5.2f} {r['sharpe']:>7.4f} {r['n_trades']:>6} "
              f"{r['score']:>7.1f} {r.get('segment','all'):>15}")


def save_results(results: list[dict], filename: str):
    """Save results to JSON."""
    with open(filename, "w") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_results": len(results),
            "top_strategies": results[:20],
        }, f, indent=2)
    logger.info(f"Results saved to {filename}")


# ---------------------------------------------------------------------------
# v82 P2-1: Per-KOL Strategy Backtester
# ---------------------------------------------------------------------------

def backtest_per_kol(days: int = 14) -> dict:
    """
    v82: Per-KOL per-strategy backtester using closed paper_trades.
    Returns {kol_group: {strategy: {n, wins, wr, pnl, avg_pnl, best_strategy}}}
    and saves to scoring_config.kol_strategy_stats for the adaptive system.
    """
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    from datetime import timedelta

    deprecated = {"MOONBAG", "WIDE_RUNNER", "SCALE_OUT", "TP100_SL30"}
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    # Fetch all closed RT trades
    all_trades = []
    page_size = 1000
    offset = 0
    while True:
        resp = sb.table("paper_trades").select(
            "kol_group, kol_tier, strategy, pnl_usd, pnl_pct, status, symbol, position_usd"
        ).neq("status", "open").filter(
            "kol_group", "not.is", "null"
        ).gte("exit_at", cutoff).range(offset, offset + page_size - 1).execute()
        rows = resp.data or []
        if not rows:
            break
        all_trades.extend(rows)
        offset += page_size
        if len(rows) < page_size:
            break

    # Filter deprecated
    all_trades = [t for t in all_trades if t.get("strategy") not in deprecated]
    if not all_trades:
        logger.info("Per-KOL backtest: no trades found")
        return {}

    logger.info(f"Per-KOL backtest: {len(all_trades)} trades from {days}d")

    # Aggregate per KOL × strategy
    kol_data = defaultdict(lambda: defaultdict(lambda: {
        "n": 0, "wins": 0, "pnl": 0.0, "invested": 0.0, "pnls": []
    }))
    for t in all_trades:
        kol = t["kol_group"]
        strat = t.get("strategy", "?")
        pnl = float(t.get("pnl_usd") or 0)
        pos = float(t.get("position_usd") or 0)
        kol_data[kol][strat]["n"] += 1
        kol_data[kol][strat]["pnl"] += pnl
        kol_data[kol][strat]["invested"] += pos
        kol_data[kol][strat]["pnls"].append(pnl)
        if pnl > 0:
            kol_data[kol][strat]["wins"] += 1

    # Build report
    report = {}
    for kol in sorted(kol_data, key=lambda k: sum(s["pnl"] for s in kol_data[k].values()), reverse=True):
        strats = kol_data[kol]
        kol_report = {"strategies": {}, "total_trades": 0, "total_pnl": 0.0}
        best_strat = None
        best_pnl = -999

        for strat, stats in strats.items():
            wr = stats["wins"] / stats["n"] if stats["n"] > 0 else 0
            roi = stats["pnl"] / stats["invested"] * 100 if stats["invested"] > 0 else 0
            avg_pnl = stats["pnl"] / stats["n"] if stats["n"] > 0 else 0

            kol_report["strategies"][strat] = {
                "n": stats["n"],
                "wins": stats["wins"],
                "wr": round(wr, 4),
                "pnl": round(stats["pnl"], 2),
                "roi": round(roi, 1),
                "avg_pnl": round(avg_pnl, 2),
            }
            kol_report["total_trades"] += stats["n"]
            kol_report["total_pnl"] += stats["pnl"]

            if stats["pnl"] > best_pnl:
                best_pnl = stats["pnl"]
                best_strat = strat

        kol_report["best_strategy"] = best_strat
        kol_report["total_pnl"] = round(kol_report["total_pnl"], 2)
        report[kol] = kol_report

    # Print summary
    print(f"\n{'='*100}")
    print(f"  PER-KOL STRATEGY BACKTEST ({days}d, {len(all_trades)} trades, {len(report)} KOLs)")
    print(f"{'='*100}")
    print(f"{'KOL':<25} {'Trades':>6} {'PnL':>10} {'Best Strategy':<15} {'N':>4} {'WR':>6} {'ROI':>7}")
    print("-" * 100)

    for kol, kr in report.items():
        best = kr["best_strategy"]
        bs = kr["strategies"].get(best, {})
        print(f"{kol:<25} {kr['total_trades']:>6} ${kr['total_pnl']:>+8.1f} "
              f"{best:<15} {bs.get('n', 0):>4} {bs.get('wr', 0)*100:>5.0f}% "
              f"{bs.get('roi', 0):>+6.1f}%")

        # Print other strategies for this KOL
        for strat, ss in kr["strategies"].items():
            if strat != best:
                print(f"{'':>25} {'':>6} {'':>10} "
                      f"{strat:<15} {ss['n']:>4} {ss['wr']*100:>5.0f}% "
                      f"{ss['roi']:>+6.1f}%")

    # Save to scoring_config for the adaptive system
    try:
        # Build compact version for DB (only KOLs with >= 3 trades)
        db_stats = {}
        for kol, kr in report.items():
            if kr["total_trades"] >= 3:
                db_stats[kol] = {
                    "best": kr["best_strategy"],
                    "pnl": kr["total_pnl"],
                    "strats": {s: {"n": d["n"], "wr": d["wr"], "pnl": d["pnl"]}
                               for s, d in kr["strategies"].items()},
                }
        sb.table("scoring_config").update({
            "kol_strategy_stats": json.dumps(db_stats),
        }).eq("id", 1).execute()
        logger.info(f"Per-KOL stats saved to scoring_config.kol_strategy_stats ({len(db_stats)} KOLs)")
    except Exception as e:
        logger.warning(f"Per-KOL stats DB save failed: {e}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Strategy Optimizer")
    parser.add_argument("--segment", choices=["mcap", "freshness", "momentum", "age"],
                        default=None, help="Segment snapshots by feature")
    parser.add_argument("--n-trials", type=int, default=300,
                        help="Number of Optuna trials per segment")
    parser.add_argument("--grid", action="store_true",
                        help="Use exhaustive grid search instead of Optuna")
    parser.add_argument("--no-kol-filter", action="store_true",
                        help="Disable KOL WR whitelist filter (optimize on ALL snapshots)")
    parser.add_argument("--wr-threshold", type=float, default=0.55,
                        help="KOL win rate threshold (default: 0.55)")
    parser.add_argument("--return-threshold", type=float, default=1.5,
                        help="KOL return threshold as multiplier (default: 1.5 = +50%%)")
    parser.add_argument("--min-calls", type=int, default=5,
                        help="Minimum KOL calls to qualify (default: 5)")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Top N tokens per cycle (default: 5)")
    parser.add_argument("--output", default="strategy_optimization.json",
                        help="Output filename")
    parser.add_argument("--per-kol", action="store_true",
                        help="Run per-KOL strategy backtest using paper_trades data")
    parser.add_argument("--per-kol-days", type=int, default=14,
                        help="Lookback days for per-KOL backtest (default: 14)")
    args = parser.parse_args()

    if args.per_kol:
        backtest_per_kol(days=args.per_kol_days)
        return

    snapshots = fetch_snapshots(
        top_n_per_cycle=args.top_n,
        use_kol_filter=not args.no_kol_filter,
        wr_threshold=args.wr_threshold,
        return_threshold=args.return_threshold,
        min_calls=args.min_calls,
    )
    if not snapshots:
        logger.error("No snapshots found")
        return

    all_results = []

    if args.segment:
        segments = segment_snapshots(snapshots, args.segment)
        logger.info(f"Segments: {', '.join(f'{k}={len(v)}' for k, v in segments.items())}")
    else:
        segments = {"all": snapshots}

    for seg_name, seg_snaps in segments.items():
        logger.info(f"\n--- Optimizing segment: {seg_name} ({len(seg_snaps)} snapshots) ---")

        if args.grid:
            results = grid_search(seg_snaps, seg_name)
            all_results.extend(results[:10])
            print_results(results, f"Grid Search: {seg_name}")
        else:
            if optuna is None:
                logger.error("optuna not installed — falling back to grid search")
                results = grid_search(seg_snaps, seg_name)
                all_results.extend(results[:10])
                print_results(results, f"Grid Search: {seg_name}")
            else:
                # Run Optuna for best single combo
                best = optuna_optimize(seg_snaps, args.n_trials, seg_name)
                if best:
                    all_results.append(best)
                    print_results([best], f"Optuna Best: {seg_name}", top_n=1)

                # Also run grid for top 10 overview
                results = grid_search(seg_snaps, seg_name)
                all_results.extend(results[:10])
                print_results(results, f"Grid Top 10: {seg_name}")

    # Deduplicate and sort
    seen = set()
    unique = []
    for r in all_results:
        key = (r["tp_pct"], r["sl_pct"], r["horizon_h"], r.get("segment", "all"))
        if key not in seen:
            seen.add(key)
            unique.append(r)
    unique.sort(key=lambda x: x["score"], reverse=True)

    print_results(unique, "OVERALL TOP STRATEGIES", top_n=20)
    save_results(unique, args.output)


if __name__ == "__main__":
    main()
