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


def fetch_snapshots(top_n_per_cycle: int = 5) -> list[dict]:
    """
    Fetch snapshots that simulate paper trader selection.

    Uses a SQL query to rank tokens by score within each hourly cycle
    and return only the top N — matching what the paper trader would actually
    trade. This avoids the bias of optimizing on all 23K+ snapshots (most of
    which are untradeable shitcoins).
    """
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info(f"Fetching top-{top_n_per_cycle} snapshots per cycle (simulating paper trader)...")

    # Use RPC or direct SQL via the REST API with a view
    # Since we can't run raw SQL through supabase-py easily, we fetch all scored
    # snapshots and do the ranking in Python
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

    logger.info(f"Fetched {len(all_rows)} scored snapshots, selecting top {top_n_per_cycle}/cycle...")

    # Group by hourly cycle and pick top N by score
    from collections import defaultdict
    cycles = defaultdict(list)
    for row in all_rows:
        # Group by hour
        ts = row.get("snapshot_at", "")[:13]  # "2026-02-28T12"
        cycles[ts].append(row)

    # Select top N per cycle, with 24h token dedup
    selected = []
    recent_tokens = {}  # token_address -> last_selected_ts

    for cycle_ts in sorted(cycles.keys()):
        cycle_snaps = cycles[cycle_ts]
        # Sort by score descending
        cycle_snaps.sort(key=lambda x: float(x.get("score_at_snapshot") or 0), reverse=True)

        picked = 0
        for snap in cycle_snaps:
            if picked >= top_n_per_cycle:
                break
            addr = snap.get("token_address")
            if not addr:
                continue
            # 24h dedup check
            last_ts = recent_tokens.get(addr)
            if last_ts and cycle_ts < last_ts:
                continue
            selected.append(snap)
            recent_tokens[addr] = cycle_ts[:10] + "T" + str(int(cycle_ts[11:13]) + 24).zfill(2) if len(cycle_ts) > 11 else cycle_ts
            picked += 1

    logger.info(f"Selected {len(selected)} snapshots after top-{top_n_per_cycle} + 24h dedup")
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


def main():
    parser = argparse.ArgumentParser(description="Strategy Optimizer")
    parser.add_argument("--segment", choices=["mcap", "freshness", "momentum", "age"],
                        default=None, help="Segment snapshots by feature")
    parser.add_argument("--n-trials", type=int, default=300,
                        help="Number of Optuna trials per segment")
    parser.add_argument("--grid", action="store_true",
                        help="Use exhaustive grid search instead of Optuna")
    parser.add_argument("--output", default="strategy_optimization.json",
                        help="Output filename")
    args = parser.parse_args()

    snapshots = fetch_snapshots()
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
