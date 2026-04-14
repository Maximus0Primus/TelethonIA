"""
Exhaustive Strategy Optimizer — Analyzes ALL shadow trades + simulates custom TP/SL combos.

Fetches closed shadow trades from paper_trades + OHLCV data from token_snapshots
to find optimal TP/SL/timeout combinations with slippage-adjusted returns.
"""

import os
import sys
import math
import warnings
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv

# Load env from scraper/.env
SCRAPER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scraper")
load_dotenv(os.path.join(SCRAPER_DIR, ".env"))

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")

# --- Config ---
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

# Slippage (same as paper_trader.py v94)
BUY_SLIP_PCT = 1.5   # 100bps slip + 50bps fee
SELL_SLIP_PCT = 2.5   # 200bps slip + 50bps fee

sb = create_client(SUPABASE_URL, SUPABASE_KEY)


# ═══════════════════════════════════════════════════════════════════════
# PART 1: Fetch ALL closed shadow trades (RT source)
# ═══════════════════════════════════════════════════════════════════════

def fetch_all_shadow_trades():
    """Fetch all closed shadow trades with pagination."""
    print("=" * 90)
    print("PART 1: FETCHING ALL CLOSED SHADOW TRADES (RT source)")
    print("=" * 90)

    columns = (
        "strategy, token_address, kol_group, pnl_pct, entry_price, exit_price, "
        "tp_price, sl_price, entry_score, high_price_seen, exit_minutes, "
        "horizon_minutes, status, created_at, kol_tier, entry_mcap, "
        "momentum_mult, unique_kols, source, is_shadow, symbol"
    )

    all_trades = []
    page_size = 1000
    offset = 0
    while True:
        resp = (
            sb.table("paper_trades")
            .select(columns)
            .eq("is_shadow", True)
            .eq("source", "rt")
            .in_("status", ["tp_hit", "sl_hit", "timeout"])
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = resp.data or []
        all_trades.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
        print(f"  ... fetched {len(all_trades)} trades so far")

    df = pd.DataFrame(all_trades)
    print(f"\nTotal RT shadow trades fetched: {len(df)}")
    if df.empty:
        print("No trades found! Trying without source filter...")
        # Fallback: fetch ALL shadow trades regardless of source
        offset = 0
        all_trades = []
        while True:
            resp = (
                sb.table("paper_trades")
                .select(columns)
                .eq("is_shadow", True)
                .in_("status", ["tp_hit", "sl_hit", "timeout"])
                .range(offset, offset + page_size - 1)
                .execute()
            )
            batch = resp.data or []
            all_trades.extend(batch)
            if len(batch) < page_size:
                break
            offset += page_size
        df = pd.DataFrame(all_trades)
        print(f"Total shadow trades (all sources): {len(df)}")

    if df.empty:
        print("FATAL: No shadow trades found at all.")
        sys.exit(1)

    # Type conversions
    for col in ["pnl_pct", "entry_price", "exit_price", "tp_price", "sl_price",
                "entry_score", "high_price_seen", "exit_minutes", "horizon_minutes",
                "entry_mcap", "momentum_mult", "unique_kols"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    print(f"Date range: {df['created_at'].min()} to {df['created_at'].max()}")
    print(f"Unique strategies: {df['strategy'].nunique()}")
    print(f"Unique tokens: {df['token_address'].nunique()}")
    print(f"Status distribution:\n{df['status'].value_counts().to_string()}")
    print(f"Source distribution:\n{df['source'].value_counts().to_string()}" if 'source' in df.columns else "")

    return df


# ═══════════════════════════════════════════════════════════════════════
# PART 2: Per-strategy performance (strategy-neutral dedup)
# ═══════════════════════════════════════════════════════════════════════

def per_strategy_performance(df):
    """Per-strategy stats with strategy-neutral dedup (first call per token)."""
    print("\n" + "=" * 90)
    print("PART 2: PER-STRATEGY PERFORMANCE (strategy-neutral dedup)")
    print("=" * 90)

    # Dedup: per token_address, keep earliest created_at trade (across all strategies)
    df_sorted = df.sort_values("created_at")
    first_per_token = df_sorted.groupby("token_address")["created_at"].min().reset_index()
    first_per_token.columns = ["token_address", "first_call_time"]

    # For each strategy, keep only trades where this token's first appearance
    # matches the trade's created_at (within 1 minute tolerance)
    df = df.merge(first_per_token, on="token_address", how="left")
    df["is_first_call"] = (df["created_at"] - df["first_call_time"]).abs() < pd.Timedelta(minutes=5)

    results = []
    for strategy, group in df.groupby("strategy"):
        # Only keep first-call trades for dedup
        deduped = group[group["is_first_call"]]
        if len(deduped) < 3:
            continue

        n = len(deduped)
        wins = (deduped["status"] == "tp_hit").sum()
        losses = (deduped["status"] == "sl_hit").sum()
        timeouts = (deduped["status"] == "timeout").sum()
        wr = wins / n * 100 if n > 0 else 0
        avg_pnl = deduped["pnl_pct"].mean()
        med_pnl = deduped["pnl_pct"].median()
        total_gain = deduped.loc[deduped["pnl_pct"] > 0, "pnl_pct"].sum()
        total_loss = abs(deduped.loc[deduped["pnl_pct"] < 0, "pnl_pct"].sum())
        pf = total_gain / total_loss if total_loss > 0 else float("inf")
        avg_exit_min = deduped["exit_minutes"].mean()

        results.append({
            "strategy": strategy,
            "N": n,
            "wins": wins,
            "losses": losses,
            "timeouts": timeouts,
            "WR%": round(wr, 1),
            "avg_pnl%": round(avg_pnl, 2),
            "med_pnl%": round(med_pnl, 2),
            "profit_factor": round(pf, 2),
            "avg_exit_min": round(avg_exit_min, 0),
        })

    result_df = pd.DataFrame(results).sort_values("avg_pnl%", ascending=False)
    print(f"\n{'='*90}")
    print(f"ALL STRATEGIES ({len(result_df)} strategies with 3+ trades)")
    print(f"{'='*90}")
    print(result_df.to_string(index=False))

    # Also show top 20 by EV * sqrt(N)
    result_df["score"] = result_df["avg_pnl%"] * np.sqrt(result_df["N"])
    top20 = result_df.sort_values("score", ascending=False).head(20)
    print(f"\n{'='*90}")
    print("TOP 20 BY EV * sqrt(N)")
    print(f"{'='*90}")
    print(top20.to_string(index=False))

    return result_df


# ═══════════════════════════════════════════════════════════════════════
# PART 3: Fetch token_snapshots OHLCV data + simulate custom combos
# ═══════════════════════════════════════════════════════════════════════

def fetch_token_snapshots(token_addresses):
    """Fetch token_snapshots with max/min price data for simulation."""
    print("\n" + "=" * 90)
    print("PART 3: FETCHING TOKEN SNAPSHOTS FOR SIMULATION")
    print("=" * 90)

    columns = (
        "token_address, price_at_snapshot, score_at_snapshot, market_cap, snapshot_at, "
        "max_price_1h, min_price_1h, "
        "max_price_6h, min_price_6h, "
        "max_price_12h, min_price_12h, "
        "max_price_24h, min_price_24h, "
        "max_price_48h, min_price_48h, "
        "unique_kols, momentum_mult, kol_freshness"
    )

    all_rows = []
    # Fetch in batches of token addresses
    addr_list = list(token_addresses)
    batch_size = 50  # Supabase IN filter limit
    for i in range(0, len(addr_list), batch_size):
        batch_addrs = addr_list[i:i+batch_size]
        offset = 0
        page_size = 1000
        while True:
            resp = (
                sb.table("token_snapshots")
                .select(columns)
                .in_("token_address", batch_addrs)
                .not_.is_("price_at_snapshot", "null")
                .range(offset, offset + page_size - 1)
                .execute()
            )
            rows = resp.data or []
            all_rows.extend(rows)
            if len(rows) < page_size:
                break
            offset += page_size
        if (i // batch_size) % 10 == 0:
            print(f"  ... fetched snapshots for {min(i + batch_size, len(addr_list))}/{len(addr_list)} tokens ({len(all_rows)} rows)")

    df = pd.DataFrame(all_rows)
    print(f"Total token snapshots fetched: {len(df)}")

    if df.empty:
        return df

    # Type conversions
    numeric_cols = [c for c in df.columns if c not in ("token_address", "snapshot_at")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["snapshot_at"] = pd.to_datetime(df["snapshot_at"], errors="coerce")

    # Dedup: per token, keep the one with the most complete max_price data
    # (prefer earliest snapshot with data)
    df["has_data"] = df[["max_price_1h", "max_price_6h", "max_price_12h", "max_price_24h"]].notna().sum(axis=1)
    df = df.sort_values(["token_address", "has_data", "snapshot_at"], ascending=[True, False, True])
    df = df.drop_duplicates(subset="token_address", keep="first")
    df = df.drop(columns=["has_data"])

    print(f"Unique tokens with snapshot data: {len(df)}")
    print(f"Tokens with max_price_1h: {df['max_price_1h'].notna().sum()}")
    print(f"Tokens with max_price_6h: {df['max_price_6h'].notna().sum()}")
    print(f"Tokens with max_price_12h: {df['max_price_12h'].notna().sum()}")
    print(f"Tokens with max_price_24h: {df['max_price_24h'].notna().sum()}")

    return df


def simulate_tp_sl_combos(snap_df):
    """Simulate TP/SL/timeout combos using token_snapshots OHLCV data."""
    print("\n" + "=" * 90)
    print("PART 3b: SIMULATING CUSTOM TP/SL/TIMEOUT COMBOS")
    print("=" * 90)

    if snap_df.empty:
        print("No snapshot data available for simulation.")
        return pd.DataFrame()

    # TP/SL grid
    tp_pcts = [10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100]
    sl_pcts = [10, 15, 20, 25, 30, 40, 50, 60, 70]

    # Timeout mapping to horizon columns
    # We approximate timeouts by using the closest available horizon
    # Available: 1h, 6h, 12h, 24h, 48h
    timeout_map = {
        30: "1h",    # 30min → use 1h data (conservative)
        60: "1h",    # 1h → 1h
        120: "6h",   # 2h → use 6h data (no 2h column, conservative)
        240: "6h",   # 4h → 6h
        360: "6h",   # 6h → 6h
        720: "12h",  # 12h → 12h
        1440: "24h", # 24h → 24h
    }

    # For simulation: we use max_price_{h} and min_price_{h} to determine outcomes
    # If max_price >= entry * (1 + tp_pct/100) → TP hit
    # If min_price <= entry * (1 - sl_pct/100) → SL hit
    # If both could hit within the horizon, we can't know order → use heuristic:
    #   - If TP hit AND SL hit in same horizon: assume SL hit first (conservative)
    #   - For shorter horizons nested in longer ones, check shorter first

    results = []
    total_combos = len(tp_pcts) * len(sl_pcts) * len(timeout_map)
    combo_idx = 0

    for tp_pct, sl_pct, (timeout_min, horizon_col) in product(tp_pcts, sl_pcts, timeout_map.items()):
        combo_idx += 1
        if combo_idx % 200 == 0:
            print(f"  Simulating combo {combo_idx}/{total_combos}...")

        max_col = f"max_price_{horizon_col}"
        min_col = f"min_price_{horizon_col}"

        if max_col not in snap_df.columns or min_col not in snap_df.columns:
            continue

        # Filter to rows with both max and min data
        valid = snap_df[snap_df[max_col].notna() & snap_df[min_col].notna() & snap_df["price_at_snapshot"].notna()].copy()
        if len(valid) < 10:
            continue

        entry = valid["price_at_snapshot"].values
        max_p = valid[max_col].values
        min_p = valid[min_col].values

        # Slippage-adjusted entry
        effective_entry = entry * (1 + BUY_SLIP_PCT / 100)

        # TP and SL thresholds
        tp_threshold = effective_entry * (1 + tp_pct / 100)
        sl_threshold = effective_entry * (1 - sl_pct / 100)

        tp_hit = max_p >= tp_threshold
        sl_hit = min_p <= sl_threshold

        # Determine outcome per trade
        # Conservative: if both TP and SL could hit, check shorter horizons first
        # For simplicity in bulk sim: if both hit, assume SL wins (conservative)
        pnl = np.zeros(len(valid))

        # Case 1: TP hit only
        tp_only = tp_hit & ~sl_hit
        pnl[tp_only] = tp_pct / 100 - SELL_SLIP_PCT / 100  # net after sell slippage

        # Case 2: SL hit only
        sl_only = sl_hit & ~tp_hit
        pnl[sl_only] = -(sl_pct / 100) - SELL_SLIP_PCT / 100  # net after sell slippage

        # Case 3: Both hit → assume SL first (conservative)
        both = tp_hit & sl_hit
        pnl[both] = -(sl_pct / 100) - SELL_SLIP_PCT / 100

        # Case 4: Neither → timeout at period-end price
        neither = ~tp_hit & ~sl_hit
        # Use price at end of period (approximated by checking if price went up or down)
        # Best we can do: use mid of max and min as timeout exit estimate
        # More conservative: use the final price column if available
        price_after_col = f"price_after_{horizon_col}"
        if price_after_col in snap_df.columns:
            end_price = valid[price_after_col].values
            timeout_pnl = (end_price / effective_entry) - 1 - SELL_SLIP_PCT / 100
        else:
            # Fallback: average of max and min, discounted
            end_price = (max_p + min_p) / 2
            timeout_pnl = (end_price / effective_entry) - 1 - SELL_SLIP_PCT / 100
        pnl[neither] = timeout_pnl[neither]
        # Cap timeout PnL by SL (if SL would have saved us)
        pnl[neither] = np.maximum(pnl[neither], -(sl_pct / 100) - SELL_SLIP_PCT / 100)

        # Also account for buy slippage already baked into effective_entry
        n_trades = len(valid)
        n_wins = int((pnl > 0).sum())
        wr = n_wins / n_trades * 100
        avg_pnl = pnl.mean() * 100  # as percentage
        med_pnl = np.median(pnl) * 100
        total_gain = pnl[pnl > 0].sum()
        total_loss = abs(pnl[pnl < 0].sum())
        pf = total_gain / total_loss if total_loss > 0 else float("inf")
        ev = avg_pnl  # EV per trade in %
        std_pnl = pnl.std() * 100
        sharpe = (avg_pnl / std_pnl) if std_pnl > 0 else 0

        results.append({
            "TP%": tp_pct,
            "SL%": sl_pct,
            "timeout_min": timeout_min,
            "horizon": horizon_col,
            "N": n_trades,
            "WR%": round(wr, 1),
            "avg_pnl%": round(avg_pnl, 2),
            "med_pnl%": round(med_pnl, 2),
            "EV%": round(ev, 2),
            "profit_factor": round(pf, 3),
            "sharpe": round(sharpe, 3),
            "std%": round(std_pnl, 2),
        })

    result_df = pd.DataFrame(results)
    if result_df.empty:
        print("No simulation results!")
        return result_df

    # Score: EV * sqrt(N)
    result_df["score"] = result_df["EV%"] * np.sqrt(result_df["N"])

    # Show top 30
    top = result_df.sort_values("score", ascending=False).head(30)
    print(f"\n{'='*90}")
    print(f"TOP 30 SIMULATED COMBOS BY EV * sqrt(N)  (slippage: buy {BUY_SLIP_PCT}% / sell {SELL_SLIP_PCT}%)")
    print(f"{'='*90}")
    print(top.to_string(index=False))

    # Also show top by pure EV (min 20 trades)
    ev_top = result_df[result_df["N"] >= 20].sort_values("EV%", ascending=False).head(20)
    print(f"\n{'='*90}")
    print("TOP 20 BY PURE EV% (min 20 trades)")
    print(f"{'='*90}")
    print(ev_top.to_string(index=False))

    # Show top by Sharpe (min 20 trades)
    sharpe_top = result_df[result_df["N"] >= 20].sort_values("sharpe", ascending=False).head(20)
    print(f"\n{'='*90}")
    print("TOP 20 BY SHARPE RATIO (min 20 trades)")
    print(f"{'='*90}")
    print(sharpe_top.to_string(index=False))

    return result_df


# ═══════════════════════════════════════════════════════════════════════
# PART 4: Score threshold analysis
# ═══════════════════════════════════════════════════════════════════════

def score_threshold_analysis(snap_df, sim_df):
    """Test different entry_score thresholds on top combos."""
    print("\n" + "=" * 90)
    print("PART 4: SCORE THRESHOLD ANALYSIS (top 10 combos)")
    print("=" * 90)

    if sim_df.empty or snap_df.empty:
        print("No data for score threshold analysis.")
        return

    # Get top 10 combos by score
    top10 = sim_df.sort_values("score", ascending=False).head(10)
    score_thresholds = [20, 30, 40, 50, 60, 70]

    for _, combo in top10.iterrows():
        tp_pct = combo["TP%"]
        sl_pct = combo["SL%"]
        horizon_col = combo["horizon"]
        timeout_min = combo["timeout_min"]

        print(f"\n--- TP{tp_pct}_SL{sl_pct} / {timeout_min}min (horizon={horizon_col}) ---")

        max_col = f"max_price_{horizon_col}"
        min_col = f"min_price_{horizon_col}"

        valid = snap_df[
            snap_df[max_col].notna() &
            snap_df[min_col].notna() &
            snap_df["price_at_snapshot"].notna() &
            snap_df["score_at_snapshot"].notna()
        ].copy()

        if len(valid) < 10:
            print("  Not enough data with scores.")
            continue

        rows = []
        for threshold in score_thresholds:
            subset = valid[valid["score_at_snapshot"] >= threshold]
            if len(subset) < 3:
                rows.append({"score>=": threshold, "N": 0, "WR%": 0, "EV%": 0})
                continue

            entry = subset["price_at_snapshot"].values
            max_p = subset[max_col].values
            min_p = subset[min_col].values

            effective_entry = entry * (1 + BUY_SLIP_PCT / 100)
            tp_threshold = effective_entry * (1 + tp_pct / 100)
            sl_threshold = effective_entry * (1 - sl_pct / 100)

            tp_hit = max_p >= tp_threshold
            sl_hit = min_p <= sl_threshold

            pnl = np.zeros(len(subset))
            tp_only = tp_hit & ~sl_hit
            sl_only = sl_hit & ~tp_hit
            both = tp_hit & sl_hit
            neither = ~tp_hit & ~sl_hit

            pnl[tp_only] = tp_pct / 100 - SELL_SLIP_PCT / 100
            pnl[sl_only] = -(sl_pct / 100) - SELL_SLIP_PCT / 100
            pnl[both] = -(sl_pct / 100) - SELL_SLIP_PCT / 100

            price_after_col = f"price_after_{horizon_col}"
            if price_after_col in snap_df.columns:
                end_price = subset[price_after_col].values
                timeout_pnl = (end_price / effective_entry) - 1 - SELL_SLIP_PCT / 100
            else:
                end_price = (max_p + min_p) / 2
                timeout_pnl = (end_price / effective_entry) - 1 - SELL_SLIP_PCT / 100
            pnl[neither] = timeout_pnl[neither]
            pnl[neither] = np.maximum(pnl[neither], -(sl_pct / 100) - SELL_SLIP_PCT / 100)

            n = len(subset)
            wr = (pnl > 0).sum() / n * 100
            ev = pnl.mean() * 100

            rows.append({
                "score>=": threshold,
                "N": n,
                "WR%": round(wr, 1),
                "EV%": round(ev, 2),
                "EV*sqrt(N)": round(ev * math.sqrt(n), 2),
            })

        print(pd.DataFrame(rows).to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════
# PART 5: Entry mcap analysis
# ═══════════════════════════════════════════════════════════════════════

def mcap_analysis(snap_df, sim_df):
    """Test mcap filters on top combos."""
    print("\n" + "=" * 90)
    print("PART 5: ENTRY MCAP ANALYSIS (top 10 combos)")
    print("=" * 90)

    if sim_df.empty or snap_df.empty:
        print("No data for mcap analysis.")
        return

    top10 = sim_df.sort_values("score", ascending=False).head(10)
    mcap_buckets = [
        ("< 500K", 0, 500_000),
        ("500K-2M", 500_000, 2_000_000),
        ("2M-10M", 2_000_000, 10_000_000),
        ("> 10M", 10_000_000, 1e15),
    ]

    for _, combo in top10.iterrows():
        tp_pct = combo["TP%"]
        sl_pct = combo["SL%"]
        horizon_col = combo["horizon"]
        timeout_min = combo["timeout_min"]

        print(f"\n--- TP{tp_pct}_SL{sl_pct} / {timeout_min}min (horizon={horizon_col}) ---")

        max_col = f"max_price_{horizon_col}"
        min_col = f"min_price_{horizon_col}"

        valid = snap_df[
            snap_df[max_col].notna() &
            snap_df[min_col].notna() &
            snap_df["price_at_snapshot"].notna() &
            snap_df["market_cap"].notna()
        ].copy()

        if len(valid) < 10:
            print("  Not enough data with mcap.")
            continue

        rows = []
        for label, lo, hi in mcap_buckets:
            subset = valid[(valid["market_cap"] >= lo) & (valid["market_cap"] < hi)]
            if len(subset) < 3:
                rows.append({"mcap": label, "N": 0, "WR%": 0, "EV%": 0})
                continue

            entry = subset["price_at_snapshot"].values
            max_p = subset[max_col].values
            min_p = subset[min_col].values

            effective_entry = entry * (1 + BUY_SLIP_PCT / 100)
            tp_threshold = effective_entry * (1 + tp_pct / 100)
            sl_threshold = effective_entry * (1 - sl_pct / 100)

            tp_hit = max_p >= tp_threshold
            sl_hit = min_p <= sl_threshold

            pnl = np.zeros(len(subset))
            tp_only = tp_hit & ~sl_hit
            sl_only = sl_hit & ~tp_hit
            both = tp_hit & sl_hit
            neither = ~tp_hit & ~sl_hit

            pnl[tp_only] = tp_pct / 100 - SELL_SLIP_PCT / 100
            pnl[sl_only] = -(sl_pct / 100) - SELL_SLIP_PCT / 100
            pnl[both] = -(sl_pct / 100) - SELL_SLIP_PCT / 100

            price_after_col = f"price_after_{horizon_col}"
            if price_after_col in snap_df.columns:
                end_price = subset[price_after_col].values
                timeout_pnl = (end_price / effective_entry) - 1 - SELL_SLIP_PCT / 100
            else:
                end_price = (max_p + min_p) / 2
                timeout_pnl = (end_price / effective_entry) - 1 - SELL_SLIP_PCT / 100
            pnl[neither] = timeout_pnl[neither]
            pnl[neither] = np.maximum(pnl[neither], -(sl_pct / 100) - SELL_SLIP_PCT / 100)

            n = len(subset)
            wr = (pnl > 0).sum() / n * 100
            ev = pnl.mean() * 100

            rows.append({
                "mcap": label,
                "N": n,
                "WR%": round(wr, 1),
                "EV%": round(ev, 2),
                "EV*sqrt(N)": round(ev * math.sqrt(n), 2),
            })

        print(pd.DataFrame(rows).to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════
# PART 6: Final ranking
# ═══════════════════════════════════════════════════════════════════════

def final_ranking(sim_df, strat_df):
    """Combine shadow trade actuals + simulated combos into final ranking."""
    print("\n" + "=" * 90)
    print("PART 6: FINAL RANKING — ALL COMBOS SORTED BY EV * sqrt(N)")
    print("=" * 90)

    # From actual shadow trades
    if strat_df is not None and not strat_df.empty:
        actual = strat_df[["strategy", "N", "WR%", "avg_pnl%", "med_pnl%", "profit_factor"]].copy()
        actual["source"] = "actual_shadow"
        actual["EV%"] = actual["avg_pnl%"]
        actual["score"] = actual["EV%"] * np.sqrt(actual["N"])
        actual = actual.rename(columns={"strategy": "combo"})
    else:
        actual = pd.DataFrame()

    # From simulated combos
    if sim_df is not None and not sim_df.empty:
        sim = sim_df.copy()
        sim["combo"] = sim.apply(lambda r: f"SIM_TP{int(r['TP%'])}_SL{int(r['SL%'])}_{int(r['timeout_min'])}min", axis=1)
        sim["source"] = "simulated"
        sim = sim.rename(columns={"avg_pnl%": "avg_pnl%"})
    else:
        sim = pd.DataFrame()

    # Combine
    combined_rows = []
    if not actual.empty:
        for _, r in actual.iterrows():
            combined_rows.append({
                "combo": r["combo"],
                "source": "actual",
                "N": r["N"],
                "WR%": r["WR%"],
                "EV%": r["EV%"],
                "PF": r["profit_factor"],
                "score": r["score"],
            })

    if not sim.empty:
        for _, r in sim.iterrows():
            combined_rows.append({
                "combo": r["combo"],
                "source": "sim",
                "N": r["N"],
                "WR%": r["WR%"],
                "EV%": r["EV%"],
                "PF": r["profit_factor"],
                "score": r["score"],
            })

    if not combined_rows:
        print("No data to rank.")
        return

    combined = pd.DataFrame(combined_rows).sort_values("score", ascending=False)

    # Top 20
    top20 = combined.head(20)
    print(f"\nTOP 20 OVERALL (actual shadow + simulated):")
    print(top20.to_string(index=False))

    # Also show top 20 actual only
    actual_only = combined[combined["source"] == "actual"].head(20)
    if not actual_only.empty:
        print(f"\nTOP 20 ACTUAL SHADOW TRADES:")
        print(actual_only.to_string(index=False))

    # Top 20 simulated only (min 20 trades)
    sim_only = combined[(combined["source"] == "sim") & (combined["N"] >= 20)].head(20)
    if not sim_only.empty:
        print(f"\nTOP 20 SIMULATED (min 20 trades):")
        print(sim_only.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 90)
    print("EXHAUSTIVE STRATEGY OPTIMIZER")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 90)

    # PART 1: Fetch shadow trades
    trades_df = fetch_all_shadow_trades()

    # PART 2: Per-strategy performance
    strat_df = per_strategy_performance(trades_df)

    # PART 3: Fetch snapshots + simulate combos
    token_addresses = trades_df["token_address"].dropna().unique().tolist()
    print(f"\nFetching snapshots for {len(token_addresses)} unique tokens...")
    snap_df = fetch_token_snapshots(token_addresses)

    sim_df = pd.DataFrame()
    if not snap_df.empty:
        sim_df = simulate_tp_sl_combos(snap_df)

        # PART 4: Score threshold analysis
        score_threshold_analysis(snap_df, sim_df)

        # PART 5: Mcap analysis
        mcap_analysis(snap_df, sim_df)

    # PART 6: Final ranking
    final_ranking(sim_df, strat_df)

    print("\n" + "=" * 90)
    print("OPTIMIZATION COMPLETE")
    print("=" * 90)
