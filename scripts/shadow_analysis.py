"""Shadow trade strategy analysis — comprehensive report."""

import os
import sys
from datetime import datetime

# Add scraper dir to path for potential imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))

import pandas as pd
from supabase import create_client

# Load credentials
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# ─── Fetch all closed shadow trades since Mar 11 with 2h timeout ───
print("Fetching shadow trades from Supabase...")

all_data = []
page_size = 1000
offset = 0
while True:
    resp = (
        sb.table("paper_trades")
        .select("id,strategy,status,pnl_pct,pnl_usd,entry_score,exit_minutes,source,created_at,is_shadow,horizon_minutes,position_usd")
        .eq("is_shadow", True)
        .eq("horizon_minutes", 120)
        .neq("status", "open")
        .gte("created_at", "2026-03-11T00:00:00Z")
        .order("created_at")
        .range(offset, offset + page_size - 1)
        .execute()
    )
    batch = resp.data or []
    all_data.extend(batch)
    if len(batch) < page_size:
        break
    offset += page_size

print(f"Fetched {len(all_data)} closed shadow trades (horizon=120min, since Mar 11)\n")

if not all_data:
    print("No data found. Exiting.")
    sys.exit(0)

df = pd.DataFrame(all_data)
df["pnl_pct"] = pd.to_numeric(df["pnl_pct"], errors="coerce")
df["pnl_usd"] = pd.to_numeric(df["pnl_usd"], errors="coerce")
df["entry_score"] = pd.to_numeric(df["entry_score"], errors="coerce")
df["exit_minutes"] = pd.to_numeric(df["exit_minutes"], errors="coerce")
df["created_at"] = pd.to_datetime(df["created_at"], utc=True)

VIRTUAL_POS = 15.0  # $15 per trade

# ═══════════════════════════════════════════════════════════════════
# 1. SHADOW TRADE STRATEGY PERFORMANCE
# ═══════════════════════════════════════════════════════════════════
print("=" * 90)
print("1. SHADOW TRADE STRATEGY PERFORMANCE (post-v101, horizon=120min, since Mar 11)")
print("=" * 90)


def compute_stats(group):
    n = len(group)
    tp_count = (group["status"] == "tp_hit").sum()
    sl_count = (group["status"] == "sl_hit").sum()
    timeout_count = (group["status"] == "timeout").sum()
    win_rate = tp_count / n if n else 0

    avg_pnl = group["pnl_pct"].mean()
    med_pnl = group["pnl_pct"].median()

    # Total PnL with $15 virtual position
    total_pnl = (group["pnl_pct"] * VIRTUAL_POS).sum()

    # Profit factor
    wins_sum = (group["pnl_pct"][group["pnl_pct"] > 0] * VIRTUAL_POS).sum()
    losses_sum = abs((group["pnl_pct"][group["pnl_pct"] < 0] * VIRTUAL_POS).sum())
    pf = wins_sum / losses_sum if losses_sum > 0 else float("inf")

    # Max consecutive losses
    is_loss = (group["pnl_pct"] < 0).values
    max_consec = 0
    current = 0
    for v in is_loss:
        if v:
            current += 1
            max_consec = max(max_consec, current)
        else:
            current = 0

    avg_exit_min = group["exit_minutes"].mean()

    return pd.Series({
        "N": n,
        "WinRate": f"{win_rate:.1%}",
        "AvgPnL%": f"{avg_pnl * 100:.2f}%",
        "MedPnL%": f"{med_pnl * 100:.2f}%",
        "TotalPnL$": f"${total_pnl:.2f}",
        "PF": f"{pf:.2f}" if pf != float("inf") else "inf",
        "MaxConsLoss": max_consec,
        "AvgExitMin": f"{avg_exit_min:.0f}",
        "TP": tp_count,
        "SL": sl_count,
        "Timeout": timeout_count,
        "_total_pnl": total_pnl,  # for sorting
    })


stats = df.groupby("strategy").apply(compute_stats, include_groups=False).reset_index()
stats = stats.sort_values("_total_pnl", ascending=False)

display_cols = ["strategy", "N", "WinRate", "AvgPnL%", "MedPnL%", "TotalPnL$", "PF", "MaxConsLoss", "AvgExitMin", "TP", "SL", "Timeout"]
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 200)
print(stats[display_cols].to_string(index=False))
print()

# ═══════════════════════════════════════════════════════════════════
# 2. RT VS BATCH COMPARISON (top 10 strategies by total PnL)
# ═══════════════════════════════════════════════════════════════════
print("=" * 90)
print("2. RT VS BATCH COMPARISON (top 10 strategies)")
print("=" * 90)

top10 = stats.head(10)["strategy"].tolist()
df_top10 = df[df["strategy"].isin(top10)]

for strat in top10:
    sub = df_top10[df_top10["strategy"] == strat]
    print(f"\n--- {strat} ---")
    for src in ["rt", "batch"]:
        g = sub[sub["source"] == src]
        if g.empty:
            print(f"  {src:>5}: no trades")
            continue
        n = len(g)
        tp = (g["status"] == "tp_hit").sum()
        wr = tp / n if n else 0
        avg_pnl = g["pnl_pct"].mean()
        total_pnl = (g["pnl_pct"] * VIRTUAL_POS).sum()
        avg_exit = g["exit_minutes"].mean()
        print(f"  {src:>5}: N={n:>4}  WR={wr:.1%}  AvgPnL={avg_pnl*100:+.2f}%  TotalPnL=${total_pnl:+.2f}  AvgExit={avg_exit:.0f}min")

    # Also show any other sources
    other_sources = sub[~sub["source"].isin(["rt", "batch"])]["source"].unique()
    for src in other_sources:
        g = sub[sub["source"] == src]
        n = len(g)
        tp = (g["status"] == "tp_hit").sum()
        wr = tp / n if n else 0
        total_pnl = (g["pnl_pct"] * VIRTUAL_POS).sum()
        print(f"  {src:>5}: N={n:>4}  WR={wr:.1%}  TotalPnL=${total_pnl:+.2f}")

print()

# ═══════════════════════════════════════════════════════════════════
# 3. BEST STRATEGIES (N>=30, WR>35%, best PnL)
# ═══════════════════════════════════════════════════════════════════
print("=" * 90)
print("3. BEST STRATEGIES (N >= 30, WR > 35%)")
print("=" * 90)

stats_num = df.groupby("strategy").apply(
    lambda g: pd.Series({
        "N": len(g),
        "WinRate": (g["status"] == "tp_hit").sum() / len(g) if len(g) else 0,
        "TotalPnL": (g["pnl_pct"] * VIRTUAL_POS).sum(),
        "AvgPnL": g["pnl_pct"].mean(),
    }),
    include_groups=False
).reset_index()

best = stats_num[(stats_num["N"] >= 30) & (stats_num["WinRate"] > 0.35)]
best = best.sort_values("TotalPnL", ascending=False)

if best.empty:
    # Relax criteria
    print("No strategies meet N>=30 + WR>35%. Relaxing to N>=15 + WR>30%:")
    best = stats_num[(stats_num["N"] >= 15) & (stats_num["WinRate"] > 0.30)]
    best = best.sort_values("TotalPnL", ascending=False)

if best.empty:
    print("No strategies meet relaxed criteria either. Showing top 10 by N:")
    best = stats_num.sort_values("N", ascending=False).head(10)

for _, row in best.iterrows():
    print(f"  {row['strategy']:>20}: N={int(row['N']):>4}  WR={row['WinRate']:.1%}  TotalPnL=${row['TotalPnL']:+.2f}  AvgPnL={row['AvgPnL']*100:+.2f}%")

print()

# ═══════════════════════════════════════════════════════════════════
# 4. STRATEGY CORRELATION WITH ENTRY SCORE
# ═══════════════════════════════════════════════════════════════════
print("=" * 90)
print("4. ENTRY SCORE: WINNERS VS LOSERS (per strategy)")
print("=" * 90)

# Take top 15 strategies by trade count
top15_strats = stats_num.sort_values("N", ascending=False).head(15)["strategy"].tolist()

for strat in top15_strats:
    sub = df[df["strategy"] == strat]
    winners = sub[sub["pnl_pct"] > 0]
    losers = sub[sub["pnl_pct"] <= 0]
    w_score = winners["entry_score"].mean() if not winners.empty else 0
    l_score = losers["entry_score"].mean() if not losers.empty else 0
    delta = w_score - l_score
    print(f"  {strat:>20}: Win AvgScore={w_score:.1f}  Loss AvgScore={l_score:.1f}  Delta={delta:+.1f}  (N_win={len(winners)}, N_loss={len(losers)})")

print()

# ═══════════════════════════════════════════════════════════════════
# 5. TIME-OF-DAY ANALYSIS (top 5 strategies, UTC hours)
# ═══════════════════════════════════════════════════════════════════
print("=" * 90)
print("5. TIME-OF-DAY WIN RATE (top 5 strategies by total PnL, UTC hours)")
print("=" * 90)

top5_strats = stats.head(5)["strategy"].tolist()
df["hour"] = df["created_at"].dt.hour

for strat in top5_strats:
    sub = df[df["strategy"] == strat]
    print(f"\n--- {strat} ---")
    hourly = sub.groupby("hour").apply(
        lambda g: pd.Series({
            "N": len(g),
            "WR": f"{(g['status'] == 'tp_hit').sum() / len(g):.0%}" if len(g) else "n/a",
            "AvgPnL": f"{g['pnl_pct'].mean()*100:+.1f}%",
            "PnL$": f"${(g['pnl_pct'] * VIRTUAL_POS).sum():+.1f}",
        }),
        include_groups=False
    )
    # Only show hours with trades
    hourly = hourly[hourly["N"] > 0]
    for hour_val in range(24):
        if hour_val in hourly.index:
            r = hourly.loc[hour_val]
            print(f"  {hour_val:02d}:00  N={int(r['N']):>3}  WR={r['WR']:>4}  AvgPnL={r['AvgPnL']:>8}  PnL={r['PnL$']:>10}")

print()
print("=" * 90)
print("ANALYSIS COMPLETE")
print("=" * 90)
