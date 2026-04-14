"""Compare DTRAIL3_ACT5_SL60 (LAZY) vs DTRAIL10_ACT15_SL70 stats."""
import os, json, sys
from collections import defaultdict
from supabase import create_client

sb = create_client(os.environ["SUPABASE_URL"], os.environ.get("SUPABASE_KEY") or os.environ["SUPABASE_SERVICE_ROLE_KEY"])

strats = ["DTRAIL3_ACT5_SL60", "DTRAIL10_ACT15_SL70", "DTRAIL10_ACT10_SL70"]
results = {}

for strat in strats:
    data = []
    offset = 0
    while True:
        res = (sb.table("paper_trades")
               .select("id,strategy,source,status,pnl_pct,pnl_usd,position_usd,exit_minutes,high_price_seen,entry_price,exit_price,created_at")
               .eq("strategy", strat).neq("status", "open")
               .gte("created_at", "2026-04-04T00:00:00Z")  # same period for fair comparison
               .order("created_at").range(offset, offset + 999).execute())
        data.extend(res.data)
        if len(res.data) < 1000:
            break
        offset += 1000
    results[strat] = data

for strat in strats:
    trades = results[strat]
    hybrid = [t for t in trades if float(t.get("position_usd") or 0) > 0]
    shadow = [t for t in trades if float(t.get("position_usd") or 0) == 0]

    is_lazy = strat in ["DTRAIL3_ACT5_SL60", "DTRAIL5_ACT10_SL60"]
    for label, subset in [(f"HYBRID {'(LAZY)' if is_lazy else ''}", hybrid),
                           ("SHADOW (control)", shadow)]:
        if not subset:
            continue
        pnls = [float(t.get("pnl_pct") or 0) for t in subset]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total_usd = sum(float(t.get("pnl_usd") or 0) for t in subset)
        avg_pnl = sum(pnls) / len(pnls) if pnls else 0
        median_pnl = sorted(pnls)[len(pnls) // 2] if pnls else 0
        wr = len(wins) / len(pnls) * 100 if pnls else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        avg_min = sum(float(t.get("exit_minutes") or 0) for t in subset) / len(subset)

        statuses = defaultdict(int)
        for t in subset:
            statuses[t["status"]] += 1

        print(f"\n{'=' * 60}")
        print(f"{strat} -- {label}")
        print(f"{'=' * 60}")
        print(f"  Trades:     {len(subset)}")
        print(f"  Win rate:   {wr:.1f}%")
        print(f"  Avg PnL:    {avg_pnl * 100:+.1f}%")
        print(f"  Median PnL: {median_pnl * 100:+.1f}%")
        print(f"  Total $:    ${total_usd:+.2f}")
        print(f"  Avg win:    {avg_win * 100:+.1f}%")
        print(f"  Avg loss:   {avg_loss * 100:+.1f}%")
        print(f"  Avg time:   {avg_min:.0f}min")
        print(f"  Exits:      {dict(statuses)}")

        by_pnl = sorted(subset, key=lambda t: float(t.get("pnl_pct") or 0), reverse=True)
        print(f"  Best 5:")
        for t in by_pnl[:5]:
            p = float(t.get("pnl_pct") or 0)
            print(f"    {t['created_at'][:16]} {p * 100:+.1f}%")
        print(f"  Worst 5:")
        for t in by_pnl[-5:]:
            p = float(t.get("pnl_pct") or 0)
            print(f"    {t['created_at'][:16]} {p * 100:+.1f}%")
