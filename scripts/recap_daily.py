"""Recap : paper actuel + live actuel + projection post-changes v144."""
import os, sys, json, statistics as st
from collections import defaultdict
from datetime import datetime, timedelta, timezone
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv; load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

SINCE_7D = (datetime.now(timezone.utc) - timedelta(hours=168)).isoformat()
DAYS = 7

def fetch_all(tbl, sel, **f):
    out, step, off = [], 1000, 0
    while True:
        q = sb.table(tbl).select(sel)
        for k, v in f.items():
            if k.startswith("gte_"): q = q.gte(k[4:], v)
            elif k.startswith("eq_"): q = q.eq(k[3:], v)
        r = q.range(off, off+step-1).execute()
        if not r.data: break
        out.extend(r.data)
        if len(r.data) < step: break
        off += step
    return out

# 1. Load current config
r = sb.table("scoring_config").select("rt_trade_config").eq("id",1).execute()
cfg = r.data[0]["rt_trade_config"]
if isinstance(cfg, str): cfg = json.loads(cfg)
hybrid_active = sorted(cfg["hybrid_strategy"]["allocations"].keys())
live_active = cfg["live_trading"]["allocations"]

# 2. Paper 7d per-strat
rows = fetch_all("paper_trades", "strategy,status,source,pnl_pct,pnl_usd",
                 gte_created_at=SINCE_7D, eq_source="rt")
closed = [r for r in rows if r.get("status") in ("sl_hit","trail_stop","tp_hit","timeout","be_stop")]
by_strat = defaultdict(list)
for r in closed:
    by_strat[r["strategy"]].append(r)

def stats_for(strat):
    xs = by_strat.get(strat, [])
    n = len(xs)
    if n == 0: return None
    pnls = [float(x.get("pnl_pct") or 0)*100 for x in xs]
    usd = sum(float(x.get("pnl_usd") or 0) for x in xs)
    wr = 100 * sum(1 for p in pnls if p>0) / n
    return {"n":n, "wr":wr, "avg":st.mean(pnls), "usd_7d":usd, "usd_day":usd/DAYS}

print("=" * 80)
print("PAPER — hybrid_strategy.allocations (16 strats actives post-v144)")
print("=" * 80)
print(f"{'Strategy':<32}{'N 7d':>7}{'WR':>7}{'avg%':>9}{'$ 7d':>10}{'$/jour':>10}")
print("-" * 75)
total_7d = 0
for s in hybrid_active:
    st_ = stats_for(s)
    if st_ is None:
        print(f"{s:<32}{'—':>7}{'—':>7}{'—':>9}{'—':>10}{'—':>10}")
        continue
    total_7d += st_["usd_7d"]
    print(f"{s:<32}{st_['n']:>7}{st_['wr']:>6.1f}%{st_['avg']:>+8.2f}%{st_['usd_7d']:>+9.2f}{st_['usd_day']:>+9.2f}")
print(f"{'TOTAL':<32}{'':>7}{'':>7}{'':>9}{total_7d:>+9.2f}{total_7d/DAYS:>+9.2f}")

# 3. Live 7d
live_rows = fetch_all("paper_trades", "strategy,status,pnl_pct,pnl_usd",
                      gte_created_at=SINCE_7D, eq_source="rt_live")
live_closed = [r for r in live_rows if r.get("status") in ("sl_hit","trail_stop","tp_hit","timeout","be_stop")]
live_by_strat = defaultdict(list)
for r in live_closed:
    live_by_strat[r["strategy"]].append(r)

print()
print("=" * 80)
print("LIVE — real money last 7d (BEFORE v144 swap)")
print("=" * 80)
print(f"{'Strategy':<30}{'N 7d':>7}{'WR':>7}{'avg%':>9}{'$ 7d':>10}{'$/jour':>10}")
print("-" * 73)
total_live = 0
for s, xs in sorted(live_by_strat.items()):
    pnls = [float(x.get("pnl_pct") or 0)*100 for x in xs]
    usd = sum(float(x.get("pnl_usd") or 0) for x in xs)
    n = len(xs); wr = 100*sum(1 for p in pnls if p>0)/n if n else 0
    total_live += usd
    print(f"{s:<30}{n:>7}{wr:>6.1f}%{st.mean(pnls):>+8.2f}%{usd:>+9.2f}{usd/DAYS:>+9.2f}")
print(f"{'TOTAL LIVE':<30}{'':>7}{'':>7}{'':>9}{total_live:>+9.2f}{total_live/DAYS:>+9.2f}")

# 4. Projection LIVE post-swap (BE15 -> FAST_TP50_SL30)
# Live uses $1.70/trade, paper $50/trade. Scale paper $/jour by ratio.
# Actual live position from rt_bankroll config
live_pos = 1.70  # per todo
paper_pos = 50.0
scale = live_pos / paper_pos  # = 0.034

be25_paper_day = stats_for("BE25_TP80_SL30")["usd_day"] if stats_for("BE25_TP80_SL30") else 0
fast_paper_day = stats_for("FAST_TP50_SL30")["usd_day"] if stats_for("FAST_TP50_SL30") else 0
# Live allocation is 50/50, so each strat fires on ~half the tokens
projected_be25 = be25_paper_day * scale * 0.5  # half allocations
projected_fast = fast_paper_day * scale * 0.5

# Actual live pnl_usd already reflects $1.70 position sizing, so direct comparison:
be25_live_day = sum(float(x.get("pnl_usd") or 0) for x in live_by_strat.get("BE25_TP80_SL30",[])) / DAYS
be15_live_day = sum(float(x.get("pnl_usd") or 0) for x in live_by_strat.get("BE15_TP100_SL50",[])) / DAYS

print()
print("=" * 80)
print("LIVE — projection POST-SWAP (BE15 → FAST_TP50_SL30)")
print("=" * 80)
print(f"Live position: ${live_pos}/trade  (scale vs paper ${paper_pos}: {scale:.4f})")
print()
print(f"{'Strategy':<24}{'7d actual':>11}{'$/j actual':>11}{'$/j projeté':>13}{'source':>18}")
print("-" * 76)
print(f"{'BE25_TP80_SL30':<24}{be25_live_day*DAYS:>+10.2f}{be25_live_day:>+10.2f}{be25_live_day:>+12.2f}  {'live real (kept)':>16}")
print(f"{'BE15 (OUT)':<24}{be15_live_day*DAYS:>+10.2f}{be15_live_day:>+10.2f}{'0.00':>12}  {'removed':>16}")
print(f"{'FAST_TP50_SL30 (IN)':<24}{'—':>11}{'—':>11}{projected_fast:>+12.2f}  {'paper×scale×0.5':>16}")
total_before = be25_live_day + be15_live_day
total_after = be25_live_day + projected_fast
print("-" * 76)
print(f"{'TOTAL before (actual)':<24}{total_before*DAYS:>+10.2f}{total_before:>+10.2f}")
print(f"{'TOTAL after (projected)':<24}{'':>11}{'':>11}{total_after:>+12.2f}")
delta_day = total_after - total_before
print(f"{'Delta $/jour':<24}{'':>11}{'':>11}{delta_day:>+12.2f}  ({delta_day*30:+.2f}/mois)")

print()
print("Notes:")
print("  * Paper 50/trade, live 1.70/trade => scale 0.034x per trade")
print("  * FAST prediction assumes paper->live perfect match; real slip, fills, competition may reduce")
print("  * N live trades/week is small (~5-15), projection has wide CI")
