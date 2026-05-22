"""Simulate live PnL at various position sizes from paper-main outcomes + measured live drift.
Transparent assumptions (printed). Two scenarios: typical (median drift) vs stress (mean drift)."""
import os
from datetime import datetime, timezone
from collections import defaultdict
from supabase import create_client
from dotenv import load_dotenv
load_dotenv("scraper/.env")
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

LIVE_STRATS = ["FAST60_TP70_SL50_NZ_S40", "FAST_TP50_SL30_MCAP_S40"]
EXITS = ["tp_hit", "sl_hit", "timeout", "trail_stop", "be_stop"]
FEE = 0.033          # $/trade, network+priority (rent now recovered via v14e.66)
DAYS = 14

# Measured live<->paper drift by exit type (paired test, N=39, in pp = fraction)
DRIFT_MEDIAN = {"tp_hit": 0.0061, "timeout": -0.0007, "sl_hit": -0.0353, "be_stop": -0.02, "trail_stop": -0.02}
DRIFT_MEAN   = {"tp_hit": -0.0145, "timeout": -0.0489, "sl_hit": -0.1795, "be_stop": -0.08, "trail_stop": -0.08}

since = "2026-05-07T00:00:00Z"
rows = []
for strat in LIVE_STRATS:
    off = 0
    while True:
        b = sb.table("paper_trades").select("pnl_pct,status,created_at").eq("source","rt").eq("is_shadow",False)\
            .eq("chain","solana").eq("strategy",strat).in_("status",EXITS).gte("created_at",since)\
            .order("created_at").range(off,off+999).execute().data or []
        rows += b
        if len(b) < 1000: break
        off += 1000
rows = [r for r in rows if r.get("pnl_pct") is not None]
N = len(rows)
days_span = (datetime.now(timezone.utc) - datetime.fromisoformat(rows[0]["created_at"].replace("Z","+00:00"))).days or 1
tpd = N / days_span
paper_avg = sum(float(r["pnl_pct"]) for r in rows)/N
print(f"Base: {N} paper-main trades (2 live strats), {days_span}d, {tpd:.1f} trades/day")
print(f"Paper-main avg gross: {paper_avg*100:+.2f}%/trade   (fee=${FEE}/trade, rent recovered)")
print(f"Exit mix: ", {k: sum(1 for r in rows if r['status']==k) for k in EXITS})

def sim(drift, label):
    print(f"\n=== Scenario: {label} ===")
    print(f"  {'pos$':>5} | {'live%/trade':>11} | {'net$/trade':>10} | {'net$/day':>9} | {'net$/14d':>9} | viable")
    for size in [1, 5, 10, 25, 50]:
        net_trade = 0.0
        for r in rows:
            live_pct = float(r["pnl_pct"]) + drift.get(r["status"], -0.02)
            net_trade += size*live_pct - FEE
        net_trade /= N
        net_day = net_trade * tpd
        print(f"  {size:>5} | {(paper_avg+sum(drift[s]*sum(1 for r in rows if r['status']==s) for s in drift)/N)*100:>10.2f}% | {net_trade:>+10.3f} | {net_day:>+9.2f} | {net_day*14:>+9.2f} | {'YES' if net_trade>0 else 'no'}")

sim(DRIFT_MEDIAN, "TYPICAL (median drift)")
sim(DRIFT_MEAN, "STRESS (mean drift / rug-tail heavy)")
print("\nNote: failed-buy slippage misses NOT modeled (was hurting live, now eased 225->500).")
print("Note: larger positions also get BETTER Jupiter routes (less entry slip) — NOT modeled = conservative.")
