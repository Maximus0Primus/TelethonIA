"""Compare live trade exit patterns pre vs post v123."""
import os
from supabase import create_client

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

# All DTRAIL10 live trades (tx_signature not null)
res = (sb.table("paper_trades")
    .select("symbol,entry_price,exit_price,high_price_seen,pnl_pct,exit_minutes,status,created_at,exit_at")
    .eq("strategy", "DTRAIL10_ACT15_SL70")
    .not_.is_("tx_signature", "null")
    .gte("created_at", "2026-04-08T00:00:00Z")
    .in_("status", ["trail_stop", "sl_hit", "timeout"])
    .order("created_at", desc=True)
    .limit(40)
    .execute())

print(f"{'Symbol':12s} | {'Mins':>4s} | {'High/Entry':>10s} | {'PnL%':>8s} | {'Exit/Trig':>9s} | {'Created':19s} | Phase")
print("-" * 95)

v123_cutoff = "2026-04-10T15:47"  # approximate deploy time

for t in res.data:
    entry = float(t["entry_price"] or 1)
    exit_p = float(t["exit_price"] or 0)
    high = float(t["high_price_seen"] or 0)
    pnl = float(t["pnl_pct"] or 0)
    mins = t["exit_minutes"] or 0
    created = (t["created_at"] or "")[:19]
    phase = "POST-v123" if created >= v123_cutoff else "pre-v123"

    high_entry_ratio = high / entry if entry > 0 else 0
    act_price = entry * 1.15
    trail_trigger = high * 0.90 if high >= act_price else 0
    exit_trig_ratio = exit_p / trail_trigger if trail_trigger > 0 else 0

    sym = t["symbol"]
    status = t["status"]

    print(f"{sym:12s} | {mins:4d} | {high_entry_ratio:9.1f}x | {pnl*100:+7.1f}% | {exit_trig_ratio:9.3f} | {created} | {phase}")

# Stats
pre = [t for t in res.data if (t["created_at"] or "") < v123_cutoff]
post = [t for t in res.data if (t["created_at"] or "") >= v123_cutoff]

print(f"\n--- PRE-v123 ({len(pre)} trades) ---")
if pre:
    avg_pnl = sum(float(t["pnl_pct"] or 0) for t in pre) / len(pre)
    avg_mins = sum(int(t["exit_minutes"] or 0) for t in pre) / len(pre)
    quick = sum(1 for t in pre if int(t["exit_minutes"] or 0) <= 2)
    print(f"  Avg PnL: {avg_pnl*100:+.2f}%, Avg duration: {avg_mins:.0f}min, Quick exits (<=2min): {quick}/{len(pre)}")

print(f"\n--- POST-v123 ({len(post)} trades) ---")
if post:
    avg_pnl = sum(float(t["pnl_pct"] or 0) for t in post) / len(post)
    avg_mins = sum(int(t["exit_minutes"] or 0) for t in post) / len(post)
    quick = sum(1 for t in post if int(t["exit_minutes"] or 0) <= 2)
    print(f"  Avg PnL: {avg_pnl*100:+.2f}%, Avg duration: {avg_mins:.0f}min, Quick exits (<=2min): {quick}/{len(post)}")
