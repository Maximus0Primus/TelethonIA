"""Debug trades closing immediately after v123 Jupiter changes."""
import os
from supabase import create_client

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

# All live trades since v123 deploy
res = (sb.table("paper_trades")
    .select("*")
    .not_.is_("tx_signature", "null")
    .gte("created_at", "2026-04-10T15:45:00Z")
    .order("created_at", desc=True)
    .limit(10)
    .execute())

print(f"LIVE TRADES SINCE v123 DEPLOY")
print("=" * 100)
for t in res.data:
    sym = t["symbol"]
    pnl_pct = float(t["pnl_pct"] or 0)
    entry = float(t["entry_price"] or 0)
    exec_p = t.get("execution_price")
    exit_p = float(t["exit_price"] or 0)
    high = float(t["high_price_seen"] or 0)
    dex_spot = t.get("dex_spot_price_at_entry")
    status = t["status"]
    created = (t["created_at"] or "")[:19]
    exit_at = (t["exit_at"] or "")[:19]
    mins = t.get("exit_minutes", "?")
    paper_exit = t.get("paper_exit_price")
    div = t.get("price_divergence_pct")

    # Compute trail trigger
    act_price = entry * 1.15  # ACT15
    trail_trigger = high * 0.90 if high > 0 else 0  # DTRAIL10

    print(f"\n--- ${sym} ({status}) ---")
    print(f"  Created:   {created}")
    print(f"  Exit:      {exit_at} ({mins}min)")
    print(f"  Entry:     {entry:.10f}  (exec={exec_p}, dex_spot={dex_spot})")
    print(f"  Exit:      {exit_p:.10f}")
    print(f"  High seen: {high:.10f}")
    print(f"  PnL:       {pnl_pct*100:+.2f}%")
    print(f"  --- Trail analysis ---")
    print(f"  Activation price (entry*1.15): {act_price:.10f}")
    print(f"  High >= activation?  {high >= act_price} (high/entry = {high/entry*100:.1f}%)")
    print(f"  Trail trigger (high*0.90):     {trail_trigger:.10f}")
    print(f"  Exit price <= trigger?  {exit_p <= trail_trigger}")
    if trail_trigger > 0:
        print(f"  Exit/trigger ratio: {exit_p/trail_trigger:.4f}")
    print(f"  Paper exit: {paper_exit}  Divergence: {div}")

# Also get the price_ticks for these tokens around entry time
print("\n\n" + "=" * 100)
print("PRICE TICKS around entry for these tokens")
print("=" * 100)
for t in res.data[:3]:
    addr = t["token_address"]
    sym = t["symbol"]
    created = t["created_at"]
    ticks = (sb.table("price_ticks")
        .select("price_usd,source,created_at")
        .eq("token_address", addr)
        .gte("created_at", "2026-04-10T15:45:00Z")
        .order("created_at", desc=False)
        .limit(20)
        .execute())
    print(f"\n--- ${sym} ({addr[:8]}) ---")
    for tick in ticks.data:
        ts = (tick["created_at"] or "")[:19]
        src = tick["source"]
        price = tick["price_usd"]
        print(f"  {ts} | {src:8s} | {price}")
