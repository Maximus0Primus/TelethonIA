import os
from supabase import create_client

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

# Paper DTRAIL10 closed since Apr 8
res_paper = (sb.table("paper_trades")
    .select("symbol,pnl_pct,pnl_usd,status,entry_price,exit_price,position_usd,exit_minutes,is_shadow,created_at")
    .eq("strategy", "DTRAIL10_ACT15_SL70")
    .eq("is_shadow", False)
    .gte("created_at", "2026-04-08T00:00:00Z")
    .in_("status", ["sl_hit", "tp_hit", "timeout", "trail_stop"])
    .order("created_at", desc=True)
    .execute())

# Live DTRAIL10
res_live = (sb.table("paper_trades")
    .select("symbol,pnl_pct,pnl_usd,status,entry_price,exit_price,position_usd,position_sol,exit_minutes,created_at")
    .eq("strategy", "DTRAIL10_ACT15_SL70")
    .not_.is_("tx_signature", "null")
    .gte("created_at", "2026-04-08T00:00:00Z")
    .order("created_at", desc=True)
    .execute())

print("=== PAPER DTRAIL10 (since Apr 8, non-shadow) ===")
print(f"Total: {len(res_paper.data)}")
paper_pnl = sum(float(r["pnl_usd"] or 0) for r in res_paper.data)
paper_wins = sum(1 for r in res_paper.data if float(r["pnl_usd"] or 0) > 0)
statuses = {}
for r in res_paper.data:
    statuses[r["status"]] = statuses.get(r["status"], 0) + 1
print(f"PnL: ${paper_pnl:.2f}, Wins: {paper_wins}/{len(res_paper.data)}")
print(f"Exit breakdown: {statuses}")
print()
for r in res_paper.data[:15]:
    pnl = float(r["pnl_usd"] or 0)
    pnl_pct = float(r["pnl_pct"] or 0)
    pos = float(r["position_usd"] or 0)
    mins = r["exit_minutes"] or "?"
    sym = r["symbol"]
    st = r["status"]
    dt = (r["created_at"] or "")[:16]
    print(f"  {sym:12s} | {st:10s} | pnl={pnl_pct:+.1f}% ${pnl:+.2f} | pos=${pos:.0f} | {mins}min | {dt}")

print()
print("=== LIVE DTRAIL10 (since Apr 8) ===")
print(f"Total: {len(res_live.data)}")
live_pnl = sum(float(r["pnl_usd"] or 0) for r in res_live.data)
live_wins = sum(1 for r in res_live.data if float(r["pnl_usd"] or 0) > 0)
statuses2 = {}
for r in res_live.data:
    statuses2[r["status"]] = statuses2.get(r["status"], 0) + 1
print(f"PnL: ${live_pnl:.2f}, Wins: {live_wins}/{len(res_live.data)}")
print(f"Exit breakdown: {statuses2}")
print()
for r in res_live.data[:15]:
    pnl = float(r["pnl_usd"] or 0)
    pnl_pct = float(r["pnl_pct"] or 0)
    pos = float(r["position_usd"] or 0)
    sol = r["position_sol"] or "?"
    mins = r["exit_minutes"] or "?"
    sym = r["symbol"]
    st = r["status"]
    dt = (r["created_at"] or "")[:16]
    print(f"  {sym:12s} | {st:10s} | pnl={pnl_pct:+.1f}% ${pnl:+.2f} | pos=${pos:.0f} ({sol} SOL) | {mins}min | {dt}")

# Same tokens
paper_tokens = {}
for r in res_paper.data:
    if r["symbol"] not in paper_tokens:
        paper_tokens[r["symbol"]] = r
live_tokens = {}
for r in res_live.data:
    if r["symbol"] not in live_tokens:
        live_tokens[r["symbol"]] = r
both = set(paper_tokens) & set(live_tokens)
print(f"\n=== SAME TOKEN ({len(both)} in both) ===")
for sym in sorted(both):
    p = paper_tokens[sym]
    l = live_tokens[sym]
    pp = float(p["pnl_pct"] or 0)
    lp = float(l["pnl_pct"] or 0)
    p_pos = float(p["position_usd"] or 0)
    l_pos = float(l["position_usd"] or 0)
    print(f"  {sym:12s} | paper: {pp:+.1f}% ({p['status']}) pos=${p_pos:.0f} | live: {lp:+.1f}% ({l['status']}) pos=${l_pos:.0f} | delta={lp-pp:+.1f}%")
