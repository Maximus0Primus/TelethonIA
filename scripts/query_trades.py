import os
from supabase import create_client
from collections import defaultdict, Counter

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

# 1) Paper trades - 3 main strategies (last 7 days, closed = sl_hit/tp_hit/timeout)
closed_statuses = ["sl_hit", "tp_hit", "timeout"]
main3 = ["DTRAIL10_ACT15_SL70", "DTRAIL3_ACT5_SL60", "DIP30_B5_T5_A20_SL70_240m"]
stats = defaultdict(lambda: {"count":0, "wins":0, "pnl_usd":0, "pnl_pcts":[], "exits": defaultdict(int)})

for status in closed_statuses:
    for strat in main3:
        res = sb.table("paper_trades").select("strategy,pnl_pct,pnl_usd,status,is_shadow").eq("status", status).eq("strategy", strat).eq("is_shadow", False).gte("created_at", "2026-04-03T00:00:00Z").execute()
        for t in res.data:
            s = t["strategy"]
            stats[s]["count"] += 1
            pnl = float(t["pnl_usd"] or 0)
            pnl_pct = float(t["pnl_pct"] or 0)
            stats[s]["pnl_usd"] += pnl
            stats[s]["pnl_pcts"].append(pnl_pct)
            if pnl > 0:
                stats[s]["wins"] += 1
            stats[s]["exits"][t["status"]] += 1

print("=" * 60)
print("PAPER TRADES - 3 Main Strategies (last 7 days, non-shadow)")
print("=" * 60)
for s in main3:
    d = stats[s]
    wr = (d["wins"] / d["count"] * 100) if d["count"] else 0
    avg = (sum(d["pnl_pcts"]) / len(d["pnl_pcts"])) if d["pnl_pcts"] else 0
    print(f"\n--- {s} ---")
    print(f"  Trades: {d['count']}, WR: {wr:.1f}%, Total PnL: ${d['pnl_usd']:.2f}, Avg PnL%: {avg:.1f}%")
    print(f"  Exits: {dict(d['exits'])}")

# 2) Live trades (tx_signature not null = on-chain)
print("\n" + "=" * 60)
print("LIVE TRADES (on-chain, tx_signature != null)")
print("=" * 60)
res = sb.table("paper_trades").select("symbol,strategy,status,pnl_pct,pnl_usd,position_sol,tx_signature,created_at,exit_at").not_.is_("tx_signature", "null").order("created_at", desc=True).limit(30).execute()
print(f"Total live trades: {len(res.data)}")
total_live_pnl = 0
for t in res.data:
    pnl = float(t["pnl_usd"] or 0)
    pnl_pct = float(t["pnl_pct"] or 0)
    total_live_pnl += pnl
    sol = t["position_sol"] or "?"
    dt = (t["created_at"] or "")[:16]
    exit_dt = (t["exit_at"] or "OPEN")[:16]
    print(f"  {t['symbol']:12s} | {t['strategy']:25s} | {t['status']:8s} | pnl={pnl_pct:+.1f}% ${pnl:+.2f} | {sol} SOL | {dt} -> {exit_dt}")
print(f"\nTotal live realized PnL: ${total_live_pnl:.2f}")

# 3) Currently open positions
print("\n" + "=" * 60)
print("CURRENTLY OPEN POSITIONS")
print("=" * 60)
res_open = sb.table("paper_trades").select("strategy,symbol,pnl_pct,is_shadow,created_at,tx_signature").eq("status", "open").execute()
live_open = [r for r in res_open.data if r.get("tx_signature")]
paper_open = [r for r in res_open.data if not r.get("tx_signature") and not r.get("is_shadow")]
shadow_open = [r for r in res_open.data if r.get("is_shadow")]

print(f"Live (on-chain): {len(live_open)}")
for t in live_open:
    print(f"  {t['symbol']:12s} | {t['strategy']:25s} | since {(t['created_at'] or '')[:16]}")

print(f"\nPaper (main, non-shadow): {len(paper_open)}")
strats = Counter(r["strategy"] for r in paper_open)
for s, c in strats.most_common(10):
    print(f"  {s}: {c} open")

print(f"\nShadow: {len(shadow_open)}")
