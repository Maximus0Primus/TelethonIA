"""Analyze last 30h drawdown vs previous period."""
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from supabase import create_client

sb = create_client(os.environ["SUPABASE_URL"], os.environ.get("SUPABASE_KEY") or os.environ["SUPABASE_SERVICE_ROLE_KEY"])

now = datetime.now(timezone.utc)
cutoff_bad = (now - timedelta(hours=30)).isoformat()
cutoff_good = (now - timedelta(days=7)).isoformat()
SEP = "=" * 80

for label, gte, lte in [("LAST 30h", cutoff_bad, None), ("BEFORE 5.5d", cutoff_good, cutoff_bad)]:
    q = (sb.table("paper_trades")
         .select("id,strategy,source,status,pnl_pct,pnl_usd,position_usd,kol_group,created_at,token_address,symbol")
         .neq("status", "open")
         .gte("created_at", gte)
         .gt("position_usd", "0"))
    if lte:
        q = q.lt("created_at", lte)
    data = []
    offset = 0
    while True:
        res = q.order("created_at").range(offset, offset + 999).execute()
        data.extend(res.data)
        if len(res.data) < 1000:
            break
        offset += 1000

    kol_stats = defaultdict(lambda: {"wins": 0, "total": 0, "pnl_sum": 0.0, "usd": 0.0})
    for t in data:
        kol = t.get("kol_group") or "unknown"
        pnl = float(t.get("pnl_pct") or 0)
        usd = float(t.get("pnl_usd") or 0)
        kol_stats[kol]["total"] += 1
        kol_stats[kol]["pnl_sum"] += pnl
        kol_stats[kol]["usd"] += usd
        if pnl > 0:
            kol_stats[kol]["wins"] += 1

    all_pnl = [float(t.get("pnl_pct") or 0) for t in data]
    all_usd = sum(float(t.get("pnl_usd") or 0) for t in data)
    all_wr = sum(1 for p in all_pnl if p > 0) / len(all_pnl) * 100 if all_pnl else 0
    all_avg = sum(all_pnl) / len(all_pnl) * 100 if all_pnl else 0

    print(f"\n{SEP}")
    print(f"{label} -- {len(data)} hybrid trades | WR={all_wr:.1f}% | avg={all_avg:+.1f}% | ${all_usd:+.2f}")
    print(SEP)

    sorted_kols = sorted(kol_stats.items(), key=lambda x: x[1]["usd"])
    for kol, s in sorted_kols:
        wr = s["wins"] / s["total"] * 100
        avg = s["pnl_sum"] / s["total"] * 100
        marker = " <<<" if s["total"] >= 3 and wr < 40 else ""
        print(f"  {kol:30s} N={s['total']:3d}  WR={wr:5.1f}%  avg={avg:+6.1f}%  ${s['usd']:+8.2f}{marker}")

# Token-level last 30h
print(f"\n{SEP}")
print("WORST TOKENS -- LAST 30h (hybrid, by total $ lost)")
print(SEP)
q = (sb.table("paper_trades")
     .select("id,strategy,status,pnl_pct,pnl_usd,position_usd,kol_group,symbol,token_address,created_at,exit_minutes")
     .neq("status", "open")
     .gte("created_at", cutoff_bad)
     .gt("position_usd", "0"))
data2 = []
offset = 0
while True:
    res = q.order("created_at").range(offset, offset + 999).execute()
    data2.extend(res.data)
    if len(res.data) < 1000:
        break
    offset += 1000

tok_stats = defaultdict(lambda: {"trades": 0, "usd": 0.0, "pnl_sum": 0.0, "kol": "", "sym": ""})
for t in data2:
    ca = t.get("token_address", "")
    pnl = float(t.get("pnl_pct") or 0)
    usd = float(t.get("pnl_usd") or 0)
    tok_stats[ca]["trades"] += 1
    tok_stats[ca]["usd"] += usd
    tok_stats[ca]["pnl_sum"] += pnl
    tok_stats[ca]["kol"] = t.get("kol_group", "")
    tok_stats[ca]["sym"] = t.get("symbol", "")

sorted_tok = sorted(tok_stats.items(), key=lambda x: x[1]["usd"])
for ca, s in sorted_tok[:15]:
    avg = s["pnl_sum"] / s["trades"] * 100
    print(f"  {s['sym']:15s} N={s['trades']:2d}  avg={avg:+6.1f}%  ${s['usd']:+8.2f}  kol={s['kol']}")

# Hourly breakdown last 30h
print(f"\n{SEP}")
print("HOURLY PnL -- LAST 30h (all hybrid trades)")
print(SEP)
hourly = defaultdict(lambda: {"n": 0, "wins": 0, "usd": 0.0})
for t in data2:
    h = t["created_at"][:13]  # YYYY-MM-DDTHH
    pnl = float(t.get("pnl_pct") or 0)
    usd = float(t.get("pnl_usd") or 0)
    hourly[h]["n"] += 1
    hourly[h]["usd"] += usd
    if pnl > 0:
        hourly[h]["wins"] += 1

for h in sorted(hourly.keys()):
    s = hourly[h]
    wr = s["wins"] / s["n"] * 100 if s["n"] else 0
    bar = "#" * max(1, int(abs(s["usd"]) / 10))
    sign = "+" if s["usd"] >= 0 else "-"
    print(f"  {h}  N={s['n']:3d}  WR={wr:5.1f}%  ${s['usd']:+8.2f}  {sign}{bar}")
