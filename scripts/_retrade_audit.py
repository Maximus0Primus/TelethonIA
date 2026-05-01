"""Audit retrades: combien de tokens sont buy plusieurs fois en live, quel PnL pour
les 1er/2eme/3eme trades, et est-ce que retrader le meme token est profitable ?

Le dedup actuel est cooldown=1800s (30min) PER (KOL x token), pas global token.
Donc :
- Meme KOL re-call meme token apres 30min: AUTORISE
- KOL different call le meme token: AUTORISE (n'importe quand)
"""
import os, sys, statistics as st
from collections import defaultdict, Counter
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

NOW = datetime.now(timezone.utc)
SINCE_LIVE = "2026-04-29T21:10:00+00:00"  # depuis live deploy
CLOSED = ("sl_hit", "trail_stop", "tp_hit", "timeout", "be_stop")

def fetch():
    out, off, step = [], 0, 1000
    while True:
        r = sb.table("paper_trades").select(
            "id,strategy,status,pnl_pct,pnl_usd,kol_group,token_address,symbol,"
            "created_at,position_usd,is_shadow"
        ).eq("source", "rt_live").eq("chain", "solana").gte("created_at", SINCE_LIVE).range(off, off+step-1).execute()
        if not r.data:
            break
        out.extend(r.data)
        if len(r.data) < step:
            break
        off += step
    return [r for r in out if r.get("status") in CLOSED and not r.get("is_shadow")]

trades = fetch()
print("=" * 100)
print(f"RETRADE AUDIT — rt_live SOL since live deploy ({SINCE_LIVE[:10]}, N={len(trades)} closed trades)")
print("=" * 100)

# ============= 1. TOKENS RE-TRADED — count + PnL aggregated =============
by_token = defaultdict(list)
for t in trades:
    by_token[t["token_address"]].append(t)

# Sort each token's trades chronologically
for ca in by_token:
    by_token[ca].sort(key=lambda r: r["created_at"])

# Distribution: combien de tokens sont trade 1×, 2×, 3×, ...
trade_count_dist = Counter(len(v) for v in by_token.values())
total_unique_tokens = len(by_token)
total_repeats = sum(len(v) for v in by_token.values() if len(v) > 1)
total_first_trades = total_unique_tokens
total_retrade_only = sum(len(v) - 1 for v in by_token.values() if len(v) > 1)
print(f"\n1. DISTRIBUTION trade count per token:")
for n, count in sorted(trade_count_dist.items()):
    pct = 100*count/total_unique_tokens
    print(f"   {n} trades: {count} tokens ({pct:.1f}%)")
print(f"   total unique tokens: {total_unique_tokens}")
print(f"   total trades: {len(trades)}")
print(f"   retrade events (excludes 1st): {total_retrade_only}")

# ============= 2. PNL by trade ORDER (1st, 2nd, 3rd, ...) =============
print(f"\n2. PnL by trade ORDER (1st time we buy a token vs 2nd, 3rd...):")
by_order = defaultdict(list)
for ca, ts in by_token.items():
    for i, t in enumerate(ts, 1):
        by_order[i].append(t)

print(f"   {'order':<8}{'N':>5}{'WR':>7}{'avg%':>9}{'med%':>9}{'$tot':>9}{'$/trade':>10}")
for order in sorted(by_order.keys()):
    rows = by_order[order]
    pcts = [float(r.get("pnl_pct") or 0)*100 for r in rows]
    usd = sum(float(r.get("pnl_usd") or 0) for r in rows)
    wr = 100*sum(1 for p in pcts if p > 0)/len(rows)
    print(f"   #{order:<7}{len(rows):>5}{wr:>6.0f}%{st.mean(pcts):>+8.2f}%{st.median(pcts):>+8.2f}%{usd:>+8.2f}{usd/len(rows):>+9.3f}")

# ============= 3. SAME TOKEN — DIFFERENT KOL or SAME KOL ? =============
print(f"\n3. RETRADE source — meme KOL ou KOL different ?")
same_kol_pairs, diff_kol_pairs = [], []
for ca, ts in by_token.items():
    if len(ts) < 2:
        continue
    for i in range(1, len(ts)):
        prev = ts[i-1]
        curr = ts[i]
        if prev.get("kol_group") == curr.get("kol_group"):
            same_kol_pairs.append((prev, curr))
        else:
            diff_kol_pairs.append((prev, curr))
print(f"   meme KOL re-call: {len(same_kol_pairs)} retrades")
print(f"   KOL different:    {len(diff_kol_pairs)} retrades")

# ============= 4. RETRADES PnL conditioned on PREVIOUS outcome =============
print(f"\n4. RETRADE PnL conditione au resultat du trade PRECEDENT:")
print(f"   Si trade #N etait perdant, est-ce que trade #N+1 sur le meme token marche ?")
prev_was_loss, prev_was_win = [], []
for ca, ts in by_token.items():
    if len(ts) < 2:
        continue
    for i in range(1, len(ts)):
        prev_pnl = float(ts[i-1].get("pnl_pct") or 0)
        curr = ts[i]
        if prev_pnl < 0:
            prev_was_loss.append(curr)
        else:
            prev_was_win.append(curr)

for label, rows in (("prev_loss", prev_was_loss), ("prev_win", prev_was_win)):
    if not rows:
        continue
    pcts = [float(r.get("pnl_pct") or 0)*100 for r in rows]
    usd = sum(float(r.get("pnl_usd") or 0) for r in rows)
    wr = 100*sum(1 for p in pcts if p > 0)/len(rows)
    print(f"   {label:<12} N={len(rows):<3} WR={wr:.0f}% avg={st.mean(pcts):+.2f}% med={st.median(pcts):+.2f}% $tot={usd:+.2f} $/trade={usd/len(rows):+.3f}")

# ============= 5. TIME GAP between retrades =============
print(f"\n5. TIME GAP entre 1er et 2eme trade sur meme token:")
gaps_min = []
for ca, ts in by_token.items():
    if len(ts) < 2:
        continue
    t1 = datetime.fromisoformat(ts[0]["created_at"].replace("Z","+00:00"))
    t2 = datetime.fromisoformat(ts[1]["created_at"].replace("Z","+00:00"))
    gap = (t2 - t1).total_seconds() / 60
    gaps_min.append(gap)

if gaps_min:
    print(f"   min gap:    {min(gaps_min):.1f} min")
    print(f"   median gap: {st.median(gaps_min):.1f} min")
    print(f"   mean gap:   {st.mean(gaps_min):.1f} min")
    print(f"   max gap:    {max(gaps_min):.1f} min")

    # Buckets
    buckets = [(0,5), (5,15), (15,30), (30,60), (60,120), (120,360), (360,720), (720,1500)]
    print(f"\n   Distribution du gap:")
    for lo, hi in buckets:
        n = sum(1 for g in gaps_min if lo <= g < hi)
        bar = "#" * int(50 * n / len(gaps_min)) if gaps_min else ""
        print(f"   [{lo:>4}-{hi:>4}min[: {n:>3} {bar}")

# ============= 6. RETRADE PnL conditione au time gap =============
print(f"\n6. RETRADE PnL par TIME GAP (proche vs loin du 1er trade):")
gap_pnl = defaultdict(list)
for ca, ts in by_token.items():
    if len(ts) < 2:
        continue
    t1 = datetime.fromisoformat(ts[0]["created_at"].replace("Z","+00:00"))
    for i in range(1, len(ts)):
        ti = datetime.fromisoformat(ts[i]["created_at"].replace("Z","+00:00"))
        gap = (ti - t1).total_seconds() / 60
        if gap < 30:
            bucket = "<30min"
        elif gap < 120:
            bucket = "30-120min"
        elif gap < 360:
            bucket = "2-6h"
        elif gap < 720:
            bucket = "6-12h"
        else:
            bucket = ">12h"
        gap_pnl[bucket].append(ts[i])

print(f"   {'gap':<14}{'N':>5}{'WR':>7}{'avg%':>9}{'med%':>9}{'$tot':>9}")
for bucket in ("<30min", "30-120min", "2-6h", "6-12h", ">12h"):
    rows = gap_pnl.get(bucket, [])
    if not rows:
        continue
    pcts = [float(r.get("pnl_pct") or 0)*100 for r in rows]
    usd = sum(float(r.get("pnl_usd") or 0) for r in rows)
    wr = 100*sum(1 for p in pcts if p > 0)/len(rows)
    print(f"   {bucket:<14}{len(rows):>5}{wr:>6.0f}%{st.mean(pcts):>+8.2f}%{st.median(pcts):>+8.2f}%{usd:>+8.2f}")

# ============= 7. TOP HEAVILY RETRADED TOKENS =============
print(f"\n7. TOP 10 tokens les plus retraded (>=3 trades):")
heavy = [(ca, ts) for ca, ts in by_token.items() if len(ts) >= 3]
heavy.sort(key=lambda x: -len(x[1]))
print(f"   {'symbol':<14}{'N':>4}{'WR':>5}{'$tot':>9}{'first_pnl%':>11}{'last_pnl%':>11}")
for ca, ts in heavy[:10]:
    pcts = [float(r.get("pnl_pct") or 0)*100 for r in ts]
    wr = 100*sum(1 for p in pcts if p > 0)/len(ts)
    usd = sum(float(r.get("pnl_usd") or 0) for r in ts)
    sym = ts[0].get("symbol", "?")
    print(f"   {sym:<14}{len(ts):>4}{wr:>4.0f}%{usd:>+8.2f}{pcts[0]:>+10.2f}%{pcts[-1]:>+10.2f}%")
