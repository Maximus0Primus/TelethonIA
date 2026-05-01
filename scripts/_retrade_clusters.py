"""Vrai audit retrades : cluster les trades par session (gap >= 5min = nouveau cluster).
Repond a: est-ce qu'un token est RE-CALL plus tard (par meme/autre KOL) ?
Et est-ce que les strats AGE24/AGE48 capture ces retrades ?"""
import os, sys, statistics as st
from collections import defaultdict, Counter
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

NOW = datetime.now(timezone.utc)
SINCE_LIVE = "2026-04-29T21:10:00+00:00"
SINCE_7D = (NOW - timedelta(days=7)).isoformat()
SINCE_14D = (NOW - timedelta(days=14)).isoformat()
CLOSED = ("sl_hit", "trail_stop", "tp_hit", "timeout", "be_stop")
CLUSTER_GAP_MIN = 5  # gap >= 5min = nouveau cluster de call

def fetch(src, since, chain="solana"):
    out, off, step = [], 0, 1000
    while True:
        r = sb.table("paper_trades").select(
            "id,strategy,status,pnl_pct,pnl_usd,kol_group,token_address,symbol,"
            "created_at,position_usd,is_shadow"
        ).eq("source", src).eq("chain", chain).gte("created_at", since).range(off, off+step-1).execute()
        if not r.data:
            break
        out.extend(r.data)
        if len(r.data) < step:
            break
        off += step
    return [r for r in out if r.get("status") in CLOSED and not r.get("is_shadow")]

def parse_dt(s):
    return datetime.fromisoformat(s.replace("Z","+00:00"))

def cluster_token_trades(trades, gap_min=5):
    """Group trades on same token into clusters (gap >= gap_min between clusters)."""
    by_token = defaultdict(list)
    for t in trades:
        by_token[t["token_address"]].append(t)
    out = {}
    for ca, ts in by_token.items():
        ts.sort(key=lambda r: r["created_at"])
        clusters = [[ts[0]]]
        for i in range(1, len(ts)):
            prev_t = parse_dt(clusters[-1][-1]["created_at"])
            curr_t = parse_dt(ts[i]["created_at"])
            if (curr_t - prev_t).total_seconds() / 60 >= gap_min:
                clusters.append([ts[i]])
            else:
                clusters[-1].append(ts[i])
        out[ca] = clusters
    return out

# ============================================================
# LIVE rt_live SOL
# ============================================================
live_trades = fetch("rt_live", SINCE_LIVE)
print("=" * 100)
print(f"LIVE retrade clusters analysis — rt_live SOL since live deploy ({SINCE_LIVE[:10]})")
print(f"  N={len(live_trades)} closed trades on {len(set(t['token_address'] for t in live_trades))} unique tokens")
print("=" * 100)

clusters_live = cluster_token_trades(live_trades, gap_min=CLUSTER_GAP_MIN)
n_tokens = len(clusters_live)
n_single_cluster = sum(1 for cs in clusters_live.values() if len(cs) == 1)
n_multi_cluster = sum(1 for cs in clusters_live.values() if len(cs) > 1)
print(f"\n1. CLUSTER count par token (cluster = trades avec gap < {CLUSTER_GAP_MIN}min):")
print(f"   tokens avec 1 cluster (call unique):   {n_single_cluster}/{n_tokens} ({100*n_single_cluster/n_tokens:.0f}%)")
print(f"   tokens avec >=2 clusters (RE-CALL):    {n_multi_cluster}/{n_tokens} ({100*n_multi_cluster/n_tokens:.0f}%)")

# Distribution # clusters
dist = Counter(len(cs) for cs in clusters_live.values())
for n, count in sorted(dist.items()):
    print(f"   {n} cluster(s): {count} tokens")

# ============================================================
# 2. Si retrades existent: gap entre clusters
# ============================================================
multi_tokens = [(ca, cs) for ca, cs in clusters_live.items() if len(cs) > 1]
print(f"\n2. GAP entre clusters consecutifs (vrais re-calls):")
all_gaps_h = []
for ca, cs in multi_tokens:
    for i in range(1, len(cs)):
        end_prev = parse_dt(cs[i-1][-1]["created_at"])
        start_curr = parse_dt(cs[i][0]["created_at"])
        gap_h = (start_curr - end_prev).total_seconds() / 3600
        all_gaps_h.append(gap_h)

if all_gaps_h:
    print(f"   N gaps: {len(all_gaps_h)}")
    print(f"   median gap: {st.median(all_gaps_h):.2f}h")
    print(f"   mean gap:   {st.mean(all_gaps_h):.2f}h")
    print(f"   min/max:    {min(all_gaps_h):.2f}h / {max(all_gaps_h):.2f}h")
    buckets = [(0, 0.5), (0.5, 2), (2, 6), (6, 12), (12, 24), (24, 72), (72, 168)]
    print(f"\n   Distribution:")
    for lo, hi in buckets:
        n = sum(1 for g in all_gaps_h if lo <= g < hi)
        bar = "#" * int(40 * n / len(all_gaps_h))
        print(f"   [{lo:>5.1f}-{hi:>5.1f}h[: {n:>3} {bar}")
else:
    print("   AUCUN re-cluster — tous les tokens sont call une seule fois en live")

# ============================================================
# 3. Re-call: meme KOL ou KOL different ?
# ============================================================
if multi_tokens:
    same_kol_recalls, diff_kol_recalls = 0, 0
    for ca, cs in multi_tokens:
        for i in range(1, len(cs)):
            kol_prev = cs[i-1][0].get("kol_group")
            kol_curr = cs[i][0].get("kol_group")
            if kol_prev == kol_curr:
                same_kol_recalls += 1
            else:
                diff_kol_recalls += 1
    print(f"\n3. RE-CALL source:")
    print(f"   meme KOL re-call:   {same_kol_recalls}")
    print(f"   KOL different:      {diff_kol_recalls}")

# ============================================================
# 4. PnL re-cluster: 2eme cluster vs 1er
# ============================================================
if multi_tokens:
    print(f"\n4. PnL des re-clusters (2eme cluster sur le meme token) vs 1er cluster:")
    first_pnls, second_pnls = [], []
    for ca, cs in multi_tokens:
        first_pnls.extend(float(t.get("pnl_pct") or 0)*100 for t in cs[0])
        second_pnls.extend(float(t.get("pnl_pct") or 0)*100 for t in cs[1])
    print(f"   1er cluster:    N={len(first_pnls):<3} avg={st.mean(first_pnls):+.2f}%  med={st.median(first_pnls):+.2f}%")
    print(f"   2eme cluster:   N={len(second_pnls):<3} avg={st.mean(second_pnls):+.2f}%  med={st.median(second_pnls):+.2f}%")

# ============================================================
# 5. SHADOW: meme analyse pour mesurer si AGE24/AGE48 captureraient les retrades
# ============================================================
print(f"\n{'='*100}\n5. SHADOW PAPER 7d — meme analyse (capture les retrades AGE24/AGE48 ?)\n{'='*100}")
shadow_trades = fetch("rt", SINCE_7D)
print(f"\n   N={len(shadow_trades)} closed trades on {len(set(t['token_address'] for t in shadow_trades))} unique tokens (7d)")

clusters_sh = cluster_token_trades(shadow_trades, gap_min=CLUSTER_GAP_MIN)
n_sh_tokens = len(clusters_sh)
n_sh_multi = sum(1 for cs in clusters_sh.values() if len(cs) > 1)
print(f"   tokens avec >=2 clusters: {n_sh_multi}/{n_sh_tokens} ({100*n_sh_multi/n_sh_tokens:.1f}%)")

# Gap distribution
sh_gaps_h = []
for ca, cs in clusters_sh.items():
    if len(cs) < 2:
        continue
    for i in range(1, len(cs)):
        end_prev = parse_dt(cs[i-1][-1]["created_at"])
        start_curr = parse_dt(cs[i][0]["created_at"])
        sh_gaps_h.append((start_curr - end_prev).total_seconds() / 3600)

if sh_gaps_h:
    print(f"\n   Distribution gap entre re-clusters (paper main shadow):")
    buckets = [(0, 0.5), (0.5, 2), (2, 6), (6, 12), (12, 24), (24, 48), (48, 168)]
    for lo, hi in buckets:
        n = sum(1 for g in sh_gaps_h if lo <= g < hi)
        bar = "#" * int(40 * n / len(sh_gaps_h)) if sh_gaps_h else ""
        print(f"   [{lo:>5.1f}-{hi:>5.1f}h[: {n:>4} {bar}")

# ============================================================
# 6. AGE24/AGE48 strats — combien de trades, PnL ?
# ============================================================
print(f"\n{'='*100}\n6. AGE24/AGE48 strategies — performance recente (capture des retrades age 12-48h)\n{'='*100}")
def fetch_strat_shadow(strat, since):
    out, off, step = [], 0, 1000
    while True:
        r = sb.table("paper_trades").select(
            "strategy,status,pnl_pct,pnl_usd,kol_group,token_address,symbol,created_at,is_shadow,chain"
        ).eq("source", "rt").eq("chain", "solana").eq("strategy", strat).gte("created_at", since).range(off, off+step-1).execute()
        if not r.data:
            break
        out.extend(r.data)
        if len(r.data) < step:
            break
        off += step
    return [r for r in out if r.get("status") in CLOSED]

AGE_STRATS = ["AGE24_FAST_TP50_SL30", "AGE48_FAST_TP50_SL30",
              "AGE24_BE25_LOCK10_TP80_SL30", "AGE48_BE25_LOCK10_TP80_SL30",
              "AGE24_BE50_LOCK20_TP150_SL30", "AGE48_BE50_LOCK20_TP150_SL30"]

print(f"\n  {'strategy':<36}{'window':>7}{'N':>5}{'WR':>6}{'avg%':>9}{'med%':>9}{'$tot':>10}{'$/d':>9}")
for strat in AGE_STRATS:
    for label, since, days in (("7d", SINCE_7D, 7), ("14d", SINCE_14D, 14)):
        rows = fetch_strat_shadow(strat, since)
        if not rows:
            print(f"  {strat:<34}{label:>7}{0:>5}")
            continue
        pcts = [float(r.get("pnl_pct") or 0)*100 for r in rows]
        usd = sum(float(r.get("pnl_usd") or 0) for r in rows)
        wr = 100*sum(1 for p in pcts if p > 0)/len(rows)
        is_shadow_count = sum(1 for r in rows if r.get("is_shadow"))
        print(f"  {strat:<34}{label:>7}{len(rows):>5}{wr:>5.0f}%{st.mean(pcts):>+8.2f}%{st.median(pcts):>+8.2f}%{usd:>+9.2f}{usd/days:>+8.2f}")
