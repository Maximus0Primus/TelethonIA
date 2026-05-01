"""Audit complet retrades + RECALL strats sur N grand (14d/30d).
- Cluster retrades sur paper main + shadow 14d/30d
- Top RECALL_* strats sur 14d/30d (perf + N)
- Confirme: max_age_hours appliqué en live ? (oui depuis v144.16, safe_scraper.py:1718)
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
SINCE_14D = (NOW - timedelta(days=14)).isoformat()
SINCE_30D = (NOW - timedelta(days=30)).isoformat()
CLOSED = ("sl_hit", "trail_stop", "tp_hit", "timeout", "be_stop")
CLUSTER_GAP_MIN = 30  # gap >= 30min = nouveau cluster

def fetch_paged(query):
    out, off, step = [], 0, 1000
    while True:
        r = query.range(off, off+step-1).execute()
        if not r.data:
            break
        out.extend(r.data)
        if len(r.data) < step:
            break
        off += step
    return out

def parse_dt(s):
    return datetime.fromisoformat(s.replace("Z","+00:00"))

# ============================================================
# 1. RETRADES analysis sur paper main 14d/30d (gros N)
# ============================================================
def analyze_retrades(label, since, source_filter):
    """source_filter: list of (source, is_shadow_value) tuples"""
    rows = []
    for src in source_filter:
        q = sb.table("paper_trades").select(
            "strategy,status,pnl_pct,pnl_usd,kol_group,token_address,symbol,created_at,is_shadow"
        ).eq("source", src).eq("chain", "solana").gte("created_at", since)
        rows.extend(fetch_paged(q))
    rows = [r for r in rows if r.get("status") in CLOSED and not r.get("is_shadow")]

    print(f"\n{'='*100}\nRETRADES {label} (paper main rt, SOL, {len(rows)} trades)\n{'='*100}")

    # cluster by token (gap >= 30min)
    by_token = defaultdict(list)
    for r in rows:
        by_token[r["token_address"]].append(r)
    for ca in by_token:
        by_token[ca].sort(key=lambda r: r["created_at"])

    clusters_per_token = {}
    for ca, ts in by_token.items():
        clusters = [[ts[0]]]
        for i in range(1, len(ts)):
            prev_t = parse_dt(clusters[-1][-1]["created_at"])
            curr_t = parse_dt(ts[i]["created_at"])
            if (curr_t - prev_t).total_seconds() / 60 >= CLUSTER_GAP_MIN:
                clusters.append([ts[i]])
            else:
                clusters[-1].append(ts[i])
        clusters_per_token[ca] = clusters

    n_tokens = len(clusters_per_token)
    n_multi = sum(1 for cs in clusters_per_token.values() if len(cs) > 1)
    print(f"\nUnique tokens: {n_tokens}, with >=2 clusters: {n_multi} ({100*n_multi/n_tokens:.1f}%)")

    # Cluster count distribution
    dist = Counter(len(cs) for cs in clusters_per_token.values())
    for n_cl in sorted(dist.keys()):
        if n_cl <= 5 or dist[n_cl] >= 5:
            print(f"  {n_cl} cluster(s): {dist[n_cl]} tokens")

    # PnL by cluster ORDER (1st cluster vs 2nd vs 3rd...)
    by_cluster_idx = defaultdict(list)
    for ca, clusters in clusters_per_token.items():
        for ci, cluster in enumerate(clusters, 1):
            for trade in cluster:
                by_cluster_idx[ci].append(trade)

    print(f"\nPnL by cluster ORDER (1st cluster on a token vs 2nd, 3rd...):")
    print(f"  {'order':<8}{'N':>5}{'WR':>7}{'avg%':>9}{'med%':>9}{'$tot':>10}")
    for order in sorted(by_cluster_idx.keys()):
        if order > 5:
            break
        rs = by_cluster_idx[order]
        if len(rs) < 5:
            continue
        pcts = [float(r.get("pnl_pct") or 0)*100 for r in rs]
        usd = sum(float(r.get("pnl_usd") or 0) for r in rs)
        wr = 100*sum(1 for p in pcts if p > 0)/len(rs)
        print(f"  #{order:<7}{len(rs):>5}{wr:>6.0f}%{st.mean(pcts):>+8.2f}%{st.median(pcts):>+8.2f}%{usd:>+9.2f}")

    # Gap distribution between cluster 1 and cluster 2
    gaps_h = []
    same_kol_count = 0
    diff_kol_count = 0
    for ca, clusters in clusters_per_token.items():
        if len(clusters) < 2:
            continue
        for i in range(1, len(clusters)):
            end_prev = parse_dt(clusters[i-1][-1]["created_at"])
            start_curr = parse_dt(clusters[i][0]["created_at"])
            gaps_h.append((start_curr - end_prev).total_seconds() / 3600)
            kol_prev = clusters[i-1][0].get("kol_group")
            kol_curr = clusters[i][0].get("kol_group")
            if kol_prev == kol_curr:
                same_kol_count += 1
            else:
                diff_kol_count += 1

    if gaps_h:
        print(f"\nGap distribution (between consecutive clusters):")
        print(f"  N gaps: {len(gaps_h)}, median {st.median(gaps_h):.1f}h, mean {st.mean(gaps_h):.1f}h")
        buckets = [(0, 0.5), (0.5, 2), (2, 6), (6, 12), (12, 24), (24, 48), (48, 168), (168, 1000)]
        for lo, hi in buckets:
            n = sum(1 for g in gaps_h if lo <= g < hi)
            bar = "#" * int(50 * n / len(gaps_h)) if gaps_h else ""
            print(f"  [{lo:>5.1f}-{hi:>5.1f}h[: {n:>4} {bar}")
        print(f"\n  same KOL re-call: {same_kol_count}, different KOL: {diff_kol_count}")

# Run on 14d and 30d
analyze_retrades("14d", SINCE_14D, ["rt"])
analyze_retrades("30d", SINCE_30D, ["rt"])

# ============================================================
# 2. RECALL_* strats — top performers 14d
# ============================================================
print(f"\n{'='*100}\nRECALL_* shadow strats — TOP 14d performers\n{'='*100}")

q = (sb.table("paper_trades").select(
    "strategy,status,pnl_pct,pnl_usd,token_address,created_at,is_shadow,chain"
).eq("source", "rt").gte("created_at", SINCE_14D).like("strategy", "RECALL_%"))
recall = fetch_paged(q)
recall = [r for r in recall if r.get("status") in CLOSED]

by_strat = defaultdict(list)
for r in recall:
    by_strat[r["strategy"]].append(r)

print(f"\nTotal RECALL trades 14d: {len(recall)} on {len(by_strat)} strategies")
print(f"\n  {'strategy':<42}{'N':>4}{'WR':>6}{'avg%':>9}{'med%':>9}{'$tot':>10}{'$/d':>9}")
ranked = sorted(by_strat.items(), key=lambda kv: -sum(float(x.get("pnl_usd") or 0) for x in kv[1]))
for strat, rs in ranked[:25]:
    if len(rs) < 5:
        continue
    pcts = [float(r.get("pnl_pct") or 0)*100 for r in rs]
    usd = sum(float(r.get("pnl_usd") or 0) for r in rs)
    wr = 100*sum(1 for p in pcts if p > 0)/len(rs)
    print(f"  {strat:<40}{len(rs):>4}{wr:>5.0f}%{st.mean(pcts):>+8.2f}%{st.median(pcts):>+8.2f}%{usd:>+9.2f}{usd/14:>+8.2f}")

# Bottom (negative)
print(f"\nBottom 10 RECALL_* (worst 14d):")
print(f"  {'strategy':<42}{'N':>4}{'WR':>6}{'avg%':>9}{'$tot':>10}")
for strat, rs in ranked[-10:]:
    if len(rs) < 5:
        continue
    pcts = [float(r.get("pnl_pct") or 0)*100 for r in rs]
    usd = sum(float(r.get("pnl_usd") or 0) for r in rs)
    wr = 100*sum(1 for p in pcts if p > 0)/len(rs)
    print(f"  {strat:<40}{len(rs):>4}{wr:>5.0f}%{st.mean(pcts):>+8.2f}%{usd:>+9.2f}")
