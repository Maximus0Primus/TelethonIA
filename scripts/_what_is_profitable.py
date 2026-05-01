"""Audit: qu'est-ce qui est VRAIMENT rentable ?
1. Compare strats par BUCKET (RECALL vs AGE vs baseline) sur shadow 14d/30d
2. Decompose retrades par age du token au moment du 2eme call
3. Identifie quelles nouvelles strats valent la peine d'ajouter
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

def fetch(strat_filter=None, since=SINCE_14D, source="rt"):
    out, off, step = [], 0, 1000
    while True:
        q = sb.table("paper_trades").select(
            "strategy,status,pnl_pct,pnl_usd,token_address,kol_group,created_at,is_shadow,chain,rt_token_age_hours"
        ).eq("source", source).eq("chain", "solana").gte("created_at", since)
        if strat_filter:
            q = q.like("strategy", strat_filter)
        r = q.range(off, off+step-1).execute()
        if not r.data:
            break
        out.extend(r.data)
        if len(r.data) < step:
            break
        off += step
    return [r for r in out if r.get("status") in CLOSED]

# ============================================================
# 1. PERFORMANCE par CATEGORIE de strat (14d shadow)
# ============================================================
print("=" * 100)
print("1. PERF par CATEGORIE shadow 14d")
print("=" * 100)

categories = {
    "BASELINE_FAST_BE":    lambda s: (s.startswith("FAST_") or s.startswith("BE25_") or s.startswith("BE15_") or s.startswith("BE30_") or s.startswith("BE50_")) and not s.startswith("AGE") and not s.startswith("RECALL_"),
    "AGE24":               lambda s: s.startswith("AGE24_"),
    "AGE48":               lambda s: s.startswith("AGE48_"),
    "AGE72":               lambda s: s.startswith("AGE72_"),
    "RECALL_DIP30":        lambda s: s.startswith("RECALL_DIP30_") and "AGE" not in s,
    "RECALL_DIP50":        lambda s: s.startswith("RECALL_DIP50_"),
    "RECALL_DIP30_AGE":    lambda s: s.startswith("RECALL_DIP30_AGE"),
    "RECALL_PEAK30":       lambda s: s.startswith("RECALL_PEAK30_"),
    "RECALL_PEAK50":       lambda s: s.startswith("RECALL_PEAK50_"),
    "RECALL_PEAK70":       lambda s: s.startswith("RECALL_PEAK70_"),
    "SCALP":               lambda s: s.startswith("SCALP_") and not s.startswith("RECALL"),
    "SLOW":                lambda s: s.startswith("SLOW") and not s.startswith("RECALL"),
}

# Pull all 14d
all_rows = []
for off in range(0, 200000, 1000):
    r = sb.table("paper_trades").select(
        "strategy,status,pnl_pct,pnl_usd,token_address,kol_group,created_at,is_shadow,chain"
    ).eq("source","rt").eq("chain","solana").gte("created_at", SINCE_14D).range(off, off+999).execute()
    if not r.data:
        break
    all_rows.extend(r.data)
    if len(r.data) < 1000:
        break
all_rows = [r for r in all_rows if r.get("status") in CLOSED]
print(f"\nTotal SOL trades 14d: {len(all_rows)}")

print(f"\n  {'category':<24}{'N':>7}{'WR':>7}{'avg%':>9}{'med%':>9}{'$tot':>10}{'$/d':>9}")
for cat_name, cat_fn in categories.items():
    rows = [r for r in all_rows if cat_fn(r["strategy"])]
    if not rows:
        print(f"  {cat_name:<24}{0:>7}")
        continue
    pcts = [float(r.get("pnl_pct") or 0)*100 for r in rows]
    usd = sum(float(r.get("pnl_usd") or 0) for r in rows)
    wr = 100*sum(1 for p in pcts if p > 0)/len(rows)
    n_strats = len(set(r["strategy"] for r in rows))
    print(f"  {cat_name:<24}{len(rows):>7}{wr:>6.0f}%{st.mean(pcts):>+8.2f}%{st.median(pcts):>+8.2f}%{usd:>+9.2f}{usd/14:>+8.2f}  ({n_strats} strats)")

# ============================================================
# 2. SHADOW vs PAPER MAIN comparison sur la meme strat
# ============================================================
print(f"\n{'='*100}\n2. RECALL & AGE — top performers detail (N>=10 shadow OR is_shadow=True)\n{'='*100}")
strat_perf = defaultdict(list)
for r in all_rows:
    strat_perf[r["strategy"]].append(r)

# RECALL_* + AGE_*
ranked = []
for strat, rows in strat_perf.items():
    if not (strat.startswith("RECALL_") or strat.startswith("AGE")):
        continue
    if len(rows) < 5:
        continue
    pcts = [float(r.get("pnl_pct") or 0)*100 for r in rows]
    usd = sum(float(r.get("pnl_usd") or 0) for r in rows)
    wr = 100*sum(1 for p in pcts if p > 0)/len(rows)
    n_shadow = sum(1 for r in rows if r.get("is_shadow"))
    ranked.append((strat, len(rows), wr, st.mean(pcts), st.median(pcts), usd, usd/14, n_shadow))

ranked.sort(key=lambda x: -x[5])
print(f"\n  {'strategy':<46}{'N':>5}{'WR':>6}{'avg%':>8}{'med%':>8}{'$tot':>10}{'$/d':>8}  shadow_pct")
for strat, n, wr, avg, med, usd, daily, n_sh in ranked:
    sh_pct = 100*n_sh/n
    print(f"  {strat:<46}{n:>5}{wr:>5.0f}%{avg:>+7.2f}%{med:>+7.2f}%{usd:>+9.2f}{daily:>+7.2f}  {sh_pct:>3.0f}%")

# ============================================================
# 3. DECOMPOSE RETRADES par AGE du TOKEN au moment du 2eme call
# ============================================================
print(f"\n{'='*100}\n3. RETRADES — decompose par age du TOKEN au 2eme cluster\n{'='*100}")

def parse_dt(s):
    return datetime.fromisoformat(s.replace("Z","+00:00"))

# group by token
by_token = defaultdict(list)
for r in all_rows:
    by_token[r["token_address"]].append(r)
for ca in by_token:
    by_token[ca].sort(key=lambda r: r["created_at"])

# cluster by 30min gap
clusters_per_token = {}
for ca, ts in by_token.items():
    if not ts:
        continue
    clusters = [[ts[0]]]
    for i in range(1, len(ts)):
        prev_t = parse_dt(clusters[-1][-1]["created_at"])
        curr_t = parse_dt(ts[i]["created_at"])
        if (curr_t - prev_t).total_seconds() / 60 >= 30:
            clusters.append([ts[i]])
        else:
            clusters[-1].append(ts[i])
    clusters_per_token[ca] = clusters

# Pour chaque 2eme/3eme cluster, fetch token age via paper_trades.rt_token_age_hours du 1er trade
# (the field is set at trade open time)
print(f"\nPnL des retrades (>=2eme cluster) decompose par TIME GAP depuis 1er cluster:")
buckets = [(0, 2), (2, 6), (6, 12), (12, 24), (24, 48), (48, 168)]
print(f"  {'gap':<14}{'N':>5}{'WR':>7}{'avg%':>9}{'med%':>9}{'$tot':>10}")

retrade_by_gap = defaultdict(list)
for ca, clusters in clusters_per_token.items():
    if len(clusters) < 2:
        continue
    first_t = parse_dt(clusters[0][-1]["created_at"])
    for ci in range(1, len(clusters)):
        ci_t = parse_dt(clusters[ci][0]["created_at"])
        gap_h = (ci_t - first_t).total_seconds() / 3600
        for trade in clusters[ci]:
            for lo, hi in buckets:
                if lo <= gap_h < hi:
                    retrade_by_gap[(lo, hi)].append(trade)
                    break

for lo, hi in buckets:
    rs = retrade_by_gap.get((lo, hi), [])
    if not rs:
        continue
    pcts = [float(r.get("pnl_pct") or 0)*100 for r in rs]
    usd = sum(float(r.get("pnl_usd") or 0) for r in rs)
    wr = 100*sum(1 for p in pcts if p > 0)/len(rs)
    print(f"  [{lo:>3}-{hi:>3}h[ : {len(rs):>5}{wr:>6.0f}%{st.mean(pcts):>+8.2f}%{st.median(pcts):>+8.2f}%{usd:>+9.2f}")

# Et par MECANIQUE de strat sur les retrades
print(f"\nPnL retrades >=2eme cluster decompose par MECANIQUE (ALL gap):")
all_retrades = []
for cs in clusters_per_token.values():
    if len(cs) >= 2:
        for ci in range(1, len(cs)):
            all_retrades.extend(cs[ci])

mech_groups = defaultdict(list)
for r in all_retrades:
    s = r["strategy"]
    if s.startswith("RECALL_"):
        mech_groups["RECALL_*"].append(r)
    elif s.startswith("AGE24_"):
        mech_groups["AGE24_*"].append(r)
    elif s.startswith("AGE48_"):
        mech_groups["AGE48_*"].append(r)
    elif s.startswith("AGE72_"):
        mech_groups["AGE72_*"].append(r)
    elif "LOCK" in s:
        mech_groups["BE_LOCK_*"].append(r)
    elif s.startswith("FAST"):
        mech_groups["FAST_*"].append(r)
    elif s.startswith("BE"):
        mech_groups["BE_*"].append(r)
    elif s.startswith("SCALP"):
        mech_groups["SCALP_*"].append(r)
    elif s.startswith("SLOW"):
        mech_groups["SLOW_*"].append(r)
    else:
        mech_groups["OTHER"].append(r)

print(f"  {'mech':<14}{'N':>5}{'WR':>7}{'avg%':>9}{'med%':>9}{'$tot':>10}")
for mech, rs in sorted(mech_groups.items(), key=lambda kv: -sum(float(x.get("pnl_usd") or 0) for x in kv[1])):
    if len(rs) < 5:
        continue
    pcts = [float(r.get("pnl_pct") or 0)*100 for r in rs]
    usd = sum(float(r.get("pnl_usd") or 0) for r in rs)
    wr = 100*sum(1 for p in pcts if p > 0)/len(rs)
    print(f"  {mech:<14}{len(rs):>5}{wr:>6.0f}%{st.mean(pcts):>+8.2f}%{st.median(pcts):>+8.2f}%{usd:>+9.2f}")

# ============================================================
# 4. 1ST CLUSTERS by AGE band — does the AGE of FIRST CALL matter ?
# ============================================================
print(f"\n{'='*100}\n4. 1ER cluster decompose par AGE de TOKEN (rt_token_age_hours)\n{'='*100}")

# Need rt_token_age_hours field
all_with_age, off = [], 0
while True:
    r = sb.table("paper_trades").select(
        "strategy,status,pnl_pct,pnl_usd,token_address,created_at,is_shadow,rt_token_age_hours"
    ).eq("source","rt").eq("chain","solana").gte("created_at", SINCE_14D).range(off, off+999).execute()
    if not r.data:
        break
    all_with_age.extend(r.data)
    if len(r.data) < 1000:
        break
    off += 1000
all_with_age = [r for r in all_with_age if r.get("status") in CLOSED and r.get("rt_token_age_hours") is not None]
print(f"trades 14d with rt_token_age_hours: {len(all_with_age)}")

# Filter to baseline only (no AGE/RECALL), take 1st cluster trades
by_t2 = defaultdict(list)
for r in all_with_age:
    by_t2[r["token_address"]].append(r)
for ca in by_t2:
    by_t2[ca].sort(key=lambda r: r["created_at"])

age_buckets = [(0, 1), (1, 3), (3, 6), (6, 12), (12, 24), (24, 48), (48, 72), (72, 168)]
print(f"\nBASELINE (non-AGE non-RECALL) trades par age du token:")
print(f"  {'age':<14}{'N':>5}{'WR':>7}{'avg%':>9}{'med%':>9}{'$tot':>10}")
for lo, hi in age_buckets:
    bucket_trades = [r for r in all_with_age
                     if not r["strategy"].startswith("AGE")
                     and not r["strategy"].startswith("RECALL_")
                     and lo <= float(r["rt_token_age_hours"] or 0) < hi]
    if not bucket_trades:
        continue
    pcts = [float(r.get("pnl_pct") or 0)*100 for r in bucket_trades]
    usd = sum(float(r.get("pnl_usd") or 0) for r in bucket_trades)
    wr = 100*sum(1 for p in pcts if p > 0)/len(bucket_trades)
    print(f"  [{lo:>3}-{hi:>3}h[ : {len(bucket_trades):>5}{wr:>6.0f}%{st.mean(pcts):>+8.2f}%{st.median(pcts):>+8.2f}%{usd:>+9.2f}")
