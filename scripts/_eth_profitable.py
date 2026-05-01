"""ETH profitability audit 14d/30d - mirror du SOL audit pour identifier
les sweet spots ETH avant de cloner les age bands."""
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

def fetch_eth(since):
    out, off, step = [], 0, 1000
    while True:
        r = sb.table("paper_trades").select(
            "strategy,status,pnl_pct,pnl_usd,token_address,kol_group,created_at,is_shadow,chain,rt_token_age_hours"
        ).eq("source","rt").eq("chain","ethereum").gte("created_at", since).range(off, off+step-1).execute()
        if not r.data:
            break
        out.extend(r.data)
        if len(r.data) < step:
            break
        off += step
    return [r for r in out if r.get("status") in CLOSED]

# 14d ETH
all_eth = fetch_eth(SINCE_14D)
print("=" * 100)
print(f"ETH PROFITABILITY 14d — N={len(all_eth)} trades")
print("=" * 100)

# 1. CATEGORIES
cats = {
    "BASELINE_ETH":     lambda s: s.startswith("ETH_") and not s.startswith("ETH_RECALL") and not "BSR" in s and not "_KW" in s,
    "AGE24_ETH":        lambda s: s.startswith("AGE24_ETH"),
    "AGE48_ETH":        lambda s: s.startswith("AGE48_ETH"),
    "ETH_RECALL_DIP30": lambda s: s.startswith("ETH_RECALL_DIP30_"),
    "ETH_RECALL_DIP50": lambda s: s.startswith("ETH_RECALL_DIP50_"),
    "ETH_RECALL_PEAK":  lambda s: s.startswith("ETH_RECALL_PEAK"),
    "ETH_BSR":          lambda s: "BSR" in s and s.startswith("ETH"),
    "ETH_KW26":         lambda s: "_KW26" in s,
}

print(f"\n1. PERF par CATEGORIE ETH 14d")
print(f"  {'category':<22}{'N':>7}{'WR':>7}{'avg%':>9}{'med%':>9}{'$tot':>10}{'$/d':>9}  n_strats")
for cat_name, cat_fn in cats.items():
    rows = [r for r in all_eth if cat_fn(r["strategy"])]
    if not rows:
        print(f"  {cat_name:<22}{0:>7}")
        continue
    pcts = [float(r.get("pnl_pct") or 0)*100 for r in rows]
    usd = sum(float(r.get("pnl_usd") or 0) for r in rows)
    wr = 100*sum(1 for p in pcts if p > 0)/len(rows)
    n_strats = len(set(r["strategy"] for r in rows))
    print(f"  {cat_name:<22}{len(rows):>7}{wr:>6.0f}%{st.mean(pcts):>+8.2f}%{st.median(pcts):>+8.2f}%{usd:>+9.2f}{usd/14:>+8.2f}  ({n_strats})")

# 2. AGE distribution
print(f"\n2. ETH 1ER cluster par AGE de TOKEN (rt_token_age_hours)")
age_buckets = [(0, 1), (1, 3), (3, 6), (6, 12), (12, 24), (24, 48), (48, 168)]
print(f"  {'age':<14}{'N':>5}{'WR':>7}{'avg%':>9}{'med%':>9}{'$tot':>10}")
for lo, hi in age_buckets:
    bucket = [r for r in all_eth
              if not r["strategy"].startswith("AGE")
              and not r["strategy"].startswith("ETH_RECALL")
              and r.get("rt_token_age_hours") is not None
              and lo <= float(r["rt_token_age_hours"]) < hi]
    if not bucket:
        continue
    pcts = [float(r.get("pnl_pct") or 0)*100 for r in bucket]
    usd = sum(float(r.get("pnl_usd") or 0) for r in bucket)
    wr = 100*sum(1 for p in pcts if p > 0)/len(bucket)
    print(f"  [{lo:>3}-{hi:>3}h[ : {len(bucket):>5}{wr:>6.0f}%{st.mean(pcts):>+8.2f}%{st.median(pcts):>+8.2f}%{usd:>+9.2f}")

# 3. Top ETH strats 14d
print(f"\n3. TOP 25 ETH strats 14d (par $tot)")
by_strat = defaultdict(list)
for r in all_eth:
    by_strat[r["strategy"]].append(r)
ranked = []
for s, rs in by_strat.items():
    if len(rs) < 3:
        continue
    pcts = [float(r.get("pnl_pct") or 0)*100 for r in rs]
    usd = sum(float(r.get("pnl_usd") or 0) for r in rs)
    wr = 100*sum(1 for p in pcts if p > 0)/len(rs)
    ranked.append((s, len(rs), wr, st.mean(pcts), st.median(pcts), usd, usd/14))
ranked.sort(key=lambda x: -x[5])
print(f"  {'strategy':<48}{'N':>4}{'WR':>5}{'avg%':>9}{'med%':>9}{'$tot':>10}{'$/d':>8}")
for s, n, wr, avg, med, usd, daily in ranked[:25]:
    print(f"  {s:<48}{n:>4}{wr:>4.0f}%{avg:>+8.2f}%{med:>+8.2f}%{usd:>+9.2f}{daily:>+7.2f}")

# 4. RETRADES decompose
print(f"\n4. ETH RETRADES (>=2eme cluster) par GAP")
def parse_dt(s):
    return datetime.fromisoformat(s.replace("Z","+00:00"))
by_token = defaultdict(list)
for r in all_eth:
    by_token[r["token_address"]].append(r)
for ca in by_token:
    by_token[ca].sort(key=lambda r: r["created_at"])

clusters_per_token = {}
for ca, ts in by_token.items():
    if not ts: continue
    clusters = [[ts[0]]]
    for i in range(1, len(ts)):
        prev_t = parse_dt(clusters[-1][-1]["created_at"])
        curr_t = parse_dt(ts[i]["created_at"])
        if (curr_t - prev_t).total_seconds() / 60 >= 30:
            clusters.append([ts[i]])
        else:
            clusters[-1].append(ts[i])
    clusters_per_token[ca] = clusters

n_multi = sum(1 for cs in clusters_per_token.values() if len(cs) > 1)
print(f"  Tokens uniques ETH: {len(clusters_per_token)}, with >=2 clusters: {n_multi}")

retrade_by_gap = defaultdict(list)
buckets = [(0, 2), (2, 6), (6, 12), (12, 24), (24, 48), (48, 168)]
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

if any(retrade_by_gap.values()):
    print(f"  {'gap':<14}{'N':>5}{'WR':>7}{'avg%':>9}{'med%':>9}{'$tot':>10}")
    for lo, hi in buckets:
        rs = retrade_by_gap.get((lo, hi), [])
        if not rs: continue
        pcts = [float(r.get("pnl_pct") or 0)*100 for r in rs]
        usd = sum(float(r.get("pnl_usd") or 0) for r in rs)
        wr = 100*sum(1 for p in pcts if p > 0)/len(rs)
        print(f"  [{lo:>3}-{hi:>3}h[ : {len(rs):>5}{wr:>6.0f}%{st.mean(pcts):>+8.2f}%{st.median(pcts):>+8.2f}%{usd:>+9.2f}")
else:
    print("  AUCUN retrade ETH 14d (tokens call une seule fois)")

# 5. AGE-clones existing perf
print(f"\n5. ETH AGE clones (existing AGE24_ETH/AGE48_ETH) - perf par strat")
age_strats = [s for s in by_strat.keys() if s.startswith("AGE24_ETH") or s.startswith("AGE48_ETH")]
print(f"  {'strat':<48}{'N':>4}{'WR':>5}{'avg%':>9}{'$tot':>10}{'$/d':>8}")
for s in sorted(age_strats):
    rs = by_strat[s]
    if len(rs) < 3:
        continue
    pcts = [float(r.get("pnl_pct") or 0)*100 for r in rs]
    usd = sum(float(r.get("pnl_usd") or 0) for r in rs)
    wr = 100*sum(1 for p in pcts if p > 0)/len(rs)
    print(f"  {s:<48}{len(rs):>4}{wr:>4.0f}%{st.mean(pcts):>+8.2f}%{usd:>+9.2f}{usd/14:>+7.2f}")
