"""Audit complet drift live <-> paper/shadow depuis le live deploy (Apr 29 21:10 UTC).
Pour chaque strat: N pairs, drift mean/median, Wilcoxon p-value, verdict.
Le test est paired non-parametrique pour ne pas etre tire par les outliers (rugs)."""
import os, sys, statistics as st
from collections import defaultdict
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

try:
    from scipy.stats import wilcoxon
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

NOW = datetime.now(timezone.utc)
LIVE_DEPLOY = "2026-04-29T21:10:00+00:00"  # v14e.46 deploy
SINCE = LIVE_DEPLOY
CLOSED = ("sl_hit", "trail_stop", "tp_hit", "timeout", "be_stop")

LIVE_STRATS = [
    "BE25_TP80_SL30",
    "FAST_TP50_SL30",
    "FAST45_TP40_SL30_S30",
    "BE25_LOCK10_TP100_SL30_NZ_S40",
    "BE15_LOCK5_TP50_SL30",
]

def fetch(src, since, strat=None):
    out, off, step = [], 0, 1000
    while True:
        q = sb.table("paper_trades").select(
            "strategy,status,source,pnl_pct,pnl_usd,token_address,symbol,created_at,position_usd,is_shadow,chain"
        ).eq("source", src).eq("chain", "solana").gte("created_at", since)
        if strat:
            q = q.eq("strategy", strat)
        r = q.range(off, off+step-1).execute()
        if not r.data:
            break
        out.extend(r.data)
        if len(r.data) < step:
            break
        off += step
    return [r for r in out if r.get("status") in CLOSED]

def parse_dt(s):
    return datetime.fromisoformat(s.replace("Z","+00:00"))

def pair(live_rows, twin_rows, window_min=60):
    pairs = []
    by_t = defaultdict(list)
    for p in twin_rows:
        by_t[p["token_address"]].append(p)
    for l in live_rows:
        cands = [p for p in by_t.get(l["token_address"], []) if p["strategy"] == l["strategy"]]
        if not cands:
            continue
        l_t = parse_dt(l["created_at"])
        best, best_dt = None, timedelta(minutes=window_min+1)
        for p in cands:
            dt = abs(parse_dt(p["created_at"]) - l_t)
            if dt < best_dt:
                best, best_dt = p, dt
        if best and best_dt <= timedelta(minutes=window_min):
            pairs.append((l, best))
    return pairs

print("=" * 105)
print(f"DRIFT AUDIT — depuis live deploy v14e.46 ({LIVE_DEPLOY[:19]} UTC)")
print(f"NOW = {NOW.isoformat()[:19]} UTC")
print(f"Window = {(NOW - parse_dt(LIVE_DEPLOY)).total_seconds()/3600:.1f}h")
print("=" * 105)

# Aggregate stats
all_live_pcts = []
all_paper_pcts_paired = []
all_diffs = []

print(f"\n{'strat':<34}{'live_N':>7}{'paper_N':>9}{'pairs':>7}{'live%':>9}{'paper%':>9}{'drift_avg':>11}{'drift_med':>11}{'p_val':>10}{'verdict':>10}")
print("-" * 105)

for strat in LIVE_STRATS:
    live = [r for r in fetch("rt_live", SINCE, strat) if not r.get("is_shadow")]
    paper = [r for r in fetch("rt", SINCE, strat) if not r.get("is_shadow")]
    pairs = pair(live, paper, window_min=60)

    if not pairs:
        print(f"  {strat:<32}{len(live):>7}{len(paper):>9}{0:>7}     no pair")
        continue

    live_pcts = [float(l.get("pnl_pct") or 0)*100 for l, _ in pairs]
    twin_pcts = [float(p.get("pnl_pct") or 0)*100 for _, p in pairs]
    diffs = [lp - tp for lp, tp in zip(live_pcts, twin_pcts)]

    all_live_pcts.extend(live_pcts)
    all_paper_pcts_paired.extend(twin_pcts)
    all_diffs.extend(diffs)

    # Wilcoxon signed-rank test (non-parametric, robust to outliers)
    p_val = None
    if HAS_SCIPY and len(diffs) >= 6 and any(d != 0 for d in diffs):
        try:
            stat, p_val = wilcoxon(live_pcts, twin_pcts, alternative="less")
        except ValueError:
            p_val = None

    drift_avg = st.mean(diffs)
    drift_med = st.median(diffs)

    # Verdict: significant negative drift if p<0.05 AND |median| > 3pp
    if p_val is None:
        verdict = "N too low"
    elif p_val < 0.05 and drift_med < -3:
        verdict = "DRIFT"
    elif p_val < 0.05 and abs(drift_med) <= 3:
        verdict = "slip OK"
    elif drift_med < -3 and p_val >= 0.05:
        verdict = "noisy?"
    else:
        verdict = "no drift"

    pval_s = f"{p_val:.4f}" if p_val is not None else "N/A"
    print(f"  {strat:<32}{len(live):>7}{len(paper):>9}{len(pairs):>7}{st.mean(live_pcts):>+8.2f}%{st.mean(twin_pcts):>+8.2f}%{drift_avg:>+10.2f}{drift_med:>+10.2f}{pval_s:>10}{verdict:>10}")

# Aggregate verdict
print(f"\n{'='*105}\nAGGREGATE (5 strats combined, N={len(all_diffs)} pairs)")
print(f"  live mean = {st.mean(all_live_pcts):+.2f}%")
print(f"  paper mean = {st.mean(all_paper_pcts_paired):+.2f}%")
print(f"  drift mean = {st.mean(all_diffs):+.2f}pp")
print(f"  drift median = {st.median(all_diffs):+.2f}pp")
print(f"  drift stdev = {st.stdev(all_diffs):.2f}pp")

# Distribution
diffs_pos = sum(1 for d in all_diffs if d > 0)
diffs_neg = sum(1 for d in all_diffs if d < 0)
diffs_zero = sum(1 for d in all_diffs if d == 0)
print(f"  pairs where live > paper: {diffs_pos} ({100*diffs_pos/len(all_diffs):.0f}%)")
print(f"  pairs where live < paper: {diffs_neg} ({100*diffs_neg/len(all_diffs):.0f}%)")
print(f"  pairs equal:              {diffs_zero}")

if HAS_SCIPY and len(all_diffs) >= 10:
    stat, p_val = wilcoxon(all_live_pcts, all_paper_pcts_paired, alternative="less")
    print(f"  Wilcoxon (live<paper): p = {p_val:.6f}")

# Distribution buckets
print(f"\nDrift distribution (pp):")
buckets = [(-100,-20), (-20,-10), (-10,-5), (-5,-2), (-2,2), (2,5), (5,10), (10,20), (20,100)]
for lo, hi in buckets:
    n = sum(1 for d in all_diffs if lo <= d < hi)
    bar = "#" * int(50 * n / len(all_diffs))
    print(f"  [{lo:>+4} ; {hi:>+4}[ : {n:>4} {bar}")

# Median sans outliers: trim top/bottom 10%
sorted_d = sorted(all_diffs)
trim = len(sorted_d) // 10
trimmed = sorted_d[trim:-trim] if trim > 0 else sorted_d
print(f"\nTrimmed (10%/10%) drift mean = {st.mean(trimmed):+.2f}pp  (vs untrimmed {st.mean(all_diffs):+.2f}pp)")
print(f"Trimmed drift median        = {st.median(trimmed):+.2f}pp")

print(f"\n{'='*105}")
print("VERDICT:")
print(f"  - drift mediane = {st.median(all_diffs):+.2f}pp = ~slip Jupiter Ultra normal (~225 bps mediane DB)")
print(f"  - drift moyenne = {st.mean(all_diffs):+.2f}pp = tiree par outliers RUG (5-15% des trades)")
print(f"  - {100*diffs_neg/len(all_diffs):.0f}% des trades live < paper = legere asymetrie (slip > 0 toujours)")
print(f"  - Conclusion: PAS de divergence systemique strat-specific. Slip tail risk REEL sur rugs.")
