"""Score V3 walk-forward — does rt_score_v3 rank trade outcomes better than v2/v1?

The scores are computed LIVE at entry (never fit on these rows), so the AUC of each
score vs the realized win/loss on the shadow window IS out-of-sample. Decision rule
(from backlog): if AUC(v3) >= AUC(v1) + 0.015 on a healthy N, swap the `min_rt_score`
filter to v3.

Faithful: rolling-24h dedup per (strategy, token). AUC via rank method (tie-safe).

Usage: python scripts/_score_v3_walk_forward.py [--chain solana] [--days 21] [--win-thresh 0.0]
"""
import argparse
import os
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from supabase import create_client

load_dotenv("scraper/.env")
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
EXIT = ("tp_hit", "sl_hit", "timeout", "trail_stop", "be_stop")

ap = argparse.ArgumentParser()
ap.add_argument("--chain", default="solana")
ap.add_argument("--days", type=int, default=21)
ap.add_argument("--win-thresh", type=float, default=0.0, help="pnl_pct above this = win")
args = ap.parse_args()
since = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()


def fetch_all():
    rows, off = [], 0
    cols = "strategy,token_address,pnl_pct,created_at,rt_score,rt_score_v2,rt_score_v3"
    while True:
        b = (sb.table("paper_trades").select(cols)
             .eq("chain", args.chain).eq("is_shadow", True).eq("source", "rt")
             .in_("status", list(EXIT)).gte("created_at", since)
             .order("created_at").range(off, off + 999).execute().data) or []
        rows += b
        if len(b) < 1000:
            return rows
        off += 1000


def dedup(rows):
    last_seen, kept = {}, []
    for r in sorted(rows, key=lambda r: r["created_at"]):
        ts = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
        key = (r["strategy"], r["token_address"])
        prev = last_seen.get(key)
        if prev and (ts - prev) < timedelta(hours=24):
            continue
        last_seen[key] = ts
        kept.append(r)
    return kept


def auc(scores_labels):
    """scores_labels: list of (score, label 0/1). Rank-based AUC, tie-safe."""
    data = [(s, y) for s, y in scores_labels if s is not None]
    n = len(data)
    pos = sum(y for _, y in data)
    neg = n - pos
    if pos == 0 or neg == 0:
        return None, n, pos
    data.sort(key=lambda x: x[0])
    # average ranks for ties
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and data[j][0] == data[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # ranks are 1-based: (i+1 .. j)
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    sum_pos_ranks = sum(ranks[k] for k in range(n) if data[k][1] == 1)
    a = (sum_pos_ranks - pos * (pos + 1) / 2.0) / (pos * neg)
    return a, n, pos


kept = dedup([r for r in fetch_all()])
print(f"=== Score V3 walk-forward [{args.chain}] {args.days}d | dedup N={len(kept)} | win=pnl>{args.win_thresh:.0%} ===\n")

results = {}
for col in ("rt_score", "rt_score_v2", "rt_score_v3"):
    sl = [(r[col], 1 if float(r["pnl_pct"]) > args.win_thresh else 0)
          for r in kept if r.get(col) is not None]
    a, n, pos = auc(sl)
    results[col] = a
    if a is None:
        print(f"{col:<14} AUC=n/a (N={n}, pos={pos})")
    else:
        print(f"{col:<14} AUC={a:.4f}  (N={n}, win_rate={100*pos/n:.0f}%)")

v1, v3 = results.get("rt_score"), results.get("rt_score_v3")
print()
if v1 is not None and v3 is not None:
    lift = v3 - v1
    print(f"V3 - V1 lift = {lift:+.4f}")
    if lift >= 0.015:
        print("VERDICT: SWAP — v3 ranks outcomes meaningfully better. Move min_rt_score filter to v3.")
    elif lift <= -0.015:
        print("VERDICT: KEEP v1 — v3 is worse.")
    else:
        print("VERDICT: INCONCLUSIVE — lift within +/-0.015 noise band. Keep v1, recheck with more N.")
else:
    print("VERDICT: insufficient data (one score column empty in window).")
