"""Drift live <-> paper paired-test, segmente par TYPE D'EXIT et par MECANIQUE.
Repond a: lock5 cree-t-il plus de drift que lock10/lock15/be/tp/sl ?"""
import os, sys, statistics as st
from collections import defaultdict
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

NOW = datetime.now(timezone.utc)
LIVE_DEPLOY = "2026-04-29T21:10:00+00:00"
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
            "strategy,status,source,pnl_pct,pnl_usd,token_address,created_at,is_shadow,chain"
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

# === Build all pairs across all 5 strats ===
all_pairs = []
for strat in LIVE_STRATS:
    live = [r for r in fetch("rt_live", LIVE_DEPLOY, strat) if not r.get("is_shadow")]
    paper = [r for r in fetch("rt", LIVE_DEPLOY, strat) if not r.get("is_shadow")]
    all_pairs.extend(pair(live, paper, window_min=60))

print("=" * 100)
print(f"DRIFT BY EXIT TYPE — N={len(all_pairs)} pairs since {LIVE_DEPLOY[:19]} UTC")
print("=" * 100)

# ====================================================================
# 1. Drift par STATUS DU LIVE EXIT
# ====================================================================
print(f"\n--- 1. DRIFT par status d'exit LIVE (qu'est-ce qui a ferme le trade live) ---\n")
print(f"  {'live_status':<14}{'pairs':>7}{'live%':>9}{'paper%':>9}{'drift_avg':>11}{'drift_med':>11}")
by_live_status = defaultdict(list)
for l, p in all_pairs:
    by_live_status[l["status"]].append((l, p))
for status, pairs_ in sorted(by_live_status.items(), key=lambda kv: -len(kv[1])):
    diffs = [(float(l.get("pnl_pct") or 0)*100) - (float(p.get("pnl_pct") or 0)*100) for l, p in pairs_]
    live_pct = st.mean([float(l.get("pnl_pct") or 0)*100 for l, _ in pairs_])
    paper_pct = st.mean([float(p.get("pnl_pct") or 0)*100 for _, p in pairs_])
    print(f"  {status:<14}{len(pairs_):>7}{live_pct:>+8.2f}%{paper_pct:>+8.2f}%{st.mean(diffs):>+10.2f}{st.median(diffs):>+10.2f}")

# ====================================================================
# 2. Drift par DIVERGENCE de status (live vs paper sur meme token)
# ====================================================================
print(f"\n--- 2. STATUS divergence (live ferme avec X, paper ferme avec Y) ---\n")
print(f"  {'live_st':<13}{'paper_st':<13}{'pairs':>7}{'live%':>9}{'paper%':>9}{'drift_avg':>11}")
combos = defaultdict(list)
for l, p in all_pairs:
    combos[(l["status"], p["status"])].append((l, p))
for (ls, ps), pairs_ in sorted(combos.items(), key=lambda kv: -len(kv[1])):
    if len(pairs_) < 5:
        continue
    diffs = [(float(l.get("pnl_pct") or 0)*100) - (float(p.get("pnl_pct") or 0)*100) for l, p in pairs_]
    live_pct = st.mean([float(l.get("pnl_pct") or 0)*100 for l, _ in pairs_])
    paper_pct = st.mean([float(p.get("pnl_pct") or 0)*100 for _, p in pairs_])
    same = "  SAME" if ls == ps else "DIFF"
    print(f"  {ls:<13}{ps:<13}{len(pairs_):>7}{live_pct:>+8.2f}%{paper_pct:>+8.2f}%{st.mean(diffs):>+10.2f}  {same}")

# ====================================================================
# 3. Drift par MECANIQUE (LOCK5 / LOCK10 / BE / TP / SL / FAST)
# ====================================================================
def classify(strat):
    if "LOCK5" in strat:
        return "LOCK5"
    if "LOCK10" in strat:
        return "LOCK10"
    if "LOCK15" in strat or "LOCK20" in strat:
        return "LOCK15+"
    if strat.startswith("BE25") or strat.startswith("BE15"):
        return "BE_only"
    if strat.startswith("FAST"):
        return "FAST"
    return "OTHER"

print(f"\n--- 3. DRIFT par MECANIQUE de strat (LOCK5 vs LOCK10 vs BE vs FAST) ---\n")
print(f"  {'mechanic':<14}{'pairs':>7}{'live%':>9}{'paper%':>9}{'drift_avg':>11}{'drift_med':>11}{'WR_live':>9}{'WR_paper':>10}")
by_mech = defaultdict(list)
for l, p in all_pairs:
    by_mech[classify(l["strategy"])].append((l, p))
for mech, pairs_ in sorted(by_mech.items(), key=lambda kv: -len(kv[1])):
    diffs = [(float(l.get("pnl_pct") or 0)*100) - (float(p.get("pnl_pct") or 0)*100) for l, p in pairs_]
    live_pcts = [float(l.get("pnl_pct") or 0)*100 for l, _ in pairs_]
    paper_pcts = [float(p.get("pnl_pct") or 0)*100 for _, p in pairs_]
    wr_l = 100*sum(1 for x in live_pcts if x > 0)/len(pairs_)
    wr_p = 100*sum(1 for x in paper_pcts if x > 0)/len(pairs_)
    print(f"  {mech:<14}{len(pairs_):>7}{st.mean(live_pcts):>+8.2f}%{st.mean(paper_pcts):>+8.2f}%{st.mean(diffs):>+10.2f}{st.median(diffs):>+10.2f}{wr_l:>+8.0f}%{wr_p:>+9.0f}%")

# ====================================================================
# 4. Comparison LOCK5 vs LOCK10 vs BE_only spécifiquement
# ====================================================================
print(f"\n--- 4. LOCK5 vs LOCK10 — meme token, meme moment ---\n")
print(f"  Comparing the 3 BE strats (BE25_TP80_SL30, BE15_LOCK5, BE25_LOCK10_NZ_S40)\n  on the SAME tokens at the same time.\n")

# Find tokens where ALL 3 BE strats fired in same window
by_token = defaultdict(dict)
for l, p in all_pairs:
    mech = classify(l["strategy"])
    if mech in ("BE_only", "LOCK5", "LOCK10"):
        # Use created_at floor to 30min bucket
        bucket = l["created_at"][:13]  # hour bucket
        by_token[(l["token_address"], bucket)][mech] = (l, p)

triplets = [v for v in by_token.values() if {"BE_only", "LOCK5", "LOCK10"}.issubset(v.keys())]
print(f"  {len(triplets)} tokens where all 3 BE-family strats fired in same hour")

if triplets:
    print(f"  {'mechanic':<10}{'live_avg%':>11}{'paper_avg%':>12}{'drift_avg':>11}{'drift_med':>11}")
    for mech in ("BE_only", "LOCK5", "LOCK10"):
        live_p = [float(t[mech][0].get("pnl_pct") or 0)*100 for t in triplets]
        paper_p = [float(t[mech][1].get("pnl_pct") or 0)*100 for t in triplets]
        diffs = [lp - pp for lp, pp in zip(live_p, paper_p)]
        print(f"  {mech:<10}{st.mean(live_p):>+10.2f}%{st.mean(paper_p):>+11.2f}%{st.mean(diffs):>+10.2f}{st.median(diffs):>+10.2f}")

# ====================================================================
# 5. Drift par SIGN — drift only on LOSSES, drift only on WINS
# ====================================================================
print(f"\n--- 5. Drift conditionne au PnL (loss vs win) ---\n")
print(f"  Question: le drift est-il symetrique ou apparait-il que sur les pertes ?\n")
losses = [(l, p) for l, p in all_pairs if float(l.get("pnl_pct") or 0)*100 < 0]
wins = [(l, p) for l, p in all_pairs if float(l.get("pnl_pct") or 0)*100 >= 0]
breakeven = [(l, p) for l, p in all_pairs if abs(float(l.get("pnl_pct") or 0)*100) < 0.1]

print(f"  {'subset':<14}{'pairs':>7}{'live%':>9}{'paper%':>9}{'drift_avg':>11}{'drift_med':>11}")
for label, pairs_ in (("ALL", all_pairs), ("LIVE_LOSS", losses), ("LIVE_WIN", wins), ("LIVE_BE0", breakeven)):
    if not pairs_:
        continue
    diffs = [(float(l.get("pnl_pct") or 0)*100) - (float(p.get("pnl_pct") or 0)*100) for l, p in pairs_]
    live_pct = st.mean([float(l.get("pnl_pct") or 0)*100 for l, _ in pairs_])
    paper_pct = st.mean([float(p.get("pnl_pct") or 0)*100 for _, p in pairs_])
    print(f"  {label:<14}{len(pairs_):>7}{live_pct:>+8.2f}%{paper_pct:>+8.2f}%{st.mean(diffs):>+10.2f}{st.median(diffs):>+10.2f}")
