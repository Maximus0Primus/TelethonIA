"""Paired-test every v144 shadow vs its base on intersection tokens.

Reports per (variant, base) :
  - pair_N : tokens where both traded (last trade per token if duplicates)
  - median dpp = pnl_pct_variant − pnl_pct_base
  - mean dpp
  - sum d$
  - wins V/B/T
  - verdict : VARIANT WIN | TIE | BASE WIN | PREMATURE (pair_N<10)

Then prints A/B GAPS : v144 shadow suffixes that exist for some bases but not others.
"""
import os, sys, statistics as st
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

NOW = datetime.now(timezone.utc)
SINCE = (NOW - timedelta(days=14)).isoformat()

# Active mains = bases candidates
MAINS = [
    "FAST_TP40_SL30","FAST_TP80_SL25","FAST_TP50_SL30","BE25_TP80_SL30",
    "TP50_SL15","FAST_TP100_SL20","BE25_TP80_SL30_S30_HYST","FAST_TP50_SL30_HYST",
    "BE15_TP70_SL50_NZ","BE25_TP80_SL30_NZS30_HYST","HIGHSCORE_TP200_SL40","NOZEROLIQ_TP200_SL40",
]

# v144 suffix dimensions to test
SUFFIXES = ["_NOLAZY","_LAZYFAST","_LAZYMED","_LAZYSLOW","_LAZYXSLOW",
            "_BOTH","_JUPITER","_DS","_MED3","_S30","_S40","_MCAP_S40","_COMBO"]

# Get all distinct strategy names that exist in DB
print("Loading strategy universe...")
all_strats = set()
off = 0
while True:
    r = sb.table("paper_trades").select("strategy").eq("source","rt").gte("created_at", SINCE).range(off, off+999).execute().data
    for x in r: all_strats.add(x["strategy"])
    if len(r) < 1000: break
    off += 1000
print(f"  {len(all_strats)} distinct strategies in last 14d\n")

def fetch_dedup_per_token(strat):
    out = {}
    off = 0
    while True:
        r = sb.table("paper_trades").select(
            "token_address,pnl_pct,pnl_usd,created_at"
        ).eq("source","rt").eq("strategy",strat).gte("created_at",SINCE).neq("status","open").neq("status","closing").range(off, off+999).execute().data
        for row in r:
            tok = row["token_address"]
            if tok not in out or row["created_at"] > out[tok]["created_at"]:
                out[tok] = row
        if len(r) < 1000: break
        off += 1000
    return out

# Per-base × suffix matrix
results = []
gaps = []
print(f"{'Base':<25} {'Suffix':<12} {'pair_N':>7} {'med_dpp':>9} {'sum_d$':>9} {'V/B/T':>10} {'Verdict':>14}")
print("-" * 95)
for base in MAINS:
    base_idx = fetch_dedup_per_token(base)
    if not base_idx:
        continue
    for suf in SUFFIXES:
        v_name = base + suf
        if v_name not in all_strats:
            gaps.append((base, suf))
            continue
        v_idx = fetch_dedup_per_token(v_name)
        common = set(base_idx) & set(v_idx)
        n = len(common)
        if n == 0:
            print(f"{base:<25} {suf:<12} {n:>7} {'—':>9} {'—':>9} {'—':>10} {'NO PAIR':>14}")
            results.append((base, suf, 0, None, None, None, "NO PAIR"))
            continue
        deltas_pp = [(v_idx[t]["pnl_pct"] or 0)*100 - (base_idx[t]["pnl_pct"] or 0)*100 for t in common]
        deltas_usd = [(v_idx[t]["pnl_usd"] or 0) - (base_idx[t]["pnl_usd"] or 0) for t in common]
        med_pp = st.median(deltas_pp)
        sum_d = sum(deltas_usd)
        wins_v = sum(1 for d in deltas_pp if d > 0.5)
        wins_b = sum(1 for d in deltas_pp if d < -0.5)
        ties = n - wins_v - wins_b
        if n < 10:
            verdict = "PREMATURE"
        elif med_pp > 1.0:
            verdict = "VARIANT WIN"
        elif med_pp < -1.0:
            verdict = "BASE WIN"
        else:
            verdict = "TIE"
        results.append((base, suf, n, med_pp, sum_d, (wins_v,wins_b,ties), verdict))
        print(f"{base:<25} {suf:<12} {n:>7} {med_pp:>+8.2f} {sum_d:>+8.2f} {wins_v:>3}/{wins_b}/{ties:<3} {verdict:>14}")

print()
print("="*95)
print("GAPS — v144 dimensions NOT tested for these bases:")
print("="*95)
gap_by_base = defaultdict(list)
for b,s in gaps:
    gap_by_base[b].append(s)
for b in MAINS:
    if b in gap_by_base:
        print(f"  {b:<30} missing: {', '.join(gap_by_base[b])}")

# Group findings
print()
print("="*95)
print("WINNERS (VARIANT WIN, pair_N>=10):")
print("="*95)
for base, suf, n, med, sd, wbt, v in results:
    if v == "VARIANT WIN":
        print(f"  {base+suf:<40} pair_N={n} med_dpp={med:+.2f} sum_d$={sd:+.2f} V/B/T={wbt[0]}/{wbt[1]}/{wbt[2]}")

print()
print("LOSERS (BASE WIN, pair_N>=10) — candidates for shadow removal:")
for base, suf, n, med, sd, wbt, v in results:
    if v == "BASE WIN":
        print(f"  {base+suf:<40} pair_N={n} med_dpp={med:+.2f} sum_d$={sd:+.2f} V/B/T={wbt[0]}/{wbt[1]}/{wbt[2]}")

print()
print("PREMATURE (need more data):")
for base, suf, n, med, sd, wbt, v in results:
    if v == "PREMATURE":
        print(f"  {base+suf:<40} pair_N={n}/10")
