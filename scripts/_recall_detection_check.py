"""Pourquoi RECALL_* fire si peu ? Audit de l'is_recall flag rate + drift distribution."""
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

# Check: kol_call_outcomes is the source for recall detection (safe_scraper.py:1547)
print("=" * 100)
print("RECALL DETECTION audit — is the _rt_is_recall flag firing as expected ?")
print("=" * 100)

# 1. count outcomes records 14d
out, off = [], 0
while True:
    r = sb.table("kol_call_outcomes").select("kol_group,token_address,call_timestamp,entry_price,ath_after_call").gte("call_timestamp", SINCE_14D).range(off, off+999).execute()
    if not r.data:
        break
    out.extend(r.data)
    if len(r.data) < 1000:
        break
    off += 1000
print(f"\nkol_call_outcomes 14d: N={len(out)}")

# 2. Tokens tradés 14d en paper main
out2, off = [], 0
while True:
    r = sb.table("paper_trades").select("token_address,strategy,kol_group,created_at").eq("source","rt").eq("chain","solana").gte("created_at", SINCE_14D).range(off, off+999).execute()
    if not r.data:
        break
    out2.extend(r.data)
    if len(r.data) < 1000:
        break
    off += 1000
unique_tokens = set(r["token_address"] for r in out2)
recall_strats_fired = sum(1 for r in out2 if r["strategy"].startswith("RECALL_"))
print(f"paper main rt 14d: N={len(out2)} trades on {len(unique_tokens)} unique tokens")
print(f"  RECALL_* strat fires: {recall_strats_fired}")

# 3. For each unique token, was there a prior call in kol_call_outcomes ?
outcomes_by_token = defaultdict(list)
for o in out:
    outcomes_by_token[o["token_address"]].append(o)

tokens_with_prior_outcome = sum(1 for ca in unique_tokens if ca in outcomes_by_token)
print(f"\ntokens (paper main 14d) with a kol_call_outcomes record: {tokens_with_prior_outcome}/{len(unique_tokens)}")

# 4. Pour chaque token paper main, est-ce qu'il y a un PRIOR call (avant le 1er trade) ?
#    Le check recall demande first_call >= 30min ago (safe_scraper.py:1547)
import collections
prior_call_count = 0
no_prior_count = 0
prior_drifts = []
for ca in unique_tokens:
    trades_for_ca = [r for r in out2 if r["token_address"] == ca]
    if not trades_for_ca:
        continue
    trades_for_ca.sort(key=lambda r: r["created_at"])
    first_trade_t = datetime.fromisoformat(trades_for_ca[0]["created_at"].replace("Z","+00:00"))
    outcomes = outcomes_by_token.get(ca, [])
    prior_outcomes = [o for o in outcomes if datetime.fromisoformat(o["call_timestamp"].replace("Z","+00:00")) < first_trade_t - timedelta(minutes=30)]
    if prior_outcomes:
        prior_call_count += 1
    else:
        no_prior_count += 1

print(f"  tokens with PRIOR call (>=30min before 1st trade): {prior_call_count}")
print(f"  tokens without prior call: {no_prior_count}")
print(f"  → only {prior_call_count} would be eligible for RECALL strats (need is_recall=True)")

# 5. Distribution drift_vs_first quand recall fire
print(f"\nNow check actual recall_drift_pct distribution on TRADES that DID fire RECALL strats:")
recall_trades = [r for r in out2 if r["strategy"].startswith("RECALL_")]
print(f"  RECALL trades 14d: {len(recall_trades)}")
recall_unique_tokens = set(r["token_address"] for r in recall_trades)
print(f"  unique tokens hit by RECALL strats: {len(recall_unique_tokens)}")

# How many tokens have BOTH a prior call AND a 2nd call leading to a non-RECALL trade ?
# Those are MISSED OPPORTUNITIES — RECALL should have fired but didn't because filter
# (drift bounds, age window) excluded them
print(f"\nMissed recall opportunities (prior call exists but RECALL didn't fire):")
missed = []
for ca in unique_tokens:
    if ca in recall_unique_tokens:
        continue  # RECALL did fire
    trades_for_ca = [r for r in out2 if r["token_address"] == ca]
    trades_for_ca.sort(key=lambda r: r["created_at"])
    if len(trades_for_ca) < 2:
        continue
    first_trade_t = datetime.fromisoformat(trades_for_ca[0]["created_at"].replace("Z","+00:00"))
    outcomes = outcomes_by_token.get(ca, [])
    prior = [o for o in outcomes if datetime.fromisoformat(o["call_timestamp"].replace("Z","+00:00")) < first_trade_t - timedelta(minutes=30)]
    if prior:
        # has prior call, never tagged recall — why ?
        missed.append((ca, len(trades_for_ca), prior))

print(f"  tokens missed: {len(missed)}")
print(f"  examples (top 10):")
for ca, n_trades, prior in missed[:10]:
    sym = next((r.get("symbol", "?") for r in out2 if r["token_address"] == ca and r.get("symbol")), ca[:10])
    print(f"    {ca[:12]} ({n_trades} trades, {len(prior)} prior calls)")

# Compare: trades_per_token in retrade clusters vs RECALL actually firing
print(f"\nSummary: RECALL_* fires on {len(recall_unique_tokens)} tokens, but {prior_call_count} tokens had a prior call.")
print(f"  → ~{100*len(recall_unique_tokens)/max(prior_call_count,1):.0f}% capture rate")
