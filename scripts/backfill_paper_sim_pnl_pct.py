"""Backfill paper_sim_pnl_pct on closed paper_trades that lack it.

paper_sim_pnl_pct (v143.6) stores the sim-projected PnL for each closed live/paper
trade so verify_sim_live_alignment.py can do per-trade joins. New v144 shadows
opened post-deploy haven't had this column populated yet — this script fills the gap.

Strategy: for each closed trade with paper_sim_pnl_pct IS NULL, compute a
deterministic "sim-equivalent" PnL using the same logic as the production
sim from-trades mode: just copy pnl_pct (since paper IS the ground truth for
sim from-trades — the sim "replay" of a paper trade just re-uses its PnL).

For LIVE trades, the sim equivalent is the matching paper trade's pnl_pct
(token+strategy join). For PAPER/shadow trades, it's the trade's own pnl_pct
(self-reference for from-trades sim).

Idempotent: only updates rows where paper_sim_pnl_pct IS NULL.
"""
import os, sys
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from dotenv import load_dotenv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

WINDOW_DAYS = int(os.environ.get("BACKFILL_DAYS", "14"))
DRY_RUN = "--dry-run" in sys.argv

since = (datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)).isoformat()

print(f"=== Backfill paper_sim_pnl_pct (last {WINDOW_DAYS}d) ===")
print(f"Mode: {'DRY-RUN' if DRY_RUN else 'WRITE'}")
print(f"Since: {since}\n")

def fetch(source):
    out, off = [], 0
    while True:
        r = sb.table("paper_trades").select(
            "id,token_address,strategy,pnl_pct,paper_sim_pnl_pct,is_shadow,source"
        ).eq("source", source).gte("created_at", since).neq("status", "open").neq("status", "closing").is_("paper_sim_pnl_pct", "null").range(off, off+999).execute().data
        out += r
        if len(r) < 1000:
            break
        off += 1000
    return out

# Load missing rows
rt_missing = fetch("rt")
live_missing = fetch("rt_live")
print(f"rt rows missing paper_sim_pnl_pct:      {len(rt_missing):,}")
print(f"rt_live rows missing paper_sim_pnl_pct: {len(live_missing):,}\n")

# For LIVE rows, look up matching paper trade's pnl_pct (token + strategy)
# Build paper index for the same window first
paper_idx = defaultdict(list)
all_paper, off = [], 0
while True:
    r = sb.table("paper_trades").select("token_address,strategy,pnl_pct,is_shadow").eq("source", "rt").gte("created_at", since).neq("status", "open").range(off, off+999).execute().data
    all_paper += r
    if len(r) < 1000:
        break
    off += 1000
for r in all_paper:
    paper_idx[(r["token_address"], r["strategy"])].append(r["pnl_pct"])

# Build update batches
def avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None

updates = []

# rt rows: self-reference (sim from-trades replays the paper PnL)
for r in rt_missing:
    if r.get("pnl_pct") is None:
        continue
    updates.append({"id": r["id"], "paper_sim_pnl_pct": r["pnl_pct"]})

# rt_live rows: join with paper sibling on (token, strategy)
matched_live = 0
for r in live_missing:
    sibs = paper_idx.get((r["token_address"], r["strategy"]))
    if not sibs:
        continue
    sim_pnl = avg(sibs)
    if sim_pnl is None:
        continue
    updates.append({"id": r["id"], "paper_sim_pnl_pct": sim_pnl})
    matched_live += 1

print(f"Updates planned:  {len(updates):,}")
print(f"  rt self-ref:    {len(rt_missing):,}")
print(f"  rt_live joined: {matched_live:,}/{len(live_missing):,} (others have no paper sibling)")

if DRY_RUN:
    print("\nDRY-RUN — no writes. Re-run without --dry-run to commit.")
    sys.exit(0)

# Apply in batches of 100 via individual updates (Supabase doesn't support bulk update by id)
batch_size = 100
done = 0
for i in range(0, len(updates), batch_size):
    chunk = updates[i:i+batch_size]
    for u in chunk:
        try:
            sb.table("paper_trades").update({"paper_sim_pnl_pct": u["paper_sim_pnl_pct"]}).eq("id", u["id"]).execute()
            done += 1
        except Exception as e:
            print(f"  ERROR id={u['id']}: {e}")
    print(f"  {done:,}/{len(updates):,} updated")

print(f"\nBackfill complete: {done:,} rows updated.")
