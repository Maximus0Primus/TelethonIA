"""v14e.56: idempotent fix-up for kol_call_outcomes rows that should be marked
`dead_no_ohlcv` but stuck at NULL outcome_status.

Diagnostic (May 3): 1507 NULL outcome_status on 14d. Of these, 1292 have
entry_price (Phase B succeeded) but 0 have ath_after_call (Phase C silently
skipped them). Root cause: Phase C marks `dead_no_ohlcv` only when call_age>72h
AND DexPaprika+GeckoTerminal+Birdeye all return no candles. Many fresh-mint
pump.fun rugs have entry_price (filled at spawn second) but no historical OHLCV
once dead — they fall through Phase C indefinitely.

What this script does (DRY-RUN by default):
- Find rows with outcome_status IS NULL AND call_age > 72h.
- Mark them `dead_no_ohlcv` with last_checked_at=now(). NO ATH attempt — they
  are confirmed dead by age + Phase C history.
- Touches only the status field. Reversible by setting status back to NULL.

Why standalone script vs touching outcome_tracker.py:
- outcome_tracker is a 3-phase pipeline (fill_kol_outcomes) that ML training
  depends on. Modifying it during a session = high regression risk.
- This script is invocation-once, idempotent, hits only the orphan rows.
- Can be re-run anytime (won't re-update rows already marked).

Usage:
    python scripts/_kco_resurrect_dead.py              # dry-run (default)
    python scripts/_kco_resurrect_dead.py --apply      # write to DB
    python scripts/_kco_resurrect_dead.py --apply --batch 200
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scraper"))

try:
    from dotenv import load_dotenv
    # Project keeps .env under scraper/, mirroring how outcome_tracker.py loads it
    load_dotenv(REPO_ROOT / "scraper" / ".env")
    load_dotenv(REPO_ROOT / ".env")  # fallback if also at root
except ImportError:
    pass

from supabase import create_client


def get_client():
    url = os.environ["SUPABASE_URL"]
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ["SUPABASE_KEY"]
    return create_client(url, key)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="Actually write to DB. Without this flag, prints what would change.")
    ap.add_argument("--age-hours", type=float, default=72,
                    help="Min call age in hours before marking dead (default: 72).")
    ap.add_argument("--batch", type=int, default=500,
                    help="Update batch size (default: 500).")
    ap.add_argument("--limit", type=int, default=10000,
                    help="Max rows to process per run (default: 10000).")
    args = ap.parse_args()

    client = get_client()

    # Find orphan rows: NULL outcome_status, call_age > N hours.
    # FIFO order so we resolve oldest first.
    cutoff = datetime.now(timezone.utc).timestamp() - args.age_hours * 3600
    cutoff_iso = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat()

    print(f"Scanning kol_call_outcomes for outcome_status IS NULL AND call_timestamp < {cutoff_iso} ({args.age_hours}h ago)")

    page_size = 1000
    offset = 0
    all_rows = []
    while True:
        resp = (
            client.table("kol_call_outcomes")
            .select("id, symbol, kol_group, call_timestamp, entry_price, ath_after_call")
            .is_("outcome_status", "null")
            .lt("call_timestamp", cutoff_iso)
            .order("call_timestamp")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            break
        all_rows.extend(rows)
        offset += page_size
        if len(all_rows) >= args.limit:
            all_rows = all_rows[:args.limit]
            print(f"  reached --limit {args.limit}, stopping scan")
            break
        if len(rows) < page_size:
            break

    if not all_rows:
        print("No orphan rows found. Nothing to do.")
        return 0

    n_total = len(all_rows)
    n_with_entry = sum(1 for r in all_rows if r.get("entry_price"))
    n_with_ath = sum(1 for r in all_rows if r.get("ath_after_call"))
    print(f"Found {n_total} orphan rows ({n_with_entry} have entry_price, {n_with_ath} have ath_after_call)")

    # Sample for sanity
    print("\nSample (first 5):")
    for r in all_rows[:5]:
        print(f"  id={r['id']} {r.get('symbol','?'):12s} {r.get('kol_group','?'):20s} "
              f"call={r['call_timestamp'][:19]} entry={r.get('entry_price')} ath={r.get('ath_after_call')}")

    if not args.apply:
        print(f"\n[DRY-RUN] Would mark {n_total} rows as outcome_status='dead_no_ohlcv'.")
        print("Re-run with --apply to write.")
        return 0

    # Write in batches
    now_iso = datetime.now(timezone.utc).isoformat()
    update_payload = {
        "outcome_status": "dead_no_ohlcv",
        "last_checked_at": now_iso,
    }
    n_done = 0
    for i in range(0, n_total, args.batch):
        chunk_ids = [r["id"] for r in all_rows[i:i + args.batch]]
        try:
            client.table("kol_call_outcomes").update(update_payload).in_("id", chunk_ids).execute()
            n_done += len(chunk_ids)
            print(f"  updated {n_done}/{n_total}")
        except Exception as e:
            print(f"  batch {i}-{i+len(chunk_ids)} FAILED: {e}")
            return 2

    print(f"\nDone. {n_done} rows marked dead_no_ohlcv.")
    print("To revert: UPDATE kol_call_outcomes SET outcome_status=NULL "
          f"WHERE outcome_status='dead_no_ohlcv' AND last_checked_at='{now_iso}';")
    return 0


if __name__ == "__main__":
    sys.exit(main())
