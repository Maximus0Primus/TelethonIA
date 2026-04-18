"""
Audit rt_bankroll vs paper_trades ground truth.

rt_bankroll is updated incrementally (read-modify-write) on each trade close.
If trades close without triggering _rt_update_bankroll (e.g., reconcile auto-close
bypass, crashed updates), the stored state drifts from ground truth.

Strategy: rather than taking a fixed --since date (which can include pre-reset
trades and falsely report drift), this script auto-detects the reset point per
strategy by matching the N most-recent main-RT closed trades against the stored
(pnl, trades) pair. If the sum matches, the stored state is consistent with
"no bypassed updates since the reset that put that count at 0".

Any strategy where no match is found signals a real drift (missing updates).

Dry-run by default. --apply writes corrections (only if drift is detected and
the delta can be derived from a known source — currently only the global
v133-D cleanup delta is accepted).
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / "scraper" / ".env")
except ImportError:
    pass

from supabase import create_client


CLOSED = ["sl_hit", "tp_hit", "timeout", "trail_stop", "be_stop",
          "reconciled", "tp_partial", "moonbag_tp"]

# Known one-shot corrections applied manually after DB cleanup scripts run.
# Each entry documents the delta that should be applied once. Re-running the
# script must not apply them twice — idempotency is the caller's responsibility.
KNOWN_CLEANUPS = {
    "v133-D": {
        "delta_usd": -2.99,
        "description": "v133-D hybrid-ATA sell pollution cleanup (net delta over 30 rows)",
    },
}


def fetch_recent_closed(sb, strategy: str, limit: int = 2000):
    rows = []
    offset = 0
    while True:
        q = (
            sb.table("paper_trades")
            .select("exit_at, pnl_usd")
            .eq("source", "rt").eq("is_shadow", False)
            .eq("strategy", strategy).in_("status", CLOSED)
            .order("exit_at", desc=True)
            .range(offset, offset + 999)
        )
        r = q.execute()
        rows.extend(r.data or [])
        if len(r.data or []) < 1000 or len(rows) >= limit:
            break
        offset += 1000
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply-cleanup", metavar="TAG",
                        choices=list(KNOWN_CLEANUPS.keys()),
                        help="Apply a known cleanup delta to global total_pnl and current_balance.")
    args = parser.parse_args()

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not (url and key):
        print("SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY required", file=sys.stderr)
        sys.exit(1)
    sb = create_client(url, key)

    cur = sb.table("rt_bankroll").select("*").limit(1).execute()
    if not cur.data:
        print("No rt_bankroll row.")
        return
    row = cur.data[0]
    row_id = row["id"]
    strat_stored = row.get("strategy_bankrolls") or {}

    print("=" * 78)
    print(f"rt_bankroll id={row_id} — per-strategy audit")
    print("=" * 78)
    print(f"  {'strategy':<40s} {'stored pnl':>11s} {'sum recent-N':>13s}  {'status':<10s} {'reset >':s}")

    any_drift = False
    for strategy, s in strat_stored.items():
        n = int(s.get("trades") or 0)
        stored_pnl = round(float(s.get("pnl") or 0), 2)
        if n == 0:
            print(f"  {strategy:<40s} ${stored_pnl:+10.2f} {'(no trades)':>13s}  {'OK':<10s}")
            continue
        rows = fetch_recent_closed(sb, strategy, limit=max(n * 2, 2000))
        if len(rows) < n:
            print(f"  {strategy:<40s} ${stored_pnl:+10.2f} {'insuf rows':>13s}  {'SKIP':<10s} (have {len(rows)} < {n})")
            continue
        sum_pnl = round(sum(float(r.get("pnl_usd") or 0) for r in rows[:n]), 2)
        cutoff = rows[n - 1].get("exit_at") if rows else None
        delta = round(sum_pnl - stored_pnl, 2)
        status = "OK" if abs(delta) < 0.1 else f"DRIFT {delta:+.2f}"
        if status != "OK":
            any_drift = True
        print(f"  {strategy:<40s} ${stored_pnl:+10.2f} ${sum_pnl:+10.2f}  {status:<10s} {cutoff}")

    print("\nPer-strategy verdict: " + ("DRIFT DETECTED — investigate" if any_drift else "CLEAN"))

    # Global section
    print("\n" + "=" * 78)
    print("Global fields")
    print("=" * 78)
    for k in ("starting_capital", "current_balance", "total_pnl", "total_trades",
              "peak_balance", "max_drawdown_pct", "last_updated_at"):
        print(f"  {k:<22s} {row.get(k)}")

    if not args.apply_cleanup:
        print("\nDry-run. Pass --apply-cleanup <tag> to apply a known cleanup delta.")
        print(f"Known cleanups: {', '.join(KNOWN_CLEANUPS.keys())}")
        return

    spec = KNOWN_CLEANUPS[args.apply_cleanup]
    delta = float(spec["delta_usd"])
    old_balance = float(row["current_balance"])
    old_pnl = float(row["total_pnl"])
    new_balance = round(old_balance + delta, 2)
    new_pnl = round(old_pnl + delta, 2)
    print(f"\nApplying {args.apply_cleanup}: {spec['description']}")
    print(f"  total_pnl:       ${old_pnl:+,.2f} -> ${new_pnl:+,.2f}")
    print(f"  current_balance: ${old_balance:+,.2f} -> ${new_balance:+,.2f}")
    sb.table("rt_bankroll").update({
        "current_balance": new_balance,
        "total_pnl": new_pnl,
        "last_updated_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", row_id).execute()
    print("Applied.")


if __name__ == "__main__":
    main()
