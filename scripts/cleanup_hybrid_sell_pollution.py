"""
v133-D cleanup: recompute exit_price / pnl_pct / pnl_usd for live trades polluted by
the hybrid-allocation shared-ATA bug.

Symptom 1 (winner leg): sell_input_tokens ~= 2 x buy_output_tokens because execute_sell
was called without amount_tokens and drained the full ATA. exit_price is back-derived
from the full SOL received, so pnl is ~2x inflated.

Symptom 2 (loser leg): sibling drained the ATA first; reconcile_positions marked the
trade 'reconciled' with hardcoded pnl_pct = -1.0, pnl_usd = -position_usd.

Fix: for each token where BOTH a "drained sibling" and a "reconciled sibling" exist
within the same trade window, recompute using the winner's sell_sol_received /
sell_input_tokens ratio (SOL-per-token from the actual on-chain swap) and each leg's
own buy_output_tokens. Result: both legs reflect the real per-token fill.

DRY RUN by default — prints planned changes. Pass --apply to write.
"""

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    # scraper/.env holds SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY.
    load_dotenv(Path(__file__).resolve().parent.parent / "scraper" / ".env")
except ImportError:
    pass

from supabase import create_client

LAMPORTS_PER_SOL = 1_000_000_000


def parse_iso(s):
    if not s:
        return None
    if isinstance(s, datetime):
        return s
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Write changes to DB")
    parser.add_argument("--days", type=int, default=14, help="Look-back window (days)")
    args = parser.parse_args()

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        print("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set", file=sys.stderr)
        sys.exit(1)
    sb = create_client(url, key)

    cutoff = f"now() - interval '{args.days} days'"
    # Supabase client doesn't accept raw SQL in filter; compute ISO cutoff client-side.
    from datetime import timedelta
    iso_cutoff = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()

    # Pull all live trades (closed) in window, grouped by token_address
    resp = (
        sb.table("paper_trades")
        .select(
            "id, symbol, token_address, strategy, status, entry_price, position_usd, "
            "exit_price, pnl_pct, pnl_usd, buy_output_tokens, sell_input_tokens, "
            "sell_output_lamports, sell_sol_received, sol_price_at_exit, "
            "tx_signature_exit, created_at, exit_at"
        )
        .eq("source", "rt_live")
        .gte("created_at", iso_cutoff)
        .execute()
    )
    trades = resp.data or []

    by_token = defaultdict(list)
    for t in trades:
        if t.get("token_address"):
            by_token[t["token_address"]].append(t)

    planned = []  # list of (trade, new_payload, reason)

    for ca, rows in by_token.items():
        if len(rows) < 2:
            continue  # need at least 2 legs to have a sibling artefact

        # Identify a "drain winner": sell_input_tokens significantly > buy_output_tokens
        winners = [
            r for r in rows
            if r.get("sell_input_tokens") and r.get("buy_output_tokens")
            and int(r["sell_input_tokens"]) >= 1.5 * int(r["buy_output_tokens"])
            and r.get("tx_signature_exit")
        ]
        # Identify reconciled losers
        losers = [r for r in rows if r.get("status") == "reconciled"]

        if not winners and not losers:
            continue

        # Use the winner with the largest absolute drain (most reliable SOL-per-token)
        winner = None
        if winners:
            winner = max(winners, key=lambda r: int(r["sell_input_tokens"]))

        if not winner:
            # Loser without a visible winner (winner row already re-fixed or missing data).
            # Without SOL-per-token we can't recompute; skip and log.
            for loser in losers:
                planned.append((loser, None, "no_winner_found_skip"))
            continue

        # Compute SOL-per-token from the winner's realized fill.
        sol_received = winner.get("sell_sol_received")
        if not sol_received and winner.get("sell_output_lamports"):
            sol_received = float(winner["sell_output_lamports"]) / LAMPORTS_PER_SOL
        sol_received = float(sol_received or 0)
        tokens_sold = int(winner.get("sell_input_tokens") or 0)
        sol_price = float(winner.get("sol_price_at_exit") or 0)
        if not (sol_received > 0 and tokens_sold > 0 and sol_price > 0):
            for loser in losers:
                planned.append((loser, None, "winner_missing_fields"))
            continue

        sol_per_token = sol_received / tokens_sold

        def recompute(r):
            entry = float(r.get("entry_price") or 0)
            pos_usd = float(r.get("position_usd") or 0)
            tokens = int(r.get("buy_output_tokens") or 0)
            if not (entry > 0 and pos_usd > 0 and tokens > 0):
                return None
            our_sol = tokens * sol_per_token
            our_usd = our_sol * sol_price
            new_exit = entry * (our_usd / pos_usd)
            new_pnl_pct = round((new_exit / entry) - 1, 4)
            new_pnl_usd = round(pos_usd * new_pnl_pct, 2)
            return {
                "exit_price": new_exit,
                "pnl_pct": new_pnl_pct,
                "pnl_usd": new_pnl_usd,
                "sell_sol_received": round(our_sol, 6),
                "sol_price_at_exit": sol_price,
            }

        # Fix the winner (deflate 2x -> real single-leg)
        new_w = recompute(winner)
        if new_w:
            old = (winner.get("pnl_pct"), winner.get("pnl_usd"), winner.get("exit_price"))
            if abs((new_w["pnl_pct"] or 0) - (winner.get("pnl_pct") or 0)) > 0.005:
                planned.append((winner, new_w, "winner_deflated"))

        # Fix each loser
        for loser in losers:
            new_l = recompute(loser)
            if new_l:
                planned.append((loser, new_l, "loser_from_sibling"))
            else:
                planned.append((loser, None, "loser_missing_fields"))

    # Report
    print("=" * 72)
    print(f"v133-D CLEANUP — {'APPLY' if args.apply else 'DRY RUN'} — {len(planned)} rows")
    print("=" * 72)
    win_delta = 0.0
    los_delta = 0.0
    for trade, payload, reason in planned:
        old_pnl_usd = float(trade.get("pnl_usd") or 0)
        if payload:
            new_pnl_usd = payload["pnl_usd"]
            delta = new_pnl_usd - old_pnl_usd
            if reason == "winner_deflated":
                win_delta += delta
            else:
                los_delta += delta
            print(
                f"[{reason:24s}] {trade['symbol']:12s} {trade['strategy']:25s} "
                f"pnl_pct {float(trade.get('pnl_pct') or 0):+.4f} -> {payload['pnl_pct']:+.4f} "
                f"| pnl_usd ${old_pnl_usd:+.2f} -> ${new_pnl_usd:+.2f} "
                f"(D ${delta:+.2f})"
            )
        else:
            print(f"[{reason:24s}] {trade['symbol']:12s} {trade['strategy']:25s} SKIP (insufficient data)")

    print("-" * 72)
    print(f"Winner rows net D: ${win_delta:+.2f}")
    print(f"Loser  rows net D: ${los_delta:+.2f}")
    print(f"Total        net D: ${win_delta + los_delta:+.2f}")
    print("(Real on-chain SOL unchanged — this just unwinds double-count in DB.)")

    if not args.apply:
        print("\nDRY RUN — re-run with --apply to write.")
        return

    written = 0
    for trade, payload, reason in planned:
        if not payload:
            continue
        try:
            sb.table("paper_trades").update(payload).eq("id", trade["id"]).execute()
            written += 1
        except Exception as e:
            print(f"FAIL update {trade['symbol']} ({trade['id']}): {e}", file=sys.stderr)
    print(f"\nWrote {written} rows.")


if __name__ == "__main__":
    main()
