"""Backfill Solana on-chain fees for live trades.

Scans paper_trades for rt_live SOL rows where gas_usd_buy/gas_usd_sell are NULL
and queries Solana RPC `getTransaction` to fetch the actual `meta.fee` (lamports).
Converts to USD using sol_price_at_entry/exit and writes back to DB.

Usage:
    python scripts/_backfill_solana_fees.py            # backfill all NULL
    python scripts/_backfill_solana_fees.py --hours=24 # only last 24h
    python scripts/_backfill_solana_fees.py --dry-run  # show what would be updated

Schedule on VPS via cron (every 15 min):
    */15 * * * * cd /path/to/repo && python scripts/_backfill_solana_fees.py --hours=2

Note: meta.fee includes BASE FEE (5000 lamports) + PRIORITY FEE only.
Jupiter Ultra platform fee (0.20%) is already in pnl_usd via execution_price.
"""
import os
import sys
import time
import argparse
from datetime import datetime, timedelta, timezone

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client

LAMPORTS_PER_SOL = 1_000_000_000
RPC_URL = os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
HELIUS_KEY = os.environ.get("HELIUS_API_KEY", "")
if HELIUS_KEY and "mainnet-beta" in RPC_URL:
    RPC_URL = f"https://mainnet.helius-rpc.com/?api-key={HELIUS_KEY}"

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])


def fetch_tx_fee_lamports(signature: str, retries: int = 3) -> int | None:
    """Query Solana RPC getTransaction to get meta.fee.
    Returns fee in lamports (base 5000 + priority), or None if not found."""
    if not signature or len(signature) < 80:
        return None
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTransaction",
        "params": [
            signature,
            {"maxSupportedTransactionVersion": 0, "encoding": "json", "commitment": "confirmed"},
        ],
    }
    for attempt in range(retries):
        try:
            r = requests.post(RPC_URL, json=body, timeout=10)
            j = r.json()
            if "result" in j and j["result"]:
                return int(j["result"]["meta"]["fee"])
            if "error" in j:
                err = j["error"].get("message", "")
                if "not found" in err.lower():
                    return None  # tx genuinely missing or too old
            time.sleep(0.5 * (attempt + 1))
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return None


def fetch_rows(hours: int | None):
    """Fetch rt_live SOL rows missing gas data."""
    out, off, step = [], 0, 1000
    since_iso = None
    if hours:
        since_iso = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    while True:
        q = (sb.table("paper_trades")
             .select("id,tx_signature,tx_signature_exit,sol_price_at_entry,sol_price_at_exit,"
                     "gas_usd_buy,gas_usd_sell,status,created_at,exit_at,symbol,strategy")
             .eq("source", "rt_live").eq("chain", "solana"))
        if since_iso:
            q = q.gte("created_at", since_iso)
        r = q.range(off, off+step-1).execute()
        if not r.data:
            break
        out.extend(r.data)
        if len(r.data) < step:
            break
        off += step
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=None,
                    help="Only backfill rows opened in last N hours (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="Show updates without writing")
    ap.add_argument("--max", type=int, default=500, help="Max rows to process (default: 500)")
    args = ap.parse_args()

    print(f"RPC: {RPC_URL[:50]}...")
    rows = fetch_rows(args.hours)
    print(f"Found {len(rows)} rt_live SOL rows {'(last %dh)' % args.hours if args.hours else '(all-time)'}")

    # Filter: needs backfill
    to_process = []
    for r in rows:
        needs_buy = r.get("gas_usd_buy") is None and r.get("tx_signature")
        needs_sell = r.get("gas_usd_sell") is None and r.get("tx_signature_exit")
        if needs_buy or needs_sell:
            to_process.append(r)
    print(f"Rows needing backfill: {len(to_process)} (limited to {args.max})")
    to_process = to_process[:args.max]

    updated = 0
    skipped = 0
    total_fee_usd = 0.0
    started = time.time()

    for i, r in enumerate(to_process, 1):
        update = {}
        # Buy fee
        if r.get("gas_usd_buy") is None and r.get("tx_signature"):
            fee_lamports = fetch_tx_fee_lamports(r["tx_signature"])
            if fee_lamports is not None:
                sol_price = float(r.get("sol_price_at_entry") or 0)
                if sol_price > 0:
                    fee_usd = (fee_lamports / LAMPORTS_PER_SOL) * sol_price
                    update["gas_usd_buy"] = round(fee_usd, 6)
                    total_fee_usd += fee_usd

        # Sell fee
        if r.get("gas_usd_sell") is None and r.get("tx_signature_exit"):
            fee_lamports = fetch_tx_fee_lamports(r["tx_signature_exit"])
            if fee_lamports is not None:
                sol_price = float(r.get("sol_price_at_exit") or r.get("sol_price_at_entry") or 0)
                if sol_price > 0:
                    fee_usd = (fee_lamports / LAMPORTS_PER_SOL) * sol_price
                    update["gas_usd_sell"] = round(fee_usd, 6)
                    total_fee_usd += fee_usd

        if not update:
            skipped += 1
            continue

        if args.dry_run:
            print(f"  [{i}/{len(to_process)}] DRY id={str(r['id'])[:8]} {r.get('symbol','?'):<10} {r.get('strategy','?'):<28} -> {update}")
        else:
            try:
                sb.table("paper_trades").update(update).eq("id", r["id"]).execute()
                updated += 1
                if i % 25 == 0:
                    print(f"  [{i}/{len(to_process)}] processed, updated={updated}, total_fee_usd=${total_fee_usd:.4f}")
            except Exception as e:
                print(f"  ERROR id={r['id']}: {e}")

        # Rate limit polite (Helius / public RPC)
        time.sleep(0.05)

    elapsed = time.time() - started
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Updated: {updated}")
    print(f"  Skipped (tx not found): {skipped}")
    print(f"  Total fees backfilled: ${total_fee_usd:.4f}")
    if updated > 0:
        print(f"  Average fee per swap: ${total_fee_usd / updated:.4f}")


if __name__ == "__main__":
    main()
