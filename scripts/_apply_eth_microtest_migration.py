"""Apply ETH microtest DB migration via Supabase. Idempotent.

Adds 6 columns to paper_trades for fine-grained slippage/gas/latency tracking.
Read scripts/_eth_microtest_migration.sql for rationale per column.

Usage:
  python scripts/_apply_eth_microtest_migration.py            # dry-run, prints SQL
  python scripts/_apply_eth_microtest_migration.py --apply    # executes
"""
import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))

SQL = """
ALTER TABLE paper_trades
  ADD COLUMN IF NOT EXISTS gas_usd_buy        NUMERIC,
  ADD COLUMN IF NOT EXISTS gas_usd_sell       NUMERIC,
  ADD COLUMN IF NOT EXISTS quote_slip_bps_buy   INTEGER,
  ADD COLUMN IF NOT EXISTS quote_slip_bps_sell  INTEGER,
  ADD COLUMN IF NOT EXISTS block_number_buy   BIGINT,
  ADD COLUMN IF NOT EXISTS block_number_sell  BIGINT;
""".strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    print("Migration SQL:")
    print(SQL)
    if not args.apply:
        print("\n[DRY RUN] pass --apply to execute. Or run the SQL manually in Supabase Studio.")
        return

    # Supabase Python client doesn't expose raw SQL — use the postgrest
    # ad-hoc RPC if available, else require manual application.
    print("\nNOTE: supabase-py has no raw-SQL endpoint. Apply this migration via:")
    print("  1) Supabase Studio -> SQL Editor (paste the SQL above)")
    print("  2) OR psql with DATABASE_URL")
    print("  3) OR `supabase db push` if you maintain the migration locally")
    print("\nOnce applied, re-run scripts/_check_eth_infra.py — all 6 cols should show OK.")


if __name__ == "__main__":
    main()
