"""One-shot / manual empty-ATA rent sweep for the live wallet.

Closes every 0-balance token account (legacy SPL + Token-2022) and recovers the
~0.002 SOL rent each holds. Uses the exact same code path as the in-service periodic
sweep (live_trader._sweep_empty_token_accounts), so this is also the canonical manual
recovery tool. Requires SOLANA_PRIVATE_KEY in scraper/.env (present on the VPS).

Usage (on the VPS, where the key lives):
    python scripts/_sweep_empty_atas.py
"""
import os, sys, logging
from dotenv import load_dotenv

load_dotenv("scraper/.env")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

if not os.environ.get("SOLANA_PRIVATE_KEY"):
    sys.exit("SOLANA_PRIVATE_KEY not set — run this on the VPS (key is not synced locally).")

import live_trader  # noqa: E402

n, rent = live_trader._sweep_empty_token_accounts(max_close=200, return_rent=True)
print(f"\nClosed {n} empty ATA(s), recovered ~{rent:.5f} SOL = ${rent * 84.5:.2f}")
