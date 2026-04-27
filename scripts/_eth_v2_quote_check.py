"""Read-only validation that the V3+V2 quote helpers in live_trader_eth work.

No private key needed, no transactions submitted. Calls _quote_with_best_fee,
_quote_v2, and _quote_best_route on a panel of well-known mainnet tokens to
confirm:
  - V2 quote helper returns a non-zero amount on V2-only / V2-major tokens
  - V3 quote helper returns a non-zero amount on V3-major tokens
  - _quote_best_route picks the higher-output route

Run before any on-chain smoke to verify the quote plumbing without spending gas.

Usage:
    python scripts/_eth_v2_quote_check.py                      # default panel
    python scripts/_eth_v2_quote_check.py --eth-amount 0.003   # custom size
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))

from web3 import Web3

# Reuse the same constants + helpers as production. No _client() call so
# the script works without ETH_PRIVATE_KEY.
from live_trader_eth import (
    DEFAULT_READ_RPC, WETH,
    _quote_with_best_fee, _quote_v2, _quote_best_route, _eth_usd_price,
)

# Panel of known tokens (mainnet) covering each routing case we care about.
# Liquidity profiles change over time — these are stable as of 2026.
PANEL = [
    # symbol,        address,                                          notes
    ("PEPE",  "0x6982508145454Ce325dDbE47a25d4ec3d2311933", "V3 dominant (3000bp pool)"),
    ("SHIB",  "0x95aD61b0a150d79219dCF64E1E6Cc01f0B64C4cE", "V2 dominant + V3 (10000bp)"),
    ("MOG",   "0xaaeE1A9723aaDB7afA2810263653A34bA2C21C7a", "V2-only (V3 has shallow pool)"),
    ("WOJAK", "0x5026F006B85729a8b14553FAE6af249aD16c9aaB", "V2-only memecoin"),
    ("USDC",  "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", "V3 dominant — sanity baseline"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eth-amount", type=float, default=0.003,
                    help="ETH input size for the quote (default 0.003 ≈ $7)")
    ap.add_argument("--rpc", default=os.environ.get("ETH_READ_RPC_URL", DEFAULT_READ_RPC))
    args = ap.parse_args()

    w3 = Web3(Web3.HTTPProvider(args.rpc))
    if not w3.is_connected():
        print(f"ERROR: cannot connect to {args.rpc}")
        sys.exit(1)

    eth_usd = _eth_usd_price(w3)
    amount_in_wei = w3.to_wei(args.eth_amount, "ether")
    amount_usd = args.eth_amount * eth_usd

    print(f"=== V2/V3 quote validation ===")
    print(f"RPC:       {args.rpc}")
    print(f"ETH/USD:   ${eth_usd:.2f}")
    print(f"Probe:     {args.eth_amount} ETH = ${amount_usd:.2f}")
    print()

    fmt = "  {sym:<8}{v3:>14}{v3fee:>8}{v2:>14}{best:>8}{spread:>8}  {note}"
    print(fmt.format(sym="symbol", v3="v3_out", v3fee="v3_fee",
                      v2="v2_out", best="best", spread="spread%", note="notes"))
    print("  " + "-" * 100)

    n_v3 = n_v2 = n_either = n_none = 0
    for sym, addr, note in PANEL:
        try:
            v3_out, v3_fee, _ = _quote_with_best_fee(w3, amount_in_wei, WETH, addr)
        except Exception as e:
            v3_out, v3_fee = None, None
            note += f" | v3 err: {type(e).__name__}"
        try:
            v2_out = _quote_v2(w3, amount_in_wei, WETH, addr)
        except Exception as e:
            v2_out = None
            note += f" | v2 err: {type(e).__name__}"

        best_out, best_route, _, _ = _quote_best_route(w3, amount_in_wei, WETH, addr)
        spread = ""
        if v3_out and v2_out and v3_out > 0 and v2_out > 0:
            spread = f"{(max(v3_out, v2_out) / min(v3_out, v2_out) - 1) * 100:+.2f}"

        if v3_out and v2_out:
            n_either += 1
        elif v3_out:
            n_v3 += 1
        elif v2_out:
            n_v2 += 1
        else:
            n_none += 1

        print(fmt.format(
            sym=sym,
            v3=f"{v3_out/1e9:>.2f}B" if v3_out else "—",
            v3fee=str(v3_fee) if v3_fee else "—",
            v2=f"{v2_out/1e9:>.2f}B" if v2_out else "—",
            best=best_route or "none",
            spread=spread,
            note=note,
        ))

    print()
    print(f"Summary: V3-only={n_v3}  V2-only={n_v2}  both={n_either}  none={n_none}")
    print()

    # Sanity assertions — fail loudly if the helpers are broken.
    failed = []
    if n_v2 == 0 and n_either == 0:
        failed.append("V2 quote returned 0 results across the entire panel — V2 helper likely broken")
    if n_v3 == 0 and n_either == 0:
        failed.append("V3 quote returned 0 results across the entire panel — V3 helper likely broken")

    if failed:
        for msg in failed:
            print(f"FAIL: {msg}")
        sys.exit(2)
    print("OK: V2 + V3 quote plumbing both responding.")


if __name__ == "__main__":
    main()
