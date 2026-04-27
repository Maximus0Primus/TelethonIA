"""Run THIS on the VPS where ETH_PRIVATE_KEY is configured.

Quick read-only audit — confirms wallet is funded and reachable before flipping
eth_live_enabled. Does NOT mutate anything.

Usage on VPS:
  cd /opt/telethonia  (or wherever the repo is on VPS)
  python scripts/_vps_eth_wallet_check.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))

MIN_USD = 80
RECOMMENDED_USD = 150


def main():
    print("=" * 80)
    print("VPS ETH WALLET CHECK")
    print("=" * 80)

    pk = os.environ.get("ETH_PRIVATE_KEY")
    if not pk:
        print("FAIL: ETH_PRIVATE_KEY not set in environment")
        print("  Check scraper/.env on this VPS")
        sys.exit(1)
    print(f"  ETH_PRIVATE_KEY: present (len={len(pk)})")

    rpc = os.environ.get("ETH_RPC_URL") or os.environ.get("ETH_HTTP_RPC")
    print(f"  ETH_RPC_URL: {'present' if rpc else 'NOT SET (defaults inside live_trader_eth)'}")

    try:
        from live_trader_eth import _client, _eth_usd_price
        w3r, _, acct = _client()
    except Exception as e:
        print(f"FAIL: web3 client init failed: {e}")
        sys.exit(1)

    print(f"  wallet address: {acct.address}")

    try:
        bal_wei = w3r.eth.get_balance(acct.address)
        bal_eth = bal_wei / 1e18
        eth_usd = _eth_usd_price(w3r)
        bal_usd = bal_eth * eth_usd
    except Exception as e:
        print(f"FAIL: balance read failed: {e}")
        sys.exit(1)

    print(f"  ETH/USD spot: ${eth_usd:.2f}")
    print(f"  balance: {bal_eth:.6f} ETH ≈ ${bal_usd:.2f}")

    block = w3r.eth.block_number
    base_fee_gwei = w3r.eth.get_block("latest")["baseFeePerGas"] / 1e9
    print(f"  current block: {block}, base_fee: {base_fee_gwei:.2f} gwei")

    print()
    if bal_usd < MIN_USD:
        print(f"VERDICT: BLOCK ($ {bal_usd:.2f} < ${MIN_USD} minimum)")
        print(f"  top up to at least ${MIN_USD} (recommended ${RECOMMENDED_USD}) before going live")
        sys.exit(1)
    elif bal_usd < RECOMMENDED_USD:
        print(f"VERDICT: OK MARGINAL (${bal_usd:.2f}, recommended ${RECOMMENDED_USD})")
        print(f"  Phase 1 viable but no buffer. Top up to ${RECOMMENDED_USD} for safety.")
    else:
        print(f"VERDICT: OK ({bal_usd:.2f} comfortable for Phase 1 microtest)")

    print()
    print("Next: from local machine, run:")
    print("  python scripts/_enable_eth_phase1_microtest.py --local --apply")
    print("Then on this VPS:")
    print("  systemctl restart kol-scraper.service")


if __name__ == "__main__":
    main()
