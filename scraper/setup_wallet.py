"""
One-time Solana wallet generator for live trading.

Usage:
    python setup_wallet.py

Prints the public key (to fund) and private key (for .env).
"""

from solders.keypair import Keypair
import base58


def main():
    kp = Keypair()
    pubkey = str(kp.pubkey())
    privkey = base58.b58encode(bytes(kp)).decode()

    print("=" * 60)
    print("NEW SOLANA WALLET GENERATED")
    print("=" * 60)
    print(f"\nPublic key:  {pubkey}")
    print(f"Private key: {privkey}")
    print(f"\nAdd to .env on VPS:")
    print(f"  SOLANA_PRIVATE_KEY={privkey}")
    print(f"\nSend SOL to: {pubkey}")
    print("=" * 60)


if __name__ == "__main__":
    main()
