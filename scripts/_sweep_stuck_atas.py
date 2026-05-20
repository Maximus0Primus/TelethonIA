"""v14e.66: One-shot sweep — close all empty token accounts (legacy SPL + Token-2022)
to recover locked ATA rent. Uses the fixed live_trader._close_token_account path.
Run on the VPS (needs SOLANA_PRIVATE_KEY). Dry-run by default; pass --apply to execute."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
sys.path.insert(0, "scraper")
from dotenv import load_dotenv
load_dotenv("scraper/.env")
import requests
import base58
from solders.keypair import Keypair
from live_trader import _close_token_account, _solana_rpc_url

APPLY = "--apply" in sys.argv
RENT = 0.00203928
LEGACY = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
T2022 = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
URL = _solana_rpc_url()


def rpc(m, p):
    return requests.post(URL, json={"jsonrpc": "2.0", "id": 1, "method": m, "params": p}, timeout=20).json().get("result")


kp = Keypair.from_bytes(base58.b58decode(os.environ["SOLANA_PRIVATE_KEY"]))
owner = str(kp.pubkey())
bal0 = (rpc("getBalance", [owner]) or {}).get("value", 0) / 1e9
print(f"Wallet: {owner}  balance={bal0:.5f} SOL")

empty = []  # (mint, program_label)
for prog, label in [(LEGACY, "legacy"), (T2022, "token2022")]:
    res = rpc("getTokenAccountsByOwner", [owner, {"programId": prog}, {"encoding": "jsonParsed"}]) or {}
    for a in res.get("value", []):
        info = a["account"]["data"]["parsed"]["info"]
        if int(info["tokenAmount"]["amount"]) == 0:
            empty.append((info["mint"], label))

print(f"Empty closeable accounts: {len(empty)}  (rent locked ~{len(empty)*RENT:.5f} SOL = ${len(empty)*RENT*84.5:.2f})")
by_prog = {}
for _, l in empty:
    by_prog[l] = by_prog.get(l, 0) + 1
print(f"  by program: {by_prog}")

if not APPLY:
    print("\nDRY-RUN. Re-run with --apply to close them.")
    sys.exit(0)

ok = 0
for mint, label in empty:
    res = _close_token_account(mint)
    print(f"  {'OK ' if res else 'FAIL'} [{label}] {mint[:14]}")
    if res:
        ok += 1
    time.sleep(0.3)

bal1 = (rpc("getBalance", [owner]) or {}).get("value", 0) / 1e9
print(f"\nClosed {ok}/{len(empty)}.  Balance {bal0:.5f} -> {bal1:.5f} SOL  (recovered {bal1-bal0:+.5f} SOL = ${(bal1-bal0)*84.5:+.2f})")
