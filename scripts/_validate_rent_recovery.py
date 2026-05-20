"""Validate close_ata rent recovery on-chain (post v14e.65). Isolates STANDALONE closes."""
import os, time, requests
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv("scraper/.env")
KEY = os.environ["HELIUS_API_KEY"]
URL = f"https://mainnet.helius-rpc.com/?api-key={KEY}"
RENT = 0.00203928
def rpc(m,p): return requests.post(URL, json={"jsonrpc":"2.0","id":1,"method":m,"params":p}, timeout=25).json().get("result")

SEED="fFzwr3cgGMBqrGP7u8b1sHLi4oMgjWo2ZzNeDZPiZZAEePrBMJyYGSxy59RtNh1rb8Djren6uxALAxJknhoAAbF"
tx0=rpc("getTransaction",[SEED,{"maxSupportedTransactionVersion":0,"encoding":"jsonParsed"}])
owner=tx0["transaction"]["message"]["accountKeys"][0]["pubkey"]
bal=rpc("getBalance",[owner]) or {}
print(f"Wallet {owner}  bal={bal.get('value',0)/1e9:.5f} SOL")
sigs=rpc("getSignaturesForAddress",[owner,{"limit":300}]) or []

pure_close=[]; n_buy=n_sell=n_swapclose=0
for s in sigs:
    bt=s.get("blockTime")
    if not bt or (datetime.now(timezone.utc).timestamp()-bt)/3600>72: continue
    tx=rpc("getTransaction",[s["signature"],{"maxSupportedTransactionVersion":0,"encoding":"jsonParsed"}])
    if not tx or not tx.get("meta"): continue
    top=tx["transaction"]["message"].get("instructions",[])
    # program names of top-level instrs, ignore computeBudget
    progs=[]; types=[]
    for i in top:
        if not isinstance(i,dict): continue
        prog=i.get("program") or i.get("programId","")
        if prog=="computeBudget": continue
        progs.append(prog); types.append(i.get("parsed",{}).get("type"))
    delta=(tx["meta"]["postBalances"][0]-tx["meta"]["preBalances"][0])/1e9
    err=tx["meta"].get("err") is not None
    # PURE close = every non-computeBudget top instr is a spl-token closeAccount
    if progs and all(t=="closeAccount" for t in types):
        pure_close.append((datetime.fromtimestamp(bt,timezone.utc).strftime("%m-%d %H:%M"),delta,err,s["signature"][:10]))
    elif delta < -0.005: n_buy+=1
    elif delta > 0.003: n_sell+=1
    time.sleep(0.02)

print(f"\nBuys~{n_buy}  Sells~{n_sell}  Standalone close_ata txns={len(pure_close)}")
ok=[d for _,d,e,_ in pure_close if not e and d>0.0015]
fail=[d for _,d,e,_ in pure_close if e or d<=0.0015]
print(f"  recovered OK (delta>+0.0015): {len(ok)}  sum={sum(ok):+.5f} SOL = ${sum(ok)*84.5:+.2f}")
print(f"  failed/no-recovery:           {len(fail)} sum={sum(fail):+.5f} SOL")
print("  --- detail ---")
for ts,d,e,sig in pure_close[:30]:
    print(f"  {ts} | {d:+.6f} SOL | {'ERR' if e else 'ok '} | {sig}..")
