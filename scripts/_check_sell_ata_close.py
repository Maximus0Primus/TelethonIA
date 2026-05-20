"""Does the SELL txn itself close the meme-token ATA and return rent? (settles post-fix rent recovery)"""
import os, time, requests
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv("scraper/.env")
KEY=os.environ["HELIUS_API_KEY"]; URL=f"https://mainnet.helius-rpc.com/?api-key={KEY}"
WSOL="So11111111111111111111111111111111111111112"
def rpc(m,p): return requests.post(URL,json={"jsonrpc":"2.0","id":1,"method":m,"params":p},timeout=25).json().get("result")

# recent post-fix sell exits (from DB query, newest first)
EXITS=["644g8ahiqLHPCLtqaCvFt74RfySj4YeFjPyNizxDLSANtfnLf2Dauxvm3H4UgEZPKCPoACecKetVQdD9C3GQH6qb",
       "3PWC2qcry4tYbStzVjZs641oZCCJ2CfQKzk1hbsGQHdYEwahx5RSJhFb88tWG7GYLHE9G5nLrJGPdfNHAGtKjubq",
       "2HfzqGjNVpAfzMUcDfHn6Eigq3Ts9dzg849pNbrf1maGviwtvtm9wPhuwTKVxG6YDbLkeKXwD4r9E4VZBt4SQER3",
       "2HfzqGjNVpAfzMUcDfHn6Eigq3Ts9dzg849pNbrf1maGviwtvtm9wPhuwTKVxG6YDbLkeKXwD4r9E4VZBt4SQER3"]
for sig in EXITS[:3]:
    tx=rpc("getTransaction",[sig,{"maxSupportedTransactionVersion":0,"encoding":"jsonParsed"}])
    if not tx: print(f"{sig[:10]} no tx"); continue
    ts=datetime.fromtimestamp(tx["blockTime"],timezone.utc).strftime("%m-%d %H:%M")
    instrs=tx["transaction"]["message"].get("instructions",[])
    inner=[]
    for ii in (tx["meta"].get("innerInstructions") or []): inner+=ii.get("instructions",[])
    closes=[]
    for i in instrs+inner:
        if isinstance(i,dict) and i.get("parsed",{}).get("type")=="closeAccount":
            info=i["parsed"]["info"]
            closes.append(info.get("account","?")[:6])
    # which mints had ATAs closed? check pre/postTokenBalances owned by us
    pre={b["accountIndex"]:b for b in tx["meta"].get("preTokenBalances",[])}
    post={b["accountIndex"]:b for b in tx["meta"].get("postTokenBalances",[])}
    meme_mints_pre={b["mint"] for b in pre.values() if b["mint"]!=WSOL}
    meme_mints_post={b["mint"] for b in post.values() if b["mint"]!=WSOL}
    closed_meme = meme_mints_pre - meme_mints_post
    print(f"{ts} | {sig[:10]} | closeAccount instrs={len(closes)} | meme-ATA disappeared post-tx: {bool(closed_meme)} ({[m[:6] for m in closed_meme]})")
    time.sleep(0.05)
