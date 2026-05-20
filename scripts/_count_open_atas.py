"""Count open token ATAs in the live wallet + rent locked. Definitive close_ata health check."""
import os, requests
from dotenv import load_dotenv
load_dotenv("scraper/.env")
KEY=os.environ["HELIUS_API_KEY"]; URL=f"https://mainnet.helius-rpc.com/?api-key={KEY}"
def rpc(m,p): return requests.post(URL,json={"jsonrpc":"2.0","id":1,"method":m,"params":p},timeout=25).json().get("result")
SEED="fFzwr3cgGMBqrGP7u8b1sHLi4oMgjWo2ZzNeDZPiZZAEePrBMJyYGSxy59RtNh1rb8Djren6uxALAxJknhoAAbF"
tx0=rpc("getTransaction",[SEED,{"maxSupportedTransactionVersion":0,"encoding":"jsonParsed"}])
owner=tx0["transaction"]["message"]["accountKeys"][0]["pubkey"]
TOKEN="TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
res=rpc("getTokenAccountsByOwner",[owner,{"programId":TOKEN},{"encoding":"jsonParsed"}]) or {}
accts=res.get("value",[])
empty=nonempty=0; rent_locked=0.0
for a in accts:
    info=a["account"]["data"]["parsed"]["info"]
    amt=int(info["tokenAmount"]["amount"])
    lam=a["account"]["lamports"]/1e9
    if amt==0: empty+=1; rent_locked+=lam
    else: nonempty+=1
print(f"Wallet: {owner}")
print(f"Open token ATAs: {len(accts)}  (empty={empty}, still-holding={nonempty})")
print(f"Rent locked in EMPTY ATAs (closeable, not recovered): {rent_locked:.5f} SOL = ${rent_locked*84.5:.2f}")
print(f"  -> {empty} empty ATAs x 0.00204 = each one is $0.17 of unrecovered rent")
