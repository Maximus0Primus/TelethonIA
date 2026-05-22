"""Count open token ATAs in the live wallet + rent locked. Definitive close_ata health check.
v14e.68: covers BOTH token programs (legacy SPL + Token-2022). The prior version queried
only the legacy program and silently under-reported Token-2022 stuck rent (e.g. showed 6
when 17 ATAs were actually open)."""
import os, requests
from dotenv import load_dotenv
load_dotenv("scraper/.env")
KEY=os.environ["HELIUS_API_KEY"]; URL=f"https://mainnet.helius-rpc.com/?api-key={KEY}"
def rpc(m,p): return requests.post(URL,json={"jsonrpc":"2.0","id":1,"method":m,"params":p},timeout=25).json().get("result")
SEED="fFzwr3cgGMBqrGP7u8b1sHLi4oMgjWo2ZzNeDZPiZZAEePrBMJyYGSxy59RtNh1rb8Djren6uxALAxJknhoAAbF"
tx0=rpc("getTransaction",[SEED,{"maxSupportedTransactionVersion":0,"encoding":"jsonParsed"}])
owner=tx0["transaction"]["message"]["accountKeys"][0]["pubkey"]
PROGRAMS={"legacy":"TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
          "token2022":"TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"}
print(f"Wallet: {owner}")
tot=tot_empty=tot_nonempty=0; tot_rent=0.0
for name,pid in PROGRAMS.items():
    res=rpc("getTokenAccountsByOwner",[owner,{"programId":pid},{"encoding":"jsonParsed"}]) or {}
    accts=res.get("value",[])
    empty=nonempty=0; rent=0.0
    for a in accts:
        info=a["account"]["data"]["parsed"]["info"]
        amt=int(info["tokenAmount"]["amount"])
        lam=a["account"]["lamports"]/1e9
        if amt==0: empty+=1; rent+=lam
        else: nonempty+=1
    tot+=len(accts); tot_empty+=empty; tot_nonempty+=nonempty; tot_rent+=rent
    print(f"  [{name:9}] total={len(accts):3}  empty(closeable)={empty:3}  still-holding={nonempty:3}  rent_locked={rent:.5f} SOL")
print(f"TOTAL open token ATAs: {tot}  (empty={tot_empty}, still-holding={tot_nonempty})")
print(f"Rent locked in EMPTY ATAs (closeable, not recovered): {tot_rent:.5f} SOL = ${tot_rent*84.5:.2f}")
print(f"  -> {tot_empty} empty ATAs x 0.00204 = each one is ~$0.17 of unrecovered rent")
