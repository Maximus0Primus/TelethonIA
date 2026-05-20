"""Diagnose why CloseAccount fails: check token program (legacy vs 2022) + ATA derivation match."""
import os, requests
from dotenv import load_dotenv
from solders.pubkey import Pubkey
load_dotenv("scraper/.env")
KEY=os.environ["HELIUS_API_KEY"]; URL=f"https://mainnet.helius-rpc.com/?api-key={KEY}"
def rpc(m,p): return requests.post(URL,json={"jsonrpc":"2.0","id":1,"method":m,"params":p},timeout=25).json().get("result")
SEED="fFzwr3cgGMBqrGP7u8b1sHLi4oMgjWo2ZzNeDZPiZZAEePrBMJyYGSxy59RtNh1rb8Djren6uxALAxJknhoAAbF"
tx0=rpc("getTransaction",[SEED,{"maxSupportedTransactionVersion":0,"encoding":"jsonParsed"}])
owner=tx0["transaction"]["message"]["accountKeys"][0]["pubkey"]
LEGACY="TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
T2022="TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
ATA_PROG=Pubkey.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")
ownerpk=Pubkey.from_string(owner)

def derive(mint, tokenprog):
    ata,_=Pubkey.find_program_address([bytes(ownerpk),bytes(Pubkey.from_string(tokenprog)),bytes(Pubkey.from_string(mint))],ATA_PROG)
    return str(ata)

for prog,label in [(LEGACY,"legacy"),(T2022,"token2022")]:
    res=rpc("getTokenAccountsByOwner",[owner,{"programId":prog},{"encoding":"jsonParsed"}]) or {}
    for a in res.get("value",[]):
        info=a["account"]["data"]["parsed"]["info"]
        if int(info["tokenAmount"]["amount"])!=0: continue
        ata_actual=a["pubkey"]; mint=info["mint"]; prog_owner=a["account"]["owner"]
        code_derives=derive(mint, LEGACY)  # code ALWAYS uses legacy
        match = "MATCH" if code_derives==ata_actual else "MISMATCH"
        print(f"[{label}] mint={mint[:10]} ata={ata_actual[:10]} owner_prog={'2022' if prog_owner==T2022 else 'legacy'} | code(legacy)-derive={match}")
