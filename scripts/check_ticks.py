import os
from supabase import create_client
from collections import Counter

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

res = sb.table("price_ticks").select("token_address,price_usd,source").order("id", desc=True).limit(30).execute()
for r in res.data:
    addr = r["token_address"][:12]
    src = r["source"]
    price = r["price_usd"]
    print(f"  {addr}... | {src:8s} | {price}")

c = Counter(r["source"] for r in res.data)
print(f"\nSource distribution (last 30): {dict(c)}")

# Check same token has both dex and jupiter
tokens = {}
for r in res.data:
    a = r["token_address"]
    s = r["source"]
    tokens.setdefault(a, set()).add(s)

print("\nPer-token coverage:")
for a, sources in tokens.items():
    print(f"  {a[:12]}... : {sources}")
