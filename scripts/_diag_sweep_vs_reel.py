"""Rejoue le sweep sur des trades REELS et compare, trade par trade."""
import os, sys, json
from datetime import datetime, timedelta
sys.path.insert(0, "/opt/TelethonIA/scraper")
os.chdir("/opt/TelethonIA/scraper")
from dotenv import load_dotenv; load_dotenv()
from supabase import create_client
import sim
from sim import _mega_replay_one

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
STRAT, TP, SL, H, BE = "BE25_TP80_SL30", 1.80, 0.70, 30, 0.25

rows = (sb.table("paper_trades")
        .select("token_address,created_at,entry_price,exit_price,pnl_pct,status,exit_minutes,rt_liquidity_usd")
        .eq("strategy", STRAT).eq("chain", "solana").eq("source", "rt")
        .neq("status", "open").gte("created_at", "2026-06-01")
        .order("created_at", desc=True).limit(120).execute().data)
rows = [r for r in rows if r.get("entry_price") and r.get("pnl_pct") is not None]
print(f"{len(rows)} trades reels")

seen, res = set(), []
for r in rows:
    a = r["token_address"]
    if a in seen: continue
    seen.add(a)
    e = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
    tk = (sb.table("price_ticks").select("price_usd,fetched_at,source")
          .eq("token_address", a).eq("chain", "solana")
          .gte("fetched_at", (e - timedelta(minutes=5)).isoformat())
          .lte("fetched_at", (e + timedelta(minutes=H + 30)).isoformat())
          .order("fetched_at").limit(3000).execute().data)
    jp = sorted([t for t in tk if t["source"] == "jupiter"], key=lambda t: t["fetched_at"])
    ds = sorted([t for t in tk if t["source"] in ("fast","full","live")], key=lambda t: t["fetched_at"])
    if not jp: continue
    p = _mega_replay_one(TP, SL, H, BE, jp, ds, float(r["entry_price"]),
                         r["created_at"], "jupiter", "raw", "lazy_fast",
                         r.get("rt_liquidity_usd"), STRAT)
    if p is None: continue
    res.append({"tok": a[:8], "reel": float(r["pnl_pct"]), "sweep": float(p),
                "raison": r.get("status"), "n_jp": len(jp)})

import statistics as st
print(f"\n{len(res)} paires rejouees")
print(f"  EV reelle : {100*st.mean([x['reel'] for x in res]):+.2f} %")
print(f"  EV sweep  : {100*st.mean([x['sweep'] for x in res]):+.2f} %")
print(f"  ECART     : {100*st.mean([x['sweep']-x['reel'] for x in res]):+.2f} pp")
print(f"  ticks jupiter par trade: median {st.median([x['n_jp'] for x in res]):.0f}")
print("\n=== ecart par raison de sortie REELLE ===")
par = {}
for x in res: par.setdefault(x["raison"], []).append(x)
for k, v in sorted(par.items(), key=lambda kv: -len(kv[1])):
    print(f"  {str(k):<12} n={len(v):>3}  reel {100*st.mean([y['reel'] for y in v]):+7.2f}  "
          f"sweep {100*st.mean([y['sweep'] for y in v]):+7.2f}  "
          f"ecart {100*st.mean([y['sweep']-y['reel'] for y in v]):+7.2f} pp")
print("\n=== 8 plus gros ecarts ===")
for x in sorted(res, key=lambda y: -(y["sweep"]-y["reel"]))[:8]:
    print(f"  {x['tok']}  reel {100*x['reel']:+8.2f}  sweep {100*x['sweep']:+8.2f}  "
          f"({x['raison']}, {x['n_jp']} ticks)")
