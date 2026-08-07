"""Meme replay, mais on capture PAR QUEL CHEMIN le sweep sort."""
import os, sys
from datetime import datetime, timedelta
sys.path.insert(0, "/opt/TelethonIA/scraper"); os.chdir("/opt/TelethonIA/scraper")
from dotenv import load_dotenv; load_dotenv()
from supabase import create_client
from paper_trader import _evaluate_trade_exit, _last_eval_ts, _dynamic_sell_slip_factor
from sim import _mega_poll_offsets, _mega_latest_at_or_before, _mega_smooth, _MegaSmState, _MEGA_SELL_SLIP_BASE

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
STRAT, TP, SL, H, BE = "BE25_TP80_SL30", 1.80, 0.70, 30, 0.25

def replay(jp, ds, entry_price, entry_iso, liq):
    """Copie de _mega_replay_one, instrumentee."""
    entry_time = datetime.fromisoformat(entry_iso.replace("Z","+00:00"))
    tid = f"d_{id(jp)}"; _last_eval_ts.pop(tid, None)
    offs = _mega_poll_offsets("lazy_fast", H*60)
    ft = {"id": tid, "entry_price": entry_price, "sl_price": entry_price*SL,
          "tp_price": entry_price*TP, "position_usd": 10.0, "strategy": STRAT,
          "tranche_label": "main", "horizon_minutes": H, "created_at": entry_iso,
          "high_price_seen": entry_price, "rt_liquidity_usd": liq,
          "dex_spot_price_at_entry": entry_price}
    st = _MegaSmState(); last = None
    for o in offs:
        pt = entry_time + timedelta(seconds=o)
        p = _mega_latest_at_or_before(jp, pt.isoformat().replace("+00:00","Z"))
        if p is None: continue
        last = p
        ev = _evaluate_trade_exit(ft, p, pt, _MEGA_SELL_SLIP_BASE, sell_fee_bps=0,
                                  decision_price=_mega_smooth(st, p, "raw", ft["sl_price"], ft["tp_price"]))
        if ev is None: continue
        if ev.get("high_price_seen"): ft["high_price_seen"] = max(ft["high_price_seen"], ev["high_price_seen"])
        if ev.get("status"): return ev["status"], ev.get("pnl_pct", 0), p
    return "FALLTHROUGH", (round(last/entry_price-1,4) if last else None), last

rows = (sb.table("paper_trades")
        .select("token_address,created_at,entry_price,exit_price,pnl_pct,status,rt_liquidity_usd")
        .eq("strategy",STRAT).eq("chain","solana").eq("source","rt").neq("status","open")
        .gte("created_at","2026-06-01").order("created_at",desc=True).limit(150).execute().data)
seen, res = set(), []
for r in rows:
    a = r["token_address"]
    if a in seen or not r.get("entry_price") or r.get("pnl_pct") is None: continue
    seen.add(a)
    e = datetime.fromisoformat(r["created_at"].replace("Z","+00:00"))
    tk = (sb.table("price_ticks").select("price_usd,fetched_at,source").eq("token_address",a)
          .eq("chain","solana").gte("fetched_at",(e-timedelta(minutes=5)).isoformat())
          .lte("fetched_at",(e+timedelta(minutes=H+30)).isoformat())
          .order("fetched_at").limit(3000).execute().data)
    jp = sorted([t for t in tk if t["source"]=="jupiter"], key=lambda t:t["fetched_at"])
    if not jp: continue
    stt, pnl, lastp = replay(jp, [], float(r["entry_price"]), r["created_at"], r.get("rt_liquidity_usd"))
    if pnl is None: continue
    res.append({"reel_st": r["status"], "reel": float(r["pnl_pct"]), "sw_st": stt, "sw": float(pnl),
                "liq": r.get("rt_liquidity_usd")})

import statistics as s
print(f"{len(res)} paires\n")
print("=== chemin de sortie: REEL -> SWEEP ===")
m = {}
for x in res: m.setdefault((x["reel_st"], x["sw_st"]), []).append(x)
for k, v in sorted(m.items(), key=lambda kv:-len(kv[1])):
    print(f"  {k[0]:<10} -> {k[1]:<12} n={len(v):>3}  reel {100*s.mean([y['reel'] for y in v]):+7.2f}  "
          f"sweep {100*s.mean([y['sw'] for y in v]):+7.2f}")
liq0 = [x for x in res if not x["liq"]]
print(f"\ntokens a liquidite nulle/absente: {len(liq0)}/{len(res)}")
if liq0:
    print(f"  ecart moyen sur ceux-la : {100*s.mean([x['sw']-x['reel'] for x in liq0]):+.2f} pp")
liqok = [x for x in res if x["liq"]]
if liqok:
    print(f"  ecart moyen sur les autres: {100*s.mean([x['sw']-x['reel'] for x in liqok]):+.2f} pp")
