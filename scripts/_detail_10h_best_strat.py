"""Detail every paper RT shadow trade in the last 10h for the best strat per chain.
Shows token-by-token PnL and aggregates total $ at $50/trade paper position.
"""
import os, sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

NOW = datetime.now(timezone.utc)
SINCE = (NOW - timedelta(hours=10)).isoformat()
CLOSED = ("sl_hit", "trail_stop", "tp_hit", "timeout", "be_stop")


def fetch_all(tbl, sel, **filters):
    out, step, off = [], 1000, 0
    while True:
        q = sb.table(tbl).select(sel).gte("created_at", SINCE)
        for k, v in filters.items():
            q = q.eq(k, v)
        r = q.range(off, off + step - 1).execute()
        if not r.data:
            break
        out.extend(r.data)
        if len(r.data) < step:
            break
        off += step
    return out


def show(chain_db, chain_label, candidates):
    rows = fetch_all(
        "paper_trades",
        "strategy,status,source,pnl_pct,pnl_usd,chain,created_at,symbol,token_address,entry_price,exit_price,position_usd,exit_at,exit_minutes",
        source="rt", chain=chain_db,
    )
    print()
    print("=" * 100)
    print(f"{chain_label} — last 10h, top candidate strats")
    print("=" * 100)

    for strat in candidates:
        sub = [r for r in rows if r.get("strategy") == strat and r.get("status") in CLOSED]
        if not sub:
            print(f"\n[{strat}]  no closed trades in 10h")
            continue
        sub.sort(key=lambda r: r.get("created_at") or "")
        print(f"\n[{strat}]  N={len(sub)}")
        print(f"  {'opened (UTC)':<20}{'symbol':<14}{'status':<13}{'pnl%':>9}{'$pnl':>9}{'pos$':>8}{'mins':>7}")
        total_pnl = 0.0
        total_pos = 0.0
        for r in sub:
            sym = (r.get("symbol") or r.get("token_address") or "?")[:12]
            st_ = (r.get("status") or "?")
            pct = float(r.get("pnl_pct") or 0) * 100
            usd = float(r.get("pnl_usd") or 0)
            pos = float(r.get("position_usd") or 0)
            mins = r.get("exit_minutes")
            mins_s = f"{int(mins)}" if mins is not None else "—"
            opened = (r.get("created_at") or "")[:19].replace("T", " ")
            print(f"  {opened:<20}{sym:<14}{st_:<13}{pct:>+8.2f}%{usd:>+8.2f}{pos:>8.2f}{mins_s:>7}")
            total_pnl += usd
            total_pos += pos
        n = len(sub)
        wr = 100 * sum(1 for r in sub if float(r.get("pnl_pct") or 0) > 0) / n
        avg = sum(float(r.get("pnl_pct") or 0) * 100 for r in sub) / n
        print(f"  {'TOTAL':<20}{'':<14}{'':<13}{avg:>+8.2f}%{total_pnl:>+8.2f}{total_pos:>8.2f}{'':>7}"
              f"   WR {wr:.0f}%   $/h {total_pnl/10:+.2f}   $/j {total_pnl*24/10:+.2f}")


# Best of each chain on 10h (per the previous ranking) + the actual live strats for context
show("solana", "SOL", [
    "DECAY_TP100_SL50_E30",   # #1 10h
    "TP60_SL50",              # #2 10h, 100% WR
    "BE15_TP70_SL50_NZ",      # #3 10h
    "TP100_NOSL",             # #4
    "BE25_TP80_SL30",         # actual live (for compare)
    "BOND_FAST_TP50_SL20_T20",# actual live
])

show("ethereum", "ETH", [
    "ETH_SLOW6H_TP200_SL40_NZ_S40",  # #1 10h
    "ETH_BE35_LOCK15_TP150_SL40_T2H",
    "ETH_BE25_LOCK10_TP100_SL20",
    "ETH_FAST_TP100_SL20",            # 7d/3d winner
    "ETH_TP80_SL40_T2H",              # ALL winner
])
