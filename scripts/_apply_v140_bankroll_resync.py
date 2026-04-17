"""v140 bankroll re-sync — inclure les trades pré-reset qui ont été wipés.

Contexte : v140 reset à 16:55 UTC a wipé les gains du call ReaperGems $ELONMUSK
qui avait généré +$819 sur 6 strats actives avant le reset. Puis le trade $DFV
de caniscooks à 17:19 a perdu -$194. Re-sync depuis 16:00 UTC pour tout
récupérer.

Méthode :
  - reset effectif = 2026-04-17T16:00:00Z (avant ReaperGems)
  - Pour chaque strategy : somme pnl_usd des trades MAIN fermés depuis ce timestamp
  - Appliquer à rt_bankroll.strategy_bankrolls[strategy].{pnl, trades, balance}
  - Nouvelles strats v140 (8 HYST/NZS30/MCAP variants) : pas concernées,
    restent à $1000 (n'existaient pas avant 16:55)

Run : python scripts/_apply_v140_bankroll_resync.py [--apply]
"""
from __future__ import annotations
import os
import sys
from datetime import datetime, timezone
from collections import defaultdict

from dotenv import load_dotenv
from supabase import create_client

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SCRAPER = os.path.join(ROOT, "scraper")
load_dotenv(os.path.join(SCRAPER, ".env"))
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

RESET_CUTOFF = "2026-04-17T16:00:00Z"  # before ReaperGems $ELONMUSK call
PER_STRAT_BANKROLL = 1000.0


def main(apply: bool):
    br = sb.table("rt_bankroll").select("*").eq("id", 1).execute().data[0]
    sb_alloc = dict(br.get("strategy_bankrolls") or {})

    print(f"Fetching closed MAIN trades since {RESET_CUTOFF}...")
    rows = []
    off = 0
    while True:
        r = (sb.table("paper_trades")
               .select("strategy,pnl_usd,pnl_pct,position_usd,exit_at,is_shadow,status")
               .eq("source", "rt").eq("is_shadow", False)
               .gte("exit_at", RESET_CUTOFF)
               .order("exit_at").range(off, off + 999).execute().data)
        if not r: break
        rows.extend(r)
        if len(r) < 1000: break
        off += 1000
    closed = [r for r in rows if r.get("status") in
              ("tp_hit", "sl_hit", "timeout", "trail_stop", "trail_crash")]
    print(f"  {len(closed)} closed main trades post-cutoff")

    # Aggregate per strategy
    agg = defaultdict(lambda: {"pnl": 0.0, "trades": 0})
    for t in closed:
        s = t.get("strategy")
        if not s: continue
        agg[s]["pnl"] += float(t.get("pnl_usd") or 0)
        agg[s]["trades"] += 1

    print(f"\n=== Per-strategy PnL since {RESET_CUTOFF} ===")
    total_pnl = 0.0
    total_trades = 0
    new_alloc = dict(sb_alloc)
    for s in sorted(sb_alloc.keys()):
        agg_data = agg.get(s, {"pnl": 0.0, "trades": 0})
        new_pnl = round(agg_data["pnl"], 2)
        new_trades = agg_data["trades"]
        new_balance = round(PER_STRAT_BANKROLL + new_pnl, 2)
        total_pnl += new_pnl
        total_trades += new_trades
        old_pnl = sb_alloc[s].get("pnl", 0)
        delta = new_pnl - old_pnl
        flag = "⭐" if new_pnl > 50 else ("⚠️" if new_pnl < -50 else "  ")
        print(f"  {flag} {s:<32}  N={new_trades:>2}  pnl=${new_pnl:>+8.2f}  (was ${old_pnl:>+7.2f}, Δ${delta:>+7.2f})")
        new_alloc[s] = {
            "pnl": new_pnl, "trades": new_trades,
            "balance": new_balance,
            "starting_balance": PER_STRAT_BANKROLL,
        }

    # Global
    new_total = PER_STRAT_BANKROLL * len(sb_alloc)
    new_current = round(new_total + total_pnl, 2)
    new_peak = max(new_current, float(br.get("peak_balance", new_total)))
    new_dd_pct = round(-min(0, (new_current - new_peak) / new_peak * 100), 2)

    print(f"\n=== Global bankroll ===")
    print(f"OLD  starting=${br['starting_capital']:.0f}  current=${br['current_balance']:.2f}  pnl=${br['total_pnl']:+.2f}  trades={br['total_trades']}")
    print(f"NEW  starting=${new_total:.0f}  current=${new_current:.2f}  pnl=${total_pnl:+.2f}  trades={total_trades}")

    if not apply:
        print("\nDRY RUN — pass --apply to commit.")
        return

    sb.table("rt_bankroll").update({
        "starting_capital": new_total,
        "current_balance": new_current,
        "total_trades": total_trades,
        "total_pnl": round(total_pnl, 2),
        "peak_balance": new_peak,
        "max_drawdown_pct": new_dd_pct,
        "strategy_bankrolls": new_alloc,
        "last_updated_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", 1).execute()
    print(f"\n✅ APPLIED. Bankroll resynced from {RESET_CUTOFF}.")


if __name__ == "__main__":
    main("--apply" in sys.argv)
