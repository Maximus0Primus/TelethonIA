"""Per-KOL evidence for the 3 graduates: mad_apes, TheReaperGems, bat_gamble.

For each:
  - count live trades since Apr 26 graduation (chain split)
  - paper vs live drift on every strat where they have live N>=1
  - per-token detail to see if the loss is on copytrader-front-running pattern
    (= entry price drifts hard between paper sim and live fill = high paper
    high_price > live execution_price by big margin).
"""
import os, sys, statistics as st
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

CLOSED = ("sl_hit", "trail_stop", "tp_hit", "timeout", "be_stop")
SINCE = "2026-04-26T00:00:00+00:00"
TARGETS = ["mad_apes_gambles", "TheReaperGems", "bat_gamble"]


def fetch(kol, src):
    out, off, step = [], 0, 1000
    while True:
        r = sb.table("paper_trades").select(
            "strategy,status,source,pnl_pct,pnl_usd,kol_group,token_address,symbol,"
            "created_at,is_shadow,chain,entry_price,exit_price,high_price_seen,position_usd"
        ).eq("kol_group", kol).eq("source", src).gte("created_at", SINCE) \
         .range(off, off+step-1).execute()
        if not r.data:
            break
        out.extend(r.data)
        if len(r.data) < step:
            break
        off += step
    return out


for kol in TARGETS:
    print()
    print("=" * 100)
    print(f"  {kol}  since Apr 26 graduation")
    print("=" * 100)
    paper = fetch(kol, "rt")
    live = fetch(kol, "rt_live")
    paper_c = [r for r in paper if r.get("status") in CLOSED and not r.get("is_shadow")]
    live_c = [r for r in live if r.get("status") in CLOSED and not r.get("is_shadow")]
    print(f"  paper rows={len(paper)}  closed_main={len(paper_c)}  shadows={len(paper)-len(paper_c)}")
    print(f"  live  rows={len(live)}   closed={len(live_c)}")
    if not live:
        print("  -> NO LIVE TRADES at all")
        continue

    # chain split
    by_chain_l = defaultdict(int)
    for r in live:
        by_chain_l[r.get("chain") or "—"] += 1
    print(f"  live chains: {dict(by_chain_l)}")

    if not live_c:
        continue

    # per-strat drift
    by_strat_p = defaultdict(list)
    by_strat_l = defaultdict(list)
    for r in paper_c:
        by_strat_p[r["strategy"]].append(float(r.get("pnl_pct") or 0) * 100)
    for r in live_c:
        by_strat_l[r["strategy"]].append(float(r.get("pnl_pct") or 0) * 100)
    strats = sorted(set(by_strat_p) | set(by_strat_l))
    print(f"\n  per-strat (paper vs live):")
    print(f"  {'strat':<32}{'Np':>5}{'avgP':>9}{'Nl':>5}{'avgL':>9}{'drift':>9}{'$L':>9}")
    for s in strats:
        p, l = by_strat_p.get(s, []), by_strat_l.get(s, [])
        ap = f"{st.mean(p):+.2f}%" if p else "—"
        al = f"{st.mean(l):+.2f}%" if l else "—"
        d = (st.mean(l) - st.mean(p)) if (p and l) else None
        d_s = f"{d:+.1f}" if d is not None else "—"
        usd_l = sum(float(x.get("pnl_usd") or 0) for x in live_c if x["strategy"] == s)
        print(f"  {s:<32}{len(p):>5}{ap:>9}{len(l):>5}{al:>9}{d_s:>9}{usd_l:>+9.2f}")

    # token-level: paper high_price vs live entry — copytrader / front-run signal
    by_tok = defaultdict(lambda: {"paper": [], "live": []})
    for r in paper_c:
        by_tok[r["token_address"]]["paper"].append(r)
    for r in live_c:
        by_tok[r["token_address"]]["live"].append(r)
    tokens_with_both = [t for t, d in by_tok.items() if d["paper"] and d["live"]]
    print(f"\n  tokens traded in BOTH paper-main and live: {len(tokens_with_both)}")
    if tokens_with_both:
        print(f"  {'symbol':<14}{'strat':<26}{'entryP':>11}{'entryL':>11}{'fillD%':>8}{'pnlP%':>8}{'pnlL%':>8}")
        for t in tokens_with_both[:20]:
            d = by_tok[t]
            for rl in d["live"]:
                # find paper trade same strat same token
                rp = next((x for x in d["paper"] if x["strategy"] == rl["strategy"]), None)
                if not rp:
                    continue
                ep = float(rp.get("entry_price") or 0)
                el = float(rl.get("entry_price") or 0)
                fill_drift = ((el - ep) / ep * 100) if ep > 0 else 0
                pp = float(rp.get("pnl_pct") or 0) * 100
                pl = float(rl.get("pnl_pct") or 0) * 100
                sym = (rl.get("symbol") or "")[:13]
                print(f"  {sym:<14}{rl['strategy'][:25]:<26}{ep:>11.6f}{el:>11.6f}{fill_drift:>+7.2f}%{pp:>+7.2f}%{pl:>+7.2f}%")
