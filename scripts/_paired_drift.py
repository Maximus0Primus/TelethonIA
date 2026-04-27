"""Apples-to-apples drift: paired test on tokens that BOTH paper-main and live
traded for the same strategy in the same time window.

Aggregate drift = mean(all live) - mean(all paper) is biased by selection (live
fires on a subset of tokens). The CORRECT metric is per-token paired drift:
  for each token traded by BOTH sides, drift_t = pnl_live(t) - pnl_paper(t)
  paired_drift = mean(drift_t over intersection tokens)
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
A_FROM, A_TO = "2026-04-22T00:00:00+00:00", "2026-04-25T23:59:59+00:00"
B_FROM, B_TO = "2026-04-26T00:00:00+00:00", "2026-04-27T23:59:59+00:00"
STRAT = "BE25_TP80_SL30"


def fetch(src, since, until):
    out, off, step = [], 0, 1000
    while True:
        r = sb.table("paper_trades").select(
            "strategy,status,source,pnl_pct,kol_group,token_address,symbol,created_at,is_shadow,chain"
        ).eq("strategy", STRAT).eq("source", src).eq("chain", "solana") \
         .gte("created_at", since).lte("created_at", until) \
         .range(off, off+step-1).execute()
        if not r.data:
            break
        out.extend(r.data)
        if len(r.data) < step:
            break
        off += step
    return [r for r in out if r.get("status") in CLOSED and not r.get("is_shadow")]


def paired(paper, live):
    """Group by token_address. Keep only tokens with at least 1 trade each side.
    For multiple trades same token same side, take the first (by created_at)."""
    p_by_tok = {}
    l_by_tok = {}
    for r in sorted(paper, key=lambda x: x["created_at"]):
        p_by_tok.setdefault(r["token_address"], r)
    for r in sorted(live, key=lambda x: x["created_at"]):
        l_by_tok.setdefault(r["token_address"], r)
    common = sorted(set(p_by_tok) & set(l_by_tok))
    pairs = []
    for tok in common:
        rp, rl = p_by_tok[tok], l_by_tok[tok]
        pp = float(rp.get("pnl_pct") or 0) * 100
        pl = float(rl.get("pnl_pct") or 0) * 100
        pairs.append((tok, rp.get("symbol", "")[:10], rp.get("kol_group") or "—", pp, pl, pl - pp))
    return pairs


for label, since, until in (("A (Apr 22-25)", A_FROM, A_TO),
                            ("B (Apr 26-27)", B_FROM, B_TO)):
    print()
    print("=" * 100)
    print(f"WINDOW {label}  strat={STRAT}  PAIRED on intersection tokens")
    print("=" * 100)
    paper = fetch("rt", since, until)
    live = fetch("rt_live", since, until)
    pairs = paired(paper, live)
    if not pairs:
        print(f"  no intersection. Nlive={len(live)} Npaper={len(paper)}")
        continue
    drifts = [p[5] for p in pairs]
    pnl_p = [p[3] for p in pairs]
    pnl_l = [p[4] for p in pairs]
    print(f"  intersection N={len(pairs)}  Npaper_total={len(paper)} Nlive_total={len(live)}")
    print(f"  paired drift  mean={st.mean(drifts):+.2f}pp  median={st.median(drifts):+.2f}pp  std={st.pstdev(drifts):.2f}pp")
    print(f"  paper avg on intersection  = {st.mean(pnl_p):+.2f}%")
    print(f"  live  avg on intersection  = {st.mean(pnl_l):+.2f}%")
    print(f"  AGGREGATE (biased)  paper_all={st.mean([float(r.get('pnl_pct') or 0)*100 for r in paper]):+.2f}%  "
          f"live_all={st.mean([float(r.get('pnl_pct') or 0)*100 for r in live]):+.2f}%  "
          f"diff={st.mean([float(r.get('pnl_pct') or 0)*100 for r in live])-st.mean([float(r.get('pnl_pct') or 0)*100 for r in paper]):+.2f}pp")
    print(f"\n  per-pair detail:")
    print(f"  {'symbol':<12}{'KOL':<22}{'paper%':>9}{'live%':>9}{'drift':>9}")
    for tok, sym, kol, pp, pl, d in sorted(pairs, key=lambda x: x[5]):
        print(f"  {sym:<12}{kol[:21]:<22}{pp:>+8.2f}%{pl:>+8.2f}%{d:>+8.2f}")
    # drift excluding worst single outlier
    if len(drifts) >= 2:
        sorted_d = sorted(drifts)
        excl_min = sorted_d[1:]
        print(f"\n  paired drift excluding worst outlier: mean={st.mean(excl_min):+.2f}pp  N={len(excl_min)}")
