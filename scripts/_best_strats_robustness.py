"""Cross-window robustness 14d/7d/3d — find best SOL + ETH strats for live promotion.
A strat is ROBUST if positive on all 3 windows AND N>=15 (or N>=10 for shorter window).
Sorts by 14d $/d weighted by robustness."""
import os, sys, statistics as st
from collections import defaultdict
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

NOW = datetime.now(timezone.utc)
SINCE_14D = (NOW - timedelta(days=14)).isoformat()
SINCE_7D = (NOW - timedelta(days=7)).isoformat()
SINCE_3D = (NOW - timedelta(days=3)).isoformat()
CLOSED = ("sl_hit", "trail_stop", "tp_hit", "timeout", "be_stop")

def fetch(chain, since):
    out, off, step = [], 0, 1000
    while True:
        r = sb.table("paper_trades").select(
            "strategy,status,pnl_pct,pnl_usd,token_address,created_at,is_shadow"
        ).eq("source","rt").eq("chain",chain).gte("created_at", since).range(off, off+step-1).execute()
        if not r.data:
            break
        out.extend(r.data)
        if len(r.data) < step:
            break
        off += step
    return [r for r in out if r.get("status") in CLOSED]


def aggregate_by_strat(rows, days):
    by_strat = defaultdict(list)
    for r in rows:
        by_strat[r["strategy"]].append(r)
    out = {}
    for s, rs in by_strat.items():
        if not rs:
            continue
        pcts = [float(r.get("pnl_pct") or 0)*100 for r in rs]
        usd = sum(float(r.get("pnl_usd") or 0) for r in rs)
        wr = 100*sum(1 for p in pcts if p > 0)/len(rs)
        is_shadow = sum(1 for r in rs if r.get("is_shadow")) / len(rs)  # fraction shadow
        out[s] = {
            "N": len(rs),
            "WR": wr,
            "avg": st.mean(pcts),
            "med": st.median(pcts),
            "$tot": round(usd, 2),
            "$/d": round(usd/days, 2),
            "shadow_frac": is_shadow,
        }
    return out


def analyze(chain):
    print(f"\n{'='*110}")
    print(f"CHAIN = {chain.upper()} — robustness cross-window")
    print(f"{'='*110}")
    rows14 = fetch(chain, SINCE_14D)
    rows7 = [r for r in rows14 if r["created_at"] >= SINCE_7D]
    rows3 = [r for r in rows14 if r["created_at"] >= SINCE_3D]
    print(f"\nTotals — 14d: {len(rows14)}, 7d: {len(rows7)}, 3d: {len(rows3)}")

    s14 = aggregate_by_strat(rows14, 14)
    s7 = aggregate_by_strat(rows7, 7)
    s3 = aggregate_by_strat(rows3, 3)

    # ROBUST = positive on all 3 windows AND N >= 10 each window AND N>=15 cumul
    candidates = []
    for s in s14.keys():
        if s14[s]["N"] < 15:
            continue
        d14 = s14[s]
        d7 = s7.get(s)
        d3 = s3.get(s)
        # Need data on all 3 windows
        if not d7 or not d3:
            continue
        if d7["N"] < 8 or d3["N"] < 5:
            continue
        # Robust = $/d positive on all 3 windows
        all_positive = (d14["$/d"] > 0) and (d7["$/d"] > 0) and (d3["$/d"] > 0)
        candidates.append({
            "strat": s,
            "robust": all_positive,
            "14d": d14,
            "7d": d7,
            "3d": d3,
            "is_paper_main": d14["shadow_frac"] < 0.5,
        })

    # Sort by robustness then by 14d $/d
    candidates.sort(key=lambda c: (-int(c["robust"]), -c["14d"]["$/d"]))

    # Print
    print(f"\nROBUST candidates (positive $/d on 14d AND 7d AND 3d, N>=15/10/5):")
    print(f"  {'strategy':<46}{'main':>5}{'N14d':>5}{'$/d14':>7}{'$/d7':>7}{'$/d3':>7}{'WR14':>6}{'avg14':>8}{'med14':>8}")
    n_robust = 0
    for c in candidates:
        if not c["robust"]:
            continue
        n_robust += 1
        if n_robust > 30:
            break
        m = "*" if c["is_paper_main"] else " "
        print(f"  {c['strat']:<46}{m:>5}{c['14d']['N']:>5}{c['14d']['$/d']:>+7.2f}{c['7d']['$/d']:>+7.2f}{c['3d']['$/d']:>+7.2f}{c['14d']['WR']:>5.0f}%{c['14d']['avg']:>+7.2f}%{c['14d']['med']:>+7.2f}%")
    print(f"\n  >>> {n_robust} robust candidates (positive sur 3 windows)")

    # Now show non-robust top 14d (positives 14d but failing on 7d or 3d) — these are FADING
    print(f"\nFADING strats (positive 14d but went negative on 7d or 3d — probably regime-dependent):")
    print(f"  {'strategy':<46}{'main':>5}{'N14d':>5}{'$/d14':>7}{'$/d7':>7}{'$/d3':>7}")
    fading = [c for c in candidates if not c["robust"] and c["14d"]["$/d"] > 0]
    fading.sort(key=lambda c: -c["14d"]["$/d"])
    for c in fading[:15]:
        m = "*" if c["is_paper_main"] else " "
        print(f"  {c['strat']:<46}{m:>5}{c['14d']['N']:>5}{c['14d']['$/d']:>+7.2f}{c['7d']['$/d']:>+7.2f}{c['3d']['$/d']:>+7.2f}")

    # And EMERGING — robust 7d+3d but small/negative 14d (newer signal)
    print(f"\nEMERGING strats (positive 7d AND 3d, N small or 14d weak):")
    print(f"  {'strategy':<46}{'main':>5}{'N14d':>5}{'$/d14':>7}{'$/d7':>7}{'$/d3':>7}")
    emerging = []
    for c in candidates:
        if c["robust"]:
            continue
        if c["7d"]["$/d"] > 0 and c["3d"]["$/d"] > 0 and c["14d"]["$/d"] <= 0:
            emerging.append(c)
    emerging.sort(key=lambda c: -c["3d"]["$/d"])
    for c in emerging[:15]:
        m = "*" if c["is_paper_main"] else " "
        print(f"  {c['strat']:<46}{m:>5}{c['14d']['N']:>5}{c['14d']['$/d']:>+7.2f}{c['7d']['$/d']:>+7.2f}{c['3d']['$/d']:>+7.2f}")

    return candidates


# Run on both chains
sol_candidates = analyze("solana")
eth_candidates = analyze("ethereum")

# Final shortlist for live
print(f"\n\n{'='*110}\nSHORTLIST LIVE PROMOTION — top 5 SOL + top 5 ETH (robust + N>=20 + WR>=45%)\n{'='*110}")
def shortlist(cands, label):
    qualified = [c for c in cands if c["robust"]
                 and c["14d"]["N"] >= 20 and c["14d"]["WR"] >= 45]
    qualified.sort(key=lambda c: -c["14d"]["$/d"])
    print(f"\n{label}:")
    if not qualified:
        print(f"  AUCUN candidat ne passe les criteres robust+N20+WR45")
        return []
    print(f"  {'strategy':<46}{'main':>5}{'N':>4}{'WR':>5}{'$/d 14d':>9}{'$/d 7d':>8}{'$/d 3d':>8}")
    for c in qualified[:5]:
        m = "*" if c["is_paper_main"] else " "
        print(f"  {c['strat']:<46}{m:>5}{c['14d']['N']:>4}{c['14d']['WR']:>4.0f}%{c['14d']['$/d']:>+8.2f}{c['7d']['$/d']:>+7.2f}{c['3d']['$/d']:>+7.2f}")
    return qualified[:5]

shortlist(sol_candidates, "SOL")
shortlist(eth_candidates, "ETH")
