"""Root cause: why did per-KOL drift on BE25 explode Apr 26-27 vs Apr 22-25?

Two windows:
  - WINDOW A (good): Apr 22 00:00 -> Apr 25 23:59 UTC (drift ~ -3pp avg)
  - WINDOW B (bad ): Apr 26 00:00 -> Apr 27 23:59 UTC (drift ~ -15pp avg)

For each KOL present in BOTH windows: compute drift(A) and drift(B). If a KOL
that was 0pp in A is suddenly -20pp in B, the explanation is exec/data quality,
NOT KOL composition.

Also: enumerate KOLs PRESENT only in B (newly-added Apr 26 5aa8ee5 commit).
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


def fetch(src, strat, since, until):
    out, off, step = [], 0, 1000
    while True:
        r = sb.table("paper_trades").select(
            "strategy,status,source,pnl_pct,kol_group,token_address,created_at,is_shadow,chain,entry_price,position_usd"
        ).eq("strategy", strat).eq("source", src).eq("chain", "solana") \
         .gte("created_at", since).lte("created_at", until) \
         .range(off, off+step-1).execute()
        if not r.data:
            break
        out.extend(r.data)
        if len(r.data) < step:
            break
        off += step
    return [r for r in out if r.get("status") in CLOSED and not r.get("is_shadow")]


def per_kol(rows):
    d = defaultdict(list)
    for r in rows:
        d[r.get("kol_group") or "—"].append(float(r.get("pnl_pct") or 0) * 100)
    return d


for strat in ("BE25_TP80_SL30",):
    print()
    print("=" * 100)
    print(f"WINDOW COMPARE  {strat}")
    print(f"  A = Apr 22-25 (good)   B = Apr 26-27 (bad)")
    print("=" * 100)

    A_p = per_kol(fetch("rt", strat, A_FROM, A_TO))
    A_l = per_kol(fetch("rt_live", strat, A_FROM, A_TO))
    B_p = per_kol(fetch("rt", strat, B_FROM, B_TO))
    B_l = per_kol(fetch("rt_live", strat, B_FROM, B_TO))

    # Drift per KOL each window
    def drift(p, l, k):
        pp, ll = p.get(k, []), l.get(k, [])
        if not pp or not ll:
            return None, len(pp), len(ll)
        return st.mean(ll) - st.mean(pp), len(pp), len(ll)

    all_kols = set(A_p) | set(A_l) | set(B_p) | set(B_l)
    rows = []
    for k in all_kols:
        dA, NpA, NlA = drift(A_p, A_l, k)
        dB, NpB, NlB = drift(B_p, B_l, k)
        in_A = (NpA + NlA) > 0
        in_B = (NpB + NlB) > 0
        rows.append((k, in_A, in_B, dA, NpA, NlA, dB, NpB, NlB))

    # SECTION 1: KOLs in BOTH windows — drift evolution A vs B
    print("\n[1] KOLs in BOTH windows (compare drift A vs B):")
    print(f"  {'KOL':<28}{'NpA':>4}{'NlA':>4}{'driftA':>9}{'NpB':>4}{'NlB':>4}{'driftB':>9}{'d_drift':>9}")
    both = [r for r in rows if r[1] and r[2]]
    both.sort(key=lambda r: (r[6] - r[3]) if (r[3] is not None and r[6] is not None) else 999)
    for k, _, _, dA, NpA, NlA, dB, NpB, NlB in both:
        dA_s = f"{dA:+.1f}" if dA is not None else "—"
        dB_s = f"{dB:+.1f}" if dB is not None else "—"
        if dA is not None and dB is not None:
            delta = f"{dB - dA:+.1f}"
        else:
            delta = "—"
        print(f"  {k[:27]:<28}{NpA:>4}{NlA:>4}{dA_s:>9}{NpB:>4}{NlB:>4}{dB_s:>9}{delta:>9}")

    # SECTION 2: KOLs ONLY in B (newly added or first live trades after Apr 26)
    print("\n[2] KOLs ONLY in B (newly-active live):")
    print(f"  {'KOL':<28}{'NpB':>4}{'NlB':>4}{'driftB':>9}")
    only_b = [r for r in rows if (not r[1]) and r[2]]
    only_b.sort(key=lambda r: r[6] if r[6] is not None else 999)
    for k, _, _, _, _, _, dB, NpB, NlB in only_b:
        dB_s = f"{dB:+.1f}" if dB is not None else "—"
        print(f"  {k[:27]:<28}{NpB:>4}{NlB:>4}{dB_s:>9}")

    # SECTION 3: aggregate drift per window
    def agg(p, l):
        all_p = [x for v in p.values() for x in v]
        all_l = [x for v in l.values() for x in v]
        if not all_p or not all_l:
            return None, len(all_p), len(all_l)
        return st.mean(all_l) - st.mean(all_p), len(all_p), len(all_l)
    dA_all, NpA_all, NlA_all = agg(A_p, A_l)
    dB_all, NpB_all, NlB_all = agg(B_p, B_l)
    print(f"\n[3] Aggregate drift per window:")
    print(f"  A: Np={NpA_all} Nl={NlA_all} drift={dA_all:+.2f}pp")
    print(f"  B: Np={NpB_all} Nl={NlB_all} drift={dB_all:+.2f}pp")

    # SECTION 4: drift if we EXCLUDE Apr 26-graduated KOLs from B
    # commit 5aa8ee5 added: mad_apes, reapergamble, bat_gamble. Drop ryoshikdegen.
    GRADUATED = {"mad_apes_gambles", "TheReaperGems", "bat_gamble"}
    B_p_x = {k: v for k, v in B_p.items() if k not in GRADUATED}
    B_l_x = {k: v for k, v in B_l.items() if k not in GRADUATED}
    dBx, NpBx, NlBx = agg(B_p_x, B_l_x)
    print(f"\n[4] B drift EXCLUDING Apr 26 graduates (mad_apes_gambles, TheReaperGems, bat_gamble):")
    print(f"  B*: Np={NpBx} Nl={NlBx} drift={dBx:+.2f}pp  (was {dB_all:+.2f}pp)")
