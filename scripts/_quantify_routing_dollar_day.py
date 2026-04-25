"""Quantify $/day uplift for the 2 promising routing strategies.

Scenarios:
  A. Baseline   = current (BE25_TP80_SL30 always) — what we run today
  B. Per-KOL    = route to KOL's preferred family (23 KOLs mapped, fallback BE)
  C. Score-gate = TP50_SL15 if score<30, BE25 elsewhere
  D. Combined   = score<30 → TP50_SL15, else per-KOL family

For each scenario, on the 8d window since v138.3 reset (paper main only,
exclu bat_gamble, single-exit only):
  - matched trades count
  - avg pnl_pct
  - sum $ PnL realized at avg position size
  - $/day rate
  - Live extrapolation at $1.74/trade (current BE25 live size)
"""
import os, sys, re
from collections import defaultdict
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv; load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

CLOSED = ["sl_hit","trail_stop","tp_hit","timeout","be_stop"]
RESET = "2026-04-17T14:36:00+00:00"

ARTEFACT = re.compile(r"^(DTRAIL|TRAIL|DIP|PTRAIL|SPLIT|SCALE_OUT|MOONBAG|WIDE_RUNNER)", re.IGNORECASE)
def is_clean(s):
    if ARTEFACT.match(s): return False
    if s.endswith("_HYST") and not re.search(r"_(S\d+|NZ\w*?)_HYST$", s): return False
    return True
def fam(s):
    if s.startswith("BE"): return "BE_FAMILY"
    if s.startswith("FAST60"): return "FAST60"
    if s.startswith("FAST45"): return "FAST45"
    if s.startswith("FAST"): return "FAST"
    if s.startswith("SLOW"): return "SLOW"
    if s.startswith("TP"): return "TP_PURE"
    if s.startswith("AGE"): return "AGE"
    if s.startswith("BOND"): return "BOND"
    return "OTHER"

KOL_MAPPING = {
    "DoxxedChannel":"FAST", "Luca_Apes":"TP_PURE", "papicall":"OTHER",
    "slingoorioyaps":"SLOW", "mad_apes_gambles":"OTHER", "BatmanSafuCalls":"OTHER",
    "darkocalls":"TP_PURE", "ChairmanDN1":"OTHER", "DegenSeals":"OTHER",
    "trishhhxy":"FAST", "aliensalphacalls":"FAST", "gubbinscalls":"FAST",
    "caniscooks":"OTHER", "CarnagecallsGambles":"OTHER", "eunicalls":"OTHER",
    "eveesL":"FAST", "PowsGemCalls":"FAST", "TheReaperGems":"TP_PURE",
    "shahlito":"FAST", "robo_gambles":"FAST", "chiggajogambles":"OTHER",
    "explorer_gems":"OTHER", "FrenzGems":"SLOW",
}

# Fetch
out, off, step = [], 0, 1000
while True:
    r = sb.table("paper_trades").select(
        "strategy,pnl_pct,pnl_usd,kol_group,is_shadow,exit_at,position_usd,rt_score,status,token_address"
    ).eq("source","rt").in_("status", CLOSED).gte("exit_at", RESET).range(off,off+step-1).execute()
    if not r.data: break
    out.extend(r.data)
    if len(r.data) < step: break
    off += step

rows = [r for r in out if r.get("pnl_pct") is not None
        and r.get("kol_group") != "bat_gamble"
        and r.get("pnl_usd") is not None]

emin, emax = min(r["exit_at"] for r in rows), max(r["exit_at"] for r in rows)
days = max(0.001, (datetime.fromisoformat(emax.replace("Z","+00:00"))
                  - datetime.fromisoformat(emin.replace("Z","+00:00"))).total_seconds()/86400)
print(f"Window: {emin[:10]} -> {emax[:10]} = {days:.2f}d")
mains = [r for r in rows if not r.get("is_shadow") and is_clean(r["strategy"])]
print(f"clean mains: {len(mains)} | shadows: {sum(1 for r in rows if r.get('is_shadow'))}")

# Index shadows by (kol, token, family) — proxy for "what would this strat have done"
key_fam = defaultdict(dict)
for r in rows:
    if not is_clean(r["strategy"]): continue
    k = (r["kol_group"], r["token_address"])
    f = fam(r["strategy"])
    key_fam[k].setdefault(f, []).append(float(r["pnl_pct"]))

# Index by exact strat name
key_strat = defaultdict(dict)
for r in rows:
    if not is_clean(r["strategy"]): continue
    k = (r["kol_group"], r["token_address"])
    key_strat[k][r["strategy"]] = float(r["pnl_pct"])

avg_pos = sum(float(r.get("position_usd") or 50) for r in mains) / max(1,len(mains))
trades_per_day = len(mains) / days
print(f"avg position size: ${avg_pos:.0f} | trades/d: {trades_per_day:.1f}")

# APPLES-TO-APPLES: for each main trade event, compute the pnl_pct each scenario
# WOULD have realized using shadow data on the same (KOL, token). Discard events
# where any of the 4 scenarios cannot be computed (no matching shadow), so all
# 4 are compared on the SAME opportunity set.
def get_be25_pct(k):
    return key_strat[k].get("BE25_TP80_SL30")
def get_tp50sl15_pct(k):
    return key_strat[k].get("TP50_SL15")
def get_perkol_pct(k, kol):
    target_fam = KOL_MAPPING.get(kol, "BE_FAMILY")
    if target_fam in key_fam[k] and key_fam[k][target_fam]:
        return sum(key_fam[k][target_fam])/len(key_fam[k][target_fam])
    return None

a_pcts, b_pcts, c_pcts, d_pcts = [], [], [], []
events = set()
for r in mains:
    k = (r["kol_group"], r["token_address"])
    if k in events: continue  # 1 event per (kol, token)
    events.add(k)
    score = float(r.get("rt_score") or 0)
    a = get_be25_pct(k)
    b = get_perkol_pct(k, r["kol_group"])
    tp50 = get_tp50sl15_pct(k)
    c = tp50 if score < 30 and tp50 is not None else a
    d = tp50 if score < 30 and tp50 is not None else b
    if a is None or b is None or c is None or d is None: continue
    a_pcts.append(a); b_pcts.append(b); c_pcts.append(c); d_pcts.append(d)

n = len(a_pcts)
avg_be25 = sum(a_pcts)/n*100 if n else 0
sum_be25 = sum(p * avg_pos for p in a_pcts)
n_be25 = n
avg_b = sum(b_pcts)/n*100
sum_b = sum(p * avg_pos for p in b_pcts)
avg_c = sum(c_pcts)/n*100
sum_c = sum(p * avg_pos for p in c_pcts)
avg_d = sum(d_pcts)/n*100
sum_d = sum(p * avg_pos for p in d_pcts)
print(f"\nApples-to-apples N (intersection of all 4 scenarios): {n} unique (KOL, token) events")

print()
print("=" * 100)
print(f"{'Scenario':<45}{'N':>6}{'avg%':>10}{'$ total':>12}{'$/day':>10}{'Δ/d vs A':>12}")
print("-" * 100)
print(f"  A. BASELINE (BE25 always)                  {n_be25:>6}{avg_be25:>+9.2f}%{sum_be25:>+12.2f}{sum_be25/days:>+10.2f}{0:>+12.2f}")
print(f"  B. PER-KOL family routing                  {len(b_pcts):>6}{avg_b:>+9.2f}%{sum_b:>+12.2f}{sum_b/days:>+10.2f}{(sum_b-sum_be25)/days:>+12.2f}")
print(f"  C. SCORE-GATE (TP50_SL15 if score<30)      {len(c_pcts):>6}{avg_c:>+9.2f}%{sum_c:>+12.2f}{sum_c/days:>+10.2f}{(sum_c-sum_be25)/days:>+12.2f}")
print(f"  D. COMBINED (score<30→TP50, else per-KOL)  {len(d_pcts):>6}{avg_d:>+9.2f}%{sum_d:>+12.2f}{sum_d/days:>+10.2f}{(sum_d-sum_be25)/days:>+12.2f}")
print()
print("=" * 100)
print("LIVE EXTRAPOLATION at $1.74 per trade (current live BE25 size, ~218 mains/d)")
print("-" * 100)
LIVE_POS = 1.74
n_per_d = trades_per_day
def pd(avg): return avg/100 * LIVE_POS * n_per_d
print(f"  A. BASELINE BE25                           {avg_be25:>+9.2f}% × $1.74 × {n_per_d:.0f}/d = ${pd(avg_be25):+.2f}/d")
print(f"  B. PER-KOL family routing                  {avg_b:>+9.2f}% × $1.74 × {n_per_d:.0f}/d = ${pd(avg_b):+.2f}/d  (Δ ${pd(avg_b)-pd(avg_be25):+.2f}/d)")
print(f"  C. SCORE-GATE                              {avg_c:>+9.2f}% × $1.74 × {n_per_d:.0f}/d = ${pd(avg_c):+.2f}/d  (Δ ${pd(avg_c)-pd(avg_be25):+.2f}/d)")
print(f"  D. COMBINED                                {avg_d:>+9.2f}% × $1.74 × {n_per_d:.0f}/d = ${pd(avg_d):+.2f}/d  (Δ ${pd(avg_d)-pd(avg_be25):+.2f}/d)")
print()
print("Annualized projection (live $1.74 size, 365d):")
print(f"  A. BASELINE: ${pd(avg_be25)*365:+,.0f}/yr")
print(f"  B. PER-KOL:  ${pd(avg_b)*365:+,.0f}/yr  (uplift ${(pd(avg_b)-pd(avg_be25))*365:+,.0f}/yr)")
print(f"  D. COMBINED: ${pd(avg_d)*365:+,.0f}/yr  (uplift ${(pd(avg_d)-pd(avg_be25))*365:+,.0f}/yr)")
