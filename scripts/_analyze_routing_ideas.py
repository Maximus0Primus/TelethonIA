"""Test 4 routing ideas against all closed paper trades (main + shadow):

  1. SCORE-BAND ROUTING: does the optimal strategy differ across score bands?
  2. COOCCURRENCE GATE: does a 2+ KOL confirmation outperform single-KOL?
  3. PER-KOL TIMEOUT: do some KOLs need shorter/longer holds?
  4. PER-KOL STRATEGY SPECIALIZATION: do some KOLs prefer non-default strats?

For each idea, compute: (a) baseline (BE25_TP80_SL30 uniform), (b) the
"smart-route" alternative, (c) Δ avg pnl_pct on the same intersection.
Flag PROMISING if Δ ≥ +3pp at N≥30, NEGATIVE if Δ ≤ +1pp at N≥30 (worth
documenting as a dead end), INCONCLUSIVE otherwise.

Pure analysis, zero mutation. Writes summary to console + json to data/.
"""
import os, sys, json
from collections import defaultdict, Counter
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv; load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

CLOSED = ["sl_hit","trail_stop","tp_hit","timeout","be_stop"]
RESET = "2026-04-17T14:36:00+00:00"
MIN_N_SIGNAL = 30  # cells with smaller N are noisy
PROMISING_DELTA = 3.0   # +3pp Δ avg → worth implementing
NEGATIVE_DELTA = 1.0    # ≤+1pp → not worth, document as dead end

# v14e.22: filter out documented sim-artefact families before any "best strat"
# selection (cf. dtrail_shadow_artifact_apr20, hyst_artifacts_apr20). These
# pollute every ranking we run.
import re
ARTEFACT_RE = re.compile(
    r"^(DTRAIL|TRAIL|DIP|PTRAIL|SPLIT|SCALE_OUT|MOONBAG|WIDE_RUNNER)|_HYST$",
    re.IGNORECASE,
)
def is_clean(strat: str) -> bool:
    """Single-exit, no trail/dip/split/standalone-hyst artefacts."""
    if ARTEFACT_RE.match(strat) or ARTEFACT_RE.search(strat):
        # second match catches `_HYST$`
        if strat.endswith("_HYST") and not re.search(r"_(S\d+|NZ\w*?)_HYST$", strat):
            return False
        if re.match(r"^(DTRAIL|TRAIL|DIP|PTRAIL|SPLIT|SCALE_OUT|MOONBAG|WIDE_RUNNER)", strat):
            return False
    return True


def fetch_all():
    out, off, step = [], 0, 1000
    cols = "strategy,pnl_pct,pnl_usd,kol_group,is_shadow,chain,source,exit_at,created_at,status,rt_score,rt_liquidity_usd,token_address,exit_minutes"
    while True:
        r = sb.table("paper_trades").select(cols)\
            .eq("source","rt").in_("status", CLOSED).gte("exit_at", RESET)\
            .range(off, off+step-1).execute()
        if not r.data: break
        out.extend(r.data)
        if len(r.data) < step: break
        off += step
    return out


print("Fetching closed paper_trades since reset...")
rows = fetch_all()
rows = [r for r in rows
        if r.get("pnl_pct") is not None
        and r.get("kol_group") != "bat_gamble"]
print(f"Total non-bg closed trades: {len(rows)}")

main = [r for r in rows if not r.get("is_shadow")]
shadow = [r for r in rows if r.get("is_shadow")]
print(f"  main: {len(main)} | shadow: {len(shadow)}")
print(f"  chains: {dict(Counter((r.get('chain') or 'sol') for r in rows))}")
print(f"  unique tokens: {len(set(r['token_address'] for r in rows))}")
print(f"  unique KOLs: {len(set(r['kol_group'] for r in rows))}")
print(f"  unique strats: {len(set(r['strategy'] for r in rows))}")
print()

results = {}

# ==============================================================================
# 1. SCORE-BAND ROUTING
# ==============================================================================
print("="*100)
print("IDEA 1 — SCORE-BAND ROUTING")
print("="*100)
print("Hypothesis: the optimal strategy differs across rt_score bands.")
print()

BANDS = [(0, 30, "<30"), (30, 40, "30-40"), (40, 50, "40-50"),
         (50, 60, "50-60"), (60, 200, "60+")]

def band_for(s):
    for lo, hi, label in BANDS:
        if lo <= s < hi: return label
    return "?"

# (strategy, band) → list of pnl_pct (single-exit, artefacts excluded)
sb_agg = defaultdict(list)
for r in rows:
    if r.get("rt_score") is None: continue
    if not is_clean(r["strategy"]): continue
    band = band_for(float(r["rt_score"]))
    sb_agg[(r["strategy"], band)].append(float(r["pnl_pct"]))

# For each band, find best strat with N >= MIN_N_SIGNAL
print(f"{'band':<8}{'best strat':<38}{'best avg%':>10}{'N':>5}{'BE25 avg%':>11}{'BE25 N':>8}{'Δ pp':>8}")
print("-" * 90)
score_band_smart = {}
for _, _, band in BANDS:
    candidates = [(strat, sum(v)/len(v)*100, len(v)) for (strat, b), v in sb_agg.items()
                  if b == band and len(v) >= MIN_N_SIGNAL]
    if not candidates: continue
    candidates.sort(key=lambda x: -x[1])
    best_strat, best_avg, best_n = candidates[0]
    be25_data = sb_agg.get(("BE25_TP80_SL30", band), [])
    be25_avg = (sum(be25_data)/len(be25_data)*100) if be25_data else None
    delta = best_avg - be25_avg if be25_avg is not None else None
    if delta is not None:
        score_band_smart[band] = (best_strat, best_avg, best_n, be25_avg, len(be25_data), delta)
        print(f"  {band:<6}{best_strat:<38}{best_avg:>+9.1f}%{best_n:>5}{be25_avg:>+10.1f}%{len(be25_data):>8}{delta:>+7.1f}")

# Test: smart-route hypothetical vs always-BE25 on full intersection
print("\nFull smart-route hypothetical (per-band best vs always-BE25):")
smart_pnls = []
be25_pnls = []
for r in rows:
    if r.get("rt_score") is None: continue
    band = band_for(float(r["rt_score"]))
    if band not in score_band_smart: continue
    best_strat = score_band_smart[band][0]
    if r["strategy"] == best_strat:
        smart_pnls.append(float(r["pnl_pct"]))
    if r["strategy"] == "BE25_TP80_SL30":
        be25_pnls.append(float(r["pnl_pct"]))
smart_avg = sum(smart_pnls)/len(smart_pnls)*100 if smart_pnls else None
be25_avg_full = sum(be25_pnls)/len(be25_pnls)*100 if be25_pnls else None
print(f"  Smart-route: avg={smart_avg:+.2f}% (N={len(smart_pnls)})")
print(f"  Always BE25: avg={be25_avg_full:+.2f}% (N={len(be25_pnls)})")
delta_overall = (smart_avg or 0) - (be25_avg_full or 0)
print(f"  Δ overall: {delta_overall:+.2f}pp")
results["score_band"] = {
    "smart_avg": smart_avg, "be25_avg": be25_avg_full, "delta_pp": delta_overall,
    "smart_N": len(smart_pnls), "be25_N": len(be25_pnls),
    "per_band": {k: {"best_strat": v[0], "best_avg": v[1], "be25_avg": v[3], "delta_pp": v[5]}
                 for k, v in score_band_smart.items()},
    "verdict": "PROMISING" if delta_overall >= PROMISING_DELTA else
               ("NEGATIVE" if delta_overall <= NEGATIVE_DELTA else "INCONCLUSIVE")
}
print(f"  → VERDICT: {results['score_band']['verdict']}")

# ==============================================================================
# 2. COOCCURRENCE GATE (multi-KOL confirmation)
# ==============================================================================
print("\n" + "="*100)
print("IDEA 2 — COOCCURRENCE / MULTI-KOL CONFIRMATION GATE")
print("="*100)
print("Hypothesis: trades on tokens called by ≥2 KOLs in last 4h outperform single-KOL.")
print()

# For each trade, count distinct KOLs that called the same token within [created-4h, created]
# Build a token → list of (created_at, kol_group, trade_id) lookup
tok_calls = defaultdict(list)
for r in rows:
    tok_calls[r["token_address"]].append((r["created_at"], r["kol_group"], r["strategy"], float(r["pnl_pct"]), bool(r.get("is_shadow"))))

cooccur_buckets = defaultdict(list)  # cooccur_count → list of pnl_pct (main only)
for r in main:
    if r["token_address"] not in tok_calls: continue
    t0 = datetime.fromisoformat(r["created_at"].replace("Z","+00:00"))
    distinct_kols = set()
    for (ts, kol, _, _, _) in tok_calls[r["token_address"]]:
        ts_dt = datetime.fromisoformat(ts.replace("Z","+00:00"))
        if (t0 - ts_dt).total_seconds() <= 4 * 3600 and ts_dt <= t0:
            distinct_kols.add(kol)
    cooccur_count = len(distinct_kols)  # incl. self → 1=solo, 2=confirmed once, etc.
    bucket = "1" if cooccur_count <= 1 else ("2" if cooccur_count == 2 else "3+")
    cooccur_buckets[bucket].append(float(r["pnl_pct"]))

print(f"{'cooccur':<10}{'N':>6}{'avg%':>9}{'WR':>6}")
print("-"*40)
for b in ["1","2","3+"]:
    v = cooccur_buckets[b]
    if not v: continue
    avg = sum(v)/len(v)*100
    wr = sum(1 for x in v if x>0)/len(v)*100
    print(f"  {b:<8}{len(v):>6}{avg:>+8.1f}%{wr:>5.0f}%")

avg_solo = (sum(cooccur_buckets["1"])/len(cooccur_buckets["1"])*100) if cooccur_buckets["1"] else 0
avg_2plus = ((sum(cooccur_buckets["2"]) + sum(cooccur_buckets["3+"])) /
             max(1, len(cooccur_buckets["2"]) + len(cooccur_buckets["3+"]))) * 100
delta_co = avg_2plus - avg_solo
print(f"\n  Δ (2+ KOLs vs solo): {delta_co:+.2f}pp on N≥{MIN_N_SIGNAL}")
results["cooccurrence"] = {
    "solo_avg": avg_solo, "solo_N": len(cooccur_buckets["1"]),
    "confirmed_avg": avg_2plus,
    "confirmed_N": len(cooccur_buckets["2"]) + len(cooccur_buckets["3+"]),
    "delta_pp": delta_co,
    "verdict": "PROMISING" if delta_co >= PROMISING_DELTA else
               ("NEGATIVE" if delta_co <= NEGATIVE_DELTA else "INCONCLUSIVE")
}
print(f"  → VERDICT: {results['cooccurrence']['verdict']}")

# ==============================================================================
# 3. PER-KOL TIMEOUT
# ==============================================================================
print("\n" + "="*100)
print("IDEA 3 — PER-KOL TIMEOUT")
print("="*100)
print("Hypothesis: some KOLs win/lose much earlier or later than the 30min default.")
print()

# For each KOL, look at exit_minutes of TP_HIT only (real wins, not capped by timeout)
# Use ALL rows (main + shadow) to maximize sample.
from statistics import median
kol_timing = defaultdict(lambda: {"tp_hit_mins": [], "all_pnl": [], "n_tp": 0, "n_total": 0})
for r in rows:
    em = r.get("exit_minutes")
    if em is None: continue
    pnl = float(r["pnl_pct"])
    kol = r["kol_group"]
    kol_timing[kol]["all_pnl"].append(pnl)
    kol_timing[kol]["n_total"] += 1
    if r.get("status") == "tp_hit":
        kol_timing[kol]["tp_hit_mins"].append(int(em))
        kol_timing[kol]["n_tp"] += 1

print(f"{'kol':<25}{'N_total':>9}{'N_tp':>6}{'tp_hit median (min)':>22}{'avg pnl%':>10}")
print("-" * 80)
fast_kols, slow_kols = [], []
for kol, d in sorted(kol_timing.items(), key=lambda x: -x[1]["n_total"]):
    n_total = d["n_total"]
    if n_total < 100: continue  # need solid sample for shadows too
    if d["n_tp"] < 5: continue
    tp_med = median(d["tp_hit_mins"])
    avg_pnl = sum(d["all_pnl"])/n_total*100
    print(f"  {kol:<23}{n_total:>9}{d['n_tp']:>6}{tp_med:>22}{avg_pnl:>+9.1f}%")
    if tp_med < 10: fast_kols.append((kol, n_total, tp_med, avg_pnl))
    elif tp_med > 30: slow_kols.append((kol, n_total, tp_med, avg_pnl))

print(f"\nFAST-burn KOLs (win-median <15min, candidates for tight timeout):")
for k, n, wm, ap in sorted(fast_kols, key=lambda x: x[2])[:10]:
    print(f"  {k:<25} N={n} win_med={wm}min avg_pnl={ap:+.1f}%")
print(f"\nSLOW-burn KOLs (win-median >60min, candidates for extended timeout):")
for k, n, wm, ap in sorted(slow_kols, key=lambda x: -x[2])[:10]:
    print(f"  {k:<25} N={n} win_med={wm}min avg_pnl={ap:+.1f}%")

# Estimate uplift: how many wins did we miss with default 30min on slow KOLs?
total_main = len(main)
slow_kol_set = {k for (k,_,_,_) in slow_kols}
slow_main = [r for r in main if r["kol_group"] in slow_kol_set]
slow_timeouts = [r for r in slow_main if r["status"] == "timeout"]
slow_timeouts_loss = [r for r in slow_timeouts if float(r["pnl_pct"]) < 0]
n_slow_to = len(slow_timeouts)
n_slow_to_loss = len(slow_timeouts_loss)
print(f"\nSlow KOL timeout closures (extended timeout could rescue some):")
print(f"  total slow-KOL timeouts: {n_slow_to}")
print(f"  of which closing at loss: {n_slow_to_loss}")
delta_to = (n_slow_to_loss / max(1,total_main)) * 5  # rough estimate: 5pp uplift if recovered
print(f"  estimated uplift (rough): ~{delta_to:.1f}pp on full population")
results["per_kol_timeout"] = {
    "fast_kols": [{"kol": k, "N": n, "win_med": wm, "avg_pnl": ap} for k,n,wm,ap in fast_kols],
    "slow_kols": [{"kol": k, "N": n, "win_med": wm, "avg_pnl": ap} for k,n,wm,ap in slow_kols],
    "estimated_uplift_pp": delta_to,
    "verdict": "PROMISING" if delta_to >= PROMISING_DELTA else
               ("NEGATIVE" if delta_to <= NEGATIVE_DELTA else "INCONCLUSIVE")
}
print(f"  → VERDICT: {results['per_kol_timeout']['verdict']}")

# ==============================================================================
# 4. PER-KOL STRATEGY SPECIALIZATION
# ==============================================================================
print("\n" + "="*100)
print("IDEA 4 — PER-KOL STRATEGY SPECIALIZATION")
print("="*100)
print("Hypothesis: some KOLs significantly outperform with a non-default strategy.")
print()

# Group strategies by FAMILY rather than individual strat — increases N per cell.
# Family heuristic from name prefix.
def strat_family(s):
    if s.startswith("BE"): return "BE_FAMILY"
    if s.startswith("FAST60"): return "FAST60"
    if s.startswith("FAST45"): return "FAST45"
    if s.startswith("FAST"): return "FAST"
    if s.startswith("SLOW"): return "SLOW"
    if s.startswith("TP"): return "TP_PURE"
    if s.startswith("AGE"): return "AGE"
    if s.startswith("BOND"): return "BOND"
    return "OTHER"

# For each (kol, family) compute avg pnl on CLEAN strats only
ks_agg = defaultdict(list)
for r in rows:
    if not is_clean(r["strategy"]): continue
    fam = strat_family(r["strategy"])
    ks_agg[(r["kol_group"], fam)].append(float(r["pnl_pct"]))

# For each KOL with total clean N≥100, find best family AND BE_FAMILY baseline
kol_total = defaultdict(int)
for r in rows:
    if is_clean(r["strategy"]): kol_total[r["kol_group"]] += 1
qualified_kols = [k for k, n in kol_total.items() if n >= 100]
print(f"Qualified KOLs (clean-strat N≥100): {len(qualified_kols)}")

print(f"\n{'kol':<25}{'best family':<14}{'best avg%':>10}{'N':>5}{'BE_FAM avg%':>13}{'BE_FAM N':>10}{'Δ pp':>8}")
print("-" * 95)
strong_kol_pref = []
for kol in qualified_kols:
    candidates = [(fam, sum(v)/len(v)*100, len(v)) for (k, fam), v in ks_agg.items()
                  if k == kol and len(v) >= 30]
    if not candidates: continue
    candidates.sort(key=lambda x: -x[1])
    best_fam, best_avg, best_n = candidates[0]
    be_fam = ks_agg.get((kol, "BE_FAMILY"), [])
    be_avg = (sum(be_fam)/len(be_fam)*100) if len(be_fam) >= 20 else None
    delta = (best_avg - be_avg) if be_avg is not None else None
    if delta is not None and delta >= 3.0:
        strong_kol_pref.append((kol, best_fam, best_avg, best_n, be_avg, len(be_fam), delta))
        print(f"  {kol:<23}{best_fam:<14}{best_avg:>+9.1f}%{best_n:>5}{be_avg:>+12.1f}%{len(be_fam):>10}{delta:>+7.1f}")

print(f"\n  KOLs with significant strat preference (Δ ≥ 3pp): {len(strong_kol_pref)}/{len(qualified_kols)}")

# Smart route: for each main trade, use the KOL's best strat IF kol qualifies, else BE25
# Compute hypothetical avg
mapping = {}  # kol → best_strat
for k, bs, _, _, _, _, _ in strong_kol_pref:
    mapping[k] = bs

route_pnls = []
base_pnls = []
for r in rows:
    if not is_clean(r["strategy"]): continue
    kol = r["kol_group"]
    fam = strat_family(r["strategy"])
    target_fam = mapping.get(kol, "BE_FAMILY")
    if fam == target_fam:
        route_pnls.append(float(r["pnl_pct"]))
    if fam == "BE_FAMILY":
        base_pnls.append(float(r["pnl_pct"]))

route_avg = (sum(route_pnls)/len(route_pnls)*100) if route_pnls else None
base_avg = (sum(base_pnls)/len(base_pnls)*100) if base_pnls else None
delta_kol = (route_avg or 0) - (base_avg or 0)
print(f"\n  Smart-route hypothetical (KOL→best strat OR BE25 default):")
print(f"  Route avg: {route_avg:+.2f}% (N={len(route_pnls)})")
print(f"  BE25 base: {base_avg:+.2f}% (N={len(base_pnls)})")
print(f"  Δ overall: {delta_kol:+.2f}pp")
results["per_kol_strat"] = {
    "kol_mapping": [{"kol":k, "strat":bs, "avg":ba, "N":bn, "be25_avg":be25a, "delta_pp":d}
                    for k,bs,ba,bn,be25a,_,d in strong_kol_pref],
    "route_avg": route_avg, "base_avg": base_avg, "delta_pp": delta_kol,
    "qualified_kols": len(qualified_kols),
    "kols_with_pref": len(strong_kol_pref),
    "verdict": "PROMISING" if delta_kol >= PROMISING_DELTA else
               ("NEGATIVE" if delta_kol <= NEGATIVE_DELTA else "INCONCLUSIVE")
}
print(f"  → VERDICT: {results['per_kol_strat']['verdict']}")

# ==============================================================================
# Summary
# ==============================================================================
print("\n" + "="*100)
print("SUMMARY")
print("="*100)
for idea, data in results.items():
    print(f"  {idea:<25} {data['verdict']:<14} Δ={data.get('delta_pp', data.get('estimated_uplift_pp', 0)):+.2f}pp")

# Save full json
import datetime as _dt
ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
out = os.path.join(os.path.dirname(__file__), "..", "data", f"routing_ideas_{ts}.json")
with open(out, "w") as f:
    json.dump({"timestamp": ts, "results": results}, f, indent=2, default=str)
print(f"\nFull results saved: {out}")
