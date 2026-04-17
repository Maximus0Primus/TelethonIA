"""rt_score multi-variant A/B audit.

V1 baseline correlation = +0.2072 with pnl_pct (N=74).

Test 6 variants WITHOUT breaking V1 structure (lesson: weight rebalancing broke
interactions). Instead, ADD signals or FIX known bugs surgically.

  V2a : V1 + age U-curve (fix linear bug: fresh tokens <0.3h = rug, 0.3-1.3h = sweet)
  V2b : V1 + fresh-age BONUS (+10 if 0.3 < age < 1.3) — additive, preserves V1
  V2c : V1 + BSR-strong BONUS (+10 if bsr > 0.7) — additive
  V2d : V1 + combo BONUS (+15 if bsr>0.6 AND 0.3<age<1.3) — conjunction signal
  V2e : V1 - pump_fun penalty (remove blanket -10, corr is_pf = -0.003 noise)
  V2f : Geometric combiner (multiplicative interactions preserved)

Pick the winner (corr > V1).
"""
from __future__ import annotations
import os
import sys
import math
import statistics

from dotenv import load_dotenv
from supabase import create_client

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SCRAPER = os.path.join(ROOT, "scraper")
sys.path.insert(0, SCRAPER)
load_dotenv(os.path.join(SCRAPER, ".env"))
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

from safe_scraper import (  # noqa
    _rt_compute_kol_quality, _rt_compute_momentum,
    _rt_compute_token_safety,
)


# ---------------- V1 baseline replication ----------------
def score_v1(ki, ti, tier, n_confs):
    kol_q = _rt_compute_kol_quality(ki, tier)
    safety = _rt_compute_token_safety(ti)
    momentum = _rt_compute_momentum(ti)
    confirm = {0:0, 1:40, 2:70}.get(n_confs, 100 if n_confs >= 3 else 0)
    return round(max(0, min(100, kol_q*0.35 + safety*0.30 + momentum*0.20 + confirm*0.15)), 1)


# ---------------- Safety variants ----------------
def _safety_u_curve(ti):
    """Safety V1 but with age U-curve instead of linear."""
    liq = ti.get("liquidity_usd", 0)
    mcap = max(ti.get("mcap", 1), 1)
    vol = ti.get("volume_24h", 0)
    bsr = ti.get("buy_sell_ratio", 0.5)
    age_h = ti.get("token_age_hours", 0)
    is_pf = ti.get("is_pump_fun", 0)
    liq_mcap = liq / mcap
    if liq < 5000: liq_s = (liq/5000)*10
    elif liq < 50000: liq_s = 10 + (liq-5000)/45000*10
    else: liq_s = min(30, 20 + (liq-50000)/450000*10)
    if vol < 10000: vol_s = (vol/10000)*6
    elif vol < 100000: vol_s = 6 + (vol-10000)/90000*8
    else: vol_s = min(20, 14 + (vol-100000)/900000*6)
    bsr_s = min(15, max(0, bsr/2.0*15))
    liq_mcap_s = min(15, max(0, liq_mcap/0.2*15))
    age_s = 10 * math.exp(-((age_h - 0.8)/0.5)**2)
    age_s = max(0, min(10, age_s))
    pf_pen = -10 if (is_pf and liq < 10000) else 0
    return max(0, min(100, liq_s + vol_s + bsr_s + liq_mcap_s + age_s + pf_pen))


def _safety_no_pf_penalty(ti):
    """Safety V1 but drop the blanket pump.fun penalty."""
    liq = ti.get("liquidity_usd", 0)
    mcap = max(ti.get("mcap", 1), 1)
    vol = ti.get("volume_24h", 0)
    bsr = ti.get("buy_sell_ratio", 0.5)
    age_h = ti.get("token_age_hours", 0)
    liq_mcap = liq / mcap
    if liq < 5000: liq_s = (liq/5000)*10
    elif liq < 50000: liq_s = 10 + (liq-5000)/45000*10
    else: liq_s = min(30, 20 + (liq-50000)/450000*10)
    if vol < 10000: vol_s = (vol/10000)*6
    elif vol < 100000: vol_s = 6 + (vol-10000)/90000*8
    else: vol_s = min(20, 14 + (vol-100000)/900000*6)
    bsr_s = min(15, max(0, bsr/2.0*15))
    liq_mcap_s = min(15, max(0, liq_mcap/0.2*15))
    age_s = min(10, max(0, age_h/168*10))  # linear (V1)
    # no pf penalty
    return max(0, min(100, liq_s + vol_s + bsr_s + liq_mcap_s + age_s))


# ---------------- V2 variants ----------------
def score_v2a(ki, ti, tier, n_confs):
    """V1 + age U-curve (minimal fix)."""
    kol_q = _rt_compute_kol_quality(ki, tier)
    safety = _safety_u_curve(ti)
    momentum = _rt_compute_momentum(ti)
    confirm = {0:0, 1:40, 2:70}.get(n_confs, 100 if n_confs >= 3 else 0)
    return round(max(0, min(100, kol_q*0.35 + safety*0.30 + momentum*0.20 + confirm*0.15)), 1)


def score_v2b(ki, ti, tier, n_confs):
    """V1 + fresh-age BONUS (+10 if 0.3 < age < 1.3)."""
    base = score_v1(ki, ti, tier, n_confs)
    age_h = ti.get("token_age_hours", 0)
    bonus = 10 if 0.3 < age_h < 1.3 else 0
    return round(min(100, base + bonus), 1)


def score_v2c(ki, ti, tier, n_confs):
    """V1 + BSR-strong BONUS (+10 if bsr > 0.7)."""
    base = score_v1(ki, ti, tier, n_confs)
    bsr = ti.get("buy_sell_ratio", 0.5)
    bonus = 10 if bsr > 0.7 else 0
    return round(min(100, base + bonus), 1)


def score_v2d(ki, ti, tier, n_confs):
    """V1 + combo BONUS (+15 if bsr>0.6 AND 0.3<age<1.3) — conjunction signal."""
    base = score_v1(ki, ti, tier, n_confs)
    bsr = ti.get("buy_sell_ratio", 0.5)
    age_h = ti.get("token_age_hours", 0)
    bonus = 15 if (bsr > 0.6 and 0.3 < age_h < 1.3) else 0
    return round(min(100, base + bonus), 1)


def score_v2e(ki, ti, tier, n_confs):
    """V1 minus pump.fun blanket penalty."""
    kol_q = _rt_compute_kol_quality(ki, tier)
    safety = _safety_no_pf_penalty(ti)
    momentum = _rt_compute_momentum(ti)
    confirm = {0:0, 1:40, 2:70}.get(n_confs, 100 if n_confs >= 3 else 0)
    return round(max(0, min(100, kol_q*0.35 + safety*0.30 + momentum*0.20 + confirm*0.15)), 1)


def score_v2f(ki, ti, tier, n_confs):
    """Geometric (multiplicative interactions): 100 × ∏(score_i/100)^w_i."""
    kol_q = _rt_compute_kol_quality(ki, tier)
    safety = _rt_compute_token_safety(ti)
    momentum = _rt_compute_momentum(ti)
    confirm = {0:0, 1:40, 2:70}.get(n_confs, 100 if n_confs >= 3 else 0)
    # Avoid zeros — add 1e-3 floor
    kol_r = max(kol_q/100, 0.001)
    safety_r = max(safety/100, 0.001)
    momentum_r = max(momentum/100, 0.001)
    confirm_r = max((confirm + 10)/110, 0.001)  # confirm often 0, add baseline
    geom = 100 * (kol_r**0.35 * safety_r**0.30 * momentum_r**0.20 * confirm_r**0.15)
    return round(max(0, min(100, geom)), 1)


def score_v2g(ki, ti, tier, n_confs):
    """V1 + BOTH fresh-age bonus + BSR-strong bonus (stacking additives)."""
    base = score_v1(ki, ti, tier, n_confs)
    age_h = ti.get("token_age_hours", 0)
    bsr = ti.get("buy_sell_ratio", 0.5)
    bonus = 0
    if 0.3 < age_h < 1.3: bonus += 8
    if bsr > 0.7: bonus += 8
    if ti.get("liquidity_usd", 0) <= 0: bonus -= 5  # mild penalty liq=0
    return round(max(0, min(100, base + bonus)), 1)


VARIANTS = {
    "V1_baseline":            score_v1,
    "V2a_age_u_curve":        score_v2a,
    "V2b_fresh_bonus":        score_v2b,
    "V2c_bsr_bonus":          score_v2c,
    "V2d_combo_bonus":        score_v2d,
    "V2e_no_pf_penalty":      score_v2e,
    "V2f_geometric":          score_v2f,
    "V2g_stacked_bonuses":    score_v2g,
}


# Pure additive variants — apply bonus ON TOP of stored rt_score (accurate V1)
def add_fresh_bonus(stored_v1, ti, n_confs):
    age_h = ti.get("token_age_hours", 0)
    return round(min(100, stored_v1 + (10 if 0.3 < age_h < 1.3 else 0)), 1)


def add_bsr_bonus(stored_v1, ti, n_confs):
    bsr = ti.get("buy_sell_ratio", 0.5)
    return round(min(100, stored_v1 + (10 if bsr > 0.7 else 0)), 1)


def add_combo_bonus(stored_v1, ti, n_confs):
    bsr = ti.get("buy_sell_ratio", 0.5)
    age_h = ti.get("token_age_hours", 0)
    return round(min(100, stored_v1 + (15 if (bsr > 0.6 and 0.3 < age_h < 1.3) else 0)), 1)


def add_stacked_bonus(stored_v1, ti, n_confs):
    age_h = ti.get("token_age_hours", 0)
    bsr = ti.get("buy_sell_ratio", 0.5)
    bonus = 0
    if 0.3 < age_h < 1.3: bonus += 8
    if bsr > 0.7: bonus += 8
    if ti.get("liquidity_usd", 0) <= 0: bonus -= 5
    return round(max(0, min(100, stored_v1 + bonus)), 1)


def add_liq_gate_bonus(stored_v1, ti, n_confs):
    """Anti-zero-liq penalty (liq=0 is historically toxic)."""
    return round(max(0, stored_v1 - (15 if ti.get("liquidity_usd", 0) <= 0 else 0)), 1)


def add_confirm_bonus(stored_v1, ti, n_confs):
    """Boost via n_kol_confirmations — currently in V1 but only weighted 15% × 40pts = 6pts."""
    # Add stacking bonus for 2+ confirmations
    extra = 5 * max(0, n_confs - 1)  # 0 for 0-1, 5 for 2, 10 for 3+
    return round(min(100, stored_v1 + extra), 1)


ADDITIVE_VARIANTS = {
    "V1_stored":                 lambda v1, ti, n: v1,
    "V2h_fresh_bonus_pure":      add_fresh_bonus,
    "V2i_bsr_bonus_pure":        add_bsr_bonus,
    "V2j_combo_bonus_pure":      add_combo_bonus,
    "V2k_stacked_bonuses_pure":  add_stacked_bonus,
    "V2l_liq_gate_penalty":      add_liq_gate_bonus,
    "V2m_confirm_bonus":         add_confirm_bonus,
}


def pearson(xs, ys):
    n = len(xs)
    if n < 3: return 0
    mx, my = sum(xs)/n, sum(ys)/n
    num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    dx = (sum((x-mx)**2 for x in xs))**0.5
    dy = (sum((y-my)**2 for y in ys))**0.5
    return num/(dx*dy) if dx*dy > 0 else 0


def main():
    print("=" * 100)
    print("rt_score multi-variant audit — keep V1 structure, iterate on specific fixes")
    print("=" * 100)

    rows = []; off = 0
    while True:
        r = sb.table("paper_trades").select(
            "id,token_address,status,pnl_pct,rt_score,rt_liquidity_usd,"
            "kol_score,kol_win_rate,kol_tier,rt_volume_24h,rt_buy_sell_ratio,"
            "rt_token_age_hours,rt_is_pump_fun,entry_mcap,n_kol_confirmations"
        ).eq("source", "rt").eq("is_shadow", False).gte(
            "created_at", "2026-04-13T20:00:00Z"
        ).range(off, off+999).execute().data
        if not r: break
        rows.extend(r)
        if len(r) < 1000: break
        off += 1000

    closed = [r for r in rows if r.get("status") in
              ("tp_hit", "sl_hit", "timeout", "trail_stop", "trail_crash")]
    seen = set(); uniq = []
    for r in sorted(closed, key=lambda x: x["id"]):
        if r["token_address"] not in seen:
            seen.add(r["token_address"])
            uniq.append(r)
    print(f"N = {len(uniq)} unique closed MAIN tokens post-v132\n")

    # Compute all variants
    results = {name: [] for name in {**VARIANTS, **ADDITIVE_VARIANTS}}
    pnls = []
    for r in uniq:
        ki = {"score": float(r.get("kol_score") or 0),
              "win_rate": float(r.get("kol_win_rate") or 0),
              "total_calls": 0}
        ti = {"liquidity_usd": float(r.get("rt_liquidity_usd") or 0),
              "mcap": float(r.get("entry_mcap") or 1),
              "volume_24h": float(r.get("rt_volume_24h") or 0),
              "buy_sell_ratio": float(r.get("rt_buy_sell_ratio") or 0.5),
              "token_age_hours": float(r.get("rt_token_age_hours") or 0),
              "is_pump_fun": int(r.get("rt_is_pump_fun") or 0),
              "price_change_1h": 0, "price_change_5m": 0}
        tier = r.get("kol_tier") or "A"
        n_confs = int(r.get("n_kol_confirmations") or 0)
        pnl = float(r.get("pnl_pct") or 0)
        stored_v1 = float(r.get("rt_score") or 0)
        pnls.append(pnl)
        for name, fn in VARIANTS.items():
            results[name].append(fn(ki, ti, tier, n_confs))
        for name, fn in ADDITIVE_VARIANTS.items():
            results[name].append(fn(stored_v1, ti, n_confs))

    # Ranking
    print(f"{'Variant':<26}{'Corr':>9}{'Q1_avg%':>10}{'Q4_avg%':>10}{'spread':>10}{'Q4_WR':>8}{'filter≥30 avg%':>16}{'N_pass':>8}")
    print("-" * 100)
    baseline = None
    rows_out = []
    all_names = list(VARIANTS.keys()) + list(ADDITIVE_VARIANTS.keys())
    for name in all_names:
        scores = results[name]
        corr = pearson(scores, pnls)
        sdata = sorted(zip(scores, pnls))
        n = len(sdata)
        q1 = sdata[:n//4]; q4 = sdata[3*n//4:]
        q1_pnls = [d[1] for d in q1]; q4_pnls = [d[1] for d in q4]
        q1_avg = statistics.mean(q1_pnls)*100 if q1_pnls else 0
        q4_avg = statistics.mean(q4_pnls)*100 if q4_pnls else 0
        q4_wr = sum(1 for p in q4_pnls if p>0)/max(1,len(q4_pnls))*100
        spread = q4_avg - q1_avg
        passed = [(s,p) for s,p in zip(scores, pnls) if s >= 30]
        pass_avg = statistics.mean([p for _,p in passed])*100 if passed else 0
        n_pass = len(passed)
        rows_out.append((name, corr, q1_avg, q4_avg, spread, q4_wr, pass_avg, n_pass))
        if name == "V1_stored":
            baseline = (corr, spread, pass_avg)

    # Sort by corr desc
    rows_out.sort(key=lambda x: -x[1])
    for name, corr, q1_avg, q4_avg, spread, q4_wr, pass_avg, n_pass in rows_out:
        flag = ""
        if name not in ("V1_baseline", "V1_stored") and baseline:
            better = sum([corr > baseline[0], spread > baseline[1], pass_avg > baseline[2]])
            if better >= 2: flag = "⭐ BETTER"
            elif corr < baseline[0] - 0.03: flag = "❌ WORSE"
        print(f"{name:<26}{corr:>+8.4f}{q1_avg:>+9.2f}%{q4_avg:>+9.2f}%{spread:>+9.2f}%{q4_wr:>7.1f}%{pass_avg:>+15.2f}%{n_pass:>8}  {flag}")


if __name__ == "__main__":
    main()
