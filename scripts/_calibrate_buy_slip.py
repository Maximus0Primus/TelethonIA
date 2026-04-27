"""Empirically fit BUY slippage model from live trades + compare to 3 existing models.

Goal
----
The paper_trader uses BUY_SLIPPAGE_BPS=10 fixed (Jupiter Ultra RFQ assumption).
Real fill divergence on live trades shows 100-1800 bps depending on liq, age,
latency. This script:

  1. Exports buy_slippage_bps from paper_trades source=rt_live
  2. Compares to predictions of 3 existing slip models:
       - strategies.BUY_SLIPPAGE_BPS = 10 (current paper buy)
       - sim_engines.compute_buy_slippage(pos, liq) (AMM theoretical)
       - sim.py inline: max(2%, min(10%, 0.02 + 0.08*(1-min(liq,50k)/50k)))
       - paper_trader._liq_slip_multiplier(liq) * 100 bps default type
  3. Fits OLS multi-factor model (numpy lstsq, no sklearn dep)
  4. Reports R², residuals, recommended coefficients

Run:  python scripts/_calibrate_buy_slip.py
Out:  data/buy_slip_calibration.json
"""
import os
import sys
import io
import json
import math
from collections import Counter
from dotenv import load_dotenv

# Force UTF-8 stdout (Windows cp1252 default breaks on unicode arrows)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))

import numpy as np
from supabase import create_client

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

SINCE = os.environ.get("SINCE", "2026-04-08T00:00:00+00:00")  # post-v121


def fetch_all(tbl, sel, **f):
    out, step, off = [], 1000, 0
    while True:
        q = sb.table(tbl).select(sel)
        for k, v in f.items():
            if k.startswith("gte_"):
                q = q.gte(k[4:], v)
            elif k.startswith("eq_"):
                q = q.eq(k[3:], v)
            elif k.startswith("not_"):
                q = q.not_.is_(k[4:], v)
        r = q.range(off, off + step - 1).execute()
        if not r.data:
            break
        out.extend(r.data)
        if len(r.data) < step:
            break
        off += step
    return out


# --- Existing model implementations (for comparison) -----------------------

def model_paper_constant(_liq, _pos, _age, _lat, _pump):
    """strategies.BUY_SLIPPAGE_BPS = 10 (the prod paper buy slip)."""
    return 10.0


def model_sim_engines(liq, pos, _age, _lat, _pump):
    """sim_engines.compute_buy_slippage — AMM theoretical."""
    if liq <= 0:
        return 1.5 * 100  # 150 bps fallback
    impact = pos / (liq + pos)
    return (0.010 + impact) * 10000  # bps


def model_sim_inline(liq, _pos, _age, _lat, _pump):
    """sim.py:1247-1248 — linear liq → slip."""
    slip_pct = max(0.02, min(0.10, 0.02 + 0.08 * (1 - min(liq, 50_000) / 50_000)))
    return slip_pct * 10000  # bps


def model_paper_dynamic_sell(liq, _pos, _age, _lat, _pump):
    """paper_trader._dynamic_sell_slip_factor reused as buy proxy.
    base type=100bps (default), liq_mult, GLOBAL_OFFSET=-100."""
    if liq <= 0:
        liq = 50_000
    liq_clamped = max(500, min(50_000, liq))
    liq_mult = 1.0 + 0.5 * max(0, math.log10(50_000 / liq_clamped))
    liq_mult = max(1.0, min(2.5, liq_mult))
    type_bps = 100  # default exit_type
    return int(type_bps * liq_mult) - 100  # GLOBAL_OFFSET = -100


MODELS = {
    "paper_const_10bps": model_paper_constant,
    "sim_engines_amm": model_sim_engines,
    "sim_inline_linear": model_sim_inline,
    "paper_dynamic_sell_proxy": model_paper_dynamic_sell,
}


# --- Fetch ------------------------------------------------------------------

print(f"Fetching live trades since {SINCE}...")
rows = fetch_all(
    "paper_trades",
    "id,token_address,symbol,strategy,created_at,status,source,"
    "position_usd,rt_liquidity_usd,rt_volume_24h,rt_buy_sell_ratio,"
    "rt_token_age_hours,rt_is_pump_fun,buy_slippage_bps,sell_slippage_bps,"
    "message_to_buy_seconds,entry_price,dex_spot_price_at_entry,"
    "buy_input_lamports,buy_output_tokens,sol_price_at_entry,chain",
    eq_source="rt_live",
    gte_created_at=SINCE,
)
print(f"  raw rt_live rows: {len(rows)}")


def usable(r):
    if r.get("chain") not in (None, "solana"):
        return False  # Solana-only; EVM has its own model
    bs = r.get("buy_slippage_bps")
    if bs is None:
        return False
    if r.get("rt_liquidity_usd") in (None, 0):
        return False
    if r.get("position_usd") in (None, 0):
        return False
    return True


data = [r for r in rows if usable(r)]
print(f"  usable (non-null buy_slip + liq + pos): {len(data)}")

if len(data) < 20:
    print("ERROR: insufficient data for fit (need ≥20 trades).")
    sys.exit(1)


# --- Feature extraction ----------------------------------------------------

def extract(r):
    return {
        "y_bps": float(r["buy_slippage_bps"]),
        "liq": float(r["rt_liquidity_usd"]),
        "pos": float(r["position_usd"]),
        "age_h": float(r.get("rt_token_age_hours") or 24.0),
        "lat_s": float(r.get("message_to_buy_seconds") or 30.0),
        "pump": 1.0 if r.get("rt_is_pump_fun") else 0.0,
        "vol_24h": float(r.get("rt_volume_24h") or 0),
        "bsr": float(r.get("rt_buy_sell_ratio") or 0.5),
        "symbol": r.get("symbol", "?"),
        "strategy": r.get("strategy", "?"),
    }


feats = [extract(r) for r in data]
y = np.array([f["y_bps"] for f in feats])

print(f"\n{'='*70}")
print("DESCRIPTIVE STATS - buy_slippage_bps observed (live trades)")
print(f"{'='*70}")
print(f"N            = {len(y)}")
print(f"mean         = {y.mean():+.0f} bps")
print(f"median       = {np.median(y):+.0f} bps")
print(f"p25          = {np.percentile(y, 25):+.0f} bps")
print(f"p75          = {np.percentile(y, 75):+.0f} bps")
print(f"p95          = {np.percentile(y, 95):+.0f} bps")
print(f"max          = {y.max():+.0f} bps")
print(f"min          = {y.min():+.0f} bps")
print(f"std          = {y.std():.0f} bps")

# Outlier flag (e.g., ASSHOLES rug-at-trade)
outliers = [f for f in feats if abs(f["y_bps"]) > 5000]
if outliers:
    print(f"\n[!] {len(outliers)} outliers > 5000 bps:")
    for o in outliers[:5]:
        print(f"     {o['symbol']:<12} {o['strategy']:<28} {o['y_bps']:+.0f} bps "
              f"liq=${o['liq']:.0f} lat={o['lat_s']:.0f}s")
    print("     (likely rug-at-trade or very stale messages — included in fit)")


# --- Compare existing models -----------------------------------------------

print(f"\n{'='*70}")
print("EXISTING MODELS — predicted vs actual")
print(f"{'='*70}")
print(f"{'model':<28}{'mean_pred':>12}{'MAE_bps':>12}{'bias_bps':>12}{'R²':>10}")
print("-" * 74)

ss_tot = ((y - y.mean()) ** 2).sum()
results = {"observed": {"n": len(y), "mean": float(y.mean()),
                        "median": float(np.median(y)),
                        "p95": float(np.percentile(y, 95))}}

for name, fn in MODELS.items():
    preds = np.array([fn(f["liq"], f["pos"], f["age_h"], f["lat_s"], f["pump"]) for f in feats])
    mae = np.abs(preds - y).mean()
    bias = (preds - y).mean()  # negative = model under-predicts (paper too optimistic)
    ss_res = ((y - preds) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    print(f"{name:<28}{preds.mean():>+12.0f}{mae:>12.0f}{bias:>+12.0f}{r2:>+10.3f}")
    results[name] = {"mean_pred": float(preds.mean()), "mae_bps": float(mae),
                     "bias_bps": float(bias), "r2": float(r2)}

print("\nNote: bias < 0 → model UNDER-predicts → paper assumes less slip than reality.")


# --- Fit empirical OLS model -----------------------------------------------

print(f"\n{'='*70}")
print("EMPIRICAL FIT — OLS multi-factor (numpy lstsq)")
print(f"{'='*70}")

# Features (all log-transformed where positive-skewed)
# x1 = log10(liq/1000) -- liq in K
# x2 = log10(pos)
# x3 = log10(1 + age_h)
# x4 = log10(1 + lat_s)
# x5 = pump (binary)
# x6 = pos/liq (impact ratio)
X_full = np.column_stack([
    np.ones(len(feats)),                                   # intercept
    np.array([math.log10(max(f["liq"], 100) / 1000) for f in feats]),
    np.array([math.log10(max(f["pos"], 0.5)) for f in feats]),
    np.array([math.log10(1 + f["age_h"]) for f in feats]),
    np.array([math.log10(1 + f["lat_s"]) for f in feats]),
    np.array([f["pump"] for f in feats]),
    np.array([f["pos"] / max(f["liq"], 100) for f in feats]),
])
feat_names = ["intercept", "log10(liq/1k)", "log10(pos)",
              "log10(1+age_h)", "log10(1+lat_s)", "pump_fun", "pos/liq"]

beta, residuals, rank, sv = np.linalg.lstsq(X_full, y, rcond=None)
preds_full = X_full @ beta
ss_res_full = ((y - preds_full) ** 2).sum()
r2_full = 1 - ss_res_full / ss_tot if ss_tot > 0 else 0
mae_full = np.abs(preds_full - y).mean()

print(f"\nFull model (all features): R²={r2_full:.3f}  MAE={mae_full:.0f} bps")
print(f"\n{'feature':<22}{'coef':>14}{'|coef|*std(x)':>18}")
print("-" * 56)
for nm, b, col in zip(feat_names, beta, X_full.T):
    impact = abs(b) * col.std()
    print(f"{nm:<22}{b:>+14.1f}{impact:>+18.1f}")


# --- Simplified 2-feature model (production-friendly) ----------------------

print(f"\n{'='*70}")
print("SIMPLIFIED FIT — 2 features (drop-in replacement for BUY_SLIPPAGE_BPS)")
print(f"{'='*70}")

# Just liq + pos/liq (most production-friendly, no need for latency at quote-time)
X_simple = np.column_stack([
    np.ones(len(feats)),
    np.array([math.log10(max(f["liq"], 100) / 1000) for f in feats]),
    np.array([f["pos"] / max(f["liq"], 100) for f in feats]),
])
beta_s, _, _, _ = np.linalg.lstsq(X_simple, y, rcond=None)
preds_s = X_simple @ beta_s
ss_res_s = ((y - preds_s) ** 2).sum()
r2_s = 1 - ss_res_s / ss_tot if ss_tot > 0 else 0
mae_s = np.abs(preds_s - y).mean()
print(f"\nSimple model: R²={r2_s:.3f}  MAE={mae_s:.0f} bps")
print(f"  buy_slip_bps = {beta_s[0]:+.0f} + {beta_s[1]:+.0f} * log10(liq/1000) + {beta_s[2]:+.1f} * (pos/liq)")
print(f"\n  Examples (pos=$1.72):")
for liq in (1000, 5000, 17000, 50000, 200000):
    pred = beta_s[0] + beta_s[1] * math.log10(liq / 1000) + beta_s[2] * (1.72 / liq)
    print(f"    liq=${liq:>8,d}  →  predicted slip = {pred:+.0f} bps")


# --- Per-bucket descriptive (sanity) ---------------------------------------

print(f"\n{'='*70}")
print("PER-LIQ-BUCKET MEDIAN (sanity check)")
print(f"{'='*70}")
print(f"{'liq bucket':<20}{'N':>6}{'median_bps':>14}{'p95_bps':>12}")
print("-" * 52)
buckets = [(0, 5_000), (5_000, 20_000), (20_000, 50_000),
           (50_000, 100_000), (100_000, 1e9)]
bucket_stats = {}
for lo, hi in buckets:
    sub = [f["y_bps"] for f in feats if lo <= f["liq"] < hi]
    if not sub:
        continue
    label = f"${lo/1000:.0f}K-${hi/1000:.0f}K" if hi < 1e8 else f"${lo/1000:.0f}K+"
    print(f"{label:<20}{len(sub):>6}{np.median(sub):>+14.0f}{np.percentile(sub, 95):>+12.0f}")
    bucket_stats[label] = {"n": len(sub), "median": float(np.median(sub)),
                           "p95": float(np.percentile(sub, 95))}


# --- Robust fit (winsorize ±5000 bps, exclude rug-at-trade like ASSHOLES) ---

print(f"\n{'='*70}")
print("ROBUST FIT - excluding |slip| > 5000 bps (rug-at-trade outliers)")
print(f"{'='*70}")

mask_robust = np.abs(y) <= 5000
y_r = y[mask_robust]
X_r = X_full[mask_robust]
print(f"  N before: {len(y)}, after winsorize: {len(y_r)} (dropped {len(y) - len(y_r)})")

beta_r, _, _, _ = np.linalg.lstsq(X_r, y_r, rcond=None)
preds_r = X_r @ beta_r
ss_tot_r = ((y_r - y_r.mean()) ** 2).sum()
ss_res_r = ((y_r - preds_r) ** 2).sum()
r2_r = 1 - ss_res_r / ss_tot_r if ss_tot_r > 0 else 0
mae_r = np.abs(preds_r - y_r).mean()
print(f"\n  Robust full model: R²={r2_r:.3f}  MAE={mae_r:.0f} bps  median(y)={np.median(y_r):+.0f} bps")
print(f"\n  {'feature':<22}{'coef':>14}")
for nm, b in zip(feat_names, beta_r):
    print(f"  {nm:<22}{b:>+14.1f}")


# --- Constant median fallback (production-ready honest baseline) -----------

print(f"\n{'='*70}")
print("HONEST BASELINE - constant = robust median(y)")
print(f"{'='*70}")
median_robust = float(np.median(y_r))
mae_const = float(np.abs(y_r - median_robust).mean())
print(f"  buy_slip = {median_robust:+.0f} bps (constant)")
print(f"  MAE = {mae_const:.0f} bps  (vs full-fit MAE = {mae_r:.0f} bps)")
print(f"  Improvement of full-fit over constant: {(mae_const - mae_r) / mae_const * 100:.1f}%")
print(f"\n  -> If full-fit improves <5% over constant, the features don't help.")
print(f"     In that case, recommend: BUY_SLIPPAGE_BPS = {round(median_robust)} (replace 10)")

results["robust_full"] = {
    "n": int(len(y_r)),
    "features": feat_names,
    "coefficients": [float(b) for b in beta_r],
    "r2": float(r2_r),
    "mae_bps": float(mae_r),
}
results["constant_median"] = {
    "median_bps": median_robust,
    "mae_bps": mae_const,
    "recommendation": f"BUY_SLIPPAGE_BPS = {round(median_robust)}",
}


# --- Save -----------------------------------------------------------------

results["empirical_full"] = {
    "features": feat_names,
    "coefficients": [float(b) for b in beta],
    "r2": float(r2_full),
    "mae_bps": float(mae_full),
}
results["empirical_simple"] = {
    "formula": "buy_slip_bps = c0 + c1 * log10(liq/1000) + c2 * (pos/liq)",
    "coefficients": {"c0": float(beta_s[0]), "c1": float(beta_s[1]), "c2": float(beta_s[2])},
    "r2": float(r2_s),
    "mae_bps": float(mae_s),
}
results["per_liq_bucket"] = bucket_stats

out_path = os.path.join(os.path.dirname(__file__), "..", "data", "buy_slip_calibration.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {out_path}")


# --- Propagate to all 4 layers atomically --------------------------------
# v14e.34: close the boucle. strategies.py is the source of truth, but the
# scoring_config.paper_trade_config JSONB OVERRIDES it at runtime
# (paper_trader.py:1095, 2393, 2815). Forgetting to update the JSONB caused
# v14e.24-25 to silently drift for 2 days (Apr 25-27): all batch SOL trades
# kept using buy=100 from JSONB while strategies.py was at 225.
#
# Auto-update the JSONB to match the recommended median, then remind the
# operator to manually edit strategies.py + sim_engines.py + sim.py too
# (those are imports, not DB-driven, so they need a code commit).
recommended = round(median_robust)
print(f"\n{'='*70}")
print("PROPAGATION CHECK - all 4 layers must match")
print(f"{'='*70}")

apply = "--apply" in sys.argv
if apply:
    try:
        cur = sb.table("scoring_config").select("paper_trade_config").eq("id", 1).execute()
        ptc = (cur.data or [{}])[0].get("paper_trade_config") or {}
        if isinstance(ptc, str):
            ptc = json.loads(ptc)
        prev = ptc.get("buy_slippage_bps")
        ptc["buy_slippage_bps"] = recommended
        sb.table("scoring_config").update({
            "paper_trade_config": ptc,
            "updated_by": "_calibrate_buy_slip",
            "change_reason": f"buy_slip recalibration: {prev} -> {recommended} bps",
        }).eq("id", 1).execute()
        print(f"  [OK] scoring_config.paper_trade_config.buy_slippage_bps {prev} -> {recommended}")
    except Exception as e:
        print(f"  [ERR] DB update failed: {e}")
else:
    print(f"  Run with --apply to update scoring_config.paper_trade_config.buy_slippage_bps -> {recommended}")

print(f"\nMANUAL STEPS still required (code, not DB):")
print(f"  - scraper/strategies.py:        BUY_SLIPPAGE_BPS = {recommended}")
print(f"  - scraper/sim.py / sim_engines: imports from strategies (auto-aligned via import)")
print(f"  - scraper/optimize_strategies.py: imports from strategies (auto-aligned)")
print(f"  Commit these together to keep all 4 layers (paper / sim / shadow / live) coherent.")
