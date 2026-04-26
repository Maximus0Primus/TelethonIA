"""Empirically fit SELL slippage model from live trades + compare to current dynamic model.

Goal
----
The paper_trader uses a dynamic sell slip via `_dynamic_sell_slip_factor(exit_type, liq)`
that combines an exit-type baseline (sl_hit=435, tp_hit=-300, timeout=120, etc.) with
a log-linear liq multiplier and a -100 bps global offset.

Live trades since v121 (2026-04-08) populate `sell_slippage_bps` directly. This script:

  1. Exports observed sell_slippage_bps from rt_live closed trades
  2. Compares actual vs current `_dynamic_sell_slip_factor` predictions
  3. Per-exit-type breakdown — does the type_bps lookup still fit?
  4. Fits OLS (intercept + log10(liq/1k) + log10(pos) + per-exit-type one-hots)
  5. Reports R², residuals, recommended type_bps recalibration
  6. Outputs `data/sell_slip_calibration.json`

Run:  python scripts/_calibrate_sell_slip.py
Out:  data/sell_slip_calibration.json
"""
import os
import sys
import io
import json
import math
from collections import defaultdict
from dotenv import load_dotenv

# Force UTF-8 stdout (Windows cp1252 default breaks on unicode arrows)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))

import numpy as np
from supabase import create_client

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

SINCE = os.environ.get("SINCE", "2026-04-08T00:00:00+00:00")  # post-v121 (when sell_slip started logging)

# Exit types tracked by the dynamic model (paper_trader.py:1641-1656)
EXIT_TYPE_BPS_CURRENT = {
    "trail_crash": 1000,
    "sl_hit":       435,
    "trail_stop":   250,
    "tp_hit":      -300,
    "timeout":      120,
    "be_stop":      200,
    "tp_late":       80,
}
GLOBAL_OFFSET_BPS = -100
DEFAULT_TYPE_BPS = 100

# status -> exit_type. trail_stop maps to itself; reconciled/orphan_bag excluded
# from fit (closed by reconciler, didn't go through the slip model).
STATUS_TO_EXIT_TYPE = {
    "take_profit":     "tp_hit",
    "stop_loss":       "sl_hit",
    "sl_hit":          "sl_hit",
    "tp_hit":          "tp_hit",
    "timeout":         "timeout",
    "breakeven_stop":  "be_stop",
    "be_stop":         "be_stop",
    "trail_stop":      "trail_stop",
    "trail_crash":     "trail_crash",
}


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


# Mirror of paper_trader._liq_slip_multiplier
def liq_slip_mult(liq_usd: float) -> float:
    """v14e.6: continuous log-linear liquidity → slip multiplier (paper_trader.py:1575)."""
    if liq_usd <= 0:
        liq_usd = 50_000
    liq_clamped = max(500, min(50_000, liq_usd))
    mult = 1.0 + 0.5 * max(0, math.log10(50_000 / liq_clamped))
    return max(1.0, min(2.5, mult))


def model_dynamic_current(exit_type: str, liq: float) -> float:
    """Replicate paper_trader._dynamic_sell_slip_factor — Solana branch only."""
    type_bps = EXIT_TYPE_BPS_CURRENT.get(exit_type, DEFAULT_TYPE_BPS)
    mult = liq_slip_mult(liq)
    adjusted = int(type_bps * mult) + GLOBAL_OFFSET_BPS
    if exit_type == "trail_crash":
        adjusted = max(-1000, min(2500, adjusted))
    else:
        adjusted = max(-1000, min(1500, adjusted))
    return float(adjusted)


# --- Fetch ------------------------------------------------------------------

print(f"Fetching live trades since {SINCE}...")
rows = fetch_all(
    "paper_trades",
    "id,token_address,symbol,strategy,created_at,status,source,"
    "position_usd,rt_liquidity_usd,rt_volume_24h,rt_buy_sell_ratio,"
    "rt_token_age_hours,rt_is_pump_fun,sell_slippage_bps,buy_slippage_bps,"
    "message_to_buy_seconds,exit_price,paper_exit_price,"
    "tp_price,sl_price,chain",
    eq_source="rt_live",
    gte_created_at=SINCE,
)
print(f"  raw rt_live rows: {len(rows)}")


def usable(r):
    if r.get("chain") not in (None, "solana"):
        return False  # Solana-only; EVM has its own gas-based model
    if r.get("status") not in STATUS_TO_EXIT_TYPE:
        return False  # exclude reconciled, orphan_bag, open
    if r.get("sell_slippage_bps") is None:
        return False
    if r.get("rt_liquidity_usd") in (None, 0):
        return False
    return True


data = [r for r in rows if usable(r)]
print(f"  usable (closed, non-null sell_slip + liq, slip-modelled status): {len(data)}")

if len(data) < 20:
    print("ERROR: insufficient data for fit (need ≥20 trades).")
    sys.exit(1)


# --- Feature extraction ----------------------------------------------------

def extract(r):
    return {
        "y_bps": float(r["sell_slippage_bps"]),
        "liq": float(r["rt_liquidity_usd"]),
        "pos": float(r.get("position_usd") or 1.72),
        "age_h": float(r.get("rt_token_age_hours") or 24.0),
        "lat_s": float(r.get("message_to_buy_seconds") or 30.0),
        "pump": 1.0 if r.get("rt_is_pump_fun") else 0.0,
        "exit_type": STATUS_TO_EXIT_TYPE[r["status"]],
        "symbol": r.get("symbol", "?"),
        "strategy": r.get("strategy", "?"),
    }


feats = [extract(r) for r in data]
y = np.array([f["y_bps"] for f in feats])

print(f"\n{'='*70}")
print("DESCRIPTIVE STATS - sell_slippage_bps observed (live trades)")
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

outliers = [f for f in feats if abs(f["y_bps"]) > 5000]
if outliers:
    print(f"\n[!] {len(outliers)} outliers > 5000 bps:")
    for o in outliers[:5]:
        print(f"     {o['symbol']:<12} {o['strategy']:<22} {o['exit_type']:<12} "
              f"{o['y_bps']:+.0f} bps liq=${o['liq']:.0f}")


# --- Per-exit-type breakdown -----------------------------------------------

print(f"\n{'='*70}")
print("PER-EXIT-TYPE OBSERVED MEDIAN vs CURRENT DYNAMIC MODEL TYPE_BPS")
print(f"{'='*70}")
print(f"{'exit_type':<14}{'N':>5}{'obs_median':>12}{'obs_p95':>10}"
      f"{'curr_type_bps':>16}{'recommend':>12}")
print("-" * 70)
per_exit = {}
for et in sorted(set(f["exit_type"] for f in feats)):
    sub = [f["y_bps"] for f in feats if f["exit_type"] == et]
    obs_med = float(np.median(sub))
    obs_p95 = float(np.percentile(sub, 95))
    curr = EXIT_TYPE_BPS_CURRENT.get(et, DEFAULT_TYPE_BPS)
    # recommended type_bps such that median(predict) ≈ obs_median
    # predict = type * mult + offset → type = (median + 100) / median(mult)
    mults = [liq_slip_mult(f["liq"]) for f in feats if f["exit_type"] == et]
    avg_mult = float(np.mean(mults))
    recommended = round((obs_med - GLOBAL_OFFSET_BPS) / avg_mult)
    print(f"{et:<14}{len(sub):>5}{obs_med:>+12.0f}{obs_p95:>+10.0f}"
          f"{curr:>+16d}{recommended:>+12d}")
    per_exit[et] = {
        "n": len(sub),
        "obs_median_bps": obs_med,
        "obs_p95_bps": obs_p95,
        "current_type_bps": curr,
        "recommended_type_bps": recommended,
        "avg_liq_mult": avg_mult,
    }


# --- Compare current dynamic model to actuals ------------------------------

print(f"\n{'='*70}")
print("CURRENT DYNAMIC MODEL — predicted vs actual")
print(f"{'='*70}")

preds_dyn = np.array([model_dynamic_current(f["exit_type"], f["liq"]) for f in feats])
mae_dyn = float(np.abs(preds_dyn - y).mean())
bias_dyn = float((preds_dyn - y).mean())
ss_tot = ((y - y.mean()) ** 2).sum()
ss_res_dyn = ((y - preds_dyn) ** 2).sum()
r2_dyn = float(1 - ss_res_dyn / ss_tot) if ss_tot > 0 else 0.0
print(f"  mean_pred = {preds_dyn.mean():+.0f} bps")
print(f"  MAE       = {mae_dyn:.0f} bps")
print(f"  bias      = {bias_dyn:+.0f} bps  (pred − actual; <0 = model under-predicts)")
print(f"  R²        = {r2_dyn:+.3f}")


# --- Constant median fallback ----------------------------------------------

mask_robust = np.abs(y) <= 5000
y_r = y[mask_robust]
median_robust = float(np.median(y_r))
mae_const = float(np.abs(y_r - median_robust).mean())

print(f"\n{'='*70}")
print("CONSTANT MEDIAN BASELINE (winsorized |y|<=5000)")
print(f"{'='*70}")
print(f"  N         = {len(y_r)} (dropped {len(y) - len(y_r)} extreme outliers)")
print(f"  median    = {median_robust:+.0f} bps")
print(f"  MAE       = {mae_const:.0f} bps")


# --- OLS empirical fit (intercept + log10(liq) + per-exit-type one-hots) ---

print(f"\n{'='*70}")
print("EMPIRICAL FIT — OLS (intercept + log liq + log pos + exit-type one-hots)")
print(f"{'='*70}")

exit_types_observed = sorted(set(f["exit_type"] for f in feats))
# Use first as reference (drop one to avoid collinearity with intercept)
ref_type = exit_types_observed[0]
dummy_types = exit_types_observed[1:]

cols = [
    np.ones(len(feats)),
    np.array([math.log10(max(f["liq"], 100) / 1000) for f in feats]),
    np.array([math.log10(max(f["pos"], 0.5)) for f in feats]),
]
col_names = ["intercept", "log10(liq/1k)", "log10(pos)"]
for et in dummy_types:
    cols.append(np.array([1.0 if f["exit_type"] == et else 0.0 for f in feats]))
    col_names.append(f"is_{et}")

X = np.column_stack(cols)
beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
preds_ols = X @ beta
ss_res_ols = ((y - preds_ols) ** 2).sum()
r2_ols = float(1 - ss_res_ols / ss_tot) if ss_tot > 0 else 0.0
mae_ols = float(np.abs(preds_ols - y).mean())

print(f"\n  Reference exit_type (absorbed in intercept): {ref_type}")
print(f"  R²={r2_ols:.3f}  MAE={mae_ols:.0f} bps")
print(f"\n  {'feature':<22}{'coef':>14}")
for nm, b in zip(col_names, beta):
    print(f"  {nm:<22}{b:>+14.1f}")


# --- Robust OLS (winsorize) -----------------------------------------------

print(f"\n{'='*70}")
print("ROBUST OLS — exclude |slip| > 5000 bps")
print(f"{'='*70}")
y_rob = y[mask_robust]
X_rob = X[mask_robust]
beta_r, _, _, _ = np.linalg.lstsq(X_rob, y_rob, rcond=None)
preds_rob = X_rob @ beta_r
ss_tot_r = ((y_rob - y_rob.mean()) ** 2).sum()
r2_rob = float(1 - ((y_rob - preds_rob) ** 2).sum() / ss_tot_r) if ss_tot_r > 0 else 0.0
mae_rob = float(np.abs(preds_rob - y_rob).mean())
print(f"  N={len(y_rob)}  R²={r2_rob:.3f}  MAE={mae_rob:.0f} bps")
print(f"\n  {'feature':<22}{'coef':>14}")
for nm, b in zip(col_names, beta_r):
    print(f"  {nm:<22}{b:>+14.1f}")


# --- Recommendation -------------------------------------------------------

print(f"\n{'='*70}")
print("RECOMMENDATION")
print(f"{'='*70}")
improvement_dyn = (mae_const - mae_dyn) / mae_const * 100 if mae_const > 0 else 0
improvement_ols = (mae_const - mae_rob) / mae_const * 100 if mae_const > 0 else 0
print(f"\n  Constant median MAE   = {mae_const:.0f} bps  (baseline)")
print(f"  Current dynamic MAE   = {mae_dyn:.0f} bps  ({improvement_dyn:+.1f}% vs constant)")
print(f"  Robust OLS MAE        = {mae_rob:.0f} bps  ({improvement_ols:+.1f}% vs constant)")
print()
if improvement_dyn < 5 and improvement_ols < 10:
    print(f"  -> Features explain <10% of variance over a constant median.")
    print(f"     Recommend: SELL_SLIPPAGE_BPS = {round(median_robust)} (constant, replace 10).")
    print(f"     Drop _dynamic_sell_slip_factor — it's overfit to noise.")
elif improvement_ols >= 10 and improvement_ols > improvement_dyn + 5:
    print(f"  -> OLS beats both constant and current dynamic model.")
    print(f"     Recommend: keep dynamic structure but recalibrate type_bps")
    print(f"     to the per-exit-type recommended values above.")
else:
    print(f"  -> Current dynamic model is close to OLS optimum.")
    print(f"     Recommend: minor type_bps tweaks per the per-exit-type table.")


# --- Save -----------------------------------------------------------------

results = {
    "generated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
    "since": SINCE,
    "n_total": len(y),
    "observed": {
        "mean_bps": float(y.mean()),
        "median_bps": float(np.median(y)),
        "p25_bps": float(np.percentile(y, 25)),
        "p75_bps": float(np.percentile(y, 75)),
        "p95_bps": float(np.percentile(y, 95)),
        "std_bps": float(y.std()),
    },
    "current_dynamic_model": {
        "mean_pred_bps": float(preds_dyn.mean()),
        "mae_bps": mae_dyn,
        "bias_bps": bias_dyn,
        "r2": r2_dyn,
        "type_bps_lookup": EXIT_TYPE_BPS_CURRENT,
        "global_offset_bps": GLOBAL_OFFSET_BPS,
    },
    "per_exit_type": per_exit,
    "constant_median": {
        "n_robust": int(len(y_r)),
        "median_bps": median_robust,
        "mae_bps": mae_const,
        "recommendation": f"SELL_SLIPPAGE_BPS = {round(median_robust)}",
    },
    "ols_robust": {
        "features": col_names,
        "reference_exit_type": ref_type,
        "coefficients": [float(b) for b in beta_r],
        "r2": r2_rob,
        "mae_bps": mae_rob,
    },
    "improvement_pct": {
        "dynamic_vs_constant": improvement_dyn,
        "ols_vs_constant": improvement_ols,
    },
}

out_path = os.path.join(os.path.dirname(__file__), "..", "data", "sell_slip_calibration.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {out_path}")
