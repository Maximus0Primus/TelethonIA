"""SELL slip drift health-check via twin-pair (live vs in-loop paper sim) PnL delta.

Approach
--------
The right way to measure paper-vs-live SELL slip drift is NOT raw sell_slippage_bps
(opposite sign convention vs the BUY column + dominated by 5% rug outliers), but
the delta `pnl_pct − paper_sim_pnl_pct` recorded on every live trade by the
v143.6 in-loop paper sim companion. That's the ground-truth "did paper predict
my live PnL".

This mirrors the v144 (Apr 19) calibration methodology that produced the current
`_dynamic_sell_slip_factor` type_bps lookup. v144 used 77 twin trades; this
script automates the same analysis to detect drift over time.

Reads
-----
paper_trades rows where:
  source='rt_live' AND chain='solana' AND exit_at IS NOT NULL
  AND paper_sim_pnl_pct IS NOT NULL AND status in {tp_hit, sl_hit, timeout, trail_stop, be_stop}

Computes
--------
For each exit_type:
  delta_pp = (pnl_pct − paper_sim_pnl_pct) × 100
  metrics: N, mean, median, std, p25, p75
Aggregate: weighted median across exit types (weighted by N).

Decision rule
-------------
The current GLOBAL_OFFSET_BPS = −100 in `_dynamic_sell_slip_factor`. If the
weighted-median delta is:
  |median| < 1.0pp → no drift, leave model alone
  median < −1.0pp → live underperforms paper → shift GLOBAL_OFFSET ↑ by |median|*100 bps
  median > +1.0pp → live outperforms paper → shift GLOBAL_OFFSET ↓ by median*100 bps

Run:  python scripts/_calibrate_sell_slip.py [--since YYYY-MM-DD]
Out:  data/sell_slip_drift.json
"""
import os
import sys
import io
import json
import argparse
from dotenv import load_dotenv

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))

import numpy as np
from supabase import create_client

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

ap = argparse.ArgumentParser()
ap.add_argument("--since", default="2026-04-19T11:42:00+00:00",
                help="ISO date — default = post-v144 calibration window")
ap.add_argument("--threshold-pp", type=float, default=1.0,
                help="abs median pp threshold to recommend GLOBAL_OFFSET shift")
args = ap.parse_args()


def fetch_all(tbl, sel, **f):
    out, step, off = [], 1000, 0
    while True:
        q = sb.table(tbl).select(sel)
        for k, v in f.items():
            if k.startswith("gte_"):
                q = q.gte(k[4:], v)
            elif k.startswith("eq_"):
                q = q.eq(k[3:], v)
            elif k.startswith("in_"):
                q = q.in_(k[3:], v)
        r = q.range(off, off + step - 1).execute()
        if not r.data:
            break
        out.extend(r.data)
        if len(r.data) < step:
            break
        off += step
    return out


VALID_STATUSES = ["tp_hit", "sl_hit", "timeout", "trail_stop", "be_stop"]

print(f"Fetching twin pairs since {args.since}...")
rows = fetch_all(
    "paper_trades",
    "id,token_address,symbol,strategy,created_at,status,"
    "pnl_pct,paper_sim_pnl_pct,rt_liquidity_usd",
    eq_source="rt_live",
    gte_created_at=args.since,
    in_status=VALID_STATUSES,
)
print(f"  raw rt_live rows: {len(rows)}")

data = [r for r in rows if r.get("paper_sim_pnl_pct") is not None
        and r.get("pnl_pct") is not None]
print(f"  with paper_sim_pnl_pct: {len(data)}")

if len(data) < 20:
    print("ERROR: insufficient twin pairs for drift detection (need ≥20).")
    sys.exit(1)

deltas_by_status = {}
for r in data:
    delta_pp = (float(r["pnl_pct"]) - float(r["paper_sim_pnl_pct"])) * 100
    deltas_by_status.setdefault(r["status"], []).append(delta_pp)

print(f"\n{'='*70}")
print("PAPER vs LIVE PNL DRIFT — twin-pair delta (live − paper_sim)")
print(f"{'='*70}")
print(f"{'status':<14}{'N':>5}{'mean':>9}{'median':>10}{'std':>9}{'p25':>9}{'p75':>9}")
print("-" * 65)
results = {"by_status": {}, "since": args.since}
weighted_n_total = 0
weighted_median_sum = 0.0
all_deltas = []
for status in sorted(deltas_by_status.keys()):
    arr = np.array(deltas_by_status[status])
    n = len(arr)
    mean = float(arr.mean())
    median = float(np.median(arr))
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    p25 = float(np.percentile(arr, 25))
    p75 = float(np.percentile(arr, 75))
    print(f"{status:<14}{n:>5}{mean:>+9.2f}{median:>+10.2f}{std:>9.2f}{p25:>+9.2f}{p75:>+9.2f}")
    results["by_status"][status] = {
        "n": n, "mean_pp": mean, "median_pp": median,
        "std_pp": std, "p25_pp": p25, "p75_pp": p75,
    }
    weighted_n_total += n
    weighted_median_sum += median * n
    all_deltas.extend(arr.tolist())

weighted_median = weighted_median_sum / weighted_n_total
overall_median = float(np.median(all_deltas))
overall_mean = float(np.mean(all_deltas))

print(f"\n{'='*70}")
print(f"AGGREGATE (N={weighted_n_total} live trades)")
print(f"{'='*70}")
print(f"  Weighted median Δ (per-status, N-weighted) = {weighted_median:+.2f} pp")
print(f"  Overall median Δ (pooled)                  = {overall_median:+.2f} pp")
print(f"  Overall mean Δ (pooled)                    = {overall_mean:+.2f} pp")

results["aggregate"] = {
    "n": weighted_n_total,
    "weighted_median_pp": weighted_median,
    "overall_median_pp": overall_median,
    "overall_mean_pp": overall_mean,
}

print(f"\n{'='*70}")
print("DECISION")
print(f"{'='*70}")
threshold = args.threshold_pp
if abs(weighted_median) < threshold:
    decision = (
        f"NO DRIFT (|{weighted_median:+.2f}pp| < {threshold}pp threshold). "
        "v144 calibration still valid. No code change."
    )
elif weighted_median < -threshold:
    shift = round(-weighted_median * 100)  # bps
    decision = (
        f"LIVE UNDERPERFORMS PAPER by {abs(weighted_median):.2f}pp. "
        f"Shift GLOBAL_OFFSET_BPS from −100 to {-100 + shift} "
        f"(less favorable bonus to paper)."
    )
else:
    shift = round(weighted_median * 100)  # bps
    decision = (
        f"LIVE OUTPERFORMS PAPER by {weighted_median:.2f}pp. "
        f"Shift GLOBAL_OFFSET_BPS from −100 to {-100 - shift} "
        f"(more favorable bonus to paper)."
    )

print(f"  → {decision}")
results["decision"] = decision
results["threshold_pp"] = threshold

# Note on tp_hit asymmetry
if "tp_hit" in deltas_by_status:
    arr = np.array(deltas_by_status["tp_hit"])
    if len(arr) >= 5 and arr.mean() > 5 and abs(np.median(arr)) < 5:
        print(f"\n  [NOTE] tp_hit shows asymmetric upside (mean +{arr.mean():.1f}pp, "
              f"median {np.median(arr):+.1f}pp).")
        print("         Jupiter Ultra fills above TP during pumps. Paper conservative")
        print("         here is FINE — the favorable surprise lifts live PnL.")

out_path = os.path.join(os.path.dirname(__file__), "..", "data",
                        "sell_slip_drift.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {out_path}")
