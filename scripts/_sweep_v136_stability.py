"""v136 stability sweep — runs the realistic sweep on a SHORTER, more recent
window (last 3 days only). Compare top-10 strats vs the 5-day v136 to detect
overfit / regime change.

If top-10 overlap >= 7, ranking is stable.
If overlap < 5, ranking is regime-dependent — don't trust either.
"""
from __future__ import annotations
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SCRAPER = os.path.join(ROOT, "scraper")
sys.path.insert(0, SCRAPER)
sys.path.insert(0, HERE)

# Override POST_V132 cutoff to last 3 days BEFORE importing the sweep module
import _sweep_v136_realistic as srm  # noqa: E402

cutoff = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
srm.POST_V132 = cutoff
srm.CSV_OUT = os.path.join(SCRAPER, "_sweep_v136_stability.csv")

print(f"Stability window: last 3 days (>= {cutoff})")
srm.main()

# Compare top-10 against the 5-day sweep
print("\n" + "=" * 90)
print("STABILITY CHECK — top-10 overlap vs 5-day sweep")
print("=" * 90)
try:
    full = pd.read_csv(os.path.join(SCRAPER, "_sweep_v136_realistic.csv"))
    short = pd.read_csv(srm.CSV_OUT)
    full_top = (full[full["n"] >= 30]
                .sort_values("kelly_pct", ascending=False)
                .drop_duplicates(subset=["strategy"]).head(10)["strategy"].tolist())
    short_top = (short[short["n"] >= 10]  # lower threshold for short window
                 .sort_values("kelly_pct", ascending=False)
                 .drop_duplicates(subset=["strategy"]).head(10)["strategy"].tolist())
    overlap = set(full_top) & set(short_top)
    print(f"5-day top-10 : {full_top}")
    print(f"3-day top-10 : {short_top}")
    print(f"Overlap      : {len(overlap)}/10  ({sorted(overlap)})")
    if len(overlap) >= 7:
        print("STABLE — ranking holds across windows")
    elif len(overlap) >= 5:
        print("MODERATELY STABLE — top families consistent, micro-config drifts")
    else:
        print("UNSTABLE — regime-dependent, re-evaluate before deploying")
except Exception as e:
    print(f"Comparison failed: {e}")
