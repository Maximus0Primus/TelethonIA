"""Post-process mega_sweep_extended.csv with multi-test correction + family realism flag.

Adds 4 columns:
  p_value         — one-sample t-test vs zero (avg_pnl_pct = 0)
  p_corrected     — Bonferroni (× N_configs)
  fdr_q           — Benjamini-Hochberg q-value
  family_realism  — 1.0 = clean (TP/SL/BE/FAST/HIGHSCORE), 0.5 = HYST nu, 0.1 = trail/dtrail/dip/split

Outputs:
  _mega_sweep_extended_annotated.csv   — full table with new columns
  _mega_sweep_top_robust.csv           — top 30 with fdr_q < 0.05 AND family_realism >= 0.5

Usage:
  python scripts/analyze_mega_sweep.py [--csv _mega_sweep_extended.csv] [--alpha 0.05]
"""
import argparse, math, sys
from pathlib import Path
import pandas as pd

# Family realism scores — calibrated from Apr 20 audit
# DTRAIL10_ACT15_SL70: sim top vs live actual = 47x slip + 65% reconciler early-close
# DIP30/TRAIL/SPLIT: same root cause (multiple sells per trade)
# HYST nu: paired-test showed -$185 to -$489 vs base on N=38-69
# HYST + filter (S30/NZ): real signal, treat as clean
TRAIL_KEYS = ("DTRAIL", "DIP", "TRAIL", "SPLIT", "PTRAIL", "BOND_FAST", "TD2")
HYST_KEYS = ("_HYST",)
HYST_FILTERED_KEYS = ("S30_HYST", "NZS30_HYST", "_S30", "_S40", "_NZ", "_MCAP")

def family_realism(strategy: str, filter_name: str = "") -> float:
    """0.1 = artifact, 0.5 = suspicious, 1.0 = clean."""
    s = strategy.upper()
    f = (filter_name or "").upper()
    # Trail/dip/split family: multiple sells → 47x slip + reconciler 65% early
    if any(k in s for k in TRAIL_KEYS):
        return 0.1
    # HYST + quality filter survives — bagged with score/liq filter that catches whipsaw
    if any(k in s for k in HYST_FILTERED_KEYS) or f in ("SCORE30", "SCORE35", "SCORE40", "SCORE45", "SCORE50", "MCAP_MID", "NOZEROLIQ_SCORE30", "NOZEROLIQ_SCORE40", "MCAP_MID_SCORE40"):
        return 1.0 if "HYST" not in s else 0.8
    # HYST nu (no filter) = artifact per paired test
    if any(k in s for k in HYST_KEYS):
        return 0.5
    return 1.0


def t_test_pvalue(mean: float, n: int, std: float = None) -> float:
    """One-sample t-test pvalue against H0: mean=0. Two-tailed.
    If std unknown, approximate from typical memecoin pnl_pct std ~30%."""
    if not n or n < 2:
        return 1.0
    if std is None or std <= 0:
        std = 30.0  # typical memecoin pnl_pct std
    se = std / math.sqrt(n)
    if se == 0:
        return 1.0
    t = abs(mean) / se
    # Approximate normal CDF (close enough for n>30)
    z = t
    # Two-tailed
    p = math.erfc(z / math.sqrt(2))
    return min(1.0, max(0.0, p))


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    """BH FDR-controlled q-values."""
    n = len(pvals)
    if n == 0:
        return []
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    q = [0.0] * n
    prev = 1.0
    for rank_minus_1, (orig_idx, p) in enumerate(reversed(indexed)):
        rank = n - rank_minus_1
        q_i = min(prev, p * n / rank)
        prev = q_i
        q[orig_idx] = q_i
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="scraper/_mega_sweep_extended.csv",
                    help="Path to mega_sweep CSV (default: scraper/_mega_sweep_extended.csv)")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="Significance threshold for FDR + Bonferroni (default 0.05)")
    ap.add_argument("--top", type=int, default=30,
                    help="Number of robust top configs to extract (default 30)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = Path(__file__).resolve().parent.parent / csv_path
    if not csv_path.exists():
        print(f"ERROR: CSV not found at {csv_path}")
        sys.exit(1)

    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df):,} configs loaded")

    # Filter to N>=30 for stat tests
    df_eligible = df[df["n"] >= 30].copy()
    print(f"  {len(df_eligible):,} configs with N>=30 (stat-eligible)")

    # P-value per config (vs H0: mean=0)
    df_eligible["p_value"] = df_eligible.apply(
        lambda r: t_test_pvalue(r["avg_pnl_pct"], int(r["n"])), axis=1
    )

    # Bonferroni: p × N_configs
    n_total = len(df_eligible)
    df_eligible["p_corrected"] = (df_eligible["p_value"] * n_total).clip(upper=1.0)

    # BH FDR q-values
    df_eligible["fdr_q"] = benjamini_hochberg(df_eligible["p_value"].tolist())

    # Family realism flag
    df_eligible["family_realism"] = df_eligible.apply(
        lambda r: family_realism(str(r["strategy"]), str(r.get("filter", ""))),
        axis=1
    )

    # Save annotated CSV
    out_full = csv_path.parent / f"{csv_path.stem}_annotated.csv"
    df_eligible.to_csv(out_full, index=False)
    print(f"  -> {out_full} ({len(df_eligible):,} rows)")

    # Top robust: positive avg, fdr<alpha, family_realism>=0.5
    robust = df_eligible[
        (df_eligible["avg_pnl_pct"] > 0)
        & (df_eligible["fdr_q"] < args.alpha)
        & (df_eligible["family_realism"] >= 0.5)
    ].sort_values("avg_pnl_pct", ascending=False).head(args.top)
    out_top = csv_path.parent / f"{csv_path.stem.replace('extended','top_robust')}.csv"
    robust.to_csv(out_top, index=False)

    # Print summary
    print()
    print("=" * 100)
    print(f"SUMMARY (alpha={args.alpha})")
    print("=" * 100)
    n_signif_uncorrected = (df_eligible["p_value"] < args.alpha).sum()
    n_signif_bonferroni = (df_eligible["p_corrected"] < args.alpha).sum()
    n_signif_fdr = (df_eligible["fdr_q"] < args.alpha).sum()
    n_clean_family = (df_eligible["family_realism"] == 1.0).sum()
    n_artifact = (df_eligible["family_realism"] == 0.1).sum()

    print(f"Configs eligible (N>=30):       {n_total:>10,}")
    print(f"  significant uncorrected:      {n_signif_uncorrected:>10,}  ({100*n_signif_uncorrected/n_total:>5.1f}%)")
    print(f"  significant Bonferroni:       {n_signif_bonferroni:>10,}  ({100*n_signif_bonferroni/n_total:>5.1f}%)")
    print(f"  significant FDR (q<{args.alpha:.2f}):       {n_signif_fdr:>10,}  ({100*n_signif_fdr/n_total:>5.1f}%)")
    print(f"  family clean (realism=1.0):   {n_clean_family:>10,}")
    print(f"  family artifact (realism=0.1):{n_artifact:>10,}  (DTRAIL/TRAIL/DIP/SPLIT/BOND/TD2)")
    print()
    print(f"TOP {min(args.top, len(robust))} ROBUST CONFIGS (positive, FDR<{args.alpha}, family>=0.5):")
    print("-" * 100)
    if len(robust) == 0:
        print("  (none — try relaxing --alpha or check sweep data)")
    else:
        cols = ["strategy", "filter", "source", "smoothing", "polling_mode",
                "n", "wr_pct", "avg_pnl_pct", "fdr_q", "family_realism", "dollars_per_day"]
        cols = [c for c in cols if c in robust.columns]
        with pd.option_context("display.max_rows", None, "display.width", 200, "display.float_format", "{:,.4f}".format):
            print(robust[cols].to_string(index=False))
    print()
    print(f"Files written:")
    print(f"  - {out_full}")
    print(f"  - {out_top}")
    print()
    print("Interpretation:")
    print("  - p_corrected (Bonferroni): very strict; passing => robust against multiple testing")
    print("  - fdr_q (Benjamini-Hochberg): false discovery rate; q<0.05 = at most 5% expected to be noise")
    print("  - family_realism: 0.1=trail/dtrail/dip/split (sim over-estimates 5-50x in real),")
    print("                    0.5=HYST nu (paired test confirmed loss vs base),")
    print("                    0.8=HYST + filter (real signal),")
    print("                    1.0=TP/SL/BE/FAST (clean, no smoothing trickery)")


if __name__ == "__main__":
    main()
