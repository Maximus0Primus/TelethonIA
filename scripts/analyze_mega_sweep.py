"""Post-process mega_sweep_extended.csv with multi-test correction + family realism flag.

Adds columns:
  p_value         — one-sample t-test vs zero (avg_pnl_pct = 0)
  p_corrected     — Bonferroni (× N_configs)
  fdr_q           — Benjamini-Hochberg q-value
  family_realism  — 1.0 = clean (TP/SL/BE/FAST/HIGHSCORE), 0.5 = HYST nu, 0.1 = trail/dtrail/dip/split

v14e.26 — Feature 4 (bootstrap rank stability) + Feature 5 (cross-regime robust):
  bootstrap_rank_pct  — median rank percentile across N resamples of trading days
                        (1.0 = always top, 0.5 = median, 0.0 = always bottom).
                        Computed only if `daily_pnl_json` column present.
  rank_stability      — 1.0 - rank_std (normalized std of rank across resamples).
                        High (>0.7) = rank is stable across day permutations.
  cross_regime_robust — bool: pnl_active>0 AND pnl_quiet>0 AND wf_consistent

Outputs:
  _mega_sweep_extended_annotated.csv   — full table with new columns
  _mega_sweep_top_robust.csv           — top 30 robust (FDR<0.05, family>=0.5,
                                          cross_regime_robust if regime data present)

Usage:
  python scripts/analyze_mega_sweep.py [--csv _mega_sweep_extended.csv] [--alpha 0.05]
"""
import argparse, math, sys, json, os
from pathlib import Path
import pandas as pd
import numpy as np

# v14e.45: import the registered STRATEGIES dict to filter out phantom sweep
# combinations that don't have a runnable Python implementation. Without this,
# the sweep can rank a strat (e.g. BE15_TP300_SL50_4H) that will never produce
# paper trades because it's not in strategies.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scraper"))
try:
    from strategies import STRATEGIES as _REGISTERED_STRATS
    _REGISTERED_STRATS_SET = set(_REGISTERED_STRATS.keys())
    print(f"[strategies] {len(_REGISTERED_STRATS_SET)} registered strategies loaded for phantom filter")
except Exception as _e:
    print(f"[strategies] WARNING: could not import STRATEGIES ({_e}) — phantom filter DISABLED")
    _REGISTERED_STRATS_SET = None

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
    ap.add_argument("--require-fdr", action="store_true",
                    help="Apply FDR<alpha gate to top_robust (v14e.27 default OFF — "
                         "Bonferroni-corrected FDR on 371k tests was nuking the entire "
                         "top robust list to zero rows. Regime-aware gate alone is the "
                         "actual signal; FDR is a strict statistical bar that re-enables "
                         "with this flag when N is large enough).")
    ap.add_argument("--csv", default="scraper/_mega_sweep_extended.csv",
                    help="Path to mega_sweep CSV (default: scraper/_mega_sweep_extended.csv)")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="Significance threshold for FDR + Bonferroni (default 0.05)")
    ap.add_argument("--top", type=int, default=30,
                    help="Number of robust top configs to extract (default 30)")
    ap.add_argument("--persist", action="store_true",
                    help="v14e.34: also INSERT top configs into mega_sweep_runs table "
                         "(Supabase). Lets the calibration script later compare each "
                         "predicted $/day against realized paper P&L in the post-run window.")
    ap.add_argument("--persist-extra", type=int, default=50,
                    help="When --persist set, insert top-N robust rows AS top_robust=true "
                         "PLUS the next N rows by avg_pnl_pct as top_robust=false (default 50). "
                         "Total per run = top + persist-extra. Keeps DB small but tracks runners-up.")
    ap.add_argument("--chain", default="solana",
                    help="Chain tag written to mega_sweep_runs.chain when --persist is set "
                         "(default: solana). ETH workflow passes --chain ethereum.")
    # v14e.45: bi-ranking — second top-N optimisé pour détecter les strats fraîchement
    # déployées qui surperforment sur la fenêtre récente mais que `cross_regime_robust`
    # exclut (besoin ≥2 régimes ≈ ≥7-10 jours d'historique). Audit Apr 29 a montré
    # que 19/20 top shadow strats (4j) étaient absentes du top sim 14j.
    ap.add_argument("--recent-days", type=int, default=7,
                    help="v14e.45: window (days) for the recent ranking — uses daily_pnl_json "
                         "to compute avg_pnl on the last N days only. Default 7.")
    ap.add_argument("--top-recent", type=int, default=30,
                    help="v14e.45: number of recent-top configs to extract (default 30). "
                         "Uses relaxed gates (no cross_regime_robust, n_recent>=15).")
    ap.add_argument("--min-n-recent", type=int, default=15,
                    help="v14e.45: minimum trades in the recent window to be eligible (default 15).")
    ap.add_argument("--min-wr-recent", type=float, default=0.25,
                    help="v14e.45: minimum win rate (fraction) on the full sweep window for "
                         "top_recent eligibility — guards against pure fat-tail strats with "
                         "1-2 moonshots boosting recent_avg. Default 0.25 (25%%).")
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

    # v14e.26 — Feature 4: Bootstrap rank stability via resampling days.
    # Skipped silently if `daily_pnl_json` not in CSV (run pre-v14e.26).
    if "daily_pnl_json" in df_eligible.columns:
        print("\n[v14e.26 Feature 4] Bootstrap rank stability (300 resamples of days)...")
        # Parse daily_pnl_json once per row
        daily_dicts = df_eligible["daily_pnl_json"].apply(
            lambda s: json.loads(s) if isinstance(s, str) and s else {}
        ).tolist()
        # Universe of all observed days
        all_days = sorted({d for dd in daily_dicts for d in dd})
        n_days = len(all_days)
        if n_days >= 5 and len(daily_dicts) >= 10:
            n_resamples = 300
            rng = np.random.default_rng(42)
            n_configs = len(daily_dicts)
            # Pre-build matrix: rows = configs, cols = days, values = avg pnl_pct
            M = np.full((n_configs, n_days), np.nan)
            day_idx = {d: i for i, d in enumerate(all_days)}
            for ci, dd in enumerate(daily_dicts):
                for d, p in dd.items():
                    j = day_idx.get(d)
                    if j is not None:
                        M[ci, j] = float(p)
            # rank_pct accumulator
            rank_sum = np.zeros(n_configs)
            rank_sq = np.zeros(n_configs)
            for _ in range(n_resamples):
                sample_cols = rng.integers(0, n_days, n_days)  # bootstrap days w/ replacement
                sub = M[:, sample_cols]
                # Mean per config (ignoring NaN)
                with np.errstate(invalid="ignore"):
                    means = np.nanmean(sub, axis=1)
                # Rank: higher mean = better. argsort gives ascending; pct = rank / N
                order = np.argsort(np.nan_to_num(means, nan=-1e9))
                ranks = np.empty(n_configs)
                ranks[order] = np.arange(n_configs)
                pct = ranks / max(1, n_configs - 1)
                rank_sum += pct
                rank_sq += pct ** 2
            rank_mean = rank_sum / n_resamples
            rank_var = rank_sq / n_resamples - rank_mean ** 2
            rank_std = np.sqrt(np.clip(rank_var, 0, None))
            df_eligible["bootstrap_rank_pct"] = np.round(rank_mean, 4)
            df_eligible["rank_stability"] = np.round(1.0 - 2 * rank_std, 4)  # 1.0 = perfectly stable
            print(f"  computed for {n_configs:,} configs across {n_days} days")
        else:
            print(f"  skipped: only {n_days} days, {len(daily_dicts)} configs (need >=5 days, >=10 configs)")

    # v14e.45: Recent-window PnL — extracts avg_pnl on the last N days from
    # daily_pnl_json so newly-deployed strats (LOCK / NZ_S40 / MCAP_S40 deployed
    # 04-26 with 3-4d of data) have a fair shot at the bi-ranking. The 14d
    # robust ranking inherently penalizes them via cross_regime_robust gate.
    if "daily_pnl_json" in df_eligible.columns:
        from datetime import datetime, timedelta, timezone
        cutoff = (datetime.now(timezone.utc) - timedelta(days=args.recent_days)).date()
        cutoff_str = cutoff.isoformat()

        def _recent_stats(s):
            try:
                dd = json.loads(s) if isinstance(s, str) and s else {}
            except Exception:
                return (0, None)
            if not dd:
                return (0, None)
            recent_pnls = [float(p) for d, p in dd.items() if d >= cutoff_str]
            if not recent_pnls:
                return (0, None)
            return (len(recent_pnls), float(np.mean(recent_pnls)))

        recent_results = df_eligible["daily_pnl_json"].apply(_recent_stats)
        df_eligible["recent_n_days"] = recent_results.apply(lambda t: t[0])
        df_eligible["recent_avg_pnl_pct"] = recent_results.apply(lambda t: t[1])
        n_recent_eligible = (df_eligible["recent_n_days"] > 0).sum()
        print(f"[v14e.45 recent ranking] window={args.recent_days}d (since {cutoff_str}) — "
              f"{n_recent_eligible:,} configs have data in window")

    # v14e.26 — Feature 5: Cross-regime robust flag
    if all(c in df_eligible.columns for c in ("pnl_active_pct", "pnl_quiet_pct", "wf_consistent")):
        df_eligible["cross_regime_robust"] = (
            (df_eligible["pnl_active_pct"].fillna(-99) > 0)
            & (df_eligible["pnl_quiet_pct"].fillna(-99) > 0)
            & (df_eligible["wf_consistent"].fillna(False))
        )
        n_cr = int(df_eligible["cross_regime_robust"].sum())
        print(f"[v14e.26 Feature 5] cross_regime_robust = True: {n_cr:,} configs")

    # Save annotated CSV
    out_full = csv_path.parent / f"{csv_path.stem}_annotated.csv"
    df_eligible.to_csv(out_full, index=False)
    print(f"  -> {out_full} ({len(df_eligible):,} rows)")

    # Top robust: positive avg, family_realism>=0.5, cross_regime_robust if
    # available (v14e.26). FDR<alpha is OPT-IN via --require-fdr (v14e.27):
    # the Bonferroni-corrected FDR on 371k tests nuked the entire top_robust
    # list to zero on the Apr 26 run, so the regime-aware filter is now the
    # primary gate. Re-enable FDR once N grows large enough that q-values
    # actually discriminate.
    base_filter = (
        (df_eligible["avg_pnl_pct"] > 0)
        & (df_eligible["family_realism"] >= 0.5)
    )
    if args.require_fdr:
        base_filter = base_filter & (df_eligible["fdr_q"] < args.alpha)
        print(f"  applying FDR<{args.alpha} gate (--require-fdr)")
    if "cross_regime_robust" in df_eligible.columns:
        base_filter = base_filter & df_eligible["cross_regime_robust"]
        print(f"  applying cross_regime_robust filter to top robust selection")
    # v14e.45: phantom filter — exclure strats non registered dans strategies.py.
    # Le grid sweep peut ranker des combos qui n'existent pas en code (ex: run
    # 25116811803 a ranké BE15_TP300_SL50_4H qui n'est pas dans STRATEGIES,
    # donc 0 paper trade en réalité — ranking sim trompeur).
    if _REGISTERED_STRATS_SET is not None and "strategy" in df_eligible.columns:
        n_before = base_filter.sum()
        is_registered = df_eligible["strategy"].isin(_REGISTERED_STRATS_SET)
        n_phantom = (base_filter & ~is_registered).sum()
        base_filter = base_filter & is_registered
        if n_phantom > 0:
            phantom_strats = df_eligible[base_filter | (~is_registered & (df_eligible["avg_pnl_pct"] > 0))]
            phantom_names = sorted(phantom_strats[~phantom_strats["strategy"].isin(_REGISTERED_STRATS_SET)]["strategy"].unique())
            print(f"  phantom filter: {n_phantom} rows excluded ({len(phantom_names)} non-registered strats)")
            for ps in phantom_names[:10]:
                print(f"    - {ps}")
    # v14e.45: dédup par (strategy, filter) avant head(top). Sans ça, le top-30
    # est saturé par les variants source/smoothing/polling de la même strat de
    # base (run 25116811803: 30 rows = 4 strats × 7-8 variants chacune).
    robust = (
        df_eligible[base_filter]
        .sort_values("avg_pnl_pct", ascending=False)
        .drop_duplicates(subset=["strategy", "filter"], keep="first")
        .head(args.top)
    )
    out_top = csv_path.parent / f"{csv_path.stem.replace('extended','top_robust')}.csv"
    robust.to_csv(out_top, index=False)

    # v14e.45: bi-ranking — second top-N optimisé pour les déploiements récents.
    # Gates relâchés: pas de cross_regime_robust, n_recent>=min_n_recent (15 par
    # défaut), recent_avg_pnl_pct>0. Laisse remonter les LOCK/NZ_S40/MCAP_S40 que
    # le ranking 14d élimine à cause du gate cross_regime (besoin ≥7-10j).
    recent = pd.DataFrame()
    if "recent_avg_pnl_pct" in df_eligible.columns:
        recent_filter = (
            (df_eligible["recent_avg_pnl_pct"].fillna(-99) > 0)
            & (df_eligible["recent_n_days"].fillna(0) >= 1)  # any data in window
            & (df_eligible["family_realism"] >= 0.5)
        )
        # Phantom filter aussi sur recent (cohérence avec robust)
        if _REGISTERED_STRATS_SET is not None and "strategy" in df_eligible.columns:
            recent_filter = recent_filter & df_eligible["strategy"].isin(_REGISTERED_STRATS_SET)
        # Eligibilité par n_sim_trades >= min_n_recent — abaisse la barre vs robust
        if "n" in df_eligible.columns:
            recent_filter = recent_filter & (df_eligible["n"] >= args.min_n_recent)
        # v14e.45: WR gate sur la fenêtre full sweep — garde-fou anti fat-tail.
        # Une strat qui ranke +30% recent_avg avec WR 5% (= 1-2 moonshots) est
        # un mirage qui va revenir à la moyenne. Le gate WR>=25% sur la fenêtre
        # complete sweep (14d) écarte ces cas avant promotion.
        if "wr_pct" in df_eligible.columns:
            recent_filter = recent_filter & (df_eligible["wr_pct"] >= args.min_wr_recent * 100)
            print(f"[v14e.45 recent ranking] WR gate >= {args.min_wr_recent*100:.0f}% applied")
        recent = (
            df_eligible[recent_filter]
            .sort_values("recent_avg_pnl_pct", ascending=False)
            .drop_duplicates(subset=["strategy", "filter"], keep="first")
            .head(args.top_recent)
        )
        out_recent = csv_path.parent / f"{csv_path.stem.replace('extended','top_recent')}.csv"
        recent.to_csv(out_recent, index=False)
        print(f"\n[v14e.45 bi-ranking] top_recent ({args.recent_days}d, n>={args.min_n_recent}, "
              f"no cross_regime gate): {len(recent)} configs -> {out_recent.name}")

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
    _gate_desc = f"FDR<{args.alpha}, " if args.require_fdr else ""
    print(f"TOP {min(args.top, len(robust))} ROBUST CONFIGS (positive, {_gate_desc}family>=0.5):")
    print("-" * 100)
    if len(robust) == 0:
        print("  (none — try relaxing --alpha or check sweep data)")
    else:
        cols = ["strategy", "filter", "source", "smoothing", "polling_mode",
                "n", "wr_pct", "avg_pnl_pct", "fdr_q", "family_realism", "dollars_per_day",
                # v14e.26 columns (shown if present)
                "pnl_active_pct", "pnl_quiet_pct", "pnl_dead_pct",
                "wf_train_pnl_pct", "wf_test_pnl_pct", "wf_consistent",
                "rank_stability", "cross_regime_robust"]
        cols = [c for c in cols if c in robust.columns]
        with pd.option_context("display.max_rows", None, "display.width", 240, "display.float_format", "{:,.4f}".format):
            print(robust[cols].to_string(index=False))
    print()
    # v14e.34: persist top-N to Supabase mega_sweep_runs for sim/actual calibration.
    # Writes one row per (strategy, filter) at the current run timestamp. Calibration
    # script later joins with paper_trades over the post-run window to compute drift.
    if args.persist:
        try:
            import os
            from datetime import datetime, timezone
            from supabase import create_client
            url = os.environ.get("SUPABASE_URL")
            key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
            if not (url and key):
                print("[persist] SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY missing, skipping DB write")
            else:
                sb = create_client(url, key)
                run_at = datetime.now(timezone.utc).isoformat()
                run_id = os.environ.get("GITHUB_RUN_ID") or os.environ.get("RUN_ID")
                run_source = "gh-actions" if os.environ.get("GITHUB_ACTIONS") else "local"

                def _row_to_record(rr, rank, is_top_robust, is_top_recent=False):
                    g = lambda k: rr[k] if k in rr.index and pd.notna(rr[k]) else None
                    return {
                        "run_at": run_at,
                        "run_source": run_source,
                        "run_id": run_id,
                        "chain": args.chain,
                        "strategy": str(g("strategy") or ""),
                        "filter_name": str(g("filter") or "") or None,
                        "rank_at_run": int(rank),
                        "n_sim_trades": int(g("n")) if g("n") is not None else None,
                        "avg_pnl_pct": float(g("avg_pnl_pct")) if g("avg_pnl_pct") is not None else None,
                        "median_pnl_pct": float(g("median_pnl_pct")) if g("median_pnl_pct") is not None else None,
                        "win_rate": float(g("wr_pct")) / 100.0 if g("wr_pct") is not None else None,
                        "pnl_per_day": float(g("dollars_per_day")) if g("dollars_per_day") is not None else None,
                        "family_realism": float(g("family_realism")) if g("family_realism") is not None else None,
                        "fdr_q": float(g("fdr_q")) if g("fdr_q") is not None else None,
                        "p_corrected": float(g("p_corrected")) if g("p_corrected") is not None else None,
                        "bootstrap_rank_pct": float(g("bootstrap_rank_pct")) if g("bootstrap_rank_pct") is not None else None,
                        "rank_stability": float(g("rank_stability")) if g("rank_stability") is not None else None,
                        "cross_regime_robust": bool(g("cross_regime_robust")) if g("cross_regime_robust") is not None else None,
                        "is_top_robust": bool(is_top_robust),
                        # v14e.45: bi-ranking flags + window data
                        "is_top_recent": bool(is_top_recent),
                        "recent_n_trades": int(g("recent_n_days")) if g("recent_n_days") is not None else None,
                        "recent_avg_pnl_pct": float(g("recent_avg_pnl_pct")) if g("recent_avg_pnl_pct") is not None else None,
                        "recent_window_days": int(args.recent_days),
                        "metadata": {
                            "smoothing": str(g("smoothing") or "") or None,
                            "polling_mode": str(g("polling_mode") or "") or None,
                            "source": str(g("source") or "") or None,
                            "wf_consistent": bool(g("wf_consistent")) if g("wf_consistent") is not None else None,
                            "dollars_per_day_active": float(g("pnl_active_pct")) if g("pnl_active_pct") is not None else None,
                            "dollars_per_day_quiet": float(g("pnl_quiet_pct")) if g("pnl_quiet_pct") is not None else None,
                        },
                    }

                # v14e.45: build set of (strategy, filter) keys for recent so we
                # can flag rows that are in BOTH selections (one row inserted, both
                # booleans true) instead of inserting duplicates.
                def _key(rr):
                    return (str(rr.get("strategy") or ""), str(rr.get("filter") or "") or None)
                recent_keys = set()
                if len(recent) > 0:
                    recent_keys = {_key(rr) for _, rr in recent.iterrows()}

                records = []
                seen_keys = set()
                # top robust rows (is_top_robust=true, is_top_recent if also in recent)
                for i, (_, rr) in enumerate(robust.iterrows(), start=1):
                    k = _key(rr)
                    seen_keys.add(k)
                    records.append(_row_to_record(rr, i, True, is_top_recent=(k in recent_keys)))
                # v14e.45: top recent rows that aren't already in robust
                if len(recent) > 0:
                    rank_offset = len(records)
                    for _, rr in recent.iterrows():
                        k = _key(rr)
                        if k in seen_keys:
                            continue
                        seen_keys.add(k)
                        rank_offset += 1
                        records.append(_row_to_record(rr, rank_offset, False, is_top_recent=True))
                # runners-up (next N by avg_pnl_pct, family>=0.5, not in robust ∪ recent)
                if args.persist_extra > 0:
                    runners = (
                        df_eligible[
                            (df_eligible["family_realism"] >= 0.5)
                            & (~df_eligible.index.isin(robust.index))
                            & (~df_eligible.index.isin(recent.index if len(recent) > 0 else []))
                        ]
                        .sort_values("avg_pnl_pct", ascending=False)
                        .drop_duplicates(subset=["strategy", "filter"], keep="first")
                        .head(args.persist_extra)
                    )
                    for _, rr in runners.iterrows():
                        k = _key(rr)
                        if k in seen_keys:
                            continue
                        seen_keys.add(k)
                        records.append(_row_to_record(rr, len(records) + 1, False, is_top_recent=False))

                if records:
                    # Batch insert (chunks of 100 to be polite)
                    for k in range(0, len(records), 100):
                        sb.table("mega_sweep_runs").insert(records[k:k + 100]).execute()
                    print(f"[persist] inserted {len(records)} rows into mega_sweep_runs (run_at={run_at}, run_id={run_id})")
        except Exception as e:
            print(f"[persist] DB write FAILED: {e}")

    print(f"Files written:")
    print(f"  - {out_full}")
    print(f"  - {out_top}")
    if "recent_avg_pnl_pct" in df_eligible.columns and len(recent) > 0:
        out_recent_path = csv_path.parent / f"{csv_path.stem.replace('extended', 'top_recent')}.csv"
        print(f"  - {out_recent_path}")
    print()
    print("Interpretation:")
    print("  - p_corrected (Bonferroni): very strict; passing => robust against multiple testing")
    print("  - fdr_q (Benjamini-Hochberg): false discovery rate; q<0.05 = at most 5% expected to be noise")
    print("  - family_realism: 0.1=trail/dtrail/dip/split (sim over-estimates 5-50x in real),")
    print("                    0.5=HYST nu (paired test confirmed loss vs base),")
    print("                    0.8=HYST + filter (real signal),")
    print("                    1.0=TP/SL/BE/FAST (clean, no smoothing trickery)")
    print("  - pnl_{active,quiet,dead}_pct: per-regime avg PnL. Active days = pump_rate>=30%,")
    print("                    quiet = 15-30%, dead = <15% (fraction of tokens hitting peak >= +50%)")
    print("  - wf_{train,test}_pnl_pct: walk-forward split. Train = all days except last 3.")
    print("                    Test = last 3 days. wf_consistent = same sign + within 60% magnitude.")
    print("  - rank_stability: bootstrap stability of rank across day resampling. >0.7 = stable.")
    print("  - cross_regime_robust: pnl_active>0 AND pnl_quiet>0 AND wf_consistent.")
    print("                    Top robust list NOW REQUIRES this flag (v14e.26).")


if __name__ == "__main__":
    main()
