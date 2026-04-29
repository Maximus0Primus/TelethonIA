"""v14e.43b — reverse-engineer the scoring system, optimizing $/day not WR.

==== BIG REWRITE — fixes the v14e.43a mistake ====
v14e.43a optimized `WR × sqrt(N)` per-feature/per-combo. That metric
maximized winrate × volume, not profit. Cross-checked the BSR>=0.52
"finding" against actual $/day at $20/trade — it LOST $2-4/d on 7/7 top
SOL strats (high-WR but kills fat-tail moonshots). Lesson: a filter that
improves WR can still hurt $$ if it removes positive-EV high-variance
trades.

This rewrite changes the target metric to **sum_$ / day** at simulated
$20/trade, which is what we actually care about for live deployment.
Adds Optuna search for global score formula (vs the static quantile grid
scan), and a walk-forward CV that flags overfit candidates.

Pipeline:
  1. Pull paper_trades 30d closed clean (no artefact, post-blacklist)
  2. Per-feature: scan thresholds on $/d-at-position-size — keep only
     filters that BOTH improve $/d on train AND on test set.
  3. 2-feature AND combos on top strats — same $/d target.
  4. Optuna global search: linear weighted score = Σ w_i × norm(feat_i),
     find threshold T maximizing $/d post-filter. Walk-forward CV.
  5. Output: actionable filters with $/d delta (vs base) — NOT WR lift.

Run:
    python scripts/_score_reverse_engineer.py --days 30 --chain solana \
        --min-n 80 --pos-usd 20 [--combos] [--optuna 200]

Dependencies: scikit-learn, xgboost, pandas, numpy, optuna.
"""
import os, sys, json, argparse, statistics as st, re
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
CLOSED = ("sl_hit", "trail_stop", "tp_hit", "timeout", "be_stop")
ARTEFACT_RE = re.compile(r"^(DTRAIL|PTRAIL|SPLIT|TRAIL|TD2|DIP|MOONBAG|WIDE_RUNNER|SCALE_OUT|MCAP_DTRAIL|BOND)")
SUSPECT_SUFFIX = re.compile(r"_(HYST|LAZYXSLOW|LAZYSLOW|LAZYMED|LAZY|COMBO|BOTH|JUPITER)$")
SOL_BL = {"mad_apes_gambles", "papicall", "markdegens", "MaybachGambleCalls",
          "ramcalls", "leoclub69", "CarnagecallsGambles", "explorer_gems",
          "ChairmanDN1", "chiggajogambles", "bounty_journal", "DegenSeals",
          "aliensalphacalls", "LevisAlpha"}


def is_artefact(s):
    return bool(ARTEFACT_RE.search(s) or SUSPECT_SUFFIX.search(s))


def fetch_all(tbl, sel, **f):
    out, step, off = [], 1000, 0
    while True:
        q = sb.table(tbl).select(sel)
        for k, v in f.items():
            if k.startswith("gte_"): q = q.gte(k[4:], v)
            elif k.startswith("eq_"): q = q.eq(k[3:], v)
            elif k.startswith("neq_"): q = q.neq(k[4:], v)
        r = q.range(off, off+step-1).execute()
        if not r.data: break
        out.extend(r.data)
        if len(r.data) < step: break
        off += step
    return out


def dollar_per_day(df, mask, pos_usd, days):
    """Compute sum_$ per day on the filtered subset."""
    if mask is None:
        sub = df
    else:
        sub = df[mask]
    if sub.empty: return 0.0, 0
    sum_pct = float(sub["pnl_pct"].sum())  # already in % terms
    return sum_pct / 100.0 * pos_usd / days, len(sub)


def find_optimal_threshold_dollar(df, feature, pos_usd, days, target="dollar"):
    """For a continuous feature, find the >= threshold that maximizes
    actual $/day on the kept subset. Returns delta_dollar_per_day vs
    the baseline (no filter), plus N_kept and WR for transparency.
    """
    if df[feature].isna().all() or df[feature].nunique() < 5:
        return None
    base_dollar_d, base_n = dollar_per_day(df, None, pos_usd, days)
    best = {"delta_dollar_d": -np.inf}
    quantiles = np.linspace(0.05, 0.95, 19)
    for q in quantiles:
        thr = df[feature].quantile(q)
        mask = df[feature] >= thr
        n_above = int(mask.sum())
        n_below = len(df) - n_above
        if n_above < 10 or n_below < 10:
            continue
        d_kept, n_kept = dollar_per_day(df, mask, pos_usd, days)
        delta = d_kept - base_dollar_d
        wr = float(df.loc[mask, "won"].mean() * 100)
        if delta > best["delta_dollar_d"]:
            best = {
                "feature": feature,
                "threshold": float(thr),
                "delta_dollar_d": float(delta),
                "kept_dollar_d": float(d_kept),
                "base_dollar_d": float(base_dollar_d),
                "n_kept": n_kept,
                "n_excluded": int(n_below),
                "wr_kept": wr,
                "kept_pct": round(n_kept / len(df) * 100, 1),
            }
    return best if best.get("feature") else None


def find_optimal_combo_dollar(df, features, pos_usd, days, min_n=10):
    """2-feature AND combo scan on $/d delta. Coarse 5x5 quantile grid."""
    base_dollar_d, _ = dollar_per_day(df, None, pos_usd, days)
    best = {"delta_dollar_d": -np.inf}
    quantiles = [0.10, 0.30, 0.50, 0.70, 0.90]
    for i, f1 in enumerate(features):
        if df[f1].isna().all() or df[f1].nunique() < 5: continue
        for f2 in features[i+1:]:
            if df[f2].isna().all() or df[f2].nunique() < 5: continue
            for q1 in quantiles:
                thr1 = df[f1].quantile(q1)
                for q2 in quantiles:
                    thr2 = df[f2].quantile(q2)
                    mask = (df[f1] >= thr1) & (df[f2] >= thr2)
                    n_in = int(mask.sum())
                    if n_in < min_n or (len(df) - n_in) < min_n: continue
                    d_kept, _ = dollar_per_day(df, mask, pos_usd, days)
                    delta = d_kept - base_dollar_d
                    if delta > best["delta_dollar_d"]:
                        wr = float(df.loc[mask, "won"].mean() * 100)
                        best = {
                            "f1": f1, "thr1": float(thr1),
                            "f2": f2, "thr2": float(thr2),
                            "delta_dollar_d": float(delta),
                            "kept_dollar_d": float(d_kept),
                            "base_dollar_d": float(base_dollar_d),
                            "n_kept": n_in, "n_excluded": len(df) - n_in,
                            "wr_kept": wr,
                            "kept_pct": round(n_in / len(df) * 100, 1),
                        }
    return best if best.get("f1") else None


def optuna_score_formula(train_df, test_df, features, pos_usd, days_train,
                          days_test, n_trials=200, seed=42):
    """Optuna search for a linear-weighted score formula that maximizes
    $/d on train, validated on test. Score = sum(w_i × zscore(feat_i)).
    Filter: score >= percentile_threshold (search 50-95%).
    Returns (best_weights, train_delta_d, test_delta_d).
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Pre-compute z-scores on train so weights are scale-invariant
    z_train, z_test = {}, {}
    for f in features:
        m = train_df[f].mean()
        sd = train_df[f].std() or 1.0
        z_train[f] = (train_df[f].fillna(m) - m) / sd
        z_test[f] = (test_df[f].fillna(m) - m) / sd

    base_train, _ = dollar_per_day(train_df, None, pos_usd, days_train)

    def objective(trial):
        w = {f: trial.suggest_float(f"w_{f}", -1.0, 1.0) for f in features}
        thr_q = trial.suggest_float("thr_q", 0.20, 0.90)
        score_train = sum(w[f] * z_train[f] for f in features)
        thr = score_train.quantile(thr_q)
        mask = score_train >= thr
        n_in = int(mask.sum())
        if n_in < 50 or (len(train_df) - n_in) < 50:
            return -1e6
        d_kept, _ = dollar_per_day(train_df, mask, pos_usd, days_train)
        return d_kept - base_train

    study = optuna.create_study(direction="maximize",
                                  sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    bp = study.best_params
    w_best = {f: bp[f"w_{f}"] for f in features}
    thr_q = bp["thr_q"]

    # Apply on test set
    score_test = sum(w_best[f] * z_test[f] for f in features)
    score_train = sum(w_best[f] * z_train[f] for f in features)
    thr_train_val = score_train.quantile(thr_q)
    mask_test = score_test >= thr_train_val
    test_kept_d, n_test_kept = dollar_per_day(test_df, mask_test, pos_usd, days_test)
    test_base_d, _ = dollar_per_day(test_df, None, pos_usd, days_test)
    test_delta = test_kept_d - test_base_d

    return w_best, thr_q, study.best_value, test_delta, n_test_kept, len(test_df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--chain", default="solana")
    ap.add_argument("--min-n", type=int, default=80)
    ap.add_argument("--pos-usd", type=float, default=20.0,
                    help="position size USD for $/d simulation (default $20)")
    ap.add_argument("--combos", action="store_true")
    ap.add_argument("--optuna", type=int, default=0,
                    help="run Optuna global search with N trials")
    ap.add_argument("--top-strats", type=int, default=15)
    args = ap.parse_args()

    since = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()
    print(f"=== Score reverse-engineer v2 — chain={args.chain} window={args.days}d "
          f"pos=${args.pos_usd:.0f} TARGET=$/day ===\n")

    cols = ("token_address,strategy,status,pnl_pct,position_usd,entry_price,exit_price,"
            "rt_score,rt_liquidity_usd,rt_volume_24h,rt_buy_sell_ratio,"
            "rt_token_age_hours,rt_is_pump_fun,kol_score,kol_win_rate,kol_tier,"
            "kol_group,n_kol_confirmations,entry_score,entry_mcap,"
            "created_at,is_shadow")
    rows = fetch_all("paper_trades", cols, eq_chain=args.chain,
                     gte_created_at=since, neq_source="rt_live")
    closed = [r for r in rows
              if r.get("status") in CLOSED
              and not is_artefact(r.get("strategy", ""))
              and (args.chain != "solana" or r.get("kol_group") not in SOL_BL)]
    print(f"  {len(closed)} closed clean rows (post-blacklist={args.chain=='solana'})")

    df = pd.DataFrame(closed)
    df["pnl_pct"] = df["pnl_pct"].astype(float) * 100
    df["won"] = (df["pnl_pct"] > 0).astype(int)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df = df.sort_values("created_at").reset_index(drop=True)

    tier_map = {"S": 2, "A": 1, "B": 0}
    df["kol_tier_num"] = df["kol_tier"].map(lambda x: tier_map.get(x, 0))

    feats = ["rt_score", "rt_liquidity_usd", "rt_volume_24h", "rt_buy_sell_ratio",
             "rt_token_age_hours", "rt_is_pump_fun", "kol_score", "kol_win_rate",
             "kol_tier_num", "n_kol_confirmations", "entry_score", "entry_mcap"]
    for c in feats:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["rt_liquidity_usd", "rt_volume_24h", "entry_mcap"]:
        df[c + "_log"] = np.log1p(df[c].fillna(0))
    feats_used = feats + ["rt_liquidity_usd_log", "rt_volume_24h_log", "entry_mcap_log"]
    feats_used = [f for f in feats_used if df[f].notna().any()]

    base_dollar_d, base_n = dollar_per_day(df, None, args.pos_usd, args.days)
    base_wr = df["won"].mean() * 100
    print(f"  baseline: N={base_n}  WR={base_wr:.1f}%  $/d=${base_dollar_d:+.2f} (at ${args.pos_usd:.0f}/trade)\n")

    # === Per-feature single-threshold scan optimizing $/d ===
    print(f"=== GLOBAL feature threshold scan — TARGET $/d ===")
    print(f"  {'feature':<28} {'thresh':>10} {'WR_kept':>7} {'kept%':>6} {'kept_$/d':>9} {'delta_$/d':>9}")
    global_results = []
    for f in feats_used:
        r = find_optimal_threshold_dollar(df, f, args.pos_usd, args.days)
        if r:
            global_results.append(r)
            sign = "+" if r["delta_dollar_d"] > 0 else ""
            print(f"  {f:<28} {r['threshold']:>9.3g} {r['wr_kept']:>5.1f}% "
                  f"{r['kept_pct']:>5.1f}% ${r['kept_dollar_d']:>+7.2f} ${sign}{r['delta_dollar_d']:>+7.2f}")
    global_results.sort(key=lambda x: -x["delta_dollar_d"])
    if global_results:
        winners = [r for r in global_results if r["delta_dollar_d"] > 0]
        print(f"\n  positive-delta features: {len(winners)}/{len(global_results)}")

    # === Walk-forward CV ===
    print(f"\n=== WALK-FORWARD CV — train first 70%, test last 30% ===")
    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    days_train = max(1, (train_df["created_at"].max() - train_df["created_at"].min()).total_seconds() / 86400)
    days_test = max(1, (test_df["created_at"].max() - test_df["created_at"].min()).total_seconds() / 86400)
    base_train_d, _ = dollar_per_day(train_df, None, args.pos_usd, days_train)
    base_test_d, _ = dollar_per_day(test_df, None, args.pos_usd, days_test)
    print(f"  train: N={len(train_df)} days={days_train:.1f} base $/d=${base_train_d:+.2f}")
    print(f"  test:  N={len(test_df)} days={days_test:.1f} base $/d=${base_test_d:+.2f}")

    # Best feature on train, applied to test
    print(f"\n  TOP 10 single features (train→test, in-sample vs out-of-sample $/d):")
    print(f"  {'feature':<28} {'train_d_$/d':>12} {'test_d_$/d':>12} {'overfit?':>9}")
    train_results = []
    for f in feats_used:
        r = find_optimal_threshold_dollar(train_df, f, args.pos_usd, days_train)
        if not r: continue
        # Apply same threshold on test
        thr = r["threshold"]
        mask_test = test_df[f] >= thr
        d_test, _ = dollar_per_day(test_df, mask_test, args.pos_usd, days_test)
        delta_test = d_test - base_test_d
        train_results.append({"feature": f, "thr": thr,
                              "train_delta": r["delta_dollar_d"],
                              "test_delta": delta_test})
    train_results.sort(key=lambda x: -x["train_delta"])
    for r in train_results[:10]:
        overfit = "OVERFIT" if (r["train_delta"] > 0 and r["test_delta"] < -0.5) else (
            "OK" if (r["train_delta"] > 0 and r["test_delta"] > 0) else "neutral"
        )
        print(f"  {r['feature']:<28} ${r['train_delta']:>+10.2f} ${r['test_delta']:>+10.2f} {overfit:>9}")

    # === Per-strategy threshold scan ===
    print(f"\n=== PER-STRATEGY threshold scan — TARGET $/d (top strats N>={args.min_n}) ===")
    strat_sizes = df.groupby("strategy").size().sort_values(ascending=False)
    eligible = strat_sizes[strat_sizes >= args.min_n].index.tolist()
    print(f"  eligible strats: {len(eligible)}")
    out_rows = []
    for strat in eligible[:25]:
        sub = df[df["strategy"] == strat].copy()
        base_sub_d = float(sub["pnl_pct"].sum()) / 100 * args.pos_usd / args.days
        best = {"delta_dollar_d": -np.inf}
        for f in feats_used:
            r = find_optimal_threshold_dollar(sub, f, args.pos_usd, args.days)
            if r and r["delta_dollar_d"] > best["delta_dollar_d"]:
                best = dict(r)
        out_rows.append({
            "strategy": strat,
            "N_total": len(sub),
            "base_dollar_d": round(base_sub_d, 2),
            "best_feature": best.get("feature"),
            "threshold": best.get("threshold"),
            "kept_dollar_d": round(best.get("kept_dollar_d", 0), 2),
            "delta_dollar_d": round(best.get("delta_dollar_d", -np.inf), 2),
            "n_kept": best.get("n_kept"),
            "wr_kept": round(best.get("wr_kept", 0), 1),
        })

    out_df = pd.DataFrame(out_rows).sort_values("delta_dollar_d", ascending=False)
    print(f"  TOP 15 strategies by $/d delta:")
    print(f"  {'strategy':<32} {'N':>4} {'base_$/d':>9} {'feat':<20} {'thr':>10} {'kept_$/d':>9} {'delta_$/d':>9}")
    for _, x in out_df.head(15).iterrows():
        feat = (x["best_feature"] or "")[:18]
        thr = f"{x['threshold']:.2g}" if x["threshold"] is not None else "-"
        print(f"  {x['strategy']:<32} {x['N_total']:>4} ${x['base_dollar_d']:>+7.2f} "
              f"{feat:<20} {thr:>10} ${x['kept_dollar_d']:>+7.2f} ${x['delta_dollar_d']:>+7.2f}")

    # Save CSV
    out_path = os.path.join(os.path.dirname(__file__), "..", "data",
                              f"score_re_v2_dollar_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"  saved -> {out_path}")

    # === 2-feature combos optimizing $/d ===
    if args.combos:
        print(f"\n=== 2-FEATURE AND combos — TARGET $/d on top {args.top_strats} strats ===")
        combo_rows = []
        for strat in out_df.head(args.top_strats)["strategy"]:
            sub = df[df["strategy"] == strat]
            if len(sub) < 30: continue
            local_feats = [f for f in feats_used if sub[f].notna().mean() >= 0.5]
            if len(local_feats) < 2: continue
            base_sub_d = float(sub["pnl_pct"].sum()) / 100 * args.pos_usd / args.days
            r = find_optimal_combo_dollar(sub, local_feats, args.pos_usd, args.days)
            if r:
                combo_rows.append({
                    "strategy": strat,
                    "N_total": len(sub),
                    "base_dollar_d": round(base_sub_d, 2),
                    "f1": r["f1"], "thr1": round(r["thr1"], 4),
                    "f2": r["f2"], "thr2": round(r["thr2"], 4),
                    "kept_dollar_d": round(r["kept_dollar_d"], 2),
                    "delta_dollar_d": round(r["delta_dollar_d"], 2),
                    "n_kept": r["n_kept"], "wr_kept": round(r["wr_kept"], 1),
                })
        cdf = pd.DataFrame(combo_rows).sort_values("delta_dollar_d", ascending=False)
        print(f"  {'strategy':<32} {'N':>3} {'base_$/d':>8} {'f1':<20} {'thr1':>9} {'f2':<20} {'thr2':>9} {'delta_$/d':>9}")
        for _, x in cdf.iterrows():
            f1, f2 = x["f1"][:18], x["f2"][:18]
            t1, t2 = f"{x['thr1']:.2g}", f"{x['thr2']:.2g}"
            print(f"  {x['strategy']:<32} {x['N_total']:>3} ${x['base_dollar_d']:>+6.2f} "
                  f"{f1:<20} {t1:>9} {f2:<20} {t2:>9} ${x['delta_dollar_d']:>+7.2f}")
        cp = os.path.join(os.path.dirname(__file__), "..", "data",
                           f"score_re_v2_combos_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv")
        cdf.to_csv(cp, index=False)
        print(f"  saved combo CSV -> {cp}")

    # === Optuna global score formula search ===
    if args.optuna > 0:
        print(f"\n=== OPTUNA global score formula search ({args.optuna} trials) ===")
        w_best, thr_q, train_delta, test_delta, n_test_kept, n_test = optuna_score_formula(
            train_df, test_df, feats_used, args.pos_usd, days_train, days_test,
            n_trials=args.optuna,
        )
        print(f"\n  Best score formula (trained on train, tested on test):")
        print(f"  threshold: keep top {(1-thr_q)*100:.0f}% by score")
        for f, w in sorted(w_best.items(), key=lambda x: -abs(x[1])):
            sign = "+" if w > 0 else ""
            print(f"    weight[{f:<28}] = {sign}{w:+.3f}")
        print(f"\n  train delta $/d: ${train_delta:+.2f}  base $/d: ${base_train_d:+.2f}")
        print(f"  test  delta $/d: ${test_delta:+.2f}   base $/d: ${base_test_d:+.2f}  (kept {n_test_kept}/{n_test})")
        if test_delta > 0:
            print(f"  → SIGNAL HOLDS OUT-OF-SAMPLE ✓")
        elif test_delta > -0.5:
            print(f"  → marginal — close to break-even")
        else:
            print(f"  → OVERFIT (test delta worse than baseline)")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
