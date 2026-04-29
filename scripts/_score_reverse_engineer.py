"""v14e.43 — reverse-engineer the scoring system.

Goal: instead of relying on the static SCORE30/35/40/45/50 filter, find which
raw features (rt_score, liquidity, buy/sell ratio, KOL win rate, age, ...)
actually predict outcome (WR + pnl_pct) on the closed paper-trade pool, and
what the optimal threshold per feature looks like per strategy.

Pipeline:
  1. Pull paper_trades 30d closed (any source, any strategy, no artefact)
  2. Train a logistic regression (target = win 0/1) and an XGBoost regressor
     (target = pnl_pct) on the per-strategy pool.
  3. Walk-forward CV: train on first 70%, test on last 30% (chronologically).
  4. For each feature (continuous), find the threshold that maximizes
     `WR_above × N_above` on the train set; check it holds on test.
  5. Compare with the static SCORE30/35/40/45/50 grid: how much better can we do?
  6. Output: data/score_reverse_engineer_<ts>.csv + console summary

Run:
    python scripts/_score_reverse_engineer.py [--days 30] [--chain solana]
                                               [--min-n 80]

Dependencies: scikit-learn, xgboost, pandas, numpy.
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
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
CLOSED = ("sl_hit", "trail_stop", "tp_hit", "timeout", "be_stop")
ARTEFACT_RE = re.compile(r"^(DTRAIL|PTRAIL|SPLIT|TRAIL|TD2|DIP|MOONBAG|WIDE_RUNNER|SCALE_OUT|MCAP_DTRAIL|BOND)")
SUSPECT_SUFFIX = re.compile(r"_(HYST|LAZYXSLOW|LAZYSLOW|LAZYMED|LAZY|COMBO|BOTH|JUPITER)$")

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


def find_optimal_threshold(df, feature, target="won"):
    """For a continuous feature, find the >= threshold that maximizes
    WR × N_above on (df[target]==1) ratio. Returns dict with stats.
    """
    if df[feature].isna().all() or df[feature].nunique() < 5:
        return None
    best = {"threshold": None, "score": -np.inf}
    quantiles = np.linspace(0.05, 0.95, 19)
    for q in quantiles:
        thr = df[feature].quantile(q)
        above = df[df[feature] >= thr]
        below = df[df[feature] < thr]
        if len(above) < 10 or len(below) < 10:
            continue
        wr_above = above[target].mean()
        wr_below = below[target].mean()
        n_above = len(above)
        n_below_v = len(below)
        lift = wr_above - wr_below
        score = lift * np.sqrt(n_above)
        if score > best["score"]:
            best = {
                "threshold": float(thr),
                "wr_above": float(wr_above),
                "wr_below": float(wr_below),
                "lift_pp": float((wr_above - wr_below) * 100),
                "n_above": int(n_above),
                "n_below": int(n_below_v),
                "score": float(score),
            }
    return best if best["threshold"] is not None else None


def find_optimal_combo(df, features, target="won", min_n=10):
    """v14e.43b: 2-feature AND combo scan. For every ordered pair (f1, f2),
    scan a coarse 5x5 quantile grid and return the (thr1, thr2) that
    maximizes WR_intersect × sqrt(N_intersect).
    """
    best = {"score": -np.inf}
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
                    wr_in = df.loc[mask, target].mean()
                    wr_out = df.loc[~mask, target].mean()
                    lift = wr_in - wr_out
                    score = lift * np.sqrt(n_in)
                    if score > best["score"]:
                        best = {
                            "score": float(score),
                            "f1": f1, "thr1": float(thr1),
                            "f2": f2, "thr2": float(thr2),
                            "wr_in": float(wr_in),
                            "wr_out": float(wr_out),
                            "lift_pp": float(lift * 100),
                            "n_in": n_in,
                            "n_out": len(df) - n_in,
                        }
    return best if best.get("f1") else None


def join_token_snapshots(df, batch_size=200):
    """v14e.43b: join token_snapshots for richer features. Match on
    (token_address, snapshot_at closest BEFORE created_at). Only keep
    numeric snapshot features with >= 50% non-NaN coverage.
    """
    addrs = df["token_address"].dropna().unique().tolist() if "token_address" in df.columns else []
    if not addrs:
        print("  no token_address column — snapshots join skipped")
        return df, []

    print(f"  fetching token_snapshots for {len(addrs)} unique tokens...")
    # Pull recent snapshots — use the same time window
    max_ts = df["created_at"].max()
    min_ts = df["created_at"].min() - pd.Timedelta(hours=2)  # 2h buffer for pre-trade snapshot

    # Numeric features to consider (curated subset of the 298 cols, focusing on
    # known signal-rich domains: sentiment, velocity, whale, breadth, lifecycle)
    snap_features = [
        "score", "score_velocity", "mention_velocity", "volume_velocity", "price_velocity",
        "buy_sell_ratio_5m", "buy_sell_ratio_1h", "buy_sell_ratio_24h", "buy_sell_ratio_delta",
        "whale_count", "whale_dominance", "top10_holder_pct", "txn_velocity",
        "breadth", "breadth_val", "consensus_val", "conviction_val",
        "bb_pct_b", "bb_width", "ath_ratio",
        "sentiment", "avg_conviction", "data_confidence",
        "boosts_active", "buys_24h", "txns_5m", "txns_1h", "txns_24h",
        "lifecycle_phase_num", "phase_duration_cycles",
        "ca_mention_count", "ca_mention_ratio",
        "social_velocity", "mention_acceleration", "volume_acceleration",
        "volatility_proxy", "is_dead", "is_pump",
        "best_kol_win_rate", "best_kol_total_calls",
        "bundle_count", "bundle_pct", "bubblemaps_score", "wash_trading_score",
        "score_ema3", "mention_ema3", "volume_ema3",
        "rsi_14", "macd_hist",  # if present
    ]
    snap_sel = "token_address,snapshot_at," + ",".join(snap_features)

    # Fetch in batches by token_address
    snaps = []
    seen = set()
    BATCH = 50
    for i in range(0, len(addrs), BATCH):
        chunk = addrs[i:i+BATCH]
        try:
            r = (sb.table("token_snapshots").select(snap_sel)
                 .in_("token_address", chunk)
                 .gte("snapshot_at", min_ts.isoformat())
                 .lte("snapshot_at", max_ts.isoformat())
                 .execute())
            for x in (r.data or []):
                key = (x["token_address"], x["snapshot_at"])
                if key in seen: continue
                seen.add(key)
                snaps.append(x)
        except Exception as e:
            # Some columns may not exist — try without them
            print(f"    snap fetch batch {i} failed ({str(e)[:80]}), retrying with reduced cols")
            safe_sel = "token_address,snapshot_at,score,sentiment,whale_count,breadth_val,buy_sell_ratio_5m,mention_velocity,lifecycle_phase_num"
            try:
                r = (sb.table("token_snapshots").select(safe_sel)
                     .in_("token_address", chunk)
                     .gte("snapshot_at", min_ts.isoformat())
                     .lte("snapshot_at", max_ts.isoformat())
                     .execute())
                for x in (r.data or []):
                    key = (x["token_address"], x["snapshot_at"])
                    if key in seen: continue
                    seen.add(key); snaps.append(x)
            except Exception as e2:
                print(f"    fallback also failed: {e2}")
    if not snaps:
        print("  no snapshots returned")
        return df, []
    sdf = pd.DataFrame(snaps)
    sdf["snapshot_at"] = pd.to_datetime(sdf["snapshot_at"])
    print(f"  snapshots loaded: {len(sdf)} rows")

    # For each trade, find the latest snapshot BEFORE created_at
    df = df.sort_values("created_at").reset_index(drop=True)
    sdf = sdf.sort_values("snapshot_at").reset_index(drop=True)
    snap_cols = [c for c in sdf.columns if c not in ("token_address", "snapshot_at")]

    # Use merge_asof per token for efficiency
    merged_chunks = []
    for tok, g_trade in df.groupby("token_address"):
        g_snap = sdf[sdf["token_address"] == tok].sort_values("snapshot_at")
        if g_snap.empty:
            continue
        merged = pd.merge_asof(
            g_trade.sort_values("created_at"),
            g_snap[["snapshot_at"] + snap_cols].rename(columns={"snapshot_at": "_snap_at"}),
            left_on="created_at", right_on="_snap_at", direction="backward",
            tolerance=pd.Timedelta(hours=2),
        )
        merged_chunks.append(merged)
    if not merged_chunks:
        print("  no trade × snapshot match")
        return df, []
    merged_df = pd.concat(merged_chunks, ignore_index=True)
    # Filter snap features by NaN coverage >= 50%
    kept_snap_cols = []
    for c in snap_cols:
        if c in merged_df.columns:
            cov = merged_df[c].notna().mean()
            if cov >= 0.5:
                merged_df[c] = pd.to_numeric(merged_df[c], errors="coerce")
                if merged_df[c].notna().any():
                    kept_snap_cols.append(c)
    print(f"  snapshot features kept (>=50% coverage): {len(kept_snap_cols)} of {len(snap_cols)}")
    return merged_df, kept_snap_cols


def train_logreg(X_train, y_train, X_test, y_test, feature_names):
    """Fit logistic regression, return AUC + coef."""
    sc = StandardScaler()
    Xs_train = sc.fit_transform(X_train)
    Xs_test = sc.transform(X_test)
    m = LogisticRegression(max_iter=1000, C=1.0)
    m.fit(Xs_train, y_train)
    auc_train = roc_auc_score(y_train, m.predict_proba(Xs_train)[:, 1]) if len(set(y_train)) > 1 else float("nan")
    auc_test = roc_auc_score(y_test, m.predict_proba(Xs_test)[:, 1]) if len(set(y_test)) > 1 else float("nan")
    coef = dict(zip(feature_names, m.coef_[0]))
    return {"auc_train": auc_train, "auc_test": auc_test, "coef": coef}


def train_xgb_reg(X_train, y_train, X_test, y_test, feature_names):
    """Fit XGBoost regressor (target=pnl_pct), return importance + r2 test."""
    m = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=42,
                         verbosity=0)
    m.fit(X_train, y_train)
    score_test = m.score(X_test, y_test)  # R²
    imp = dict(zip(feature_names, m.feature_importances_))
    return {"r2_test": score_test, "importance": imp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--chain", default="solana")
    ap.add_argument("--min-n", type=int, default=80,
                    help="min samples per strategy to include")
    ap.add_argument("--snapshots", action="store_true",
                    help="join token_snapshots for richer features (slow)")
    ap.add_argument("--combos", action="store_true",
                    help="scan 2-feature AND combos (slower, +mins)")
    ap.add_argument("--top-strats", type=int, default=15,
                    help="how many top strats to combo-scan")
    args = ap.parse_args()

    since = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()
    print(f"=== Score reverse-engineer — chain={args.chain} window={args.days}d ===")

    # Pull paper_trades + features
    cols = ("token_address,strategy,status,pnl_pct,position_usd,entry_price,exit_price,"
            "rt_score,rt_liquidity_usd,rt_volume_24h,rt_buy_sell_ratio,"
            "rt_token_age_hours,rt_is_pump_fun,kol_score,kol_win_rate,kol_tier,"
            "kol_group,n_kol_confirmations,entry_score,entry_mcap,"
            "message_to_buy_seconds,kol_ml_pred,created_at,is_shadow")
    rows = fetch_all("paper_trades", cols, eq_chain=args.chain,
                     gte_created_at=since, neq_source="rt_live")
    closed = [r for r in rows if r.get("status") in CLOSED
              and not is_artefact(r.get("strategy", ""))]
    print(f"  {len(closed)} closed clean rows")

    df = pd.DataFrame(closed)
    df["pnl_pct"] = df["pnl_pct"].astype(float) * 100  # convert to %
    df["won"] = (df["pnl_pct"] > 0).astype(int)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df = df.sort_values("created_at").reset_index(drop=True)

    # Encode kol_tier S=2, A=1, B=0
    tier_map = {"S": 2, "A": 1, "B": 0}
    df["kol_tier_num"] = df["kol_tier"].map(lambda x: tier_map.get(x, 0))

    # Numeric features only
    feats = ["rt_score", "rt_liquidity_usd", "rt_volume_24h", "rt_buy_sell_ratio",
             "rt_token_age_hours", "rt_is_pump_fun", "kol_score", "kol_win_rate",
             "kol_tier_num", "n_kol_confirmations", "entry_score", "entry_mcap",
             "message_to_buy_seconds", "kol_ml_pred"]
    for c in feats:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Log-scale heavy-tail features
    for c in ["rt_liquidity_usd", "rt_volume_24h", "entry_mcap"]:
        df[c + "_log"] = np.log1p(df[c].fillna(0))
    feats_used = feats + ["rt_liquidity_usd_log", "rt_volume_24h_log", "entry_mcap_log"]

    # Drop fully-missing features (e.g. message_to_buy_seconds, kol_ml_pred)
    feats_used = [f for f in feats_used if df[f].notna().any()]

    # v14e.43b: optional join with token_snapshots
    if args.snapshots:
        print(f"\n=== Joining token_snapshots ===")
        df, snap_cols = join_token_snapshots(df)
        feats_used = feats_used + snap_cols
        feats_used = [f for f in feats_used if df[f].notna().any()]
        print(f"  total features after snapshots: {len(feats_used)}")

    print(f"  features available (non-empty): {len(feats_used)}")
    print(f"  feature NaN coverage:")
    for f in feats_used:
        nan_pct = df[f].isna().mean() * 100
        if nan_pct > 5:
            print(f"    {f}: {nan_pct:.1f}% NaN")

    # === Global threshold scan per feature ===
    print(f"\n=== GLOBAL feature threshold scan (target=won) — N={len(df)} ===")
    print(f"  baseline WR (all): {df['won'].mean()*100:.1f}%")
    print(f"  {'feature':<28} {'thresh':>10} {'WR_>=':>6} {'WR_<':>6} {'lift':>6} {'N>=':>6}")
    global_thresholds = {}
    for f in feats_used:
        result = find_optimal_threshold(df, f)
        if result and result["lift_pp"] > 1:
            global_thresholds[f] = result
            print(f"  {f:<28} {result['threshold']:>9.2f} {result['wr_above']*100:>5.1f}% "
                  f"{result['wr_below']*100:>5.1f}% {result['lift_pp']:>+5.1f} {result['n_above']:>6}")

    # === Walk-forward CV: train 70% / test 30% ===
    print(f"\n=== WALK-FORWARD CV — train first 70%, test last 30% ===")
    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    print(f"  train: {len(train_df)} rows ({train_df['created_at'].min()} → {train_df['created_at'].max()})")
    print(f"  test:  {len(test_df)} rows ({test_df['created_at'].min()} → {test_df['created_at'].max()})")
    print(f"  train WR: {train_df['won'].mean()*100:.1f}%   test WR: {test_df['won'].mean()*100:.1f}%")

    X_train = train_df[feats_used].fillna(train_df[feats_used].median()).values
    X_test = test_df[feats_used].fillna(train_df[feats_used].median()).values
    y_train_clf = train_df["won"].values
    y_test_clf = test_df["won"].values
    y_train_reg = train_df["pnl_pct"].values
    y_test_reg = test_df["pnl_pct"].values

    print(f"\n  --- LogReg (target=won) ---")
    lr = train_logreg(X_train, y_train_clf, X_test, y_test_clf, feats_used)
    print(f"    AUC train={lr['auc_train']:.3f}  test={lr['auc_test']:.3f}  "
          f"(0.5=random, >0.6=signal, >0.7=strong)")
    coef_sorted = sorted(lr["coef"].items(), key=lambda x: -abs(x[1]))
    print(f"    Top 8 standardized coefficients:")
    for f, c in coef_sorted[:8]:
        sign = "↑" if c > 0 else "↓"
        print(f"      {f:<28} {sign} coef={c:+.3f}")

    print(f"\n  --- XGBoost (target=pnl_pct) ---")
    xg = train_xgb_reg(X_train, y_train_reg, X_test, y_test_reg, feats_used)
    print(f"    R² test = {xg['r2_test']:+.4f}  (0=random, >0.05=weak signal)")
    imp_sorted = sorted(xg["importance"].items(), key=lambda x: -x[1])
    print(f"    Top 8 feature importance:")
    for f, i in imp_sorted[:8]:
        print(f"      {f:<28} importance={i:.3f}")

    # === Per-strategy scan ===
    print(f"\n=== PER-STRATEGY threshold scan (top strats with N>={args.min_n}) ===")
    by_strat = df.groupby("strategy")
    strat_sizes = by_strat.size().sort_values(ascending=False)
    eligible = strat_sizes[strat_sizes >= args.min_n].index.tolist()
    print(f"  eligible strats (N>={args.min_n}): {len(eligible)}")

    out_rows = []
    for strat in eligible[:25]:
        sub = df[df["strategy"] == strat]
        baseline_wr = sub["won"].mean()
        baseline_avg = sub["pnl_pct"].mean()
        # find best feature × threshold for this strat
        best = {"feature": None, "lift_pp": 0}
        for f in feats_used:
            r = find_optimal_threshold(sub, f)
            if r and r["lift_pp"] > best["lift_pp"]:
                best = dict(r); best["feature"] = f
        out_rows.append({
            "strategy": strat,
            "N_total": len(sub),
            "baseline_wr": round(baseline_wr * 100, 1),
            "baseline_avg_pnl": round(baseline_avg, 1),
            "best_feature": best.get("feature", ""),
            "best_threshold": best.get("threshold"),
            "wr_above": round(best.get("wr_above", 0) * 100, 1) if best.get("feature") else None,
            "wr_below": round(best.get("wr_below", 0) * 100, 1) if best.get("feature") else None,
            "lift_pp": round(best.get("lift_pp", 0), 1),
            "n_above": best.get("n_above"),
        })

    out_df = pd.DataFrame(out_rows).sort_values("lift_pp", ascending=False)

    # v14e.43b: 2-feature combo AND scan on the top strats
    if args.combos:
        print(f"\n=== 2-FEATURE AND combo scan on top {args.top_strats} strats ===")
        combo_rows = []
        for strat in out_df.head(args.top_strats)["strategy"]:
            sub = df[df["strategy"] == strat]
            if len(sub) < 30: continue
            # Restrict to features with enough non-NaN
            local_feats = [f for f in feats_used if sub[f].notna().mean() >= 0.5]
            if len(local_feats) < 2: continue
            best_combo = find_optimal_combo(sub, local_feats, min_n=10)
            if best_combo:
                base_wr = sub["won"].mean()
                combo_rows.append({
                    "strategy": strat,
                    "N_total": len(sub),
                    "base_wr": round(base_wr * 100, 1),
                    "f1": best_combo["f1"],
                    "thr1": best_combo["thr1"],
                    "f2": best_combo["f2"],
                    "thr2": best_combo["thr2"],
                    "wr_in": round(best_combo["wr_in"] * 100, 1),
                    "wr_out": round(best_combo["wr_out"] * 100, 1),
                    "lift_pp": round(best_combo["lift_pp"], 1),
                    "n_in": best_combo["n_in"],
                })
        cdf = pd.DataFrame(combo_rows).sort_values("lift_pp", ascending=False)
        print(f"  TOP {len(cdf)} 2-feature combos:")
        print(f"  {'strategy':<32} {'base_WR':>7} {'f1':<22} {'thr1':>9} {'f2':<22} {'thr2':>9} {'WR_in':>6} {'lift_pp':>7} {'N_in':>5}")
        for _, x in cdf.iterrows():
            f1s, f2s = x["f1"][:20], x["f2"][:20]
            t1 = f"{x['thr1']:.2g}"; t2 = f"{x['thr2']:.2g}"
            print(f"  {x['strategy']:<32} {x['base_wr']:>+5.1f}% {f1s:<22} {t1:>9} {f2s:<22} {t2:>9} {x['wr_in']:>+5.1f}% {x['lift_pp']:>+5.1f} {x['n_in']:>5}")
        # Save combo CSV
        combo_path = os.path.join(os.path.dirname(__file__), "..", "data",
                                   f"score_re_combos_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv")
        cdf.to_csv(combo_path, index=False)
        print(f"  saved combo CSV -> {combo_path}")

    print(f"\n  TOP 15 strategies by best single-feature lift:")
    print(f"  {'strategy':<32} {'N':>4} {'base_WR':>7} {'base_avg':>8} "
          f"{'best_feat':<22} {'thresh':>10} {'WR_>=':>6} {'lift_pp':>7} {'N>=':>5}")
    for _, x in out_df.head(15).iterrows():
        feat_short = (x["best_feature"] or "")[:20]
        thr = x["best_threshold"]
        thr_s = f"{thr:.2f}" if thr is not None else "-"
        wr_a = f"{x['wr_above']:.1f}%" if x["wr_above"] is not None else "-"
        print(f"  {x['strategy']:<32} {x['N_total']:>4} {x['baseline_wr']:>+5.1f}% "
              f"{x['baseline_avg_pnl']:>+6.1f}% {feat_short:<22} {thr_s:>10} "
              f"{wr_a:>6} {x['lift_pp']:>+5.1f} {x['n_above'] or 0:>5}")

    # === Compare static SCORE30/35/40 grid vs feature-driven ===
    print(f"\n=== Static rt_score grid vs OPTIMAL — global ===")
    print(f"  {'gate':<18} {'N':>5} {'WR':>6} {'avg_pnl':>8} {'note':<28}")
    print(f"  {'baseline (all)':<18} {len(df):>5} {df['won'].mean()*100:>5.1f}% "
          f"{df['pnl_pct'].mean():>+6.1f}%")
    for thr in [30, 35, 40, 45, 50]:
        s = df[df["rt_score"] >= thr]
        if len(s) > 0:
            print(f"  {'rt_score >= '+str(thr):<18} {len(s):>5} {s['won'].mean()*100:>5.1f}% "
                  f"{s['pnl_pct'].mean():>+6.1f}% {'static gate':<28}")
    # Apply best threshold from global scan (the feature with highest lift × n_above)
    best_global = max((r for r in global_thresholds.values()),
                      key=lambda r: r["lift_pp"] * np.sqrt(r["n_above"])) if global_thresholds else None
    if best_global:
        f_best = [k for k, v in global_thresholds.items() if v is best_global][0]
        s_opt = df[df[f_best] >= best_global["threshold"]]
        print(f"  OPTIMAL ({f_best:<10}): {len(s_opt):>4} {s_opt['won'].mean()*100:>5.1f}% "
              f"{s_opt['pnl_pct'].mean():>+6.1f}% {'feature-driven':<28}")

    # Save CSV
    out_path = os.path.join(os.path.dirname(__file__), "..", "data",
                            f"score_reverse_engineer_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved per-strategy table -> {out_path}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
