"""ML, reformulated. The first attempt (_ml_axis_test.py) was the weakest possible
framing and its negative result only condemns that framing.

What was wrong with it
----------------------
Regression on `pnl_pct` — a fat-tailed continuous target — then take the top 20%.
Fat tails make the squared-error objective chase outliers, and the top-k metric on
such a target has a ~10pp wide noise floor, so nothing below a huge effect is
detectable. Concluding "ML does not work" from that was premature.

What this does instead
----------------------
1. CLASSIFICATION on binary, path-aware targets. Base rates on 1545 tokens:
     2x           = 24.8%      (does the token double at all)
     clean 2x     = 10.7%      (doubles WITHOUT first dumping -40%)
     no_dump      = 38.7%      (never drops below -50%)
   `clean 2x` is the one that matters operationally: the Aug-5 regime finding is
   that 74-82% of runners now dump ~-50% before running, which is exactly what
   takes an SL out. Predicting which ones DON'T is worth more than predicting
   which ones eventually run.

2. TOKEN-LEVEL, not strategy-level. The old target conflated "was this a good
   token" with "was this exit right for it". Separating them is the point.

3. Judged by PRECISION@K against the base rate, plus the realized EV of trading
   the top-k. AUC is reported but never decisive — we only ever act on the top.

4. Every cell gets N_NULL permutation draws and we read the p10-p90 SPREAD.
   One draw is not enough on a fat-tailed selection metric; that lesson cost a
   wrong verdict earlier today.

    python scripts/_ml_reformulation.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scraper"))

WINDOW_DAYS = 120
TRAIN_FRAC = 0.60
TOP_FRAC = 0.20
N_NULL = 12
SEED = 7
MIN_STRATS_PER_TOKEN = 20          # token must be covered by enough exits to trust max/min


def fetch() -> pd.DataFrame:
    from datetime import datetime, timedelta, timezone
    from supabase import create_client

    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    since = (datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)).isoformat()
    cfg = sb.table("scoring_config").select("paper_trade_config").eq("id", 1).execute().data[0]
    blacklist = set((cfg["paper_trade_config"].get("kol_chain_blacklist") or {}).get("solana") or [])

    def page(table, select, extra, key):
        out, cur = [], None
        while True:
            q = sb.table(table).select(select).gte(key, cur or since).order(key).limit(1000)
            for k, v in extra:
                q = q.eq(k, v)
            rows = q.execute().data
            if not rows:
                break
            out.extend(rows)
            if len(rows) < 1000 or rows[-1][key] == cur:
                break
            cur = rows[-1][key]
        return out

    # One reference exit carries the per-token FEATURES (they are identical across
    # exits for a given token); outcomes come from the whole grid via an aggregate.
    feat = page("paper_trades",
                "token_address,kol_group,created_at,entry_score,rt_score,rt_liquidity_usd,"
                "rt_token_age_hours,rt_buy_sell_ratio,entry_mcap,kol_win_rate,momentum_mult,"
                "rt_volume_24h,rt_is_pump_fun",
                [("chain", "solana"), ("source", "rt"), ("strategy", "TP50_SL30")], "created_at")
    men = page("kol_mentions", "kol_group,resolved_ca,sentiment,msg_conviction_score,message_date",
               [("chain", "solana")], "message_date")

    df = pd.DataFrame(feat)
    df = df[~df["kol_group"].isin(blacklist)]
    df = df.sort_values("created_at").drop_duplicates("token_address", keep="first")

    m = pd.DataFrame(men).sort_values("message_date").drop_duplicates(
        ["kol_group", "resolved_ca"], keep="first")
    df = df.merge(m.rename(columns={"resolved_ca": "token_address"})[
        ["kol_group", "token_address", "sentiment", "msg_conviction_score"]],
        on=["kol_group", "token_address"], how="left")

    df["created_at"] = pd.to_datetime(df["created_at"], format="ISO8601")
    df = df.sort_values(["kol_group", "created_at"])
    df["kol_gap_h"] = df.groupby("kol_group")["created_at"].diff().dt.total_seconds() / 3600
    return df.sort_values("created_at").reset_index(drop=True)


FEAT_TOKEN = ["entry_score", "rt_score", "rt_liquidity_usd", "rt_token_age_hours",
              "rt_buy_sell_ratio", "entry_mcap", "kol_win_rate", "momentum_mult",
              "rt_volume_24h", "rt_is_pump_fun"]
FEAT_KOL = ["kol_gap_h", "sentiment", "msg_conviction_score"]


def run(df, cols, ident, target, shuffle=False, seed=SEED):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    n_tr = int(len(df) * TRAIN_FRAC)
    tr, te = df.iloc[:n_tr].copy(), df.iloc[n_tr:].copy()

    # KOL identity target-encoded on TRAIN ONLY, against the same binary target.
    gm = tr[target].mean()
    e = tr.groupby("kol_group")[target].agg(["mean", "count"])
    e["te"] = (e["mean"] * e["count"] + gm * 10) / (e["count"] + 10)
    for part in (tr, te):
        part["_te"] = part["kol_group"].map(e["te"]).fillna(gm)

    def X(p):
        out = p[cols].apply(pd.to_numeric, errors="coerce") if cols else pd.DataFrame(index=p.index)
        if ident:
            out = out.assign(kol_te=p["_te"].values)
        return out.fillna(out.median(numeric_only=True)).fillna(0.0)

    ytr = tr[target].values.astype(int).copy()
    if shuffle:
        np.random.default_rng(seed).shuffle(ytr)
    if ytr.sum() < 10 or ytr.sum() == len(ytr):
        return None

    mdl = HistGradientBoostingClassifier(max_iter=200, max_depth=4, learning_rate=0.06,
                                         random_state=SEED)
    mdl.fit(X(tr), ytr)
    p = mdl.predict_proba(X(te))[:, 1]

    k = max(int(len(te) * TOP_FRAC), 25)
    top = te.iloc[np.argsort(-p)[:k]]
    return {
        "prec_top": 100 * top[target].mean(),
        "base": 100 * te[target].mean(),
        "auc": roc_auc_score(te[target].astype(int), p) if te[target].nunique() > 1 else 0.5,
        "ev_top": 100 * top["best"].mean(),
        "ev_base": 100 * te["best"].mean(),
    }


def main():
    df = fetch()
    print(f"Tokens avec features: {len(df)}")

    # Outcomes per token: max/min pnl across the WHOLE exit grid, materialised
    # server-side in ml_token_outcomes (PostgREST cannot GROUP BY, and pulling
    # 1.1M raw rows to aggregate client-side would be absurd).
    from supabase import create_client
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    rows, off = [], 0
    while True:
        r = sb.table("ml_token_outcomes").select("token_address,best,pire,n_strats")               .range(off, off + 999).execute().data
        if not r:
            break
        rows.extend(r)
        if len(r) < 1000:
            break
        off += 1000
    out = pd.DataFrame(rows)
    print(f"Outcomes charges: {len(out)} tokens")
    df = df.merge(out, on="token_address", how="inner")
    df = df[df["n_strats"] >= MIN_STRATS_PER_TOKEN].reset_index(drop=True)

    df["y_2x"] = (df["best"] > 1.0)
    df["y_clean2x"] = (df["best"] > 1.0) & (df["pire"] > -0.4)
    df["y_nodump"] = (df["pire"] > -0.5)

    n_tr = int(len(df) * TRAIN_FRAC)
    print(f"Tokens retenus: {len(df)} | split temporel train {n_tr} / test {len(df)-n_tr}")
    print(f"  taux de base — 2x {100*df['y_2x'].mean():.1f}% | "
          f"2x propre {100*df['y_clean2x'].mean():.1f}% | sans dump {100*df['y_nodump'].mean():.1f}%\n")

    sets = [("token seules", FEAT_TOKEN, False),
            ("token + axe KOL", FEAT_TOKEN + FEAT_KOL, True),
            ("axe KOL seul", FEAT_KOL, True)]

    for target, tname in [("y_2x", "2x"), ("y_clean2x", "2x PROPRE (sans dump -40%)"),
                          ("y_nodump", "pas de dump -50%")]:
        print(f"=== cible: {tname} ===")
        print(f"{'features':<20}{'prec@top20%':>13}{'base':>8}{'AUC':>7}"
              f"{'hasard p10-p90':>20}{'verdict':>9}")
        for label, cols, ident in sets:
            real = run(df.copy(), cols, ident, target)
            if real is None:
                continue
            nulls = sorted(r["prec_top"] for s in range(N_NULL)
                           if (r := run(df.copy(), cols, ident, target, True, s)))
            lo, hi = nulls[1], nulls[-2]
            verdict = "SIGNAL" if real["prec_top"] > hi else "bruit"
            print(f"{label:<20}{real['prec_top']:>12.1f}%{real['base']:>7.1f}%"
                  f"{real['auc']:>7.2f}{lo:>11.1f}% a {hi:>4.1f}%{verdict:>9}")
        print()

    # ---- Traduction en termes de trading -------------------------------------
    # La precision ne paie pas les factures. Question reelle: si on ne trade QUE
    # les tokens que le modele juge survivants, l'EV monte-t-elle ?
    print("=== TRADUCTION TRADING — cible 'pas de dump -50%' ===")
    from sklearn.ensemble import HistGradientBoostingClassifier
    n_tr = int(len(df) * TRAIN_FRAC)
    tr, te = df.iloc[:n_tr].copy(), df.iloc[n_tr:].copy()
    gm = tr["y_nodump"].mean()
    e = tr.groupby("kol_group")["y_nodump"].agg(["mean", "count"])
    e["te"] = (e["mean"] * e["count"] + gm * 10) / (e["count"] + 10)
    for part in (tr, te):
        part["_te"] = part["kol_group"].map(e["te"]).fillna(gm)
    cols = FEAT_TOKEN + FEAT_KOL
    def X(p):
        o = p[cols].apply(pd.to_numeric, errors="coerce").assign(kol_te=p["_te"].values)
        return o.fillna(o.median(numeric_only=True)).fillna(0.0)
    mdl = HistGradientBoostingClassifier(max_iter=200, max_depth=4, learning_rate=0.06,
                                         random_state=SEED).fit(X(tr), tr["y_nodump"].astype(int))
    te = te.assign(p_survie=mdl.predict_proba(X(te))[:, 1])

    rr, off = [], 0
    while True:
        r = sb.table("ml_token_real_pnl").select("token_address,pnl_reel").range(off, off+999).execute().data
        if not r:
            break
        rr.extend(r)
        if len(r) < 1000:
            break
        off += 1000
    real = pd.DataFrame(rr)
    te = te.merge(real, on="token_address", how="inner")
    print(f"tokens de test apparies a un PnL reel: {len(te)}")
    print(f"{'selection':<34}{'n':>6}{'EV reelle':>12}{'% survivants':>14}{'best moyen':>12}")
    for lab, sub in [("tous (aucun filtre)", te),
                     ("top 50% proba survie", te.nlargest(int(len(te)*0.5), "p_survie")),
                     ("top 30% proba survie", te.nlargest(int(len(te)*0.3), "p_survie")),
                     ("top 20% proba survie", te.nlargest(int(len(te)*0.2), "p_survie")),
                     ("BAS 20% (a eviter)",   te.nsmallest(int(len(te)*0.2), "p_survie"))]:
        print(f"{lab:<34}{len(sub):>6}{100*sub['pnl_reel'].mean():>11.2f}%"
              f"{100*sub['y_nodump'].mean():>13.1f}%{100*sub['best'].mean():>11.1f}%")

    # Redondant avec la bande de sentiment, ou additif ? Croisement 2x2.
    print("")
    print("=== CROISEMENT survie predite x bande de sentiment ===")
    te["in_band"] = te["sentiment"].between(0.30, 0.70, inclusive="left")
    seuil = te["p_survie"].quantile(0.5)
    te["survie_haute"] = te["p_survie"] >= seuil
    print(f"{chr(32):<24}{'bande NON':>22}{'bande OUI':>22}")
    for sv, lab in [(False, "survie basse"), (True, "survie haute")]:
        cells = []
        for bd in (False, True):
            s = te[(te["survie_haute"] == sv) & (te["in_band"] == bd)]
            cells.append(f"{100*s['pnl_reel'].mean():>7.2f}% (n={len(s):>3})" if len(s) else "     -      ")
        print(f"{lab:<24}{cells[0]:>22}{cells[1]:>22}")
    both = te[te["survie_haute"] & te["in_band"]]
    print("")
    print(f"les deux filtres: n={len(both)}, EV={100*both['pnl_reel'].mean():.2f}%, "
          f"soit {len(both)/len(te)*100:.0f}% du flux conserve")


if __name__ == "__main__":
    main()
