"""E23 — faire varier la taille par trade selon la proba de survie.

E22 a montré que le levier est le sizing (f=1 → −99 %, f=0.10 → +313 %) avec un f
CONSTANT. E05 a montré que la survie est prédictible (AUC 0.72). E14 a tué le SL
conditionnel, mais pas la TAILLE conditionnelle — ce sont deux choses différentes:
le SL agit sur la forme du trade, la taille agit sur l'allocation de capital.

Test propre
-----------
Mêmes trades, même f MOYEN déployé — seule la répartition entre trades change.
Sinon on comparerait "plus de capital" à "mieux réparti", ce qui ne prouverait rien.
Chaque règle est renormalisée pour que mean(f_i) == f_base.

Hors échantillon: le modèle est entraîné sur les 60 % les plus anciens, l'évaluation
ne porte que sur les 40 % récents. Un sizing calé sur des prédictions in-sample
serait sans valeur.

Contrôle: la même règle appliquée à des p_survie MÉLANGÉES. Si le mélange fait
aussi bien, la variation de taille n'exploite rien — elle ajoute juste de la
variance.

    python scripts/_variable_sizing.py
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
F_BASE = 0.10          # quart-Kelly retenu en E22
LIVE_COST = 0.004
SEED = 7
FEAT_TOKEN = ["entry_score", "rt_score", "rt_liquidity_usd", "rt_token_age_hours",
              "rt_buy_sell_ratio", "entry_mcap", "kol_win_rate", "momentum_mult",
              "rt_volume_24h", "rt_is_pump_fun"]
FEAT_KOL = ["kol_gap_h", "sentiment", "msg_conviction_score"]


def load():
    from datetime import datetime, timedelta, timezone
    from supabase import create_client
    cli = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    since = (datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)).isoformat()
    cfg = cli.table("scoring_config").select("paper_trade_config").eq("id", 1).execute().data[0]
    bl = set((cfg["paper_trade_config"].get("kol_chain_blacklist") or {}).get("solana") or [])

    def page(table, select, eq, key):
        out, cur = [], None
        while True:
            q = cli.table(table).select(select)
            for k, v in eq:
                q = q.eq(k, v)
            rows = q.gte(key, cur or since).order(key).limit(1000).execute().data
            if not rows:
                break
            out.extend(rows)
            if len(rows) < 1000 or rows[-1][key] == cur:
                break
            cur = rows[-1][key]
        return out

    tr = page("paper_trades",
              "token_address,kol_group,created_at,pnl_pct,status,entry_score,rt_score,"
              "rt_liquidity_usd,rt_token_age_hours,rt_buy_sell_ratio,entry_mcap,"
              "kol_win_rate,momentum_mult,rt_volume_24h,rt_is_pump_fun",
              [("chain", "solana"), ("source", "rt"),
               ("strategy", "FAST_TP50_SL30_MCAP_S40")], "created_at")
    men = page("kol_mentions", "kol_group,resolved_ca,sentiment,msg_conviction_score,message_date",
               [("chain", "solana")], "message_date")

    df = pd.DataFrame(tr)
    df = df[df["status"].isin(["tp_hit", "sl_hit", "timeout", "be_stop", "trail_stop"])]
    df = df[(df["pnl_pct"].notna()) & (df["pnl_pct"] <= 20)]
    df = df[~df["kol_group"].isin(bl)]
    m = pd.DataFrame(men).sort_values("message_date").drop_duplicates(["kol_group", "resolved_ca"])
    df = df.merge(m.rename(columns={"resolved_ca": "token_address"})[
        ["kol_group", "token_address", "sentiment", "msg_conviction_score"]],
        on=["kol_group", "token_address"], how="left")
    df["created_at"] = pd.to_datetime(df["created_at"], format="ISO8601")
    df = df.sort_values(["kol_group", "created_at"])
    df["kol_gap_h"] = df.groupby("kol_group")["created_at"].diff().dt.total_seconds() / 3600
    df = df.sort_values("created_at")
    # dédup 24h par token
    df["g"] = df.groupby("token_address")["created_at"].diff()
    df = df[df["g"].isna() | (df["g"] > pd.Timedelta("24h"))]

    out, off = [], 0
    while True:
        r = cli.table("ml_token_outcomes").select("token_address,pire,n_strats") \
              .range(off, off + 999).execute().data
        if not r:
            break
        out.extend(r)
        if len(r) < 1000:
            break
        off += 1000
    df = df.merge(pd.DataFrame(out), on="token_address", how="inner")
    df = df[df["n_strats"] >= 20]
    df["y_nodump"] = df["pire"] > -0.5
    return df.sort_values("created_at").reset_index(drop=True)


def predict_survival(df):
    from sklearn.ensemble import HistGradientBoostingClassifier
    n_tr = int(len(df) * TRAIN_FRAC)
    tr, te = df.iloc[:n_tr].copy(), df.iloc[n_tr:].copy()
    gm = tr["y_nodump"].mean()
    e = tr.groupby("kol_group")["y_nodump"].agg(["mean", "count"])
    e["te"] = (e["mean"] * e["count"] + gm * 10) / (e["count"] + 10)
    for p in (tr, te):
        p["_te"] = p["kol_group"].map(e["te"]).fillna(gm)
    cols = FEAT_TOKEN + FEAT_KOL

    def X(p):
        o = p[cols].apply(pd.to_numeric, errors="coerce").assign(kol_te=p["_te"].values)
        return o.fillna(o.median(numeric_only=True)).fillna(0.0)

    mdl = HistGradientBoostingClassifier(max_iter=200, max_depth=4, learning_rate=0.06,
                                         random_state=SEED).fit(X(tr), tr["y_nodump"].astype(int))
    return te.assign(p_survie=mdl.predict_proba(X(te))[:, 1])


def equity(r, f):
    """Courbe de capital, f par trade. Retourne (final, drawdown_max)."""
    cap, peak, mdd = 1.0, 1.0, 0.0
    for ri, fi in zip(r, f):
        cap *= max(1 + fi * ri, 1e-4)
        peak = max(peak, cap)
        mdd = max(mdd, (peak - cap) / peak)
    return cap, mdd


def norm(w):
    """Renormalise pour que la fraction MOYENNE deployee reste F_BASE."""
    w = np.clip(w, 0.0, None)
    return w * (F_BASE / w.mean()) if w.mean() > 0 else np.full(len(w), F_BASE)


def main():
    df = load()
    te = predict_survival(df)
    band = te[(te["sentiment"] >= 0.30) & (te["sentiment"] < 0.70)].copy()
    band = band.sort_values("created_at")
    r = (band["pnl_pct"] - LIVE_COST).values
    p = band["p_survie"].values
    print(f"Trades hors echantillon dans la bande: {len(band)}")
    print(f"proba de survie: min {p.min():.2f} / med {np.median(p):.2f} / max {p.max():.2f}\n")

    def rules(pv):
        med = np.median(pv)
        return {
            "constant (E22)":            np.full(len(pv), F_BASE),
            "proportionnel a p":         norm(pv),
            "proportionnel a p^2":       norm(pv ** 2),
            "binaire 1.5x / 0.5x":       norm(np.where(pv >= med, 1.5, 0.5)),
            "binaire 2x / 0x (skip)":    norm(np.where(pv >= med, 1.0, 0.0)),
            "top 30% seulement":         norm((pv >= np.quantile(pv, 0.70)).astype(float)),
        }

    rng = np.random.default_rng(SEED)
    print(f"{'regle de taille':<26}{'capital':>10}{'DD max':>9}{'f moyen':>9}"
          f"{'hasard (5 tirages)':>26}")
    print("-" * 80)
    real = rules(p)
    for name, w in real.items():
        cap, mdd = equity(r, w)
        if name == "constant (E22)":
            ctrl = "  (invariant au melange)"
        else:
            sims = []
            for s in range(5):
                w_s = rules(rng.permutation(p))[name]
                sims.append(equity(r, w_s)[0])
            ctrl = f"{100*(min(sims)-1):>9.0f}% a {100*(max(sims)-1):>5.0f}%"
        print(f"{name:<26}{100*(cap-1):>9.0f}%{100*mdd:>8.1f}%{w.mean():>9.3f}{ctrl:>26}")


if __name__ == "__main__":
    main()
