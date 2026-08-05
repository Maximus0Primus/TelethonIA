"""Conditional stop-loss: does the OPTIMAL SL depend on the predicted dump risk?

Every strategy in the grid applies one fixed SL to every token. But Aug 5 showed
two things that together make that look wrong:

  - 74-82% of tokens that eventually run first dump ~-50%, which is exactly what
    takes a 30-40% stop out before the move.
  - Whether a token dumps is PREDICTABLE (AUC 0.72), while whether it runs is not.

So the question is not "which SL is best" but "which SL is best GIVEN the dump
risk". Unconditionally, tighter is better: TP50_SL30 -1.72% down to TP50_NOSL
-4.20% over 2816 tokens. The hypothesis is that this ordering INVERTS on tokens
predicted to survive — there, a wide stop should let you sit through the flush.

No new simulation needed: the shadow grid already holds the same TP50 at six SL
levels on the identical token set. We only add the survival probability.

Falsification: if the best SL is the same in every survival bucket, the idea is
dead. And the whole interaction is re-run with p_survie SHUFFLED — if the shuffled
version shows the same pattern, it is an artefact of bucketing, not of prediction.

    python scripts/_conditional_sl.py
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
SEED = 7
SL_VARIANTS = ["TP50_SL30", "TP50_SL40", "TP50_SL50", "TP50_SL60", "TP50_SL70", "TP50_NOSL"]
SL_LABEL = {"TP50_SL30": "SL 30%", "TP50_SL40": "SL 40%", "TP50_SL50": "SL 50%",
            "TP50_SL60": "SL 60%", "TP50_SL70": "SL 70%", "TP50_NOSL": "SL 80% (NOSL)"}
FEAT_TOKEN = ["entry_score", "rt_score", "rt_liquidity_usd", "rt_token_age_hours",
              "rt_buy_sell_ratio", "entry_mcap", "kol_win_rate", "momentum_mult",
              "rt_volume_24h", "rt_is_pump_fun"]
FEAT_KOL = ["kol_gap_h", "sentiment", "msg_conviction_score"]


def sb():
    from supabase import create_client
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])


def page(cli, table, select, eq=(), key="created_at", since=None, inq=None):
    out, cur = [], None
    while True:
        q = cli.table(table).select(select)
        for k, v in eq:
            q = q.eq(k, v)
        if inq:
            q = q.in_(*inq)
        if since:
            q = q.gte(key, cur or since)
        rows = q.order(key).limit(1000).execute().data
        if not rows:
            break
        out.extend(rows)
        if len(rows) < 1000 or rows[-1][key] == cur:
            break
        cur = rows[-1][key]
    return out


def load():
    from datetime import datetime, timedelta, timezone
    cli = sb()
    since = (datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)).isoformat()
    cfg = cli.table("scoring_config").select("paper_trade_config").eq("id", 1).execute().data[0]
    bl = set((cfg["paper_trade_config"].get("kol_chain_blacklist") or {}).get("solana") or [])

    feat = page(cli, "paper_trades",
                "token_address,kol_group,created_at,entry_score,rt_score,rt_liquidity_usd,"
                "rt_token_age_hours,rt_buy_sell_ratio,entry_mcap,kol_win_rate,momentum_mult,"
                "rt_volume_24h,rt_is_pump_fun",
                eq=[("chain", "solana"), ("source", "rt"), ("strategy", "TP50_SL30")], since=since)
    men = page(cli, "kol_mentions", "kol_group,resolved_ca,sentiment,msg_conviction_score,message_date",
               eq=[("chain", "solana")], key="message_date", since=since)

    df = pd.DataFrame(feat)
    df = df[~df["kol_group"].isin(bl)].sort_values("created_at").drop_duplicates("token_address")
    m = pd.DataFrame(men).sort_values("message_date").drop_duplicates(["kol_group", "resolved_ca"])
    df = df.merge(m.rename(columns={"resolved_ca": "token_address"})[
        ["kol_group", "token_address", "sentiment", "msg_conviction_score"]],
        on=["kol_group", "token_address"], how="left")
    df["created_at"] = pd.to_datetime(df["created_at"], format="ISO8601")
    df = df.sort_values(["kol_group", "created_at"])
    df["kol_gap_h"] = df.groupby("kol_group")["created_at"].diff().dt.total_seconds() / 3600
    df = df.sort_values("created_at").reset_index(drop=True)

    # survival label, from the whole-grid aggregate
    out, off = [], 0
    while True:
        r = cli.table("ml_token_outcomes").select("token_address,best,pire,n_strats") \
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

    # the six SL variants, same TP, same tokens
    sl = page(cli, "paper_trades", "token_address,strategy,pnl_pct,status,created_at",
              eq=[("chain", "solana"), ("source", "rt")],
              inq=("strategy", SL_VARIANTS), since=since)
    s = pd.DataFrame(sl)
    s = s[s["status"].isin(["tp_hit", "sl_hit", "timeout", "be_stop", "trail_stop"])]
    s = s[s["pnl_pct"] <= 20]
    s = s.sort_values("created_at").drop_duplicates(["token_address", "strategy"])
    wide = s.pivot(index="token_address", columns="strategy", values="pnl_pct")
    return df.merge(wide, on="token_address", how="inner").reset_index(drop=True)


def fit_survival(df, shuffle=False, seed=SEED):
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

    y = tr["y_nodump"].astype(int).values
    mdl = HistGradientBoostingClassifier(max_iter=200, max_depth=4, learning_rate=0.06,
                                         random_state=SEED).fit(X(tr), y)
    te = te.assign(p_survie=mdl.predict_proba(X(te))[:, 1])
    if shuffle:
        te["p_survie"] = np.random.default_rng(seed).permutation(te["p_survie"].values)
    return te


def table(te, title):
    te = te.copy()
    te["bucket"] = pd.qcut(te["p_survie"], 3, labels=["risque ELEVE", "moyen", "SURVIE probable"])
    print(f"\n{title}")
    print(f"{'bucket':<20}{'n':>5}" + "".join(f"{SL_LABEL[s]:>15}" for s in SL_VARIANTS) + f"{'meilleur':>17}")
    print("-" * (25 + 15 * len(SL_VARIANTS) + 17))
    rows = {}
    for b in ["risque ELEVE", "moyen", "SURVIE probable"]:
        sub = te[te["bucket"] == b]
        evs = {s: 100 * sub[s].mean() for s in SL_VARIANTS if s in sub}
        rows[b] = evs
        best = max(evs, key=evs.get)
        cells = "".join(f"{evs[s]:>14.2f}%" for s in SL_VARIANTS)
        print(f"{str(b):<20}{len(sub):>5}{cells}{SL_LABEL[best]:>17}")
    return rows


def main():
    df = load()
    print(f"Tokens avec les 6 variantes de SL + features: {len(df)}")
    te = fit_survival(df)
    n_tr = int(len(df) * TRAIN_FRAC)
    print(f"Modele entraine sur {n_tr}, evalue sur {len(te)} (split temporel)")

    real = table(te, "=== EV par (risque de dump predit) x (niveau de SL) — REEL ===")
    fake = table(fit_survival(df, shuffle=True), "=== MEME CHOSE, p_survie MELANGEE (controle) ===")

    print("\n=== VERDICT ===")
    b_hi = max(real["SURVIE probable"], key=real["SURVIE probable"].get)
    b_lo = max(real["risque ELEVE"], key=real["risque ELEVE"].get)
    print(f"Meilleur SL si survie probable : {SL_LABEL[b_hi]}")
    print(f"Meilleur SL si risque eleve    : {SL_LABEL[b_lo]}")
    if b_hi == b_lo:
        print("=> MEME SL optimal dans les deux cas: l'hypothese du SL conditionnel est MORTE.")
    else:
        print("=> SL optimal DIFFERENT selon le risque: l'hypothese tient, a confirmer vs controle.")

    # Politique conditionnelle vs meilleur SL fixe, sur les memes tokens
    print("")
    fixed_best = max(SL_VARIANTS, key=lambda s: te[s].mean())
    ev_fixed = 100 * te[fixed_best].mean()
    te2 = te.copy()
    te2["bucket"] = pd.qcut(te2["p_survie"], 3, labels=["risque ELEVE", "moyen", "SURVIE probable"])
    ev_cond = 100 * np.mean([
        te2[te2["bucket"] == b][max(real[b], key=real[b].get)].mean()
        for b in ["risque ELEVE", "moyen", "SURVIE probable"]])
    print(f"Meilleur SL FIXE ({SL_LABEL[fixed_best]}) sur tout le test : {ev_fixed:>6.2f}%")
    print(f"Politique CONDITIONNELLE (meilleur SL par bucket)          : {ev_cond:>6.2f}%")
    print(f"Gain                                                        : {ev_cond-ev_fixed:>+6.2f}pp")
    print("\n(le gain conditionnel est optimiste par construction: le meilleur SL par bucket")
    print(" est choisi SUR le test. A ne lire que comme une borne haute.)")


if __name__ == "__main__":
    main()
