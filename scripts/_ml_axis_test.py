"""Decisive cheap test: does the KOL axis carry predictive power the token
features don't?

Context (2026-08-05). Permutation tests this session showed:
  - token features carry ZERO edge (3 real survivors vs 6 under pure chance)
  - the KOL axis does (slingoorioyaps: 90 surviving strategies vs 12)
  - KOL call cadence is a monotone dose-response (-5.43% -> +2.22%)
  - message sentiment is an inverted U (0.5-0.6 = +7.97%, >=0.7 = -11.71%)

train_model.py has ~143 features, ~120 of them on the dead axis, and NEITHER
kol_group nor the call gap. This asks whether that omission is the reason ML
never improved — before spending days reworking the pipeline.

Method
------
One row per (kol, token) first call. Target = pnl_pct on a broad-coverage exit.
Temporal split: oldest 60% train, newest 40% test — never random, the whole
question is whether it generalises FORWARD.

Judged in trading terms, not R²: train the model, take its top-20% picks on the
test half, and read the realized mean pnl_pct. That is directly comparable to the
+7.2%/trade the sentiment band delivers.

Every feature set is run against a PERMUTATION CONTROL (y shuffled in train).
With ~2800 rows and a top-20% selection, a model fitting noise still produces a
plausible-looking number; the control is what makes the real one readable.

    python scripts/_ml_axis_test.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scraper"))

REF_STRATEGY = "TP50_SL30"      # bare grid exit: no score/mcap gate = max coverage
WINDOW_DAYS = 120
TRAIN_FRAC = 0.60
TOP_FRAC = 0.20                 # fraction of test rows the model is allowed to pick
SEED = 7


def fetch() -> pd.DataFrame:
    from supabase import create_client
    from datetime import datetime, timedelta, timezone

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

    trades = page(
        "paper_trades",
        "token_address,kol_group,created_at,pnl_pct,status,entry_score,rt_liquidity_usd,"
        "rt_token_age_hours,rt_buy_sell_ratio,entry_mcap,kol_win_rate,momentum_mult,"
        "rt_volume_24h,rt_is_pump_fun,rt_score",
        [("chain", "solana"), ("source", "rt"), ("strategy", REF_STRATEGY)],
        "created_at")
    mentions = page(
        "kol_mentions", "kol_group,resolved_ca,sentiment,msg_conviction_score,message_date",
        [("chain", "solana")], "message_date")

    df = pd.DataFrame(trades)
    df = df[df["status"].isin(["tp_hit", "sl_hit", "timeout", "be_stop", "trail_stop"])]
    df = df[(df["pnl_pct"].notna()) & (df["pnl_pct"] <= 20)]          # drop corrupt exit_price rows
    df = df[~df["kol_group"].isin(blacklist)]
    df = df.sort_values("created_at").drop_duplicates(["kol_group", "token_address"], keep="first")

    m = pd.DataFrame(mentions).sort_values("message_date")
    m = m.drop_duplicates(["kol_group", "resolved_ca"], keep="first")
    df = df.merge(m.rename(columns={"resolved_ca": "token_address"})[
        ["kol_group", "token_address", "sentiment", "msg_conviction_score"]],
        on=["kol_group", "token_address"], how="left")

    # Call cadence: hours since the SAME KOL's previous call. Causal by construction.
    df["created_at"] = pd.to_datetime(df["created_at"], format="ISO8601")
    df = df.sort_values(["kol_group", "created_at"])
    df["kol_gap_h"] = df.groupby("kol_group")["created_at"].diff().dt.total_seconds() / 3600
    return df.sort_values("created_at").reset_index(drop=True)


AXE_KOL = ["kol_gap_h", "sentiment", "msg_conviction_score"]
AXE_TOKEN = ["entry_score", "rt_score", "rt_liquidity_usd", "rt_token_age_hours",
             "rt_buy_sell_ratio", "entry_mcap", "kol_win_rate", "momentum_mult",
             "rt_volume_24h", "rt_is_pump_fun"]


def build(df, cols, with_identity):
    X = df[cols].apply(pd.to_numeric, errors="coerce").copy()
    if with_identity:
        # Target-encode KOL identity on TRAIN ONLY (done by the caller via mask)
        X["kol_id"] = df["_kol_te"].values
    return X.fillna(X.median(numeric_only=True)).fillna(0.0)


def evaluate(df, cols, with_identity, label, shuffle_y=False, seed=SEED):
    from sklearn.ensemble import HistGradientBoostingRegressor

    n_tr = int(len(df) * TRAIN_FRAC)
    tr, te = df.iloc[:n_tr].copy(), df.iloc[n_tr:].copy()

    # KOL target encoding fitted on train only — leaking it would fake the result.
    gm = tr["pnl_pct"].mean()
    enc = tr.groupby("kol_group")["pnl_pct"].agg(["mean", "count"])
    enc["te"] = (enc["mean"] * enc["count"] + gm * 10) / (enc["count"] + 10)   # smoothed
    df.loc[tr.index, "_kol_te"] = tr["kol_group"].map(enc["te"]).fillna(gm).values
    df.loc[te.index, "_kol_te"] = te["kol_group"].map(enc["te"]).fillna(gm).values
    tr, te = df.iloc[:n_tr], df.iloc[n_tr:]

    Xtr, Xte = build(tr, cols, with_identity), build(te, cols, with_identity)
    ytr = tr["pnl_pct"].values.copy()
    if shuffle_y:
        rng = np.random.default_rng(seed)
        rng.shuffle(ytr)

    mdl = HistGradientBoostingRegressor(max_iter=250, max_depth=4, learning_rate=0.05,
                                        random_state=SEED)
    mdl.fit(Xtr, ytr)
    pred = mdl.predict(Xte)

    k = max(int(len(te) * TOP_FRAC), 20)
    top = te.iloc[np.argsort(-pred)[:k]]
    return {
        "modele": label, "n_test": len(te), "n_pris": k,
        "ev_top": 100 * top["pnl_pct"].mean(),
        "ev_base": 100 * te["pnl_pct"].mean(),
        "wr_top": 100 * (top["pnl_pct"] > 0).mean(),
    }


def main():
    df = fetch()
    print(f"Univers: {len(df)} premiers calls (kol, token) | exit={REF_STRATEGY} | {WINDOW_DAYS}j")
    print(f"  sentiment joint: {df['sentiment'].notna().mean()*100:.1f}%"
          f" | gap calculable: {df['kol_gap_h'].notna().mean()*100:.1f}%")
    n_tr = int(len(df) * TRAIN_FRAC)
    print(f"  split temporel: train {n_tr} (jusqu'au {df.iloc[n_tr-1]['created_at'].date()})"
          f" / test {len(df)-n_tr}\n")

    runs = [
        ("A. axe KOL seul (gap+sentiment+identite)", AXE_KOL, True),
        ("B. features token seules (~ le modele actuel)", AXE_TOKEN, False),
        ("C. token + axe KOL", AXE_TOKEN + AXE_KOL, True),
        ("D. identite KOL seule", [], True),
    ]
    # Un seul tirage hasard ne suffit pas: sur une distribution a queues epaisses,
    # choisir 125 lignes parmi 628 a une variance enorme. On mesure la DISPERSION
    # du bruit pour savoir si la metrique sait discriminer quoi que ce soit.
    N_NULL = 12
    print(f"{'modele':<44}{'EV top20%':>11}{'hasard median':>15}{'hasard p10-p90':>18}{'verdict':>10}")
    print("-" * 98)
    for label, cols, ident in runs:
        real = evaluate(df.copy(), cols, ident, label)
        nulls = sorted(evaluate(df.copy(), cols, ident, label, shuffle_y=True, seed=s)["ev_top"]
                       for s in range(N_NULL))
        lo, med, hi = nulls[1], nulls[N_NULL // 2], nulls[-2]
        verdict = "signal" if real["ev_top"] > hi else "bruit"
        print(f"{label:<44}{real['ev_top']:>10.2f}%{med:>14.2f}%"
              f"{lo:>9.2f}% a {hi:>5.2f}%{verdict:>10}")
    print(f"\nEV de base sur le test: {100*df.iloc[int(len(df)*TRAIN_FRAC):]['pnl_pct'].mean():.2f}%")
    print(f"'hasard' = meme modele, labels melanges, {N_NULL} tirages.")
    print("Si le reel n'est pas AU-DESSUS du p90 du hasard, le modele n'apprend rien de reel.")
    print("Si la fourchette p10-p90 du hasard est large, la metrique est trop bruitee pour trancher.")


if __name__ == "__main__":
    main()
