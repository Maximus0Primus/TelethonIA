"""E17 — le TEXTE du message porte-t-il un signal au-delà du sentiment ?

`kol_mentions.message_text` est rempli à 100 % (51 k messages) et n'a jamais servi
autrement que via le `sentiment` déjà calculé. Les features textuelles simples
(longueur, ratio de majuscules, "!", 🚀, "Nx") ont déjà été testées et sont mortes
ou au niveau de la référence.

Reste le contenu sémantique. Test au coût le plus bas qui soit honnête : TF-IDF
mots + bigrammes, régression logistique, cible = survie (la seule cible qui ait
montré du signal en E05).

Trois modèles, pour isoler l'apport propre du texte :
  A. sentiment seul          (ce qu'on exploite déjà)
  B. texte seul              (TF-IDF)
  C. sentiment + texte       (le texte ajoute-t-il quelque chose ?)

Split TEMPOREL et non aléatoire — la question est de généraliser vers l'avant, et
le vocabulaire des memecoins tourne vite (les tickers de mai n'existent plus en
juillet), ce qui est précisément le risque de surapprentissage ici.

Contrôle : mêmes modèles, labels mélangés, 8 tirages, lecture de la fourchette.

    python scripts/_text_signal.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scraper"))

WINDOW_DAYS = 120
TRAIN_FRAC = 0.60
N_NULL = 8
SEED = 7


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

    tr = page("paper_trades", "token_address,kol_group,created_at",
              [("chain", "solana"), ("source", "rt"), ("strategy", "TP50_SL30")], "created_at")
    men = page("kol_mentions", "kol_group,resolved_ca,sentiment,message_text,message_date",
               [("chain", "solana")], "message_date")

    df = pd.DataFrame(tr)
    df = df[~df["kol_group"].isin(bl)].sort_values("created_at").drop_duplicates("token_address")
    m = pd.DataFrame(men).sort_values("message_date").drop_duplicates(["kol_group", "resolved_ca"])
    df = df.merge(m.rename(columns={"resolved_ca": "token_address"})[
        ["kol_group", "token_address", "sentiment", "message_text"]],
        on=["kol_group", "token_address"], how="inner")

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
    df = df[(df["n_strats"] >= 20) & df["message_text"].notna()]
    df["y"] = (df["pire"] > -0.5).astype(int)
    return df.sort_values("created_at").reset_index(drop=True)


def clean(s: str) -> str:
    """Retire URLs, adresses de contrat et $TICKERS. Sans ça le modèle apprend
    des tokens précis (mémorisation pure) au lieu du langage."""
    s = re.sub(r"https?://\S+", " ", str(s))
    s = re.sub(r"\b[1-9A-HJ-NP-Za-km-z]{32,44}\b", " ", s)   # adresses base58
    s = re.sub(r"\$[A-Za-z0-9_]+", " ", s)
    return s.lower()


def evaluate(df, mode, shuffle=False, seed=SEED):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from scipy.sparse import hstack, csr_matrix

    n_tr = int(len(df) * TRAIN_FRAC)
    tr, te = df.iloc[:n_tr], df.iloc[n_tr:]
    ytr = tr["y"].values.copy()
    if shuffle:
        np.random.default_rng(seed).shuffle(ytr)

    blocks_tr, blocks_te = [], []
    if mode in ("texte", "les deux"):
        vec = TfidfVectorizer(preprocessor=clean, ngram_range=(1, 2), min_df=5,
                              max_features=4000, sublinear_tf=True)
        blocks_tr.append(vec.fit_transform(tr["message_text"]))
        blocks_te.append(vec.transform(te["message_text"]))
    if mode in ("sentiment", "les deux"):
        s_tr = tr["sentiment"].fillna(tr["sentiment"].median()).values.reshape(-1, 1)
        s_te = te["sentiment"].fillna(tr["sentiment"].median()).values.reshape(-1, 1)
        blocks_tr.append(csr_matrix(s_tr))
        blocks_te.append(csr_matrix(s_te))

    Xtr = hstack(blocks_tr).tocsr() if len(blocks_tr) > 1 else blocks_tr[0]
    Xte = hstack(blocks_te).tocsr() if len(blocks_te) > 1 else blocks_te[0]

    mdl = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED).fit(Xtr, ytr)
    p = mdl.predict_proba(Xte)[:, 1]
    k = max(int(len(te) * 0.20), 25)
    top = te.iloc[np.argsort(-p)[:k]]
    return {"auc": roc_auc_score(te["y"], p), "prec": 100 * top["y"].mean(),
            "base": 100 * te["y"].mean()}


def main():
    df = load()
    n_tr = int(len(df) * TRAIN_FRAC)
    print(f"Messages avec texte + outcome: {len(df)}")
    print(f"  split temporel: train {n_tr} / test {len(df)-n_tr}")
    print(f"  taux de base (survie): {100*df['y'].mean():.1f}%")
    print(f"  exemple nettoye: {clean(df.iloc[0]['message_text'])[:90]!r}\n")

    print(f"{'modele':<22}{'AUC':>7}{'prec@top20%':>13}{'base':>8}"
          f"{'hasard AUC p10-p90':>24}{'verdict':>9}")
    print("-" * 84)
    for mode in ["sentiment", "texte", "les deux"]:
        real = evaluate(df, mode)
        nulls = sorted(evaluate(df, mode, True, s)["auc"] for s in range(N_NULL))
        lo, hi = nulls[1], nulls[-2]
        verdict = "SIGNAL" if real["auc"] > hi else "bruit"
        print(f"{mode:<22}{real['auc']:>7.3f}{real['prec']:>12.1f}%{real['base']:>7.1f}%"
              f"{lo:>13.3f} a {hi:>5.3f}{verdict:>9}")
    print("\nLe texte n'apporte que s'il BAT 'sentiment' — sinon il ne fait que le re-apprendre.")


if __name__ == "__main__":
    main()
