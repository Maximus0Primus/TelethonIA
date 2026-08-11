"""v14e.87 — Recherche EXHAUSTIVE (stratégie x sous-ensemble de KOL) maximisant l'argent.

Objectif: trouver la configuration qui rapporte le plus, **hors échantillon**, sur
l'univers COMPLET — toutes les stratégies, tous les KOL (blacklistés inclus), et
toutes leurs associations.

────────────────────────────────────────────────────────────────────────────────
POURQUOI CETTE RECHERCHE EST EXACTE ET NON HEURISTIQUE
────────────────────────────────────────────────────────────────────────────────
À mise plafonnée (~$100/token, contrainte de liquidité memecoin), l'argent d'une
config est `n x EV`, c'est-à-dire **la somme des pnl_pct des trades retenus**.
Cette somme est **additive sur les KOL** :

    argent(S) = somme_{k dans S} argent(k)

Donc maximiser sur les 2^87 sous-ensembles S se résout exactement :

    S* = { k : argent_train(k) > 0 }

Aucune heuristique, aucun greedy, aucune exploration partielle : toutes les
associations de KOL sont réellement couvertes, en O(nb_kols) au lieu de O(2^n).
C'est ce qui rend « tout tester » faisable ici.

────────────────────────────────────────────────────────────────────────────────
POURQUOI UN PROTOCOLE WALK-FORWARD, ET PAS UN CLASSEMENT
────────────────────────────────────────────────────────────────────────────────
Le 11/08, le même univers a produit un « meilleur » couple KOL x stratégie à
+23.19 % qui était SOUS son plancher de permutation (H0: +29.84 puis +39.67).
Avec 5 471 cellules à queue épaisse, un maximum spectaculaire est garanti même
sans aucun signal. Donc ici :

  * la whitelist est construite UNIQUEMENT sur le train de chaque fold ;
  * elle est notée UNIQUEMENT sur le test, jamais regardé pendant la sélection ;
  * on somme l'argent de test sur tous les folds (walk-forward, fenêtre
    expansive) — c'est la seule quantité qu'on classe ;
  * un CONTRÔLE PAR PERMUTATION rejoue TOUT le pipeline (sélection incluse) sur
    des étiquettes KOL mélangées. Si le meilleur réel ne dépasse pas le maximum
    obtenu sous H0, il n'y a rien, quelle que soit la beauté du chiffre.

Le contrôle porte sur la PROCÉDURE, pas sur une cellule: c'est ce qui manquait
aux classements précédents.

Usage:
    python scripts/kol_strategy_search.py [--since 2026-06-01] [--folds 4]
                                          [--perms 30] [--min-train-tokens 5]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SCRAPER = Path(__file__).resolve().parent.parent / "scraper"
sys.path.insert(0, str(SCRAPER))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(SCRAPER / ".env")
from supabase import create_client  # noqa: E402

PAGE = 1000
# Garde-fou anti-exit_price corrompu (règle projet): pnl_pct > 20 = +2000 %.
PNL_CAP = 20.0


def fetch_all(sb, since: str, start_id: int) -> list[dict]:
    """Pagine paper_trades en KEYSET sur la clé primaire.

    ⚠️ Ne PAS utiliser `.range()` ici: l'offset croissant + les filtres
    non indexés (chain, status, pnl_pct) déclenchent le timeout `57014`
    chronique de ce projet (cf. supabase_disk_io_budget_jul5). Le keyset
    `id > dernier_id` reste sur l'index de la PK et tient la charge.
    Tout le filtrage secondaire se fait côté client, c'est nettement moins
    cher que de faire scanner Postgres.
    """
    cols = "id,strategy,kol_group,token_address,pnl_pct,created_at,status,chain"
    # Point d'entrée: le premier id à partir de la date voulue. Cette table subit
    # des timeouts 57014 chroniques -> on retente au lieu d'abandonner le run.
    # ⚠️ Ne PAS amorcer par `.gte("created_at", ...)`: created_at n'est pas
    # indexe, la requete scanne et meurt en 57014 des que la DB est chargee.
    # L'id de depart est passe en argument (--start-id), obtenu une fois par
    # une agregation cote serveur. A defaut, on part du debut de la table.
    last_id = int(start_id) - 1
    out, seen = [], 0
    while True:
        rows = None
        for attempt in range(6):
            try:
                rows = (sb.table("paper_trades").select(cols)
                        .gt("id", last_id).order("id").limit(PAGE)
                        .execute().data)
                break
            except Exception as e:
                print(f"    [retry {attempt + 1}/6] id>{last_id}: {e}", flush=True)
                time.sleep(5 * (attempt + 1))
        if not rows:
            break
        last_id = rows[-1]["id"]
        seen += len(rows)
        for r in rows:
            if (r.get("chain") == "solana" and r.get("status") != "open"
                    and r.get("pnl_pct") is not None and r.get("kol_group")
                    and float(r["pnl_pct"]) <= PNL_CAP):
                out.append(r)
        if seen % 100_000 < PAGE:
            print(f"    ... {seen:,} lues / {len(out):,} retenues", flush=True)
        if len(rows) < PAGE:
            break
    return out


def dedup(rows: list[dict]) -> list[tuple]:
    """Un seul trade par (stratégie, KOL, token): le PREMIER.

    Règle projet: sans ça, les ré-entrées gonflent artificiellement les gagnants.
    """
    best: dict[tuple, tuple] = {}
    for r in rows:
        key = (r["strategy"], r["kol_group"], r["token_address"])
        ts = r["created_at"]
        cur = best.get(key)
        if cur is None or ts < cur[0]:
            best[key] = (ts, float(r["pnl_pct"]))
    return [(k[0], k[1], k[2], v[0], v[1]) for k, v in best.items()]


def build_folds(times: np.ndarray, n_folds: int) -> np.ndarray:
    """Découpe temporelle en n_folds blocs d'effectifs égaux (quantiles de date)."""
    qs = np.quantile(times, np.linspace(0, 1, n_folds + 1)[1:-1])
    return np.searchsorted(qs, times, side="right")


def walk_forward_money(strat_idx, kol_idx, fold, pnl, n_strats, n_kols, n_folds,
                       min_train_tokens):
    """Argent de test cumulé par stratégie, whitelist re-choisie à chaque fold.

    Retourne (total_test[strat], test_par_fold[strat, fold], whitelist_finale).
    """
    # money[s, k, f] et count[s, k, f] — tout le reste s'en déduit par sommes.
    money = np.zeros((n_strats, n_kols, n_folds))
    count = np.zeros((n_strats, n_kols, n_folds))
    np.add.at(money, (strat_idx, kol_idx, fold), pnl)
    np.add.at(count, (strat_idx, kol_idx, fold), 1.0)

    total_test = np.zeros(n_strats)
    baseline = np.zeros(n_strats)   # même stratégie, AUCUNE sélection de KOL
    per_fold = np.zeros((n_strats, n_folds))
    for f in range(1, n_folds):  # fold 0 sert de train initial
        train_money = money[:, :, :f].sum(axis=2)
        train_count = count[:, :, :f].sum(axis=2)
        # Sélection EXACTE du meilleur sous-ensemble (additivité de l'argent).
        keep = (train_money > 0) & (train_count >= min_train_tokens)
        gain = (money[:, :, f] * keep).sum(axis=1)
        per_fold[:, f] = gain
        total_test += gain
        baseline += money[:, :, f].sum(axis=1)

    train_money = money.sum(axis=2)
    train_count = count.sum(axis=2)
    whitelist = (train_money > 0) & (train_count >= min_train_tokens)
    return total_test, per_fold, whitelist, baseline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-06-01")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--perms", type=int, default=30)
    ap.add_argument("--min-train-tokens", type=int, default=5)
    ap.add_argument("--min-strat-tokens", type=int, default=200)
    ap.add_argument("--cache", default="")
    ap.add_argument("--start-id", type=int, default=577468,
                    help="premier id >= --since (agregation serveur, evite un scan)")
    a = ap.parse_args()

    sb = create_client(os.environ["SUPABASE_URL"],
                       os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    # Le fetch coute ~800k lignes sur une table qui timeout facilement: on le
    # met en cache pour pouvoir iterer sur l'ALGO sans re-marteler la DB.
    cache = Path(a.cache) if a.cache else None
    print(f"[1/5] Chargement depuis {a.since} ...", flush=True)
    if cache and cache.exists():
        ded = [tuple(x) for x in json.loads(cache.read_text())]
        print(f"      {len(ded):,} lignes relues du cache {cache}")
    else:
        rows = fetch_all(sb, a.since, a.start_id)
        print(f"      {len(rows):,} lignes brutes")
        ded = dedup(rows)
        if cache:
            cache.write_text(json.dumps(ded))
            print(f"      cache ecrit -> {cache}")
    print(f"      {len(ded):,} apres dedup (strategie, KOL, token)")

    strats = sorted({d[0] for d in ded})
    kols = sorted({d[1] for d in ded})
    tokens = sorted({d[2] for d in ded})
    si = {s: i for i, s in enumerate(strats)}
    ki = {k: i for i, k in enumerate(kols)}
    ti = {t: i for i, t in enumerate(tokens)}
    print(f"      {len(strats)} strategies x {len(kols)} KOL x {len(tokens)} tokens")

    strat_idx = np.fromiter((si[d[0]] for d in ded), int, len(ded))
    kol_idx = np.fromiter((ki[d[1]] for d in ded), int, len(ded))
    tok_idx = np.fromiter((ti[d[2]] for d in ded), int, len(ded))
    times = np.fromiter((np.datetime64(d[3][:19]).astype("int64") for d in ded),
                        "int64", len(ded))
    pnl = np.fromiter((d[4] for d in ded), float, len(ded))

    print(f"[2/5] Decoupe walk-forward en {a.folds} blocs ...", flush=True)
    fold = build_folds(times, a.folds)
    for f in range(a.folds):
        m = fold == f
        print(f"      fold {f}: {m.sum():,} trades")

    ns, nk, nf = len(strats), len(kols), a.folds
    print("[3/5] Recherche exacte du meilleur sous-ensemble de KOL par fold ...",
          flush=True)
    total, per_fold, whitelist, baseline = walk_forward_money(
        strat_idx, kol_idx, fold, pnl, ns, nk, nf, a.min_train_tokens)
    # LA quantité qui décide: ce que la sélection de KOL AJOUTE par rapport à
    # « on prend tout ». Une whitelist qui rapporte gros parce que la stratégie
    # est bonne n'a rien prouvé sur l'axe KOL.
    uplift = total - baseline

    # Un bras qui ne trade presque jamais n'est pas exploitable: garde de volume.
    strat_tokens = np.zeros(ns)
    np.add.at(strat_tokens, strat_idx, 1.0)
    eligible = strat_tokens >= a.min_strat_tokens
    total_elig = np.where(eligible, total, -np.inf)
    uplift_elig = np.where(eligible, uplift, -np.inf)
    print(f"      {int(eligible.sum())}/{ns} strategies avec >= "
          f"{a.min_strat_tokens} tokens")

    print(f"[4/5] Controle par permutation ({a.perms} tirages, "
          f"pipeline COMPLET rejoue) ...", flush=True)
    rng = np.random.default_rng(20260811)
    h0_max, h0_tot = [], []
    for p in range(a.perms):
        # On casse le lien token -> KOL en gardant la structure par ailleurs:
        # chaque token garde ses trades, mais herite du KOL d'un autre token.
        perm = rng.permutation(len(tokens))
        fake_kol_of_token = np.empty(len(tokens), dtype=int)
        # KOL "canonique" de chaque token = celui de sa premiere occurrence.
        canon = np.full(len(tokens), -1, dtype=int)
        for t, k in zip(tok_idx, kol_idx):
            if canon[t] < 0:
                canon[t] = k
        fake_kol_of_token[:] = canon[perm]
        fake_kol_idx = fake_kol_of_token[tok_idx]
        tot_h0, _, _, base_h0 = walk_forward_money(strat_idx, fake_kol_idx, fold,
                                                   pnl, ns, nk, nf,
                                                   a.min_train_tokens)
        h0_max.append(np.where(eligible, tot_h0 - base_h0, -np.inf).max())
        h0_tot.append(np.where(eligible, tot_h0, -np.inf).max())
        if (p + 1) % 10 == 0:
            print(f"      ... {p + 1}/{a.perms}", flush=True)

    # pnl_pct est une FRACTION (0.5 = +50 %). À $100/trade, 1 unité = $100.
    D = 100.0
    jours = (times.max() - times.min()) / 86400.0 * (nf - 1) / nf

    h0_max = np.array(h0_max)
    plancher = float(np.quantile(h0_max, 0.95))
    best_uplift = float(uplift_elig.max())
    best_total = float(total_elig.max())

    print("\n[5/5] RESULTAT")
    print("=" * 78)
    print(f"  Fenetre de test: {jours:.0f} jours cumules sur {nf - 1} folds\n")
    print("  --- CE QUE LA SELECTION DE KOL AJOUTE (la seule question qui compte) ---")
    print(f"  meilleur uplift REEL (whitelist - tout prendre) : "
          f"{best_uplift * D:+,.0f} $  ({best_uplift * D / jours:+.2f} $/j)")
    print(f"  plancher H0 (p95 du max, {a.perms} permutations)  : "
          f"{plancher * D:+,.0f} $")
    print(f"  max H0 observe                                  : "
          f"{h0_max.max() * D:+,.0f} $")
    p_val = float((h0_max >= best_uplift).mean())
    if best_uplift <= plancher:
        print(f"\n  >> L'uplift REEL ne depasse PAS le plancher (p ~ {p_val:.3f}).")
        print("     Selectionner des KOL ne rapporte pas plus que tout prendre.")
        print("     La recherche est exhaustive: il n'y a rien a trouver sur cet axe.")
    else:
        print(f"\n  >> L'uplift REEL depasse le plancher (p ~ {p_val:.3f}).")

    h0_tot_a = np.array(h0_tot)
    plancher_tot = float(np.quantile(h0_tot_a, 0.95))
    p_tot = float((h0_tot_a >= best_total).mean())
    print("\n  --- ARGENT ABSOLU (ce qu'on gagnerait vraiment) ---")
    print(f"  meilleure config REELLE          : {best_total * D:+,.0f} $"
          f"   ({best_total * D / jours:+.2f} $/j)")
    print(f"  plancher H0 (p95 du max)         : {plancher_tot * D:+,.0f} $")
    print(f"  max H0 observe                   : {h0_tot_a.max() * D:+,.0f} $")
    marge = (best_total / plancher_tot - 1) * 100 if plancher_tot > 0 else float("nan")
    print(f"  >> {'DEPASSE' if best_total > plancher_tot else 'NE DEPASSE PAS'}"
          f" le plancher (p ~ {p_tot:.3f}, marge {marge:+.0f} % sur le p95)")

    # ⚠️ Le tri par UPLIFT favorise mécaniquement les stratégies catastrophiques
    # sans filtre (uplift +13 998 $ pour un total de +25 $). Le classement qui
    # répond à « laquelle rapporte le plus » est celui-ci.
    ordre_abs = np.argsort(-total_elig)[:15]
    print("\n  ### TOP 15 PAR ARGENT ABSOLU HORS ECHANTILLON (la reponse) ###")
    print(f"  {'strategie':<34}{'test$':>9}{'sans WL$':>9}{'$/j':>8}"
          f"{'nKOL':>6}{'folds+':>7}")
    print("  " + "-" * 76)
    for i in ordre_abs:
        if not np.isfinite(total_elig[i]):
            continue
        print(f"  {strats[i][:34]:<34}{total[i] * D:>+9.0f}{baseline[i] * D:>+9.0f}"
              f"{total[i] * D / jours:>+8.1f}{int(whitelist[i].sum()):>6}"
              f"{int((per_fold[i, 1:] > 0).sum()):>4}/{nf - 1}")

    order = np.argsort(-uplift_elig)[:15]
    print(f"\n  Top 15 par UPLIFT de la whitelist (hors echantillon, $ a mise $100):")
    print(f"  {'strategie':<34}{'uplift$':>9}{'total$':>9}{'sans WL$':>9}"
          f"{'nKOL':>6}{'folds+':>7}")
    print("  " + "-" * 76)
    for i in order:
        if not np.isfinite(uplift_elig[i]):
            continue
        print(f"  {strats[i][:34]:<34}{uplift[i] * D:>+9.0f}{total[i] * D:>+9.0f}"
              f"{baseline[i] * D:>+9.0f}{int(whitelist[i].sum()):>6}"
              f"{int((per_fold[i, 1:] > 0).sum()):>4}/{nf - 1}")

    best = int(np.argmax(total_elig))
    wl = [kols[j] for j in range(nk) if whitelist[best, j]]
    print(f"\n  Meilleure config par argent absolu: {strats[best]}")
    print(f"    test {total[best] * D:+,.0f} $  |  sans whitelist "
          f"{baseline[best] * D:+,.0f} $  |  uplift {uplift[best] * D:+,.0f} $")
    print(f"    whitelist ({len(wl)}/{nk} KOL): {', '.join(wl[:30])}"
          f"{' ...' if len(wl) > 30 else ''}")
    print(f"    folds de test positifs: "
          f"{int((per_fold[best, 1:] > 0).sum())}/{nf - 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
