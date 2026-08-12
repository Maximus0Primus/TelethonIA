"""v14e.93 (volet 2) - Piloter par le REGIME plutot que par le CLASSEMENT.

Le volet 1 (`regime_rotation_test.py`) teste "le meilleur d'hier sera le
meilleur demain". C'est la version FAIBLE de l'idee, et c'est celle que
ro = 0.019 a deja mise a mal.

Ici on teste la version FORTE, qui ne demande PAS que le classement persiste,
seulement qu'il existe un MECANISME:

    "quand le marche produit des runners, les TP larges paient;
     quand il n'en produit pas, ils saignent et les TP serres paient."

Ce mecanisme est deja documente dans le projet, en creux: TP50 touche sa cible
36 % du temps, TP200 seulement 5 %, et le cout du SL est ~-45 % quel que soit
son niveau (gap-through). Donc la rentabilite d'un TP large est une fonction
quasi mecanique du TAUX DE RUNNERS du marche. Si ce taux est persistant, il est
predictible, et la rotation devient exploitable.

TROIS CONDITIONS, testees separement (une seule qui casse suffit):

  C1 MECANISME   : regime(t) explique-t-il quelle famille paie en t ?
                   (correlation CONTEMPORAINE - pas encore exploitable)
  C2 PERSISTANCE : regime(t-1) predit-il regime(t) ?
                   (sans ca, on apprend le regime trop tard)
  C3 EXPLOITABLE : regle PRE-SPECIFIEE, walk-forward, sans aucun parametre
                   ajuste apres coup:
                     runners(t-1) > mediane historique -> famille TP LARGE
                     sinon                             -> famille TP SERRE
                   comparee a: toujours-serre, toujours-large, et 500 tirages
                   pile-ou-face (le vrai plancher: une regle qui alterne au
                   hasard capture deja une part de la variance).

Le contraste est a DEUX familles pre-specifiees, jamais un maximum sur une
grille: c'est la seule forme lisible d'apres experiment_register (les maxima
sur 5 000 cellules sont garantis sous H0).

Usage:
    python scripts/regime_signal_test.py --cache <chemin.pkl> [--gran semaine]
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

SCRAPER = Path(__file__).resolve().parent.parent / "scraper"
sys.path.insert(0, str(SCRAPER))

STAKE = 100.0


def family_of(name: str):
    tp = re.search(r"TP(\d+)", name.upper())
    sl = re.search(r"SL(\d+)", name.upper())
    if not tp or not sl:
        return None, None, None
    return int(tp.group(1)), int(sl.group(1)), True


def period_key(dt: datetime, gran: str) -> str:
    if gran == "mois":
        return f"{dt.year}-{dt.month:02d}"
    if gran == "jour":
        return dt.strftime("%Y-%m-%d")
    iso = dt.isocalendar()
    if gran == "semaine":
        return f"{iso[0]}-W{iso[1]:02d}"
    return f"{iso[0]}-Q{iso[1] // 2:02d}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--gran", default="semaine")
    ap.add_argument("--min-n", type=int, default=15)
    ap.add_argument("--perms", type=int, default=500)
    a = ap.parse_args()

    # Cache LOCAL ecrit par regime_rotation_test.py (jamais une source externe).
    ded = pickle.loads(Path(a.cache).read_bytes())
    try:
        from strategies import _is_artifact_family, _DEFAULT_DEPRECATED
    except Exception:
        _is_artifact_family, _DEFAULT_DEPRECATED = (lambda _n: False), set()
    ded = [d for d in ded
           if not _is_artifact_family(d[0]) and d[0] not in _DEFAULT_DEPRECATED]

    # --- par (strategie, periode): argent, n, et distribution des pnl --------
    money = defaultdict(float)
    cnt = defaultdict(int)
    pnls = defaultdict(list)
    for s, _tok, ts, pnl in ded:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
        p = period_key(dt, a.gran)
        money[(s, p)] += pnl * STAKE
        cnt[(s, p)] += 1
        pnls[(s, p)].append(pnl)
    periods = sorted({k[1] for k in money})
    strats = sorted({k[0] for k in money})

    # --- REGIME: mesure de MARCHE, pas de strategie -------------------------
    # Reference = le bras le plus volumineux present sur TOUTES les periodes.
    # On lit sur SES tokens le taux de runners: c'est l'observable qui decide
    # mecaniquement si un TP large peut etre atteint.
    # ⚠️ PIEGE (attrape le 12/08): une sonde a TP serre ne peut PAS voir un
    # runner. `BE15_TP50_SL30` sort a +50 %, donc sa part de tokens a >= +100 %
    # ne mesure pas le marche, elle mesure les gaps d'execution (0-3 %). La
    # sonde doit etre le bras le plus LARGE possible: seul lui laisse courir
    # assez longtemps pour que le taux de runners du marche soit observable.
    # Couverture >= 70 % des periodes (a la journee, aucun bras n'est present
    # tous les jours: certains jours le flux est trop mince).
    full = [s for s in strats
            if sum(cnt.get((s, p), 0) >= a.min_n for p in periods)
            >= 0.7 * len(periods)]
    if not full:
        print("Aucun bras couvrant 70 % des periodes.")
        return 1
    larges = [s for s in full
              if (lambda t: t[2] and t[0] >= 200)(family_of(s))]
    pool = larges or full
    ref = max(pool, key=lambda s: sum(cnt[(s, p)] for p in periods))
    tous = np.concatenate([np.array(pnls[(ref, p)]) for p in periods])
    print(f"Sonde de regime = bras le plus LARGE present partout ; "
          f"part globale a >= +100 % : {(tous >= 1.0).mean():.1%}")
    print(f"Reference de regime : {ref}  "
          f"({sum(cnt[(ref, p)] for p in periods)} tokens, {len(periods)} periodes)")

    runners, med = [], []
    for p in periods:
        v = np.array(pnls[(ref, p)])
        if len(v) < a.min_n:            # sonde indisponible cette periode
            runners.append(np.nan)
            med.append(np.nan)
            continue
        runners.append(float((v >= 1.0).mean()))     # part de tokens a >= +100 %
        med.append(float(np.median(v)))
    runners, med = np.array(runners), np.array(med)

    # --- familles contrastees, PRE-SPECIFIEES -------------------------------
    def fam_money(pred):
        out = np.full(len(periods), np.nan)
        for j, p in enumerate(periods):
            vals = [money[(s, p)] for s in strats
                    if cnt.get((s, p), 0) >= a.min_n
                    and (lambda t: t[2] and pred(t[0], t[1]))(family_of(s))]
            if vals:
                out[j] = float(np.mean(vals))
            del vals
        return out

    large = fam_money(lambda tp, sl: tp >= 200)
    serre = fam_money(lambda tp, sl: tp <= 80 and sl <= 30)
    ok = ~np.isnan(large) & ~np.isnan(serre) & ~np.isnan(runners)
    ecart = large - serre

    print("\nperiode   runners  med_pnl    TP>=200    TP<=80/SL<=30    ecart")
    for j, p in enumerate(periods):
        if ok[j]:
            print(f"{p}  {runners[j]:6.1%}  {med[j]:+7.2%}  "
                  f"${large[j]:+9,.0f}  ${serre[j]:+9,.0f}  ${ecart[j]:+9,.0f}")

    # --- CO-MOUVEMENT: y a-t-il seulement "un bras qui gagne pendant que les
    # autres perdent" ? Tous les bras tradent LES MEMES tokens; s'ils montent
    # et descendent ensemble, aucune rotation ne peut sauver une mauvaise
    # periode, et l'oracle n'est qu'un maximum de bruit.
    part_pos = []
    for p in periods:
        v = [money[(s, p)] for s in strats if cnt.get((s, p), 0) >= a.min_n]
        if v:
            part_pos.append(float(np.mean(np.array(v) > 0)))
    part_pos = np.array(part_pos)
    print(f"\n[CO-MOUVEMENT] part des bras GAGNANTS par periode : "
          f"min {part_pos.min():.0%}  median {np.median(part_pos):.0%}  "
          f"max {part_pos.max():.0%}")
    print(f"     periodes ou > 90 % des bras perdent : "
          f"{int((part_pos < 0.10).sum())}/{len(part_pos)}  |  "
          f"ou > 90 % gagnent : {int((part_pos > 0.90).sum())}/{len(part_pos)}")

    # --- C1: MECANISME (contemporain) ---------------------------------------
    r1 = spearmanr(runners[ok], ecart[ok])
    print(f"\n[C1] MECANISME  runners(t) vs (TP large - TP serre)(t) : "
          f"rho = {r1.statistic:+.3f}  p = {r1.pvalue:.3f}  (n={int(ok.sum())})")

    # --- C2: PERSISTANCE ----------------------------------------------------
    pr = ~np.isnan(runners[:-1]) & ~np.isnan(runners[1:])
    pm = ~np.isnan(med[:-1]) & ~np.isnan(med[1:])
    r2 = spearmanr(runners[:-1][pr], runners[1:][pr])
    r2m = spearmanr(med[:-1][pm], med[1:][pm])
    print(f"[C2] PERSISTANCE runners(t-1) -> runners(t)             : "
          f"rho = {r2.statistic:+.3f}  p = {r2.pvalue:.3f}  (n={int(pr.sum())})")
    print(f"     idem sur la mediane de marche                      : "
          f"rho = {r2m.statistic:+.3f}  p = {r2m.pvalue:.3f}")

    # --- C3: regle PRE-SPECIFIEE, walk-forward ------------------------------
    idx = np.flatnonzero(ok)
    gains, choix, dec_t = [], [], []
    for t in idx:
        if t == 0:
            continue
        hist = runners[:t]                      # strictement le passe
        # ⚠️ NaN-SAFE. A la journee, la sonde manque certains jours: un seul NaN
        # dans `hist` rend np.median NaN, `x > NaN` est False, et la regle
        # degenere silencieusement en "toujours serre" (0 x large sur 109
        # decisions le 12/08). Une regle qui ne choisit jamais n'est pas une
        # regle: on saute la decision au lieu de la fausser.
        hist = hist[~np.isnan(hist)]
        if len(hist) < 2 or np.isnan(runners[t - 1]):
            continue
        pick_large = runners[t - 1] > np.median(hist)
        gains.append(large[t] if pick_large else serre[t])
        choix.append("large" if pick_large else "serre")
        dec_t.append(t)
    gains = np.array(gains)
    tot = float(np.nansum(gains))
    n_dec = len(gains)
    # Les comparateurs doivent porter sur EXACTEMENT les memes periodes que les
    # decisions, sinon on compare des paniers differents.
    tj_serre = float(np.nansum(serre[dec_t]))
    tj_large = float(np.nansum(large[dec_t]))

    rng = np.random.default_rng(20260812)
    null = []
    for _ in range(a.perms):
        g = [large[t] if rng.random() < 0.5 else serre[t] for t in dec_t]
        null.append(float(np.nansum(g)))
    null = np.array(null)

    print(f"\n[C3] ROTATION CONDITIONNELLE walk-forward ({n_dec} decisions)")
    print(f"     regle regime (pre-specifiee)  : ${tot:+,.0f}   "
          f"({choix.count('large')} x large, {choix.count('serre')} x serre)")
    print(f"     toujours TP serre             : ${tj_serre:+,.0f}")
    print(f"     toujours TP large             : ${tj_large:+,.0f}")
    print(f"     pile-ou-face ({a.perms} tirages)   : moyen ${null.mean():+,.0f}  "
          f"p95 ${np.percentile(null, 95):+,.0f}  max ${null.max():+,.0f}")
    verdict = ("DEPASSE le hasard" if tot > np.percentile(null, 95)
               else "ne depasse PAS le hasard")
    print(f"     -> {verdict}; et {'BAT' if tot > tj_serre else 'NE BAT PAS'} "
          f"la meilleure famille fixe")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
