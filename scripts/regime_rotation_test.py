"""v14e.93 - La rotation de strategie par regime de marche est-elle exploitable ?

QUESTION (user, 12/08): "les meilleures strategies changent selon les moments;
si on detectait ces ruptures et qu'on adaptait la strategie a chaque periode,
on gagnerait beaucoup plus".

L'idee se decompose en TROIS conditions qui doivent TOUTES tenir. Ce script les
teste separement, parce qu'une seule qui casse suffit a tuer la strategie de
rotation - et on veut savoir LAQUELLE casse.

  A. Y a-t-il de l'argent a prendre ?  (borne haute)
     Rotation PARFAITE (oracle, choisit avec le futur) vs meilleure strategie
     FIXE. Si l'ecart est faible, l'idee est morte meme avec un predicteur
     parfait: inutile d'aller plus loin.

  B. La selection retardee marche-t-elle ?  (realisable)
     "meilleur(s) de la periode t-1, trade en t", walk-forward, a 3
     granularites (semaine / quinzaine / mois), au niveau STRATEGIE et au
     niveau FAMILLE (regle projet: decider au niveau famille).
     Controle: 200 tirages ALEATOIRES de K strategies -> distribution sous H0.
     Si la rotation ne depasse pas le p95 du hasard, elle n'a aucune skill.

  C. Le REFRAME - piloter par le REGIME, pas par le classement.
     Une rotation par classement suppose que "qui a gagne gagnera". Un pilotage
     par regime suppose seulement un MECANISME: "quand le marche est X, la
     famille Y paie". Deux sous-conditions, testees separement:
        C1. MECANISME  : regime(t) explique-t-il quelle famille paie en t ?
        C2. PERSISTANCE: regime(t-1) predit-il regime(t) ?
     C1 sans C2 = mecanisme reel mais inexploitable (on apprend le regime trop
     tard). C2 sans C1 = rien a exploiter. Il faut les deux, et le produit se
     mesure par une rotation conditionnelle walk-forward (C3).

METRIQUE. A mise plafonnee (~$100/token, liquidite memecoin), l'argent d'un bras
sur une periode est `n x EV` = somme des pnl_pct de ses tokens. C'est la seule
quantite classee ici (regle mean_vs_median_ranking_rule + plafond de capacite).

Usage:
    python scripts/regime_rotation_test.py [--since 2026-04-13] [--min-n 15]
                                           [--perms 200] [--cache chemin.pkl]
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRAPER = Path(__file__).resolve().parent.parent / "scraper"
sys.path.insert(0, str(SCRAPER))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(SCRAPER / ".env")
from supabase import create_client  # noqa: E402

PAGE = 1000
PNL_CAP = 20.0          # garde-fou exit_price corrompu (regle projet)
STAKE = 100.0           # $ par token, mise plafonnee


# ---------------------------------------------------------------------------
# Chargement (keyset sur la PK: cf. supabase_disk_io_budget_jul5, timeout 57014)
# ---------------------------------------------------------------------------
def fetch_all(sb, start_id: int) -> list[dict]:
    cols = "id,strategy,token_address,pnl_pct,created_at,status,chain"
    last_id, out, seen = int(start_id) - 1, [], 0
    while True:
        rows = None
        for attempt in range(6):
            try:
                rows = (sb.table("paper_trades").select(cols)
                        .gt("id", last_id).order("id").limit(PAGE)
                        .execute().data)
                break
            except Exception as e:
                print(f"    [retry {attempt+1}/6] id>{last_id}: {e}", flush=True)
                time.sleep(5 * (attempt + 1))
        if not rows:
            break
        last_id = rows[-1]["id"]
        seen += len(rows)
        for r in rows:
            if (r.get("chain") == "solana" and r.get("status") != "open"
                    and r.get("pnl_pct") is not None
                    and float(r["pnl_pct"]) <= PNL_CAP):
                out.append((r["strategy"], r["token_address"],
                            r["created_at"], float(r["pnl_pct"])))
        if seen % 200_000 < PAGE:
            print(f"    ... {seen:,} lues / {len(out):,} retenues", flush=True)
        if len(rows) < PAGE:
            break
    return out


def dedup(rows):
    """Un seul trade par (strategie, token): le PREMIER (regle projet)."""
    best = {}
    for s, tok, ts, pnl in rows:
        key = (s, tok)
        cur = best.get(key)
        if cur is None or ts < cur[0]:
            best[key] = (ts, pnl)
    return [(k[0], k[1], v[0], v[1]) for k, v in best.items()]


# ---------------------------------------------------------------------------
# Familles: c'est le niveau auquel le projet a le droit de decider (ro=0.019
# au niveau config, mais l'ordre des familles tient d'une periode a l'autre).
# ---------------------------------------------------------------------------
def family_of(name: str) -> str | None:
    tp = re.search(r"TP(\d+)", name.upper())
    sl = re.search(r"SL(\d+)", name.upper())
    if not tp or not sl:
        return None
    tp, sl = int(tp.group(1)), int(sl.group(1))
    tb = ("TP<=50" if tp <= 50 else "TP51-80" if tp <= 80 else
          "TP81-120" if tp <= 120 else "TP121-199" if tp < 200 else "TP>=200")
    sb_ = ("SL<=20" if sl <= 20 else "SL21-30" if sl <= 30 else
           "SL31-40" if sl <= 40 else "SL>40")
    return f"{tb}/{sb_}"


def period_key(dt: datetime, gran: str) -> str:
    if gran == "mois":
        return f"{dt.year}-{dt.month:02d}"
    iso = dt.isocalendar()
    if gran == "semaine":
        return f"{iso[0]}-W{iso[1]:02d}"
    return f"{iso[0]}-Q{iso[1] // 2:02d}"          # quinzaine


# ---------------------------------------------------------------------------
# A + B: oracle, fixe, rotation lag-1, et son plancher aleatoire
# ---------------------------------------------------------------------------
def build_matrix(ded, gran, min_n):
    """money[s, t] et count[s, t] pour une granularite donnee."""
    money, count = defaultdict(float), defaultdict(int)
    for s, _tok, ts, pnl in ded:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
        p = period_key(dt, gran)
        money[(s, p)] += pnl * STAKE
        count[(s, p)] += 1
    strats = sorted({k[0] for k in money})
    periods = sorted({k[1] for k in money})
    si = {s: i for i, s in enumerate(strats)}
    pi = {p: i for i, p in enumerate(periods)}
    M = np.full((len(strats), len(periods)), np.nan)
    N = np.zeros((len(strats), len(periods)))
    for (s, p), v in money.items():
        M[si[s], pi[p]] = v
        N[si[s], pi[p]] = count[(s, p)]
    M[N < min_n] = np.nan          # bras non tradeable cette periode
    return strats, periods, M, N


def rotation(M, K, rng=None):
    """Argent cumule de la regle 'top-K de t-1, trade en t' (moyenne des K).

    rng non nul => selection ALEATOIRE parmi les eligibles (null H0).
    """
    total, per_period = 0.0, []
    for t in range(1, M.shape[1]):
        ok = ~np.isnan(M[:, t - 1]) & ~np.isnan(M[:, t])
        idx = np.flatnonzero(ok)
        if len(idx) < K:
            per_period.append(np.nan)
            continue
        if rng is None:
            pick = idx[np.argsort(-M[idx, t - 1])[:K]]
        else:
            pick = rng.choice(idx, size=K, replace=False)
        g = float(np.mean(M[pick, t]))
        total += g
        per_period.append(g)
    return total, per_period


def oracle_and_fixed(M):
    """Borne haute (rotation parfaite) et meilleure strategie FIXE in-sample."""
    T = M.shape[1]
    orac = float(np.nansum([np.nanmax(M[:, t]) for t in range(1, T)
                            if np.any(~np.isnan(M[:, t]))]))
    # fixe: seulement les bras presents sur TOUTES les periodes (comparable)
    full = ~np.isnan(M[:, 1:]).any(axis=1)
    fixed_tot = M[full, 1:].sum(axis=1) if full.any() else np.array([0.0])
    moyen = float(np.nanmean(np.nansum(M[:, 1:], axis=1)))
    return orac, float(fixed_tot.max()), float(np.median(fixed_tot)), moyen, int(full.sum())


def oracle_null(M, perms, rng):
    """Oracle sous H0 'aucune variation de regime'.

    ⚠️ L'oracle est un MAXIMUM sur ~440 bras bruites a queue epaisse: il est
    positif meme sans le moindre effet de periode (cf. le +23.19 % du 11/08 qui
    etait SOUS son plancher). On permute donc les etiquettes de periode
    INDEPENDAMMENT par stratégie: chaque bras garde sa distribution de gains,
    mais toute structure temporelle COMMUNE est detruite. Ce qui reste de
    l'oracle sous H0 est du pur bruit de maximum.
    """
    out = []
    for _ in range(perms):
        P = M.copy()
        for i in range(P.shape[0]):
            ok = np.flatnonzero(~np.isnan(P[i]))
            if len(ok) > 1:
                P[i, ok] = P[i, rng.permutation(ok)]
        out.append(float(np.nansum([np.nanmax(P[:, t]) for t in range(1, P.shape[1])
                                    if np.any(~np.isnan(P[:, t]))])))
    return np.array(out)


def decile_decomposition(M):
    """Argent moyen en t selon le DECILE d'appartenance en t-1.

    C'est le test qui separe les deux moities de la loi du projet: si le bas du
    classement persiste (bras nuls restent nuls) mais que le haut ne persiste
    pas, alors ro global > 0 ET top-K perd — ce qui parait contradictoire ne
    l'est pas.
    """
    acc = defaultdict(list)
    for t in range(1, M.shape[1]):
        ok = np.flatnonzero(~np.isnan(M[:, t - 1]) & ~np.isnan(M[:, t]))
        if len(ok) < 20:
            continue
        order = ok[np.argsort(-M[ok, t - 1])]
        for d in range(10):
            lo, hi = int(d * len(order) / 10), int((d + 1) * len(order) / 10)
            if hi > lo:
                acc[d].append(float(np.mean(M[order[lo:hi], t])))
    return {d: float(np.mean(v)) for d, v in sorted(acc.items())}


def exclusion_rule(M, perms, rng):
    """Regle d'EXCLUSION: trader la moyenne des bras qui ont gagne en t-1.

    C'est l'analogue au niveau STRATEGIE de la whitelist KOL (`argent passe > 0`)
    qui, elle, a fonctionne (+$1 410) la ou toutes les selections d'elite
    perdaient. On compare a la moyenne de TOUS les bras, et a un sous-ensemble
    ALEATOIRE de meme taille (le plancher honnete).
    """
    real, base, null = 0.0, 0.0, np.zeros(perms)
    for t in range(1, M.shape[1]):
        ok = np.flatnonzero(~np.isnan(M[:, t - 1]) & ~np.isnan(M[:, t]))
        if len(ok) < 20:
            continue
        keep = ok[M[ok, t - 1] > 0]
        base += float(np.mean(M[ok, t]))
        if len(keep) == 0:
            continue
        real += float(np.mean(M[keep, t]))
        for i in range(perms):
            pick = rng.choice(ok, size=len(keep), replace=False)
            null[i] += float(np.mean(M[pick, t]))
    return real, base, null


def robustness_table(M, fam_rows):
    """[D] ADAPTATIF vs FIXE, a armes egales, et note sur la ROBUSTESSE.

    Question user (12/08): "s'adapter donnerait plus de robustesse, car une
    strategie excellente sur 4 mois d'affilee, ca n'existe pas".

    Le [A] comparait l'adaptatif a la meilleure fixe choisie AVEC le futur —
    comparateur malhonnete, aucun systeme n'y a acces. Ici toutes les regles
    sont deployables en t: elles ne lisent que les periodes < t.

    Et on ne note plus seulement le TOTAL: une regle plus robuste peut perdre
    en moyenne tout en encaissant mieux les mauvaises periodes. On sort donc
    aussi l'ecart-type, la pire periode et la part de periodes positives.
    """
    T = M.shape[1]

    def elig(t):
        return np.flatnonzero(~np.isnan(M[:, t - 1]) & ~np.isnan(M[:, t]))

    def serie(sel):
        out = np.full(T, np.nan)
        for t in range(1, T):
            idx = sel(t)
            if idx is None or len(idx) == 0:
                continue
            out[t] = float(np.nanmean(M[idx, t]))
        return out

    # FIXE TRICHE: le bras au meilleur total sur TOUTE la periode (in-sample).
    full = np.flatnonzero(~np.isnan(M[:, 1:]).any(axis=1))
    triche = full[np.argmax(M[full, 1:].sum(axis=1))] if len(full) else None

    # FIXE HONNETE: choisi apres la 1re periode, puis JAMAIS retouche.
    e1 = elig(1)
    honnete = e1[np.argmax(M[e1, 1])] if len(e1) else None

    def best_so_far(t):
        """Le meilleur CUMULE depuis le debut (fenetre expansive).

        Version intermediaire entre le fixe et la rotation lag-1: elle s'adapte,
        mais en moyennant tout l'historique au lieu de ne lire que t-1. C'est la
        forme la plus defendable de l'idee de l'user.
        """
        idx = elig(t)
        if len(idx) == 0:
            return None
        cum = np.nansum(M[np.ix_(idx, range(max(1, t - 99), t))], axis=1)
        return np.array([idx[int(np.argmax(cum))]])

    regles = {
        "fixe TRICHE (in-sample)": serie(lambda t: np.array([triche])) if triche is not None else np.full(T, np.nan),
        "fixe HONNETE (choisi en t=1)": serie(lambda t: np.array([honnete])) if honnete is not None else np.full(T, np.nan),
        "meilleur CUMULE (expansif)": serie(best_so_far),
        "rotation top-1 de t-1": serie(lambda t: (lambda i: i[np.argsort(-M[i, t - 1])[:1]] if len(i) else None)(elig(t))),
        "rotation top-10 de t-1": serie(lambda t: (lambda i: i[np.argsort(-M[i, t - 1])[:10]] if len(i) >= 10 else None)(elig(t))),
        "PANIER exclusion (>0 en t-1)": serie(lambda t: (lambda i: i[M[i, t - 1] > 0] if len(i) else None)(elig(t))),
        "PANIER tous les bras": serie(lambda t: elig(t)),
        "PANIER famille TP<=80/SL<=30": serie(lambda t: np.array([r for r in fam_rows if not np.isnan(M[r, t])])) if fam_rows else np.full(T, np.nan),
    }
    print(f"\n[D] ADAPTATIF vs FIXE, a armes egales (seules les periodes < t sont lues)")
    print(f"    {'regle':<32}{'total':>10}{'moy/per':>10}{'ecart-t':>9}"
          f"{'pire':>10}{'% per +':>9}")
    for nom, s in regles.items():
        v = s[~np.isnan(s)]
        if len(v) == 0:
            continue
        print(f"    {nom:<32}${v.sum():>+9,.0f}${v.mean():>+9,.0f}"
              f"{v.std():>9,.0f}${v.min():>+9,.0f}{100 * (v > 0).mean():>8.0f}%")
    return regles


def single_arm_policies(M, fam_of_s, T_switch_costs=True):
    """[E] UN SEUL BRAS EN LIVE — la vraie contrainte (user, 12/08).

    Le panier du [D] n'est PAS deployable: en live il n'y a qu'une strategie,
    donc "exclure les mauvais bras" ne veut rien dire. La seule question qui
    reste est: **quelle politique de choix d'un bras unique**, sachant que
      - le classement sur fenetre courte est du bruit ([B], ro = 0.019),
      - mais une fixe choisie une fois pour toutes est fragile ([D]).

    On compare des politiques qui produisent TOUTES un seul bras par periode et
    ne lisent que le passe. On note aussi le NOMBRE DE CHANGEMENTS: chaque
    changement est un cout reel (re-cablage des 4 endroits, perte de
    comparabilite, et en live un risque d'erreur).

    Idee testee au passage: le meilleur bras d'une famille est probablement le
    plus CHANCEUX de la famille. Prendre le bras MEDIAN de la meilleure famille
    devrait etre plus robuste que prendre son maximum.
    """
    T = M.shape[1]
    fams = sorted({f for f in fam_of_s if f})
    rows_of = {f: np.array([i for i, ff in enumerate(fam_of_s) if ff == f])
               for f in fams}

    def elig(t):
        return np.flatnonzero(~np.isnan(M[:, t]))

    def cum(idx, t):
        """Argent cumule sur TOUT le passe (periodes 0..t-1)."""
        return np.nansum(M[np.ix_(idx, range(0, t))], axis=1)

    def best_family(t):
        best, bf = -np.inf, None
        for f in fams:
            r = rows_of[f]
            v = np.nanmean(cum(r, t)) if len(r) else np.nan
            if not np.isnan(v) and v > best:
                best, bf = v, f
        return bf

    def pol_fixe(t, state):
        if state.get("row") is None:
            e = elig(1)
            state["row"] = int(e[np.argmax(M[e, 1])]) if len(e) else None
        return state["row"]

    def pol_cum_top(t, state):
        e = elig(t)
        return int(e[np.argmax(cum(e, t))]) if len(e) else None

    def pol_cum_top_every(k):
        def f(t, state):
            if t % k == 1 or state.get("row") is None:
                state["row"] = pol_cum_top(t, {})
            return state["row"]
        return f

    def pol_fam_median(t, state):
        bf = best_family(t)
        if bf is None:
            return None
        r = np.intersect1d(rows_of[bf], elig(t))
        if len(r) == 0:
            return None
        c = cum(r, t)
        return int(r[np.argsort(c)[len(c) // 2]])       # le bras TYPE, pas le max

    def pol_fam_top(t, state):
        bf = best_family(t)
        if bf is None:
            return None
        r = np.intersect1d(rows_of[bf], elig(t))
        return int(r[np.argmax(cum(r, t))]) if len(r) else None

    def pol_lag1(t, state):
        e = np.flatnonzero(~np.isnan(M[:, t - 1]) & ~np.isnan(M[:, t]))
        return int(e[np.argmax(M[e, t - 1])]) if len(e) else None

    politiques = {
        "fixe: choisi en t=1, jamais change": pol_fixe,
        "cumule: top-1 sur TOUT le passe": pol_cum_top,
        "cumule: re-choisi 1 periode sur 2": pol_cum_top_every(2),
        "cumule: re-choisi 1 periode sur 4": pol_cum_top_every(4),
        "meilleure FAMILLE -> son bras MEDIAN": pol_fam_median,
        "meilleure FAMILLE -> son bras TOP": pol_fam_top,
        "rotation: top-1 de la periode t-1": pol_lag1,
    }
    print("\n[E] UN SEUL BRAS EN LIVE — politiques deployables")
    print(f"    {'politique':<38}{'total':>10}{'ecart-t':>9}{'pire':>10}"
          f"{'% per +':>9}{'chgts':>7}")
    out = {}
    for nom, pol in politiques.items():
        state, vals, prev, chg = {}, [], None, 0
        for t in range(1, T):
            r = pol(t, state)
            if r is None or np.isnan(M[r, t]):
                continue
            if prev is not None and r != prev:
                chg += 1
            prev = r
            vals.append(float(M[r, t]))
        v = np.array(vals)
        if len(v) == 0:
            continue
        out[nom] = v
        print(f"    {nom:<38}${v.sum():>+9,.0f}{v.std():>9,.0f}${v.min():>+9,.0f}"
              f"{100 * (v > 0).mean():>8.0f}%{chg:>7}")
    return out


def within_family(M, fam_rows, strats, perms, rng):
    """[F] LA FAMILLE EST FIXEE — quand changer de bras, et pour lequel ?

    Question user (12/08), une fois la famille actee: "comment je sais quand
    changer de bras et par lequel, pour maximiser les gains ?"

    L'ordre des questions compte, et il n'est pas celui qu'on croit:
      F1. EST-CE QUE CA CHANGE QUELQUE CHOSE ? Les bras d'une meme famille
          tradent LES MEMES tokens avec des sorties voisines. Si l'ecart entre
          eux est petit devant le niveau commun, la reponse a "lequel" est
          "peu importe" et il n'y a pas de question F3.
      F2. Y A-T-IL DE QUOI CHOISIR ? Oracle INTRA-famille (choisit avec le
          futur) contre son plancher de permutation. S'il ne depasse pas, aucune
          regle de changement ne peut aider — inutile d'en chercher une.
      F3. Seulement si F1 et F2 passent: quelle regle de changement.
    """
    F = M[fam_rows, :]
    noms = [strats[i] for i in fam_rows]
    T = F.shape[1]
    print(f"\n[F] FAMILLE FIXEE — {len(fam_rows)} bras, quand/comment changer ?")

    # --- F1: l'ecart intra-famille est-il grand devant le niveau commun ? ----
    ecarts, niveaux, spreads = [], [], []
    for t in range(T):
        v = F[:, t][~np.isnan(F[:, t])]
        if len(v) < 3:
            continue
        ecarts.append(float(v.std()))
        niveaux.append(float(abs(v.mean())))
        spreads.append(float(v.max() - v.min()))
    print(f"    [F1] par periode: niveau commun |moy| ${np.mean(niveaux):,.0f}  "
          f"| dispersion INTRA-famille ecart-type ${np.mean(ecarts):,.0f}  "
          f"| meilleur-pire ${np.mean(spreads):,.0f}")

    # Correlation moyenne entre bras: s'ils bougent ensemble, choisir est vain.
    ok_rows = [i for i in range(F.shape[0]) if np.sum(~np.isnan(F[i])) >= 4]
    cors = []
    for i in range(len(ok_rows)):
        for j in range(i + 1, len(ok_rows)):
            a, b = F[ok_rows[i]], F[ok_rows[j]]
            m = ~np.isnan(a) & ~np.isnan(b)
            if m.sum() >= 4 and a[m].std() > 0 and b[m].std() > 0:
                cors.append(float(np.corrcoef(a[m], b[m])[0, 1]))
    if cors:
        print(f"         correlation moyenne entre 2 bras de la famille: "
              f"{np.mean(cors):+.3f}  (1.0 = choisir ne sert a rien)")

    # --- F2: borne haute INTRA-famille, avec son plancher --------------------
    orac = float(np.nansum([np.nanmax(F[:, t]) for t in range(1, T)
                            if np.any(~np.isnan(F[:, t]))]))
    null = []
    for _ in range(perms):
        P = F.copy()
        for i in range(P.shape[0]):
            o = np.flatnonzero(~np.isnan(P[i]))
            if len(o) > 1:
                P[i, o] = P[i, rng.permutation(o)]
        null.append(float(np.nansum([np.nanmax(P[:, t]) for t in range(1, T)
                                     if np.any(~np.isnan(P[:, t]))])))
    null = np.array(null)
    moyen = float(np.nansum([np.nanmean(F[:, t]) for t in range(1, T)
                             if np.any(~np.isnan(F[:, t]))]))
    print(f"    [F2] oracle INTRA-famille ${orac:+,.0f}  |  H0 p95 "
          f"${np.percentile(null, 95):+,.0f}  |  bras moyen ${moyen:+,.0f}")
    print(f"         -> {'il y a de quoi choisir' if orac > np.percentile(null, 95) else 'RIEN a choisir: meme un oracle ne depasse pas le hasard'}")

    # --- F3: les regles de changement ---------------------------------------
    def elig(t):
        return np.flatnonzero(~np.isnan(F[:, t]))

    def cum(idx, t):
        return np.nansum(F[np.ix_(idx, range(0, t))], axis=1)

    def run(pol):
        state, vals, prev, chg = {}, [], None, 0
        for t in range(1, T):
            r = pol(t, state)
            if r is None or np.isnan(F[r, t]):
                continue
            if prev is not None and r != prev:
                chg += 1
            prev = r
            vals.append(float(F[r, t]))
        return np.array(vals), chg

    def p_hold(t, state):
        if state.get("row") is None:
            e = elig(1)
            state["row"] = int(e[np.argmax(F[e, 1])]) if len(e) else None
        return state["row"]

    def p_cum(t, state):
        e = elig(t)
        return int(e[np.argmax(cum(e, t))]) if len(e) else None

    def p_lag1(t, state):
        e = np.flatnonzero(~np.isnan(F[:, t - 1]) & ~np.isnan(F[:, t]))
        return int(e[np.argmax(F[e, t - 1])]) if len(e) else None

    def p_seuil(n_bad):
        """Ne changer QUE si le bras courant est sous la mediane de la famille
        n_bad periodes de suite. C'est la reponse directe a "quand changer".
        """
        def f(t, state):
            e = elig(t)
            if len(e) == 0:
                return None
            r = state.get("row")
            if r is None or np.isnan(F[r, t]):
                r = int(e[np.argmax(cum(e, t))])
                state["row"], state["bad"] = r, 0
                return r
            prev_ok = ~np.isnan(F[:, t - 1])
            if prev_ok.sum() >= 3 and not np.isnan(F[r, t - 1]):
                med = float(np.nanmedian(F[prev_ok, t - 1]))
                state["bad"] = state.get("bad", 0) + (1 if F[r, t - 1] < med else -1)
                state["bad"] = max(0, state["bad"])
            if state["bad"] >= n_bad:
                r = int(e[np.argmax(cum(e, t))])
                state["bad"] = 0
            state["row"] = r
            return r
        return f

    def p_alea(t, state):
        e = elig(t)
        return int(rng.choice(e)) if len(e) else None

    regles = {
        "garder le meme bras, toujours": p_hold,
        "top-1 sur tout l'historique": p_cum,
        "top-1 de la periode precedente": p_lag1,
        "changer si sous la mediane 1 fois": p_seuil(1),
        "changer si sous la mediane 2x de suite": p_seuil(2),
        "changer si sous la mediane 3x de suite": p_seuil(3),
        "bras AU HASARD chaque periode": p_alea,
    }
    print(f"    [F3] {'regle de changement':<40}{'total':>10}{'ecart-t':>9}"
          f"{'pire':>10}{'% per +':>9}{'chgts':>7}")
    for nom, pol in regles.items():
        v, chg = run(pol)
        if len(v) == 0:
            continue
        print(f"         {nom:<40}${v.sum():>+9,.0f}{v.std():>9,.0f}"
              f"${v.min():>+9,.0f}{100 * (v > 0).mean():>8.0f}%{chg:>7}")
    return noms


def rank_persistence(M):
    """Spearman entre periodes consecutives (le coeur du probleme)."""
    from scipy.stats import spearmanr
    out = []
    for t in range(1, M.shape[1]):
        ok = ~np.isnan(M[:, t - 1]) & ~np.isnan(M[:, t])
        if ok.sum() >= 10:
            r = spearmanr(M[ok, t - 1], M[ok, t]).statistic
            out.append((t, int(ok.sum()), float(r)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-04-13")
    ap.add_argument("--start-id", type=int, default=158262)
    ap.add_argument("--min-n", type=int, default=15)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--cache", default="")
    a = ap.parse_args()

    cache = Path(a.cache) if a.cache else None
    if cache and cache.exists():
        # Cache LOCAL ecrit par ce script lui-meme (jamais une source externe):
        # le fetch coute ~1.2 M lignes sur une table qui timeout facilement, on
        # le met de cote pour pouvoir iterer sur l'ALGO sans re-marteler la DB.
        ded = pickle.loads(cache.read_bytes())
        print(f"[1/5] {len(ded):,} lignes relues du cache", flush=True)
    else:
        print(f"[1/5] Chargement depuis {a.since} (id >= {a.start_id}) ...", flush=True)
        sb = create_client(os.environ["SUPABASE_URL"],
                           os.environ["SUPABASE_SERVICE_ROLE_KEY"])
        rows = fetch_all(sb, a.start_id)
        print(f"      {len(rows):,} lignes retenues", flush=True)
        ded = dedup(rows)
        if cache:
            cache.write_bytes(pickle.dumps(ded))
    print(f"      {len(ded):,} apres dedup (strategie, token)", flush=True)

    try:
        from strategies import _is_artifact_family, _DEFAULT_DEPRECATED
    except Exception:
        _is_artifact_family, _DEFAULT_DEPRECATED = (lambda _n: False), set()
    n0 = len({d[0] for d in ded})
    ded = [d for d in ded
           if not _is_artifact_family(d[0]) and d[0] not in _DEFAULT_DEPRECATED]
    print(f"      {n0 - len({d[0] for d in ded})} strategies ecartees (artefact)",
          flush=True)

    rng = np.random.default_rng(20260812)
    for gran in ("mois", "quinzaine", "semaine"):
        strats, periods, M, N = build_matrix(ded, gran, a.min_n)
        print("\n" + "=" * 78)
        print(f"GRANULARITE = {gran.upper()}  |  {len(strats)} strategies x "
              f"{len(periods)} periodes  |  min_n = {a.min_n}")
        print("=" * 78)
        print("periodes:", ", ".join(periods))

        # ---- A. borne haute -------------------------------------------------
        orac, fx_best, fx_med, fx_moy, n_full = oracle_and_fixed(M)
        print(f"\n[A] BORNE HAUTE (argent cumule, mise ${STAKE:.0f})")
        print(f"    rotation PARFAITE (oracle, triche)      : ${orac:+,.0f}")
        print(f"    meilleure strategie FIXE (in-sample)    : ${fx_best:+,.0f}"
              f"   ({n_full} bras presents partout)")
        print(f"    strategie FIXE mediane                  : ${fx_med:+,.0f}")
        print(f"    moyenne de tous les bras (= hasard)     : ${fx_moy:+,.0f}")
        if fx_best > 0:
            print(f"    -> l'oracle vaut x{orac / fx_best:.1f} le meilleur fixe")

        # ---- B. rotation lag-1 vs hasard ------------------------------------
        print(f"\n[B] ROTATION LAG-1 (top-K de t-1 trade en t) vs {a.perms} "
              f"tirages ALEATOIRES")
        print(f"    {'K':>4} {'rotation':>12} {'H0 moyen':>12} {'H0 p95':>12} "
              f"{'H0 max':>12}  verdict")
        for K in (1, 3, 5, 10):
            real, _ = rotation(M, K)
            null = np.array([rotation(M, K, rng)[0] for _ in range(a.perms)])
            p95, mx = np.percentile(null, 95), null.max()
            verdict = "DEPASSE" if real > p95 else "sous le hasard"
            print(f"    {K:>4} ${real:>+11,.0f} ${null.mean():>+11,.0f} "
                  f"${p95:>+11,.0f} ${mx:>+11,.0f}  {verdict}")

        # ---- B bis. au niveau FAMILLE ---------------------------------------
        fam_of_s = [family_of(s) for s in strats]
        fams = sorted({f for f in fam_of_s if f})
        if fams:
            FM = np.full((len(fams), len(periods)), np.nan)
            for i, f in enumerate(fams):
                rowsel = [j for j, ff in enumerate(fam_of_s) if ff == f]
                sub = M[rowsel, :]
                with np.errstate(invalid="ignore"):
                    FM[i, :] = np.nanmean(sub, axis=0)   # argent d'un bras type
            print(f"\n[B bis] MEME TEST AU NIVEAU FAMILLE ({len(fams)} familles)")
            for K in (1, 2):
                real, _ = rotation(FM, K)
                null = np.array([rotation(FM, K, rng)[0] for _ in range(a.perms)])
                verdict = "DEPASSE" if real > np.percentile(null, 95) else "sous le hasard"
                print(f"    K={K}: ${real:+,.0f}  vs H0 moyen ${null.mean():+,.0f} "
                      f"/ p95 ${np.percentile(null, 95):+,.0f}  -> {verdict}")

        # ---- A bis. l'oracle est-il autre chose qu'un maximum chanceux ? ----
        on = oracle_null(M, min(a.perms, 100), rng)
        print(f"\n[A bis] L'ORACLE SOUS H0 (periodes melangees par bras, "
              f"{len(on)} tirages)")
        print(f"    oracle reel ${orac:+,.0f}  |  H0 moyen ${on.mean():+,.0f}  "
              f"p95 ${np.percentile(on, 95):+,.0f}  max ${on.max():+,.0f}")
        print(f"    -> {'variation de regime REELLE' if orac > np.percentile(on, 95) else 'PUR bruit de maximum'}"
              f" ; part imputable au bruit ~{100 * on.mean() / orac:.0f} %")

        # ---- B ter. haut vs bas du classement -------------------------------
        dec = decile_decomposition(M)
        if dec:
            print("\n[B ter] ARGENT MOYEN EN t SELON LE DECILE EN t-1 "
                  "(D1 = meilleurs de t-1)")
            print("    " + "  ".join(f"D{d+1}:${v:+,.0f}" for d, v in dec.items()))

        # ---- B quater. la regle d'EXCLUSION ---------------------------------
        rx, bx, nx = exclusion_rule(M, min(a.perms, 100), rng)
        print(f"\n[B quater] REGLE D'EXCLUSION (garder les bras a argent > 0 en t-1)")
        print(f"    exclusion ${rx:+,.0f}  |  tous les bras ${bx:+,.0f}  |  "
              f"H0 meme taille: moyen ${nx.mean():+,.0f} p95 ${np.percentile(nx, 95):+,.0f}")
        print(f"    -> {'DEPASSE le hasard' if rx > np.percentile(nx, 95) else 'ne depasse pas le hasard'}")

        # ---- D. adaptatif vs fixe, a armes egales ---------------------------
        fam_rows = [i for i, s in enumerate(strats)
                    if (lambda f: f and f.startswith(("TP<=50", "TP51-80"))
                        and ("SL<=20" in f or "SL21-30" in f))(family_of(s))]
        robustness_table(M, fam_rows)
        single_arm_policies(M, [family_of(x) for x in strats])
        if len(fam_rows) >= 3:
            within_family(M, fam_rows, strats, min(a.perms, 100), rng)

        # ---- persistance de rang --------------------------------------------
        rp = rank_persistence(M)
        if rp:
            rs = [r for _, _, r in rp]
            print(f"\n[persistance] Spearman t-1 -> t : "
                  + ", ".join(f"{r:+.3f}" for r in rs))
            print(f"              moyenne {np.mean(rs):+.3f} "
                  f"(0 = le classement ne se reproduit pas)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
