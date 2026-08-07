"""v14e.80 — le test apparie applique a l'axe STRATEGIE.

Le sweep possedait deja l'instrument qui marche (test apparie + plancher de bruit)
et ne l'appliquait qu'aux FILTRES. Les strategies n'avaient qu'un classement, et
c'est le classement qui est du bruit : run 31089886117, meilleure config 23.90 pts
pour un plancher a 24.88 — le sommet SOUS le plancher, avec 4 mois de donnees.

La cause est mecanique : le classement compare des configs mesurees sur des tokens
DIFFERENTS. Cette variance inter-cellules est ce que le maximum de ~1 M de tests va
chercher. L'apparie la differencie : dans une cellule, toutes les sorties voient les
memes tokens.

Ce que ces tests verrouillent :
  1. le plancher de permutation REJETTE quand les sorties sont echangeables
     (sinon l'instrument recree le probleme qu'il est cense resoudre) ;
  2. il RETIENT une sortie reellement superieure plantee dans les memes donnees ;
  3. les disqualifications temporelles mordent (un seul mois porteur = rejet) ;
  4. les clones ETH_* sont regroupes et ne comptent pas comme des decouvertes
     independantes.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from analyze_mega_sweep import verdict_par_exit  # noqa: E402
from strategies import STRATEGIES, _is_artifact_family  # noqa: E402

FILTRES = [f"F{i}" for i in range(12)]
AGES = ["ALL", "AGE24"]

# v14e.82: `verdict_par_exit` ecarte les strats non enregistrees et les familles
# artefact, comme le fait le classement. Les tests doivent donc porter sur de
# VRAIS noms du registre, sinon ils testent un chemin qui n'existe pas en prod.
_PROPRES = sorted(n for n in STRATEGIES
                  if not _is_artifact_family(n) and not n.startswith("ETH_"))
SORTIES = _PROPRES[:40]
CIBLE = SORTIES[7]          # la sortie a qui on plante un edge
ARTEFACT = next(n for n in STRATEGIES if n.upper().startswith("TD2_"))
# Vraie paire SOL/ETH du registre : les deux doivent survivre au filtre
# "strat enregistree", sinon le test du regroupement ne teste rien.
_PAIRE = next(n for n in _PROPRES if "ETH_" + n in STRATEGIES)
CLONE = "ETH_" + _PAIRE


def _daily(moyenne, mois=5, jours_par_mois=20, concentre=False):
    """PnL journalier etale sur `mois`, ou concentre sur un seul mois."""
    d = {}
    for m in range(1, mois + 1):
        for j in range(1, jours_par_mois + 1):
            v = 0.0 if (concentre and m != 2) else moyenne
            d[f"2026-{m + 3:02d}-{j:02d}"] = v
    if concentre:
        for j in range(1, jours_par_mois + 1):
            d[f"2026-05-{j:02d}"] = moyenne * mois
    return json.dumps(d)


def _csv(strategies, rng, ev_par_strat=None, concentre=None):
    """Une ligne par (strategie, filtre, bande d'age) dans la cellule canonique."""
    lignes = []
    for f_i, f in enumerate(FILTRES):
        for a in AGES:
            # Effet de cellule volontairement ENORME devant l'ecart entre sorties :
            # c'est exactement ce qui noie le classement et que l'apparie retire.
            base_cellule = 50.0 * f_i
            for s in strategies:
                ev = base_cellule + rng.normal(0, 1.0) + (ev_par_strat or {}).get(s, 0.0)
                lignes.append({
                    "strategy": s, "filter": f, "age_band": a,
                    "source": "jupiter", "smoothing": "raw", "polling_mode": "lazy_fast",
                    "n": 200, "avg_pnl_pct": ev,
                    "daily_pnl_json": _daily(ev, concentre=(concentre == s)),
                })
    return pd.DataFrame(lignes)


@pytest.fixture
def tmp_csv(tmp_path):
    return tmp_path / "_mega_sweep_extended.csv"


class TestPlancherDePermutation:
    """Le controle doit rejeter quand il n'y a rien a trouver."""

    def test_sorties_echangeables_ne_donnent_aucun_retenu(self, tmp_csv, capsys):
        rng = np.random.default_rng(7)
        df = _csv(SORTIES, rng)  # aucune sortie n'est meilleure qu'une autre
        verdict_par_exit(df, tmp_csv, n_permutations=60)
        out = capsys.readouterr().out
        assert "AUCUNE sortie ne depasse le plancher" in out, out

    def test_une_sortie_superieure_est_retrouvee(self, tmp_csv, capsys):
        rng = np.random.default_rng(7)
        # +6 pp d'EV constants sur toutes les cellules : un edge, pas un coup de chance.
        df = _csv(SORTIES, rng, ev_par_strat={CIBLE: 6.0})
        verdict_par_exit(df, tmp_csv, n_permutations=60)
        out = capsys.readouterr().out
        assert "AUCUNE sortie ne depasse le plancher" not in out, out
        assert CIBLE in out, out

    def test_l_effet_de_cellule_ne_cree_pas_de_faux_positif(self, tmp_csv):
        """La variance inter-cellules (x50 entre filtres) ne doit rien produire."""
        rng = np.random.default_rng(11)
        df = _csv(SORTIES, rng)
        verdict_par_exit(df, tmp_csv, n_permutations=60)
        res = pd.read_csv(tmp_csv.parent / "_mega_sweep_verdict_sortie.csv", index_col=0)
        assert not res["retenue"].any()


class TestDisqualificationsTemporelles:
    def test_un_seul_mois_porteur_est_rejete(self, tmp_csv):
        rng = np.random.default_rng(7)
        # CIBLE a le meme total, mais tout vient d'un seul mois -> doit etre ecarte.
        df = _csv(SORTIES, rng, ev_par_strat={CIBLE: 6.0}, concentre=CIBLE)
        verdict_par_exit(df, tmp_csv, n_permutations=60)
        res = pd.read_csv(tmp_csv.parent / "_mega_sweep_verdict_sortie.csv", index_col=0)
        assert not bool(res.loc[CIBLE, "retenue"])


class TestRegroupementDesClones:
    def test_les_clones_eth_ne_comptent_pas_deux_fois(self, tmp_csv, capsys):
        rng = np.random.default_rng(7)
        df = _csv(SORTIES + [_PAIRE], rng, ev_par_strat={CIBLE: 6.0, _PAIRE: 6.0})
        # Clone parfait : memes chiffres, autre nom (cas reel des ETH_* sur SOL).
        clone = df[df["strategy"] == _PAIRE].copy()
        clone["strategy"] = CLONE
        verdict_par_exit(pd.concat([df, clone], ignore_index=True), tmp_csv,
                         n_permutations=60)
        out = capsys.readouterr().out
        ligne = [l for l in out.splitlines() if "apres regroupement des clones" in l]
        assert ligne, out
        # "=> N sorties au-dessus du plancher (M apres regroupement)" avec M < N
        n = int(ligne[0].split("=>")[1].split("sorties")[0].strip())
        m = int(ligne[0].split("(")[1].split("apres")[0].strip())
        assert m < n, ligne[0]


class TestExclusionDesArtefacts:
    """v14e.82 — le verdict doit ecarter ce que le classement ecarte deja.

    Oubli de la v14e.80 : `verdict_par_exit` lisait le CSV brut alors que le
    classement filtre `family_realism >= 0.5` et les strats non enregistrees.
    Sur le run 31147456647, `TD2_BE5_TP120_SL44_T25` ressortait 1er par EV
    (10.07 %) alors que `TD2_` est une famille artefact declaree — « prefixes
    whose shadow $/day cannot translate to live ». Un instrument cense trier le
    vrai du bruit ne doit pas mettre en tete une famille dont on sait deja
    qu'elle ne se transpose pas.
    """

    def test_une_famille_artefact_meme_excellente_est_ecartee(self, tmp_csv, capsys):
        rng = np.random.default_rng(7)
        # On lui donne le MEILLEUR edge du lot : elle doit quand meme disparaitre.
        df = _csv(SORTIES + [ARTEFACT], rng,
                  ev_par_strat={CIBLE: 6.0, ARTEFACT: 12.0})
        verdict_par_exit(df, tmp_csv, n_permutations=60)
        out = capsys.readouterr().out
        assert ARTEFACT not in out, f"{ARTEFACT} ne doit pas apparaitre:\n{out}"
        assert CIBLE in out, out

    def test_une_strat_non_enregistree_est_ecartee(self, tmp_csv, capsys):
        rng = np.random.default_rng(7)
        df = _csv(SORTIES + ["STRAT_FANTOME_QUI_NEXISTE_PAS"], rng,
                  ev_par_strat={CIBLE: 6.0, "STRAT_FANTOME_QUI_NEXISTE_PAS": 12.0})
        verdict_par_exit(df, tmp_csv, n_permutations=60)
        out = capsys.readouterr().out
        assert "STRAT_FANTOME_QUI_NEXISTE_PAS" not in out, out
