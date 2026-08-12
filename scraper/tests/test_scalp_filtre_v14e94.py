"""v14e.94 — le scalp filtre: 3 bras SHADOW, appaires aux bras du deck.

D'ou ils viennent (12/08, scripts/regime_rotation_test.py):
  - dans une famille, choisir le BRAS est du bruit (oracle intra-famille SOUS
    son plancher de permutation aux 3 granularites);
  - ce qui separe les bras, c'est le FILTRE D'ENTREE (volume vs argent:
    Spearman -0.328, p < 0.001);
  - gradient PRE-SPECIFIE sur la cible, SL <= 30: TP<=20 +$618 (seule tranche
    positive, 3/5 mois) > TP21-40 -$1 521 ~ TP41-60 -$1 502 > TP61-80 -$1 960.

Ce que ces tests verrouillent, et pourquoi chacun compte:
  1. les 3 bras existent, sont SHADOW, et ne sont dans AUCUNE allocation main —
     le classement par bras ne persiste pas, donc on MESURE avant de promouvoir;
  2. chaque bras garde les filtres de sa reference A LA LETTRE: si un filtre
     derive, l'ecart ne mesure plus la sortie mais un melange des deux, et la
     comparaison appariee est perdue (c'est la raison d'etre du bloc);
  3. les references du deck restent intactes;
  4. AUCUN be_activation sur une cible a +20 %: un BE a +25 % s'armerait APRES
     le TP, donc jamais — c'est le no-op des LOCK sur TP200 (11/08), et un
     no-op silencieux fait croire qu'on teste quelque chose;
  5. la sortie est bien un scalp (tp_mult <= 1.20) et le SL reste dans la
     tranche mesuree (SL <= 30, soit sl_mult >= 0.70).
"""

import pytest

from strategies import (SHADOW_STRATEGIES, STRATEGIES, STRATEGY_FILTERS,
                        _DEFAULT_DEPRECATED, _is_artifact_family)

SCALP = ["PFSC_TP20_SL30_S35", "PFSC_TP20_SL30_BANDE", "PFSC_TP15_SL25_S35"]

# (bras scalp, bras de reference du deck) — le filtre doit etre IDENTIQUE.
APPARIES = [
    ("PFSC_TP20_SL30_S35", "PF_TP50_SL40_S35"),
    ("PFSC_TP20_SL30_BANDE", "PF_BE25_TP80_SL30"),
    ("PFSC_TP15_SL25_S35", "PF_TP50_SL40_S35"),
]


def test_les_trois_bras_existent_et_sont_shadow(subtests):
    for nom in SCALP:
        with subtests.test(bras=nom):
            assert nom in STRATEGIES, f"{nom} absent du registre"
            assert nom in SHADOW_STRATEGIES, f"{nom} doit tourner en shadow"
            assert nom not in _DEFAULT_DEPRECATED
            assert not _is_artifact_family(nom), (
                f"{nom} tomberait dans le filtre famille-artefact et serait "
                "ecarte de tous les classements")
            assert STRATEGY_FILTERS[nom]["chain"] == "solana"


def test_filtre_identique_a_la_reference(subtests):
    """Le filtre d'entree doit etre copie A LA LETTRE depuis la reference.

    C'est la condition qui rend l'ecart interpretable: si le filtre bouge aussi,
    on ne mesure plus la sortie mais un melange, et l'apparie ne vaut rien.
    """
    for scalp, ref in APPARIES:
        with subtests.test(bras=scalp, reference=ref):
            assert STRATEGY_FILTERS[scalp] == STRATEGY_FILTERS[ref], (
                f"{scalp} doit avoir EXACTEMENT le filtre de {ref}; "
                f"obtenu {STRATEGY_FILTERS[scalp]} vs {STRATEGY_FILTERS[ref]}")


def test_horizon_identique_a_la_reference(subtests):
    """Meme raison: un horizon different ferait varier une 2e chose a la fois."""
    for scalp, ref in APPARIES:
        with subtests.test(bras=scalp, reference=ref):
            assert (STRATEGIES[scalp][0]["horizon_min"]
                    == STRATEGIES[ref][0]["horizon_min"])


def test_sortie_bien_dans_la_tranche_mesuree(subtests):
    """tp <= +20 % (la tranche gagnante) et SL <= 30 % (celle du gradient)."""
    for nom in SCALP:
        with subtests.test(bras=nom):
            spec = STRATEGIES[nom][0]
            assert spec["tp_mult"] <= 1.20, (
                f"{nom}: tp_mult {spec['tp_mult']} sort de la tranche TP<=20 "
                "sur laquelle le gradient a ete mesure")
            assert spec["sl_mult"] >= 0.70, (
                f"{nom}: sl_mult {spec['sl_mult']} = SL > 30 %, hors de la "
                "tranche SL<=30 du gradient")


def test_aucun_be_ni_lock_sur_une_cible_a_20_pct(subtests):
    """Un BE/LOCK a +25 % s'armerait APRES un TP a +20 %: no-op garanti.

    C'est exactement ce qui s'est passe le 11/08 avec les LOCK sur TP200
    (medianne -45 %, le lock ne s'armait jamais): un declencheur qui ne se
    declenche pas ne teste rien mais donne l'illusion d'un bras different.
    """
    for nom in SCALP:
        with subtests.test(bras=nom):
            spec = STRATEGIES[nom][0]
            for cle in ("be_activation", "lock_activation", "lock_pct"):
                if cle in spec:
                    assert spec[cle] < spec["tp_mult"] - 1.0, (
                        f"{nom}: {cle}={spec[cle]} >= TP "
                        f"(+{100 * (spec['tp_mult'] - 1):.0f} %) => no-op")


def test_toute_tranche_a_un_label(subtests):
    """GARDE GLOBALE, nee d'un bug de ce meme commit.

    `paper_trader.py` fait `tranche["label"]` en ACCES DIRECT (lignes 1739 et
    1929), pas `.get()`. Une tranche sans `label` ne casse donc rien au
    chargement ni aux tests de structure — elle leve un KeyError a la PREMIERE
    ouverture de trade, en production uniquement. Le bras ajoute ici etait le
    seul des 734 a en manquer un. Cette garde vaut pour tout le registre, pas
    seulement pour ce bloc.
    """
    for nom, tranches in STRATEGIES.items():
        for i, tr in enumerate(tranches):
            with subtests.test(bras=nom, tranche=i):
                assert "label" in tr, (
                    f"{nom} tranche {i}: `label` manquant => KeyError a "
                    "l'ouverture (paper_trader.py:1739)")


def test_les_bras_du_deck_sont_intacts(subtests):
    """Les references sont la base appariee: les ecraser detruit la mesure."""
    attendu = {
        "PF_TP50_SL40_S35": (1.50, 0.60, 120),
        "PF_BE25_TP80_SL30": (1.80, 0.70, 30),
        "PF_FAST_TP50_SL30_MCAP_S40": (1.50, 0.70, 30),
    }
    for nom, (tp, sl, h) in attendu.items():
        with subtests.test(bras=nom):
            spec = STRATEGIES[nom][0]
            assert (spec["tp_mult"], spec["sl_mult"], spec["horizon_min"]) == (tp, sl, h)


def test_aucun_scalp_en_main():
    """Shadow par construction: le classement par bras ne persiste pas (ro=0.019
    au niveau config, et oracle intra-famille SOUS son plancher) => on mesure
    avant de promouvoir, jamais l'inverse."""
    import json
    import os
    from pathlib import Path

    cfg = Path(__file__).resolve().parents[2] / "scraper" / "strategies.py"
    assert cfg.exists()
    # Le deck main vit en DB (rt_trade_config.hybrid_strategy.allocations); ici
    # on verrouille le seul endroit du CODE qui pourrait les y pousser.
    src = cfg.read_text(encoding="utf-8", errors="ignore")
    for nom in SCALP:
        assert f'"{nom}"' in src
    assert "PFSC" not in src.split("_PFSC_MEMBRES")[0], (
        "un bras PFSC est reference avant sa definition — risque d'allocation")
