"""v14e.77 — la fenetre de ticks du mega sweep doit suivre les TRADES, pas l'horloge.

Le run 31040338036 annoncait `Universe: 2717 unique tokens since 2026-04-13` puis
`240 with ticks`. La cause etait une fenetre codee en dur dans
`_mega_sweep_run_extended` :

    start = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()

Un token appele en juin n'a evidemment aucun tick dans [now-8j, now+1h], donc il
sortait de `ticks`, donc le replay le sautait en silence. Mesure en SQL sur la
base reelle le 06/08 :

    mois      tokens   fenetre 8j fixe   fenetre par token
    2026-04      308            0                308
    2026-05      667            0                667
    2026-06      876            0                876
    2026-07      730           89                730
    2026-08      153          153                153
    TOTAL       2734          242 (8.9%)        2734 (100%)

Soit 91 % de l'univers jete et 9 jours de profondeur au lieu de 4 mois. Un sweep
qui ne voit que 9 jours ne peut pas trouver de portefeuille multi-regimes: tout
l'interet d'E30 est qu'une strategie porte mai pendant que l'autre porte juin.

Ce test verrouille la forme du correctif au niveau du source, faute de pouvoir
rejouer un sweep de 5h en CI.
"""

import re
from pathlib import Path

import pytest

SIM = Path(__file__).resolve().parents[1] / "sim.py"


def _corps_fetch_mega() -> str:
    """Le bloc 'Fetching ticks...' de _mega_sweep_run_extended."""
    src = SIM.read_text(encoding="utf-8", errors="replace")
    i = src.index('print(f"Fetching ticks...')
    j = src.index("with ticks (", i)
    return src[i:j]


class TestFenetreTicks:
    def test_pas_de_fenetre_ancree_sur_l_horloge(self):
        """`now() - timedelta(days=N)` comme borne basse = le bug d'origine."""
        corps = _corps_fetch_mega()
        fautif = re.search(r"now\(timezone\.utc\)\s*-\s*timedelta\(\s*days\s*=", corps)
        assert fautif is None, (
            "la borne basse de la fenetre de ticks est ancree sur l'horloge: "
            "tout token plus ancien que N jours sera silencieusement exclu du "
            "replay, quel que soit --mega-since"
        )

    def test_la_fenetre_part_de_l_entree_du_token(self):
        corps = _corps_fetch_mega()
        assert 'u["created_at"]' in corps, (
            "la fenetre doit etre derivee de l'entree du token (created_at)"
        )
        assert "gte." in corps and "lte." in corps, "bornes gte/lte attendues"

    def test_couverture_faible_signalee(self):
        """Si on reperd la profondeur, ca doit se voir dans le log du run."""
        src = SIM.read_text(encoding="utf-8", errors="replace")
        assert "_couverture" in src and "de l'univers a des ticks" in src, (
            "un garde-fou doit alerter quand une minorite de l'univers a des ticks"
        )


class TestLeanGrid:
    def test_flag_expose_et_documente(self):
        src = SIM.read_text(encoding="utf-8", errors="replace")
        assert '"--mega-lean-grid"' in src
        assert "mega_lean_grid" in src

    # SOL et ETH partagent sim.py: les deux workflows doivent porter les memes
    # garde-fous, sinon le correctif ne vaut que pour la chaine qu'on a regardee.
    @pytest.mark.parametrize("nom", ["mega-sweep-48h.yml", "mega-sweep-eth-48h.yml"])
    def test_les_deux_workflows_utilisent_le_flag(self, nom):
        wf = (SIM.resolve().parents[1] / ".github" / "workflows"
              / nom).read_text(encoding="utf-8")
        assert "--mega-lean-grid" in wf, (
            f"{nom}: sans lean-grid, la fenetre elargie ne tient pas sous le cap GH "
            f"et les 70 quasi-doublons par config faussent les tests apparies"
        )

    @pytest.mark.parametrize("nom", ["mega-sweep-48h.yml", "mega-sweep-eth-48h.yml"])
    def test_l_analyse_suit_master(self, nom):
        """Le merge doit analyser avec le script courant, pas celui du SHA declencheur."""
        wf = (SIM.resolve().parents[1] / ".github" / "workflows"
              / nom).read_text(encoding="utf-8")
        merge = wf[wf.index("merge_and_analyze:"):]
        assert "ref: master" in merge.split("Set up Python")[0], (
            f"{nom}: merge_and_analyze doit checkout master pour utiliser la "
            f"derniere version du script d'analyse"
        )
