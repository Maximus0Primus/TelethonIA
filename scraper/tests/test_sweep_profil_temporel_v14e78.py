"""v14e.78 — `total_at_cap` est un TOTAL, donc il depend de la DUREE couverte.

Tant que le sweep ne voyait que 9 jours (cf. v14e.77), toutes les configs
partageaient exactement la meme fenetre et comparer des totaux etait legitime.
Sur 4 mois ce n'est plus vrai: un filtre peut etre correle au calendrier — une
bande de mcap laisse passer beaucoup de tokens quand le marche est chaud et
presque rien sinon — donc son `n` se concentre sur une periode. Deux configs a
n=200 n'ont alors pas la meme valeur selon qu'elles etalent leurs trades sur
120 jours ou les concentrent sur trois semaines favorables, et le total ne fait
pas la difference.

D'ou trois colonnes exposees a cote du total (duree couverte, mois positifs,
part du meilleur mois) et un garde-fou qui ne s'applique QUE si la fenetre porte
au moins 3 mois — sinon « un seul mois fait le resultat » serait vrai par
construction et disqualifierait tout le monde.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from analyze_mega_sweep import profil_temporel  # noqa: E402


def _serie(jours: dict) -> str:
    return json.dumps(jours)


class TestProfilTemporel:
    def test_config_reguliere_sur_quatre_mois(self):
        jours = {}
        for mois in ("2026-04", "2026-05", "2026-06", "2026-07"):
            for d in range(1, 21):
                jours[f"{mois}-{d:02d}"] = 2.0
        n_jours, n_mois, positifs, part = profil_temporel(_serie(jours))
        assert n_jours == 80
        assert n_mois == 4
        assert positifs == 4
        # Gains repartis a l'identique sur 4 mois => aucun ne domine.
        assert part == pytest.approx(0.25, abs=0.01)

    def test_config_portee_par_un_seul_mois(self):
        jours = {f"2026-04-{d:02d}": 0.1 for d in range(1, 21)}
        jours.update({f"2026-06-{d:02d}": 40.0 for d in range(1, 21)})
        n_jours, n_mois, positifs, part = profil_temporel(_serie(jours))
        assert n_mois == 2 and positifs == 2
        # Meme avec deux mois positifs, quasi tout le gain vient de juin.
        assert part > 0.99

    def test_mois_negatifs_comptes(self):
        jours = {f"2026-04-{d:02d}": -5.0 for d in range(1, 11)}
        jours.update({f"2026-05-{d:02d}": 3.0 for d in range(1, 11)})
        jours.update({f"2026-06-{d:02d}": -2.0 for d in range(1, 11)})
        _, n_mois, positifs, _ = profil_temporel(_serie(jours))
        assert n_mois == 3 and positifs == 1

    def test_serie_absente_ou_illisible(self):
        assert profil_temporel(None) == (0, 0, 0, 1.0)
        assert profil_temporel("pas du json") == (0, 0, 0, 1.0)
        assert profil_temporel("{}") == (0, 0, 0, 1.0)


class TestGardeFouRegularite:
    """Le controle de regularite ne doit pas s'appliquer sur une fenetre courte."""

    def test_le_seuil_de_trois_mois_est_code(self):
        src = (Path(__file__).resolve().parents[2] / "scripts"
               / "analyze_mega_sweep.py").read_text(encoding="utf-8")
        assert "_regularite_jugeable" in src
        assert "_mois_max >= 3" in src, (
            "sur moins de 3 mois, 'un seul mois fait le resultat' est vrai par "
            "construction: le garde-fou doit se desactiver, pas tout recaler"
        )

    def test_les_trois_disqualifications_existent(self):
        src = (Path(__file__).resolve().parents[2] / "scripts"
               / "analyze_mega_sweep.py").read_text(encoding="utf-8")
        for motif in ("duree_conservee", "mois_positifs", "part_meilleur_mois"):
            assert motif in src, f"critere {motif} absent"
