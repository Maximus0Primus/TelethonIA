"""v14e.84 — le sweep bookait ses stops au niveau THEORIQUE du stop.

Origine principale d'un ecart de **+11.71 pp** entre le sweep et les trades
reellement enregistres. Mesure sur 104 paires rejouees (memes tokens, meme
entree, meme strategie, `BE25_TP80_SL30`, juin-aout) :

    EV reelle -9.58 %   EV sweep +2.13 %   ecart +11.71 pp

Les DECISIONS de sortie concordent (`sl_hit` -> `sl_hit` dans 85 % des cas) ;
c'est le PRIX booke pour la meme sortie qui diverge :

    sl_hit   n=41   reel -51.21 %   sweep -32.68 %
    be_stop  n=11   reel -24.16 %   sweep  -1.23 %

Cote production, sur 608 `sl_hit` de la meme strategie :

    stop theorique           -30.00 %
    sortie reellement bookee -49.09 %   =  -27.27 % SOUS le stop
    416/608 sortent a plus de 10 % sous le stop

C'est le gap-through, et ce n'est pas un artefact : un memecoin qui declenche un
stop a -30 % est deja en train de s'effondrer quand la vente passe. La production
le modelise et se tient a -1.90 pp du live (145 paires sim<->live) : c'est elle
la reference calibree. `_evaluate_trade_exit` ancre son `exit_price` sur
`sl_price` + quelques bps, ce qui suppose une sortie AU niveau du stop --
hypothese qui ne tient que sur un marche liquide.

Apres correctif, sur les memes 104 paires : ecart **-1.40 pp** (et desormais du
cote prudent).

⚠️ Consequence : l'EV absolue des trois runs de sweep deja depouilles est
invalide, et le classement appariee peut bouger lui aussi -- une strategie qui
s'appuie beaucoup sur ses stops etait flattee par rapport a une qui n'en a pas.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim import _mega_gap_through  # noqa: E402

SLIP = 1 - 10 / 10_000
ENTRY = 100.0


class TestGapThrough:
    """Sur un stop, on ne peut pas vendre mieux que le marche."""

    def test_le_prix_observe_sous_le_stop_est_retenu(self):
        # Stop theorique a -30 %, mais le marche est a -80 % quand on sort.
        ev = {"status": "sl_hit", "pnl_pct": -0.30}
        r = _mega_gap_through(ev, exec_p=20.0, entry_price=ENTRY, sell_slip=SLIP)
        assert r == pytest.approx(-0.8002, abs=1e-4)

    def test_be_stop_aussi(self):
        # Le cas le plus spectaculaire: un "break-even" booke a -1 % alors que
        # le marche est a -60 %.
        ev = {"status": "be_stop", "pnl_pct": -0.01}
        r = _mega_gap_through(ev, exec_p=40.0, entry_price=ENTRY, sell_slip=SLIP)
        assert r == pytest.approx(-0.6004, abs=1e-3)

    def test_on_ne_degrade_pas_un_stop_deja_pessimiste(self):
        """Si le booke est DEJA sous le marche, on le garde: min(), pas max()."""
        ev = {"status": "sl_hit", "pnl_pct": -0.55}
        r = _mega_gap_through(ev, exec_p=90.0, entry_price=ENTRY, sell_slip=SLIP)
        assert r == pytest.approx(-0.55)

    def test_marche_juste_au_stop_ne_change_presque_rien(self):
        ev = {"status": "sl_hit", "pnl_pct": -0.30}
        r = _mega_gap_through(ev, exec_p=70.0, entry_price=ENTRY, sell_slip=SLIP)
        assert r == pytest.approx(-0.3007, abs=1e-3)


class TestNonRegression:
    """Le correctif ne doit toucher QUE les sorties par stop."""

    @pytest.mark.parametrize("statut", ["tp_hit", "timeout", "tp_late", "trail_stop"])
    def test_les_autres_sorties_sont_intactes(self, statut):
        ev = {"status": statut, "pnl_pct": 0.80}
        # meme avec un prix observe tres bas, on ne retouche pas ces sorties
        assert _mega_gap_through(ev, 10.0, ENTRY, SLIP) == 0.80

    def test_tp_hit_reste_positif(self):
        """Un TP se declenche par le HAUT: le prix observe est au-dessus."""
        ev = {"status": "tp_hit", "pnl_pct": 0.80}
        assert _mega_gap_through(ev, 185.0, ENTRY, SLIP) == 0.80

    @pytest.mark.parametrize("exec_p,entry", [(None, ENTRY), (50.0, 0), (50.0, None)])
    def test_entrees_degenerees_ne_levent_pas(self, exec_p, entry):
        ev = {"status": "sl_hit", "pnl_pct": -0.30}
        assert _mega_gap_through(ev, exec_p, entry, SLIP) == -0.30

    def test_statut_absent(self):
        assert _mega_gap_through({"pnl_pct": -0.30}, 20.0, ENTRY, SLIP) == -0.30
