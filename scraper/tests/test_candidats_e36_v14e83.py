"""v14e.83 — les candidats E34/E36 ajoutes au deck, et leur controle.

Deux acquis du sweep sur 4 mois, tous deux obtenus par TEST APPARIE (le
classement, lui, est reste sous son plancher de bruit sur 3 runs) :

  E34  la borne HAUTE du sentiment paie, la bande non. `SENT_NOHYPE` (s < 0.70)
       rend la MEME EV que la bande 0.30-0.70 en gardant 96 % du volume au lieu
       de 25 %. Sur les 3 exits du deck : 27 327 contre 7 047.
  E36  meilleure sortie connue = BE25_LOCK15_TP200_SL40 (4H, NZ, S40) : EV
       10.16 %, 5/5 mois positifs, ecart jupiter<->dexscreener de -0.04.

⚠️ Et la reserve qui structure tout le bloc : le sweep ne modelise pas le cout
d'execution des declencheurs intra-trade. Les deux seules LOCK jamais passees en
live sont devenues NEGATIVES (drift -4.89 et -3.54 pp, contre -1.90 et -0.09 pp
pour les sorties sans LOCK). D'ou `PFS_TP200_SANSLOCK_NOHYPE`, qui est le
controle : meme sortie, meme TP, meme horizon, sans le lock.

Ce que ces tests verrouillent :
  1. la borne haute seule laisse passer le bas du sentiment (c'est TOUT l'enjeu
     d'E34 : le volume) et coupe toujours la hype ;
  2. les 3 PF_* d'origine restent intacts — ce sont la reference appariee, les
     ecraser detruirait la comparaison ;
  3. chaque bras ne change QU'UNE variable par rapport a son comparateur ;
  4. le controle sans lock est bien identique au candidat, lock excepte.
"""

import pytest

import paper_trader
from paper_trader import _passes_strategy_filter
from strategies import STRATEGIES, STRATEGY_FILTERS, _DEFAULT_DEPRECATED

MAIN = ["PF2_LOCK15_TP200_NOHYPE", "PF2_BE25_TP80_NOHYPE"]
SHADOW = ["PFS_TP200_SANSLOCK_NOHYPE", "PFS_LOCK10_TP200_NOHYPE",
          "PFS_LOCK15_TP200_BANDE", "PFS_LOCK15_TP150_T2H_NOHYPE"]
DECK_ORIGINE = ["PF_BE25_TP80_SL30", "PF_FAST_TP50_SL30_MCAP_S40", "PF_TP50_SL40_S35"]


def token(sentiment):
    """Token RT complet, qui passe tous les gates sauf eventuellement le sentiment."""
    return {
        "chain": "solana",
        "token_address": "D63HYXyihS11v1UrSETWmTMHZGpaKo7NGxzuLGnJpump",
        "market_cap": 120_000,
        "_rt_liquidity_usd": 40_000,
        "_rt_score": 55,
        "_rt_kol_group": "FrenzGems",
        "_rt_token_age_hours": 0.2,
        "_rt_msg_sentiment": sentiment,
    }


@pytest.fixture(autouse=True)
def _cache_propre():
    paper_trader._sentiment_cache.clear()
    yield
    paper_trader._sentiment_cache.clear()


class TestBorneHauteSeule:
    """E34 : tout le gain vient du volume laisse passer en bas de bande."""

    @pytest.mark.parametrize("s", [-0.50, 0.00, 0.15, 0.29])
    def test_le_bas_du_sentiment_passe_desormais(self, s):
        # C'est exactement le volume que la bande jetait (75 % des trades).
        assert _passes_strategy_filter(token(s), "PF2_LOCK15_TP200_NOHYPE") is True
        assert _passes_strategy_filter(token(s), "PF2_BE25_TP80_NOHYPE") is True

    @pytest.mark.parametrize("s", [0.70, 0.85, 0.99])
    def test_la_hype_reste_coupee(self, s):
        assert _passes_strategy_filter(token(s), "PF2_LOCK15_TP200_NOHYPE") is False
        assert _passes_strategy_filter(token(s), "PF2_BE25_TP80_NOHYPE") is False

    def test_sentiment_absent_ne_passe_pas(self, monkeypatch):
        """Contrat inchange : sans sentiment on n'ouvre pas (moyenne -28.3 %/trade)."""
        monkeypatch.setattr(paper_trader, "_msg_sentiment", lambda k, c: None)
        t = token(0.4)
        del t["_rt_msg_sentiment"]
        assert _passes_strategy_filter(t, "PF2_LOCK15_TP200_NOHYPE") is False


class TestReferenceAppariee:
    """Les 3 PF_* d'origine sont la BASE de comparaison : intactes."""

    def test_le_deck_origine_garde_sa_borne_basse(self):
        for s in DECK_ORIGINE:
            assert "min_sentiment" in STRATEGY_FILTERS[s], s

    def test_le_deck_origine_rejette_toujours_le_bas(self):
        # Si ces trois se mettaient a accepter 0.15, on perdrait le comparateur.
        for s in DECK_ORIGINE:
            assert _passes_strategy_filter(token(0.15), s) is False, s

    def test_les_nouveaux_bras_ne_sont_pas_deprecies(self):
        for s in MAIN + SHADOW:
            assert s not in _DEFAULT_DEPRECATED, s


class TestUneSeuleVariableParBras:
    def test_le_controle_est_le_candidat_sans_lock(self):
        cand = STRATEGIES["PF2_LOCK15_TP200_NOHYPE"][0]
        ctrl = STRATEGIES["PFS_TP200_SANSLOCK_NOHYPE"][0]
        assert cand["be_lock_pct"] == 0.15
        assert "be_lock_pct" not in ctrl, "le controle ne doit PAS verrouiller"
        # tout le reste identique, sinon l'ecart ne mesure plus le lock
        for k in ("tp_mult", "sl_mult", "horizon_min", "be_activation"):
            assert cand[k] == ctrl[k], k
        assert STRATEGY_FILTERS["PF2_LOCK15_TP200_NOHYPE"] == \
               STRATEGY_FILTERS["PFS_TP200_SANSLOCK_NOHYPE"]

    def test_lock10_ne_differe_que_par_le_lock(self):
        a = STRATEGIES["PF2_LOCK15_TP200_NOHYPE"][0]
        b = STRATEGIES["PFS_LOCK10_TP200_NOHYPE"][0]
        assert (a["be_lock_pct"], b["be_lock_pct"]) == (0.15, 0.10)
        for k in ("tp_mult", "sl_mult", "horizon_min", "be_activation"):
            assert a[k] == b[k], k

    def test_le_bras_bande_ne_differe_que_par_le_filtre(self):
        assert STRATEGIES["PF2_LOCK15_TP200_NOHYPE"][0] == \
               STRATEGIES["PFS_LOCK15_TP200_BANDE"][0]
        f = STRATEGY_FILTERS["PFS_LOCK15_TP200_BANDE"]
        assert f["min_sentiment"] == 0.30 and f["max_sentiment"] == 0.70

    def test_le_bras_bande_rejette_ce_que_le_candidat_accepte(self):
        """La preuve vivante que les deux bras mesurent bien la meme chose."""
        assert _passes_strategy_filter(token(0.15), "PF2_LOCK15_TP200_NOHYPE") is True
        assert _passes_strategy_filter(token(0.15), "PFS_LOCK15_TP200_BANDE") is False
        # dans la bande commune, les deux passent
        assert _passes_strategy_filter(token(0.50), "PF2_LOCK15_TP200_NOHYPE") is True
        assert _passes_strategy_filter(token(0.50), "PFS_LOCK15_TP200_BANDE") is True
