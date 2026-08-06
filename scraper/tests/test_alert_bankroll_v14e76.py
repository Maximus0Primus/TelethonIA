"""v14e.76 — l'alerte KOL CALL affichait le mauvais bankroll.

Symptome (06/08, $QUENTIN par dylansdirtydiary) :

    💰 $62934 bankroll | $300 deploye (4 pos) | $62634 dispo

alors que le deck E30 tourne sur trois bankrolls de $1000 avec une mise fixe de
$100. Le montant affiche est le solde GLOBAL de rt_bankroll, cumule sur toutes
les strategies et toutes les chaines depuis avril : il ne represente rien de ce
qui est engage sur ce trade.

Deux causes chainees, une par classe de test ici :

1. safe_scraper reconstruisait un dict de 3 cles (chain / liquidite / rt_score)
   pour rejouer _passes_strategy_filter et deviner quelles strategies avaient
   ouvert. Les trois PF_* de v14e.75 filtrent sur market_cap et sur la bande de
   sentiment (qui a besoin de kol_group + token_address) : aucune de ces cles
   n'etait dans le dict, donc les trois etaient rejetees.
2. strategy_positions revenant vide, alert_kol_trade tombait sur sa branche de
   repli `elif bankroll > 0` et imprimait le bankroll global.
"""

import alerter
from paper_trader import _passes_strategy_filter

PF = "PF_FAST_TP50_SL30_MCAP_S40"


class TestFiltreAppauvri:
    """Le dict a 3 cles de l'ancienne alerte rejette les strategies du deck."""

    def test_dict_minimal_rejette_les_pf(self):
        appauvri = {
            "chain": "solana",
            "_rt_liquidity_usd": 16_000,
            "_rt_score": 42,
        }
        # C'est exactement ce que l'ancien code passait -> aucune PF_ ne passe,
        # donc l'alerte n'avait plus rien a afficher.
        assert _passes_strategy_filter(appauvri, PF) is False

    def test_le_mcap_manquant_suffit_a_faire_echouer(self):
        """Meme avec kol_group + token_address, market_cap absent = rejet."""
        sans_mcap = {
            "chain": "solana",
            "_rt_liquidity_usd": 16_000,
            "_rt_score": 42,
            "_rt_kol_group": "dylansdirtydiary",
            "token_address": "9uNefL6BciwknzLCkJ8BbXyf6CEuhrYeF7biyngHpump",
        }
        assert _passes_strategy_filter(sans_mcap, PF) is False


class TestRenduAlerte:
    """alert_kol_trade ne doit jamais afficher le bankroll global en mode hybride."""

    def test_bankroll_global_affiche_quand_positions_vides(self, mock_telegram):
        """Reproduction du bug: positions vides + bankroll global -> $62934."""
        alerter.alert_kol_trade(
            "QUENTIN", "dylansdirtydiary", 0.0000479, 100.0, 42, 16_000,
            ca="9uNefL6BciwknzLCkJ8BbXyf6CEuhrYeF7biyngHpump", mcap=47_000,
            bankroll=62_934, deployed_usd=200, open_count=3,
            strategy_positions={},
        )
        assert "<b>$62934</b> bankroll" in mock_telegram[-1]["text"]
        assert "$62634 dispo" in mock_telegram[-1]["text"]

    def test_bankroll_zero_supprime_la_ligne(self, mock_telegram):
        """Le correctif passe bankroll=0 en mode hybride: aucune ligne bankroll."""
        alerter.alert_kol_trade(
            "QUENTIN", "dylansdirtydiary", 0.0000479, 100.0, 42, 16_000,
            ca="9uNefL6BciwknzLCkJ8BbXyf6CEuhrYeF7biyngHpump", mcap=47_000,
            bankroll=0, deployed_usd=200, open_count=3,
            strategy_positions={},
        )
        texte = mock_telegram[-1]["text"]
        assert "bankroll" not in texte
        assert "dispo" not in texte

    def test_positions_reelles_affichent_les_seeds_de_1000(self, mock_telegram):
        """Avec les lignes relues depuis paper_trades, on voit les vrais bankrolls."""
        alerter.alert_kol_trade(
            "QUENTIN", "dylansdirtydiary", 0.0000479, 300.0, 42, 16_000,
            ca="9uNefL6BciwknzLCkJ8BbXyf6CEuhrYeF7biyngHpump", mcap=47_000,
            bankroll=0,
            strategy_positions={
                "PF_BE25_TP80_SL30": {"pos": 100.0, "balance": 1090.11},
                "PF_FAST_TP50_SL30_MCAP_S40": {"pos": 100.0, "balance": 1072.26},
                "PF_TP50_SL40_S35": {"pos": 100.0, "balance": 1098.38},
            },
        )
        texte = mock_telegram[-1]["text"]
        assert "Positions ouvertes" in texte
        assert "$1090" in texte and "$1072" in texte and "$1098" in texte
        assert "62934" not in texte
