"""v14e.79 — le deck E30 n'ouvrait AUCUN trade main, donc plus aucune alerte.

Symptome, du 06/08 09:36 UTC au 07/08 : 58 detections RT, zero alerte Telegram,
et dans les logs a chaque call :

    paper_trader: opened 0 rows + 314 shadow, $100 budget -> $PERIWINKLE=$100.0

Le shadow grid ouvre, le deck main n'ouvre rien. Comme safe_scraper ne poste
l'alerte KOL CALL que `if opened and opened > 0`, un deck qui n'ouvre jamais
rend le systeme totalement silencieux -- sans une seule ligne d'erreur.

Cause : les trois strategies du portefeuille E30 (v14e.75) filtrent toutes sur
la bande de sentiment. `_passes_strategy_filter` lit ce sentiment via
`_msg_sentiment`, qui fait un SELECT sur kol_mentions (kol_group, resolved_ca).
Or cette ligne est ecrite par le batch, PAS par le RT :

    1724 mentions sur 7 jours, lag message -> ecriture :
        mediane 29.3 min, p05 13.8 min, ZERO ligne sous 60 s

Le RT decide en ~7 s. Le SELECT ne trouve donc jamais rien, `_msg_sentiment`
renvoie None, et le contrat de la fonction ("None = gate non satisfait") rejette
les trois strategies a chaque call. Le gate etait insatisfiable par construction
dans le chemin RT.

L'information, elle, existe bien a l'instant du call : le sentiment est une
fonction PURE du texte du message, que le listener RT a deja en main. Seul son
STOCKAGE est tardif. Le correctif calcule donc le sentiment en ligne et le pose
sur le token, le SELECT ne servant plus que de repli pour le chemin batch.

⚠️ Parite : le batch calcule le sentiment sur `message.message` AUGMENTE des URLs
des entites Telegram. Calculer le RT sur le seul `message.message` produirait un
sentiment different de celui mesure au backtest. D'ou le constructeur de texte
partage, teste ici.
"""

import pytest

import paper_trader
from paper_trader import _passes_strategy_filter

# Les trois membres du portefeuille E30 (v14e.75), tous a bande de sentiment.
PF_BE25 = "PF_BE25_TP80_SL30"
PF_FAST = "PF_FAST_TP50_SL30_MCAP_S40"
PF_TP50 = "PF_TP50_SL40_S35"
DECK_E30 = [PF_BE25, PF_FAST, PF_TP50]


def token_rt():
    """Un token RT complet, qui doit passer tous les gates SAUF le sentiment.

    Volontairement genereux : mcap dans la fenetre de PF_FAST, rt_score au-dessus
    des deux planchers (40 / 35), token jeune. Si un rejet survient, il ne peut
    venir que de la bande de sentiment.
    """
    return {
        "chain": "solana",
        "token_address": "D63HYXyihS11v1UrSETWmTMHZGpaKo7NGxzuLGnJpump",
        "symbol": "PERIWINKLE",
        "market_cap": 120_000,
        "_rt_liquidity_usd": 40_000,
        "_rt_score": 55,
        "_rt_kol_group": "FrenzGems",
        "_rt_token_age_hours": 0.2,
    }


@pytest.fixture(autouse=True)
def _vide_le_cache_sentiment():
    """_sentiment_cache memorise les None : il doit repartir propre a chaque test."""
    paper_trader._sentiment_cache.clear()
    yield
    paper_trader._sentiment_cache.clear()


class TestGateInsatisfiableEnRT:
    """La reproduction : en RT la ligne kol_mentions n'existe pas encore."""

    def test_les_trois_strategies_du_deck_sont_rejetees(self, monkeypatch):
        # Le SELECT ne trouve rien -> None, exactement ce qui se passe a t+7s.
        monkeypatch.setattr(paper_trader, "_msg_sentiment", lambda kol, ca: None)
        tok = token_rt()
        rejetees = [s for s in DECK_E30 if not _passes_strategy_filter(tok, s)]
        # C'est le bug : le deck complet est rejete -> "opened 0 rows".
        assert rejetees == DECK_E30

    def test_donc_aucune_alerte_possible(self, monkeypatch):
        """opened == 0 ferme la garde `if opened and opened > 0` de safe_scraper."""
        monkeypatch.setattr(paper_trader, "_msg_sentiment", lambda kol, ca: None)
        tok = token_rt()
        opened = sum(1 for s in DECK_E30 if _passes_strategy_filter(tok, s))
        assert opened == 0


class TestSentimentEnLigne:
    """Le correctif : le sentiment calcule en RT est pose sur le token."""

    def test_sentiment_en_ligne_dans_la_bande_ouvre_le_deck(self, monkeypatch):
        # La DB reste muette (cas RT reel) : seule la valeur en ligne compte.
        monkeypatch.setattr(paper_trader, "_msg_sentiment", lambda kol, ca: None)
        tok = token_rt()
        tok["_rt_msg_sentiment"] = 0.42  # dans les trois bandes
        assert all(_passes_strategy_filter(tok, s) for s in DECK_E30)

    def test_la_bande_continue_de_filtrer(self, monkeypatch):
        """Le correctif ne doit pas transformer le gate en passe-plat."""
        monkeypatch.setattr(paper_trader, "_msg_sentiment", lambda kol, ca: None)
        tok = token_rt()
        tok["_rt_msg_sentiment"] = 0.15  # sous les trois planchers (0.25/0.30/0.35)
        assert not any(_passes_strategy_filter(tok, s) for s in DECK_E30)

        tok["_rt_msg_sentiment"] = 0.90  # au-dessus des trois plafonds
        assert not any(_passes_strategy_filter(tok, s) for s in DECK_E30)

    def test_la_borne_haute_reste_exclusive(self, monkeypatch):
        """PF_BE25 = [0.25, 0.75[ : 0.75 dehors, 0.25 dedans."""
        monkeypatch.setattr(paper_trader, "_msg_sentiment", lambda kol, ca: None)
        tok = token_rt()
        tok["_rt_msg_sentiment"] = 0.75
        assert _passes_strategy_filter(tok, PF_BE25) is False
        tok["_rt_msg_sentiment"] = 0.25
        assert _passes_strategy_filter(tok, PF_BE25) is True

    def test_la_valeur_en_ligne_prime_sur_la_db(self, monkeypatch):
        """Le RT ne doit pas payer le SELECT ni dependre de son resultat."""
        appels = []

        def _espion(kol, ca):
            appels.append((kol, ca))
            return 0.05  # hors bande : si la DB primait, le deck serait rejete

        monkeypatch.setattr(paper_trader, "_msg_sentiment", _espion)
        tok = token_rt()
        tok["_rt_msg_sentiment"] = 0.42
        assert all(_passes_strategy_filter(tok, s) for s in DECK_E30)
        assert appels == [], "le SELECT ne doit pas etre fait quand le RT a la valeur"

    def test_le_chemin_batch_utilise_toujours_la_db(self, monkeypatch):
        """Sans valeur en ligne (batch), le repli DB doit rester en place."""
        monkeypatch.setattr(paper_trader, "_msg_sentiment", lambda kol, ca: 0.42)
        tok = token_rt()  # pas de _rt_msg_sentiment
        assert all(_passes_strategy_filter(tok, s) for s in DECK_E30)


class TestPariteDuTexte:
    """Le sentiment RT doit porter sur le MEME texte que celui du batch."""

    def test_les_urls_des_entites_sont_ajoutees(self):
        from safe_scraper import _message_text_with_entity_urls

        class _Entite:
            def __init__(self, url):
                self.url = url

        class _Msg:
            message = "  Nouveau gem, entrez maintenant  "
            entities = [_Entite("https://dexscreener.com/solana/abc"), _Entite(None)]

        # Exactement la construction du batch : strip + une URL par ligne.
        assert _message_text_with_entity_urls(_Msg()) == (
            "Nouveau gem, entrez maintenant\nhttps://dexscreener.com/solana/abc"
        )

    def test_sans_entites_le_texte_est_juste_strippe(self):
        from safe_scraper import _message_text_with_entity_urls

        class _Msg:
            message = "  gm  "
            entities = None

        assert _message_text_with_entity_urls(_Msg()) == "gm"

    def test_message_vide_ou_absent_ne_leve_pas(self):
        from safe_scraper import _message_text_with_entity_urls

        assert _message_text_with_entity_urls(None) == ""

        class _Msg:
            message = None
            entities = None

        assert _message_text_with_entity_urls(_Msg()) == ""
