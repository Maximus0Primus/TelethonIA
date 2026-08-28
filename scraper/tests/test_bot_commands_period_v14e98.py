"""v14e.98 — `/stats 7d PFW_TP50S30` rendait de l'All-time sans le dire.

Deux fautes EMPILEES, chacune silencieuse, meme famille que v14e.97 :

1. Le nom raccourci n'etait re-parsable qu'en EGALITE EXACTE. v14e.97 avait
   ajoute `_short_strat(s) == arg`, mais pas le prefixe : le bot affiche
   `PFW_TP50S30_LM_WL`, l'utilisateur tape le debut utile `PFW_TP50S30` et
   plus rien ne matchait (le nom canonique est `PFW_TP50_SL30_LM_WL`, qui ne
   commence pas par `PFW_TP50S30`). Toutes les formes du nom -- canonique ET
   raccourcie -- doivent accepter prefixe et sous-chaine.

2. `_parse_period` rendait (0, "All-time") pour TOUT token non reconnu. Le
   bras n'ayant pas resolu, il restait "7d PFW_TP50S30" comme periode : non
   reconnue => All-time. L'utilisateur demande 7 jours et lit 4 mois, sans
   aucun avertissement. Un argument non reconnu doit etre DIT.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bot_commands as bc


ACTIVE = [
    "PF_TP50_SL40_S35", "PF_BE25_TP80_SL30", "PFW_TP50_SL30_LM_WL",
    "PF_FAST_TP50_SL30_MCAP_S40", "ETH_TP100_SL50",
]


@pytest.fixture(autouse=True)
def _fixed_active(monkeypatch):
    monkeypatch.setattr(bc, "_get_active_strategies", lambda sb: list(ACTIVE))


# ── 1. le prefixe du nom RACCOURCI doit resoudre ──

def test_short_name_prefix_resolves():
    """Le bug du rapport : `PFW_TP50S30` est le debut de ce que le bot affiche."""
    assert bc._short_strat("PFW_TP50_SL30_LM_WL") == "PFW_TP50S30_LM_WL"
    assert bc._parse_strategy("PFW_TP50S30", None) == "PFW_TP50_SL30_LM_WL"


def test_short_name_substring_resolves():
    assert bc._parse_strategy("TP50S30_LM", None) == "PFW_TP50_SL30_LM_WL"


def test_exact_and_canonical_forms_still_resolve():
    assert bc._parse_strategy("PFW_TP50S30_LM_WL", None) == "PFW_TP50_SL30_LM_WL"
    assert bc._parse_strategy("PFW_TP50_SL30_LM_WL", None) == "PFW_TP50_SL30_LM_WL"
    assert bc._parse_strategy("pfw", None) == "PFW_TP50_SL30_LM_WL"


def test_ambiguity_still_reported_not_swallowed():
    strat, candidates = bc._resolve_strategy("PF_", None)
    assert strat is None and len(candidates) > 1


# ── 2. le split doit consommer le bras et laisser la periode seule ──

def test_split_consumes_short_prefix_leaving_period():
    remaining, strat, err = bc._split_strategy_args("7d PFW_TP50S30", None)
    assert err is None
    assert strat == "PFW_TP50_SL30_LM_WL"
    assert remaining == "7d"


# ── 3. une periode non reconnue ne doit plus devenir All-time en silence ──

def test_unknown_period_is_reported():
    hours, label, err = bc._parse_period("PFW_TP99S99")
    assert err is not None, "un token inconnu ne doit pas passer pour All-time"
    assert "PFW_TP99S99" in err


def test_known_periods_still_parse():
    assert bc._parse_period("7d")[:2] == (168, "7d")
    assert bc._parse_period("24h")[:2] == (24, "24h")
    assert bc._parse_period("all")[:2] == (0, "All-time")
    assert bc._parse_period("")[:2] == (0, "All-time")
    assert bc._parse_period("", default=(24, "24h"))[:2] == (24, "24h")
    assert bc._parse_period("7d")[2] is None


# ── 4. bout en bout : /stats 7d PFW_TP50S30 doit interroger 168 h ──

def test_handle_stats_queries_7d_not_all_time(monkeypatch):
    seen = []

    def _fake_query(sb, hours=0, strategy="", chain=None, **kw):
        seen.append({"hours": hours, "strategy": strategy})
        return []

    monkeypatch.setattr(bc, "_query_trades", _fake_query)
    out = bc._handle_stats(None, "7d PFW_TP50S30")

    assert seen, "aucune requete emise"
    assert all(c["hours"] == 168 for c in seen), f"fenetre demandee: {seen}"
    assert all(c["strategy"] == "PFW_TP50_SL30_LM_WL" for c in seen), seen
    assert "7d" in out


def test_handle_stats_typo_says_so_instead_of_all_time(monkeypatch):
    monkeypatch.setattr(bc, "_query_trades",
                        lambda *a, **k: pytest.fail("ne doit pas interroger"))
    out = bc._handle_stats(None, "7dd PFW_TP50S30")
    assert "❓" in out
    assert "All-time" not in out


def test_error_names_only_the_offending_token(monkeypatch):
    """Sur `7d PFW_TP99S99` le `7d` etait compris : ne pas l'accuser."""
    monkeypatch.setattr(bc, "_query_trades",
                        lambda *a, **k: pytest.fail("ne doit pas interroger"))
    out = bc._handle_stats(None, "7d PFW_TP99S99")
    assert "PFW_TP99S99" in out
    assert "<b>7d PFW_TP99S99</b>" not in out
