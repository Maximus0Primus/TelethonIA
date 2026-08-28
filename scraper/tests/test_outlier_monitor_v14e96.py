"""v14e.96 — le nightly outlier monitor ALERTAIT SUR SES PROPRES CRASHES.

Runs 31567923907 (12/08) et 31671860468 (13/08) : le script meurt sur

    postgrest.exceptions.APIError: {'message': 'canceling statement due to
    statement timeout', 'code': '57014'}

...et le step d'alerte, garde `if: failure()`, poste quand meme

    PAPER-LIVE OUTLIER (sync=True)
     outlier(s) |L-P|>10pp on  pairs
    First:  /  Δ=pp

Tous les champs sont VIDES parce que le script est mort AVANT d'ecrire
$GITHUB_OUTPUT. Un plantage d'infra s'est donc deguise en signal de bug de
logique metier -- alors que le live est coupe depuis le 05/06 et qu'il y a
ZERO ligne `rt_live` : aucun outlier paper<->live n'est meme possible.

Deux fautes, deux gardes ici :

1. `fetch_all` tirait TOUT `paper_trades` sur 48 h pour n'apparier qu'une
   poignee de lignes live. La grille shadow est passee de ~10 k a ~33 k
   lignes/jour (bras v14e.93-95) => 64 k lignes, 64 pages, OFFSET sans
   `order()` => statement timeout. Le cote live BORNE le travail : on le lit
   d'abord, et sans live on ne touche pas au cote paper.
2. L'alerte doit distinguer « outlier trouve » de « monitor plante ».
"""

import re
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[2]
SCRIPT = RACINE / "scripts" / "nightly_outlier_monitor.py"
WORKFLOW = RACINE / ".github" / "workflows" / "nightly-outlier-monitor.yml"


# --------------------------------------------------------------------------
# 1. Le script : le cote live borne le travail, la pagination est ordonnee
# --------------------------------------------------------------------------

def test_le_cote_live_est_lu_en_premier_et_filtre_serveur():
    """Sans filtre `rt_live` server-side, on rapatrie 64 k lignes pour rien."""
    src = SCRIPT.read_text(encoding="utf-8")
    assert "eq_source=\"rt_live\"" in src or "eq_source='rt_live'" in src, (
        "le monitor doit filtrer le cote live EN SQL (eq_source='rt_live'), "
        "pas rapatrier toute la grille shadow puis trier en Python"
    )


def test_sans_trade_live_le_cote_paper_nest_jamais_tire():
    """Live coupe depuis le 05/06 => la lecture lourde doit etre court-circuitee."""
    src = SCRIPT.read_text(encoding="utf-8")
    corps = src.split("def main(")[1]
    avant_paper = corps.split("in_token_address")[0]
    assert re.search(r"if not live", avant_paper), (
        "main() doit sortir tot quand il n'y a aucun trade live dans la fenetre "
        "— sinon on scanne toute la grille shadow pour construire 0 paire"
    )


def test_pagination_ordonnee():
    """OFFSET sans ORDER BY = pages instables (doublons/trous) en plus du cout."""
    src = SCRIPT.read_text(encoding="utf-8")
    assert ".order(" in src, (
        "fetch_all pagine avec .range() : sans .order() stable, PostgREST peut "
        "renvoyer deux fois la meme ligne et en sauter une autre"
    )


def test_retry_sur_statement_timeout():
    """57014 est chronique sur ce projet (~13/h) : une seule tentative ne suffit pas."""
    src = SCRIPT.read_text(encoding="utf-8")
    assert "57014" in src, (
        "le monitor doit retenter sur statement timeout (57014) au lieu de "
        "mourir et de declencher une fausse alerte trading"
    )


def test_outputs_toujours_emis_meme_sans_outlier():
    """Le workflow doit pouvoir lire 'le script est alle au bout'."""
    src = SCRIPT.read_text(encoding="utf-8")
    assert "monitor_status=completed" in src, (
        "le script doit emettre monitor_status=completed dans $GITHUB_OUTPUT "
        "pour que l'alerte distingue 'pas d'outlier' de 'crash'"
    )


# --------------------------------------------------------------------------
# 2. Le workflow : ne jamais poster un verdict trading sur un crash
# --------------------------------------------------------------------------

def _step(nom: str) -> dict:
    """Renvoie le step `- name: <nom>` via un vrai parse YAML.

    Surtout pas de regex ligne-a-ligne ici : la condition est un scalaire
    replie (`if: >-`) etale sur plusieurs lignes, qu'un regex `if:\\s*(.+)`
    lirait comme la chaine `>-`.
    """
    yaml = pytest.importorskip("yaml")
    steps = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]["monitor"]["steps"]
    trouve = [s for s in steps if s.get("name") == nom]
    assert trouve, f"step '{nom}' absent du workflow"
    return trouve[0]


def test_alerte_outlier_pas_declenchee_par_un_crash():
    """LE BUG : `if: failure()` seul poste un verdict trading sur un plantage SQL."""
    condition = str(_step("Alert on sync=True outlier").get("if", ""))
    assert "sync_true_count" in condition, (
        f"le step d'alerte outlier est garde par `if: {condition.strip()}` : il "
        f"se declenche sur N'IMPORTE QUEL echec du job, y compris un timeout "
        f"SQL, et poste un message aux champs vides. Gardez-le sur "
        f"steps.monitor.outputs.sync_true_count > 0."
    )
    assert "monitor_status" in condition, (
        "gardez aussi sur monitor_status == 'completed' : des outputs absents "
        "ne doivent jamais etre lus comme '0 outlier'"
    )
    assert not re.fullmatch(r"\s*failure\(\)\s*", condition), (
        "condition revenue a `failure()` nu"
    )


def test_un_crash_a_sa_propre_alerte_distincte():
    """Un monitor mort doit se signaler — mais comme panne, pas comme outlier."""
    crash = _step("Alert on monitor crash")
    condition = str(crash.get("if", ""))
    assert "monitor_status" in condition, (
        "l'alerte de crash doit se declencher quand le script n'est PAS alle au "
        "bout (monitor_status != 'completed')"
    )
    assert "PAPER-LIVE OUTLIER" not in str(crash.get("run", "")), (
        "l'alerte de crash ne doit pas reutiliser le libelle d'outlier trading"
    )
