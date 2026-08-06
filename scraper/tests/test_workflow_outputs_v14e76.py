"""v14e.76 — un step qui redirige stdout vers $GITHUB_OUTPUT ne doit pas
imprimer de commandes workflow (`::notice::`, `::warning::`, ...).

Le run 31054936700 (Mega Sweep Extended ETH, 05/08) est ROUGE alors que sa garde
de fraicheur avait correctement decide de sauter le sweep :

    ##[error]Unable to process file command 'output' successfully.
    ##[error]Invalid format '::notice::SKIP — 15 tokens ETH distincts sur 14j (seuil 20)'

La garde etait ecrite `python - <<'PY' >> "$GITHUB_OUTPUT"`, donc TOUT le stdout
partait dans le fichier d'outputs -- y compris la ligne `::notice::`, que le
runner refuse de parser comme une paire cle=valeur. Le workflow SOL, lui, ouvre
$GITHUB_OUTPUT explicitement depuis Python et n'a jamais eu le probleme.

Le test scanne les deux workflows de sweep pour que la forme fautive ne revienne
pas par copier-coller.
"""

import re
from pathlib import Path

import pytest

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
FICHIERS = sorted(WORKFLOWS.glob("mega-sweep-*.yml"))

# `>> "$GITHUB_OUTPUT"` accroche a une commande (heredoc python, echo groupe...)
REDIRECTION = re.compile(r'>>\s*"?\$\{?GITHUB_OUTPUT\}?"?')
COMMANDE_WORKFLOW = re.compile(r'::(notice|warning|error|group|debug)::')


def _blocs_run(texte: str) -> list[str]:
    """Decoupe grossierement le YAML en blocs `run:` (indentation-based)."""
    blocs, courant, indent_run = [], [], None
    for ligne in texte.splitlines():
        if courant is not None and indent_run is not None:
            nue = ligne.strip()
            indent = len(ligne) - len(ligne.lstrip())
            if nue and indent <= indent_run and not ligne.lstrip().startswith("#"):
                blocs.append("\n".join(courant))
                courant, indent_run = None, None
            else:
                courant.append(ligne)
                continue
        if re.match(r"\s*run:\s*\|", ligne):
            indent_run = len(ligne) - len(ligne.lstrip())
            courant = []
    if courant:
        blocs.append("\n".join(courant))
    return blocs


@pytest.mark.parametrize("chemin", FICHIERS, ids=lambda p: p.name)
def test_pas_de_commande_workflow_dans_github_output(chemin):
    assert FICHIERS, "aucun workflow mega-sweep-*.yml trouve"
    for bloc in _blocs_run(chemin.read_text(encoding="utf-8")):
        if REDIRECTION.search(bloc) and COMMANDE_WORKFLOW.search(bloc):
            faute = COMMANDE_WORKFLOW.search(bloc).group(0)
            pytest.fail(
                f"{chemin.name}: un step redirige stdout vers $GITHUB_OUTPUT et "
                f"imprime {faute} — le runner rejettera le fichier d'outputs. "
                f"Ecrire dans $GITHUB_OUTPUT depuis Python a la place."
            )
