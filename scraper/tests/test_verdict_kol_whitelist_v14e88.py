"""v14e.88 — la recherche exhaustive KOL x sortie trouve un signal planté,
et ne trouve RIEN dans du bruit pur.

Le deuxième test est le plus important : le 11/08, un classement KOL x stratégie
a produit un « meilleur » à +23.19 % qui était sous son plancher de permutation.
Un chercheur de maximum qui ne sait pas dire « rien » est un générateur de faux
gagnants. On vérifie donc les deux sens.
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scraper"))

from analyze_mega_sweep import verdict_kol_whitelist  # noqa: E402


MOIS = ["2026-04", "2026-05", "2026-06", "2026-07"]


def _df(payloads):
    return pd.DataFrame([
        {"strategy": s, "kol_month_json": json.dumps(p)}
        for s, p in payloads.items()
    ])


def _bruit(rng, n_kols=20, n_strats=25):
    """Aucune structure : chaque (KOL, mois) tire une somme centrée sur zéro."""
    out = {}
    for s in range(n_strats):
        d = {}
        for k in range(n_kols):
            d[f"kol{k}"] = {m: [12, float(rng.normal(0, 3))] for m in MOIS}
        out[f"STRAT{s}"] = d
    return out


class TestVerdictKolWhitelist(unittest.TestCase):
    def _run(self, df):
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            verdict_kol_whitelist(df, Path("/tmp/_mega_sweep_extended.csv"),
                                  n_permutations=40)
        return buf.getvalue()

    def test_bruit_pur_taux_de_faux_positifs_calibre(self):
        """Sans signal, le verdict doit être négatif *presque toujours*.

        ⚠️ Ne PAS tester une seule réalisation : la procédure est stochastique
        et un seuil à p95 se trompe 5 % du temps par construction. Un test
        déterministe sur un tirage serait flaky et, pire, donnerait une fausse
        assurance. On mesure donc le TAUX de faux positifs sur 12 tirages :
        attendu ~0.6, on tolère jusqu'à 4. Un null casse (comme la v1, qui
        permutait les KOL identiquement pour toutes les sorties) se detecte
        par l'autre sens — cf. test_signal_plante_est_trouve.
        """
        faux = 0
        for seed in range(12):
            out = self._run(_df(_bruit(np.random.default_rng(seed))))
            if "DEPASSE le plancher" in out:
                faux += 1
        self.assertLessEqual(
            faux, 4,
            f"{faux}/12 faux positifs sur du bruit pur: le plancher est trop bas")

    def test_signal_plante_est_trouve(self):
        """Trois KOL systématiquement rentables sur une sortie: doit ressortir."""
        rng = np.random.default_rng(11)
        payloads = _bruit(rng)
        # Signal franc et STABLE dans le temps: c'est ce que le walk-forward
        # est censé savoir distinguer d'un coup de chance sur un seul mois.
        for k in ("kol0", "kol1", "kol2"):
            payloads["STRAT0"][k] = {m: [12, 40.0] for m in MOIS}
        out = self._run(_df(payloads))
        self.assertIn("DEPASSE le plancher", out, out)
        self.assertIn("STRAT0", out, out)

    def test_signal_sur_un_seul_mois_ne_passe_pas(self):
        """Un mois exceptionnel ne doit PAS suffire (piège du top-1 par outlier).

        La whitelist se construit sur les mois ANTERIEURS: un gain concentre sur
        le premier mois n'est jamais capitalisable en test.
        """
        rng = np.random.default_rng(13)
        payloads = _bruit(rng)
        for k in ("kol0", "kol1", "kol2"):
            payloads["STRAT0"][k] = {MOIS[0]: [12, 400.0]}
        out = self._run(_df(payloads))
        self.assertIn("AUCUNE config ne depasse le plancher", out, out)

    def test_famille_artefact_est_ecartee(self):
        """DTRAIL/TRAIL/DIP/SPLIT/BOND ne doivent JAMAIS remonter en tete.

        Le 1er run 4 mois a sorti TD2_BE5_TP120_SL44_T25 en n°1 faute de ce
        filtre: un artefact connu (slippage mal calibre x47) presente comme la
        meilleure strategie. `verdict_par_exit` l'ecartait deja, pas celui-ci.
        """
        rng = np.random.default_rng(3)
        payloads = _bruit(rng)
        payloads["DTRAIL9_ACT5_SL70"] = {
            f"kol{k}": {m: [12, 60.0] for m in MOIS} for k in range(20)}
        out = self._run(_df(payloads))
        self.assertNotIn("DTRAIL9_ACT5_SL70", out,
                         "famille artefact remontee en tete:\n" + out)
        self.assertIn("ecartees (famille artefact)", out, out)

    def test_colonne_absente_ne_casse_pas(self):
        """Sweep tournant sur un SHA anterieur: la section doit s'ignorer."""
        out = self._run(pd.DataFrame([{"strategy": "X"}]))
        self.assertEqual(out.strip(), "")

    def test_trop_peu_de_mois_est_signale(self):
        """Moins de 3 mois: pas de walk-forward possible, il faut le DIRE."""
        payloads = {"S": {"k0": {"2026-07": [12, 5.0]}}}
        out = self._run(_df(payloads))
        self.assertIn("mois seulement", out, out)


if __name__ == "__main__":
    unittest.main()
