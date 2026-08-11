"""v14e.91 — une `kol_whitelist` explicite prime sur `kol_chain_blacklist`.

Sans ça, un bras à whitelist perd silencieusement les KOL bannis de sa liste et
ne peut PAS reproduire les chiffres qui ont justifié sa création. Cas concret:
`PFW_TP50_SL30_LM_WL` porte 30 KOL dont 9 bannis (`mad_apes_gambles`, `zcallz`,
`unemployedDegen`… parmi ses plus gros contributeurs) — il aurait tourné sur 21
en prétendant en tester 30.

La non-régression compte autant: les 13 autres bras main ne déclarent aucune
whitelist et doivent continuer à respecter la blacklist à la lettre.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import paper_trader  # noqa: E402
from paper_trader import _strategy_overrides_blacklist  # noqa: E402


class TestWhitelistOverridesBlacklist(unittest.TestCase):
    def setUp(self):
        self._orig = dict(paper_trader.STRATEGY_FILTERS)
        paper_trader.STRATEGY_FILTERS["_T_AVEC_WL"] = {
            "chain": "solana", "kol_whitelist": ["banni_A", "propre_B"]}
        paper_trader.STRATEGY_FILTERS["_T_SANS_WL"] = {"chain": "solana"}
        paper_trader.STRATEGY_FILTERS["_T_WL_RESPECTE_BAN"] = {
            "chain": "solana", "kol_whitelist": ["banni_A"],
            "respect_chain_blacklist": True}

    def tearDown(self):
        paper_trader.STRATEGY_FILTERS.clear()
        paper_trader.STRATEGY_FILTERS.update(self._orig)

    def test_whitelist_prime_sur_le_ban(self):
        self.assertTrue(_strategy_overrides_blacklist("_T_AVEC_WL", "banni_A"))

    def test_kol_hors_whitelist_ne_passe_pas(self):
        """Le privilège vaut pour les KOL listés, pas pour toute la stratégie."""
        self.assertFalse(_strategy_overrides_blacklist("_T_AVEC_WL", "autre_C"))

    def test_strategie_sans_whitelist_inchangee(self):
        """NON-RÉGRESSION: tout le deck actuel est dans ce cas."""
        self.assertFalse(_strategy_overrides_blacklist("_T_SANS_WL", "banni_A"))

    def test_opt_out_explicite(self):
        """Un bras témoin peut redemander à subir la blacklist."""
        self.assertFalse(
            _strategy_overrides_blacklist("_T_WL_RESPECTE_BAN", "banni_A"))

    def test_kol_vide_ne_passe_pas(self):
        self.assertFalse(_strategy_overrides_blacklist("_T_AVEC_WL", ""))

    def test_strategie_inconnue_ne_casse_pas(self):
        self.assertFalse(_strategy_overrides_blacklist("_INCONNUE", "banni_A"))

    def test_le_bras_reel_porte_bien_ses_30_kols(self):
        """Le bras main déployé doit whitelister ses KOL bannis, sinon il ne
        reproduit pas la mesure qui l'a justifié."""
        paper_trader.STRATEGY_FILTERS.clear()
        paper_trader.STRATEGY_FILTERS.update(self._orig)
        wl = paper_trader.STRATEGY_FILTERS["PFW_TP50_SL30_LM_WL"]["kol_whitelist"]
        self.assertEqual(len(wl), 30)
        for kol in ("mad_apes_gambles", "zcallz", "unemployedDegen"):
            self.assertIn(kol, wl)
            self.assertTrue(
                _strategy_overrides_blacklist("PFW_TP50_SL30_LM_WL", kol),
                f"{kol} serait perdu par le bras main")


if __name__ == "__main__":
    unittest.main()
