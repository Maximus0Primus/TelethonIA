"""v14e.92 — vérifie qu'un bras MAIN est câblé aux QUATRE endroits nécessaires.

Une promotion en main n'est pas un seul geste. Le 11/08, `PFW_TP50_SL30_LM_WL`
a été déployé avec 3 des 4 branchements faits, et l'alerte Telegram a affiché
**bankroll $0** : le seed avait été écrit dans `rt_bankroll.strategy_bankrolls`
(dict plat) alors que les lectures passent d'abord par
`rt_bankroll.strategy_bankrolls_per_chain[chain]` (`safe_scraper.py:1003`), qui
n'a le dict plat qu'en *fallback legacy*. Rien ne plantait — le bras tradait,
seul le chiffre affiché était faux. C'est exactement le type de panne
silencieuse que ce projet paie cher.

Les quatre endroits :
  1. `strategies.py`                                    — la stratégie existe
  2. `rt_trade_config.hybrid_strategy.allocations`      — elle est allouée
  3. `rt_bankroll.strategy_bankrolls_per_chain[chain]`  — **celui qui est lu**
  4. `rt_bankroll.strategy_bankrolls`                   — fallback legacy
Et, pour toute stratégie clonée d'une variante d'évaluation :
  5. `rt_trade_config.strategy_overrides`               — sinon ce n'est pas
     la même stratégie que celle qui a été mesurée.

Usage:  python scripts/check_main_arm_wiring.py [--chain solana]
Sortie: code 0 si tout est câblé, 1 sinon (utilisable en garde pre-deploy).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCRAPER = Path(__file__).resolve().parent.parent / "scraper"
sys.path.insert(0, str(SCRAPER))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(SCRAPER / ".env")
from supabase import create_client  # noqa: E402

import strategies as S  # noqa: E402

# Préfixes de chaîne: sur solana, tout ce qui n'est pas préfixé ETH_/BSC_/BASE_.
_CHAIN_PREFIX = {"ethereum": "ETH_", "bsc": "BSC_", "base": "BASE_"}


def _for_chain(name: str, chain: str) -> bool:
    pref = _CHAIN_PREFIX.get(chain)
    if pref:
        return name.startswith(pref)
    return not any(name.startswith(p) for p in _CHAIN_PREFIX.values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", default="solana")
    a = ap.parse_args()

    sb = create_client(os.environ["SUPABASE_URL"],
                       os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    cfg = (sb.table("scoring_config").select("rt_trade_config")
           .eq("id", 1).execute().data[0]["rt_trade_config"])
    br = (sb.table("rt_bankroll").select("strategy_bankrolls,"
                                         "strategy_bankrolls_per_chain")
          .eq("id", 1).execute().data[0])

    allocations = ((cfg.get("hybrid_strategy") or {}).get("allocations") or {})
    overrides = cfg.get("strategy_overrides") or {}
    flat = br.get("strategy_bankrolls") or {}
    per_chain = (br.get("strategy_bankrolls_per_chain") or {}).get(a.chain) or {}

    arms = sorted(n for n in allocations if _for_chain(n, a.chain))
    print(f"{len(arms)} bras main sur {a.chain}\n")
    print(f"  {'bras':<34}{'strat':>7}{'alloc':>7}{'per_chain':>11}"
          f"{'flat':>7}{'solde':>9}")
    print("  " + "-" * 76)

    problemes: list[str] = []
    for n in arms:
        in_code = n in S.STRATEGIES
        in_pc = n in per_chain
        in_flat = n in flat
        solde = float((per_chain.get(n) or flat.get(n) or {}).get("balance", 0))
        # Le solde LU est celui de per_chain; flat n'est qu'un fallback legacy.
        ok = in_code and in_pc
        print(f"  {n[:34]:<34}{'ok' if in_code else 'MANQUE':>7}"
              f"{'ok':>7}{'ok' if in_pc else 'MANQUE':>11}"
              f"{'ok' if in_flat else '-':>7}{solde:>9.0f}")
        if not in_code:
            problemes.append(f"{n}: absent de strategies.py")
        if not in_pc:
            problemes.append(
                f"{n}: absent de strategy_bankrolls_per_chain[{a.chain}] "
                f"-> l'alerte affichera un bankroll a 0")
        if in_code and not ok:
            pass

    # Un clone de variante doit porter son override, sinon il mesure autre chose.
    print()
    for n in arms:
        filt = S.STRATEGY_FILTERS.get(n) or {}
        if filt.get("kol_whitelist") and n not in overrides:
            problemes.append(
                f"{n}: whitelist declaree mais aucun strategy_overrides "
                f"-> mode d'evaluation different de celui mesure")

    if problemes:
        print("PROBLEMES:")
        for p in problemes:
            print(f"  !! {p}")
        return 1
    print("Tous les bras main sont cables aux 4 endroits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
