"""v14e.95 — Sizing et risk management, sur les distributions REELLES du deck.

Question user (12/08): "dans les memecoins il y a deux choses importantes, le
risk management et le sizing — comment les appliquer pour ameliorer les gains ?"

Ce script ne recite pas Kelly, il le calcule sur les trades reellement encaisses,
et il verifie surtout la chose qui rend Kelly INAPPLICABLE ici: le plafond de
liquidite. Trois sorties:

  1. KELLY EMPIRIQUE. f* qui maximise la croissance log sur la distribution
     reelle, pas sur une gaussienne. Les memecoins ont une queue epaisse: la
     formule fermee (p*b - q)/b surestime f* de plusieurs points.
  2. LE PLAFOND MORD AVANT KELLY. La mise est bornee par la liquidite du token
     (~$100), pas par le bankroll. Des que f x bankroll > $100, f cesse d'etre
     un choix: on est cappe par le marche. Le script dit a partir de quel
     bankroll ca arrive — c'est la vraie frontiere.
  3. RISQUE DE RUINE ET DRAWDOWN par bootstrap: 2 000 re-tirages de l'ordre des
     trades. L'ordre reel n'est qu'UNE realisation; la ruine se mesure sur la
     distribution des chemins, jamais sur le chemin observe.

⚠️ Toutes les mesures sont faites AUSSI en NET (friction Solana -3.5 pp/trade,
`solana_fees_per_trade`, plus le drift paper->live -1.90 pp). `pnl_pct` en paper
est BRUT: raisonner en brut surestime f* et sous-estime la ruine.

Usage:
    python scripts/sizing_risk_analysis.py [--cache chemin.pkl] [--friction 5.4]
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

SCRAPER = Path(__file__).resolve().parent.parent / "scraper"
sys.path.insert(0, str(SCRAPER))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(SCRAPER / ".env")

PNL_CAP = 20.0
LIQUIDITE_MAX = 100.0     # $ par token (plafond de capacite documente)


def kelly_empirique(r: np.ndarray, grille=None) -> tuple[float, np.ndarray, np.ndarray]:
    """f* qui maximise la croissance log MOYENNE sur la distribution reelle.

    On borne f a 0.99: a f=1 un seul -100 % ruine definitivement, et log(0)
    diverge. C'est aussi la lecon f=1 -> -99 % du projet.
    """
    if grille is None:
        grille = np.linspace(0.005, 0.99, 199)
    g = np.array([np.mean(np.log1p(np.clip(f * r, -0.999999, None))) for f in grille])
    return float(grille[int(np.argmax(g))]), grille, g


def bootstrap(r: np.ndarray, f: float, n_paths=2000, n_trades=None, rng=None):
    """Distribution des chemins: multiple final, pire drawdown, ruine.

    Re-tire l'ORDRE des trades. L'ordre observe n'est qu'un tirage parmi
    beaucoup; conclure sur lui seul, c'est confondre chance et strategie.
    """
    rng = rng or np.random.default_rng(20260812)
    n = n_trades or len(r)
    idx = rng.integers(0, len(r), size=(n_paths, n))
    steps = np.log1p(np.clip(f * r[idx], -0.999999, None))
    eq = np.cumsum(steps, axis=1)
    fin = np.exp(eq[:, -1])
    run_max = np.maximum.accumulate(eq, axis=1)
    dd = 1.0 - np.exp(eq - run_max)
    return {
        "median": float(np.median(fin)),
        "p05": float(np.percentile(fin, 5)),
        "p95": float(np.percentile(fin, 95)),
        "dd_median": float(np.median(dd.max(axis=1))),
        "dd_p95": float(np.percentile(dd.max(axis=1), 95)),
        "ruine_50": float(np.mean(fin < 0.5)),
        "ruine_80": float(np.mean(fin < 0.2)),
    }


def analyse(nom: str, r: np.ndarray, friction: float, horizon: int):
    net = r - friction / 100.0
    print("\n" + "=" * 76)
    print(f"{nom}   n = {len(r)} trades")
    print("=" * 76)
    for lab, x in (("BRUT (paper)", r), (f"NET (-{friction:.1f} pp/trade)", net)):
        wr = float((x > 0).mean())
        print(f"  {lab:<26} EV {100*x.mean():+6.2f} %   mediane {100*np.median(x):+7.2f} %"
              f"   WR {100*wr:4.1f} %   meilleur {100*x.max():+8.1f} %")

    for lab, x in (("BRUT", r), ("NET", net)):
        fstar, grille, g = kelly_empirique(x)
        # Kelly "manuel" (formule fermee) pour montrer l'ecart du a la queue
        gains, pertes = x[x > 0], -x[x <= 0]
        b = gains.mean() / pertes.mean() if len(pertes) and pertes.mean() > 0 else np.nan
        p = float((x > 0).mean())
        f_ferme = (p * b - (1 - p)) / b if b and not np.isnan(b) else np.nan
        print(f"\n  [{lab}] Kelly EMPIRIQUE f* = {fstar:.3f}"
              f"   |  formule fermee (p*b-q)/b = {f_ferme:+.3f}"
              f"   |  ecart {f_ferme - fstar:+.3f}")
        if fstar <= 0.02:
            print("        -> f* colle a zero: sur cette distribution, TOUTE mise "
                  "significative detruit le capital.")
        print(f"  {'f':>7}{'multiple median':>17}{'p05':>9}{'p95':>9}"
              f"{'DD median':>11}{'DD p95':>9}{'P(-50%)':>9}{'P(-80%)':>9}")
        for f in (0.02, 0.05, 0.10, 0.15, 0.25, 0.50, 1.00):
            s = bootstrap(x, f, n_trades=horizon)
            print(f"  {f:>7.2f}{s['median']:>17.2f}{s['p05']:>9.2f}{s['p95']:>9.2f}"
                  f"{100*s['dd_median']:>10.0f}%{100*s['dd_p95']:>8.0f}%"
                  f"{100*s['ruine_50']:>8.0f}%{100*s['ruine_80']:>8.0f}%")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="")
    ap.add_argument("--friction", type=float, default=5.4,
                    help="pp retires par trade: 3.5 frais Solana + 1.9 drift")
    ap.add_argument("--horizon", type=int, default=200,
                    help="nb de trades simules par chemin (~50 jours a 4/j)")
    a = ap.parse_args()

    # --- 1. les bras REELS du deck -------------------------------------------
    from supabase import create_client
    sb = create_client(os.environ["SUPABASE_URL"],
                       os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    deck = ["PF_TP50_SL40_S35", "PF_BE25_TP80_SL30", "PFW_TP50_SL30_LM_WL"]
    rows = (sb.table("paper_trades")
            .select("strategy,token_address,pnl_pct,created_at")
            .in_("strategy", deck).eq("chain", "solana").neq("status", "open")
            .gte("created_at", "2026-08-06").limit(5000).execute().data)
    par_strat: dict[str, dict] = {}
    for r in rows:
        if r["pnl_pct"] is None or float(r["pnl_pct"]) > PNL_CAP:
            continue
        d = par_strat.setdefault(r["strategy"], {})
        k = r["token_address"]
        if k not in d or r["created_at"] < d[k][0]:
            d[k] = (r["created_at"], float(r["pnl_pct"]))
    for s in deck:
        if s in par_strat and len(par_strat[s]) >= 15:
            analyse(f"{s}  [REEL depuis le 06/08]",
                    np.array([v[1] for v in par_strat[s].values()]),
                    a.friction, a.horizon)

    # --- 2. le meilleur bras sur 4 MOIS (profondeur) -------------------------
    if a.cache and Path(a.cache).exists():
        # Cache LOCAL ecrit par regime_rotation_test.py (jamais une source
        # externe): il evite de re-paginer 1,2 M lignes sur une table qui
        # timeout facilement.
        ded = pickle.loads(Path(a.cache).read_bytes())
        for cible in ("BE15_LOCK5_TP50_SL30", "FAST_TP50_SL30_LAZYMED",
                      "FAST_TP50_SL30"):
            v = np.array([d[3] for d in ded if d[0] == cible])
            if len(v) >= 100:
                analyse(f"{cible}  [SHADOW 4 mois, SANS filtre d'entree]",
                        v, a.friction, a.horizon)
                break

    # --- 3. LA FRONTIERE QUI REND TOUT CE QUI PRECEDE THEORIQUE --------------
    print("\n" + "=" * 76)
    print("PLAFOND DE LIQUIDITE — a partir de quel bankroll f cesse d'etre un choix")
    print("=" * 76)
    print(f"  Mise maximale par token (liquidite memecoin) : ${LIQUIDITE_MAX:.0f}")
    print(f"  {'bankroll':>10}{'mise voulue a f=0.10':>24}{'mise POSSIBLE':>16}"
          f"{'f REEL':>9}")
    for B in (500, 1000, 2000, 5000, 10000, 50000):
        voulue = 0.10 * B
        possible = min(voulue, LIQUIDITE_MAX)
        print(f"  ${B:>9,}{'$' + format(voulue, ',.0f'):>24}"
              f"{'$' + format(possible, ',.0f'):>16}{possible / B:>9.3f}")
    print("\n  -> Des $1 000, la mise voulue par Kelly EGALE deja le plafond du marche.")
    print("     Au-dela, f n'est plus un parametre qu'on choisit: il est impose par")
    print("     la liquidite, et il TOMBE quand le bankroll monte. C'est exactement")
    print("     le plafond de +$23/j documente (capacity_ceiling_23_per_day).")
    print("     => le seul levier qui reste est n (trades/jour), pas la mise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
