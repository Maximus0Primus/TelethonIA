"""E30 — simulation du portefeuille 3 stratégies avec CAPITAL CONTRAINT.

Ce que les simulations précédentes ignoraient : avec 2.82 positions simultanées en
moyenne (p95 = 6, pic = 20), un petit capital ne peut pas ouvrir tous les trades.
Il faut modéliser le refus faute de capital libre, sinon on surestime.

Règles simulées
---------------
- mise = min(f × capital_total, plafond)  — le plafond ~$100 est la contrainte
  structurelle des memecoins (liquidité), pas un choix
- si capital LIBRE < mise → **trade sauté**, et on le compte
- coût fixe par trade en $ (pas en %) : à $20 de mise il pèse 0.65 %, à $100
  seulement 0.13 %. C'est ce qui pénalise les petits capitaux.
- le capital est immobilisé de created_at à exit_at (chevauchements réels)

    python scripts/_portfolio_sim.py
"""

from __future__ import annotations

import os
from datetime import datetime

COUT_FIXE = 0.13      # $ aller-retour (réseau + priorité), cf note mémoire
DERIVE = 0.035        # dérive live↔paper mesurée sur SL=30 % : -2 à -5 pp


def charger():
    from supabase import create_client
    cli = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    rows, off = [], 0
    while True:
        r = (cli.table("portfolio_trades_e30")
             .select("strategy,created_at,exit_at,pnl_pct")
             .order("created_at").range(off, off + 999).execute().data)
        if not r:
            break
        rows.extend(r)
        if len(r) < 1000:
            break
        off += 1000
    out = []
    for r in rows:
        t0 = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(r["exit_at"].replace("Z", "+00:00"))
        if t1 > t0:
            out.append((t0, t1, float(r["pnl_pct"])))
    out.sort(key=lambda x: x[0])
    return out


def simuler(trades, cap0, f, plafond, derive=0.0, cout=COUT_FIXE):
    """Renvoie (capital final, DD max, n pris, n sautés, courbe mensuelle)."""
    cap, libre = cap0, cap0
    peak, mdd = cap0, 0.0
    ouvertes = []          # (exit_at, mise, pnl)
    pris = saute = 0
    par_mois = {}

    for t0, t1, pnl in trades:
        # on ferme d'abord tout ce qui est arrivé à échéance
        for o in [o for o in ouvertes if o[0] <= t0]:
            gain = o[1] * (o[2] - derive) - cout
            cap += gain
            libre += o[1] + gain
            par_mois[o[0].strftime("%m")] = par_mois.get(o[0].strftime("%m"), 0) + gain
            peak = max(peak, cap)
            mdd = max(mdd, (peak - cap) / peak) if peak > 0 else mdd
            ouvertes.remove(o)

        mise = min(cap * f, plafond)
        if mise < 5 or libre < mise:      # pas assez de capital libre → on saute
            saute += 1
            continue
        libre -= mise
        ouvertes.append((t1, mise, pnl))
        pris += 1

    for o in ouvertes:                    # solde les positions restantes
        gain = o[1] * (o[2] - derive) - cout
        cap += gain
        par_mois[o[0].strftime("%m")] = par_mois.get(o[0].strftime("%m"), 0) + gain

    return cap, mdd, pris, saute, par_mois


MOIS = {"04": "avril", "05": "mai", "06": "juin", "07": "juillet", "08": "aout"}


def main():
    tr = charger()
    print(f"Portefeuille 3 strategies — {len(tr)} trades, "
          f"{tr[0][0].date()} -> {tr[-1][1].date()}\n")

    print("CAPITAL 100$ — mise initiale 20$ (f=0.20) — plafond 100$/position")
    print(f"{'scenario':<34}{'final':>10}{'DD':>8}{'pris':>7}{'sautes':>8}{'% rate':>8}")
    print("-" * 75)
    for lab, der in [("paper (sans derive live)", 0.0),
                     ("AVEC derive live -3.5pp", DERIVE)]:
        cap, mdd, pris, saute, pm = simuler(tr, 100.0, 0.20, 100.0, der)
        pct = 100 * saute / (pris + saute)
        print(f"{lab:<34}{cap:>9,.0f}${100*mdd:>7.1f}%{pris:>7}{saute:>8}{pct:>7.0f}%")
        print(f"{'':34}par mois: " + "  ".join(
            f"{MOIS[m]} {v:+,.0f}$" for m, v in sorted(pm.items())))
    print()

    print("Effet du CAPITAL DE DEPART (f=0.20, plafond 100$, avec derive -3.5pp)")
    print(f"{'capital':<12}{'final':>10}{'x':>7}{'DD':>8}{'pris':>7}{'sautes':>8}{'% rate':>8}")
    print("-" * 62)
    for c0 in (100, 200, 500, 1000, 2000, 5000):
        cap, mdd, pris, saute, _ = simuler(tr, float(c0), 0.20, 100.0, DERIVE)
        pct = 100 * saute / (pris + saute)
        print(f"{c0:<12,}{cap:>9,.0f}${cap/c0:>6.1f}x{100*mdd:>7.1f}%{pris:>7}{saute:>8}{pct:>7.0f}%")
    print()

    print("Effet de f a capital 100$ (plafond 100$, avec derive -3.5pp)")
    print(f"{'f':<8}{'mise dep.':>11}{'final':>10}{'DD':>8}{'pris':>7}{'% rate':>8}")
    print("-" * 52)
    for f in (0.10, 0.15, 0.20, 0.30, 0.50):
        cap, mdd, pris, saute, _ = simuler(tr, 100.0, f, 100.0, DERIVE)
        pct = 100 * saute / (pris + saute)
        print(f"{f:<8.2f}{100*f:>10.0f}${cap:>9,.0f}${100*mdd:>7.1f}%{pris:>7}{pct:>7.0f}%")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Ajouts: quel est l'optimum a PETIT capital, et quel est le risque reel ?
# ---------------------------------------------------------------------------
def simuler_fixe(trades, cap0, mise_fixe, derive=0.0, cout=COUT_FIXE):
    """Taille FIXE en $: pas de spirale de ruine, pas de composition."""
    cap, libre, peak, mdd = cap0, cap0, cap0, 0.0
    ouvertes, pris, saute = [], 0, 0
    serie_perte = pire_serie = 0
    for t0, t1, pnl in trades:
        for o in [o for o in ouvertes if o[0] <= t0]:
            g = o[1] * (o[2] - derive) - cout
            cap += g; libre += o[1] + g
            serie_perte = serie_perte + 1 if g < 0 else 0
            pire_serie = max(pire_serie, serie_perte)
            peak = max(peak, cap); mdd = max(mdd, (peak - cap) / peak) if peak > 0 else mdd
            ouvertes.remove(o)
        if libre < mise_fixe or cap < mise_fixe:
            saute += 1; continue
        libre -= mise_fixe; ouvertes.append((t1, mise_fixe, pnl)); pris += 1
    for o in ouvertes:
        cap += o[1] * (o[2] - derive) - cout
    return cap, mdd, pris, saute, pire_serie


if os.environ.get("EXTRA"):
    tr = charger()
    print("\n" + "=" * 78)
    print("OPTIMUM A PETIT CAPITAL — taille FIXE (pas de composition, pas de ruine)")
    print("=" * 78)
    print(f"{'capital':<10}{'mise':<8}{'final paper':>13}{'final derive':>14}{'DD':>8}{'% rate':>8}")
    print("-" * 62)
    for cap0 in (100, 150, 200, 300, 500):
        for mise in (5, 10, 20):
            if mise * 3 > cap0:
                continue
            p, _, pr, sa, _ = simuler_fixe(tr, float(cap0), float(mise), 0.0)
            d, mdd, pr2, sa2, _ = simuler_fixe(tr, float(cap0), float(mise), DERIVE)
            pct = 100 * sa2 / max(pr2 + sa2, 1)
            print(f"{cap0:<10,}{mise:<8}{p:>12,.0f}${d:>13,.0f}${100*mdd:>7.1f}%{pct:>7.0f}%")
    print("\n" + "=" * 78)
    print("RISQUE — pire serie de pertes consecutives et drawdown, capital 500$")
    print("=" * 78)
    for lab, der in [("paper", 0.0), ("avec derive -3.5pp", DERIVE)]:
        c, mdd, pr, sa, ps = simuler_fixe(tr, 500.0, 20.0, der)
        print(f"{lab:<22} final {c:>8,.0f}$   DD max {100*mdd:>5.1f}%   "
              f"pire serie de pertes: {ps} trades d'affilee")
