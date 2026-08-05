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
