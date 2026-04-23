---
name: bat_gamble audit — likely structural, limited-confidence data
description: High-volume KOL blacklisted pre-v108. 6617 shadows × 124 strats over 5d (Mar 24-29) — zero profitable strategy IN THE STORED PNL. But data predates price_ticks (Apr 6) and eval_history (Apr 17) so cannot tick-replay. Structural signal (median -50% 4/5 days) is robust regardless.
type: project
---

# bat_gamble — verdict Apr 23 2026 (revised)

**TL;DR** : signal structurellement mauvais (4/5 jours médiane −50%), mais les
données datent d'avant les price_ticks (Apr 6) — on ne peut pas tick-replay
avec le slip model + Jupiter actuels. Confiance élevée sur "pas réhabilitable
sans refetch OHLCV + re-sim", mais pas de certitude absolue 0-strat.

## Data caveat

**bat_gamble = Mar 24-29, 2026**. Infrastructure disponible à l'époque :
- `paper_trades.pnl_pct` stockés par paper_trader v108 : DexScreener poll 15-60s
- Slip model v108 (buckets 5k/20k/50k avec valeurs différentes de v14e.6 actuel)
- Pas de Jupiter Ultra (arrivé v121, Apr 8) → pas de positive slippage sur pumps
- **Pas de price_ticks** (créé Apr 6, v118)
- **Pas de eval_history** (créé Apr 17, v138)

Donc l'analyse ci-dessous = replay du paper v108, pas un ground-truth tick replay.
Biais possibles : TP ratés sur pumps courts entre deux polls, SL trigger sur wicks
qu'un tick-replay ne confirmerait pas.

## Volume
- 20 mains (TP100_SL50) + 6617 shadows sur 5 jours (Mar 24-29, 2026)
- 124 stratégies différentes tournées en shadow
- ~1323 shadows/jour = volume énorme, ~10× la moyenne

## Performance par strat (top 10, N≥20)

| Strategy | N | avg% | WR% | avg_loss% |
|---|---|---|---|---|
| TRAIL10_TP50_SL30 | 52 | −7.2 | 44 | −27.7 |
| TRAIL10_TP100_SL50 | 52 | −7.6 | 48 | −39.0 |
| TP30_SL10 | 52 | −8.4 | 25 | −17.9 |
| TRAIL10_TP70_SL50 | 52 | −9.0 | 48 | −39.0 |
| TP50_SL15 | 52 | −9.1 | 23 | −22.5 |
| TP100_SL50 (main) | 20 | −10.9 | 25 | −33.7 |

**Aucune ≥ 0%.** La plus proche (TRAIL10_TP50_SL30) perd encore −7%/trade × 52 trades = −$180-200 sur 5j à $52 pos.

## Pattern structural

Par jour (TP100_SL50 main):

| Day | N | avg% | median% | WR | pct_50up |
|---|---|---|---|---|---|
| 2026-03-24 | 9 | +27.6% | +83.7% | 56% | 56% |
| 2026-03-25 | 2 | −30.6% | −53.7% | 0% | 0% |
| 2026-03-27 | 18 | −38.3% | −54.0% | 11% | 6% |
| 2026-03-28 | 16 | −5.1% | −43.0% | 31% | 25% |
| 2026-03-29 | 16 | −23.0% | −53.5% | 19% | 19% |

**1 jour vert isolé, 4 jours rouges consécutifs.** Médianes −43 à −54% → les tokens crashent structurellement.

## Pourquoi volume ne compense pas

`avg_loss_pct` minimum observé = **−17.8%** (TP30_SL10 tight). Plafond SL ne sauve pas car les tokens chutent en gap au call. Même le SL à −10% se fait engulfer régulièrement.

`avg_win_pct` hétérogène (+3% à +73%) → les rares wins ne compensent pas la régularité des pertes.

## Conclusion

- Bat_gamble fait de la **quantity-over-quality** extrême
- Aucun paramètre de TP/SL/horizon/trail n'a rendu ses calls profitables **dans le
  paper v108**. Pas re-testé avec slip v14e.6 + Jupiter Ultra.
- Le volume est un signal NÉGATIF (plus de garbage, pas plus d'alpha)
- **Ne pas réhabiliter automatiquement**. Pour vérifier avec rigueur :
  1. Extraire les CAs depuis `kol_mentions` pour Mar 24-29
  2. Refetch OHLCV historique (DexPaprika → Birdeye fallback)
  3. Tourner `sim.py --from-trades` avec strats actuelles + slip v14e.6
  4. Si ça reste ≤ 0% sur N≥50 en re-sim → blacklist confirmée
- La médiane −50% sur 4/5 jours reste dure à sauver même avec un meilleur sim
  (pas un artefact de slip model), donc probabilité de réhabilitation très faible.
