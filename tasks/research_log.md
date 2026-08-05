# Research Log — TelethonIA Trading Bot

> Journal de toutes les analyses, simulations, décisions et idées.
> Format : Date · Observation · Décision · Résultat

---

## Monte Carlo — Projection 30 jours (Feb 28, 2026)

**Paramètres :**
- 20 calls/jour (KOLs whitelist 60% WR, fenêtre 30j)
- $20/call total, split hybrid 70/30
- TP50_SL30 ($14) : WR 65% assumé → EV = +$7×0.65 − $4.20×0.35 = +$3.08/call
- TP100_SL50 ($6) : WR 40% hit 2x → EV = +$6×0.40 − $3×0.60 = +$0.60/call
- EV total par call : **$3.68**

**Résultats Monte Carlo :**
| Scénario | PnL 30j |
|----------|---------|
| Pessimiste (P5, -2σ) | +$1 869 |
| Bas (P16, -1σ) | +$2 038 |
| **Espérance (moyenne)** | **+$2 208** |
| Haut (P84, +1σ) | +$2 378 |
| Optimiste (P95, +2σ) | +$2 547 |

**EV/jour : $73.60 | Std 30j : ±$169**

⚠️ Simulation path-independent (ignore SL hits avant TP sur le chemin).
⚠️ Basé sur WR historique — pas garanti si marché change.

---

## Simulation KOL Top 14 — $20/call hybride (Feb 28, 2026)

**Données réelles kol_call_outcomes, 30 derniers jours :**
- Feb 16-19 : 31-130 calls/jour → **$136-$271/jour**
- Feb 20-24 : 15-21 calls/jour → $58-$136/jour
- Feb 25-28 : 1-24 calls/jour → $12-$78/jour (API credits morts)

**Conclusion :** $150/jour est réaliste sur les périodes avec données API propres.
Volume normal : 20-30 calls/jour = **$50-120/jour attendu**.

---

## Stratégies — Résultats réels RT (Feb 26-28, 2026)

N=129 trades fermés sur 3 jours VPS, source='rt' :

| Stratégie | N | WR | ROI moyen | Verdict |
|-----------|---|----|-----------|---------|
| TP50_SL30 | 18 | 50% | +10.7% | ✅ Garder |
| TP30_SL50 | 4 | 50% | +1.9% | ❌ Trop petit TP, EV théorique -10% |
| TP100_SL30 | 18 | 11% | -13.4% | ❌ SL trop serré pour 2x |
| QUICK_SCALP | 11 | 18% | -3.4% | ❌ 9/11 timeouts |
| FRESH_MICRO | 9 | 44% | -11.4% | ❌ Négatif |
| SCALE_OUT | 47 | 6% | -32.2% | ❌ Désactivé |
| MOONBAG | 13 | 8% | -57.2% | ❌ Désactivé |
| WIDE_RUNNER | 10 | 0% | -70.4% | ❌ Désactivé |

**Config actuelle (v77) :** Hybrid TP50_SL30 70% + TP100_SL50 30%
- TP100_SL50 = pas encore de données (créé Feb 28). SL -50% pour laisser respirer le token.

---

## Pourquoi WR bot (50%) < WR KOL (60%+)

3 causes structurelles :
1. **Path dependency** : KOL WR = token JAMAIS atteint 1.5x. Bot WR = atteint 1.5x SANS toucher -30% avant. Token qui dip -32% puis monte = ✅ KOL, ❌ SL bot.
2. **Prix d'entrée décalé** : Bot achète quelques secondes après le call → +5-15% déjà. TP effectivement plus loin.
3. **Horizon 24h** : KOL WR mesuré sur fenêtre longue. Token qui hit +50% à H+26 = ✅ KOL, ❌ timeout bot.

→ Écart de ~10% est normal et attendu.

---

## Filtre KOL — Optimisation (Feb 28, 2026)

**Simulation 7j sur kol_call_outcomes :**
| Seuil WR | KOLs | Trades | ROI |
|----------|------|--------|-----|
| 20% | 13 | 129 | -14.6% |
| 50% | 10 | 110 | -11.0% |
| **60%** | **6-14** | **64-140** | **-3.7%** ← optimal |

**Config actuelle :** wr_threshold=0.60, min_calls=3, lookback_days=30, return_threshold=1.5x

**14 KOLs approuvés (30j rolling) :**
papicall (88%), archercallz (86%), invacooksclub (71%), degenncabal (67%),
MaybachGambleCalls (64%), LittleMustachoCalls (64%), spidersjournal (63%),
certifiedprintor (62%), DegenSeals (61%), eveesL (60%), legerlegends (60%)...

**Changement important v77 :** Fenêtre glissante 30j au lieu de all-time.
→ Exclut KOLs bons historiquement mais mauvais récemment (ex: kweensjournal all-time 63% mais 25% sur 7j).

---

## ML — Modèle RT (Feb 28, 2026)

**Statut :** Déployé (v66, entraîné Feb 28)
- n_train=90, n_test=39 → ⚠️ Trop petit, besoin 200+ pour robustesse
- Direction accuracy : 74.4%
- Baseline avg PnL : -17.2% → Selective avg PnL : +27.8% → **Edge +45%**

**Top features (importance) :**
1. rt_liquidity_usd (103) — liquidité au moment du call
2. hour_of_day (72) — certaines heures plus profitables
3. rt_volume_24h (70)
4. kol_score (66) + kol_win_rate (51)

**Fonctionnement :**
- `scoring_mode=hybrid` : ML prédit avg PnL → multiplicateur [0.3, 2.0] sur score
- RT : avg_pred > +2% → position ×1.5 | avg_pred < -2% → position ×0.5
- Ne bloque JAMAIS le trade — ajuste seulement la taille

**A/B tracking (v77) :** `ml_pred` column ajoutée à paper_trades.
→ Maintenant possible de comparer trades ML-boosted vs ML-réduits.

---

## Problèmes critiques résolus (Feb 28, 2026)

| Problème | Impact | Fix |
|----------|--------|-----|
| outcomes.yml cancel-in-progress + cron 1h | Jobs killed avant fin | Cron → 2h |
| Helius/Birdeye credits épuisés 23/02 | whale_count=NULL, scores ~7/100 | Reset 1er mars |
| SCALE_OUT 6% WR, -32% ROI | -$59 en 24h | Désactivé |
| TP100_SL30 dans hybrid malgré multiplier=0 | -13.4% ROI | Remplacé par TP100_SL50 |
| rt_trade_config stocké comme array JSONB | Merges silencieux | Réparé |
| Whitelist KOL all-time (KOLs stale approuvés) | Mauvais KOLs filtrés | Rolling 30j |
| CA identity collision (58% snapshots) | Mauvaises métriques | Fix v40 |
| Phantom labels (16k faux labels) | ML poisonné | Fix v34 |

---

## Bankroll Compound — Config (Feb 28, 2026)

**Capital départ : $100. Mode bankroll activé (v77b).**

```
sizing.mode          = "bankroll"
kelly_fraction       = 0.10   → pos = bankroll × 10%
max_position_usd     = 175    → cap 1 SOL
min_position_usd     = 1.0
```

**Progression théorique à 62% WR, 20 calls/jour :**
| Bankroll | Pos/call | PnL/jour | Temps depuis départ |
|----------|----------|----------|---------------------|
| $100     | $10      | ~$35     | J0                  |
| $250     | $25      | ~$88     | ~J5                 |
| $500     | $50      | ~$175    | ~J8                 |
| $1 000   | $100     | ~$350    | ~J11                |
| $1 750   | **$175 (cap)** | ~$490 | **~J14**        |

**Une fois le cap atteint :** ~$400-500/jour (simulation path-indépendante). Réaliste avec path dependency : **$250-350/jour**.

WR multipliers dans le code :
- KOL WR ≥ 80% → pos ×1.5 | ≥ 70% → ×1.2 | ≥ 60% → ×1.0
- RT score 0→0.7x, 100→1.3x

## Idées à tester / backlog

> ⚠️ **2026-08-05 — 5 items ci-dessous LIQUIDES.** Toute reprise doit s'accompagner
> d'un null de permutation (voir `tasks/lessons.md` L2), sinon le résultat n'est pas
> lisible : le scan des features token a sorti 3 "gagnants" contre 6 sous pur hasard.

- [x] ~~**A/B ML**~~ — IMPOSSIBLE : `ml_pred` rempli à **0%** sur les 1.1M lignes RT (ML off).
- [x] ~~**Heure optimale**~~ — **BRUIT**. Les 4 tranches UTC inversent leur signe entre
      train et test (0-6h : +2.15 → −4.39 ; 6-12h : +2.57 → −2.60). Piste fermée.
- [x] ~~**KOL velocity**~~ — **VALIDÉ mais inversé**. Ce n'est pas "bonus après silence",
      c'est **malus en rafale** : dose-response monotone sur 600k lignes, rafale <1h
      −5.43%/trade → 24-72h +2.22% (hors olympeqg, 57% du bucket rafale).
      Déployé en `min_kol_gap_hours` (v14e.70).
- [x] ~~**Seuil KOL dynamique**~~ — **À NE PAS FAIRE**. La forme récente d'un KOL est
      ANTI-prédictive (forme>0 → −4.35% en test, forme<0 → −2.91%). Un seuil
      auto-adaptatif dégraderait. L'edge KOL est une identité STABLE.
- [x] ~~**Multi-confirmation**~~ — IMPOSSIBLE : `n_kol_confirmations` vaut **toujours 0**
      et `unique_kols` **toujours 1** côté RT (le RT part du premier call, par design).
- [ ] **Monte Carlo path simulation** (GBM) : toujours ouvert
- [ ] **TP100_SL50 backtesting** : superseded — voir le combo sentiment+gap ci-dessous

## 🆕 2026-08-05 — Sentiment du message : 2e edge validé (indépendant du KOL)

`kol_mentions.sentiment` (rempli 100%, 51k messages) a une relation en **U INVERSÉ**
avec le résultat, pas monotone :

| bande sentiment | n | moy | train | test |
|---|---|---|---|---|
| < 0.3 | 5631 | −0.57% | −0.98 | +0.11 |
| 0.3-0.5 | 1685 | +1.31% | −0.94 | +4.18 |
| **0.5-0.6** | **663** | **+7.97%** | **+6.84** | **+11.04** |
| 0.6-0.7 | 382 | −4.38% | −8.75 | +2.75 |
| **≥ 0.7** | 212 | **−11.71%** | −13.31 | −7.55 |

Lecture : zéro enthousiasme = pas de mouvement ; conviction mesurée = vrai signal ;
hype extrême = le token est déjà soufflé. Cohérent avec le "score anti-prédictif".

Validations passées :
- **Permutation** (sentiment rebattu dans chaque mois×stratégie) : réel +7.97% contre
  −3.39 / +1.71 / +0.14 sur 3 tirages. Et aucun tirage n'est positif dans les DEUX moitiés.
- **Pas un proxy KOL** : sans slingoorioyaps le pic tient à l'identique (+7.81%,
  train +5.57 / test +14.07). Il ne pèse que 4.4% du bucket. Réparti sur batman_gem (+10.5%),
  spidersjournal (+6.9%), AnimeGems (+2.4%)… y compris des KOLs mauvais en moyenne.
- **4 mois sur 4** le pic bat le reste du flux (+18.1 / +1.7 / +11.9 / +9.9 pp).

Rendement sur TP90_SL40 :

| variante | n | /semaine | moy | géom | WR | train | test |
|---|---|---|---|---|---|---|---|
| aucun filtre | 1500 | 88.2 | −0.1% | −18.3% | 36% | −0.6 | +0.6 |
| **sentiment 0.5-0.6** | 117 | **6.9** | **+9.8%** | −10.5% | 47% | +10.1 | +8.9 |
| **+ gap≥24h** | 44 | 2.6 | **+18.3%** | **+5.5%** | 55% | +18.5 | +18.0 |

- [ ] Câbler un filtre `sentiment_band` en shadow (bande seule = meilleure cadence,
      empilé = meilleure qualité + géométrique positive)

---

## Corrélations features → outcome (données propres N=251, Feb 2026)

| Feature | Corrélation | Note |
|---------|-------------|------|
| whale_new_entries | +0.578 | Seul signal robuste confirmé |
| kol_arrival_rate | +0.42 (N=88) → ~0 (N=251) | Effondré avec plus de data |
| mention_velocity | +0.41 → ~0 | Idem |
| score total | -0.14 | Anti-prédictif ! |
| PA (price action) | +0.10 | Marginalement positif |

→ Les corrélations à N=88 étaient du bruit. Seul whale_new_entries tient à N=251.
→ Besoin de 500+ tokens pour des corrélations fiables.
