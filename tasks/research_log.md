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

## Idées à tester / backlog

- [ ] **Monte Carlo avec path simulation** : Modéliser le chemin prix (GBM) pour avoir une vraie proba de SL avant TP
- [ ] **A/B ML** : Comparer PnL trades ml_pred > 0 vs < 0 (besoin ~500 trades)
- [ ] **Seuil KOL dynamique** : Recalculer seuil optimal chaque semaine automatiquement via Optuna
- [ ] **TP100_SL50 backtesting** : Attendre 30+ trades pour valider la stratégie
- [ ] **Heure optimale** : hour_of_day = feature ML #2. Analyser quelles heures UTC ont le meilleur WR
- [ ] **KOL velocity** : KOL qui n'a pas callé depuis 3 jours = "froid", bonus si premier call après silence
- [ ] **Multi-confirmation** : Token callé par 2 KOLs whitelist en < 30min = bonus de taille x1.5

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
