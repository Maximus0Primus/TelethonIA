# Pipeline Status — Updated Apr 17, 2026 (v138.3)

## Current state

**Live (50/50)** — `BE25_TP80_SL30 (median_5/static_240)` + `BE15_TP100_SL50 (ds/fast)`. Position ~$1.70/trade, max 3 open.

**Paper Telegram (10 strats × $1000 bankroll, $10K total post-v139)**:

| Strat | Config | Sweep avg | Mode |
|---|---|---|---|
| FAST_TP100_SL20 | ds | +11.27% | LAZY |
| BE25_TP80_SL30 | median_5 | +10.09% | static 240s |
| FAST_TP80_SL25 | ds | +9.46% | LAZY |
| BE25_TP80_SL30_DS | ds | +8.64% | LAZY (A/B vs ema_fast) |
| FAST_TP40_SL30 | hysteresis | +7.96% | LAZY |
| FAST_TP50_SL30 | median_3 | +7.31% | LAZY |
| BE15_TP100_SL50 | ds | +7.14% | static 30s (fast) |
| TP50_SL15 | jupiter | +6.28% | LAZY |
| **NOZEROLIQ_TP200_SL40** | jupiter | (test +14.91%) | static 120s — **NEW v139** |
| **HIGHSCORE_TP200_SL40** | jupiter | (test +14.42%) | static 120s — **NEW v139** |

## v139 — TP200 asymmetric strategies (Apr 17 17:30 UTC)

Tested 19 candidate strategies on 71 post-v132 tokens (`scripts/_test_new_strategies.py`).
Top 2 added to paper portfolio:
- **NOZEROLIQ_TP200_SL40** : skip pump.fun pre-grad tokens (liq=0). N=44, WR 48%, avg +14.91%, $/jour proj +$83.
- **HIGHSCORE_TP200_SL40** : rt_score ≥ 30 gate. N=38, WR 50%, avg +14.42%, proj +$69.

Insights de la batterie de tests :
1. TP200_SL40 (3x TP, 0.6 SL, 4h horizon) > BE25_TP80_SL30 baseline systématiquement
2. `liq=0` (pump.fun bonding) = -$19.82/jour drag — skip = +9pp avg
3. `rt_score` PRÉDIT (revoir notre opinion "scoring on s'en fout") — score≥40 donne 65% WR median +22%
4. EARLY_DUMP cut DÉGRADE (coupe des winners qui auraient récupéré)
5. TOPKOLS whitelist : modeste gain (+$32 vs +$28)

Code: `STRATEGY_FILTERS` étendus avec `min_liquidity_usd` + `min_rt_score` + `min_mcap`.
Bankroll : 10 strats × $1000 = $10000 starting capital.

LAZY = 180s during first 5min, 600s after. Hardcoded in `strategies.py:LAZY_STRATEGIES`.

## $/jour projeté vs réel attendu

Sim biais mesuré sur historique (ratio réel/sim) :
- TP50_SL15 : 0.71 (fiable)
- BE15_TP100_SL50 : 1.25 (sim sous-estime, sera meilleur)
- BE25_TP80_SL30 : 0.54 (sim 2x optimiste)
- FAST_TP100_SL20 : **0.12** (sim 8x optimiste 🚨)
- 3 nouvelles FAST + BE25_DS : N trop petit, projection théorique

**Total projeté brut (sim)** : ~+$340/jour paper, ~+$11/jour live ($1.70/trade)
**Total réaliste avec ratios** : **~+$200-280/jour paper**, ~+$5-8/jour live

## 🔴 Priorités immédiates (Apr 17-19)

- [ ] Validation 24-48h post-v138.3 : `current_balance > $8000` ? Si oui combien de gain réel
- [ ] **Validation FAST family** — sim dit #1 mais historique réel = +1.01%. Si après 48h les 4 FAST génèrent < +2% avg réel → swap out
- [ ] **`--from-eval-history`** — confirmer biais=0% sur trades fermés post-v138 deploy (eval_history maintenant persisté)
- [ ] Live BE25 + BE15 : >$0.50/trade avg sur N≥10 trades
- [ ] **Validation latence post-A.2.1** (24h post-deploy) : si `msg→ds` mean baisse de 24s → 15-18s sous charge, le batch shadow fix a aussi désengorgé l'executor (effet attendu)

## 🟡 Active Exploration

### Latence live trade — diagnostic A.2 ✅ (Apr 17)

**Avant fix (N=50 buys Apr 15-17) :**
| Phase | p50 | p95 | mean | % total |
|---|---|---|---|---|
| msg→ds | 14.3s | 59.4s | 24.0s | 59% |
| ds→pre_buy | 12.2s | 33.4s | 15.5s | **38%** ← bottleneck |
| buy_exec | 0.9s | 1.5s | 1.0s | 2.4% |
| **TOTAL** | **35.6s** | **93.4s** | **40.4s** | — |

**Vrai bottleneck identifié** : `paper_trader.py:1051` faisait 217 inserts shadow séquentiels (~50ms HTTP × 217 = 10-20s par RT call). Les composants détaillés mesurés via RT timing : `detect→ds=0.04s`, `ds→open=0.10s` — les autres phases sont déjà optimales.

**A.2.1 ✅ DEPLOYED (commit `8a7a4c1`)** : batch shadow inserts → 1 HTTP call au lieu de 217. Cible : ds→pre_buy 15s → ~3s, et indirectement msg→ds réduit (executor moins congestionné sous charge).

**À valider** : 
- [ ] Re-mesurer LIVE LATENCY dans 24h post-deploy. Si on voit `msg→ds` passer de 24s à 15-18s mean sous charge → confirmé. Sinon, le bottleneck est ailleurs.
- [ ] Vérifier que les spikes `msg→detect > 50s` disparaissent (effet désengorgement executor)

**Restant (faible ROI à $1.70/trade, à reconsidérer si on scale-up) :**
- [ ] **A.2.2** — augmenter `ThreadPoolExecutor max_workers` (défaut ~8-16 → forcer 32-64) si encore des spikes sous charge
- [ ] **A.2.3** — paralléliser les 2 `open_live_trade` calls (BE25 + BE15) via ThreadPoolExecutor. Risque : race condition sur `max_open_positions` check. Bénéfice : ~2-3s. À faire si stratégies live > 3.
- [ ] **A.2.4** — paralléliser les 8 `open_paper_trades` calls hybrid. Bénéfice : ~5-8s sur les MAIN paper inserts. Code change moyen, à faire après validation A.2.1.

### Validation cadence v137
- [ ] Vérifier que la 30s throttle double bien les ticks `source='fast'`/`'full'` dans `price_ticks`

## 🟢 Deferred — `tp_touched` exit mode

Si `high_price_seen >= tp_price` mais exit fires sur timeout/SL/trail, exit rétro à `tp_price`. Analyse post-v133-D : +$1.55/sem live (sous threshold).

**Re-evaluate when ANY :**
- High-TP strat live (TP80+)
- Live volume ×3
- Jupiter Trigger V2 keepers 0 fill 7j

## 🧹 Housekeeping

- [x] **v138.5** : `.gitignore` cache files — untracked 2440 OHLCV cache files (commit 007db6b accidentally tracked)
- [x] **v138.5** : PA computation gate (skip si `w_price_action <= 0`)
- [x] reconcile_positions bypass bankroll — DÉJÀ FIXÉ en v136.2 (todo entry était obsolète)
- [ ] Jupiter LDS / Holders / CA resolution sous-fill — affectent SCORING uniquement (skip, scoring désactivé)
- [ ] Backlog labels : sert ML qui est disabled (skip)
- [ ] DIP30 entry gate (deprecated)

## 🔵 Low-priority

- [ ] Birdeye TOP_N 20→50+ — affecte scoring (skip)
- [ ] gate_mult dead compute (RugCheck) — vérifié 17 Apr : aucune référence dans live_trader/safe_scraper/pipeline → ne bloque rien en live, safe à supprimer si on veut économiser API. **Action différée** car low impact.
- [ ] v53 features — ML disabled, skip

## Sim ↔ Live/Paper coherence (reference)

**Bias hierarchy stable :**
- `--from-eval-history` (v138) = 0% mathématique pour trades post-deploy
- `--from-trades` = ground truth historique
- `--from-ticks jupiter` (v137) = +4pp residual sur trail-heavy
- `--from-ticks dexscreener` = correct proxy entry, faux pour tracking

**Per-family tool :** FAST/BE → from-ticks OK, DTRAIL/TRAIL → from-trades obligatoire

**Thresholds :** divergence per-pair <5pp normal, >10pp = bug. Live/sim edge <50% expected edge.

## Architecture summary

**Scoring :** 40.5/13.5/40.5/5.4 (consensus/conviction/breadth/PA), 16-multiplier chain.
**Trading :** Paper slip dynamic, live Jupiter Ultra RFQ ~10bps, position reconciliation sibling-aware (v133-D), loss limit 0.5 SOL/jour.
**Alerting :** ML disabled, RT listener uncapped, GH Actions failures, daily summary 8am UTC.

## Historique récent (sessions Apr 17)

- **v137** ✅ cadence fix `_replay_trade_orchestrated` (next-tick-after-gap → look-back), throttle 60→30s
- **v138** ✅ eval_history JSONB + cache_snapshots table + `--from-eval-history` mode (0% bias par construction)
- **v138.1** ✅ swap live FAST→BE15, drop 2 DTRAIL paper (perdaient -$160/j)
- **v138.2** ✅ mega-sweep 9040 configs → 8 paper actives + LAZY mode + 3 nouvelles FAST
- **v138.3** ✅ BE25 → median_5/static_240 (rerank par avg_pnl_pct), bankroll reset $1000×8
- **v133-D** (Apr 16) ✅ hybrid sell pollution fix
