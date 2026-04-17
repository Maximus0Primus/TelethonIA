# Pipeline Status — Updated Apr 17, 2026 (v138.3)

## Current state

**Live (50/50)** — `BE25_TP80_SL30 (median_5/static_240)` + `BE15_TP100_SL50 (ds/fast)`. Position ~$1.70/trade, max 3 open.

**Paper Telegram (8 strats × $1000 fresh bankroll, $8000 total post-v138.3 reset)**:

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

## 🧹 Housekeeping (non urgent)

- [ ] `.gitignore` cache files (commit 007db6b a pushé 2400+ fichiers)
- [ ] Jupiter LDS sous-fill (35% vs seuil 70%)
- [ ] Holders sous-fill (27%)
- [ ] CA resolution 71.9% (sous seuil 75%)
- [ ] Backlog labels : 24h=774, 7d=2346
- [ ] Bug `reconcile_positions` bypass bankroll (cosmétique)
- [ ] DIP30 entry gate cassé (désactivé en v136, pas urgent)

## 🔵 Low-priority

- [ ] Birdeye TOP_N 20→50+ (whale_new_entries NULL 80%)
- [ ] PA computation gate sur `SCORING_PARAMS["price_action"] > 0`
- [ ] gate_mult dead compute (RugCheck always 1.0)
- [ ] v53 features (holder_turnover, kol_cooccurrence) <6% fill

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
