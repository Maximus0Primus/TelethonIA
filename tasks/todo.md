# Pipeline Status — Updated Apr 17, 2026 (v140)

## Current state

**Live (50/50)** — `BE25_TP80_SL30 (median_5/static_240)` + `BE15_TP100_SL50 (ds/fast)`. Position ~$1.70/trade, max 3 open.

**Paper Telegram (18 strats × $1000 bankroll, $18K total post-v140) — A/B comparison board**:

| Strat | Filter | Source | Smoothing | Polling | Sim $/j | Real est (×0.55) | Status |
|---|---|---|---|---|---|---|---|
| **BE25_TP80_SL30** | — | jupiter | median_5 | static 240 | ~$92 | ~$50 | v138 baseline, real +3.89% |
| **BE25_TP80_SL30_DS** | — | ds | raw | lazy | ~$94 | ~$52 | A/B DS variant |
| **FAST_TP100_SL20** | — | ds | raw | lazy | ~$101 | ~$56 | real +1.01% (divergence haute) |
| **FAST_TP80_SL25** | — | ds | raw | lazy | ~$95 | ~$52 | real +3.54% |
| **FAST_TP50_SL30** | — | jupiter | median_3 | lazy | ~$60 | ~$33 | real -3.12% (plombe) |
| **FAST_TP40_SL30** | — | jupiter | hysteresis | lazy | ~$50 | ~$28 | real -1.93% (plombe) |
| **TP50_SL15** | — | jupiter | raw | lazy | ~$65 | ~$36 | real +4.09% |
| **BE15_TP100_SL50** | — | ds | raw | fast 30s | ~$50 | ~$35 | real +4.61% (sim sous-estime) |
| **NOZEROLIQ_TP200_SL40** | liq>0 | jupiter | raw | static 120 | $83 | ~$45 | NEW v139, test only |
| **HIGHSCORE_TP200_SL40** | score≥30 | jupiter | raw | static 120 | $69 | ~$38 | NEW v139, test only |
| **FAST_TP100_SL20_HYST** | — | hysteresis | — | lazy | **$151** | **~$83** | v140 top sweep ⭐ |
| **FAST_TP80_SL25_HYST** | — | hysteresis | — | lazy | **$140** | **~$77** | v140 top 2 ⭐ |
| **BE25_TP80_SL30_HYST** | — | hysteresis | — | lazy | **$139** | **~$76** | v140 top 3 ⭐ |
| **FAST_TP50_SL30_HYST** | — | hysteresis | — | lazy | **$135** | **~$74** | v140 top 5 ⭐ |
| **BE25_TP80_SL30_S30_HYST** | score≥30 | hysteresis | — | static 240 | $95 | ~$52 | v140 best SCORE30 |
| **BE15_TP70_SL50_NZ** | liq>0 | jupiter | raw | static 240 | $87 | ~$48 | v140 best NOZEROLIQ |
| **BE25_TP80_SL30_NZS30_HYST** | liq>0+score≥30 | hysteresis | — | static 240 | $86 | ~$48 | v140 avg **+25.67%** N=27 ⭐ |
| **BE15_TP300_SL50_MCAP** | 30K<mcap<500K | ds | raw | fast 30s | $85 | ~$47 | v140 best MCAP_MID |

**Total sim projection** : ~**+$1700/jour**. Réaliste (×0.55) : **~$900/jour**.

**Bankroll live** : **$18,722** (starting $18K + $722 gain depuis 16h UTC, 16 trades fermés). Call $ELONMUSK/ReaperGems 16h03 a fait **+$819** sur 6 strats ($FAST_TP40 timeout +247, $BE25_DS tp +134, etc.). Trade $DFV/caniscooks 17h19 a perdu -$194.

LAZY mode = 180s pendant 5min puis 600s. Hardcoded dans `strategies.py:LAZY_STRATEGIES`.

## rt_score V141 — taux WR par threshold (N=74 closed MAIN post-v132)

| Filter | Trades/jour proj | WR | avg | $/jour @$50 |
|---|---|---|---|---|
| Aucun (overall) | ~18 | 37.8% | +4.21% | +$38 |
| **≥30** (filter prod actuel) | ~9 | **51.3%** | **+19.01%** | **+$85** ⭐ |
| ≥40 | ~4.6 | 57.9% | +19.13% | +$44 |
| ≥50 | ~3.2 | **69.2%** | +32.49% | +$52 |
| ≥60 (ultra-sélectif) | ~1.9 | **75.0%** | +38.80% | +$37 |

**Gains v141 vs v1 stored (au filter ≥30)** :
- N pass : 38 → 39 (+2.6%)
- **WR : 44.7% → 51.3%** (+6.5pp)
- **avg : +12.45% → +19.01%** (+53%)
- corr pnl_pct : +0.207 → +0.236 (+14%)

**Quartiles V141** — signal monotonic :
- Q1 (score 4-20) : WR 27.8% / avg -5.13%
- Q4 (score 40-84) : WR 57.9% / avg +19.13%

Seul dip : ≥40 bucket dilué par les bonuses qui gonflent certains mid-scores 32-38 vers 40-46. Pas impactant car prod use ≥30.

## Sim biais mesuré (ratios réel/sim sur historique pré-v140)

| Strat | Ratio | Note |
|---|---|---|
| BE15_TP100_SL50 | 1.25 | sim sous-estime |
| TP50_SL15 | 0.71 | fiable |
| BE25_TP80_SL30 | 0.54 | sim 2x optimiste |
| FAST_TP100_SL20 | **0.12** | 🚨 sim 8x optimiste |

**Hypothèse v140** : `hysteresis` smoothing pourrait réduire le biais car il filtre les triggers transitoires (cause principale de divergence sim/real). À valider 24-48h.

## 🔴 Priorités immédiates (Apr 18-20)

- [ ] **Validation 24-48h post-v140** : `current_balance > $18000` ? Si oui combien de gain réel total
- [ ] **A/B hysteresis** : `BE25_TP80_SL30` (median_5/240) vs `BE25_TP80_SL30_HYST` (hysteresis/lazy) — mêmes tokens, configs différentes. Si HYST > vanilla → hysteresis sauve réellement le sim. Si égal → sim trompeur encore.
- [ ] **A/B filtres** : `BE25_TP80_SL30_HYST` vs `BE25_TP80_SL30_S30_HYST` vs `BE25_TP80_SL30_NZS30_HYST` — quel filter ajoute le plus de valeur réelle ?
- [ ] **NZS30 confirmation** : avg sim +25.67% sur N=27. Si réel ≥ +10% confirmé → meilleure config absolue (ratio à mesurer).
- [ ] **`--from-eval-history`** : biais=0% sur trades fermés post-v138 (eval_history persisté)
- [ ] Live BE25 + BE15 : >$0.50/trade avg sur N≥10 trades
- [ ] **Latence A.2.1** : `msg→ds` mean 24s → 15-18s sous charge confirmé ?
- [x] **v141 — fix mesure `paper_exit_price`** ✅ DÉPLOYÉ Apr 17 19:58 UTC (commit `16a7e8a`). Avant v141, `paper_exit_price` stocké par live_trader = niveau SL brut (slip=1, fee=0), mélangeant 3 effets dans `price_divergence_pct`. v141 calcule un 2ᵉ `ev` dédié mesure avec dynamic slip + SELL_FEE_BPS + Ultra SELL quote override — mirrors paper_trader exactement. Décision live inchangée (toujours slip=1, zéro impact sur PnL/status/sell). **Données pré-v141 biaisées** — filtrer par `created_at >= 2026-04-17T19:58:00Z` avant analyse divergence.
- [ ] **Divergence paper→live par liq bucket** — une fois v141 deployé, attendre N ≥ 15 par bucket avec le vrai `paper_exit_price`. Script : `scraper/analyze_divergence_by_liq.py --days 7`. Décompose (A) polling lag (LIVE MONITOR 3-11min, gap paper→live 60-78s sur $DFV) vs (B) slippage exec Jupiter sur curve illiquide via `corr(delay_sec, |div|)` et `|div|@delay<30s vs >120s`. Hypothèse courante : bonding/liq=$0 → div >20% mais attention ces 3 observations Apr 17 étaient biaisées par le bug de mesure, repartir de zéro. ETA Apr 22-24. **Ne rien modifier côté prod entre-temps**, garder exposition bonding pour collecter signal propre.
- [ ] **Debug Jupiter Trigger V2 — 0 keeper fills historiques** (memory v121). Les triggers `place_stop_loss` placent un `trigger_order_id` mais jamais exécutés par keeper. Investigation en 3 étapes :
  1. **Query `trigger_events` table** depuis v121 : compter `placed` / `cancelled_by_polling` / `keeper_filled` / `failed`. Si 0 filled → problème côté Jupiter (vault auth, keeper routing, token non-routable). Si filled mais polling cancel avant → polling trop agressif, à desync.
  2. **Tester sur trade éligible avec min=$5** — baisser `trigger_min_usd` de $10 → $5 dans `rt_trade_config.live_trading` JSONB côté Supabase (1 UPDATE query). Live pos actuel ~$1.80 donc encore sous le seuil — il faut aussi bumper position_usd à $5+ pour au moins 1 strat pour observer le flow complet. Candidat : `BE25_TP80_SL30` uniquement, pos $5, 1h d'obs.
  3. **Filter non-bonding pour trigger** — ajouter gate `if is_bonding_curve: skip trigger` dans `live_trader.py:836`. Les keepers Jupiter ne routent pas pumpfun bondings → fail certain. Évite le bruit dans trigger_events. Script SQL query + hypothesis : `SELECT COUNT(*), event_type FROM trigger_events WHERE ... GROUP BY event_type`.
  
  **Orthogonal à la question bonding** : Trigger V2 fixerait (A) polling lag sur liq >$25K non-bonding, mais ne résout pas (B) slippage exec ni les bondings (hors graphe Jupiter). À avancer en parallèle du recueil de data liq buckets.

## 🔬 Audit divergences sim/paper/live (Apr 17 16:00-20:00 UTC) — à confirmer avec N plus grand

Window : 4h post-deploy 18 strats. N(live closed)=10, N(paper closed)=68, N(shadow)=1091. Snapshot pré-v141 pour la plupart. **Ne pas agir sur ces findings avant N≥30 par strat / N≥15 par liq_bucket**.

### 🔴 Problèmes actionnables (confirmés sur N=10 mais N trop faible)

- [ ] **P1 — Inversion TP/SL paper vs live sur même token/strat**. $ELONMUSK BE15_TP100_SL50 : live=tp_hit +2.07%, paper=sl_hit −0.08% (inversion totale). Entry_price paper=2.514e-5 vs live=2.635e-5 (−4.58%). 1/10 paires live+paper divergent sur le status. → paper et live voient des séquences de prix différentes. **Data need** : 20+ paires live+paper post-v141 pour confirmer taux d'inversion. **Fix possible** : shadow-sync paper reuses live's entry_price (décision v130-v131 deferred, à remettre sur la table).
- [ ] **P3 — Slippage live réel vs modèle sim v138.5 : gaps des 2 côtés**. Modèle actuel (`paper_trader.py:1094-1143`, v138.5 recalibré sur 132 trades Apr 13-17) = 435 bps sl_hit, 1000 trail_crash, 250 trail_stop, −300 tp_hit, 120 timeout, avec ×2.0 si liq<$5K et ×1.3 si liq<$20K, caps [−1000, +1500/+2500]. Live mesuré (audit N=10 Apr 17) : bonding sl_hit **+1165 bps** (sim projette 870 → gap ~300 bps sous-estimé), liq >$50K sl_hit **−188 bps** (sim projette 435 → sim sur-pénalise ~620 bps en haute liq !). **Paper ne store aucun `buy/sell_slippage_bps`** (colonnes NULL sur 72 paper trades) — ça c'est le vrai problème d'observabilité, pas la calibration du modèle. **Data need** : N≥30 live par liq_bucket pour re-calibrer v142 (sur-pénalisation en liq >$50K à corriger + renforcer bondings). **Fix possible** : ajuster type_bps par bucket liq fin (pas juste <$5K / <$20K) et populate paper_slippage côté `paper_trader.check_paper_trades` pour pouvoir mesurer divergence.
- [ ] **P4 — Entry_price paper vs live diverge ±9% même token**. $DRAGON BE15 +8.99%, $QUACK BE25 −7.73%. Gap 4-27s entre les 2 inserts → 2 fetch DexScreener indépendants, prix bouge 5-10% en 10s sur bonding. **Cause racine** : double pipeline entry (live + paper) non-synchronisés. **Data need** : confirmer sur 20+ paires. **Fix possible** : shadow-sync (même fix que P1).

### 🟠 Suspects (à retester avec plus de data)

- [ ] **S1 — Top 4 HYST du sweep v140 perdent toutes en réel**. Sur N=4 par strat : FAST_TP100_SL20_HYST $25 sim → **−$43 réel** (ratio −1.7), idem FAST_TP80_SL25_HYST −2.0, BE25_TP80_SL30_HYST −1.8, FAST_TP50_SL30_HYST −2.1. **Pendant ce temps FAST_TP40_SL30 sim $8 → +$173 réel (ratio +21) 🎯**. N=4 trop faible → shot noise ou sim biais hysteresis. **Data need** : N≥15 par strat HYST (~4-7 jours). Si ratio reste <0 → hysteresis smoothing sauve le sim mais pas la réalité, revenir à vanilla.
- [ ] **S3 — `exit_price` live ≠ vrai prix de vente**. Gap `pnl_pct` vs `(exit_price/entry − 1)` jusqu'à −205% sur $ELONMUSK tp_hit, +60% sur $DFV sl_hit. `exit_price` stocké = dernier tick DS observé, pas le net Jupiter sell. **Impact** : reconstitution naïve de PnL via `exit_price` fausse, analyses post-hoc unreliables. **Data need** : aucune, c'est un bug structurel. **Fix possible** : soit `exit_price` = vrai net de vente (sell_sol_received × sol_price / token_amount), soit renommer en `last_observed_price_at_exit` et ajouter une colonne `exit_price_net`. Corréler `pnl_usd / position_usd` vs raw_return stored pour valider.
- [ ] **S4 — Bankroll +$117 réel, pas +$722 comme noté**. Peak $18,722 = instantané à un moment donné. Net Apr 17 16:00 → 20:00 = `current_balance $18,117 − $18,000 = +$117`. Somme paper closed +$120.64 + live closed +$1.31 = $121.95 (cohérent avec $117 à $4 près). **Action** : nettoyer la section "Current state" du todo pour ne plus mentionner $722 (trompeur), remplacer par net actuel.
- [ ] **S5 — Bondings gagnent, high-liq perdent (contre-intuitif)**. Paper PnL par bucket sur 4h : liq 0-100$ (bonding) 40 trades WR 49% **+$439**, liq 10-50K 15 trades WR 0% **−$144**, liq >50K 14 trades WR 0% **−$174**. **Si ça tient**, les filtres `NOZEROLIQ` / `HIGHSCORE≥30` sabotent le seul segment rentable. **Data need** : N≥50 par bucket (fenêtre 4h = shot noise probable). Retester dans 48h.

### ⚪ Non-testables tant qu'il n'y a pas plus de data

- [ ] **v141 data** : 0 live trades post-19:58 UTC dans le snapshot audité. ETA Apr 19-20 pour N≥15. Toute analyse `price_divergence_pct` avant = biaisée.
- [ ] **Paper slippage** : colonnes `buy/sell_slippage_bps` sont **NULL structurellement** côté paper (paper_trader ne les populate pas). v141 ne fixe PAS ça. Fix nécessite d'ajouter le write dans paper_trader `check_paper_trades` quand ev ferme. Low priority tant qu'on sait juste que paper = 0 slip assumé.
- [ ] **Trigger V2 fills** : 0 keeper fills historiques, non interrogé dans cet audit (déjà couvert puce debug Trigger V2 ci-dessus).

### 🧹 Nettoyage

- [ ] Delete `scraper/_audit2.py` et `scraper/_audit3.py` (scripts one-shot du subagent Apr 17). Le script permanent `scraper/analyze_divergence_by_liq.py` couvre le besoin récurrent.

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

## Workflow sim (v140 unifié dans sim.py)

| Mode | Flag | Use case |
|---|---|---|
| Grid focused | `--from-ticks` | Ranking rapide par strategy |
| Ground truth | `--from-trades` | Vérité terrain historique |
| 0% bias | `--from-eval-history` | Perfect replay post-v138 |
| **Mega sweep** | `--mega-sweep` | **Full matrix 134K configs** |

**Commandes :**
```bash
# Ranking rapide
python scraper/sim.py --from-ticks --since 2026-04-13 --top 30

# Vérité terrain
python scraper/sim.py --from-trades --since 2026-04-13

# 0% biais mathématique (trades post-v138 uniquement)
python scraper/sim.py --from-eval-history --since 2026-04-17

# Full matrix (134K configs, ~30-45 min, 12 workers)
python scraper/sim.py --mega-sweep
#   flags optionnels : --mega-workers N, --mega-csv-out PATH, --mega-since ISO
```

Mega sweep utilise `_evaluate_trade_exit` avec slip v138.5 calibré (sl_hit 435bps, trail 250bps, tp +300bps positif).

## Historique récent (sessions Apr 17)

- **v141** ✅ rt_score enrichi avec 3 bonuses data-driven (+8 fresh age 0.3-1.3h / +8 bsr>0.7 / -5 liq=0). Backfill audit N=74 : corr +0.207 → +0.236 (+14%), filter≥30 avg +12.45% → +19.01% (+53%). V1 weights inchangés — bonuses purement additifs.
- **v140** ✅ Full mega sweep 136K configs, 12 workers, 22min. Découverte `hysteresis+lazy` domine top 10. 8 nouvelles strats ajoutées + bankroll reset 18×$1000=$18K. `_BE_RE` regex relaxé pour accepter suffixes (_HYST/_NZ/_S30).
- **v139** ✅ Test 19 candidates → ajout NOZEROLIQ_TP200_SL40 + HIGHSCORE_TP200_SL40 ($83+$69 sim/jour)
- **v138.5** ✅ `_dynamic_sell_slip_factor` recalibré (sl_hit 30→435bps, trail 15→250bps, tp positive +300bps), .gitignore 2440 cache files, PA gate, audit ML (toujours cassé)
- **v138.4** ✅ batch shadow inserts (217 HTTP → 1 batch) → -12s ds→pre_buy
- **v138.3** ✅ BE25 → median_5/static_240, bankroll reset $1000×8
- **v138.2** ✅ mega-sweep 9040 configs → 8 paper actives + LAZY mode + 3 nouvelles FAST
- **v138.1** ✅ swap live FAST→BE15, drop 2 DTRAIL paper (perdaient -$160/j)
- **v138** ✅ eval_history JSONB + cache_snapshots table + `--from-eval-history` (0% bias)
- **v137** ✅ cadence fix `_replay_trade_orchestrated` (next-tick-after-gap → look-back), throttle 60→30s
- **v133-D** (Apr 16) ✅ hybrid sell pollution fix
