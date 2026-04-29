# Pipeline Status — Updated 2026-04-29 (v14e.46 LIVE TEST DECK 9 STRATS)

## v14e.46 LIVE TEST DECK 5 STRATS (Apr 29 21:10 UTC) — CURRENT
**Mission** : ground-truth catalogue divergences sim/paper/live par famille. **Slim deck à 5 strats** (vs 9 initial) pour densifier N par strat (10-15 trades/jour SOL ÷ 5 = 2-3/strat/jour → N=30-40/strat sur 14j, vs N=10-20 avec 9). **ETH désactivé** — assume "comme SOL".

**Config DB live_trading** (post 2026-04-29 21:10 UTC restart kol-scraper) :
- `max_position_sol: 0.003` (~$0.51/trade)
- `max_open_positions: 12`
- `daily_loss_limit_sol: 0.5`
- `eth_live_enabled: false`
- 5 strats `allocations` :
  1. **BE25_TP80_SL30** (baseline, N=119 cumul) — drift≈0 médian, calibration historique
  2. **FAST_TP50_SL30** (baseline, N=88 cumul) — drift +3pp avg, sim sous-estime
  3. **TP200_SL40_2H_NZ_S40** (NEW) — TP-only large + filter NZ_S40, top cross-chain
  4. **BE25_LOCK10_TP100_SL30_NZ_S40** (NEW) — famille LOCK 100% inexplorée live
  5. **SCALP_TP20_SL10_S30** (NEW) — famille SCALP (TP serré WR haut), profil totalement nouveau

**Strats DROPPED de la vague 9 → 5** (analysables offline ou redondantes) :
- FAST_TP50_SL30_S40 → A/B filter analysable depuis paper shadow (14j data dispo)
- TP300_SL50_4H_NZ_S40 → redondant avec TP200_SL40_2H_NZ_S40 (axe magnitude)
- FAST_TP500_SL40_60M → fire rare, N≈3 sur 14j = noise
- SLOW6H_TP100_SL50 → covered par paper shadow + couplé à TP-only timeout

**Backup config pre-update** (rollback) :
```sql
UPDATE scoring_config SET rt_trade_config = jsonb_set(jsonb_set(jsonb_set(rt_trade_config,
  '{live_trading,eth_live_enabled}', 'true'::jsonb),
  '{live_trading,max_position_sol}', '0.02'::jsonb),
  '{live_trading,allocations}', '{"BE25_TP80_SL30":1}'::jsonb) WHERE id = 1;
```

**⚠️ Caveats** :
- **Wallet vide depuis Apr 27 15:45 UTC** → aucun trade tant que pas refill (bot logge "insufficient SOL" et skip)
- **$0.51/trade sous le seuil testé** ($1.70/trade en v144.12). Risque que Jupiter Ultra rejette les ordres trop petits → fallback à `max_position_sol: 0.005` ($0.85) si erreurs `min trade size`
- **N target par strat sur 14j** : ~5-15 trades/strat (faible densité) — drift visible si ≥5pp, fragile pour fines conclusions
- **Décision rules** : à N≥15 ET drift > ±5pp → désactiver via JSONB

**Pourquoi cette stratégie** : tous les paper trades sont déjà sauvegardés avec `paper_sim_pnl_pct` injecté en live. Une fois la donnée accumulée, on peut **calibrer la sim sur des observations cross-famille** au lieu de re-deviner. Le coût est borné ($4.59 max simultané × 12 max_open = ~$55 budget si on perd tout).

## État précédent (snapshot pre-v14e.46)

- ~~**SOL live** : `BE25_TP80_SL30` alloc 1.0, position $1.70/trade.~~ remplacé par 9 strats à $0.51 chacun
- ~~**ETH live Phase 1** : `ETH_TP80_SL40_T2H`~~ → DÉSACTIVÉ via `eth_live_enabled=false`. Historique : 1 trade live (−$7.49). Le code paper continue de tracker ETH pour comparaison sim.
- **Sweeps en cours** :
  - SOL `25116811803` (matrix split v14e.43, 3 shards parallèles) — in_progress, ETA 17:00-17:30 UTC
  - ETH `25118472430` — in_progress, ETA <17:00 UTC
  - **Aucun ranking propre actuel** : dernier SOL réussi Apr 25 (pré-RECALL), dernier ETH Apr 27 (OLD slip 100/100)
- **Blacklist KOL SOL** : 14 KOLs filtrés, 13/14 confirmés statistiquement encore négatifs sur 7d. **Exception : `CarnagecallsGambles` recovered (CI95% +5.4 à +12.4% sur N=698)** — à un-blacklist.
- **RECALL family** : 60 strats déployées Apr 28-29, **0 trade fired** depuis. Soit gate trop strict, soit pas de pump-then-dump qualifiant. À évaluer après sweep.

## v14e.45 (Apr 29 PM) — BE/SL outlier guards

| ID | Fix | Statut |
|---|---|---|
| v14e.45.1 | Spread guard 20% Jupiter↔DEX (`paper_trader._fetch_prices_batch`) — drop tick Jupiter si diverge >20% du DEX. Catche $PAPERCLIP-style RFQ glitches | ✅ déployé |
| v14e.45.2 | 2-tick confirm SL/BE (`_evaluate_trade_exit` + `_pending_sl_be` state) — un breach doit persister 2 polls consécutifs avant fire | ✅ déployé |
| v14e.45.3 | DB JSONB `BE25_TP80_SL30.polling_sec` 240→60 (rend 2-tick lag tolérable: 120s au lieu de 480s) | ✅ déployé |
| (test fix) | `test_pipeline_eth.py` assertions 175/115 → 575/515 (stale: ETH_BUY_SLIPPAGE_BPS passé 100→500) | ✅ |

**Audit findings** (cf. `memory/v14e_45_be_outlier_guards.md`) :
- Drift live↔paper_sim BE25 be_stop : +1.16pp PRE-04-21 → **−20.10pp POST-04-22**. Sl_hit aussi (+1.74→−13.12). Timeout stable.
- Pas un bug polling (médiane délai dump→exit = 73s). Outlier exec sur quote — $PAPERCLIP DEX +33% / Jupiter −39% simultané, $OMEGA gap stale tick 1053s.
- Pourquoi BE/LOCK > TP/SL : seuil pile dans la zone d'oscillation du spot (entry ou entry×lock), TP/SL profonds sont loin de la zone de bruit memecoin.
- **CRITIQUE** : `_dynamic_sell_slip_factor` est PAPER-ONLY, n'affecte PAS la slippageBps Jupiter live (hardcoded 300/1000). Ne JAMAIS reverter v14e.6 pour fixer un drift live.

**Décisions parquées** :
- Skip #3 (outlier filter |tick−high_seen|>50%) — risque de bugs sur tokens qui dump légitimement
- Skip #5 (revert v14e.6) — log-linéaire est meilleur modèle, revert serait cosmétique
- Park #6 (recompute paper_sim_pnl_pct historique) — à faire seulement si drift résiduel post-fix mystérieux
- Pas de patch dédié LOCK : v14e.45 patches couvrent automatiquement BE_LOCK via le même `_evaluate_trade_exit` path

## v14e.43 (Apr 29) — récap fix livrés

| ID | Fix | Statut |
|---|---|---|
| v14e.42 | `closing_retry` sentinel-leak ETH+SOL + recovery 6 stuck rows | ✅ déployé |
| v14e.42 | Bigint overflow (INCOME orphan 2.91e19 raw tokens) — caps préventifs | ✅ déployé |
| T4 | `eth_daily_loss_limit_usd` câblé + `_track_eth_pnl` accumule sur close paths | ✅ |
| T5 | `BOND_FAST_TP50_SL20_T20` retiré du live SOL | ✅ |
| T6 | `ETH_FAST_TP500_SL40_60M` confirmé paper-only | ✅ |
| T7 | `kol-weekly-audit.yml` cron lundi 06:00 UTC | ✅ |
| E5 | `_eth_open_lock = threading.Lock()` autour de `open_live_trade` | ✅ |
| W8 | Mega-sweep SOL matrix split 3 shards (jupiter/dexscreener/both) + merge job | ✅ déployé |
| Live tx tolerance | `live_trader_eth.py:1437` lit `eth_sell_slippage_bps` JSONB (était hardcodé 500) | ✅ |
| ETH slip recalib | `ETH_BUY_SLIPPAGE_BPS 100→500`, `ETH_SELL 100→800` (empirique N=8) | ✅ |
| $INCOME orphan recovery | row id=270539 status=manual_recovered | ✅ |
| $MUSK orphan recovery | row id=271709 → bot retry-sell → +$10.67 net @ 16:26 UTC | ✅ |

## Actions actives

### 🟢 Faisable immédiatement (en attente du sweep)
- [x] **REV.** Reverse-engineer scoring (run #1 + #2 with 2-feature combos + token_snapshots join). Output: `docs/score_reverse_engineer_findings.md`. Top finding: BSR (rt_buy_sell_ratio) is the most universal predictor — SOL +5-7pp WR at thr>=0.52, ETH BE+LOCK +40-48pp at thr>=0.55. Refonte rt_score formula proposée et déployée v14e.43 (voir BSR_AB ci-dessous).
- [ ] **BSR_AB ⚠️ EXPECTED NEGATIVE — verdict 3j (~Mai 02)** — 10 shadow strats `_BSR52` SOL + `_BSR55` ETH déjà déployés. Re-vérif 14d post-deploy : -$2-4/d sur 7/7 top SOL strats. **Action** : laisse tourner 3j pour confirmation in-vivo, kill ensuite si confirmé. **JAMAIS direct-apply**.

- [ ] **KW_AB ⏳ verdict 3j (~Mai 02)** — v14e.43b deployed 8 shadow strats `_KW34` SOL (KW>=0.34) + `_KW26` ETH. Validated walk-forward sur target $/d : SOL train +$777 → test +$222 ✅, ETH train +$41 → test +$440 ✅. Top filtre cross-chain. Strats : `BE25_TP80_SL30_KW34`, `FAST_TP50_SL30_KW34`, `FAST_TP50_SL30_S40_KW34`, `BE15_LOCK5_TP50_SL30_KW34`, `SLOW6H_TP100_SL50_KW34` (SOL); `ETH_TP80_SL40_T2H_KW26`, `ETH_FAST_TP100_SL50_KW26`, `ETH_BE25_LOCK10_TP80_SL40_T2H_KW26` (ETH). Paired-test à N≥30 vs base.

- [ ] **BSR_MCAP_AB ⏳ verdict 3j (~Mai 02)** — v14e.43b deployed 4 shadow strats SOL avec combo `rt_buy_sell_ratio >= 0.53 AND entry_mcap >= $45K`. Validated +$20-26/d sur 4 strats SOL losing-base (SLOW6H/SLOW4H/TP100_SL60/TP80_SL70). Filtres seuls = mauvais, combo = excellent. Strats : `SLOW6H_TP100_SL50_BSR_MCAP`, `SLOW4H_TP100_SL50_BSR_MCAP`, `TP100_SL60_BSR_MCAP`, `TP80_SL70_BSR_MCAP`.
- [ ] **SCORE_V2_AB ⏳ data collection — verdict 3j (~Mai 02)** — v14e.43 deployed: `_rt_compute_score_v2()` calculé en parallèle de `rt_score`, persisted en colonne `paper_trades.rt_score_v2`. Formula: `30*kol_win_rate + 5*min(bsr,5) + 8*log10(volume+1) + 8*log10(mcap+1)`. PAS appliqué au filtrage. **Cadence verdict réduite 7-14j → 3j** : à ~9-10 paper trades/jour SOL × 3j = N=27-30, suffisant pour AUC vs `rt_score` actuel. Métrique cible = `sum_$ / day` post-filter (PAS WR — leçon BSR).
- [ ] **MEGA_FILTERS ⏳ post-prod** — `_MEGA_EXT_FILTERS` ajoute (v14e.43b) `BSR52, BSR55, NOZEROLIQ_BSR52, NOZEROLIQ_BSR55, KW34, KW26, NOZEROLIQ_KW34, NOZEROLIQ_KW26, BSR_MCAP` + filter logic `_mega_apply_filter`. Le prochain run de sweep va ranker ces nouveaux filters dans le grid. Permettra de valider out-of-sample au pool level. À retirer les BSR* après confirmation négatif au sweep level.
- [ ] **NEW.** Wire alerte Telegram immédiate sur `CRITICAL: ... bought ... but DB insert failed` (cas INCOME/MUSK auraient été notifiés en <1min au lieu de découverts H+15).
- [ ] **NEW.** Un-blacklist `CarnagecallsGambles` SOL (CI95% positive). Diff JSONB simple.
- [ ] **NEW.** Re-run `_calibrate_eth_slip.py` post-MUSK (N=9 maintenant). Si MUSK pousse les médianes, ajuster ETH_BUY/SELL_SLIPPAGE_BPS.

### 🟡 Wait sur sweep en cours (~1-2h)
- Top SOL/ETH ranking propre avec slip recalibré + RECALL family présente
- Verdict T1 (`_calibrate_sell_slip` post-merge câblage) — débloqué si W8 réussit
- Comparer Apr 27 sweep (OLD slip) vs Apr 29 sweep (NEW slip) pour voir l'impact ranking

### 🟡 Wait sur data
- [ ] **E2.** Verdict ETH microtest à N≥10-15 trades. Actuellement N=9 (incluant MUSK +53% recovered). 1-3 jours.
- [ ] **W1-W5.** Paired-tests (SCALP, AGE, LOCK, ETH winner cluster) à N≥30. ETA Mai 03-10.
- [ ] **W9b.** RECALL family verdict à N≥30 par bucket. ETA Mai 26-Juin 5.
- [ ] **T2.** Sell slip drift re-run à N≥200 SOL twin pairs (~Mai 03-05).
- [ ] **v14e.45 vérif post-deploy** — wallet refill needed first (live mute depuis Apr 27 15:45 UTC, raison: SOL low). Après refill + 7-10j d'accumulation : re-run `_strat_deep_dive.py` BE25+FAST chain=solana fenêtre Apr-29→present. **Cibles** : drift be_stop <±5pp (vs −20pp), médiane bps réels be_stop ≥ −1200 (vs −1765 POST). Compter logs `spread guard:` et `2-tick confirm: SL/BE first breach deferred` — si zéro sur 200+ trades, soit régime calme soit chemin pas exécuté (à investiguer).
- [ ] **LOCK polling alignment (parqué)** — si BE25 vérif post-v14e.45 confirme `polling_sec=60 + median_5` est le bon réglage, appliquer même override aux LOCK SOL/ETH (actuellement `_DEFAULT_ORCH = polling_sec:30, jupiter`). Pas avant validation BE25.

### 🔴 Action utilisateur requise
- [ ] **E3.** Top-up wallet ETH ($14 actuel post-MUSK ~$15 → cible $80-100). Adresse `0xC5c92E3AC207f686D09686Fe1dE79a302D9410E9`.
- [ ] **K3.** Arbitrer ETH PROBABLE_BL : `jadendegens` (N=20 WR 0%), `aliensalphacalls` (N=20 WR 0%). N<30 mais WR=0% → p<10⁻⁶. Ajouter ou wait N=30.
- [ ] **Décision deploy live SOL post-sweep** : config alloc + position sizing.

### 🔧 Backlog (pas urgent)
- [ ] **R2.** Profiler `process_and_push` si lag >30s revient.
- [ ] **T3.** `_eth_round_trip_smoke.py --execute` mensuel ou si base_fee >5 gwei.
- [ ] **T8.** Auto-apply JSONB diff via PR si signal stable 4 sem K1/K2.
- [ ] **T9-T10.** Dead-day filter (`_compute_day_regime`) — priorité basse.
- [ ] **T11-T16.** Idées mécaniques nouvelles : DELAY entry, CIRCUIT BREAKER, VOLUME drop exit, LIQ-pull exit, MULTI-KOL confirmation, TIME-based BE.

## Reverse engineering scoring (NEW)

**Objectif** : trouver les params (weights / thresholds) qui maximisent WR × $/d sur le pool des stratégies, au lieu d'utiliser le score filter discret SCORE30/35/40/45/50.

**Approche** :
1. Pull `paper_trades` 30d closed avec features : `rt_score`, `rt_liquidity_usd`, `rt_volume_24h`, `rt_buy_sell_ratio`, `rt_token_age_hours`, `kol_score`, `kol_win_rate`, `kol_tier`, `entry_mcap`, snapshot features (sentiment, mention_velocity, whale_count, etc) joinés via token_snapshots
2. Target double :
  - Binary WR (1 si pnl>0 sinon 0) → logistic regression + XGB classifier
  - Continuous pnl_pct → linear regression + XGB regressor
3. Per-strategy feature importance : pour chaque strat, trouve les top-3 features qui prédisent son outcome
4. Threshold optimization : pour chaque feature continue, trouve le seuil qui maximise `WR × N` (effective dollars edge)
5. Cross-validation : train 0-21d, test 21-30d (walk-forward, évite overfit)
6. Output : tableau "strategy → optimal feature thresholds" + "global score formula proposée"

**Ce que ça va donner** :
- Confirmation/infirmation des SCORE30/35/40 actuels
- Possibilité de remplacer le SCORE filter par un combo `liquidity + KOL_winrate + volume_bsr`
- Verdict : le rt_score est-il informatif ou peut-on faire mieux avec features brutes ?

**Pré-requis** : `pip install scikit-learn xgboost` sur l'env local. Output `data/score_reverse_engineer_<ts>.csv`.

## Rappels persistants

### Méthode statistique
- TOUJOURS paired-test sur tokens intersection paper×live, JAMAIS aggregate avg quand sample sizes diffèrent.
- N≥30 par strat avant verdict, N≥30 par (KOL, chain) avant blacklist reliable (15-29 = probable, observer 1 sem).
- Bootstrap CI 95% + sign test obligatoires sur tout verdict KOL.
- Filtrer artefacts (DTRAIL/DIP/SPLIT/TRAIL/etc) avec `--exclude-artifact-strats`.

### KOL routing v14e.38
- Per-chain blacklist active : 14 SOL, 0 ETH. JSONB `paper_trade_config.kol_chain_blacklist`.
- mad_apes_gambles : SOL ban / ETH allow (RELIABLE_WINNER N=104).
- Telemetry préservée : kol_mentions + snapshots + ticks + paper SHADOWS continuent pour blacklist.
- KOL whitelist : DISABLED.

### Bankroll
- Total : $53,409 / starting $29,000.
- SOL live position : $1.70/trade (0.01 SOL). User veut scaler à $20/trade.
- ETH live position : $20/trade Phase 1, max 1 open.

### Slippage v14e.34 + v14e.43
- SOL : `BUY_SLIPPAGE_BPS = 225` source unique (strategies.py).
- ETH : `BUY 500 / SELL 800` (recalibré v14e.43 sur N=8 empirique).
- Live tx tolerance ETH : JSONB `live_trading.eth_buy_slippage_bps=500, eth_sell_slippage_bps=600`. Empirical p75 sell = 2500 → peut-être à monter à 2000 si reverts.
- Cost model paper sim : `_evm_slip_bps_with_gas` + gas_as_bps + multipliers liq.

### Cohérence sim/paper/live
- SOL : sim drift median -0.15pp / N=249 ✅ **AGGREGATE**. **MAIS** par exit_reason POST-04-22 BE25 : be_stop −20pp, sl_hit −13pp, timeout stable (cf. v14e.45). Le drift médian global cache une cassure exit-spécifique.
- ETH : sim drift +7.65pp / N=1 ⚠️ — companion sim ratée sur 7/8 trades (path closing_retry pré-fix). Reconstruction possible via `eval_history`.
- Sim **PAS de gate `max_open=1`** → top sim $/d capacity-blind, multiplier ~30-40% pour live réel.
- **v14e.45 patches** doivent réduire drift be_stop+sl_hit. À mesurer post-refill wallet + 7-10j live.

### Mega-sweep
- ETH workflow `mega-sweep-eth-48h.yml` cron 22:00 UTC tous les 2 jours (single job, ~1h).
- SOL workflow `mega-sweep-48h.yml` cron 02:00 UTC tous les 2 jours, **matrix split 3 shards** depuis v14e.43.
- v14e.35 : persist top-30+50 dans `mega_sweep_runs` Supabase + `_mega_sweep_calibration.py`.
- v14e.34 : `_strat_slip_sensitivity.py` post-sweep flag fragile strats.

### Strats deck (577 dont 119 artefact-deprecated)
| Family | SOL | ETH |
|--------|-----|-----|
| BE | 33 | 9 |
| LOCK | 33 | 20 |
| FAST | 73 | 18 |
| SLOW | 13 | 5 |
| SCALP | 36 | 17 |
| Other | 80 | 10 |
| AGE clones | 38 | 18 |
| RECALL DIP (v14e.40+41) | 27 | 9 |
| RECALL PEAK (v14e.41) | 18 | 6 |
