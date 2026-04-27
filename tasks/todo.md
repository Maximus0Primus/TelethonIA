# Pipeline Status — Updated Apr 27, 2026 PM (v14e.32 deployed)

État courant : **ETH live Phase 1 microtest ACTIF** depuis 13:18 UTC. Live SOL = BE25_TP80_SL30 + BOND_FAST_TP50_SL20_T20. Live ETH = `eth_live_enabled=True`, 1 strat (`ETH_TP80_SL40_T2H`), pos $20, max 1 open.

Bankroll global $53,409 / starting $29,000 (post Apr 27 paper-promote +6 strats à $1000 chacun, fix bankroll bug schema).

L'historique des décisions se lit dans le git log — ce TODO ne garde que ce qui est encore à faire.

---

## 🎯 EN COURS — ETH Phase 1 microtest

Goal : mesurer **slippage empirique sur tokens KOL ETH** (le smoke test Apr 26 sur PEPE n'est pas représentatif). Position $20 max, max 1 open. **Stop quand 10 trades fermés** ou si drift > -7pp.

- [ ] **E1.** Surveiller premier trade live ETH : `journalctl -u kol-scraper -f | grep "ETH LIVE"`. Vérifier que les 6 nouvelles colonnes (`gas_usd_buy`, `quote_slip_bps_buy`, etc.) se peuplent bien.
- [ ] **E2.** Après 5-10 trades fermés (ETA 1-3 jours selon volume KOL ETH) : `python scripts/_eth_microtest_recap.py`. Verdict :
  - drift médian > -3pp → Phase 2 ($50/trade, 2 strats)
  - drift -3 à -7pp → continuer collecte
  - drift < -7pp → abort + recalibrer `ETH_BUY_SLIPPAGE_BPS` empiriquement
- [ ] **E3.** Top up wallet : actuel $43 → $80-100 (≈ 0.035 ETH supplémentaires) pour avoir buffer. Adresse : `0xC5c92E3AC207f686D09686Fe1dE79a302D9410E9`.
- [ ] **E4.** **Rotation clé wallet ETH** — la clé est compromise (transcript persistant). À faire AVANT scaling Phase 2.

---

## 🎯 EN COURS — observation des shadows

Pas de code à écrire, juste laisser la data grossir. ETA verdicts paired-test : **Mai 03-10**.

### À N≥30 par strat
- [ ] **W1.** Paired-test `SCALP_TP15_SL20_S35` vs base `SCALP_TP15_SL20`.
- [ ] **W2.** Paired-test des AGE clones SOL vs leur parent (AGE24/48_BE25 vs BE25, etc.).
- [ ] **W3.** Paired-test des AGE clones ETH (4 existants + 8 nouveaux v14e.31).
- [ ] **W4.** Paired-test **LOCK family** vs BE base : LOCK10 SOL = +1.88pp / LOCK10 ETH = +2.85pp en backtest tick-replay. Confirmer en paper N≥30.
- [ ] **W5.** Validation **ETH winner cluster** (26 strats v14e.31).

### Mega sweep auto (cron 48h)
- [ ] **W6.** SOL : prochain run cron 02:00 UTC.
- [ ] **W7.** ETH : prochain run 22:00 UTC avec chain-aware position $50.

---

## 🚨 INVESTIGATION DRIFT BE25 (à surveiller)

Apr 27 investigation : aggregate drift Apr 26-27 = -16.88pp paraissait alarmant mais le **paired-test correct** donne -7.46pp (vs -4.59pp Apr 22-25). Δ réel = -2.87pp, dans le bruit pour N=28.

- [x] Paired-test analysis fait (`_paired_drift.py`)
- [x] Mad_apes blacklist appliquée puis annulée (paired drift -1.17pp sur N=4 = noise, pas un signal)
- [ ] **D1.** Re-runner `_paired_drift.py` chaque 48h sur BE25 — surveiller si drift médian descend < -5pp avec N>50 (alors investigation cause exec).
- [ ] **D2.** Si drift persistant : recalibrer `BUY_SLIPPAGE_BPS` SOL via `_calibrate_buy_slip.py` (pas touché depuis 225bps Apr 25).

---

## 🔧 BACKLOG TECH

### Calibration drift (recurring)
- [x] SELL slip twin-pair calibrator — Apr 26 verdict : N=120, weighted median Δ = −0.16 pp → v144 SOL calibration valide.
- [ ] **T1.** Câbler `_calibrate_sell_slip.py` dans `mega-sweep-48h.yml` post-sweep.
- [ ] **T2.** Re-run sell slip drift quand N≥200 twin pairs (~Mai 03-05).
- [ ] **T3.** Re-run `_eth_round_trip_smoke.py --execute` mensuellement OU si base_fee ETH > 5 gwei. Bumper `ETH_GAS_COST_USD_PER_SIDE` ~3× si gas/side > $4.
- [ ] **T4.** Câbler `eth_daily_loss_limit_usd` dans le ETH dispatch (pas enforced actuellement, pour Phase 1 le `eth_max_open_positions=1` borne le risque).

### KOL blacklist multi-chain
- [ ] **T5.** Convertir `live_trading.kol_blacklist` en struct chain-aware. Permettre exclusion ETH-only vs SOL-only sans cross-pollution. Note : les "destroyers" identifiés Apr 26 doivent être re-vérifiés en paired-test (l'aggregate de cette époque était biaisé).

### Dead-day filter (priorité basse, exploratoire)
- [ ] **T6.** Brancher `_compute_day_regime` (sim.py) dans pipeline RT.
- [ ] **T7.** Tester en shadow avec set `DEADGATE_*`.

### Idées de nouvelles mécaniques (à coder si bandwidth)
- [ ] **T8.** **DELAY entry** (DELAY30/60) — attendre 30s après KOL call, vérifier prix tient avant d'acheter. Filtre les instant-rugs. ~30 lignes.
- [ ] **T9.** **CIRCUIT BREAKER** (CRASH5_30S) — exit si -5% en 30s. Rug-pull early-exit. ~50 lignes.
- [ ] **T10.** **VOLUME drop exit** — exit si rolling 1min volume < seuil. ~80 lignes.
- [ ] **T11.** **LIQ-pull exit** — exit si liquidity_usd drop >15%. Dev-pull detection.
- [ ] **T12.** **MULTI-KOL confirmation** — open seulement si 2+ KOLs callent dans X min.
- [ ] **T13.** **TIME-based BE** : déjà supporté via `time_be_minute` mais 0 strat l'utilise. Créer 5-10 shadows TIMEBE5_LOCK10/etc.

---

## 📌 RAPPELS PERSISTANTS

### Méthode statistique
- **TOUJOURS paired-test** sur tokens intersection paper×live, JAMAIS aggregate avg quand sample sizes diffèrent. Cf. Apr 27 leçon : aggregate -16.88pp = artefact du selection bias, paired -7.46pp = vrai signal.
- N≥30 par strat avant verdict, N≥10 par KOL avant blacklist.

### KOL routing
- KOL whitelist : DISABLED.
- Per-chain destroyers : à re-verifier en paired-test (les listes Apr 24-26 sont basées sur aggregate biaisé). Prerequisite T5.
- **MaestrosDegen** : retiré du scraping Apr 26 (1142 paper trades + 481 ticks wipés).

### Bankroll
- Current : $53,409 / starting $29,000 (Apr 27 = +6 promoted strats à $1000).
- SOL live position size : ~$1.70/trade (0.01 SOL × $170/SOL).
- ETH live position size (Phase 1) : **$20/trade**, max 1 open simultanée.

### Calibration ETH
- v14e.28 : Gas $1.50/side, Slip 100 bps base, Min position paper $50. Empirique 26 avril (ETH base_fee 0.5-1.5 gwei). Rerun smoke si base_fee > 5 gwei.
- Phase 1 microtest = collecte empirique slippage sur tokens KOL réels (≠ PEPE). Verdict E2 ci-dessus.

### Cohérence sim/paper/live ETH (v14e.31)
- Tous layers à position $50 + slip kernel `_evm_slip_bps_with_gas`. Live Phase 1 à $20 = écart cost-drag connu (+12pp), à corriger lors de scale-up.

### Mega-sweep
- ETH workflow `mega-sweep-eth-48h.yml` cron 22:00 UTC tous les 2 jours.
- SOL workflow `mega-sweep-48h.yml` cron 02:00 UTC tous les 2 jours.
- Triple gate FDR<alpha opt-in via `--require-fdr` (default OFF).

### Strats deck (484 total)
| Family | SOL | ETH |
|--------|-----|-----|
| BE | 33 | 9 |
| LOCK | 33 | 20 |
| FAST | 73 | 18 |
| SLOW | 13 | 5 |
| SCALP | 36 | 17 |
| Other | 199 | 10 |
| AGE clones | 38 | 18 |
