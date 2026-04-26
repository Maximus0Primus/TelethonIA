# Pipeline Status — Updated Apr 26, 2026 PM (v14e.31 deployed)

État courant : ETH live infra complète + recalibrée + cohérence sim/paper/live à $50. **484 strats** dont 25 LOCK, 19 AGE clones, 26 winners-cluster. MaestrosDegen retiré (1142 paper trades + 481 ticks effacés). Live SOL = BE25_TP80_SL30 + BOND_FAST_TP50_SL20_T20. Live ETH = câblé mais `eth_live_enabled=False`.

Dernière session ajouts : 85 nouveaux shadows en 4 vagues (LOCK base + LOCK extended + gap-fill + winner cluster). Mega-sweep position $10→$50 chain-aware. Alerter "slip" label corrigé en "missed peak".

L'historique des décisions se lit dans le git log — ce TODO ne garde que ce qui est encore à faire.

---

## 🎯 EN COURS — observation des shadows

Pas de code à écrire, juste laisser la data grossir. ETA verdicts paired-test : **Mai 03-10**.

### À N≥30 par strat
- [ ] **W1.** Paired-test `SCALP_TP15_SL20_S35` vs base `SCALP_TP15_SL20`. Si score gate ajoute >+2pp → promouvoir 1-2 SCALP en paper main + Telegram.
- [ ] **W2.** Paired-test des AGE clones SOL vs leur parent (AGE24/48_BE25 vs BE25, etc.).
- [ ] **W3.** Paired-test des AGE clones ETH (4 existants + 8 nouveaux v14e.31). Confirmer si ETH AGE12-band winner (mega-sweep Run 1 = +34% avg N=30) tient sur N élargi.
- [ ] **W4.** Paired-test **LOCK family** vs BE base : LOCK10 SOL = +1.88pp / LOCK10 ETH = +2.85pp en backtest tick-replay. Confirmer en paper N≥30.
- [ ] **W5.** Validation **ETH winner cluster** (26 strats v14e.31) : quel SL/TP/score gate exact donne le meilleur ratio dans la zone du winner FAST_TP100_SL20 × AGE12.

### Mega sweep auto (cron 48h)
- [ ] **W6.** **SOL** : prochain run cron 02:00 UTC. Position fixée à $10 SOL (Jupiter Ultra near-zero slip = position-indép, donc OK).
- [ ] **W7.** **ETH** : prochain run 22:00 UTC avec **chain-aware position $50** (commit `29f1870`). Premier ranking propre avec slip kernel matching paper/live.

---

## 🚧 PHASE A ETH live — INFRA COMPLÈTE, attente data paper

État : tout câblé, gated `eth_live_enabled=False`. Paper actif depuis Apr 26 13:00 UTC sur 2 strats × $50/trade × $1000 bankroll.

**Premières TP HITs Apr 26 17:56 UTC sur `$WIDE`** :
- `ETH_FAST_TP100_SL20` : +92% net (TP gross +100% − 4% slip = +96% computed, peak max +121%) → **$+46/trade** — match v14e.28 calibration parfaitement
- `ETH_TP80_SL40_T2H` : +72.8% net → **$+36.40/trade** — idem

→ **La calibration v14e.28 est validée empiriquement par les premiers TP HITs paper ETH.**

### Étapes restantes pour activer ETH live
- [ ] **A1.** Refunder le wallet : actuel $43.52 → ~$300 (4-5 positions $50 simultanées + buffer gas). **Rotation clé en même temps** (la clé du wallet est compromise via le transcript de session Apr 26).
- [ ] **A2.** Attendre que les 2 ETH paper mains atteignent N≥30 (~Mai 03-06) avec cum_pnl positif sur 7j.
- [ ] **A3.** Configurer `live_trading.eth_allocations` en DB : `{"ETH_TP80_SL40_T2H": 0.5, "ETH_FAST_TP100_SL20": 0.5}`.
- [ ] **A4.** Per-chain KOL blacklist : prérequis = T5 ci-dessous.
- [ ] **A5.** Flipper `live_trading.eth_live_enabled=True`.
- [ ] **A6.** Surveillance 24-72h : cohérence paper↔live drift, gas réel, slippage, Flashbots latency.

### Wallet état
- Balance : 0.01867 ETH = $43.52 | Adresse : `0xC5c92E3AC207f686D09686Fe1dE79a302D9410E9`
- **Sécurité** : clé privée compromise (transcript persistant). À rotater AVANT activation live.

---

## 🔧 BACKLOG TECH

### Calibration drift (recurring)
- [x] SELL slip twin-pair calibrator — Apr 26 verdict : N=120, weighted median Δ = −0.16 pp → v144 SOL calibration valide.
- [ ] **T1.** Câbler `_calibrate_sell_slip.py` dans `mega-sweep-48h.yml` post-sweep.
- [ ] **T2.** Re-run sell slip drift quand N≥200 twin pairs (~Mai 03-05).
- [ ] **T3.** Re-run `_eth_round_trip_smoke.py --execute` mensuellement OU si base_fee ETH > 5 gwei. Bumper `ETH_GAS_COST_USD_PER_SIDE` ~3× si gas/side > $4.

### KOL blacklist multi-chain
- [ ] **T5.** Convertir `live_trading.kol_blacklist` en struct chain-aware (actuellement liste plate vide). Permettre exclusion ETH-only (5 destroyers ETH) + SOL (5 destroyers SOL) sans cross-pollution. Prérequis pour A4.

### Dead-day filter (priorité basse, exploratoire)
- [ ] **T6.** Brancher `_compute_day_regime` (sim.py) dans pipeline RT : flagger `regime=dead` côté safe_scraper.
- [ ] **T7.** Tester en shadow avec set `DEADGATE_*`.

### Idées de nouvelles mécaniques (à coder si bandwidth)
- [ ] **T8.** **DELAY entry** (DELAY30/60) — attendre 30s après KOL call, vérifier prix tient avant d'acheter. Filtre les instant-rugs. ~30 lignes.
- [ ] **T9.** **CIRCUIT BREAKER** (CRASH5_30S) — exit si -5% en 30s. Rug-pull early-exit. ~50 lignes.
- [ ] **T10.** **VOLUME drop exit** — exit si rolling 1min volume < seuil. Data déjà capturée dans price_ticks. ~80 lignes.
- [ ] **T11.** **LIQ-pull exit** — exit si liquidity_usd drop >15%. Dev-pull detection. Data dans price_ticks.
- [ ] **T12.** **MULTI-KOL confirmation** — open seulement si 2+ KOLs callent dans X min. Conviction filter.
- [ ] **T13.** **TIME-based BE** activement utilisé : déjà supporté via `time_be_minute` dans tranche config mais 0 strat l'utilise. Créer 5-10 shadows TIMEBE5_LOCK10/etc.

---

## 📌 RAPPELS PERSISTANTS

### KOL routing
- **KOL whitelist** : DISABLED. **Per-chain destroyers** identifiés (à blacklist quand T5 fait) :
  - SOL : `jadendegens, CarnagecallsGambles, ChairmanDN1, bounty_journal, papicall` (papicall N=507 sum -$3984 sur 4j)
  - ETH : `marcellsfightclub, maythousdegens, batmansafucalls, neocallss, animegems` (WR=0%)
  - **Caveat** : `DegenSeals`, `explorer_gems` saignent SOL mais winners ETH → routing per-chain obligatoire.
  - **MaestrosDegen** : entièrement retiré du scraping Apr 26 (1142 paper trades + 481 ticks wipés). 5 tokens uniques à lui seul.

### KOL winners (paper/live profitables)
- ETH : `mad_apes_gambles` (N=25, WR 92%, +67%), `luca_apes` (N=63, WR 75%, +25%), `bat_gamble` (N=26, WR 50%, +14% — chain-restricted ETH only)
- SOL : `mad_apes_gambles` (N=1571, WR 72%, +11%)

### Bankroll
- Current : $48,125 / starting $23,000 (post-MaestrosDegen refund +$26,918). Peak $48,125. SOL live = $20.

### Calibration ETH (v14e.28)
- Gas $1.50/side | Slip 100 bps base | Min position $50
- Empirique du 26 avril (base_fee ETH 0.5-1.5 gwei, calme post-Pectra). Rerun smoke si base_fee > 5 gwei.

### Cohérence sim/paper/live ETH (v14e.31)
- Tous layers ETH à **position $50** + slip kernel `_evm_slip_bps_with_gas` + logic `_evaluate_trade_exit`.
- Mega-sweep ETH désormais chain-aware ($50, pas $10) → ranking valide ET valeurs absolues réalistes.

### SELL slip model SOL
- v144 calibration (`_dynamic_sell_slip_factor` avec type_bps + GLOBAL_OFFSET=−100) valide. Ne PAS toucher sans `_calibrate_sell_slip.py` empirique d'abord.

### Mega-sweep
- Triple gate FDR<alpha opt-in via `--require-fdr` (default OFF).
- ETH workflow `mega-sweep-eth-48h.yml` cron 22:00 UTC tous les 2 jours.
- SOL workflow `mega-sweep-48h.yml` cron 02:00 UTC tous les 2 jours.

### Strats deck (484 total — repartition)
| Family | SOL | ETH |
|--------|-----|-----|
| BE | 33 | 9 |
| **LOCK** (v14e.29-31) | 33 | 20 |
| FAST | 73 | 18 |
| SLOW | 13 | 5 |
| SCALP | 36 | 17 |
| Other (TP_only/TRAIL/DIP/etc.) | 199 | 10 |
| AGE clones (×N) | 38 | 18 |
