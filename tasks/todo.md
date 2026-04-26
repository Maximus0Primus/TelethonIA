# Pipeline Status — Updated Apr 26, 2026 (v14e.28 deployed)

État courant : ETH live infra déployée + recalibrée empiriquement (gas $1.50/side, slip 100 bps), close-side wired, dedup case-bug fix, AGE clones ETH ajoutés, mega-sweep ETH workflow nightly. Paper: 2 ETH strats actives à $50/trade ($1000 bankroll chacune). Live SOL = BE25_TP80_SL30 + BOND_FAST_TP50_SL20_T20.

L'historique des décisions se lit dans le git log — ce TODO ne garde que ce qui est encore à faire.

---

## 🎯 EN COURS — surveillance des shadows + ETH paper rebuild

Pas de code à écrire, juste observer la data grossir.

### À N≥30 par strat (estimé Mai 02-06)

- [ ] **W1.** Paired-test `SCALP_TP15_SL20_S35` vs base `SCALP_TP15_SL20`. Si score gate ajoute >+2pp avg/trade → promouvoir 1-2 SCALP en paper main + Telegram.
- [ ] **W2.** Paired-test des AGE clones SOL vs leur parent : `AGE24_BE25_TP80_SL30` vs `BE25_TP80_SL30`, etc. Mesurer le delta avg/WR par bande d'âge.
- [ ] **W3.** **NEW** — Paired-test des AGE clones ETH (`AGE24_ETH_TP80_SL40_T2H`, `AGE48_ETH_TP80_SL40_T2H`, `AGE24_ETH_FAST_TP100_SL20`, `AGE48_ETH_FAST_TP100_SL20`) vs leur parent. Hypothèse de base : age band 12-24h et 24-48h saigne sur ETH (rerank Apr 26 = -23.5% N=94). Re-vérifier avec les samples qui s'accumulent post-fix.

### Mega sweep auto

- [ ] **W4.** **SOL** — au prochain run cron (~Apr 28 02:00 UTC), vérifier le ranking avec age dimension activée.
- [ ] **W5.** **ETH** — premier run nightly via `mega-sweep-eth-48h.yml` à 22:00 UTC (~Apr 26-27). Vérifier les artifacts `_mega_sweep_eth_top_robust.csv`. Voir si AGE bands ETH montrent un signal différent du rerank actuel.

---

## 🚧 PHASE A ETH live — INFRA DÉPLOYÉE, attente data paper

État : tout est câblé, gated derrière `live_trading.eth_live_enabled=False` en DB. Paper actif depuis Apr 26 16h UTC sur 2 strats × $50/trade × $1000 bankroll.

### Étapes complétées (Apr 26)
- [x] Wallet ETH `0xC5c92E3AC207f686D09686Fe1dE79a302D9410E9` configuré sur VPS (env vars `ETH_PRIVATE_KEY` + `ETH_RPC_URL`).
- [x] `web3` + `eth-account` installés.
- [x] Open + close path ETH wired (`safe_scraper._rt_open_trades` + `live_trader_eth.check_live_trades_eth`).
- [x] Smoke test round-trip réel sur PEPE ($7) → gas $1.76 round-trip empirique. Data dans `data/eth_smoke_20260426T132313Z.json`.
- [x] **2 bugs critiques fixés** via le smoke test :
  - SELL retournait WETH ERC20 → multicall avec `unwrapWETH9` (commit `1d2f5af`)
  - BUY tokens_received=0 par RPC lag → parsing Swap event log
- [x] **Recalibration ETH empirique** (commit `59af9df`) : gas 7.50→1.50, slip 200→100 bps, min position 200→50. Sim/paper/live cohérence chain-aware (compute_buy_slippage + _dynamic_sell_slippage + _exit acceptent `chain` param).
- [x] **Bug dedup case-sensitive fixé** (commit `72f4ed2`) : adresses ETH normalisées lowercase via `chain_detect.normalize_address`. Bug observé Apr 26 sur $HENRY (Luca_Apes 2 positions 4min apart, mixed-case vs lowercase).
- [x] **Default `max_age_hours=12` pour EVM** (commit `72f4ed2`) : SOL convention pre-AGE clones étendue à ETH. Strats opt-out en déclarant `max_age_hours` explicitement.
- [x] **4 ETH AGE shadow clones** (commit `f3c82fc`) : AGE24/AGE48 × ETH_TP80_SL40_T2H + ETH_FAST_TP100_SL20.
- [x] **Bankrolls reset** (Apr 26) : `ETH_TP80_SL40_T2H` $1000 / `ETH_FAST_TP100_SL20` $1000. `ETH_FAST_TP40_SL30` retiré (loser -8.5%). Position $50/trade auto via cap global.
- [x] **Workflow ETH mega-sweep** (commit `f3c82fc`) : `mega-sweep-eth-48h.yml` à 22:00 UTC tous les 2 jours. Triggered manuellement Apr 26 14:08 UTC pour data initiale.

### Étapes restantes pour activer ETH live
- [ ] **A1.** Refunder le wallet : actuel $43.52 → ~$300 (pour 4-5 positions $50 simultanées + buffer gas). Wallet brûlé pour cette session — rotater la clé en même temps si possible.
- [ ] **A2.** Attendre que les 2 ETH paper mains atteignent N≥30 chacune (post-reset Apr 26 → estimé Mai 03-06) avec cum_pnl positif sur 7j glissants.
- [ ] **A3.** Configurer `live_trading.eth_allocations` en DB : `{"ETH_TP80_SL40_T2H": 0.5, "ETH_FAST_TP100_SL20": 0.5}`.
- [ ] **A4.** Per-chain KOL blacklist : ajouter `live_trading.kol_blacklist.ethereum` avec les 5 destroyers ETH (`marcellsfightclub, maythousdegens, batmansafucalls, neocallss, animegems`).
- [ ] **A5.** Flipper `live_trading.eth_live_enabled=True`.
- [ ] **A6.** Surveillance 24-72h : cohérence paper↔live drift, gas par trade, slippage, Flashbots latency.

### Wallet état (Apr 26 fin journée)
- Balance: 0.01867 ETH = $43.52
- Coût total smoke tests : $4.61
- Adresse publique : `0xC5c92E3AC207f686D09686Fe1dE79a302D9410E9`

### Sécurité (rappel critique)
- La clé privée du wallet a été collée en clair dans le transcript de la session Apr 26. À considérer compromise. Rotation prévue avant mise en prod.

---

## 🔧 BACKLOG TECH

### SELL slip drift monitoring
- [x] Calibrator twin-pair `scripts/_calibrate_sell_slip.py`. Verdict Apr 26 : N=120, weighted median Δ = −0.16 pp → **v144 calibration toujours valide**.
- [ ] **T1.** Câbler le script dans `mega-sweep-48h.yml` workflow comme step post-sweep.
- [ ] **T2.** Re-run quand N≥200 twin pairs (~Mai 03-05).

### Dead-day filter (priorité basse — exploratoire)
- [ ] **T3.** Brancher `_compute_day_regime` (sim.py) dans le pipeline RT : flagger `regime=dead` côté safe_scraper.
- [ ] **T4.** Tester en shadow d'abord avec un set `DEADGATE_*`.

### KOL blacklist multi-chain
- [ ] **T5.** Convertir `live_trading.kol_blacklist` en struct chain-aware (actuellement liste plate). Permettre exclusion ETH-only sans toucher SOL. Prérequis pour A4.

---

## 📌 RAPPELS PERSISTANTS

- **KOL whitelist** : DISABLED globalement. Per-chain destroyers identifiés :
  - SOL : `jadendegens, CarnagecallsGambles, ChairmanDN1, bounty_journal, DegenSeals, explorer_gems`
  - ETH : `marcellsfightclub, maythousdegens, batmansafucalls, neocallss, animegems` (WR=0%)
  - **Note** : `DegenSeals` et `explorer_gems` saignent SOL mais sont winners ETH (WR ≥ 83%) — confirmation que le routing per-chain est nécessaire.

- **Bankroll** : current $19,766 / starting $23,000 (DD −$3,234). SOL live = $20.

- **ETH calibration** : v14e.28 Apr 26 — gas $1.50/side, slip 100 bps, min position $50. Re-run `scripts/_eth_round_trip_smoke.py` si base_fee ETH dépasse 5 gwei (rare depuis mi-2025) — la constante gas devra ~3× bumper.

- **SELL slip model SOL** : v144 calibration (`_dynamic_sell_slip_factor` avec type_bps + GLOBAL_OFFSET=−100) valide. Ne PAS toucher sans `scripts/_calibrate_sell_slip.py` empirique.

- **Triple gate mega-sweep** : FDR<alpha opt-in via `--require-fdr` (default OFF).
