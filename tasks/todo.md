# Pipeline Status — Updated Apr 26, 2026 (v14e.27 deployed)

État courant : 34 nouvelles shadow strats post-mega-sweep + 12 AGE clones + age dimension dans le mega-sweep grid + SELL slip health-check récurrent. ETH bleeders demoted en shadow, 3 mains ETH restants. Live SOL = BE25_TP80_SL30 + BOND_FAST_TP50_SL20_T20.

L'historique des décisions se lit dans le git log (commits e8239b4, a6772df, 0deef5b, 4f6ee3c…) — ce TODO ne garde que ce qui est encore à faire.

---

## 🎯 EN COURS — surveillance des 34 nouvelles shadows + AGE clones

Pas de code à écrire, juste observer la data grossir.

### À N≥30 par strat (estimé Mai 02-06)

- [ ] **W1.** Paired-test `SCALP_TP15_SL20_S35` vs base `SCALP_TP15_SL20` (sans filter). Si score gate ajoute >+2pp avg/trade → promouvoir 1-2 SCALP en paper main + Telegram.
- [ ] **W2.** Ranking ETH-only sur les 17 ETH clones — sortir top 3 et virer les autres.
- [ ] **W3.** Paired-test des AGE clones vs leur parent : `AGE24_BE25_TP80_SL30` vs `BE25_TP80_SL30`, etc. Mesurer le delta avg/WR par bande d'âge.

### Mega sweep auto (cron 48h)

- [ ] **W4.** Au prochain run (~Apr 28 21:00 UTC), vérifier le ranking avec age dimension activée. Le top robust devrait maintenant inclure des `(strat, age_band)` couples — voir si AGE24/AGE48 surperforme `ALL` sur les top strats.

---

## 🔧 BACKLOG TECH

### SELL slip drift monitoring (recurring health-check)

- [x] Calibrator twin-pair `scripts/_calibrate_sell_slip.py` créé. Mesure `(pnl_live − paper_sim_pnl_pct)` sur chaque rt_live trade (méthodo v144). Verdict Apr 26 : N=120, weighted median Δ = −0.16 pp → **v144 calibration toujours valide**, pas de code change.
- [ ] **T1.** Câbler le script dans `mega-sweep-48h.yml` workflow comme step post-sweep — log dans artifacts. Détecte automatiquement si la dérive franchit le seuil 1pp.
- [ ] **T2.** Re-run quand N≥200 twin pairs (~Mai 03-05) pour confirmer la stabilité.

### Dead-day filter en RT (priorité basse — exploratoire)

- [ ] **T3.** Brancher le `_compute_day_regime` (sim.py:4252) dans le pipeline RT : flagger `regime=dead` côté safe_scraper et couper les ouvertures de strats non-robust quand le pump_rate live tombe <15%. Hypothèse : BE25/FAST classics arrêtent de saigner les jours dead.
- [ ] **T4.** Tester en shadow d'abord : ajouter un nouveau set de `DEADGATE_*` shadows qui n'ouvrent que les jours non-dead.

---

## 🚧 PHASE A ETH live — REPRISE Apr 26 (suspension levée par décision utilisateur)

Décision : on avance malgré le crash ETH paper du 25 avril, en gating l'ouverture derrière `live_trading.eth_live_enabled` (default False) jusqu'à validation empirique.

### État actuel
- ✅ Wallet ETH dédié `0xC5c92E3AC207f686D09686Fe1dE79a302D9410E9` configuré dans `scraper/.env` VPS (`ETH_PRIVATE_KEY` + `ETH_RPC_URL=rpc.flashbots.net`)
- ✅ `web3` + `eth-account` installés sur le venv VPS
- ✅ Smoke test dry-run validé sur PEPE — gas $0.77 one-way / $1.54 round-trip à base_fee 0.19 gwei (calme). Gas $5-15 round-trip attendu en conditions normales.
- ✅ `safe_scraper._rt_open_trades` câblé pour `chain=='ethereum'` → `live_trader_eth.open_live_trade`, gated derrière `live_trading.eth_live_enabled`
- ❌ **Close-side NON câblée** — `check_live_trades` (live_trader.py:1113) reste Solana-only. Flipper `eth_live_enabled=True` aujourd'hui = positions qui ne peuvent PAS s'auto-clôturer.

### Étapes complétées (Apr 26)
- [x] **A1.** User a envoyé 0.0207 ETH ($48) sur `0xC5c9…10E9` (Apr 26, ~13h00 UTC).
- [x] **A2.** Round-trip empirique sur PEPE ($7 swap) : data permanente dans `data/eth_smoke_*.json`. Gas réel BUY $0.89-1.12 / SELL $0.87-1.11. Slip pur ~0bps. Round-trip cost dominé par gas, swap slip négligeable.
- [x] **A3.** `check_live_trades_eth` câblé dans `live_trader_eth.py` (commit `754cd86`). Open path insert DB + close path full mirror.
- [x] **A4.** Deploy `b9c03eb` + `754cd86` + `1d2f5af` sur VPS. 117 tests passent.
- [x] **A5 bonus.** 2 bugs critiques `live_trader_eth` trouvés et fixés via le smoke test (commit `1d2f5af`) :
  - SELL retournait WETH ERC20, pas ETH natif → fix multicall avec unwrapWETH9
  - BUY reportait `tokens_received=0` (RPC read-after-write lag) → fix parsing du Swap event log
- [x] **A6.** Helper `unwrap_weth_balance()` ajouté pour récupérer le WETH orphan créé pendant le 1er smoke (avant fix). Wallet propre.

### Étapes restantes
- [ ] **A7.** Configurer `live_trading.eth_allocations` en DB. Démarrer petit : `{"ETH_TP80_SL40_T2H": 1.0}` à 100% sur 1 seul strat (le moins risqué des 3 ETH paper mains restantes).
- [ ] **A8.** Position size ETH initiale : $10/trade max (gas représente ~22% à cette taille — viable mais marge serrée). Bankroll ETH allouée du wallet : ~$40 (= 4 trades simultanés max).
- [ ] **A9.** Flipper `live_trading.eth_live_enabled=True` en DB.
- [ ] **A10.** Surveillance 24h-72h : cohérence paper↔live drift, gas réel par trade, slippage, Flashbots latency. Comparer aux numéros empiriques captured Apr 26.

### Wallet état Apr 26 (post-tests)
- Balance: 0.01867 ETH = $43.52
- Coût total des smoke tests : $4.61 (= 2 round-trips + 1 unwrap manuel + slips)
- Adresse publique : `0xC5c92E3AC207f686D09686Fe1dE79a302D9410E9`

### Sécurité (rappel critique)
- La clé privée `0xd270…da69` du wallet `0xC5c9…10E9` a été collée en clair dans le chat de cette session (transcript persistant). À considérer compromise. Déploie une rotation dès que ETH live est validé en infra.

---

## 📌 RAPPELS PERSISTANTS

- **KOL whitelist** : actuellement DISABLED. Les 6 KOL destroyers (jadendegens, CarnagecallsGambles, ChairmanDN1, bounty_journal, DegenSeals, explorer_gems) saignent en paper+live. Décision pendante : blacklist dans `live_trading.kol_blacklist` ou activer whitelist mode ?
- **Bankroll** : current $19,766 / starting $23,000 (DD −$3,234, peak $25,831). SOL live = $20 (post nuit Apr 25-26).
- **Triple gate mega-sweep** : FDR<alpha désormais opt-in via `--require-fdr` (default OFF). Top robust publie 30 SCALP_TP15_SL20 variants depuis le run Apr 26.
- **SELL slip model** : v144 calibration (`_dynamic_sell_slip_factor` avec type_bps + GLOBAL_OFFSET=−100) confirmée valide par twin-pair drift checker. Ne PAS toucher sans re-runner `scripts/_calibrate_sell_slip.py` d'abord.
