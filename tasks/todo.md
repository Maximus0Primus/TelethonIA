# Pipeline Status — Updated Apr 29, 2026 (v14e.42 deployed)

## v14e.42 (Apr 29) — closing_retry sentinel-leak + ETH bigint overflow

**Bug 1 (DB cosmétique) — `closing_retry` trap.** Quand un sell ETH reverted/timeout, status passait `open→closing`. Le retry path à `live_trader_eth.py:1248` settait `ev["status"]="closing_retry"` (sentinelle) ; après retry sell réussi, `closing_retry` était écrit comme status terminal. Le filtre de scan `in_("status",["open","closing"])` n'incluait pas `closing_retry` → row jamais retouchée. **6 ETH live (CYB, APHEX, MOMMYS, ARMA, SCAM, PARANOID)** ont été silencieusement bloqués alors que les sells avaient bien minée on-chain (vérifié via `eth_getTransactionReceipt`). Fix : (a) inclure `closing_retry` dans le filtre, (b) renommer sentinelle en `force_close`, (c) résoudre `force_close` post-sell en vrai status terminal (`tp_hit`/`sl_hit`/`timeout`) à partir de `pnl_pct` vs `STRATEGIES[strategy]` thresholds. Symétrique sur `live_trader.py` SOL. Recovery DB via `scripts/_recover_stuck_eth.py` (6 rows fixées : 1 tp_hit, 2 sl_hit, 3 timeout).

**Bug 2 (CRITIQUE silent loss) — `$INCOME` orphan.** Buy on-chain réussi (tx `0xb435ab7e...`, 29.13 INCOME reçus), mais insert DB rejected par Postgres : `value "29136565813671862919" is out of range for type bigint`. Token avec 18 decimals + price très bas → raw token amount = 2.91e19 > bigint max (9.22e18). Le `CRITICAL` log a fired mais jusqu'à ce jour aucun monitoring ne le remontait. L'utilisateur a dû vendre manuellement via Rabby ~15h plus tard. Fix : caps préventifs sur `buy_input_lamports`, `buy_output_tokens`, `sell_output_lamports` (None si > 9e18) + `rt_is_pump_fun` cast `int(bool(...))` (DB col est int2, un True natif Python crashait aussi). Row INCOME insérée rétroactivement (`status=manual_recovered`, `entry_source=manual_recover`, id=270539).

**À monitorer post-deploy v14e.42 :** plus aucun `CRITICAL: ETH live trade ... bought ... but DB insert failed` ne doit apparaître. Si ça réapparaît, c'est un autre champ ETH qui overflow → ajouter au cap.

## v14e.42 (Apr 29 second pass) — todo cleanup batch

**T5 ✅** — `BOND_FAST_TP50_SL20_T20` retiré du live SOL config. `live_trading.allocations` désormais `{BE25_TP80_SL30: 1.0}` seul. Backup `data/rt_trade_config_pre_t5_20260429T144956Z.json`. Le bot reload le config toutes 5min — propagation auto, no restart needed.

**T6 ✅** — `ETH_FAST_TP500_SL40_60M` confirmé paper-only (pas dans `eth_allocations`).

**R1 ✅** — RT lag fix v14e.34 tient : msg→detect max 24.7s sur dernière heure, < 30s threshold.

**K4 ✅** — Post-fix v14e.40 (Apr 28 15:50 UTC) : 0 SOL main de KOL SOL-blacklisté. Les 28 mains ETH `mad_apes_gambles` détectés sont ATTENDUS (mad_apes RELIABLE_WINNER ETH, blacklist par-chain).

**T7 ✅** — `.github/workflows/kol-weekly-audit.yml` créé. Cron lundi 06:00 UTC, lance `_kol_reliable_audit --exclude-artifact-strats` + `_kol_recovery_check`, alerte Telegram si nouveaux candidates, upload CSVs en artifact 30j.

**W8 ⚠️ NEW BLOCKER** — `mega_sweep_runs` ne se remplit plus depuis Apr 25 21:51 (160 rows). Les 4 derniers crons SOL ont été CANCELLED par GH après 6h05m. Le `timeout-minutes: 720` dans le YAML est ignoré : **les hosted runners GH ont une hard limit de 6h par job**. Le commit Apr 28 `34c99ce` "fix(v14e.41): mega sweep cron — SOL timeout fix" n'a PAS résolu (les workers ont été bumpés 2→4 mais le scope de sweep dépasse encore 6h). Solutions possibles :
- (a) split en matrice de jobs (chain×family) → chacun < 6h
- (b) self-hosted runner (le VPS pourrait servir mais charge déjà la prod)
- (c) réduire le scope sweep (moins de strats × moins de filters)
- ETH sweep tourne OK (1-3h, dernière run réussie Apr 27 22:56).

**T1 ⏸ DIFFÉRÉ** — `_calibrate_sell_slip.py` filtre `chain='solana'` only. Inutile de le câbler dans le ETH workflow. Le SOL workflow est cassé (W8 ci-dessus). Tant que W8 pas résolu, T1 ne tourne pas. À débloquer ensemble.

---

# Pipeline Status — Updated Apr 28, 2026 15:50 UTC (v14e.41 deployed)

État courant :
- **ETH live Phase 1 microtest ACTIF** depuis 20:12 UTC. Live SOL = BE25_TP80_SL30 + BOND_FAST_TP50_SL20_T20 (la deuxième est artifact-deprecated v14e.36 — à retirer du live config). Live ETH = `eth_live_enabled=True`, 1 strat (`ETH_TP80_SL40_T2H`), pos $20, max 1 open.
- **KOL chain-blacklist v14e.40 RÉPARÉ** : v14e.37/38 avait shipping-bug — `_load_paper_trade_config` filtrait `kol_chain_blacklist` hors du dict retourné car la clé n'était pas dans `defaults`. Fix v14e.40 : ajout dans `defaults`. Du 26 au 28 Apr, mad_apes + 12 autres KOLs blacklistés ont quand même ouvert ~343 paper mains SOL (-$2 752 paper). Live indemne (gate `live_trading.kol_blacklist` séparé). Post-fix le gate paper main + safe_scraper RT live re-fonctionnent.
- **RECALL family v14e.40 + v14e.41** : 60 strats shadow couvrant les 2 modes drift (DIP vs 1st-call entry × PEAK vs post-1st ATH) × les mécaniques validées (BE15/25/30, LOCK5/10/15, FAST45/60, SLOW4H/6H, SCALP, DECAY, NZ, S30/S40, MCAP_MID, AGE2H/6H/24H, wide-TP TP100→TP500). v14e.41 ajoute le mode `drift_vs_peak` après replay $PARANOID 27 Apr (drift_vs_1st=-21% → bloqué par DIP30, mais drift_vs_peak=-54% capture pump-then-dump → +220% post-recall). Gate temporel resserré 1800s→600s. SL≥-50% obligatoire sur PEAK strats car le dump-mèche tape SL30/SL40 avant la recovery (replay $PARANOID : SL30 fired @ t+8min, TP200 @ t+52min → seul `RECALL_PEAK50_TP200_SL50_6H` aurait gagné). 5 strats opened sur scenario PARANOID, 30 sur deep-dip scenario, 0 sur non-recall — filtres validés.
- **Trail/dip/split shadows DEPRECATED v14e.36** : 119 strats artefact retirées. Aucun nouveau shadow DTRAIL/PTRAIL/SPLIT/DIP/BOND/TD2/MCAP_DTRAIL.

L'historique des décisions se lit dans le git log — ce TODO ne garde que ce qui est encore à faire.

---

## 🚨 EN COURS — ETH live Phase 1 (ALIENPEPE résolu)

- [x] **E1.** Premier trade live ETH ouvert : **$ALIENPEPE** (Apr 27 20:12 UTC, route=v2, gas $1.35, quote_slip 0bps, ds_slip 98bps). v14e.33 = Uniswap V2 fallback validé.
- [x] **E1b.** **$ALIENPEPE désync DB ↔ on-chain — résolu Apr 28**. Le sell V2 a bien miné block 24974330 (22:20:11 UTC, tx `0x0bb165f1...`, 0.01524 ETH reçus = $34.84) mais le DB update qui suit `execute_sell` n'est jamais arrivé (probable crash/restart VPS entre mine et `.update()`). Le retry path `live_trader_eth.py:1022` re-soumettait le sell à chaque cycle alors que le wallet était à 0 token → revert silencieux indéfini. Trade resyncé manuellement via `scripts/_eth_alienpepe_db_resync.py` : status=`timeout`, exit $0.0003279, pnl **+74.22% (+$14.84)**, exit_minutes=127, bankroll ETH +$14.84. Fix systémique déployé : `_finalize_orphan_eth_sell()` dans `live_trader_eth.py` détecte (status=closing AND tx_signature_exit IS NULL AND wallet_balance=0), retrouve la sell tx via Transfer logs wallet→pool, parse ETH reçu (V3/V2/trace fallback), Chainlink ETH/USD au block exact, écrit la row + bankroll. Hooké en tête de boucle dans `process_open_trades`. **Reste à push VPS + verify next cycle.**
- [ ] **E2.** Après 5-10 trades fermés (ETA 1-3 jours selon volume KOL ETH) : `python scripts/_eth_microtest_recap.py`. Verdict :
  - drift médian > -3pp → Phase 2 ($50/trade, 2 strats)
  - drift -3 à -7pp → continuer collecte
  - drift < -7pp → abort + recalibrer `ETH_BUY_SLIPPAGE_BPS` empiriquement
- [ ] **E3.** Top up wallet ETH : actuel $43 → $80-100. Adresse : `0xC5c92E3AC207f686D09686Fe1dE79a302D9410E9`.
- [ ] **E4.** **Rotation clé wallet ETH** — la clé est compromise (transcript persistant). À faire AVANT scaling Phase 2.
- [ ] **E5.** **Race condition ETH dispatch** : `threading.Lock` autour de `live_trader_eth.open_live_trade` (~5 lignes). Pas urgent, eth_max_open=1 borne le risque.

---

## 🚨 EN COURS — KOL audit weekly cycle

Goal : confirmer blacklist tient, détecter recovery / nouveaux pourris.

- [ ] **K1.** Lundi 06:00 UTC (cron à wire) : `python scripts/_kol_reliable_audit.py --days 14 --bootstrap 10000 --exclude-artifact-strats` → propose nouveaux blacklist candidates.
- [ ] **K2.** Lundi 06:30 UTC (cron à wire) : `python scripts/_kol_recovery_check.py --days 7 --block-at 2026-04-27T21:18Z` → propose un-blacklist candidates.
- [ ] **K3.** **ETH PROBABLE_BL à arbitrer** : jadendegens (N=20 WR 0% IC[-39%, -22%]), aliensalphacalls (N=20 WR 0% IC[-22%, -17%]). N<30 mais WR=0% sur 20 = p<10⁻⁶. Counterfactual 48h gain : +$2 163 ETH. Verdict utilisateur : ajouter ou attendre N=30 pour passer en RELIABLE.
- [x] **K4.** Vérifié Apr 29 14:45 UTC — 995 shadow rows en 24h des 14 KOLs blacklistés. 0 SOL main post-deploy v14e.40. Per-chain gate fonctionne. ✅

---

## 🎯 EN COURS — observation des shadows

Pas de code à écrire, juste laisser la data grossir. ETA verdicts paired-test : **Mai 03-10**.

### À N≥30 par strat
- [ ] **W1.** Paired-test `SCALP_TP15_SL20_S35` vs base.
- [ ] **W2.** Paired-test des AGE clones SOL vs leur parent.
- [ ] **W3.** Paired-test des AGE clones ETH (4 existants + 8 v14e.31).
- [ ] **W4.** Paired-test **LOCK family** vs BE base : LOCK10 SOL +1.88pp / ETH +2.85pp en backtest.
- [ ] **W5.** Validation **ETH winner cluster** (26 strats v14e.31).
- [x] **W9a.** **RECALL sweep extended v14e.41** (`scripts/_recall_sweep.py`) — backtest 14d sur 30 exit specs × 11 universes (first_call + 6 recall buckets + 4 unions). Wired dans mega-sweep-48h.yml. **Top findings 14d** :
  - **Union beats first_call alone** : FAST60_TP40_SL30 cumul $sum +358 → +534 (+49%) en ajoutant recall_dip30 (N=295 vs 279).
  - **Recall isolés sont massivement positifs** : SLOW6H_TP50_SL30 sur recall_dip30 = +33% EV WR 81% N=16, SLOW6H_TP100_SL50 sur recall_peak30 = +22% EV N=25, SLOW4H_TP50_SL30 sur recall_peak70 = +22% EV WR 73% N=15.
  - Confirme : ajouter les recall events à n'importe quelle exit-spec déjà profitable booste son $sum cumulé sans dégrader le N.
- [ ] **W9b.** **RECALL family v14e.40 + v14e.41** (60 strats shadow — 36 DIP + 24 PEAK, SOL+ETH). Wait N≥30 par bucket avant verdict. Découpage paired-test :
  - DIP vs first_call_price : DIP10/DIP30/DIP50 × {plain TP/SL, BE+LOCK, FAST/SLOW timeouts, SCALP, DECAY, NZ/S30/S40/MCAP gates, AGE2H/AGE6H/AGE24}
  - PEAK vs post-1st-call ATH : PEAK30/PEAK50/PEAK70 × {SL≥50% mandatory, BE+LOCK, SCALP, wide-TP TP100→TP500}
  - Validation $PARANOID confirmée : `RECALL_PEAK50_TP200_SL50_6H` capture +200% TP @ t+52min, les SL30/40 strats sortent à t+8min sur la mèche. Vérifier ce pattern se reproduit sur N≥30.
  - ETA verdict : Mai 26-Juin 5 selon volume recalls.

### Mega sweep auto (cron 48h)
- [x] **W6.** SOL : prochain run cron 02:00 UTC avec `--persist` → DB `mega_sweep_runs`.
- [x] **W7.** ETH : prochain run 22:00 UTC avec chain-aware position $50.
- [⚠️] **W8.** **BLOQUÉ** — `mega_sweep_runs` arrêté à Apr 25 21:51 (160 rows, plus rien depuis). 4 derniers crons SOL CANCELLED après 6h05m (hard limit GH hosted runner). Voir batch v14e.42 en tête. ETH OK.

---

## 🚨 À VALIDER — RT lag fix v14e.34

Apr 27 20:12 UTC : burst de 5 KOL calls détectés avec `msg→detect` 700-1050s (14-17 min de retard). Fix v14e.34 : tous les blocs lourds wrappés en `await asyncio.to_thread(...)`.

- [x] **R1.** Vérifié Apr 29 14:45 UTC — msg→detect max 24.7s sur 1h. ✅
- [ ] **R2.** Si lag persiste : profiler `process_and_push` ou `check_paper_trades` pour trouver le bloc qui mange 1000s+.

---

## 🔧 BACKLOG TECH

### Calibration drift (recurring)
- [⏸] **T1.** Différé — `_calibrate_sell_slip.py` est SOL-only et le SOL sweep est cassé (W8). À débloquer ensemble.
- [ ] **T2.** Re-run sell slip drift quand N≥200 twin pairs (~Mai 03-05).
- [ ] **T3.** Re-run `_eth_round_trip_smoke.py --execute` mensuellement OU si base_fee ETH > 5 gwei.
- [ ] **T4.** Câbler `eth_daily_loss_limit_usd` dans le ETH dispatch.

### Live SOL config cleanup (post v14e.36)
- [x] **T5.** Apr 29 — retiré, `live_trading.allocations` = `{BE25_TP80_SL30: 1.0}` seul. ✅
- [x] **T6.** Apr 29 — confirmé paper-only (`eth_allocations` = `{ETH_TP80_SL40_T2H: 1.0}`). ✅

### Auto-recovery wiring
- [x] **T7.** Apr 29 — `.github/workflows/kol-weekly-audit.yml` créé (cron lundi 06:00 UTC, alerte Telegram, artifacts 30j). ✅
- [ ] **T8.** Si signal stable post-K1/K2 sur 4 semaines → auto-apply JSONB diff via PR auto-générée (review humain garde le merge).

### Dead-day filter (priorité basse)
- [ ] **T9.** Brancher `_compute_day_regime` (sim.py) dans pipeline RT.
- [ ] **T10.** Tester en shadow avec set `DEADGATE_*`.

### Idées de nouvelles mécaniques (à coder si bandwidth)
- [ ] **T11.** **DELAY entry** (DELAY30/60).
- [ ] **T12.** **CIRCUIT BREAKER** (CRASH5_30S).
- [ ] **T13.** **VOLUME drop exit**.
- [ ] **T14.** **LIQ-pull exit**.
- [ ] **T15.** **MULTI-KOL confirmation** — open seulement si 2+ KOLs callent dans X min.
- [ ] **T16.** **TIME-based BE** : déjà supporté via `time_be_minute` mais 0 strat l'utilise.

---

## 📌 RAPPELS PERSISTANTS

### Méthode statistique
- **TOUJOURS paired-test** sur tokens intersection paper×live, JAMAIS aggregate avg quand sample sizes diffèrent.
- N≥30 par strat avant verdict, N≥30 par (KOL, chain) avant blacklist reliable (N=15-29 = probable, à observer 1 semaine).
- Bootstrap CI 95% + sign test obligatoires sur tout verdict KOL (script `_kol_reliable_audit.py`).
- Filtrer artefacts (DTRAIL/DIP/etc) du dataset audit avec `--exclude-artifact-strats`.

### KOL routing v14e.38
- **Per-chain blacklist active** : 14 SOL, 0 ETH. JSONB `paper_trade_config.kol_chain_blacklist`.
- mad_apes_gambles : SOL ban / ETH allow (RELIABLE_WINNER N=104 +$1815/d, best ETH_FAST_TP100_SL20).
- Telemetry préservée pour blacklist : kol_mentions + snapshots + ticks + paper SHADOWS continuent. Re-évaluation hebdo possible.
- KOL whitelist : DISABLED.

### Bankroll
- Current : $53,409 / starting $29,000.
- SOL live position size : ~$1.70/trade (0.01 SOL).
- ETH live position size (Phase 1) : **$20/trade**, max 1 open simultanée.

### Slippage v14e.34 single source of truth
- `strategies.BUY_SLIPPAGE_BPS = 225` (recalibration empirique 229 live trades, R²=5.8% sur 6 features).
- JSONB override path REMOVED v14e.34. Tous layers (paper/sim/sim_engines/optimize) importent depuis strategies.py.
- ETH : position-aware `_evm_slip_bps_with_gas` (gas $1.50 + slip 100 bps base + multipliers liq).
- 14,953 lignes pre-fix recalculées dans columns `pnl_pct_recalc` / `pnl_usd_recalc` / `buy_slippage_bps_recalc` (originaux préservés).

### Cohérence sim/paper/live ETH (v14e.31)
- Tous layers à position $50 + slip kernel `_evm_slip_bps_with_gas`. Live Phase 1 à $20 = écart cost-drag connu (+12pp).

### Mega-sweep
- ETH workflow `mega-sweep-eth-48h.yml` cron 22:00 UTC tous les 2 jours.
- SOL workflow `mega-sweep-48h.yml` cron 02:00 UTC tous les 2 jours.
- v14e.35 : **chaque run persist top-30+50 dans `mega_sweep_runs` Supabase**. Calibration sim-vs-actual via `_mega_sweep_calibration.py`.
- v14e.34 : robustness analyzer `_strat_slip_sensitivity.py` post-sweep — flag fragile strats whose $/d sign-flips at slip 100→600 bps.
- Triple gate FDR<alpha opt-in via `--require-fdr` (default OFF).

### Strats deck (577 dont 119 artefact-deprecated)
| Family | SOL | ETH |
|--------|-----|-----|
| BE | 33 | 9 |
| LOCK | 33 | 20 |
| FAST | 73 | 18 |
| SLOW | 13 | 5 |
| SCALP | 36 | 17 |
| Other | 80 | 10 |  *# was 199, -119 artefacts retirés*
| AGE clones | 38 | 18 |
| **RECALL DIP** (v14e.40+41) | **27** | **9** |  *# drift_vs_first_call_price*
| **RECALL PEAK** (v14e.41) | **18** | **6** |  *# drift_vs_post-1st-call ATH ($PARANOID pattern)*
