# Pipeline Status — Updated Apr 27, 2026 21:50 UTC (v14e.38 deployed)

État courant :
- **ETH live Phase 1 microtest ACTIF** depuis 20:12 UTC. Live SOL = BE25_TP80_SL30 + BOND_FAST_TP50_SL20_T20 (la deuxième est artifact-deprecated v14e.36 — à retirer du live config). Live ETH = `eth_live_enabled=True`, 1 strat (`ETH_TP80_SL40_T2H`), pos $20, max 1 open.
- **KOL chain-blacklist v14e.38 ACTIVE** : 14 SOL bans (mad_apes, papicall, markdegens, MaybachGambleCalls, ramcalls, leoclub69, CarnagecallsGambles, explorer_gems, ChairmanDN1, chiggajogambles, bounty_journal, DegenSeals, aliensalphacalls, LevisAlpha) + 0 ETH. Paper main + RT live gated, shadows + telemetry intacts. Counterfactual 48h : SOL -$3 854 → -$538, ETH +$985 → $985 (sans ban ETH yet). Total swing +$3 316/48j.
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
- [ ] **K4.** **Telemetry verification post-deploy** : confirmer que les 14 KOLs blacklist génèrent toujours des SHADOW trades (kol_mentions OK à 32min, mais 0 shadow rows observés vs 95 attendus). Re-check à 12h post-deploy. Si 0 shadows → bug dans la gate (devrait être `continue` outer-loop main only, shadow loop doit tourner).

---

## 🎯 EN COURS — observation des shadows

Pas de code à écrire, juste laisser la data grossir. ETA verdicts paired-test : **Mai 03-10**.

### À N≥30 par strat
- [ ] **W1.** Paired-test `SCALP_TP15_SL20_S35` vs base.
- [ ] **W2.** Paired-test des AGE clones SOL vs leur parent.
- [ ] **W3.** Paired-test des AGE clones ETH (4 existants + 8 v14e.31).
- [ ] **W4.** Paired-test **LOCK family** vs BE base : LOCK10 SOL +1.88pp / ETH +2.85pp en backtest.
- [ ] **W5.** Validation **ETH winner cluster** (26 strats v14e.31).

### Mega sweep auto (cron 48h)
- [x] **W6.** SOL : prochain run cron 02:00 UTC avec `--persist` → DB `mega_sweep_runs`.
- [x] **W7.** ETH : prochain run 22:00 UTC avec chain-aware position $50.
- [ ] **W8.** Vérifier après 1er run que `mega_sweep_runs` table se remplit. Vérifier aussi `data/slip_sensitivity_*.csv` (post-analyse v14e.34) et `data/sim_calibration_*.csv` (v14e.35).

---

## 🚨 À VALIDER — RT lag fix v14e.34

Apr 27 20:12 UTC : burst de 5 KOL calls détectés avec `msg→detect` 700-1050s (14-17 min de retard). Fix v14e.34 : tous les blocs lourds wrappés en `await asyncio.to_thread(...)`.

- [ ] **R1.** Vérifier sur le prochain cycle batch (ETA Apr 27 ~22:00 UTC) que `msg→detect` reste < 30s. `journalctl -u kol-scraper --since '1h ago' | grep "msg.detect"`.
- [ ] **R2.** Si lag persiste : profiler `process_and_push` ou `check_paper_trades` pour trouver le bloc qui mange 1000s+.

---

## 🔧 BACKLOG TECH

### Calibration drift (recurring)
- [ ] **T1.** Câbler `_calibrate_sell_slip.py` dans `mega-sweep-48h.yml` post-sweep.
- [ ] **T2.** Re-run sell slip drift quand N≥200 twin pairs (~Mai 03-05).
- [ ] **T3.** Re-run `_eth_round_trip_smoke.py --execute` mensuellement OU si base_fee ETH > 5 gwei.
- [ ] **T4.** Câbler `eth_daily_loss_limit_usd` dans le ETH dispatch.

### Live SOL config cleanup (post v14e.36)
- [ ] **T5.** **`BOND_FAST_TP50_SL20_T20` dans live SOL** — cette strat est dans `_AUTO_DEPRECATED` v14e.36 (artefact family BOND_*). À retirer de `rt_trade_config.live_trading.allocations`. Confirmer qu'elle ne fire plus de live trades.
- [ ] **T6.** Vérifier que `ETH_FAST_TP500_SL40_60M` (perdant -$432/48h, WR 12%) est paper-only et n'est pas en live.

### Auto-recovery wiring
- [ ] **T7.** Wire `_kol_recovery_check.py` + `_kol_reliable_audit.py --exclude-artifact-strats` en cron GH (lundi 06:00 UTC). Sortie : artefacts CSV + alerte Telegram si nouveaux candidates.
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

### Strats deck (484 dont 119 artefact-deprecated)
| Family | SOL | ETH |
|--------|-----|-----|
| BE | 33 | 9 |
| LOCK | 33 | 20 |
| FAST | 73 | 18 |
| SLOW | 13 | 5 |
| SCALP | 36 | 17 |
| Other | 80 | 10 |  *# was 199, -119 artefacts retirés*
| AGE clones | 38 | 18 |
