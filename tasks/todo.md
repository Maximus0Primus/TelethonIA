# Pipeline Status — Updated Apr 18, 2026 (v142, post-revert)

## Data windows (à lire avant toute lecture de stats 7d)

- **Bases FAST/BE non-HYST** (ds/median_3/median_5/jupiter smoothing) : **~7j complets** depuis v132, N=11-123 par strat. Solide.
- **_HYST variants** (hysteresis/30s) : activées **v140 Apr 17 ~16h UTC**, seulement **~20-24h** de data → N=8 par strat, fragile.
- **3 nouvelles mains v142** (FAST_TP70_SL50, BE15_TP200_SL40_4H, MCAP_MID_DTRAIL5_ACT25_SL50_2H) : activées **Apr 18 12:07 UTC** aujourd'hui, quelques heures de data, N=0-5. Non évaluable.
- **9 shadows v142** (TD2, PTRAIL_V2, BOND_FAST, SCORE40, etc.) : activées **Apr 18 08:39-12:01 UTC**, ~4-8h de data. Non évaluable.
- **Orchestration align des mains existantes** : appliquée 12:16 UTC, **reverted à 12:30 UTC** (14 min de config erroné, impact négligeable).

**ETA verdicts** :
- HYST vs base A/B : N≥30 paired ~ Apr 22-23
- Nouvelles strats v142 : N≥15 ~ Apr 19-20

## Current state

**Live (50/50)** — `BE25_TP80_SL30` (median_5/240s) + `BE15_TP100_SL50` (ds/30s). Position ~$1.70/trade, max 3 open. Configs live identiques au paper post-revert (A/B base preservé).

**Paper Telegram — 21 strats active × $1000 bankroll ($21K seed post-v142) + 9 shadows v142**.
Orchestration per-strat : **10 mains existantes en configs pré-v142 (A/B base vs _HYST préservé)** + **11 nouvelles strats v142 en configs sim-optimal** (pas de config précédent à préserver). Voir `rt_trade_config.strategy_overrides` + `LAZY_STRATEGIES`.

### 21 Mains actives — stats 7d réelles paginées (27,223 trades total 7d)

Configs post-revert (A/B test structure intacte base vs _HYST). Stats `is_shadow=False` only.

| Strat | Orch (config actuel) | N 7d | WR | avg% | **$ 7d** | bankroll |
|---|---|---|---|---|---|---|
| **FAST_TP80_SL25** ⭐ | ds/30s + LAZY | 54 | 37% | +5.83% | **+$157** | $1029 |
| **FAST_TP40_SL30** ⭐ | hysteresis/30s + LAZY | 11 | 27% | +27.52% | **+$151** | $1158 |
| **BE25_TP80_SL30** | median_5/240s | 77 | 32% | +2.10% | **+$87** | $979 |
| **FAST_TP100_SL20** | ds/30s + LAZY | 65 | 31% | +1.61% | **+$52** | $1024 |
| **FAST_TP50_SL30** | median_3/30s + LAZY | 123 | 41% | +1.85% | **+$47** | $1100 |
| **TP50_SL15** | jupiter/30s | 11 | 45% | +4.74% | **+$26** | $1027 |
| BE25_TP80_SL30_DS | ds/30s + LAZY | — | — | — | — | $1031 |
| BE15_TP70_SL50_NZ (filter NOZEROLIQ) | jupiter/240s | — | — | — | — | $975 |
| BE15_TP300_SL50_MCAP (filter MCAP_MID) | ds/30s | — | — | — | — | $970 |
| BE15_TP100_SL50 | ds/30s | — | — | — | — | $904 |
| FAST_TP100_SL20_HYST | hysteresis/30s + LAZY | 8 | 12% | −18.25% | **−$73** | $927 |
| FAST_TP80_SL25_HYST | hysteresis/30s + LAZY | 8 | 12% | −19.67% | **−$79** | $921 |
| FAST_TP50_SL30_HYST | hysteresis/30s + LAZY | 8 | 25% | −15.81% | **−$63** | $936 |
| BE25_TP80_SL30_HYST | hysteresis/30s + LAZY | 8 | 12% | −17.66% | **−$71** | $929 |
| BE25_TP80_SL30_S30_HYST (filter SCORE30) | hysteresis/240s | — | — | — | — | $989 |
| BE25_TP80_SL30_NZS30_HYST (filter NZ+S30) | hysteresis/240s | — | — | — | — | $987 |
| HIGHSCORE_TP200_SL40 (filter SCORE30) | jupiter/120s | — | — | — | — | $956 |
| NOZEROLIQ_TP200_SL40 (filter liq>0) | jupiter/120s | — | — | — | — | $906 |
| FAST_TP70_SL50 🆕 (Apr 18 12:07) | winsor_p95/30s + LAZY | 0 | — | — | — | $1000 |
| BE15_TP200_SL40_4H 🆕 | hysteresis/60s | 0 | — | — | — | $1000 |
| MCAP_MID_DTRAIL5_ACT25_SL50_2H 🆕 | median_5/120s | 0 | — | — | — | $1000 |

**Totaux stratégiques 7d** :
- **6 bases non-HYST** : N=341, **+$520 / 7j** = **+$74/jour** ✅
- **4 HYST variants** : N=32, **−$286 / 7j** = **−$41/jour** (mais N=8 chacun, activées v140 Apr 17 ~20-24h seulement)
- **Net book** : ~+$234 / 7j sur bases + HYST combinés, + delta indeterminé des strats avec `—` (filter-gated, trop peu de données)

**Bankrolls cumulés winners** : +$158 FAST_TP40, +$100 FAST_TP50, +$31 BE25_DS, +$29 FAST_TP80, +$27 TP50_SL15, +$25 FAST_TP100
**Bankrolls cumulés losers** : −$95 BE15_TP100_SL50, −$94 NOZEROLIQ, −$79 FAST_TP80_HYST, −$73 FAST_TP100_HYST, −$71 BE25_HYST, −$63 FAST_TP50_HYST

**Global rt_bankroll** : $17,750 current / $18,722 peak / −$249 all-time sur 147 trades (cumul non-relié aux stats per-strat ci-dessus — ces dernières filtrent sur 7d).

### Paired A/B comparison HYST vs base (mêmes token/date, 7d)

| Paire | N pairs | mean Δ (HYST−base) | median Δ |
|---|---|---|---|
| FAST_TP100_SL20 | 8 | −0.47% | −0.64% |
| **FAST_TP80_SL25** | 8 | **−6.61%** | −2.75% |
| FAST_TP50_SL30 | 8 | −1.28% | −0.45% |
| BE25_TP80_SL30 | 8 | −2.07% | −1.11% |

**4/4 paires : HYST perd en mean ET median.** Direction consistante → signal vraisemblable, mais N=8/pair petit (activées ~20-24h). Need N≥30 paired pour verdict définitif (ETA Apr 22-23).

### 9 Shadows v142 (bankroll $0, observabilité pure)

| Strat | Orch | Source du pick |
|---|---|---|
| TD2_BE5_TP120_SL44_T25 | jupiter/30s | fine sweep TD2 (+$468 haircut, stab 3/4) |
| PTRAIL_V2_T10-18-30-45_SL30_T60 | ds/30s + LAZY | mega sweep ($99/j) |
| BOND_FAST_TP50_SL20_T20 | hysteresis/60s | mega sweep ($88/j) |
| SCORE40_FAST_TP50_SL30_30M | median_3/30s + LAZY | sweep SCORE40 winner (+34.5% avg N=18) |
| FAST_TP200_SL40_60M | hysteresis/60s | MEGA_NEW_STRATS |
| DIP30_B10_T10_A20_SL60_120m | jupiter/30s | DIP variant stricter |
| BE15_TP150_SL40_2H | hysteresis/60s | BE medium-horizon gap |
| FAST_TP500_SL40_60M | winsor_p95/30s + LAZY | moonshot tail-captor |

## v142 changelog (Apr 18)

- **v142 A** (`60ab314`) — paper slip bps persisté + live exit fallback log
- **v142 B** (`a3fbbd7`) — 6 shadows diversity + cleanup 4 HYST redondants
- **v142 main promote** (`1c19d0b`) — 3 mega-sweep winners en main $1K chacun (FAST_TP70, BE15_TP200_4H, MCAP_MID_DTRAIL5)
- **v142 orch align** (`0feba13`) → **REVERT** (`b559453`) — alignment a écrasé l'A/B structure existant, rollback sur 10 mains
- **v142 bankroll fix + cleanup** (`3710416`) — `current_balance` $17,750→$20,754, archive sim_sweep/sim_new_strategies → `_archive/`
- **v142 C — smoothings** (`79a0d7d`) — 3 nouveaux `price_source` : `jp_sampled_60s`, `vwap_5min`, `twin_confirm`
- **v142 D — OHLC burst port** (`6c10d24`) — `ohlc_burst_60s` port littéral de `sim_engines.candles_to_synthetic_ticks()`
- **v142 E — shadow-sync P1+P4** (`34ec4be`) — live opens first in hybrid flow, paper reuse `execution_price` via `_rt_force_entry_price`. `open_live_trade` retourne dict `{success, execution_price}`. Élimine divergence entry_price ±9% + TP/SL status inversion.

### v142 C/D finding important

Legacy OHLCV sim historique donnait **+184% sur DIP_SCALE_OUT** (DIP30_B5_SO50_100_RT10_RA50_SL70 = $500 → $1421). Tick sim sur même famille DIP = **−25%** (DIP30_B5_T5_A20_SL70_240m = $373 sur start $500).

**Gap de 209pp** = les "supers résultats" OHLCV étaient un **artifact d'estimation** :
- OHLCV 15-min bars DS agrégent tous les trades exchange → wicks extrêmes visibles
- Nos polls 30s voient 2-4 snapshots → manquent 95%+ amplitude intra-bar
- Même avec Birdeye OHLCV API (payant), exec latency Jupiter (1-3s) rate les wicks <5s
- Seul Jupiter Trigger V2 (on-chain triggers) peut les catcher — déjà implémenté, 0 fills car live_pos $1.70 < trigger_min $10

**Conclusion** : le sim OHLCV historique était **overfitted aux wicks intrabar** qu'on ne capture pas en live. Les DIP_SCALE_OUT top-ranked là-dedans ne peuvent pas reproduire leurs gains. Inutile de payer Birdeye pour reproduire ce biais. **Focus sur FAST/BE vanilla qui gagnent empiriquement (+$74/j 7d).**

### Mega sweep v142 — top 10 per-strat ($/day at $50 pos, via price_ticks 80 tokens)

| Rank | Strat | $/j | Note |
|---|---|---|---|
| 1 | FAST_TP100_SL20 | $134 | hysteresis + lazy — ACTIVE |
| 2 | FAST_TP80_SL25 | $124 | hysteresis + lazy — ACTIVE |
| 3 | BE25_TP80_SL30 (+5 variants tied) | $122 | hysteresis + lazy — ACTIVE |
| 4 | FAST_TP50_SL30 / FAST_TP40_SL30 | $118 | hysteresis + lazy — ACTIVE |
| 5 | FAST_TP70_SL50 | $100 | winsor_p95 + lazy — ACTIVE (new) |
| 6 | PTRAIL_V2 | $99 | shadow only |
| 7 | TD2 | $95 | shadow only |
| 8 | FAST_TP100_SL50 / FAST_TP50_SL50 | $95 | **NOT ACTIVE** (can add) |
| 9 | BE20_TP50_SL30 / BE15_TP50_SL30 | $89 | **NOT ACTIVE** (BE band gap) |
| 10 | BOND_FAST | $88 | shadow only |

## Findings sim/real alignment

- **Pre-v142 eval_history sim** : biaisé vers slow-losing trades (sample filter `eval_history ≥ 3 polls` = selection bias). Ranking opposé à la réalité paper.
- **v142 mega sweep via price_ticks** : ranking aligné au réel paper 7d. FAST/BE dominent, DTRAIL/DIP perdent (cohérent avec paper réel).
- **Hypothèse haircut slip** : TD2 fine winner perd 12% à haircut (slip-efficient), BOND perd 64% (bondings extra +400bps brittle), PTRAIL perd 45% (trail mid-slip).

## 📋 Ce qui reste à faire — état Apr 18 13h UTC

### 🟢 Rien à faire en code actuellement (attente de data)
- ⏳ **N≥15 sur 3 nouvelles mains v142** (FAST_TP70_SL50, BE15_TP200_SL40_4H, MCAP_MID_DTRAIL5) — ETA Apr 19-20
- ⏳ **N≥20 sur 9 shadows v142** (TD2, PTRAIL_V2, BOND_FAST, SCORE40, FAST_TP200_60M, DIP30_B10, BE15_TP150_2H, FAST_TP500_60M) — ETA Apr 20-21
- ⏳ **N≥30 paired HYST vs base** pour verdict définitif HYST — ETA Apr 22-23
- ⏳ **N≥15 live par liq_bucket** pour recalibrer slip v143 — ETA Apr 22-24

### 🔴 Actions code restantes
1. [x] ~~Bug `rt_bankroll.current_balance` drift~~ — FIX `3710416` (current $20,754)
2. [x] ~~Nettoyage scripts one-shot~~ — FAIT `3710416`
3. [x] ~~S3 exit_price DS-tick fallback~~ — CLOSED (0 occurrences en 5h post-v142 → pas de schema migration nécessaire)
4. [x] ~~P1 + P4 shadow-sync paper ↔ live entry_price~~ — FIX `34ec4be` (live opens first, paper reuse execution_price via `_rt_force_entry_price`)
5. **Debug Jupiter Trigger V2** — bloqué par live_pos $1.70 < trigger_min $10. Nécessite scale-up live d'abord.
6. **Post-N≥15 (Apr 19-20)** : analyser 3 nouvelles mains v142 vs sim
7. **Post-N≥30 (Apr 22-23)** : verdict HYST définitif
8. **Post-48h v142E (Apr 20)** : valider shadow-sync — query paires paper+live avec `entry_source='live_sync'` → `ABS(paper.entry_price - live.execution_price) < 0.5%` attendu (vs ±9% avant). P1 inversion rate attendu ~0%.

### 🔴 Open bugs (à éteindre avant tout action structurelle)
- **P1 — Inversion TP/SL paper vs live** (N=10, ligne 143) : need 20+ paires post-v141 — shadow-sync fix proposed
- **P3 — Slip calibration** (ligne 144) : need N≥30 par liq bucket — modèle actuel surpénalise high-liq, sous-pénalise bondings
- **P4 — Entry_price paper vs live ±9%** (ligne 145) : même fix que P1 (shadow-sync)
- **S3 — exit_price live edge case** (ligne 150) : schema migration deferred, monitoring via `journalctl | grep DS-tick fallback` — check si 0 en 48h → close définitif
- **S5 — Bondings profitable, filters losing** (ligne 152) : confirmer avec N≥50 par bucket (ETA 48h)

### 🟡 Hors scope v142 (à reprendre si prio change)
- Live BE25 + BE15 : >$0.50/trade avg sur N≥10 (bankroll live trop petit à $1.70/trade, scale-up d'abord)
- Latence A.2.1 re-mesure, A.2.2-4 parallélisation (low ROI à volume actuel)
- v137 cadence validation (mineur)
- tp_touched exit mode (deferred, sous threshold à +$1.55/sem)

### 🧠 Gotcha opérationnel récurrent
**Supabase PostgREST cap 1000 rows par requête même avec `.limit(10000)`**. Toujours paginer via `.range(off, off+999)` en loop. Mon query non-paginée ce matin a cru N=5 partout alors que c'était 54-123 — conclusions toutes fausses. Pattern correct : voir `scraper/sim.py::sb_get` ou `scraper/_align_orchestration_to_sim.py`.

---

## 🔴 Priorités immédiates (Apr 18-22) — historique détaillé

- [x] **Validation 48-72h post-v142 orchestration alignment** — **REVERTED** (commit `b559453` Apr 18 12:30 UTC). L'align avait écrasé les configs A/B existantes (base vs _HYST) en tout mettant en hysteresis, destruction de la diversité expérimentale. Les 10 mains existantes ont été restaurées à leur config pré-align. Les 11 nouveaux strats v142 gardent leur config sim-optimale (pas de config précédent à préserver).
- [ ] **3 nouvelles mains v142** : FAST_TP70_SL50 / BE15_TP200_SL40_4H / MCAP_MID_DTRAIL5_ACT25_SL50_2H — attendre N≥15 par strat pour juger vs sim ($100/$86/$81 par jour respectivement)
- [ ] **9 shadows v142** : TD2/PTRAIL_V2/BOND_FAST/SCORE40/FAST_TP200_60M/DIP30_B10/BE15_TP150_2H/FAST_TP500_60M — N≥20 shadow trades pour décider promotion en main
- [ ] **Bug mineur rt_bankroll.current_balance** : pas incrémenté des $3K seed v142 (addition manuelle de strategy_bankrolls, _rt_update_bankroll ne touche qu'au pnl). Reseed si besoin de cohérence global vs per-strat.
- [x] **Paper slippage populate** — FIXED v142 (commit `c6a739d`)
- [x] **Live exit_price fallback log** — FIXED v142 (commit `c6a739d`), monitoring via `journalctl | grep DS-tick`
- [x] **HYST shadows dedup** — FIXED v142 (commit `a3fbbd7`), 4 redundants retirés
- [x] **3 mega-sweep winners promus main** — FIXED v142 (commit `1c19d0b`), $1K each
- [⚠️] **Orchestration alignment 21 strats** — REVERTED (commit `0feba13` → `b559453`). Les 10 mains existantes ont retrouvé leur config pré-align pour préserver l'A/B. Seuls les 11 nouveaux strats v142 gardent leur config sim-optimal.
- [ ] Live BE25 + BE15 : >$0.50/trade avg sur N≥10 trades (hors scope v142 — bankroll live micro)
- [ ] **Latence A.2.1** : `msg→ds` mean 24s → 15-18s sous charge confirmé ? (hors scope v142)
- [x] **v141 — fix mesure `paper_exit_price`** ✅ DÉPLOYÉ Apr 17 19:58 UTC (commit `16a7e8a`). Avant v141, `paper_exit_price` stocké par live_trader = niveau SL brut (slip=1, fee=0), mélangeant 3 effets dans `price_divergence_pct`. v141 calcule un 2ᵉ `ev` dédié mesure avec dynamic slip + SELL_FEE_BPS + Ultra SELL quote override — mirrors paper_trader exactement. Décision live inchangée (toujours slip=1, zéro impact sur PnL/status/sell). **Données pré-v141 biaisées** — filtrer par `created_at >= 2026-04-17T19:58:00Z` avant analyse divergence.
- [ ] **Divergence paper→live par liq bucket** — une fois v141 deployé, attendre N ≥ 15 par bucket avec le vrai `paper_exit_price`. Script : `scraper/analyze_divergence_by_liq.py --days 7`. Décompose (A) polling lag (LIVE MONITOR 3-11min, gap paper→live 60-78s sur $DFV) vs (B) slippage exec Jupiter sur curve illiquide via `corr(delay_sec, |div|)` et `|div|@delay<30s vs >120s`. Hypothèse courante : bonding/liq=$0 → div >20% mais attention ces 3 observations Apr 17 étaient biaisées par le bug de mesure, repartir de zéro. ETA Apr 22-24. **Ne rien modifier côté prod entre-temps**, garder exposition bonding pour collecter signal propre.
- [ ] **Debug Jupiter Trigger V2 — 0 keeper fills historiques** (memory v121). Les triggers `place_stop_loss` placent un `trigger_order_id` mais jamais exécutés par keeper. Investigation en 3 étapes :
  1. **Query `trigger_events` table** depuis v121 : compter `placed` / `cancelled_by_polling` / `keeper_filled` / `failed`. Si 0 filled → problème côté Jupiter (vault auth, keeper routing, token non-routable). Si filled mais polling cancel avant → polling trop agressif, à desync.
  2. **Tester sur trade éligible avec min=$5** — baisser `trigger_min_usd` de $10 → $5 dans `rt_trade_config.live_trading` JSONB côté Supabase (1 UPDATE query). Live pos actuel ~$1.80 donc encore sous le seuil — il faut aussi bumper position_usd à $5+ pour au moins 1 strat pour observer le flow complet. Candidat : `BE25_TP80_SL30` uniquement, pos $5, 1h d'obs.
  3. **Filter non-bonding pour trigger** — ajouter gate `if is_bonding_curve: skip trigger` dans `live_trader.py:836`. Les keepers Jupiter ne routent pas pumpfun bondings → fail certain. Évite le bruit dans trigger_events. Script SQL query + hypothesis : `SELECT COUNT(*), event_type FROM trigger_events WHERE ... GROUP BY event_type`.
  
  **Orthogonal à la question bonding** : Trigger V2 fixerait (A) polling lag sur liq >$25K non-bonding, mais ne résout pas (B) slippage exec ni les bondings (hors graphe Jupiter). À avancer en parallèle du recueil de data liq buckets.

## 🔬 Audit divergences sim/paper/live (Apr 17 16:00-20:00 UTC) — partiellement résolu post-v142

Window : 4h post-deploy 18 strats. N(live closed)=10, N(paper closed)=68, N(shadow)=1091. Snapshot pré-v141 pour la plupart. **Ne pas agir sur ces findings avant N≥30 par strat / N≥15 par liq_bucket**.

**Update v142 (Apr 18)** : le sim biaisé (eval_history filter) a été remplacé par le mega sweep via `price_ticks` (sim.py `_mega_sweep_run`, commit `a3fbbd7`). Le ranking sim est maintenant aligné au réel paper 7d (FAST/BE gagnent, DTRAIL/DIP perdent). Beaucoup de points ci-dessous sont obsolètes ou résolus par v142 :

### 🔴 Problèmes actionnables (confirmés sur N=10 mais N trop faible)

- [x] **P1 — Inversion TP/SL paper vs live** — FIXED v142 E (commit `34ec4be` Apr 18 14:18 UTC). Shadow-sync implémenté : live opens first, paper reuse execution_price. Validation 48h : query paires `entry_source='live_sync'` → inversion rate attendu 0%.
- [ ] **P3 — Slippage live réel vs modèle sim v138.5 : gaps des 2 côtés**. Modèle actuel (`paper_trader.py:1094-1143`, v138.5 recalibré sur 132 trades Apr 13-17) = 435 bps sl_hit, 1000 trail_crash, 250 trail_stop, −300 tp_hit, 120 timeout, avec ×2.0 si liq<$5K et ×1.3 si liq<$20K, caps [−1000, +1500/+2500]. Live mesuré (audit N=10 Apr 17) : bonding sl_hit **+1165 bps** (sim projette 870 → gap ~300 bps sous-estimé), liq >$50K sl_hit **−188 bps** (sim projette 435 → sim sur-pénalise ~620 bps en haute liq !). ~~**Paper ne store aucun `buy/sell_slippage_bps`**~~ **FIXED v142** (Apr 18 08:39 UTC, commit `c6a739d`) : paper_trader.check_paper_trades + cascade + fast persistent désormais `buy_slippage_bps`/`sell_slippage_bps` au close depuis la config courante. **Data need** : N≥30 live par liq_bucket pour re-calibrer v143 (sur-pénalisation en liq >$50K à corriger + renforcer bondings). **Fix possible** : ajuster type_bps par bucket liq fin (pas juste <$5K / <$20K).
- [x] **P4 — Entry_price paper vs live ±9%** — FIXED v142 E (même commit `34ec4be`). Paper utilise désormais `live.execution_price` directement via `_rt_force_entry_price`. Gap attendu: 0 (bit-pour-bit identique). Validation 48h.

### 🟠 Suspects (à retester avec plus de data)

- [ ] **S1 — HYST variants perdent en réel paper** — observation CONFIRMÉE par query paginée 7d (mean/median Δ HYST-base = -0.47% à -6.61%, 4/4 paires dans direction négative). Mais N=8/pair petit car HYST activés seulement ~20-24h. **Verdict définitif ETA Apr 22-23** (N≥30 paired). **Si confirmé** : soit dégager les 4 HYST mains, soit les garder comme control group en paper pour A/B continue. Note : le mega sweep predict HYST +$134/j pour FAST_TP100_SL20 = **sim très optimiste sur HYST**, ne plus s'y fier pour les rankings entre smoothings.
- [x] **S3 — `exit_price` live fallback** — CLOSED Apr 18 14h UTC. `journalctl | grep "DS-tick fallback"` = 0 occurrences en 5h post-v142 deploy → edge case jamais triggered en prod. Warning log sert de canary, schema migration non nécessaire.
- [x] **S4 — Bankroll +$117 réel pas +$722** — **résolu** : section "Current state" réécrite avec chiffres actuels (commit `c55e29b` puis mise à jour v142).
- [ ] **S5 — Bondings gagnent, high-liq perdent (contre-intuitif)**. 7d paper : NOZEROLIQ_TP200_SL40 −$37 sur 2 trades, HIGHSCORE_TP200_SL40 −$15 sur 2 trades, BOND_FAST (shadow) en cours. Les filtres continuent de perdre. **Data need** : attendre 48h v142 orch post-alignment pour N≥20 par filter + comparer `BOND_FAST` shadow vs ces filter.

### ⚪ Non-testables tant qu'il n'y a pas plus de data

- [ ] **v141 data** : audit 12h post-deploy (N=6 live, N=43 paper closed) — `paper_exit_price` divergence mean −0.91%, p95 5.19%, 0 outlier >20%. v141 fonctionne comme prévu. Attendre N≥15 live par bucket pour ratio sim/réel stable (ETA Apr 19-20).
- [x] **Paper slippage** — FIXED v142 (commit `c6a739d`, Apr 18 08:39 UTC). Voir P3.
- [ ] **Trigger V2 fills** : 0 keeper fills historiques, non interrogé dans cet audit (déjà couvert puce debug Trigger V2 ci-dessus). Hors scope tant que live_pos $1.5 < trigger_min_usd $10.

### ✅ v142 — Observabilité slip + exit_price (Apr 18 08:39 UTC, commit `c6a739d`)

- **Paper slip populate** : `check_paper_trades` (main + SL cascade) + `check_paper_trades_fast` écrivent désormais `buy_slippage_bps`/`sell_slippage_bps` dans l'update dict au close. Valeurs tirées de la config courante via `_load_paper_trade_config(client)`. Débloque la comparaison paper slip assumé vs live slip réel.
- **Live exit_price fallback log** : `check_live_trades` track `exit_price_from_fill` et WARNING si sell_output absent. Formule reconstruction fiable documentée inline.

**Validation à J+1** :
1. `SELECT COUNT(*) FROM paper_trades WHERE status!='open' AND exit_at > '2026-04-18T08:39:17Z' AND buy_slippage_bps IS NOT NULL` — attendu ~100% des closes post-deploy.
2. `ssh vps "sudo journalctl -u kol-scraper --since '2026-04-18 08:39:17' | grep 'DS-tick fallback' | wc -l"` — si 0 → close S3 définitivement, sinon planifier schema migration.

### 🧹 Nettoyage

- [ ] Delete `scraper/_audit2.py` et `scraper/_audit3.py` (scripts one-shot du subagent Apr 17). Le script permanent `scraper/analyze_divergence_by_liq.py` couvre le besoin récurrent.
- [ ] Delete `scraper/_v141_24h_audit.py|.json|.log` (one-shot v141 audit Apr 18 matin). Garder `scraper/_activate_v142_mains.py` et `scraper/_align_orchestration_to_sim.py` (réutilisables, idempotents).
- [ ] Archiver `scraper/sim_new_strategies.py` et `scraper/sim_sweep.py` : remplacés par `sim.py --mega-sweep` (résolue le bug sim.py + preserve strategy_name) qui fait tout mieux. Garder les CSVs de résultats (`_mega_sweep_v142.csv`, `sim_sweep_top.json`) pour référence.

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
