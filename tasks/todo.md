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

## 📋 Reste à faire

### ⏳ Data wait (tout ce qui reste)
- **3 mains v142** (FAST_TP70, BE15_TP200_4H, MCAP_MID_DTRAIL5) — N≥15 Apr 19-20
- **9 shadows v142** (TD2, PTRAIL_V2, BOND_FAST, SCORE40, FAST_TP200_60M, DIP30_B10, BE15_TP150_2H, FAST_TP500_60M) — N≥20 Apr 20-21
- **HYST verdict** (paired N≥30) — Apr 22-23
- **Shadow-sync v142 E validation** — 48h, query `entry_source='live_sync'` → gap <0.5%
- **Slip calibration v143** par liq_bucket — N≥15 live/bucket — Apr 22-24

### 🔴 Open bugs (need data)
- **P3** : slip model sur-pénalise liq>$50K, sous-pénalise bondings. N≥30 par bucket.
- **S5** : NOZEROLIQ/HIGHSCORE filtres continuent de perdre. N≥50 par bucket.

### 🔒 Bloqué sur scale-up live
- **Jupiter Trigger V2** — 0 fills historiques. Débloquer quand live_pos > $10.

### 🧠 Gotcha
Supabase PostgREST cap 1000 rows même avec `.limit(10000)`. Toujours paginer via `.range(off, off+999)`. Pattern : `sim.py::sb_get`.

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
