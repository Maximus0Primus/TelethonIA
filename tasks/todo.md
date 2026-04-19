# Pipeline Status — Updated Apr 19, 2026 (v144 deployed)

## Current state

**Live (50/50)** — `BE25_TP80_SL30` (median_5/240s) + `BE15_TP100_SL50` (ds/30s). Position ~$1.70/trade, max 3 open. Configs live identiques au paper post-revert (A/B base préservé).

**Paper Telegram — 21 strats active × $1000 bankroll ($21K seed post-v142) + 9 shadows v142**. Orchestration per-strat via `rt_trade_config.strategy_overrides` + `LAZY_STRATEGIES`.

### 21 Mains actives — stats 7d (27,223 trades total)

| Strat | Orch | N 7d | WR | avg% | **$ 7d** | bankroll |
|---|---|---|---|---|---|---|
| **FAST_TP80_SL25** ⭐ | ds/30s + LAZY | 54 | 37% | +5.83% | **+$157** | $1029 |
| **FAST_TP40_SL30** ⭐ | hysteresis/30s + LAZY | 11 | 27% | +27.52% | **+$151** | $1158 |
| **BE25_TP80_SL30** | median_5/240s | 77 | 32% | +2.10% | **+$87** | $979 |
| **FAST_TP100_SL20** | ds/30s + LAZY | 65 | 31% | +1.61% | **+$52** | $1024 |
| **FAST_TP50_SL30** | median_3/30s + LAZY | 123 | 41% | +1.85% | **+$47** | $1100 |
| **TP50_SL15** | jupiter/30s | 11 | 45% | +4.74% | **+$26** | $1027 |
| BE25_TP80_SL30_DS | ds/30s + LAZY | — | — | — | — | $1031 |
| BE15_TP70_SL50_NZ (NOZEROLIQ) | jupiter/240s | — | — | — | — | $975 |
| BE15_TP300_SL50_MCAP (MCAP_MID) | ds/30s | — | — | — | — | $970 |
| BE15_TP100_SL50 | ds/30s | — | — | — | — | $904 |
| FAST_TP100_SL20_HYST | hysteresis/30s + LAZY | 8 | 12% | −18.25% | **−$73** | $927 |
| FAST_TP80_SL25_HYST | hysteresis/30s + LAZY | 8 | 12% | −19.67% | **−$79** | $921 |
| FAST_TP50_SL30_HYST | hysteresis/30s + LAZY | 8 | 25% | −15.81% | **−$63** | $936 |
| BE25_TP80_SL30_HYST | hysteresis/30s + LAZY | 8 | 12% | −17.66% | **−$71** | $929 |
| BE25_TP80_SL30_S30_HYST (SCORE30) | hysteresis/240s | — | — | — | — | $989 |
| BE25_TP80_SL30_NZS30_HYST (NZ+S30) | hysteresis/240s | — | — | — | — | $987 |
| HIGHSCORE_TP200_SL40 (SCORE30) | jupiter/120s | — | — | — | — | $956 |
| NOZEROLIQ_TP200_SL40 (liq>0) | jupiter/120s | — | — | — | — | $906 |
| FAST_TP70_SL50 🆕 (Apr 18) | winsor_p95/30s + LAZY | 0 | — | — | — | $1000 |
| BE15_TP200_SL40_4H 🆕 | hysteresis/60s | 0 | — | — | — | $1000 |
| MCAP_MID_DTRAIL5_ACT25_SL50_2H 🆕 | median_5/120s | 0 | — | — | — | $1000 |

**Totaux 7d** :
- **6 bases non-HYST** : N=341, **+$520 / 7j** = **+$74/jour** ✅
- **4 HYST variants** : N=32, **−$286 / 7j** = **−$41/jour** (N=8 chacun, activées Apr 17)
- **Net book** : ~+$234 / 7j sur bases + HYST combinés

**Global rt_bankroll** : $17,750 current / $18,722 peak.

### Paired A/B comparison HYST vs base (même token/date, 7d)

| Paire | N | mean Δ (HYST−base) | median Δ |
|---|---|---|---|
| FAST_TP100_SL20 | 8 | −0.47% | −0.64% |
| **FAST_TP80_SL25** | 8 | **−6.61%** | −2.75% |
| FAST_TP50_SL30 | 8 | −1.28% | −0.45% |
| BE25_TP80_SL30 | 8 | −2.07% | −1.11% |

**4/4 paires : HYST perd**. Direction consistante, mais N=8/pair. Need N≥30 pour verdict définitif (ETA Apr 22-23).

### Live vs Paper same-strat (depuis 2026-04-17 13:50, excl. pump.fun outliers)

| Strat | Matched | Avg diff | Median | Max | within_10pp | Same status | Entry div | Exit div |
|---|---|---|---|---|---|---|---|---|
| **BE25_TP80_SL30** | 15 | +2.90pp | +1.54pp | 23pp | **12/15 (80%)** | **100%** | 2.9% | +1.8% |
| **BE15_TP100_SL50** | 12 | +16.17pp | −1.16pp | 215pp | 6/12 (50%) | 10/12 | 2.6% | **−10.4%** |

BE25 bien aligné, BE15 a 1-2 outliers qui écrasent la moyenne.

### 9 Shadows v142 (bankroll $0)
TD2_BE5_TP120_SL44_T25, PTRAIL_V2_T10-18-30-45_SL30_T60, BOND_FAST_TP50_SL20_T20, SCORE40_FAST_TP50_SL30_30M, FAST_TP200_SL40_60M, DIP30_B10_T10_A20_SL60_120m, BE15_TP150_SL40_2H, FAST_TP500_SL40_60M.

---

## 📋 Reste à faire

### ⏳ Data wait
- **3 mains v142** (FAST_TP70, BE15_TP200_4H, MCAP_MID_DTRAIL5) — N≥15 ~ Apr 20-21
- **9 shadows v142** — N≥20 ~ Apr 21-22
- **HYST verdict** (paired N≥30) — Apr 22-23
- **v144 LAZY A/B verdict** — 4 shadows `*_NOLAZY` (FAST_TP40/TP80/TP50/TP50_SL15) vs leurs mains LAZY. Run `scripts/compare_lazy_vs_nolazy.py`. N≥50 paires ETA Apr 22-23. Seul test clean possible (paired mêmes tokens, mêmes prix de fill, delta = polling seul).
- **v144 SCORE filter isolation** — shadows `BE25_TP80_SL30_S30` + `BE25_TP80_SL30_S40` (raw smoothing, pas de HYST). S5 audit retroactive : SCORE≥40 = N=13, WR 62%, avg +34%. Confirme que SCORE≥40 > SCORE≥30 (bande 30-40 perd −5.86% pop-wide). Si S40 > S40_HYST sur N≥30, promouvoir en main. ETA Apr 28-30.
- **v144 slip offset validation** — 48h, re-run `scripts/diverge_report.py`, L−S median doit rester ≤ 2pp sur strats actives — Apr 21
- **Non-pump N≥30 pour décider split pump vs global offset** — ETA Apr 25

### 🟠 Décision live scale-up — **Plan 2 étapes**

**BE15_TP100_SL50 : à dégager complètement (live + paper).**
- Origine : A/B entre 2 profils BE — BE25 (active +25%, TP +80%, SL −30%) vs BE15 (active +15%, TP +100%, SL −50%)
- Verdict triple-confirmé :
  - Paper 7d : BE25 +$183/N=69/avg +5.31% vs BE15 +$52/N=119/avg +2.65% → BE25 gagne 3.5×
  - Sim mega sweep (400 configs chacun) : BE25 median avg +9.03% / WR 41.7% / $50/jour vs BE15 +6.01% / 27.1% / $32/jour
  - Live en cours : BE15 sous-performe (bankroll $904 vs seed $1000)
- **L'A/B est terminé, BE15 perd sur tous les angles.**

**Étape 1 — Apr 20-21** (action concrète à faire) :
1. `rt_trade_config.live_trading.allocations` → `{BE25_TP80_SL30: 0.5, FAST_TP50_SL30: 0.5}` (remplace BE15)
2. `rt_trade_config.hybrid_strategy.allocations` → retirer `BE15_TP100_SL50` de la liste (18 → 17 strats actives en paper)
3. Code : `BE15_TP100_SL50` reste dans `STRATEGIES` dict pour permettre aux trades ouverts de clôturer proprement, mais disparaît des auto-opens
4. Bankroll BE15 ($904) réallouable
5. VPS pick up config au prochain cycle, pas de restart

FAST_TP50_SL30 justifié : N=126 paper (max data), avg +3.87%, WR 41%, SL 30% raisonnable real money, LAZY déjà configuré.

**Étape 2 — après 3-5 jours + N≥30 live FAST_TP50** :
Si FAST matche ses stats paper, remplacer BE25 par une 2e FAST avec TP **différent** (FAST_TP80_SL25 ou FAST_TP100_SL20) pour décorréler par profil d'exit. Sinon garder BE25 comme hedge. Pas 2×FAST similaires = concentration de risque.

### 🔴 Open bugs (need data)
- **S5 filters audit (v144)** : NOZEROLIQ retro sur BE25 = +$217 vs base $+187 → **NZ aide**. SCORE30 capte bien >=40 (mean +11%) mais inclut 30-40 (mean -5.86%) → **SCORE40 > SCORE30** recommandé. HYST reste le vrai tueur. TP200 N=9-11 trop faible pour conclure.
- **Paper↔live outliers automatisé (v144)** : `nightly-outlier-monitor.yml` tourne à 04:30 UTC, fail + alert si outlier sync=True apparaît. Test local OK (10 historiques sync=False, 0 sync=True).
- **LAZY polling audit (v144)** — real 7d data : LAZY total +$1387, non-LAZY −$1720 (different strat pools, confounded). Top 4 earners tous LAZY (FAST_TP40 +$275, FAST_TP80 +$269, FAST_TP50 +$242, TP50_SL15 +$216). Sim bench contredit (disait FAST_30 > LAZY sur FAST_TP50) → sim a probablement biais "sur-check" (voit tous ticks 15-30s, triggers SL aberrants). **v144 action** : 4 shadows `*_NOLAZY` ajoutés (strategies.py) — FAST_TP40/TP80/TP50 + TP50_SL15. Paired comparaison via `scripts/compare_lazy_vs_nolazy.py` post-deploy. N≥50 ETA Apr 22. LAZY_XSLOW jamais best en sim → pas déployé.

### 🟡 Améliorations alignment identifiées (low-priority)
- **Tick logging 15-30s → 5-10s** : gain DTRAIL outliers maxabs −36pp → <5pp, coût Jupiter RPC 2-3x. À envisager SI outliers sync=True émergent.

### 🔒 Bloqué sur scale-up live
- **Jupiter Trigger V2** — 0 fills historiques. Débloquer quand live_pos > $10.

### 🧠 Gotcha
Supabase PostgREST cap 1000 rows même avec `.limit(10000)`. Toujours paginer via `.range(off, off+999)`. Pattern : `sim.py::sb_get`.

---

## Sim ↔ Live/Paper coherence (v144 aligned)

### Status post-v144 (Apr 19, N=56 paires live/paper, DTRAIL exclu)
- **sim ↔ live per-pair median** : L−S ≤ 2.5pp sur toutes strats actives (FAST_TP50 +0.35, BE25 −1.54, BE15 −2.45)
- **sim ↔ paper Spearman rank corr** : ρ = +0.905 (N=139 strats) → **sim prédit bien le classement paper**
- **paper ↔ live median** : ≤ 2pp per strat (FAST_TP50 +0.93, BE25 +1.00, BE15 −1.73) — gap fermé au median par v142E (entry sync) + v143.5 (exit sync)
- **Outliers restants** : 23 historiques |L−P|>10pp, 100% avec sync=False (= pré-v142E ou live swap failed). Post-sync devrait être zéro.

### Méthodologie mesure divergence sim
Trois canaux complémentaires :
1. **`paper_trades.paper_sim_pnl_pct`** (colonne v143.6) — PnL sim joint par live_trader sur ticks réels pour chaque trade live. Source directe per-trade.
2. **`scripts/verify_sim_live_alignment.py`** — replay via `_decision_price` + `_evaluate_trade_exit`. CI nightly.
3. **Mega-sweep ranking vs paper/live** (`sim.py --mega-sweep` + `scripts/ranking_compare.py`) — **toujours faire ça en complément** : compare Spearman rank sur stratégies pour vérifier que la sim classe correctement même avec biais de niveau. Sans ce check, on risque de ne détecter qu'un biais absolu et manquer un problème de structure.

### Split slip : pump vs global offset
Test (`slip_split_test.py`) : pump/liq/mcap donnent pooled std ~2920 bps, aucun ne réduit meaningfully → **offset global −100 bps** dans `_dynamic_sell_slip_factor` (v144). Monitor non-pump jusqu'à N≥30 (Apr 25) avant de décider split définitif.

### Tools
- `--from-eval-history` (v138) = 0% bias mathématique
- `--from-trades` = ground truth historique
- `--mega-sweep` = grid complet → ranking correlation sim vs paper/live
- `scripts/verify_sim_live_alignment.py` = audit sim vs live (CI nightly)
- `scripts/diverge_report.py` = tableau récap sim/paper/live unifié
- `scripts/calibrate_slip.py` = calibration per-pair delta
- `scripts/slip_split_test.py` = test splitter pump/liq/mcap
- `scripts/ranking_compare.py` = Spearman sim↔paper↔live
- `scripts/outlier_diag.py` = root-cause par outlier |L−P|>10pp

**Thresholds CI** : avg |diff| ≤ 5pp ET within_10pp ≥ 80% sinon fail + Telegram alert.

---

## Architecture summary

**Scoring :** 40.5/13.5/40.5/5.4 (consensus/conviction/breadth/PA), 16-multiplier chain.
**Trading :** Paper slip dynamic, live Jupiter Ultra RFQ ~10bps, position reconciliation sibling-aware (v133-D), loss limit 0.5 SOL/jour.
**Alerting :** ML disabled, RT listener uncapped, GH Actions failures, daily summary 8am UTC.

## Workflow sim

| Mode | Flag | Use case |
|---|---|---|
| Grid focused | `--from-ticks` | Ranking rapide par strategy |
| Ground truth | `--from-trades` | Vérité terrain historique |
| 0% bias | `--from-eval-history` | Perfect replay post-v138 |
| Mega sweep | `--mega-sweep` | Full matrix 134K configs |

```bash
python scraper/sim.py --from-ticks --since 2026-04-13 --top 30
python scraper/sim.py --from-trades --since 2026-04-13
python scraper/sim.py --from-eval-history --since 2026-04-17
python scraper/sim.py --mega-sweep  # flags: --mega-workers N, --mega-csv-out, --mega-since
```

## Historique récent

- **v144** (Apr 19, soir) ✅ 4 chantiers safe : (1) shadow-sync entry étendu au path exploration dans `safe_scraper.py` — plus aucun sync=False sur nouveaux trades même si hybrid OFF. (2) `dex_ticks` câblé dans les 4 autres callsites `_replay_with_intervals` (sim.py:3074/3315/4016/4023) — dual-stream smoothing utilisable partout. (3) S5 filter audit (voir §Open bugs). (4) `nightly-outlier-monitor.yml` déployé — CI nightly alert Telegram sur outlier sync=True.
- **v144** (Apr 19) ✅ `_dynamic_sell_slip_factor` : offset global −100 bps (shift per-pair delta mean +115 → 0). Split pump/non-pump testé (N=6 non-pump trop petit, pas de gain std) → report Apr 25.
- **v143.6** (Apr 19) ✅ DS cache TTL + `paper_sim_pnl_pct` column + CI nightly gate
- **v143.5** (Apr 19) ✅ Live exit shadow-sync : force-close paper match au fill Jupiter
- **v143.1-4** (Apr 18-19) ✅ Sim alignment fixes (`_decision_price`, `high_price_seen` reset, 7 smoothing modes ports)
- **v142 E** (Apr 18) ✅ Entry shadow-sync : paper reuse live `execution_price` via `_rt_force_entry_price`
- **v142 A-D** (Apr 18) ✅ Mega sweep 134K configs → 3 new mains + 9 shadows + 3 smoothing modes + OHLC burst port
- **v141** (Apr 17) ✅ rt_score +3 bonuses data-driven (corr +0.207 → +0.236)
- **v140** (Apr 17) ✅ 8 new strats, `_BE_RE` regex relaxé, bankroll reset $18K
- **v138.5** (Apr 17) ✅ Slip recalibration (sl_hit 435bps, trail 250bps, tp +300bps)
- **v138** (Apr 17) ✅ `eval_history` JSONB + `cache_snapshots` table + `--from-eval-history`
