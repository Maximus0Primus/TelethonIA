# Pipeline Status — Updated Apr 19, 2026 (v144 deployed)

## Current state

**Live (50/50)** — `BE25_TP80_SL30` + `FAST_TP50_SL30`. Position 0.02 SOL (~$3.40)/trade. **max_open_positions: 6** (bumped from 3 pour plus de data). Exposition max 0.12 SOL ≈ $20. Daily loss limit 0.5 SOL (~$85).

**Paper (hybrid) — 16 mains actives** + **12 shadows** (9 v142 + 3 v144 SCORE filter + 4 v144 NOLAZY + 3 v144 SCORE40 FAST).

### 16 Mains actives — stats 7d

| Strat | N 7d | WR | avg% | **$ 7d** | **$/jour** |
|---|---|---|---|---|---|
| **FAST_TP40_SL30** ⭐ | 126 | 41.3% | +3.73% | **+$276** | **+$39** |
| **FAST_TP80_SL25** ⭐ | 78 | 37.2% | +6.28% | **+$270** | **+$39** |
| **FAST_TP50_SL30** ⭐ (live) | 126 | 41.3% | +3.87% | **+$242** | **+$35** |
| **TP50_SL15** | 126 | 35.7% | +6.72% | **+$217** | **+$31** |
| **FAST_TP100_SL20** | 78 | 34.6% | +4.72% | **+$184** | **+$26** |
| **BE25_TP80_SL30** (live) | 69 | 33.3% | +5.31% | **+$183** | **+$26** |
| **BE25_TP80_SL30_S30_HYST** | 12 | 58.3% | +25.84% | **+$155** | **+$22** 🎯 |
| FAST_TP50_SL30_HYST | 21 | 42.9% | +7.40% | +$78 | +$11 |
| FAST_TP100_SL20_HYST | 21 | 33.3% | +6.53% | +$69 | +$10 |
| BE25_TP80_SL30_HYST | 21 | 38.1% | +5.14% | +$54 | +$8 |
| FAST_TP80_SL25_HYST | 21 | 38.1% | +4.46% | +$47 | +$7 |
| BE25_TP80_SL30_DS | 37 | 37.8% | +2.02% | +$37 | +$5 |
| BE15_TP70_SL50_NZ | 9 | 33.3% | +6.81% | +$31 | +$4 |
| BE25_TP80_SL30_NZS30_HYST | 7 | 28.6% | +7.13% | +$25 | +$4 |
| HIGHSCORE_TP200_SL40 | 11 | 27.3% | +2.27% | +$12 | +$2 |
| NOZEROLIQ_TP200_SL40 | 9 | 11.1% | −27.34% | **−$123** | **−$18** 🔴 |

**TOTAL paper 7d : +$1756 / 7j = +$251/jour** (positions $50/trade)

### Live 7d (N=48 FAST + N=25 BE25)
BE25 avg +12.3% / WR 44% / +$5.28 → **+$0.75/jour**
FAST_TP50 avg +5.21% / WR 45.8% / +$4.31 → **+$0.62/jour**
**Total projeté post-swap : +$1.37/jour** (avant bump max_open_positions). Avec 6 positions vs 3, potentiel ~2× si opportunités de même qualité.

Note : live avg% > paper avg% (ex. BE25 live +12.3% vs paper +5.31%) — probablement effet v142E/v143.5 shadow-sync (live réutilise fill Jupiter réel).

---

## 📋 Reste à faire

### ⏳ Data wait (rien à faire, laisser tourner)
- **v144 LAZY A/B verdict** (Apr 22-23) — shadows `*_NOLAZY` × 4 vs mains LAZY. Run `scripts/compare_lazy_vs_nolazy.py` à N≥50.
- **v144 SCORE40 family verdict** (Apr 28-30) — 5 shadows (BE25_S30/S40, FAST_TP50/TP80/TP100 _S40). Run `scripts/refresh_main_stats.py` à N≥30. Promouvoir le meilleur en main si stats confirmées.
- **HYST paired verdict** (Apr 22-23) — N≥30 pour confirmer le −2 à −6pp penalty.
- **3 mains v142** (FAST_TP70, BE15_TP200_4H, MCAP_MID_DTRAIL5) — N≥15 ~ Apr 20-21.
- **9 shadows v142** (TD2, PTRAIL_V2, BOND_FAST, etc.) — N≥20 ~ Apr 21-22.
- **Slip offset validation v144** — Apr 21, re-run `scripts/diverge_report.py` pour vérifier L−S median ≤ 2pp persiste.
- **Non-pump N≥30 pour split slip final** — ETA Apr 25.

### 🟠 Scale-up live — Étape 2 (après Étape 1 validée)

**Étape 1 : FAIT** ✅ (Apr 19) — BE15 retiré live + paper, FAST_TP50_SL30 ajouté live, max_open_positions 3→6.

**Étape 2 — après 3-5 jours + N≥30 live FAST_TP50** :
- Si FAST_TP50 matche ses stats paper : remplacer BE25 par une 2e FAST à TP différent (FAST_TP80_SL25 ou FAST_TP100_SL20) pour diversifier par profil d'exit, OU ajouter une 3e strat (33/33/33).
- Si un SCORE40 shadow a validé son alpha (N≥30) : promouvoir en main paper + envisager live.

### 🔒 Bloqué sur scale-up
- **Jupiter Trigger V2** — 0 fills historiques. Débloquer quand live_pos > $10.

### 🧠 Gotcha
Supabase PostgREST cap 1000 rows même avec `.limit(10000)`. Toujours paginer via `.range(off, off+999)`. Pattern : `sim.py::sb_get`.

---

## Sim ↔ Live/Paper coherence (v144 aligned)

### Status post-v144
- **sim ↔ live per-pair median** : L−S ≤ 2.5pp sur strats actives (FAST_TP50 +0.35, BE25 −1.54)
- **sim ↔ paper Spearman rank corr** : ρ = +0.905 (N=139 strats) → sim prédit bien le classement
- **paper ↔ live median** : ≤ 2pp per strat (gap fermé par v142E + v143.5 shadow-sync)
- **Outliers restants** : 23 historiques |L−P|>10pp, 100% avec sync=False (pré-v142E). Nouveaux outliers sync=True = vrai bug → monitor nightly actif.

### Méthodologie mesure divergence sim (3 canaux)
1. **`paper_trades.paper_sim_pnl_pct`** (v143.6) — PnL sim joint par trade live
2. **`scripts/verify_sim_live_alignment.py`** — CI nightly 04:00 UTC
3. **Mega-sweep ranking vs paper/live** (`ranking_compare.py`) — **toujours faire en complément** : Spearman rank corr pour validation structurelle en plus du biais absolu

### Slip calibration v144
`_dynamic_sell_slip_factor` : offset global −100 bps (pump/liq/mcap splits testés, pooled std identique ~2920 bps → pas de gain du split). Revisit Apr 25 avec N non-pump ≥ 30.

### Monitoring CI actif
- `sim-align-gate.yml` (04:00 UTC) — alert si sim-live drift > 5pp
- `nightly-outlier-monitor.yml` (04:30 UTC) — alert si outlier paper↔live sync=True

### Scripts utilitaires (`scripts/`)
- `recap_daily.py` — top earners + $/jour paper & live
- `refresh_main_stats.py` — ranking actuel par $ 7d
- `compare_lazy_vs_nolazy.py` — paired LAZY A/B verdict (à run Apr 22-23)
- `diverge_report.py` — tableau unifié sim/paper/live
- `calibrate_slip.py` — slip model calibration per-pair delta
- `ranking_compare.py` — Spearman sim↔paper↔live
- `outlier_diag.py`, `nightly_outlier_monitor.py` — outlier root-cause
- `analyze_s5_filters.py` — NZ/SCORE/MCAP filter analysis
- `verify_sim_live_alignment.py` — CI sim vs live audit

---

## Architecture summary

**Scoring :** 40.5/13.5/40.5/5.4 (consensus/conviction/breadth/PA), 16-multiplier chain. rt_score v141 data-driven bonuses (+3 features).
**Trading :** Paper slip dynamic (v144 offset −100bps), live Jupiter Ultra RFQ, position reconciliation sibling-aware (v133-D), loss limit 0.5 SOL/jour.
**Alerting :** ML disabled (v109, anti-predictive), RT listener uncapped, GH Actions failures, daily summary 8am UTC, sim-align + outlier nightly alerts.

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
python scraper/sim.py --mega-sweep
```

## Historique récent

- **v144** (Apr 19 — complet) ✅
  - Slip `_dynamic_sell_slip_factor` offset global −100 bps (calibration per-pair L−P delta)
  - Shadow-sync entry étendu au path exploration dans `safe_scraper.py` (plus de sync=False si hybrid OFF)
  - `dex_ticks` câblé dans 4 callsites `_replay_with_intervals` (sim.py complet)
  - 4 shadows `*_NOLAZY` pour paired A/B LAZY (FAST_TP40/TP80/TP50/TP50_SL15)
  - 5 shadows SCORE filter isolation (BE25_S30/S40, FAST_TP50/TP80/TP100_S40)
  - `nightly-outlier-monitor.yml` CI — alert Telegram sur outlier sync=True
  - DB swap live : BE15→FAST_TP50_SL30. Paper hybrid : BE15_TP100_SL50 + BE15_TP300_SL50_MCAP retirés
  - max_open_positions 3→6
- **v143.6** (Apr 19) ✅ DS cache TTL + `paper_sim_pnl_pct` column + CI sim-align-gate
- **v143.5** (Apr 19) ✅ Live exit shadow-sync : force-close paper match au fill Jupiter
- **v143.1-4** (Apr 18-19) ✅ Sim alignment fixes + 7 smoothing modes ports
- **v142 E** (Apr 18) ✅ Entry shadow-sync : paper reuse live `execution_price`
- **v142 A-D** (Apr 18) ✅ Mega sweep 134K configs → 3 new mains + 9 shadows + smoothing ports
- **v141** (Apr 17) ✅ rt_score +3 bonuses data-driven (corr +0.207 → +0.236)
- **v140** (Apr 17) ✅ 8 new strats, bankroll reset $18K
- **v138.5** (Apr 17) ✅ Slip recalibration (sl_hit 435bps, trail 250bps, tp +300bps)
- **v138** (Apr 17) ✅ `eval_history` JSONB + `--from-eval-history` 0% bias mode
