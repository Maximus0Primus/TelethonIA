# Pipeline Status — Updated Apr 19, 2026 (v144 complet)

## Current state

**Live (50/50)** — `BE25_TP80_SL30` (median_5/240s) + `FAST_TP50_SL30` (median_3/30s + LAZY). Position 0.02 SOL (~$3.40)/trade. **max_open_positions: 6**. Exposition max 0.12 SOL ≈ $20. Daily loss limit 0.5 SOL (~$85).

**Paper hybrid — 16 mains + 261 shadows** (dont 34 v144 A/B). Alignment audit: **zéro issue**.

### 16 Mains actives — stats 7d

| Strat | Orch | N | WR | avg% | **$ 7d** | **$/jour** |
|---|---|---|---|---|---|---|
| **FAST_TP40_SL30** ⭐ | hysteresis/30s + LAZY | 127 | 41% | +3.20% | **+$248** | **+$35** |
| **FAST_TP80_SL25** ⭐ | ds/30s + LAZY | 81 | 36% | +5.40% | **+$243** | **+$35** |
| **FAST_TP50_SL30** (live) | median_3/30s + LAZY | 125 | 41% | +3.64% | **+$234** | **+$33** |
| **BE25_TP80_SL30** (live) | median_5/240s | 69 | 33% | +5.31% | **+$183** | **+$26** |
| **TP50_SL15** | jupiter/30s + LAZY | 127 | 35% | +5.77% | **+$174** | **+$25** |
| **FAST_TP100_SL20** | ds/30s + LAZY | 81 | 33% | +3.74% | **+$151** | **+$22** |
| **BE25_TP80_SL30_S30_HYST** | hysteresis/240s | 13 | 54% | +21.30% | **+$138** | **+$20** |
| FAST_TP50_SL30_HYST | hysteresis/30s + LAZY | 24 | 38% | +4.05% | +$49 | +$7 |
| FAST_TP100_SL20_HYST | hysteresis/30s + LAZY | 24 | 29% | +3.45% | +$41 | +$6 |
| BE25_TP80_SL30_HYST | hysteresis/30s + LAZY | 21 | 38% | +5.14% | +$54 | +$8 |
| FAST_TP80_SL25_HYST | hysteresis/30s + LAZY | 21 | 38% | +4.46% | +$47 | +$7 |
| BE25_TP80_SL30_DS | ds/30s + LAZY | 37 | 38% | +2.02% | +$37 | +$5 |
| BE15_TP70_SL50_NZ | jupiter/240s | 9 | 33% | +6.81% | +$31 | +$4 |
| BE25_TP80_SL30_NZS30_HYST | hysteresis/240s | 7 | 29% | +7.13% | +$25 | +$4 |
| HIGHSCORE_TP200_SL40 | jupiter/120s | 11 | 27% | +2.27% | +$12 | +$2 |
| NOZEROLIQ_TP200_SL40 | jupiter/120s | 9 | 11% | −27.34% | **−$123** | **−$18** 🔴 |

**TOTAL paper 7d : +$1756 = +$251/jour** (positions $50/trade).

### Live 7d (actual)
- BE25_TP80_SL30 : N=25, WR 44%, avg +12.30%, +$5.28 → +$0.75/jour
- FAST_TP50_SL30 : N=48, WR 46%, avg +5.21%, +$4.31 → +$0.62/jour
- **Total projeté post-swap BE15→FAST_TP50 : +$1.37/jour**

Live avg% > paper avg% (BE25: live +12.3% vs paper +5.3%) — effet shadow-sync v142E/v143.5 (live réutilise fill Jupiter réel).

---

## 🧪 Shadows A/B v144 — résumé (attendent data)

| Dim | Shadows | Verdict ETA |
|---|---|---|
| **LAZY NOLAZY** (4 strats) | FAST_TP40/50/80 + TP50_SL15 `_NOLAZY` | Apr 22-23 N≥50 |
| **LAZY cadence** (FAST_TP50 seul) | `_LAZYFAST/MED/SLOW/XSLOW` (+ main LAZY_STD + NOLAZY = 6 variants) | Apr 25-27 |
| **Source** (4 strats) | `_BOTH/_JUPITER` sur FAST_TP50/BE25, `_BOTH` sur FAST_TP80/TP100 | Apr 25-27 |
| **SCORE filter** (5) | BE25 `_S30/_S40`, FAST_TP50/80/100 `_S40` | Apr 28-30 |
| **MCAP_MID_SCORE40** | `FAST_TP50_SL30_MCAP_S40` | Apr 28-30 |
| **COMBO top-sim** (4) | FAST_TP100/BE25/FAST_TP80/FAST_TP50 `_COMBO` (source+smoothing+polling du top sim) | Apr 25-30 |
| **Smoothing FAST_TP40** | `FAST_TP40_SL30_MED3` + `_DS` | Apr 25-27 |
| **LAZY_SLOW** (3 strats) | FAST_TP50/80 + BE25 `_LAZYSLOW` | Apr 25-27 |
| **Legacy v142** (9 shadows) | TD2, PTRAIL_V2, BOND_FAST, SCORE40_FAST, FAST_TP200/500, DIP30, BE15_TP150/200 | Apr 21-22 N≥20 |

**Attention** : sim sur-estime massivement certaines strats (TD2 sim $154/j vs réel $3.40/j = 45× ; BOND_FAST 57× ; HYST −2 à −6pp vs base). **Ne promouvoir aucun shadow en main sans N≥30 en paper réel.**

---

## 📋 Reste à faire

### ⏳ Data wait (rien à faire, laisser tourner)
- **v144 LAZY A/B** — run `scripts/compare_lazy_vs_nolazy.py` à N≥50 (Apr 22-23)
- **v144 cadence/source/smoothing/filter/COMBO verdicts** — run `scripts/refresh_main_stats.py` à N≥30 (Apr 25-30) pour chaque shadow
- **HYST paired verdict** (paired N≥30 sur 6 HYST variants en hybrid) — Apr 22-23 : retirer ceux qui perdent
- **Slip offset v144 validation** — Apr 21, re-run `scripts/diverge_report.py` : L−S median doit rester ≤ 2pp
- **Non-pump N≥30 pour split slip** — Apr 25

### 🟠 Actions après verdicts
- **LAZY verdict** : si NOLAZY > LAZY_STD → retirer des top earners de `LAZY_STRATEGIES`. Sinon confirmer LAZY_STD comme optimal (ou passer à la cadence gagnante).
- **HYST verdict** : retirer de `hybrid_strategy.allocations` les 4-6 _HYST mains qui perdent (gaspille bankroll ~$900/trade).
- **COMBO verdict** : si un COMBO shadow matche son sim ($100+/jour), envisager promotion en main. Sinon confirme artifact sim.
- **SCORE40 verdict** : si FAST_TP50/80/100_S40 matche leur retroactive (WR 68-74%), promouvoir en main + envisager live.
- **NOZEROLIQ_TP200_SL40** : perd −$123 en 7d (N=9). À retirer du hybrid si pattern persiste.
- **FAST_TP40_SL30 smoothing** : si `_MED3` ou `_DS` beats main hysteresis, changer l'orch du #1 earner.

### 🟡 Scale-up live (après verdict paper)
- **BE25 → remplacer par 2e FAST avec TP différent** (FAST_TP80 ou FAST_TP100) après 3-5j de FAST_TP50 live stable + N≥30
- **max_open_positions** : bumped 3→6. Si perfs OK, envisager 8-10 avec wallet plus gros.

### 🔒 Bloqué
- **Jupiter Trigger V2** — 0 fills historiques. Débloquer quand live_pos > $10.

### 🧠 Gotcha
- Supabase PostgREST cap 1000 rows même avec `.limit(10000)`. Toujours paginer via `.range(off, off+999)`.
- **Sim TD2/BOND/DTRAIL/TRAIL sur-estime 45-57×** — ne pas croire le sim sur ces familles.
- `slippage_actual_bps` column : signe opposé à `_dynamic_sell_slip_factor` (positive=overshoot vs positive=cost). Utiliser per-pair PnL delta pour calibration, pas cette colonne.

---

## Sim ↔ Live/Paper coherence (v144)

### Status
- **sim ↔ live per-pair median** : L−S ≤ 2.5pp
- **sim ↔ paper Spearman** : ρ = +0.905 (N=139)
- **paper ↔ live median** : ≤ 2pp per strat (gap fermé par v142E + v143.5)
- **Outliers** : 23 historiques sync=False (pré-v142E) ; zéro nouveaux sync=True depuis v143.5

### Méthodologie (3 canaux)
1. `paper_trades.paper_sim_pnl_pct` (v143.6) — PnL sim joint par trade live
2. `scripts/verify_sim_live_alignment.py` — CI nightly 04:00 UTC
3. `sim.py --mega-sweep` + `ranking_compare.py` — Spearman rank (toujours en complément)

### Slip calibration v144
`_dynamic_sell_slip_factor` : offset global −100 bps (pump/liq/mcap splits testés, aucun gain → pas de split). Revisit Apr 25.

### Monitoring CI
- `sim-align-gate.yml` (04:00 UTC) — alert si drift > 5pp
- `nightly-outlier-monitor.yml` (04:30 UTC) — alert si outlier sync=True

### Scripts (`scripts/`)
- `recap_daily.py` — $/jour paper & live
- `refresh_main_stats.py` — top earners ranking
- `compare_lazy_vs_nolazy.py` — paired LAZY verdict
- `diverge_report.py` — tableau sim/paper/live unifié
- `calibrate_slip.py` — slip model tuning
- `ranking_compare.py` — Spearman sim↔paper↔live
- `outlier_diag.py` + `nightly_outlier_monitor.py` — outlier root-cause
- `analyze_s5_filters.py` — NZ/SCORE/MCAP filter analysis
- `audit_strategies.py` — **audit alignement mains+live+shadows** (relancer avant/après toute modif)
- `verify_sim_live_alignment.py` — CI sim vs live audit

---

## Architecture summary

**Scoring :** rt_score v141 (40.5/13.5/40.5/5.4 + 3 bonuses data-driven).
**Trading :** Paper slip `_dynamic_sell_slip_factor` v144 (offset −100bps), live Jupiter Ultra RFQ. Position reconciliation sibling-aware (v133-D). Loss limit 0.5 SOL/jour.
**Orch v144** : price_source split en `source` + `smoothing` (backward-compat legacy auto-parsed). Nouveau `source=both` supporté.
**Alerting :** ML disabled (anti-predictive). Sim-align + outlier nightly alerts.

## Workflow sim

| Mode | Flag | Use case |
|---|---|---|
| Focused grid | `--from-ticks` | Ranking rapide |
| Ground truth | `--from-trades` | Vérité historique |
| 0% bias | `--from-eval-history` | Perfect replay |
| Standard sweep | `--mega-sweep` | 7 filters × 2 sources × 8 smooth × 5 poll |
| **Extended sweep** | `--mega-sweep-extended` | **12 filters × 3 sources × 9 smooth × 10 poll = 874K configs** (~3h) |

## Historique récent

- **v144** (Apr 19 complet) ✅
  - Slip `_dynamic_sell_slip_factor` offset global −100 bps
  - Shadow-sync entry étendu au path exploration (safe_scraper.py)
  - `dex_ticks` câblé dans 4 callsites `_replay_with_intervals`
  - Extended mega sweep `--mega-sweep-extended` (874K configs)
  - price_source split en `source` + `smoothing` avec backward-compat
  - `source=both` supporté dans `_decision_price`
  - 34 shadows A/B (LAZY cadence + source + SCORE filter + COMBO + smoothing)
  - Nightly outlier monitor CI + Telegram alert
  - DB swap live : BE15 → FAST_TP50_SL30, max_open_positions 3→6
  - BE15_TP100_SL50 + BE15_TP300_SL50_MCAP retirés hybrid
  - 7 entrées mortes LAZY_STRATEGIES purgées
  - `audit_strategies.py` outil d'alignement
- **v143.6** (Apr 19) DS cache TTL + `paper_sim_pnl_pct` column + CI gate
- **v143.5** (Apr 19) Live exit shadow-sync
- **v143.1-4** (Apr 18-19) Sim alignment fixes + 7 smoothing modes ports
- **v142 E** (Apr 18) Entry shadow-sync
- **v142 A-D** (Apr 18) Mega sweep 134K configs
- **v141** (Apr 17) rt_score +3 bonuses data-driven
- **v140** (Apr 17) 8 new strats, bankroll reset $18K
- **v138.5** (Apr 17) Slip recalibration per exit-type
- **v138** (Apr 17) `eval_history` JSONB + `--from-eval-history`
