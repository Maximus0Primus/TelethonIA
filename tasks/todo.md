# Pipeline Status — Updated Apr 21, 2026 (v144.9 sim-align overhaul)

## Current state

**Live (50/50)** — `BE25_TP80_SL30` (median_5/240s) + `FAST_TP50_SL30` (median_3/30s + LAZY). Position 0.02 SOL (~$3.40)/trade. **max_open_positions: 6**. Exposition max 0.12 SOL ≈ $20. Daily loss limit 0.5 SOL (~$85).

**Paper hybrid — 12 mains + 283 shadows** (incl. 21 v144.x A/B). Alignment audit (`verify_shadow_main_parity.py`): **0 violations sur 805 shadows post-v144.3**.

---

## v144.6-9 deployed Apr 21 (sim alignment overhaul)

### v144.6 — Fix LAZY throttling for live_sync shadows
Nightly_outlier_monitor a flaggé 4 outliers sync=True post-v144.3 (ASMORA +21pp, SAEP +25pp, TRUST x2). Cause : v144.3 a retiré le shortcut `if pos_usd==0: return True` dans `_should_evaluate_exit`, donc les paper rows `entry_source="live_sync"` (v142E shadow-sync) se sont retrouvées LAZY-throttled (180-600s) alors qu'elles doivent mirror la cadence live (30s). Fix : bypass LAZY quand `entry_source="live_sync"`. Shadows A/B purs gardent LAZY.

### v144.7 — Sim-align gate via eval_history (not price_ticks)
`sim-align-gate.yml` fail chronique 3 jours (Apr 19-20-21). Root cause : `verify_sim_live_alignment.py` reconstruisait l'input prix depuis `price_ticks` qui sample Jupiter à 3-min batch vs live 30s polling. Tokens hors rotation active → 0% coverage Jupiter → sim fallback `timeout_eod` bidons. Fix : switched to `paper_trades.eval_history` JSONB (v138+, chaque poll persisté), replay via `sim._replay_from_eval_history`. **avg=-3.78pp → -1.16pp** (3.3× mieux).

### v144.8 — Gate compares replay vs paper_sim_pnl_pct (apples-to-apples)
Encore des "divergences" trompeuses parce que le gate comparait sim_replay vs live.pnl_pct, et live.pnl_pct inclut le fill Jupiter Ultra réel (slippage positif sur spikes, ex: $CHUCHU TP=+50% fill=+120%). Fix : compare vs `paper_sim_pnl_pct` (colonne v143.6 persistée par live_trader.py:1174 — "ce que paper aurait book avec le même input"). Colonne "Jup slip" ajoutée en info. **avg=-1.16pp → -0.61pp**, max Jup slip ±0.5pp typique confirme Ultra RFQ near-zero. Aussi migré `scripts/diverge_report.py` pour préférer eval_history.

### v144.10 — 10 new shadows from EH A/B (hidden gems)
Le Spearman ρ=0.058 entre PT et EH sweeps confirme le biais structurel de price_ticks. 10 shadows ajoutées depuis les rankings EH propres :
- **7 nouvelles strats** dans STRATEGIES (TP200/TP150 cluster, rank EH 46-113) : `BE25_TP200_SL40_4H`, `TP200_SL30_2H`, `BE50_TP200_SL30_4H`, `TP200_SL30_4H`, `TP200_SL40_2H`, `TP200_SL50_4H`, `TP150_SL40_2H`
- **3 existantes** promues en shadow (MOONBAG, WIDE_RUNNER, SCALE_OUT — let-it-run profile, WR 60.9% med +8.58% sur SCORE30 subset)
- Skipped : HYST variants (v142 redundant), DIP30/DTRAIL (artifacts live), dupes TP300/500_SL50 (weak median)

ETA verdict paper paired : **Apr 28-Maj 02** (N≥30 paired vs base attendu)

### v144.9 — mega_sweep A/B price_ticks vs eval_history
Le mega_sweep (discovery de strats, dernier output = BE25_S35 + FAST_TP100_S35 v144.4/5) lisait `price_ticks` → même biais structurel 3-min Jupiter. Deux patches :
- **A (minimaliste)** : warning coverage dans `_mega_sweep_run`. Affiche `median jup ticks/token`, `% zero_jup`, `% <10_jup`. Alerte si >15% zero_jup ⇒ résultats biaisés DS fallback.
- **B (propre)** : nouveau flag `--mega-sweep-eval-history`. Universe = tokens tradés avec `eval_history`. Source forcée à jupiter (eval_history n'a pas de DS stream). Output `_mega_sweep_eh.csv`.

Usage A/B :
```
python scraper/sim.py --mega-sweep                  # legacy price_ticks
python scraper/sim.py --mega-sweep-eval-history     # ground truth
# Compare rankings; strats avec delta rank ≥ 5 = suspectes.
```

---

## v144.x deployed Apr 20

### v144.1 — 4 retraits HYST/DS losers from hybrid
Pair-test 7d (N=38-69) :
- FAST_TP80_SL25_HYST (−$62 vs base +$427)
- FAST_TP100_SL20_HYST (−$54 vs base +$137)
- BE25_TP80_SL30_HYST (+$6 vs base +$191)
- BE25_TP80_SL30_DS (−$0 vs base +$191)

### v144.2 — Bug routing paper FAST_TP50/BE25
Root cause: `paper_trader.py` open/cooldown queries n'excluaient pas `source='rt_live'` → live row bloquait paper sibling. Fix : 3 queries patchées avec `.neq("source", "rt_live")`. Avant fix, FAST_TP50 paper stoppé 32h, BE25 paper stoppé 52h.

### v144.3 — Shadow ↔ main parity
3 changements pour aligner shadows sur mains (zéro biais A/B) :
1. `_should_evaluate_exit` : LAZY throttling appliqué aux shadows aussi
2. `_override_exit_with_ultra_quote` : Ultra SELL quote sur shadows (legacy pos=0 bypass auto)
3. Shadow row creation : `position_usd = alloc_usd × tranche_pct × bot_ml_mult` (= main), entry_source tagué, ML gate appliqué

Cosmétique préservé : telegram alerts + bankroll updates restent skippés via `is_shadow=True`.

### v144.4 — `FAST_TP100_SL20_S35` shadow (top robust)
Top robust cluster sim (`analyze_mega_sweep.py` Bonferroni × 508K) : N=35, WR 62.86%, avg +28.06%, fdr_q≈0. Orch : LAZY + median_3 + jupiter.

### v144.5 — `BE25_TP80_SL30_S35` + LAZY_STRATEGIES cleanup
Sweet-spot SCORE35 sur BE25 (extrapolation FAST_TP100_S35). LAZY_STRATEGIES nettoyé : retiré 4 entrées qui référençaient des mains supprimées par v144.1.

---

## 12 Mains actives (post v144.1) — état 7d

| Strat | $/jour | Note |
|---|---|---|
| FAST_TP80_SL25 ⭐ | +$45 | top earner paper |
| FAST_TP50_SL30 (live) | +$53 | top + en live |
| BE25_TP80_SL30_S30_HYST 🚀 | +$44 | WR 56% |
| TP50_SL15 | +$40 | simple, robuste |
| HIGHSCORE_TP200_SL40 | +$35 | asymétrique |
| FAST_TP40_SL30 | +$34 | |
| BE25_TP80_SL30 (live) | +$30 | |
| FAST_TP100_SL20 | +$11 | |
| BE25_TP80_SL30_NZS30_HYST | +$8 | N=17 |
| FAST_TP50_SL30_HYST | +$8 | watch |
| BE15_TP70_SL50_NZ | +$6 | N=22 |
| NOZEROLIQ_TP200_SL40 | −$8 | 🔴 perdant N=18, retirer si pattern persiste |

**TOTAL paper 7d : ~+$2027 = +$290/jour** (positions $50/trade).

---

## Live 7d actual (avant swap v144.1)

- BE25_TP80_SL30 : N=38, WR 42%, +$4.90 → +$0.70/jour
- FAST_TP50_SL30 : N=66, WR 41%, +$1.16 → +$0.17/jour
- (legacy DTRAIL/BE15 résiduels) : −$0.30/jour
- **Total live : +$0.58/jour**, projection post-swap **+$1.4/jour**

---

## 🧪 Shadows v144.x — verdicts en attente data

| Dim | Shadows | ETA verdict |
|---|---|---|
| **NOLAZY paired** (4) | FAST_TP40/50/80, TP50_SL15 | Apr 23-25 N≥30 paired |
| **Source BOTH/JUPITER** (8) | FAST_TP40/50/80/100, BE25 | Apr 25-27 |
| **Smoothing DS/MED3** (8) | FAST_TP40/50/80/100, TP50_SL15 | Apr 25-27 |
| **SCORE filter S35/S40/S30** (10) | BE25, FAST_TP50/80/100, TP50_SL15 | Apr 25-30 |
| **MCAP_S40 / COMBO** (5) | sur top earners | Apr 25-30 |
| **LAZY cadence FAST/MED/SLOW/XSLOW** (4) | FAST_TP50_SL30 only | Apr 25-27 |
| **LAZYSLOW** (3) | FAST_TP50/80, BE25 | Apr 25-27 |
| **HIGHSCORE_*_BOTH/DS/MED3/NOLAZY** (4) | nouveaux v144.2 | Apr 27-30 |
| **v144.10 TP200/TP150 cluster** (7) | BE25_TP200_SL40_4H, TP200_SL30_2H/4H, BE50_TP200_SL30_4H, TP200_SL40_2H, TP200_SL50_4H, TP150_SL40_2H | Apr 28-Maj 02 |
| **v144.10 let-it-run** (3) | MOONBAG, WIDE_RUNNER, SCALE_OUT | Apr 28-Maj 02 |

**Règle** : N≥30 paired (pas raw) avant promotion. Re-run `paired_all_v144_shadows.py` quotidien.

---

## 📋 Reste à faire

### ⏳ Data wait (laisser tourner)
- Verdicts paired shadows v144.x (Apr 23-30)
- Slip per-cell N≥15 sur pump×tp_hit + non-pump×* (Apr 25)
- Validation FAST_TP100_SL20_S35 paper paired vs base (sim dit +28%/trade)
- Validation BE25_TP80_SL30_S35 paper paired vs base
- LIVE post-swap projection vs réel (Apr 27)

### 🟢 Maintenance rapide (faisable maintenant)
- Backfill `paper_sim_pnl_pct` historique : `python scripts/backfill_paper_sim_pnl_pct.py` (~50K updates, lent ~3h)
- Migrer `paired_all_v144_shadows.py` + `analyze_mega_sweep.py` + `verify_shadow_main_parity.py` en CI nightly
- Documenter règles HYST/DTRAIL/paired-test dans `docs/known_issues.md`

### 🔵 Sim-align follow-up (post v144.9)
- **A/B mega sweep** : run `--mega-sweep` et `--mega-sweep-eval-history` côte à côte, comparer les rankings top-40 des deux CSVs. Delta rank ≥ 5 ⇒ strat biaisée par price_ticks. Priorité sur `BE25_TP80_SL30_S35` et `FAST_TP100_SL20_S35` (ajoutées v144.4/5 depuis price_ticks sweep).
- **Vrais bugs logiques résiduels** (gate v144.8 flagged) : 4 cas à investiguer quand N > 20 paired
  - `$BUZZED BE25` +11pp : sim `timeout_eod` mais live `timeout` — sim ne trigger pas le timeout
  - `$XBT BE25` −32pp : idem
  - `$ZACHXBT BE25` +12pp : status match (be_stop/be_stop) mais exit_price diverge — bug formule be_stop
  - `$ACHI BE25` −12pp (dont 4.7pp Jup slip) : mix logique + slippage
- **Assouplir threshold gate** (optionnel) : `sim-align-gate.yml` passer `within_10pp >= 80%` → 70% si v144.8 ne suffit pas à le faire vert. Seuil actuel est cohérent avec le signal propre, laisser 80% pour forcer l'investigation des 4 bugs résiduels.
- **Attendre 48-72h** : la colonne `paper_sim_pnl_pct` se remplit progressivement (17/45 trades actuels). Gate plus stable avec N≥40.
- **Backfill eval_history** pré-v138 : N/A (pré-v138 n'a pas la colonne, laisser filtrer naturellement).

### 🟠 Actions après verdicts
- **NOZEROLIQ_TP200_SL40** : si pattern perdant à N≥30, retirer du hybrid (~+$8/j net)
- **Top winners shadow paired** : promouvoir 1-2 en main paper si Δpp ≥ +5pp
- **FAST_TP100_SL20_S35** (sim top robust) : si paper paired confirme → main paper + envisager live
- **HYST + filtre** : si confirmation N≥30, scaling de la famille S30/NZS30

### 🟡 Scale-up live (après verdict paper)
- BE25 → remplacer par 2e FAST avec TP différent (FAST_TP80 ou FAST_TP100) après FAST_TP50 stable + N≥30
- max_open_positions 6 → 8-10 si bankroll grandit
- Position size live $3.40 → $10-20/trade (gain x3-x6 attendu)

### 🔒 Bloqué
- **Jupiter Trigger V2** — 0 fills historiques. Débloquer quand live_pos > $10.

---

## 🛠 Chantiers planifiés (sprint format)

### Sprint #1 — Refinement slip model (1-2h, gain 0.3-0.5pp médiane)
**Cible** : ETA Apr 25-28 quand N≥30 par cellule pump×exit_type
**Quoi** :
- Remplacer 3 buckets liq actuels (5K/20K/50K) dans `_dynamic_sell_slip_factor` par modèle continu utilisant `price_ticks.volume_24h` + `price_ticks.liquidity_usd` à exit_time
- Calibrer sur les ~143 paires live/paper matched (`scripts/slip_per_exit_type.py`)
- Modèle suggéré : `slip_bps = base × (1 + α × log(50K / liq)) × (1 + β × volume_volatility) × exit_type_mult`
**Coût** : ~200 lignes paper_trader + tests + redéploiement

### Sprint #2 — Coherence sim trail/dtrail/dip family (post Apr 25)
**Problème** : sim mega_sweep top picks famille trail/dtrail/dip alors que paper/live confirment artefact (DTRAIL10 sim top vs live 65% reconciled, slip 47×)
**Options** :
- (a) Modéliser `position_reconciler` dans sim (~150 lignes)
- (b) Per-family slip multiplier × 5-10 sur DTRAIL/TRAIL/DIP (~50 lignes)
- (c) Post-process flag `family_realism` dans `analyze_mega_sweep.py` — **fait Apr 20**, à itérer
**Reco** : (b) data-driven simple, puis (a) si rigueur nécessaire

---

## 🧠 Gotcha
- Supabase PostgREST cap 1000 rows même avec `.limit(10000)`. Toujours paginer via `.range(off, off+999)`.
- **Sim mega_sweep over-estimates massivement** trail/dtrail/dip/HYST (45-57×). 44% du sweep est de cette famille → toujours filtrer via `family_realism` avant promotion.
- `slippage_actual_bps` column : signe opposé à `_dynamic_sell_slip_factor`. Utiliser per-pair PnL delta pour calibration.
- **Dedup paper/live asymétrique** (intentionnel) : paper exclut rt_live, live n'inclut que rt_live. Edge case sur KOL recall <24h après SL — bias ~5-10% optimiste paper. Pas un bug, design OK.
- **Per-trade Spearman ρ ≠ Per-strategy Spearman ρ** — par-trade ~0.9, par-strat ~0.7. Toujours préciser le niveau.

---

## Sim ↔ Live/Paper coherence (post v144.3)

### Status actuel
- **shadow ↔ paper main** : 100% parité (post v144.3) ✅
- **paper ↔ live médiane** : ≤2pp (target tenu)
- **paper ↔ live mean** : +5pp paper > live (queue lourde)
- **sim per-trade ↔ paper** : ρ ≈ +0.9
- **sim per-strategy ↔ paper** : ρ ≈ +0.7 (excluant shadow v144 polluants : ρ +0.71)

### Slip calibration v144
`_dynamic_sell_slip_factor` : offset global −100 bps. Splits per-cell pas faits faute de N. Revisit Apr 25-28.

### CI Monitoring
- `sim-align-gate.yml` (04:00 UTC) — alert si drift > 5pp
- `nightly-outlier-monitor.yml` (04:30 UTC) — alert si outlier sync=True

### Méthodologie 3 canaux
1. `paper_trades.paper_sim_pnl_pct` (v143.6) — PnL sim joint per-trade
2. `scripts/verify_sim_live_alignment.py` — CI nightly
3. `sim.py --mega-sweep` + `ranking_compare.py` — Spearman rank

---

## Architecture summary

**Scoring** : rt_score v141 (40.5/13.5/40.5/5.4 + 3 bonuses).
**Trading** : Paper slip `_dynamic_sell_slip_factor` v144 (offset −100bps), live Jupiter Ultra RFQ. Loss limit 0.5 SOL/jour.
**Orch v144** : `source` + `smoothing` split via `strategy_overrides` JSONB. `source=both` supporté.
**Alerting** : ML disabled (anti-predictive). Sim-align + outlier nightly alerts.
**Shadow ↔ main** : 100% parité comportementale post v144.3 (sauf alerts/bankroll).

## Workflow sim

| Mode | Flag | Source | Biais | Use case |
|---|---|---|---|---|
| Focused grid | `--from-ticks` | price_ticks | ⚠️ 3-min jup batch | Ranking rapide legacy |
| Ground truth | `--from-trades` | paper_trades.pnl_pct | ✅ exact | Vérité historique (strats déjà tradées) |
| 0% bias | `--from-eval-history` | eval_history JSONB | ✅ 30s exact | Perfect replay per-trade |
| Standard sweep | `--mega-sweep` | price_ticks | ⚠️ biaisé | Discovery legacy (warning coverage depuis v144.9) |
| Extended sweep | `--mega-sweep-extended` | price_ticks | ⚠️ biaisé | 874K configs (~3h) |
| **Ground truth sweep** | `--mega-sweep-eval-history` | eval_history | ✅ 30s | **v144.9 — A/B vs legacy, discover sans biais** |
| Annotation | `analyze_mega_sweep.py` | — | — | Multi-test correction (FDR/Bonferroni) + family_realism flag |

## Scripts (`scripts/`)

| Script | Usage |
|---|---|
| `recap_daily.py` | $/jour paper & live |
| `refresh_main_stats.py` | top earners ranking |
| `compare_lazy_vs_nolazy.py` | paired LAZY verdict |
| `paired_all_v144_shadows.py` | **paired audit + gap detection v144** |
| `verify_shadow_main_parity.py` | **invariants v144.3 shadows** |
| `diverge_report.py` | tableau sim/paper/live unifié |
| `slip_per_exit_type.py` | per pump×exit_type calibration |
| `spearman_drift_check.py` | Spearman 4×4 matrix |
| `analyze_mega_sweep.py` | **multi-test correction + family_realism** |
| `backfill_paper_sim_pnl_pct.py` | backfill `paper_sim_pnl_pct` historique |
| `audit_strategies.py` | audit alignement mains+live+shadows |
| `verify_sim_live_alignment.py` | CI sim vs live audit |

---

## Historique récent

- **v144.9** (Apr 21) mega_sweep A/B : warning coverage jup (A) + `--mega-sweep-eval-history` mode (B)
- **v144.8** (Apr 21) Sim-align gate apples-to-apples (vs `paper_sim_pnl_pct`, Jup slip info) + diverge_report migration
- **v144.7** (Apr 21) Sim-align gate switched from price_ticks to eval_history replay (−3.78pp → −1.16pp)
- **v144.6** (Apr 21) Fix LAZY throttling bypass pour live_sync shadows (4 outliers Apr 21)
- **v144.5** (Apr 20 PM) BE25_TP80_SL30_S35 + LAZY_STRATEGIES cleanup (4 dead entries)
- **v144.4** (Apr 20 PM) FAST_TP100_SL20_S35 — top robust sweep cluster
- **v144.3** (Apr 20 PM) Shadow ↔ main behavioral parity (LAZY + Ultra exit + position)
- **v144.2** (Apr 20 PM) Bug routing paper FAST_TP50/BE25 (rt_live blocking sibling) + 19 new shadows pour gaps couverture
- **v144.1** (Apr 20) 4 retraits HYST/DS losers from hybrid_strategy.allocations
- **v144** (Apr 19) Slip offset −100bps + extended mega sweep + price_source split + 34 A/B shadows + audit_strategies tool
- **v143.6** (Apr 19) DS cache TTL + `paper_sim_pnl_pct` column + CI gate
- **v143.5** (Apr 19) Live exit shadow-sync
- **v143.1-4** (Apr 18-19) Sim alignment fixes + 7 smoothing modes ports
- **v142E** (Apr 18) Entry shadow-sync
- **v141** (Apr 17) rt_score +3 bonuses data-driven
- **v140** (Apr 17) 8 new strats, bankroll reset $18K
- **v138.5** (Apr 17) Slip recalibration per exit-type
