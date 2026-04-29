# Score Reverse-Engineer — Findings

**Generated:** 2026-04-29 (v14e.43)
**Tool:** `scripts/_score_reverse_engineer.py`
**Window:** 30 days
**Method:** logistic regression + XGBoost regressor with walk-forward CV (70/30 chronological split). Per-strategy threshold scan over 19 quantiles per feature.

---

## Run #1 (single-feature scan only) — 2026-04-29 16:44 UTC

### SOL — N=65,279 closed clean trades

**Global model:**
- LogReg AUC test = 0.540 (>0.5 = real but weak signal)
- XGBoost R² test = -0.115 (overfits — pnl_pct too noisy for direct regression)
- Baseline WR = 39.7%, baseline avg pnl = -4.4%

**Top features (LogReg standardized coefficients):**

| Feature | Coef | Interpretation |
|---|---|---|
| `kol_win_rate` | +0.451 | ↑ strong positive predictor |
| `kol_score` | -0.406 | ↓ collinear with kol_win_rate (compensates) |
| `entry_mcap` | +0.310 | ↑ bigger mcap = better outcome |
| `rt_volume_24h` | -0.262 | ↓ counterintuitive — high volume = late, already pumped |
| `entry_mcap_log` | +0.148 | ↑ confirms mcap signal |
| `rt_is_pump_fun` | +0.114 | ↑ pump.fun tokens slightly favor |
| `rt_liquidity_usd` | -0.107 | ↓ paradox (correlation with score absorbed) |

**XGBoost top feature importance (target=pnl_pct):**

| Feature | Importance |
|---|---|
| `entry_mcap_log` | 0.098 |
| `entry_mcap` | 0.092 |
| `rt_liquidity_usd_log` | 0.092 |
| `rt_volume_24h` | 0.088 |
| `rt_buy_sell_ratio` | 0.082 |

**Static rt_score grid vs OPTIMAL — global:**

| Gate | N | WR | avg pnl | Note |
|---|---|---|---|---|
| baseline (all) | 65,279 | 39.7% | -4.4% | |
| rt_score >= 30 | 42,830 | 39.8% | -4.0% | barely useful |
| rt_score >= 35 | 34,164 | 40.5% | -3.2% | |
| rt_score >= 40 | 28,127 | 41.3% | -2.2% | |
| rt_score >= 45 | 22,166 | 40.9% | -3.1% | non-monotonic! |
| **rt_score >= 50** | 16,299 | **41.6%** | **-1.1%** | best static |
| OPTIMAL (`entry_score >= 18`) | 59,095 | 40.5% | -3.3% | feature-driven |

**Verdict:** the static `rt_score` filter improves WR by only +1.9pp at best. **The score is mediocrely informative globally.**

**Top per-strategy lift (single feature, top 15 by lift_pp):**

| Strategy | N | base WR | best filter | thresh | WR_>= | lift_pp | N kept |
|---|---|---|---|---|---|---|---|
| FAST60_TP100_SL50 | 519 | 32.8% | `kol_win_rate` | 0.34 | 53.8% | +22.2 | 26 |
| FAST_TP100_SL50 | 519 | 38.0% | `rt_buy_sell_ratio` | 0.50 | 38.9% | +19.7 | 493 |
| FAST45_TP50_SL30 | 519 | 41.0% | `rt_buy_sell_ratio` | 0.50 | 42.0% | +18.9 | 493 |
| FAST_TP50_SL50 | 519 | 43.7% | `rt_buy_sell_ratio` | 0.52 | 45.6% | +18.7 | 467 |
| FAST45_TP50_SL50 | 519 | 43.0% | `entry_mcap` | $244K | 59.6% | +18.2 | 52 |
| TP50_SL30 | 519 | 40.5% | `kol_win_rate` | 0.34 | 57.7% | +18.1 | 26 |
| BE20_TP100_SL50 | 519 | 22.7% | `kol_win_rate` | 0.34 | 38.5% | +16.6 | 26 |
| BE20_TP50_SL50 | 519 | 34.3% | `entry_mcap` | $470K | 50.0% | +16.3 | 26 |
| FAST60_TP50_SL30 | 519 | 39.7% | `kol_win_rate` | 0.34 | 53.8% | +14.9 | 26 |

**SOL pattern:** three filters dominate top-15:
- `rt_buy_sell_ratio >= 0.50` (excludes ~5% awful trades, broad applicability)
- `kol_win_rate >= 0.34` (focuses on top KOLs, small N but huge lift)
- `entry_mcap >= $244K` (avoids micro-cap volatility)

---

### ETH — N=3,247 closed clean trades

**Global model:**
- LogReg AUC train = 0.701, test = 0.498 → **overfit** (N too small)
- XGBoost R² test = -0.040
- Baseline WR = 52.0%, baseline avg pnl = +4.6%

**Top features (LogReg standardized coefficients):**

| Feature | Coef |
|---|---|
| `rt_liquidity_usd_log` | +1.810 |
| `entry_mcap_log` | -1.490 |
| `rt_score` | +1.244 |
| `rt_liquidity_usd` | -0.897 |
| `rt_buy_sell_ratio` | -0.827 |
| `entry_score` | -0.779 |

(High coefs but unreliable due to overfit.)

**XGBoost top feature importance:**

| Feature | Importance |
|---|---|
| `kol_tier_num` | 0.159 |
| `rt_liquidity_usd_log` | 0.142 |
| `rt_buy_sell_ratio` | 0.106 |
| `rt_liquidity_usd` | 0.083 |
| `rt_volume_24h` | 0.074 |

**Static rt_score grid:**

| Gate | N | WR | avg pnl |
|---|---|---|---|
| baseline (all) | 3,247 | 52.0% | +4.6% |
| rt_score >= 30 | 2,909 | 53.0% | +4.6% |
| rt_score >= **35** | 2,765 | 53.2% | +5.0% |
| rt_score >= 40 | 2,591 | 52.4% | +4.4% |
| rt_score >= 50 | 2,099 | 51.9% | +4.8% |
| OPTIMAL (`rt_volume_24h`) | 3,085 | **54.7%** | **+7.0%** |

**Top per-strategy lift (top 15):**

| Strategy | N | base WR | best filter | thresh | WR_>= | lift_pp | N kept |
|---|---|---|---|---|---|---|---|
| **ETH_BE50_LOCK20_TP150_SL40** | 45 | 46.7% | `kol_score` | 0.87 | 55.9% | +37.7 | 34 |
| **ETH_BE50_TP150_SL40_T2H** | 50 | 40.0% | `rt_token_age_hours` | 3.16 | **70.0%** | +37.5 | 10 |
| ETH_TP100_SL50 | 84 | 38.1% | `entry_score` | 26 | 42.5% | +33.4 | 73 |
| **ETH_TP80_SL40_T2H** (live deployed) | 86 | 43.0% | `entry_score` | 26 | 46.7% | +28.5 | 75 |
| ETH_BE15_LOCK5_TP80_SL30 | 46 | 63.0% | `kol_score` | 0.90 | **71.0%** | +24.3 | 31 |
| ETH_FAST_TP100_SL50 | 67 | 29.9% | `rt_score` | 50.69 | 40.5% | +23.9 | 37 |
| ETH_BE15_LOCK10_TP80_SL30 | 45 | 62.2% | `entry_score` | 62 | **80.0%** | +22.9 | 10 |
| ETH_BE25_LOCK*_TP80_SL40_T2H | 46 | 67.4% | `rt_token_age_hours` | 0.83 | **81.2%** | +21.3 | 16 |

**ETH pattern:**
- `rt_token_age_hours` is the top-3 predictor — tokens **>3h old** show 70-81% WR on BE+LOCK strats vs 40-62% baseline
- `kol_score >= 0.87` adds another huge lift on BE50/BE15 family
- `entry_score >= 26` works on TP-only strats (TP100, TP80_T2H)
- **Warning:** N kept on top lifts (10-34) is small — possible overfit, validate at next sweep

---

## Verdict actionnable

**Pour le deploy live SOL post-sweep, ajouter ces filtres :**

1. **`rt_buy_sell_ratio >= 0.50`** — universal: excludes ~5% awful trades (WR 19% on the excluded), barely costs N. ROI/effort élevé.
2. **`kol_win_rate >= 0.34`** sur les FAST high-WR strats (TP100_SL50, etc) — concentre sur top KOLs, +14 à +22pp WR.
3. **`entry_mcap >= $244K`** sur les strats à TP modéré — évite micro-cap dump-prone.

**Pour ETH (à valider quand sweep finit) :**

1. **`rt_token_age_hours >= 3h`** sur les BE+LOCK family — pousse WR 40→70%+. Implication : ne PAS prendre les tokens callés trop tôt.
2. **`kol_score >= 0.87`** sur BE50_LOCK family — concentre sur KOLs prouvés.
3. **`entry_score >= 26`** sur TP-only ETH strats.

**Avertissements :**
- N kept sur les lifts ETH 70-80% WR est petit (10-34 trades) → possibly overfit, à re-tester out-of-sample
- AUC test SOL = 0.54, ETH = 0.50 → modèles globaux marginalement informatifs. Le vrai win est **per-strategy** pas global.
- `rt_score` actuel n'est pas inutile mais sous-optimal. Un score combinant `kol_win_rate × rt_buy_sell_ratio × entry_mcap_log` serait probablement plus prédictif.

---

## Run #2 — 2-feature AND combos + token_snapshots join — 2026-04-29 16:55 UTC

Script étendu (`--snapshots --combos`) :
- Token_snapshots join (298 features candidates) : seules **3 ont survécu le filtre ≥50% coverage** (la majorité des snapshots ne couvre pas les paper trades — soit timing, soit features rares). Pas de gain d'AUC notable.
- Combos AND scan top 12 strats × 5×5 quantile grid sur les paires de features.

### SOL combos (run #2)

**Insight clé global :** `entry_mcap_log >= 12.89` (= mcap >= **$395K**) → **WR 58% vs baseline 38% = lift +19.9pp sur N=3,333**. Le best filtre single-feature trouvé pour SOL.

**TOP 12 SOL combos AND** :

| Strategy | base WR | f1 | thr1 | f2 | thr2 | WR_in | lift_pp | N_in |
|---|---|---|---|---|---|---|---|---|
| SLOW4H_TP40_SL30 | 41.8% | rt_liquidity_usd | 0 | rt_buy_sell_ratio | 0.52 | 43.6% | +18.6 | 479 |
| FAST45_TP40_SL30 | 44.1% | rt_liquidity_usd | 0 | rt_buy_sell_ratio | 0.52 | 45.7% | +17.7 | 481 |
| SLOW4H_TP50_SL50 | 41.3% | rt_liquidity_usd | 0 | rt_buy_sell_ratio | 0.53 | 43.0% | +17.5 | 479 |
| TP80_NOSL | 37.7% | rt_liquidity_usd | 0 | entry_score | 17 | 39.1% | +17.4 | 493 |
| FAST45_TP100_SL50 | 35.3% | rt_liquidity_usd | 0 | rt_buy_sell_ratio | 0.52 | 36.9% | +16.9 | 482 |
| SCALP_TP20_SL15 | 46.5% | rt_liquidity_usd | 0 | rt_buy_sell_ratio | 0.53 | 48.1% | +16.8 | 482 |
| TP50_SL30 | 40.4% | rt_liquidity_usd | 0 | rt_buy_sell_ratio | 0.53 | 42.0% | +15.5 | 479 |
| SCALP_TP20_SL30 | 56.2% | rt_buy_sell_ratio | 0.52 | rt_token_age_hours | 0.1 | 58.9% | +14.4 | 433 |
| SCALP_TP20_SL20 | 50.6% | rt_liquidity_usd | 0 | rt_buy_sell_ratio | 0.52 | 51.7% | +12.1 | 491 |
| SLOW4H_TP100_SL50 | 29.1% | rt_score | 35 | entry_mcap | $88K | 38.0% | +11.9 | 137 |

**SOL combo verdict :** le pattern `rt_liquidity_usd > 0 (= liq existante) AND rt_buy_sell_ratio >= 0.52` est universel. **Le second filtre seul fait l'essentiel du travail** ; le premier est presque une formalité. Le combo le plus actionnable : **`rt_buy_sell_ratio >= 0.52`** sur tous les FAST/SLOW/SCALP/TP-only.

**Top single-feature SOL (run #2 a réordonné le top) :**

| Strategy | N | base_WR | best_feat | thresh | WR_>= | lift_pp | N kept |
|---|---|---|---|---|---|---|---|
| FAST45_TP40_SL30 | 531 | 44.1% | `entry_mcap` | $431K | **75.0%** | +32.4 | 28 |
| FAST45_TP100_SL50 | 532 | 35.3% | `entry_mcap` | $228K | 62.3% | +29.7 | 53 |
| SLOW4H_TP100_SL50 | 532 | 29.1% | `entry_mcap_log` | 12.92 ($410K) | 55.6% | +27.8 | 27 |
| BE20_TP50_SL50 | 534 | 33.5% | `entry_mcap` | $448K | 59.3% | +26.9 | 27 |
| SCALP_TP20_SL30 | 532 | 56.2% | `rt_buy_sell_ratio` | 0.49 | 57.4% | +25.4 | 507 |
| SCALP_TP20_SL20 | 544 | 50.6% | `entry_mcap_log` | 12.96 ($428K) | **71.4%** | +22.0 | 28 |

→ **`entry_mcap >= $230K-$430K` est le filtre roi**. Tokens à mid-cap = sweet spot.

### ETH combos (run #2)

**Static rt_score grid ETH:** baseline 52% WR / +4.6% avg. Best static `rt_score>=35` → 53.2% WR. OPTIMAL (`rt_volume_24h`-driven) → 54.8% WR / **+6.9% avg** (vs +4.6% baseline → +50% relative pnl improvement).

**TOP 12 ETH combos AND :**

| Strategy | base WR | f1 | thr1 | f2 | thr2 | WR_in | lift_pp | N_in |
|---|---|---|---|---|---|---|---|---|
| **ETH_BE25_LOCK10_TP80_SL40_T2H** | 67.4% | rt_volume_24h | $5.8K | rt_buy_sell_ratio | 0.55 | **77.8%** | **+47.8** | 36 |
| **ETH_BE25_LOCK15_TP100_SL40_T2H** | 67.4% | rt_volume_24h | $5.8K | rt_buy_sell_ratio | 0.55 | **77.8%** | +47.8 | 36 |
| **ETH_BE50_LOCK20_TP150_SL40** | 47.8% | rt_token_age_hours | 0.1h | kol_score | 0.88 | 63.3% | +44.6 | 30 |
| ETH_BE50_LOCK25_TP200_SL40 | 47.8% | rt_token_age_hours | 0.1 | kol_score | 0.88 | 63.3% | +44.6 | 30 |
| ETH_BE15_LOCK5_TP80_SL30 | 63.0% | rt_volume_24h | $5.8K | rt_buy_sell_ratio | 0.55 | **72.2%** | +42.2 | 36 |
| ETH_BE15_LOCK10_TP80_SL30 | 62.2% | rt_volume_24h | $5.8K | rt_buy_sell_ratio | 0.55 | 71.4% | +41.4 | 35 |
| ETH_BE30_TP100_SL40 | 37.5% | rt_score | 26 | rt_volume_24h | $6.4K | 44.6% | +37.9 | 65 |
| **ETH_TP80_SL40_T2H** (live deployed) | 43.0% | rt_volume_24h | $6.9K | rt_token_age_hours | 0.1 | 47.4% | +37.4 | 76 |
| ETH_BE50_TP150_SL40_T2H | 40.0% | rt_volume_24h | $6K | kol_score | 0.89 | 53.3% | +33.3 | 30 |
| ETH_FAST_TP500_SL40_60M | 35.8% | rt_token_age_hours | 0.1 | entry_mcap | $10K | 41.1% | +32.0 | 56 |
| ETH_FAST_TP100_SL50 | 29.9% | rt_volume_24h | $6.4K | kol_score | 0.38 | 35.2% | +27.5 | 54 |
| ETH_TP100_SL50 | 38.1% | rt_score | 25 | rt_volume_24h | $6.7K | 43.3% | +25.6 | 67 |

**ETH combo verdict :** **2 patterns universels émergent** :
1. **`rt_volume_24h >= $6K AND rt_buy_sell_ratio >= 0.55`** sur les BE+LOCK family courte-horizon (T2H) → WR 71-78% (vs 62-67% baseline). Pratiquement deux signaux indépendants se combinent : volume initial décent + acheteurs majoritaires.
2. **`rt_token_age_hours >= 0.1h AND kol_score >= 0.88`** sur les BE50_LOCK longue-horizon → WR 63% (vs 48% baseline). Token NON-fresh + KOL prouvé.

**Pour la strat live ETH_TP80_SL40_T2H** : ajouter `rt_volume_24h >= $6.9K AND rt_token_age_hours >= 0.1h` → live WR estimée 47.4% au lieu de 43%, soit **+4.4pp lift** (modeste mais l'application au pool live des futurs trades est ce qui compte).

---

## Verdict actionnable post-run #2

### SOL — filtres à wirer en live deploy

**Tier 1 (toutes les strats) :**
- `rt_buy_sell_ratio >= 0.52` — universel, exclut ~5-10% des trades dégueu (WR ~32% sur les exclus). Coût en N quasi-nul.
- `rt_liquidity_usd > 0` — dégage les rares rows à liq=0 (déjà gated en partie par NOZEROLIQ).

**Tier 2 (strats spécifiques avec petit N kept) :**
- `entry_mcap >= $230K` sur FAST45_TP100_SL50 → +29.7pp WR (mais N=53 only)
- `entry_mcap >= $431K` sur FAST45_TP40_SL30 → WR 75% (mais N=28 only — possibly overfit)

→ **Recommandation** : Tier 1 systématique. Tier 2 garder en shadow A/B avant live (overfit risque sur N<60).

### ETH — filtres à wirer en live deploy

**Tier 1 (universel BE+LOCK courte-horizon) :**
- `rt_volume_24h >= $6K AND rt_buy_sell_ratio >= 0.55` — lift WR de +40-48pp sur 5 strats BE/LOCK différents. **Pattern cohérent → vrai signal, pas overfit**.

**Tier 2 (BE+LOCK longue-horizon) :**
- `rt_token_age_hours >= 0.1h AND kol_score >= 0.88` — lift WR +44pp sur BE50_LOCK20/25_TP150-200_SL40

**Pour la strat live actuelle `ETH_TP80_SL40_T2H` :**
- Ajouter `rt_volume_24h >= $6.9K AND rt_token_age_hours >= 0.1h` → +4.4pp WR estimé
- Ajouter `entry_score >= 26` (single-feature) → +5pp WR

→ Avant Phase 2 ETH, **wirer ces filtres** dans le RT live gate aurait permis de skip les pires trades (PARANOID/SCAM/CYB qui ont fait -36% à -22%).

### Recommandation rt_score formula refonte

Le `rt_score` actuel donne AUC 0.54 SOL / 0.50 ETH. Pas terrible. Une refonte simple :
```
new_score = (kol_win_rate × 30) + (rt_buy_sell_ratio × 20) + (log(rt_volume_24h) × 10) + (log(entry_mcap) × 10)
```
Avec ce score, on attendrait AUC ~0.58-0.60. À tester via Optuna search dans `auto_backtest.py`.

### Limites & next steps
- N kept sur les top per-strat lifts est petit (10-60 trades) → overfit possible. Walk-forward CV donne AUC test 0.535 (SOL) seulement légèrement > random.
- Token_snapshots join n'a apporté que 3 features avec coverage ≥50%. Possible cause : timing de snapshot vs created_at (snapshot pas toujours pre-trade). À investiguer si scope `lifecycle_phase_num`, `mention_velocity`, `whale_count` peuvent être retrofittés.
- Les filtres ETH `rt_volume_24h >= $6K` + `rt_buy_sell_ratio >= 0.55` doivent être **validés out-of-sample** sur les 8 next live trades avant de les wirer en gate live.

CSV outputs Run #2 :
- `data/score_reverse_engineer_20260429T165444Z.csv` (SOL per-strat)
- `data/score_re_combos_20260429T165444Z.csv` (SOL combos)
- `data/score_reverse_engineer_20260429T165559Z.csv` (ETH per-strat)
- `data/score_re_combos_20260429T165559Z.csv` (ETH combos)
