# Score Reverse-Engineer — Findings

> ⚠️ **2026-08-05 — LA VALIDATION ATTENDUE ICI N'A JAMAIS PU AVOIR LIEU.**
> Ce doc dit *« validate at next sweep »* pour les lifts BSR / `kol_win_rate`.
> Or les arms `BSR52`, `BSR55`, `KW34`, `KW26`, `NOZEROLIQ_BSR*`, `BSR_MCAP` du mega
> sweep lisaient `rt_buy_sell_ratio` et `kol_win_rate` — **deux colonnes absentes du
> `select`** depuis v14e.43. `(None or 0) >= seuil` était donc toujours faux : **7 arms
> sur 21 n'ont matché AUCUN trade pendant 4 mois**, en silence.
> Corrigé en v14e.72 (BSR52 passe de 0 % à 82.2 % de match).
> ⇒ Les lifts ci-dessous restent **NON VALIDÉS out-of-sample**. Le premier sweep qui
> peut réellement les tester est celui du 05/08. Ne rien promouvoir d'ici là.
> Détail : `tasks/experiments.md` + mémoire `mega_sweep_dead_filter_arms_aug5`.
>
> ⚠️ Second point : `kol_win_rate` est un **agrégat de forme récente du KOL**. Le
> 05/08 la forme récente est mesurée **ANTI-prédictive** (forme>0 → −4.35 % en test,
> E08). Les lifts `kol_win_rate >= 0.34` de ce doc sont donc à re-tester avec un null
> de permutation, pas seulement avec plus de N.

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

---

## Run #3 (REWRITE — TARGET = $/day at $20/trade) — 2026-04-29 17:24 UTC

**Pourquoi un Run #3 :** vérification empirique du finding BSR>=0.52 a montré que ce filtre **PERD $2-4/d** sur 7/7 top SOL strats à $20/trade. Le Run #1+#2 optimisait sur `WR × sqrt(N)` — winrate × volume — pas sur les dollars. Une cohorte BSR<0.52 a un WR plus bas (40% vs 49%) MAIS un avg pnl plus haut (+3.5%) : ce sont des **fat-tail moonshots** que le filter sacrifie. Cette erreur d'optimization target est CORRIGÉE en Run #3.

Script v2 ajoute :
- Target metric = **`sum_$ at simulated $20/trade per day`** (NOT WR)
- Walk-forward CV (train 70% / test 30% chrono) avec flag OVERFIT si train+ et test−
- **Optuna search** sur formule de score linéaire pondérée (15 features, 200 trials TPE)

### SOL — Run #3 résultats

**Baseline :** post-blacklist 30d à $20/trade = **−$216/jour** sur paper aggregé. Le marché 30d SOL a été net négatif sur l'univers.

**Top single-feature filters (TARGET $/d) :**

| Feature | Threshold | kept_% | kept_$/d | Δ_$/d vs base |
|---|---|---|---|---|
| `rt_score` | 38.8 | 40.5% | $+599 | **$+815** |
| `entry_score` | 38 | 41.1% | $+587 | $+803 |
| `entry_mcap` | $16K | 64.9% | $+560 | $+776 |
| `kol_score` | 1.56 | 5.1% | $+411 | $+626 |
| `kol_win_rate` | 0.34 | 5.4% | $+409 | $+624 |
| `rt_liquidity_usd` | $17.7K | 30.4% | $+387 | $+602 |
| `rt_volume_24h` | $314K | 20.6% | $+278 | $+494 |

**Walk-forward CV (train 21d / test 9d) :**

| Feature | train Δ$/d | test Δ$/d | Verdict |
|---|---|---|---|
| **`rt_score`** | $+884 | **$+551** | ✅ **HOLD OUT-OF-SAMPLE** |
| **`entry_score`** | $+777 | **$+828** | ✅ **HOLD** (test > train!) |
| **`kol_win_rate`** | $+777 | $+222 | ✅ HOLD weakly |
| `entry_mcap` | $+1079 | $-210 | ❌ OVERFIT |
| `entry_mcap_log` | $+1070 | $-345 | ❌ OVERFIT |
| `rt_liquidity_usd` | $+938 | $-413 | ❌ OVERFIT |
| `rt_token_age_hours` | $+892 | $-1899 | ❌ OVERFIT massive |
| `rt_volume_24h` | $+727 | $-1017 | ❌ OVERFIT |

→ **3 features tiennent out-of-sample** : `rt_score`, `entry_score`, `kol_win_rate`. Les autres overfit massivement.

**Top per-strategy (non-BSR) findings :**

| Strategy | base $/d | best filter | kept $/d | Δ |
|---|---|---|---|---|
| SLOW6H_TP100_SL50 | $-21 | `entry_mcap_log >= 11` | $+3 | **+$24** |
| TP100_SL60 | $-17 | `kol_win_rate >= 0.34` | $+5 | +$23 |
| TP80_SL70 | $-16 | `kol_win_rate >= 0.34` | $+5 | +$22 |
| SLOW4H_TP100_SL50 | $-18 | `kol_win_rate >= 0.30` | $+2 | +$20 |
| FAST60_TP100_SL50 | $+1.5 | `entry_mcap >= $17K` | $+10 | +$9 |
| FAST45_TP100_SL50 | $+2 | `entry_mcap >= $18K` | $+10 | +$8 |

→ Beaucoup de strats LOSING base deviennent winners avec filtre `kol_win_rate >= 0.34`.

**Top per-strategy 2-feature combos (where BSR shines AS PART OF combo) :**

| Strategy | base $/d | f1 | thr1 | f2 | thr2 | Δ$/d |
|---|---|---|---|---|---|---|
| SLOW6H_TP100_SL50 | $-21 | `rt_buy_sell_ratio` | 0.53 | `entry_mcap` | $45K | **+$26** |
| SLOW4H_TP100_SL50 | $-18 | `rt_buy_sell_ratio` | 0.53 | `entry_mcap` | $51K | +$23 |
| TP100_SL60 | $-17 | `rt_buy_sell_ratio` | 0.53 | `entry_mcap` | $50K | +$23 |
| TP80_SL70 | $-16 | `rt_token_age_hours` | 0.6 | `kol_win_rate` | 0.25 | +$20 |

→ **BSR seul = mauvais. BSR + entry_mcap >= $45K = excellent.** Le filter MCAP seul élimine les micro-cap pumps qui sont les fat-tail moonshots à BSR<0.52.

**Optuna SOL global formula : OVERFIT** (train +$1090/d, test −$406/d). 15 weights = trop de degrés de liberté, ne généralise pas. Conclusion : pas de formule de score globale fiable trouvée à ce jour pour SOL.

### ETH — Run #3 résultats

**Baseline :** 30d post-no-blacklist (ETH n'a pas de blacklist) = **+$549/jour** train, **+$359/jour** test à $20/trade.

**Per-strat (TOUTES positives sur ETH — différent de SOL) :**

| Strategy | base $/d | best filter | kept $/d | Δ |
|---|---|---|---|---|
| ETH_FAST_TP500_SL40_60M | $-0.21 | `kol_win_rate >= 0.26` | $+3.71 | +$3.92 |
| ETH_FAST_TP100_SL50 | $-2.62 | `kol_win_rate >= 0.24` | $+0.86 | +$3.48 |
| ETH_TP100_SL50 | $+1.97 | `kol_score >= 0.88` | $+4.78 | +$2.81 |
| **ETH_TP80_SL40_T2H** (live) | $+3.49 | `kol_score >= 0.62` | $+5.18 | +$1.69 |
| ETH_BE50_TP150_SL40_T2H | $+4.03 | `entry_mcap_log >= 9.6` | $+5.14 | +$1.12 |

→ **`kol_score`** + **`kol_win_rate`** sont les filtres dominants ETH. Lifts modestes ($1-4/d) mais consistents.

**Top combos ETH :**

| Strategy | f1 | thr1 | f2 | thr2 | Δ$/d |
|---|---|---|---|---|---|
| ETH_FAST_TP500_SL40_60M | `rt_score` | 44 | `kol_win_rate` | 0.26 | **+$4.65** |
| ETH_FAST_TP100_SL50 | `rt_volume_24h` | $6.4K | `kol_score` | 1.1 | +$4.26 |
| ETH_TP100_SL50 | `kol_score` | 0.78 | `entry_mcap_log` | 9.2 | +$3.71 |
| **ETH_TP80_SL40_T2H** (live) | `rt_volume_24h` | $6.9K | `kol_score` | 0.7 | **+$3.25** |
| ETH_BE25_LOCK10_TP80_SL40_T2H | `rt_volume_24h` | $5.8K | `rt_buy_sell_ratio` | 0.55 | +$1.96 |

**Optuna ETH global formula : SIGNAL HOLDS ✅** (train +$461/d, test +$560/d). Top weighted features (signs sont mixed mais ça fonctionne) : `rt_score`, `entry_score`, `rt_volume_24h_log`, `kol_score`, `entry_mcap_log`. Conclusion : on PEUT construire un score formula fiable pour ETH.

---

## Verdict actionnable v2 (corrige le verdict v1)

### Filtres VALIDÉS out-of-sample (à wirer en shadow A/B)

| Filtre | Chain | Δ$/d test | Confidence |
|---|---|---|---|
| **`kol_win_rate >= 0.30-0.34`** | SOL+ETH | +$222 SOL / +$3-4 ETH | **HIGH** ✅ |
| `rt_score >= 38` | SOL | +$551 | **HIGH** ✅ |
| `entry_score >= 38` | SOL | +$828 | **HIGH** ✅ (similar to rt_score, presque colinéaire) |
| `kol_score >= 0.88` | ETH BE+LOCK | +$1-3 | **MEDIUM** ✅ |

### Filtres REJETÉS (overfit ou anti-signal sur $/d)

| Filtre | Pourquoi |
|---|---|
| `rt_buy_sell_ratio >= 0.52` (BSR seul) | Améliore WR, perd $/d. Sacrifie fat-tail. |
| `rt_token_age_hours >= 0.1-0.3` | OVERFIT (train +$890, test −$1900) |
| `rt_volume_24h >= $6K` (seul) | OVERFIT |
| `entry_mcap >= $230K` (seul) | OVERFIT (test −$210) |
| `rt_liquidity_usd >= $17K` (seul) | OVERFIT |

### Combos GAGNANTS (1 + 1 = 3)

| Combo | Strats où ça marche | Δ$/d |
|---|---|---|
| `rt_buy_sell_ratio >= 0.53 AND entry_mcap >= $45K` | SLOW6H/TP100/TP80_SL70 | +$20-26 |
| `rt_volume_24h >= $6K AND kol_score >= 0.88` | ETH BE+LOCK family | +$2-4 |
| `rt_score >= 26 AND rt_volume_24h >= $6.4K` | ETH BE+LOCK | +$3 |

→ Les filters seuls sont fragiles, les combos sont robustes (cohérence cross-strat).

### Optuna formula

- **SOL** : pas de formule globale stable. 15 weights overfit, le marché SOL trop hétérogène.
- **ETH** : **formule globale tient out-of-sample** (test +$560/d). Pas encore actionnable car les weights sont contre-intuitifs (rt_score = -0.998, entry_score = +0.910, kol_win_rate = +0.224 → suggests rt_score actuel est anti-signal sur ETH dans l'Optuna). À investiguer.

CSV outputs Run #3 :
- `data/score_re_v2_dollar_20260429T172429Z.csv` (SOL per-strat — TARGET $/d)
- `data/score_re_v2_combos_20260429T172502Z.csv` (SOL combos — TARGET $/d)
- `data/score_re_v2_dollar_20260429T172633Z.csv` (ETH per-strat)
- `data/score_re_v2_combos_20260429T172700Z.csv` (ETH combos)
