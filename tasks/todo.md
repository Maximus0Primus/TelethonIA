# Pipeline Status — Updated Feb 21, 2026 (v49)

## Current State

The pipeline is **healthy and accumulating data**. All critical bugs from the Feb 17 audit are fixed. The scoring engine has 119 Optuna-searchable parameters with zero hardcoded constants in multiplier recomputation. The main bottleneck is **data volume** — need ~200+ unique labeled tokens for ML and Optuna to produce statistically reliable results.

---

## Completed Work (v34-v49)

### P0 — DATA RECOVERY (v34, Feb 17)
- [x] Fix phantom labels: Reset 16,418 phantom labels to NULL across all horizons
- [x] Fix `_mark_ohlcv_failed()` + `_mark_no_price()` + `_mark_dead_pool()`: leave did_2x as NULL
- [x] Add API keys to GH Actions: `BUBBLEMAPS_API_KEY` + `JUPITER_API_KEY`
- [x] Fix kol_call_outcomes: DexPaprika fallback + Phase C computes `did_2x` + `max_return`

### P1 — GATE REFORM (v34, Feb 17)
- [x] Remove anti-predictive gates: high_risk_score, wash_trading, low_liquidity (kept as ML features)
- [x] Keep valid gates: freeze_authority, mint_authority, top10_concentration, low_holders

### P2 — LABELING QUALITY (v34, Feb 17)
- [x] First-appearance dedup in backtest (was counting same token 20-30x)
- [x] Re-run auto_backtest on clean data (N=88 deduped)

### P3 — SCORING REFORM (v35, Feb 18)
- [x] Weights: 35/10/55/0 (consensus/conviction/breadth/PA)
- [x] Proxy signals: kol_freshness, mention_heat_ratio stored in snapshots
- [x] momentum_mult [0.7,1.4]: 14th multiplier (kol_fresh + mention_heat + vol_heat)
- [x] Whale boost in onchain_mult: whale>=5 -> 1.4x

### P4 — DATA PIPELINE FIXES (v36, Feb 18)
- [x] DexPaprika timestamp bug: `time_open`/`time_close` not `time`
- [x] Zombie snapshot infinite retry loop: `max_price=0` sentinel
- [x] Batch ordering: FIFO (snapshot_at ASC) instead of score-DESC

### P5 — CA IDENTITY COLLISION (v40, Feb 19)
- [x] 213 symbols with multiple CAs: pipeline now propagates CA through entire chain
- [x] `_resolve_pair_to_symbol_and_ca()`: returns (symbol, token_ca)
- [x] Cache invalidation on CA change

### P6 — FEATURE COMPUTATION FIXES (v41, Feb 19)
- [x] Consensus normalization 0.05 -> 0.10
- [x] mention_heat_ratio epsilon 0.1 -> 1.0 + cap 10
- [x] breadth_score unified /10
- [x] Helius TTL 2h -> 30min (whale detection 4x faster)
- [x] Entry price correction (>30% divergence -> use candle open)

### P7 — MOMENTUM FIX + KCO DEAD-MARKING (v42, Feb 20)
- [x] momentum_mult: zero = neutral (was penalizing 79% of tokens)
- [x] KCO Phase B: dead_no_ohlcv marking instead of infinite retry

### P8 — DYNAMIC PARAMETER OPTIMIZATION (v44-v48, Feb 21)
- [x] All hardcoded constants -> scoring_config table (JSONB for grouped breakpoints)
- [x] Raw features added to ML: activity_ratio_raw, kol_freshness, etc.
- [x] Optuna: 200 trials, TPE sampler, walk-forward split
- [x] 14/14 multipliers recomputable from raw snapshot fields
- [x] ~102 search params (v48)

### P9 — EVALUATION ENGINE OPTIMIZATION (v49, Feb 21)
- [x] Zero hardcoded constants in recompute blocks (onchain slopes, safety slopes, pump thresholds)
- [x] +17 new trial.suggest_* calls -> **119 total search params**
- [x] conviction_offset / conviction_divisor now Optuna-searchable
- [x] Objective reweight: 50/30/20 -> **30/20/50** (hit-rate-dominant for single-pick bot)
- [x] Post-Optuna bot validation: rejects params with negative #1-pick expectancy
- [x] 2-fold expanding walk-forward (was single 70/30 split)
- [x] Quality gate in train_model.py: Spearman check + dynamic p@5 threshold by sample size

---

## Remaining Work

### ACTIVE — Data Accumulation (bottleneck)
- [ ] **Accumulate 200+ unique labeled tokens** — Currently the main blocker for ML and Optuna. Scraper runs every 15min via GH Actions. Need ~1-2 more weeks.
- [ ] **Run first real Optuna optimization** — Gate: 150 unique labeled tokens minimum. Will auto-run when data threshold is met.
- [ ] **Run ML training on clean data** — Gate: 200 test samples minimum. Quality gate also requires Spearman >= 0.10 and dynamic p@5 threshold.

### KNOWN LIMITATIONS (low priority, not blocking)
- [ ] **narrative / narrative_is_hot** — Never implemented. Dead columns in DB. Could implement trend detection from Telegram message themes, but ROI unclear with current data volume.
- [ ] **entry_drift_mult mostly inert** — Logic exists but `kol_stated_entry_mcaps` ~95% empty. KOLs rarely state entry MCap in messages. Not fixable without NLP improvement or manual labeling.
- [ ] **price_drift_from_first_seen** — 0.7% coverage. Structurally limited: 95% of tokens die before cycle 2. Only useful for tokens that persist across multiple scrape cycles.
- [ ] **Bubblemaps API** — User decided NOT to use (too expensive). Removed from GH Actions workflow. May revisit later.

### FUTURE OPTIMIZATION (when data allows)
- [ ] **Optuna search space expansion** — ~32 safety/momentum/size sub-params still fixed in Optuna search space (bounded by ranges but not trial variables). Adding them would push to ~150 params but needs 500+ trials to explore effectively.
- [ ] **Multi-horizon Optuna** — Currently optimizes 12h only. Could grid-search 6h/12h/24h horizons.
- [ ] **Ensemble scoring modes** — `scoring_mode` column exists (formula/hybrid/ml_primary) but only formula is active. Hybrid mode needs reliable ML model first.

---

## Architecture Summary (v49)

### Scoring Engine
- **Weights:** auto_backtest optimized, currently 35/10/55/0 (consensus/conviction/breadth/PA)
- **Multiplier chain:** 14 multipliers, all recomputable from raw snapshot fields
- **Optuna:** 119 search params, 2-fold expanding walk-forward, hit-rate-dominant objective
- **Guard-rails:** 150 token minimum, 5% improvement gate, 20% train/test gap, 30% max param change, bot expectancy check

### Data Pipeline
- **Scraper:** 15min full loop + 3min price refresh, 59 KOL groups (13 S-tier, 46 A-tier)
- **OHLCV sources:** DexPaprika (primary) -> Birdeye (fallback) -> GeckoTerminal (last resort)
- **Enrichment:** DexScreener, RugCheck, Helius (30min TTL), Jupiter (when available)
- **Labeling:** outcome_tracker with zombie prevention, per-horizon age filter, FIFO ordering
- **Cache TTLs:** DexScreener 5min, RugCheck 2h, Birdeye 1h, Jupiter 2h, Helius 30min

### ML Pipeline
- **train_model.py:** XGBoost + LightGBM ensemble, walk-forward temporal splits
- **Quality gates:** N >= 200 test, Spearman >= 0.10, dynamic p@5 (0.25/0.30/0.35 by sample size)
- **auto_train:** grid searches 6h/12h/24h x 1.3x/1.5x/2.0x, deploys best combo

---

## Key Metrics to Watch

| Metric | Target | Current Status |
|--------|--------|---------------|
| Unique labeled tokens (24h) | 200+ | ~100-130 (accumulating) |
| Optuna gate | 150 tokens | Approaching |
| ML gate | 200 test samples | Not yet met |
| whale_new_entries correlation | >0.4 (at N=130) | +0.578 (promising but N small) |
| Score anti-predictive? | No | r=-0.140 at N=251 (still anti-predictive) |
| Best signal | TBD | whale_new_entries (+0.578), needs more data |
