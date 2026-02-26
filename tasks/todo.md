# Pipeline Status — Updated Feb 25, 2026 (v67)

## Current State

The pipeline is **production-ready with full monitoring**. Data accumulation gates are **PASSED** — 635 unique labeled tokens (24h), well above the 200 target. Both ML training and Optuna optimization are now unblocked. Paper trading is live with 4 strategies (309 trades). Real-time KOL listener + smart RT trading deployed. Monitoring via Telegram alerts (v67).

### Live Metrics (Feb 25)

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Unique labeled tokens (24h) | 200+ | **635** | PASSED |
| Unique labeled tokens (12h) | 200+ | **633** | PASSED |
| Unique labeled tokens (6h) | 200+ | **628** | PASSED |
| Optuna gate (150 tokens) | 150 | 635 | PASSED |
| ML gate (200 test samples) | 200 | 635 | PASSED |
| Hit rate 24h | >10% | **14.2%** | Healthy |
| Hit rate 12h | — | **10.0%** | OK |
| Total snapshots | — | 34,550 | Growing ~2.5K/day |
| Paper trades | — | 309 (53 open) | Active |
| Labeling backlog (24h) | 0 | ~9,678 | Normal (OHLCV API budget limited) |
| Marked failed (max_price=0) | — | 9,428 | Dead tokens, expected |

### Paper Trading Performance

| Strategy | TP Hit | SL Hit | Timeout | Open | Avg PnL% | Total PnL$ |
|----------|--------|--------|---------|------|-----------|-------------|
| TP100_SL30 | 4 | 23 | 12 | 2 | +0.6% | **+$374** |
| MOONBAG | 6 | 28 | 0 | 35 | +0.6% | **+$287** |
| SCALE_OUT | 7 | 110 | 16 | 16 | +0.4% | **+$225** |
| TP50_SL30 | 5 | 21 | 24 | 0 | -0.1% | **-$31** |

**Key insight:** TP100_SL30 and MOONBAG are the best strategies. TP50_SL30 is the only losing strategy. SCALE_OUT has too many SL hits (74% of closed trades).

---

## Completed Work (v34-v67)

### P0-P9 (v34-v49, Feb 17-21) — See git history
- [x] Data recovery, phantom label fix, gate reform, labeling quality
- [x] Scoring reform (weights 35/10/55/0), proxy signals, momentum_mult
- [x] Data pipeline fixes (DexPaprika, zombies, FIFO ordering)
- [x] CA identity collision fix (213 symbols, full CA propagation)
- [x] Feature computation fixes (consensus norm, heat ratio, Helius TTL)
- [x] momentum_mult fix (zero = neutral), KCO dead-marking
- [x] Dynamic Optuna (119 params, 14/14 multipliers, walk-forward)

### P10 — CA RESOLUTION BUG + MULTI-STRATEGY PAPER TRADING (v50, Feb 21)
- [x] Fixed 35% wrong CAs: `resolved_ca` was NULL for 50% of mentions with CAs in `extracted_cas`
- [x] Fallback to `msg_cas` when unambiguous (1 token or 1 CA per message)
- [x] Paper Trading v2: 4 strategies (TP50_SL30, TP100_SL30, SCALE_OUT, MOONBAG) with tranche model
- [x] 15 new cols on token_snapshots, 2 on paper_trades

### P11 — NEW FEATURE BLOCKS (v53, Feb 21)
- [x] 9 new columns: holder_turnover_pct, smart_money_retention, small_holder_pct, avg_tx_size_usd, kol_cooccurrence_avg, kol_combo_novelty, jup_price_impact_500, jup_price_impact_5k, liquidity_depth_score
- [x] 3 new onchain sub-factors + hype_pen co-occurrence penalty
- [x] 8 new Optuna params (~110 total)

### P12 — EGRESS CONTROL (v60-v63-v65)
- [x] v60: Optuna multi-threshold + strategy PnL objective
- [x] v63: Emergency egress fix (37GB -> ~300MB/day, 98% reduction)
- [x] v65: Throttle expensive windows to cut Supabase egress ~60%

### P13 — REAL-TIME TRADING (v64-v66)
- [x] v64: Real-time KOL listener with instant paper trading
- [x] v66: Smart RT trading with ML strategy selection + exploration mode
- [x] Scoring config: `v66_exploration_mode`, scoring_mode=formula, ml_horizon=24h, ml_threshold=1.30

### P14 — MONITORING SYSTEM (v67, Feb 25)
- [x] `monitor.py`: In-memory MetricsCollector singleton (API calls, cycles, RT events, egress, paper trades)
- [x] `alerter.py`: Telegram Bot API alerts with per-category throttling (6 alert types)
- [x] Instrumentation: 9 API call tracking points, 3 egress tracking points, paper trade tracking
- [x] `monitor_loop()`: 5min health checks (cycles, RT liveness, API errors, egress, daily summary)
- [x] Zero new pip dependencies, fail-safe imports, zero Supabase egress for monitoring

---

## Completed (v68)

### Optuna Optimization
- [x] **Snapshot mode:** 970 tokens, improvement 0.5% < 1% threshold → params already near-optimal
- [x] **KCO mode:** Fixed constraint violation bug (99.97% trials wasted). `_suggest_ordered()` helper guarantees ordering.
- [x] **First-call mode:** Skipped (only 10/970 tokens have price_at_first_call)

### ML Training (Feb 25)
- [x] **Grid search complete:** 27 combos (3h×3t×3e). Best: **12h/1.5x** (spearman=0.328, p@5=0.800)
- [x] **Bug fix:** auto_train had first-wins bias when p@5 tied. Now uses (p@5, spearman) tuple tiebreaker.
- [x] **Switched to hybrid mode:** scoring_config updated to `scoring_mode=hybrid, ml_horizon=12h, ml_threshold=1.5`
- [x] **Model persistence:** train-models.yml now commits model files to git after training
- [x] **Bot model:** 12h/TP30_SL20 (p@5=0.600) deployed
- [x] **RR model:** 12h (p@5=0.600) deployed

### Paper Trade Strategy Reform (v68)
- [x] **TP50_SL30 horizon 12h→24h** — analysis showed TP100's edge was from horizon, not TP level
- [x] **SCALE_OUT SL -30%→-50%** — 82% SL hit rate was too tight for 48h hold
- [x] **MOONBAG SL -50%→-70%** — 7d hold needs room for intraday drawdowns

---

## Actionable Now

### 1. VPS Update (IMMEDIATE)
- [ ] **`git pull` on VPS** — Needs v68 code + model files (once training workflow commits them)
- [ ] **Verify hybrid mode active** — Check logs for "ML multiplier applied (regression mode)" after next cycle

### 2. Monitor Hybrid Mode (1-2 WEEKS)
- [ ] **Compare paper trade PnL before/after** — hybrid mode started Feb 26. Mark this date.
- [ ] **Watch for ML multiplier distribution** — Should be [0.3, 2.0] range, not all clustering at 1.0.
- [ ] **DO NOT switch to ml_primary** — N still too small. Let hybrid prove itself first.

### 3. Score Calibration (AFTER 2 weeks hybrid)
- [ ] **Re-check signal correlations at N=635+** — whale_new_entries was +0.578 at N=130 but signals collapsed at N=251.
- [ ] **If hybrid PnL improves** — Consider narrowing ML bounds or increasing ML weight.
- [ ] **If hybrid PnL doesn't improve** — Investigate if model is overfitting (same predictions for all tokens).

---

## Known Limitations (low priority)

- [ ] **narrative / narrative_is_hot** — Never implemented. Dead columns. Low ROI.
- [ ] **entry_drift_mult** — kol_stated_entry_mcaps ~95% empty. Not fixable without NLP.
- [ ] **price_drift_from_first_seen** — 0.7% coverage. 95% of tokens die before cycle 2.
- [ ] **Bubblemaps API** — Not using (too expensive).
- [ ] **Labeling backlog (~9.7K snapshots)** — Limited by OHLCV API budget. Steady-state processing, not a bug.

---

## Architecture Summary (v67)

### Scoring Engine
- **Weights:** 35/10/55/0 (consensus/conviction/breadth/PA)
- **Multiplier chain:** 14 multipliers, all recomputable from raw snapshot fields
- **Optuna:** 119 search params, 2-fold expanding walk-forward, hit-rate-dominant objective
- **Guard-rails:** 150 token minimum, 5% improvement gate, 20% train/test gap, 30% max param change, bot expectancy check

### Data Pipeline
- **Scraper:** 15min full loop + 3min price refresh, 62 KOL groups
- **RT Listener:** Real-time KOL message monitoring with instant paper trading (v64-v66)
- **OHLCV sources:** DexPaprika (primary) -> Birdeye (fallback) -> GeckoTerminal (last resort)
- **Enrichment:** DexScreener, RugCheck, Helius (30min TTL), Jupiter
- **Labeling:** outcome_tracker with zombie prevention, FIFO ordering, dead-marking
- **Cache TTLs:** DexScreener 5min, RugCheck 2h, Birdeye 1h, Jupiter 2h, Helius 30min

### ML Pipeline
- **train_model.py:** XGBoost + LightGBM ensemble, walk-forward temporal splits
- **Quality gates:** N >= 200 test, Spearman >= 0.10, dynamic p@5 (0.25/0.30/0.35 by sample size)
- **auto_train:** grid searches 6h/12h/24h x 1.3x/1.5x/2.0x, deploys best combo
- **Current config:** ml_horizon=24h, ml_threshold=1.30, scoring_mode=formula

### Monitoring (v67)
- **MetricsCollector:** In-memory singleton, tracks API calls/cycles/RT/egress/paper trades
- **Alerter:** Telegram Bot API, 6 alert types with cooldown throttling
- **monitor_loop:** 5min health checks, daily summary at 8h UTC
- **Fail-safe:** monitoring crash != scraper crash
