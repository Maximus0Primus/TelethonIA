# Active TODO — Feb 14, 2026

## Current State (updated after audit)
- **Scraper:** Running every 15min full cycle + 3min price refresh
- **Snapshots:** 3,289 total | 471 labeled 12h (29 winners) | 1,677 labeled 1h (40 winners)
- **Velocity features:** 225 snapshots with score_velocity/mention_velocity/volume_velocity
- **ML Model 12h:** Regression trained on 145 samples (p@5=0.80, Spearman=0.289). Needs retrain with 3x more data.
- **ML Model 1h:** NOT YET TRAINED. 1,677 labels ready.
- **Algorithm:** v18 — auto-learning loop closed (scoring_config + dynamic weights + auto-optimization)
- **Frontend:** Next.js 16 + Tuning Lab + KOL Leaderboard + Token Detail

---

## Audit Plan: Data-First Approach (Feb 14, 2026)

### Phase 0: Accumulate Data (NOW → 2 weeks)
**Strategy:** Zero code changes. Let the system collect and auto-learn.

What happens automatically:
- [x] Velocity features accumulating (225/3289 snapshots so far)
- [ ] Labels 12h reach 500 → auto-retrain triggers (currently 471, ~2 days away)
- [ ] auto_backtest generates reports each cycle with updated correlations
- [ ] auto_apply_weights adjusts if >5pp improvement found (100+ labels)

What to monitor:
- `backtest_report.json` — feature_correlation and top1_hit_rate
- velocity features appearing in snapshots
- first ML model deployment via auto-retrain

**Immediate actions (non-destructive) — DONE Feb 14:**
- [x] Train 1h model: **p@5=0.60, p@10=0.70, Spearman=0.256** (1,000 samples used, pagination bug limited from 1,677)
  - Top SHAP: token_age_hours, liq_mcap_ratio, safety_penalty, short_term_heat
  - 28 features (extended tier), XGBoost+LightGBM ensemble
- [x] Retrain 12h LTR: **p@5=0.60, NDCG@5=0.358, Spearman=0.217** (459 samples, 15 cycles)
  - Top SHAP: token_age_hours, volatility_proxy, price_change_6h
  - Note: Previous regression model (p@5=0.80 on 145 samples) was better — small data issue
  - 12h regression attempt: Spearman 0.626 (excellent!) but p@5=0 (test window had no winners in top-5)
- [x] Fixed: `reg:squaredlogerror` → `reg:squarederror` (crash on negative returns)
- [x] Fixed: `load_labeled_data()` pagination for >1000 samples
- [x] Added: 1h horizon support in train_model.py

### Phase 1: First Bilan (in ~2 weeks, when 2000+ labels)
- [ ] Re-run audit: did PA correlation change? (currently ~0 with 2x)
- [ ] Verify ML model Spearman/precision@5
- [ ] Verify velocity features correlation with 2x
- [ ] Verify entry_timing_quality values in DB
- [ ] Decision: reduce PA weight if confirmed noise at 55%
- [ ] Decision: increase ML authority if Spearman > 0.30
- [ ] Decision: promote velocity features to multipliers if predictive
- [ ] Decision: make entry_premium a standalone multiplier if top signal

### Phase 2: Adjustments (after Phase 1 data confirms)
*Decisions driven by Phase 1 results. No pre-planning.*
- 2A: Rebalance weights if PA confirmed noise
- 2B: Widen ML bounds [0.3, 2.0] if Spearman > 0.50
- 2C: Velocity multipliers if features predictive
- 2D: Entry premium standalone if remains top signal

### Phase 3: Entry Zone + Buy Signal (1 month+)
- [ ] Validate entry_timing_quality correlation with did_2x
- [ ] If positive → soft multiplier (>0.7 = 1.15x, <0.2 = 0.85x)
- [ ] Buy zone badge on dashboard

### Phase 4: ML Advanced (2-3 months+, 5000+ labels)
- [ ] Multi-horizon ensemble (1h + 6h + 12h)
- [ ] Online learning incremental
- [ ] SHAP-driven weight optimization
- [ ] Classifier "pump en cours" vs "pre-pump"

---

## Key Diagnostic Numbers (Feb 14)

| Metric | Value | Target |
|--------|-------|--------|
| #1 token 12h hit rate | 15% (3/20) | >50% |
| Score discrimination | 1.0x | >2.0x |
| Top correlation: entry_premium | -0.393 | confirm with more data |
| Top correlation: unique_wallet_24h_change | +0.339 | confirm with more data |
| PA score correlation with 2x | -0.045 | investigate (55% weight!) |

---

## Blocked (waiting for data)
- [ ] **Weight auto-adjustment** — Need 100+ unique labeled tokens
- [ ] **Bubblemaps enrichment** — Need API key (email api@bubblemaps.io)

---

## Available Now

### Stripe integration / paywall
**Effort:** Large. **Impact:** Revenue.
