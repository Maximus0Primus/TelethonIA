# Active TODO — Feb 15, 2026

## Current State (Feb 16)

- **Scraper:** Running every 15min full cycle + 3min price refresh
- **Snapshots:** ~10,500 total | 780 labeled 24h | 122 unique tokens labeled
- **Algorithm:** v28 (soft gates, backtest alignment) + v29-v31 (labeling pipeline fixes)
- **v31:** Birdeye OHLCV fallback — recovers tokens deindexed from DexScreener/GeckoTerminal
- **scoring_config:** Balanced 30/5/10/55, ML horizon=24h, threshold=1.3x, bot_strategy=TP50_SL30
- **Frontend:** Next.js 16 + Tuning Lab + KOL Leaderboard + Token Detail
- **Backlog:** ~7,000 unlabeled snapshots (clearing via run_labeling_loop.py + GH Actions)

---

## Completed (v18 → v22 + ML v3.1)

### Algorithm Evolution (DONE)
- [x] v18: Auto-learning loop closed (scoring_config + dynamic weights + auto-optimization)
- [x] v19: Large-cap penalty ($50M→0.70x, $500M+→0.25x)
- [x] v20: 15 hardcoded constants → dynamic via scoring_config table
- [x] v21: Data quality audit — CA validation, hard gates→soft penalties (gate_mult)
- [x] v22: Dynamic ML training — auto-optimize horizon (6h/12h/24h) + threshold (1.3x/1.5x/2.0x)

### ML v2: Learning-to-Rank + Temporal Velocity + Entry Zone (DONE)
- [x] 6 temporal velocity features (score/mention/volume velocity + acceleration + kol_arrival_rate)
- [x] XGBoost rank:ndcg + walk-forward on cycles
- [x] Entry timing quality (freshness, price position, social momentum, score velocity)

### ML v3: Bot-Won Labels + Token Re-entries (DONE)
- [x] outcome_tracker: bot_won simulation (TP before SL within horizon)
- [x] train_model: `--mode bot_won` training mode
- [x] Token re-entry support in outcome tracking

### ML v3.1: Risk/Reward Prediction + Adaptive SL (DONE)
- [x] `load_risk_reward_data()` — rr_ratio = max_return / (1 + max_dd_pct / 100)
- [x] `train_risk_reward()` — XGB+LGB regression on log_rr target
- [x] Phase 3 in `auto_train()` — loops 12h/24h with min_samples=30
- [x] `--mode risk_reward` CLI
- [x] dd_by_rr_band metadata with recommended_sl_pct per band

### Temporal Features (DONE)
- [x] 4 features: day_of_week, hour_paris, is_weekend, is_prime_time
- [x] Added to train_model.py (CORE_FEATURES + prepare_features)
- [x] Added to pipeline.py (_build_feature_row for inference)
- [x] Temporal analysis in auto_backtest (day-of-week, hour bands, prime time)

### Backtest Suite v2 (DONE)
- [x] Equity curve analysis (10% risk/trade, max DD, Calmar ratio, losing streaks)
- [x] Slippage analysis (AMM model, $100-$5000 trade sizes, uses liquidity_usd)
- [x] Confidence intervals (Wilson score 95% CI, samples needed for significance)
- [x] Portfolio simulation (top 1/3/5 per cycle, equal-weight, DD tracking)
- [x] Kelly criterion (by score band, half-Kelly, edge detection)
- [x] Adaptive bot simulation (oracle SL = DD*1.2 vs fixed SL 20/30/50%)
- [x] Bot-won simulation (TP50/SL30 realistic PnL tracking)

### Data Quality Fixes (DONE)
- [x] Dedup by token_address everywhere (not symbol) — 7 files fixed
- [x] SOL price leak cleanup (17 corrupted rows)
- [x] CA .strip() + .isascii() validation
- [x] load_labeled_data() pagination for >1000 samples
- [x] numpy type sanitization for supabase-py

---

## Uncommitted Changes (pending commit)

5 files modified (2,477 lines added):
- `scraper/train_model.py` — ML v3.1 risk/reward + temporal features
- `scraper/pipeline.py` — temporal features in inference
- `scraper/auto_backtest.py` — adaptive SL + temporal analysis + 5 backtest gaps
- `scraper/outcome_tracker.py` — bot_won labels + token re-entries
- `.github/workflows/outcomes.yml` — workflow updates

---

## Active: Data Accumulation Phase

### What the system does automatically
- [x] Velocity features accumulating (4,198/7,210 = 58%)
- [x] 12h labels reached 457 (90 unique tokens)
- [ ] **12h unique tokens reach 200** → ML becomes statistically reliable (currently 90)
- [ ] **24h unique tokens reach 100** → 24h ML becomes meaningful (currently 66)
- [ ] **DD data reach 100 unique tokens** → RR model trainable (currently 53)
- [ ] auto_backtest generates reports each cycle
- [ ] auto_train grid-searches best horizon+threshold combo

### What to monitor
- `backtest_report.json` — feature_correlation, temporal analysis, confidence intervals
- ML auto-retrain deploys on p@5 improvement
- RR model appears in `model_{hz}_rr_meta.json` once DD data sufficient
- Temporal patterns (Sunday worst, Tuesday best, 19h-5h runners)

---

## Phase 1: First Bilan (when 200+ unique 12h tokens)

**Estimated:** ~2 weeks at current scraping rate

- [ ] Re-run audit: did PA correlation change? (was ~0 with 2x on 69 tokens)
- [ ] Verify ML model Spearman/precision@5 with larger test set
- [ ] Verify velocity features correlation with 2x
- [ ] Verify temporal features: does ML learn Sunday=bad, Tuesday=good?
- [ ] Check RR model dd_by_rr_band — are recommended SLs actionable?
- [ ] Review confidence intervals — are hit rates statistically significant?
- [ ] Decision: reduce PA weight if confirmed noise at 55%
- [ ] Decision: increase ML authority if Spearman > 0.30
- [ ] Decision: promote velocity features to multipliers if predictive
- [ ] Decision: make entry_premium a standalone multiplier if top signal

## Phase 2: Adjustments (after Phase 1 data confirms)

*Decisions driven by Phase 1 results. No pre-planning.*
- 2A: Rebalance weights if PA confirmed noise
- 2B: Widen ML bounds [0.3, 2.0] if Spearman > 0.50
- 2C: Velocity multipliers if features predictive
- 2D: Entry premium standalone if remains top signal
- 2E: Deploy adaptive SL to bot if RR model passes quality gate

## Phase 3: Trading Bot Integration (1 month+)

- [ ] Feed RR model recommended_sl_pct to bot per-token
- [ ] Adaptive TP based on RR band (high RR → larger TP target)
- [ ] Entry zone badge on dashboard (entry_timing_quality)
- [ ] Real-time portfolio mode: top 3/5 allocation with Kelly sizing

## Phase 4: ML Advanced (2-3 months+, 500+ unique tokens)

- [ ] Multi-horizon ensemble (1h + 12h + 24h combined signal)
- [ ] Online learning incremental
- [ ] SHAP-driven weight optimization
- [ ] Classifier "pump en cours" vs "pre-pump"
- [ ] Feature importance dashboard (SHAP waterfall per token)
- [ ] **Custom NLP model** — Fine-tune CryptoBERT on `kol_mentions.message_text → did_2x` when N > 2000 unique labeled tokens. Requires GPU (A10G+). Prématuré tant que signal sentiment ~0.02 corrélation et N < 500.

---

## Key Diagnostic Numbers (Feb 15)

| Metric | Value | Target |
|--------|-------|--------|
| Total snapshots | 7,210 | growing |
| Unique tokens (12h labeled) | 90 | 200+ for reliable ML |
| Unique tokens (DD data) | 53 | 100+ for RR model |
| Velocity coverage | 58% | 100% (new snapshots) |
| ML horizon deployed | 24h | auto-optimized |
| ML threshold deployed | 1.3x | auto-optimized |
| Best auto-train p@5 | 1.000 (N=14!) | need N>50 for confidence |

---

## Blocked (waiting for data)

- [ ] **Reliable ML metrics** — Need 200+ unique labeled tokens (currently 90 for 12h)
- [ ] **RR model deployment** — Need 100+ unique DD tokens (currently 53)
- [ ] **Bubblemaps enrichment** — Need API key (email api@bubblemaps.io)

---

## Available Now

### Stripe integration / paywall
**Effort:** Large. **Impact:** Revenue.

### Commit + deploy pending changes
**Effort:** Small. **Impact:** All v3.1 features go live.
- 5 files, 2,477 lines added
- Risk/reward ML + temporal features + backtest suite v2
