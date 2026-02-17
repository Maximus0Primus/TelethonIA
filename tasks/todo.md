# FULL PIPELINE AUDIT — Feb 17, 2026

## VERDICT: The pipeline has fundamental data quality issues that make the score essentially random. Fix these BEFORE touching weights, ML, or adding features.

---

## CRITICAL BUG #1: Phantom Labels (Poisoned Training Data)

**Location:** `outcome_tracker.py:773-786` (`_mark_ohlcv_failed()`)

**The Bug:** When all OHLCV sources fail AND snapshot is >48h old, the code sets `did_2x_*h = false` with `max_price_*h = NULL`. This means "we don't know the price" is recorded as "token did NOT 2x".

**Impact (measured from DB):**
- `did_2x_7d = false` with `max_price_7d = NULL`: **2702 snapshots** (100% of 7d labels!)
- `did_2x_72h = false` with `max_price_72h = NULL`: **2941 snapshots**
- `did_2x_48h = false` with `max_price_48h = NULL`: **3832 snapshots**
- `did_2x_24h = false` with `max_price_24h = NULL`: **3019 snapshots**
- **ALL training data is poisoned.** Many "losers" are actually unlabeled tokens.
- Backtest reports 2.55% overall hit rate, but TRUE deduped rate is **16.7%** (7x higher).

**Fix:** Never set `did_2x = false` without actual price data. Set to NULL instead.

---

## CRITICAL BUG #2: Gates Are Killing The Best Performers

**DB evidence — per-gate hit rate at 24h (with real price data):**

| Gate | N | 2x Hit Rate | Avg Return |
|------|---|------------|------------|
| **high_risk_score** | 45 | **46.7%** | 5.48x |
| **wash_trading** | 21 | **38.1%** | 1.80x |
| **early_snapshot** | 141 | **18.4%** | 1.68x |
| **low_liquidity** | 121 | **17.4%** | 1.54x |
| scored (no gate) | 1754 | 9.9% | 1.45x |
| top10_concentration | 1004 | 1.6% | 2.20x |
| single_a_tier | 1908 | 0.1% | 1.25x |
| freeze_authority | 336 | 0.0% | 1.01x |

**The "high_risk_score" gate has 46.7% hit rate** — nearly 5x better than scored tokens! We're penalizing the best performers. Safety is anti-predictive in memecoins (known since v9 but still enforced).

**Valid gates (keep):** freeze_authority, mint_authority, single_a_tier, top10_concentration
**Anti-predictive gates (remove/invert):** high_risk_score, wash_trading, early_snapshot, low_liquidity

---

## CRITICAL BUG #3: Score Is Not Predictive

**Deduped correlation analysis (N=108 unique tokens with real 24h prices):**

| Signal | Correlation with Return | Status |
|--------|------------------------|--------|
| price_action_val | +0.262 | Strongest (it's 80% of score) |
| score_at_snapshot | +0.184 | Weak overall |
| activity_mult | +0.146 | Moderate |
| breadth_val | +0.101 | Weak |
| consensus_val | +0.075 | Almost zero |
| entry_premium | -0.058 | Slightly negative |
| conviction_val | **-0.100** | **ANTI-PREDICTIVE** |

**Score band hit rates (deduped, real prices only):**

| Band | N | True 24h Hit Rate |
|------|---|-------------------|
| 80-100 | 40 | 12.5% |
| 70-79 | 24 | 8.3% |
| 60-69 | 60 | 10.0% |
| 50-59 | 84 | 2.4% |
| 40-49 | 110 | 10.9% |
| 30-39 | 223 | 9.9% |
| **0-29** | **981** | **12.6%** |

Score 0-29 (12.6%) outperforms 70-79 (8.3%). The score is essentially random noise.

**Root cause:** w_consensus=0.000, w_conviction=0.050 (anti-predictive!), w_price_action=0.800. Auto_backtest zeroed consensus because it has near-zero correlation. The algorithm is just a momentum tracker, not a KOL consensus system.

---

## CRITICAL BUG #4: 70% of Enrichment Features Are Broken/Empty

**Feature coverage on scored snapshots (N=12,917):**

| Feature | Coverage | Root Cause |
|---------|----------|-----------|
| bubblemaps_score | **0.0%** | API key missing in GH Actions |
| bubblemaps_* (5 fields) | **0.0%** | API key missing in GH Actions |
| jup_tradeable | **0.8%** | API key missing in GH Actions |
| jup_price_impact_1k | **0.8%** | API key missing in GH Actions |
| narrative | **0.0%** | Never implemented (hardcoded NULL) |
| narrative_is_hot | **0.0%** | Never implemented (hardcoded 0) |
| price_at_first_call | **0.7%** | Pool address 70% NULL |
| entry_drift_mult | **~100% but inert** | kol_stated_entry_mcaps 95% empty |
| price_drift_from_first_seen | **0.7%** | Requires 2 cycles (95% die first) |
| helius_* | 25.7% | Top N only (expected) |
| RSI/MACD/BB | 24-27% | Need OHLCV candles |
| holders (Birdeye) | 35.6% | Top 20 only |
| lp_locked_pct | 37.8% | RugCheck coverage |
| PA score | 72.1% | Good (OHLCV-dependent) |

**5-minute fix:** Add `BUBBLEMAPS_API_KEY` and `JUPITER_API_KEY` to `.github/workflows/scrape.yml`. This recovers Bubblemaps (0% → 85%) and Jupiter (0.8% → 80%).

---

## CRITICAL BUG #5: kol_call_outcomes Nearly Empty

**DB state:** 490 rows, but only **6 have entry_price** (1.2%). The v2 KOL leaderboard is useless.

**Root cause chain:**
1. Phase B needs `pair_address` → only 30% of tokens have it
2. Falls back to GeckoTerminal pool lookup → hits 429 after 3 requests
3. `_gecko_disabled = True` for **entire run** → all remaining KOLs skipped
4. Phase C computes `ath_after_call` but **never computes `did_2x`** (missing logic)

---

## CRITICAL BUG #6: Snapshot Dedup Inflation

**The same token is scored 20-30 times (every 15-min cycle):**

| Band | Snapshots | Unique Tokens | Avg Snaps/Token |
|------|-----------|---------------|-----------------|
| 70+ | 66 | 12 | 5.5x |
| 50-69 | 196 | 32 | 6.1x |
| 30-49 | 465 | 51 | 9.1x |
| 0-29 | 4200 | 152 | **27.6x** |

**Total:** 4927 scored snapshots, but only **247 unique tokens**. The ML model trains on 20x duplicated data with correlated labels. This is data leakage.

---

## USER'S QUESTION: 15-min Snapshot Price Drift

**Confirmed:** The score stays high but price moves. Example from $CORPUS:
- Snapshot 1: score=100, price=$0.000179, first_seen=$0.000157 (+14% drift)
- Snapshot 2: score=100, price=$0.000231, first_seen=$0.000157 (+47% drift)
- Snapshot 3: score=100, price=$0.000116, first_seen=$0.000157 (-26% drift)

All three get `did_2x_24h` measured from THEIR snapshot price. So:
- At $0.000179: needs $0.000358 for 2x (unlikely)
- At $0.000116: needs $0.000232 for 2x (actually happened — but only THIS snapshot counts as hit)

**The same token scores 100 at all prices.** The score doesn't account for "you're already late." `entry_premium` tries to handle this but is inert (95% missing data).

---

## WHAT DOESN'T WORK / IS USELESS

1. **w_consensus = 0.000** — Auto_backtest zeroed it. Consensus is dead.
2. **w_conviction = 0.050, negative correlation** — Higher conviction = worse returns.
3. **narrative/narrative_is_hot** — Never implemented. Dead columns.
4. **entry_drift_mult** — Inert (always 1.0).
5. **price_drift_from_first_seen** — 0.7% coverage, useless.
6. **Bubblemaps** — 0% data. Missing API key.
7. **Jupiter** — 0.8% data. Missing API key.
8. **kol_call_outcomes** — 99% empty entry_price. V2 leaderboard is theater.
9. **did_2x_7d** — 0% real labels (all phantom). All 7d labels are FALSE without price data.
10. **did_2x_72h/48h** — Heavily contaminated (2941/3832 phantom labels).
11. **Backtest** — Reports 2.55% hit rate, but true rate is 16.7%. All recommendations based on poisoned data.
12. **auto_backtest weight optimization** — Optimizing on poisoned data with duplicates. Results are meaningless.
13. **290+ hardcoded constants** — Only 15-26 are in scoring_config. Can't tune without code deploy.

---

## WHAT WORKS

1. **Token extraction** — Excellent. CA/ticker/URL extraction is solid. Dedup works. No false positives.
2. **Update/brag detection** — Good (minor gaps: "just realized", "portfolio update").
3. **Soft gates (v33)** — Correct architecture. Data collection approach is sound.
4. **price_action_score** — Best predictor (r=0.262). 72% coverage.
5. **activity_mult** — Second best predictor (r=0.146). 72% coverage.
6. **DexScreener/RugCheck enrichment** — High coverage (95%+/90%+), free.
7. **Helius enrichment** — 25.7% coverage, whale_count is top signal (r=0.468 from backtest).
8. **OHLCV fallback chain** — GeckoTerminal → DexPaprika → Birdeye works well.
9. **GH Actions infrastructure** — Runs reliably, caching works.
10. **Frontend** — Dashboard, tuning lab, KOL leaderboard all functional.

---

## PRIORITY FIX ORDER

### P0 — DATA RECOVERY ✅ DONE (Feb 17, 2026)

1. ✅ **Fix phantom labels:** Reset 16,418 phantom labels to NULL across all horizons
2. ✅ **Fix `_mark_ohlcv_failed()` + `_mark_no_price()` + `_mark_dead_pool()`:** All 3 functions now leave did_2x as NULL (unknown) instead of False
3. ✅ **Add API keys to GH Actions:** `BUBBLEMAPS_API_KEY` + `JUPITER_API_KEY` added to scrape.yml
4. ✅ **Fix kol_call_outcomes:** DexPaprika fallback in `_get_pool_address()` when GeckoTerminal disabled. Phase C now computes `did_2x` + `max_return`.

**Post-fix DB state:** 2,448 real 24h labels, 270 hits (11.0% hit rate). 0 phantom labels.

### P1 — GATE REFORM ✅ DONE (Feb 17, 2026)

5. ✅ **Removed anti-predictive gates:** high_risk_score (was 0.6x), wash_trading (was 0.5x), low_liquidity (was 0.5x). Data kept as ML features.
6. ✅ **Kept valid gates:** freeze_authority (0.05x), mint_authority (0.05x), top10_concentration (0.7x), low_holders (0.7x)

### P2 — LABELING QUALITY ✅ DONE (Feb 17, 2026)

7. ✅ **Dedup in backtest:** `_top1_hit_rate()` now uses first-appearance dedup (was counting same token 20-30x)
8. ✅ **Re-run auto_backtest on clean data:** N=88 deduped 24h labels, overall 13.6% hit rate

**CRITICAL DISCOVERY from clean data backtest:**
- **PA correlation COLLAPSED**: r=+0.262 (old poisoned) → r=-0.010 (clean). PA was an artifact of phantom labels!
- **Top real predictors**: whale_count (+0.47), kol_arrival_rate (+0.42), mention_velocity (+0.41), short_term_heat (+0.36)
- **Score 70+ hit rate: 4.3%** (worse than random 15.8%). Score 30-49: 28.6% (sweet spot)
- **Conviction still anti-predictive**: r=-0.17

### P3 — SCORING REFORM ✅ DONE (Feb 17, 2026)

9. ✅ **New weights: 35/0/40/25** (consensus/conviction/breadth/PA)
   - Consensus 0→35: captures kol_arrival_rate (+0.42), mention_velocity (+0.41)
   - Breadth 5→40: captures whale_count (+0.47, #1 predictor)
   - PA 85→25: collapsed from r=+0.26 to r=-0.01 after phantom cleanup
   - Conviction stays 0: anti-predictive (r=-0.17)
10. ✅ Updated scoring_config table + pipeline.py + auto_backtest.py defaults

### P4 — ACCUMULATE & ITERATE (ongoing)

11. **Run scraper for 2+ weeks** with all API keys working to fill enrichment data
12. **Re-run ML training** on clean data with Bubblemaps/Jupiter features
13. **Dynamize top 10 hardcoded sets** (290 constants → scoring_config)
14. **When N>100 labeled:** auto_backtest weight optimizer will run and further refine weights

---

## DIAGNOSTIC NUMBERS (CORRECTED)

| Metric | Reported | True (after audit) |
|--------|----------|--------------------|
| Total snapshots | 12,917 scored | 12,917 scored |
| Unique tokens | "616" | **247 scored** (rest are gated) |
| Labeled 24h | 5,286 | **2,267 with real prices** |
| Overall 2x hit rate | 4.6% | **16.7% (deduped, real)** |
| Score correlation | "strong" | **0.184 (weak)** |
| Top predictor | "score" | **whale_count (r=0.468)** |
| Bubblemaps coverage | "cached" | **0.0%** |
| Jupiter coverage | "cached" | **0.8%** |
| kol_call_outcomes useful | 490 | **6 (1.2%)** |
| Backtest reliability | "STRONG" | **Poisoned by phantom labels** |
| ML quality | "p@5=1.0" | **N=14, statistically meaningless** |
