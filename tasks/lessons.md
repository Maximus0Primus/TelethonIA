# Lessons Learned

## 2026-02-14: size_mult must scale aggressively for large-cap memecoins
**Mistake:** `size_mult` capped at 0.7x for ALL tokens >$20M. Pippin at $718M got only -16% penalty (0.84x with freshness boost) — scored 48/100 when it should be ~14. A $700M memecoin needs $700M NEW capital to 2x — near impossible.
**Fix:** Progressive tiers: $50M→0.70x, $200M→0.50x, $500M→0.35x, >$500M→0.25x. No freshness boost for >$50M. Floor lowered from 0.6 to 0.25.
**Rule:** When scoring tokens for 2x potential, the market cap penalty must reflect the actual capital required. $5M and $700M cannot receive the same penalty. Always ask: "how much new money must flow in for this to 2x?"

## 2026-02-14: entry_premium needs market-cap magnitude fallback
**Mistake:** `entry_premium` returned neutral (1.0) for $718M Pippin because: (a) KOLs don't write "at 700M mcap" for established tokens, (b) OHLCV candles only cover 24h so recent calls show ~same price. Both sources fail → neutral → no penalty.
**Fix:** Added third fallback: when mcap > $50M and both primary sources fail, compute implied premium from mcap magnitude (mcap / $1M typical launch). $718M → implied 718x → mult 0.35x.
**Rule:** Any signal with neutral fallback (1.0) must be audited for cases where "no data" actually means "obviously bad". A $700M token with no entry data is NOT neutral — it's already pumped 700x from launch.

## 2026-02-14: Lifecycle "boom" bonus must consider market cap
**Mistake:** Pippin ($718M, +25% 24h, 4 KOLs) classified as "boom" → got 1.1x BONUS. The lifecycle classifier didn't consider mcap, so a $700M established token got the same "boom" bonus as a $500K micro-cap.
**Fix:** Boom phase with mcap > $50M → 0.85x penalty instead of 1.1x bonus.
**Rule:** Lifecycle phases (boom/euphoria/displacement) must factor in token size. "Boom" for a micro-cap means explosive growth potential; for a mega-cap it means the train already left.

## 2026-02-14: XGBoost reg:squaredlogerror crashes on negative targets
**Mistake:** `reg:squaredlogerror` (RMSLE) requires all labels > -1. Our `log_return = np.log1p(max_return - 1)` produces negative values when tokens lose value (max_return < 1). Training crashed on first trial.
**Fix:** Changed to `reg:squarederror` which works with any real-valued target.
**Rule:** When using log-error objectives, verify target range first. For financial return prediction, `reg:squarederror` is safer since returns can be negative.

## 2026-02-14: supabase-py defaults to 1000 row limit
**Mistake:** `load_labeled_data()` fetched all rows with `.select("*").execute()` but supabase-py caps at 1000. With 1,677 1h labels, 677 were silently dropped. The model trained on 60% of available data.
**Fix:** Added pagination with `.range(offset, offset + page_size - 1)` in a while loop.
**Rule:** Always paginate supabase-py queries when data may exceed 1000 rows. Test with `len(result.data) == page_size` to detect truncation.

## 2026-02-14: p@5 is noisy with small test sets
**Mistake:** 12h regression model had Spearman=0.626 (excellent!) but p@5=0.000 because the walk-forward test split (last 30%) happened to have no 2x tokens in the top-5 by random chance. Only 29 winners out of 471 total (6.2%).
**Lesson:** With <500 labels and <10% positive rate, precision@5 on a single temporal split is essentially a coin flip. Need 1000+ labels for stable p@5 estimates. Consider using cross-validation with multiple temporal splits, or lowering the quality gate when Spearman is strong.

## 2026-02-14: Optimizer must explore from multiple starting points
**Mistake:** Rewrote autoOptimize to start cumulative sweep from user's current config only. If the config was already decent, the greedy sweep found no improvements (each phase compared against accumulated best using strict `>`). The old code started `bestHR = 0` per component so it always found something.
**Fix:** Dual-track optimization — run `cumulativeSweep()` from BOTH `baseConfig` AND `DEFAULT_CONFIG`, take the better result, then run random exploration on top. Feature impacts computed by reverting each feature from bestConfig to baseConfig (shows "what would we lose by going back to your settings").
**Rule:** Greedy optimizers stuck in local optima is a known problem. Always try multiple starting points. For feature impact, "revert and measure" is more meaningful than "marginal improvement during sweep".

## 2026-02-14: Optimizer must optimize for the metric the user sees
**Mistake:** Optimizer maximized `top10_hit_rate` but the user looks at `top5_hit_rate`. The optimizer found configs where 2x tokens land in positions 6-10 (great top10) but not 1-5 (terrible top5). User sees top5 drop from 40% to 20% after applying the "best" config.
**Fix:** Changed optimization target to combined score: `top5 * 0.5 + top10 * 0.3 + top20 * 0.2`. Top5 dominates, top10/top20 add granularity for tie-breaking. Button shows actual top5 hit rate (not internal score).
**Rule:** Always optimize for the metric that's most visible to the user. If displaying top5/top10/top20, the optimization target must prioritize top5. Test by verifying that the displayed metric doesn't DROP after applying the optimized config.

## 2026-02-13: Check ALL imports before committing
**Mistake:** Committed `safe_scraper.py` with a top-level `from auto_backtest import run_auto_backtest` but `auto_backtest.py` was untracked. GitHub Actions crashed immediately with `ModuleNotFoundError`.
**Fix:** Moved to lazy import inside the try/except block where it's actually used.
**Rule:** Before committing, verify that every import in modified files either (a) exists in the repo, (b) is in requirements.txt, or (c) is guarded by try/except. Run `python -c "import <module>"` as a smoke test when uncertain.

## 2026-02-14: ML training data must be deduplicated to one snapshot per token
**Mistake:** `train_model.py` loaded ALL snapshots (470 for 12h horizon) but many were the same token appearing 3-7x across cycles. Same token = same outcome = correlated observations. 470 snapshots were actually only 69 unique tokens. The 6.2% winner rate was really 11.6%. Feature correlations were inflated by duplicates.
**Fix:** Added `deduplicate_snapshots()` — sorts by `snapshot_at`, keeps first snapshot per `token_address` (fallback `symbol`), filters zombies >48h. Called at both `auto_train()` and CLI entry points.
**Rule:** ML training data MUST be one observation per token. Multiple snapshots of the same token are not independent samples — they have identical outcomes and correlated features. Always deduplicate before computing correlations or training.

## 2026-02-14: SOL price leaks persist across ALL horizons
**Mistake:** outcome_tracker.py had a known SOL price leak bug (OHLCV APIs returning SOL price ~$78-87 instead of token price). A sanity check was added, but only AFTER $YEE, $ZEREBRO, $LUCE were already labeled. Found 17 corrupted rows: 1 in 12h, 11 in 6h, 5 in 1h. Max prices of $79-87 for micro-cap tokens are SOL's price.
**Fix:** Cleaned all corrupted labels to NULL for re-labeling. Scanned ALL `max_price_*` columns across all horizons (not just the one where the bug was first found).
**Rule:** When cleaning data corruption, always scan ALL related columns/horizons. A bug that corrupted 12h labels almost certainly also corrupted 6h and 1h labels if they existed at the time.

## 2026-02-14: Feature correlations computed on duplicated data are unreliable
**Mistake:** price_action_score showed +0.252 correlation with did_2x_12h on raw data (470 snapshots). After deduplication (69 unique tokens), it collapsed to +0.041 — practically noise. Yet it had 55% weight in scoring. The duplicate tokens amplified PA's apparent correlation because similar tokens had similar PA scores.
**Fix:** All correlation analysis must be done on deduplicated data. Post-dedup rankings: risk_count +0.335 (#1), entry_premium -0.180, age -0.149, mentions -0.131, PA +0.041 (noise).
**Rule:** Never trust feature correlations computed on data with duplicate observations. Deduplication can completely change which features appear predictive. This is the most dangerous form of data leakage — it doesn't just inflate metrics, it points you at the wrong features.

## 2026-02-14: Tuning Lab backtester must use per-cycle evaluation, not global ranking
**Mistake:** Tuning Lab showed 80% top5 hit rate at 12h — wildly inflated. Root causes: (1) No deduplication — same winning token ($WORDSLOP with 6 snapshots) filled multiple top-5 slots across cycles. (2) `consensus_val IS NOT NULL` filter in backtest API dropped 9 out of 13 unique winners (from before component values were saved). (3) Global ranking treated all snapshots as one pool, ignoring that the scoring system produces rankings per 15-min cycle.
**Fix:** Three changes: (a) Backtest API removed `consensus_val` filter — rescorer handles nulls with defaults. (b) `backtester.ts` rewrote to per-cycle evaluation: group snapshots into 15-min cycles, rescore each cycle independently, compute hit rates per cycle, average across valid cycles (5+ tokens). (c) Global stats (base_rate, avg_score, separation) still use dedup for honest denominators.
**Rule:** A real-time ranking system must be backtested PER DECISION POINT (cycle), not as one big pool. Per-pool evaluation lets duplicate winners inflate top-K and hides that the system makes independent decisions every 15 minutes. Always ask: "at each decision point, did we rank the winner in the top K?"

## 2026-02-15: NEVER dedup by symbol — always use token_address
**Mistake:** Across 7 files, deduplication used `symbol` (ticker like $LUNA) instead of `token_address` (contract address). The same ticker can map to 3+ different contracts — $LUNA had 3, $ROCK had 3, $WIF had 3. This caused: (1) auto_backtest.py merged different tokens' outcomes (12 functions x 13 instances), (2) kol_scorer.py collapsed 45 real token-KOL pairs into 32, (3) backtester.ts inflated hit rates by losing unique tokens, (4) snapshots route dropped tokens from API responses.
**Fix:** Replaced all `drop_duplicates(subset=["symbol"])` with `subset=["token_address"]`, all `seen_symbols` sets with address-keyed sets, all `DISTINCT ON (kol, symbol)` with `DISTINCT ON (kol, token_address)` in RPC. Always fall back to symbol when token_address is null.
**Rule:** NEVER use `symbol` as a dedup key. Symbols are display names, NOT unique identifiers. Always use `token_address` for deduplication, grouping, and lookups. When writing new code that touches token identity, grep for "symbol" in dedup/groupby/set contexts and flag it.

## 2026-02-14: unique_kols must be materialized in snapshots, not left NULL
**Mistake:** 41% of token_snapshots had `unique_kols = NULL` because the column was added after many snapshots were already created. `top_kols` JSON was always populated but `unique_kols` numeric was not extracted from it.
**Fix:** Backfilled 4,201 rows from `top_kols` JSON: `json_array_length(top_kols::json)`. Ensured push_to_supabase always writes unique_kols.
**Rule:** When adding a new computed column to snapshots, always backfill from existing data. NULL features are invisible to ML and break any feature that depends on them.
