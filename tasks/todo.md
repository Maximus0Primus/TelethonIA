# Pipeline Status — Updated Apr 17, 2026 (v137)

## Current state (live config)

**Live (50/50):** `BE25_TP80_SL30` + `FAST_TP100_SL20` (both jupiter/120s). Position ~$1.70/trade, max 3 open. Swapped from DTRAIL3+DTRAIL10 after v136 realistic sim + ground-truth `--from-trades` rerank both confirmed DTRAIL_ACT10_SL70 unprofitable (-$33/trade live, -$5/trade paper avg).

**Paper Telegram (7 strats):**
- DTRAIL5_ACT10_SL60 (jupiter/120s) — kept (best DTRAIL post-v136)
- DTRAIL10_ACT5_SL50 (jupiter/120s) — added (best-of-DTRAIL ground truth)
- BE25_TP80_SL30 (ema_fast/120s, historical balance $1217.65)
- BE25_TP80_SL30_DS (ds/120s) — A/B variant
- FAST_TP100_SL20 (ds/120s)
- TP50_SL15 (jupiter/60s) — added (#1 ground truth $853 5d)
- BE15_TP100_SL50 (jupiter/120s) — added (#2 ground truth $814 5d)

All paper positions fixed $50 (kelly × bankroll capped at max_position_usd=50).

## v137 — Cadence Fix + Strategy Swap (Apr 17 14:30 UTC) ✅

**Root cause of DTRAIL sim overestimation:**
1. Sim's tick-driven subsample picked "next tick after gap" → real paper uses "latest cached tick before poll" (paper_trader._jupiter_prices_cache lookup semantics)
2. Sim used fixed 10bps slip; real uses dynamic 30-250bps via `_dynamic_sell_slip_factor` depending on exit type (already routed through `_evaluate_trade_exit`, was correct)
3. `_log_price_ticks` throttled paper-only tokens to 60s while real cache updates every 30s — sim missed half the prices real paper used

**Bias measured (sim before → sim after, vs real paper PnL):**
- DTRAIL3_ACT10_SL70: +19.80% → +4.24% (-78%)
- DTRAIL10_ACT10_SL70: +17.22% → +3.37% (-80%)
- BE25_TP80_SL30: +6.57% → +4.42% (-33%)
- FAST_TP100_SL20: +10.68% → +4.19% (-61%)

**Code changes (everything in sim.py — no standalone scripts):**
- [x] `paper_trader.py:369` throttle 60s → 30s for paper-only tokens
- [x] `sim.py:_replay_trade_orchestrated` — replaced tick-driven subsample with deterministic 30s grid + cache look-back via new `_latest_tick_at_or_before` helper
- [x] `sim.py:LOOP_SEC=30` constant aligned with paper_trader.unified_check_loop
- [x] `scripts/_apply_v137_swap.py` — atomic DB swap (one-off, kept for audit)

**Sweep workflow going forward:** `python scraper/sim.py --from-ticks --since YYYY-MM-DD --top 30` (no more standalone sweep scripts — all logic lives in sim.py).

**⚠️ Residual sim bias is STRUCTURAL** — investigated v137.1 (filtering jupiter stream to paper-logged ticks only): made MAE WORSE (17%→30%) because sparser ticks meant look-back picked up stale prices. Reverted. The +4pp bias on trail-heavy strategies comes from `price_ticks` undersampling (60s throttle pre-v137 vs 30s real fetch cadence) — sim can only see logged ticks, not what real paper's cache actually held between logs. The throttle 60→30s patch (now deployed) will halve this bias for new trades over the next 24-48h. For HISTORICAL data the bias is locked.

**Sim alignment status per family:**
- FIXED, BE: sim bias ~+4%, ranking reliable ✓
- FAST: sim bias ~+4%, ranking reliable ✓
- DTRAIL/TRAIL: sim bias +4% with HIGH MAE (~20% per-trade) → sim says break-even when real loses → **NEVER decide on these from `--from-ticks`, always cross-check with `--from-trades`**
- DIP: untested, treat like DTRAIL (trail-heavy mechanics)

**Decision rule going forward:** for any strategy swap or live deployment, the ground-truth `--from-trades` is the source of truth. `--from-ticks` is exploratory only.

**DB swap applied via `_apply_v137_swap.py --apply`:**
- paper.active_strategies: removed DTRAIL3_ACT10/10_ACT10/3_ACT20, added TP50_SL15/BE15_TP100/DTRAIL10_ACT5
- live_trading.allocations: DTRAIL3+10 → BE25+FAST (50/50)
- hybrid_strategy.allocations + strategy_overrides synced
- removed strats moved to deprecated_strategies

## v133-D — Hybrid Sell Pollution Fix (Apr 16 18:17 UTC) ✅

Hybrid (FAST + DTRAIL same token) shared one ATA. `execute_sell(addr)` without `amount_tokens` drained full wallet → winner 2× inflated pnl, loser phantom −100%.

- [x] FIX: `live_trader.py:1246` passes `amount_tokens=buy_output_tokens`
- [x] FIX reconciler: `_find_sibling_exit` + `_reconcile_close_payload` uses sibling SOL-per-token instead of phantom −100%
- [x] Cleanup: `scripts/cleanup_hybrid_sell_pollution.py` corrected 30 rows (net delta −$2.99)

Validated on $INCOME pair post-deploy (ratio 1.000 both legs).

## v136 — Strategy Swap (Apr 16 21:30 UTC) ✅

All via DB (no code change). Based on v135 full sweep: 224 strats × 48 configs × 59 post-v132 tokens = 10,752 combos.

**Swapped out:** FAST_TP50_SL30, FAST_TP80_SL25, DTRAIL10_ACT15_SL70, DTRAIL3_ACT5_SL60, 4× DIP30 variants.

**Cleanup:** removed dead `strategy_multipliers` from rt_trade_config (never read, was Optuna-only output).

### Watch list (Apr 17-21)
- [ ] Live DTRAIL3_ACT10 + DTRAIL10_ACT10 generate >$1 avg on N≥10 trades each
- [ ] **BE25 A/B resolution**: `BE25_TP80_SL30` (ema_fast, $1217.65) vs `BE25_TP80_SL30_DS` (ds/raw, $1000). After N≥30 trades each, promote winner and delete loser.
- [ ] No residual opens on removed strats (config cache refreshed <60s)
- [ ] **Apr 17-21 sim vs real check**: real daily pnl within ±40% of theoretical → sim aligned; else → sim overfit, re-sweep with larger N

### Watch list (Apr 17-21)
- [ ] Live BE25 + FAST generate >$0.50 avg/trade on N≥10 trades each (vs v136 sim +8.81% / +8.90% best avg)
- [ ] **BE25 A/B resolution**: ema_fast vs ds variant (existing test, continues)
- [ ] Verify cache cadence fix: post-v137 throttle change should yield ~2x more `source='fast'` ticks per token
- [ ] If real bias persists >10pp on BE/FAST → investigate residual (KOL filter / dedup not modeled in sim)

### Real ground-truth ranking (sim/sim_v136/realistic/from-trades cross-validation)
**5-day from-trades top 10 (real PnL):**
1. TP50_SL15 (+5.5% / 33% WR / $853)
2. BE15_TP100_SL50 (+6.0% / 26% / $814)
3. TP80_SL30 (+5.0% / 35% / $732)
4-9. BE15_TP70_SL50, TP90/70/30, BE20_TP100, FAST_TP100_SL50, FAST_TP70_SL50 ($600-720)
25. BE25_TP80_SL30 (currently active) — +4.2% / 34% / $572

**v137 sim --from-ticks top winners** (re-run any time via `sim.py --from-ticks --since 2026-04-13`):
- BE: BE25_TP80_SL30 — +7.3% / 30% WR / $753 (5d, N=53)
- FAST: FAST_TP50_SL30 — -0.1% / 44% WR / $494 (deprecated, residual main trades)
- DTRAIL: DTRAIL5_ACT10_SL50 in v136 cross-product (kelly 13.13, SL50 not SL70)
- TP: TP30_SL10 in v136 cross-product (kelly 13.92)

## Active Exploration

### Latence live trade (Plan A — msg→buy delay)

Contexte: buys live entrent avec 20-60s de retard sur le call KOL → front-running. Voir session 2026-04-14.

**Fait:**
- [x] B — fix `_rt_price_at_message` (bug clé de dict)
- [x] A.1 — instrumentation `RT timing:` + `LIVE LATENCY:` logs
- [x] A.3 — cache SOL price 5s + wallet balance 10s

**À faire:**
- [ ] **A.2** — analyser logs `LIVE LATENCY:` après N≥10 buys (msg→ds / ds→pre_buy / buy_exec) → localiser bottleneck
- [ ] A.2 — cibler 2-3 bloqueurs. Hypothèses : DS fetch sync, enrichment synchrone pré-buy, checks séquentiels open_live_trade

Command: `ssh vps "journalctl -u kol-scraper | grep 'LIVE LATENCY'"` → p50/p95.

### Deferred — tp_touched exit mode

Idea: si `high_price_seen >= tp_price` pendant horizon mais exit fires sur timeout/SL/trail, rétroactivement exit à `tp_price` (paper) ou tick-fire execute_sell (live). Analyse post-v133-D : **+$1.55/week uplift live** (5× sous threshold). 2 "missed peaks" détectés = artefacts pollution, pas réel miss. Max peak/tp ratio FAST_TP50 timeout/SL = 0.95 → TP jamais touché. Paper-only casserait alignement sim/paper/live.

**Re-evaluate when ANY:**
- [ ] High-TP strategy en live (TP80+) — FAST_TP100_SL20 paper montre +$47/14d uplift potentiel
- [ ] Live volume ×3 (actuel ~20/sem → seuil ~60/sem)
- [ ] Jupiter Trigger V2 keepers toujours 0 fill après 7j → self-built fast-path regains value

**Build path (si re-trigger):**
1. `_check_tick_tp_cross()` in paper_trader (walks price_ticks since last poll)
2. `_evaluate_trade_exit` override to tp_hit if cross detected
3. `live_trader.check_live_trades` subscribe price_ticks or 5s-poll between regular polls
4. Sim same helper on `--from-ticks` → alignment preserved
5. Re-run `verify_sim_live_alignment.py` <3pp

## Housekeeping

### Repo hygiene
- [ ] Commit `007db6b` a pushé 2400+ fichiers cache. Options : revert + force-push, ou ajouter au `.gitignore` et laisser.

### Data quality (/check-data Apr 14)
- [ ] Jupiter LDS sous-fill (35% vs seuil 70%) — quotas API / silent errors
- [ ] Holders sous-fill (27%) — idem
- [ ] CA resolution 71.9% (sous seuil 75%) — resolver messages récents
- [ ] Backlog labels 24h = 774, 7d = 2346 — outcome_tracker cadence

### Bugs connus non-fixés (non urgents)
- [ ] Reconcile bypass bankroll — `reconcile_positions` auto-close saute `_rt_update_bankroll`. Drift silencieux sur `rt_bankroll.total_pnl`. Impact cosmétique, pas runtime.
- [ ] DIP30 entry gate cassé — 0 trades post-v132. Désactivé de paper en v136. Si on veut le réactiver plus tard : investiguer `STRATEGY_FILTERS["DIP30_..."]`.

### Sim ↔ Live/Paper coherence — reference

**Bias hierarchy (stable):**
- `--from-trades` = ground truth (perfect coherence)
- Sim `--from-ticks jupiter` = coherent with prod tracking, entry bias ~3-5% vs Ultra RFQ
- Sim `--from-ticks dexscreener` = correct proxy for Ultra entry, wrong for tracking
- Sim OHLCV = 5-15% bias on trails (candles ≠ real ticks)

**Per-family tool:**
- FAST family → OHLCV backtest acceptable
- DTRAIL / hybrid → tick-replay required, OHLCV lies
- Tight trails <10% → tick-replay only (OHLCV overestimates)

**Thresholds:**
- Paper/live per-pair divergence: realistic <5pp, >10pp = bug signal
- Live/sim edge: `|live − sim| < 50% of expected edge`
- Entry calibration Ultra-vs-PriceAPI : recheck Apr 27 (~2w post-v130 `entry_source=ultra`)

## Still Pending (low priority)

- [ ] **Birdeye top N expansion** — `BIRDEYE_TOP_N = 20` means whale_new_entries NULL for 80%+. Increase to 50+ (costs CUs)
- [ ] **PA computation gated** — PA weight=0% mais OHLCV fetched. Gate on `SCORING_PARAMS["price_action"] > 0`
- [ ] **gate_mult dead compute** — RugCheck/wash-trading executed despite result always 1.0
- [ ] **v53 features** — holder_turnover, kol_cooccurrence computed mais excluded from ML (<6% fill)

## Architecture summary

**Scoring:** 40.5/13.5/40.5/5.4 (consensus/conviction/breadth/PA), 16-multiplier chain, Optuna ~48 params walk-forward.

**Trading:**
- Paper slippage: dynamic from liquidity_depth_score (base 100/200bps buy/sell)
- Live slippage: Jupiter Ultra RFQ ~10bps real
- Position reconciliation: sibling-aware (v133-D)
- Loss limits: 0.5 SOL/day

**Alerting:** ML disabled, RT listener down (uncapped), GH Actions failures, write-ahead log, daily summary 8am UTC.
