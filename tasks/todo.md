# Pipeline Status — Updated Apr 14, 2026 (v133)

## v133 — Paper/Live Divergence Fixes (Apr 14 night)

Checkpoint 24h post-v132 deploy: 26 RT tokens → paper opened 26 FAST + 26 DTRAIL10, live opened 21 FAST + **1 DTRAIL10** (broken). Divergence paper/live on FAST: avg |diff| = 10.8pp (target <2%), 3 outliers >37pp driven by paper SL firing on DS noise dips while live Jupiter quote stayed above.

### Fixes deployed
- [x] **FIX 133-A**: dedup in `live_trader.open_live_trade` was matching any `rt_live` row for the CA, blocking the 2nd strategy in the hybrid allocation loop. Scoped to `same strategy`. Effect: each RT token now opens BOTH FAST and DTRAIL10 at $1.50 each ($3 total).
- [x] **FIX 133-B**: `_evaluate_trade_exit` SL-check now uses `current_price` (exit_ref = Jupiter) instead of `eval_price` (decision = DS) in hybrid mode. TP/trail still use decision_price (DS catches pumps faster). Mirrors live execution semantics — paper and live now SL at same Jupiter quote.
- [x] **FIX 133-C**: sell slippage monitoring — prior formula `(1 - usd_received / (pos_usd*(1+pnl_pct)))` was circular (pnl_pct derived from usd_received → always 0). Now `(exit_price / current_price - 1) * 10000` measures actual fill vs spot-at-trigger. Also populates `slippage_actual_bps` (round-trip buy+sell) which was dead column.

### Watch list (next 24-48h — recheck Apr 16 evening)
- [ ] Confirm live opens DTRAIL10 alongside FAST (ratio ~1:1 post-v133-A)
- [ ] Divergence FAST paper/live drops avg |diff| < 5pp (target was 2pp, realistic after latency+slippage = 3-4pp)
- [ ] No new SL outliers >10pp on hybrid-mode strategies
- [ ] `_strategy_orchestration` never returns None,None silently (defaults fallback verified OK)
- [ ] DIP30 still at 0 trades — investigate gate separately if still blocked after 48h
- [ ] Slippage monitoring: `sell_slippage_bps` and `slippage_actual_bps` now populated (v133-C) — check distribution after first ~10 sells
- [ ] **Sim↔Live DTRAIL10 rigorous verify** — pre-v133-A: only 7 DTRAIL10 live trades since Apr 13 (N too small + contaminated by dedup bug blocking 2nd allocation). Wait for N ≥ 15 clean post-v133 DTRAIL10 live trades, then rerun `python scripts/verify_sim_live_alignment.py --strategy DTRAIL10_ACT15_SL70 --from-live-config --priority-fee-sol <val>`. Success criterion: sim matches live <3pp structural (consistent with $KUSYA 2.1pp / $GOONALD 0.9pp reference pairs).

### Known residual
- TP outlier (Cw9V +37pp) = real Ultra fill luck, not a bug. Accept as execution variance.
- Shadow-sync (paper reuses live execution_price) deferred — keeps paper independent as a benchmark to detect live-only bugs.

## v133-D — Hybrid Sell Pollution Fix (Apr 16, 18:17 UTC)

Hybrid allocation (FAST + DTRAIL on same token) shared one ATA. `execute_sell(addr)` without `amount_tokens` drained full wallet balance → winner leg showed ~2× inflated pnl, loser leg auto-reconciled at phantom −100%. 16 trades polluted over 30h. Validated on $INCOME pair post-deploy (ratio 1.000 on both legs, independent pnl).

### Fixes deployed
- [x] **FIX 133-D**: `live_trader.py:1246` passes `amount_tokens=buy_output_tokens` → each leg sells only its own share. Legacy rows missing the column fall back to full-balance.
- [x] **FIX 133-D reconciler**: new `_find_sibling_exit` + `_reconcile_close_payload`. If a sibling exit explains the missing balance, recompute pnl from its realized SOL-per-token. No sibling → pnl=0 rather than phantom −100%.
- [x] **Cleanup script**: `scripts/cleanup_hybrid_sell_pollution.py` (dry-run then `--apply`). Corrected 30 rows on 14d window. Net pnl_usd delta: −$2.99 (on-chain SOL unchanged — unwinds double-counting between sibling pairs).

### Deferred — C (tp_touched) — revisit conditions

Idea: if `high_price_seen >= tp_price` during trade horizon but exit fires on timeout/SL/trail, retroactively exit at `tp_price` (paper) or fire execute_sell on tick-crossing (live). Analysis post-v133-D shows only **+$1.55/week uplift on live** (5× below the $10/week defer threshold); 2 "missed peaks" detected were reconciled-artifact pollution, not real missed TPs. Max peak/tp ratio for FAST_TP50 timeout/sl_hit trades = 0.95 → peak genuinely never reached TP. Paper-only implementation would break paper/live/sim alignment (paper surestimates live). Sim + live must move together.

**Re-evaluate when ANY of these hits:**
- [ ] **High-TP strategy promoted live** (TP80/TP100/TP150) — paper shows +$47/14d uplift on FAST_TP100_SL20 because high TPs get touched briefly then retrace more often. Once that strategy is live, the tick-based fast-path becomes quantifiable in live.
- [ ] **Live volume grows 3× or more** — at current ~20 live trades/week, $1.55 noise is within error bars. At ~60/week, re-run the analysis; edge may clarify.
- [ ] **Post-v133-D window ≥ 20 clean FAST trades** (expected ~Apr 20) — current N=2 is too small to confirm the zero-missed-peak reading. If clean window still shows <2 missed peaks per 20 trades, close definitively.
- [ ] **Jupiter Trigger V2 keeper fills still at 0** after another 7d — memory notes "zero keeper fills yet" as of v131. If trigger orders remain unreliable, self-built tick fast-path regains value.

**If re-triggered, build path:**
1. Add `_check_tick_tp_cross()` to `paper_trader.py` (walks `price_ticks` since last poll, returns True if any tick ≥ tp_price).
2. `_evaluate_trade_exit` calls it before timeout/SL branches → if True, override to `tp_hit` at `tp_price`.
3. `live_trader.check_live_trades` subscribes to price_ticks stream (or polls it at 5s between regular polls) → fires `execute_sell` immediately on cross, bypassing next-poll wait.
4. Sim uses the same `_check_tick_tp_cross` on `--from-ticks` backtests → alignment preserved across paper/live/sim.
5. Re-run `scripts/verify_sim_live_alignment.py` after deploy, must stay <3pp divergence.

## Active Exploration

### Latence live trade (Plan A — msg→buy delay)

Contexte: buys live entrent avec 20-60s de retard sur le call KOL → front-running par bots concurrents → "slippage" apparent 20-50% (ex: $BBC call @ 10k → entré @ 14k). Voir analyse session 2026-04-14.

**Fait:**
- [x] **B** — fix `_rt_price_at_message` (bug clé de dict, toujours NULL)
- [x] **A.1** — instrumentation latence : logs `RT timing:` + `LIVE LATENCY:` (msg→ds, ds→pre_buy, buy_exec)
- [x] **A.3** — cache SOL price 5s + wallet balance 10s dans live_trader.py
- [x] **A.3** — vérifié qu'il n'y a pas de double fetch DexScreener (faux blocker)

**À faire:**
- [ ] **A.2** — analyser les logs `LIVE LATENCY:` après 24h (≥5-10 buys live) pour localiser le vrai bottleneck parmi :
  - `msg→ds` : temps entre message Telegram et fin fetch DexScreener (API externe)
  - `ds→pre_buy` : enrichment + scoring + sizing + `_rt_open_trades` → `open_live_trade` → pre-`execute_buy`
  - `buy_exec` : temps d'exécution Jupiter Ultra (signature + envoi)
- [ ] **A.2** — cibler 2-3 gros bloqueurs universels (pas de shortcut S-tier). Hypothèses :
  - `_fetch_dexscreener_by_address` (safe_scraper.py:1655) sync dans executor — voir si async HTTP client aide
  - enrichment synchrone avant buy dans `_rt_open_trades` — paralléliser ou déplacer post-buy
  - checks séquentiels dans `open_live_trade` (max_open, dedup_cooldown, min_sol_reserve) — combinable

Données à récolter: `journalctl -u kol-scraper | grep "LIVE LATENCY"` → moyenne/p50/p95 sur msg→ds, ds→pre_buy, buy_exec.

### Stratégies post-v132 (Apr 13+, régime actuel)

**Découvertes clés du sweep 2026-04-14:**
- **+11pp de WR depuis Apr 13** — attribué majoritairement aux modifs code (v130-v132: source cohérence, polling per-strat, hybrid) — pas à un "régime marché" pur. Tendance graduelle avant v132 ~+2pp/sem.
- Kelly + MC extrapolés à N<50 = overfit garanti. Traiter toute proj avec CI±5pp sur WR.
- **Le sweep sur N=133 mélangeant shadows pré/post-v132 a produit des conclusions fausses** (notamment "désactiver DTRAIL10 et DIP30"). Post-v132 uniquement, ces strats redeviennent profitables. Ne pas désactiver avant plus de données.

**Top strats post-Apr 13 (N=30/strat, shadows inclus) — profil consistent:**
| Rank | Strategy | avg | WR | med |
|---|---|---|---|---|
| 5 | **FAST_TP70_SL50** | +10.14% | **56.7%** | +8.97% |
| 6 | TP70_SL70 | +10.06% | 50.0% | +10.84% |
| 12 | FAST_TP100_SL50 | +8.55% | 53.3% | +8.02% |
| **16** | **FAST_TP50_SL30 (LIVE actuel)** | **+8.42%** | 53.3% | +6.86% |

Home-run profile (TP70/80/90_SL30): avg élevée mais médiane −20/−30% = > 50% losers. Pas recommandés pour live.

**Stratégies synthétiques v134 sweep (top candidates consistent, WR ≥53%, médiane positive):**
| Strat | Config | avg | WR | med | Note |
|---|---|---|---|---|---|
| **FAST_TP80_SL25** | Jup/120s/dual_confirm | +13.23% | 53.3% | +7.79% | **+3pp sur FAST_TP70_SL50** |
| BE25_TP80_SL30 | Jup/120s/ema_fast | +13.40% | 53.3% | +7.79% | BE ratchet safety |
| FAST_TP60_SL20 | Jup/60s/ema_slow | +11.69% | 53.3% | +3.10% | Ultra tight |
| BE15_TP70_SL30 | Jup/120s/ema_slow | +11.52% | 53.3% | +7.79% | Conservative |

Home-run (avg haute, médiane ~0): FAST_TP100_SL20 (DS/120s/raw) +16.36%, WR 50%, Kelly 31.6% — **piège à N=30**, drawdown long.

**Plan pragmatique:**
- [ ] **A/B paper FAST_TP50_SL30 (live actuel) vs FAST_TP70_SL50** — 2 semaines, allocation 50/50. FAST_TP70_SL50 a WR 56.7% vs 53.3% et +1.72pp d'edge sur N=30.
- [ ] **Ajouter FAST_TP80_SL25 à paper_trader.STRATEGIES** — candidate #1 consistent. Laisser tourner 1 semaine paper pour confirmer le +3pp vs FAST_TP70_SL50.
- [ ] **Ajouter BE25_TP80_SL30** idem — test du BE ratchet.
- [ ] Re-run sweep smoothing sur **top-10 strats consistent** (pas juste les 4 actuelles) pour voir si le gain smoothing est stable inter-strats.
- [ ] **Ne pas désactiver DTRAIL10/DIP30** — post-v132 elles redeviennent profitables (DTRAIL10 both/60s/ema_slow Kelly 5.4%, DIP30 both/60s/winsor_p95 Kelly 11.4%). Laisser tourner en paper, décision après N=100 par strat.
- [ ] Attendre **N≥100/strat en régime actuel** avant tout tuning définitif (actuel N=30-41/strat).
- [ ] Si validé en paper (N≥50), A/B live vs FAST_TP50_SL30 actuel.
- [ ] FAST_TP100_SL20 → **prudence**. Médiane négative = drawdown long. Ne tester que si on accepte volatilité.
- [ ] Re-run synthetic-sweep dans 1 semaine avec N double pour trancher.

### Smoothing — conclusions

- À N=133 mélangé : `raw` ≥ smoothed sur FAST (+2.03% vs +1.53%).
- À N=30 post-v132 : `dual_confirm` > `raw` pour FAST (Kelly 17.2% vs moins), mais N trop petit pour trancher.
- **Verdict honnête : pas de gain prouvé du smoothing. Laisser FAST en `hybrid/60s/raw` (prod actuelle) jusqu'à plus de données.**
- [ ] Si smoothing validé plus tard : étendre `paper_trader._decision_price` avec `median_3/5, winsor_p95, dual_confirm, ema_fast/slow, hysteresis, volume_gated`.

## Housekeeping

### Repo hygiene
- [ ] Nettoyer commit `007db6b` qui a pushé 2400+ fichiers cache (`scraper/ohlcv_cache/`, `scraper/jupiter_candles_cache/`, `grid_search_*.csv`). Options : revert + force-push, ou ajouter au `.gitignore` et laisser.

### Data quality (depuis /check-data 2026-04-14)
- [ ] Enrichment Jupiter LDS sous-fill (35% vs seuil 70%) — vérifier quotas API / erreurs silent
- [ ] Enrichment holders sous-fill (27%) — idem
- [ ] CA resolution 71.9% (sous seuil 75%) — revoir le resolver pour les messages récents
- [ ] Backlog labels 24h = 774, 7d = 2346 — voir outcome_tracker cadence

### Sim ↔ Live/Paper coherence — current truth (Apr 14 post-v133)

**Still valid:**
- `--from-trades` = ground truth for strategy decisions (replays real stored PnL, tick-source independent)
- `--price-source jupiter` = best default for OHLCV/tick sim (dominates DTRAIL tracking behavior)
- Sensitivity check: run backtest in `jupiter`, `dexscreener`, `both` → divergence <5% = robust, >15% = fragile (typically tight trails DTRAIL3 / DIP T5)
- Tick vs OHLCV: **FAST** family → OHLCV OK (candle-close bias ~-0.3pp). **DTRAIL / hybrid DS-Jup** → tick-replay mandatory (candle-close bias explodes on trails, ex: fake DECAY +88%).

**Bias hierarchy:**
- `--from-trades` = perfectly coherent by construction
- Sim OHLCV = approximation (candles ≠ real ticks, 5-15% bias)
- Sim `--from-ticks jupiter` = coherent with prod tracking, but entry bias ~3-5% vs Ultra RFQ (ticks = Price API, prod entry = Ultra)
- Sim `--from-ticks dexscreener` = correct proxy for Ultra entry, wrong for tracking

**Updated thresholds (v133):**
- Paper/live per-pair divergence target: previously "<2%", **realistic = <5pp** (latency + slippage residuals). >10pp = bug signal.
- Empirical Ultra-vs-Price-API entry calibration countdown **restarted Apr 13** with v130 `entry_source='ultra'`. Need ~2 weeks of live data → **recheck Apr 27** to inject entry bias into OHLCV sim.

**New since v132 (less critical now):**
- Shared hooks `_decision_price` + `_should_poll_trade` used by both prod and sim via `--from-live-config`. Sim now reproduces prod orchestration at decision layer. Remaining gap is exec-level (Ultra fill vs Price API tick), not decision-level.

**Practical use:**
- `FAST_TP50_SL30` → OHLCV backtest acceptable
- `DTRAIL*` → tick-replay required, OHLCV lies
- Hybrid DS/Jup orchestration → tick-replay required (needs both streams)

### Guard rails for live strategies (anti-overfit)
Strategies (FAST_TP50_SL30, DTRAIL10_ACT15_SL70) were picked from a 6-day backtest window with N=77-138 trades/strat — edge has ±3-4pp error bar. Keep live only while:
- [ ] **7d rolling live pnl > 0** on N ≥ 30 trades per strategy
- [ ] **|live − sim| < 50% of expected edge** (if sim says +8%, live must be +4% to +12%)
- [ ] If either breaks → re-sweep on rolling 14d window via Optuna (data now clean post-v133 dedup fix)

---

# Pipeline Status — Updated Feb 28, 2026 (v74)

## Current State

v74 = robustness + completeness release. 16 fixes addressing P0 bugs, alerting gaps, compute waste, and structural improvements on top of v73 audit.

### v74 Changes (16 Items)

#### P0 — Active Bugs Fixed
- [x] **FIX 1**: Real Jupiter fill price — `execution_price` now computed from `inputAmountResult`/`outputAmountResult` in buy AND sell
- [x] **FIX 2**: RT ML model wired — `_rt_ml_model` (LightGBM) now consulted in `_rt_on_new_message` to adjust position size
- [x] **FIX 3**: Position reconciliation on startup — `reconcile_positions()` checks DB vs on-chain balance, auto-closes orphaned trades

#### P1 — Missing Features
- [x] **FIX 7**: model_kco.json — verified code handles absence gracefully (returns None), not dead code

#### P3 — Robustness & Alerting
- [x] **FIX 12**: ML disabled alert — `alert_ml_disabled()` fires when quality gate fails (daily cooldown)
- [x] **FIX 13**: rt_listener_down uncapped — `_MAX_ALERTS` set to 0 (unlimited) to prevent silent outages
- [x] **FIX 14**: GH Actions failure alerts — all 4 workflows now have `if: failure()` step with `curl` to Telegram
- [x] **FIX 15**: Write-ahead log — `_save_failed_write()` buffers failed Supabase writes to `failed_writes.jsonl`, `retry_failed_writes()` replays at cycle start
- [x] **FIX 16**: DexPaprika daily budget — `_dexpaprika_budget_ok()` tracks 9K/day limit, skips when exhausted
- [x] **FIX 17**: Dynamic SOL price fallback — CoinGecko simple price API as secondary source before $170 static

#### P4 — Structural Improvements
- [x] **FIX 18**: Dynamic slippage — `liquidity_depth_score` scales buy slippage 1x-3x (deep→shallow liquidity)
- [x] **FIX 19**: Optuna 2-phase — 40% coarse trials + 60% fine trials seeded with best, using same study
- [x] **FIX 20**: SHAP persistence — `_persist_shap_to_db()` saves top 15 features to `scoring_config.ml_shap_history` (last 30)
- [x] **FIX 21**: Daily summary cron — `daily-summary.yml` at 8am UTC, independent `daily_summary.py` module
- [x] **FIX 22**: KOL attribution — `kol_attribution()` aggregates paper trade PnL by KOL, logs top/bottom performers

---

## Files Modified (v74)

| File | Fixes | Changes |
|------|-------|---------|
| `scraper/live_trader.py` | 1, 3, 17 | Jupiter fill price, position reconciliation, CoinGecko SOL fallback |
| `scraper/safe_scraper.py` | 2, 3, 15, 22 | RT ML model wiring, reconciliation at startup, write replay, KOL attribution |
| `scraper/alerter.py` | 12, 13 | ML disabled alert, uncapped RT listener alerts |
| `scraper/pipeline.py` | 12 | ML quality gate → Telegram alert |
| `scraper/paper_trader.py` | 18, 22 | Dynamic slippage from LDS, KOL attribution function |
| `scraper/push_to_supabase.py` | 15 | Write-ahead log + retry buffer |
| `scraper/outcome_tracker.py` | 16 | DexPaprika daily budget counter |
| `scraper/auto_backtest.py` | 19 | 2-phase Optuna optimization |
| `scraper/train_model.py` | 20 | SHAP importance persistence to DB |
| `scraper/daily_summary.py` | 21 | **NEW** standalone daily summary module |
| `.github/workflows/scrape.yml` | 14 | Failure alert step |
| `.github/workflows/outcomes.yml` | 14 | Failure alert step |
| `.github/workflows/train-models.yml` | 14 | Failure alert step |
| `.github/workflows/daily-summary.yml` | 21 | **NEW** daily summary workflow |
| `.gitignore` | 15 | Add `failed_writes.jsonl` |

---

## Still Pending (Lower Priority)

### P1 — Not Yet Addressed
- [ ] **Narrative/meta alignment** — Signal #1 in MemecoinGuide. Needs external data source (Twitter/CT trends API)
- [ ] **Birdeye top N expansion** — `BIRDEYE_TOP_N = 20` means whale_new_entries NULL for 80%+ tokens. Need to increase to 50+ (costs CUs)
- [ ] **Dashboard time-window selector** — Frontend always sends 7d. Need UI dropdown for 3h/6h/24h

### P2 — Compute Optimization
- [ ] **PA computation gated** — PA weight=0% but OHLCV still fetched. Gate on `SCORING_PARAMS["price_action"] > 0`
- [ ] **gate_mult dead compute** — RugCheck/wash-trading still executed despite result always 1.0
- [ ] **v53 features** — holder_turnover, kol_cooccurrence computed but excluded from ML (<6% fill)

---

## v73 Audit Fixes (Complete) ✅

All 15 v73 audit fixes deployed and verified. See git history for details.

---

## Architecture Summary (v74)

### Scoring Engine
- **Weights:** 35/10/55/0 (consensus/conviction/breadth/PA)
- **16-multiplier chain:** hype_pen=1.0, entry_drift=1.0 (both disabled v73)
- **Optuna:** ~48 params, 2-phase search (v74), walk-forward
- **KOL dedup:** max 2 mentions per (KOL, token) pair per cycle

### Trading Safety
- **Paper slippage:** dynamic from `liquidity_depth_score` (v74), base 150bps buy / 300bps sell
- **Loss limits:** daily 2 SOL, weekly 5 SOL, monthly 10 SOL
- **Position reconciliation:** on-chain balance verified at startup (v74)
- **Fill price:** real Jupiter amounts recorded (v74)

### Alerting
- ML disabled → Telegram (v74)
- RT listener down → unlimited alerts (v74)
- GH Actions failures → Telegram via curl (v74)
- Failed writes → local buffer + retry (v74)
- Daily summary → independent cron (v74)
