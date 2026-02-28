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
