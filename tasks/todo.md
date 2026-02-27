# Pipeline Status — Updated Feb 28, 2026 (v73)

## Current State

The pipeline is **production with real trading bot via Jupiter Ultra API**. v73 is an audit fix release addressing 15 issues causing -$264.40 total PnL. Key changes: broken dedup fixed, anti-predictive hype_pen disabled, ML overfitting reduced (139→48 Optuna params), slippage simulation added, weekly/monthly drawdown limits.

### Live Config (Feb 28)

| Setting | Value |
|---------|-------|
| KOL filter | enabled, WR >= 60%, min 5 calls, return >= 1.5x |
| Hybrid strategy | TP50_SL30 (60%) + TP100_SL30 (40%) |
| Position sizing | bankroll mode, 7% Kelly of balance |
| Starting capital | $100 |
| Min/max position | $1 / $200 |
| **Live trading** | **disabled** (flip to true after funding) |
| Max position (SOL) | 0.5 SOL per trade |
| Max open positions | 5 concurrent |
| SOL reserve | 0.05 SOL (for fees) |
| Daily loss limit | 2.0 SOL |
| Weekly loss limit | 5.0 SOL (v73) |
| Monthly loss limit | 10.0 SOL (v73) |
| Buy slippage (paper) | 150 bps (1.5%) (v73) |
| Sell slippage (paper) | 300 bps (3.0%) (v73) |
| RugCheck | disabled by default (v73) |
| Optuna params | ~48 (was ~139) (v73) |
| KOL mention cap | 2 per (KOL, token) (v73) |

---

## v73 Audit Fixes (15 Issues) ✅

### Phase 1: Infrastructure ✅
- [x] **FIX 2**: ML Training Auto-Schedule — daily cron at 3am UTC
- [x] **FIX 6**: Outcomes.yml — `continue-on-error` removed from steps 1-3
- [x] **FIX 13**: Pin requirements.txt — all `==` exact versions
- [x] **FIX 14**: ML Model Versioning — backup before overwrite + version field

### Phase 2: Scoring Formula ✅
- [x] **FIX 8**: Disable hype_pen (contradicts breadth 55% weight)
- [x] **FIX 12**: Disable entry_drift_mult (95% empty data)
- [x] **FIX 9**: Recalibrate activity_mult (softer penalty, 0.0/0.15/0.35)
- [x] **FIX 15**: Expand BARE_WORD_SUSPECTS (+13 crypto slang words)

### Phase 3: Optuna + KOL ✅
- [x] **FIX 4**: Reduce Optuna ~139→~48 params (hardcode DB best for rest)
- [x] **FIX 5**: KOL Whitelist Grid Search (WR × min_calls → Sharpe)

### Phase 4: Pipeline Dedup ✅
- [x] **FIX 1**: Per-KOL-Per-Token dedup cap (KOL_MENTION_CAP=2)

### Phase 5: ML Calibration ✅
- [x] **FIX 3**: Calibration leakage fix (70/30 train/cal split)

### Phase 6: Trading Safety ✅
- [x] **FIX 7**: Slippage simulation (buy 150bps, sell 300bps)
- [x] **FIX 10**: Weekly/monthly drawdown limits (5/10 SOL defaults)
- [x] **FIX 11**: Skip RugCheck (RUGCHECK_ENABLED=0 default)

---

## Post-Deploy Monitoring (v73)

### Immediate (Day 1)
- [ ] **Deploy to VPS** and verify logs: no crashes, no import errors
- [ ] **Check GH Actions**: train-models.yml fires daily at 3am UTC
- [ ] **Verify outcomes.yml**: steps 1-3 fail visibly when broken
- [ ] **Score distributions**: hype_pen=1.0 should boost multi-KOL tokens
- [ ] **Dedup impact**: spidersjournal flood tokens should drop in rank

### Week 1
- [ ] **Paper trade PnL**: expect ~4.5% drop from slippage (more realistic)
- [ ] **KOL grid search results**: check scoring_config.kol_filter_config
- [ ] **Optuna convergence**: 48 params should converge faster than 139
- [ ] **ML calibration**: compare Brier scores with previous

---

## Completed Work

### v72 (Feb 27) — Real Trading Bot
- [x] Jupiter Ultra API integration via `jup-python-sdk`
- [x] `live_trader.py`: buy/sell, position management, daily loss limit
- [x] 10s fast polling loop, Telegram alerts

### v71 (Feb 27) — KOL WR Filter + Hybrid Strategy + Bankroll
- [x] KOL whitelist, bankroll-based Kelly, hybrid 60/40

### v34-v70 — See git history

---

## Known Limitations (low priority)
- [ ] **narrative / narrative_is_hot** — Never implemented. Dead columns.
- [ ] **Bubblemaps API** — Not using (too expensive).
- [ ] **Labeling backlog (~9.7K snapshots)** — Limited by OHLCV API budget.

---

## Architecture Summary (v73)

### Scoring Engine
- **Weights:** 35/10/55/0 (consensus/conviction/breadth/PA)
- **Multiplier chain:** 14 multipliers — hype_pen=1.0, entry_drift=1.0 (both disabled v73)
- **Optuna:** ~48 search params (was ~139), walk-forward, hit-rate objective
- **KOL dedup:** max 2 mentions per (KOL, token) pair per cycle

### Trading Safety
- **Paper slippage:** buy 1.5%, sell 3.0% (realistic memecoin spreads)
- **Loss limits:** daily 2 SOL, weekly 5 SOL, monthly 10 SOL
- **RugCheck:** disabled by default (save ~30s/cycle)
