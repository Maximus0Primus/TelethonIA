# Pipeline Status — Updated Feb 27, 2026 (v72)

## Current State

The pipeline is **production with real trading bot via Jupiter Ultra API**. v72 adds live execution that mirrors paper trading. Runs in parallel with paper trades (`source='rt_live'` vs `'rt'`). Starts disabled — flip `live_trading.enabled = true` after funding wallet.

### Live Config (Feb 27)

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
| Buy slippage | 300 bps (3%) |
| Sell slippage | 500 bps (5%) |

### Data Gates

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Unique labeled tokens (24h) | 200+ | **635+** | PASSED |
| KCO outcomes with max_return | 50+ | ~1,014 | PASSED |
| KOLs with >= 5 calls | 10+ | ~40 | PASSED |
| Approved KOLs (WR >= 60%) | 5+ | ~10 | PASSED |
| Paper trades (7d) | — | 736+ | Active |

---

## Completed Work

### v72 (Feb 27) — Real Trading Bot
- [x] Jupiter Ultra API integration via `jup-python-sdk`
- [x] `setup_wallet.py`: one-time Solana keypair generator
- [x] `live_trader.py`: buy/sell execution, position management, daily loss limit
- [x] DB migration: `tx_signature`, `tx_signature_exit`, `execution_price`, `slippage_actual_bps` on `paper_trades`
- [x] `scoring_config.rt_trade_config.live_trading` section (starts disabled)
- [x] Integration in `safe_scraper.py`: RT open + dedicated 10s monitor loop
- [x] Telegram alerts for every live trade with Solscan link
- [x] Fast 10s polling loop for live trades (single owner, no race conditions)

### v34-v67 (Feb 17-25) — See git history
- [x] Data recovery, phantom label fix, gate reform, labeling quality
- [x] Scoring reform, proxy signals, momentum_mult, Optuna (119 params)
- [x] CA identity collision fix, feature computation fixes
- [x] Paper Trading v2-v5: 7 strategies, tranche model, dedup cooldown
- [x] Real-time KOL listener + smart RT trading (v64-v66)
- [x] Monitoring system: metrics + Telegram alerts (v67)

### v68 (Feb 25) — Strategy Reform + ML Training
- [x] Optuna optimization (snapshot + KCO modes)
- [x] ML grid search: best 12h/1.5x (spearman=0.328, p@5=0.800)
- [x] Hybrid scoring mode deployed
- [x] TP50 horizon 12h→24h, SCALE_OUT SL widened, MOONBAG SL widened

### v69-v70 (Feb 26-27) — RT Improvements + ML
- [x] Auto-join all KOL groups for RT listener
- [x] Cached IDs for JoinChannel (flood wait fix)
- [x] ML training: 10 temporal features + KCO model
- [x] Crash loop fix + paper trading v5 + alerter backoff

### v71 (Feb 27) — KOL WR Filter + Hybrid Strategy + Bankroll
- [x] DB: `rt_bankroll` table (balance, peak, drawdown tracking, RLS)
- [x] DB: `scoring_config.rt_trade_config` merged with kol_filter, hybrid_strategy, sizing
- [x] `paper_trader.py`: check_paper_trades returns pnl_usd, rt_pnl_usd, rt_closed
- [x] `safe_scraper.py`: KOL whitelist from kol_call_outcomes (1h cache, fail-open)
- [x] `safe_scraper.py`: bankroll-based Kelly position sizing (legacy fallback preserved)
- [x] `safe_scraper.py`: hybrid _rt_open_trades (60/40 TP50+TP100)
- [x] `safe_scraper.py`: bankroll update in both price_refresh_loop and run_one_cycle
- [x] Code review: 6 bugs caught and fixed before deploy

---

## Monitoring v71 (THIS WEEK)

### 1. Verify v71 is Working (TODAY)
- [ ] **Check VPS logs:** `journalctl -u kol-scraper -f | grep -E "RT KOL whitelist|RT SKIP|RT TRADE|RT bankroll"`
- [ ] **Expected:** "RT KOL whitelist: ~10/40 approved"
- [ ] **Expected:** "RT SKIP (KOL WR): ramcalls" for bad KOLs
- [ ] **Expected:** "RT TRADE [HYBRID]: ... TP50_SL30=60% + TP100_SL30=40%" for good KOLs
- [ ] **Check bankroll:** `SELECT * FROM rt_bankroll;` → balance should move after trades close

### 2. Collect A/B Test Data (1 WEEK)
- [ ] **Target: 100+ RT trades** (50+ per strategy) for statistical significance
- [ ] **Daily check:** TP50 vs TP100 head-to-head WR and PnL
```sql
SELECT strategy, COUNT(*) trades,
       ROUND(AVG(CASE WHEN pnl_pct > 0 THEN 1.0 ELSE 0 END)*100,1) AS wr,
       ROUND(SUM(pnl_usd),2) AS total_pnl
FROM paper_trades WHERE source='rt' AND status != 'open'
GROUP BY strategy;
```
- [ ] **Per-KOL breakdown:** which KOLs actually generate profit?
```sql
SELECT kol_group, strategy, COUNT(*), ROUND(AVG(pnl_usd),2) AS avg_pnl
FROM paper_trades WHERE source='rt' AND status != 'open'
GROUP BY kol_group, strategy ORDER BY avg_pnl DESC;
```

### 3. Adjust Allocations (AFTER 100+ trades)
- [ ] **If TP100 > TP50:** shift to 40/60 or 30/70
- [ ] **If TP50 > TP100:** shift to 70/30 or 80/20
- [ ] **If both positive:** increase Kelly fraction from 7% → 10%
- [ ] **If both negative:** review KOL whitelist threshold (raise to 70%?)
- [ ] **If bankroll drops >25%:** pause RT trading, review

---

## Actionable Next Steps

### 4. Auto-Optimize Allocations with Optuna (AFTER 200+ RT trades)
- [ ] Add allocation optimization to auto_backtest: try all ratios from 0/100 to 100/0
- [ ] Possibly expand to 3 strategies (add SCALE_OUT?)
- [ ] Score-dependent strategy routing (high score → TP100, low → TP50)

### 5. KOL-Level Bankroll Tracking
- [ ] Track PnL per KOL in rt_bankroll or separate table
- [ ] Auto-adjust whitelist threshold based on actual RT performance (not just KCO)
- [ ] Dynamic Kelly per KOL (high-performing KOLs get larger allocation)

### 6. Graduate to Real Trading
- [ ] **Gate:** bankroll must be > $120 (20% profit) over 2+ weeks
- [ ] **Gate:** max drawdown < 15%
- [ ] **Gate:** WR > 40% across both strategies
- [ ] Start with $50 real capital, mirror paper trade logic

---

## Known Limitations (low priority)

- [ ] **narrative / narrative_is_hot** — Never implemented. Dead columns. Low ROI.
- [ ] **entry_drift_mult** — kol_stated_entry_mcaps ~95% empty. Not fixable without NLP.
- [ ] **Bubblemaps API** — Not using (too expensive).
- [ ] **Labeling backlog (~9.7K snapshots)** — Limited by OHLCV API budget. Steady-state.

---

## Architecture Summary (v71)

### RT Trading Pipeline
- **KOL filter:** whitelist from kol_call_outcomes (WR >= 60%, min 5 calls, 1h cache)
- **Strategy:** hybrid 60/40 (TP50_SL30 + TP100_SL30), configurable via DB
- **Sizing:** bankroll-based Kelly (7% × WR mult × score mult), $1-$200 range
- **Bankroll:** rt_bankroll table, updated every 3min (price_refresh) + every 30min (cycle)
- **Kill switch:** `kol_filter.enabled = false` → reverts to exploration mode

### Scoring Engine
- **Weights:** 35/10/55/0 (consensus/conviction/breadth/PA)
- **Multiplier chain:** 14 multipliers, all config-driven
- **Optuna:** 119 search params, walk-forward, hit-rate objective

### Data Pipeline
- **Scraper:** 15min full loop + 3min price refresh, 62 KOL groups
- **RT Listener:** Real-time monitoring, instant paper trading (v66+v71)
- **OHLCV sources:** DexPaprika → Birdeye → GeckoTerminal
- **Cache TTLs:** DexScreener 5min, RugCheck 2h, Birdeye 1h, Jupiter 2h, Helius 4h
