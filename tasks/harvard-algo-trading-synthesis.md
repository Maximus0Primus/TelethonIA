# Research Synthesis: Harvard Algorithmic Trading with AI
## Source: `moondevonyt/Harvard-Algorithmic-Trading-with-AI`
## Goal: Extract patterns applicable to TelethonIA memecoin scoring

---

## EXECUTIVE SUMMARY

Educational repo (~950 lines Python) implementing the **RBI methodology** (Research → Backtest → Implement) for a **Bollinger Band Squeeze + ADX** strategy on HyperLiquid exchange. Pure technical analysis on a single asset (BTC), no sentiment/on-chain/social signals.

**Key takeaway for TelethonIA:** The repo itself is architecturally simpler than our system, but introduces 3 transferable patterns we don't use:
1. **Volatility squeeze detection** — pre-breakout signal adaptable to token volume
2. **ADX trend strength** — quantifiable directional conviction metric
3. **Multi-indicator confirmation** — requiring 2+ independent signals to agree

**What's NOT transferable:** Single-asset TA, HyperLiquid-specific live trading, fixed TP/SL (memecoins don't behave like BTC).

---

## REPO STRUCTURE

| File | Lines | Purpose |
|------|-------|---------|
| `backtest/data.py` | ~80 | Fetches OHLCV from HyperLiquid API, adds BB/KC/ADX indicators via `pandas_ta` |
| `backtest/template.py` | ~60 | Generic `backtesting.py` Strategy class with optimize() grid search |
| `backtest/bb_squeeze_adx.py` | ~120 | BB Squeeze + ADX strategy: entry when BB inside KC + ADX confirms trend |
| `implement/nice_funcs.py` | ~400 | HyperLiquid SDK wrapper (positions, orders, leverage, account info) |
| `implement/bot.py` | ~290 | Live trading bot with `schedule` library, 1-min candle loop |

---

## PART 1: TRANSFERABLE PATTERNS

### 1.1 Volatility Squeeze Detection (HIGH VALUE)

**Concept:** Bollinger Bands (BB) contract inside Keltner Channels (KC) when volatility is low. When BB expands back outside KC = **squeeze fires** = imminent breakout.

**Harvard implementation:**
```python
squeeze = (bb_lower > kc_lower) & (bb_upper < kc_upper)  # BB inside KC
fire = squeeze.shift(1) & ~squeeze  # transition from squeeze to non-squeeze
```

**Adaptation for TelethonIA:**
We already have `volume_1h`, `volume_6h`, `volume_24h`, and `volatility_proxy`. A volume squeeze would detect tokens where:
- Volume has been **compressing** (low vol relative to recent average)
- Then **explodes** (volume spike breaks the compression)

This is a pre-breakout signal — catching tokens BEFORE the pump, not during.

**Proposed implementation:**
```python
def _detect_volume_squeeze(token: dict) -> tuple[str, float]:
    """
    Detect volume squeeze/fire pattern.
    Returns (state, squeeze_score) where state is 'squeezing', 'firing', 'none'.
    squeeze_score: 0.0 (no signal) to 1.0 (strong squeeze fire)
    """
    vol_1h = token.get("volume_1h", 0)
    vol_6h = token.get("volume_6h", 0)
    vol_24h = token.get("volume_24h", 0)

    if not vol_6h or not vol_24h:
        return "none", 0.0

    # Hourly average over 6h vs 24h
    avg_hourly_6h = vol_6h / 6
    avg_hourly_24h = vol_24h / 24

    # Squeeze: 6h average LOWER than 24h average (compression)
    # Fire: 1h volume BREAKS above the compressed 6h average by 2x+
    compression_ratio = avg_hourly_6h / max(avg_hourly_24h, 1)

    if compression_ratio < 0.5:  # Volume was compressed
        expansion = vol_1h / max(avg_hourly_6h, 1)
        if expansion > 2.0:  # Breakout
            return "firing", min(1.0, expansion / 5.0)
        return "squeezing", 0.3

    return "none", 0.0
```

**Integration:** Add `squeeze_state` and `squeeze_score` to price_action scoring. A firing squeeze with KOL mentions = very strong entry signal.

### 1.2 ADX Trend Strength (MEDIUM VALUE)

**Concept:** ADX (Average Directional Index) measures HOW STRONG a trend is, regardless of direction. ADX > 25 = strong trend, < 20 = ranging.

**Harvard implementation:**
```python
entry_long = squeeze_fire & (adx > 20) & (close > bb_mid)  # trend + direction
entry_short = squeeze_fire & (adx > 20) & (close < bb_mid)
```

**Adaptation for TelethonIA:**
We compute momentum in `price_action.py` but don't quantify trend strength. We use discrete labels ("hard_pumping", "climbing", "consolidating") but no continuous strength metric.

**Proposed implementation:** Use price movement consistency as a proxy for ADX (since we don't have full OHLCV for all tokens):
```python
def _compute_trend_strength(token: dict) -> float:
    """
    ADX-like trend strength from available price data.
    Returns 0.0 (no trend) to 1.0 (very strong directional move).
    """
    pc1h = abs(token.get("priceChange_1h", 0))
    pc6h = abs(token.get("priceChange_6h", 0))
    pc24h = abs(token.get("priceChange_24h", 0))

    if not pc24h:
        return 0.0

    # Trend consistency: are shorter timeframes moving in same direction?
    # Strong trend: 1h and 6h both positive AND 6h > 1h (accelerating)
    direction_1h = 1 if token.get("priceChange_1h", 0) > 0 else -1
    direction_6h = 1 if token.get("priceChange_6h", 0) > 0 else -1
    direction_24h = 1 if token.get("priceChange_24h", 0) > 0 else -1

    # All timeframes agree = strong trend
    agreement = (direction_1h == direction_6h == direction_24h)

    # Magnitude (clamped to reasonable range for memecoins)
    magnitude = min(pc24h / 100, 1.0)  # 100%+ move = max

    if agreement:
        return min(1.0, 0.5 + magnitude * 0.5)
    else:
        return magnitude * 0.3  # Conflicting signals = weak trend
```

**Integration:** Factor into price_action_score as a multiplier. Strong trend + squeeze fire + KOL consensus = highest conviction.

### 1.3 Multi-Indicator Confirmation Pattern (HIGH VALUE — ARCHITECTURAL)

**Concept:** Harvard repo requires BOTH squeeze AND ADX to agree before entry. Neither alone triggers a trade.

**Current TelethonIA approach:** We have 5 independent components (consensus, sentiment, conviction, breadth, price_action) that are weighted-averaged. A token can score high on consensus alone even if price action is terrible.

**What changes:** Rather than pure weighted average, add a **confirmation gate** requiring minimum thresholds on key components:
```python
# After computing raw weighted score:
# Require at least 2 of 3 "confirmation pillars" above minimum
pillars_above_minimum = 0
if kol_consensus >= 0.3: pillars_above_minimum += 1       # Social confirmed
if price_action_score >= 0.4: pillars_above_minimum += 1   # Price confirmed
if breadth_score >= 0.3: pillars_above_minimum += 1        # Distribution confirmed

if pillars_above_minimum < 2:
    score *= 0.7  # Unconfirmed signal penalty
```

This prevents high scores from a single outlier component.

---

## PART 2: METHODOLOGY INSIGHTS

### 2.1 RBI Framework (Research → Backtest → Implement)

The Harvard repo's strongest contribution is its **disciplined methodology**:
1. **Research:** Study the indicator (BB Squeeze + ADX), understand theory
2. **Backtest:** Grid-search optimize parameters on historical data
3. **Implement:** Only go live after backtest validates

**Our gap:** We've been implementing features first, backtesting second. The v7 backtest tool exists but hasn't been used to validate any scoring changes yet.

**Action:** Before the next algorithm change (v8+), run `python backtest.py --sensitivity --walk-forward` to establish a baseline. Any change must improve the backtest metrics.

### 2.2 Parameter Grid Search (from `backtesting.py`)

Harvard uses `backtesting.py`'s `bt.optimize()` for grid search:
```python
stats = bt.optimize(
    bb_length=range(10, 30, 5),
    bb_std=np.arange(1.5, 3.0, 0.5),
    adx_length=range(10, 25, 5),
    maximize="Return [%]"
)
```

**Our parallel:** Our backtest.py already has `parameter_sensitivity()` which varies one weight at a time. We could extend this to grid search across all 5 weights simultaneously (combinatorial).

---

## PART 3: WHAT'S NOT USEFUL

| Harvard Pattern | Why Not Applicable |
|----------------|-------------------|
| Single-asset BTC | We score 50+ memecoins simultaneously |
| Fixed TP/SL (2%/1%) | Memecoins move 50-500% — fixed % stops are meaningless |
| HyperLiquid perps | We're not executing trades, just scoring |
| `schedule` library | We already use asyncio event loop with better timing |
| 1-minute candle resolution | Most memecoins don't have liquid 1m candle data |
| Short selling | Memecoins on Solana are spot-only (no shorts) |
| `pandas_ta` indicators | Requires full OHLCV series — we only have snapshots for top 5 |

---

## PART 4: RECOMMENDED IMPROVEMENTS (PRIORITY ORDER)

### Priority 1: Volume Squeeze Detection (NEW SIGNAL)
- **Impact:** Catches pre-breakout tokens before they pump
- **Effort:** Small — uses existing DexScreener volume fields
- **Where:** `scraper/pipeline.py` — new function + integrate into price_action scoring
- **New fields:** `squeeze_state` (str), `squeeze_score` (float 0-1)

### Priority 2: Multi-Indicator Confirmation Gate
- **Impact:** Reduces false positives from single-component outliers
- **Effort:** Small — 10 lines in scoring section
- **Where:** `scraper/pipeline.py` — after raw weighted score computation
- **Risk:** Could penalize legitimately early tokens (only 1 KOL but strong price)

### Priority 3: Trend Strength Metric
- **Impact:** Distinguishes strong directional moves from noise
- **Effort:** Small — uses existing price change fields
- **Where:** `scraper/pipeline.py` — factor into price_action_score
- **New fields:** `trend_strength` (float 0-1)

### NOT RECOMMENDED:
- Full `pandas_ta` integration (we don't have enough OHLCV data for most tokens)
- `backtesting.py` framework (our custom backtest.py is better suited to our scoring model)
- Live trading bot (out of scope — we're a scoring platform, not a trading bot)

---

## COMPARISON: OUR SYSTEM vs HARVARD

| Dimension | Harvard Repo | TelethonIA |
|-----------|-------------|------------|
| Assets | 1 (BTC) | 50+ memecoins |
| Data sources | HyperLiquid OHLCV | DexScreener + RugCheck + Helius + Jupiter + Birdeye + Bubblemaps + 59 Telegram groups |
| Signals | 2 (BB Squeeze, ADX) | 5 components + 80+ features |
| ML | Grid search on 3 params | XGBoost + Optuna + CalibratedClassifier |
| Sentiment | None | VADER + CryptoBERT + crypto lexicon |
| On-chain | None | Holder distribution, whale tracking, bundle detection, Gini coefficient |
| Social | None | 59 KOL groups, S/A tier weighting, consensus/breadth |
| Output | Buy/Sell signals | Ranked score 0-100 with interpretation |
| Sophistication | Educational/basic | Production-grade |

**Bottom line:** Our system is significantly more sophisticated. The Harvard repo's value is in 2-3 specific patterns (squeeze, trend strength, confirmation) that we can adapt, not in its overall architecture.
