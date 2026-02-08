# MEGA RESEARCH SYNTHESIS: Memecoin Scoring Algorithm v2
## Goal: Top-5 tokens have >50% chance of 2x within 12h

---

## EXECUTIVE SUMMARY

5 parallel research agents analyzed: trading bot architectures, ML prediction models, on-chain signals, alpha scoring systems, and our current codebase. This document synthesizes ALL findings into actionable improvements.

**Current state:** 104 features, XGBoost+LightGBM ensemble, 3 scoring modes (balanced/conviction/momentum), enrichment via DexScreener+RugCheck+Birdeye.

**Key finding:** Our architecture is solid but missing 6 critical signal categories that the best systems use. The research shows >50% precision@5 for 2x is achievable with the right features.

---

## PART 1: WHAT THE BEST BOTS DO THAT WE DON'T

### 1.1 Smart Money Wallet Tracking (HIGHEST IMPACT MISSING SIGNAL)

**Data:** Win rate has 0.610 correlation with trading success (strongest single predictor).

The best platforms (GMGN, Axiom, Nansen) track "smart money" wallets:
- **Definition:** Wallets with >40% win rate, >60% 7D PnL, <300 txns/7d
- **Signal:** When 3+ smart money wallets buy the same token = very strong buy signal
- **Anti-signal:** Fresh wallets as top holders = team wallets, likely dump

**What we need:**
- Track a curated list of 50-100 smart money wallets on Solana
- Via Birdeye API or GMGN API, detect when they buy tokens our KOLs mention
- New feature: `smart_money_buyers_count` (0-10+)
- New feature: `smart_money_avg_winrate` (0-1)

### 1.2 Bundle Detection (CRITICAL SAFETY SIGNAL)

**Data:** 82.8% of high-return memecoins show manipulation. Bundle detection catches most.

Jito Bundles = up to 5 transactions atomically executed in same slot. Attack pattern: deployer creates pool + buys tokens in same block.

**Detection method:**
1. Check first transactions at pool creation via Solscan/Jito Explorer
2. Flag tokens where deployer/connected wallets bought in first block
3. New feature: `bundle_pct` (% of supply bought via bundles at launch)
4. Threshold: bundle_pct > 10% = RED FLAG

**Implementation:** Use Helius `getSignaturesForAddress` on the token mint, check first ~20 txns for Jito bundle patterns.

### 1.3 Wash Trading Detection (VOLUME QUALITY)

**Data:** 41.4% of Solana volume is wash trading. Raw volume is unreliable.

**Detection methods (implement at least 2):**

| Method | Formula | Threshold |
|--------|---------|-----------|
| Volume/Liquidity mismatch | vol_24h / liquidity_usd | > 50x = wash traded |
| Unique makers vs volume | volume_24h / unique_makers_24h | If avg txn > $50K on micro-cap = suspicious |
| Circular txn detection | Same wallets buying/selling repeatedly | wash_score > 0.5 = bot |
| Volume/MCap extreme | vol_24h / market_cap | > 5x on established token = artificial |

**New features:**
- `wash_trading_score` (0-1): composite of above methods
- `real_volume_estimate`: volume * (1 - wash_trading_score)
- `unique_makers_ratio`: unique_makers / total_txns (higher = more organic)

### 1.4 KOL Size as NEGATIVE Signal

**Data (Bitget, 1500+ tokens, 377 influencers):**
- Large KOLs (>200K followers): **-39% after 1 week, -89% after 3 months**
- Small KOLs (<50K followers): **+25% after 1 week, +141% cumulative**

**Why:** Large KOLs are paid shills. Small KOLs have skin in the game.

**Current gap:** Our `kol_scorer.py` only tracks hit_rate. No size weighting.

**Fix:** Add inverse size weighting:
```python
size_modifier = 1.0 / (1 + log10(max(1, follower_count / 10000)))
effective_kol_score = hit_rate * size_modifier
```

### 1.5 Dynamic Per-Message Conviction (vs Static Per-Group)

**Current:** Static conviction score per group (6-10 scale, manually assigned).

**What research shows works better:**

| Conviction Level | Text Indicators | Score |
|-----------------|-----------------|-------|
| Casual mention | Token in list, passing reference | 1.0x |
| Positive sentiment | "Looks good", "interesting" | 1.2x |
| Active recommendation | "Worth watching", "good entry" | 1.5x |
| Strong conviction | "Loading bags", "my top pick", price target | 2.0x |
| All-in call | "Life-changing", "biggest play", personal stake | 2.5x |

**New features:**
- `msg_conviction_score`: per-message conviction from NLP (1.0-2.5)
- `has_price_target`: boolean (mentions specific price or multiplier)
- `has_personal_stake`: boolean ("I bought", "my position", "loaded")
- `has_urgency`: boolean ("NOW", "before it's too late")
- `has_hedging`: boolean ("might", "could", "NFA") = reduces conviction

### 1.6 Two-Phase Decay Model

**Current:** Single exponential decay lambda=0.3, half-life 2.3 hours.

**Empirical data (Pump.fun lifecycle):**
- 40% of graduating tokens do so in < 5 minutes
- 75% of tokens dead after Day 1
- KOL promotion half-life: 2-3 days
- Effective hype half-life: 4-8 hours

**Proposed two-phase:**
```python
if hours_ago <= 6:
    weight = exp(-0.15 * hours_ago)   # half-life ~4.6h (proving ground)
else:
    weight = exp(-0.15 * 6) * exp(-0.5 * (hours_ago - 6))  # half-life ~1.4h (rapid decay)
```

This gives new tokens more time to prove themselves while aggressively discounting stale signals.

---

## PART 2: ML MODEL IMPROVEMENTS

### 2.1 Best Model Architecture (Validated by Research)

**Our choice of XGBoost + LightGBM ensemble is CORRECT.**

Research confirms:
- XGBoost outperforms LSTM/Transformer for small datasets (<10K samples)
- LSTM+XGBoost hybrid is best when you have temporal data
- Transformer needs >>10K samples to avoid overfitting
- LightGBM ensemble adds diversity

**One addition worth considering:** LSTM pre-processing of temporal features (mentions_delta, sentiment_delta, volume_delta, holder_delta) fed as additional features to XGBoost. This captures temporal patterns that tree models miss.

### 2.2 Feature Priority (Research-Validated Ranking)

**Tier 1 - Highest Predictive Power:**
1. `top10_holder_pct` / `insider_pct` (strongest rug pull predictor)
2. `smart_money_buyers_count` (0.610 correlation with success)
3. `unique_wallet_24h` / `unique_makers_ratio` (organic activity)
4. `social_velocity` / `mention_acceleration` (momentum)
5. `buy_sell_ratio_1h` (trending direction)

**Tier 2 - Strong Signal:**
6. `token_age_hours` (sweet spot: 1-24h)
7. `kol_consensus` (breadth across independent KOLs)
8. `volume_mcap_ratio` (> 0.1 = healthy)
9. `narrative_is_hot` (meta alignment)
10. `pump_graduated` (survived bonding curve)

**Tier 3 - Supporting:**
11. `liq_mcap_ratio` (liquidity adequacy)
12. `volume_acceleration` (volume ramping up)
13. `sentiment_score` (positive but noisy)
14. `risk_score` (RugCheck composite)
15. `lp_locked_pct` (safety)

**Tier 4 - Noise (consider removing):**
- Raw `volume_24h` (too easily wash-traded, use `real_volume_estimate` instead)
- `price_change_24h` (lagging indicator)
- `pair_count` (weak signal)

### 2.3 New Features to Engineer

| Feature | Source | Formula | Expected Impact |
|---------|--------|---------|-----------------|
| `smart_money_count` | Birdeye/GMGN | Count of known smart wallets buying | HIGH |
| `bundle_pct` | Helius/Solscan | % of supply from bundled txns at launch | HIGH (safety) |
| `wash_trading_score` | Computed | vol_liq_mismatch + circular_txn + maker_ratio | HIGH (quality) |
| `msg_conviction_avg` | NLP | Average per-message conviction from text patterns | MEDIUM |
| `kol_size_weighted_score` | Computed | hit_rate / log(followers) | MEDIUM |
| `scoring_mode_disagreement` | Computed | std(balanced, conviction, momentum) / mean | MEDIUM |
| `holder_growth_rate` | Birdeye | unique_wallet_24h_change normalized | MEDIUM |
| `time_since_graduation` | DexScreener | Hours since Pump.fun graduation | MEDIUM |
| `narrative_pvp_count` | Computed | Count of tokens in same narrative this cycle | MEDIUM (negative) |
| `dev_wallet_sold` | On-chain | Boolean: has dev wallet sold any tokens? | HIGH (safety) |

### 2.4 Hard Gates (Before ML Scoring)

Research shows some signals should be binary gates, not soft features:

```python
# AUTOMATIC SCORE = 0 (hard reject)
if has_mint_authority: return 0
if has_freeze_authority and not is_spl2022: return 0
if top10_holder_pct > 70: return 0
if insider_pct > 50: return 0
if bundle_pct > 30: return 0
if risk_score > 8000: return 0

# AUTOMATIC PENALTY (soft gate)
if token_age_hours < 0.5: score *= 0.3  # too new, no data
if wash_trading_score > 0.7: score *= 0.4  # mostly fake volume
```

### 2.5 Backtesting Corrections

**Critical pitfall: Survivorship Bias**
- Only 1.4% of Pump.fun tokens survive. Our training data MUST include dead tokens.
- Current `token_snapshots` table should capture ALL scored tokens, not just top ones.
- Use MemeChain dataset (34,988 tokens across 4 chains) for supplementary training data.

**Slippage Modeling:**
- Standard crypto: 0.1-0.5% slippage
- New memecoins: 1-3% minimum
- During volatility: 5-10%+
- Add slippage cost to backtesting: `real_return = price_return - 2 * slippage` (buy + sell)

**Walk-Forward:** Our current 70/30 temporal split is correct. Do NOT use random shuffle.

---

## PART 3: SCORING FORMULA v2

### 3.1 Current Formula (for reference)

```
Momentum: 0.40*recency + 0.30*sentiment + 0.20*consensus + 0.10*breadth
```

### 3.2 Proposed Formula v2

**Phase 1: Safety Gate**
```python
safety_pass = (
    not has_mint_authority
    and not has_freeze_authority
    and top10_holder_pct < 70
    and insider_pct < 50
    and bundle_pct < 30
    and risk_score < 8000
)
if not safety_pass: return score=0
```

**Phase 2: Quality-Adjusted Signals**
```python
# Replace raw volume with quality-adjusted
real_volume = volume_24h * (1 - wash_trading_score)
real_vmcr = real_volume / market_cap

# KOL quality over quantity
kol_quality = mean([hit_rate / log10(max(1, followers/10000)) for kol in kols])

# Dynamic conviction from text
effective_conviction = msg_conviction_avg * (0.5 + kol_quality)
```

**Phase 3: Composite Score (Momentum v2)**
```python
raw_momentum = (
    0.25 * recency_score_v2          # two-phase decay
    + 0.20 * kol_quality_consensus    # quality-weighted KOL breadth
    + 0.15 * effective_conviction      # text-derived conviction * KOL quality
    + 0.15 * smart_money_signal       # min(1, smart_money_count / 3)
    + 0.10 * sentiment_score          # NLP sentiment (reduced weight)
    + 0.10 * volume_quality_score     # real_vmcr normalized
    + 0.05 * narrative_bonus          # hot meta alignment
)
```

**Phase 4: On-Chain Multipliers**
```python
onchain_mult = mean([
    age_mult,           # 1-24h = 1.2x
    liquidity_mult,     # adequate liq = 1.0-1.2x
    holder_health_mult, # good distribution = 1.0-1.3x
    graduation_mult,    # graduated = 1.1x
])

safety_pen = product([
    insider_pen,        # >30% = penalty
    top10_pen,          # >50% = penalty
    risk_pen,           # >5000 = penalty
])

# PVP penalty (multiple tokens same narrative)
pvp_penalty = 1.0 / (1 + 0.1 * max(0, narrative_pvp_count - 2))

final_score = int(raw_momentum * onchain_mult * safety_pen * pvp_penalty * 100)
```

### 3.3 Key Differences from v1

| Aspect | v1 | v2 |
|--------|----|----|
| KOL weighting | Equal weight per group | Quality-weighted (hit_rate / log(size)) |
| Conviction | Static per-group (6-10) | Dynamic per-message NLP |
| Volume | Raw volume_24h | Wash-trading adjusted |
| Smart money | Not tracked | Core signal (0.15 weight) |
| Bundle detection | Not checked | Hard gate + feature |
| Decay model | Single lambda=0.3 | Two-phase (0.15 then 0.5) |
| PVP detection | Not implemented | Narrative competition penalty |
| Safety | Soft multipliers only | Hard gates + soft multipliers |
| Sentiment weight | 0.30 (high) | 0.10 (reduced, noisy) |

---

## PART 4: IMPLEMENTATION ROADMAP

### Sprint 1: Quick Wins (1-2 days)
- [ ] Two-phase decay model in pipeline.py
- [ ] Hard safety gates (mint/freeze/top10>70% = score 0)
- [ ] Reduce sentiment weight from 0.30 to 0.10-0.15
- [ ] Add `scoring_mode_disagreement` feature
- [ ] PVP penalty (count tokens per narrative)

### Sprint 2: Volume Quality (2-3 days)
- [ ] Wash trading score (vol/liq mismatch + maker ratio)
- [ ] Replace raw volume with quality-adjusted volume
- [ ] Add `unique_makers_ratio` feature
- [ ] Better DexScreener data: fetch unique makers count

### Sprint 3: Dynamic Conviction (2-3 days)
- [ ] Per-message conviction NLP (price targets, personal stake, urgency, hedging)
- [ ] Replace static group conviction with dynamic average
- [ ] Add conviction-related features to ML model

### Sprint 4: KOL Quality v2 (2-3 days)
- [ ] Track KOL follower counts (scrape from Telegram or manual)
- [ ] Inverse size weighting in kol_scorer.py
- [ ] Known grifter blacklist (auto-score = 0 for their calls)
- [ ] KOL disagreement feature (when usually-agreeing KOLs diverge)

### Sprint 5: Smart Money Integration (3-5 days)
- [ ] Curate list of 50-100 smart money wallets
- [ ] Birdeye API: check if smart wallets hold our scored tokens
- [ ] New features: smart_money_count, smart_money_avg_winrate
- [ ] Integrate into scoring formula

### Sprint 6: Bundle Detection (3-5 days)
- [ ] Helius/Solscan API: check first transactions at token creation
- [ ] Detect Jito bundle patterns at launch
- [ ] New feature: bundle_pct
- [ ] Add as hard gate (>30% = reject)

### Sprint 7: ML Model Retrain (2-3 days)
- [ ] Add all new features to token_snapshots
- [ ] Collect 2-4 weeks of enriched data
- [ ] Retrain XGBoost+LightGBM with new features
- [ ] Evaluate precision@5 improvement
- [ ] SHAP analysis on feature importance
- [ ] Prune low-importance features

---

## PART 5: KEY NUMBERS TO REMEMBER

| Metric | Value | Source |
|--------|-------|--------|
| Pump.fun graduation rate | 1.4% | Research |
| Avg memecoin lifespan | 12 days | ChainPlay.gg |
| 75% tokens dead after | Day 1 | Pump.fun data |
| KOL promoted tokens -70% after | 1 week | Bitget (1500 tokens) |
| Large KOL (>200K) avg return | -39% 1w, -89% 3mo | Bitget |
| Small KOL (<50K) avg return | +25% 1w, +141% cum | Bitget |
| Smart money win rate correlation | 0.610 | 1080 wallet study |
| Wash trading % on Solana | 41.4% | Research |
| Manipulated high-return tokens | 82.8% | Academic paper |
| Healthy vol/mcap ratio | 10-50% | Multiple sources |
| Safe top10 holder threshold | <20% | Aggregated research |
| Danger top10 threshold | >50% | Aggregated research |
| Kelly fraction for 55% edge | 10% (use half = 5%) | Kelly criterion |
| Slippage on new memecoins | 1-3% minimum | Solana data |
| Axiom bot execution speed | <0.4 seconds | Axiom docs |
| Jito bundle auction interval | 200ms | Jito docs |

---

## PART 6: SOURCES (Selected)

### Trading Bots
- Axiom Trade: 57% Solana market share, $200M revenue in 202 days
- Photon: $421M in fees, analytics-first terminal
- BONKbot: 74% win rate (experienced), 41.3% monthly ROI
- GMGN: Smart money tracking + copy trading

### Academic Papers
- CoinCLIP (CIKM '25): Multimodal memecoin viability, best AUC
- Resisting Manipulative Bots (arXiv 2601.08641): Multi-agent LLM, >70% precision
- MemeChain (arXiv 2601.22185): 34,988 token dataset
- A Midsummer Meme's Dream (arXiv 2507.01963): 82.8% manipulation rate
- ME2F Framework (arXiv 2512.00377): Fragility scoring

### Tools
- RugCheck API: Risk score 0-10000, holder distribution, security flags
- GoPlus Security API: Free, covers Solana, mint/freeze/holder checks
- BubbleMaps: Supply distribution visualization, cluster detection
- Solana Bundler Detector (GitHub): Bundle detection algorithm
- Helius gRPC: Sub-50ms event streaming from Solana validators

### Key Benchmarks
- XGBoost > LSTM/Transformer for <10K samples
- CryptoBERT: 70% 3-class accuracy (19% improvement over VADER alone)
- VADER + crypto lexicon: competitive with BERT per 2025 research
- Multi-agent LLM CoT: >70% precision, 3% avg return per trade
