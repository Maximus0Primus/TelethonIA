# Algorithm v4 Implementation — COMPLETE

## Sprint 1: Price Action Analysis via Birdeye OHLCV ✅
- [x] `scraper/enrich_birdeye_ohlcv.py` — Birdeye OHLCV 15m candles (top 15 tokens)
- [x] `scraper/price_action.py` — price position, momentum direction, volume confirm, support
- [x] `scraper/pipeline.py` — OHLCV enrichment + price action + v4 balanced weights
- [x] `scraper/push_to_supabase.py` — new snapshot fields
- [x] `supabase/migrations/v4_price_action.sql` — DB migration

## Sprint 2: Fix Incohérences ✅
- [x] Balanced: 30% consensus + 10% sentiment + 20% conviction + 20% breadth + 20% price_action
- [x] Token age: 6-48h optimal (was 1-24h)
- [x] Volatility proxy + whale dominance integrated
- [x] Pump detection degressive (was binary 0.2x)
- [x] Quality-weighted breadth (KOL reputation * mentions)
- [x] Removed blend slider (ViewControls, HomeClient, route.ts)

## Sprint 3: Micro-Refresh Prix (5 min) ✅
- [x] `scraper/price_refresh.py` — DexScreener refresh for top 20
- [x] `scraper/safe_scraper.py` — 5min background asyncio task
- [x] base_score columns in tokens table

## Sprint 4: Whale Direction Tracking ✅
- [x] `scraper/enrich_helius.py` — whale_change_history + whale_direction
- [x] distributing → 0.6x safety, accumulating → 1.15x on-chain

## Algorithm v8: Harvard Adaptations (Volume Squeeze, Trend Strength, Confirmation Gate) ✅
- [x] `_detect_volume_squeeze()` — BB Squeeze adaptation: compression → expansion detection
- [x] `_compute_trend_strength()` — ADX adaptation: directional conviction across timeframes
- [x] Squeeze bonus (up to 1.2x for firing) + Trend bonus (up to 1.15x for strong trend)
- [x] Multi-indicator confirmation gate: 2 of 3 pillars required (consensus/price_action/breadth)
- [x] Unconfirmed tokens get 0.7x penalty across all 3 scoring modes
- [x] New DB columns: squeeze_state, squeeze_score, trend_strength, confirmation_pillars
- [x] push_to_supabase.py updated: snapshot inserts + NUMERIC_LIMITS
- [x] Migration applied to Supabase

## Pending Actions
- [ ] Run `supabase/migrations/v4_price_action.sql` in Supabase SQL editor
- [ ] Add `BIRDEYE_API_KEY=<key>` to `scraper/.env`

---

# Consensus - Phantom Style Redesign

## Completed Tasks

### Phase 1: Theme Update
- [x] Update globals.css with Phantom dark theme
  - Pure black background (#000000)
  - Dark card surfaces (#111111)
  - White text and accents
  - Custom CSS for glass effects, 3D perspective, token cards

### Phase 2: Layout Update
- [x] Update layout.tsx
  - Replaced fonts (Inter instead of Geist/Playfair)
  - Updated metadata

### Phase 3: Layout Components
- [x] IntroReveal.tsx - New Phantom-style intro animation (logo zoom/fade)
- [x] Header.tsx - Ultra-minimal (logo only, transparent)
- [x] FloatingNav.tsx - Pills navigation at bottom center (Tokens | About)
- [x] ViewControls.tsx - Grid/List toggle + Filter button

### Phase 4: Token Components
- [x] TokenCard.tsx - 3D card with symbol, score, trend
- [x] TokenGrid.tsx - Perspective grid with parallax + list view

### Phase 5: About Page
- [x] /about - Manifesto style page with light background
- [x] Large typography sections
- [x] Stats and "How It Works" sections

### Phase 6: Token Detail Page
- [x] Dark theme styling
- [x] Clean layout with stats cards
- [x] Top KOLs and Recent Mentions sections

### Cleanup
- [x] Deleted ranking/ folder (RankingTable, TokenRow, HeroStats, TimeFilter)
- [x] Deleted BackgroundMesh.tsx
- [x] Deleted GlassCard.tsx

### Build Fixes
- [x] Fixed TimeWindow type import in API route
- [x] Fixed TokenData import in mockData.ts
- [x] Build verification: `npm run build` passes

---

## Files Changed/Created

### Modified
- `src/app/globals.css` - Complete dark theme overhaul
- `src/app/layout.tsx` - New fonts, simplified structure
- `src/app/page.tsx` - New component structure with grid
- `src/app/token/[symbol]/page.tsx` - Dark theme styling
- `src/app/api/ranking/route.ts` - Local TimeWindow type
- `src/data/mockData.ts` - Updated import
- `src/components/layout/IntroReveal.tsx` - Phantom-style animation
- `src/components/layout/Header.tsx` - Minimal design

### Created
- `src/app/about/page.tsx` - Manifesto page
- `src/components/layout/FloatingNav.tsx` - Bottom navigation
- `src/components/layout/ViewControls.tsx` - View toggles
- `src/components/tokens/TokenCard.tsx` - Token card component
- `src/components/tokens/TokenGrid.tsx` - 3D grid component
- `src/hooks/useParallax.ts` - Parallax hook

### Deleted
- `src/components/ranking/` (entire folder)
- `src/components/layout/BackgroundMesh.tsx`
- `src/components/shared/GlassCard.tsx`

---

## Verification Checklist

- [x] Animation intro "Consensus" fonctionne
- [x] Grille 3D avec perspective visible
- [x] Navigation en bas (Tokens | About)
- [x] Header minimal avec logo seul
- [x] Toggles Grid/List et Filter présents
- [x] Hover sur cartes fonctionne
- [x] Click ouvre page détail
- [x] Page About style manifesto
- [x] List view mode works
- [ ] Responsive mobile OK (needs manual testing)
- [x] Parallax au scroll (desktop) - implemented

---

## Auto-Scraping + Auto-Update + Wave/Shuffle Animation

### Phase C — Frontend animation (DONE)
- [x] Create `src/hooks/useAutoRefresh.ts` — polls `/api/ranking/updated-at` every 60s
- [x] Modify `src/app/page.tsx` — animation state machine (idle/glitching/shuffling), debug button
- [x] Modify `src/components/tokens/TokenGrid.tsx` — LayoutGroup, prevRankMap, isNew detection
- [x] Modify `src/components/tokens/TokenCard.tsx` — layoutId, glitch cascade, score scramble, rank delta badge
- [x] Modify `src/app/globals.css` — scanlines, glitch-border, glitch-text-shadow keyframes

### Phase B — API Supabase + polling endpoint (DONE)
- [x] Rewrite `src/app/api/ranking/route.ts` — Supabase RPC via REST instead of fs.readFileSync
- [x] Create `src/app/api/ranking/updated-at/route.ts` — lightweight polling endpoint
- [x] Add `scrape_metadata` table to `supabase/schema.sql`
- [x] Update `src/lib/supabase/types.ts` with scrape_metadata types

### Phase A — Safe scraper + pipeline (DONE)
- [x] Create `scraper/safe_scraper.py` — 30min loop, FloodWait handling, random shuffle, jitter
- [x] Create `scraper/pipeline.py` — extract_tokens, calculate_sentiment, aggregate_ranking
- [x] Create `scraper/push_to_supabase.py` — upsert tokens + scrape_metadata
- [x] Create `scraper/.env` template
- [x] Create `scraper/requirements.txt`

### Verification
- [x] `npm run build` — 0 errors
- [ ] Phase C: Click debug button in dev mode -> verify glitch cascade + shuffle
- [ ] Phase B: Deploy schema, run `curl /api/ranking?window=24h` -> verify Supabase data
- [ ] Phase A: Fill `.env` with real creds, run `python safe_scraper.py`, check Supabase SQL editor

---

## TODO MANUEL — Ce qu'il te reste a faire

### Etape 1 : Deployer le schema Supabase

Va dans **Supabase Dashboard** > **SQL Editor** et execute tout le contenu de `supabase/schema.sql`.

Ca va creer :
- `tokens` (ranking par time_window)
- `groups` (config KOL)
- `mentions` (raw data)
- `profiles` / `subscriptions` / `api_keys` (auth + stripe)
- `scrape_metadata` (singleton pour le polling)
- `get_token_ranking()` RPC function (locked to service_role)
- RLS sur toutes les tables, zero policy publique

### Etape 2 : Configurer `.env.local` du frontend

Fichier : `crypto-kol-ranking/.env.local`

```
NEXT_PUBLIC_SUPABASE_URL=https://xbcasrywqqmnotknzbpg.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhiY2Fzcnl3cXFtbm90a256YnBnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzA0MTY4OTQsImV4cCI6MjA4NTk5Mjg5NH0.Tld7Qz6ofaB-99NMGrnJpz7hTl7fWYuHuvbwu2k0uZQ
SUPABASE_SERVICE_ROLE_KEY=<ta clef service_role depuis Dashboard > Settings > API>
```

La clef `service_role` se trouve dans :
**Supabase Dashboard** > **Settings** > **API** > **Project API keys** > `service_role` (celle qui commence par `eyJ...`)

### Etape 3 : Configurer `scraper/.env`

Fichier : `scraper/.env`

```
TELEGRAM_API_ID=<ton api_id Telegram>
TELEGRAM_API_HASH=<ton api_hash Telegram>
TELEGRAM_SESSION_NAME=scraper_session
SUPABASE_URL=https://xbcasrywqqmnotknzbpg.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<meme clef service_role que ci-dessus>
```

### Etape 4 : Tester le frontend (animation)

```bash
cd crypto-kol-ranking
npm run dev
```

- Ouvre http://localhost:3000
- Apres l'intro, un bouton vert **"Trigger Update"** apparait en bas a droite (dev mode only)
- Clique dessus pour voir : glitch cascade (vert matrix) → shuffle des cartes
- Note : sans data Supabase, l'API retournera une erreur — c'est normal

### Etape 5 : Installer les deps du scraper

```bash
cd scraper
pip install -r requirements.txt
```

### Etape 6 : Lancer le scraper

```bash
cd scraper
python safe_scraper.py
```

- Premiere fois : Telethon demandera ton numero de telephone + code
- Le scraper va tourner en boucle toutes les 30 min
- Logs : 3-7s entre chaque groupe, FloodWait gere automatiquement
- Apres chaque cycle : upsert dans Supabase, le frontend se refresh en ~60s

### Etape 7 : Verifier que tout marche

1. **Supabase SQL Editor** : `SELECT * FROM tokens WHERE time_window = '24h' ORDER BY score DESC LIMIT 5;`
2. **API** : `curl http://localhost:3000/api/ranking?window=24h&limit=5`
3. **Frontend** : Les cartes apparaissent, et se mettent a jour automatiquement toutes les 60s
4. **Animation** : Quand le scraper push de nouvelles donnees, la cascade glitch + shuffle se declenche

### Deploiement scraper (plus tard)

Pour que le scraper tourne 24/7 :
- **Railway** : Docker + persistent volume pour la session Telethon (~$5/mois)
- **Render** : Background Worker
- **Fly.io** : Machine always-on
- **VPS** : n'importe quel VPS avec `tmux` ou `systemd`

---

## ML Scoring Algorithm — XGBoost + Data Enrichment (DONE)

### Implementation Status

- [x] **Step 1:** `scraper/enrich.py` — DexScreener + RugCheck enrichment module
- [x] **Step 2:** Supabase migration — `token_snapshots` table created
- [x] **Step 3:** `scraper/push_to_supabase.py` — Added `insert_snapshots()` function
- [x] **Step 4:** `scraper/pipeline.py` — Calls `enrich_tokens()` + exposes ML features
- [x] **Step 5:** `scraper/outcome_tracker.py` — Fills price labels after 6h/12h/24h
- [x] **Step 6:** `scraper/safe_scraper.py` — Calls `insert_snapshots()` + `fill_outcomes()` in loop
- [x] **Step 7:** `scraper/train_model.py` — XGBoost + Optuna training script
- [x] **Step 8:** `scraper/pipeline.py` — ML model integration (auto-loads model_12h.json)
- [x] **Step 9:** `scraper/requirements.txt` — Added xgboost, optuna, pandas, numpy, scikit-learn
- [x] **Step 10:** `supabase/schema.sql` — Documented token_snapshots table

### Next Steps (Manual)

1. `pip install -r scraper/requirements.txt` — install new ML dependencies
2. Run scraper normally — starts collecting snapshots + enrichment automatically
3. Wait ~1-2 weeks for ~200+ labeled snapshots
4. `python scraper/train_model.py --horizon 12h --trials 100` — train first model
5. Model auto-loads on next scraper cycle (pipeline.py detects model_12h.json)

---

## Phase 2: Algorithm Intelligence Upgrade (DONE)

### Implementation Status

- [x] **Step 1:** `scraper/kol_scorer.py` — KOL reputation scoring from historical 2x outcomes
- [x] **Step 2:** `scraper/pipeline.py` — Narrative/meta classification (ai_agent, animal, politics, etc.)
- [x] **Step 3:** `scraper/pipeline.py` — CryptoBERT sentiment (optional, falls back to VADER)
- [x] **Step 4:** `scraper/enrich.py` — Pump.fun graduation detection (bonding → graduated)
- [x] **Step 5:** `scraper/push_to_supabase.py` — Temporal features (mentions/sentiment/volume/holder deltas)
- [x] **Step 6:** `scraper/train_model.py` — ML calibration (isotonic regression via CalibratedClassifierCV)
- [x] **Step 7:** `scraper/train_model.py` — Phase 1+2 features added to ALL_FEATURE_COLS
- [x] **Step 8:** Supabase migration — 9 new columns on token_snapshots

### Files Modified
- `scraper/pipeline.py` — KOL reputation weighting, narrative classification, CryptoBERT optional, pump.fun/narrative score bonuses
- `scraper/enrich.py` — pump_graduation_status field (bonding/graduated/null)
- `scraper/push_to_supabase.py` — temporal deltas, Phase 2 columns in snapshots
- `scraper/train_model.py` — +11 features (Phase 1+2), pump_graduated derived, calibration
- `scraper/requirements.txt` — joblib, optional transformers/torch

### Files Created
- `scraper/kol_scorer.py` — computes KOL hit rates, caches to kol_scores.json

### DB Migration
- token_snapshots: +9 columns (top_kols, narrative, kol_reputation_avg, narrative_is_hot, pump_graduation_status, mentions_delta, sentiment_delta, volume_delta, holder_delta)

### Verification
1. `python kol_scorer.py --verbose` — check kol_scores.json has plausible hit rates
2. Run scraper twice → second run should have non-null temporal deltas
3. Check logs for "CryptoBERT loaded" (if transformers installed) or "VADER fallback"
4. Check enriched tokens for pump_graduation_status
5. `python train_model.py --horizon 12h --trials 5` → verify new features appear

---

## Phase 3: Helius Smart Money & Bundle Detection (DONE)

### Implementation Status

- [x] **Step 1:** Supabase migration — +10 columns on token_snapshots (helius_holder_count, helius_top5_pct, helius_top20_pct, helius_gini, bundle_detected, bundle_count, bundle_pct, helius_recent_tx_count, helius_unique_buyers, helius_onchain_bsr)
- [x] **Step 2:** `scraper/enrich_helius.py` — Full Helius API module (getTokenAccounts, getSignaturesForAddress, bundle detection, Gini coefficient, holder quality analysis)
- [x] **Step 3:** `scraper/pipeline.py` — Helius integration (import, call after enrich_tokens, safety penalty for bundles/gini/low holders, on-chain multiplier for tx activity/BSR)
- [x] **Step 4:** `scraper/push_to_supabase.py` — 10 new Helius fields in insert_snapshots()
- [x] **Step 5:** `scraper/train_model.py` — +10 features in ALL_FEATURE_COLS, helius_holder_count log transform
- [x] **Step 6:** `scraper/.env` — HELIUS_API_KEY added

### Files Modified
- `scraper/pipeline.py` — import enrich_helius, call in aggregate_ranking, 3 new safety checks (bundle, gini, holder count), 2 new on-chain factors (tx count, BSR)
- `scraper/push_to_supabase.py` — 10 new Helius fields in snapshot rows
- `scraper/train_model.py` — 10 new features + helius_holder_count log transform

### Files Created
- `scraper/enrich_helius.py` — Complete Helius API module (bundle detection, holder quality, transaction analysis, 2h cache, rate limiting)

### DB Migration (RUN MANUALLY)
```sql
ALTER TABLE token_snapshots
  ADD COLUMN IF NOT EXISTS helius_holder_count integer,
  ADD COLUMN IF NOT EXISTS helius_top5_pct real,
  ADD COLUMN IF NOT EXISTS helius_top20_pct real,
  ADD COLUMN IF NOT EXISTS helius_gini real,
  ADD COLUMN IF NOT EXISTS bundle_detected smallint,
  ADD COLUMN IF NOT EXISTS bundle_count integer,
  ADD COLUMN IF NOT EXISTS bundle_pct real,
  ADD COLUMN IF NOT EXISTS helius_recent_tx_count integer,
  ADD COLUMN IF NOT EXISTS helius_unique_buyers integer,
  ADD COLUMN IF NOT EXISTS helius_onchain_bsr real;
```

### Verification
- [x] Syntax: all 4 files compile (py_compile)
- [x] Unit test: Gini (equal=0.0, whale=0.99) PASS
- [x] Unit test: Bundle detection (10 similar=detected, diverse=not detected) PASS
- [x] Integration: enrich_tokens_helius graceful fallback without API key PASS
- [ ] DB migration: run SQL above in Supabase SQL Editor
- [ ] Live test: run scraper → check token_snapshots for non-null helius_holder_count on top 10
- [ ] ML test: `python train_model.py --horizon 12h --trials 5` → new features in metadata

### Credit Budget
- Free tier: 1M credits/month
- Estimated usage: ~22K/month (2.2% of budget)
- getTokenAccounts: 10 credits/page, top 10 tokens/cycle
- getSignaturesForAddress: 10 credits/call, top 5 tokens/cycle

---

## Phase 3B: Jupiter, Whale Tracking, Semantic Narratives (DONE)

### Implementation Status

- [x] **Step 1:** Supabase migration — +9 columns on token_snapshots (jup_tradeable, jup_price_impact_1k, jup_route_count, jup_price_usd, whale_count, whale_total_pct, whale_change, whale_new_entries, narrative_confidence)
- [x] **Step 2:** `scraper/enrich_jupiter.py` — Jupiter API module (quote for tradeability + price impact, batch price lookup, 30min cache)
- [x] **Step 3:** `scraper/enrich_helius.py` — Whale tracking (_analyze_whales, cross-cycle comparison, no extra API calls)
- [x] **Step 4:** `scraper/pipeline.py` — Semantic narrative classifier (all-MiniLM-L6-v2, 12 narratives, keyword fallback), Jupiter + whale scoring factors
- [x] **Step 5:** `scraper/push_to_supabase.py` — 9 new fields in insert_snapshots()
- [x] **Step 6:** `scraper/train_model.py` — +8 features in ALL_FEATURE_COLS (69 total)
- [x] **Step 7:** `scraper/requirements.txt` — Optional sentence-transformers

### Files Created
- `scraper/enrich_jupiter.py` — Jupiter Quote + Price API module

### Files Modified
- `scraper/enrich_helius.py` — _analyze_whales(), _empty_helius_result() +4 whale fields
- `scraper/pipeline.py` — semantic narrative (12 categories), Jupiter import + call, onchain_multiplier +2 factors (Jupiter liquidity, whale accumulation), safety_penalty +1 check (whale concentration)
- `scraper/push_to_supabase.py` — +9 columns in snapshot rows
- `scraper/train_model.py` — +8 ML features
- `scraper/requirements.txt` — optional sentence-transformers

### API Budget: $0
- Jupiter Quote + Price: FREE, no auth, 10 req/s
- Whale tracking: reuses existing Helius holder data (0 extra API calls)
- sentence-transformers: local CPU model (22MB)

### Verification
- [x] Syntax: all 5 files compile (py_compile)
- [x] Unit test: Jupiter quote parsing (400=not tradeable, 200=tradeable with price impact)
- [x] Unit test: Jupiter batch price lookup
- [x] Unit test: Whale tracking first cycle (count=2, change=None)
- [x] Unit test: Whale tracking second cycle (change detected, new_entries=1)
- [x] Unit test: Narrative keyword fallback (ai_agent, animal, no match)
- [x] Unit test: classify_narrative returns (str|None, float) tuple
- [x] Unit test: _compute_onchain_multiplier Jupiter penalty (0.50 for not tradeable)
- [x] Unit test: _compute_onchain_multiplier whale accumulation bonus (1.30)
- [x] Unit test: _compute_safety_penalty whale concentration (0.50 for 80%)
- [x] Feature count: 8 Phase 3B features in train_model.py (69 total)
- [x] DB migration: applied via Supabase MCP
- [ ] Live test: run scraper → check token_snapshots for non-null jup_tradeable + whale_count
- [ ] ML test: `python train_model.py --horizon 12h --trials 5` → 8 new features in metadata

---

## Algorithm v3: Research-Informed Improvements (DONE)

### Sprint A: Quick Wins (no new API calls) — `pipeline.py`
- [x] **A1:** Short-term volume heat (1h/6h acceleration in onchain_multiplier)
- [x] **A2:** Transaction velocity (txn/holder activity density in onchain_multiplier)
- [x] **A3:** Sentiment consistency (std dev of KOL sentiments — ML feature)
- [x] **A4:** Liquidity floor ($10K) + holder floor (30) hard gates
- [x] **A5:** Artificial pump detection (price pump + no organic growth → 0.2x penalty)
- [x] **A6:** Enhanced wash trading (volume spike + flat price divergence signal)

### Sprint B: Architecture Fixes — `pipeline.py`
- [x] **B1:** ML blend (70% ML + 30% manual) instead of pure ML override
- [x] **B2:** Missing data penalty (no factors → 0.7 instead of 1.0)

### Sprint C: ME2F-Inspired ML Features — `pipeline.py`
- [x] **C1:** Volatility proxy (multi-timeframe price change std dev)
- [x] **C2:** Whale dominance (top10_pct * gini — single concentration metric)
- [x] **C3:** Sentiment amplification (sentiment volatility * price reaction)

### Storage & ML Updates
- [x] `scraper/push_to_supabase.py` — 7 new snapshot fields
- [x] `scraper/train_model.py` — 7 new ML features in ALL_FEATURE_COLS
- [x] Supabase migration — 7 new columns on token_snapshots

### Verification
- [x] Syntax: all 3 files compile (ast.parse)
- [x] DB migration: 7 columns verified via information_schema query
- [ ] Live test: run `python safe_scraper.py --once` → check logs for new gates + features
- [ ] After 2+ weeks: retrain ML model, check if new features appear in SHAP top-15

---

## Algorithm v3.1: User Feedback Improvements (DONE)

### 1. 5m Volume Granularity
- [x] `scraper/enrich.py` — Extract `volume_5m` + `buy_sell_ratio_5m` from DexScreener (already in response, was ignored)
- [x] `scraper/pipeline.py` — `ultra_short_heat = (vol_5m * 12) / vol_1h` in onchain_multiplier

### 2. "Already Pumped" Penalty
- [x] `scraper/pipeline.py` — Degressive penalty: 200% → 1.0, 350% → 0.7, 500%+ → 0.4 floor
- [x] Stored as `already_pumped_penalty` for ML feature

### 3. Fix A5 Pump Detection (Supply Control ≠ Manipulation)
- [x] `scraper/pipeline.py` — If `whale_change > 0` during pump, skip artificial pump flag (whales accumulating = bullish supply control)

### 4. Bubblemaps Wallet Clustering
- [x] `scraper/enrich_bubblemaps.py` — NEW FILE: Bubblemaps API integration (decentralization_score, clusters, CEX/DEX supply breakdown)
- [x] `scraper/pipeline.py` — Import + call after Jupiter, safety penalty for low decentralization + large clusters
- [x] Graceful fallback: no API key = silently skipped

### Storage & ML Updates
- [x] `scraper/push_to_supabase.py` — 7 new snapshot fields (volume_5m, buy_sell_ratio_5m, ultra_short_heat, already_pumped_penalty, bubblemaps_score, bubblemaps_cluster_max_pct, bubblemaps_cluster_count)
- [x] `scraper/train_model.py` — 7 new ML features + volume_5m log transform
- [x] Supabase migration — 7 new columns on token_snapshots

### Verification
- [x] Syntax: all 5 files compile (ast.parse)
- [x] DB migration: 7 new columns verified via information_schema query
- [ ] Live test: run scraper → check ultra_short_heat values + already_pumped_penalty in logs
- [ ] Email `api@bubblemaps.io` to request beta API key
- [ ] After key obtained: set `BUBBLEMAPS_API_KEY` in `.env`, verify bubblemaps_score populated

---

## Outcome Tracker Fix: Max Price via OHLCV (DONE)

**Problem:** `outcome_tracker.py` checked the price at a single instant (when `fill_outcomes()` runs). A token that pumped to 3x then dumped back was labeled `did_2x = False`. This polluted ML training data with false negatives.

**Fix:** Use GeckoTerminal OHLCV candles (5-min resolution) to find the **max high price** during the 6h/12h/24h window.

### Implementation
- [x] `scraper/enrich.py` — Extract `pair_address` (pool address) from DexScreener for OHLCV lookups
- [x] `scraper/push_to_supabase.py` — Store `pair_address` in snapshots
- [x] `scraper/outcome_tracker.py` — Full rewrite:
  - GeckoTerminal `/tokens/{addr}/pools` to find pool address (cached 7 days)
  - GeckoTerminal OHLCV 5-min candles → `max(high)` during window
  - Fallback to DexScreener current price if OHLCV fails
  - Rate limiting: 2.1s between GeckoTerminal calls (30 req/min free tier)
  - Stores `max_price_6h`, `max_price_12h`, `max_price_24h` alongside labels
- [x] Supabase migration — `pair_address`, `max_price_6h`, `max_price_12h` columns

### Verification
- [x] Syntax: all 3 modified files compile (ast.parse)
- [x] DB migration: 4 new columns verified
- [ ] Live test: run scraper → wait 6h+ → check logs for "OHLCV max" vs "current price" in outcomes

---

## Algorithm v7: Scoring Improvements (DONE)

### Part 1: Backtest Pipeline
- [x] `scraper/backtest.py` — Standalone CLI: fetch_snapshots, compute_score, evaluate_weights, parameter_sensitivity, walk_forward
- [x] Weight renormalization in score recomputation
- [x] CLI: `python backtest.py [--threshold N] [--sensitivity] [--walk-forward]`

### Part 2: Minsky Lifecycle Phases
- [x] `_classify_lifecycle_phase()` — 5-phase model (displacement/boom/euphoria/profit_taking/panic/unknown)
- [x] Replaces flat `already_pumped_penalty` with phase-aware penalties [0.25, 1.1]
- [x] Boom phase gives 1.1x bonus (only phase with positive multiplier)
- [x] `lifecycle_phase` field on token dict + snapshots

### Part 3: Weakest Component + Interpretation Bands
- [x] `_identify_weakest_component()` — finds lowest-scoring of 5 components
- [x] `_interpret_score()` — strong/good/moderate/weak/low_conviction bands
- [x] TokenCard: colored interpretation badge + weakest component label
- [x] API route: passes weakestComponent + scoreInterpretation to frontend

### Part 4: Missing-Data Weight Renormalization
- [x] `_compute_score_with_renormalization()` — redistributes missing component weight
- [x] `data_confidence` field (1.0 = all data, 0.6 = missing price_action)
- [x] Replaces hardcoded 0.5 price_action placeholder with proper renormalization
- [x] `_compute_onchain_multiplier` no-data fallback 0.7 → 0.85

### DB Migration
- [x] `supabase/migrations/v7_scoring_improvements.sql` — applied via MCP
- [x] 5 new columns on token_snapshots, 3 on tokens
- [x] Updated `get_token_ranking` RPC with new fields

### Verification
- [x] Syntax: pipeline.py, push_to_supabase.py, backtest.py compile (ast.parse)
- [x] DB: all columns verified via information_schema
- [x] RPC: get_token_ranking returns weakest_component, score_interpretation, data_confidence
- [x] Frontend: `npm run build` passes with 0 errors
- [ ] Live test: run scraper → verify lifecycle_phase, weakest_component, data_confidence populated
- [ ] Backtest: `python backtest.py` with labeled snapshots → report generated

---

## Algorithm v5.1: Post-Scraping Fixes (DONE)

### Fix 1: Soften Safety Penalty — `pipeline.py`
- [x] All safety penalty factors softened (higher floors, gentler curves)
- [x] Global floor: `max(0.3, penalty)` — never destroy a token completely
- [x] Key changes: insider 0.2→0.5, top10 0.3→0.7, risk 0.2→0.5, jito 0.2→0.4, gini 0.6→0.8

### Fix 2: Enrichment After Gates — `pipeline.py`
- [x] Moved Helius/Jupiter/Bubblemaps/OHLCV enrichment AFTER quality gates
- [x] DexScreener + RugCheck still runs on ALL tokens (needed for gates)
- [x] Expensive enrichment now targets SURVIVORS only → 100% coverage for displayed tokens

### Fix 3: Numeric Clamping — `push_to_supabase.py`
- [x] Added NUMERIC_LIMITS dict for 15 bounded numeric fields
- [x] `_sanitize_row()` clamps values to prevent `numeric(6,3)` overflow
- [x] Fixes $COMPANY crash on `volatility_proxy`

### Verification
- [x] Syntax: both files compile (ast.parse)
- [ ] Live test: `python safe_scraper.py --once --dump` → safety penalties >= 0.3
- [ ] Live test: top 10 tokens in 7d window have ath_ratio, helius_holder_count, jup_tradeable
- [ ] Live test: no `numeric field overflow` errors in Supabase logs
