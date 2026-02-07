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
