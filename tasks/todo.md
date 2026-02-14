# Active TODO — Feb 14, 2026

## Current State
- **Scraper:** Running every 30min via GH Actions + 3min price refresh
- **Snapshots:** ~1454 total, 410 labeled 12h. Now captures gated tokens too (gate_reason column).
- **ML Model:** Trained (80% precision@5 on 145 samples). Auto-retrain enabled (500+ labels, 7-day cooldown).
- **Algorithm:** v17 scoring — anti-pump fixes (pump_momentum_pen, consensus_discount, activity_cap, euphoria range, combined cap 2.0)
- **Auto-learning loop:** CLOSED. scoring_config table → pipeline.py reads weights → auto_backtest writes optimal → Tuning Lab can override.
- **Frontend:** Next.js 16 + Tuning Lab (Apply to Prod button) + KOL Leaderboard + enriched Token Detail
- **Data quality:** extraction_method + extracted_cas per mention, peak_hour per outcome, min_price, time_to_2x, SOL price, oldest_mention_hours, price_at_first_call
- **KOL scoring:** Dynamic win rates from live RPC (replaces static KOL_SCORES). CA-only tracking in kol_scorer.py + RPC + backtest.

---

## Auto-Learning Loop Status (v18)
- [x] `scoring_config` table in Supabase (single-row, audited)
- [x] `pipeline.py` reads weights from DB at cycle start (fallback: hardcoded)
- [x] `auto_backtest.py` writes optimal weights when 100+ labels + >5pp improvement
- [x] `train_model.py --auto` mode with A/B comparison
- [x] `safe_scraper.py` auto-retrain (500+ labels, 7-day cooldown)
- [x] `/api/tuning/config` GET/POST endpoint
- [x] Tuning Lab "Apply to Production" button
- [ ] **Waiting:** 100 unique labeled tokens → auto_backtest starts optimizing weights
- [ ] **Waiting:** 500 labeled snapshots → auto-retrain triggers

---

## Blocked (waiting for data)
- [ ] **ML auto-retrain** — Need 500+ labeled snapshots (~5 days from now). Auto-train enabled.
- [ ] **Weight auto-adjustment** — Need 100+ unique labeled tokens. Auto-apply enabled.
- [ ] **Bubblemaps enrichment** — Need API key (email api@bubblemaps.io)

---

## Available Now

### Stripe integration / paywall
**Effort:** Large. **Impact:** Revenue.
