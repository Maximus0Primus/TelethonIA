# Active TODO — Feb 14, 2026

## Current State
- **Scraper:** Running every 30min via GH Actions + 3min price refresh
- **Snapshots:** ~1454 total, 410 labeled 12h. Now captures gated tokens too (gate_reason column).
- **ML Model:** Trained (80% precision@5 on 145 samples) but NOT deployed — waiting for 500+ samples
- **Algorithm:** v16 scoring, 59 KOL groups (13 S-tier, 46 A-tier)
- **Frontend:** Next.js 16 + Tuning Lab + KOL Leaderboard (dynamic scores + CA toggle) + enriched Token Detail
- **Data quality:** extraction_method + extracted_cas per mention, peak_hour per outcome, min_price, time_to_2x, SOL price, oldest_mention_hours, price_at_first_call
- **KOL scoring:** Dynamic win rates from live RPC (replaces static KOL_SCORES). CA-only tracking in kol_scorer.py + RPC + backtest.

---

## Blocked (waiting for data)
- [ ] **ML production blend** — Need 500+ labeled snapshots (~5 days). Model exists, just not deployed.
- [ ] **Weekly training GH Action** — Pointless until data accumulates
- [ ] **Bubblemaps enrichment** — Need API key (email api@bubblemaps.io)

---

## Available Now

### Stripe integration / paywall
**Effort:** Large. **Impact:** Revenue.
