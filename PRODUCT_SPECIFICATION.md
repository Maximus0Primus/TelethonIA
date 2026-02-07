# Product Specification: Crypto KOL Consensus Ranking Platform

**Version:** 1.0
**Date:** 2026-02-04
**Status:** Ready for Development

---

## Executive Summary

### The Product in One Sentence
> A dead-simple web app that shows a real-time ranking of crypto tokens based on aggregated KOL consensus from 60+ Telegram groups, with adjustable time filters.

### Why This Will Work
| Factor | Evidence |
|--------|----------|
| **Validated Gap** | No tool aggregates KOL consensus with time filters (7 research agents confirmed) |
| **Existing Moat** | 60 pre-scored Telegram groups with conviction ratings (6-10) already configured |
| **Trust Advantage** | 27% trust AI analysis > 21% trust human KOLs (CoinGecko survey, n=2,632) |
| **Timing** | 3-6 month window before market correction (Q1-Q2 2026 optimal) |
| **Low Risk** | Core tech exists, 3-4 weeks to MVP, minimal investment required |

### Key Metrics
| Metric | Target |
|--------|--------|
| **MVP Timeline** | 3-4 weeks (5-10h/week available) |
| **Launch Validation** | 200 active users in 30 days |
| **Pricing** | Free tier + $9.99/month Pro |
| **Year 1 Target** | 2,000-5,000 paying users |
| **Year 1 ARR** | $120K-$360K |

---

## 1. Product Vision

### 1.1 Problem Statement

Crypto traders face three critical problems:

1. **Information Overload**
   - 50+ Telegram groups to monitor
   - Hundreds of messages per day
   - No way to see consensus across KOLs

2. **Trust Deficit**
   - 67% of signal providers are fraudulent
   - $6B+ lost to rug pulls in Q1 2025 alone
   - 97% of day traders lose money

3. **Tool Complexity**
   - DexTools is "overwhelming for beginners"
   - Santiment has 750+ metrics
   - No simple "what should I look at" answer

### 1.2 Solution

**One page. One score. Adjustable time filter.**

The platform aggregates mentions from 60+ curated KOL Telegram groups, calculates a composite score for each token, and displays a simple ranking that answers: **"What are the top KOLs talking about right now?"**

### 1.3 Positioning

| Aspect | Our Approach |
|--------|--------------|
| **Core Message** | "See what top KOLs are actually talking about" |
| **Primary Value** | Simplicity + Aggregation |
| **Secondary Value** | Time savings + Confidence |
| **NOT Positioning As** | Trading signals, financial advice, guaranteed gains |

### 1.4 Differentiation

| Competitor | Their Approach | Our Advantage |
|------------|----------------|---------------|
| **DexScreener** | Price/volume data, no social | KOL consensus layer |
| **LunarCrush** | Twitter/Reddit sentiment | Telegram KOL focus, simpler |
| **DexTools** | Complex trader dashboard | One-page simplicity |
| **Nansen** | Whale tracking, $99-999/mo | 10x cheaper, KOL-focused |
| **Signal Groups** | Pay $100-1000/mo for one KOL | Aggregate 60 KOLs for $9.99 |

---

## 2. Target Market

### 2.1 Primary Market

**International (English-first)**

| Segment | Size | Characteristics |
|---------|------|-----------------|
| **Global crypto traders** | 52M active | Seeking alpha, time-poor |
| **Memecoin focused** | 5-7M | High risk tolerance, FOMO-driven |
| **Would pay for tools** | 260K-780K | Already spending on signals/analytics |

### 2.2 Target User Persona

**"Alex the Aspiring Trader"**

| Attribute | Detail |
|-----------|--------|
| **Demographics** | 25-40 years old, male (70%), tech-savvy |
| **Experience** | 6-24 months in crypto |
| **Portfolio** | $1K-$50K in crypto, 10-30% in memecoins |
| **Time** | 30min-2h/day for crypto research |
| **Pain** | Misses good calls, overwhelmed by groups, fears rug pulls |
| **Current Tools** | DexScreener (free), 5-15 Telegram groups |
| **Willingness to Pay** | $10-50/month for clear value |

### 2.3 User Journey

```
1. DISCOVER
   - Sees Twitter thread "Top tokens by KOL consensus this week"
   - Clicks link to website

2. EXPLORE (Free Tier)
   - Views Top 20 ranking with 24h filter
   - Clicks on token to see basic details
   - Realizes value of aggregated KOL data

3. HIT LIMIT
   - Wants to see 6h or 3h filter (Pro only)
   - Wants to see full top 50 (Pro only)
   - Wants alerts when token enters top 10 (Pro only)

4. CONVERT
   - Signs up for $9.99/month
   - Gets full access + alerts

5. RETAIN
   - Daily habit: check ranking each morning
   - Weekly: adjust portfolio based on consensus shifts
   - Shares discoveries on Twitter (organic growth)
```

---

## 3. Product Specification

### 3.1 Core Feature: The Ranking

**Main View (One Page)**

```
+----------------------------------------------------------+
|  [LOGO]  Crypto KOL Consensus Ranking    [Free] [Sign In] |
+----------------------------------------------------------+
|                                                           |
|  Time Filter:  [3h] [6h] [12h] [24h*] [48h] [7d]         |
|                        * = selected                       |
|                                                           |
+----------------------------------------------------------+
|  #  | Token    | Score | Mentions | KOLs | Trend | Detail |
+----------------------------------------------------------+
|  1  | $PEPE    |  87   |   142    |  23  |  ↑↑   |  [>]   |
|  2  | $WIF     |  82   |   98     |  19  |  ↑    |  [>]   |
|  3  | $BONK    |  79   |   87     |  17  |  →    |  [>]   |
|  4  | $DOGE    |  75   |   134    |  21  |  ↓    |  [>]   |
|  5  | $SHIB    |  72   |   76     |  15  |  →    |  [>]   |
|  ... (20 shown free, 50 for Pro)                         |
+----------------------------------------------------------+
|                                                           |
|  [Upgrade to Pro - $9.99/mo] - All filters, alerts, API  |
|                                                           |
+----------------------------------------------------------+
```

### 3.2 Score Calculation

**Composite Score (0-100)**

The score combines multiple factors into one easy-to-understand number:

```
SCORE = normalize_to_100(
    0.35 × KOL_Consensus +      // How many KOLs mentioned it
    0.25 × Sentiment_Score +     // Positive vs negative sentiment
    0.20 × Conviction_Weight +   // Weighted by KOL reliability (6-10)
    0.15 × Momentum +            // Trend direction (increasing/decreasing)
    0.05 × Breadth               // Spread across different KOL groups
)
```

**Component Breakdown:**

| Component | Weight | Description | Source |
|-----------|--------|-------------|--------|
| **KOL Consensus** | 35% | Number of unique KOLs mentioning token | Telegram message count |
| **Sentiment** | 25% | Positive/negative tone of mentions | CryptoBERT + VADER + Lexicon |
| **Conviction Weight** | 20% | Weighted by KOL reliability score (6-10) | Pre-assigned per group |
| **Momentum** | 15% | Is mention rate increasing or decreasing? | Time-series slope |
| **Breadth** | 5% | Spread across different KOL categories | Group diversity |

**Score Display:**
- Primary: Single score 0-100
- Optional detail (on click): Mini-bars showing component breakdown

### 3.3 Data Sources

**60+ Telegram Groups (Existing TelethonIA Configuration)**

| Conviction | Groups (Examples) | Weight |
|------------|-------------------|--------|
| **10/10** | overdose_gems_calls, cryptorugmuncher, thetonymoontana | 1.5x |
| **9/10** | marcellcooks, PoseidonTAA, Carnagecalls, MarkGems | 1.3x |
| **8/10** | ghastlygems, slingdeez, archercallz, LevisAlpha | 1.1x |
| **7/10** | shahlito, sadcatgamble, veigarcalls | 1.0x |
| **6/10** | houseofdegeneracy | 0.9x |

**Data Pipeline:**
- CRON job runs every 2-4 hours
- Collects last 50 messages per group
- Processes through sentiment pipeline
- Updates database and recalculates scores

### 3.4 Time Filters

| Filter | Free | Pro | Use Case |
|--------|------|-----|----------|
| **3h** | No | Yes | Catch very early momentum |
| **6h** | No | Yes | Short-term trading |
| **12h** | No | Yes | Intraday view |
| **24h** | Yes | Yes | Default daily view |
| **48h** | No | Yes | Longer trends |
| **7d** | No | Yes | Weekly overview |

### 3.5 Token Detail View (Click to Expand)

```
+----------------------------------------------------------+
|  $PEPE - PepeCoin                          Score: 87/100  |
+----------------------------------------------------------+
|                                                           |
|  Component Breakdown:                                     |
|  [████████░░] Consensus: 82%                             |
|  [███████░░░] Sentiment: 71%                             |
|  [█████████░] Conviction: 89%                            |
|  [██████░░░░] Momentum: 65%                              |
|  [████████░░] Breadth: 78%                               |
|                                                           |
|  Stats (24h):                                            |
|  • 142 mentions across 23 KOLs                           |
|  • Average sentiment: +0.42 (Positive)                   |
|  • Trend: ↑↑ Strong increase                             |
|                                                           |
|  Top KOL Mentions:                                        |
|  • overdose_gems (10/10): 12 mentions, +0.8 sentiment    |
|  • marcellcooks (9/10): 8 mentions, +0.6 sentiment       |
|  • LevisAlpha (8/10): 6 mentions, +0.5 sentiment         |
|                                                           |
|  [View on DexScreener] [View on CoinGecko]               |
|                                                           |
+----------------------------------------------------------+
```

---

## 4. Technical Architecture

### 4.1 Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| **Frontend** | Next.js 14 + React | Modern, SEO-friendly, fast |
| **Styling** | Tailwind CSS | Rapid UI development |
| **Hosting** | Vercel | Free tier, easy deployment |
| **Database** | Supabase (PostgreSQL) | Free tier, real-time, auth built-in |
| **Auth** | Supabase Auth | Email + Google OAuth |
| **Payments** | Stripe | Industry standard, easy integration |
| **Data Pipeline** | Python (existing) | Reuse TelethonIA codebase |
| **CRON** | Vercel Cron or Railway | Scheduled data collection |

### 4.2 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA COLLECTION                          │
├─────────────────────────────────────────────────────────────┤
│  CRON Job (every 2-4h)                                      │
│  └─> exportfinaljson.py (TelethonIA)                        │
│      └─> 60+ Telegram Groups                                │
│          └─> Raw messages JSON                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     PROCESSING                               │
├─────────────────────────────────────────────────────────────┤
│  Python Pipeline (utils.py reuse)                           │
│  ├─> Token extraction (regex + alias detection)             │
│  ├─> Sentiment analysis (CryptoBERT + VADER + Lexicon)      │
│  ├─> Score calculation (composite formula)                  │
│  └─> Save to Supabase                                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     DATABASE (Supabase)                      │
├─────────────────────────────────────────────────────────────┤
│  Tables:                                                     │
│  ├─> tokens (symbol, score, mentions, sentiment, updated_at)│
│  ├─> messages (id, group, token, sentiment, timestamp)      │
│  ├─> groups (name, conviction, category)                    │
│  ├─> users (id, email, plan, created_at)                    │
│  └─> subscriptions (user_id, stripe_id, status)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     FRONTEND (Next.js)                       │
├─────────────────────────────────────────────────────────────┤
│  Pages:                                                      │
│  ├─> / (Landing + Ranking)                                  │
│  ├─> /token/[symbol] (Token detail)                         │
│  ├─> /pricing (Pro upgrade)                                 │
│  ├─> /login (Auth)                                          │
│  └─> /dashboard (Pro features)                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 Database Schema

```sql
-- Tokens table (main ranking data)
CREATE TABLE tokens (
  id SERIAL PRIMARY KEY,
  symbol VARCHAR(20) NOT NULL UNIQUE,
  score DECIMAL(5,2),
  mentions INTEGER,
  unique_kols INTEGER,
  sentiment DECIMAL(4,3),
  momentum DECIMAL(4,3),
  breadth DECIMAL(4,3),
  conviction_weighted DECIMAL(5,2),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Messages table (raw data)
CREATE TABLE messages (
  id SERIAL PRIMARY KEY,
  telegram_id BIGINT,
  group_name VARCHAR(100),
  token VARCHAR(20),
  text TEXT,
  sentiment DECIMAL(4,3),
  timestamp TIMESTAMP,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Groups table (KOL configuration)
CREATE TABLE groups (
  id SERIAL PRIMARY KEY,
  name VARCHAR(100) NOT NULL UNIQUE,
  conviction INTEGER CHECK (conviction >= 6 AND conviction <= 10),
  category VARCHAR(50),
  remark TEXT
);

-- Users table
CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  email VARCHAR(255) NOT NULL UNIQUE,
  plan VARCHAR(20) DEFAULT 'free',
  created_at TIMESTAMP DEFAULT NOW()
);

-- Subscriptions table
CREATE TABLE subscriptions (
  id SERIAL PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  stripe_customer_id VARCHAR(100),
  stripe_subscription_id VARCHAR(100),
  status VARCHAR(20),
  current_period_end TIMESTAMP
);
```

### 4.4 API Endpoints

```
GET  /api/ranking
     ?timeframe=24h (3h, 6h, 12h, 24h, 48h, 7d)
     ?limit=20 (free) or 50 (pro)
     Returns: Array of tokens with scores

GET  /api/token/[symbol]
     Returns: Detailed token data with component breakdown

POST /api/auth/login
     Body: { email, password }
     Returns: JWT token

POST /api/stripe/create-checkout
     Body: { priceId }
     Returns: Stripe checkout URL

POST /api/stripe/webhook
     Handles subscription events
```

---

## 5. Monetization

### 5.1 Pricing Tiers

| Feature | Free | Pro ($9.99/mo) |
|---------|------|----------------|
| Top tokens visible | 20 | 50 |
| Time filters | 24h only | All (3h to 7d) |
| Token detail view | Basic | Full breakdown |
| Refresh rate | Every 4h | Every 2h |
| Alerts | No | Email + (future) TG |
| API access | No | Yes |
| Ads | Yes | No |

### 5.2 Revenue Projections

**Conservative Case (3-4 week MVP + 30-day validation):**

| Month | Free Users | Conversion | Paid Users | MRR |
|-------|------------|------------|------------|-----|
| 1 | 500 | 3% | 15 | $150 |
| 3 | 3,000 | 3.5% | 105 | $1,050 |
| 6 | 10,000 | 4% | 400 | $4,000 |
| 12 | 30,000 | 4% | 1,200 | $12,000 |

**Year 1 ARR (Conservative):** ~$100K

**Optimistic Case (with viral growth):**

| Month | Free Users | Conversion | Paid Users | MRR |
|-------|------------|------------|------------|-----|
| 1 | 1,000 | 4% | 40 | $400 |
| 3 | 8,000 | 5% | 400 | $4,000 |
| 6 | 25,000 | 5% | 1,250 | $12,500 |
| 12 | 80,000 | 5% | 4,000 | $40,000 |

**Year 1 ARR (Optimistic):** ~$350K

### 5.3 Unit Economics

| Metric | Value | Notes |
|--------|-------|-------|
| **Price** | $9.99/mo | Aggressive for volume |
| **Gross Margin** | ~90% | Minimal infrastructure cost |
| **CAC Target** | <$15 | Organic + content marketing |
| **LTV (6mo avg)** | ~$50 | 5-month average retention |
| **LTV/CAC** | >3x | Healthy ratio |

---

## 6. Development Roadmap

### 6.1 MVP (Weeks 1-4)

**Week 1: Foundation**
- [ ] Set up Next.js project with Tailwind
- [ ] Create Supabase database with schema
- [ ] Implement basic ranking page UI
- [ ] Set up Vercel deployment

**Week 2: Data Pipeline**
- [ ] Adapt TelethonIA export script for CRON
- [ ] Implement score calculation in Python
- [ ] Create API endpoint for ranking data
- [ ] Connect frontend to API

**Week 3: Core Features**
- [ ] Implement time filter functionality
- [ ] Add token detail view
- [ ] Implement basic auth (Supabase)
- [ ] Add free tier limitations

**Week 4: Monetization + Polish**
- [ ] Integrate Stripe checkout
- [ ] Implement Pro tier access control
- [ ] UI polish and mobile responsiveness
- [ ] Deploy and test end-to-end

### 6.2 Post-MVP (Month 2-3)

**If 200+ active users achieved:**
- [ ] Email alerts for Pro users
- [ ] Historical charts per token
- [ ] More granular time filters
- [ ] Performance optimization

### 6.3 Future (Month 4-6)

**If growth continues:**
- [ ] Telegram bot for alerts
- [ ] API access for Pro users
- [ ] Mobile app consideration
- [ ] Additional KOL sources (Twitter)

---

## 7. Go-to-Market

### 7.1 Launch Strategy

**Phase 1: Soft Launch (Week 4)**
- Deploy MVP to production
- Share with 10-20 friendly crypto traders for feedback
- Fix critical bugs

**Phase 2: Public Launch (Week 5)**
- Post on Twitter with screenshots
- Share on relevant subreddits
- No paid marketing initially

### 7.2 Organic Growth Channels

| Channel | Effort | Expected Impact |
|---------|--------|-----------------|
| **Crypto Twitter** | Low | High - viral potential |
| **Reddit** (r/CryptoCurrency, r/memecoin) | Low | Medium |
| **ProductHunt** | Medium | Medium - one-time spike |
| **SEO** | Medium | Long-term traffic |

### 7.3 Content Strategy (Post-Validation)

**Weekly (after 200 users):**
- Twitter thread: "Top 5 tokens by KOL consensus this week"
- Screenshot of ranking with commentary

**This builds:**
- Organic backlinks
- Social proof
- User acquisition funnel

---

## 8. Success Metrics

### 8.1 Validation Criteria (30 Days Post-Launch)

| Metric | Target | Decision |
|--------|--------|----------|
| **Active users** | 200+ | Continue |
| **Active users** | 50-199 | Iterate |
| **Active users** | <50 | Pivot or abandon |

**Definition of "Active":** Visited site at least 2x in the past 7 days

### 8.2 Key Metrics to Track

| Category | Metric | Tool |
|----------|--------|------|
| **Acquisition** | Daily signups | Supabase |
| **Activation** | % who view 3+ tokens | PostHog/Vercel Analytics |
| **Retention** | Weekly active users | PostHog |
| **Revenue** | MRR, conversion rate | Stripe Dashboard |
| **Engagement** | Avg session duration | Vercel Analytics |

---

## 9. Risks & Mitigations

### 9.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Telegram API rate limits** | Medium | High | Respect limits, queue requests, cache aggressively |
| **Data freshness issues** | Medium | Medium | Clear "last updated" timestamp, user expectations |
| **Sentiment accuracy** | Low | Medium | Hybrid model already tested, iterate based on feedback |

### 9.2 Market Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Market crash Q3-Q4 2026** | High (60%) | High | Launch fast, validate in Q1-Q2, pivot if needed |
| **Competitor launches similar** | Medium | Medium | Speed to market, community building |
| **Low conversion rate** | Medium | High | Iterate on value prop, test different limitations |

### 9.3 Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Time availability** | Medium | Medium | Keep scope minimal, ruthless prioritization |
| **Burnout** | Low | High | Side project pace, don't over-commit |

---

## 10. Legal & Compliance

### 10.1 Disclaimers Required

**On every page:**
> "This platform aggregates publicly available information from crypto communities. It is not financial advice. Always do your own research before making investment decisions."

**On ranking page:**
> "Scores are calculated algorithmically based on KOL mention frequency and sentiment. Past performance does not guarantee future results."

### 10.2 Terms of Service

- No financial advice provided
- Data is aggregated from public sources
- No guarantee of accuracy
- User assumes all trading risks

### 10.3 Privacy Policy

- Minimal data collection (email, usage analytics)
- No selling of user data
- GDPR compliant (email deletion on request)

---

## 11. Decision Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Positioning** | Simplicity + Aggregated Score | Research showed users want "one answer", not 750 metrics |
| **Market** | International EN | Larger TAM, FR too small alone |
| **Pricing** | $9.99/mo freemium | Aggressive for volume, 42% of FR users earn <30k |
| **Distribution** | Web first, TG later | Reduce initial scope |
| **MVP Features** | Ranking + time filters only | Minimal viable, validate fast |
| **Stack** | Next.js + Vercel + Supabase | Modern, free tiers, fast deployment |
| **Data Source** | Existing 60 TG groups | Unique moat, no rebuild needed |
| **Transparency** | Partial methodology | Balance trust vs competitive advantage |
| **Timeline** | 3-4 weeks MVP | Realistic for 5-10h/week |
| **Success Metric** | 200 active users in 30 days | Clear validation threshold |

---

## 12. Next Steps

### Immediate Actions (This Week)

1. [ ] **Choose product name** (you decide)
2. [ ] **Set up development environment**
   - Create Next.js project
   - Set up Supabase account
   - Create Vercel account
3. [ ] **Adapt TelethonIA pipeline**
   - Modify export script for automated runs
   - Test score calculation logic
4. [ ] **Domain + branding**
   - Register domain
   - Create basic logo (can use AI tools)

### Week 1 Goals

- [ ] Basic ranking page displaying mock data
- [ ] Database schema created and seeded
- [ ] Data pipeline running locally
- [ ] Deployed to Vercel (even if incomplete)

---

## Appendix A: Research Summary

This specification is based on comprehensive research from 7 parallel agents:

| Agent | Key Finding |
|-------|-------------|
| **Pain Hunter** | 20+ pain points found, but users leaving market |
| **Competitor Analyst** | No KOL consensus tool exists - validated gap |
| **Trend Scout** | 3-6 month window optimal, crash likely Q3-Q4 2026 |
| **Market Sizing** | 2-5K paid users Year 1 realistic |
| **User Psychology** | Trust AI (27%) > Trust humans (21%) |
| **French Market** | FR too small, go international |
| **Community Sources** | Transparency = key differentiator |

**Full research available in:** `tasks/research/`

---

## Appendix B: Existing Code Assets

**From TelethonIA (ready to reuse):**

| File | Purpose | Reuse |
|------|---------|-------|
| `exportfinaljson.py` | Telegram data collection | Direct |
| `utils.py` | Token extraction, sentiment, scoring | Adapt |
| `sentiment_local.py` | CryptoBERT wrapper | Direct |
| `group_cache.json` | 60 group IDs | Direct |

**Estimated code reuse:** 60-70% of data pipeline

---

## Appendix C: Competitor Reference

| Tool | Price | Gap We Fill |
|------|-------|-------------|
| DexScreener | Free | No KOL data |
| LunarCrush | $24.99/mo | No TG KOLs, complex |
| Nansen | $99-999/mo | Too expensive |
| Signal groups | $100-1000/mo | One KOL only |
| **Us** | $9.99/mo | 60 KOLs, simple, affordable |

---

**Document Status:** Ready for Development
**Next Review:** After MVP launch (Week 5)
