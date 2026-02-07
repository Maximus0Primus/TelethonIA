-- KOL Consensus Database Schema
-- This schema follows the security rules: RLS enabled, no public policies

-- ============================================
-- TOKENS TABLE
-- Stores the current ranking for each token per time window
-- ============================================
CREATE TABLE IF NOT EXISTS tokens (
  id SERIAL PRIMARY KEY,
  symbol VARCHAR(20) NOT NULL,
  score INTEGER CHECK (score >= 0 AND score <= 100),
  mentions INTEGER DEFAULT 0,
  unique_kols INTEGER DEFAULT 0,
  sentiment DECIMAL(4,3) CHECK (sentiment >= -1 AND sentiment <= 1),
  momentum DECIMAL(4,3) CHECK (momentum >= -1 AND momentum <= 1),
  breadth DECIMAL(4,3) CHECK (breadth >= 0 AND breadth <= 1),
  conviction_weighted DECIMAL(5,2),
  trend VARCHAR(10) CHECK (trend IN ('up', 'down', 'stable')),
  time_window VARCHAR(10) NOT NULL CHECK (time_window IN ('3h', '6h', '12h', '24h', '48h', '7d')),
  change_24h DECIMAL(8,2),
  change_7d DECIMAL(8,2),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  UNIQUE(symbol, time_window)
);

-- Index for fast queries by time window and score
CREATE INDEX IF NOT EXISTS idx_tokens_window_score ON tokens(time_window, score DESC);
CREATE INDEX IF NOT EXISTS idx_tokens_symbol ON tokens(symbol);

-- Enable RLS (will deny all access by default since we create no policies)
ALTER TABLE tokens ENABLE ROW LEVEL SECURITY;

-- ============================================
-- GROUPS TABLE
-- Configuration for KOL Telegram groups
-- ============================================
CREATE TABLE IF NOT EXISTS groups (
  id SERIAL PRIMARY KEY,
  name VARCHAR(100) NOT NULL UNIQUE,
  telegram_id BIGINT,
  telegram_username VARCHAR(100),
  conviction INTEGER CHECK (conviction >= 6 AND conviction <= 10),
  category VARCHAR(50),
  active BOOLEAN DEFAULT true,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE groups ENABLE ROW LEVEL SECURITY;

-- ============================================
-- MENTIONS TABLE
-- Raw mention data for detailed analysis
-- ============================================
CREATE TABLE IF NOT EXISTS mentions (
  id SERIAL PRIMARY KEY,
  symbol VARCHAR(20) NOT NULL,
  group_id INTEGER REFERENCES groups(id),
  message_id BIGINT,
  message_text TEXT,
  sentiment DECIMAL(4,3),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for querying mentions
CREATE INDEX IF NOT EXISTS idx_mentions_symbol ON mentions(symbol);
CREATE INDEX IF NOT EXISTS idx_mentions_created_at ON mentions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_mentions_group ON mentions(group_id);

-- Enable RLS
ALTER TABLE mentions ENABLE ROW LEVEL SECURITY;

-- ============================================
-- PROFILES TABLE
-- Extends Supabase auth.users with app-specific data
-- ============================================
CREATE TABLE IF NOT EXISTS profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  email VARCHAR(255),
  plan VARCHAR(20) DEFAULT 'free' CHECK (plan IN ('free', 'pro', 'enterprise')),
  stripe_customer_id VARCHAR(100),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only read their own profile
CREATE POLICY "Users can view own profile" ON profiles
  FOR SELECT USING (auth.uid() = id);

-- ============================================
-- SUBSCRIPTIONS TABLE
-- Stripe subscription tracking
-- ============================================
CREATE TABLE IF NOT EXISTS subscriptions (
  id SERIAL PRIMARY KEY,
  user_id UUID REFERENCES profiles(id) ON DELETE CASCADE,
  stripe_subscription_id VARCHAR(100) UNIQUE,
  stripe_price_id VARCHAR(100),
  status VARCHAR(20) CHECK (status IN ('active', 'canceled', 'past_due', 'incomplete', 'trialing')),
  current_period_start TIMESTAMP WITH TIME ZONE,
  current_period_end TIMESTAMP WITH TIME ZONE,
  cancel_at_period_end BOOLEAN DEFAULT false,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for user lookups
CREATE INDEX IF NOT EXISTS idx_subscriptions_user ON subscriptions(user_id);
CREATE INDEX IF NOT EXISTS idx_subscriptions_status ON subscriptions(status);

-- Enable RLS
ALTER TABLE subscriptions ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only read their own subscriptions
CREATE POLICY "Users can view own subscriptions" ON subscriptions
  FOR SELECT USING (auth.uid() = user_id);

-- ============================================
-- API KEYS TABLE (for Pro users)
-- ============================================
CREATE TABLE IF NOT EXISTS api_keys (
  id SERIAL PRIMARY KEY,
  user_id UUID REFERENCES profiles(id) ON DELETE CASCADE,
  key_hash VARCHAR(64) NOT NULL UNIQUE, -- SHA-256 hash of the API key
  name VARCHAR(100),
  last_used_at TIMESTAMP WITH TIME ZONE,
  requests_today INTEGER DEFAULT 0,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for key lookups
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id);

-- Enable RLS
ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only read their own API keys
CREATE POLICY "Users can view own api keys" ON api_keys
  FOR SELECT USING (auth.uid() = user_id);

-- ============================================
-- FUNCTIONS
-- ============================================

-- Function to auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to relevant tables
CREATE TRIGGER update_profiles_updated_at
  BEFORE UPDATE ON profiles
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_subscriptions_updated_at
  BEFORE UPDATE ON subscriptions
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_groups_updated_at
  BEFORE UPDATE ON groups
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_tokens_updated_at
  BEFORE UPDATE ON tokens
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

-- Function to get ranking (called from API routes with service_role)
CREATE OR REPLACE FUNCTION get_token_ranking(
  p_time_window VARCHAR(10) DEFAULT '24h',
  p_limit INTEGER DEFAULT 10,
  p_offset INTEGER DEFAULT 0
)
RETURNS TABLE (
  rank BIGINT,
  symbol VARCHAR(20),
  score INTEGER,
  mentions INTEGER,
  unique_kols INTEGER,
  sentiment DECIMAL(4,3),
  trend VARCHAR(10),
  change_24h DECIMAL(8,2)
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    ROW_NUMBER() OVER (ORDER BY t.score DESC) as rank,
    t.symbol,
    t.score,
    t.mentions,
    t.unique_kols,
    t.sentiment,
    t.trend,
    t.change_24h
  FROM tokens t
  WHERE t.time_window = p_time_window
  ORDER BY t.score DESC
  LIMIT p_limit
  OFFSET p_offset;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Lock down the function to service_role only
REVOKE EXECUTE ON FUNCTION get_token_ranking FROM public;
REVOKE EXECUTE ON FUNCTION get_token_ranking FROM anon;
GRANT EXECUTE ON FUNCTION get_token_ranking TO service_role;

-- ============================================
-- SCRAPE METADATA TABLE
-- Singleton row tracking the latest scrape cycle
-- ============================================
CREATE TABLE IF NOT EXISTS scrape_metadata (
  id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  stats JSONB DEFAULT '{}'
);

ALTER TABLE scrape_metadata ENABLE ROW LEVEL SECURITY;

-- Seed the singleton row
INSERT INTO scrape_metadata (id) VALUES (1) ON CONFLICT DO NOTHING;

-- ============================================
-- INITIAL DATA: Sample groups from TelethonIA
-- ============================================
INSERT INTO groups (name, conviction, category, active) VALUES
('CryptoKingSignals', 10, 'alpha', true),
('MoonHunterCalls', 9, 'alpha', true),
('AlphaSeekerPro', 9, 'alpha', true),
('GemFinderVIP', 8, 'gems', true),
('WhaleWatchAlerts', 8, 'whales', true),
('MemecoinMasters', 8, 'memes', true),
('DeFiDegen', 7, 'defi', true),
('NFTAlpha', 7, 'nft', true),
('SolanaGems', 7, 'solana', true),
('BaseBuilders', 7, 'base', true)
ON CONFLICT (name) DO NOTHING;

-- ============================================
-- TOKEN SNAPSHOTS TABLE (ML Training Data)
-- Each row = one token at one point in time, with features + eventual outcome labels
-- ============================================
CREATE TABLE IF NOT EXISTS token_snapshots (
  id SERIAL PRIMARY KEY,
  symbol VARCHAR(20) NOT NULL,
  snapshot_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  -- Telegram features (from pipeline.py)
  mentions INTEGER,
  sentiment DECIMAL(4,3),
  breadth DECIMAL(4,3),
  avg_conviction DECIMAL(4,2),
  recency_score DECIMAL(4,3),

  -- On-chain features (DexScreener)
  volume_24h DECIMAL(18,2),
  liquidity_usd DECIMAL(18,2),
  market_cap DECIMAL(18,2),
  txn_count_24h INTEGER,
  price_change_1h DECIMAL(8,4),

  -- Safety features (RugCheck)
  risk_score INTEGER,
  top10_holder_pct DECIMAL(5,2),
  insider_pct DECIMAL(5,2),

  -- Price at snapshot (for outcome calculation)
  price_at_snapshot DECIMAL(18,10),
  token_address VARCHAR(60),

  -- Outcome labels (filled later by outcome_tracker)
  price_after_6h DECIMAL(18,10),
  price_after_12h DECIMAL(18,10),
  price_after_24h DECIMAL(18,10),
  max_price_24h DECIMAL(18,10),
  did_2x_6h BOOLEAN,
  did_2x_12h BOOLEAN,
  did_2x_24h BOOLEAN
);

CREATE INDEX IF NOT EXISTS idx_snapshots_pending ON token_snapshots(snapshot_at)
  WHERE did_2x_24h IS NULL;
CREATE INDEX IF NOT EXISTS idx_snapshots_labeled ON token_snapshots(snapshot_at)
  WHERE did_2x_24h IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_snapshots_symbol ON token_snapshots(symbol);

ALTER TABLE token_snapshots ENABLE ROW LEVEL SECURITY;
