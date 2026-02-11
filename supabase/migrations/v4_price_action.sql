-- Algorithm v4 Sprint 1: Price Action columns on token_snapshots
ALTER TABLE token_snapshots ADD COLUMN IF NOT EXISTS ath_24h NUMERIC;
ALTER TABLE token_snapshots ADD COLUMN IF NOT EXISTS ath_ratio NUMERIC;
ALTER TABLE token_snapshots ADD COLUMN IF NOT EXISTS price_action_score NUMERIC;
ALTER TABLE token_snapshots ADD COLUMN IF NOT EXISTS momentum_direction TEXT;
ALTER TABLE token_snapshots ADD COLUMN IF NOT EXISTS support_level NUMERIC;

-- Algorithm v4 Sprint 3: Base scores for micro-refresh
ALTER TABLE tokens ADD COLUMN IF NOT EXISTS base_score INTEGER;
ALTER TABLE tokens ADD COLUMN IF NOT EXISTS base_score_conviction INTEGER;
ALTER TABLE tokens ADD COLUMN IF NOT EXISTS base_score_momentum INTEGER;

-- Algorithm v4 Sprint 4: Whale direction tracking
ALTER TABLE token_snapshots ADD COLUMN IF NOT EXISTS whale_direction TEXT;
