-- v10: scrape_history table + audit fixes (v95)

-- Fix 7: scrape_history (append-only cycle log)
CREATE TABLE IF NOT EXISTS scrape_history (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    cycle_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    duration_seconds INT,
    tokens_scored INT,
    tokens_enriched INT,
    tokens_pushed INT,
    total_kols INT,
    total_mentions NUMERIC(10,2),
    avg_sentiment NUMERIC(5,2),
    errors JSONB DEFAULT '[]',
    windows JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_scrape_history_cycle_ts ON scrape_history(cycle_ts DESC);
ALTER TABLE scrape_history ENABLE ROW LEVEL SECURITY;

-- Fix 2: Disable ML multiplier (anti-predictive -9.3% PnL)
UPDATE scoring_config SET ml_mult_floor = 1.0, ml_mult_cap = 1.0,
  updated_at = NOW(), updated_by = 'audit_v95' WHERE id = 1;

-- Fix 4: Soften momentum_config pvfc penalties (75.5% at floor)
UPDATE scoring_config
SET momentum_config = jsonb_set(jsonb_set(jsonb_set(jsonb_set(jsonb_set(
  momentum_config,
  '{pvfc_penalty_3x}', '0.75'),
  '{pvfc_penalty_2x}', '0.85'),
  '{pvfc_penalty_1_5x}', '0.95'),
  '{late_penalty_360min}', '0.90'),
  '{floor}', '0.65'),
  updated_at = NOW(), updated_by = 'audit_v95'
WHERE id = 1;
