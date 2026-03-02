-- v88: bot_ml_predictions — precomputed ML gate predictions (GH Actions → Supabase → VPS)
-- VPS has no xgboost/lightgbm, so GH Actions predicts and stores results here.

CREATE TABLE IF NOT EXISTS bot_ml_predictions (
    token_address TEXT NOT NULL,
    strategy TEXT NOT NULL,
    win_prob REAL NOT NULL,
    gate_mult REAL NOT NULL,         -- 0.0 (skip), 0.5 (half), or 1.0 (full)
    model_horizon TEXT,
    predicted_at TIMESTAMPTZ DEFAULT NOW(),
    snapshot_id BIGINT,
    CONSTRAINT uq_bot_pred UNIQUE (token_address, strategy)
);

CREATE INDEX IF NOT EXISTS idx_bot_pred_at ON bot_ml_predictions(predicted_at);

ALTER TABLE bot_ml_predictions ENABLE ROW LEVEL SECURITY;
-- No public policies = deny-all for anon/public. Only service_role can access.
