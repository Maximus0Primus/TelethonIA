-- v6: Add pandas-ta technical indicator columns to token_snapshots
-- RSI, MACD histogram, Bollinger Bands (width + %B), OBV slope

ALTER TABLE token_snapshots
  ADD COLUMN IF NOT EXISTS rsi_14 NUMERIC,
  ADD COLUMN IF NOT EXISTS macd_histogram NUMERIC,
  ADD COLUMN IF NOT EXISTS bb_width NUMERIC,
  ADD COLUMN IF NOT EXISTS bb_pct_b NUMERIC,
  ADD COLUMN IF NOT EXISTS obv_slope_norm NUMERIC;
