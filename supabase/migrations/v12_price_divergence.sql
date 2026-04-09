-- v12/v122: Track paper vs live price divergence for bonding curve analysis
-- paper_exit_price: DexScreener-based exit price (from _evaluate_trade_exit) saved on live trades
-- price_divergence_pct: (jupiter_exit / paper_exit - 1) — positive = Jupiter got a better price

ALTER TABLE paper_trades
  ADD COLUMN IF NOT EXISTS paper_exit_price DOUBLE PRECISION,
  ADD COLUMN IF NOT EXISTS price_divergence_pct DOUBLE PRECISION;

COMMENT ON COLUMN paper_trades.paper_exit_price IS 'DexScreener-based exit price for live trades (what paper would have gotten)';
COMMENT ON COLUMN paper_trades.price_divergence_pct IS 'Divergence: (jupiter_exit / paper_exit - 1). Positive = live outperformed paper.';
