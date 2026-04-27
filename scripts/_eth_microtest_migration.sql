-- v14e.32+ ETH live instrumentation — adds 6 columns to paper_trades for
-- fine-grained slippage / gas / latency analysis on ETH live trades.
--
-- Why each column:
--   gas_usd_buy/sell     : real gas paid in USD per side (already returned by
--                          execute_buy/sell, was discarded after a log line).
--                          Lets us compute "% of position eaten by gas" per
--                          trade to validate the $50 floor in real conditions.
--
--   quote_slip_bps_buy/sell : Uniswap quote vs actual fill (price-impact /
--                             router-internal slippage). DIFFERENT from the
--                             existing buy_slippage_bps which is DexScreener
--                             mid vs fill (data-quality slippage). Decoupling
--                             them isolates "is DexScreener wrong?" from
--                             "did the pool slip?".
--
--   block_number_buy/sell : block at which receipt landed. Combined with
--                           buy_exec_ms gives latency profile + lets us
--                           check for sandwich attacks (e.g. our trade is
--                           between two from the same searcher).
--
-- Run this BEFORE deploying the live_trader_eth.py patch. The patch is
-- defensive — it'll degrade gracefully if cols are missing — but you'll lose
-- the data. Idempotent: ADD COLUMN IF NOT EXISTS.

ALTER TABLE paper_trades
  ADD COLUMN IF NOT EXISTS gas_usd_buy        NUMERIC,
  ADD COLUMN IF NOT EXISTS gas_usd_sell       NUMERIC,
  ADD COLUMN IF NOT EXISTS quote_slip_bps_buy   INTEGER,
  ADD COLUMN IF NOT EXISTS quote_slip_bps_sell  INTEGER,
  ADD COLUMN IF NOT EXISTS block_number_buy   BIGINT,
  ADD COLUMN IF NOT EXISTS block_number_sell  BIGINT;

-- Quick verification (run after):
-- SELECT column_name FROM information_schema.columns
-- WHERE table_name='paper_trades'
--   AND column_name IN ('gas_usd_buy','gas_usd_sell',
--                       'quote_slip_bps_buy','quote_slip_bps_sell',
--                       'block_number_buy','block_number_sell');
