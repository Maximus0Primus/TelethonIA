-- v143.6 — persist paper-accurate PnL per live trade row.
-- live_trader already computes _paper_sim_ev (paper_trader's exact pipeline:
-- dynamic slip + SELL_FEE_BPS + Ultra SELL quote override) for divergence
-- measurement. Storing it alongside pnl_pct unlocks direct per-trade
-- divergence analysis without re-running verify_sim_live_alignment.py.
ALTER TABLE paper_trades
  ADD COLUMN IF NOT EXISTS paper_sim_pnl_pct numeric(10, 4);

COMMENT ON COLUMN paper_trades.paper_sim_pnl_pct IS
  'Live rows only. PnL that paper_trader would have booked at the same '
  'decision tick (dynamic slip + sell fee + Ultra SELL quote). Divergence '
  'vs pnl_pct isolates the slippage+routing impact from the decision ev.';
