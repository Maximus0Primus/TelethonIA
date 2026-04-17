-- v138: Perfect sim/real alignment via eval_history (per-trade) + cache_snapshots (global).
--
-- B) eval_history JSONB on paper_trades:
--    Each poll in paper_trader appends {t, d, e, h} = (timestamp, decision_price,
--    exec_price, high_seen_at_poll). Persisted on trade close. Sim can replay
--    EXACTLY the prices real paper used -> 0% divergence by construction.
--
-- D) cache_snapshots table:
--    Every loop tick, dump _jupiter_prices_cache + _dex_prices_cache (full state).
--    Sim can reconstruct what the cache held at any moment for ANY token, even
--    those without active trades. Replaces fragile price_ticks reconstruction.

ALTER TABLE paper_trades
    ADD COLUMN IF NOT EXISTS eval_history JSONB;

CREATE TABLE IF NOT EXISTS cache_snapshots (
    id          BIGSERIAL PRIMARY KEY,
    snapshot_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    jp_prices   JSONB NOT NULL,           -- {token_addr: price_usd}
    ds_prices   JSONB,                    -- {token_addr: price_usd}
    n_tokens    INT
);

CREATE INDEX IF NOT EXISTS idx_cache_snapshots_at
    ON cache_snapshots (snapshot_at DESC);

-- Lockdown: service_role only
REVOKE ALL ON cache_snapshots FROM PUBLIC;
REVOKE ALL ON cache_snapshots FROM anon;
GRANT ALL ON cache_snapshots TO service_role;
GRANT USAGE, SELECT ON SEQUENCE cache_snapshots_id_seq TO service_role;
