-- v14 — Multi-chain support. Add chain column to all token/trade tables.
-- All existing rows default to 'solana'. ETH rows will insert 'ethereum'.
-- CHECK constraint locks the enum — unrecognized chains rejected at write time.
--
-- Index strategy: compound (chain, token_address) because 99% of reads go
-- "find this token on this chain". Keeps Solana queries fast while isolating
-- ETH rows when filtered.

-- paper_trades
ALTER TABLE paper_trades
  ADD COLUMN IF NOT EXISTS chain TEXT NOT NULL DEFAULT 'solana'
    CHECK (chain IN ('solana', 'ethereum'));
CREATE INDEX IF NOT EXISTS idx_paper_trades_chain_token
  ON paper_trades (chain, token_address);

-- token_snapshots
ALTER TABLE token_snapshots
  ADD COLUMN IF NOT EXISTS chain TEXT NOT NULL DEFAULT 'solana'
    CHECK (chain IN ('solana', 'ethereum'));
CREATE INDEX IF NOT EXISTS idx_token_snapshots_chain_token
  ON token_snapshots (chain, token_address);

-- tokens
ALTER TABLE tokens
  ADD COLUMN IF NOT EXISTS chain TEXT NOT NULL DEFAULT 'solana'
    CHECK (chain IN ('solana', 'ethereum'));
CREATE INDEX IF NOT EXISTS idx_tokens_chain_symbol
  ON tokens (chain, symbol);

-- price_ticks
ALTER TABLE price_ticks
  ADD COLUMN IF NOT EXISTS chain TEXT NOT NULL DEFAULT 'solana'
    CHECK (chain IN ('solana', 'ethereum'));
CREATE INDEX IF NOT EXISTS idx_price_ticks_chain_token
  ON price_ticks (chain, token_address, fetched_at DESC);

-- kol_mentions (may not exist in all envs — wrapped in DO block for idempotency)
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.tables
             WHERE table_schema = 'public' AND table_name = 'kol_mentions') THEN
    EXECUTE '
      ALTER TABLE kol_mentions
        ADD COLUMN IF NOT EXISTS chain TEXT NOT NULL DEFAULT ''solana''
          CHECK (chain IN (''solana'', ''ethereum''));
      CREATE INDEX IF NOT EXISTS idx_kol_mentions_chain_token
        ON kol_mentions (chain, token_address);
    ';
  END IF;
END $$;

-- Note: No UPDATE needed — DEFAULT 'solana' handles existing rows at
-- ADD COLUMN time. Every row already present is Solana by definition
-- since this is the first chain column introduction.
