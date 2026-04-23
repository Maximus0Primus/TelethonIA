-- v14e — Per-chain bankroll + strategy isolation.
--
-- Motivation: v14 put a `chain` column on every trade/token table, but the
-- money side (rt_bankroll.strategy_bankrolls) stayed a single flat JSONB
-- dict keyed by strategy name. That "works" today only because ETH strats
-- happen to be prefixed ETH_, which is a naming convention, not a hard
-- isolation. BSC/Base rollout would break this — same strategy names on
-- different chains would share a bankroll silently.
--
-- This migration:
--   1. Widens the chain CHECK constraint to allow bsc + base (forward compat).
--   2. Adds a new JSONB column `strategy_bankrolls_per_chain` storing the
--      nested structure {chain: {strategy: {balance, pnl, trades}}}.
--   3. Backfills it from the current flat dict by routing each strategy to
--      its chain via naming heuristic (ETH_ prefix → ethereum; everything
--      else → solana — the only two chains live today).
--   4. Adds per-chain daily_loss_limit + max_open_positions JSONB for
--      independent risk controls per chain.
--
-- The old `strategy_bankrolls` column is kept for one release cycle as a
-- read-only mirror so any rollback is trivial. Writers go to the new column;
-- readers fall back to the old one only if the new one is empty.

-- 1. Widen chain CHECK on every table that has it.
ALTER TABLE paper_trades   DROP CONSTRAINT IF EXISTS paper_trades_chain_check;
ALTER TABLE paper_trades   ADD  CONSTRAINT paper_trades_chain_check   CHECK (chain IN ('solana','ethereum','bsc','base'));
ALTER TABLE token_snapshots DROP CONSTRAINT IF EXISTS token_snapshots_chain_check;
ALTER TABLE token_snapshots ADD  CONSTRAINT token_snapshots_chain_check CHECK (chain IN ('solana','ethereum','bsc','base'));
ALTER TABLE tokens         DROP CONSTRAINT IF EXISTS tokens_chain_check;
ALTER TABLE tokens         ADD  CONSTRAINT tokens_chain_check         CHECK (chain IN ('solana','ethereum','bsc','base'));
ALTER TABLE price_ticks    DROP CONSTRAINT IF EXISTS price_ticks_chain_check;
ALTER TABLE price_ticks    ADD  CONSTRAINT price_ticks_chain_check    CHECK (chain IN ('solana','ethereum','bsc','base'));
ALTER TABLE kol_mentions   DROP CONSTRAINT IF EXISTS kol_mentions_chain_check;
ALTER TABLE kol_mentions   ADD  CONSTRAINT kol_mentions_chain_check   CHECK (chain IN ('solana','ethereum','bsc','base'));

-- 2. Add the per-chain bankroll column.
ALTER TABLE rt_bankroll
  ADD COLUMN IF NOT EXISTS strategy_bankrolls_per_chain JSONB NOT NULL DEFAULT '{}'::jsonb;

-- 3. Backfill from flat dict. Routing heuristic:
--   - Strategy name starts with "ETH_" → ethereum
--   - Everything else → solana (the only other chain live on this release)
--   - BSC_/BASE_ are reserved for future rollout (no rows to migrate today)
DO $$
DECLARE
  r RECORD;
  flat JSONB;
  nested JSONB;
  k TEXT;
  v JSONB;
  target_chain TEXT;
BEGIN
  FOR r IN SELECT id, strategy_bankrolls FROM rt_bankroll LOOP
    flat := COALESCE(r.strategy_bankrolls, '{}'::jsonb);
    nested := jsonb_build_object('solana', '{}'::jsonb, 'ethereum', '{}'::jsonb,
                                  'bsc', '{}'::jsonb, 'base', '{}'::jsonb);
    FOR k, v IN SELECT * FROM jsonb_each(flat) LOOP
      IF k LIKE 'ETH\_%' ESCAPE '\' THEN
        target_chain := 'ethereum';
      ELSIF k LIKE 'BSC\_%' ESCAPE '\' THEN
        target_chain := 'bsc';
      ELSIF k LIKE 'BASE\_%' ESCAPE '\' THEN
        target_chain := 'base';
      ELSE
        target_chain := 'solana';
      END IF;
      nested := jsonb_set(nested, ARRAY[target_chain, k], v, true);
    END LOOP;
    UPDATE rt_bankroll
      SET strategy_bankrolls_per_chain = nested
      WHERE id = r.id;
  END LOOP;
END $$;

-- 4. Per-chain risk controls. Each chain gets its own daily loss limit +
-- max open positions. Defaults mirror the existing Solana config so
-- behaviour doesn't change until the user edits them explicitly.
ALTER TABLE rt_bankroll
  ADD COLUMN IF NOT EXISTS risk_limits_per_chain JSONB NOT NULL DEFAULT
    jsonb_build_object(
      'solana',   jsonb_build_object('daily_loss_limit_sol', 0.5, 'max_open_positions', 12),
      'ethereum', jsonb_build_object('daily_loss_limit_usd', 100, 'max_open_positions', 3),
      'bsc',      jsonb_build_object('daily_loss_limit_usd', 100, 'max_open_positions', 3),
      'base',     jsonb_build_object('daily_loss_limit_usd', 100, 'max_open_positions', 3)
    );

-- Verification: pretty-print the nested structure for one row. Will show up
-- in migration output so you can eyeball the backfill.
SELECT jsonb_pretty(strategy_bankrolls_per_chain) AS bankroll_per_chain,
       jsonb_pretty(risk_limits_per_chain) AS risk_limits
  FROM rt_bankroll
  LIMIT 1;
