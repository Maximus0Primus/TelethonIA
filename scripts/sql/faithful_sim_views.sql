-- Faithful sim layer — canonical lens for shadow↔paper↔live coherence.
--
-- WHY: shadow, paper main and live trade DIFFERENT token sets because of different
-- entry gates (strategy-set funnel, KOL blacklist skipped by shadow, promoted-strat
-- companion shadows that bypass dedup, position caps). Raw shadow aggregates therefore
-- LIE about live profitability — e.g. FAST60_TP70_SL50_NZ_S40 showed raw-shadow +5.19%
-- but real live −10.87% / paper_main −2.19%. This view bakes the ground-truth logic from
-- scripts/_ground_truth_strat.py so every query is faithful by default:
--   1. PREFER MAIN: a strategy with paper_main rows (promoted) is scored on paper_main
--      only — its un-deduped companion shadow rows (paper_trader.py:1671-1678) are ignored.
--      Pure-shadow strategies fall back to their shadow rows (already deduped at insert).
--   2. BLACKLIST FILTER: drop rows whose kol_group is in paper_trade_config.kol_chain_blacklist
--      for the row's chain (shadow records blacklisted KOLs, paper/live don't).
--   3. ROLLING-24h DEDUP: collapse re-entries on the same (strategy, token) within 24h
--      (calendar-day dedup is WRONG — phantom retrades across the day boundary).
--
-- Usage: SELECT * FROM v_strategy_faithful_perf WHERE strategy = '<X>';
-- Caveat: blacklist is applied with the CURRENT config (not point-in-time). Phase A adds
-- per-row persisted tags (live_eligible, is_companion_shadow) for point-in-time accuracy.

-- security_invoker=true: run with the querying role's privileges (service_role bypasses RLS),
-- not the view owner's — avoids the SECURITY DEFINER advisor without changing query results.
CREATE OR REPLACE VIEW v_strategy_faithful_trades
  WITH (security_invoker = true) AS
WITH base AS (
  SELECT pt.id, pt.strategy, pt.chain, pt.token_address, pt.kol_group,
         pt.is_shadow, pt.source, pt.status, pt.pnl_pct, pt.pnl_usd, pt.created_at,
         EXISTS (
           SELECT 1 FROM paper_trades m
           WHERE m.strategy = pt.strategy AND m.is_shadow = false AND m.source = 'rt'
             AND m.status IN ('tp_hit','sl_hit','timeout','trail_stop','be_stop')
             AND m.created_at >= now() - interval '30 days'
         ) AS strat_has_main
  FROM paper_trades pt
  WHERE pt.source = 'rt'
    AND pt.status IN ('tp_hit','sl_hit','timeout','trail_stop','be_stop')
    AND pt.created_at >= now() - interval '30 days'
),
picked AS (
  SELECT b.* FROM base b
  WHERE ((b.strat_has_main AND b.is_shadow = false)
      OR (NOT b.strat_has_main AND b.is_shadow = true))
    AND NOT EXISTS (
      SELECT 1
      FROM scoring_config sc,
           jsonb_array_elements_text(
             coalesce(sc.paper_trade_config->'kol_chain_blacklist'->b.chain, '[]'::jsonb)
           ) AS bl(kol)
      WHERE sc.id = 1 AND bl.kol = b.kol_group
    )
),
dedup AS (
  SELECT p.*,
    p.created_at - lag(p.created_at) OVER (
      PARTITION BY p.strategy, p.token_address ORDER BY p.created_at
    ) AS gap
  FROM picked p
)
SELECT * FROM dedup WHERE gap IS NULL OR gap > interval '24 hours';

CREATE OR REPLACE VIEW v_strategy_faithful_perf
  WITH (security_invoker = true) AS
SELECT strategy, chain,
  bool_or(strat_has_main) AS uses_main_book,
  count(*) AS n,
  round((100.0 * count(*) FILTER (WHERE pnl_pct > 0) / count(*))::numeric, 1) AS wr_pct,
  round((avg(pnl_pct) * 100)::numeric, 2) AS avg_pnlpct,
  round((sum(pnl_pct) * 100)::numeric, 1) AS sum_pnlpct
FROM v_strategy_faithful_trades
GROUP BY strategy, chain;
