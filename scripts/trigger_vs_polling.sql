-- v122: Trigger V2 vs Polling comparison analysis
-- Run after accumulating enough live trades with experiment_id = 'trigger_v2_ab'

-- 1. A/B summary: trigger vs polling performance
SELECT
  variant_id,
  count(*) as trades,
  count(*) FILTER (WHERE status NOT IN ('open', 'closing')) as closed,
  round(avg(pnl_pct) FILTER (WHERE status NOT IN ('open', 'closing'))::numeric, 4) as avg_pnl_pct,
  round(sum(pnl_usd) FILTER (WHERE status NOT IN ('open', 'closing'))::numeric, 2) as total_pnl_usd,
  round(avg(exit_minutes) FILTER (WHERE status NOT IN ('open', 'closing'))::numeric, 1) as avg_hold_min,
  round(avg(price_divergence_pct) FILTER (WHERE price_divergence_pct IS NOT NULL)::numeric, 4) as avg_divergence,
  count(*) FILTER (WHERE pnl_pct > 0 AND status NOT IN ('open', 'closing')) as winners,
  round(100.0 * count(*) FILTER (WHERE pnl_pct > 0 AND status NOT IN ('open', 'closing')) /
    NULLIF(count(*) FILTER (WHERE status NOT IN ('open', 'closing')), 0), 1) as win_rate_pct
FROM paper_trades
WHERE source = 'rt_live'
  AND experiment_id = 'trigger_v2_ab'
GROUP BY variant_id
ORDER BY total_pnl_usd DESC;

-- 2. Trigger events breakdown (placed/failed/patched/cancelled/filled)
SELECT event_type, count(*),
       max(recorded_at) as last_event,
       jsonb_agg(DISTINCT details->'reason') FILTER (WHERE event_type = 'failed') as fail_reasons
FROM trigger_events
GROUP BY event_type
ORDER BY count(*) DESC;

-- 3. Price divergence distribution (live vs paper exit prices)
SELECT
  CASE
    WHEN price_divergence_pct IS NULL THEN 'no_data'
    WHEN price_divergence_pct > 0.20 THEN 'live_much_better (>+20%)'
    WHEN price_divergence_pct > 0.05 THEN 'live_better (+5-20%)'
    WHEN price_divergence_pct > -0.05 THEN 'similar (+-5%)'
    WHEN price_divergence_pct > -0.20 THEN 'paper_better (+5-20%)'
    ELSE 'paper_much_better (>+20%)'
  END as divergence_bucket,
  count(*) as trades,
  round(avg(pnl_usd)::numeric, 2) as avg_pnl_usd
FROM paper_trades
WHERE source = 'rt_live'
GROUP BY 1
ORDER BY 2 DESC;
