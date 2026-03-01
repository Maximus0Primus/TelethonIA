-- v8: KOL paper trade stats RPC
-- Returns per-KOL paper trade performance (RT trades only, last 7d, active strategies)

CREATE OR REPLACE FUNCTION get_kol_paper_stats()
RETURNS TABLE (
  kol_name    TEXT,
  rt_trades   BIGINT,
  rt_wins     BIGINT,
  rt_wins_50  BIGINT,
  rt_pnl      NUMERIC,
  rt_wr       NUMERIC,
  rt_wr_50    NUMERIC
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    pt.kol_group                                       AS kol_name,
    COUNT(*)                                           AS rt_trades,
    COUNT(*) FILTER (WHERE pt.pnl_pct > 0)             AS rt_wins,
    COUNT(*) FILTER (WHERE pt.pnl_pct >= 0.50)         AS rt_wins_50,
    ROUND(SUM(pt.pnl_usd)::NUMERIC, 2)                AS rt_pnl,
    ROUND(
      COUNT(*) FILTER (WHERE pt.pnl_pct > 0)::NUMERIC
      / NULLIF(COUNT(*), 0), 4
    )                                                  AS rt_wr,
    ROUND(
      COUNT(*) FILTER (WHERE pt.pnl_pct >= 0.50)::NUMERIC
      / NULLIF(COUNT(*), 0), 4
    )                                                  AS rt_wr_50
  FROM paper_trades pt
  WHERE pt.source = 'rt'
    AND pt.kol_group IS NOT NULL
    AND pt.status IN ('tp_hit', 'sl_hit', 'timeout')
    AND pt.strategy IN ('TP50_SL30', 'TP30_SL50', 'TP50_SL50')
    AND pt.created_at >= NOW() - INTERVAL '7 days'
  GROUP BY pt.kol_group
  ORDER BY rt_trades DESC;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Lock down: service_role only
REVOKE EXECUTE ON FUNCTION get_kol_paper_stats FROM public;
REVOKE EXECUTE ON FUNCTION get_kol_paper_stats FROM anon;
GRANT EXECUTE ON FUNCTION get_kol_paper_stats TO service_role;
