-- v9: KOL leaderboard RPCs with dynamic time range
-- All 3 RPCs now accept p_days (0 = all time).
-- Paper stats: trail_stop included, slippage-adjusted thresholds (0.40/0.90).

-- === Paper trade stats ===
DROP FUNCTION IF EXISTS get_kol_paper_stats();
DROP FUNCTION IF EXISTS get_kol_paper_stats_v2(INT);

CREATE OR REPLACE FUNCTION get_kol_paper_stats_v2(p_days INT DEFAULT 7)
RETURNS TABLE (
  kol_name    TEXT,
  rt_trades   BIGINT,
  rt_wins     BIGINT,
  rt_wins_50  BIGINT,
  rt_wins_2x  BIGINT,
  rt_pnl      NUMERIC,
  rt_wr       NUMERIC,
  rt_wr_50    NUMERIC,
  rt_wr_2x    NUMERIC
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    pt.kol_group                                       AS kol_name,
    COUNT(*)                                           AS rt_trades,
    COUNT(*) FILTER (WHERE pt.pnl_pct > 0)             AS rt_wins,
    -- 0.40 instead of 0.50: TP50 yields ~45% after buy+sell slippage
    COUNT(*) FILTER (WHERE pt.pnl_pct >= 0.40)         AS rt_wins_50,
    -- 0.90 instead of 1.00: 2x yields ~94% after slippage
    COUNT(*) FILTER (WHERE pt.pnl_pct >= 0.90)         AS rt_wins_2x,
    ROUND(SUM(pt.pnl_usd)::NUMERIC, 2)                AS rt_pnl,
    ROUND(
      COUNT(*) FILTER (WHERE pt.pnl_pct > 0)::NUMERIC
      / NULLIF(COUNT(*), 0), 4
    )                                                  AS rt_wr,
    ROUND(
      COUNT(*) FILTER (WHERE pt.pnl_pct >= 0.40)::NUMERIC
      / NULLIF(COUNT(*), 0), 4
    )                                                  AS rt_wr_50,
    ROUND(
      COUNT(*) FILTER (WHERE pt.pnl_pct >= 0.90)::NUMERIC
      / NULLIF(COUNT(*), 0), 4
    )                                                  AS rt_wr_2x
  FROM paper_trades pt
  WHERE pt.source = 'rt'
    AND pt.kol_group IS NOT NULL
    AND pt.status IN ('tp_hit', 'sl_hit', 'timeout', 'trail_stop')
    AND pt.strategy IN ('TP50_SL30', 'TP30_SL50', 'TP50_SL50')
    AND (p_days = 0 OR pt.created_at >= NOW() - INTERVAL '1 day' * p_days)
  GROUP BY pt.kol_group
  ORDER BY rt_trades DESC;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

REVOKE EXECUTE ON FUNCTION get_kol_paper_stats_v2(INT) FROM public;
REVOKE EXECUTE ON FUNCTION get_kol_paper_stats_v2(INT) FROM anon;
GRANT EXECUTE ON FUNCTION get_kol_paper_stats_v2(INT) TO service_role;

-- === V1: Snapshot-based leaderboard ===
DROP FUNCTION IF EXISTS get_kol_leaderboard(BOOLEAN);

CREATE OR REPLACE FUNCTION get_kol_leaderboard(p_ca_only BOOLEAN DEFAULT FALSE, p_days INT DEFAULT 0)
RETURNS TABLE (
  kol_name      TEXT,
  unique_tokens BIGINT,
  labeled_calls BIGINT,
  hits_any      BIGINT,
  labeled_12h   BIGINT,
  labeled_24h   BIGINT,
  labeled_48h   BIGINT,
  labeled_72h   BIGINT,
  labeled_7d    BIGINT,
  hits_12h      BIGINT,
  hits_24h      BIGINT,
  hits_48h      BIGINT,
  hits_72h      BIGINT,
  hits_7d       BIGINT,
  last_active   TIMESTAMPTZ
) AS $$
BEGIN
  RETURN QUERY
  WITH unwrapped AS (
    SELECT
      ts.snapshot_at,
      ts.symbol,
      ts.token_address,
      ts.did_2x_12h,
      ts.did_2x_24h,
      ts.did_2x_48h,
      ts.did_2x_72h,
      ts.did_2x_7d,
      ts.max_price_12h,
      ts.max_price_24h,
      ts.max_price_48h,
      ts.max_price_72h,
      ts.max_price_7d,
      jsonb_array_elements_text(
        (ts.top_kols #>> '{}')::jsonb
      ) AS kol
    FROM token_snapshots ts
    WHERE ts.top_kols IS NOT NULL
      AND ts.top_kols::text <> 'null'
      AND ts.top_kols::text <> '[]'
      AND (p_days = 0 OR ts.snapshot_at >= NOW() - INTERVAL '1 day' * p_days)
      AND (NOT p_ca_only OR ts.has_ca_mention = true)
  ),
  first_calls AS (
    SELECT DISTINCT ON (kol, token_address)
      kol, symbol, token_address, snapshot_at,
      did_2x_12h, did_2x_24h, did_2x_48h, did_2x_72h, did_2x_7d,
      max_price_12h, max_price_24h, max_price_48h, max_price_72h, max_price_7d
    FROM unwrapped
    ORDER BY kol, token_address, snapshot_at ASC
  )
  SELECT
    fc.kol AS kol_name,
    COUNT(DISTINCT fc.token_address) AS unique_tokens,
    COUNT(*) FILTER (WHERE
      (fc.did_2x_12h IS NOT NULL AND fc.max_price_12h IS NOT NULL) OR
      (fc.did_2x_24h IS NOT NULL AND fc.max_price_24h IS NOT NULL) OR
      (fc.did_2x_48h IS NOT NULL AND fc.max_price_48h IS NOT NULL) OR
      (fc.did_2x_72h IS NOT NULL AND fc.max_price_72h IS NOT NULL) OR
      (fc.did_2x_7d  IS NOT NULL AND fc.max_price_7d  IS NOT NULL)
    ) AS labeled_calls,
    COUNT(*) FILTER (WHERE
      (fc.did_2x_12h = true AND fc.max_price_12h IS NOT NULL) OR
      (fc.did_2x_24h = true AND fc.max_price_24h IS NOT NULL) OR
      (fc.did_2x_48h = true AND fc.max_price_48h IS NOT NULL) OR
      (fc.did_2x_72h = true AND fc.max_price_72h IS NOT NULL) OR
      (fc.did_2x_7d  = true AND fc.max_price_7d  IS NOT NULL)
    ) AS hits_any,
    COUNT(*) FILTER (WHERE fc.did_2x_12h IS NOT NULL AND fc.max_price_12h IS NOT NULL) AS labeled_12h,
    COUNT(*) FILTER (WHERE fc.did_2x_24h IS NOT NULL AND fc.max_price_24h IS NOT NULL) AS labeled_24h,
    COUNT(*) FILTER (WHERE fc.did_2x_48h IS NOT NULL AND fc.max_price_48h IS NOT NULL) AS labeled_48h,
    COUNT(*) FILTER (WHERE fc.did_2x_72h IS NOT NULL AND fc.max_price_72h IS NOT NULL) AS labeled_72h,
    COUNT(*) FILTER (WHERE fc.did_2x_7d  IS NOT NULL AND fc.max_price_7d  IS NOT NULL) AS labeled_7d,
    COUNT(*) FILTER (WHERE fc.did_2x_12h = true AND fc.max_price_12h IS NOT NULL) AS hits_12h,
    COUNT(*) FILTER (WHERE fc.did_2x_24h = true AND fc.max_price_24h IS NOT NULL) AS hits_24h,
    COUNT(*) FILTER (WHERE fc.did_2x_48h = true AND fc.max_price_48h IS NOT NULL) AS hits_48h,
    COUNT(*) FILTER (WHERE fc.did_2x_72h = true AND fc.max_price_72h IS NOT NULL) AS hits_72h,
    COUNT(*) FILTER (WHERE fc.did_2x_7d  = true AND fc.max_price_7d  IS NOT NULL) AS hits_7d,
    MAX(fc.snapshot_at) AS last_active
  FROM first_calls fc
  GROUP BY fc.kol
  ORDER BY fc.kol;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

REVOKE EXECUTE ON FUNCTION get_kol_leaderboard(BOOLEAN, INT) FROM public;
REVOKE EXECUTE ON FUNCTION get_kol_leaderboard(BOOLEAN, INT) FROM anon;
GRANT EXECUTE ON FUNCTION get_kol_leaderboard(BOOLEAN, INT) TO service_role;

-- === V2: Call-price-based leaderboard ===
DROP FUNCTION IF EXISTS get_kol_leaderboard_v2();

CREATE OR REPLACE FUNCTION get_kol_leaderboard_v2(p_days INT DEFAULT 0)
RETURNS TABLE (
  kol_name         TEXT,
  total_calls      BIGINT,
  with_entry_price BIGINT,
  hits_2x          BIGINT,
  hits_1_5x        BIGINT,
  avg_max_return   NUMERIC,
  best_return      NUMERIC,
  unique_tokens    BIGINT,
  last_active      TIMESTAMPTZ
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    kco.kol_group AS kol_name,
    COUNT(*) AS total_calls,
    COUNT(*) FILTER (WHERE kco.entry_price IS NOT NULL AND kco.ath_after_call IS NOT NULL) AS with_entry_price,
    COUNT(*) FILTER (WHERE kco.did_2x = TRUE) AS hits_2x,
    COUNT(*) FILTER (WHERE kco.max_return >= 1.5) AS hits_1_5x,
    ROUND(AVG(kco.max_return) FILTER (WHERE kco.max_return IS NOT NULL), 2) AS avg_max_return,
    ROUND(MAX(kco.max_return) FILTER (WHERE kco.max_return IS NOT NULL), 2) AS best_return,
    COUNT(DISTINCT kco.token_address) AS unique_tokens,
    MAX(kco.call_timestamp) AS last_active
  FROM kol_call_outcomes kco
  WHERE (p_days = 0 OR kco.call_timestamp >= NOW() - INTERVAL '1 day' * p_days)
  GROUP BY kco.kol_group
  ORDER BY kco.kol_group;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

REVOKE EXECUTE ON FUNCTION get_kol_leaderboard_v2(INT) FROM public;
REVOKE EXECUTE ON FUNCTION get_kol_leaderboard_v2(INT) FROM anon;
GRANT EXECUTE ON FUNCTION get_kol_leaderboard_v2(INT) TO service_role;
