-- v7: Scoring improvements — Minsky lifecycle, weakest component, data confidence
-- Adds columns for lifecycle phase classification, score interpretation,
-- weakest component analysis, and missing-data confidence tracking.

-- token_snapshots: ML training data columns
ALTER TABLE token_snapshots
  ADD COLUMN IF NOT EXISTS lifecycle_phase VARCHAR(20),
  ADD COLUMN IF NOT EXISTS weakest_component VARCHAR(20),
  ADD COLUMN IF NOT EXISTS weakest_component_value NUMERIC,
  ADD COLUMN IF NOT EXISTS score_interpretation VARCHAR(20),
  ADD COLUMN IF NOT EXISTS data_confidence NUMERIC;

-- tokens: live ranking display columns
ALTER TABLE tokens
  ADD COLUMN IF NOT EXISTS weakest_component VARCHAR(20),
  ADD COLUMN IF NOT EXISTS score_interpretation VARCHAR(20),
  ADD COLUMN IF NOT EXISTS data_confidence NUMERIC;

-- Drop ALL old function overloads explicitly
DROP FUNCTION IF EXISTS get_token_ranking(character varying, integer, integer);
DROP FUNCTION IF EXISTS get_token_ranking(text, integer, integer, integer);

-- Recreate with updated signature including new fields
CREATE OR REPLACE FUNCTION get_token_ranking(
  p_time_window VARCHAR(10) DEFAULT '24h',
  p_limit INTEGER DEFAULT 10,
  p_offset INTEGER DEFAULT 0,
  p_blend INTEGER DEFAULT 0
)
RETURNS TABLE (
  rank BIGINT,
  symbol VARCHAR(20),
  score INTEGER,
  score_conviction INTEGER,
  score_momentum INTEGER,
  mentions INTEGER,
  unique_kols INTEGER,
  sentiment DECIMAL(4,3),
  trend VARCHAR(10),
  change_24h DECIMAL(8,2),
  weakest_component VARCHAR(20),
  score_interpretation VARCHAR(20),
  data_confidence NUMERIC
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    ROW_NUMBER() OVER (ORDER BY
      CASE WHEN p_blend = 0 THEN t.score
           WHEN p_blend = 1 THEN t.score_conviction
           ELSE t.score_momentum
      END DESC
    ) as rank,
    t.symbol,
    t.score,
    t.score_conviction,
    t.score_momentum,
    t.mentions,
    t.unique_kols,
    t.sentiment,
    t.trend,
    t.change_24h,
    t.weakest_component,
    t.score_interpretation,
    t.data_confidence
  FROM tokens t
  WHERE t.time_window = p_time_window
  ORDER BY
    CASE WHEN p_blend = 0 THEN t.score
         WHEN p_blend = 1 THEN t.score_conviction
         ELSE t.score_momentum
    END DESC
  LIMIT p_limit
  OFFSET p_offset;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Lock down the function to service_role only
REVOKE EXECUTE ON FUNCTION get_token_ranking FROM public;
REVOKE EXECUTE ON FUNCTION get_token_ranking FROM anon;
GRANT EXECUTE ON FUNCTION get_token_ranking TO service_role;
