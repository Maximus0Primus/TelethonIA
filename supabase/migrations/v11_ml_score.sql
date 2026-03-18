-- v11/v107: Add standalone ML score and bot win probability columns
-- ml_score: percentile rank (0-100) from regression ensemble, independent of formula score
-- ml_win_prob: bot win probability from classification ensemble (0.0-1.0)

-- tokens table: live ranking display
ALTER TABLE tokens
  ADD COLUMN IF NOT EXISTS ml_score REAL,
  ADD COLUMN IF NOT EXISTS ml_win_prob REAL;

-- token_snapshots table: ML training data
ALTER TABLE token_snapshots
  ADD COLUMN IF NOT EXISTS ml_score REAL,
  ADD COLUMN IF NOT EXISTS ml_win_prob REAL;

-- Drop ALL old function overloads to avoid ambiguity
DROP FUNCTION IF EXISTS get_token_ranking(character varying, integer, integer);
DROP FUNCTION IF EXISTS get_token_ranking(character varying, integer, integer, integer);
DROP FUNCTION IF EXISTS get_token_ranking(text, integer, integer, integer);

-- Recreate with ml_score and ml_win_prob included
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
  data_confidence NUMERIC,
  ml_score REAL,
  ml_win_prob REAL
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    ROW_NUMBER() OVER (ORDER BY
      CASE WHEN p_blend = 0 THEN COALESCE(t.ml_score::INTEGER, t.score)
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
    t.data_confidence,
    t.ml_score,
    t.ml_win_prob
  FROM tokens t
  WHERE t.time_window = p_time_window
  ORDER BY
    CASE WHEN p_blend = 0 THEN COALESCE(t.ml_score::INTEGER, t.score)
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
