/**
 * Client-side re-scoring engine.
 * Mirrors the scoring logic from pipeline.py for instant parameter tuning.
 */

// ─── Types ───────────────────────────────────────────────────────────────────

export interface ScoringConfig {
  weights: {
    consensus: number;
    sentiment: number;
    conviction: number;
    breadth: number;
    price_action: number;
  };
  multipliers: {
    safety_penalty: { enabled: boolean };
    onchain_multiplier: { enabled: boolean };
    death_penalty: { enabled: boolean };
    already_pumped: { enabled: boolean };
    crash_penalty: { enabled: boolean };
    activity_mult: { enabled: boolean };
    squeeze: { enabled: boolean; max_bonus: number };
    trend: { enabled: boolean; bonus: number };
    entry_premium: { enabled: boolean };
    pump_bonus: { enabled: boolean };
    wash_pen: { enabled: boolean };
    pvp_pen: { enabled: boolean };
    pump_pen: { enabled: boolean };
    breadth_pen: { enabled: boolean };
    stale_pen: { enabled: boolean };
  };
  confirmation_gate: {
    enabled: boolean;
    min_pillars: number;
    penalty: number;
    thresholds: {
      consensus: number;
      price_action: number;
      breadth: number;
    };
  };
}

export interface TokenSnapshot {
  symbol: string;
  token_address?: string;
  snapshot_at?: string;
  // 5 component values [0,1]
  consensus_val: number | null;
  sentiment_val: number | null;
  conviction_val: number | null;
  breadth_val: number | null;
  price_action_val: number | null;
  // Multipliers (stored from scraper)
  safety_penalty: number | null;
  onchain_multiplier: number | null;
  death_penalty: number | null;
  already_pumped_penalty: number | null;
  crash_pen: number | null;
  activity_mult: number | null;
  squeeze_score: number | null;
  squeeze_state: string | null;
  trend_strength: number | null;
  confirmation_pillars: number | null;
  entry_premium_mult: number | null;
  pump_bonus: number | null;
  wash_pen: number | null;
  pvp_pen: number | null;
  pump_pen: number | null;
  breadth_pen: number | null;
  stale_pen: number | null;
  // Context data
  price_change_24h: number | null;
  volume_24h: number | null;
  liquidity_usd: number | null;
  mentions: number | null;
  freshest_mention_hours: number | null;
  top_kols: string | null;
  price_at_snapshot: number | null;
  market_cap: number | null;
  // Outcomes (for backtest)
  did_2x_1h?: boolean | null;
  did_2x_6h?: boolean | null;
  did_2x_12h?: boolean | null;
  did_2x_24h?: boolean | null;
}

export interface ScoredToken {
  symbol: string;
  newScore: number;
  prodScore: number;
  delta: number;
  rank: number;
  // Component breakdown
  components: {
    consensus: number;
    sentiment: number;
    conviction: number;
    breadth: number;
    price_action: number;
    weighted_sum: number;
  };
  // Multiplier breakdown
  multipliers: {
    safety_penalty: number;
    onchain_multiplier: number;
    crash_pen: number;
    activity_mult: number;
    squeeze: number;
    trend: number;
    pump_bonus: number;
    wash_pen: number;
    pvp_pen: number;
    pump_pen: number;
    breadth_pen: number;
    stale_pen: number;
    confirmation_gate: number;
    combined: number;
  };
  snapshot: TokenSnapshot;
}

// ─── Default Config (production values) ──────────────────────────────────────

export const DEFAULT_CONFIG: ScoringConfig = {
  weights: {
    consensus: 0.30,
    sentiment: 0.00, // v13: removed from balanced
    conviction: 0.15,
    breadth: 0.25,
    price_action: 0.30,
  },
  multipliers: {
    safety_penalty: { enabled: true },
    onchain_multiplier: { enabled: true },
    death_penalty: { enabled: true },
    already_pumped: { enabled: true },
    crash_penalty: { enabled: true },
    activity_mult: { enabled: true },
    squeeze: { enabled: true, max_bonus: 0.2 },
    trend: { enabled: true, bonus: 1.15 },
    entry_premium: { enabled: true },
    pump_bonus: { enabled: true },
    wash_pen: { enabled: true },
    pvp_pen: { enabled: true },
    pump_pen: { enabled: true },
    breadth_pen: { enabled: true },
    stale_pen: { enabled: true },
  },
  confirmation_gate: {
    enabled: true,
    min_pillars: 2,
    penalty: 0.8,
    thresholds: {
      consensus: 0.2,
      price_action: 0.35,
      breadth: 0.08,
    },
  },
};

// ─── Scoring Functions ───────────────────────────────────────────────────────

export function rescore(snapshot: TokenSnapshot, config: ScoringConfig): ScoredToken {
  const w = config.weights;

  // 1. Component values (with fallbacks)
  const cv = snapshot.consensus_val ?? 0;
  const sv = snapshot.sentiment_val ?? 0;
  const kv = snapshot.conviction_val ?? 0;
  const bv = snapshot.breadth_val ?? 0;
  const pv = snapshot.price_action_val ?? 0.5;

  // Weight renormalization: only count weights for available components
  const available: { value: number; weight: number }[] = [];
  if (snapshot.consensus_val !== null) available.push({ value: cv, weight: w.consensus });
  if (snapshot.sentiment_val !== null) available.push({ value: sv, weight: w.sentiment });
  if (snapshot.conviction_val !== null) available.push({ value: kv, weight: w.conviction });
  if (snapshot.breadth_val !== null) available.push({ value: bv, weight: w.breadth });
  if (snapshot.price_action_val !== null) available.push({ value: pv, weight: w.price_action });

  const totalWeight = available.reduce((s, a) => s + a.weight, 0);
  const weightedSum = totalWeight > 0
    ? available.reduce((s, a) => s + a.value * (a.weight / totalWeight), 0)
    : 0;

  const raw = weightedSum * 100;

  // 2. Multiplier chain
  const m = config.multipliers;
  let mult = 1.0;
  const safetyMult = m.safety_penalty.enabled ? (snapshot.safety_penalty ?? 1.0) : 1.0;
  const onchainMult = m.onchain_multiplier.enabled ? (snapshot.onchain_multiplier ?? 1.0) : 1.0;

  // crash_pen in DB is min(already_pumped, death, entry_premium)
  // When individual toggles are off, we reconstruct
  let crashMult = 1.0;
  if (m.crash_penalty.enabled) {
    crashMult = snapshot.crash_pen ?? 1.0;
  } else {
    // If crash_penalty toggled off but sub-components toggled on, use them individually
    const parts: number[] = [];
    if (m.death_penalty.enabled) parts.push(snapshot.death_penalty ?? 1.0);
    if (m.already_pumped.enabled) parts.push(snapshot.already_pumped_penalty ?? 1.0);
    if (m.entry_premium.enabled) parts.push(snapshot.entry_premium_mult ?? 1.0);
    if (parts.length > 0) crashMult = Math.min(...parts);
  }

  const activityMult = m.activity_mult.enabled ? (snapshot.activity_mult ?? 1.0) : 1.0;
  const pumpBonusMult = m.pump_bonus.enabled ? (snapshot.pump_bonus ?? 1.0) : 1.0;
  const washMult = m.wash_pen.enabled ? (snapshot.wash_pen ?? 1.0) : 1.0;
  const pvpMult = m.pvp_pen.enabled ? (snapshot.pvp_pen ?? 1.0) : 1.0;
  const pumpPenMult = m.pump_pen.enabled ? (snapshot.pump_pen ?? 1.0) : 1.0;
  const breadthPenMult = m.breadth_pen.enabled ? (snapshot.breadth_pen ?? 1.0) : 1.0;
  const stalePenMult = m.stale_pen.enabled ? (snapshot.stale_pen ?? 1.0) : 1.0;

  let squeezeMult = 1.0;
  if (m.squeeze.enabled && (snapshot.squeeze_score ?? 0) > 0) {
    squeezeMult = 1.0 + (snapshot.squeeze_score ?? 0) * m.squeeze.max_bonus;
  }

  let trendMult = 1.0;
  if (m.trend.enabled && (snapshot.trend_strength ?? 0) > 0.5) {
    trendMult = m.trend.bonus;
  }

  mult = safetyMult * onchainMult * crashMult * activityMult * squeezeMult * trendMult
    * pumpBonusMult * washMult * pvpMult * pumpPenMult * breadthPenMult * stalePenMult;

  // 3. Confirmation gate
  let gateMult = 1.0;
  const cg = config.confirmation_gate;
  if (cg.enabled) {
    let pillars = 0;
    if ((snapshot.consensus_val ?? 0) >= cg.thresholds.consensus) pillars++;
    if ((snapshot.price_action_val ?? 0.5) >= cg.thresholds.price_action) pillars++;
    if ((snapshot.breadth_val ?? 0) >= cg.thresholds.breadth) pillars++;
    if (pillars < cg.min_pillars) gateMult = cg.penalty;
  }

  const combined = mult * gateMult;
  const newScore = Math.min(100, Math.max(0, Math.round(raw * combined)));

  // Estimate production score: same formula with default config
  // Actually, we just use the raw score from snapshot if available
  const prodScore = estimateProdScore(snapshot);

  return {
    symbol: snapshot.symbol,
    newScore,
    prodScore,
    delta: newScore - prodScore,
    rank: 0, // will be set by rescoreAll
    components: {
      consensus: cv,
      sentiment: sv,
      conviction: kv,
      breadth: bv,
      price_action: pv,
      weighted_sum: Math.round(raw * 10) / 10,
    },
    multipliers: {
      safety_penalty: safetyMult,
      onchain_multiplier: onchainMult,
      crash_pen: crashMult,
      activity_mult: activityMult,
      squeeze: squeezeMult,
      trend: trendMult,
      pump_bonus: pumpBonusMult,
      wash_pen: washMult,
      pvp_pen: pvpMult,
      pump_pen: pumpPenMult,
      breadth_pen: breadthPenMult,
      stale_pen: stalePenMult,
      confirmation_gate: gateMult,
      combined: Math.round(combined * 1000) / 1000,
    },
    snapshot,
  };
}

function estimateProdScore(snapshot: TokenSnapshot): number {
  // Re-score with default config to get a comparable production score
  const w = DEFAULT_CONFIG.weights;
  const cv = snapshot.consensus_val ?? 0;
  const sv = snapshot.sentiment_val ?? 0;
  const kv = snapshot.conviction_val ?? 0;
  const bv = snapshot.breadth_val ?? 0;
  const pv = snapshot.price_action_val ?? 0.5;

  const available: { value: number; weight: number }[] = [];
  if (snapshot.consensus_val !== null) available.push({ value: cv, weight: w.consensus });
  if (snapshot.sentiment_val !== null) available.push({ value: sv, weight: w.sentiment });
  if (snapshot.conviction_val !== null) available.push({ value: kv, weight: w.conviction });
  if (snapshot.breadth_val !== null) available.push({ value: bv, weight: w.breadth });
  if (snapshot.price_action_val !== null) available.push({ value: pv, weight: w.price_action });

  const totalWeight = available.reduce((s, a) => s + a.weight, 0);
  const weightedSum = totalWeight > 0
    ? available.reduce((s, a) => s + a.value * (a.weight / totalWeight), 0)
    : 0;

  const raw = weightedSum * 100;

  let mult = (snapshot.safety_penalty ?? 1.0)
    * (snapshot.onchain_multiplier ?? 1.0)
    * (snapshot.crash_pen ?? 1.0)
    * (snapshot.activity_mult ?? 1.0)
    * (snapshot.pump_bonus ?? 1.0)
    * (snapshot.wash_pen ?? 1.0)
    * (snapshot.pvp_pen ?? 1.0)
    * (snapshot.pump_pen ?? 1.0)
    * (snapshot.breadth_pen ?? 1.0)
    * (snapshot.stale_pen ?? 1.0);

  if ((snapshot.squeeze_score ?? 0) > 0) {
    mult *= 1.0 + (snapshot.squeeze_score ?? 0) * 0.2;
  }
  if ((snapshot.trend_strength ?? 0) > 0.5) {
    mult *= 1.15;
  }

  // Confirmation gate
  let pillars = 0;
  if ((snapshot.consensus_val ?? 0) >= 0.2) pillars++;
  if ((snapshot.price_action_val ?? 0.5) >= 0.35) pillars++;
  if ((snapshot.breadth_val ?? 0) >= 0.08) pillars++;
  if (pillars < 2) mult *= 0.8;

  return Math.min(100, Math.max(0, Math.round(raw * mult)));
}

export function rescoreAll(
  snapshots: TokenSnapshot[],
  config: ScoringConfig
): ScoredToken[] {
  const scored = snapshots.map((s) => rescore(s, config));
  scored.sort((a, b) => b.newScore - a.newScore || (b.snapshot.mentions ?? 0) - (a.snapshot.mentions ?? 0));
  scored.forEach((t, i) => { t.rank = i + 1; });
  return scored;
}

export function normalizeWeights(weights: ScoringConfig["weights"]): ScoringConfig["weights"] {
  const total = Object.values(weights).reduce((s, v) => s + v, 0);
  if (total === 0) return weights;
  return {
    consensus: Math.round((weights.consensus / total) * 100) / 100,
    sentiment: Math.round((weights.sentiment / total) * 100) / 100,
    conviction: Math.round((weights.conviction / total) * 100) / 100,
    breadth: Math.round((weights.breadth / total) * 100) / 100,
    price_action: Math.round((weights.price_action / total) * 100) / 100,
  };
}
