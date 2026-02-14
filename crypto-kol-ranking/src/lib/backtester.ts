/**
 * Backtesting engine for the tuning platform.
 * Re-scores historical labeled snapshots and computes hit rates.
 */

import {
  type ScoringConfig,
  type TokenSnapshot,
  DEFAULT_CONFIG,
  rescore,
  normalizeWeights,
  filterByExtractionMode,
} from "./rescorer";

// ─── Types ───────────────────────────────────────────────────────────────────

export type Horizon = "1h" | "6h" | "12h" | "24h" | "48h" | "72h" | "7d";

export interface BacktestResult {
  total_snapshots: number;
  total_2x: number;
  base_rate: number;
  top5_hit_rate: number;
  top10_hit_rate: number;
  top20_hit_rate: number;
  avg_score_2x: number;
  avg_score_no2x: number;
  separation: number;
}

export interface FeatureImpact {
  feature: string;
  label: string;
  impact_on_hit_rate: number;
  optimal_value: string;
  current_value: string;
}

// ─── Helpers ────────────────────────────────────────────────────────────────

function deepClone<T>(obj: T): T {
  return JSON.parse(JSON.stringify(obj));
}

/** Combined optimization score: top5 dominates, top10/top20 add granularity */
function optScore(snapshots: TokenSnapshot[], config: ScoringConfig, horizon: Horizon): number {
  const r = runBacktest(snapshots, config, horizon);
  return r.top5_hit_rate * 0.5 + r.top10_hit_rate * 0.3 + r.top20_hit_rate * 0.2;
}

const COMPONENTS = ["consensus", "sentiment", "conviction", "breadth", "price_action"] as const;

const MULTS = [
  "safety_penalty", "onchain_multiplier", "death_penalty",
  "already_pumped", "crash_penalty", "activity_mult",
  "squeeze", "trend", "entry_premium",
  "pump_bonus", "wash_pen", "pvp_pen", "pump_pen", "breadth_pen",
  "stale_pen",
] as const;

// ─── Backtest Runner ─────────────────────────────────────────────────────────

function getOutcome(snap: TokenSnapshot, horizon: Horizon): boolean | null {
  switch (horizon) {
    case "1h": return snap.did_2x_1h ?? null;
    case "6h": return snap.did_2x_6h ?? null;
    case "12h": return snap.did_2x_12h ?? null;
    case "24h": return snap.did_2x_24h ?? null;
    case "48h": return snap.did_2x_48h ?? null;
    case "72h": return snap.did_2x_72h ?? null;
    case "7d": return snap.did_2x_7d ?? null;
  }
}

export function runBacktest(
  snapshots: TokenSnapshot[],
  config: ScoringConfig,
  horizon: Horizon
): BacktestResult {
  const filtered = filterByExtractionMode(snapshots, config.extraction_mode);
  const labeled = filtered.filter((s) => getOutcome(s, horizon) !== null);
  if (labeled.length === 0) {
    return {
      total_snapshots: 0, total_2x: 0, base_rate: 0,
      top5_hit_rate: 0, top10_hit_rate: 0, top20_hit_rate: 0,
      avg_score_2x: 0, avg_score_no2x: 0, separation: 0,
    };
  }

  const scored = labeled
    .map((s) => ({ score: rescore(s, config).newScore, did2x: getOutcome(s, horizon)! }))
    .sort((a, b) => b.score - a.score);

  const total = scored.length;
  const total2x = scored.filter((s) => s.did2x).length;
  const baseRate = total2x / total;

  const hitRate = (k: number) => {
    const topK = scored.slice(0, Math.min(k, total));
    if (topK.length === 0) return 0;
    return topK.filter((s) => s.did2x).length / topK.length;
  };

  const scores2x = scored.filter((s) => s.did2x).map((s) => s.score);
  const scoresNo2x = scored.filter((s) => !s.did2x).map((s) => s.score);
  const avg2x = scores2x.length > 0 ? scores2x.reduce((a, b) => a + b, 0) / scores2x.length : 0;
  const avgNo2x = scoresNo2x.length > 0 ? scoresNo2x.reduce((a, b) => a + b, 0) / scoresNo2x.length : 0;

  return {
    total_snapshots: total,
    total_2x: total2x,
    base_rate: Math.round(baseRate * 1000) / 10,
    top5_hit_rate: Math.round(hitRate(5) * 1000) / 10,
    top10_hit_rate: Math.round(hitRate(10) * 1000) / 10,
    top20_hit_rate: Math.round(hitRate(20) * 1000) / 10,
    avg_score_2x: Math.round(avg2x * 10) / 10,
    avg_score_no2x: Math.round(avgNo2x * 10) / 10,
    separation: avgNo2x > 0 ? Math.round((avg2x / avgNo2x) * 100) / 100 : 0,
  };
}

// ─── Internal: cumulative sweep (phases 1-4) ────────────────────────────────

function cumulativeSweep(
  snapshots: TokenSnapshot[],
  horizon: Horizon,
  startConfig: ScoringConfig,
): { config: ScoringConfig; hitRate: number } {
  let best = deepClone(startConfig);
  let bestHR = optScore(snapshots, best, horizon);

  // Phase 1: Cumulative weight sweep
  for (const comp of COMPONENTS) {
    let phaseBestHR = bestHR;
    let phaseBestW = best.weights[comp];

    for (let w = 0; w <= 0.50; w += 0.05) {
      const test = deepClone(best);
      test.weights[comp] = w;
      const others = COMPONENTS.filter((c) => c !== comp);
      const remaining = 1 - w;
      const otherSum = others.reduce((s, c) => s + best.weights[c], 0);
      for (const o of others) {
        test.weights[o] = otherSum > 0
          ? (best.weights[o] / otherSum) * remaining
          : remaining / others.length;
      }
      test.weights = normalizeWeights(test.weights);

      const hr = optScore(snapshots, test, horizon);
      if (hr > phaseBestHR) {
        phaseBestHR = hr;
        phaseBestW = w;
      }
    }

    // Apply best weight
    const others = COMPONENTS.filter((c) => c !== comp);
    const remaining = 1 - phaseBestW;
    const otherSum = others.reduce((s, c) => s + best.weights[c], 0);
    best.weights[comp] = phaseBestW;
    for (const o of others) {
      best.weights[o] = otherSum > 0
        ? (best.weights[o] / otherSum) * remaining
        : remaining / others.length;
    }
    best.weights = normalizeWeights(best.weights);
    bestHR = phaseBestHR;
  }

  // Phase 2: Cumulative multiplier toggle
  for (const mult of MULTS) {
    const configOn = deepClone(best);
    const configOff = deepClone(best);
    (configOn.multipliers[mult] as { enabled: boolean }).enabled = true;
    (configOff.multipliers[mult] as { enabled: boolean }).enabled = false;

    const hitOn = optScore(snapshots, configOn, horizon);
    const hitOff = optScore(snapshots, configOff, horizon);

    const bestState = hitOn >= hitOff;
    (best.multipliers[mult] as { enabled: boolean }).enabled = bestState;
    bestHR = Math.max(hitOn, hitOff);
  }

  // Phase 3: Confirmation gate
  const cgOn = deepClone(best);
  const cgOff = deepClone(best);
  cgOn.confirmation_gate.enabled = true;
  cgOff.confirmation_gate.enabled = false;
  const cgHitOn = optScore(snapshots, cgOn, horizon);
  const cgHitOff = optScore(snapshots, cgOff, horizon);
  best.confirmation_gate.enabled = cgHitOn >= cgHitOff;
  bestHR = Math.max(cgHitOn, cgHitOff);

  // Phase 4: KOL tuning

  // 4a: conviction_dampening
  {
    let pBest = bestHR;
    let pEnabled = best.kol_tuning.conviction_dampening.enabled;
    let pMinKols = best.kol_tuning.conviction_dampening.min_kols;

    for (const enabled of [true, false]) {
      const mkValues = enabled ? [1, 2, 3, 4] : [best.kol_tuning.conviction_dampening.min_kols];
      for (const mk of mkValues) {
        const test = deepClone(best);
        test.kol_tuning.conviction_dampening.enabled = enabled;
        test.kol_tuning.conviction_dampening.min_kols = mk;
        const hr = optScore(snapshots, test, horizon);
        if (hr > pBest) { pBest = hr; pEnabled = enabled; pMinKols = mk; }
      }
    }
    best.kol_tuning.conviction_dampening.enabled = pEnabled;
    best.kol_tuning.conviction_dampening.min_kols = pMinKols;
    bestHR = pBest;
  }

  // 4b: s_tier_bonus
  {
    let pBest = bestHR;
    let pEnabled = best.kol_tuning.s_tier_bonus.enabled;
    let pBonus = best.kol_tuning.s_tier_bonus.bonus;

    // OFF
    const testOff = deepClone(best);
    testOff.kol_tuning.s_tier_bonus.enabled = false;
    const hrOff = optScore(snapshots, testOff, horizon);
    if (hrOff > pBest) { pBest = hrOff; pEnabled = false; }

    // ON with sweep
    for (let b = 1.0; b <= 2.0; b = Math.round((b + 0.1) * 10) / 10) {
      const test = deepClone(best);
      test.kol_tuning.s_tier_bonus.enabled = true;
      test.kol_tuning.s_tier_bonus.bonus = b;
      const hr = optScore(snapshots, test, horizon);
      if (hr > pBest) { pBest = hr; pEnabled = true; pBonus = b; }
    }

    best.kol_tuning.s_tier_bonus.enabled = pEnabled;
    best.kol_tuning.s_tier_bonus.bonus = pBonus;
    bestHR = pBest;
  }

  // 4c: freshness_cutoff
  {
    let pBest = bestHR;
    let pEnabled = best.kol_tuning.freshness_cutoff.enabled;
    let pHours = best.kol_tuning.freshness_cutoff.max_hours;
    let pPenalty = best.kol_tuning.freshness_cutoff.penalty;

    // OFF
    const testOff = deepClone(best);
    testOff.kol_tuning.freshness_cutoff.enabled = false;
    const hrOff = optScore(snapshots, testOff, horizon);
    if (hrOff > pBest) { pBest = hrOff; pEnabled = false; }

    // ON with sweep
    for (const h of [6, 12, 24, 48]) {
      for (const p of [0.3, 0.5, 0.7]) {
        const test = deepClone(best);
        test.kol_tuning.freshness_cutoff.enabled = true;
        test.kol_tuning.freshness_cutoff.max_hours = h;
        test.kol_tuning.freshness_cutoff.penalty = p;
        const hr = optScore(snapshots, test, horizon);
        if (hr > pBest) { pBest = hr; pEnabled = true; pHours = h; pPenalty = p; }
      }
    }

    best.kol_tuning.freshness_cutoff.enabled = pEnabled;
    best.kol_tuning.freshness_cutoff.max_hours = pHours;
    best.kol_tuning.freshness_cutoff.penalty = pPenalty;
    bestHR = pBest;
  }

  return { config: best, hitRate: bestHR };
}

// ─── Auto-Optimizer (dual-track + random exploration) ───────────────────────

export function autoOptimize(
  snapshots: TokenSnapshot[],
  horizon: Horizon,
  baseConfig: ScoringConfig
): { features: FeatureImpact[]; bestConfig: ScoringConfig; bestHitRate: number } {
  const baseHR = optScore(snapshots, baseConfig, horizon);

  // ── Track A: sweep from user's current config ─────────────────────────────
  const trackA = cumulativeSweep(snapshots, horizon, baseConfig);

  // ── Track B: sweep from DEFAULT_CONFIG ────────────────────────────────────
  const trackB = cumulativeSweep(snapshots, horizon, DEFAULT_CONFIG);

  // Take the better starting point
  let best = trackA.hitRate >= trackB.hitRate ? trackA.config : trackB.config;
  let bestHR = Math.max(trackA.hitRate, trackB.hitRate);

  // ── Phase 5: Random exploration around the best ───────────────────────────
  for (let i = 0; i < 40; i++) {
    const trial = deepClone(best);

    // Perturb weights: +-0.15 from best
    for (const comp of COMPONENTS) {
      const delta = (Math.random() - 0.5) * 0.30;
      trial.weights[comp] = Math.max(0, Math.min(0.60, trial.weights[comp] + delta));
    }
    trial.weights = normalizeWeights(trial.weights);

    // Flip multiplier toggles with 20% probability each
    for (const mult of MULTS) {
      if (Math.random() < 0.20) {
        const m = trial.multipliers[mult] as { enabled: boolean };
        m.enabled = !m.enabled;
      }
    }

    // Flip confirmation gate with 15% probability
    if (Math.random() < 0.15) {
      trial.confirmation_gate.enabled = !trial.confirmation_gate.enabled;
    }

    const hr = optScore(snapshots, trial, horizon);
    if (hr > bestHR) { best = trial; bestHR = hr; }
  }

  // ── Compute feature impacts: compare bestConfig vs baseConfig ─────────────
  const results: FeatureImpact[] = [];

  // Weight impacts: what happens if we revert each weight to baseConfig's value?
  for (const comp of COMPONENTS) {
    const reverted = deepClone(best);
    reverted.weights[comp] = baseConfig.weights[comp];
    // Renormalize proportionally
    const others = COMPONENTS.filter((c) => c !== comp);
    const remaining = 1 - reverted.weights[comp];
    const otherSum = others.reduce((s, c) => s + best.weights[c], 0);
    for (const o of others) {
      reverted.weights[o] = otherSum > 0
        ? (best.weights[o] / otherSum) * remaining
        : remaining / others.length;
    }
    reverted.weights = normalizeWeights(reverted.weights);

    const revertedHR = optScore(snapshots, reverted, horizon);
    // Impact = how much we lose by reverting to the original value
    const impact = bestHR - revertedHR;

    results.push({
      feature: `weight:${comp}`,
      label: `${comp} weight`,
      impact_on_hit_rate: Math.round(impact * 10) / 10,
      optimal_value: best.weights[comp].toFixed(2),
      current_value: baseConfig.weights[comp].toFixed(2),
    });
  }

  // Multiplier impacts: what happens if we revert each toggle?
  for (const mult of MULTS) {
    const bestState = (best.multipliers[mult] as { enabled: boolean }).enabled;
    const baseState = (baseConfig.multipliers[mult] as { enabled: boolean }).enabled;

    const reverted = deepClone(best);
    (reverted.multipliers[mult] as { enabled: boolean }).enabled = baseState;
    const revertedHR = optScore(snapshots, reverted, horizon);
    const impact = bestHR - revertedHR;

    results.push({
      feature: `mult:${mult}`,
      label: mult.replace(/_/g, " "),
      impact_on_hit_rate: Math.round(impact * 10) / 10,
      optimal_value: bestState ? "ON" : "OFF",
      current_value: baseState ? "ON" : "OFF",
    });
  }

  // Confirmation gate impact
  {
    const reverted = deepClone(best);
    reverted.confirmation_gate.enabled = baseConfig.confirmation_gate.enabled;
    const revertedHR = optScore(snapshots, reverted, horizon);
    results.push({
      feature: "gate:confirmation",
      label: "confirmation gate",
      impact_on_hit_rate: Math.round((bestHR - revertedHR) * 10) / 10,
      optimal_value: best.confirmation_gate.enabled ? "ON" : "OFF",
      current_value: baseConfig.confirmation_gate.enabled ? "ON" : "OFF",
    });
  }

  // KOL tuning impacts
  {
    // conviction_dampening
    const revCD = deepClone(best);
    revCD.kol_tuning.conviction_dampening = deepClone(baseConfig.kol_tuning.conviction_dampening);
    const hrCD = optScore(snapshots, revCD, horizon);
    results.push({
      feature: "kol:conviction_dampening",
      label: "conviction dampening",
      impact_on_hit_rate: Math.round((bestHR - hrCD) * 10) / 10,
      optimal_value: best.kol_tuning.conviction_dampening.enabled
        ? `ON (min_kols=${best.kol_tuning.conviction_dampening.min_kols})` : "OFF",
      current_value: baseConfig.kol_tuning.conviction_dampening.enabled
        ? `ON (min_kols=${baseConfig.kol_tuning.conviction_dampening.min_kols})` : "OFF",
    });

    // s_tier_bonus
    const revST = deepClone(best);
    revST.kol_tuning.s_tier_bonus = deepClone(baseConfig.kol_tuning.s_tier_bonus);
    const hrST = optScore(snapshots, revST, horizon);
    results.push({
      feature: "kol:s_tier_bonus",
      label: "S-tier bonus",
      impact_on_hit_rate: Math.round((bestHR - hrST) * 10) / 10,
      optimal_value: best.kol_tuning.s_tier_bonus.enabled
        ? `ON (${best.kol_tuning.s_tier_bonus.bonus.toFixed(1)}x)` : "OFF",
      current_value: baseConfig.kol_tuning.s_tier_bonus.enabled
        ? `ON (${baseConfig.kol_tuning.s_tier_bonus.bonus.toFixed(1)}x)` : "OFF",
    });

    // freshness_cutoff
    const revFC = deepClone(best);
    revFC.kol_tuning.freshness_cutoff = deepClone(baseConfig.kol_tuning.freshness_cutoff);
    const hrFC = optScore(snapshots, revFC, horizon);
    results.push({
      feature: "kol:freshness_cutoff",
      label: "freshness cutoff",
      impact_on_hit_rate: Math.round((bestHR - hrFC) * 10) / 10,
      optimal_value: best.kol_tuning.freshness_cutoff.enabled
        ? `ON (${best.kol_tuning.freshness_cutoff.max_hours}h, ${best.kol_tuning.freshness_cutoff.penalty}x)` : "OFF",
      current_value: baseConfig.kol_tuning.freshness_cutoff.enabled
        ? `ON (${baseConfig.kol_tuning.freshness_cutoff.max_hours}h, ${baseConfig.kol_tuning.freshness_cutoff.penalty}x)` : "OFF",
    });
  }

  // Sort by absolute impact descending
  results.sort((a, b) => Math.abs(b.impact_on_hit_rate) - Math.abs(a.impact_on_hit_rate));

  // Return actual top5 hit rate for UI display (not the internal optimization score)
  const finalResult = runBacktest(snapshots, best, horizon);
  return { features: results, bestConfig: best, bestHitRate: finalResult.top5_hit_rate };
}
