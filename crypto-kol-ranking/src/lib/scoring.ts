/**
 * Cryptosensus score calculation for token ranking
 * Based on the formula from the plan:
 *
 * Score = 0.35 * kolConsensus +
 *         0.25 * ((sentiment + 1) / 2) +
 *         0.20 * convictionWeight +
 *         0.15 * ((momentum + 1) / 2) +
 *         0.05 * breadth
 */

export interface ScoreComponents {
  /** 0-1: Proportion of total KOLs mentioning the token */
  kolConsensus: number;
  /** -1 to 1: Average sentiment across mentions */
  sentiment: number;
  /** 0-1: Average conviction weight of mentioning KOLs */
  convictionWeight: number;
  /** -1 to 1: Trend in mention frequency */
  momentum: number;
  /** 0-1: Diversity of groups mentioning */
  breadth: number;
}

const WEIGHTS = {
  kolConsensus: 0.35,
  sentiment: 0.25,
  convictionWeight: 0.20,
  momentum: 0.15,
  breadth: 0.05,
} as const;

/**
 * Calculate the final score (0-100) from components
 */
export function calculateScore(components: ScoreComponents): number {
  const {
    kolConsensus,
    sentiment,
    convictionWeight,
    momentum,
    breadth,
  } = components;

  // Normalize sentiment and momentum from [-1, 1] to [0, 1]
  const normalizedSentiment = (sentiment + 1) / 2;
  const normalizedMomentum = (momentum + 1) / 2;

  const raw =
    WEIGHTS.kolConsensus * kolConsensus +
    WEIGHTS.sentiment * normalizedSentiment +
    WEIGHTS.convictionWeight * convictionWeight +
    WEIGHTS.momentum * normalizedMomentum +
    WEIGHTS.breadth * breadth;

  return Math.round(raw * 100);
}

/**
 * Determine trend based on momentum and recent changes
 */
export function determineTrend(
  momentum: number,
  change24h: number
): "up" | "down" | "stable" {
  if (momentum > 0.2 && change24h > 0) return "up";
  if (momentum < -0.2 && change24h < 0) return "down";
  return "stable";
}

/**
 * Calculate KOL consensus score
 * @param uniqueKols Number of unique KOLs mentioning
 * @param totalKols Total number of active KOLs
 */
export function calculateKolConsensus(
  uniqueKols: number,
  totalKols: number
): number {
  if (totalKols === 0) return 0;
  // Use a log scale to prevent dominance by very popular tokens
  return Math.min(1, Math.log2(uniqueKols + 1) / Math.log2(totalKols + 1));
}

/**
 * Calculate breadth score (diversity of groups)
 * @param groupsMentioning Number of unique groups mentioning
 * @param totalGroups Total number of active groups
 */
export function calculateBreadth(
  groupsMentioning: number,
  totalGroups: number
): number {
  if (totalGroups === 0) return 0;
  return Math.min(1, groupsMentioning / totalGroups);
}

/**
 * Calculate conviction-weighted score
 * @param mentionsByConviction Map of conviction level (6-10) to mention count
 */
export function calculateConvictionWeight(
  mentionsByConviction: Record<number, number>
): number {
  let totalWeightedMentions = 0;
  let totalMentions = 0;

  for (const [conviction, count] of Object.entries(mentionsByConviction)) {
    const convictionNum = parseInt(conviction, 10);
    totalWeightedMentions += convictionNum * count;
    totalMentions += count;
  }

  if (totalMentions === 0) return 0;

  // Normalize to 0-1 range (conviction ranges from 6-10, so divide by 10)
  return (totalWeightedMentions / totalMentions) / 10;
}

/**
 * Calculate momentum based on mention frequency change
 * @param currentMentions Mentions in current period
 * @param previousMentions Mentions in previous period
 */
export function calculateMomentum(
  currentMentions: number,
  previousMentions: number
): number {
  if (previousMentions === 0) {
    return currentMentions > 0 ? 1 : 0;
  }

  const change = (currentMentions - previousMentions) / previousMentions;

  // Clamp to [-1, 1]
  return Math.max(-1, Math.min(1, change));
}
