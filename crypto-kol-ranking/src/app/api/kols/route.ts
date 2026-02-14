import { NextResponse } from "next/server";
import { KOL_TIERS, KOL_SCORES } from "@/lib/kol-tiers";

interface RpcRow {
  kol_name: string;
  unique_tokens: number;
  labeled_calls: number;
  labeled_12h: number;
  labeled_24h: number;
  labeled_48h: number;
  labeled_72h: number;
  labeled_7d: number;
  hits_12h: number;
  hits_24h: number;
  hits_48h: number;
  hits_72h: number;
  hits_7d: number;
  last_active: string | null;
}

export interface KolLeaderboardEntry {
  name: string;
  tier: "S" | "A";
  conviction: number;
  score: number | null;
  uniqueTokens: number;
  labeledCalls: number;
  hits12h: number;
  hits24h: number;
  hits48h: number;
  hits72h: number;
  hits7d: number;
  winRate12h: number | null;
  winRate24h: number | null;
  winRateAny: number | null;
  lastActive: string | null;
}

async function callRpc(caOnly: boolean): Promise<{
  data: RpcRow[] | null;
  error: string | null;
}> {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY;

  if (!url || !key) {
    return { data: null, error: "Missing Supabase configuration" };
  }

  const res = await fetch(`${url}/rest/v1/rpc/get_kol_leaderboard`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      apikey: key,
      Authorization: `Bearer ${key}`,
    },
    body: JSON.stringify({ p_ca_only: caOnly }),
    cache: "no-store",
  });

  if (!res.ok) {
    const errText = await res.text();
    return { data: null, error: errText };
  }

  const data: RpcRow[] = await res.json();
  return { data, error: null };
}

/**
 * Best per-horizon win rate for a KOL.
 * Uses per-horizon labeled counts so denominator matches the horizon.
 * Requires MIN_HORIZON_CALLS per horizon to count.
 */
function bestHorizonWinRate(row: RpcRow, minCalls: number): number | null {
  const horizons: [number, number][] = [
    [Number(row.hits_12h), Number(row.labeled_12h)],
    [Number(row.hits_24h), Number(row.labeled_24h)],
    [Number(row.hits_48h), Number(row.labeled_48h)],
    [Number(row.hits_72h), Number(row.labeled_72h)],
    [Number(row.hits_7d), Number(row.labeled_7d)],
  ];
  let best: number | null = null;
  for (const [hits, labeled] of horizons) {
    if (labeled >= minCalls) {
      const rate = hits / labeled;
      if (best === null || rate > best) best = rate;
    }
  }
  return best;
}

/**
 * Compute dynamic normalized scores from RPC data.
 * Uses per-horizon win rates so incomplete horizons don't deflate scores.
 * Matches kol_scorer.py logic: score = (hit_rate / baseline), capped [0.1, 3.0].
 */
function computeDynamicScores(
  rpcRows: RpcRow[]
): Map<string, number> {
  const MIN_CALLS = 5;
  const scores = new Map<string, number>();

  // Compute baseline: best per-horizon win rate across all KOLs
  let totalHits = 0;
  let totalLabeled = 0;
  for (const row of rpcRows) {
    // Use the longest labeled horizon for baseline (most complete data)
    const labeled24 = Number(row.labeled_24h);
    const hits24 = Number(row.hits_24h);
    const labeled12 = Number(row.labeled_12h);
    const hits12 = Number(row.hits_12h);
    if (labeled24 >= 1) {
      totalLabeled += labeled24;
      totalHits += hits24;
    } else if (labeled12 >= 1) {
      totalLabeled += labeled12;
      totalHits += hits12;
    }
  }

  const baseline = totalLabeled > 0 ? totalHits / totalLabeled : 0.1;
  const safeBaseline = Math.max(baseline, 0.01);

  for (const row of rpcRows) {
    const best = bestHorizonWinRate(row, MIN_CALLS);
    if (best === null) continue;

    const normalized = best / safeBaseline;
    scores.set(
      row.kol_name,
      Math.round(Math.max(0.1, Math.min(3.0, normalized)) * 1000) / 1000
    );
  }

  return scores;
}

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const caOnly = searchParams.get("ca_only") === "true";

    const { data: rpcRows, error: rpcError } = await callRpc(caOnly);

    if (rpcError || !rpcRows) {
      console.error("KOL leaderboard RPC error:", rpcError);
      return NextResponse.json(
        { error: "Failed to fetch KOL leaderboard" },
        { status: 500 }
      );
    }

    // Compute dynamic scores from live RPC data
    const dynamicScores = computeDynamicScores(rpcRows);

    // Build a map from RPC data
    const rpcMap = new Map<string, RpcRow>();
    for (const row of rpcRows) {
      rpcMap.set(row.kol_name, row);
    }

    // Merge all 59 KOLs (even those with no snapshots)
    const entries: KolLeaderboardEntry[] = Object.entries(KOL_TIERS).map(
      ([name, tierInfo]) => {
        const rpc = rpcMap.get(name);
        const labeledCalls = rpc ? Number(rpc.labeled_calls) : 0;
        const labeled12h = rpc ? Number(rpc.labeled_12h) : 0;
        const labeled24h = rpc ? Number(rpc.labeled_24h) : 0;
        const hits12h = rpc ? Number(rpc.hits_12h) : 0;
        const hits24h = rpc ? Number(rpc.hits_24h) : 0;
        const hits48h = rpc ? Number(rpc.hits_48h) : 0;
        const hits72h = rpc ? Number(rpc.hits_72h) : 0;
        const hits7d = rpc ? Number(rpc.hits_7d) : 0;

        // Dynamic score primary, static KOL_SCORES fallback
        const score = dynamicScores.get(name) ?? KOL_SCORES[name] ?? null;

        // Per-horizon win rates (use per-horizon labeled counts as denominator)
        const winRate12h = labeled12h >= 3 ? hits12h / labeled12h : null;
        const winRate24h = labeled24h >= 3 ? hits24h / labeled24h : null;
        // Best available horizon win rate
        const winRateAny = rpc ? bestHorizonWinRate(rpc, 3) : null;

        return {
          name,
          tier: tierInfo.tier,
          conviction: tierInfo.conviction,
          score,
          uniqueTokens: rpc ? Number(rpc.unique_tokens) : 0,
          labeledCalls,
          hits12h,
          hits24h,
          hits48h,
          hits72h,
          hits7d,
          winRate12h,
          winRate24h,
          winRateAny,
          lastActive: rpc?.last_active ?? null,
        };
      }
    );

    // Default sort: by score desc (null last), then by tier (S first)
    entries.sort((a, b) => {
      if (a.score !== null && b.score !== null) return b.score - a.score;
      if (a.score !== null) return -1;
      if (b.score !== null) return 1;
      if (a.tier !== b.tier) return a.tier === "S" ? -1 : 1;
      return a.name.localeCompare(b.name);
    });

    return NextResponse.json({ data: entries });
  } catch (error) {
    console.error("KOL API error:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
