import { NextResponse } from "next/server";
import { KOL_TIERS, KOL_SCORES } from "@/lib/kol-tiers";

interface RpcRowV1 {
  kol_name: string;
  unique_tokens: number;
  labeled_calls: number;
  hits_any: number;
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

interface RpcRowV2 {
  kol_name: string;
  total_calls: number;
  with_entry_price: number;
  hits_2x: number;
  avg_max_return: number | null;
  best_return: number | null;
  unique_tokens: number;
  last_active: string | null;
}

export interface KolLeaderboardEntry {
  name: string;
  tier: "S" | "A";
  conviction: number;
  score: number | null;
  uniqueTokens: number;
  labeledCalls: number;
  hitsAny: number;
  hits12h: number;
  hits24h: number;
  hits48h: number;
  hits72h: number;
  hits7d: number;
  winRateAll: number | null;
  lastActive: string | null;
  // v2: exact call-price based metrics
  totalCalls: number;
  withEntryPrice: number;
  hits2xExact: number;
  winRate2xExact: number | null;
  avgMaxReturn: number | null;
  bestReturn: number | null;
}

async function callRpc(caOnly: boolean): Promise<{
  data: RpcRowV1[] | null;
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

  const data: RpcRowV1[] = await res.json();
  return { data, error: null };
}

async function callRpcV2(): Promise<{
  data: RpcRowV2[] | null;
  error: string | null;
}> {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY;

  if (!url || !key) {
    return { data: null, error: "Missing Supabase configuration" };
  }

  const res = await fetch(`${url}/rest/v1/rpc/get_kol_leaderboard_v2`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      apikey: key,
      Authorization: `Bearer ${key}`,
    },
    body: "{}",
    cache: "no-store",
  });

  if (!res.ok) {
    const errText = await res.text();
    return { data: null, error: errText };
  }

  const data: RpcRowV2[] = await res.json();
  return { data, error: null };
}

/**
 * Compute dynamic normalized scores from RPC data.
 * Uses per-horizon win rates so incomplete horizons don't deflate scores.
 * Matches kol_scorer.py logic: score = (hit_rate / baseline), capped [0.1, 3.0].
 */
function computeDynamicScores(
  rpcRows: RpcRowV1[]
): Map<string, number> {
  const MIN_CALLS = 5;
  const scores = new Map<string, number>();

  // Compute baseline: overall hit rate (any horizon) across all KOLs
  let totalHits = 0;
  let totalLabeled = 0;
  for (const row of rpcRows) {
    const labeled = Number(row.labeled_calls);
    const hits = Number(row.hits_any);
    if (labeled >= 1) {
      totalLabeled += labeled;
      totalHits += hits;
    }
  }

  const baseline = totalLabeled > 0 ? totalHits / totalLabeled : 0.1;
  const safeBaseline = Math.max(baseline, 0.01);

  for (const row of rpcRows) {
    const labeled = Number(row.labeled_calls);
    const hits = Number(row.hits_any);
    if (labeled < MIN_CALLS) continue;

    const winRate = hits / labeled;
    const normalized = winRate / safeBaseline;
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

    // Fetch both v1 (snapshot-based) and v2 (call-price-based) leaderboards
    const [v1Result, v2Result] = await Promise.all([
      callRpc(caOnly),
      callRpcV2(),
    ]);

    if (v1Result.error || !v1Result.data) {
      console.error("KOL leaderboard RPC error:", v1Result.error);
      return NextResponse.json(
        { error: "Failed to fetch KOL leaderboard" },
        { status: 500 }
      );
    }

    const rpcRows = v1Result.data;

    // Compute dynamic scores from live RPC data
    const dynamicScores = computeDynamicScores(rpcRows);

    // Build maps from RPC data
    const rpcMap = new Map<string, RpcRowV1>();
    for (const row of rpcRows) {
      rpcMap.set(row.kol_name, row);
    }

    const v2Map = new Map<string, RpcRowV2>();
    if (v2Result.data) {
      for (const row of v2Result.data) {
        v2Map.set(row.kol_name, row);
      }
    }

    // Merge all 59 KOLs (even those with no snapshots)
    const entries: KolLeaderboardEntry[] = Object.entries(KOL_TIERS).map(
      ([name, tierInfo]) => {
        const rpc = rpcMap.get(name);
        const v2 = v2Map.get(name);
        const labeledCalls = rpc ? Number(rpc.labeled_calls) : 0;
        const hitsAny = rpc ? Number(rpc.hits_any) : 0;
        const hits12h = rpc ? Number(rpc.hits_12h) : 0;
        const hits24h = rpc ? Number(rpc.hits_24h) : 0;
        const hits48h = rpc ? Number(rpc.hits_48h) : 0;
        const hits72h = rpc ? Number(rpc.hits_72h) : 0;
        const hits7d = rpc ? Number(rpc.hits_7d) : 0;

        // Dynamic score primary, static KOL_SCORES fallback
        const score = dynamicScores.get(name) ?? KOL_SCORES[name] ?? null;

        // Overall win rate: hits at ANY horizon / all labeled calls (no minimum threshold)
        const winRateAll = labeledCalls >= 1 ? hitsAny / labeledCalls : null;

        // v2: exact call-price based metrics
        const totalCalls = v2 ? Number(v2.total_calls) : 0;
        const withEntryPrice = v2 ? Number(v2.with_entry_price) : 0;
        const hits2xExact = v2 ? Number(v2.hits_2x) : 0;
        // v2 only reliable with enough samples — prevent 1-sample override of v1
        const winRate2xExact = withEntryPrice >= 5 ? hits2xExact / withEntryPrice : null;

        return {
          name,
          tier: tierInfo.tier,
          conviction: tierInfo.conviction,
          score,
          uniqueTokens: rpc ? Number(rpc.unique_tokens) : 0,
          labeledCalls,
          hitsAny,
          hits12h,
          hits24h,
          hits48h,
          hits72h,
          hits7d,
          winRateAll,
          lastActive: rpc?.last_active ?? v2?.last_active ?? null,
          totalCalls,
          withEntryPrice,
          hits2xExact,
          winRate2xExact,
          avgMaxReturn: v2?.avg_max_return != null ? Number(v2.avg_max_return) : null,
          bestReturn: v2?.best_return != null ? Number(v2.best_return) : null,
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
