import { NextRequest, NextResponse } from "next/server";
import { createServiceRoleClient } from "@/lib/supabase/server";

interface RankingRow {
  rank: number;
  symbol: string;
  score: number;
  score_conviction: number;
  score_momentum: number;
  mentions: number;
  unique_kols: number;
  sentiment: number;
  trend: string;
  change_24h: number;
  weakest_component: string | null;
  score_interpretation: string | null;
  data_confidence: number | null;
  token_address: string | null;
}

async function callRpc(
  limit: number,
  offset: number,
): Promise<{ data: RankingRow[] | null; error: string | null }> {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY;

  if (!url || !key) {
    return { data: null, error: "Missing Supabase configuration" };
  }

  const res = await fetch(`${url}/rest/v1/rpc/get_token_ranking`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      apikey: key,
      Authorization: `Bearer ${key}`,
    },
    body: JSON.stringify({
      p_time_window: "7d",
      p_limit: limit,
      p_offset: offset,
      p_blend: 0,
    }),
    cache: "no-store",
  });

  if (!res.ok) {
    const errText = await res.text();
    return { data: null, error: errText };
  }

  const data: RankingRow[] = await res.json();
  return { data, error: null };
}

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;

  const limit = parseInt(searchParams.get("limit") || "10", 10);
  const offset = parseInt(searchParams.get("offset") || "0", 10);

  // Validate limit and offset
  if (isNaN(limit) || limit < 1 || limit > 100) {
    return NextResponse.json(
      { error: "Limit must be a number between 1 and 100" },
      { status: 400 }
    );
  }

  if (isNaN(offset) || offset < 0) {
    return NextResponse.json(
      { error: "Offset must be a non-negative number" },
      { status: 400 }
    );
  }

  try {
    const supabase = createServiceRoleClient();

    // Fetch ranked tokens via the locked-down RPC (direct REST call)
    const { data: tokens, error: rpcError } = await callRpc(limit, offset);

    if (rpcError || !tokens) {
      console.error("RPC error:", rpcError);
      return NextResponse.json(
        { error: "Failed to fetch ranking data" },
        { status: 500 }
      );
    }

    // Get total count for pagination
    const { count } = await supabase
      .from("tokens")
      .select("*", { count: "exact", head: true })
      .eq("time_window", "7d");

    const total = count ?? 0;

    // Map DB column names to camelCase for frontend
    const mappedTokens = tokens.map((t) => ({
      rank: Number(t.rank),
      symbol: t.symbol,
      score: t.score,
      mentions: t.mentions,
      uniqueKols: t.unique_kols,
      sentiment: Number(t.sentiment),
      trend: t.trend,
      change24h: Number(t.change_24h ?? 0),
      weakestComponent: t.weakest_component ?? null,
      scoreInterpretation: t.score_interpretation ?? null,
      dataConfidence: t.data_confidence != null ? Number(t.data_confidence) : null,
      tokenAddress: t.token_address ?? null,
    }));

    // Fetch stats from scrape_metadata (use REST to avoid type issues)
    let metaStats: Record<string, unknown> = {
      totalTokens: total,
      totalMentions: 0,
      avgSentiment: 0,
      totalKols: 0,
    };
    let metaUpdatedAt = new Date().toISOString();

    try {
      const metaRes = await fetch(
        `${process.env.NEXT_PUBLIC_SUPABASE_URL}/rest/v1/scrape_metadata?id=eq.1&select=updated_at,stats`,
        {
          headers: {
            apikey: process.env.SUPABASE_SERVICE_ROLE_KEY!,
            Authorization: `Bearer ${process.env.SUPABASE_SERVICE_ROLE_KEY!}`,
          },
          cache: "no-store",
        }
      );
      if (metaRes.ok) {
        const rows = await metaRes.json();
        if (rows.length > 0) {
          metaUpdatedAt = rows[0].updated_at ?? metaUpdatedAt;
          metaStats = rows[0].stats ?? metaStats;
        }
      }
    } catch {
      // Fall through with defaults
    }

    return NextResponse.json({
      data: mappedTokens,
      pagination: {
        limit,
        offset,
        total,
        hasMore: total > offset + limit,
      },
      stats: metaStats,
      updated_at: metaUpdatedAt,
    });
  } catch (error) {
    console.error("API error:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
