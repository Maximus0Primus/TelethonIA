import { NextRequest, NextResponse } from "next/server";

/**
 * GET /api/tuning/snapshots
 * Fetches the latest token_snapshots with all component values for client-side re-scoring.
 * Returns one snapshot per symbol (most recent).
 */
export async function GET(request: NextRequest) {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY;

  if (!url || !key) {
    return NextResponse.json(
      { error: "Missing Supabase configuration" },
      { status: 500 }
    );
  }

  // Fetch recent snapshots — get enough to deduplicate by symbol
  // PostgREST doesn't support DISTINCT ON, so we fetch extras and dedupe in JS
  const columns = [
    "symbol",
    "token_address",
    "snapshot_at",
    // 5 component values
    "consensus_val",
    "sentiment_val",
    "conviction_val",
    "breadth_val",
    "price_action_val",
    // Multipliers
    "safety_penalty",
    "onchain_multiplier",
    "death_penalty",
    "already_pumped_penalty",
    "crash_pen",
    "activity_mult",
    "squeeze_score",
    "squeeze_state",
    "trend_strength",
    "confirmation_pillars",
    "entry_premium_mult",
    "pump_bonus",
    "wash_pen",
    "pvp_pen",
    "pump_pen",
    "breadth_pen",
    "stale_pen",
    // Raw data for context
    "price_change_24h",
    "volume_24h",
    "liquidity_usd",
    "mentions",
    "freshest_mention_hours",
    "top_kols",
    "price_at_snapshot",
    "market_cap",
    // Scores (original production scores)
    "score",
    "breadth",
  ].join(",");

  try {
    // We need the score field — it's stored in tokens table, not snapshots.
    // Instead, fetch snapshots and sort by snapshot_at desc, dedupe by symbol.
    const res = await fetch(
      `${url}/rest/v1/token_snapshots?select=${columns}&order=snapshot_at.desc&limit=200`,
      {
        headers: {
          apikey: key,
          Authorization: `Bearer ${key}`,
        },
        cache: "no-store",
      }
    );

    if (!res.ok) {
      const errText = await res.text();
      return NextResponse.json({ error: errText }, { status: res.status });
    }

    const allSnapshots = await res.json();

    // Deduplicate: keep only the most recent snapshot per symbol
    const seen = new Set<string>();
    const uniqueSnapshots = [];
    for (const snap of allSnapshots) {
      if (!seen.has(snap.symbol)) {
        seen.add(snap.symbol);
        uniqueSnapshots.push(snap);
      }
    }

    return NextResponse.json({
      data: uniqueSnapshots,
      total: uniqueSnapshots.length,
    });
  } catch (error) {
    console.error("Tuning snapshots API error:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
