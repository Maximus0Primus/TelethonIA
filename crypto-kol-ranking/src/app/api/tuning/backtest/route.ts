import { NextRequest, NextResponse } from "next/server";

/**
 * GET /api/tuning/backtest?horizon=12h
 * Fetches all labeled snapshots (with outcome data) for backtesting.
 * These have did_2x_6h/12h/24h filled by outcome_tracker.
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

  const horizon = request.nextUrl.searchParams.get("horizon") || "12h";
  const validHorizons = ["1h", "6h", "12h", "24h", "48h", "72h", "7d"];
  if (!validHorizons.includes(horizon)) {
    return NextResponse.json(
      { error: `Invalid horizon. Use: ${validHorizons.join(", ")}` },
      { status: 400 }
    );
  }

  const outcomeCol = `did_2x_${horizon}`;

  const columns = [
    "symbol",
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
    // Extraction source counts
    "ca_mention_count",
    "ticker_mention_count",
    "url_mention_count",
    "has_ca_mention",
    // Outcomes (all 7 horizons)
    "did_2x_1h",
    "did_2x_6h",
    "did_2x_12h",
    "did_2x_24h",
    "did_2x_48h",
    "did_2x_72h",
    "did_2x_7d",
    "price_at_snapshot",
    "price_after_1h",
    "price_after_6h",
    "price_after_12h",
    "price_after_24h",
    "price_after_48h",
    "price_after_72h",
    "price_after_7d",
    "max_price_1h",
    "max_price_6h",
    "max_price_12h",
    "max_price_24h",
    "max_price_48h",
    "max_price_72h",
    "max_price_7d",
  ].join(",");

  try {
    // Only fetch snapshots where the relevant outcome column is NOT null
    // (meaning outcome_tracker has already labeled them)
    const res = await fetch(
      `${url}/rest/v1/token_snapshots?select=${columns}&${outcomeCol}=not.is.null&consensus_val=not.is.null&order=snapshot_at.desc&limit=5000`,
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

    const snapshots = await res.json();

    return NextResponse.json({
      data: snapshots,
      total: snapshots.length,
      horizon,
    });
  } catch (error) {
    console.error("Backtest API error:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
