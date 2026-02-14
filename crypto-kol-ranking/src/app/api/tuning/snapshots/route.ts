import { NextRequest, NextResponse } from "next/server";

/**
 * GET /api/tuning/snapshots
 * Fetches token_snapshots with all component values for client-side re-scoring.
 *
 * Query params:
 *   ?cycle=<ISO timestamp>  — return snapshots from that specific scrape cycle
 *   ?cycles=true            — return list of available scrape cycle timestamps
 *   (no params)             — return latest snapshot per symbol (default)
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

  const headers = {
    apikey: key,
    Authorization: `Bearer ${key}`,
  };

  const { searchParams } = request.nextUrl;

  // --- Mode 1: List available scrape cycles ---
  if (searchParams.get("cycles") === "true") {
    try {
      const res = await fetch(
        `${url}/rest/v1/token_snapshots?select=snapshot_at&order=snapshot_at.desc&limit=2000`,
        { headers, cache: "no-store" },
      );
      if (!res.ok) {
        return NextResponse.json({ error: await res.text() }, { status: res.status });
      }
      const rows: { snapshot_at: string }[] = await res.json();

      // Group by scrape cycle: snapshots within 2 minutes are the same cycle
      const cycles: { ts: string; count: number }[] = [];
      for (const row of rows) {
        const ts = new Date(row.snapshot_at).getTime();
        const last = cycles[cycles.length - 1];
        if (last && Math.abs(ts - new Date(last.ts).getTime()) < 2 * 60 * 1000) {
          last.count++;
        } else {
          cycles.push({ ts: row.snapshot_at, count: 1 });
        }
      }

      return NextResponse.json({ cycles });
    } catch (error) {
      console.error("Cycles API error:", error);
      return NextResponse.json({ error: "Internal server error" }, { status: 500 });
    }
  }

  // --- Columns for snapshot queries ---
  const columns = [
    "symbol", "token_address", "snapshot_at",
    "consensus_val", "sentiment_val", "conviction_val", "breadth_val", "price_action_val",
    "safety_penalty", "onchain_multiplier", "death_penalty", "already_pumped_penalty",
    "crash_pen", "activity_mult", "squeeze_score", "squeeze_state", "trend_strength",
    "confirmation_pillars", "entry_premium_mult", "pump_bonus", "wash_pen", "pvp_pen",
    "pump_pen", "breadth_pen", "stale_pen", "pump_momentum_pen",
    "price_change_24h", "volume_24h", "liquidity_usd", "mentions",
    "freshest_mention_hours", "top_kols", "price_at_snapshot", "market_cap", "unique_kols", "s_tier_count",
    "score_at_snapshot", "breadth", "size_mult", "s_tier_mult",
    "ca_mention_count", "ticker_mention_count", "url_mention_count", "has_ca_mention",
    "did_2x_1h", "did_2x_6h", "did_2x_12h", "did_2x_24h", "did_2x_48h", "did_2x_72h", "did_2x_7d",
  ].join(",");

  // --- Mode 2: Snapshots for a specific cycle ---
  const cycleParam = searchParams.get("cycle");
  if (cycleParam) {
    try {
      const cycleTs = new Date(cycleParam);
      const windowStart = new Date(cycleTs.getTime() - 2 * 60 * 1000).toISOString();
      const windowEnd = new Date(cycleTs.getTime() + 2 * 60 * 1000).toISOString();

      const res = await fetch(
        `${url}/rest/v1/token_snapshots?select=${columns}` +
        `&snapshot_at=gte.${windowStart}&snapshot_at=lte.${windowEnd}` +
        `&order=snapshot_at.desc&limit=200`,
        { headers, cache: "no-store" },
      );
      if (!res.ok) {
        return NextResponse.json({ error: await res.text() }, { status: res.status });
      }
      const snapshots = await res.json();

      // Dedupe by token_address within this cycle (fallback to symbol)
      const seen = new Set<string>();
      const unique = [];
      for (const snap of snapshots) {
        const key = snap.token_address || snap.symbol;
        if (!seen.has(key)) {
          seen.add(key);
          unique.push(snap);
        }
      }

      return NextResponse.json({ data: unique, total: unique.length });
    } catch (error) {
      console.error("Cycle snapshots API error:", error);
      return NextResponse.json({ error: "Internal server error" }, { status: 500 });
    }
  }

  // --- Mode 3: Latest snapshots (default) ---
  try {
    const res = await fetch(
      `${url}/rest/v1/token_snapshots?select=${columns}&order=snapshot_at.desc&limit=200`,
      { headers, cache: "no-store" },
    );
    if (!res.ok) {
      return NextResponse.json({ error: await res.text() }, { status: res.status });
    }

    const allSnapshots = await res.json();

    const seen = new Set<string>();
    const uniqueSnapshots = [];
    for (const snap of allSnapshots) {
      const key = snap.token_address || snap.symbol;
      if (!seen.has(key)) {
        seen.add(key);
        uniqueSnapshots.push(snap);
      }
    }

    return NextResponse.json({
      data: uniqueSnapshots,
      total: uniqueSnapshots.length,
    });
  } catch (error) {
    console.error("Tuning snapshots API error:", error);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
