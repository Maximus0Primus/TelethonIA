import { NextRequest, NextResponse } from "next/server";

const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL;
const SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

const HEADERS = {
  apikey: SERVICE_KEY!,
  Authorization: `Bearer ${SERVICE_KEY}`,
  "Content-Type": "application/json",
};

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ symbol: string }> }
) {
  const { symbol } = await params;

  if (!SUPABASE_URL || !SERVICE_KEY) {
    return NextResponse.json(
      { error: "Server configuration error" },
      { status: 500 }
    );
  }

  // Normalize: NFC Unicode form (browser may send NFD), add $ prefix, uppercase
  const normalized = symbol.normalize("NFC").toUpperCase().replace(/^\$/, "");
  const dbSymbol = `$${normalized}`;

  try {
    // Fetch token summary from tokens table (7d window)
    const tokenRes = await fetch(
      `${SUPABASE_URL}/rest/v1/tokens?select=symbol,score,score_conviction,score_momentum,mentions,unique_kols,sentiment,conviction_weighted,trend,change_24h,momentum,breadth&symbol=eq.${encodeURIComponent(dbSymbol)}&time_window=eq.7d&limit=1`,
      { headers: HEADERS, cache: "no-store" }
    );

    if (!tokenRes.ok) {
      return NextResponse.json(
        { error: "Failed to fetch token data" },
        { status: 502 }
      );
    }

    const tokenRows = await tokenRes.json();
    if (!tokenRows || tokenRows.length === 0) {
      return NextResponse.json({ error: "Token not found" }, { status: 404 });
    }

    const token = tokenRows[0];

    // Fetch latest snapshot (on-chain details)
    const snapRes = await fetch(
      `${SUPABASE_URL}/rest/v1/token_snapshots?select=token_address,price_at_snapshot,market_cap,liquidity_usd,volume_24h,holder_count,top10_holder_pct,price_change_5m,price_change_1h,price_change_6h,price_change_24h,risk_score,has_mint_authority,has_freeze_authority,bundle_detected,bundle_count,bundle_pct,whale_count,whale_total_pct,whale_direction,wash_trading_score,token_age_hours,is_pump_fun,ath_ratio,momentum_direction,price_action_score,top_kols,narrative,bubblemaps_score,consensus_val,conviction_val,breadth_val,price_action_val,safety_penalty,onchain_multiplier,crash_pen,activity_mult,squeeze_score,squeeze_state,trend_strength,entry_premium_mult,s_tier_mult,size_mult,stale_pen,rsi_14,macd_histogram,bb_width,bb_pct_b,has_twitter,has_telegram,has_website,boosts_active,social_count,lifecycle_phase,weakest_component,score_interpretation,data_confidence,unique_wallet_24h_change,v_buy_24h_usd,v_sell_24h_usd,whale_new_entries,freshest_mention_hours,s_tier_count&symbol=eq.${encodeURIComponent(dbSymbol)}&order=snapshot_at.desc&limit=1`,
      { headers: HEADERS, cache: "no-store" }
    );

    const snapshot = snapRes.ok
      ? ((await snapRes.json()) as Record<string, unknown>[])?.[0] ?? null
      : null;

    return NextResponse.json({ token, snapshot });
  } catch (error) {
    console.error("Token API error:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
