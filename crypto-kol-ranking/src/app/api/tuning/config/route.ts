import { NextRequest, NextResponse } from "next/server";

/**
 * GET /api/tuning/config
 * Returns current production scoring_config (weights, floors, caps, dynamic constants).
 *
 * POST /api/tuning/config
 * Updates production scoring_config. Body: { weights, combined_floor?, combined_cap?, safety_floor?, ...15 dynamic constants, reason? }
 * Requires valid TUNING_SECRET header for auth.
 */

const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL;
const SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
const TUNING_SECRET = process.env.TUNING_SECRET; // optional auth for POST

// v20: 15 dynamic scoring constants (key in DB → key in API response)
const DYNAMIC_CONSTANT_KEYS = [
  "decay_lambda",
  "activity_mult_floor",
  "activity_mult_cap",
  "pa_norm_floor",
  "pa_norm_cap",
  "onchain_mult_floor",
  "onchain_mult_cap",
  "death_pc24_severe",
  "death_pc24_moderate",
  "pump_pc1h_hard",
  "pump_pc5m_hard",
  "stale_hours_severe",
  "gate_top10_pct",
  "gate_min_liquidity",
  "gate_min_holders",
] as const;

function supabaseHeaders() {
  return {
    apikey: SERVICE_KEY!,
    Authorization: `Bearer ${SERVICE_KEY!}`,
    "Content-Type": "application/json",
    Prefer: "return=representation",
  };
}

export async function GET() {
  if (!SUPABASE_URL || !SERVICE_KEY) {
    return NextResponse.json(
      { error: "Missing Supabase configuration" },
      { status: 500 },
    );
  }

  try {
    const res = await fetch(
      `${SUPABASE_URL}/rest/v1/scoring_config?id=eq.1&select=*`,
      { headers: supabaseHeaders(), cache: "no-store" },
    );

    if (!res.ok) {
      return NextResponse.json({ error: await res.text() }, { status: res.status });
    }

    const rows = await res.json();
    if (!rows.length) {
      return NextResponse.json({ error: "No scoring_config found" }, { status: 404 });
    }

    const row = rows[0];

    // Build dynamic constants object
    const constants: Record<string, number> = {};
    for (const key of DYNAMIC_CONSTANT_KEYS) {
      constants[key] = parseFloat(row[key]);
    }

    return NextResponse.json({
      weights: {
        consensus: parseFloat(row.w_consensus),
        conviction: parseFloat(row.w_conviction),
        breadth: parseFloat(row.w_breadth),
        price_action: parseFloat(row.w_price_action),
      },
      combined_floor: parseFloat(row.combined_floor),
      combined_cap: parseFloat(row.combined_cap),
      safety_floor: parseFloat(row.safety_floor),
      constants,
      updated_at: row.updated_at,
      updated_by: row.updated_by,
      change_reason: row.change_reason,
    });
  } catch (e) {
    return NextResponse.json(
      { error: `Failed to fetch config: ${e}` },
      { status: 500 },
    );
  }
}

interface ConfigBody {
  weights?: {
    consensus?: number;
    conviction?: number;
    breadth?: number;
    price_action?: number;
  };
  combined_floor?: number;
  combined_cap?: number;
  safety_floor?: number;
  constants?: Partial<Record<(typeof DYNAMIC_CONSTANT_KEYS)[number], number>>;
  reason?: string;
}

export async function POST(request: NextRequest) {
  if (!SUPABASE_URL || !SERVICE_KEY) {
    return NextResponse.json(
      { error: "Missing Supabase configuration" },
      { status: 500 },
    );
  }

  // Auth: require TUNING_SECRET if configured
  if (TUNING_SECRET) {
    const authHeader = request.headers.get("x-tuning-secret");
    if (authHeader !== TUNING_SECRET) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
  }

  let body: ConfigBody;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  // Build update payload
  const update: Record<string, unknown> = {
    updated_at: new Date().toISOString(),
    updated_by: "tuning_lab",
    change_reason: body.reason || "Manual update from Tuning Lab",
  };

  if (body.weights) {
    const w = body.weights;
    // Validate all 4 weights present
    if (
      w.consensus === undefined ||
      w.conviction === undefined ||
      w.breadth === undefined ||
      w.price_action === undefined
    ) {
      return NextResponse.json(
        { error: "All 4 weights required: consensus, conviction, breadth, price_action" },
        { status: 400 },
      );
    }
    // Validate sum ≈ 1.0
    const sum = w.consensus + w.conviction + w.breadth + w.price_action;
    if (Math.abs(sum - 1.0) > 0.02) {
      return NextResponse.json(
        { error: `Weights must sum to 1.0 (got ${sum.toFixed(3)})` },
        { status: 400 },
      );
    }
    // Validate range [0, 1]
    for (const [k, v] of Object.entries(w)) {
      if (v < 0 || v > 1) {
        return NextResponse.json(
          { error: `Weight ${k} must be between 0 and 1 (got ${v})` },
          { status: 400 },
        );
      }
    }
    update.w_consensus = w.consensus;
    update.w_conviction = w.conviction;
    update.w_breadth = w.breadth;
    update.w_price_action = w.price_action;
  }

  if (body.combined_floor !== undefined) update.combined_floor = body.combined_floor;
  if (body.combined_cap !== undefined) update.combined_cap = body.combined_cap;
  if (body.safety_floor !== undefined) update.safety_floor = body.safety_floor;

  // v20: 15 dynamic scoring constants
  if (body.constants) {
    for (const key of DYNAMIC_CONSTANT_KEYS) {
      if (body.constants[key] !== undefined) {
        update[key] = body.constants[key];
      }
    }
  }

  try {
    const res = await fetch(
      `${SUPABASE_URL}/rest/v1/scoring_config?id=eq.1`,
      {
        method: "PATCH",
        headers: supabaseHeaders(),
        body: JSON.stringify(update),
      },
    );

    if (!res.ok) {
      const errText = await res.text();
      return NextResponse.json({ error: errText }, { status: res.status });
    }

    const rows = await res.json();
    return NextResponse.json({
      success: true,
      config: rows[0],
    });
  } catch (e) {
    return NextResponse.json(
      { error: `Failed to update config: ${e}` },
      { status: 500 },
    );
  }
}
