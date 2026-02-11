import { NextRequest, NextResponse } from "next/server";
import { createHmac, timingSafeEqual } from "crypto";

// ---------------------------------------------------------------------------
// Rate-limiting state (in-memory, resets on restart)
// ---------------------------------------------------------------------------
interface RateLimitEntry {
  attempts: number;
  tier: number;
  blockedUntil: number | null;
}

const rateLimitMap = new Map<string, RateLimitEntry>();

const MAX_ATTEMPTS = 10;
const TIER_DURATIONS = [
  10 * 60 * 1000, // Tier 0 → 10 min
  60 * 60 * 1000, // Tier 1 → 1 hour
  24 * 60 * 60 * 1000, // Tier 2+ → 24 hours
];

function getClientIP(req: NextRequest): string {
  return (
    req.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ||
    req.headers.get("x-real-ip") ||
    "unknown"
  );
}

function getLockDuration(tier: number): number {
  return TIER_DURATIONS[Math.min(tier, TIER_DURATIONS.length - 1)];
}

// ---------------------------------------------------------------------------
// Token generation
// ---------------------------------------------------------------------------
function generateToken(secret: string): { token: string; expiresAt: number } {
  const expiresAt = Date.now() + 30 * 24 * 60 * 60 * 1000; // 30 days
  const payload = `cryptosensus_access:${expiresAt}`;
  const hmac = createHmac("sha256", secret).update(payload).digest("hex");
  const token = `${hmac}:${expiresAt}`;
  return { token, expiresAt };
}

// ---------------------------------------------------------------------------
// Constant-time password comparison using HMAC
// ---------------------------------------------------------------------------
function passwordMatches(input: string, expected: string): boolean {
  const a = createHmac("sha256", "pw-check").update(input).digest();
  const b = createHmac("sha256", "pw-check").update(expected).digest();
  return timingSafeEqual(a, b);
}

// ---------------------------------------------------------------------------
// POST handler
// ---------------------------------------------------------------------------
export async function POST(req: NextRequest) {
  const ACCESS_PASSWORD = process.env.ACCESS_PASSWORD;
  const ACCESS_TOKEN_SECRET = process.env.ACCESS_TOKEN_SECRET;

  if (!ACCESS_PASSWORD || !ACCESS_TOKEN_SECRET) {
    return NextResponse.json(
      { error: "Server misconfigured" },
      { status: 500 }
    );
  }

  // Parse body
  let password: string;
  try {
    const body = await req.json();
    password = typeof body.password === "string" ? body.password.trim() : "";
  } catch {
    return NextResponse.json({ error: "Invalid request" }, { status: 400 });
  }

  const ip = getClientIP(req);

  // Rate-limit check
  let entry = rateLimitMap.get(ip);
  if (!entry) {
    entry = { attempts: 0, tier: 0, blockedUntil: null };
    rateLimitMap.set(ip, entry);
  }

  if (entry.blockedUntil) {
    const remaining = entry.blockedUntil - Date.now();
    if (remaining > 0) {
      return NextResponse.json(
        {
          error: "Too many attempts",
          blocked: true,
          retryAfter: Math.ceil(remaining / 1000),
        },
        { status: 429 }
      );
    }
    // Block expired — reset attempts but keep tier
    entry.attempts = 0;
    entry.blockedUntil = null;
  }

  // Verify password
  if (!passwordMatches(password, ACCESS_PASSWORD)) {
    entry.attempts += 1;
    const remaining = MAX_ATTEMPTS - entry.attempts;

    if (entry.attempts >= MAX_ATTEMPTS) {
      const lockMs = getLockDuration(entry.tier);
      entry.blockedUntil = Date.now() + lockMs;
      entry.tier += 1;

      return NextResponse.json(
        {
          error: "Too many attempts",
          blocked: true,
          retryAfter: Math.ceil(lockMs / 1000),
        },
        { status: 429 }
      );
    }

    return NextResponse.json(
      { error: "Wrong password", attemptsRemaining: remaining },
      { status: 401 }
    );
  }

  // Correct password — reset rate limit and return token
  rateLimitMap.delete(ip);

  const { token, expiresAt } = generateToken(ACCESS_TOKEN_SECRET);
  return NextResponse.json({ token, expiresAt });
}
