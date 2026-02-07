import { NextRequest, NextResponse } from "next/server";
import { createHmac, timingSafeEqual } from "crypto";

export async function POST(req: NextRequest) {
  const ACCESS_TOKEN_SECRET = process.env.ACCESS_TOKEN_SECRET;

  if (!ACCESS_TOKEN_SECRET) {
    return NextResponse.json(
      { error: "Server misconfigured" },
      { status: 500 }
    );
  }

  let token: string;
  try {
    const body = await req.json();
    token = typeof body.token === "string" ? body.token : "";
  } catch {
    return NextResponse.json({ valid: false });
  }

  // Token format: "hmac_hex:expiresAt"
  const separatorIndex = token.lastIndexOf(":");
  if (separatorIndex === -1) {
    return NextResponse.json({ valid: false });
  }

  const hmacHex = token.slice(0, separatorIndex);
  const expiresAtStr = token.slice(separatorIndex + 1);
  const expiresAt = Number(expiresAtStr);

  if (!expiresAt || isNaN(expiresAt)) {
    return NextResponse.json({ valid: false });
  }

  // Check expiry
  if (Date.now() > expiresAt) {
    return NextResponse.json({ valid: false });
  }

  // Recompute HMAC and compare
  const payload = `cryptosensus_access:${expiresAt}`;
  const expected = createHmac("sha256", ACCESS_TOKEN_SECRET)
    .update(payload)
    .digest("hex");

  try {
    const a = Buffer.from(hmacHex, "hex");
    const b = Buffer.from(expected, "hex");
    if (a.length !== b.length || !timingSafeEqual(a, b)) {
      return NextResponse.json({ valid: false });
    }
  } catch {
    return NextResponse.json({ valid: false });
  }

  return NextResponse.json({ valid: true });
}
