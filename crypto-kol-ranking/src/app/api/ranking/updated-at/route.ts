import { NextResponse } from "next/server";

export async function GET() {
  try {
    const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
    const key = process.env.SUPABASE_SERVICE_ROLE_KEY;

    if (!url || !key) {
      return NextResponse.json({ updated_at: null });
    }

    const res = await fetch(
      `${url}/rest/v1/scrape_metadata?id=eq.1&select=updated_at`,
      {
        headers: {
          apikey: key,
          Authorization: `Bearer ${key}`,
        },
        cache: "no-store",
      }
    );

    if (!res.ok) {
      return NextResponse.json({ updated_at: null });
    }

    const rows = await res.json();
    return NextResponse.json({
      updated_at: rows.length > 0 ? rows[0].updated_at : null,
    });
  } catch {
    return NextResponse.json({ updated_at: null });
  }
}
