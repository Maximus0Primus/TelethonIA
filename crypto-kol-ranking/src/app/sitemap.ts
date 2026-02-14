import type { MetadataRoute } from "next";

const BASE_URL = process.env.NEXT_PUBLIC_SITE_URL || "https://cryptosensus.org";

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const staticPages: MetadataRoute.Sitemap = [
    {
      url: BASE_URL,
      lastModified: new Date(),
      changeFrequency: "hourly",
      priority: 1,
    },
    {
      url: `${BASE_URL}/about`,
      changeFrequency: "monthly",
      priority: 0.5,
    },
    {
      url: `${BASE_URL}/disclaimer`,
      changeFrequency: "yearly" as const,
      priority: 0.3,
    },
    {
      url: `${BASE_URL}/privacy`,
      changeFrequency: "yearly" as const,
      priority: 0.3,
    },
    {
      url: `${BASE_URL}/terms`,
      changeFrequency: "yearly" as const,
      priority: 0.3,
    },
  ];

  // Fetch active tokens for dynamic pages
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

  if (supabaseUrl && supabaseKey) {
    try {
      const res = await fetch(
        `${supabaseUrl}/rest/v1/tokens?select=symbol,updated_at&time_window=eq.7d&score=gt.0&order=score.desc&limit=100`,
        {
          headers: {
            apikey: supabaseKey,
            Authorization: `Bearer ${supabaseKey}`,
          },
          cache: "no-store",
        }
      );
      if (res.ok) {
        const tokens: { symbol: string; updated_at: string }[] =
          await res.json();
        for (const t of tokens) {
          const sym = t.symbol.replace("$", "").toLowerCase();
          staticPages.push({
            url: `${BASE_URL}/token/${sym}`,
            lastModified: new Date(t.updated_at),
            changeFrequency: "hourly",
            priority: 0.7,
          });
        }
      }
    } catch {
      // Silently fail — static pages still returned
    }
  }

  return staticPages;
}
