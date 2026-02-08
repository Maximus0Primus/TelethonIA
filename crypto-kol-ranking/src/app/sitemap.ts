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
  ];

  // Fetch token symbols for dynamic pages
  let tokenPages: MetadataRoute.Sitemap = [];
  try {
    const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
    const key = process.env.SUPABASE_SERVICE_ROLE_KEY;
    if (url && key) {
      const res = await fetch(
        `${url}/rest/v1/tokens?select=symbol&order=score.desc&limit=50`,
        {
          headers: {
            apikey: key,
            Authorization: `Bearer ${key}`,
          },
          next: { revalidate: 3600 },
        }
      );
      if (res.ok) {
        const tokens: { symbol: string }[] = await res.json();
        tokenPages = tokens.map((t) => ({
          url: `${BASE_URL}/token/${t.symbol.replace("$", "")}`,
          lastModified: new Date(),
          changeFrequency: "hourly" as const,
          priority: 0.7,
        }));
      }
    }
  } catch {
    // Sitemap still works with static pages only
  }

  return [...staticPages, ...tokenPages];
}
