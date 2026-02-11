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

  return staticPages;
}
