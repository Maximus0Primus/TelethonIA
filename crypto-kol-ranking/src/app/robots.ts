import type { MetadataRoute } from "next";

const BASE_URL = process.env.NEXT_PUBLIC_SITE_URL || "https://cryptosensus.org";

export default function robots(): MetadataRoute.Robots {
  return {
    rules: [
      {
        userAgent: "*",
        allow: "/",
        disallow: ["/api/*", "/token/*"],
      },
    ],
    sitemap: `${BASE_URL}/sitemap.xml`,
  };
}
