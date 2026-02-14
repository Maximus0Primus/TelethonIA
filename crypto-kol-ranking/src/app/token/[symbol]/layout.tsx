import type { Metadata } from "next";

interface TokenLayoutProps {
  params: Promise<{ symbol: string }>;
  children: React.ReactNode;
}

export async function generateMetadata({
  params,
}: TokenLayoutProps): Promise<Metadata> {
  const { symbol } = await params;
  const normalized = symbol.toUpperCase().replace("$", "");

  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY;

  let tokenData: {
    symbol: string;
    score: number;
    mentions: number;
    unique_kols: number;
    sentiment: number;
    trend: string;
  } | null = null;

  if (url && key) {
    try {
      const res = await fetch(
        `${url}/rest/v1/tokens?select=symbol,score,mentions,unique_kols,sentiment,trend&symbol=eq.$${normalized}&time_window=eq.7d&limit=1`,
        {
          headers: {
            apikey: key,
            Authorization: `Bearer ${key}`,
          },
          cache: "no-store",
        }
      );
      if (res.ok) {
        const rows = await res.json();
        if (rows.length > 0) tokenData = rows[0];
      }
    } catch {
      // Fall through to default metadata
    }
  }

  const canonicalSymbol = normalized.toLowerCase();

  if (!tokenData) {
    return {
      title: `$${normalized} Analysis`,
      description: `View real-time KOL onchain analysis and conviction score for $${normalized} on Cryptosensus.`,
      alternates: { canonical: `/token/${canonicalSymbol}` },
    };
  }

  const sentimentPct = (tokenData.sentiment * 100).toFixed(0);
  const trendLabel =
    tokenData.trend === "up"
      ? "Rising"
      : tokenData.trend === "down"
        ? "Falling"
        : "Stable";

  const title = `$${tokenData.symbol} — Score ${tokenData.score} | ${trendLabel}`;
  const description = `$${tokenData.symbol} has a Cryptosensus score of ${tokenData.score} with ${tokenData.mentions} mentions from ${tokenData.unique_kols} KOLs. Sentiment: ${sentimentPct}%. Trend: ${trendLabel}.`;

  return {
    title,
    description,
    alternates: { canonical: `/token/${canonicalSymbol}` },
    openGraph: {
      title: `$${tokenData.symbol} — Score ${tokenData.score} | Cryptosensus`,
      description,
      type: "website",
      url: `/token/${canonicalSymbol}`,
    },
    twitter: {
      card: "summary_large_image",
      title: `$${tokenData.symbol} Score: ${tokenData.score} ${trendLabel}`,
      description,
      creator: "@Maximus0Primus",
    },
  };
}

export default function TokenLayout({ children }: TokenLayoutProps) {
  return children;
}
