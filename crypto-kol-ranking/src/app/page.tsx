import { HomeClient } from "@/components/HomeClient";
import type { TokenCardData } from "@/components/tokens/TokenCard";

interface RankingRow {
  rank: number;
  symbol: string;
  score: number;
  score_conviction: number;
  score_momentum: number;
  mentions: number;
  unique_kols: number;
  sentiment: number;
  trend: string;
  change_24h: number;
}

async function fetchInitialTokens(): Promise<TokenCardData[]> {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY;

  if (!url || !key) return [];

  try {
    const res = await fetch(`${url}/rest/v1/rpc/get_token_ranking`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        apikey: key,
        Authorization: `Bearer ${key}`,
      },
      body: JSON.stringify({
        p_time_window: "7d",
        p_limit: 30,
        p_offset: 0,
        p_blend: 0,
      }),
      cache: "no-store",
    });

    if (!res.ok) return [];

    const rows: RankingRow[] = await res.json();
    return rows.map((t) => ({
      rank: Number(t.rank),
      symbol: t.symbol,
      score: t.score,
      mentions: t.mentions,
      uniqueKols: t.unique_kols,
      sentiment: Number(t.sentiment),
      trend: t.trend as "up" | "down" | "stable",
      change24h: Number(t.change_24h ?? 0),
    }));
  } catch {
    return [];
  }
}

export default async function HomePage() {
  const initialTokens = await fetchInitialTokens();

  return (
    <>
      {/* Static SEO content — crawlable by search engines, visually hidden */}
      <div className="sr-only" aria-hidden="false">
        <h2>Real-Time Crypto Lowcap Buy Score — Powered by KOL Sentiment</h2>
        <p>
          Cryptosensus analyzes sentiment from 50+ crypto KOL Telegram groups in
          real-time to score lowcap tokens by conviction, mentions, and community
          sentiment. Find trending memecoins and high-conviction plays before the
          crowd. Data-driven crypto trading signals updated every 30 minutes.
        </p>
        {initialTokens.length > 0 && (
          <>
            <h3>Top Ranked Tokens</h3>
            <ul>
              {initialTokens.slice(0, 15).map((token) => (
                <li key={token.symbol}>
                  #{token.rank} {token.symbol} — Score: {token.score},
                  Mentions: {token.mentions}, KOLs: {token.uniqueKols},
                  Sentiment: {(token.sentiment * 100).toFixed(0)}%
                </li>
              ))}
            </ul>
          </>
        )}
      </div>

      <HomeClient initialTokens={initialTokens} />
    </>
  );
}
