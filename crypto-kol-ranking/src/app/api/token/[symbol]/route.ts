import { NextRequest, NextResponse } from "next/server";
import { readFileSync, existsSync } from "fs";
import { join } from "path";

interface TokenData {
  rank: number;
  symbol: string;
  score: number;
  mentions: number;
  uniqueKols: number;
  sentiment: number;
  trend: string;
  change24h: number;
  topKols: string[];
}

interface RankingData {
  updated_at: string;
  stats: {
    totalTokens: number;
    totalMentions: number;
    avgSentiment: number;
    totalKols: number;
  };
  tokens: Record<string, TokenData[]>;
}

// Groups with their conviction scores (matching Python pipeline)
const GROUPS_CONVICTION: Record<string, number> = {
  overdose_gems_calls: 10,
  cryptorugmuncher: 10,
  thetonymoontana: 10,
  marcellcooks: 9,
  Carnagecalls: 9,
  PoseidonTAA: 9,
  MarkGems: 9,
  slingdeez: 8,
  ghastlygems: 8,
  archercallz: 8,
  LevisAlpha: 8,
  darkocalls: 8,
  kweensjournal: 8,
  ArcaneGems: 8,
  dylansdegens: 8,
  ALSTEIN_GEMCLUB: 8,
  jsdao: 8,
  MaybachCalls: 8,
  inside_calls: 8,
  BossmanCallsOfficial: 8,
  bounty_journal: 8,
  StereoCalls: 8,
  PowsGemCalls: 8,
  CatfishcallsbyPoe: 8,
  spidersjournal: 8,
  cryptolyxecalls: 8,
  izzycooks: 8,
  wulfcryptocalls: 8,
  OnyxxGems: 8,
  eunicalls: 8,
  TheCabalCalls: 8,
  sugarydick: 8,
  certifiedprintor: 8,
  LittleMustachoCalls: 8,
};

function loadRankingData(): RankingData | null {
  const dataPath = join(process.cwd(), "public", "data", "ranking_data.json");

  if (!existsSync(dataPath)) {
    return null;
  }

  try {
    const rawData = readFileSync(dataPath, "utf-8");
    return JSON.parse(rawData) as RankingData;
  } catch (error) {
    console.error("Error reading ranking data:", error);
    return null;
  }
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ symbol: string }> }
) {
  const { symbol } = await params;
  const searchParams = request.nextUrl.searchParams;
  const timeWindow = searchParams.get("window") || "24h";

  // Normalize symbol (add $ prefix if not present)
  const normalizedSymbol = symbol.startsWith("$") ? symbol.toUpperCase() : `$${symbol.toUpperCase()}`;

  try {
    // Load data from JSON file
    const data = loadRankingData();

    if (!data) {
      return NextResponse.json(
        { error: "Ranking data not available. Run the pipeline script first." },
        { status: 503 }
      );
    }

    const tokens = data.tokens[timeWindow] || [];
    const token = tokens.find((t) => t.symbol.toUpperCase() === normalizedSymbol);

    if (!token) {
      return NextResponse.json(
        { error: "Token not found" },
        { status: 404 }
      );
    }

    // Build KOL breakdown with conviction scores
    const topKols = token.topKols.map((name) => ({
      name,
      mentions: 1, // We don't have per-KOL mention counts in the simplified data
      conviction: GROUPS_CONVICTION[name] || 7,
    }));

    return NextResponse.json({
      token: {
        symbol: token.symbol,
        score: token.score,
        mentions: token.mentions,
        unique_kols: token.uniqueKols,
        sentiment: token.sentiment,
        trend: token.trend,
        change_24h: token.change24h,
        rank: token.rank,
      },
      topKols,
      recentMentions: [], // Not available in simplified data
      timeWindow,
      updated_at: data.updated_at,
    });
  } catch (error) {
    console.error("API error:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
