"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { motion } from "framer-motion";
import { Header } from "@/components/layout/Header";
import { FloatingNav } from "@/components/layout/FloatingNav";
import { HeroSection } from "@/components/layout/HeroSection";
import { CyclingHeading } from "@/components/layout/CyclingHeading";
import { ViewControls } from "@/components/layout/ViewControls";
import { TokenGrid } from "@/components/tokens/TokenGrid";
import type { TokenCardData } from "@/components/tokens/TokenCard";
import type { AnimationPhase } from "@/components/tokens/TokenCard";
import { useAutoRefresh } from "@/hooks/useAutoRefresh";
import { GridDistortion } from "@/components/effects/GridDistortion";

interface ApiResponse {
  data: TokenCardData[];
  pagination: {
    limit: number;
    offset: number;
    total: number;
    hasMore: boolean;
  };
  stats: {
    totalTokens: number;
    totalMentions: number;
    avgSentiment: number;
    totalKols: number;
  };
  updated_at: string;
  error?: string;
}

interface HomeClientProps {
  initialTokens: TokenCardData[];
}

export function HomeClient({ initialTokens }: HomeClientProps) {
  const hasInitialData = initialTokens.length > 0;
  const [tokens, setTokens] = useState<TokenCardData[]>(initialTokens);
  const [prevTokens, setPrevTokens] = useState<TokenCardData[]>([]);
  const [loading, setLoading] = useState(!hasInitialData);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const [introDone, setIntroDone] = useState(false);
  const [animationPhase, setAnimationPhase] = useState<AnimationPhase>("idle");
  const introHandled = useRef(false);

  const handleIntroComplete = useCallback(() => {
    if (introHandled.current) return;
    introHandled.current = true;
    setIntroDone(true);
  }, []);

  const fetchRanking = useCallback(async (): Promise<TokenCardData[]> => {
    const response = await fetch(`/api/ranking?limit=30`);
    const data: ApiResponse = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Failed to fetch ranking");
    }

    return data.data.map((token) => ({
      rank: token.rank,
      symbol: token.symbol,
      score: token.score,
      mentions: token.mentions,
      uniqueKols: token.uniqueKols,
      sentiment: token.sentiment,
      trend: token.trend as "up" | "down" | "stable",
      change24h: token.change24h,
    }));
  }, []);

  // Initial load — skip if we have SSR data
  useEffect(() => {
    if (hasInitialData) return;

    (async () => {
      setLoading(true);
      setError(null);
      try {
        const mapped = await fetchRanking();
        setTokens(mapped);
      } catch (err) {
        const message = err instanceof Error ? err.message : "Failed to load data";
        setError(message);
      } finally {
        setLoading(false);
      }
    })();
  }, [fetchRanking, hasInitialData]);

  // Animation sequence: glitch → fetch → shuffle → idle
  const runAnimation = useCallback(async () => {
    if (animationPhase !== "idle") return;

    // Phase 1: Glitch cascade (1.2s)
    setAnimationPhase("glitching");
    await new Promise((r) => setTimeout(r, 1200));

    try {
      const newTokens = await fetchRanking();

      // Store current tokens as previous for rank delta
      setPrevTokens(tokens);

      // Phase 2: Shuffle (update data + layout animation)
      setTokens(newTokens);
      setAnimationPhase("shuffling");

      // Let shuffle animation play (0.8s)
      await new Promise((r) => setTimeout(r, 800));
    } catch {
      // If fetch fails during animation, just go idle
    }

    // Phase 3: Back to idle
    setAnimationPhase("idle");
  }, [animationPhase, fetchRanking, tokens]);

  // Auto-refresh polling
  useAutoRefresh({
    interval: 60_000,
    enabled: introDone,
    onDataChange: () => runAnimation(),
  });

  return (
    <>
      <GridDistortion />

      <Header />

      <HeroSection onIntroComplete={handleIntroComplete} />

      {introDone && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.7, ease: "easeOut" }}
        >
          <FloatingNav />
          <ViewControls
            viewMode={viewMode}
            onViewModeChange={setViewMode}
          />

          <main>
            {/* Error Banner */}
            {error && (
              <div className="fixed top-20 left-1/2 -translate-x-1/2 z-40">
                <div className="rounded-lg border border-red-500/30 bg-red-500/10 backdrop-blur-sm px-4 py-3 text-sm text-red-400">
                  {error}
                  <button
                    onClick={() => {
                      setLoading(true);
                      setError(null);
                      fetchRanking()
                        .then(setTokens)
                        .catch((e) => setError(e instanceof Error ? e.message : "Failed"))
                        .finally(() => setLoading(false));
                    }}
                    className="ml-2 underline hover:no-underline"
                  >
                    Retry
                  </button>
                </div>
              </div>
            )}

            <CyclingHeading />

            {/* Loading State */}
            {loading && tokens.length === 0 ? (
              <div className="flex items-center justify-center min-h-[50vh]">
                <div className="h-8 w-8 animate-spin rounded-full border-2 border-white border-t-transparent" />
              </div>
            ) : (
              <TokenGrid
                tokens={tokens}
                viewMode={viewMode}
                animationPhase={animationPhase}
                prevTokens={prevTokens}
              />
            )}
          </main>
        </motion.div>
      )}
    </>
  );
}
