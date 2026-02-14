"use client";

import { useRef, useMemo } from "react";
import { motion, useScroll, useTransform, LayoutGroup } from "framer-motion";
import { TokenCard, type TokenCardData, type AnimationPhase } from "./TokenCard";

interface TokenGridProps {
  tokens: TokenCardData[];
  viewMode?: "grid" | "list";
  animationPhase?: AnimationPhase;
  prevTokens?: TokenCardData[];
}

export function TokenGrid({
  tokens,
  viewMode = "grid",
  animationPhase = "idle",
  prevTokens = [],
}: TokenGridProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start start", "end start"],
  });

  // Parallax effect - subtle rotation change on scroll
  const rotateX = useTransform(scrollYProgress, [0, 1], [12, 8]);

  // Build a map of previous ranks for delta badges
  const prevRankMap = useMemo(() => {
    const map = new Map<string, number>();
    for (const t of prevTokens) {
      map.set(t.symbol, t.rank);
    }
    return map;
  }, [prevTokens]);

  // Set of previously known symbols to detect new tokens
  const prevSymbolSet = useMemo(() => {
    return new Set(prevTokens.map((t) => t.symbol));
  }, [prevTokens]);

  if (viewMode === "list") {
    return (
      <div className="pt-24 pb-32 px-4 sm:px-6 max-w-4xl mx-auto space-y-3">
        {tokens.map((token, index) => (
          <TokenListItem key={token.symbol} token={token} index={index} />
        ))}
      </div>
    );
  }

  return (
    <div ref={containerRef} className="perspective-container pt-20 pb-32">
      <LayoutGroup>
        <motion.div
          style={{ rotateX }}
          className="grid-3d grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-2 sm:gap-4 px-3 sm:px-4 md:px-6"
        >
          {tokens.map((token, index) => (
            <TokenCard
              key={token.symbol}
              token={token}
              index={index}
              animationPhase={animationPhase}
              previousRank={prevRankMap.get(token.symbol) ?? null}
              isNew={prevTokens.length > 0 && !prevSymbolSet.has(token.symbol)}
            />
          ))}
        </motion.div>
      </LayoutGroup>
    </div>
  );
}

// Simple list item for list view
function TokenListItem({ token, index }: { token: TokenCardData; index: number }) {
  return (
    <motion.a
      href={`/token/${token.symbol.replace("$", "")}`}
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.02, duration: 0.3 }}
      className="flex items-center justify-between p-4 rounded-xl border border-card-border bg-card hover:border-white/20 transition-colors"
    >
      <div className="flex items-center gap-4">
        <span className="text-muted-foreground text-sm font-mono w-6">
          #{token.rank}
        </span>
        <span className="text-lg font-bold">{token.symbol}</span>
      </div>
      <div className="flex items-center gap-6">
        <span className="text-sm text-muted-foreground">
          {token.uniqueKols} KOLs
        </span>
        <span className="text-2xl font-bold tabular-nums w-16 text-right">
          {token.score}
        </span>
      </div>
    </motion.a>
  );
}
