"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { motion } from "framer-motion";
import Link from "next/link";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import { cn } from "@/lib/utils";

export type AnimationPhase = "idle" | "glitching" | "shuffling";

export interface TokenCardData {
  rank: number;
  symbol: string;
  score: number;
  mentions: number;
  uniqueKols: number;
  sentiment: number;
  trend: "up" | "down" | "stable";
  change24h: number;
}

interface TokenCardProps {
  token: TokenCardData;
  index: number;
  animationPhase?: AnimationPhase;
  previousRank?: number | null;
  isNew?: boolean;
}

const PODIUM_STYLES: Record<number, {
  border: string;
  shadow: string;
  hover: string;
  bg: string;
  score: string;
  badge: string;
  badgeBg: string;
}> = {
  1: {
    border: "border-[#c9a962]/30",
    shadow: "shadow-[0_0_30px_rgba(201,169,98,0.10)]",
    hover: "border-[#c9a962]/25",
    bg: "bg-gradient-to-b from-[#c9a962]/[0.06] to-transparent",
    score: "text-[#e2c878]",
    badge: "text-[#c9a962]",
    badgeBg: "bg-[#c9a962]/10 border-[#c9a962]/20",
  },
  2: {
    border: "border-[#a8aeb8]/25",
    shadow: "shadow-[0_0_30px_rgba(168,174,184,0.08)]",
    hover: "border-[#a8aeb8]/20",
    bg: "bg-gradient-to-b from-[#a8aeb8]/[0.05] to-transparent",
    score: "text-[#bcc2cc]",
    badge: "text-[#a8aeb8]",
    badgeBg: "bg-[#a8aeb8]/10 border-[#a8aeb8]/15",
  },
  3: {
    border: "border-[#a17a56]/25",
    shadow: "shadow-[0_0_30px_rgba(161,122,86,0.08)]",
    hover: "border-[#a17a56]/20",
    bg: "bg-gradient-to-b from-[#a17a56]/[0.05] to-transparent",
    score: "text-[#c49b74]",
    badge: "text-[#a17a56]",
    badgeBg: "bg-[#a17a56]/10 border-[#a17a56]/15",
  },
};

// Characters used for the glitch scramble effect
const GLITCH_CHARS = "!@#$%&*?/<>{}[]~";

function useScrambleText(text: string, active: boolean, duration = 300) {
  const [display, setDisplay] = useState(text);
  const frameRef = useRef<number | null>(null);

  useEffect(() => {
    if (!active) {
      setDisplay(text);
      return;
    }

    const startTime = performance.now();
    const scramble = (now: number) => {
      const elapsed = now - startTime;
      if (elapsed >= duration) {
        setDisplay(text);
        return;
      }
      const scrambled = text
        .split("")
        .map((ch) =>
          ch === "$" || ch === " "
            ? ch
            : GLITCH_CHARS[Math.floor(Math.random() * GLITCH_CHARS.length)]
        )
        .join("");
      setDisplay(scrambled);
      frameRef.current = requestAnimationFrame(scramble);
    };
    frameRef.current = requestAnimationFrame(scramble);

    return () => {
      if (frameRef.current) cancelAnimationFrame(frameRef.current);
    };
  }, [text, active, duration]);

  return display;
}

function useScrambleNumber(value: number, active: boolean, frames = 8, interval = 40) {
  const [display, setDisplay] = useState(value);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!active) {
      setDisplay(value);
      return;
    }

    let count = 0;
    const tick = () => {
      count++;
      if (count >= frames) {
        setDisplay(value);
        return;
      }
      setDisplay(Math.floor(Math.random() * 100));
      timerRef.current = setTimeout(tick, interval);
    };
    timerRef.current = setTimeout(tick, interval);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [value, active, frames, interval]);

  return display;
}

export function TokenCard({
  token,
  index,
  animationPhase = "idle",
  previousRank = null,
  isNew = false,
}: TokenCardProps) {
  const TrendIcon = token.trend === "up" ? TrendingUp : token.trend === "down" ? TrendingDown : Minus;
  const podium = PODIUM_STYLES[token.rank];

  // Stagger delay for glitch cascade: 40ms per card
  const glitchDelay = index * 40;
  const [isGlitching, setIsGlitching] = useState(false);
  const [scoreChanged, setScoreChanged] = useState(false);
  const prevScoreRef = useRef(token.score);

  // Trigger per-card glitch with stagger
  useEffect(() => {
    if (animationPhase !== "glitching") {
      setIsGlitching(false);
      return;
    }

    const startTimer = setTimeout(() => setIsGlitching(true), glitchDelay);
    const endTimer = setTimeout(() => setIsGlitching(false), glitchDelay + 300);

    return () => {
      clearTimeout(startTimer);
      clearTimeout(endTimer);
    };
  }, [animationPhase, glitchDelay]);

  // Detect score change for scramble
  useEffect(() => {
    if (prevScoreRef.current !== token.score) {
      setScoreChanged(true);
      prevScoreRef.current = token.score;
      const timer = setTimeout(() => setScoreChanged(false), 360);
      return () => clearTimeout(timer);
    }
  }, [token.score]);

  const scrambledSymbol = useScrambleText(token.symbol, isGlitching);
  const scrambledScore = useScrambleNumber(token.score, isGlitching || scoreChanged);

  // Rank delta calculation
  const rankDelta = previousRank !== null ? previousRank - token.rank : 0;
  const showRankDelta = animationPhase === "shuffling" && rankDelta !== 0;

  return (
    <motion.div
      layout="position"
      layoutId={token.symbol}
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{
        layout: { type: "spring", stiffness: 300, damping: 30 },
        opacity: { delay: index * 0.03, duration: 0.4 },
        y: { delay: index * 0.03, duration: 0.4, ease: "easeOut" },
      }}
    >
      <Link href={`/token/${token.symbol.replace("$", "")}`}>
        <div
          className={cn(
            "token-card group relative flex flex-col justify-between",
            "h-[200px] sm:h-[220px] p-5",
            "rounded-2xl border",
            "cursor-pointer",
            podium
              ? [podium.border, podium.shadow, podium.bg]
              : "border-card-border bg-card",
            isGlitching && "glitch-active"
          )}
        >
          {/* Scanlines overlay during glitch */}
          {isGlitching && (
            <div className="absolute inset-0 rounded-2xl pointer-events-none z-10 overflow-hidden">
              <div className="absolute inset-0 scanlines opacity-30" />
            </div>
          )}

          {/* Rank badge for top 3 */}
          {podium && (
            <span className={cn(
              "absolute top-3 right-3 text-[10px] font-semibold tracking-wider uppercase",
              "px-2 py-0.5 rounded-full border",
              podium.badgeBg, podium.badge
            )}>
              #{token.rank}
            </span>
          )}

          {/* Rank delta badge during shuffle */}
          {showRankDelta && (
            <motion.span
              initial={{ opacity: 0, scale: 0.5 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0 }}
              className={cn(
                "absolute top-3 left-3 text-[10px] font-bold px-1.5 py-0.5 rounded-md z-20",
                rankDelta > 0
                  ? "bg-bullish/20 text-bullish"
                  : "bg-bearish/20 text-bearish"
              )}
            >
              {rankDelta > 0 ? `+${rankDelta}` : rankDelta}
            </motion.span>
          )}

          {/* NEW badge for newly appeared tokens */}
          {isNew && animationPhase === "shuffling" && (
            <motion.span
              initial={{ opacity: 0, scale: 0.5 }}
              animate={{ opacity: 1, scale: 1 }}
              className="absolute top-3 left-3 text-[10px] font-bold px-1.5 py-0.5 rounded-md z-20 bg-[#00ff41]/20 text-[#00ff41]"
            >
              NEW
            </motion.span>
          )}

          {/* Symbol */}
          <div className="overflow-hidden">
            <span
              className={cn(
                "font-bold block truncate",
                token.symbol.length > 8 ? "text-lg sm:text-xl" : "text-2xl sm:text-3xl",
                isGlitching
                  ? "text-[#00ff41] glitch-text-shadow"
                  : "text-white"
              )}
            >
              {scrambledSymbol}
            </span>
          </div>

          {/* Score - Large centered */}
          <div className="flex-1 flex items-center justify-center">
            <span className={cn(
              "text-5xl sm:text-6xl font-bold tabular-nums",
              podium ? podium.score : "text-white"
            )}>
              {scrambledScore}
            </span>
          </div>

          {/* Bottom: Trend indicator */}
          <div className="flex items-center justify-between">
            <div
              className={cn(
                "flex items-center gap-1.5 text-sm font-medium",
                token.change24h > 0
                  ? "text-bullish"
                  : token.change24h < 0
                  ? "text-bearish"
                  : "text-muted-foreground"
              )}
            >
              <TrendIcon className="h-4 w-4" />
              <span className="tabular-nums">
                {token.change24h > 0 ? "+" : ""}
                {token.change24h.toFixed(1)}%
              </span>
            </div>

            {/* KOL count */}
            <span className="text-xs text-muted-foreground">
              {token.uniqueKols} KOLs
            </span>
          </div>

          {/* Subtle hover border glow */}
          <div className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
            <div className={cn(
              "absolute inset-0 rounded-2xl border",
              podium ? podium.hover : "border-white/10"
            )} />
          </div>
        </div>
      </Link>
    </motion.div>
  );
}
