"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface SentimentBarProps {
  /** Sentiment value from -1 (bearish) to 1 (bullish) */
  value: number;
  showLabel?: boolean;
  size?: "sm" | "md";
  className?: string;
}

const sizeConfig = {
  sm: {
    height: "h-1.5",
    width: "w-20",
  },
  md: {
    height: "h-2",
    width: "w-28",
  },
};

export function SentimentBar({
  value,
  showLabel = false,
  size = "sm",
  className,
}: SentimentBarProps) {
  const config = sizeConfig[size];

  // Normalize value from -1..1 to 0..100
  const percentage = ((value + 1) / 2) * 100;

  const getColor = () => {
    if (percentage >= 60) return "#22C55E"; // Green
    if (percentage >= 40) return "#5C7C5C"; // Sage green
    return "#DC2626"; // Red
  };

  const color = getColor();

  return (
    <div className={cn("flex items-center gap-2", className)}>
      <div
        className={cn(
          "relative overflow-hidden rounded-full bg-muted",
          config.height,
          config.width
        )}
      >
        <motion.div
          className={cn("absolute inset-y-0 left-0 rounded-full", config.height)}
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          style={{ background: color }}
        />
      </div>
      {showLabel && (
        <span
          className="text-xs font-mono tabular-nums"
          style={{ color }}
        >
          {value > 0 ? "+" : ""}
          {(value * 100).toFixed(0)}%
        </span>
      )}
    </div>
  );
}
