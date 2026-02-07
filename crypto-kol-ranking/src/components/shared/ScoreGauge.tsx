"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface ScoreGaugeProps {
  score: number;
  size?: "sm" | "md" | "lg";
  showLabel?: boolean;
  className?: string;
}

const sizeConfig = {
  sm: {
    container: "h-10 w-10",
    text: "text-xs",
    strokeWidth: 3,
  },
  md: {
    container: "h-14 w-14",
    text: "text-sm",
    strokeWidth: 4,
  },
  lg: {
    container: "h-20 w-20",
    text: "text-lg",
    strokeWidth: 5,
  },
};

function getScoreColor(score: number) {
  if (score >= 80) return "#22C55E"; // Green
  if (score >= 60) return "#5C7C5C"; // Sage green
  if (score >= 40) return "#C9A962"; // Gold
  return "#DC2626"; // Red
}

export function ScoreGauge({
  score,
  size = "md",
  showLabel = true,
  className,
}: ScoreGaugeProps) {
  const config = sizeConfig[size];
  const color = getScoreColor(score);

  const radius = 45;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (score / 100) * circumference;

  return (
    <div className={cn("relative", config.container, className)}>
      <svg
        viewBox="0 0 100 100"
        className="h-full w-full -rotate-90 transform"
      >
        {/* Background circle */}
        <circle
          cx="50"
          cy="50"
          r={radius}
          fill="none"
          stroke="#E8E4DE"
          strokeWidth={config.strokeWidth}
        />

        {/* Progress circle */}
        <motion.circle
          cx="50"
          cy="50"
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={config.strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset }}
          transition={{ duration: 1, ease: "easeOut" }}
        />
      </svg>

      {/* Score text */}
      {showLabel && (
        <motion.div
          className="absolute inset-0 flex items-center justify-center"
          initial={{ opacity: 0, scale: 0.5 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3, duration: 0.3 }}
        >
          <span
            className={cn(
              "font-mono font-bold tabular-nums",
              config.text
            )}
            style={{ color }}
          >
            {score}
          </span>
        </motion.div>
      )}
    </div>
  );
}
