"use client";

import { motion } from "framer-motion";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import { cn } from "@/lib/utils";

interface TrendBadgeProps {
  trend: "up" | "down" | "stable";
  value?: string;
  size?: "sm" | "md";
  className?: string;
}

const trendConfig = {
  up: {
    icon: TrendingUp,
    color: "#22C55E",
    bgColor: "rgba(34, 197, 94, 0.08)",
    borderColor: "rgba(34, 197, 94, 0.15)",
  },
  down: {
    icon: TrendingDown,
    color: "#DC2626",
    bgColor: "rgba(220, 38, 38, 0.08)",
    borderColor: "rgba(220, 38, 38, 0.15)",
  },
  stable: {
    icon: Minus,
    color: "#6B6B6B",
    bgColor: "rgba(107, 107, 107, 0.08)",
    borderColor: "rgba(107, 107, 107, 0.15)",
  },
};

const sizeConfig = {
  sm: {
    padding: "px-1.5 py-0.5",
    iconSize: 12,
    text: "text-xs",
  },
  md: {
    padding: "px-2 py-1",
    iconSize: 14,
    text: "text-sm",
  },
};

export function TrendBadge({
  trend,
  value,
  size = "sm",
  className,
}: TrendBadgeProps) {
  const { icon: Icon, color, bgColor, borderColor } = trendConfig[trend];
  const { padding, iconSize, text } = sizeConfig[size];

  return (
    <motion.div
      className={cn(
        "inline-flex items-center gap-1 rounded-md border font-mono",
        padding,
        text,
        className
      )}
      style={{
        backgroundColor: bgColor,
        borderColor: borderColor,
        color: color,
      }}
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.2 }}
    >
      <Icon size={iconSize} />
      {value && <span className="tabular-nums">{value}</span>}
    </motion.div>
  );
}
