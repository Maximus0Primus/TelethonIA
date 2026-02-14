"use client";

import { cn } from "@/lib/utils";
import type { FeatureImpact } from "@/lib/backtester";

interface FeatureImportanceProps {
  features: FeatureImpact[];
}

export function FeatureImportance({ features }: FeatureImportanceProps) {
  if (features.length === 0) return null;

  const maxImpact = Math.max(...features.map((f) => Math.abs(f.impact_on_hit_rate)), 1);

  return (
    <div className="space-y-2">
      <h4 className="text-[10px] text-white/30 uppercase tracking-wider">
        Feature Impact on Hit Rate
      </h4>

      {features.map((f, i) => {
        const isPositive = f.impact_on_hit_rate >= 0;
        const barWidth = (Math.abs(f.impact_on_hit_rate) / maxImpact) * 100;
        const changed = f.optimal_value !== f.current_value;

        return (
          <div key={f.feature} className="flex items-center gap-2 text-xs">
            <span className="text-white/20 font-mono w-4 text-right shrink-0">
              {i + 1}
            </span>
            <span className="text-white/60 w-32 truncate shrink-0">{f.label}</span>

            {/* Impact bar */}
            <div className="flex-1 h-3 bg-white/5 rounded-sm relative overflow-hidden">
              <div
                className={cn(
                  "h-full rounded-sm transition-all",
                  isPositive ? "bg-green-500/40" : "bg-red-500/30"
                )}
                style={{ width: `${Math.max(barWidth, 2)}%` }}
              />
            </div>

            {/* Impact value */}
            <span
              className={cn(
                "font-mono w-14 text-right shrink-0",
                isPositive ? "text-green-400" : "text-red-400/70"
              )}
            >
              {isPositive ? "+" : ""}{f.impact_on_hit_rate}%
            </span>

            {/* Optimal value */}
            <span
              className={cn(
                "font-mono w-12 text-right shrink-0 text-[10px]",
                changed ? "text-yellow-400" : "text-white/20"
              )}
            >
              {f.optimal_value}
            </span>
          </div>
        );
      })}
    </div>
  );
}
