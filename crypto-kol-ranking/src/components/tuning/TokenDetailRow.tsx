"use client";

import { useState } from "react";
import { cn } from "@/lib/utils";
import type { ScoredToken } from "@/lib/rescorer";

interface TokenDetailRowProps {
  token: ScoredToken;
}

export function TokenDetailRow({ token }: TokenDetailRowProps) {
  const [expanded, setExpanded] = useState(false);

  const delta = token.delta;
  const deltaColor = delta > 0 ? "text-green-400" : delta < 0 ? "text-red-400" : "text-white/40";

  // Parse top_kols JSON
  let topKols: string[] = [];
  if (token.snapshot.top_kols) {
    try {
      const parsed = JSON.parse(token.snapshot.top_kols);
      topKols = Array.isArray(parsed) ? parsed.slice(0, 3) : [];
    } catch {
      topKols = [];
    }
  }

  return (
    <div className="border-b border-white/5">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-3 px-3 py-2.5 hover:bg-white/[0.02] transition-colors text-left"
      >
        {/* Rank */}
        <span className="text-xs font-mono text-white/30 w-6 text-right shrink-0">
          {token.rank}
        </span>

        {/* Symbol */}
        <span className="text-sm font-semibold text-white w-24 truncate">
          ${token.symbol}
        </span>

        {/* New Score */}
        <span className="text-sm font-mono text-white w-10 text-right">
          {token.newScore}
        </span>

        {/* Prod Score */}
        <span className="text-xs font-mono text-white/40 w-10 text-right">
          {token.prodScore}
        </span>

        {/* Delta */}
        <span className={cn("text-xs font-mono w-10 text-right", deltaColor)}>
          {delta > 0 ? "+" : ""}{delta}
        </span>

        {/* KOL badges */}
        <div className="flex-1 flex gap-1 overflow-hidden">
          {topKols.map((kol) => (
            <span
              key={kol}
              className="text-[10px] px-1.5 py-0.5 rounded bg-white/5 text-white/40 truncate max-w-[80px]"
            >
              {kol}
            </span>
          ))}
        </div>

        {/* Expand arrow */}
        <span className="text-white/20 text-xs shrink-0">
          {expanded ? "\u25B2" : "\u25BC"}
        </span>
      </button>

      {expanded && (
        <div className="px-3 pb-3 pt-1 ml-9 space-y-3">
          {/* Component breakdown */}
          <div>
            <div className="text-[10px] text-white/30 uppercase tracking-wider mb-1.5">
              Components
            </div>
            <div className="grid grid-cols-5 gap-2">
              {(["consensus", "sentiment", "conviction", "breadth", "price_action"] as const).map(
                (comp) => (
                  <div key={comp} className="text-center">
                    <div className="text-[10px] text-white/30 mb-0.5">
                      {comp.slice(0, 4).toUpperCase()}
                    </div>
                    <div className="text-xs font-mono text-white/80">
                      {(token.components[comp] * 100).toFixed(0)}%
                    </div>
                    <div className="h-1 bg-white/5 rounded-full mt-0.5">
                      <div
                        className="h-full bg-white/30 rounded-full"
                        style={{ width: `${token.components[comp] * 100}%` }}
                      />
                    </div>
                  </div>
                )
              )}
            </div>
            <div className="text-[10px] text-white/20 mt-1 font-mono text-right">
              weighted sum: {token.components.weighted_sum}
            </div>
          </div>

          {/* Multiplier breakdown */}
          <div>
            <div className="text-[10px] text-white/30 uppercase tracking-wider mb-1.5">
              Multipliers
            </div>
            <div className="flex flex-wrap gap-x-3 gap-y-1">
              {Object.entries(token.multipliers)
                .filter(([k]) => k !== "combined")
                .map(([key, val]) => {
                  const isNeutral = val === 1.0;
                  const isBuff = val > 1.0;
                  return (
                    <div key={key} className="flex items-center gap-1 text-[10px]">
                      <span className="text-white/30">{key.replace(/_/g, " ")}:</span>
                      <span
                        className={cn(
                          "font-mono",
                          isNeutral ? "text-white/30" : isBuff ? "text-green-400" : "text-red-400"
                        )}
                      >
                        {val.toFixed(3)}
                      </span>
                    </div>
                  );
                })}
            </div>
            <div className="text-[10px] text-white/20 mt-1 font-mono text-right">
              combined: {token.multipliers.combined}x
            </div>
          </div>

          {/* Context data */}
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-[10px] text-white/30">
            {token.snapshot.volume_24h != null && (
              <span>vol24h: ${(token.snapshot.volume_24h / 1000).toFixed(0)}K</span>
            )}
            {token.snapshot.liquidity_usd != null && (
              <span>liq: ${(token.snapshot.liquidity_usd / 1000).toFixed(0)}K</span>
            )}
            {token.snapshot.market_cap != null && (
              <span>mcap: ${(token.snapshot.market_cap / 1000).toFixed(0)}K</span>
            )}
            {token.snapshot.price_change_24h != null && (
              <span>pc24h: {token.snapshot.price_change_24h.toFixed(0)}%</span>
            )}
            {token.snapshot.mentions != null && (
              <span>mentions: {token.snapshot.mentions}</span>
            )}
            {token.snapshot.unique_kols != null && (
              <span>KOLs: {token.snapshot.unique_kols}</span>
            )}
            {token.snapshot.s_tier_count != null && token.snapshot.s_tier_count > 0 && (
              <span className="text-yellow-400/60">S-tier: {token.snapshot.s_tier_count}</span>
            )}
            {token.snapshot.freshest_mention_hours != null && (
              <span>fresh: {token.snapshot.freshest_mention_hours.toFixed(1)}h</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
