"use client";

import type { ScoredToken } from "@/lib/rescorer";
import { TokenDetailRow } from "./TokenDetailRow";

interface RankingPreviewProps {
  tokens: ScoredToken[];
}

export function RankingPreview({ tokens }: RankingPreviewProps) {
  if (tokens.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-white/20 text-sm">
        No snapshot data available. Run the scraper first.
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <div className="min-w-[560px]">
        {/* Table header */}
        <div className="flex items-center gap-3 px-3 py-2 text-[10px] text-white/30 uppercase tracking-wider border-b border-white/10">
          <span className="w-6 text-right">#</span>
          <span className="w-24">Symbol</span>
          <span className="w-10 text-right">New</span>
          <span className="w-10 text-right">Prod</span>
          <span className="w-10 text-right">Delta</span>
          <span className="flex-1">KOLs</span>
          <span className="w-4" />
        </div>

        {/* Token rows */}
        <div className="max-h-[calc(100vh-200px)] overflow-y-auto">
          {tokens.map((t, i) => (
            <TokenDetailRow key={`${t.symbol}-${i}`} token={t} />
          ))}
        </div>
      </div>
    </div>
  );
}
