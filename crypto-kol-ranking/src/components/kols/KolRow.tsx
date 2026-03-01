"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

export interface KolRowData {
  name: string;
  tier: "S" | "A";
  conviction: number;
  score: number | null;
  uniqueTokens: number;
  labeledCalls: number;
  hitsAny: number;
  hits12h: number;
  hits24h: number;
  hits48h: number;
  hits72h: number;
  hits7d: number;
  winRateAll: number | null;
  lastActive: string | null;
  // v2: exact call-price based metrics
  totalCalls: number;
  withEntryPrice: number;
  hits2xExact: number;
  winRate2xExact: number | null;
  avgMaxReturn: number | null;
  bestReturn: number | null;
  // Paper trade (RT) metrics
  rtTrades: number;
  rtWr: number | null;
  rtWr50: number | null;
  rtPnl: number | null;
}

export type KolMode = "paper" | "kco";

interface KolRowProps {
  kol: KolRowData;
  rank: number;
  index: number;
  mode: KolMode;
}

function formatRelativeTime(iso: string | null): string {
  if (!iso) return "Never";
  const diff = Date.now() - new Date(iso).getTime();
  const hours = Math.floor(diff / 3_600_000);
  if (hours < 1) return "< 1h ago";
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days === 1) return "1d ago";
  return `${days}d ago`;
}

function scoreColor(score: number | null): string {
  if (score === null) return "text-white/30";
  if (score >= 2.0) return "text-[#22C55E]";
  if (score >= 1.0) return "text-[#22D3EE]";
  if (score >= 0.5) return "text-[#F97316]";
  return "text-[#EF4444]";
}

const PODIUM_BORDER: Record<number, string> = {
  1: "border-l-[#c9a962]",
  2: "border-l-[#a8aeb8]",
  3: "border-l-[#cd7f32]",
};

function PnlBadge({ pnl }: { pnl: number }) {
  const positive = pnl >= 0;
  return (
    <span
      className={cn(
        "shrink-0 text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded-sm",
        positive
          ? "text-[#22C55E] bg-[#22C55E]/10"
          : "text-[#EF4444] bg-[#EF4444]/10"
      )}
    >
      {positive ? "+" : ""}{pnl < 1000 && pnl > -1000 ? `$${pnl.toFixed(0)}` : `$${(pnl / 1000).toFixed(1)}k`}
    </span>
  );
}

function WinRateBar({ rate, label }: { rate: number; label?: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <div className="flex-1 h-1.5 bg-white/5 rounded-full overflow-hidden">
        <div
          className={cn(
            "h-full rounded-full transition-all",
            rate >= 0.3
              ? "bg-[#22C55E]"
              : rate >= 0.15
                ? "bg-[#F97316]"
                : "bg-[#EF4444]"
          )}
          style={{ width: `${Math.min(rate * 100, 100)}%` }}
        />
      </div>
      <span className="text-xs text-white/50 tabular-nums font-mono text-right whitespace-nowrap">
        {(rate * 100).toFixed(0)}%
        {label && <span className="text-white/25 ml-0.5">{label}</span>}
      </span>
    </div>
  );
}

export function KolRow({ kol, rank, index, mode }: KolRowProps) {
  const podiumBorder = PODIUM_BORDER[rank];
  const isPaper = mode === "paper";

  // Paper mode: RT win rate. KCO mode: v2 exact preferred, v1 fallback
  const winRate = isPaper ? kol.rtWr : (kol.winRate2xExact ?? kol.winRateAll);
  const callCount = isPaper
    ? kol.rtTrades
    : (kol.winRate2xExact !== null ? kol.totalCalls : kol.labeledCalls);

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.02 }}
      className={cn(
        "grid grid-cols-[2rem_1fr_3.5rem_4.5rem_3rem] md:grid-cols-[3rem_1fr_5rem_6rem_5rem_6rem] items-center gap-1.5 md:gap-4 px-2 md:px-4 py-2 md:py-3 border-b border-white/5 hover:bg-white/[0.02] transition-colors",
        podiumBorder && `border-l-2 ${podiumBorder}`
      )}
    >
      {/* Rank */}
      <span className="text-white/40 text-sm tabular-nums text-center font-mono">
        {rank}
      </span>

      {/* Name + Tier Badge + PnL Badge (paper mode) */}
      <div className="flex items-center gap-2 min-w-0">
        <span
          className={cn(
            "shrink-0 text-[10px] font-bold px-1.5 py-0.5 rounded-sm border",
            kol.tier === "S"
              ? "text-[#c9a962] bg-[#c9a962]/10 border-[#c9a962]/20"
              : "text-[#a8aeb8] bg-[#a8aeb8]/10 border-[#a8aeb8]/20"
          )}
        >
          {kol.tier}
        </span>
        <span className="text-sm font-medium text-white truncate">
          {kol.name}
        </span>
        {isPaper && kol.rtPnl !== null && kol.rtTrades > 0 && (
          <PnlBadge pnl={kol.rtPnl} />
        )}
      </div>

      {/* Score */}
      <span
        className={cn(
          "text-sm font-mono tabular-nums text-right",
          scoreColor(kol.score)
        )}
      >
        {kol.score !== null ? kol.score.toFixed(2) : "\u2014"}
      </span>

      {/* Win Rate */}
      <div>
        {isPaper && kol.rtTrades === 0 ? (
          <span className="text-xs text-white/20 block text-right">No RT</span>
        ) : winRate !== null ? (
          <div className="space-y-0.5">
            <WinRateBar rate={winRate} />
            {isPaper && kol.rtWr50 !== null && (
              <div className="text-[10px] text-white/30 font-mono text-right">
                +50%: {(kol.rtWr50 * 100).toFixed(0)}%
              </div>
            )}
          </div>
        ) : (
          <span className="text-xs text-white/20 block text-right">
            {callCount > 0
              ? `${callCount} call${callCount > 1 ? "s" : ""}`
              : "\u2014"}
          </span>
        )}
      </div>

      {/* Calls */}
      <span className="text-sm text-white/50 tabular-nums font-mono text-right">
        {callCount}
      </span>

      {/* Last Active — hidden on mobile */}
      <span className="hidden md:block text-xs text-white/30 text-right">
        {formatRelativeTime(kol.lastActive)}
      </span>
    </motion.div>
  );
}
