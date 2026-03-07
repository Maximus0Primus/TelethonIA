"use client";

import { useState, useMemo, useEffect, useCallback } from "react";
import { motion } from "framer-motion";
import { ChevronUp, ChevronDown, FlaskConical, BarChart3, TestTube2, Calendar } from "lucide-react";
import { cn } from "@/lib/utils";
import { KolRow } from "./KolRow";
import type { KolRowData, KolMode, WrThreshold } from "./KolRow";

const TIME_RANGES = [
  { label: "1d", days: 1 },
  { label: "3d", days: 3 },
  { label: "7d", days: 7 },
  { label: "14d", days: 14 },
  { label: "30d", days: 30 },
  { label: "All", days: 0 },
] as const;

type SortKey =
  | "rank"
  | "name"
  | "tier"
  | "score"
  | "winRate"
  | "calls"
  | "lastActive";

const COLUMNS: { key: SortKey; label: string; labelShort?: string; className: string }[] = [
  { key: "rank", label: "#", className: "text-center w-[2.5rem] md:w-[3rem]" },
  { key: "name", label: "KOL", className: "text-left flex-1" },
  { key: "score", label: "Score", className: "text-right w-[4rem] md:w-[5rem]" },
  { key: "winRate", label: "Win Rate", labelShort: "Win%", className: "text-right w-[5rem] md:w-[6rem]" },
  { key: "calls", label: "Calls", className: "text-right w-[4rem] md:w-[5rem]" },
  { key: "lastActive", label: "Last Active", labelShort: "Active", className: "text-right w-[5rem] md:w-[6rem]" },
];

function getWr(k: KolRowData, mode: KolMode, wrThreshold: WrThreshold): number {
  if (mode === "paper") {
    if (wrThreshold === "2x") return k.rtWr2x ?? -1;
    if (wrThreshold === "50") return k.rtWr50 ?? -1;
    return k.rtWr ?? -1;
  }
  return (wrThreshold === "2x" ? (k.winRate2xExact ?? k.winRateAll) : k.winRate1_5xExact) ?? -1;
}

function getCalls(k: KolRowData, mode: KolMode): number {
  if (mode === "paper") return k.rtTrades;
  return k.totalCalls || k.labeledCalls;
}

function compare(a: KolRowData, b: KolRowData, key: SortKey, dir: "asc" | "desc", mode: KolMode, wrThreshold: WrThreshold): number {
  const m = dir === "asc" ? 1 : -1;
  switch (key) {
    case "rank":
      return m * ((b.score ?? -1) - (a.score ?? -1));
    case "name":
      return m * a.name.localeCompare(b.name);
    case "tier":
      return m * (a.tier === b.tier ? 0 : a.tier === "S" ? -1 : 1);
    case "score":
      return m * ((a.score ?? -1) - (b.score ?? -1));
    case "winRate":
      return m * (getWr(a, mode, wrThreshold) - getWr(b, mode, wrThreshold));
    case "calls":
      return m * (getCalls(a, mode) - getCalls(b, mode));
    case "lastActive": {
      const ta = a.lastActive ? new Date(a.lastActive).getTime() : 0;
      const tb = b.lastActive ? new Date(b.lastActive).getTime() : 0;
      return m * (ta - tb);
    }
    default:
      return 0;
  }
}

export function KolLeaderboard() {
  const [kols, setKols] = useState<KolRowData[]>([]);
  const [loading, setLoading] = useState(true);
  const [caOnly, setCaOnly] = useState(false);
  const [days, setDays] = useState(7);
  const [mode, setMode] = useState<KolMode>("paper");
  const [wrThreshold, setWrThreshold] = useState<WrThreshold>("wr");
  const [sortKey, setSortKey] = useState<SortKey>("score");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  const fetchData = useCallback(async (ca: boolean, d: number) => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (ca) params.set("ca_only", "true");
      params.set("days", String(d));
      const res = await fetch(`/api/kols?${params}`);
      if (!res.ok) throw new Error("Failed to fetch");
      const json = await res.json();
      setKols(json.data ?? []);
    } catch {
      // Keep existing data on error
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData(caOnly, days);
  }, [caOnly, days, fetchData]);

  const sorted = useMemo(() => {
    const arr = [...kols];
    arr.sort((a, b) => compare(a, b, sortKey, sortDir, mode, wrThreshold));
    return arr;
  }, [kols, sortKey, sortDir, mode, wrThreshold]);

  const handleSort = (key: SortKey) => {
    if (key === sortKey) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir(key === "name" ? "asc" : "desc");
    }
  };

  const sTierCount = kols.filter((k) => k.tier === "S").length;
  const scoredKols = kols.filter((k) => k.score !== null);
  const avgScore =
    scoredKols.length > 0
      ? scoredKols.reduce((s, k) => s + (k.score ?? 0), 0) / scoredKols.length
      : 0;

  // Mode-aware win rate stats
  const getDisplayWr = (k: KolRowData) => getWr(k, mode, wrThreshold);
  const withWinRate = mode === "paper"
    ? kols.filter((k) => k.rtTrades > 0 && getDisplayWr(k) >= 0)
    : kols.filter((k) => getDisplayWr(k) >= 0);
  const avgWinRate =
    withWinRate.length > 0
      ? withWinRate.reduce((s, k) => s + Math.max(0, getDisplayWr(k)), 0) / withWinRate.length
      : 0;
  const totalRtPnl = mode === "paper"
    ? kols.reduce((s, k) => s + (k.rtPnl ?? 0), 0)
    : null;

  return (
    <div className="w-full max-w-4xl mx-auto">
      {/* Summary Stats */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6"
      >
        {[
          { label: "Total KOLs", value: kols.length || 59 },
          {
            label: "S-Tier",
            value: sTierCount || 13,
            accent: "text-[#c9a962]",
          },
          mode === "paper" && totalRtPnl !== null
            ? {
                label: `RT PnL (${days === 0 ? "All" : days + "d"})`,
                value: `${totalRtPnl >= 0 ? "+" : ""}$${totalRtPnl.toFixed(0)}`,
                accent: totalRtPnl >= 0 ? "text-[#22C55E]" : "text-[#EF4444]",
              }
            : {
                label: "Avg Score",
                value: avgScore > 0 ? avgScore.toFixed(2) : "--",
                accent: "text-[#22D3EE]",
              },
          {
            label: wrThreshold === "2x" ? "Avg WR 2x"
              : wrThreshold === "50" ? "Avg WR +50%"
              : "Avg WR >0%",
            value:
              withWinRate.length > 0
                ? `${(avgWinRate * 100).toFixed(0)}%`
                : "N/A",
            accent: avgWinRate >= 0.2 ? "text-[#22C55E]" : "text-[#F97316]",
          },
        ].map((stat) => (
          <div
            key={stat.label}
            className="surface-card rounded-lg px-4 py-3 text-center"
          >
            <div className="text-xs text-white/40 uppercase tracking-wider mb-1">
              {stat.label}
            </div>
            <div
              className={cn(
                "text-lg font-mono font-semibold tabular-nums",
                stat.accent ?? "text-white"
              )}
            >
              {stat.value}
            </div>
          </div>
        ))}
      </motion.div>

      {/* Toggles: Time Range + Mode + Thresholds + CA-Only */}
      <div className="flex flex-wrap items-center justify-end gap-2 mb-3">
        {/* Time range selector */}
        <div className="flex items-center gap-1.5">
          <Calendar className="w-3.5 h-3.5 text-white/30" />
          <div className="flex rounded-md border border-white/10 overflow-hidden">
            {TIME_RANGES.map((tr) => (
              <button
                key={tr.days}
                onClick={() => setDays(tr.days)}
                className={cn(
                  "px-2 py-1.5 text-xs font-medium transition-all border-l border-white/10 first:border-l-0",
                  days === tr.days
                    ? "bg-[#A78BFA]/10 text-[#A78BFA]"
                    : "bg-white/5 text-white/40 hover:text-white/60"
                )}
              >
                {tr.label}
              </button>
            ))}
          </div>
        </div>

        {/* Paper / KCO toggle */}
        <div className="flex rounded-md border border-white/10 overflow-hidden">
          <button
            onClick={() => { setMode("paper"); setWrThreshold("wr"); }}
            className={cn(
              "flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium transition-all",
              mode === "paper"
                ? "bg-[#22C55E]/10 text-[#22C55E]"
                : "bg-white/5 text-white/40 hover:text-white/60"
            )}
          >
            <BarChart3 className="w-3.5 h-3.5" />
            Paper WR
          </button>
          <button
            onClick={() => { setMode("kco"); setWrThreshold("50"); }}
            className={cn(
              "flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium transition-all border-l border-white/10",
              mode === "kco"
                ? "bg-[#22D3EE]/10 text-[#22D3EE]"
                : "bg-white/5 text-white/40 hover:text-white/60"
            )}
          >
            <TestTube2 className="w-3.5 h-3.5" />
            KCO WR
          </button>
        </div>
        {/* WR threshold toggle — Paper: WR / +50% / 2x, KCO: +50% / 2x */}
        <div className="flex rounded-md border border-white/10 overflow-hidden">
          {mode === "paper" ? (
            <>
              <button
                onClick={() => setWrThreshold("wr")}
                className={cn(
                  "px-2.5 py-1.5 text-xs font-medium transition-all",
                  wrThreshold === "wr"
                    ? "bg-[#F97316]/10 text-[#F97316]"
                    : "bg-white/5 text-white/40 hover:text-white/60"
                )}
              >
                {"> 0%"}
              </button>
              <button
                onClick={() => setWrThreshold("50")}
                className={cn(
                  "px-2.5 py-1.5 text-xs font-medium transition-all border-l border-white/10",
                  wrThreshold === "50"
                    ? "bg-[#F97316]/10 text-[#F97316]"
                    : "bg-white/5 text-white/40 hover:text-white/60"
                )}
              >
                +50%
              </button>
              <button
                onClick={() => setWrThreshold("2x")}
                className={cn(
                  "px-2.5 py-1.5 text-xs font-medium transition-all border-l border-white/10",
                  wrThreshold === "2x"
                    ? "bg-[#F97316]/10 text-[#F97316]"
                    : "bg-white/5 text-white/40 hover:text-white/60"
                )}
              >
                2x
              </button>
            </>
          ) : (
            <>
              <button
                onClick={() => setWrThreshold("50")}
                className={cn(
                  "px-2.5 py-1.5 text-xs font-medium transition-all",
                  wrThreshold === "50"
                    ? "bg-[#F97316]/10 text-[#F97316]"
                    : "bg-white/5 text-white/40 hover:text-white/60"
                )}
              >
                +50%
              </button>
              <button
                onClick={() => setWrThreshold("2x")}
                className={cn(
                  "px-2.5 py-1.5 text-xs font-medium transition-all border-l border-white/10",
                  wrThreshold === "2x"
                    ? "bg-[#F97316]/10 text-[#F97316]"
                    : "bg-white/5 text-white/40 hover:text-white/60"
                )}
              >
                2x
              </button>
            </>
          )}
        </div>
        <button
          onClick={() => setCaOnly(!caOnly)}
          className={cn(
            "flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all border",
            caOnly
              ? "bg-[#22D3EE]/10 border-[#22D3EE]/30 text-[#22D3EE]"
              : "bg-white/5 border-white/10 text-white/40 hover:text-white/60 hover:border-white/20"
          )}
        >
          <FlaskConical className="w-3.5 h-3.5" />
          CA Only
        </button>
      </div>

      {/* Table */}
      <div className={cn("surface-card rounded-lg overflow-hidden", loading && "opacity-60 transition-opacity")}>
        {/* Header */}
        <div className="grid grid-cols-[2rem_1fr_3.5rem_4.5rem_3rem] md:grid-cols-[3rem_1fr_5rem_6rem_5rem_6rem] items-center gap-1.5 md:gap-4 px-2 md:px-4 py-2 border-b border-white/10 bg-white/[0.02]">
          {COLUMNS.map((col) => (
            <button
              key={col.key}
              onClick={() => handleSort(col.key)}
              className={cn(
                "flex items-center gap-0.5 text-xs font-medium text-white/50 hover:text-white/80 transition-colors cursor-pointer select-none",
                col.key === "lastActive" && "hidden md:flex",
                col.className
              )}
            >
              <span className="md:hidden">{col.labelShort ?? col.label}</span>
              <span className="hidden md:inline">{col.label}</span>
              {sortKey === col.key &&
                (sortDir === "desc" ? (
                  <ChevronDown className="w-3 h-3" />
                ) : (
                  <ChevronUp className="w-3 h-3" />
                ))}
            </button>
          ))}
        </div>

        {/* Rows */}
        <div>
          {sorted.map((kol, i) => (
            <KolRow key={kol.name} kol={kol} rank={i + 1} index={i} mode={mode} wrThreshold={wrThreshold} />
          ))}
        </div>
      </div>
    </div>
  );
}
