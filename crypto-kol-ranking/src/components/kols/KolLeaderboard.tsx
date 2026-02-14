"use client";

import { useState, useMemo, useEffect, useCallback } from "react";
import { motion } from "framer-motion";
import { ChevronUp, ChevronDown, FlaskConical } from "lucide-react";
import { cn } from "@/lib/utils";
import { KolRow } from "./KolRow";
import type { KolRowData } from "./KolRow";

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

function compare(a: KolRowData, b: KolRowData, key: SortKey, dir: "asc" | "desc"): number {
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
      return m * (((a.winRateAny ?? a.winRate24h ?? a.winRate12h) ?? -1) - ((b.winRateAny ?? b.winRate24h ?? b.winRate12h) ?? -1));
    case "calls":
      return m * (a.uniqueTokens - b.uniqueTokens);
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
  const [sortKey, setSortKey] = useState<SortKey>("score");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  const fetchData = useCallback(async (ca: boolean) => {
    setLoading(true);
    try {
      const res = await fetch(`/api/kols${ca ? "?ca_only=true" : ""}`);
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
    fetchData(caOnly);
  }, [caOnly, fetchData]);

  const sorted = useMemo(() => {
    const arr = [...kols];
    arr.sort((a, b) => compare(a, b, sortKey, sortDir));
    return arr;
  }, [kols, sortKey, sortDir]);

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
  const withWinRate = kols.filter((k) => (k.winRateAny ?? k.winRate24h ?? k.winRate12h) !== null);
  const avgWinRate =
    withWinRate.length > 0
      ? withWinRate.reduce((s, k) => s + (k.winRateAny ?? k.winRate24h ?? k.winRate12h ?? 0), 0) /
        withWinRate.length
      : 0;

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
          {
            label: "Avg Score",
            value: avgScore > 0 ? avgScore.toFixed(2) : "--",
            accent: "text-[#22D3EE]",
          },
          {
            label: "Avg Win Rate",
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

      {/* CA-Only Toggle */}
      <div className="flex items-center justify-end gap-2 mb-3">
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
            <KolRow key={kol.name} kol={kol} rank={i + 1} index={i} />
          ))}
        </div>
      </div>
    </div>
  );
}
