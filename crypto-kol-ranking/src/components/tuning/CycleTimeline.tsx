"use client";

import { useState, useEffect } from "react";
import { cn } from "@/lib/utils";

interface Cycle {
  ts: string;
  count: number;
}

interface CycleTimelineProps {
  onCycleChange: (cycleTs: string | null) => void;
  currentCycle: string | null;
}

function formatTime(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" });
}

function formatDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString("fr-FR", { day: "2-digit", month: "short" });
}

function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 60) return `il y a ${mins}min`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `il y a ${hours}h`;
  return `il y a ${Math.floor(hours / 24)}j`;
}

export function CycleTimeline({ onCycleChange, currentCycle }: CycleTimelineProps) {
  const [cycles, setCycles] = useState<Cycle[]>([]);
  const [loading, setLoading] = useState(true);
  // selectedIdx: 0 = newest (right of slider), maxIdx = oldest (left of slider)
  const [selectedIdx, setSelectedIdx] = useState(0);

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch("/api/tuning/snapshots?cycles=true");
        if (!res.ok) return;
        const data = await res.json();
        setCycles(data.cycles || []);
      } catch {
        /* ignore */
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  // On initial load, select latest
  useEffect(() => {
    if (cycles.length > 0 && !currentCycle) {
      setSelectedIdx(0);
    }
  }, [cycles, currentCycle]);

  const handleSelect = (idx: number) => {
    setSelectedIdx(idx);
    onCycleChange(cycles[idx]?.ts ?? null);
  };

  // Slider value is inverted: left=oldest (maxIdx), right=newest (0)
  // sliderValue = maxIdx - selectedIdx
  const maxIdx = Math.max(0, cycles.length - 1);

  const handleSlider = (e: React.ChangeEvent<HTMLInputElement>) => {
    const sliderVal = parseInt(e.target.value, 10);
    const idx = maxIdx - sliderVal;
    handleSelect(idx);
  };

  if (loading) {
    return (
      <div className="text-[10px] text-white/20 py-2">Loading timeline...</div>
    );
  }

  if (cycles.length === 0) return null;

  const selected = cycles[selectedIdx];
  const sliderValue = maxIdx - selectedIdx; // 0=oldest on left, maxIdx=newest on right

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-white/90 uppercase tracking-wider">
          Timeline
        </h3>
        <button
          onClick={() => handleSelect(0)}
          className={cn(
            "text-[10px] px-2 py-0.5 rounded transition-colors",
            selectedIdx === 0
              ? "bg-cyan-500/20 text-cyan-300"
              : "bg-white/5 text-white/30 hover:text-white/60",
          )}
        >
          Latest
        </button>
      </div>

      {/* Current cycle info */}
      <div className="flex items-center gap-3 text-xs">
        <span className="font-mono text-white/80">
          {formatDate(selected.ts)} {formatTime(selected.ts)}
        </span>
        <span className="text-white/30">{relativeTime(selected.ts)}</span>
        <span className="text-white/20">{selected.count} tokens</span>
      </div>

      {/* Slider — left=oldest, right=newest */}
      <div>
        <input
          type="range"
          min={0}
          max={maxIdx}
          step={1}
          value={sliderValue}
          onChange={handleSlider}
          className="w-full h-2 bg-white/10 rounded-full appearance-none cursor-pointer
            [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4
            [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full
            [&::-webkit-slider-thumb]:bg-cyan-400 [&::-webkit-slider-thumb]:cursor-pointer
            [&::-webkit-slider-thumb]:shadow-[0_0_10px_rgba(0,200,255,0.5)]"
        />
        <div className="flex justify-between text-[9px] text-white/20 mt-1 px-0.5">
          <span>{cycles.length > 1 ? relativeTime(cycles[maxIdx].ts) : ""}</span>
          <span>maintenant</span>
        </div>
      </div>

      {/* Quick-jump buttons for recent cycles */}
      <div className="flex gap-1 overflow-x-auto pb-1 scrollbar-none">
        {cycles.slice(0, 12).map((c, i) => (
          <button
            key={c.ts}
            onClick={() => handleSelect(i)}
            className={cn(
              "shrink-0 px-2 py-1 rounded text-[10px] font-mono transition-all",
              i === selectedIdx
                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 shadow-[0_0_8px_rgba(0,200,255,0.15)]"
                : "bg-white/[0.03] text-white/25 hover:text-white/50 hover:bg-white/5 border border-transparent",
            )}
            title={`${new Date(c.ts).toLocaleString()} — ${c.count} tokens`}
          >
            {formatTime(c.ts)}
          </button>
        ))}
      </div>
    </div>
  );
}
