"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { LayoutGrid, List, SlidersHorizontal } from "lucide-react";
import { cn } from "@/lib/utils";

interface ViewControlsProps {
  viewMode: "grid" | "list";
  onViewModeChange: (mode: "grid" | "list") => void;
  blend: number;
  onApplyBlend: (blend: number) => void;
}

export function ViewControls({
  viewMode,
  onViewModeChange,
  blend,
  onApplyBlend,
}: ViewControlsProps) {
  const [filterOpen, setFilterOpen] = useState(false);
  const [localBlend, setLocalBlend] = useState(blend);
  const filterRef = useRef<HTMLDivElement>(null);

  // Sync local slider when external blend changes
  useEffect(() => {
    setLocalBlend(blend);
  }, [blend]);

  // Close on click outside
  useEffect(() => {
    if (!filterOpen) return;
    const handler = (e: MouseEvent) => {
      if (filterRef.current && !filterRef.current.contains(e.target as Node)) {
        setFilterOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [filterOpen]);

  const handleApply = () => {
    setFilterOpen(false);
    onApplyBlend(localBlend);
  };

  const label =
    localBlend <= 20
      ? "Conviction"
      : localBlend >= 80
      ? "Quick Gains"
      : "Mixed";

  return (
    <>
      {/* Grid/List Toggle - Bottom Left */}
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.5, delay: 0.6 }}
        className="fixed bottom-8 left-6 z-40 hidden sm:flex"
      >
        <div className="glass-control flex items-center rounded-lg p-1">
          <button
            onClick={() => onViewModeChange("grid")}
            className={cn(
              "flex h-9 w-9 items-center justify-center rounded-md transition-colors",
              viewMode === "grid"
                ? "bg-white text-black"
                : "text-white/60 hover:text-white"
            )}
            aria-label="Grid view"
          >
            <LayoutGrid className="h-4 w-4" />
          </button>
          <button
            onClick={() => onViewModeChange("list")}
            className={cn(
              "flex h-9 w-9 items-center justify-center rounded-md transition-colors",
              viewMode === "list"
                ? "bg-white text-black"
                : "text-white/60 hover:text-white"
            )}
            aria-label="List view"
          >
            <List className="h-4 w-4" />
          </button>
        </div>
      </motion.div>

      {/* Filter Button + Slider Popover - Bottom Right */}
      <motion.div
        ref={filterRef}
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.5, delay: 0.6 }}
        className="fixed bottom-8 right-6 z-40 hidden sm:flex flex-col items-end gap-2"
      >
        <AnimatePresence>
          {filterOpen && (
            <motion.div
              initial={{ opacity: 0, y: 8, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 8, scale: 0.95 }}
              transition={{ duration: 0.15 }}
              className="glass-control rounded-xl p-4 flex flex-col gap-3 w-[280px]"
            >
              {/* Labels */}
              <div className="flex items-center justify-between text-[11px] font-medium">
                <span className={cn(
                  "transition-colors",
                  localBlend <= 30 ? "text-white" : "text-white/40"
                )}>
                  Conviction
                </span>
                <span className={cn(
                  "transition-colors",
                  localBlend >= 70 ? "text-white" : "text-white/40"
                )}>
                  Quick Gains
                </span>
              </div>

              {/* Slider */}
              <input
                type="range"
                min={0}
                max={100}
                step={5}
                value={localBlend}
                onChange={(e) => setLocalBlend(parseInt(e.target.value, 10))}
                className="w-full h-1.5 rounded-full appearance-none cursor-pointer bg-white/10
                  [&::-webkit-slider-thumb]:appearance-none
                  [&::-webkit-slider-thumb]:h-4
                  [&::-webkit-slider-thumb]:w-4
                  [&::-webkit-slider-thumb]:rounded-full
                  [&::-webkit-slider-thumb]:bg-white
                  [&::-webkit-slider-thumb]:shadow-[0_0_8px_rgba(255,255,255,0.4)]
                  [&::-webkit-slider-thumb]:transition-shadow
                  [&::-webkit-slider-thumb]:hover:shadow-[0_0_12px_rgba(255,255,255,0.6)]
                  [&::-moz-range-thumb]:h-4
                  [&::-moz-range-thumb]:w-4
                  [&::-moz-range-thumb]:rounded-full
                  [&::-moz-range-thumb]:bg-white
                  [&::-moz-range-thumb]:border-0
                  [&::-moz-range-thumb]:shadow-[0_0_8px_rgba(255,255,255,0.4)]
                "
              />

              {/* OK button */}
              <button
                onClick={handleApply}
                className="w-full py-2 text-xs font-semibold rounded-lg bg-white text-black hover:bg-white/90 transition-colors"
              >
                OK
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        <button
          onClick={() => setFilterOpen(!filterOpen)}
          className={cn(
            "glass-control flex h-11 items-center gap-2 rounded-lg px-4 text-sm font-medium transition-colors",
            filterOpen
              ? "text-white bg-white/10"
              : "text-white/80 hover:text-white"
          )}
        >
          <SlidersHorizontal className="h-4 w-4" />
          <span>Filter</span>
        </button>
      </motion.div>
    </>
  );
}
