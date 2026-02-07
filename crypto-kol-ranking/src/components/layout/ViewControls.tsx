"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { LayoutGrid, List, SlidersHorizontal } from "lucide-react";
import { cn } from "@/lib/utils";

export type TimeWindow = "3h" | "6h" | "12h" | "24h" | "48h" | "7d";

const TIME_WINDOWS: TimeWindow[] = ["3h", "6h", "12h", "24h", "48h", "7d"];

interface ViewControlsProps {
  viewMode: "grid" | "list";
  onViewModeChange: (mode: "grid" | "list") => void;
  timeWindow: TimeWindow;
  onTimeWindowChange: (window: TimeWindow) => void;
}

export function ViewControls({
  viewMode,
  onViewModeChange,
  timeWindow,
  onTimeWindowChange,
}: ViewControlsProps) {
  const [filterOpen, setFilterOpen] = useState(false);
  const filterRef = useRef<HTMLDivElement>(null);

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

      {/* Filter Button + Time Window Popover - Bottom Right */}
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
              className="glass-control rounded-lg p-1.5 flex items-center gap-0.5"
            >
              {TIME_WINDOWS.map((w) => (
                <button
                  key={w}
                  onClick={() => {
                    onTimeWindowChange(w);
                    setFilterOpen(false);
                  }}
                  className={cn(
                    "px-3 py-1.5 text-xs font-mono rounded-md transition-colors",
                    timeWindow === w
                      ? "bg-white text-black"
                      : "text-white/60 hover:text-white"
                  )}
                >
                  {w}
                </button>
              ))}
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
          <span>{timeWindow}</span>
        </button>
      </motion.div>
    </>
  );
}
