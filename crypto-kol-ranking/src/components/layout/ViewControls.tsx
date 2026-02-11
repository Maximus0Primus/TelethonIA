"use client";

import { motion } from "framer-motion";
import { LayoutGrid, List } from "lucide-react";
import { cn } from "@/lib/utils";

interface ViewControlsProps {
  viewMode: "grid" | "list";
  onViewModeChange: (mode: "grid" | "list") => void;
}

export function ViewControls({
  viewMode,
  onViewModeChange,
}: ViewControlsProps) {
  return (
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
  );
}
