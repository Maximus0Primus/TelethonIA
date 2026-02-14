"use client";

import { type ScoringConfig, type ExtractionMode, normalizeWeights } from "@/lib/rescorer";
import { cn } from "@/lib/utils";

interface WeightSlidersProps {
  config: ScoringConfig;
  onChange: (config: ScoringConfig) => void;
}

const WEIGHT_LABELS: Record<string, string> = {
  consensus: "Consensus",
  sentiment: "Sentiment",
  conviction: "Conviction",
  breadth: "Breadth",
  price_action: "Price Action",
};

const EXTRACTION_MODES: { value: ExtractionMode; label: string }[] = [
  { value: "both", label: "Both" },
  { value: "ca_only", label: "CA Only" },
  { value: "ticker_only", label: "Ticker Only" },
];

export function WeightSliders({ config, onChange }: WeightSlidersProps) {
  const weights = config.weights;
  const total = Object.values(weights).reduce((s, v) => s + v, 0);

  const handleChange = (key: keyof typeof weights, rawValue: number) => {
    const newWeights = { ...weights, [key]: rawValue };
    onChange({ ...config, weights: normalizeWeights(newWeights, key) });
  };

  return (
    <div className="space-y-4">
      {/* Extraction mode toggle */}
      <div className="space-y-2">
        <h3 className="text-sm font-semibold text-white/90 uppercase tracking-wider">
          Extraction Mode
        </h3>
        <div className="flex bg-white/5 rounded-full p-0.5">
          {EXTRACTION_MODES.map(({ value, label }) => (
            <button
              key={value}
              onClick={() => onChange({ ...config, extraction_mode: value })}
              className={cn(
                "flex-1 px-3 py-1.5 text-xs rounded-full transition-colors",
                config.extraction_mode === value
                  ? "bg-white text-black font-semibold"
                  : "text-white/50 hover:text-white"
              )}
            >
              {label}
            </button>
          ))}
        </div>
        <p className="text-[10px] text-white/25 leading-snug">
          CA Only = tokens with contract address. Ticker Only = $TICKER mentions without CA.
        </p>
      </div>

      {/* Weight sliders */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-white/90 uppercase tracking-wider">
          Component Weights
        </h3>
        <span className="text-xs text-white/40 font-mono">
          sum: {total.toFixed(2)}
        </span>
      </div>

      {(Object.keys(weights) as (keyof typeof weights)[]).map((key) => {
        const val = weights[key];
        const pct = Math.round(val * 100);
        return (
          <div key={key} className="space-y-1">
            <div className="flex items-center justify-between">
              <label className="text-sm text-white/70">{WEIGHT_LABELS[key]}</label>
              <span className="text-sm font-mono text-white/90 w-14 text-right">
                {pct}%
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={100}
              step={1}
              value={pct}
              onChange={(e) => handleChange(key, parseInt(e.target.value, 10) / 100)}
              className="w-full h-1.5 bg-white/10 rounded-full appearance-none cursor-pointer
                [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3.5
                [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:rounded-full
                [&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:cursor-pointer
                [&::-webkit-slider-thumb]:shadow-[0_0_8px_rgba(255,255,255,0.3)]"
            />
          </div>
        );
      })}
    </div>
  );
}
