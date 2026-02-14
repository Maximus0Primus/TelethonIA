"use client";

import { useState, useCallback, useEffect, useRef, useMemo } from "react";
import { cn } from "@/lib/utils";
import type { ScoringConfig, TokenSnapshot } from "@/lib/rescorer";
import {
  type Horizon,
  type BacktestResult,
  type FeatureImpact,
  runBacktest,
  autoOptimize,
} from "@/lib/backtester";
import { FeatureImportance } from "./FeatureImportance";

const MIN_RELIABLE = 200; // unique tokens needed for reliable optimization

function DataHealth({
  uniqueTokens,
  rawSnapshots,
  hits,
  baseRate,
  numCycles,
  snapshots,
}: {
  uniqueTokens: number;
  rawSnapshots: number;
  hits: number;
  baseRate: number;
  numCycles: number;
  snapshots: TokenSnapshot[];
}) {
  const { oldest, newest } = useMemo(() => {
    const dates = snapshots
      .map((s) => s.snapshot_at)
      .filter(Boolean)
      .sort();
    return {
      oldest: dates[0] ? new Date(dates[0]!).toLocaleDateString() : "—",
      newest: dates.length
        ? new Date(dates[dates.length - 1]!).toLocaleDateString()
        : "—",
    };
  }, [snapshots]);

  const pct = Math.min(100, Math.round((uniqueTokens / MIN_RELIABLE) * 100));
  const ready = uniqueTokens >= MIN_RELIABLE;

  return (
    <div className="rounded-lg border border-white/10 bg-white/[0.02] p-3 space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-white/30">
          Data Health
        </span>
        <span
          className={cn(
            "text-[10px] px-2 py-0.5 rounded-full font-medium",
            ready
              ? "bg-green-500/20 text-green-400"
              : "bg-yellow-500/20 text-yellow-400"
          )}
        >
          {ready ? "Ready for optimization" : `${pct}% to reliable`}
        </span>
      </div>

      {/* Progress bar */}
      <div className="h-1 rounded-full bg-white/5 overflow-hidden">
        <div
          className={cn(
            "h-full rounded-full transition-all duration-500",
            ready ? "bg-green-500/60" : "bg-yellow-500/50"
          )}
          style={{ width: `${pct}%` }}
        />
      </div>

      <div className="grid grid-cols-5 gap-2 text-center">
        <div>
          <div className="text-lg font-mono text-white">{uniqueTokens}</div>
          <div className="text-[9px] text-white/30">Unique Tokens</div>
        </div>
        <div>
          <div className="text-lg font-mono text-green-400">{hits}</div>
          <div className="text-[9px] text-white/30">2x Hits</div>
        </div>
        <div>
          <div className="text-lg font-mono text-white/70">{baseRate}%</div>
          <div className="text-[9px] text-white/30">Base Rate</div>
        </div>
        <div>
          <div className="text-lg font-mono text-cyan-400">{numCycles}</div>
          <div className="text-[9px] text-white/30">Cycles</div>
        </div>
        <div>
          <div className="text-[11px] font-mono text-white/50 leading-tight mt-0.5">
            {oldest}
            <br />
            {newest}
          </div>
          <div className="text-[9px] text-white/30">Date Range</div>
        </div>
      </div>

      {rawSnapshots > uniqueTokens && (
        <div className="text-[10px] text-white/15">
          {rawSnapshots} raw snapshots deduped to {uniqueTokens} unique tokens
        </div>
      )}

      {!ready && (
        <div className="text-[10px] text-white/20">
          Need {MIN_RELIABLE - uniqueTokens} more unique tokens for reliable
          optimization. ~{Math.ceil((MIN_RELIABLE - uniqueTokens) / 15)}d at current
          rate.
        </div>
      )}
    </div>
  );
}

interface BacktestPanelProps {
  config: ScoringConfig;
  onApplyBestConfig: (config: ScoringConfig) => void;
}

const HORIZONS: Horizon[] = ["1h", "6h", "12h", "24h"];

export function BacktestPanel({ config, onApplyBestConfig }: BacktestPanelProps) {
  const [horizon, setHorizon] = useState<Horizon>("12h");
  const [snapshots, setSnapshots] = useState<TokenSnapshot[]>([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [features, setFeatures] = useState<FeatureImpact[]>([]);
  const [optimizing, setOptimizing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [optimizedHitRate, setOptimizedHitRate] = useState<number | null>(null);
  const [applied, setApplied] = useState(false);
  const bestConfigRef = useRef<ScoringConfig | null>(null);

  // Fetch labeled snapshots
  const fetchSnapshots = useCallback(async (h: Horizon) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/tuning/backtest?horizon=${h}`);
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setSnapshots(data.data || []);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to fetch");
      setSnapshots([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSnapshots(horizon);
  }, [horizon, fetchSnapshots]);

  // Re-run backtest whenever config or snapshots change
  useEffect(() => {
    if (snapshots.length === 0) {
      setResult(null);
      return;
    }
    const r = runBacktest(snapshots, config, horizon);
    setResult(r);
  }, [snapshots, config, horizon]);

  // Clear optimization results when horizon changes
  useEffect(() => {
    setFeatures([]);
    setOptimizedHitRate(null);
    bestConfigRef.current = null;
    setApplied(false);
  }, [horizon]);

  const handleOptimize = useCallback(() => {
    if (snapshots.length === 0) return;
    setOptimizing(true);
    setApplied(false);
    // Run in next tick to not block UI
    setTimeout(() => {
      const { features: f, bestConfig, bestHitRate } = autoOptimize(snapshots, horizon, config);
      setFeatures(f);
      setOptimizedHitRate(bestHitRate);
      setOptimizing(false);
      bestConfigRef.current = bestConfig;
    }, 10);
  }, [snapshots, horizon, config]);

  const handleApply = useCallback(() => {
    if (bestConfigRef.current) {
      onApplyBestConfig(bestConfigRef.current);
      setApplied(true);
      setTimeout(() => setApplied(false), 3000);
    }
  }, [onApplyBestConfig]);

  return (
    <div className="space-y-4">
      {/* Horizon selector */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-white/40">Horizon:</span>
        {HORIZONS.map((h) => (
          <button
            key={h}
            onClick={() => setHorizon(h)}
            className={cn(
              "px-3 py-1 text-xs rounded-full transition-colors",
              h === horizon
                ? "bg-white text-black font-semibold"
                : "bg-white/5 text-white/50 hover:text-white hover:bg-white/10"
            )}
          >
            {h}
          </button>
        ))}
      </div>

      {loading && (
        <div className="text-white/30 text-sm py-4 text-center">Loading labeled snapshots...</div>
      )}

      {error && (
        <div className="text-red-400/70 text-sm py-2">{error}</div>
      )}

      {!loading && snapshots.length === 0 && !error && (
        <div className="text-white/20 text-sm py-8 text-center">
          No labeled snapshots yet. The outcome tracker needs 6-24h after scraping starts
          to label tokens with 2x outcomes.
        </div>
      )}

      {result && snapshots.length > 0 && (
        <div className="space-y-4">
          {/* Data Health */}
          <DataHealth
            uniqueTokens={result.total_snapshots}
            rawSnapshots={result.total_raw_snapshots}
            hits={result.total_2x}
            baseRate={result.base_rate}
            numCycles={result.num_cycles}
            snapshots={snapshots}
          />

          {/* Hit rates */}
          <div className="grid grid-cols-3 gap-3">
            {[
              { label: "Top 5", value: result.top5_hit_rate },
              { label: "Top 10", value: result.top10_hit_rate },
              { label: "Top 20", value: result.top20_hit_rate },
            ].map(({ label, value }) => (
              <div key={label} className="rounded-lg bg-white/[0.03] border border-white/10 p-3 text-center">
                <div className="text-[10px] text-white/30 uppercase">{label} Hit Rate</div>
                <div className="text-xl font-mono text-white mt-1">
                  {value}%
                </div>
              </div>
            ))}
          </div>

          {result.num_cycles > 0 && (
            <div className="text-[10px] text-white/20 -mt-2">
              Hit rates averaged across {result.num_cycles} cycles (5+ tokens each)
            </div>
          )}

          {/* Separation */}
          <div className="flex items-center gap-4 text-xs">
            <div className="text-white/40">
              Avg score (2x): <span className="text-green-400 font-mono">{result.avg_score_2x}</span>
            </div>
            <div className="text-white/40">
              Avg score (no 2x): <span className="text-red-400/70 font-mono">{result.avg_score_no2x}</span>
            </div>
            <div className="text-white/40">
              Separation: <span className="text-white font-mono">{result.separation}x</span>
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center gap-2">
            <button
              onClick={handleOptimize}
              disabled={optimizing}
              className={cn(
                "px-4 py-2 text-sm rounded-lg transition-colors",
                optimizing
                  ? "bg-white/5 text-white/30 cursor-wait"
                  : "bg-white/10 text-white hover:bg-white/20"
              )}
            >
              {optimizing ? "Optimizing..." : "Auto-Optimize"}
            </button>

            {features.length > 0 && (
              <button
                onClick={handleApply}
                className={cn(
                  "px-4 py-2 text-sm rounded-lg transition-all duration-300",
                  applied
                    ? "bg-green-500/40 text-green-300 ring-1 ring-green-400/50"
                    : "bg-green-500/20 text-green-400 hover:bg-green-500/30"
                )}
              >
                {applied
                  ? "Config Applied"
                  : `Apply Best Config${optimizedHitRate !== null ? ` (top5: ${optimizedHitRate}%)` : ""}`}
              </button>
            )}
          </div>

          {/* Feature importance */}
          {features.length > 0 && <FeatureImportance features={features} />}
        </div>
      )}
    </div>
  );
}
