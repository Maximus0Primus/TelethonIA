"use client";

import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  type ScoringConfig,
  type TokenSnapshot,
  DEFAULT_CONFIG,
  rescoreAll,
} from "@/lib/rescorer";
import { WeightSliders } from "@/components/tuning/WeightSliders";
import { MultiplierToggles } from "@/components/tuning/MultiplierToggles";
import { RankingPreview } from "@/components/tuning/RankingPreview";
import { BacktestPanel } from "@/components/tuning/BacktestPanel";
import { CycleTimeline } from "@/components/tuning/CycleTimeline";
import { cn } from "@/lib/utils";

type Tab = "live" | "backtest";

const PRESETS_KEY = "tuning-presets";

interface Preset {
  name: string;
  config: ScoringConfig;
}

function configDiffersFromDefault(config: ScoringConfig): boolean {
  return JSON.stringify(config) !== JSON.stringify(DEFAULT_CONFIG);
}

export default function TuningPage() {
  const [config, setConfig] = useState<ScoringConfig>(
    JSON.parse(JSON.stringify(DEFAULT_CONFIG))
  );
  const [snapshots, setSnapshots] = useState<TokenSnapshot[]>([]);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<Tab>("live");
  const [presetName, setPresetName] = useState("");
  const [presets, setPresets] = useState<Preset[]>([]);
  const [currentCycle, setCurrentCycle] = useState<string | null>(null);
  const [configApplied, setConfigApplied] = useState(false);
  const [applyingToProd, setApplyingToProd] = useState(false);
  const [prodPushStatus, setProdPushStatus] = useState<"idle" | "success" | "error">("idle");
  const [prodConfig, setProdConfig] = useState<{ updated_by?: string; updated_at?: string } | null>(null);
  const appliedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Load current production config on mount
  useEffect(() => {
    fetch("/api/tuning/config")
      .then((r) => r.json())
      .then((data) => {
        if (data.updated_by) setProdConfig(data);
      })
      .catch(() => {});
  }, []);

  // Load presets from localStorage
  useEffect(() => {
    try {
      const stored = localStorage.getItem(PRESETS_KEY);
      if (stored) setPresets(JSON.parse(stored));
    } catch { /* ignore */ }
  }, []);

  // Fetch snapshots (latest or for a specific cycle)
  const fetchSnapshots = useCallback(async (cycleTs: string | null) => {
    setLoading(true);
    try {
      const url = cycleTs
        ? `/api/tuning/snapshots?cycle=${encodeURIComponent(cycleTs)}`
        : "/api/tuning/snapshots";
      const res = await fetch(url);
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setSnapshots(data.data || []);
    } catch (e) {
      console.error("Failed to load snapshots:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  // Initial load
  useEffect(() => {
    fetchSnapshots(null);
  }, [fetchSnapshots]);

  const handleCycleChange = useCallback((cycleTs: string | null) => {
    setCurrentCycle(cycleTs);
    if (cycleTs) fetchSnapshots(cycleTs);
  }, [fetchSnapshots]);

  // Re-score whenever config or snapshots change
  const scored = useMemo(() => {
    if (snapshots.length === 0) return [];
    return rescoreAll(snapshots, config);
  }, [snapshots, config]);

  const handleReset = useCallback(() => {
    setConfig(JSON.parse(JSON.stringify(DEFAULT_CONFIG)));
  }, []);

  const handleSavePreset = useCallback(() => {
    if (!presetName.trim()) return;
    const newPresets = [...presets.filter((p) => p.name !== presetName), { name: presetName, config }];
    setPresets(newPresets);
    localStorage.setItem(PRESETS_KEY, JSON.stringify(newPresets));
    setPresetName("");
  }, [presetName, config, presets]);

  const handleLoadPreset = useCallback(
    (preset: Preset) => {
      // Merge with defaults to handle presets saved before kol_tuning existed
      const loaded = JSON.parse(JSON.stringify(preset.config));
      setConfig({ ...JSON.parse(JSON.stringify(DEFAULT_CONFIG)), ...loaded });
    },
    []
  );

  const handleDeletePreset = useCallback(
    (name: string) => {
      const newPresets = presets.filter((p) => p.name !== name);
      setPresets(newPresets);
      localStorage.setItem(PRESETS_KEY, JSON.stringify(newPresets));
    },
    [presets]
  );

  const handleApplyFromBacktest = useCallback((newConfig: ScoringConfig) => {
    setConfig(newConfig);
    setConfigApplied(true);
    if (appliedTimerRef.current) clearTimeout(appliedTimerRef.current);
    appliedTimerRef.current = setTimeout(() => setConfigApplied(false), 3000);
  }, []);

  const handlePushToProduction = useCallback(async () => {
    if (applyingToProd) return;
    setApplyingToProd(true);
    setProdPushStatus("idle");
    try {
      const res = await fetch("/api/tuning/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          weights: {
            consensus: config.weights.consensus,
            conviction: config.weights.conviction,
            breadth: config.weights.breadth,
            price_action: config.weights.price_action,
          },
          reason: "Tuning Lab manual push",
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setProdPushStatus("success");
      setProdConfig({ updated_by: "tuning_lab", updated_at: new Date().toISOString() });
      setTimeout(() => setProdPushStatus("idle"), 4000);
    } catch (e) {
      console.error("Push to prod failed:", e);
      setProdPushStatus("error");
      setTimeout(() => setProdPushStatus("idle"), 4000);
    } finally {
      setApplyingToProd(false);
    }
  }, [config, applyingToProd]);

  const isCustomConfig = configDiffersFromDefault(config);

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="border-b border-white/10 px-4 sm:px-6 py-4"
      >
        <div className="max-w-[1400px] mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold tracking-tight">Tuning Lab</h1>
            <p className="text-xs text-white/30 mt-0.5">
              {snapshots.length} tokens{currentCycle ? ` @ ${new Date(currentCycle).toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" })}` : " (latest)"}
            </p>
          </div>
          <div className="flex items-center gap-3">
            {/* Tab switcher */}
            <div className="flex bg-white/5 rounded-full p-0.5">
              {(["live", "backtest"] as Tab[]).map((t) => (
                <button
                  key={t}
                  onClick={() => setTab(t)}
                  className={cn(
                    "px-4 py-1.5 text-xs rounded-full transition-colors capitalize relative",
                    tab === t
                      ? "bg-white text-black font-semibold"
                      : "text-white/50 hover:text-white"
                  )}
                >
                  {t === "live" ? "Live Tuning" : "Backtest"}
                  {t === "live" && isCustomConfig && tab !== "live" && (
                    <span className="absolute -top-1 -right-1 w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                  )}
                </button>
              ))}
            </div>
            {/* Applied flash */}
            <AnimatePresence>
              {configApplied && (
                <motion.span
                  initial={{ opacity: 0, x: 10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0 }}
                  className="text-xs text-green-400 font-medium"
                >
                  Config updated
                </motion.span>
              )}
            </AnimatePresence>
            <button
              onClick={handleReset}
              className="px-3 py-1.5 text-xs rounded-lg bg-white/5 text-white/50 hover:text-white hover:bg-white/10 transition-colors"
            >
              Reset
            </button>
          </div>
        </div>
      </motion.div>

      {/* Main content */}
      <div className="max-w-[1400px] mx-auto px-4 sm:px-6 py-6">
        {tab === "live" ? (
          <div className="grid grid-cols-1 lg:grid-cols-[320px_1fr] gap-6">
            {/* Controls panel — always mounted so timeline keeps state */}
            <motion.div
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              className="space-y-6 lg:max-h-[calc(100vh-140px)] lg:overflow-y-auto lg:pr-2"
            >
              {/* Timeline */}
              <div className="rounded-xl border border-white/10 bg-white/[0.02] p-4">
                <CycleTimeline
                  onCycleChange={handleCycleChange}
                  currentCycle={currentCycle}
                />
              </div>

              {/* Weight sliders */}
              <div className="rounded-xl border border-white/10 bg-white/[0.02] p-4">
                <WeightSliders config={config} onChange={setConfig} />
              </div>

              {/* Multiplier toggles */}
              <div className="rounded-xl border border-white/10 bg-white/[0.02] p-4">
                <MultiplierToggles config={config} onChange={setConfig} />
              </div>

              {/* Presets */}
              <div className="rounded-xl border border-white/10 bg-white/[0.02] p-4 space-y-3">
                <h3 className="text-sm font-semibold text-white/90 uppercase tracking-wider">
                  Presets
                </h3>

                {/* Save */}
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={presetName}
                    onChange={(e) => setPresetName(e.target.value)}
                    placeholder="Preset name..."
                    className="flex-1 bg-white/5 border border-white/10 rounded-lg px-3 py-1.5 text-sm text-white placeholder:text-white/20 focus:outline-none focus:border-white/30"
                    onKeyDown={(e) => e.key === "Enter" && handleSavePreset()}
                  />
                  <button
                    onClick={handleSavePreset}
                    disabled={!presetName.trim()}
                    className="px-3 py-1.5 text-xs rounded-lg bg-white/10 text-white hover:bg-white/20 transition-colors disabled:opacity-30"
                  >
                    Save
                  </button>
                </div>

                {/* Load */}
                {presets.length > 0 && (
                  <div className="space-y-1">
                    {presets.map((p) => (
                      <div
                        key={p.name}
                        className="flex items-center justify-between px-3 py-1.5 rounded-lg bg-white/[0.02] hover:bg-white/5 transition-colors"
                      >
                        <button
                          onClick={() => handleLoadPreset(p)}
                          className="text-sm text-white/70 hover:text-white"
                        >
                          {p.name}
                        </button>
                        <button
                          onClick={() => handleDeletePreset(p.name)}
                          className="text-[10px] text-red-400/50 hover:text-red-400"
                        >
                          del
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Apply to Production */}
              {isCustomConfig && (
                <div className="rounded-xl border border-orange-500/20 bg-orange-500/[0.03] p-4 space-y-3">
                  <h3 className="text-sm font-semibold text-orange-400/90 uppercase tracking-wider">
                    Production
                  </h3>
                  <p className="text-[11px] text-white/40 leading-relaxed">
                    Push current weights to production. The scraper will pick them up on its next cycle (~15 min).
                  </p>
                  {prodConfig && (
                    <p className="text-[10px] text-white/25">
                      Last update: {prodConfig.updated_by} @ {prodConfig.updated_at ? new Date(prodConfig.updated_at).toLocaleString("fr-FR") : "?"}
                    </p>
                  )}
                  <button
                    onClick={handlePushToProduction}
                    disabled={applyingToProd}
                    className={cn(
                      "w-full px-4 py-2 text-sm font-medium rounded-lg transition-all",
                      prodPushStatus === "success"
                        ? "bg-green-600 text-white"
                        : prodPushStatus === "error"
                          ? "bg-red-600 text-white"
                          : "bg-orange-500/20 text-orange-400 hover:bg-orange-500/30 border border-orange-500/20",
                      applyingToProd && "opacity-50 cursor-wait",
                    )}
                  >
                    {applyingToProd
                      ? "Applying..."
                      : prodPushStatus === "success"
                        ? "Applied!"
                        : prodPushStatus === "error"
                          ? "Failed"
                          : "Apply to Production"}
                  </button>
                </div>
              )}
            </motion.div>

            {/* Ranking preview — loading indicator stays here */}
            <motion.div
              initial={{ opacity: 0, x: 10 }}
              animate={{ opacity: 1, x: 0 }}
              className="rounded-xl border border-white/10 bg-white/[0.02] overflow-hidden"
            >
              {loading ? (
                <div className="flex items-center justify-center h-64 text-white/20">
                  Loading snapshots...
                </div>
              ) : (
                <RankingPreview tokens={scored} />
              )}
            </motion.div>
          </div>
        ) : (
          /* Backtest tab */
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="rounded-xl border border-white/10 bg-white/[0.02] p-6"
          >
            <BacktestPanel config={config} onApplyBestConfig={handleApplyFromBacktest} />
          </motion.div>
        )}
      </div>
    </div>
  );
}
