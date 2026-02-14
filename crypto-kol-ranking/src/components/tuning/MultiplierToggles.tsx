"use client";

import { type ScoringConfig } from "@/lib/rescorer";
import { cn } from "@/lib/utils";

interface MultiplierTogglesProps {
  config: ScoringConfig;
  onChange: (config: ScoringConfig) => void;
}

const MULT_LABELS: Record<string, string> = {
  safety_penalty: "Safety Penalty",
  onchain_multiplier: "On-chain Mult",
  death_penalty: "Death Penalty",
  already_pumped: "Already Pumped",
  crash_penalty: "Crash Penalty",
  activity_mult: "Activity Mult",
  squeeze: "Volume Squeeze",
  trend: "Trend Strength",
  entry_premium: "Entry Premium",
  pump_bonus: "Pump.fun Bonus",
  wash_pen: "Wash Trading",
  pvp_pen: "PVP Penalty",
  pump_pen: "Artificial Pump",
  breadth_pen: "Breadth Floor",
  stale_pen: "Stale Token Penalty",
  pump_momentum_pen: "Pump Momentum",
};

export function MultiplierToggles({ config, onChange }: MultiplierTogglesProps) {
  const mults = config.multipliers;

  const toggle = (key: keyof typeof mults) => {
    const current = mults[key];
    onChange({
      ...config,
      multipliers: {
        ...mults,
        [key]: { ...current, enabled: !current.enabled },
      },
    });
  };

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold text-white/90 uppercase tracking-wider">
        Multipliers
      </h3>

      <div className="grid grid-cols-1 gap-2">
        {(Object.keys(mults) as (keyof typeof mults)[]).map((key) => {
          const isOn = mults[key].enabled;
          return (
            <button
              key={key}
              onClick={() => toggle(key)}
              className={cn(
                "flex items-center justify-between px-3 py-2 rounded-lg text-sm transition-all",
                "border",
                isOn
                  ? "border-white/20 bg-white/5 text-white"
                  : "border-white/5 bg-transparent text-white/30"
              )}
            >
              <span>{MULT_LABELS[key]}</span>
              <span
                className={cn(
                  "text-[10px] font-mono px-1.5 py-0.5 rounded",
                  isOn ? "bg-green-500/20 text-green-400" : "bg-red-500/10 text-red-400/50"
                )}
              >
                {isOn ? "ON" : "OFF"}
              </span>
            </button>
          );
        })}
      </div>

      {/* Squeeze max bonus slider */}
      {mults.squeeze.enabled && (
        <div className="space-y-1 pl-3 border-l border-white/10">
          <div className="flex justify-between text-xs text-white/50">
            <span>Squeeze max bonus</span>
            <span className="font-mono">{mults.squeeze.max_bonus.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min={0}
            max={0.5}
            step={0.05}
            value={mults.squeeze.max_bonus}
            onChange={(e) =>
              onChange({
                ...config,
                multipliers: {
                  ...mults,
                  squeeze: { ...mults.squeeze, max_bonus: parseFloat(e.target.value) },
                },
              })
            }
            className="w-full h-1 bg-white/10 rounded-full appearance-none cursor-pointer
              [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3
              [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full
              [&::-webkit-slider-thumb]:bg-white/70 [&::-webkit-slider-thumb]:cursor-pointer"
          />
        </div>
      )}

      {/* Trend bonus slider */}
      {mults.trend.enabled && (
        <div className="space-y-1 pl-3 border-l border-white/10">
          <div className="flex justify-between text-xs text-white/50">
            <span>Trend bonus</span>
            <span className="font-mono">{mults.trend.bonus.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min={1.0}
            max={1.3}
            step={0.01}
            value={mults.trend.bonus}
            onChange={(e) =>
              onChange({
                ...config,
                multipliers: {
                  ...mults,
                  trend: { ...mults.trend, bonus: parseFloat(e.target.value) },
                },
              })
            }
            className="w-full h-1 bg-white/10 rounded-full appearance-none cursor-pointer
              [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3
              [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full
              [&::-webkit-slider-thumb]:bg-white/70 [&::-webkit-slider-thumb]:cursor-pointer"
          />
        </div>
      )}

      {/* Confirmation Gate */}
      <div className="space-y-3 pt-2 border-t border-white/10">
        <button
          onClick={() =>
            onChange({
              ...config,
              confirmation_gate: {
                ...config.confirmation_gate,
                enabled: !config.confirmation_gate.enabled,
              },
            })
          }
          className={cn(
            "flex items-center justify-between w-full px-3 py-2 rounded-lg text-sm transition-all border",
            config.confirmation_gate.enabled
              ? "border-white/20 bg-white/5 text-white"
              : "border-white/5 bg-transparent text-white/30"
          )}
        >
          <span>Confirmation Gate</span>
          <span
            className={cn(
              "text-[10px] font-mono px-1.5 py-0.5 rounded",
              config.confirmation_gate.enabled
                ? "bg-green-500/20 text-green-400"
                : "bg-red-500/10 text-red-400/50"
            )}
          >
            {config.confirmation_gate.enabled ? "ON" : "OFF"}
          </span>
        </button>

        {config.confirmation_gate.enabled && (
          <div className="space-y-2 pl-3 border-l border-white/10">
            <div className="flex justify-between text-xs text-white/50">
              <span>Min pillars</span>
              <select
                value={config.confirmation_gate.min_pillars}
                onChange={(e) =>
                  onChange({
                    ...config,
                    confirmation_gate: {
                      ...config.confirmation_gate,
                      min_pillars: parseInt(e.target.value),
                    },
                  })
                }
                className="bg-white/5 border border-white/10 rounded px-2 py-0.5 text-white text-xs"
              >
                <option value={1}>1</option>
                <option value={2}>2</option>
                <option value={3}>3</option>
              </select>
            </div>
            <div className="flex justify-between text-xs text-white/50">
              <span>Penalty</span>
              <span className="font-mono">{config.confirmation_gate.penalty.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min={0.5}
              max={1.0}
              step={0.05}
              value={config.confirmation_gate.penalty}
              onChange={(e) =>
                onChange({
                  ...config,
                  confirmation_gate: {
                    ...config.confirmation_gate,
                    penalty: parseFloat(e.target.value),
                  },
                })
              }
              className="w-full h-1 bg-white/10 rounded-full appearance-none cursor-pointer
                [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3
                [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full
                [&::-webkit-slider-thumb]:bg-white/70 [&::-webkit-slider-thumb]:cursor-pointer"
            />
          </div>
        )}
      </div>

      {/* KOL & Freshness */}
      <div className="space-y-3 pt-2 border-t border-white/10">
        <h3 className="text-sm font-semibold text-white/90 uppercase tracking-wider">
          KOL &amp; Freshness
        </h3>

        {/* Conviction Dampening */}
        <button
          onClick={() =>
            onChange({
              ...config,
              kol_tuning: {
                ...config.kol_tuning,
                conviction_dampening: {
                  ...config.kol_tuning.conviction_dampening,
                  enabled: !config.kol_tuning.conviction_dampening.enabled,
                },
              },
            })
          }
          className={cn(
            "flex items-center justify-between w-full px-3 py-2 rounded-lg text-sm transition-all border",
            config.kol_tuning.conviction_dampening.enabled
              ? "border-white/20 bg-white/5 text-white"
              : "border-white/5 bg-transparent text-white/30"
          )}
        >
          <span>Conviction Dampening</span>
          <span
            className={cn(
              "text-[10px] font-mono px-1.5 py-0.5 rounded",
              config.kol_tuning.conviction_dampening.enabled
                ? "bg-green-500/20 text-green-400"
                : "bg-red-500/10 text-red-400/50"
            )}
          >
            {config.kol_tuning.conviction_dampening.enabled ? "ON" : "OFF"}
          </span>
        </button>
        {config.kol_tuning.conviction_dampening.enabled && (
          <div className="space-y-1 pl-3 border-l border-white/10">
            <div className="flex justify-between text-xs text-white/50">
              <span>Min KOLs for full conviction</span>
              <span className="font-mono">{config.kol_tuning.conviction_dampening.min_kols}</span>
            </div>
            <input
              type="range"
              min={1}
              max={5}
              step={1}
              value={config.kol_tuning.conviction_dampening.min_kols}
              onChange={(e) =>
                onChange({
                  ...config,
                  kol_tuning: {
                    ...config.kol_tuning,
                    conviction_dampening: {
                      ...config.kol_tuning.conviction_dampening,
                      min_kols: parseInt(e.target.value),
                    },
                  },
                })
              }
              className="w-full h-1 bg-white/10 rounded-full appearance-none cursor-pointer
                [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3
                [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full
                [&::-webkit-slider-thumb]:bg-white/70 [&::-webkit-slider-thumb]:cursor-pointer"
            />
          </div>
        )}

        {/* S-tier Bonus */}
        <button
          onClick={() =>
            onChange({
              ...config,
              kol_tuning: {
                ...config.kol_tuning,
                s_tier_bonus: {
                  ...config.kol_tuning.s_tier_bonus,
                  enabled: !config.kol_tuning.s_tier_bonus.enabled,
                },
              },
            })
          }
          className={cn(
            "flex items-center justify-between w-full px-3 py-2 rounded-lg text-sm transition-all border",
            config.kol_tuning.s_tier_bonus.enabled
              ? "border-white/20 bg-white/5 text-white"
              : "border-white/5 bg-transparent text-white/30"
          )}
        >
          <span>S-tier Bonus</span>
          <span
            className={cn(
              "text-[10px] font-mono px-1.5 py-0.5 rounded",
              config.kol_tuning.s_tier_bonus.enabled
                ? "bg-green-500/20 text-green-400"
                : "bg-red-500/10 text-red-400/50"
            )}
          >
            {config.kol_tuning.s_tier_bonus.enabled ? "ON" : "OFF"}
          </span>
        </button>
        {config.kol_tuning.s_tier_bonus.enabled && (
          <div className="space-y-1 pl-3 border-l border-white/10">
            <div className="flex justify-between text-xs text-white/50">
              <span>Bonus multiplier</span>
              <span className="font-mono">{config.kol_tuning.s_tier_bonus.bonus.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min={1.0}
              max={2.0}
              step={0.05}
              value={config.kol_tuning.s_tier_bonus.bonus}
              onChange={(e) =>
                onChange({
                  ...config,
                  kol_tuning: {
                    ...config.kol_tuning,
                    s_tier_bonus: {
                      ...config.kol_tuning.s_tier_bonus,
                      bonus: parseFloat(e.target.value),
                    },
                  },
                })
              }
              className="w-full h-1 bg-white/10 rounded-full appearance-none cursor-pointer
                [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3
                [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full
                [&::-webkit-slider-thumb]:bg-white/70 [&::-webkit-slider-thumb]:cursor-pointer"
            />
          </div>
        )}

        {/* Freshness Cutoff */}
        <button
          onClick={() =>
            onChange({
              ...config,
              kol_tuning: {
                ...config.kol_tuning,
                freshness_cutoff: {
                  ...config.kol_tuning.freshness_cutoff,
                  enabled: !config.kol_tuning.freshness_cutoff.enabled,
                },
              },
            })
          }
          className={cn(
            "flex items-center justify-between w-full px-3 py-2 rounded-lg text-sm transition-all border",
            config.kol_tuning.freshness_cutoff.enabled
              ? "border-white/20 bg-white/5 text-white"
              : "border-white/5 bg-transparent text-white/30"
          )}
        >
          <span>Freshness Cutoff</span>
          <span
            className={cn(
              "text-[10px] font-mono px-1.5 py-0.5 rounded",
              config.kol_tuning.freshness_cutoff.enabled
                ? "bg-green-500/20 text-green-400"
                : "bg-red-500/10 text-red-400/50"
            )}
          >
            {config.kol_tuning.freshness_cutoff.enabled ? "ON" : "OFF"}
          </span>
        </button>
        {config.kol_tuning.freshness_cutoff.enabled && (
          <div className="space-y-2 pl-3 border-l border-white/10">
            <div className="flex justify-between text-xs text-white/50">
              <span>Max hours</span>
              <span className="font-mono">{config.kol_tuning.freshness_cutoff.max_hours}h</span>
            </div>
            <input
              type="range"
              min={6}
              max={72}
              step={1}
              value={config.kol_tuning.freshness_cutoff.max_hours}
              onChange={(e) =>
                onChange({
                  ...config,
                  kol_tuning: {
                    ...config.kol_tuning,
                    freshness_cutoff: {
                      ...config.kol_tuning.freshness_cutoff,
                      max_hours: parseInt(e.target.value),
                    },
                  },
                })
              }
              className="w-full h-1 bg-white/10 rounded-full appearance-none cursor-pointer
                [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3
                [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full
                [&::-webkit-slider-thumb]:bg-white/70 [&::-webkit-slider-thumb]:cursor-pointer"
            />
            <div className="flex justify-between text-xs text-white/50">
              <span>Penalty</span>
              <span className="font-mono">{config.kol_tuning.freshness_cutoff.penalty.toFixed(2)}x</span>
            </div>
            <input
              type="range"
              min={0.1}
              max={1.0}
              step={0.05}
              value={config.kol_tuning.freshness_cutoff.penalty}
              onChange={(e) =>
                onChange({
                  ...config,
                  kol_tuning: {
                    ...config.kol_tuning,
                    freshness_cutoff: {
                      ...config.kol_tuning.freshness_cutoff,
                      penalty: parseFloat(e.target.value),
                    },
                  },
                })
              }
              className="w-full h-1 bg-white/10 rounded-full appearance-none cursor-pointer
                [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3
                [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full
                [&::-webkit-slider-thumb]:bg-white/70 [&::-webkit-slider-thumb]:cursor-pointer"
            />
          </div>
        )}
      </div>
    </div>
  );
}
