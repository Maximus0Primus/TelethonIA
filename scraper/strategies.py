"""
Centralized strategy definitions for the TelethonIA trading system.

Single source of truth for:
- STRATEGIES dict (all strategy definitions with tranches)
- SHADOW_STRATEGIES list
- Strategy name regex parsers
- Trail/decay config extraction
- LAZY check mode settings
- Fee constants
- sim_cfg_to_fake_trade() converter for unified simulation

Imported by: paper_trader.py, sim.py, sim_engines.py, live_trader.py, etc.
"""

import re

# ---------------------------------------------------------------------------
# Fee constants — Solana (v121: Jupiter Ultra RFQ — near-zero slippage)
# ---------------------------------------------------------------------------
BUY_SLIPPAGE_BPS = 10    # 0.1% — Jupiter Ultra platform fee
SELL_SLIPPAGE_BPS = 10   # 0.1% — Jupiter Ultra platform fee
BUY_FEE_BPS = 0          # 0% — folded into slippage
SELL_FEE_BPS = 0          # 0% — folded into slippage

# ---------------------------------------------------------------------------
# Fee constants — Ethereum L1 (v14: shadow-only cost model)
#
# Budget assumes mainnet ETH at ~20 gwei baseline. Single Uniswap V3 swap
# = ~150k gas = ~$7-8 at $2500/ETH. Round-trip (buy + sell) = $15.
# MEV sandwich attacks commonly take 100-300 bps on memecoin swaps; we
# budget 200 bps per side to reflect realistic post-Flashbots conditions.
# If we move to private relay for live (Phase 3), drop MEV to ~50 bps.
# ---------------------------------------------------------------------------
ETH_GAS_COST_USD_PER_SIDE = 7.50   # $7.50 each side, $15 round-trip
ETH_BUY_SLIPPAGE_BPS = 200         # 2% slippage on entry (MEV + pool impact)
ETH_SELL_SLIPPAGE_BPS = 200        # 2% slippage on exit
ETH_MIN_POSITION_USD = 200         # below $200, fees eat >7.5% of trade


# ---------------------------------------------------------------------------
# Default deprecated strategies — overridable via scoring_config JSONB
# ---------------------------------------------------------------------------
_DEFAULT_DEPRECATED = {
    "MOONBAG", "WIDE_RUNNER", "SCALE_OUT", "TP100_SL30",
    "QUICK_SCALP", "TP30_SL10", "TP50_SL15", "TP30_SL30",
}

# ---------------------------------------------------------------------------
# Shadow strategies list (single-tranche strategies eligible for $0 shadows)
# ---------------------------------------------------------------------------
SHADOW_STRATEGIES = [
    "TP30_SL50", "TP50_SL30", "TP100_SL30", "TP100_SL50",
    "TP50_SL15", "TP30_SL30", "TP50_SL50", "FRESH_MICRO", "QUICK_SCALP",
    "TP30_SL10",
]

# ---------------------------------------------------------------------------
# Strategy Definitions
# ---------------------------------------------------------------------------
# Each strategy has a list of tranches. Moonbag tranches have tp_mult=None.
STRATEGIES = {
    "TP30_SL50": [
        {"pct": 1.0, "tp_mult": 1.30, "sl_mult": 0.50, "horizon_min": 120, "label": "main"},
    ],
    "TP30_SL10": [
        {"pct": 1.0, "tp_mult": 1.30, "sl_mult": 0.90, "horizon_min": 120, "label": "main"},
    ],
    "TP50_SL30": [
        {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.70, "horizon_min": 120, "label": "main"},
    ],
    "TP100_SL30": [
        {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.70, "horizon_min": 120, "label": "main"},
    ],
    "TP100_SL50": [
        {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.50, "horizon_min": 120, "label": "main"},
    ],
    "SCALE_OUT": [
        {"pct": 0.25, "tp_mult": 2.00, "sl_mult": 0.50, "horizon_min": 120, "label": "tp_2x"},
        {"pct": 0.25, "tp_mult": 3.00, "sl_mult": 0.50, "horizon_min": 120, "label": "tp_3x"},
        {"pct": 0.25, "tp_mult": 5.00, "sl_mult": 0.50, "horizon_min": 120, "label": "tp_5x"},
        {"pct": 0.25, "tp_mult": None, "sl_mult": 0.50, "horizon_min": 120, "label": "moonbag"},
    ],
    "MOONBAG": [
        {"pct": 0.80, "tp_mult": 2.00, "sl_mult": 0.30, "horizon_min": 120, "label": "main"},
        {"pct": 0.20, "tp_mult": None, "sl_mult": 0.30, "horizon_min": 120, "label": "moonbag"},
    ],
    "TP50_SL15": [
        {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.85, "horizon_min": 120, "label": "main"},
    ],
    "TP30_SL30": [
        {"pct": 1.0, "tp_mult": 1.30, "sl_mult": 0.70, "horizon_min": 120, "label": "main"},
    ],
    "TP50_SL50": [
        {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.50, "horizon_min": 120, "label": "main"},
    ],
    "FRESH_MICRO": [
        {"pct": 1.0, "tp_mult": 1.30, "sl_mult": 0.70, "horizon_min": 120, "label": "main"},
    ],
    "QUICK_SCALP": [
        {"pct": 1.0, "tp_mult": 1.30, "sl_mult": 0.85, "horizon_min": 120, "label": "main"},
    ],
    "WIDE_RUNNER": [
        {"pct": 0.60, "tp_mult": 2.00, "sl_mult": 0.30, "horizon_min": 120, "label": "main"},
        {"pct": 0.40, "tp_mult": 3.00, "sl_mult": 0.30, "horizon_min": 120, "label": "runner"},
    ],
}

# --- Strategy Entry Filters ---
STRATEGY_FILTERS = {
    "FRESH_MICRO": {
        "min_score": 10, "max_score": 49,
        "min_kol_freshness": 0.01, "min_momentum_mult": 1.0,
        "max_mcap": 5_000_000,
    },
    "QUICK_SCALP": {
        "min_score": 10, "max_score": 49,
        "min_momentum_mult": 1.0,
    },
    "WIDE_RUNNER": {
        "min_score": 10, "max_score": 49,
        "min_kol_freshness": 0.01, "max_mcap": 5_000_000,
    },
    # v139: skip pump.fun pre-graduation tokens (liq=0 in DexScreener) —
    # measured -19.82$/jour drag on baseline. Filter cuts 38% of universe
    # but boosts avg PnL by +9pp on TP200_SL40.
    "NOZEROLIQ_TP200_SL40": {
        "min_liquidity_usd": 1.0,  # any non-zero liquidity
    },
    # v139: gate on rt_score >= 30. Tested 50% WR vs 41% baseline,
    # avg +14.42% vs +5.88%. Score signal IS predictive at 30+ threshold.
    "HIGHSCORE_TP200_SL40": {
        "min_rt_score": 30,
    },
    # v140 filtered variants
    "BE25_TP80_SL30_S30_HYST": {"min_rt_score": 30},
    "BE15_TP70_SL50_NZ": {"min_liquidity_usd": 1.0},
    "BE25_TP80_SL30_NZS30_HYST": {"min_liquidity_usd": 1.0, "min_rt_score": 30},
    "BE15_TP300_SL50_MCAP": {"min_mcap": 30_000, "max_mcap": 500_000},
    # v144 — isolate SCORE filter alpha from HYST noise (S5 audit: SCORE>=40
    # on BE25 retroactive = N=13, WR 62%, avg +34% on rt_score v141).
    "BE25_TP80_SL30_S30": {"min_rt_score": 30},
    "BE25_TP80_SL30_S40": {"min_rt_score": 40},
    # v144 — SCORE>=40 on FAST_TP80_SL25 retroactive = N=16, WR 68.8%, avg +29.76%
    "FAST_TP80_SL25_S40": {"min_rt_score": 40},
    # v144 — Sim mega sweep v142 SCORE40 top: FAST_TP50_SL30 (WR 72.2%, avg +34.5%)
    # and FAST_TP100_SL20 (WR 66.7%, avg +32.0%) — complete the SCORE40 family.
    "FAST_TP50_SL30_S40": {"min_rt_score": 40},
    "FAST_TP100_SL20_S40": {"min_rt_score": 40},
    # v144 — MCAP_MID_SCORE40 combo on FAST_TP50_SL30 — sim extended top WR 74%
    "FAST_TP50_SL30_MCAP_S40": {
        "min_mcap": 30_000, "max_mcap": 500_000, "min_rt_score": 40,
    },
}

# --- Grid strategies (v93) ---
_GRID_STRATEGIES = {}
for _tp in range(40, 110, 10):
    for _sl in range(30, 80, 10):
        _name = f"TP{_tp}_SL{_sl}"
        if _name not in STRATEGIES:
            _GRID_STRATEGIES[_name] = [
                {"pct": 1.0, "tp_mult": 1 + _tp / 100, "sl_mult": 1 - _sl / 100,
                 "horizon_min": 120, "label": "main"},
            ]
for _tp_nosl in [50, 60, 70, 80, 90, 100]:
    _GRID_STRATEGIES[f"TP{_tp_nosl}_NOSL"] = [
        {"pct": 1.0, "tp_mult": 1 + _tp_nosl / 100, "sl_mult": 0.20,
         "horizon_min": 120, "label": "main"},
    ]
STRATEGIES.update(_GRID_STRATEGIES)
SHADOW_STRATEGIES.extend(_GRID_STRATEGIES.keys())

# --- Scalp grid (v105) ---
_SCALP_STRATEGIES = {}
for _tp_s in [10, 15, 20]:
    for _sl_s in [10, 15, 20, 30]:
        if _sl_s > _tp_s * 2:
            continue
        _sname = f"SCALP_TP{_tp_s}_SL{_sl_s}"
        _SCALP_STRATEGIES[_sname] = [
            {"pct": 1.0, "tp_mult": 1 + _tp_s / 100, "sl_mult": 1 - _sl_s / 100,
             "horizon_min": 120, "label": "main"},
        ]
    _SCALP_STRATEGIES[f"SCALP_TP{_tp_s}_NOSL"] = [
        {"pct": 1.0, "tp_mult": 1 + _tp_s / 100, "sl_mult": 0.20,
         "horizon_min": 120, "label": "main"},
    ]
STRATEGIES.update(_SCALP_STRATEGIES)
SHADOW_STRATEGIES.extend(_SCALP_STRATEGIES.keys())

# --- Decay grid (v106) ---
_DECAY_STRATEGIES = {}
for _tp_d, _sl_d, _end_d in [
    (100, 50, 20), (100, 50, 15), (100, 50, 30),
    (70, 50, 15), (50, 30, 15), (50, 50, 15),
]:
    _dname = f"DECAY_TP{_tp_d}_SL{_sl_d}_E{_end_d}"
    _DECAY_STRATEGIES[_dname] = [
        {"pct": 1.0, "tp_mult": 1 + _tp_d / 100,
         "sl_mult": 1 - _sl_d / 100, "horizon_min": 120,
         "tp_decay_end": 1 + _end_d / 100, "label": "main"},
    ]
STRATEGIES.update(_DECAY_STRATEGIES)
SHADOW_STRATEGIES.extend(_DECAY_STRATEGIES.keys())

# --- Fast timeout grids (v106) ---
# v134: + (80, 25) and (100, 20) — candidates from synthetic-sweep robustness test
_FAST_STRATEGIES = {}
for _tp_f, _sl_f in [(100, 50), (50, 30), (50, 50), (70, 50), (40, 30), (80, 25), (100, 20)]:
    _fname = f"FAST_TP{_tp_f}_SL{_sl_f}"
    _FAST_STRATEGIES[_fname] = [
        {"pct": 1.0, "tp_mult": 1 + _tp_f / 100,
         "sl_mult": 1 - _sl_f / 100, "horizon_min": 30, "label": "main"},
    ]
STRATEGIES.update(_FAST_STRATEGIES)
SHADOW_STRATEGIES.extend(_FAST_STRATEGIES.keys())

_FAST60_STRATEGIES = {}
for _tp_f6, _sl_f6 in [(100, 50), (50, 30), (50, 50), (70, 50), (40, 30)]:
    _f6name = f"FAST60_TP{_tp_f6}_SL{_sl_f6}"
    _FAST60_STRATEGIES[_f6name] = [
        {"pct": 1.0, "tp_mult": 1 + _tp_f6 / 100,
         "sl_mult": 1 - _sl_f6 / 100, "horizon_min": 60, "label": "main"},
    ]
STRATEGIES.update(_FAST60_STRATEGIES)
SHADOW_STRATEGIES.extend(_FAST60_STRATEGIES.keys())

_FAST45_STRATEGIES = {}
for _tp_f45, _sl_f45 in [(100, 50), (50, 30), (50, 50), (70, 50), (40, 30)]:
    _f45name = f"FAST45_TP{_tp_f45}_SL{_sl_f45}"
    _FAST45_STRATEGIES[_f45name] = [
        {"pct": 1.0, "tp_mult": 1 + _tp_f45 / 100,
         "sl_mult": 1 - _sl_f45 / 100, "horizon_min": 45, "label": "main"},
    ]
STRATEGIES.update(_FAST45_STRATEGIES)
SHADOW_STRATEGIES.extend(_FAST45_STRATEGIES.keys())

# --- Slow timeout grids (v107) ---
_SLOW4H_STRATEGIES = {}
for _tp_s4, _sl_s4 in [(100, 50), (50, 30), (50, 50), (70, 50), (40, 30)]:
    _s4name = f"SLOW4H_TP{_tp_s4}_SL{_sl_s4}"
    _SLOW4H_STRATEGIES[_s4name] = [
        {"pct": 1.0, "tp_mult": 1 + _tp_s4 / 100,
         "sl_mult": 1 - _sl_s4 / 100, "horizon_min": 240, "label": "main"},
    ]
STRATEGIES.update(_SLOW4H_STRATEGIES)
SHADOW_STRATEGIES.extend(_SLOW4H_STRATEGIES.keys())

_SLOW6H_STRATEGIES = {}
for _tp_s6, _sl_s6 in [(100, 50), (50, 30), (50, 50), (70, 50), (40, 30)]:
    _s6name = f"SLOW6H_TP{_tp_s6}_SL{_sl_s6}"
    _SLOW6H_STRATEGIES[_s6name] = [
        {"pct": 1.0, "tp_mult": 1 + _tp_s6 / 100,
         "sl_mult": 1 - _sl_s6 / 100, "horizon_min": 360, "label": "main"},
    ]
STRATEGIES.update(_SLOW6H_STRATEGIES)
SHADOW_STRATEGIES.extend(_SLOW6H_STRATEGIES.keys())

# --- Breakeven stop grid (v106) ---
_BE_STRATEGIES = {}
for _be_act in [15, 20, 30]:
    for _tp_be, _sl_be in [(100, 50), (50, 30), (50, 50), (70, 50)]:
        _bename = f"BE{_be_act}_TP{_tp_be}_SL{_sl_be}"
        _BE_STRATEGIES[_bename] = [
            {"pct": 1.0, "tp_mult": 1 + _tp_be / 100,
             "sl_mult": 1 - _sl_be / 100, "horizon_min": 120,
             "be_activation": _be_act / 100, "label": "main"},
        ]
STRATEGIES.update(_BE_STRATEGIES)
SHADOW_STRATEGIES.extend(_BE_STRATEGIES.keys())

# v134: FAST-horizon BE variant — candidate from synthetic-sweep (#2 post-v132)
# BE25 activates breakeven once peak ≥ entry*1.25, then SL moves to entry.
STRATEGIES["BE25_TP80_SL30"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "label": "main"},
]
SHADOW_STRATEGIES.append("BE25_TP80_SL30")

# v136: BE25 A/B variant — identical exit rules, different price_source config.
# BE25_TP80_SL30     → ema_fast (original, +$217 accumulated pnl on $1000 baseline)
# BE25_TP80_SL30_DS  → ds/raw   (v135 sweep winner, Kelly 22% vs ~19%)
# Both run in parallel to resolve the config preference empirically.
STRATEGIES["BE25_TP80_SL30_DS"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "label": "main"},
]
SHADOW_STRATEGIES.append("BE25_TP80_SL30_DS")

# v139: Asymmetric payoff bets — TP200 (3x) with 4h horizon, gated by quality filters.
# Tested on 71 post-v132 tokens: NOZEROLIQ +14.91%/48% WR, HIGHSCORE +14.42%/50% WR
# vs BASELINE +5.88%/41% WR. Skip toxic flow (liq=0) + use score signal = clear edge.
STRATEGIES["TP200_SL40_4H"] = [
    {"pct": 1.0, "tp_mult": 3.00, "sl_mult": 0.60, "horizon_min": 240, "label": "main"},
]
SHADOW_STRATEGIES.append("TP200_SL40_4H")

# Filtered variants — same exit but different entry gates
STRATEGIES["NOZEROLIQ_TP200_SL40"] = [
    {"pct": 1.0, "tp_mult": 3.00, "sl_mult": 0.60, "horizon_min": 240, "label": "main"},
]
SHADOW_STRATEGIES.append("NOZEROLIQ_TP200_SL40")

STRATEGIES["HIGHSCORE_TP200_SL40"] = [
    {"pct": 1.0, "tp_mult": 3.00, "sl_mult": 0.60, "horizon_min": 240, "label": "main"},
]
SHADOW_STRATEGIES.append("HIGHSCORE_TP200_SL40")

# ============================================================
# v140 — full mega sweep winners (hysteresis + lazy dominates)
# A/B test variants vs existing strats, each gets $1000 bankroll
# ============================================================
# v142 (Apr 18): _HYST variants of non-filtered strats produce IDENTICAL results
# to their vanilla counterparts in sim.py mega sweep (same tp_mult/sl_mult; the
# "HYST" naming carried a smoothing hint that is orchestrated separately). Kept
# in STRATEGIES so any in-flight open trades can close cleanly, but removed
# from SHADOW_STRATEGIES so no NEW duplicate shadows open. Filtered _HYST
# variants (S30_HYST, NZS30_HYST) differ via entry filter and stay active.
STRATEGIES["FAST_TP100_SL20_HYST"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.80, "horizon_min": 30, "label": "main"},
]
# SHADOW_STRATEGIES.append("FAST_TP100_SL20_HYST")  # v142: redundant, dropped

STRATEGIES["FAST_TP80_SL25_HYST"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.75, "horizon_min": 30, "label": "main"},
]
# SHADOW_STRATEGIES.append("FAST_TP80_SL25_HYST")  # v142: redundant, dropped

STRATEGIES["BE25_TP80_SL30_HYST"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "label": "main"},
]
# SHADOW_STRATEGIES.append("BE25_TP80_SL30_HYST")  # v142: redundant, dropped

STRATEGIES["FAST_TP50_SL30_HYST"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.70, "horizon_min": 30, "label": "main"},
]
# SHADOW_STRATEGIES.append("FAST_TP50_SL30_HYST")  # v142: redundant, dropped

# Best per filter — each combo tests different (filter, config) point
STRATEGIES["BE25_TP80_SL30_S30_HYST"] = [  # SCORE30 filter + hysteresis/static_240
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "label": "main"},
]
SHADOW_STRATEGIES.append("BE25_TP80_SL30_S30_HYST")

STRATEGIES["BE15_TP70_SL50_NZ"] = [  # NOZEROLIQ filter + jupiter/raw/static_240
    {"pct": 1.0, "tp_mult": 1.70, "sl_mult": 0.50, "horizon_min": 120,
     "be_activation": 0.15, "label": "main"},
]
SHADOW_STRATEGIES.append("BE15_TP70_SL50_NZ")

STRATEGIES["BE25_TP80_SL30_NZS30_HYST"] = [  # NOZEROLIQ+SCORE30 + hysteresis/static_240
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "label": "main"},
]
SHADOW_STRATEGIES.append("BE25_TP80_SL30_NZS30_HYST")

STRATEGIES["BE15_TP300_SL50_MCAP"] = [  # MCAP_MID filter + ds/raw/fast
    {"pct": 1.0, "tp_mult": 4.00, "sl_mult": 0.50, "horizon_min": 240,
     "be_activation": 0.15, "label": "main"},
]
SHADOW_STRATEGIES.append("BE15_TP300_SL50_MCAP")

# ============================================================
# v142 — Data-driven candidates (NOT active, shadow/sim only)
#
# Observation (7d, 881 trades): FAST+tight-SL family wins (+$405), HYST all lose
# (−$284), filters (NZ/HS/MCAP) cut the rentable segment (bondings +$439/4h).
# Four hypotheses to validate before shipping:
#   1. Bondings are the alpha — gate FAST on low-liq/bonding only
#   2. Memecoin pumps fade fast — decay TP + lock breakeven early in horizon
#   3. Winners need breathing room — tier-scaled trail (tight on small peaks,
#      loose on big peaks) beats fixed DTRAIL
#   4. Entry price validation — 60s confirm before open filters dead calls
# ============================================================

# Sweep-tuned configs (sim_sweep.py, 14d post-v138 eval_history, 4-fold
# walk-forward, post-haircut per-exit-type slippage). Ex-ante expectation
# (14d, same trade universe where paper actual = −$455):
#   - TD2  (fine grid 14400 configs, haircut ON): +$468 → delta +$923 vs paper
#   - PTRAIL_V2 (coarse 432, haircut ON):          +$161 → delta +$616
#   - BOND_FAST (coarse 864, haircut ON):          +$56  → delta +$511
# Stability 3/4 folds positive on TD2 + PTRAIL; 2/4 on BOND (brittle to slip).
# All 3 enter as SHADOW (in SHADOW_STRATEGIES but NOT in active_strategies) —
# observe N≥20 actual trades before main-bankroll activation.

# 1. TD2_BE5_TP120_SL44_T25 — TIME_DECAY_V2 fine winner.
#    TP schedule: +120% at t=0, decays to +40% at t=5min, then +0% (late = take
#    any profit). BE moves SL to entry at t=5min regardless of peak.
STRATEGIES["TD2_BE5_TP120_SL44_T25"] = [
    {"pct": 1.0, "tp_mult": None,  # TP handled by tp_schedule, not scalar
     "sl_mult": 0.55, "horizon_min": 25,
     "time_be_minute": 5,
     "tp_schedule": [(0, 2.20), (5, 1.40), (15, 1.00), (25, 1.00)],
     "label": "main"},
]
SHADOW_STRATEGIES.append("TD2_BE5_TP120_SL44_T25")

# 2. PTRAIL_V2_T10-18-30-45_SL30_T60 — tiered trail winner.
STRATEGIES["PTRAIL_V2_T10-18-30-45_SL30_T60"] = [
    {"pct": 1.0, "tp_mult": None, "sl_mult": 0.70, "horizon_min": 60,
     "trail_tiers": [(1.30, 0.10), (1.80, 0.18), (3.00, 0.30), (6.00, 0.45)],
     "trail_activation_pct": 0.15, "label": "main"},
]
SHADOW_STRATEGIES.append("PTRAIL_V2_T10-18-30-45_SL30_T60")

# 3. BOND_FAST_TP50_SL20_T20 — bonding/low-liq isolated, tight stops.
STRATEGIES["BOND_FAST_TP50_SL20_T20"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.80, "horizon_min": 20,
     "trail_pct": 0.10, "trail_activation_pct": 0.15, "label": "main"},
]
SHADOW_STRATEGIES.append("BOND_FAST_TP50_SL20_T20")

# Entry filter for BOND_FAST — only bondings / ultra-low liq.
# v14: `chain: solana` gate ADDED. Without it, an ETH token with liq <$3k
# would match this filter and get traded with Solana fee model (10bps slip).
# ETH fees are $15 round-trip + 2% MEV, so a $3k-liq ETH token is far too
# small to profitably trade. BOND_FAST is Solana-pump.fun-specific by design.
STRATEGY_FILTERS["BOND_FAST_TP50_SL20_T20"] = {
    "chain": "solana",
    "max_liquidity_usd": 3_000,
}
# (TD2 and PTRAIL_V2 have no entry gate — all tokens eligible)

# MOMENTUM_CONFIRM_ENTRY is not a strategy — sim falsified it (paired delta
# −24% vs baseline). Hypothèse morte. Do not ship.

# ============================================================
# v142 (Apr 18, cont.) — Diversity pack, shadow-only (bankroll $0)
#
# 6 new strategies covering gaps in the current lineup, derived from mega
# sweep signals. Each targets a distinct axis of the parameter space.
# Shadow-only: observability without bankroll risk. Promote individually
# after N>=15 real shadow trades confirm sim expectations.
# ============================================================

# 1) SCORE40 ultra-selective — mega sweep best per-trade alpha (+34.5% N=18)
STRATEGIES["SCORE40_FAST_TP50_SL30_30M"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.70, "horizon_min": 30, "label": "main"},
]
SHADOW_STRATEGIES.append("SCORE40_FAST_TP50_SL30_30M")
STRATEGY_FILTERS["SCORE40_FAST_TP50_SL30_30M"] = {"min_rt_score": 40}

# 2) MCAP_MID + DTRAIL — mega sweep MCAP filter winner ($81/j, +19.35% N=37)
STRATEGIES["MCAP_MID_DTRAIL5_ACT25_SL50_2H"] = [
    {"pct": 1.0, "tp_mult": None, "sl_mult": 0.50, "horizon_min": 120,
     "trail_pct": 0.05, "trail_activation_pct": 0.25, "label": "main"},
]
SHADOW_STRATEGIES.append("MCAP_MID_DTRAIL5_ACT25_SL50_2H")
STRATEGY_FILTERS["MCAP_MID_DTRAIL5_ACT25_SL50_2H"] = {
    "min_mcap": 30_000, "max_mcap": 500_000,
}

# 3) Moonshot at 60min horizon — gap between FAST (30min) and SLOW (4h)
STRATEGIES["FAST_TP200_SL40_60M"] = [
    {"pct": 1.0, "tp_mult": 3.00, "sl_mult": 0.60, "horizon_min": 60, "label": "main"},
]
SHADOW_STRATEGIES.append("FAST_TP200_SL40_60M")

# 4) DIP variant with stricter bounce threshold — vs existing B5 bounces
# that trigger on false dips. Fits DIP_RE pattern so _get_trail_config
# picks up trail/act correctly.
STRATEGIES["DIP30_B10_T10_A20_SL60_120m"] = [
    {"pct": 0.5, "tp_mult": None, "sl_mult": 0.40, "horizon_min": 120,
     "trail_pct": 0.10, "trail_activation_pct": 0.20, "label": "dip_p1"},
]
SHADOW_STRATEGIES.append("DIP30_B10_T10_A20_SL60_120m")

# 5) BE on medium horizon — current BE suite is all 30min; add 2h to catch
# whale-sized tokens that take 30-60min to develop
STRATEGIES["BE15_TP150_SL40_2H"] = [
    {"pct": 1.0, "tp_mult": 2.50, "sl_mult": 0.60, "horizon_min": 120,
     "be_activation": 0.15, "label": "main"},
]
SHADOW_STRATEGIES.append("BE15_TP150_SL40_2H")

# 6) Pure moonshot tail-captor — TP500 (x6), wide SL, short horizon.
# Negative EV per trade expected but captures the rare 5x+
STRATEGIES["FAST_TP500_SL40_60M"] = [
    {"pct": 1.0, "tp_mult": 6.00, "sl_mult": 0.60, "horizon_min": 60, "label": "main"},
]
SHADOW_STRATEGIES.append("FAST_TP500_SL40_60M")

# v142 — mega sweep pick: BE long-horizon moonshot. Top config (NONE + DS +
# hysteresis + static_60) projects $86/day at $50 pos. Promoted to active
# main paper with $1000 bankroll.
STRATEGIES["BE15_TP200_SL40_4H"] = [
    {"pct": 1.0, "tp_mult": 3.00, "sl_mult": 0.60, "horizon_min": 240,
     "be_activation": 0.15, "label": "main"},
]
SHADOW_STRATEGIES.append("BE15_TP200_SL40_4H")

# ============================================================
# v144.10 — EH A/B hidden gems (Apr 21)
# The eval_history mega sweep (Spearman rho=0.058 vs price_ticks) flagged a
# "let-it-run" cluster at rank 44-128 with TP150-200 + tight SL + 2-4h horizons
# that the price_ticks sweep had missed. Added as shadows — N=42-47 per config
# in EH, needs paper paired confirmation. Plus 3 existing STRATEGIES not yet
# in SHADOW (MOONBAG, WIDE_RUNNER, SCALE_OUT — WR 60.9%, med +8.58% on SCORE30
# subset in EH).
# ============================================================

# Tier 1 — create new TP200/TP150 variants flagged by EH
STRATEGIES["BE25_TP200_SL40_4H"] = [  # EH rank 46, dpd $129.5
    {"pct": 1.0, "tp_mult": 3.00, "sl_mult": 0.60, "horizon_min": 240,
     "be_activation": 0.25, "label": "main"},
]
SHADOW_STRATEGIES.append("BE25_TP200_SL40_4H")

STRATEGIES["TP200_SL30_2H"] = [  # EH rank 47, dpd $129.5
    {"pct": 1.0, "tp_mult": 3.00, "sl_mult": 0.70, "horizon_min": 120, "label": "main"},
]
SHADOW_STRATEGIES.append("TP200_SL30_2H")

STRATEGIES["BE50_TP200_SL30_4H"] = [  # EH rank 48, dpd $128.2 — strong BE activation
    {"pct": 1.0, "tp_mult": 3.00, "sl_mult": 0.70, "horizon_min": 240,
     "be_activation": 0.50, "label": "main"},
]
SHADOW_STRATEGIES.append("BE50_TP200_SL30_4H")

STRATEGIES["TP200_SL30_4H"] = [  # EH rank 49, dpd $127.8
    {"pct": 1.0, "tp_mult": 3.00, "sl_mult": 0.70, "horizon_min": 240, "label": "main"},
]
SHADOW_STRATEGIES.append("TP200_SL30_4H")

STRATEGIES["TP200_SL40_2H"] = [  # EH rank 57, dpd $118.3 — 2H variant of TP200_SL40_4H
    {"pct": 1.0, "tp_mult": 3.00, "sl_mult": 0.60, "horizon_min": 120, "label": "main"},
]
SHADOW_STRATEGIES.append("TP200_SL40_2H")

STRATEGIES["TP200_SL50_4H"] = [  # EH rank 82, dpd $113.7 — SL50 (wider safety)
    {"pct": 1.0, "tp_mult": 3.00, "sl_mult": 0.50, "horizon_min": 240, "label": "main"},
]
SHADOW_STRATEGIES.append("TP200_SL50_4H")

STRATEGIES["TP150_SL40_2H"] = [  # EH rank 113, dpd $107.8 — no-BE counterpart of BE15_TP150_SL40_2H
    {"pct": 1.0, "tp_mult": 2.50, "sl_mult": 0.60, "horizon_min": 120, "label": "main"},
]
SHADOW_STRATEGIES.append("TP150_SL40_2H")

# Tier 2 — promote existing STRATEGIES to SHADOW (let-it-run profile, WR 60.9%
# on SCORE30 in EH — paper will tell us if it holds without the filter).
SHADOW_STRATEGIES.append("MOONBAG")       # EH rank 246, dpd $91.1, med +8.58%
SHADOW_STRATEGIES.append("WIDE_RUNNER")   # EH rank 243, dpd $91.1, med +8.58%
SHADOW_STRATEGIES.append("SCALE_OUT")     # EH rank 232, dpd $94.5, med -0.13%

# --- Trailing stop grid (v106) ---
_TRAIL_STRATEGIES = {}
for _trail_pct in [10, 15, 20, 25]:
    for _tp_tr, _sl_tr in [(100, 50), (50, 30), (50, 50), (70, 50)]:
        _tname = f"TRAIL{_trail_pct}_TP{_tp_tr}_SL{_sl_tr}"
        _TRAIL_STRATEGIES[_tname] = [
            {"pct": 1.0, "tp_mult": 1 + _tp_tr / 100,
             "sl_mult": 1 - _sl_tr / 100, "horizon_min": 120,
             "trail_pct": _trail_pct / 100, "label": "main"},
        ]
STRATEGIES.update(_TRAIL_STRATEGIES)
SHADOW_STRATEGIES.extend(_TRAIL_STRATEGIES.keys())

# --- Dynamic trail grid (v110) ---
_DTRAIL_STRATEGIES = {}
for _dt_trail in [3, 5, 10, 15, 20]:
    for _dt_act in [5, 10, 15, 20, 25, 30]:
        for _dt_sl in [50, 60, 70]:
            _dtname = f"DTRAIL{_dt_trail}_ACT{_dt_act}_SL{_dt_sl}"
            _DTRAIL_STRATEGIES[_dtname] = [
                {"pct": 1.0, "tp_mult": None,
                 "sl_mult": 1 - _dt_sl / 100, "horizon_min": 120,
                 "trail_pct": _dt_trail / 100,
                 "trail_activation_pct": _dt_act / 100, "label": "main"},
            ]
STRATEGIES.update(_DTRAIL_STRATEGIES)
SHADOW_STRATEGIES.extend(_DTRAIL_STRATEGIES.keys())

# --- Split exit grid (v106) ---
_SPLIT_STRATEGIES = {}
for _sl_sp in [30, 50]:
    _SPLIT_STRATEGIES[f"SPLIT_50_100_SL{_sl_sp}"] = [
        {"pct": 0.5, "tp_mult": 1.50, "sl_mult": 1 - _sl_sp / 100,
         "horizon_min": 120, "label": "tp50_half"},
        {"pct": 0.5, "tp_mult": 2.00, "sl_mult": 1 - _sl_sp / 100,
         "horizon_min": 120, "label": "runner_half"},
    ]
    _SPLIT_STRATEGIES[f"SPLIT_50_TRAIL_SL{_sl_sp}"] = [
        {"pct": 0.5, "tp_mult": 1.50, "sl_mult": 1 - _sl_sp / 100,
         "horizon_min": 120, "label": "tp50_half"},
        {"pct": 0.5, "tp_mult": None, "sl_mult": 1 - _sl_sp / 100,
         "horizon_min": 120, "trail_pct": 0.20, "label": "trail_half"},
    ]
STRATEGIES.update(_SPLIT_STRATEGIES)
SHADOW_STRATEGIES.extend(_SPLIT_STRATEGIES.keys())

# --- DIP_BUY strategies (v115-v118) ---
STRATEGIES["DIP30_B5_T5_A15_SL70_240m"] = [
    {"pct": 0.5, "tp_mult": None, "sl_mult": 0.30,
     "horizon_min": 240, "trail_pct": 0.05,
     "trail_activation_pct": 0.15, "label": "dip_p1"},
]
STRATEGIES["DIP30_B5_P1T5A10S70_P2T10A15S60_240m"] = [
    {"pct": 0.5, "tp_mult": None, "sl_mult": 0.30,
     "horizon_min": 240, "trail_pct": 0.05,
     "trail_activation_pct": 0.10, "label": "dip_p1"},
]
STRATEGIES["DIP30_B5_T5_A20_SL70_240m"] = [
    {"pct": 0.5, "tp_mult": None, "sl_mult": 0.30,
     "horizon_min": 240, "trail_pct": 0.05,
     "trail_activation_pct": 0.20, "label": "dip_p1"},
]
STRATEGIES["DIP30_B5_T10_A30_SL60_240m"] = [
    {"pct": 0.5, "tp_mult": None, "sl_mult": 0.40,
     "horizon_min": 240, "trail_pct": 0.10,
     "trail_activation_pct": 0.30, "label": "dip_p1"},
]

# ---------------------------------------------------------------------------
# Strategy name regex parsers
# v125: Allow optional _\d+m timeout suffix for sim grid compatibility
# ---------------------------------------------------------------------------
_DECAY_RE = re.compile(r"^DECAY_TP\d+_SL\d+_E(\d+)(?:_\d+m)?$")
_TRAIL_RE = re.compile(r"^TRAIL(\d+)_TP\d+_SL\d+(?:_\d+m)?$")
_DTRAIL_RE = re.compile(r"^DTRAIL(\d+)_ACT(\d+)_SL\d+(?:_\d+m)?$")
_DIP_RE = re.compile(r"^DIP(\d+)_B(\d+)_T(\d+)_A(\d+)_SL(\d+)_(\d+)m$")
_DIP_SPLIT_RE = re.compile(
    r"^DIP(\d+)_B(\d+)_P1T(\d+)A(\d+)S(\d+)_P2T(\d+)A(\d+)S(\d+)_(\d+)m$"
)
_BE_RE = re.compile(r"^BE(\d+)_TP\d+_SL\d+")  # v140: no end anchor → accepts any suffix (_HYST, _NZ, _S30, etc.)

# Cache for _get_trail_config() to avoid regex per-tick in sim
_trail_config_cache: dict[str, tuple] = {}


def _get_decay_end(strategy_name: str) -> float | None:
    """Extract tp_decay_end multiplier from strategy name, or None."""
    m = _DECAY_RE.match(strategy_name)
    if m:
        return 1 + int(m.group(1)) / 100
    return None


def _get_trail_config(trade: dict) -> tuple[float | None, float | None]:
    """Get (trail_pct, activation_pct) from strategy name or tranche config.
    Returns (trail_pct, activation_pct) or (None, None).
    For DTRAIL strategies: activation_pct is the gain threshold before trail activates.
    For TRAIL strategies: activation_pct = trail_pct (legacy behavior).
    Uses cache to avoid regex on every tick during simulation."""
    strat = trade.get("strategy", "")
    label = trade.get("tranche_label", "")
    cache_key = f"{strat}|{label}"

    cached = _trail_config_cache.get(cache_key)
    if cached is not None:
        return cached

    result = _get_trail_config_uncached(strat, label)
    _trail_config_cache[cache_key] = result
    return result


# v142: tranche config lookup — used by eval extensions (time_be, tp_schedule,
# trail_tiers). Returns the specific tranche dict matching (strategy, label), or
# None. Gracefully falls back to label-less first tranche when only one exists.
def _find_tranche_config(strategy_name: str, tranche_label: str = "main") -> dict | None:
    tranches = STRATEGIES.get(strategy_name)
    if not tranches:
        return None
    for t in tranches:
        if t.get("label") == tranche_label:
            return t
    if len(tranches) == 1:
        return tranches[0]
    return None


def _get_trail_config_uncached(strat: str, label: str) -> tuple[float | None, float | None]:
    """Uncached trail config extraction."""
    # v116: DIP_BUY split — return P1 or P2 params based on tranche_label
    m = _DIP_SPLIT_RE.match(strat)
    if m:
        if label == "dip_p2":
            return int(m.group(6)) / 100, int(m.group(7)) / 100
        return int(m.group(3)) / 100, int(m.group(4)) / 100
    # v115: DIP_BUY shared
    m = _DIP_RE.match(strat)
    if m:
        return int(m.group(3)) / 100, int(m.group(4)) / 100
    # v110: Dynamic trail — DTRAIL{trail}_ACT{act}_SL{sl}
    m = _DTRAIL_RE.match(strat)
    if m:
        return int(m.group(1)) / 100, int(m.group(2)) / 100
    # Legacy: TRAIL{pct}_TP{tp}_SL{sl} — activation = trail_pct
    m = _TRAIL_RE.match(strat)
    if m:
        pct = int(m.group(1)) / 100
        return pct, pct
    # SPLIT strategy runner tranche
    if "trail" in label:
        return 0.20, 0.20
    return None, None


# ---------------------------------------------------------------------------
# LAZY check mode (v118)
# ---------------------------------------------------------------------------
LAZY_STRATEGIES: set[str] = {
    # v138.2: lazy dominates static on these.
    "FAST_TP100_SL20",
    "FAST_TP80_SL25",
    "FAST_TP50_SL30",
    "FAST_TP40_SL30",
    "TP50_SL15",
    # v140 filtered HYST variants kept (still in hybrid_strategy.allocations)
    "FAST_TP50_SL30_HYST",
    # v144.5: removed 4 entries that were retired from hybrid_strategy.allocations
    # by v144.1 cleanup (Apr 20): BE25_TP80_SL30_DS, FAST_TP100_SL20_HYST,
    # FAST_TP80_SL25_HYST, BE25_TP80_SL30_HYST. LAZY membership without hybrid
    # presence had no behavioral effect since shadow path now respects LAZY
    # (v144.3) but those names produced 0 shadow trades anyway.
    # Re-add if any of these are promoted back to hybrid.
}
LAZY_FAST_SEC = 180     # 3 min during fast phase
LAZY_FAST_WINDOW = 300  # 5 min fast phase
LAZY_SLOW_SEC = 600     # 10 min after fast phase


# ---------------------------------------------------------------------------
# v144 — LAZY vs non-LAZY A/B shadows. Top 4 real 7d earners ($275/$270/$242/$217
# all LAZY) vs identical TP/SL but NOT in LAZY_STRATEGIES. Shadows have
# position_usd=0 which forces CURRENT interval in _should_poll_trade, giving a
# clean paired control. Goal: paired N≥50 to confirm LAZY domination isn't an
# artifact of which strats got assigned LAZY.
# ---------------------------------------------------------------------------
STRATEGIES["FAST_TP40_SL30_NOLAZY"] = [
    {"pct": 1.0, "tp_mult": 1.40, "sl_mult": 0.70, "horizon_min": 30, "label": "main"},
]
SHADOW_STRATEGIES.append("FAST_TP40_SL30_NOLAZY")

STRATEGIES["FAST_TP80_SL25_NOLAZY"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.75, "horizon_min": 30, "label": "main"},
]
SHADOW_STRATEGIES.append("FAST_TP80_SL25_NOLAZY")

STRATEGIES["FAST_TP50_SL30_NOLAZY"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.70, "horizon_min": 30, "label": "main"},
]
SHADOW_STRATEGIES.append("FAST_TP50_SL30_NOLAZY")

STRATEGIES["TP50_SL15_NOLAZY"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.85, "horizon_min": 120, "label": "main"},
]
SHADOW_STRATEGIES.append("TP50_SL15_NOLAZY")


# ---------------------------------------------------------------------------
# v144 — SCORE filter isolation shadows.
# BE25_TP80_SL30_S30_HYST (N=12) showed WR 58%, avg +25.84% — but HYST bagage
# confounds. Retroactive audit on BE25 base (N=69):
#   SCORE>=30: N=33 WR 39% avg +16.20%
#   SCORE>=35: N=22 WR 50% avg +27.68%
#   SCORE>=40: N=13 WR 62% avg +33.92%
#   SCORE>=45: N=11 WR 64% avg +38.52%
# Threshold >=40 concentrates alpha (the 30-40 band loses −5.86% pop-wide).
# These shadows isolate the filter from HYST noise — same TP/SL/horizon as
# BE25_TP80_SL30 base, only difference is the min_rt_score filter.
# ---------------------------------------------------------------------------
STRATEGIES["BE25_TP80_SL30_S30"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "label": "main"},
]
SHADOW_STRATEGIES.append("BE25_TP80_SL30_S30")

STRATEGIES["BE25_TP80_SL30_S40"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "label": "main"},
]
SHADOW_STRATEGIES.append("BE25_TP80_SL30_S40")

# FAST_TP80_SL25 + SCORE>=40 filter. Retroactive (N=16): WR 68.8%, avg +29.76%,
# $+232 — strongest score-filtered FAST on rt_score v141. No HYST, no LAZY
# override (LAZY_STRATEGIES kept empty for this variant so CURRENT polling
# applies, giving a second paired point alongside FAST_TP80_SL25 main LAZY).
STRATEGIES["FAST_TP80_SL25_S40"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.75, "horizon_min": 30, "label": "main"},
]
SHADOW_STRATEGIES.append("FAST_TP80_SL25_S40")

# FAST_TP50_SL30 + SCORE>=40. Sim mega sweep v142 TOP SCORE40 candidate:
# N=18, WR 72.22%, avg +34.53%, $69.93/jour. Paper retroactive (N=37):
# WR 51%, avg +8.91% — sim much more optimistic, real N≥30 will arbitrate.
STRATEGIES["FAST_TP50_SL30_S40"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.70, "horizon_min": 30, "label": "main"},
]
SHADOW_STRATEGIES.append("FAST_TP50_SL30_S40")

# FAST_TP100_SL20 + SCORE>=40. Sim top5 SCORE40: N=18, WR 66.67%, avg +32.0%,
# $64.82/jour. Highest TP in the score-filtered family — tests whether wide TP
# captures the outliers that SCORE>=40 tokens typically produce.
STRATEGIES["FAST_TP100_SL20_S40"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.80, "horizon_min": 30, "label": "main"},
]
SHADOW_STRATEGIES.append("FAST_TP100_SL20_S40")


# ---------------------------------------------------------------------------
# v144 — LAZYSLOW variants. Extended mega sweep v142 flagged lazy_slow
# (300/600/900s) as dominant on several strats BUT sim over-estimates trail/BE
# strategies by ~45x (TD2 sim \$154/day vs real \$3.40/day) so this needs
# shadow validation. These shadows use polling_sec=600 via strategy_overrides
# (approx lazy_slow) instead of LAZY_STRATEGIES membership (which shadows
# bypass anyway). Not a true LAZY profile (no fast-burst window) but closest
# approximation without refactoring _should_evaluate_exit.
# ---------------------------------------------------------------------------
STRATEGIES["FAST_TP50_SL30_LAZYSLOW"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.70, "horizon_min": 30, "label": "main"},
]
SHADOW_STRATEGIES.append("FAST_TP50_SL30_LAZYSLOW")

STRATEGIES["FAST_TP80_SL25_LAZYSLOW"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.75, "horizon_min": 30, "label": "main"},
]
SHADOW_STRATEGIES.append("FAST_TP80_SL25_LAZYSLOW")

STRATEGIES["BE25_TP80_SL30_LAZYSLOW"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "label": "main"},
]
SHADOW_STRATEGIES.append("BE25_TP80_SL30_LAZYSLOW")


# ---------------------------------------------------------------------------
# v144 — LAZY cadence family A/B on FAST_TP50_SL30 (live + top paper earner).
# Tests full LAZY profile spread: LAZYFAST / LAZYMED / LAZYSTD (LAZY_STRATEGIES
# main) / LAZYSLOW / LAZYXSLOW. Polling_sec overrides approximate the slow-phase
# interval of each LAZY profile. NOLAZY (polling 30s) already exists.
# ---------------------------------------------------------------------------
STRATEGIES["FAST_TP50_SL30_LAZYFAST"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.70, "horizon_min": 30, "label": "main"},
]
SHADOW_STRATEGIES.append("FAST_TP50_SL30_LAZYFAST")

STRATEGIES["FAST_TP50_SL30_LAZYMED"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.70, "horizon_min": 30, "label": "main"},
]
SHADOW_STRATEGIES.append("FAST_TP50_SL30_LAZYMED")

STRATEGIES["FAST_TP50_SL30_LAZYXSLOW"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.70, "horizon_min": 30, "label": "main"},
]
SHADOW_STRATEGIES.append("FAST_TP50_SL30_LAZYXSLOW")


# ---------------------------------------------------------------------------
# v144 — Source family A/B: BOTH (merge jp+ds) / JUPITER / DS on live strats.
# Extended sweep flagged "both" as dominant on top configs. Validation via
# shadows before orchestration override on mains.
# ---------------------------------------------------------------------------
STRATEGIES["FAST_TP50_SL30_BOTH"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.70, "horizon_min": 30, "label": "main"},
]
SHADOW_STRATEGIES.append("FAST_TP50_SL30_BOTH")

STRATEGIES["FAST_TP50_SL30_JUPITER"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.70, "horizon_min": 30, "label": "main"},
]
SHADOW_STRATEGIES.append("FAST_TP50_SL30_JUPITER")

STRATEGIES["BE25_TP80_SL30_BOTH"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "label": "main"},
]
SHADOW_STRATEGIES.append("BE25_TP80_SL30_BOTH")

STRATEGIES["BE25_TP80_SL30_JUPITER"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "label": "main"},
]
SHADOW_STRATEGIES.append("BE25_TP80_SL30_JUPITER")

STRATEGIES["FAST_TP80_SL25_BOTH"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.75, "horizon_min": 30, "label": "main"},
]
SHADOW_STRATEGIES.append("FAST_TP80_SL25_BOTH")

STRATEGIES["FAST_TP100_SL20_BOTH"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.80, "horizon_min": 30, "label": "main"},
]
SHADOW_STRATEGIES.append("FAST_TP100_SL20_BOTH")


# ---------------------------------------------------------------------------
# v144 — MCAP_MID_SCORE40 combo filter shadow. Sim extended: FAST_TP50_SL30
# + MCAP_MID_SCORE40 = N=19 WR 74% avg +33.6%. Already in STRATEGY_FILTERS.
# ---------------------------------------------------------------------------
STRATEGIES["FAST_TP50_SL30_MCAP_S40"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.70, "horizon_min": 30, "label": "main"},
]
SHADOW_STRATEGIES.append("FAST_TP50_SL30_MCAP_S40")


# ---------------------------------------------------------------------------
# v144 — Top sim combo shadows. Uses the new split source+smoothing fields
# (requires paper_trader._decision_price v144 refactor). Each shadow replicates
# the exact top config found by the extended mega sweep for validation.
# ---------------------------------------------------------------------------
# Sim #1: FAST_TP100_SL20 + both + median_3 + lazy_slow → \$149/day
STRATEGIES["FAST_TP100_SL20_COMBO"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.80, "horizon_min": 30, "label": "main"},
]
SHADOW_STRATEGIES.append("FAST_TP100_SL20_COMBO")

# Sim #2: BE25_TP80_SL30 + both + median_3 + lazy_slow → \$141/day
STRATEGIES["BE25_TP80_SL30_COMBO"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "label": "main"},
]
SHADOW_STRATEGIES.append("BE25_TP80_SL30_COMBO")

# Sim #3: FAST_TP80_SL25 + ds + winsor_p95 + lazy → \$138/day
STRATEGIES["FAST_TP80_SL25_COMBO"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.75, "horizon_min": 30, "label": "main"},
]
SHADOW_STRATEGIES.append("FAST_TP80_SL25_COMBO")

# Sim #4: FAST_TP50_SL30 + ds + winsor_p95 + lazy_slow → \$127/day
STRATEGIES["FAST_TP50_SL30_COMBO"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.70, "horizon_min": 30, "label": "main"},
]
SHADOW_STRATEGIES.append("FAST_TP50_SL30_COMBO")


# ---------------------------------------------------------------------------
# v144 — FAST_TP40_SL30 smoothing A/B. Main uses hysteresis smoothing but has
# no paired _HYST variant for control. Other _HYST paired tests show −2 to −6pp
# vs base, suggesting FAST_TP40 with hysteresis may leave money on the table.
# Shadow with median_3 (same as FAST_TP50_SL30 main) to isolate smoothing effect.
# ---------------------------------------------------------------------------
STRATEGIES["FAST_TP40_SL30_MED3"] = [
    {"pct": 1.0, "tp_mult": 1.40, "sl_mult": 0.70, "horizon_min": 30, "label": "main"},
]
SHADOW_STRATEGIES.append("FAST_TP40_SL30_MED3")

STRATEGIES["FAST_TP40_SL30_DS"] = [
    {"pct": 1.0, "tp_mult": 1.40, "sl_mult": 0.70, "horizon_min": 30, "label": "main"},
]
SHADOW_STRATEGIES.append("FAST_TP40_SL30_DS")


# ---------------------------------------------------------------------------
# v144.2 (Apr 20) — Coverage extension for paired A/B testing.
# Audit `paired_all_v144_shadows.py` showed gaps:
#   - TP50_SL15: only _NOLAZY paired (won +$40, med +3.80pp). Missing source/smoothing.
#   - FAST_TP40_SL30: missing source dim (_BOTH/_JUPITER) and SCORE filter (_S40)
#   - FAST_TP100_SL20: missing _NOLAZY, full smoothing dim
#   - FAST_TP80_SL25: missing smoothing dim (_DS/_MED3/_JUPITER)
#   - HIGHSCORE_TP200_SL40 (#5 earner +$35/j): zero v144 coverage
# Filters inherited via STRATEGY_FILTERS lookup below.
# ---------------------------------------------------------------------------

# TP50_SL15 family extension (5 new)
STRATEGIES["TP50_SL15_BOTH"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.85, "horizon_min": 120, "label": "main"},
]
SHADOW_STRATEGIES.append("TP50_SL15_BOTH")

STRATEGIES["TP50_SL15_JUPITER"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.85, "horizon_min": 120, "label": "main"},
]
SHADOW_STRATEGIES.append("TP50_SL15_JUPITER")

STRATEGIES["TP50_SL15_DS"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.85, "horizon_min": 120, "label": "main"},
]
SHADOW_STRATEGIES.append("TP50_SL15_DS")

STRATEGIES["TP50_SL15_MED3"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.85, "horizon_min": 120, "label": "main"},
]
SHADOW_STRATEGIES.append("TP50_SL15_MED3")

STRATEGIES["TP50_SL15_S40"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.85, "horizon_min": 120, "label": "main"},
]
SHADOW_STRATEGIES.append("TP50_SL15_S40")

# FAST_TP40_SL30 source + filter (3 new)
STRATEGIES["FAST_TP40_SL30_BOTH"] = [
    {"pct": 1.0, "tp_mult": 1.40, "sl_mult": 0.70, "horizon_min": 30, "label": "main"},
]
SHADOW_STRATEGIES.append("FAST_TP40_SL30_BOTH")

STRATEGIES["FAST_TP40_SL30_JUPITER"] = [
    {"pct": 1.0, "tp_mult": 1.40, "sl_mult": 0.70, "horizon_min": 30, "label": "main"},
]
SHADOW_STRATEGIES.append("FAST_TP40_SL30_JUPITER")

STRATEGIES["FAST_TP40_SL30_S40"] = [
    {"pct": 1.0, "tp_mult": 1.40, "sl_mult": 0.70, "horizon_min": 30, "label": "main"},
]
SHADOW_STRATEGIES.append("FAST_TP40_SL30_S40")

# FAST_TP100_SL20 NOLAZY + smoothing (4 new)
STRATEGIES["FAST_TP100_SL20_NOLAZY"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.80, "horizon_min": 30, "label": "main"},
]
SHADOW_STRATEGIES.append("FAST_TP100_SL20_NOLAZY")

STRATEGIES["FAST_TP100_SL20_DS"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.80, "horizon_min": 30, "label": "main"},
]
SHADOW_STRATEGIES.append("FAST_TP100_SL20_DS")

STRATEGIES["FAST_TP100_SL20_MED3"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.80, "horizon_min": 30, "label": "main"},
]
SHADOW_STRATEGIES.append("FAST_TP100_SL20_MED3")

STRATEGIES["FAST_TP100_SL20_JUPITER"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.80, "horizon_min": 30, "label": "main"},
]
SHADOW_STRATEGIES.append("FAST_TP100_SL20_JUPITER")

# FAST_TP80_SL25 smoothing (3 new)
STRATEGIES["FAST_TP80_SL25_DS"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.75, "horizon_min": 30, "label": "main"},
]
SHADOW_STRATEGIES.append("FAST_TP80_SL25_DS")

STRATEGIES["FAST_TP80_SL25_MED3"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.75, "horizon_min": 30, "label": "main"},
]
SHADOW_STRATEGIES.append("FAST_TP80_SL25_MED3")

STRATEGIES["FAST_TP80_SL25_JUPITER"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.75, "horizon_min": 30, "label": "main"},
]
SHADOW_STRATEGIES.append("FAST_TP80_SL25_JUPITER")

# HIGHSCORE_TP200_SL40 (#5 earner +$35/j) — full v144 coverage (4 new)
# Inherits min_rt_score:30 filter via STRATEGY_FILTERS extension below.
STRATEGIES["HIGHSCORE_TP200_SL40_BOTH"] = [
    {"pct": 1.0, "tp_mult": 3.00, "sl_mult": 0.60, "horizon_min": 240, "label": "main"},
]
SHADOW_STRATEGIES.append("HIGHSCORE_TP200_SL40_BOTH")

STRATEGIES["HIGHSCORE_TP200_SL40_DS"] = [
    {"pct": 1.0, "tp_mult": 3.00, "sl_mult": 0.60, "horizon_min": 240, "label": "main"},
]
SHADOW_STRATEGIES.append("HIGHSCORE_TP200_SL40_DS")

STRATEGIES["HIGHSCORE_TP200_SL40_MED3"] = [
    {"pct": 1.0, "tp_mult": 3.00, "sl_mult": 0.60, "horizon_min": 240, "label": "main"},
]
SHADOW_STRATEGIES.append("HIGHSCORE_TP200_SL40_MED3")

STRATEGIES["HIGHSCORE_TP200_SL40_NOLAZY"] = [
    {"pct": 1.0, "tp_mult": 3.00, "sl_mult": 0.60, "horizon_min": 240, "label": "main"},
]
SHADOW_STRATEGIES.append("HIGHSCORE_TP200_SL40_NOLAZY")

# Filter inheritance for v144.2 score-gated shadows
STRATEGY_FILTERS["TP50_SL15_S40"] = {"min_rt_score": 40}
STRATEGY_FILTERS["FAST_TP40_SL30_S40"] = {"min_rt_score": 40}
STRATEGY_FILTERS["HIGHSCORE_TP200_SL40_BOTH"] = {"min_rt_score": 30}
STRATEGY_FILTERS["HIGHSCORE_TP200_SL40_DS"] = {"min_rt_score": 30}
STRATEGY_FILTERS["HIGHSCORE_TP200_SL40_MED3"] = {"min_rt_score": 30}
STRATEGY_FILTERS["HIGHSCORE_TP200_SL40_NOLAZY"] = {"min_rt_score": 30}


# ---------------------------------------------------------------------------
# v144.4 (Apr 20) — SCORE35 sweet spot from mega_sweep_top_robust
# Top robust cluster (analyze_mega_sweep.py): FAST_TP100_SL20 + SCORE35 + LAZY
# + median_3 + jupiter — N=35, WR 62.86%, avg +28.06%, fdr_q=0.0000.
# Single distinct pattern that survives Bonferroni × 508K eligible configs.
# Existing shadows test SCORE>=30 and SCORE>=40 — S35 fills the sweet-spot gap.
# ---------------------------------------------------------------------------
STRATEGIES["FAST_TP100_SL20_S35"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.80, "horizon_min": 30, "label": "main"},
]
SHADOW_STRATEGIES.append("FAST_TP100_SL20_S35")
STRATEGY_FILTERS["FAST_TP100_SL20_S35"] = {"min_rt_score": 35}
LAZY_STRATEGIES.add("FAST_TP100_SL20_S35")

# v144.5 — Sweet spot SCORE35 also worth testing on BE25 (live strat).
# Existing: BE25_TP80_SL30_S30 (broad), BE25_TP80_SL30_S40 (strict). S35 fills the gap.
# Pattern extrapolated from FAST_TP100_SL20_S35 robust cluster.
STRATEGIES["BE25_TP80_SL30_S35"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "label": "main"},
]
SHADOW_STRATEGIES.append("BE25_TP80_SL30_S35")
STRATEGY_FILTERS["BE25_TP80_SL30_S35"] = {"min_rt_score": 35}


# ============================================================
# v14 (Sprint #ETH-1) — Ethereum L1 shadow strategies.
# Shadow-only, zero capital. Entry filters require chain='ethereum'.
# TP/SL widened vs Solana because ETH fees ($15 round-trip gas + 2% MEV
# slippage) make tight TP30/TP50 unprofitable. Phase 2 verdict at N≥50 /
# 14 days: go live if WR ≥ 65% AND EV net ≥ +10%/trade; else archive.
#
# min_liquidity_usd 25k because ETH memecoin slippage above 2% kicks in
# below ~$20k pool depth — we'd eat our edge before TP fires.
# ============================================================

# 1) ETH_TP100_SL50 — let-it-run classic. 4h horizon because ETH moves slower
#    than Solana memecoins (bigger pools, fewer snipers).
# v14b: promoted to MAIN paper (not shadow) — user wants Telegram alerts +
# paper_trades.is_shadow=False so the flow is identical to a Solana main.
# Must be added to scoring_config.rt_trade_config.hybrid_strategy.allocations
# for RT path to actually open them.
STRATEGIES["ETH_TP100_SL50"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.50, "horizon_min": 240, "label": "main"},
]
STRATEGY_FILTERS["ETH_TP100_SL50"] = {
    "chain": "ethereum",
    "min_liquidity_usd": 25_000,
}

# 2) ETH_TP80_SL40_T2H — conservative. Shorter horizon, tighter SL.
STRATEGIES["ETH_TP80_SL40_T2H"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.60, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["ETH_TP80_SL40_T2H"] = {
    "chain": "ethereum",
    "min_liquidity_usd": 25_000,
}

# 3) ETH_BE50_TP150_SL50 — breakeven protection at +50%, TP +150%.
#    For KOLs whose ETH calls tend to 2-3x. BE activation prevents round-trip
#    on tokens that pump then dump.
STRATEGIES["ETH_BE50_TP150_SL50"] = [
    {"pct": 1.0, "tp_mult": 2.50, "sl_mult": 0.50, "horizon_min": 240,
     "be_activation": 0.50, "label": "main"},
]
STRATEGY_FILTERS["ETH_BE50_TP150_SL50"] = {
    "chain": "ethereum",
    "min_liquidity_usd": 25_000,
}


# ---------------------------------------------------------------------------
# Sim config → fake trade converter
# ---------------------------------------------------------------------------
def sim_cfg_to_fake_trade(cfg: dict, entry_price: float, created_at: str,
                          liquidity_usd: float = 50_000,
                          trade_id: str = "sim_0") -> dict:
    """Convert sim grid config → _evaluate_trade_exit()-compatible trade dict.

    Bridges the gap between sim.py grid configs (numeric dicts with 'type',
    'tp_mult', etc.) and paper_trader trade dicts (DB-style with 'strategy',
    'sl_price', 'tp_price', etc.).

    For multi-tranche strategies, call once per tranche with appropriate
    tranche_label and tranche-specific params.
    """
    tp_mult = cfg.get("tp_mult")
    sl_mult = cfg.get("sl_mult", 0.50)
    horizon = cfg.get("horizon") or cfg.get("horizon_min") or cfg.get("timeout") or 120
    name = cfg.get("name", "UNKNOWN")

    tp_price = entry_price * tp_mult if tp_mult else None
    sl_price = entry_price * sl_mult

    return {
        "id": trade_id,
        "entry_price": entry_price,
        "sl_price": sl_price,
        "tp_price": tp_price,
        "position_usd": 10.0,
        "strategy": name,
        "tranche_label": cfg.get("tranche_label", "main"),
        "horizon_minutes": horizon,
        "created_at": created_at,
        "high_price_seen": entry_price,
        "rt_liquidity_usd": liquidity_usd,
        "dex_spot_price_at_entry": entry_price,
    }
