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
# Fee constants — Solana
#
# v121 (Apr 8): set to 10 bps assuming Jupiter Ultra RFQ near-zero fills.
# v14e.24 (Apr 25): empirical recalibration. 229 live rt_live trades since
# Apr 8 yielded median buy slip = +225 bps (vs assumed 10), p95 = +2598 bps,
# std = 1424 bps. OLS fit on 6 features (liq, pos, age, latency, pump, pos/liq)
# yields R² = 5.8% — slip is dominated by price-drift between DexScreener
# spot fetch and Ultra fill (~700ms) which we cannot observe ex-ante.
# Full-fit MAE 717 bps vs constant-median MAE 738 bps → 2.8% gain only.
# Honest baseline: use the median, accept variance as Monte Carlo noise.
# Sell slip stays dynamic via _dynamic_sell_slip_factor (liq-aware).
# Calibration script: scripts/_calibrate_buy_slip.py → data/buy_slip_calibration.json
# ---------------------------------------------------------------------------
BUY_SLIPPAGE_BPS = 225   # 2.25% — empirical median (v14e.24)
SELL_SLIPPAGE_BPS = 10   # 0.1% — base; dynamic adjuster applies liq_mult + type_bps on top
BUY_FEE_BPS = 0          # 0% — folded into slippage
SELL_FEE_BPS = 0          # 0% — folded into slippage

# ---------------------------------------------------------------------------
# Fee constants — Ethereum L1 (v14e.28: empirical recalibration Apr 26)
#
# Calibrated against 2 real round-trips on PEPE (Apr 26, base_fee 0.5-1.5 gwei,
# ETH @ $2330). Measured: $0.85-1.15 gas/side, ~0 bps slip on PEPE 0.3% pool.
# Defensive buffers above measured to account for shallower memecoin pools.
# Empirical run captured in data/eth_smoke_20260426T132313Z.json.
#
# Gas regime assumption: post-Pectra mainnet sub-2 gwei. ETH base_fee tracked
# 0.12-0.83 gwei across 7d window (Apr 19-26). If base_fee returns >5 gwei
# (rare since mid-2025), ETH_GAS_COST_USD_PER_SIDE will need ~3x bump.
# Re-run scripts/_eth_round_trip_smoke.py to recalibrate.
#
# Slip 100 bps = defensive vs PEPE-measured 0 bps. Real memecoins on Uniswap
# V3 ($50-500K liq) typically take 50-200 bps. Drop to ~30 bps once we have
# N≥20 ETH live trades to fit empirically (mirror SOL v14e.24 methodology).
# ---------------------------------------------------------------------------
ETH_GAS_COST_USD_PER_SIDE = 1.50   # $1.50 each side, $3 round-trip (Apr 26 empirical + 25% buffer)
# v14e.43 (Apr 29): empirical recalibration on N=8 KOL memecoin live trades.
# Adverse-side median was 892 bps buy / 1600 bps sell, expected (signed mean)
# 749 / 1216. Applied SHRINK=0.65 on the expected to keep paper sim from over-
# pessimizing on small N — proposed by scripts/_calibrate_eth_slip.py and
# also persisted in scoring_config.paper_trade_config.eth_*_slippage_bps so
# live_trader_eth picks up the same numbers. Pre-fix sim drag was 17% (way
# below empirical 30%); after this bump the implied drag rises to ~28%, much
# closer to ground truth. Revisit at N≥20.
ETH_BUY_SLIPPAGE_BPS = 350         # v14e.49b: recalibrated on N=10 (post-MUSK), shrinkage 0.65 — empirical buy adverse mean 554 bps
ETH_SELL_SLIPPAGE_BPS = 650        # v14e.49b: recalibrated on N=10, empirical sell adverse mean 973 bps. Median total cost reality 21.5%, paper sim now 25.0%.
ETH_MIN_POSITION_USD = 50          # gas = 6% of $50 — viable; below $30 gas dominates >10%

# ---------------------------------------------------------------------------
# Fee constants — BSC (v14e: paper-only cost model)
#
# BSC gas ~3 gwei at 5-figure gas = ~$0.15-0.30 per PancakeSwap swap. Pools
# are typically deeper than ETH L1 but MEV sandwiching is rampant on
# PancakeSwap (no private relay equivalent). Budget 250 bps/side for MEV.
# Position floor $50 — gas dominates under that. PancakeSwap V3 fee tier
# (0.25% usually for memecoins) is folded into the slippage budget.
# ---------------------------------------------------------------------------
BSC_GAS_COST_USD_PER_SIDE = 0.30
BSC_BUY_SLIPPAGE_BPS = 250
BSC_SELL_SLIPPAGE_BPS = 250
BSC_MIN_POSITION_USD = 50

# ---------------------------------------------------------------------------
# Fee constants — Base L2 (v14e: paper-only cost model)
#
# Base is an L2 → gas ~$0.05-0.10 per swap (basefee in the 0.001 gwei range).
# MEV is less aggressive than L1/BSC but present on Aerodrome/Uniswap V3.
# Budget 150 bps/side + $0.10 gas. Position floor $50 by parity with BSC.
# ---------------------------------------------------------------------------
BASE_GAS_COST_USD_PER_SIDE = 0.10
BASE_BUY_SLIPPAGE_BPS = 150
BASE_SELL_SLIPPAGE_BPS = 150
BASE_MIN_POSITION_USD = 50


# ---------------------------------------------------------------------------
# Default deprecated strategies — overridable via scoring_config JSONB
# v14e.36: trail/dip/split families dropped to deprecated set so they stop
# polluting shadow analytics. Per dtrail_shadow_artifact_apr20 audit, sim
# over-estimates these by 47x (paper models 200 bps sell slip; live actual
# 9429 bps from multi-step sells), and the position_reconciler closes 50-65%
# of these trades before the trail can fire. They cannot be promoted to
# live, and their "winning" $/day in shadow polluted every sweep ranking
# (top robust on Apr 27 10h: top 17 = all DTRAIL20_*).
# Specific trail/dip/split names are appended below the family loops.
_DEFAULT_DEPRECATED = {
    "MOONBAG", "WIDE_RUNNER", "SCALE_OUT", "TP100_SL30",
    "QUICK_SCALP", "TP30_SL10", "TP50_SL15", "TP30_SL30",
}


def _is_artifact_family(name: str) -> bool:
    """Trail/dip/split prefixes whose shadow $/day cannot translate to live.
    Matches the family_realism=0.1 filter in analyze_mega_sweep.py."""
    s = name.upper()
    return (
        s.startswith("DTRAIL")
        or s.startswith("PTRAIL")
        or s.startswith("TRAIL")
        or s.startswith("SPLIT_")
        or s.startswith("DIP30_")
        or s.startswith("DIP_")
        or "MCAP_MID_DTRAIL" in s
        or s.startswith("BOND_")
        or s.startswith("TD2_")
    )

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

# ============================================================
# v14e.29 (Apr 26) — BE+LOCK profit-lock variants.
# Backtest on 100 SOL + 37 ETH BE25/BE30 closed trades (replay of price_ticks):
#   LOCK10 SOL: +1.88pp avg vs BE25 (19 trades better, 0 worse)
#   LOCK10 ETH: +2.85pp avg vs ETH_BE30 (10 better, 3 worse)
#   LOCK15-20: increasingly hurt (TP exits sacrificed) — sweet spot LOCK10
# Mechanic: when peak ≥ entry*(1+be_act), SL ratchets to entry*(1+lock_pct)
# instead of plain entry. Single-event ratchet (not continuous trail) so it
# escapes the 47x slip + reconciler-race pitfalls of TRAIL/DTRAIL.
# All shadows. Will paired-test vs BE base when N≥30.
# ============================================================

# SOL — clones of BE25_TP80_SL30 with various lock pcts
STRATEGIES["BE25_LOCK10_TP80_SL30"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.10, "label": "main"},
]
SHADOW_STRATEGIES.append("BE25_LOCK10_TP80_SL30")

STRATEGIES["BE25_LOCK5_TP80_SL30"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.05, "label": "main"},
]
SHADOW_STRATEGIES.append("BE25_LOCK5_TP80_SL30")

STRATEGIES["BE25_LOCK15_TP100_SL30"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.15, "label": "main"},
]
SHADOW_STRATEGIES.append("BE25_LOCK15_TP100_SL30")

# Higher activation pre-lock — only locks once we have a real pump
STRATEGIES["BE50_LOCK20_TP150_SL30"] = [
    {"pct": 1.0, "tp_mult": 2.50, "sl_mult": 0.70, "horizon_min": 60,
     "be_activation": 0.50, "be_lock_pct": 0.20, "label": "main"},
]
SHADOW_STRATEGIES.append("BE50_LOCK20_TP150_SL30")

STRATEGIES["BE50_LOCK25_TP200_SL40_4H"] = [
    {"pct": 1.0, "tp_mult": 3.00, "sl_mult": 0.60, "horizon_min": 240,
     "be_activation": 0.50, "be_lock_pct": 0.25, "label": "main"},
]
SHADOW_STRATEGIES.append("BE50_LOCK25_TP200_SL40_4H")

# ETH — clones of the 2 active ETH paper mains with LOCK10 (validated by backtest)
STRATEGIES["ETH_BE25_LOCK10_TP80_SL40_T2H"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.60, "horizon_min": 120,
     "be_activation": 0.25, "be_lock_pct": 0.10, "label": "main"},
]
STRATEGY_FILTERS["ETH_BE25_LOCK10_TP80_SL40_T2H"] = {"chain": "ethereum"}
SHADOW_STRATEGIES.append("ETH_BE25_LOCK10_TP80_SL40_T2H")

STRATEGIES["ETH_BE30_LOCK10_TP100_SL40"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.60, "horizon_min": 240,
     "be_activation": 0.30, "be_lock_pct": 0.10, "label": "main"},
]
STRATEGY_FILTERS["ETH_BE30_LOCK10_TP100_SL40"] = {"chain": "ethereum"}
SHADOW_STRATEGIES.append("ETH_BE30_LOCK10_TP100_SL40")

STRATEGIES["ETH_BE30_LOCK15_TP100_SL40"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.60, "horizon_min": 240,
     "be_activation": 0.30, "be_lock_pct": 0.15, "label": "main"},
]
STRATEGY_FILTERS["ETH_BE30_LOCK15_TP100_SL40"] = {"chain": "ethereum"}
SHADOW_STRATEGIES.append("ETH_BE30_LOCK15_TP100_SL40")

STRATEGIES["ETH_BE50_LOCK20_TP150_SL40"] = [
    {"pct": 1.0, "tp_mult": 2.50, "sl_mult": 0.60, "horizon_min": 240,
     "be_activation": 0.50, "be_lock_pct": 0.20, "label": "main"},
]
STRATEGY_FILTERS["ETH_BE50_LOCK20_TP150_SL40"] = {"chain": "ethereum"}
SHADOW_STRATEGIES.append("ETH_BE50_LOCK20_TP150_SL40")

# ============================================================
# v14e.29 (Apr 26 PM) — extended LOCK grid for mega-sweep coverage.
# First 9 LOCK clones validated the mechanic. These extend the parameter
# space (be_activation × lock_pct × tp × horizon) so the mega-sweep can
# rank them against ALL strats, not just classic-BE base. If LOCK is
# competitive globally, the top robust will surface multiple LOCK configs;
# if not, mega-sweep ranking will tell us LOCK is just a BE optim and not
# a top-tier family. Either way, more shadows = more signal.
# ============================================================

# SOL — early-activation variants (BE15) for fast scalp profile
STRATEGIES["BE15_LOCK5_TP50_SL30"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.15, "be_lock_pct": 0.05, "label": "main"},
]
SHADOW_STRATEGIES.append("BE15_LOCK5_TP50_SL30")

STRATEGIES["BE15_LOCK10_TP80_SL30"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.15, "be_lock_pct": 0.10, "label": "main"},
]
SHADOW_STRATEGIES.append("BE15_LOCK10_TP80_SL30")

# SOL — BE25 LOCK grid covering more TP/SL combinations
STRATEGIES["BE25_LOCK10_TP100_SL30"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.10, "label": "main"},
]
SHADOW_STRATEGIES.append("BE25_LOCK10_TP100_SL30")

STRATEGIES["BE25_LOCK10_TP60_SL30"] = [
    {"pct": 1.0, "tp_mult": 1.60, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.10, "label": "main"},
]
SHADOW_STRATEGIES.append("BE25_LOCK10_TP60_SL30")

# Same with SL20 (tighter — locks profit harder + smaller initial loss bound)
STRATEGIES["BE25_LOCK10_TP80_SL20"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.80, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.10, "label": "main"},
]
SHADOW_STRATEGIES.append("BE25_LOCK10_TP80_SL20")

# SOL — longer horizon variants (T2H) capturing slow-pumps
STRATEGIES["BE25_LOCK10_TP100_SL40_T2H"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.60, "horizon_min": 120,
     "be_activation": 0.25, "be_lock_pct": 0.10, "label": "main"},
]
SHADOW_STRATEGIES.append("BE25_LOCK10_TP100_SL40_T2H")

STRATEGIES["BE25_LOCK15_TP150_SL40_T2H"] = [
    {"pct": 1.0, "tp_mult": 2.50, "sl_mult": 0.60, "horizon_min": 120,
     "be_activation": 0.25, "be_lock_pct": 0.15, "label": "main"},
]
SHADOW_STRATEGIES.append("BE25_LOCK15_TP150_SL40_T2H")

# SOL — late activation (BE35) — only locks after a real pump
STRATEGIES["BE35_LOCK15_TP100_SL30"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.35, "be_lock_pct": 0.15, "label": "main"},
]
SHADOW_STRATEGIES.append("BE35_LOCK15_TP100_SL30")

STRATEGIES["BE35_LOCK20_TP150_SL40_T2H"] = [
    {"pct": 1.0, "tp_mult": 2.50, "sl_mult": 0.60, "horizon_min": 120,
     "be_activation": 0.35, "be_lock_pct": 0.20, "label": "main"},
]
SHADOW_STRATEGIES.append("BE35_LOCK20_TP150_SL40_T2H")

# SOL — SCALP-style LOCK (small TP, small lock — very tight)
STRATEGIES["BE15_LOCK5_TP30_SL20"] = [
    {"pct": 1.0, "tp_mult": 1.30, "sl_mult": 0.80, "horizon_min": 30,
     "be_activation": 0.15, "be_lock_pct": 0.05, "label": "main"},
]
SHADOW_STRATEGIES.append("BE15_LOCK5_TP30_SL20")

# ETH — extended grid covering shorter/longer horizons + LOCK5
STRATEGIES["ETH_BE25_LOCK5_TP80_SL40_T2H"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.60, "horizon_min": 120,
     "be_activation": 0.25, "be_lock_pct": 0.05, "label": "main"},
]
STRATEGY_FILTERS["ETH_BE25_LOCK5_TP80_SL40_T2H"] = {"chain": "ethereum"}
SHADOW_STRATEGIES.append("ETH_BE25_LOCK5_TP80_SL40_T2H")

STRATEGIES["ETH_BE25_LOCK15_TP100_SL40_T2H"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.60, "horizon_min": 120,
     "be_activation": 0.25, "be_lock_pct": 0.15, "label": "main"},
]
STRATEGY_FILTERS["ETH_BE25_LOCK15_TP100_SL40_T2H"] = {"chain": "ethereum"}
SHADOW_STRATEGIES.append("ETH_BE25_LOCK15_TP100_SL40_T2H")

# ETH — fast horizon FAST family LOCK clones
STRATEGIES["ETH_BE25_LOCK10_TP100_SL20"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.80, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.10, "label": "main"},
]
STRATEGY_FILTERS["ETH_BE25_LOCK10_TP100_SL20"] = {"chain": "ethereum"}
SHADOW_STRATEGIES.append("ETH_BE25_LOCK10_TP100_SL20")

STRATEGIES["ETH_BE15_LOCK5_TP80_SL30"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.15, "be_lock_pct": 0.05, "label": "main"},
]
STRATEGY_FILTERS["ETH_BE15_LOCK5_TP80_SL30"] = {"chain": "ethereum"}
SHADOW_STRATEGIES.append("ETH_BE15_LOCK5_TP80_SL30")

# ETH — late activation (BE35/BE50) for slow-pump regime
STRATEGIES["ETH_BE35_LOCK15_TP150_SL40_T2H"] = [
    {"pct": 1.0, "tp_mult": 2.50, "sl_mult": 0.60, "horizon_min": 120,
     "be_activation": 0.35, "be_lock_pct": 0.15, "label": "main"},
]
STRATEGY_FILTERS["ETH_BE35_LOCK15_TP150_SL40_T2H"] = {"chain": "ethereum"}
SHADOW_STRATEGIES.append("ETH_BE35_LOCK15_TP150_SL40_T2H")

STRATEGIES["ETH_BE50_LOCK25_TP200_SL40"] = [
    {"pct": 1.0, "tp_mult": 3.00, "sl_mult": 0.60, "horizon_min": 240,
     "be_activation": 0.50, "be_lock_pct": 0.25, "label": "main"},
]
STRATEGY_FILTERS["ETH_BE50_LOCK25_TP200_SL40"] = {"chain": "ethereum"}
SHADOW_STRATEGIES.append("ETH_BE50_LOCK25_TP200_SL40")

# ==========================================================================
# v14e.30 (Apr 26 PM) — combinatorial gap-fill: LOCK × filters, AGE × LOCK,
# ETH SLOW family, ETH × SCALP filters, ETH × HYST/LAZY/AGE×SCALP combos.
#
# Audit identified holes in the strategy lattice:
#   ETH × LAZY = 0   ETH × HYST = 0   ETH × SLOW = 0
#   LOCK × score-filter = 0 anywhere   LOCK × AGE = 0 anywhere
#   AGE × SCALP × ETH = 0   AGE × BE × ETH = 0
# These ~40 shadows cover the most impactful missing combos. All paper-only.
# ==========================================================================

# ---- Bloc A: SOL LOCK × score/liq filters (6) ----
STRATEGIES["BE25_LOCK10_TP80_SL30_S35"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.10, "label": "main"},
]
STRATEGY_FILTERS["BE25_LOCK10_TP80_SL30_S35"] = {"chain": "solana", "min_rt_score": 35}
SHADOW_STRATEGIES.append("BE25_LOCK10_TP80_SL30_S35")

STRATEGIES["BE25_LOCK10_TP80_SL30_S40"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.10, "label": "main"},
]
STRATEGY_FILTERS["BE25_LOCK10_TP80_SL30_S40"] = {"chain": "solana", "min_rt_score": 40}
SHADOW_STRATEGIES.append("BE25_LOCK10_TP80_SL30_S40")

STRATEGIES["BE25_LOCK10_TP100_SL30_NZ_S40"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.10, "label": "main"},
]
STRATEGY_FILTERS["BE25_LOCK10_TP100_SL30_NZ_S40"] = {
    "chain": "solana", "min_liquidity_usd": 1, "min_rt_score": 40,
}
SHADOW_STRATEGIES.append("BE25_LOCK10_TP100_SL30_NZ_S40")

STRATEGIES["BE50_LOCK20_TP150_SL30_S40"] = [
    {"pct": 1.0, "tp_mult": 2.50, "sl_mult": 0.70, "horizon_min": 60,
     "be_activation": 0.50, "be_lock_pct": 0.20, "label": "main"},
]
STRATEGY_FILTERS["BE50_LOCK20_TP150_SL30_S40"] = {"chain": "solana", "min_rt_score": 40}
SHADOW_STRATEGIES.append("BE50_LOCK20_TP150_SL30_S40")

STRATEGIES["BE50_LOCK25_TP200_SL40_4H_NZ_S40"] = [
    {"pct": 1.0, "tp_mult": 3.00, "sl_mult": 0.60, "horizon_min": 240,
     "be_activation": 0.50, "be_lock_pct": 0.25, "label": "main"},
]
STRATEGY_FILTERS["BE50_LOCK25_TP200_SL40_4H_NZ_S40"] = {
    "chain": "solana", "min_liquidity_usd": 1, "min_rt_score": 40,
}
SHADOW_STRATEGIES.append("BE50_LOCK25_TP200_SL40_4H_NZ_S40")

STRATEGIES["BE50_LOCK25_TP200_SL40_4H_MCAP"] = [
    {"pct": 1.0, "tp_mult": 3.00, "sl_mult": 0.60, "horizon_min": 240,
     "be_activation": 0.50, "be_lock_pct": 0.25, "label": "main"},
]
STRATEGY_FILTERS["BE50_LOCK25_TP200_SL40_4H_MCAP"] = {
    "chain": "solana", "min_mcap": 30_000, "max_mcap": 500_000,
}
SHADOW_STRATEGIES.append("BE50_LOCK25_TP200_SL40_4H_MCAP")

# ---- Bloc B: ETH LOCK × score/liq filters (6) ----
STRATEGIES["ETH_BE25_LOCK10_TP80_SL40_T2H_S35"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.60, "horizon_min": 120,
     "be_activation": 0.25, "be_lock_pct": 0.10, "label": "main"},
]
STRATEGY_FILTERS["ETH_BE25_LOCK10_TP80_SL40_T2H_S35"] = {"chain": "ethereum", "min_rt_score": 35}
SHADOW_STRATEGIES.append("ETH_BE25_LOCK10_TP80_SL40_T2H_S35")

STRATEGIES["ETH_BE30_LOCK10_TP100_SL40_S40"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.60, "horizon_min": 240,
     "be_activation": 0.30, "be_lock_pct": 0.10, "label": "main"},
]
STRATEGY_FILTERS["ETH_BE30_LOCK10_TP100_SL40_S40"] = {"chain": "ethereum", "min_rt_score": 40}
SHADOW_STRATEGIES.append("ETH_BE30_LOCK10_TP100_SL40_S40")

STRATEGIES["ETH_BE30_LOCK10_TP100_SL40_NZ_S40"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.60, "horizon_min": 240,
     "be_activation": 0.30, "be_lock_pct": 0.10, "label": "main"},
]
STRATEGY_FILTERS["ETH_BE30_LOCK10_TP100_SL40_NZ_S40"] = {
    "chain": "ethereum", "min_liquidity_usd": 1, "min_rt_score": 40,
}
SHADOW_STRATEGIES.append("ETH_BE30_LOCK10_TP100_SL40_NZ_S40")

STRATEGIES["ETH_BE50_LOCK20_TP150_SL40_S40"] = [
    {"pct": 1.0, "tp_mult": 2.50, "sl_mult": 0.60, "horizon_min": 240,
     "be_activation": 0.50, "be_lock_pct": 0.20, "label": "main"},
]
STRATEGY_FILTERS["ETH_BE50_LOCK20_TP150_SL40_S40"] = {"chain": "ethereum", "min_rt_score": 40}
SHADOW_STRATEGIES.append("ETH_BE50_LOCK20_TP150_SL40_S40")

STRATEGIES["ETH_BE50_LOCK20_TP150_SL40_NZ_S40"] = [
    {"pct": 1.0, "tp_mult": 2.50, "sl_mult": 0.60, "horizon_min": 240,
     "be_activation": 0.50, "be_lock_pct": 0.20, "label": "main"},
]
STRATEGY_FILTERS["ETH_BE50_LOCK20_TP150_SL40_NZ_S40"] = {
    "chain": "ethereum", "min_liquidity_usd": 1, "min_rt_score": 40,
}
SHADOW_STRATEGIES.append("ETH_BE50_LOCK20_TP150_SL40_NZ_S40")

STRATEGIES["ETH_BE50_LOCK25_TP200_SL40_MCAP_S40"] = [
    {"pct": 1.0, "tp_mult": 3.00, "sl_mult": 0.60, "horizon_min": 240,
     "be_activation": 0.50, "be_lock_pct": 0.25, "label": "main"},
]
STRATEGY_FILTERS["ETH_BE50_LOCK25_TP200_SL40_MCAP_S40"] = {
    "chain": "ethereum", "min_mcap": 30_000, "max_mcap": 500_000, "min_rt_score": 40,
}
SHADOW_STRATEGIES.append("ETH_BE50_LOCK25_TP200_SL40_MCAP_S40")

# ---- Bloc C: AGE × LOCK SOL (4) ----
STRATEGIES["AGE24_BE25_LOCK10_TP80_SL30"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.10, "label": "main"},
]
STRATEGY_FILTERS["AGE24_BE25_LOCK10_TP80_SL30"] = {
    "chain": "solana", "min_age_hours": 12, "max_age_hours": 24,
}
SHADOW_STRATEGIES.append("AGE24_BE25_LOCK10_TP80_SL30")

STRATEGIES["AGE48_BE25_LOCK10_TP80_SL30"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.10, "label": "main"},
]
STRATEGY_FILTERS["AGE48_BE25_LOCK10_TP80_SL30"] = {
    "chain": "solana", "min_age_hours": 24, "max_age_hours": 48,
}
SHADOW_STRATEGIES.append("AGE48_BE25_LOCK10_TP80_SL30")

STRATEGIES["AGE24_BE50_LOCK20_TP150_SL30"] = [
    {"pct": 1.0, "tp_mult": 2.50, "sl_mult": 0.70, "horizon_min": 60,
     "be_activation": 0.50, "be_lock_pct": 0.20, "label": "main"},
]
STRATEGY_FILTERS["AGE24_BE50_LOCK20_TP150_SL30"] = {
    "chain": "solana", "min_age_hours": 12, "max_age_hours": 24,
}
SHADOW_STRATEGIES.append("AGE24_BE50_LOCK20_TP150_SL30")

STRATEGIES["AGE48_BE50_LOCK20_TP150_SL30"] = [
    {"pct": 1.0, "tp_mult": 2.50, "sl_mult": 0.70, "horizon_min": 60,
     "be_activation": 0.50, "be_lock_pct": 0.20, "label": "main"},
]
STRATEGY_FILTERS["AGE48_BE50_LOCK20_TP150_SL30"] = {
    "chain": "solana", "min_age_hours": 24, "max_age_hours": 48,
}
SHADOW_STRATEGIES.append("AGE48_BE50_LOCK20_TP150_SL30")

# ---- Bloc D: AGE × LOCK ETH (4) ----
STRATEGIES["AGE24_ETH_BE25_LOCK10_TP80_SL40_T2H"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.60, "horizon_min": 120,
     "be_activation": 0.25, "be_lock_pct": 0.10, "label": "main"},
]
STRATEGY_FILTERS["AGE24_ETH_BE25_LOCK10_TP80_SL40_T2H"] = {
    "chain": "ethereum", "min_age_hours": 12, "max_age_hours": 24,
}
SHADOW_STRATEGIES.append("AGE24_ETH_BE25_LOCK10_TP80_SL40_T2H")

STRATEGIES["AGE48_ETH_BE25_LOCK10_TP80_SL40_T2H"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.60, "horizon_min": 120,
     "be_activation": 0.25, "be_lock_pct": 0.10, "label": "main"},
]
STRATEGY_FILTERS["AGE48_ETH_BE25_LOCK10_TP80_SL40_T2H"] = {
    "chain": "ethereum", "min_age_hours": 24, "max_age_hours": 48,
}
SHADOW_STRATEGIES.append("AGE48_ETH_BE25_LOCK10_TP80_SL40_T2H")

STRATEGIES["AGE24_ETH_BE50_LOCK20_TP150_SL40"] = [
    {"pct": 1.0, "tp_mult": 2.50, "sl_mult": 0.60, "horizon_min": 240,
     "be_activation": 0.50, "be_lock_pct": 0.20, "label": "main"},
]
STRATEGY_FILTERS["AGE24_ETH_BE50_LOCK20_TP150_SL40"] = {
    "chain": "ethereum", "min_age_hours": 12, "max_age_hours": 24,
}
SHADOW_STRATEGIES.append("AGE24_ETH_BE50_LOCK20_TP150_SL40")

STRATEGIES["AGE48_ETH_BE50_LOCK20_TP150_SL40"] = [
    {"pct": 1.0, "tp_mult": 2.50, "sl_mult": 0.60, "horizon_min": 240,
     "be_activation": 0.50, "be_lock_pct": 0.20, "label": "main"},
]
STRATEGY_FILTERS["AGE48_ETH_BE50_LOCK20_TP150_SL40"] = {
    "chain": "ethereum", "min_age_hours": 24, "max_age_hours": 48,
}
SHADOW_STRATEGIES.append("AGE48_ETH_BE50_LOCK20_TP150_SL40")

# ---- Bloc E: AGE × SCALP × ETH (4) ----
STRATEGIES["AGE24_ETH_SCALP_TP15_SL20_S35"] = [
    {"pct": 1.0, "tp_mult": 1.15, "sl_mult": 0.80, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["AGE24_ETH_SCALP_TP15_SL20_S35"] = {
    "chain": "ethereum", "min_age_hours": 12, "max_age_hours": 24, "min_rt_score": 35,
}
SHADOW_STRATEGIES.append("AGE24_ETH_SCALP_TP15_SL20_S35")

STRATEGIES["AGE48_ETH_SCALP_TP15_SL20_S35"] = [
    {"pct": 1.0, "tp_mult": 1.15, "sl_mult": 0.80, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["AGE48_ETH_SCALP_TP15_SL20_S35"] = {
    "chain": "ethereum", "min_age_hours": 24, "max_age_hours": 48, "min_rt_score": 35,
}
SHADOW_STRATEGIES.append("AGE48_ETH_SCALP_TP15_SL20_S35")

STRATEGIES["AGE24_ETH_SCALP_TP20_SL10_S30"] = [
    {"pct": 1.0, "tp_mult": 1.20, "sl_mult": 0.90, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["AGE24_ETH_SCALP_TP20_SL10_S30"] = {
    "chain": "ethereum", "min_age_hours": 12, "max_age_hours": 24, "min_rt_score": 30,
}
SHADOW_STRATEGIES.append("AGE24_ETH_SCALP_TP20_SL10_S30")

STRATEGIES["AGE48_ETH_SCALP_TP20_SL10_S30"] = [
    {"pct": 1.0, "tp_mult": 1.20, "sl_mult": 0.90, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["AGE48_ETH_SCALP_TP20_SL10_S30"] = {
    "chain": "ethereum", "min_age_hours": 24, "max_age_hours": 48, "min_rt_score": 30,
}
SHADOW_STRATEGIES.append("AGE48_ETH_SCALP_TP20_SL10_S30")

# ---- Bloc F: AGE × BE × ETH (4) ----
STRATEGIES["AGE24_ETH_BE25_TP80_SL40_T2H"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.60, "horizon_min": 120,
     "be_activation": 0.25, "label": "main"},
]
STRATEGY_FILTERS["AGE24_ETH_BE25_TP80_SL40_T2H"] = {
    "chain": "ethereum", "min_age_hours": 12, "max_age_hours": 24,
}
SHADOW_STRATEGIES.append("AGE24_ETH_BE25_TP80_SL40_T2H")

STRATEGIES["AGE48_ETH_BE25_TP80_SL40_T2H"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.60, "horizon_min": 120,
     "be_activation": 0.25, "label": "main"},
]
STRATEGY_FILTERS["AGE48_ETH_BE25_TP80_SL40_T2H"] = {
    "chain": "ethereum", "min_age_hours": 24, "max_age_hours": 48,
}
SHADOW_STRATEGIES.append("AGE48_ETH_BE25_TP80_SL40_T2H")

STRATEGIES["AGE24_ETH_BE30_TP100_SL40"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.60, "horizon_min": 240,
     "be_activation": 0.30, "label": "main"},
]
STRATEGY_FILTERS["AGE24_ETH_BE30_TP100_SL40"] = {
    "chain": "ethereum", "min_age_hours": 12, "max_age_hours": 24,
}
SHADOW_STRATEGIES.append("AGE24_ETH_BE30_TP100_SL40")

STRATEGIES["AGE48_ETH_BE30_TP100_SL40"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.60, "horizon_min": 240,
     "be_activation": 0.30, "label": "main"},
]
STRATEGY_FILTERS["AGE48_ETH_BE30_TP100_SL40"] = {
    "chain": "ethereum", "min_age_hours": 24, "max_age_hours": 48,
}
SHADOW_STRATEGIES.append("AGE48_ETH_BE30_TP100_SL40")

# ---- Bloc G: ETH SLOW family (5) — long horizon, no BE ----
STRATEGIES["ETH_SLOW4H_TP100_SL50"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.50, "horizon_min": 240, "label": "main"},
]
STRATEGY_FILTERS["ETH_SLOW4H_TP100_SL50"] = {"chain": "ethereum"}
SHADOW_STRATEGIES.append("ETH_SLOW4H_TP100_SL50")

STRATEGIES["ETH_SLOW4H_TP150_SL40"] = [
    {"pct": 1.0, "tp_mult": 2.50, "sl_mult": 0.60, "horizon_min": 240, "label": "main"},
]
STRATEGY_FILTERS["ETH_SLOW4H_TP150_SL40"] = {"chain": "ethereum"}
SHADOW_STRATEGIES.append("ETH_SLOW4H_TP150_SL40")

STRATEGIES["ETH_SLOW6H_TP150_SL50"] = [
    {"pct": 1.0, "tp_mult": 2.50, "sl_mult": 0.50, "horizon_min": 360, "label": "main"},
]
STRATEGY_FILTERS["ETH_SLOW6H_TP150_SL50"] = {"chain": "ethereum"}
SHADOW_STRATEGIES.append("ETH_SLOW6H_TP150_SL50")

STRATEGIES["ETH_SLOW6H_TP200_SL40"] = [
    {"pct": 1.0, "tp_mult": 3.00, "sl_mult": 0.60, "horizon_min": 360, "label": "main"},
]
STRATEGY_FILTERS["ETH_SLOW6H_TP200_SL40"] = {"chain": "ethereum"}
SHADOW_STRATEGIES.append("ETH_SLOW6H_TP200_SL40")

STRATEGIES["ETH_SLOW6H_TP200_SL40_NZ_S40"] = [
    {"pct": 1.0, "tp_mult": 3.00, "sl_mult": 0.60, "horizon_min": 360, "label": "main"},
]
STRATEGY_FILTERS["ETH_SLOW6H_TP200_SL40_NZ_S40"] = {
    "chain": "ethereum", "min_liquidity_usd": 1, "min_rt_score": 40,
}
SHADOW_STRATEGIES.append("ETH_SLOW6H_TP200_SL40_NZ_S40")

# ---- Bloc H: ETH × SCALP additional filter combos (4) ----
STRATEGIES["ETH_SCALP_TP10_SL10_S30"] = [
    {"pct": 1.0, "tp_mult": 1.10, "sl_mult": 0.90, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["ETH_SCALP_TP10_SL10_S30"] = {"chain": "ethereum", "min_rt_score": 30}
SHADOW_STRATEGIES.append("ETH_SCALP_TP10_SL10_S30")

STRATEGIES["ETH_SCALP_TP15_NOSL_S40"] = [
    {"pct": 1.0, "tp_mult": 1.15, "sl_mult": 0.20, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["ETH_SCALP_TP15_NOSL_S40"] = {"chain": "ethereum", "min_rt_score": 40}
SHADOW_STRATEGIES.append("ETH_SCALP_TP15_NOSL_S40")

STRATEGIES["ETH_SCALP_TP20_NOSL_S40"] = [
    {"pct": 1.0, "tp_mult": 1.20, "sl_mult": 0.20, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["ETH_SCALP_TP20_NOSL_S40"] = {"chain": "ethereum", "min_rt_score": 40}
SHADOW_STRATEGIES.append("ETH_SCALP_TP20_NOSL_S40")

STRATEGIES["ETH_SCALP_TP25_SL15_S35"] = [
    {"pct": 1.0, "tp_mult": 1.25, "sl_mult": 0.85, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["ETH_SCALP_TP25_SL15_S35"] = {"chain": "ethereum", "min_rt_score": 35}
SHADOW_STRATEGIES.append("ETH_SCALP_TP25_SL15_S35")

# ---- Bloc I: ETH let-it-run TP200/TP300 with filters (3) ----
STRATEGIES["ETH_TP200_SL40_4H_NZ_S40"] = [
    {"pct": 1.0, "tp_mult": 3.00, "sl_mult": 0.60, "horizon_min": 240, "label": "main"},
]
STRATEGY_FILTERS["ETH_TP200_SL40_4H_NZ_S40"] = {
    "chain": "ethereum", "min_liquidity_usd": 1, "min_rt_score": 40,
}
SHADOW_STRATEGIES.append("ETH_TP200_SL40_4H_NZ_S40")

STRATEGIES["ETH_TP300_SL50_4H"] = [
    {"pct": 1.0, "tp_mult": 4.00, "sl_mult": 0.50, "horizon_min": 240, "label": "main"},
]
STRATEGY_FILTERS["ETH_TP300_SL50_4H"] = {"chain": "ethereum"}
SHADOW_STRATEGIES.append("ETH_TP300_SL50_4H")

STRATEGIES["ETH_TP300_SL40_4H_MCAP_S40"] = [
    {"pct": 1.0, "tp_mult": 4.00, "sl_mult": 0.60, "horizon_min": 240, "label": "main"},
]
STRATEGY_FILTERS["ETH_TP300_SL40_4H_MCAP_S40"] = {
    "chain": "ethereum", "min_mcap": 30_000, "max_mcap": 500_000, "min_rt_score": 40,
}
SHADOW_STRATEGIES.append("ETH_TP300_SL40_4H_MCAP_S40")

# ==========================================================================
# v14e.30 (Apr 26 PM, suite) — SOL balance: 19 more shadows to match the
# combinatorial depth applied to ETH. Fills the remaining SOL gaps:
#   AGE × SCALP × S30/S40    (current AGE×SCALP only has S35)
#   AGE × LOCK × score-gated (triple combo, never tested)
#   SOL LOCK × let-it-run    (LOCK was only 30-60min horizon)
#   SCALP × NZ filter        (SCALP had S30/35/40 but no NZ)
#   SOL TP200/TP300 × MCAP/NZ (let-it-run with quality gates)
# ==========================================================================

# ---- Bloc J: AGE × SCALP × S30/S40 SOL (4) ----
STRATEGIES["AGE24_SCALP_TP20_SL10_S30"] = [
    {"pct": 1.0, "tp_mult": 1.20, "sl_mult": 0.90, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["AGE24_SCALP_TP20_SL10_S30"] = {
    "chain": "solana", "min_age_hours": 12, "max_age_hours": 24, "min_rt_score": 30,
}
SHADOW_STRATEGIES.append("AGE24_SCALP_TP20_SL10_S30")

STRATEGIES["AGE48_SCALP_TP20_SL10_S30"] = [
    {"pct": 1.0, "tp_mult": 1.20, "sl_mult": 0.90, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["AGE48_SCALP_TP20_SL10_S30"] = {
    "chain": "solana", "min_age_hours": 24, "max_age_hours": 48, "min_rt_score": 30,
}
SHADOW_STRATEGIES.append("AGE48_SCALP_TP20_SL10_S30")

STRATEGIES["AGE24_SCALP_TP15_SL15_S40"] = [
    {"pct": 1.0, "tp_mult": 1.15, "sl_mult": 0.85, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["AGE24_SCALP_TP15_SL15_S40"] = {
    "chain": "solana", "min_age_hours": 12, "max_age_hours": 24, "min_rt_score": 40,
}
SHADOW_STRATEGIES.append("AGE24_SCALP_TP15_SL15_S40")

STRATEGIES["AGE48_SCALP_TP15_SL15_S40"] = [
    {"pct": 1.0, "tp_mult": 1.15, "sl_mult": 0.85, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["AGE48_SCALP_TP15_SL15_S40"] = {
    "chain": "solana", "min_age_hours": 24, "max_age_hours": 48, "min_rt_score": 40,
}
SHADOW_STRATEGIES.append("AGE48_SCALP_TP15_SL15_S40")

# ---- Bloc K: AGE × LOCK × score filter SOL (4) — triple combo ----
STRATEGIES["AGE24_BE25_LOCK10_TP80_SL30_S35"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.10, "label": "main"},
]
STRATEGY_FILTERS["AGE24_BE25_LOCK10_TP80_SL30_S35"] = {
    "chain": "solana", "min_age_hours": 12, "max_age_hours": 24, "min_rt_score": 35,
}
SHADOW_STRATEGIES.append("AGE24_BE25_LOCK10_TP80_SL30_S35")

STRATEGIES["AGE48_BE25_LOCK10_TP80_SL30_S35"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.10, "label": "main"},
]
STRATEGY_FILTERS["AGE48_BE25_LOCK10_TP80_SL30_S35"] = {
    "chain": "solana", "min_age_hours": 24, "max_age_hours": 48, "min_rt_score": 35,
}
SHADOW_STRATEGIES.append("AGE48_BE25_LOCK10_TP80_SL30_S35")

STRATEGIES["AGE24_BE50_LOCK20_TP150_SL30_S40"] = [
    {"pct": 1.0, "tp_mult": 2.50, "sl_mult": 0.70, "horizon_min": 60,
     "be_activation": 0.50, "be_lock_pct": 0.20, "label": "main"},
]
STRATEGY_FILTERS["AGE24_BE50_LOCK20_TP150_SL30_S40"] = {
    "chain": "solana", "min_age_hours": 12, "max_age_hours": 24, "min_rt_score": 40,
}
SHADOW_STRATEGIES.append("AGE24_BE50_LOCK20_TP150_SL30_S40")

STRATEGIES["AGE48_BE50_LOCK20_TP150_SL30_S40"] = [
    {"pct": 1.0, "tp_mult": 2.50, "sl_mult": 0.70, "horizon_min": 60,
     "be_activation": 0.50, "be_lock_pct": 0.20, "label": "main"},
]
STRATEGY_FILTERS["AGE48_BE50_LOCK20_TP150_SL30_S40"] = {
    "chain": "solana", "min_age_hours": 24, "max_age_hours": 48, "min_rt_score": 40,
}
SHADOW_STRATEGIES.append("AGE48_BE50_LOCK20_TP150_SL30_S40")

# ---- Bloc L: SOL LOCK × let-it-run TP200/TP300 (4) ----
STRATEGIES["BE25_LOCK10_TP200_SL40_4H"] = [
    {"pct": 1.0, "tp_mult": 3.00, "sl_mult": 0.60, "horizon_min": 240,
     "be_activation": 0.25, "be_lock_pct": 0.10, "label": "main"},
]
SHADOW_STRATEGIES.append("BE25_LOCK10_TP200_SL40_4H")

STRATEGIES["BE25_LOCK15_TP200_SL40_4H_NZ_S40"] = [
    {"pct": 1.0, "tp_mult": 3.00, "sl_mult": 0.60, "horizon_min": 240,
     "be_activation": 0.25, "be_lock_pct": 0.15, "label": "main"},
]
STRATEGY_FILTERS["BE25_LOCK15_TP200_SL40_4H_NZ_S40"] = {
    "chain": "solana", "min_liquidity_usd": 1, "min_rt_score": 40,
}
SHADOW_STRATEGIES.append("BE25_LOCK15_TP200_SL40_4H_NZ_S40")

STRATEGIES["BE50_LOCK25_TP300_SL40_4H"] = [
    {"pct": 1.0, "tp_mult": 4.00, "sl_mult": 0.60, "horizon_min": 240,
     "be_activation": 0.50, "be_lock_pct": 0.25, "label": "main"},
]
SHADOW_STRATEGIES.append("BE50_LOCK25_TP300_SL40_4H")

STRATEGIES["BE50_LOCK25_TP300_SL40_4H_MCAP_S40"] = [
    {"pct": 1.0, "tp_mult": 4.00, "sl_mult": 0.60, "horizon_min": 240,
     "be_activation": 0.50, "be_lock_pct": 0.25, "label": "main"},
]
STRATEGY_FILTERS["BE50_LOCK25_TP300_SL40_4H_MCAP_S40"] = {
    "chain": "solana", "min_mcap": 30_000, "max_mcap": 500_000, "min_rt_score": 40,
}
SHADOW_STRATEGIES.append("BE50_LOCK25_TP300_SL40_4H_MCAP_S40")

# ---- Bloc M: SCALP × NZ filter SOL (4) ----
STRATEGIES["SCALP_TP15_SL20_S35_NZ"] = [
    {"pct": 1.0, "tp_mult": 1.15, "sl_mult": 0.80, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["SCALP_TP15_SL20_S35_NZ"] = {
    "chain": "solana", "min_liquidity_usd": 1, "min_rt_score": 35,
}
SHADOW_STRATEGIES.append("SCALP_TP15_SL20_S35_NZ")

STRATEGIES["SCALP_TP20_SL10_S30_NZ"] = [
    {"pct": 1.0, "tp_mult": 1.20, "sl_mult": 0.90, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["SCALP_TP20_SL10_S30_NZ"] = {
    "chain": "solana", "min_liquidity_usd": 1, "min_rt_score": 30,
}
SHADOW_STRATEGIES.append("SCALP_TP20_SL10_S30_NZ")

STRATEGIES["SCALP_TP15_NOSL_S35_NZ"] = [
    {"pct": 1.0, "tp_mult": 1.15, "sl_mult": 0.20, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["SCALP_TP15_NOSL_S35_NZ"] = {
    "chain": "solana", "min_liquidity_usd": 1, "min_rt_score": 35,
}
SHADOW_STRATEGIES.append("SCALP_TP15_NOSL_S35_NZ")

STRATEGIES["SCALP_TP20_NOSL_S40_NZ"] = [
    {"pct": 1.0, "tp_mult": 1.20, "sl_mult": 0.20, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["SCALP_TP20_NOSL_S40_NZ"] = {
    "chain": "solana", "min_liquidity_usd": 1, "min_rt_score": 40,
}
SHADOW_STRATEGIES.append("SCALP_TP20_NOSL_S40_NZ")

# ---- Bloc N: SOL TP200/TP300 × MCAP/NZ (3) ----
STRATEGIES["TP200_SL40_4H_MCAP_S40"] = [
    {"pct": 1.0, "tp_mult": 3.00, "sl_mult": 0.60, "horizon_min": 240, "label": "main"},
]
STRATEGY_FILTERS["TP200_SL40_4H_MCAP_S40"] = {
    "chain": "solana", "min_mcap": 30_000, "max_mcap": 500_000, "min_rt_score": 40,
}
SHADOW_STRATEGIES.append("TP200_SL40_4H_MCAP_S40")

STRATEGIES["TP300_SL50_4H_NZ_S40"] = [
    {"pct": 1.0, "tp_mult": 4.00, "sl_mult": 0.50, "horizon_min": 240, "label": "main"},
]
STRATEGY_FILTERS["TP300_SL50_4H_NZ_S40"] = {
    "chain": "solana", "min_liquidity_usd": 1, "min_rt_score": 40,
}
SHADOW_STRATEGIES.append("TP300_SL50_4H_NZ_S40")

STRATEGIES["TP300_SL40_6H_MCAP_S40"] = [
    {"pct": 1.0, "tp_mult": 4.00, "sl_mult": 0.60, "horizon_min": 360, "label": "main"},
]
STRATEGY_FILTERS["TP300_SL40_6H_MCAP_S40"] = {
    "chain": "solana", "min_mcap": 30_000, "max_mcap": 500_000, "min_rt_score": 40,
}
SHADOW_STRATEGIES.append("TP300_SL40_6H_MCAP_S40")

# ==========================================================================
# v14e.31 (Apr 26 PM, suite) — expansion around mega-sweep ETH winner.
# Run 24958587941 finished: top 30 robust ALL = FAST_TP100_SL20 × AGE12 band
# (12-24h tokens). +34.35% avg / WR 60% / $232/day / cross-regime robust.
# Independent of smoothing/source = signal très propre.
#
# ~25 variants clustered around this winning combo to find the exact sweet
# spot (TP variants, SL variants, score gates, NZ filter, horizons, LOCK
# combo, ETH parallels). All shadow-only.
# ==========================================================================

# ---- Bloc P: AGE24/AGE48 FAST_TP100_SL20 + score gates SOL (6) ----
STRATEGIES["AGE24_FAST_TP100_SL20_S30"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.80, "horizon_min": 30, "label": "main"},
]
STRATEGY_FILTERS["AGE24_FAST_TP100_SL20_S30"] = {
    "chain": "solana", "min_age_hours": 12, "max_age_hours": 24, "min_rt_score": 30,
}
SHADOW_STRATEGIES.append("AGE24_FAST_TP100_SL20_S30")

STRATEGIES["AGE24_FAST_TP100_SL20_S35"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.80, "horizon_min": 30, "label": "main"},
]
STRATEGY_FILTERS["AGE24_FAST_TP100_SL20_S35"] = {
    "chain": "solana", "min_age_hours": 12, "max_age_hours": 24, "min_rt_score": 35,
}
SHADOW_STRATEGIES.append("AGE24_FAST_TP100_SL20_S35")

STRATEGIES["AGE24_FAST_TP100_SL20_S40"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.80, "horizon_min": 30, "label": "main"},
]
STRATEGY_FILTERS["AGE24_FAST_TP100_SL20_S40"] = {
    "chain": "solana", "min_age_hours": 12, "max_age_hours": 24, "min_rt_score": 40,
}
SHADOW_STRATEGIES.append("AGE24_FAST_TP100_SL20_S40")

STRATEGIES["AGE24_FAST_TP100_SL20_NZ"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.80, "horizon_min": 30, "label": "main"},
]
STRATEGY_FILTERS["AGE24_FAST_TP100_SL20_NZ"] = {
    "chain": "solana", "min_age_hours": 12, "max_age_hours": 24, "min_liquidity_usd": 1,
}
SHADOW_STRATEGIES.append("AGE24_FAST_TP100_SL20_NZ")

STRATEGIES["AGE24_FAST_TP100_SL20_NZ_S40"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.80, "horizon_min": 30, "label": "main"},
]
STRATEGY_FILTERS["AGE24_FAST_TP100_SL20_NZ_S40"] = {
    "chain": "solana", "min_age_hours": 12, "max_age_hours": 24,
    "min_liquidity_usd": 1, "min_rt_score": 40,
}
SHADOW_STRATEGIES.append("AGE24_FAST_TP100_SL20_NZ_S40")

STRATEGIES["AGE48_FAST_TP100_SL20_NZ_S40"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.80, "horizon_min": 30, "label": "main"},
]
STRATEGY_FILTERS["AGE48_FAST_TP100_SL20_NZ_S40"] = {
    "chain": "solana", "min_age_hours": 24, "max_age_hours": 48,
    "min_liquidity_usd": 1, "min_rt_score": 40,
}
SHADOW_STRATEGIES.append("AGE48_FAST_TP100_SL20_NZ_S40")

# ---- Bloc Q: TP/SL variants of the winner SOL (6) ----
STRATEGIES["AGE24_FAST_TP80_SL20"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.80, "horizon_min": 30, "label": "main"},
]
STRATEGY_FILTERS["AGE24_FAST_TP80_SL20"] = {
    "chain": "solana", "min_age_hours": 12, "max_age_hours": 24,
}
SHADOW_STRATEGIES.append("AGE24_FAST_TP80_SL20")

STRATEGIES["AGE24_FAST_TP120_SL20"] = [
    {"pct": 1.0, "tp_mult": 2.20, "sl_mult": 0.80, "horizon_min": 30, "label": "main"},
]
STRATEGY_FILTERS["AGE24_FAST_TP120_SL20"] = {
    "chain": "solana", "min_age_hours": 12, "max_age_hours": 24,
}
SHADOW_STRATEGIES.append("AGE24_FAST_TP120_SL20")

STRATEGIES["AGE24_FAST_TP150_SL20"] = [
    {"pct": 1.0, "tp_mult": 2.50, "sl_mult": 0.80, "horizon_min": 30, "label": "main"},
]
STRATEGY_FILTERS["AGE24_FAST_TP150_SL20"] = {
    "chain": "solana", "min_age_hours": 12, "max_age_hours": 24,
}
SHADOW_STRATEGIES.append("AGE24_FAST_TP150_SL20")

STRATEGIES["AGE24_FAST_TP100_SL10"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.90, "horizon_min": 30, "label": "main"},
]
STRATEGY_FILTERS["AGE24_FAST_TP100_SL10"] = {
    "chain": "solana", "min_age_hours": 12, "max_age_hours": 24,
}
SHADOW_STRATEGIES.append("AGE24_FAST_TP100_SL10")

STRATEGIES["AGE24_FAST_TP100_SL30"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.70, "horizon_min": 30, "label": "main"},
]
STRATEGY_FILTERS["AGE24_FAST_TP100_SL30"] = {
    "chain": "solana", "min_age_hours": 12, "max_age_hours": 24,
}
SHADOW_STRATEGIES.append("AGE24_FAST_TP100_SL30")

STRATEGIES["AGE24_FAST_TP100_NOSL"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.20, "horizon_min": 30, "label": "main"},
]
STRATEGY_FILTERS["AGE24_FAST_TP100_NOSL"] = {
    "chain": "solana", "min_age_hours": 12, "max_age_hours": 24,
}
SHADOW_STRATEGIES.append("AGE24_FAST_TP100_NOSL")

# ---- Bloc R: Horizon variants on the winner SOL (3) ----
STRATEGIES["AGE24_FAST60_TP100_SL20"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.80, "horizon_min": 60, "label": "main"},
]
STRATEGY_FILTERS["AGE24_FAST60_TP100_SL20"] = {
    "chain": "solana", "min_age_hours": 12, "max_age_hours": 24,
}
SHADOW_STRATEGIES.append("AGE24_FAST60_TP100_SL20")

STRATEGIES["AGE24_TP100_SL20_T2H"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.80, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["AGE24_TP100_SL20_T2H"] = {
    "chain": "solana", "min_age_hours": 12, "max_age_hours": 24,
}
SHADOW_STRATEGIES.append("AGE24_TP100_SL20_T2H")

STRATEGIES["AGE24_FAST15_TP100_SL20"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.80, "horizon_min": 15, "label": "main"},
]
STRATEGY_FILTERS["AGE24_FAST15_TP100_SL20"] = {
    "chain": "solana", "min_age_hours": 12, "max_age_hours": 24,
}
SHADOW_STRATEGIES.append("AGE24_FAST15_TP100_SL20")

# ---- Bloc S: AGE × LOCK on the FAST_TP100_SL20 base SOL (3) ----
STRATEGIES["AGE24_BE25_LOCK10_TP100_SL20"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.80, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.10, "label": "main"},
]
STRATEGY_FILTERS["AGE24_BE25_LOCK10_TP100_SL20"] = {
    "chain": "solana", "min_age_hours": 12, "max_age_hours": 24,
}
SHADOW_STRATEGIES.append("AGE24_BE25_LOCK10_TP100_SL20")

STRATEGIES["AGE24_BE25_LOCK15_TP100_SL20"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.80, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.15, "label": "main"},
]
STRATEGY_FILTERS["AGE24_BE25_LOCK15_TP100_SL20"] = {
    "chain": "solana", "min_age_hours": 12, "max_age_hours": 24,
}
SHADOW_STRATEGIES.append("AGE24_BE25_LOCK15_TP100_SL20")

STRATEGIES["AGE24_BE50_LOCK20_TP100_SL20"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.80, "horizon_min": 30,
     "be_activation": 0.50, "be_lock_pct": 0.20, "label": "main"},
]
STRATEGY_FILTERS["AGE24_BE50_LOCK20_TP100_SL20"] = {
    "chain": "solana", "min_age_hours": 12, "max_age_hours": 24,
}
SHADOW_STRATEGIES.append("AGE24_BE50_LOCK20_TP100_SL20")

# ---- Bloc T: ETH parallels of the winning combo (8) ----
STRATEGIES["AGE24_ETH_FAST_TP100_SL20_S35"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.80, "horizon_min": 30, "label": "main"},
]
STRATEGY_FILTERS["AGE24_ETH_FAST_TP100_SL20_S35"] = {
    "chain": "ethereum", "min_age_hours": 12, "max_age_hours": 24, "min_rt_score": 35,
}
SHADOW_STRATEGIES.append("AGE24_ETH_FAST_TP100_SL20_S35")

STRATEGIES["AGE24_ETH_FAST_TP100_SL20_NZ_S40"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.80, "horizon_min": 30, "label": "main"},
]
STRATEGY_FILTERS["AGE24_ETH_FAST_TP100_SL20_NZ_S40"] = {
    "chain": "ethereum", "min_age_hours": 12, "max_age_hours": 24,
    "min_liquidity_usd": 1, "min_rt_score": 40,
}
SHADOW_STRATEGIES.append("AGE24_ETH_FAST_TP100_SL20_NZ_S40")

STRATEGIES["AGE48_ETH_FAST_TP100_SL20_NZ_S40"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.80, "horizon_min": 30, "label": "main"},
]
STRATEGY_FILTERS["AGE48_ETH_FAST_TP100_SL20_NZ_S40"] = {
    "chain": "ethereum", "min_age_hours": 24, "max_age_hours": 48,
    "min_liquidity_usd": 1, "min_rt_score": 40,
}
SHADOW_STRATEGIES.append("AGE48_ETH_FAST_TP100_SL20_NZ_S40")

STRATEGIES["AGE24_ETH_FAST_TP120_SL20"] = [
    {"pct": 1.0, "tp_mult": 2.20, "sl_mult": 0.80, "horizon_min": 30, "label": "main"},
]
STRATEGY_FILTERS["AGE24_ETH_FAST_TP120_SL20"] = {
    "chain": "ethereum", "min_age_hours": 12, "max_age_hours": 24,
}
SHADOW_STRATEGIES.append("AGE24_ETH_FAST_TP120_SL20")

STRATEGIES["AGE24_ETH_FAST_TP150_SL20"] = [
    {"pct": 1.0, "tp_mult": 2.50, "sl_mult": 0.80, "horizon_min": 30, "label": "main"},
]
STRATEGY_FILTERS["AGE24_ETH_FAST_TP150_SL20"] = {
    "chain": "ethereum", "min_age_hours": 12, "max_age_hours": 24,
}
SHADOW_STRATEGIES.append("AGE24_ETH_FAST_TP150_SL20")

STRATEGIES["AGE24_ETH_FAST_TP100_SL30"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.70, "horizon_min": 30, "label": "main"},
]
STRATEGY_FILTERS["AGE24_ETH_FAST_TP100_SL30"] = {
    "chain": "ethereum", "min_age_hours": 12, "max_age_hours": 24,
}
SHADOW_STRATEGIES.append("AGE24_ETH_FAST_TP100_SL30")

STRATEGIES["AGE24_ETH_FAST60_TP100_SL20"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.80, "horizon_min": 60, "label": "main"},
]
STRATEGY_FILTERS["AGE24_ETH_FAST60_TP100_SL20"] = {
    "chain": "ethereum", "min_age_hours": 12, "max_age_hours": 24,
}
SHADOW_STRATEGIES.append("AGE24_ETH_FAST60_TP100_SL20")

STRATEGIES["AGE24_ETH_BE25_LOCK10_TP100_SL20"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.80, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.10, "label": "main"},
]
STRATEGY_FILTERS["AGE24_ETH_BE25_LOCK10_TP100_SL20"] = {
    "chain": "ethereum", "min_age_hours": 12, "max_age_hours": 24,
}
SHADOW_STRATEGIES.append("AGE24_ETH_BE25_LOCK10_TP100_SL20")

# ==========================================================================
# v14e.32 (Apr 26 PM, Run 2 verdict) — winner cluster expansion.
# Run 24959544627 finished: top 30 robust ETH dominated by 3 LOCK strats:
#   #1  BE25_LOCK10_TP100_SL30  18 cfg  +37.40% avg  WR 76.67%  $264/d
#   #2  BE25_LOCK15_TP100_SL30   8 cfg  +36.78% avg  WR 76.67%  $249/d
#   #3  BE15_LOCK10_TP80_SL30    4 cfg  +36.68% avg  WR 80.00%  $247/d
# All on AGE12 band (12-24h) × median_5 × lazy_med polling. NONE/NOZEROLIQ
# both work. Run 2 used position=$10 (lowballed), real +12pp expected at $50.
#
# 35 new shadows:
#  - ETH versions of the 3 winners (Bloc U) — exact strat name parity SOL/ETH
#  - AGE24/AGE48 explicit clones SOL+ETH (Bloc V)
#  - Filter variants on the top winner BE25_LOCK10_TP100_SL30 (Bloc W)
#  - BE-range variants BE15/BE20/BE30 × LOCK10 (Bloc X)
#  - SL variants BE25_LOCK10_TP100 × SL20/40 (Bloc Y)
#  - Wider TP variants (TP120/TP150 + LOCK10/15) (Bloc Z)
# ==========================================================================

# ---- Bloc U: ETH versions of the 3 Run 2 winners (3) ----
STRATEGIES["ETH_BE25_LOCK10_TP100_SL30"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.10, "label": "main"},
]
STRATEGY_FILTERS["ETH_BE25_LOCK10_TP100_SL30"] = {"chain": "ethereum"}
SHADOW_STRATEGIES.append("ETH_BE25_LOCK10_TP100_SL30")

STRATEGIES["ETH_BE25_LOCK15_TP100_SL30"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.15, "label": "main"},
]
STRATEGY_FILTERS["ETH_BE25_LOCK15_TP100_SL30"] = {"chain": "ethereum"}
SHADOW_STRATEGIES.append("ETH_BE25_LOCK15_TP100_SL30")

STRATEGIES["ETH_BE15_LOCK10_TP80_SL30"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.15, "be_lock_pct": 0.10, "label": "main"},
]
STRATEGY_FILTERS["ETH_BE15_LOCK10_TP80_SL30"] = {"chain": "ethereum"}
SHADOW_STRATEGIES.append("ETH_BE15_LOCK10_TP80_SL30")

# ---- Bloc V: AGE24 explicit clones of winners SOL+ETH (6) ----
STRATEGIES["AGE24_BE25_LOCK10_TP100_SL30"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.10, "label": "main"},
]
STRATEGY_FILTERS["AGE24_BE25_LOCK10_TP100_SL30"] = {
    "chain": "solana", "min_age_hours": 12, "max_age_hours": 24,
}
SHADOW_STRATEGIES.append("AGE24_BE25_LOCK10_TP100_SL30")

STRATEGIES["AGE24_BE25_LOCK15_TP100_SL30"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.15, "label": "main"},
]
STRATEGY_FILTERS["AGE24_BE25_LOCK15_TP100_SL30"] = {
    "chain": "solana", "min_age_hours": 12, "max_age_hours": 24,
}
SHADOW_STRATEGIES.append("AGE24_BE25_LOCK15_TP100_SL30")

STRATEGIES["AGE24_BE15_LOCK10_TP80_SL30"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.15, "be_lock_pct": 0.10, "label": "main"},
]
STRATEGY_FILTERS["AGE24_BE15_LOCK10_TP80_SL30"] = {
    "chain": "solana", "min_age_hours": 12, "max_age_hours": 24,
}
SHADOW_STRATEGIES.append("AGE24_BE15_LOCK10_TP80_SL30")

STRATEGIES["AGE24_ETH_BE25_LOCK10_TP100_SL30"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.10, "label": "main"},
]
STRATEGY_FILTERS["AGE24_ETH_BE25_LOCK10_TP100_SL30"] = {
    "chain": "ethereum", "min_age_hours": 12, "max_age_hours": 24,
}
SHADOW_STRATEGIES.append("AGE24_ETH_BE25_LOCK10_TP100_SL30")

STRATEGIES["AGE24_ETH_BE25_LOCK15_TP100_SL30"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.15, "label": "main"},
]
STRATEGY_FILTERS["AGE24_ETH_BE25_LOCK15_TP100_SL30"] = {
    "chain": "ethereum", "min_age_hours": 12, "max_age_hours": 24,
}
SHADOW_STRATEGIES.append("AGE24_ETH_BE25_LOCK15_TP100_SL30")

STRATEGIES["AGE24_ETH_BE15_LOCK10_TP80_SL30"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.15, "be_lock_pct": 0.10, "label": "main"},
]
STRATEGY_FILTERS["AGE24_ETH_BE15_LOCK10_TP80_SL30"] = {
    "chain": "ethereum", "min_age_hours": 12, "max_age_hours": 24,
}
SHADOW_STRATEGIES.append("AGE24_ETH_BE15_LOCK10_TP80_SL30")

# ---- Bloc W: filter variants on top winner BE25_LOCK10_TP100_SL30 (8) ----
STRATEGIES["BE25_LOCK10_TP100_SL30_S30"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.10, "label": "main"},
]
STRATEGY_FILTERS["BE25_LOCK10_TP100_SL30_S30"] = {"chain": "solana", "min_rt_score": 30}
SHADOW_STRATEGIES.append("BE25_LOCK10_TP100_SL30_S30")

STRATEGIES["BE25_LOCK10_TP100_SL30_S35"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.10, "label": "main"},
]
STRATEGY_FILTERS["BE25_LOCK10_TP100_SL30_S35"] = {"chain": "solana", "min_rt_score": 35}
SHADOW_STRATEGIES.append("BE25_LOCK10_TP100_SL30_S35")

STRATEGIES["BE25_LOCK10_TP100_SL30_S40"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.10, "label": "main"},
]
STRATEGY_FILTERS["BE25_LOCK10_TP100_SL30_S40"] = {"chain": "solana", "min_rt_score": 40}
SHADOW_STRATEGIES.append("BE25_LOCK10_TP100_SL30_S40")

STRATEGIES["BE25_LOCK10_TP100_SL30_NZ_S40"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.10, "label": "main"},
]
STRATEGY_FILTERS["BE25_LOCK10_TP100_SL30_NZ_S40"] = {
    "chain": "solana", "min_liquidity_usd": 1, "min_rt_score": 40,
}
SHADOW_STRATEGIES.append("BE25_LOCK10_TP100_SL30_NZ_S40")

STRATEGIES["ETH_BE25_LOCK10_TP100_SL30_S35"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.10, "label": "main"},
]
STRATEGY_FILTERS["ETH_BE25_LOCK10_TP100_SL30_S35"] = {"chain": "ethereum", "min_rt_score": 35}
SHADOW_STRATEGIES.append("ETH_BE25_LOCK10_TP100_SL30_S35")

STRATEGIES["ETH_BE25_LOCK10_TP100_SL30_S40"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.10, "label": "main"},
]
STRATEGY_FILTERS["ETH_BE25_LOCK10_TP100_SL30_S40"] = {"chain": "ethereum", "min_rt_score": 40}
SHADOW_STRATEGIES.append("ETH_BE25_LOCK10_TP100_SL30_S40")

STRATEGIES["ETH_BE25_LOCK10_TP100_SL30_NZ_S40"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.10, "label": "main"},
]
STRATEGY_FILTERS["ETH_BE25_LOCK10_TP100_SL30_NZ_S40"] = {
    "chain": "ethereum", "min_liquidity_usd": 1, "min_rt_score": 40,
}
SHADOW_STRATEGIES.append("ETH_BE25_LOCK10_TP100_SL30_NZ_S40")

STRATEGIES["ETH_BE25_LOCK10_TP100_SL30_MCAP_S40"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.10, "label": "main"},
]
STRATEGY_FILTERS["ETH_BE25_LOCK10_TP100_SL30_MCAP_S40"] = {
    "chain": "ethereum", "min_mcap": 30_000, "max_mcap": 500_000, "min_rt_score": 40,
}
SHADOW_STRATEGIES.append("ETH_BE25_LOCK10_TP100_SL30_MCAP_S40")

# ---- Bloc X: BE-range variants around the winner (8) ----
STRATEGIES["BE20_LOCK10_TP100_SL30"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.20, "be_lock_pct": 0.10, "label": "main"},
]
SHADOW_STRATEGIES.append("BE20_LOCK10_TP100_SL30")

STRATEGIES["BE30_LOCK10_TP100_SL30"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.30, "be_lock_pct": 0.10, "label": "main"},
]
SHADOW_STRATEGIES.append("BE30_LOCK10_TP100_SL30")

STRATEGIES["BE25_LOCK20_TP100_SL30"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.20, "label": "main"},
]
SHADOW_STRATEGIES.append("BE25_LOCK20_TP100_SL30")

STRATEGIES["BE15_LOCK15_TP80_SL30"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.15, "be_lock_pct": 0.15, "label": "main"},
]
SHADOW_STRATEGIES.append("BE15_LOCK15_TP80_SL30")

STRATEGIES["ETH_BE20_LOCK10_TP100_SL30"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.20, "be_lock_pct": 0.10, "label": "main"},
]
STRATEGY_FILTERS["ETH_BE20_LOCK10_TP100_SL30"] = {"chain": "ethereum"}
SHADOW_STRATEGIES.append("ETH_BE20_LOCK10_TP100_SL30")

STRATEGIES["ETH_BE30_LOCK10_TP100_SL30"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.30, "be_lock_pct": 0.10, "label": "main"},
]
STRATEGY_FILTERS["ETH_BE30_LOCK10_TP100_SL30"] = {"chain": "ethereum"}
SHADOW_STRATEGIES.append("ETH_BE30_LOCK10_TP100_SL30")

STRATEGIES["ETH_BE25_LOCK20_TP100_SL30"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.20, "label": "main"},
]
STRATEGY_FILTERS["ETH_BE25_LOCK20_TP100_SL30"] = {"chain": "ethereum"}
SHADOW_STRATEGIES.append("ETH_BE25_LOCK20_TP100_SL30")

STRATEGIES["ETH_BE15_LOCK15_TP80_SL30"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.15, "be_lock_pct": 0.15, "label": "main"},
]
STRATEGY_FILTERS["ETH_BE15_LOCK15_TP80_SL30"] = {"chain": "ethereum"}
SHADOW_STRATEGIES.append("ETH_BE15_LOCK15_TP80_SL30")

# ---- Bloc Y: SL variants around BE25_LOCK10_TP100 (4) ----
STRATEGIES["BE25_LOCK10_TP100_SL20"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.80, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.10, "label": "main"},
]
SHADOW_STRATEGIES.append("BE25_LOCK10_TP100_SL20")

STRATEGIES["BE25_LOCK10_TP100_SL40"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.60, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.10, "label": "main"},
]
SHADOW_STRATEGIES.append("BE25_LOCK10_TP100_SL40")

STRATEGIES["ETH_BE25_LOCK10_TP100_SL20"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.80, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.10, "label": "main"},
]
STRATEGY_FILTERS["ETH_BE25_LOCK10_TP100_SL20"] = {"chain": "ethereum"}
SHADOW_STRATEGIES.append("ETH_BE25_LOCK10_TP100_SL20")

STRATEGIES["ETH_BE25_LOCK10_TP100_SL40"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.60, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.10, "label": "main"},
]
STRATEGY_FILTERS["ETH_BE25_LOCK10_TP100_SL40"] = {"chain": "ethereum"}
SHADOW_STRATEGIES.append("ETH_BE25_LOCK10_TP100_SL40")

# ---- Bloc Z: wider TP variants with LOCK (6) ----
STRATEGIES["BE25_LOCK10_TP120_SL30"] = [
    {"pct": 1.0, "tp_mult": 2.20, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.10, "label": "main"},
]
SHADOW_STRATEGIES.append("BE25_LOCK10_TP120_SL30")

STRATEGIES["BE25_LOCK15_TP120_SL30"] = [
    {"pct": 1.0, "tp_mult": 2.20, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.15, "label": "main"},
]
SHADOW_STRATEGIES.append("BE25_LOCK15_TP120_SL30")

STRATEGIES["BE25_LOCK15_TP150_SL30"] = [
    {"pct": 1.0, "tp_mult": 2.50, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.15, "label": "main"},
]
SHADOW_STRATEGIES.append("BE25_LOCK15_TP150_SL30")

STRATEGIES["ETH_BE25_LOCK10_TP120_SL30"] = [
    {"pct": 1.0, "tp_mult": 2.20, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.10, "label": "main"},
]
STRATEGY_FILTERS["ETH_BE25_LOCK10_TP120_SL30"] = {"chain": "ethereum"}
SHADOW_STRATEGIES.append("ETH_BE25_LOCK10_TP120_SL30")

STRATEGIES["ETH_BE25_LOCK15_TP120_SL30"] = [
    {"pct": 1.0, "tp_mult": 2.20, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.15, "label": "main"},
]
STRATEGY_FILTERS["ETH_BE25_LOCK15_TP120_SL30"] = {"chain": "ethereum"}
SHADOW_STRATEGIES.append("ETH_BE25_LOCK15_TP120_SL30")

STRATEGIES["ETH_BE25_LOCK15_TP150_SL30"] = [
    {"pct": 1.0, "tp_mult": 2.50, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "be_lock_pct": 0.15, "label": "main"},
]
STRATEGY_FILTERS["ETH_BE25_LOCK15_TP150_SL30"] = {"chain": "ethereum"}
SHADOW_STRATEGIES.append("ETH_BE25_LOCK15_TP150_SL30")


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

# v14e.22 — LOWSCORE gate test (Apr 25). Apples-to-apples on N=52 events
# showed TP50_SL15 on score<30 calls beats BE25_TP80_SL30 by +1.93pp avg.
# Live extrapolation at $1.74/trade = +$3.34/d (+$1,217/yr). Modest signal,
# N too small for tight CI — running as SHADOW to grow N before deciding to
# promote to main paper. NOT in hybrid_strategy.allocations / not in
# paper_trade_config.active_strategies — gated to shadow auto-open.
# Mécanisme : SL serré (-15%) évite drawdown long sur low-conviction calls,
# TP +50% suffit (pumps score<30 rarement >+50%).
STRATEGIES["LOWSCORE_TP50_SL15"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.85, "horizon_min": 120, "label": "main"},
]
# horizon_min=120 mirrors base TP50_SL15 — the apples-to-apples test used the
# base strat directly, so the shadow must match for valid paired comparison.
STRATEGY_FILTERS["LOWSCORE_TP50_SL15"] = {"chain": "solana", "max_score": 29}
SHADOW_STRATEGIES.append("LOWSCORE_TP50_SL15")

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
# v14e.29: BE+LOCK variants — when BE arms, SL ratchets to entry*(1+lock_pct)
# instead of plain entry. Captures the case where the user wants to lock a
# guaranteed profit at activation rather than just breakeven. Pattern matches
# e.g. BE25_LOCK10_TP80_SL30 (group 1=25, group 2=10) or BE50_LOCK25_TP150_SL30.
# Also matches the chain-prefixed variants ETH_BE25_LOCK10_TP80_SL30.
_BE_LOCK_RE = re.compile(r"^(?:ETH_|BSC_|BASE_)?BE(\d+)_LOCK(\d+)_TP\d+_SL\d+")

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
# v14e.16 (Apr 24) — Age-window A/B shadows.
#
# Goal: measure the marginal PnL of relaxing the 12h token-age gate without
# risking the baseline. Three paper-only shadows, each covering a DISJOINT
# age band (12-24h, 24-48h, 48-72h). Clone of FAST_TP50_SL30 — our top live
# earner — to isolate the age-window variable. Sum of the 3 buckets = total
# impact of raising the gate to 72h.
#
# Why disjoint: with overlapping windows (e.g. AGE48 covering 0-48h) all
# three would trade the same <=12h tokens redundantly. Disjoint bands give
# a clean incremental-win measurement per band.
#
# Not in live_trading.allocations → paper-only by construction. Existing
# strats keep max_age_hours=12 default in _passes_strategy_filter (zero
# regression). Global gate in safe_scraper relaxed 12h -> 72h so tokens in
# these bands actually reach the per-strategy filter.
# ============================================================

STRATEGIES["AGE24_FAST_TP50_SL30"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.70, "horizon_min": 30, "label": "main"},
]
STRATEGY_FILTERS["AGE24_FAST_TP50_SL30"] = {
    "chain": "solana", "min_age_hours": 12, "max_age_hours": 24,
}

STRATEGIES["AGE48_FAST_TP50_SL30"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.70, "horizon_min": 30, "label": "main"},
]
STRATEGY_FILTERS["AGE48_FAST_TP50_SL30"] = {
    "chain": "solana", "min_age_hours": 24, "max_age_hours": 48,
}

STRATEGIES["AGE72_FAST_TP50_SL30"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.70, "horizon_min": 30, "label": "main"},
]
STRATEGY_FILTERS["AGE72_FAST_TP50_SL30"] = {
    "chain": "solana", "min_age_hours": 48, "max_age_hours": 72,
}


# ============================================================
# v14e.27 (Apr 26) — AGE clones across other top strats.
# AGE24_FAST_TP50_SL30 surfaced +20% avg / WR 62.5% on N=8 (Apr 24-26 dead
# regime), hinting the 12-24h age band beats the default 0-12h universe when
# the macro pump-rate is low. Cloning the top-earning + new SCALP picks across
# the same 2 bands (12-24h, 24-48h) to test whether the age signal is a
# strat-specific quirk or a general edge. AGE72 (48-72h) deliberately skipped
# — it bled −$7.55/N=2 on the existing FAST clone, signal points the wrong
# way on tokens that old. All shadow-only, paper data only.
# ============================================================

# BE25_TP80_SL30 — live earner, +$166 post-reset / N=75. Test if age gate
# improves dead-day resilience.
STRATEGIES["AGE24_BE25_TP80_SL30"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "label": "main"},
]
STRATEGY_FILTERS["AGE24_BE25_TP80_SL30"] = {
    "chain": "solana", "min_age_hours": 12, "max_age_hours": 24,
}
SHADOW_STRATEGIES.append("AGE24_BE25_TP80_SL30")

STRATEGIES["AGE48_BE25_TP80_SL30"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "label": "main"},
]
STRATEGY_FILTERS["AGE48_BE25_TP80_SL30"] = {
    "chain": "solana", "min_age_hours": 24, "max_age_hours": 48,
}
SHADOW_STRATEGIES.append("AGE48_BE25_TP80_SL30")

# TP50_SL15 — fast scalp +$60 post-reset / N=100.
STRATEGIES["AGE24_TP50_SL15"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.85, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["AGE24_TP50_SL15"] = {
    "chain": "solana", "min_age_hours": 12, "max_age_hours": 24,
}
SHADOW_STRATEGIES.append("AGE24_TP50_SL15")

STRATEGIES["AGE48_TP50_SL15"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.85, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["AGE48_TP50_SL15"] = {
    "chain": "solana", "min_age_hours": 24, "max_age_hours": 48,
}
SHADOW_STRATEGIES.append("AGE48_TP50_SL15")

# FAST_TP100_SL20 — high-TP fast +$49 post-reset / N=100.
STRATEGIES["AGE24_FAST_TP100_SL20"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.80, "horizon_min": 30, "label": "main"},
]
STRATEGY_FILTERS["AGE24_FAST_TP100_SL20"] = {
    "chain": "solana", "min_age_hours": 12, "max_age_hours": 24,
}
SHADOW_STRATEGIES.append("AGE24_FAST_TP100_SL20")

STRATEGIES["AGE48_FAST_TP100_SL20"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.80, "horizon_min": 30, "label": "main"},
]
STRATEGY_FILTERS["AGE48_FAST_TP100_SL20"] = {
    "chain": "solana", "min_age_hours": 24, "max_age_hours": 48,
}
SHADOW_STRATEGIES.append("AGE48_FAST_TP100_SL20")

# FAST_TP80_SL25 — fast TP80 +$25 post-reset / N=100.
STRATEGIES["AGE24_FAST_TP80_SL25"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.75, "horizon_min": 30, "label": "main"},
]
STRATEGY_FILTERS["AGE24_FAST_TP80_SL25"] = {
    "chain": "solana", "min_age_hours": 12, "max_age_hours": 24,
}
SHADOW_STRATEGIES.append("AGE24_FAST_TP80_SL25")

STRATEGIES["AGE48_FAST_TP80_SL25"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.75, "horizon_min": 30, "label": "main"},
]
STRATEGY_FILTERS["AGE48_FAST_TP80_SL25"] = {
    "chain": "solana", "min_age_hours": 24, "max_age_hours": 48,
}
SHADOW_STRATEGIES.append("AGE48_FAST_TP80_SL25")

# SCALP_TP15_SL20 SCORE35 — mega-sweep top robust (dead-day +7.74%, rs 0.88).
STRATEGIES["AGE24_SCALP_TP15_SL20_S35"] = [
    {"pct": 1.0, "tp_mult": 1.15, "sl_mult": 0.80, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["AGE24_SCALP_TP15_SL20_S35"] = {
    "chain": "solana", "min_age_hours": 12, "max_age_hours": 24, "min_rt_score": 35,
}
SHADOW_STRATEGIES.append("AGE24_SCALP_TP15_SL20_S35")

STRATEGIES["AGE48_SCALP_TP15_SL20_S35"] = [
    {"pct": 1.0, "tp_mult": 1.15, "sl_mult": 0.80, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["AGE48_SCALP_TP15_SL20_S35"] = {
    "chain": "solana", "min_age_hours": 24, "max_age_hours": 48, "min_rt_score": 35,
}
SHADOW_STRATEGIES.append("AGE48_SCALP_TP15_SL20_S35")

# SCALP_TP15_NOSL SCORE35 — mega-sweep robust (dead-day +7.99%, WR 76%).
STRATEGIES["AGE24_SCALP_TP15_NOSL_S35"] = [
    {"pct": 1.0, "tp_mult": 1.15, "sl_mult": 0.20, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["AGE24_SCALP_TP15_NOSL_S35"] = {
    "chain": "solana", "min_age_hours": 12, "max_age_hours": 24, "min_rt_score": 35,
}
SHADOW_STRATEGIES.append("AGE24_SCALP_TP15_NOSL_S35")

STRATEGIES["AGE48_SCALP_TP15_NOSL_S35"] = [
    {"pct": 1.0, "tp_mult": 1.15, "sl_mult": 0.20, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["AGE48_SCALP_TP15_NOSL_S35"] = {
    "chain": "solana", "min_age_hours": 24, "max_age_hours": 48, "min_rt_score": 35,
}
SHADOW_STRATEGIES.append("AGE48_SCALP_TP15_NOSL_S35")


# ============================================================
# v14e.28 (Apr 26) — ETH AGE clones across the 2 active ETH paper mains.
# Apr 26 rerank surfaced ETH AGE24-48 = -23.5% / WR 15% on N=94 — INVERSE
# of Solana where AGE24/48 outperforms. Default ETH max_age now 12h. These
# 4 clones disable that default by declaring an explicit age band: keep
# collecting data in case the regime shifts (or a specific strat profile
# proves the inverse signal wrong). Shadow-only by construction.
# ============================================================

# ETH_TP80_SL40_T2H — top by N (26 trades) + paper main candidate.
STRATEGIES["AGE24_ETH_TP80_SL40_T2H"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.60, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["AGE24_ETH_TP80_SL40_T2H"] = {
    "chain": "ethereum", "min_age_hours": 12, "max_age_hours": 24,
}
SHADOW_STRATEGIES.append("AGE24_ETH_TP80_SL40_T2H")

STRATEGIES["AGE48_ETH_TP80_SL40_T2H"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.60, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["AGE48_ETH_TP80_SL40_T2H"] = {
    "chain": "ethereum", "min_age_hours": 24, "max_age_hours": 48,
}
SHADOW_STRATEGIES.append("AGE48_ETH_TP80_SL40_T2H")

# ETH_FAST_TP100_SL20 — second active ETH main, top rerank avg pnl.
STRATEGIES["AGE24_ETH_FAST_TP100_SL20"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.80, "horizon_min": 30, "label": "main"},
]
STRATEGY_FILTERS["AGE24_ETH_FAST_TP100_SL20"] = {
    "chain": "ethereum", "min_age_hours": 12, "max_age_hours": 24,
}
SHADOW_STRATEGIES.append("AGE24_ETH_FAST_TP100_SL20")

STRATEGIES["AGE48_ETH_FAST_TP100_SL20"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.80, "horizon_min": 30, "label": "main"},
]
STRATEGY_FILTERS["AGE48_ETH_FAST_TP100_SL20"] = {
    "chain": "ethereum", "min_age_hours": 24, "max_age_hours": 48,
}
SHADOW_STRATEGIES.append("AGE48_ETH_FAST_TP100_SL20")


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
# v14e.27 (Apr 26) — ETH BE20 variants confirmed BE never armed (peak <+20%
# on ETH memecoins): paper PnL was identical row-pour-row vs the non-BE
# parents (ETH_TP100_SL50 vs ETH_BE20_TP100_SL50, ETH_TP80_SL40_T2H vs
# ETH_BE20_TP80_SL40_T2H). Removing the BE20 duplicates and adding a single
# higher-trigger BE shadow (BE50_TP150_SL40_T2H) below — testing whether a
# BE that can plausibly arm on the +50%+ runners adds value vs the pure TP.
STRATEGIES["ETH_TP100_SL50"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.50, "horizon_min": 240, "label": "main"},
]
# v14e.4: min_liquidity_usd removed for Phase 1 — let fee model (gas + dynamic
# slippage) encode the real cost on shallow pools instead of pre-filtering.
# Same rationale for BSC/Base below. The chain gate is the only hard filter.
STRATEGY_FILTERS["ETH_TP100_SL50"] = {"chain": "ethereum"}
# v14e.27: bleeding −$872 last 12h — demote to shadow, keep N growing.
SHADOW_STRATEGIES.append("ETH_TP100_SL50")

STRATEGIES["ETH_TP80_SL40_T2H"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.60, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["ETH_TP80_SL40_T2H"] = {"chain": "ethereum"}

# Higher-trigger BE shadow — tests whether BE@+50% (plausibly reached on
# real ETH runners) plus tight SL40 outperforms the pure TP100/TP80 parents.
STRATEGIES["ETH_BE50_TP150_SL40_T2H"] = [
    {"pct": 1.0, "tp_mult": 2.50, "sl_mult": 0.60, "horizon_min": 120,
     "be_activation": 0.50, "label": "main"},
]
STRATEGY_FILTERS["ETH_BE50_TP150_SL40_T2H"] = {"chain": "ethereum"}
SHADOW_STRATEGIES.append("ETH_BE50_TP150_SL40_T2H")

# Kept in code for in-flight close (legacy), removed from SHADOW_STRATEGIES so
# no NEW shadows open. v14e.27: BE20 never armed on ETH (peak <+20% in regime),
# so these were dead duplicates of the TP-pure parents.
STRATEGIES["ETH_BE50_TP150_SL50"] = [
    {"pct": 1.0, "tp_mult": 2.50, "sl_mult": 0.50, "horizon_min": 240,
     "be_activation": 0.50, "label": "main"},
]
STRATEGY_FILTERS["ETH_BE50_TP150_SL50"] = {"chain": "ethereum"}
# Not in SHADOW_STRATEGIES — replaced by ETH_BE50_TP150_SL40_T2H.

STRATEGIES["ETH_BE20_TP100_SL50"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.50, "horizon_min": 240,
     "be_activation": 0.20, "label": "main"},
]
STRATEGY_FILTERS["ETH_BE20_TP100_SL50"] = {"chain": "ethereum"}
# Not in SHADOW_STRATEGIES — duplicate of ETH_TP100_SL50, BE20 never arms.

STRATEGIES["ETH_BE30_TP100_SL40"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.60, "horizon_min": 240,
     "be_activation": 0.30, "label": "main"},
]
STRATEGY_FILTERS["ETH_BE30_TP100_SL40"] = {"chain": "ethereum"}
# v14e.27: bleeding −$525 last 12h, BE30 rarely armed — demote to shadow.
SHADOW_STRATEGIES.append("ETH_BE30_TP100_SL40")

STRATEGIES["ETH_BE20_TP80_SL40_T2H"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.60, "horizon_min": 120,
     "be_activation": 0.20, "label": "main"},
]
STRATEGY_FILTERS["ETH_BE20_TP80_SL40_T2H"] = {"chain": "ethereum"}
# Not in SHADOW_STRATEGIES — duplicate of ETH_TP80_SL40_T2H, BE20 never arms.

# ============================================================
# v14e.21 — ETH FAST family (sim mega-sweep top picks Apr 25, post v14e.20
# clean-up of trail/dip/hyst artefacts). Hypothesis: ETH memecoins pump and
# dump faster than 2-4h timeouts capture; shorter timeouts (30-60min) +
# higher TP targets convert peaks to fills before reversal.
# Sim ranks vs current ETH mains (avg% / $/d on N=23 universe):
#   ETH_FAST_TP100_SL50    +30.7% / $277  (#1 sim)
#   ETH_FAST_TP100_SL20    +30.1% / $271
#   ETH_FAST_TP500_SL40_60M +30.1% / $270  (moonshot — TP 500%, SL 40%, 60min)
#   ETH_FAST60_TP100_SL50  +29.8% / $268  (60min timeout variant)
#   ETH_FAST_TP40_SL30 (SCORE30) +42.8% / $235  N=14
# Activated as MAIN paper (Telegram alerts on, paper_trades.is_shadow=False)
# alongside the 5 existing ETH mains. Paired-test verdict at N>=15-20 each
# expected ~Mai 02-09. NOT in live_trading.allocations — paper only.
# ============================================================

# v14e.27 (Apr 26) — ETH_FAST family triage after 17h post-seed:
#   ETH_FAST_TP100_SL20 +$81 / +6.8% N=6 (last 6h)  → KEEP main
#   ETH_FAST_TP40_SL30  +$6  / +0.5% N=7            → KEEP main
#   ETH_FAST_TP100_SL50 −$497 / −14.6% N=17         → DEMOTE shadow
#   ETH_FAST60_TP100_SL50 −$671 / −19.8% N=17       → DEMOTE shadow
#   ETH_FAST_TP500_SL40_60M −$785 / −23.1% N=17     → DEMOTE shadow
# The 3 demoted variants stay in SHADOW_STRATEGIES so we keep collecting
# paper data, but are dropped from rt_trade_config.hybrid_strategy.allocations
# (DB-side update). Shadow keeps N growing; if regime flips we have data to
# reverse the call.

STRATEGIES["ETH_FAST_TP100_SL50"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.50, "horizon_min": 30, "label": "main"},
]
STRATEGY_FILTERS["ETH_FAST_TP100_SL50"] = {"chain": "ethereum"}
SHADOW_STRATEGIES.append("ETH_FAST_TP100_SL50")

STRATEGIES["ETH_FAST_TP100_SL20"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.80, "horizon_min": 30, "label": "main"},
]
STRATEGY_FILTERS["ETH_FAST_TP100_SL20"] = {"chain": "ethereum"}

STRATEGIES["ETH_FAST60_TP100_SL50"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.50, "horizon_min": 60, "label": "main"},
]
STRATEGY_FILTERS["ETH_FAST60_TP100_SL50"] = {"chain": "ethereum"}
SHADOW_STRATEGIES.append("ETH_FAST60_TP100_SL50")

STRATEGIES["ETH_FAST_TP500_SL40_60M"] = [
    {"pct": 1.0, "tp_mult": 6.00, "sl_mult": 0.60, "horizon_min": 60, "label": "main"},
]
STRATEGY_FILTERS["ETH_FAST_TP500_SL40_60M"] = {"chain": "ethereum"}
SHADOW_STRATEGIES.append("ETH_FAST_TP500_SL40_60M")

STRATEGIES["ETH_FAST_TP40_SL30"] = [
    {"pct": 1.0, "tp_mult": 1.40, "sl_mult": 0.70, "horizon_min": 30, "label": "main"},
]
# SCORE30 filter applied — sim showed +42.8% gated vs +24% un-gated on the same
# TP/SL/horizon, so we keep the gate to avoid washing the signal on weak callers.
STRATEGY_FILTERS["ETH_FAST_TP40_SL30"] = {"chain": "ethereum", "min_score": 30}

# ============================================================
# v14e — BSC L1 paper strats (Phase 1 shadow, zero capital).
#
# Three strategies symmetrical to ETH's so we can compare apples-to-apples
# across chains once data starts flowing. Min liquidity raised to $20k —
# PancakeSwap memecoins below that routinely slip >5% on $50 trades.
# Prefix `BSC_` is load-bearing: it's what _build_chain_strategies + the
# v14e migration routing heuristic use to partition the bankroll bucket.
# ============================================================

STRATEGIES["BSC_TP100_SL50"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.50, "horizon_min": 240, "label": "main"},
]
STRATEGY_FILTERS["BSC_TP100_SL50"] = {"chain": "bsc"}

STRATEGIES["BSC_TP80_SL40_T2H"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.60, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["BSC_TP80_SL40_T2H"] = {"chain": "bsc"}

STRATEGIES["BSC_BE50_TP150_SL50"] = [
    {"pct": 1.0, "tp_mult": 2.50, "sl_mult": 0.50, "horizon_min": 240,
     "be_activation": 0.50, "label": "main"},
]
STRATEGY_FILTERS["BSC_BE50_TP150_SL50"] = {"chain": "bsc"}

# ============================================================
# v14e — Base L2 paper strats (Phase 1 shadow, zero capital).
#
# L2 → cheaper gas → smaller min_liquidity ($15k) works without fees eating
# the edge. Same 3 shapes as ETH/BSC for coherent cross-chain comparison.
# ============================================================

STRATEGIES["BASE_TP100_SL50"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.50, "horizon_min": 240, "label": "main"},
]
STRATEGY_FILTERS["BASE_TP100_SL50"] = {"chain": "base"}

STRATEGIES["BASE_TP80_SL40_T2H"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.60, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["BASE_TP80_SL40_T2H"] = {"chain": "base"}

STRATEGIES["BASE_BE50_TP150_SL50"] = [
    {"pct": 1.0, "tp_mult": 2.50, "sl_mult": 0.50, "horizon_min": 240,
     "be_activation": 0.50, "label": "main"},
]
STRATEGY_FILTERS["BASE_BE50_TP150_SL50"] = {"chain": "base"}


# ============================================================
# v14e.27 (Apr 26) — Mega-sweep regime-aware top picks shadow rollout.
#
# Run 24941515693 (521,640 configs simulated, 9d window 4 active / 3 quiet
# / 1 dead) surfaced 24 strategy_ids that pass cross_regime_robust + family
# realism gate. Highlight: SCALP family dominates (10/20 top robust), with
# SCALP_TP15_SL20 SCORE35 the only strat positive on dead-days (+7.74%) and
# rank_stability 0.88. FAST60 niche secondary; classic TP/SL with score gate
# rounds out.
#
# All added below as SHADOW (paper-only, no live alloc, no bankroll alloc).
# Filter is embedded in the strategy name (`_S30`, `_S35`, `_S40`,
# `_NZ_S40`, `_MCAP_S40`) so the mega-sweep result reproduces 1:1.
#
# Expected verdict at N≥50 / 14d (~Mai 09): paired-test vs base SCALP/FAST60
# without filter → if score gate adds >+2pp avg/trade, promote 1-2 to main.
# ============================================================

# --- SOL shadow rollout (17 strats) ---

# SCALP family — robust dead-day winner per v14e.26 sweep.
STRATEGIES["SCALP_TP15_SL20_S35"] = [
    {"pct": 1.0, "tp_mult": 1.15, "sl_mult": 0.80, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["SCALP_TP15_SL20_S35"] = {"min_rt_score": 35}
SHADOW_STRATEGIES.append("SCALP_TP15_SL20_S35")

STRATEGIES["SCALP_TP15_SL15_S40"] = [
    {"pct": 1.0, "tp_mult": 1.15, "sl_mult": 0.85, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["SCALP_TP15_SL15_S40"] = {"min_rt_score": 40}
SHADOW_STRATEGIES.append("SCALP_TP15_SL15_S40")

STRATEGIES["SCALP_TP15_NOSL_S35"] = [
    {"pct": 1.0, "tp_mult": 1.15, "sl_mult": 0.20, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["SCALP_TP15_NOSL_S35"] = {"min_rt_score": 35}
SHADOW_STRATEGIES.append("SCALP_TP15_NOSL_S35")

STRATEGIES["SCALP_TP15_SL10_S30"] = [
    {"pct": 1.0, "tp_mult": 1.15, "sl_mult": 0.90, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["SCALP_TP15_SL10_S30"] = {"min_rt_score": 30}
SHADOW_STRATEGIES.append("SCALP_TP15_SL10_S30")

STRATEGIES["SCALP_TP10_SL10_S35"] = [
    {"pct": 1.0, "tp_mult": 1.10, "sl_mult": 0.90, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["SCALP_TP10_SL10_S35"] = {"min_rt_score": 35}
SHADOW_STRATEGIES.append("SCALP_TP10_SL10_S35")

STRATEGIES["SCALP_TP10_SL15_S40"] = [
    {"pct": 1.0, "tp_mult": 1.10, "sl_mult": 0.85, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["SCALP_TP10_SL15_S40"] = {"min_rt_score": 40}
SHADOW_STRATEGIES.append("SCALP_TP10_SL15_S40")

STRATEGIES["SCALP_TP20_SL10_S30"] = [
    {"pct": 1.0, "tp_mult": 1.20, "sl_mult": 0.90, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["SCALP_TP20_SL10_S30"] = {"min_rt_score": 30}
SHADOW_STRATEGIES.append("SCALP_TP20_SL10_S30")

STRATEGIES["SCALP_TP20_NOSL_S35"] = [
    {"pct": 1.0, "tp_mult": 1.20, "sl_mult": 0.20, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["SCALP_TP20_NOSL_S35"] = {"min_rt_score": 35}
SHADOW_STRATEGIES.append("SCALP_TP20_NOSL_S35")

STRATEGIES["SCALP_TP20_SL15_S35"] = [
    {"pct": 1.0, "tp_mult": 1.20, "sl_mult": 0.85, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["SCALP_TP20_SL15_S35"] = {"min_rt_score": 35}
SHADOW_STRATEGIES.append("SCALP_TP20_SL15_S35")

# FAST60 family — rapid-timeout robust picks.
STRATEGIES["FAST60_TP50_SL50_S30"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.50, "horizon_min": 60, "label": "main"},
]
STRATEGY_FILTERS["FAST60_TP50_SL50_S30"] = {"min_rt_score": 30}
SHADOW_STRATEGIES.append("FAST60_TP50_SL50_S30")

STRATEGIES["FAST60_TP100_SL50_NZ_S40"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.50, "horizon_min": 60, "label": "main"},
]
STRATEGY_FILTERS["FAST60_TP100_SL50_NZ_S40"] = {"min_liquidity_usd": 1.0, "min_rt_score": 40}
SHADOW_STRATEGIES.append("FAST60_TP100_SL50_NZ_S40")

STRATEGIES["FAST60_TP70_SL50_NZ_S40"] = [
    {"pct": 1.0, "tp_mult": 1.70, "sl_mult": 0.50, "horizon_min": 60, "label": "main"},
]
STRATEGY_FILTERS["FAST60_TP70_SL50_NZ_S40"] = {"min_liquidity_usd": 1.0, "min_rt_score": 40}
SHADOW_STRATEGIES.append("FAST60_TP70_SL50_NZ_S40")

STRATEGIES["FAST45_TP40_SL30_S30"] = [
    {"pct": 1.0, "tp_mult": 1.40, "sl_mult": 0.70, "horizon_min": 45, "label": "main"},
]
STRATEGY_FILTERS["FAST45_TP40_SL30_S30"] = {"min_rt_score": 30}
SHADOW_STRATEGIES.append("FAST45_TP40_SL30_S30")

# Classic TP/SL with score gate.
STRATEGIES["TP30_SL10_S35"] = [
    {"pct": 1.0, "tp_mult": 1.30, "sl_mult": 0.90, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["TP30_SL10_S35"] = {"min_rt_score": 35}
SHADOW_STRATEGIES.append("TP30_SL10_S35")

STRATEGIES["TP50_SL40_S35"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.60, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["TP50_SL40_S35"] = {"min_rt_score": 35}
SHADOW_STRATEGIES.append("TP50_SL40_S35")

STRATEGIES["TP200_SL40_2H_NZ_S40"] = [
    {"pct": 1.0, "tp_mult": 3.00, "sl_mult": 0.60, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["TP200_SL40_2H_NZ_S40"] = {"min_liquidity_usd": 1.0, "min_rt_score": 40}
SHADOW_STRATEGIES.append("TP200_SL40_2H_NZ_S40")

STRATEGIES["FAST_TP200_SL40_60M_MCAP_S40"] = [
    {"pct": 1.0, "tp_mult": 3.00, "sl_mult": 0.60, "horizon_min": 60, "label": "main"},
]
STRATEGY_FILTERS["FAST_TP200_SL40_60M_MCAP_S40"] = {
    "min_mcap": 30_000, "max_mcap": 500_000, "min_rt_score": 40,
}
SHADOW_STRATEGIES.append("FAST_TP200_SL40_60M_MCAP_S40")

# --- ETH clones of the SOL shadow rollout (17 strats, chain-gated) ---

STRATEGIES["ETH_SCALP_TP15_SL20_S35"] = [
    {"pct": 1.0, "tp_mult": 1.15, "sl_mult": 0.80, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["ETH_SCALP_TP15_SL20_S35"] = {"chain": "ethereum", "min_rt_score": 35}
SHADOW_STRATEGIES.append("ETH_SCALP_TP15_SL20_S35")

STRATEGIES["ETH_SCALP_TP15_SL15_S40"] = [
    {"pct": 1.0, "tp_mult": 1.15, "sl_mult": 0.85, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["ETH_SCALP_TP15_SL15_S40"] = {"chain": "ethereum", "min_rt_score": 40}
SHADOW_STRATEGIES.append("ETH_SCALP_TP15_SL15_S40")

STRATEGIES["ETH_SCALP_TP15_NOSL_S35"] = [
    {"pct": 1.0, "tp_mult": 1.15, "sl_mult": 0.20, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["ETH_SCALP_TP15_NOSL_S35"] = {"chain": "ethereum", "min_rt_score": 35}
SHADOW_STRATEGIES.append("ETH_SCALP_TP15_NOSL_S35")

STRATEGIES["ETH_SCALP_TP15_SL10_S30"] = [
    {"pct": 1.0, "tp_mult": 1.15, "sl_mult": 0.90, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["ETH_SCALP_TP15_SL10_S30"] = {"chain": "ethereum", "min_rt_score": 30}
SHADOW_STRATEGIES.append("ETH_SCALP_TP15_SL10_S30")

STRATEGIES["ETH_SCALP_TP10_SL10_S35"] = [
    {"pct": 1.0, "tp_mult": 1.10, "sl_mult": 0.90, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["ETH_SCALP_TP10_SL10_S35"] = {"chain": "ethereum", "min_rt_score": 35}
SHADOW_STRATEGIES.append("ETH_SCALP_TP10_SL10_S35")

STRATEGIES["ETH_SCALP_TP10_SL15_S40"] = [
    {"pct": 1.0, "tp_mult": 1.10, "sl_mult": 0.85, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["ETH_SCALP_TP10_SL15_S40"] = {"chain": "ethereum", "min_rt_score": 40}
SHADOW_STRATEGIES.append("ETH_SCALP_TP10_SL15_S40")

STRATEGIES["ETH_SCALP_TP20_SL10_S30"] = [
    {"pct": 1.0, "tp_mult": 1.20, "sl_mult": 0.90, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["ETH_SCALP_TP20_SL10_S30"] = {"chain": "ethereum", "min_rt_score": 30}
SHADOW_STRATEGIES.append("ETH_SCALP_TP20_SL10_S30")

STRATEGIES["ETH_SCALP_TP20_NOSL_S35"] = [
    {"pct": 1.0, "tp_mult": 1.20, "sl_mult": 0.20, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["ETH_SCALP_TP20_NOSL_S35"] = {"chain": "ethereum", "min_rt_score": 35}
SHADOW_STRATEGIES.append("ETH_SCALP_TP20_NOSL_S35")

STRATEGIES["ETH_SCALP_TP20_SL15_S35"] = [
    {"pct": 1.0, "tp_mult": 1.20, "sl_mult": 0.85, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["ETH_SCALP_TP20_SL15_S35"] = {"chain": "ethereum", "min_rt_score": 35}
SHADOW_STRATEGIES.append("ETH_SCALP_TP20_SL15_S35")

STRATEGIES["ETH_FAST60_TP50_SL50_S30"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.50, "horizon_min": 60, "label": "main"},
]
STRATEGY_FILTERS["ETH_FAST60_TP50_SL50_S30"] = {"chain": "ethereum", "min_rt_score": 30}
SHADOW_STRATEGIES.append("ETH_FAST60_TP50_SL50_S30")

STRATEGIES["ETH_FAST60_TP100_SL50_NZ_S40"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.50, "horizon_min": 60, "label": "main"},
]
STRATEGY_FILTERS["ETH_FAST60_TP100_SL50_NZ_S40"] = {
    "chain": "ethereum", "min_liquidity_usd": 1.0, "min_rt_score": 40,
}
SHADOW_STRATEGIES.append("ETH_FAST60_TP100_SL50_NZ_S40")

STRATEGIES["ETH_FAST60_TP70_SL50_NZ_S40"] = [
    {"pct": 1.0, "tp_mult": 1.70, "sl_mult": 0.50, "horizon_min": 60, "label": "main"},
]
STRATEGY_FILTERS["ETH_FAST60_TP70_SL50_NZ_S40"] = {
    "chain": "ethereum", "min_liquidity_usd": 1.0, "min_rt_score": 40,
}
SHADOW_STRATEGIES.append("ETH_FAST60_TP70_SL50_NZ_S40")

STRATEGIES["ETH_FAST45_TP40_SL30_S30"] = [
    {"pct": 1.0, "tp_mult": 1.40, "sl_mult": 0.70, "horizon_min": 45, "label": "main"},
]
STRATEGY_FILTERS["ETH_FAST45_TP40_SL30_S30"] = {"chain": "ethereum", "min_rt_score": 30}
SHADOW_STRATEGIES.append("ETH_FAST45_TP40_SL30_S30")

STRATEGIES["ETH_TP30_SL10_S35"] = [
    {"pct": 1.0, "tp_mult": 1.30, "sl_mult": 0.90, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["ETH_TP30_SL10_S35"] = {"chain": "ethereum", "min_rt_score": 35}
SHADOW_STRATEGIES.append("ETH_TP30_SL10_S35")

STRATEGIES["ETH_TP50_SL40_S35"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.60, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["ETH_TP50_SL40_S35"] = {"chain": "ethereum", "min_rt_score": 35}
SHADOW_STRATEGIES.append("ETH_TP50_SL40_S35")

STRATEGIES["ETH_TP200_SL40_2H_NZ_S40"] = [
    {"pct": 1.0, "tp_mult": 3.00, "sl_mult": 0.60, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["ETH_TP200_SL40_2H_NZ_S40"] = {
    "chain": "ethereum", "min_liquidity_usd": 1.0, "min_rt_score": 40,
}
SHADOW_STRATEGIES.append("ETH_TP200_SL40_2H_NZ_S40")

STRATEGIES["ETH_FAST_TP200_SL40_60M_MCAP_S40"] = [
    {"pct": 1.0, "tp_mult": 3.00, "sl_mult": 0.60, "horizon_min": 60, "label": "main"},
]
STRATEGY_FILTERS["ETH_FAST_TP200_SL40_60M_MCAP_S40"] = {
    "chain": "ethereum", "min_mcap": 30_000, "max_mcap": 500_000, "min_rt_score": 40,
}
SHADOW_STRATEGIES.append("ETH_FAST_TP200_SL40_60M_MCAP_S40")


# ============================================================
# v14e.40 — RECALL_DIP shadow family (INVA pattern A/B test).
# Hypothesis: when a token already called by a KOL gets recalled
# AFTER a meaningful dip vs the first call price, smart money may
# be re-entering. Historical 21d sample (N=15-20 per bucket) showed
# +8-12% EV on TP50/SL30 for dip≥30%, but evidence is too thin to
# ship live — these run as shadows to gather labeled data.
#
# Filter contract (paper_trader._passes_strategy_filter v14e.40):
#   require_recall=True       — must be a 2nd+ call
#   min_recall_drift / max_recall_drift  — bounds on (price/first - 1)
#   max_hours_since_first     — lookback window
# RT detection: safe_scraper._rt_open_trades queries kol_call_outcomes
# for the FIRST call ≥30 min ago, populates _rt_is_recall + drift fields.
#
# Variants (14): dip × age × TP/SL/timeout × chain.
# All shadows, no live exposure.
# ============================================================

# --- SOL recall — TP50/SL30 (positive EV bucket from 21d backtest) ---
STRATEGIES["RECALL_DIP30_TP50_SL30_2H"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.70, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["RECALL_DIP30_TP50_SL30_2H"] = {
    "require_recall": True, "min_recall_drift": -0.50, "max_recall_drift": -0.30,
    "max_hours_since_first": 24,
}
SHADOW_STRATEGIES.append("RECALL_DIP30_TP50_SL30_2H")

STRATEGIES["RECALL_DIP30_TP50_SL30_6H"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.70, "horizon_min": 360, "label": "main"},
]
STRATEGY_FILTERS["RECALL_DIP30_TP50_SL30_6H"] = {
    "require_recall": True, "min_recall_drift": -0.50, "max_recall_drift": -0.30,
    "max_hours_since_first": 24,
}
SHADOW_STRATEGIES.append("RECALL_DIP30_TP50_SL30_6H")

# Deeper dip (-50 to -70%) — N=5 historic, 40% TP50 hit but tiny sample
STRATEGIES["RECALL_DIP50_TP50_SL30_2H"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.70, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["RECALL_DIP50_TP50_SL30_2H"] = {
    "require_recall": True, "min_recall_drift": -0.70, "max_recall_drift": -0.50,
    "max_hours_since_first": 48,
}
SHADOW_STRATEGIES.append("RECALL_DIP50_TP50_SL30_2H")

STRATEGIES["RECALL_DIP50_TP100_SL40_6H"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.60, "horizon_min": 360, "label": "main"},
]
STRATEGY_FILTERS["RECALL_DIP50_TP100_SL40_6H"] = {
    "require_recall": True, "min_recall_drift": -0.70, "max_recall_drift": -0.50,
    "max_hours_since_first": 48,
}
SHADOW_STRATEGIES.append("RECALL_DIP50_TP100_SL40_6H")

# Wide dip range -30 to -70 with BE (lock the bounce, no full SL)
STRATEGIES["RECALL_DIP30_BE25_TP80_SL30"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.70, "horizon_min": 120,
     "be_activation": 0.25, "label": "main"},
]
STRATEGY_FILTERS["RECALL_DIP30_BE25_TP80_SL30"] = {
    "require_recall": True, "min_recall_drift": -0.70, "max_recall_drift": -0.30,
    "max_hours_since_first": 24,
}
SHADOW_STRATEGIES.append("RECALL_DIP30_BE25_TP80_SL30")

# Age-segmented (6h after first call = freshest dip recovery, 44% TP50 in N=18)
STRATEGIES["RECALL_DIP30_AGE6_TP50_SL30"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.70, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["RECALL_DIP30_AGE6_TP50_SL30"] = {
    "require_recall": True, "min_recall_drift": -0.70, "max_recall_drift": -0.30,
    "max_hours_since_first": 6,
}
SHADOW_STRATEGIES.append("RECALL_DIP30_AGE6_TP50_SL30")

# Older recalls (6-24h) — stale bag bounce hypothesis
STRATEGIES["RECALL_DIP30_AGE24_TP50_SL30"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.70, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["RECALL_DIP30_AGE24_TP50_SL30"] = {
    "require_recall": True, "min_recall_drift": -0.70, "max_recall_drift": -0.30,
    "min_hours_since_first": 6, "max_hours_since_first": 24,
}
SHADOW_STRATEGIES.append("RECALL_DIP30_AGE24_TP50_SL30")

# Shallow dip (-10 to -30%) — control / negative-control bucket
STRATEGIES["RECALL_DIP10_TP50_SL30"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.70, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["RECALL_DIP10_TP50_SL30"] = {
    "require_recall": True, "min_recall_drift": -0.30, "max_recall_drift": -0.10,
    "max_hours_since_first": 24,
}
SHADOW_STRATEGIES.append("RECALL_DIP10_TP50_SL30")

# Wider TP for the deep-dip recoveries that historically went 5x-10x
STRATEGIES["RECALL_DIP50_TP200_SL50_6H"] = [
    {"pct": 1.0, "tp_mult": 3.00, "sl_mult": 0.50, "horizon_min": 360, "label": "main"},
]
STRATEGY_FILTERS["RECALL_DIP50_TP200_SL50_6H"] = {
    "require_recall": True, "min_recall_drift": -0.85, "max_recall_drift": -0.50,
    "max_hours_since_first": 72,
}
SHADOW_STRATEGIES.append("RECALL_DIP50_TP200_SL50_6H")

# --- ETH recall variants ---
STRATEGIES["ETH_RECALL_DIP30_TP50_SL30_2H"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.70, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["ETH_RECALL_DIP30_TP50_SL30_2H"] = {
    "chain": "ethereum",
    "require_recall": True, "min_recall_drift": -0.50, "max_recall_drift": -0.30,
    "max_hours_since_first": 24,
}
SHADOW_STRATEGIES.append("ETH_RECALL_DIP30_TP50_SL30_2H")

STRATEGIES["ETH_RECALL_DIP30_BE25_TP80_SL40_T2H"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.60, "horizon_min": 120,
     "be_activation": 0.25, "label": "main"},
]
STRATEGY_FILTERS["ETH_RECALL_DIP30_BE25_TP80_SL40_T2H"] = {
    "chain": "ethereum",
    "require_recall": True, "min_recall_drift": -0.70, "max_recall_drift": -0.30,
    "max_hours_since_first": 24,
}
SHADOW_STRATEGIES.append("ETH_RECALL_DIP30_BE25_TP80_SL40_T2H")

STRATEGIES["ETH_RECALL_DIP50_TP100_SL40_6H"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.60, "horizon_min": 360, "label": "main"},
]
STRATEGY_FILTERS["ETH_RECALL_DIP50_TP100_SL40_6H"] = {
    "chain": "ethereum",
    "require_recall": True, "min_recall_drift": -0.70, "max_recall_drift": -0.50,
    "max_hours_since_first": 48,
}
SHADOW_STRATEGIES.append("ETH_RECALL_DIP50_TP100_SL40_6H")

# Combo: recall dip + score gate (only if rt_score >= 30 — quality filter)
STRATEGIES["RECALL_DIP30_S30_TP50_SL30"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.70, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["RECALL_DIP30_S30_TP50_SL30"] = {
    "require_recall": True, "min_recall_drift": -0.70, "max_recall_drift": -0.30,
    "max_hours_since_first": 24, "min_rt_score": 30,
}
SHADOW_STRATEGIES.append("RECALL_DIP30_S30_TP50_SL30")

# Same but ETH
STRATEGIES["ETH_RECALL_DIP30_S30_TP50_SL30"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.70, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["ETH_RECALL_DIP30_S30_TP50_SL30"] = {
    "chain": "ethereum",
    "require_recall": True, "min_recall_drift": -0.70, "max_recall_drift": -0.30,
    "max_hours_since_first": 24, "min_rt_score": 30,
}
SHADOW_STRATEGIES.append("ETH_RECALL_DIP30_S30_TP50_SL30")

# ============================================================
# v14e.41 — PEAK-based recall family (pump-then-dump pattern).
# Catches the $PARANOID 27 Apr scenario: 1st KOL calls, pump +72%,
# then dumps -54% from peak before next KOL recalls. Drift_vs_1st
# only -21% (filtered out by DIP30) but drift_vs_peak = -54%
# (passes DIP30/DIP50 peak filters). Post-recall ran +220% (3x).
#
# Uses kol_call_outcomes.ath_after_call as the peak reference. If
# null (peak not yet computed), strats requiring peak drift skip.
# ============================================================

# SOL — recall after deep dip from peak (-50 to -30%)
STRATEGIES["RECALL_PEAK30_TP50_SL30"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.70, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["RECALL_PEAK30_TP50_SL30"] = {
    "require_recall": True,
    "min_drift_vs_peak": -0.50, "max_drift_vs_peak": -0.30,
    "max_hours_since_first": 6,
}
SHADOW_STRATEGIES.append("RECALL_PEAK30_TP50_SL30")

# Deeper peak dump (-70 to -50%) — the $PARANOID sweet spot
STRATEGIES["RECALL_PEAK50_TP100_SL40_6H"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.60, "horizon_min": 360, "label": "main"},
]
STRATEGY_FILTERS["RECALL_PEAK50_TP100_SL40_6H"] = {
    "require_recall": True,
    "min_drift_vs_peak": -0.70, "max_drift_vs_peak": -0.50,
    "max_hours_since_first": 6,
}
SHADOW_STRATEGIES.append("RECALL_PEAK50_TP100_SL40_6H")

# Wide TP for the moonshot end of pump-dump-rebuy
STRATEGIES["RECALL_PEAK50_TP200_SL50_6H"] = [
    {"pct": 1.0, "tp_mult": 3.00, "sl_mult": 0.50, "horizon_min": 360, "label": "main"},
]
STRATEGY_FILTERS["RECALL_PEAK50_TP200_SL50_6H"] = {
    "require_recall": True,
    "min_drift_vs_peak": -0.85, "max_drift_vs_peak": -0.50,
    "max_hours_since_first": 12,
}
SHADOW_STRATEGIES.append("RECALL_PEAK50_TP200_SL50_6H")

# Quick recovery scalp (15-30 min after 1st, dip from peak ≥30%)
STRATEGIES["RECALL_PEAK30_FAST_TP50_SL30"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.70, "horizon_min": 30, "label": "main"},
]
STRATEGY_FILTERS["RECALL_PEAK30_FAST_TP50_SL30"] = {
    "require_recall": True,
    "min_drift_vs_peak": -0.70, "max_drift_vs_peak": -0.30,
    "min_hours_since_first": 0.15, "max_hours_since_first": 2,
}
SHADOW_STRATEGIES.append("RECALL_PEAK30_FAST_TP50_SL30")

# Same with BE25 lock for the ones that pop then dump again
STRATEGIES["RECALL_PEAK30_BE25_TP80_SL30"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.70, "horizon_min": 120,
     "be_activation": 0.25, "label": "main"},
]
STRATEGY_FILTERS["RECALL_PEAK30_BE25_TP80_SL30"] = {
    "require_recall": True,
    "min_drift_vs_peak": -0.85, "max_drift_vs_peak": -0.30,
    "max_hours_since_first": 6,
}
SHADOW_STRATEGIES.append("RECALL_PEAK30_BE25_TP80_SL30")

# ETH peak variant
STRATEGIES["ETH_RECALL_PEAK30_TP50_SL30"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.70, "horizon_min": 120, "label": "main"},
]
STRATEGY_FILTERS["ETH_RECALL_PEAK30_TP50_SL30"] = {
    "chain": "ethereum",
    "require_recall": True,
    "min_drift_vs_peak": -0.70, "max_drift_vs_peak": -0.30,
    "max_hours_since_first": 6,
}
SHADOW_STRATEGIES.append("ETH_RECALL_PEAK30_TP50_SL30")

# ============================================================
# v14e.41 — RECALL × proven mechanics matrix.
# Combines the 2 recall modes (DIP_vs_1st / PEAK_vs_ATH) with the
# strategy mechanics that paper-tested positive on 14d (BE25/30,
# LOCK10/15, FAST45/60, SLOW4H/6H, SCALP, DECAY, NZ, S30/S40,
# MCAP_MID, AGE windows, wide-TP moonshots).
# Insight from $PARANOID 27 Apr replay: SL≥-50% is mandatory on
# pump-then-dump recalls because the dump mèche fires SL30/SL40
# BEFORE the recovery pump arrives. All PEAK variants below use
# SL40-SL60 to survive the dip-trough.
# ============================================================

# Helper to register a shadow strat with filter in 1 line
def _add_recall(name, tp_mult, sl_mult, horizon, filt_extra, be_act=None, be_lock=None):
    spec = {"pct": 1.0, "tp_mult": tp_mult, "sl_mult": sl_mult,
            "horizon_min": horizon, "label": "main"}
    if be_act is not None:
        spec["be_activation"] = be_act
    if be_lock is not None:
        spec["be_lock_pct"] = be_lock
    STRATEGIES[name] = [spec]
    base = {"require_recall": True}
    base.update(filt_extra)
    STRATEGY_FILTERS[name] = base
    SHADOW_STRATEGIES.append(name)


# --- DIP family (drift_vs_first_call_price) — broaden mechanic coverage ---
# BE + LOCK on dip recalls
_add_recall("RECALL_DIP30_BE15_LOCK5_TP50_SL30", 1.50, 0.70, 120,
    {"min_recall_drift": -0.70, "max_recall_drift": -0.30, "max_hours_since_first": 24},
    be_act=0.15, be_lock=0.05)
_add_recall("RECALL_DIP30_BE25_LOCK10_TP80_SL30", 1.80, 0.70, 120,
    {"min_recall_drift": -0.70, "max_recall_drift": -0.30, "max_hours_since_first": 24},
    be_act=0.25, be_lock=0.10)
_add_recall("RECALL_DIP30_BE25_LOCK10_TP100_SL40", 2.00, 0.60, 240,
    {"min_recall_drift": -0.70, "max_recall_drift": -0.30, "max_hours_since_first": 24},
    be_act=0.25, be_lock=0.10)
_add_recall("RECALL_DIP30_BE30_LOCK15_TP100_SL40", 2.00, 0.60, 240,
    {"min_recall_drift": -0.70, "max_recall_drift": -0.30, "max_hours_since_first": 24},
    be_act=0.30, be_lock=0.15)

# FAST timeouts on dip recalls
_add_recall("RECALL_DIP30_FAST45_TP50_SL30", 1.50, 0.70, 45,
    {"min_recall_drift": -0.70, "max_recall_drift": -0.30, "max_hours_since_first": 24})
_add_recall("RECALL_DIP30_FAST60_TP40_SL30", 1.40, 0.70, 60,
    {"min_recall_drift": -0.70, "max_recall_drift": -0.30, "max_hours_since_first": 24})

# SLOW timeouts (let the moonshot run)
_add_recall("RECALL_DIP30_SLOW4H_TP100_SL50", 2.00, 0.50, 240,
    {"min_recall_drift": -0.70, "max_recall_drift": -0.30, "max_hours_since_first": 24})
_add_recall("RECALL_DIP30_SLOW6H_TP150_SL50", 2.50, 0.50, 360,
    {"min_recall_drift": -0.70, "max_recall_drift": -0.30, "max_hours_since_first": 48})
_add_recall("RECALL_DIP30_SLOW6H_TP200_SL50", 3.00, 0.50, 360,
    {"min_recall_drift": -0.70, "max_recall_drift": -0.30, "max_hours_since_first": 48})

# SCALP on dip — tight TP, capture the bounce
_add_recall("RECALL_DIP30_SCALP_TP15_SL15", 1.15, 0.85, 120,
    {"min_recall_drift": -0.70, "max_recall_drift": -0.30, "max_hours_since_first": 24})
_add_recall("RECALL_DIP30_SCALP_TP20_SL15", 1.20, 0.85, 120,
    {"min_recall_drift": -0.70, "max_recall_drift": -0.30, "max_hours_since_first": 24})

# DECAY — TP decays from 50% to 15% over the horizon
STRATEGIES["RECALL_DIP30_DECAY_TP50_E15"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.70, "horizon_min": 120,
     "tp_decay_end": 1.15, "label": "main"},
]
STRATEGY_FILTERS["RECALL_DIP30_DECAY_TP50_E15"] = {
    "require_recall": True,
    "min_recall_drift": -0.70, "max_recall_drift": -0.30, "max_hours_since_first": 24,
}
SHADOW_STRATEGIES.append("RECALL_DIP30_DECAY_TP50_E15")

# Quality gates on dip
_add_recall("RECALL_DIP30_NZ_TP50_SL30", 1.50, 0.70, 120,
    {"min_recall_drift": -0.70, "max_recall_drift": -0.30, "max_hours_since_first": 24,
     "min_liquidity_usd": 1.0})
_add_recall("RECALL_DIP30_S40_TP100_SL40", 2.00, 0.60, 240,
    {"min_recall_drift": -0.70, "max_recall_drift": -0.30, "max_hours_since_first": 24,
     "min_rt_score": 40})
_add_recall("RECALL_DIP30_MCAP_TP100_SL40", 2.00, 0.60, 240,
    {"min_recall_drift": -0.70, "max_recall_drift": -0.30, "max_hours_since_first": 24,
     "min_mcap": 30_000, "max_mcap": 500_000})

# Wide-TP moonshots on dip
_add_recall("RECALL_DIP30_TP100_SL50", 2.00, 0.50, 240,
    {"min_recall_drift": -0.70, "max_recall_drift": -0.30, "max_hours_since_first": 24})
_add_recall("RECALL_DIP30_TP150_SL50", 2.50, 0.50, 240,
    {"min_recall_drift": -0.70, "max_recall_drift": -0.30, "max_hours_since_first": 24})
_add_recall("RECALL_DIP30_TP200_SL50_6H", 3.00, 0.50, 360,
    {"min_recall_drift": -0.70, "max_recall_drift": -0.30, "max_hours_since_first": 48})
_add_recall("RECALL_DIP50_TP200_SL50_6H_OVERRIDE", 3.00, 0.50, 360,
    {"min_recall_drift": -0.85, "max_recall_drift": -0.50, "max_hours_since_first": 72})

# AGE-windowed dip recalls
_add_recall("RECALL_DIP30_AGE2H_BE25_TP80_SL30", 1.80, 0.70, 120,
    {"min_recall_drift": -0.70, "max_recall_drift": -0.30,
     "min_hours_since_first": 0.15, "max_hours_since_first": 2},
    be_act=0.25)
_add_recall("RECALL_DIP30_AGE6H_LOCK10_TP100_SL40", 2.00, 0.60, 240,
    {"min_recall_drift": -0.70, "max_recall_drift": -0.30,
     "min_hours_since_first": 0.15, "max_hours_since_first": 6},
    be_act=0.25, be_lock=0.10)


# --- PEAK family (drift_vs_post-1st-call ATH) — SL≥50% mandatory ---
# Wider SL variants of the existing peak strats (mèche-survivable)
_add_recall("RECALL_PEAK30_TP50_SL50", 1.50, 0.50, 120,
    {"min_drift_vs_peak": -0.70, "max_drift_vs_peak": -0.30, "max_hours_since_first": 6})
_add_recall("RECALL_PEAK30_TP80_SL50", 1.80, 0.50, 240,
    {"min_drift_vs_peak": -0.70, "max_drift_vs_peak": -0.30, "max_hours_since_first": 6})

# BE/LOCK on peak dump
_add_recall("RECALL_PEAK30_BE25_LOCK10_TP100_SL50", 2.00, 0.50, 240,
    {"min_drift_vs_peak": -0.70, "max_drift_vs_peak": -0.30, "max_hours_since_first": 6},
    be_act=0.25, be_lock=0.10)
_add_recall("RECALL_PEAK50_BE25_LOCK15_TP150_SL50_6H", 2.50, 0.50, 360,
    {"min_drift_vs_peak": -0.85, "max_drift_vs_peak": -0.50, "max_hours_since_first": 12},
    be_act=0.25, be_lock=0.15)

# FAST/SLOW timeouts on peak
_add_recall("RECALL_PEAK30_FAST45_TP50_SL50", 1.50, 0.50, 45,
    {"min_drift_vs_peak": -0.70, "max_drift_vs_peak": -0.30,
     "min_hours_since_first": 0.15, "max_hours_since_first": 2})
_add_recall("RECALL_PEAK50_SLOW4H_TP100_SL50", 2.00, 0.50, 240,
    {"min_drift_vs_peak": -0.85, "max_drift_vs_peak": -0.50, "max_hours_since_first": 6})

# SCALP on peak (capture quick bounce off the dip)
_add_recall("RECALL_PEAK30_SCALP_TP15_SL20", 1.15, 0.80, 120,
    {"min_drift_vs_peak": -0.70, "max_drift_vs_peak": -0.30, "max_hours_since_first": 6})
_add_recall("RECALL_PEAK50_SCALP_TP20_SL30", 1.20, 0.70, 120,
    {"min_drift_vs_peak": -0.85, "max_drift_vs_peak": -0.50, "max_hours_since_first": 6})

# Wide-TP / extreme moonshots on deep peak dumps (the $PARANOID +220% pattern)
_add_recall("RECALL_PEAK50_TP100_SL50_6H", 2.00, 0.50, 360,
    {"min_drift_vs_peak": -0.85, "max_drift_vs_peak": -0.50, "max_hours_since_first": 12})
_add_recall("RECALL_PEAK50_TP200_SL60_6H", 3.00, 0.40, 360,
    {"min_drift_vs_peak": -0.85, "max_drift_vs_peak": -0.50, "max_hours_since_first": 12})
_add_recall("RECALL_PEAK70_TP200_SL50_6H", 3.00, 0.50, 360,
    {"min_drift_vs_peak": -0.95, "max_drift_vs_peak": -0.70, "max_hours_since_first": 24})
_add_recall("RECALL_PEAK70_TP500_SL60_6H", 6.00, 0.40, 360,
    {"min_drift_vs_peak": -0.95, "max_drift_vs_peak": -0.70, "max_hours_since_first": 24})

# Score-gated peak
_add_recall("RECALL_PEAK30_S30_TP80_SL50", 1.80, 0.50, 240,
    {"min_drift_vs_peak": -0.70, "max_drift_vs_peak": -0.30,
     "max_hours_since_first": 6, "min_rt_score": 30})


# --- ETH variants of the proven mechanics ---
_add_recall("ETH_RECALL_DIP30_BE25_LOCK10_TP80_SL40_T2H", 1.80, 0.60, 120,
    {"chain": "ethereum",
     "min_recall_drift": -0.70, "max_recall_drift": -0.30, "max_hours_since_first": 24},
    be_act=0.25, be_lock=0.10)
_add_recall("ETH_RECALL_DIP30_TP150_SL50_4H", 2.50, 0.50, 240,
    {"chain": "ethereum",
     "min_recall_drift": -0.70, "max_recall_drift": -0.30, "max_hours_since_first": 24})
_add_recall("ETH_RECALL_DIP30_SLOW4H_TP100_SL50", 2.00, 0.50, 240,
    {"chain": "ethereum",
     "min_recall_drift": -0.70, "max_recall_drift": -0.30, "max_hours_since_first": 48})
_add_recall("ETH_RECALL_PEAK30_TP100_SL50", 2.00, 0.50, 240,
    {"chain": "ethereum",
     "min_drift_vs_peak": -0.70, "max_drift_vs_peak": -0.30, "max_hours_since_first": 6})
_add_recall("ETH_RECALL_PEAK50_TP200_SL50_6H", 3.00, 0.50, 360,
    {"chain": "ethereum",
     "min_drift_vs_peak": -0.85, "max_drift_vs_peak": -0.50, "max_hours_since_first": 12})
_add_recall("ETH_RECALL_PEAK50_BE25_LOCK10_TP150_SL40_T4H", 2.50, 0.60, 240,
    {"chain": "ethereum",
     "min_drift_vs_peak": -0.85, "max_drift_vs_peak": -0.50, "max_hours_since_first": 12},
    be_act=0.25, be_lock=0.10)


# ---------------------------------------------------------------------------
# v14e — Chain-indexed strategy registry.
# ---------------------------------------------------------------------------
# Until BSC/Base go live this is purely bookkeeping: STRATEGY_FILTERS already
# gates each strategy by chain ("chain": "ethereum" etc.). This registry gives
# callers (bankroll, allocations, daily summary) an O(1) "what strategies run
# on chain X?" lookup without scanning every filter dict.
#
# Convention:
#   - Solana strategies have NO "chain" key in their filter (implicit solana).
#   - Non-Solana strategies MUST declare their chain in STRATEGY_FILTERS.
#   - Adding a new chain: add its list here + tag each strategy with
#     "chain": "<chain>" in STRATEGY_FILTERS. Enforcement lives in
#     _passes_strategy_filter (paper_trader.py:153-166).

def _build_chain_strategies() -> dict:
    """Partition STRATEGIES by declared chain filter. Built on import;
    keep the dict static — mutating at runtime would desync from DB allocs."""
    buckets: dict = {"solana": [], "ethereum": [], "bsc": [], "base": []}
    for sname in STRATEGIES.keys():
        flt = STRATEGY_FILTERS.get(sname) or {}
        c = flt.get("chain", "solana")
        buckets.setdefault(c, []).append(sname)
    return buckets


CHAIN_STRATEGIES = _build_chain_strategies()


def strategies_for_chain(chain: str) -> list[str]:
    """Return the strategy names allowed on one chain. Stable ordering for
    bankroll seeding + daily summary display."""
    return sorted(CHAIN_STRATEGIES.get((chain or "solana").lower(), []))


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


# ---------------------------------------------------------------------------
# v14e.43 — BSR (rt_buy_sell_ratio) filter A/B shadows.
# Driven by `scripts/_score_reverse_engineer.py` findings on N=65K SOL +
# N=3.2K ETH closed shadow trades:
#   SOL: `rt_buy_sell_ratio >= 0.52` lifts WR by +5-7pp on cross-strat
#        single-feature scan (universal pattern, ~5% of trades excluded).
#   ETH: combo `rt_volume_24h >= $6K AND rt_buy_sell_ratio >= 0.55` lifts
#        WR by +40-48pp on BE+LOCK T2H family (4 strats agree → real signal).
# We do NOT direct-apply the gate — the CLAUDE.md rule is "TOUJOURS paired-
# test". Instead, clone the top 5 SOL + top 5 ETH strats with `_BSR52`/
# `_BSR55` suffix and let them run as shadows ~7-14d. Verdict at N>=30
# paired-test vs base. If positive in live conditions too, add as filter
# variant in the strategy registry.
# ---------------------------------------------------------------------------

# SOL — BSR52 KILLED v14e.49 (Apr 30 2026). 7d shadow on N=33 each:
# BE25_TP80_SL30_BSR52 avg -5.41% WR 27.3%, FAST_TP50_SL30_BSR52 avg -5.75% WR 33.3%.
# Confirmed the "EXPECTED NEGATIVE" warning in the original deploy note —
# BSR alone sacrifices fat-tail moonshots without enough WR uplift.
# Combo `BSR + entry_mcap` (see BSR_MCAP_AB) still under test.

# ETH — BSR55 clones of top 5 BE+LOCK family (where the combo signal is
# strongest per reverse-engineer findings). KEPT — N=11 each, all 5 positive
# (avg +5 to +19%, WR 36-64%). Let run for verdict at N>=30.
_BSR_ETH_CLONES = [
    ("ETH_TP80_SL40_T2H_BSR55", 1.80, 0.60, 120, {}),
    ("ETH_BE25_LOCK10_TP80_SL40_T2H_BSR55", 1.80, 0.60, 120,
        {"be_activation": 0.25, "be_lock_pct": 0.10}),
    ("ETH_BE15_LOCK5_TP80_SL30_BSR55", 1.80, 0.70, 30,
        {"be_activation": 0.15, "be_lock_pct": 0.05}),
    ("ETH_BE15_LOCK10_TP80_SL30_BSR55", 1.80, 0.70, 30,
        {"be_activation": 0.15, "be_lock_pct": 0.10}),
    ("ETH_BE25_LOCK15_TP100_SL40_T2H_BSR55", 2.00, 0.60, 120,
        {"be_activation": 0.25, "be_lock_pct": 0.15}),
]
for name, tp, sl, h, extra in _BSR_ETH_CLONES:
    spec = {"pct": 1.0, "tp_mult": tp, "sl_mult": sl, "horizon_min": h, "label": "main"}
    spec.update(extra)
    STRATEGIES[name] = [spec]
    STRATEGY_FILTERS[name] = {"chain": "ethereum", "min_buy_sell_ratio": 0.55}
    SHADOW_STRATEGIES.append(name)

# ---------------------------------------------------------------------------
# v14e.43b — KW34 (kol_win_rate >= 0.34) shadow A/B family.
# Driven by `scripts/_score_reverse_engineer.py` v2 with TARGET=$/day (not WR).
# Walk-forward CV (train 21d → test 9d):
#   SOL: train +$777/d, test +$222/d → ✅ HOLD out-of-sample
#   ETH: train +$41/d,  test +$440/d → ✅ HOLD strongly
# This is the cross-chain validated single-feature filter. Adding 5 SOL +
# 3 ETH shadow clones to validate in vivo. Verdict at N>=30 paired-test.
# ---------------------------------------------------------------------------

_KW34_SOL_CLONES = [
    ("BE25_TP80_SL30_KW34", 1.80, 0.70, 30, {"be_activation": 0.25}, {}),
    ("FAST_TP50_SL30_KW34", 1.50, 0.70, 30, {}, {}),
    ("FAST_TP50_SL30_S40_KW34", 1.50, 0.70, 30, {}, {"min_rt_score": 40}),
    ("BE15_LOCK5_TP50_SL30_KW34", 1.50, 0.70, 30,
        {"be_activation": 0.15, "be_lock_pct": 0.05}, {}),
    ("SLOW6H_TP100_SL50_KW34", 2.00, 0.50, 360, {}, {}),
]
for name, tp, sl, h, extra, filt_extra in _KW34_SOL_CLONES:
    spec = {"pct": 1.0, "tp_mult": tp, "sl_mult": sl, "horizon_min": h, "label": "main"}
    spec.update(extra)
    STRATEGIES[name] = [spec]
    STRATEGY_FILTERS[name] = {"chain": "solana", "min_kol_win_rate": 0.34, **filt_extra}
    SHADOW_STRATEGIES.append(name)

_KW26_ETH_CLONES = [
    ("ETH_TP80_SL40_T2H_KW26", 1.80, 0.60, 120, {}),
    ("ETH_FAST_TP100_SL50_KW26", 2.00, 0.50, 30, {}),
    ("ETH_BE25_LOCK10_TP80_SL40_T2H_KW26", 1.80, 0.60, 120,
        {"be_activation": 0.25, "be_lock_pct": 0.10}),
]
for name, tp, sl, h, extra in _KW26_ETH_CLONES:
    spec = {"pct": 1.0, "tp_mult": tp, "sl_mult": sl, "horizon_min": h, "label": "main"}
    spec.update(extra)
    STRATEGIES[name] = [spec]
    STRATEGY_FILTERS[name] = {"chain": "ethereum", "min_kol_win_rate": 0.26}
    SHADOW_STRATEGIES.append(name)

# ---------------------------------------------------------------------------
# v14e.43b — BSR_MCAP combo shadows. The single BSR filter loses $/d but the
# combo `rt_buy_sell_ratio >= 0.53 AND entry_mcap >= $45K` validated +$20-26/d
# on SLOW6H_TP100_SL50, SLOW4H_TP100_SL50, TP100_SL60, TP80_SL70 (currently
# losing strats that flip positive with the combo). 4 shadow clones SOL only.
# ---------------------------------------------------------------------------
_BSR_MCAP_SOL_CLONES = [
    ("SLOW6H_TP100_SL50_BSR_MCAP", 2.00, 0.50, 360, {}),
    ("SLOW4H_TP100_SL50_BSR_MCAP", 2.00, 0.50, 240, {}),
    ("TP100_SL60_BSR_MCAP", 2.00, 0.40, 120, {}),
    ("TP80_SL70_BSR_MCAP", 1.80, 0.30, 120, {}),
]
for name, tp, sl, h, extra in _BSR_MCAP_SOL_CLONES:
    spec = {"pct": 1.0, "tp_mult": tp, "sl_mult": sl, "horizon_min": h, "label": "main"}
    spec.update(extra)
    STRATEGIES[name] = [spec]
    STRATEGY_FILTERS[name] = {
        "chain": "solana",
        "min_buy_sell_ratio": 0.53,
        "min_entry_mcap": 45000,
    }
    SHADOW_STRATEGIES.append(name)

# ---------------------------------------------------------------------------
# v14e.36 — auto-deprecate every artifact-family strategy registered above.
# Trail/dip/split/bond shadows pollute analytics (sim ranks them top via
# 47x slip miscalibration) and cannot be promoted to live. This finalization
# step ensures any name matching the artifact prefixes — present or future —
# is dropped to deprecated without manual maintenance.
# ---------------------------------------------------------------------------
_AUTO_DEPRECATED = {name for name in STRATEGIES.keys() if _is_artifact_family(name)}
_DEFAULT_DEPRECATED |= _AUTO_DEPRECATED
