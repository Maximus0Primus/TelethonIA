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
# Fee constants (v121: Jupiter Ultra RFQ — near-zero slippage)
# ---------------------------------------------------------------------------
BUY_SLIPPAGE_BPS = 10    # 0.1% — Jupiter Ultra platform fee
SELL_SLIPPAGE_BPS = 10   # 0.1% — Jupiter Ultra platform fee
BUY_FEE_BPS = 0          # 0% — folded into slippage
SELL_FEE_BPS = 0          # 0% — folded into slippage

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
# Top 10 (NONE filter, hysteresis/lazy) — the big surprise
STRATEGIES["FAST_TP100_SL20_HYST"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.80, "horizon_min": 30, "label": "main"},
]
SHADOW_STRATEGIES.append("FAST_TP100_SL20_HYST")

STRATEGIES["FAST_TP80_SL25_HYST"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.75, "horizon_min": 30, "label": "main"},
]
SHADOW_STRATEGIES.append("FAST_TP80_SL25_HYST")

STRATEGIES["BE25_TP80_SL30_HYST"] = [
    {"pct": 1.0, "tp_mult": 1.80, "sl_mult": 0.70, "horizon_min": 30,
     "be_activation": 0.25, "label": "main"},
]
SHADOW_STRATEGIES.append("BE25_TP80_SL30_HYST")

STRATEGIES["FAST_TP50_SL30_HYST"] = [
    {"pct": 1.0, "tp_mult": 1.50, "sl_mult": 0.70, "horizon_min": 30, "label": "main"},
]
SHADOW_STRATEGIES.append("FAST_TP50_SL30_HYST")

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
    "BE25_TP80_SL30_DS",
    "FAST_TP100_SL20",
    "FAST_TP80_SL25",
    "FAST_TP50_SL30",
    "FAST_TP40_SL30",
    "TP50_SL15",
    # v140: new hysteresis+lazy variants (full sweep top 10)
    "FAST_TP100_SL20_HYST",
    "FAST_TP80_SL25_HYST",
    "BE25_TP80_SL30_HYST",
    "FAST_TP50_SL30_HYST",
    # legacy
    "DTRAIL3_ACT5_SL60",
    "DTRAIL5_ACT10_SL60",
    "DIP30_B5_T5_A20_SL70_240m",
}
LAZY_FAST_SEC = 180     # 3 min during fast phase
LAZY_FAST_WINDOW = 300  # 5 min fast phase
LAZY_SLOW_SEC = 600     # 10 min after fast phase


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
