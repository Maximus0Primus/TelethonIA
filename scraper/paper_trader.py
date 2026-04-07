"""
Paper Trading System v6 — Multi-strategy with tranche support + portfolio allocation.

Strategies run in parallel per token:
- TP30_SL10:    100% at 1.3x, -10% SL, 12h horizon (v86: replaces TP30_SL50, 3:1 R:R)
- TP50_SL30:    100% at 1.5x, -30% SL, 24h horizon (RT best 7d: +8% ROI, 50% WR)
- TP100_SL50:   100% at 2x,   -50% SL, 48h horizon (v6: wide SL to survive memecoin volatility)
- FRESH_MICRO:  100% at 1.3x, -70% SL, 24h (score 10-49, fresh KOL, micro-cap)
- QUICK_SCALP:  100% at 1.5x, ~no SL, 6h timeout (score 10-49, momentum)
- TP50_SL15:    100% at 1.5x, -15% SL, 24h horizon (tight SL, best EV on KOL-filtered tokens)
- TP30_SL30:    100% at 1.3x, -30% SL, 12h horizon (symmetric risk/reward, cuts losses early)
- TP50_SL50:    100% at 1.5x, -50% SL, 24h horizon (symmetric risk/reward, room to recover)

Deprecated (data shows negative EV — kept in code for backtesting, removed from active_strategies):
- TP30_SL50:   -2.5% ROI, 49% WR. Asymmetric R:R — wins too small vs losses.
- TP100_SL30:  -14% ROI live, 12% WR. Tight SL (-30%) too small for 2x target.
- MOONBAG:     -45.6% ROI live, 8% WR.
- WIDE_RUNNER: -70.4% ROI live, 0% WR in RT.
- SCALE_OUT:   -29% ROI live, 6% WR.
- QUICK_SCALP: v92 deprecated — 33% WR, negative PnL. 6h too short for TP30.
- TP30_SL10:   v92 deprecated — tight SL (-10%) stops out too fast.
- TP50_SL15:   v92 deprecated — tight SL (-15%) kills trades that would recover.
- TP30_SL30:   v92 deprecated — TP30 too small to overcome slippage.

Each tranche = 1 row in paper_trades. SL triggers close ALL open tranches
for the same token+strategy. Moonbag tranches (tp_price=NULL) only close
on SL or timeout.

v3: Score-weighted portfolio allocation. $50 budget per cycle split
proportionally by token score. Tracks position_usd and pnl_usd.
v4: Data-driven strategies with entry filters (STRATEGY_FILTERS).
v5: +TP30_SL50, dedup cooldown 24h default, MOONBAG/WIDE_RUNNER deprecated.
"""

import logging
from datetime import datetime, timezone, timedelta

import requests
import time as _time_mod

logger = logging.getLogger(__name__)

# v67: Monitoring — conditional import
try:
    from monitor import metrics as _metrics, estimate_egress as _estimate_egress
    _monitoring = True
except ImportError:
    _monitoring = False

DEXSCREENER_BATCH_URL = "https://api.dexscreener.com/tokens/v1/solana/{addresses}"
BATCH_SIZE = 30

# v88: Bot ML predictions — precomputed in GH Actions, read from Supabase
_BOT_PREDICTIONS: dict = {}  # {(token_address, strategy): gate_mult}

# --- Defaults (overridden by scoring_config.paper_trade_config) ---
TOP_N = 5
PORTFOLIO_BUDGET = 200.0  # v94: USD per cycle, score-weighted across top N (was 50)
DEDUP_COOLDOWN_HOURS = 24  # v5: was 0 — re-trading same token across cycles was the #1 PnL killer
CA_FILTER = True

# v92: Default deprecated strategies — overridable via paper_trade_config JSONB
_DEFAULT_DEPRECATED = {"MOONBAG", "WIDE_RUNNER", "SCALE_OUT", "TP100_SL30", "QUICK_SCALP", "TP30_SL10", "TP50_SL15", "TP30_SL30"}

# Shadow trading: single-tranche strategies eligible for $0 shadow trades.
# Excluded: multi-tranche (SCALE_OUT, MOONBAG, WIDE_RUNNER) and legacy deprecated.
SHADOW_STRATEGIES = [
    "TP30_SL50", "TP50_SL30", "TP100_SL30", "TP100_SL50",
    "TP50_SL15", "TP30_SL30", "TP50_SL50", "FRESH_MICRO", "QUICK_SCALP",
    "TP30_SL10",
]

# v73: Slippage simulation — realistic entry/exit price adjustments
BUY_SLIPPAGE_BPS = 100   # 1.0% buy slippage (v92: was 150 — too aggressive)
SELL_SLIPPAGE_BPS = 200   # 2.0% sell slippage (v92: was 300 — TP needed +54% raw move)

# v94: Fee simulation — Jupiter priority fees on buy + sell
BUY_FEE_BPS = 50    # 0.5% Jupiter priority fee on buy
SELL_FEE_BPS = 50   # 0.5% Jupiter priority fee on sell

# --- Strategy Definitions ---
# Each strategy has a list of tranches. Moonbag tranches have tp_mult=None.
STRATEGIES = {
    "TP30_SL50": [
        # v5: backtest best — TP30_SL50_12h: +8.5% expectancy, 61% WR, PF 1.51 (report #602)
        # KCO sim: 63.2% hit 1.3x across all KOLs, 71.4% on top 10
        # DEPRECATED v86: 49% WR but negative PnL (-2.5% ROI) — asymmetric R:R kills it
        # v101: ALL horizons → 2h (120min). Data proof: <2h trades = +$1,432 (48% WR), >2h = -$6,273 (15% WR)
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
# If a token doesn't pass the filter, that strategy is skipped (other strategies still apply).
STRATEGY_FILTERS = {
    "FRESH_MICRO": {
        "min_score": 10,
        "max_score": 49,
        "min_kol_freshness": 0.01,
        "min_momentum_mult": 1.0,
        "max_mcap": 5_000_000,
    },
    "QUICK_SCALP": {
        "min_score": 10,
        "max_score": 49,
        "min_momentum_mult": 1.0,
    },
    "WIDE_RUNNER": {
        "min_score": 10,
        "max_score": 49,
        "min_kol_freshness": 0.01,
        "max_mcap": 5_000_000,
    },
}

# v93: Grid search — all TP/SL combos for shadow trading optimization
# TP 40-100 (step 10) × SL 30-70 (step 10) = 35 combos + 6 no-SL baselines (TP50-100)
_GRID_STRATEGIES = {}
for _tp in range(40, 110, 10):    # 40, 50, 60, 70, 80, 90, 100
    for _sl in range(30, 80, 10):  # 30, 40, 50, 60, 70
        _name = f"TP{_tp}_SL{_sl}"
        if _name not in STRATEGIES:
            _GRID_STRATEGIES[_name] = [
                {"pct": 1.0, "tp_mult": 1 + _tp / 100, "sl_mult": 1 - _sl / 100,
                 "horizon_min": 120, "label": "main"},
            ]
# No-SL baseline: only exits via TP or timeout (SL at -80% = nearly unreachable)
for _tp_nosl in [50, 60, 70, 80, 90, 100]:
    _GRID_STRATEGIES[f"TP{_tp_nosl}_NOSL"] = [
        {"pct": 1.0, "tp_mult": 1 + _tp_nosl / 100, "sl_mult": 0.20,
         "horizon_min": 120, "label": "main"},
    ]

STRATEGIES.update(_GRID_STRATEGIES)
SHADOW_STRATEGIES.extend(_GRID_STRATEGIES.keys())

# v105: Scalp grid — small TP, standard 2h timeout (shadow-only, test volume strategy)
# Hypothesis: catch the initial 10-20% pump post KOL call. TP triggers fast if it works.
# Timeout stays 2h like all strategies — analyze exit_minutes to see speed of TP hits.
_SCALP_STRATEGIES = {}
for _tp_s in [10, 15, 20]:
    for _sl_s in [10, 15, 20, 30]:
        if _sl_s > _tp_s * 2:
            continue  # skip nonsensical combos like TP10_SL30
        _sname = f"SCALP_TP{_tp_s}_SL{_sl_s}"
        _SCALP_STRATEGIES[_sname] = [
            {"pct": 1.0, "tp_mult": 1 + _tp_s / 100, "sl_mult": 1 - _sl_s / 100,
             "horizon_min": 120, "label": "main"},
        ]
    # NOSL variant
    _SCALP_STRATEGIES[f"SCALP_TP{_tp_s}_NOSL"] = [
        {"pct": 1.0, "tp_mult": 1 + _tp_s / 100, "sl_mult": 0.20,
         "horizon_min": 120, "label": "main"},
    ]

STRATEGIES.update(_SCALP_STRATEGIES)
SHADOW_STRATEGIES.extend(_SCALP_STRATEGIES.keys())

# v106: Time-decay TP grid — shadow-only A/B test.
# Concept: TP threshold decays linearly from tp_mult to tp_decay_end over the
# second half of the horizon. Captures "wasted pumps" that spike then fade.
# Field `tp_decay_end` on a tranche activates this; absence = normal TP.
_DECAY_STRATEGIES = {}
for _tp_d, _sl_d, _end_d in [
    (100, 50, 20),   # TP100 decay→20%, SL50
    (100, 50, 15),   # TP100 decay→15%, SL50
    (100, 50, 30),   # TP100 decay→30%, SL50
    (70,  50, 15),   # TP70 decay→15%, SL50
    (50,  30, 15),   # TP50 decay→15%, SL30
    (50,  50, 15),   # TP50 decay→15%, SL50
]:
    _dname = f"DECAY_TP{_tp_d}_SL{_sl_d}_E{_end_d}"
    _DECAY_STRATEGIES[_dname] = [
        {"pct": 1.0, "tp_mult": 1 + _tp_d / 100,
         "sl_mult": 1 - _sl_d / 100, "horizon_min": 120,
         "tp_decay_end": 1 + _end_d / 100,   # e.g., 1.20 for +20%
         "label": "main"},
    ]

STRATEGIES.update(_DECAY_STRATEGIES)
SHADOW_STRATEGIES.extend(_DECAY_STRATEGIES.keys())

# v106: Fast timeout grid — 30min horizon A/B test vs standard 2h.
# Hypothesis: 81% of TP hits happen in first 30min. Shorter timeout cuts losses.
_FAST_STRATEGIES = {}
for _tp_f, _sl_f in [
    (100, 50),  # FAST_TP100_SL50
    (50,  30),  # FAST_TP50_SL30
    (50,  50),  # FAST_TP50_SL50
    (70,  50),  # FAST_TP70_SL50
    (40,  30),  # FAST_TP40_SL30
]:
    _fname = f"FAST_TP{_tp_f}_SL{_sl_f}"
    _FAST_STRATEGIES[_fname] = [
        {"pct": 1.0, "tp_mult": 1 + _tp_f / 100,
         "sl_mult": 1 - _sl_f / 100, "horizon_min": 30,
         "label": "main"},
    ]

STRATEGIES.update(_FAST_STRATEGIES)
SHADOW_STRATEGIES.extend(_FAST_STRATEGIES.keys())

# v106: 1h timeout grid — between 30min (FAST) and 2h (standard).
_FAST60_STRATEGIES = {}
for _tp_f6, _sl_f6 in [
    (100, 50), (50, 30), (50, 50), (70, 50), (40, 30),
]:
    _f6name = f"FAST60_TP{_tp_f6}_SL{_sl_f6}"
    _FAST60_STRATEGIES[_f6name] = [
        {"pct": 1.0, "tp_mult": 1 + _tp_f6 / 100,
         "sl_mult": 1 - _sl_f6 / 100, "horizon_min": 60,
         "label": "main"},
    ]

STRATEGIES.update(_FAST60_STRATEGIES)
SHADOW_STRATEGIES.extend(_FAST60_STRATEGIES.keys())

# v106: 45min timeout grid — fills gap between 30min and 60min.
_FAST45_STRATEGIES = {}
for _tp_f45, _sl_f45 in [
    (100, 50), (50, 30), (50, 50), (70, 50), (40, 30),
]:
    _f45name = f"FAST45_TP{_tp_f45}_SL{_sl_f45}"
    _FAST45_STRATEGIES[_f45name] = [
        {"pct": 1.0, "tp_mult": 1 + _tp_f45 / 100,
         "sl_mult": 1 - _sl_f45 / 100, "horizon_min": 45,
         "label": "main"},
    ]

STRATEGIES.update(_FAST45_STRATEGIES)
SHADOW_STRATEGIES.extend(_FAST45_STRATEGIES.keys())

# v107: Slow timeout grid — 4h horizon A/B test vs standard 2h.
# Hypothesis: some tokens need more time to pump. Test if longer timeout recovers value.
_SLOW4H_STRATEGIES = {}
for _tp_s4, _sl_s4 in [
    (100, 50), (50, 30), (50, 50), (70, 50), (40, 30),
]:
    _s4name = f"SLOW4H_TP{_tp_s4}_SL{_sl_s4}"
    _SLOW4H_STRATEGIES[_s4name] = [
        {"pct": 1.0, "tp_mult": 1 + _tp_s4 / 100,
         "sl_mult": 1 - _sl_s4 / 100, "horizon_min": 240,
         "label": "main"},
    ]

STRATEGIES.update(_SLOW4H_STRATEGIES)
SHADOW_STRATEGIES.extend(_SLOW4H_STRATEGIES.keys())

# v107: Slow timeout grid — 6h horizon A/B test vs standard 2h.
# If 6h is significantly better than 2h, we're leaving money on the table with early timeouts.
_SLOW6H_STRATEGIES = {}
for _tp_s6, _sl_s6 in [
    (100, 50), (50, 30), (50, 50), (70, 50), (40, 30),
]:
    _s6name = f"SLOW6H_TP{_tp_s6}_SL{_sl_s6}"
    _SLOW6H_STRATEGIES[_s6name] = [
        {"pct": 1.0, "tp_mult": 1 + _tp_s6 / 100,
         "sl_mult": 1 - _sl_s6 / 100, "horizon_min": 360,
         "label": "main"},
    ]

STRATEGIES.update(_SLOW6H_STRATEGIES)
SHADOW_STRATEGIES.extend(_SLOW6H_STRATEGIES.keys())

# v106: Breakeven stop grid — once price exceeds entry*(1+be_activation),
# the SL is moved UP to entry price (breakeven). You can't lose anymore.
# Encoded in strategy name: BE{activation}_TP{tp}_SL{sl}
# be_activation = minimum gain before breakeven activates (e.g., 20 = +20%)
_BE_STRATEGIES = {}
for _be_act in [15, 20, 30]:
    for _tp_be, _sl_be in [(100, 50), (50, 30), (50, 50), (70, 50)]:
        _bename = f"BE{_be_act}_TP{_tp_be}_SL{_sl_be}"
        _BE_STRATEGIES[_bename] = [
            {"pct": 1.0, "tp_mult": 1 + _tp_be / 100,
             "sl_mult": 1 - _sl_be / 100, "horizon_min": 120,
             "be_activation": _be_act / 100,  # e.g., 0.20
             "label": "main"},
        ]

STRATEGIES.update(_BE_STRATEGIES)
SHADOW_STRATEGIES.extend(_BE_STRATEGIES.keys())

# v106: Trailing stop grid — exit when price drops X% from high_price_seen.
# Activates only after price is above entry * (1 + trail_pct) to avoid
# triggering on the initial dip right after entry.
# Encoded in strategy name: TRAIL{pct}_TP{tp}_SL{sl}
_TRAIL_STRATEGIES = {}
for _trail_pct in [10, 15, 20, 25]:
    for _tp_tr, _sl_tr in [(100, 50), (50, 30), (50, 50), (70, 50)]:
        _tname = f"TRAIL{_trail_pct}_TP{_tp_tr}_SL{_sl_tr}"
        _TRAIL_STRATEGIES[_tname] = [
            {"pct": 1.0, "tp_mult": 1 + _tp_tr / 100,
             "sl_mult": 1 - _sl_tr / 100, "horizon_min": 120,
             "trail_pct": _trail_pct / 100,  # e.g., 0.20 for -20% from peak
             "label": "main"},
        ]

STRATEGIES.update(_TRAIL_STRATEGIES)
SHADOW_STRATEGIES.extend(_TRAIL_STRATEGIES.keys())

# v110: Dynamic trail — trail activates only after a gain threshold (activation_pct).
# No TP cap: lets runners run. SL and timeout still apply.
# Encoded: DTRAIL{trail}_ACT{act}_SL{sl}
_DTRAIL_STRATEGIES = {}
for _dt_trail in [3, 5, 10, 15, 20]:  # v116: added trail=3 (sim best)
    for _dt_act in [5, 10, 15, 20, 25, 30]:  # v116: added act=5 (sim best)
        for _dt_sl in [50, 60, 70]:
            _dtname = f"DTRAIL{_dt_trail}_ACT{_dt_act}_SL{_dt_sl}"
            _DTRAIL_STRATEGIES[_dtname] = [
                {"pct": 1.0, "tp_mult": None,  # no TP cap
                 "sl_mult": 1 - _dt_sl / 100, "horizon_min": 120,
                 "trail_pct": _dt_trail / 100,
                 "trail_activation_pct": _dt_act / 100,  # trail activates after this gain
                 "label": "main"},
            ]

STRATEGIES.update(_DTRAIL_STRATEGIES)
SHADOW_STRATEGIES.extend(_DTRAIL_STRATEGIES.keys())

# v106: Split exit — 50% at TP50, 50% runner (TP100 or trailing stop).
# Simpler version of SCALE_OUT (which had 4 tranches and -29% ROI).
# Multi-tranche shadow support added in shadow insertion code.
_SPLIT_STRATEGIES = {}
for _sl_sp in [30, 50]:
    # SPLIT_50_100: 50% at TP50, 50% at TP100
    _SPLIT_STRATEGIES[f"SPLIT_50_100_SL{_sl_sp}"] = [
        {"pct": 0.5, "tp_mult": 1.50, "sl_mult": 1 - _sl_sp / 100,
         "horizon_min": 120, "label": "tp50_half"},
        {"pct": 0.5, "tp_mult": 2.00, "sl_mult": 1 - _sl_sp / 100,
         "horizon_min": 120, "label": "runner_half"},
    ]
    # SPLIT_50_TRAIL: 50% at TP50, 50% runner with trailing -20%
    _SPLIT_STRATEGIES[f"SPLIT_50_TRAIL_SL{_sl_sp}"] = [
        {"pct": 0.5, "tp_mult": 1.50, "sl_mult": 1 - _sl_sp / 100,
         "horizon_min": 120, "label": "tp50_half"},
        {"pct": 0.5, "tp_mult": None, "sl_mult": 1 - _sl_sp / 100,
         "horizon_min": 120, "trail_pct": 0.20, "label": "trail_half"},
    ]

STRATEGIES.update(_SPLIT_STRATEGIES)
SHADOW_STRATEGIES.extend(_SPLIT_STRATEGIES.keys())

# v115: DIP_BUY strategy — P1 enters at KOL call (50%), P2 enters on dip+bounce (50%)
# DIP30 = wait for -30% dip, B5 = +5% bounce confirmation, T5/A15 = trail 5% / activation 15%
STRATEGIES["DIP30_B5_T5_A15_SL70_240m"] = [
    {"pct": 0.5, "tp_mult": None, "sl_mult": 0.30,
     "horizon_min": 240, "trail_pct": 0.05,
     "trail_activation_pct": 0.15, "label": "dip_p1"},
]

# v116: DIP_BUY split — P1 and P2 have independent trail/act/sl params.
# P1: trail 5%, activation 10%, SL 70% (enter at KOL call — tight trail, fast activation)
# P2: trail 10%, activation 15%, SL 60% (enter at dip bounce — wider trail, tighter SL)
STRATEGIES["DIP30_B5_P1T5A10S70_P2T10A15S60_240m"] = [
    {"pct": 0.5, "tp_mult": None, "sl_mult": 0.30,  # P1: SL70 → sl_mult=0.30
     "horizon_min": 240, "trail_pct": 0.05,
     "trail_activation_pct": 0.10, "label": "dip_p1"},
]

# v118: DIP_BUY T5 LAZY — best overall strategy in sim with LAZY interval ($11,457)
# Trail 5% is unreliable with CURRENT checks (exits too fast on micro-pullbacks)
# but with LAZY (3min/10min) the check frequency matches the sim → results should hold
STRATEGIES["DIP30_B5_T5_A20_SL70_240m"] = [
    {"pct": 0.5, "tp_mult": None, "sl_mult": 0.30,
     "horizon_min": 240, "trail_pct": 0.05,
     "trail_activation_pct": 0.20, "label": "dip_p1"},
]

# v118: DIP_BUY T10 — best reliable strategy from sim (trail 10%, activation 30%, SL 60%)
# Shared params: both P1 and P2 use the same trail/act/sl
STRATEGIES["DIP30_B5_T10_A30_SL60_240m"] = [
    {"pct": 0.5, "tp_mult": None, "sl_mult": 0.40,
     "horizon_min": 240, "trail_pct": 0.10,
     "trail_activation_pct": 0.30, "label": "dip_p1"},
]

# v115: DIP_BUY in-memory watchlist — tracks tokens waiting for dip+bounce to open P2
# Key: (token_address, strategy_name) → tracking state
_dip_watchlist: dict[tuple, dict] = {}
_dip_watchlist_rebuilt = False


def _load_bot_predictions(client) -> None:
    """v88: Load precomputed bot ML predictions from Supabase (one query per cycle)."""
    global _BOT_PREDICTIONS
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        result = (
            client.table("bot_ml_predictions")
            .select("token_address, strategy, gate_mult")
            .gte("predicted_at", cutoff)
            .execute()
        )
        _BOT_PREDICTIONS = {
            (r["token_address"], r["strategy"]): float(r["gate_mult"])
            for r in (result.data or [])
        }
        if _BOT_PREDICTIONS:
            logger.info("bot_ml: loaded %d predictions", len(_BOT_PREDICTIONS))
    except Exception as e:
        logger.warning("bot_ml: load failed: %s", e)
        _BOT_PREDICTIONS = {}


def _bot_ml_gate(token: dict, strategy_name: str, config: dict | None = None) -> float:
    """v90: ML gate with inversion support. Model is anti-predictive (7d data N=189):
    blocked=60% WR, full=35% WR. Modes: disabled/normal/inverted."""
    mode = (config or {}).get("ml_gate_mode", "disabled")
    if mode == "disabled":
        return 1.0

    raw = _BOT_PREDICTIONS.get(
        (token.get("token_address"), strategy_name), 1.0
    )

    if mode == "inverted":
        # Anti-predictive model: what it blocks performs best. Flip the signal.
        if raw <= 0.0:
            return 1.5   # model said SKIP → BOOST 50%
        elif raw < 1.0:
            return 1.3   # model said HALF → BOOST 30%
        else:
            return 0.7   # model said FULL → REDUCE 30%

    return raw  # normal mode


def _passes_strategy_filter(token: dict, strategy_name: str) -> bool:
    """Check if a token passes the entry filter for a given strategy."""
    filt = STRATEGY_FILTERS.get(strategy_name)
    if not filt:
        return True  # no filter = always pass

    score = token.get("score", 0)
    if score < filt.get("min_score", 0) or score > filt.get("max_score", 100):
        return False
    mcap = float(token.get("market_cap") or 0)
    if filt.get("max_mcap") and mcap > filt["max_mcap"]:
        return False
    kf = float(token.get("kol_freshness") or 0)
    if kf < filt.get("min_kol_freshness", 0):
        return False
    mm = float(token.get("momentum_mult") or 1.0)
    if mm < filt.get("min_momentum_mult", 0):
        return False
    return True


def _fetch_price_fallback(address: str) -> float | None:
    """Fallback: fetch price from GeckoTerminal (no API key needed, on-chain data).
    Catches pool migrations and tokens DexScreener hasn't indexed yet."""
    try:
        resp = requests.get(
            f"https://api.geckoterminal.com/api/v2/networks/solana/tokens/{address}",
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        price = resp.json().get("data", {}).get("attributes", {}).get("price_usd")
        if price:
            return float(price)
    except Exception:
        pass
    return None


# v118: Price tick logger — records DexScreener spot prices for future backtesting
_last_tick_log: dict[str, float] = {}  # token_address -> last log timestamp


def _log_price_ticks(client, prices: dict[str, float], source: str = "check"):
    """Log DexScreener spot prices to price_ticks table. Throttled to 1/min per token."""
    if not prices or not client:
        return
    now = _time_mod.time()
    rows = []
    for addr, price in prices.items():
        if price <= 0:
            continue
        last = _last_tick_log.get(addr, 0)
        if now - last < 60:  # throttle: max 1 insert per token per minute
            continue
        _last_tick_log[addr] = now
        rows.append({"token_address": addr, "price_usd": price, "source": source})
    if not rows:
        return
    try:
        client.table("price_ticks").insert(rows).execute()
    except Exception as e:
        logger.debug("price_ticks insert failed (%d rows): %s", len(rows), e)


def _fetch_prices_batch(addresses: list[str]) -> dict[str, float]:
    """Batch fetch current USD prices. DexScreener primary, Jupiter fallback.
    v107: Jupiter fallback catches pool migrations that DexScreener misses."""
    if not addresses:
        return {}
    prices = {}
    for i in range(0, len(addresses), BATCH_SIZE):
        chunk = addresses[i:i + BATCH_SIZE]
        addr_str = ",".join(chunk)
        try:
            resp = requests.get(
                DEXSCREENER_BATCH_URL.format(addresses=addr_str),
                timeout=15,
            )
            if resp.status_code != 200:
                logger.warning("paper_trader: DexScreener batch %d", resp.status_code)
                continue
            data = resp.json()
            pairs = data if isinstance(data, list) else data.get("pairs", [])
            if not isinstance(pairs, list):
                continue
            # Pick highest-volume pair per base token address
            by_addr: dict[str, list] = {}
            for p in pairs:
                addr = p.get("baseToken", {}).get("address", "")
                if addr:
                    by_addr.setdefault(addr, []).append(p)
            for addr, token_pairs in by_addr.items():
                best = max(token_pairs, key=lambda p: float(p.get("volume", {}).get("h24", 0) or 0))
                price = best.get("priceUsd")
                if price:
                    try:
                        prices[addr] = float(price)
                    except (ValueError, TypeError):
                        pass
        except requests.RequestException as e:
            logger.warning("paper_trader: DexScreener batch error: %s", e)

    # v107: GeckoTerminal fallback for tokens DexScreener missed (pool migration, deindexing)
    missing = [a for a in addresses if a not in prices]
    if missing:
        gecko_recovered = 0
        for addr in missing:
            fallback_price = _fetch_price_fallback(addr)
            if fallback_price and fallback_price > 0:
                prices[addr] = fallback_price
                gecko_recovered += 1
        if gecko_recovered:
            logger.info("paper_trader: GeckoTerminal fallback recovered %d/%d missing prices", gecko_recovered, len(missing))

    return prices


def _load_paper_trade_config(client) -> dict:
    """
    Load paper_trade_config from scoring_config table.
    Returns config dict with keys: top_n, budget_usd, active_strategies,
    dedup_cooldown_hours, ca_filter. Falls back to module defaults on error.
    """
    defaults = {
        "top_n": TOP_N,
        "budget_usd": PORTFOLIO_BUDGET,
        "active_strategies": list(STRATEGIES.keys()),
        "dedup_cooldown_hours": DEDUP_COOLDOWN_HOURS,
        "ca_filter": CA_FILTER,
        "buy_slippage_bps": BUY_SLIPPAGE_BPS,
        "sell_slippage_bps": SELL_SLIPPAGE_BPS,
        "buy_fee_bps": BUY_FEE_BPS,
        "sell_fee_bps": SELL_FEE_BPS,
        "ml_gate_mode": "disabled",  # v90: "disabled" | "normal" | "inverted"
        "experiment_id": None,       # v92: A/B testing
        "variant_id": None,          # v92: A/B testing
        "deprecated_strategies": list(_DEFAULT_DEPRECATED),  # v92: dynamic deprecated
    }
    try:
        result = client.table("scoring_config").select("paper_trade_config").eq("id", 1).execute()
        if result.data and result.data[0].get("paper_trade_config"):
            raw = result.data[0]["paper_trade_config"]
            if isinstance(raw, str):
                import json
                raw = json.loads(raw)
            # Merge with defaults (unknown keys ignored, missing keys use default)
            config = {k: raw.get(k, v) for k, v in defaults.items()}
            # Type safety: JSONB stores numbers as float, but top_n must be int
            config["top_n"] = int(config["top_n"])
            config["budget_usd"] = float(config["budget_usd"])
            config["dedup_cooldown_hours"] = int(config.get("dedup_cooldown_hours", DEDUP_COOLDOWN_HOURS))
            # Validate active_strategies against known strategies
            config["active_strategies"] = [
                s for s in config["active_strategies"] if s in STRATEGIES
            ]
            if not config["active_strategies"]:
                config["active_strategies"] = defaults["active_strategies"]
            # v92: pass through A/B testing keys from raw config
            for extra_key in ("experiment_id", "variant_id"):
                if raw.get(extra_key):
                    config[extra_key] = raw[extra_key]
            logger.info("paper_trader: loaded config from DB: top_n=%d, budget=$%.0f, strategies=%s, dedup=%dh, ca_filter=%s",
                        config["top_n"], config["budget_usd"], config["active_strategies"],
                        config["dedup_cooldown_hours"], config["ca_filter"])
            return config
    except Exception as e:
        logger.warning("paper_trader: failed to load config from scoring_config: %s", e)
    return defaults


def get_deprecated_strategies(client=None, config=None) -> set:
    """v92: Load deprecated_strategies from paper_trade_config. Fallback to hardcoded."""
    if config and config.get("deprecated_strategies"):
        return set(config["deprecated_strategies"])
    if client:
        try:
            r = client.table("scoring_config").select("paper_trade_config").eq("id", 1).execute()
            if r.data and r.data[0].get("paper_trade_config"):
                ptc = r.data[0]["paper_trade_config"]
                if isinstance(ptc, str):
                    import json
                    ptc = json.loads(ptc)
                return set(ptc.get("deprecated_strategies", _DEFAULT_DEPRECATED))
        except Exception:
            pass
    return set(_DEFAULT_DEPRECATED)


def get_open_portfolio(client) -> dict:
    """Return snapshot of open main (non-shadow) positions for alert context.
    Returns {open_count, deployed_usd}. Lightweight: single aggregate query."""
    try:
        result = (
            client.table("paper_trades")
            .select("position_usd")
            .eq("status", "open")
            .eq("is_shadow", False)
            .execute()
        )
        rows = result.data or []
        total = sum(float(r.get("position_usd") or 0) for r in rows)
        return {"open_count": len(rows), "deployed_usd": round(total, 2)}
    except Exception as e:
        logger.warning("get_open_portfolio failed: %s", e)
        return {"open_count": 0, "deployed_usd": 0}


def open_paper_trades(client, ranking: list[dict], cycle_ts: datetime, config: dict | None = None) -> int:
    """
    Open paper trades for top N tokens across configured strategies.
    Each strategy may have multiple tranches (e.g. SCALE_OUT has 4 rows per token).
    Dedup: skip if token_address + strategy already has an open trade.
    Cooldown dedup: skip if same (token, strategy) closed within dedup_cooldown_hours.
    Returns number of new trade rows opened.
    """
    # v88: Load precomputed bot ML predictions (one query per cycle)
    # v109: Skip loading — ML gate is disabled (ml_gate_mode=disabled, ml_threshold=99).
    # Saves one Supabase query per cycle. Re-enable when ML is useful.
    # _load_bot_predictions(client)

    if config is None:
        config = {
            "top_n": TOP_N,
            "budget_usd": PORTFOLIO_BUDGET,
            "active_strategies": list(STRATEGIES.keys()),
            "dedup_cooldown_hours": DEDUP_COOLDOWN_HOURS,
            "ca_filter": CA_FILTER,
        }

    top_n = config["top_n"]
    budget_usd = config["budget_usd"]
    deprecated = set(config.get("deprecated_strategies", _DEFAULT_DEPRECATED))
    active_strategies = [s for s in config["active_strategies"] if s in STRATEGIES and s not in deprecated]
    dedup_cooldown_h = config.get("dedup_cooldown_hours", DEDUP_COOLDOWN_HOURS)
    ca_filter = config.get("ca_filter", True)
    buy_slip_bps_base = int(config.get("buy_slippage_bps", BUY_SLIPPAGE_BPS))
    sell_slip_bps = int(config.get("sell_slippage_bps", SELL_SLIPPAGE_BPS))
    buy_fee_bps = int(config.get("buy_fee_bps", BUY_FEE_BPS))

    # Filter candidates
    base_filter = [
        t for t in ranking
        if t.get("score", 0) > 0
        and t.get("token_address")
        and t.get("price_usd") and float(t["price_usd"]) > 0
    ]
    if ca_filter:
        base_filter = [
            t for t in base_filter
            if (t.get("ca_mention_count", 0) or 0) > 0 or (t.get("url_mention_count", 0) or 0) > 0
        ]
    candidates = base_filter[:top_n]

    if not candidates:
        return 0

    # Score-weighted portfolio allocation
    scores = [max(t.get("score", 1), 1) for t in candidates]
    total_score = sum(scores)
    for i, token in enumerate(candidates):
        token["_alloc_usd"] = round(budget_usd * scores[i] / total_score, 2)

    # Check which (token_address, strategy) combos already have open trades
    addrs = [t["token_address"] for t in candidates]
    try:
        existing = (
            client.table("paper_trades")
            .select("token_address, strategy")
            .eq("status", "open")
            .in_("token_address", addrs)
            .execute()
        )
        open_combos = {
            (r["token_address"], r["strategy"]) for r in (existing.data or [])
        }
    except Exception as e:
        logger.error("paper_trader: failed to check open trades: %s", e)
        open_combos = set()

    # Cooldown dedup: check recently closed trades (main + shadow)
    # v105: Apply cooldown to ALL trades (not just main). Shadow re-entries on the same
    # token pollute data — a KOL re-calling a dead token generates 47 losing shadow trades.
    cooldown_combos = set()
    if dedup_cooldown_h > 0:
        cooldown_since = (cycle_ts - timedelta(hours=dedup_cooldown_h)).isoformat()
        try:
            recent = (
                client.table("paper_trades")
                .select("token_address, strategy")
                .neq("status", "open")
                .gte("exit_at", cooldown_since)
                .in_("token_address", addrs)
                .execute()
            )
            cooldown_combos = {
                (r["token_address"], r["strategy"]) for r in (recent.data or [])
            }
        except Exception as e:
            logger.warning("paper_trader: cooldown dedup query failed: %s", e)

    # Lookup snapshot_id for candidates missing it (batch query)
    need_snap = [t for t in candidates if not t.get("snapshot_id") and t.get("token_address")]
    if need_snap:
        snap_addrs = [t["token_address"] for t in need_snap]
        try:
            snap_res = (
                client.table("token_snapshots")
                .select("id, token_address, snapshot_at")
                .in_("token_address", snap_addrs)
                .gte("snapshot_at", (cycle_ts - timedelta(minutes=10)).isoformat())
                .lte("snapshot_at", (cycle_ts + timedelta(minutes=10)).isoformat())
                .order("snapshot_at", desc=True)
                .execute()
            )
            # Keep closest snapshot per token_address
            snap_map = {}
            for s in (snap_res.data or []):
                addr = s["token_address"]
                if addr not in snap_map:
                    snap_map[addr] = s["id"]
            for t in need_snap:
                sid = snap_map.get(t["token_address"])
                if sid:
                    t["snapshot_id"] = sid
        except Exception as e:
            logger.debug("paper_trader: snapshot_id lookup failed: %s", e)

    opened = 0
    for rank_idx, token in enumerate(candidates, 1):
        addr = token["token_address"]
        raw_price = float(token["price_usd"])
        # v74: Dynamic slippage — scale with liquidity depth score
        # liquidity_depth_score: 1.0 = deep liquidity, 0.1 = shallow
        lds = float(token.get("liquidity_depth_score") or token.get("_rt_liquidity_depth_score") or 0)
        # v94: RT fallback — derive LDS proxy from rt_liquidity_usd when Jupiter LDS unavailable
        if not lds and token.get("_rt_liquidity_usd"):
            rt_liq = float(token["_rt_liquidity_usd"])
            # $50K+ = 1.0, $10K = 0.5, $5K = 0.25, $1K = 0.05
            lds = min(1.0, rt_liq / 50_000) if rt_liq > 0 else 0.1
        lds = max(0.1, min(1.0, lds)) if lds else 1.0
        # Shallow liquidity → up to 3x base slippage; deep → 1x
        slip_mult = 1.0 + 2.0 * (1.0 - lds)  # 1.0 for lds=1.0, 3.0 for lds=0.0
        buy_slip_bps = int(buy_slip_bps_base * slip_mult)
        # v94: entry_price includes slippage + Jupiter priority fee
        entry_price = raw_price * (1 + (buy_slip_bps + buy_fee_bps) / 10_000)
        alloc_usd = token.get("_alloc_usd", budget_usd / top_n)

        # Common fields for all tranches of this token
        base_row = {
            "cycle_ts": cycle_ts.isoformat(),
            "symbol": token.get("symbol", "???"),
            "token_address": addr,
            "rank_in_cycle": rank_idx,
            "entry_price": entry_price,
            "entry_score": int(token.get("score", 0)),
            "entry_mcap": float(token["market_cap"]) if token.get("market_cap") else None,
            "status": "open",
            "unique_kols": token.get("unique_kols"),
            "whale_new_entries": token.get("whale_new_entries"),
            "momentum_mult": float(token["momentum_mult"]) if token.get("momentum_mult") else None,
            "portfolio_budget": budget_usd,
        }
        # v96: Batch KOL attribution — propagate top_kol as kol_group + source="batch"
        if not token.get("_rt_source"):
            top_kols = token.get("top_kols") or []
            if isinstance(top_kols, list) and top_kols:
                base_row["kol_group"] = top_kols[0]
            base_row["source"] = "batch"
        # v92: A/B testing — thread experiment/variant into trade rows
        if config.get("experiment_id"):
            base_row["experiment_id"] = config["experiment_id"]
        if config.get("variant_id"):
            base_row["variant_id"] = config["variant_id"]
        if token.get("snapshot_id"):
            base_row["snapshot_id"] = int(token["snapshot_id"])

        # v66: RT metadata propagation (keys prefixed _rt_ in token dict → DB columns)
        _rt_col_map = {
            "_rt_source": "source",
            "_rt_kol_group": "kol_group",
            "_rt_kol_tier": "kol_tier",
            "_rt_kol_score": "kol_score",
            "_rt_kol_win_rate": "kol_win_rate",
            "_rt_score": "rt_score",
            "_rt_liquidity_usd": "rt_liquidity_usd",
            "_rt_volume_24h": "rt_volume_24h",
            "_rt_buy_sell_ratio": "rt_buy_sell_ratio",
            "_rt_token_age_hours": "rt_token_age_hours",
            "_rt_is_pump_fun": "rt_is_pump_fun",
            "_rt_pair_address": "pair_address",  # v109: track pool for migration debug
            "_rt_ml_pred": "ml_pred",        # v77: ML predicted avg PnL — enables A/B analysis
            "_rt_kol_ml_pred": "kol_ml_pred", # v78: KOL ML predicted return — enables KCO A/B
            "_rt_n_kol_confirmations": "n_kol_confirmations",  # v80: multi-KOL confirmation count
            "_rt_experiment_id": "experiment_id",    # v92: A/B testing
            "_rt_variant_id": "variant_id",          # v92: A/B testing
        }
        for src_key, db_col in _rt_col_map.items():
            val = token.get(src_key)
            if val is not None:
                base_row[db_col] = val

        for strat_name in active_strategies:
            if not _passes_strategy_filter(token, strat_name):
                continue  # token doesn't qualify for this strategy
            tranches = STRATEGIES[strat_name]

            if (addr, strat_name) in open_combos:
                continue
            if (addr, strat_name) in cooldown_combos:
                continue

            # v108: Defensive re-check right before insert to catch race conditions
            # (e.g. batch + RT overlap, or two RT calls that slipped past in-flight lock)
            try:
                recheck = (
                    client.table("paper_trades")
                    .select("id", count="exact")
                    .eq("token_address", addr)
                    .eq("strategy", strat_name)
                    .eq("status", "open")
                    .execute()
                )
                if recheck.count and recheck.count > 0:
                    open_combos.add((addr, strat_name))
                    continue
            except Exception:
                pass  # proceed with insert on query failure

            # v87: Bot ML gate — position sizing or skip (v89: disabled by default)
            bot_ml_mult = _bot_ml_gate(token, strat_name, config)
            if bot_ml_mult <= 0.0:
                logger.info("bot_ml_gate: SKIP %s/%s (win_prob < 0.30)", token.get("symbol"), strat_name)
                continue
            if bot_ml_mult < 1.0:
                logger.info("bot_ml_gate: HALF %s/%s (mult=%.1f)", token.get("symbol"), strat_name, bot_ml_mult)

            for tranche in tranches:
                tp_price = entry_price * tranche["tp_mult"] if tranche["tp_mult"] else None
                sl_price = entry_price * tranche["sl_mult"]

                row = {
                    **base_row,
                    "strategy": strat_name,
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                    "horizon_minutes": tranche["horizon_min"],
                    "tranche_pct": tranche["pct"],
                    "tranche_label": tranche["label"],
                    "position_usd": round(alloc_usd * tranche["pct"] * bot_ml_mult, 2),
                }

                try:
                    client.table("paper_trades").insert(row).execute()
                    opened += 1
                    # v115: DIP_BUY — add to watchlist for P2 monitoring
                    if tranche["label"] == "dip_p1":
                        # v116: Try split regex first, then shared
                        ms = _DIP_SPLIT_RE.match(strat_name)
                        m = _DIP_RE.match(strat_name) if not ms else None
                        if ms:
                            _dip_watchlist[(addr, strat_name)] = {
                                "entry_price": entry_price,
                                "low_seen": entry_price,
                                "dip_pct": int(ms.group(1)) / 100,
                                "bounce_pct": int(ms.group(2)) / 100,
                                "sl_pct": int(ms.group(8)) / 100,  # P2 SL
                                "horizon_min": int(ms.group(9)),
                                "alloc_usd": round(alloc_usd * tranche["pct"] * bot_ml_mult, 2),
                                "base_row": {k: v for k, v in base_row.items()},
                                "created_at": datetime.now(timezone.utc),
                            }
                            logger.info("dip_watch: watching %s/%s for -%.0f%% dip (split P2: T%sA%sS%s)",
                                        token.get("symbol"), strat_name, int(ms.group(1)),
                                        ms.group(6), ms.group(7), ms.group(8))
                        elif m:
                            _dip_watchlist[(addr, strat_name)] = {
                                "entry_price": entry_price,
                                "low_seen": entry_price,
                                "dip_pct": int(m.group(1)) / 100,
                                "bounce_pct": int(m.group(2)) / 100,
                                "sl_pct": int(m.group(5)) / 100,
                                "horizon_min": int(m.group(6)),
                                "alloc_usd": round(alloc_usd * tranche["pct"] * bot_ml_mult, 2),
                                "base_row": {k: v for k, v in base_row.items()},
                                "created_at": datetime.now(timezone.utc),
                            }
                            logger.info("dip_watch: watching %s/%s for -%.0f%% dip",
                                        token.get("symbol"), strat_name, int(m.group(1)))
                except Exception as e:
                    logger.error(
                        "paper_trader: insert failed for %s/%s/%s: %s",
                        token.get("symbol"), strat_name, tranche["label"], e,
                    )
            # v108: Mark as opened so subsequent iterations in this call don't re-insert
            open_combos.add((addr, strat_name))

    # ── Shadow trades: open $0 trades for ALL other single-tranche strategies ──
    # This lets us compare strategies on the SAME tokens (apples-to-apples).
    shadow_enabled = config.get("shadow_enabled", True)
    shadow_opened = 0
    if shadow_enabled:
        for rank_idx, token in enumerate(candidates, 1):
            addr = token["token_address"]
            raw_price = float(token["price_usd"])
            lds = float(token.get("liquidity_depth_score") or token.get("_rt_liquidity_depth_score") or 0)
            if not lds and token.get("_rt_liquidity_usd"):
                rt_liq = float(token["_rt_liquidity_usd"])
                lds = min(1.0, rt_liq / 50_000) if rt_liq > 0 else 0.1
            lds = max(0.1, min(1.0, lds)) if lds else 1.0
            slip_mult = 1.0 + 2.0 * (1.0 - lds)
            buy_slip_bps = int(buy_slip_bps_base * slip_mult)
            entry_price = raw_price * (1 + (buy_slip_bps + buy_fee_bps) / 10_000)

            shadow_base = {
                "cycle_ts": cycle_ts.isoformat(),
                "symbol": token.get("symbol", "???"),
                "token_address": addr,
                "rank_in_cycle": rank_idx,
                "entry_price": entry_price,
                "entry_score": int(token.get("score", 0)),
                "entry_mcap": float(token["market_cap"]) if token.get("market_cap") else None,
                "status": "open",
                "is_shadow": True,
                "position_usd": 0,
                "portfolio_budget": budget_usd,
            }
            # v92: A/B testing — thread experiment/variant into shadow rows
            if config.get("experiment_id"):
                shadow_base["experiment_id"] = config["experiment_id"]
            if config.get("variant_id"):
                shadow_base["variant_id"] = config["variant_id"]
            if token.get("snapshot_id"):
                shadow_base["snapshot_id"] = int(token["snapshot_id"])
            # Propagate RT metadata to shadows too (for KOL attribution)
            for src_key, db_col in _rt_col_map.items():
                val = token.get(src_key)
                if val is not None:
                    shadow_base[db_col] = val

            # all_real_strategies: full list of strategies opened as real (across all calls)
            # In RT hybrid mode, this includes all hybrid allocations (not just this call's active_strategies)
            real_strats = set(config.get("all_real_strategies", active_strategies))
            for strat_name in SHADOW_STRATEGIES:
                if strat_name in real_strats:
                    continue  # opened as real trade (this call or sibling call)
                if not _passes_strategy_filter(token, strat_name):
                    continue
                if (addr, strat_name) in open_combos:
                    continue
                if (addr, strat_name) in cooldown_combos:
                    continue

                tranches = STRATEGIES[strat_name]
                for tranche in tranches:
                    tp_price = entry_price * tranche["tp_mult"] if tranche.get("tp_mult") else None
                    sl_price = entry_price * tranche["sl_mult"]

                    row = {
                        **shadow_base,
                        "strategy": strat_name,
                        "tp_price": tp_price,
                        "sl_price": sl_price,
                        "horizon_minutes": tranche["horizon_min"],
                        "tranche_pct": tranche["pct"],
                        "tranche_label": tranche["label"],
                    }
                    try:
                        client.table("paper_trades").insert(row).execute()
                        shadow_opened += 1
                    except Exception as e:
                        logger.error("paper_trader: shadow insert failed %s/%s/%s: %s",
                                     token.get("symbol"), strat_name, tranche["label"], e)
                # v108: Mark as opened so subsequent tokens in this call don't re-insert
                open_combos.add((addr, strat_name))

    allocs = [f"{t.get('symbol','?')}=${t.get('_alloc_usd',0):.1f}" for t in candidates]
    logger.info(
        "paper_trader: opened %d rows + %d shadow, $%.0f budget → %s (%d strategies, dedup=%dh)",
        opened, shadow_opened, budget_usd, ", ".join(allocs), len(active_strategies), dedup_cooldown_h,
    )
    if _monitoring and opened > 0:
        _metrics.record_paper_trade_open(opened)
    return opened


def _dynamic_sell_slip_factor(trade: dict, exit_type: str, base_bps: int = 200,
                              fee_bps: int = SELL_FEE_BPS) -> float:
    """v119: Dynamic sell slippage calibrated from live Jupiter data.

    Live observations (17 trades):
      trail_UP (sell during uptrend):  ~0-2% slippage — Jupiter fills well
      trail_CRASH (sell during crash): ~20-55% slippage — liquidity vanishes
      sl_hit:                          ~5-10% slippage — moderate dump
      tp_hit / timeout:                ~0-2% slippage — normal conditions

    fee_bps = Jupiter priority fee (flat, added on top of slippage).
    Batch trades (no rt_liquidity_usd) fall back to 50K default = 2% base."""
    liq_usd = float(trade.get("rt_liquidity_usd") or 50_000)

    # Liquidity multiplier: $50K+ = 1x, $5K = 2x, $1K = 4x
    liq_mult = max(1.0, min(4.0, 50_000 / max(liq_usd, 1_000)))

    # v119: Exit type multiplier calibrated from live data
    if exit_type == "trail_crash":
        exit_mult = 3.0   # crash: live shows 20-55% effective slippage
    elif exit_type == "sl_hit":
        exit_mult = 2.0   # SL dump: live shows 5-10% slippage
    elif exit_type == "trail_stop":
        exit_mult = 1.2   # normal pullback: slightly worse than calm sell
    else:
        exit_mult = 1.0   # tp_hit, timeout: normal conditions

    adjusted_bps = int(base_bps * liq_mult * exit_mult) + fee_bps
    # v119: Higher cap for crash (50%) vs normal (15%)
    max_bps = 5000 if exit_type == "trail_crash" else 1500
    adjusted_bps = min(adjusted_bps, max_bps)

    return 1 - adjusted_bps / 10_000


# v106: Parse tp_decay_end from DECAY strategy names
import re as _re
_DECAY_RE = _re.compile(r"^DECAY_TP\d+_SL\d+_E(\d+)$")
_TRAIL_RE = _re.compile(r"^TRAIL(\d+)_TP\d+_SL\d+$")
_DTRAIL_RE = _re.compile(r"^DTRAIL(\d+)_ACT(\d+)_SL\d+$")
_DIP_RE = _re.compile(r"^DIP(\d+)_B(\d+)_T(\d+)_A(\d+)_SL(\d+)_(\d+)m$")  # v115
# v116: Split DIP — P1 and P2 have independent params
# DIP{dip}_B{bounce}_P1T{t}A{a}S{sl}_P2T{t}A{a}S{sl}_{timeout}m
_DIP_SPLIT_RE = _re.compile(
    r"^DIP(\d+)_B(\d+)_P1T(\d+)A(\d+)S(\d+)_P2T(\d+)A(\d+)S(\d+)_(\d+)m$"
)
_BE_RE = _re.compile(r"^BE(\d+)_TP\d+_SL\d+$")

def _get_decay_end(strategy_name: str) -> float | None:
    """Extract tp_decay_end multiplier from strategy name, or None if not a decay strategy."""
    m = _DECAY_RE.match(strategy_name)
    if m:
        return 1 + int(m.group(1)) / 100  # e.g., E20 → 1.20
    return None

def _get_trail_config(trade: dict) -> tuple[float | None, float | None]:
    """Get (trail_pct, activation_pct) from strategy name or tranche config.
    Returns (trail_pct, activation_pct) or (None, None).
    For DTRAIL strategies: activation_pct is the gain threshold before trail activates.
    For TRAIL strategies: activation_pct = trail_pct (legacy behavior).
    """
    strat = trade.get("strategy", "")
    # v116: DIP_BUY split — return P1 or P2 params based on tranche_label
    m = _DIP_SPLIT_RE.match(strat)
    if m:
        label = trade.get("tranche_label", "")
        if label == "dip_p2":
            return int(m.group(6)) / 100, int(m.group(7)) / 100  # P2 trail, P2 act
        return int(m.group(3)) / 100, int(m.group(4)) / 100  # P1 trail, P1 act
    # v115: DIP_BUY shared — DIP{dip}_B{bounce}_T{trail}_A{act}_SL{sl}_{timeout}m
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
    label = trade.get("tranche_label", "")
    if "trail" in label:
        return 0.20, 0.20
    return None, None


# v118: LAZY check mode — strategies with check_mode="lazy" skip exit evaluation
# during throttle windows, but still update high_price_seen.
# LAZY: first 5min check every 3min, after that every 10min.
# This lets trail stops ride longer without triggering on micro-pullbacks.
LAZY_STRATEGIES = {
    # v118: LAZY helps trails < 10% by skipping micro-pullback exits.
    # A/B test: these run LAZY on hybrid (pos>0), shadows keep CURRENT.
    "DTRAIL3_ACT5_SL60",
    "DTRAIL5_ACT10_SL60",
    "DIP30_B5_T5_A20_SL70_240m",  # best overall sim strat — LAZY makes T5 viable
}
_last_eval_ts: dict[str, float] = {}  # trade_id -> last evaluation timestamp
LAZY_FAST_SEC = 180     # 3 min during fast phase
LAZY_FAST_WINDOW = 300  # 5 min fast phase
LAZY_SLOW_SEC = 600     # 10 min after fast phase


def _should_evaluate_exit(trade: dict, now: datetime) -> bool:
    """v118: Check if this trade should be evaluated for exit.
    LAZY strategies are throttled to check less frequently.
    Only applies to hybrid trades (position_usd > 0), NOT shadows.
    Shadows keep CURRENT interval = control group for A/B test."""
    strat = trade.get("strategy", "")
    if strat not in LAZY_STRATEGIES:
        return True
    # Shadows (position_usd=0) always use CURRENT interval (control group)
    if float(trade.get("position_usd") or 0) == 0:
        return True

    trade_id = str(trade.get("id", ""))
    now_ts = now.timestamp()

    # How old is the trade?
    try:
        created = datetime.fromisoformat(trade["created_at"].replace("Z", "+00:00"))
        age_sec = (now - created).total_seconds()
    except Exception:
        return True

    # Determine throttle interval
    if age_sec < LAZY_FAST_WINDOW:
        interval = LAZY_FAST_SEC
    else:
        interval = LAZY_SLOW_SEC

    last = _last_eval_ts.get(trade_id, 0)
    if now_ts - last < interval:
        return False  # skip this evaluation

    _last_eval_ts[trade_id] = now_ts
    return True


def _evaluate_trade_exit(trade: dict, current_price: float | None,
                         now: datetime, sell_slip_factor: float,
                         sl_cascade: bool = False,
                         sell_fee_bps: int = SELL_FEE_BPS) -> dict | None:
    """v94: Shared exit logic for check_paper_trades + check_paper_trades_fast.

    Checks in order: SL → TP → timeout.
    Updates high_price_seen on every call (even when no exit).
    sell_slip_factor is used as base_bps source; dynamic slippage + fee applied per exit type.

    v118: LAZY mode — returns high_price_seen update only (no exit eval)
    when _should_evaluate_exit() says to skip.

    Returns dict with keys {status, exit_price, pnl_pct, pnl_usd, exit_minutes,
    high_price_seen} or None if no action. Caller handles DB update.
    """
    entry_price = float(trade["entry_price"])
    sl_price = float(trade["sl_price"])
    tp_price = float(trade["tp_price"]) if trade.get("tp_price") is not None else None
    pos_usd = float(trade.get("position_usd") or 0)

    # Derive base_bps from the flat sell_slip_factor passed by caller
    base_bps = int(round((1 - sell_slip_factor) * 10_000))

    created_str = trade["created_at"]
    try:
        created_at = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
    except Exception:
        return None

    elapsed_minutes = (now - created_at).total_seconds() / 60
    horizon = trade.get("horizon_minutes", 120)

    # Track high_price_seen (always, even in LAZY skip mode)
    high_seen = float(trade.get("high_price_seen") or 0)
    if current_price is not None and current_price > high_seen:
        high_seen = current_price

    # v118: LAZY mode — update high_price_seen but skip exit evaluation
    if not _should_evaluate_exit(trade, now):
        return {"high_price_seen": high_seen}  # caller updates DB but no exit

    new_status = None
    exit_price = None

    # 1) SL cascade from sibling tranche
    if sl_cascade:
        new_status = "sl_hit"
        exit_price = sl_price * _dynamic_sell_slip_factor(trade, "sl_hit", base_bps, sell_fee_bps)

    elif current_price is not None:
        # 2) SL check — with breakeven stop override
        #    BE strategies: once peak exceeded entry*(1+be_act), SL moves to entry price
        effective_sl = sl_price
        be_match = _BE_RE.match(trade.get("strategy", ""))
        if be_match and entry_price > 0 and high_seen > 0:
            be_act = int(be_match.group(1)) / 100  # e.g., BE20 → 0.20
            if high_seen >= entry_price * (1 + be_act):
                # Breakeven activated — SL is now entry price
                effective_sl = entry_price

        if current_price <= effective_sl:
            new_status = "sl_hit"
            exit_price = effective_sl * _dynamic_sell_slip_factor(trade, "sl_hit", base_bps, sell_fee_bps)
        # 3) TP check (only tranches with TP target)
        elif tp_price is not None and current_price >= tp_price:
            new_status = "tp_hit"
            exit_price = tp_price * _dynamic_sell_slip_factor(trade, "tp_hit", base_bps, sell_fee_bps)
        # 3b) v106: Time-decay TP — threshold decreases in second half of horizon
        #     Derive tp_decay_end from strategy name (DECAY_TPxx_SLyy_Ezz → 1 + zz/100)
        elif tp_price is not None and _get_decay_end(trade.get("strategy", "")) is not None:
            tp_decay_end = _get_decay_end(trade["strategy"])
            tp_mult = tp_price / entry_price if entry_price > 0 else 1
            activation_frac = 0.5  # decay starts at 50% of horizon
            if elapsed_minutes > horizon * activation_frac:
                decay_progress = (elapsed_minutes - horizon * activation_frac) / (horizon * (1 - activation_frac))
                decay_progress = min(decay_progress, 1.0)
                # Linear interpolation: tp_mult → tp_decay_end
                decayed_mult = tp_mult - (tp_mult - tp_decay_end) * decay_progress
                decayed_price = entry_price * decayed_mult
                if current_price >= decayed_price:
                    new_status = "tp_hit"
                    # Exit at the decayed threshold, not the peak
                    exit_price = decayed_price * _dynamic_sell_slip_factor(trade, "tp_hit", base_bps, sell_fee_bps)
    # 3c) v106/v110: Trailing stop — exit when price drops trail_pct% from peak.
    #     v106 TRAIL: activates once peak > entry * (1 + trail_pct).
    #     v110 DTRAIL: activates once peak > entry * (1 + activation_pct).
    if new_status is None and current_price is not None:
        trail_pct, activation_pct = _get_trail_config(trade)
        if trail_pct is not None and high_seen > 0 and entry_price > 0:
            activation_price = entry_price * (1 + activation_pct)
            if high_seen >= activation_price:
                trail_trigger = high_seen * (1 - trail_pct)
                if current_price <= trail_trigger and trail_trigger > entry_price:
                    new_status = "trail_stop"
                    # v119: Detect crash vs normal pullback using price-to-trigger ratio
                    # If current_price is >5% below trail_trigger, it's a crash
                    crash_ratio = current_price / trail_trigger if trail_trigger > 0 else 1.0
                    if crash_ratio < 0.95:
                        # Crash: price already well below trigger — use current_price as base
                        exit_price = current_price * _dynamic_sell_slip_factor(trade, "trail_crash", base_bps, sell_fee_bps)
                    else:
                        # Normal pullback: use current_price with mild penalty
                        exit_price = current_price * _dynamic_sell_slip_factor(trade, "trail_stop", base_bps, sell_fee_bps)
    # 4) Timeout
    if new_status is None and elapsed_minutes >= horizon:
        new_status = "timeout"
        exit_price = (current_price if current_price else entry_price) * _dynamic_sell_slip_factor(trade, "timeout", base_bps, sell_fee_bps)

    # Always return high_price_seen update (even without exit)
    result = {"high_price_seen": high_seen}

    if new_status is None:
        return result  # no exit, but may need to update high_price_seen

    pnl_pct = round((exit_price / entry_price) - 1, 4) if exit_price and entry_price else 0
    # v99: Shadow trades (pos_usd=0) get simulated $10 pnl_usd so stats aren't NULL
    effective_usd = pos_usd if pos_usd else 10.0
    pnl_usd = round(effective_usd * pnl_pct, 2)

    result.update({
        "status": new_status,
        "exit_price": exit_price,
        "exit_at": now.isoformat(),
        "pnl_pct": pnl_pct,
        "pnl_usd": pnl_usd,
        "exit_minutes": int(elapsed_minutes),
    })
    return result


# ---------------------------------------------------------------------------
# v115: DIP_BUY watchlist — monitor for dip+bounce to open P2
# ---------------------------------------------------------------------------

def _rebuild_dip_watchlist(client):
    """Rebuild dip watchlist from open P1 trades on startup."""
    global _dip_watchlist, _dip_watchlist_rebuilt
    if _dip_watchlist_rebuilt:
        return
    _dip_watchlist_rebuilt = True

    try:
        result = (
            client.table("paper_trades")
            .select("*")
            .eq("status", "open")
            .eq("tranche_label", "dip_p1")
            .execute()
        )
        for trade in (result.data or []):
            addr = trade["token_address"]
            strat = trade["strategy"]
            key = (addr, strat)

            # Skip if P2 already opened
            p2 = (
                client.table("paper_trades")
                .select("id", count="exact")
                .eq("token_address", addr)
                .eq("strategy", strat)
                .eq("tranche_label", "dip_p2")
                .neq("status", "open")  # if P2 exists and closed, also skip
                .execute()
            )
            p2_open = (
                client.table("paper_trades")
                .select("id", count="exact")
                .eq("token_address", addr)
                .eq("strategy", strat)
                .eq("tranche_label", "dip_p2")
                .eq("status", "open")
                .execute()
            )
            if (p2.count and p2.count > 0) or (p2_open.count and p2_open.count > 0):
                continue

            # v116: Try split regex first, then shared
            ms = _DIP_SPLIT_RE.match(strat)
            m = _DIP_RE.match(strat) if not ms else None
            if not ms and not m:
                continue

            created = datetime.fromisoformat(trade["created_at"].replace("Z", "+00:00"))
            if ms:
                dip_pct = int(ms.group(1)) / 100
                bounce_pct = int(ms.group(2)) / 100
                sl_pct = int(ms.group(8)) / 100  # P2 SL
                horizon_min = int(ms.group(9))
            else:
                dip_pct = int(m.group(1)) / 100
                bounce_pct = int(m.group(2)) / 100
                sl_pct = int(m.group(5)) / 100
                horizon_min = int(m.group(6))

            _dip_watchlist[key] = {
                "entry_price": float(trade["entry_price"]),
                "low_seen": float(trade["entry_price"]),  # conservative reset
                "dip_pct": dip_pct,
                "bounce_pct": bounce_pct,
                "sl_pct": sl_pct,
                "horizon_min": horizon_min,
                "alloc_usd": float(trade["position_usd"]),  # P1 pos = P2 pos (50/50)
                "base_row": {
                    "cycle_ts": trade.get("cycle_ts"),
                    "symbol": trade.get("symbol"),
                    "token_address": addr,
                    "pair_address": trade.get("pair_address"),
                    "rank_in_cycle": trade.get("rank_in_cycle"),
                    "entry_score": trade.get("entry_score"),
                    "entry_mcap": trade.get("entry_mcap"),
                    "source": trade.get("source"),
                    "kol_group": trade.get("kol_group"),
                    "is_shadow": trade.get("is_shadow", False),
                },
                "created_at": created,
            }

        if _dip_watchlist:
            logger.info("dip_watch: rebuilt watchlist with %d entries", len(_dip_watchlist))
    except Exception as e:
        logger.warning("dip_watch: rebuild failed: %s", e)


def _process_dip_watchlist(client, prices: dict, now: datetime):
    """Check dip+bounce conditions for DIP_BUY P2 entries."""
    to_remove = []

    for key, watch in _dip_watchlist.items():
        addr, strat_name = key
        current_price = prices.get(addr)
        if current_price is None:
            continue

        # Update low_seen
        if current_price < watch["low_seen"]:
            watch["low_seen"] = current_price

        # Check timeout — watchlist shouldn't outlive the strategy horizon
        elapsed = (now - watch["created_at"]).total_seconds() / 60
        if elapsed >= watch["horizon_min"]:
            to_remove.append(key)
            logger.debug("dip_watch: timeout for %s/%s after %dmin", addr[:8], strat_name, int(elapsed))
            continue

        # Check if P1 is still open
        try:
            p1_check = (
                client.table("paper_trades")
                .select("id", count="exact")
                .eq("token_address", addr)
                .eq("strategy", strat_name)
                .eq("tranche_label", "dip_p1")
                .eq("status", "open")
                .execute()
            )
            if not p1_check.count or p1_check.count == 0:
                to_remove.append(key)
                logger.info("dip_watch: P1 closed for %s/%s, cancelling P2 watch", addr[:8], strat_name)
                continue
        except Exception:
            continue

        # Check dip condition: price dropped dip_pct from P1 entry
        dip_level = watch["entry_price"] * (1 - watch["dip_pct"])
        if watch["low_seen"] <= dip_level:
            # Dip triggered — check bounce from low
            if watch["low_seen"] > 0:
                bounce_from_low = current_price / watch["low_seen"] - 1
            else:
                bounce_from_low = 0
            if bounce_from_low >= watch["bounce_pct"]:
                # BOUNCE CONFIRMED — open P2
                _open_dip_p2(client, key, watch, current_price, now)
                to_remove.append(key)

    for key in to_remove:
        _dip_watchlist.pop(key, None)


def _open_dip_p2(client, key: tuple, watch: dict, current_price: float, now: datetime):
    """Open DIP_BUY P2 trade on confirmed bounce."""
    addr, strat_name = key

    # P2 entry = low * (1 + bounce) * (1 + slippage + fee)
    buy_slip = BUY_SLIPPAGE_BPS / 10_000 + BUY_FEE_BPS / 10_000
    p2_entry = watch["low_seen"] * (1 + watch["bounce_pct"]) * (1 + buy_slip)
    p2_sl = p2_entry * (1 - watch["sl_pct"])

    # P2 timeout = remaining time from P1 entry
    elapsed_since_p1 = (now - watch["created_at"]).total_seconds() / 60
    remaining_horizon = max(int(watch["horizon_min"] - elapsed_since_p1), 10)  # min 10min

    row = {
        **watch["base_row"],
        "strategy": strat_name,
        "entry_price": p2_entry,
        "sl_price": p2_sl,
        "tp_price": None,
        "horizon_minutes": remaining_horizon,
        "tranche_pct": 0.5,
        "tranche_label": "dip_p2",
        "position_usd": round(watch["alloc_usd"], 2),
        "status": "open",
        "high_price_seen": current_price,
    }

    try:
        client.table("paper_trades").insert(row).execute()
        logger.info(
            "dip_watch: OPENED P2 %s/%s — entry=$%.8f (low=$%.8f, bounce=+%.0f%%), alloc=$%.0f, horizon=%dmin",
            watch["base_row"].get("symbol", "?"), strat_name,
            p2_entry, watch["low_seen"], watch["bounce_pct"] * 100,
            watch["alloc_usd"], remaining_horizon,
        )
        # Alert for P2 open
        if not watch["base_row"].get("is_shadow"):
            try:
                from alerter import _send as _alert_send
                dip_pct = (1 - watch["low_seen"] / watch["entry_price"]) * 100
                msg = (
                    f"🔄 DIP BUY P2\n\n"
                    f"${watch['base_row'].get('symbol', '?')} dipped <b>-{dip_pct:.0f}%</b> → bounced +{watch['bounce_pct']*100:.0f}%\n"
                    f"💰 P2 entry: ${p2_entry:.8f} | ${watch['alloc_usd']:.0f}\n"
                    f"⏱ Remaining: {remaining_horizon}min"
                )
                _alert_send(msg, "dip_buy_p2")
            except Exception:
                pass
    except Exception as e:
        logger.error("dip_watch: P2 insert failed for %s/%s: %s", addr[:8], strat_name, e)


def _precache_ohlcv_for_closed(closed_ids: set, trades: list[dict]):
    """v114: Pre-cache OHLCV for recently closed trades so sim.py has data ready.

    Uses sim.py's cache format (ohlcv_cache/{pool}_{ts}_{window}_{hash}.json).
    Only fetches for trades with pair_address that don't already have cached data.
    Non-blocking: errors are silently logged and don't affect trade processing.
    """
    import hashlib
    import json
    import os
    from pathlib import Path

    cache_dir = Path(__file__).resolve().parent / "ohlcv_cache"
    cache_dir.mkdir(exist_ok=True)
    window = 365  # match sim.py MAX_WINDOW_MIN

    closed_trades = [t for t in trades if t["id"] in closed_ids and t.get("pair_address")]
    # Dedup by (pair_address, created_at) — multiple strategies for same token
    seen = set()
    to_fetch = []
    for t in closed_trades:
        pool = t["pair_address"]
        created = t["created_at"]
        key = (pool, created)
        if key in seen:
            continue
        seen.add(key)

        dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
        start_ts = int(dt.timestamp())
        h = hashlib.md5(f"{pool}_{start_ts}_{window}".encode()).hexdigest()[:12]
        cache_file = cache_dir / f"{pool[:12]}_{start_ts}_{window}_{h}.json"
        if cache_file.exists():
            continue  # already cached
        to_fetch.append({"pool": pool, "token": t["token_address"], "start_ts": start_ts,
                         "end_ts": start_ts + window * 60, "cache_file": cache_file})

    if not to_fetch:
        return

    logger.info("ohlcv_precache: fetching %d OHLCV windows for closed trades", len(to_fetch))
    fetched = 0
    for entry in to_fetch[:10]:  # limit to 10 per cycle to avoid rate limits
        try:
            # Try DexPaprika first (free, uses pool address)
            start_iso = datetime.fromtimestamp(entry["start_ts"], tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            end_iso = datetime.fromtimestamp(entry["end_ts"], tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            r = requests.get(
                f"https://api.dexpaprika.com/networks/solana/pools/{entry['pool']}/ohlcv",
                params={"start": start_iso, "end": end_iso, "interval": "15m"},
                timeout=15,
            )
            candles = []
            if r.status_code == 200:
                for c in r.json():
                    ts_str = c.get("time_open") or c.get("time_close") or c.get("timestamp")
                    if not ts_str:
                        continue
                    ts = int(datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()) \
                        if isinstance(ts_str, str) else int(ts_str)
                    candles.append({"timestamp": ts, "open": float(c.get("open", 0)),
                                    "high": float(c.get("high", 0)), "low": float(c.get("low", 0)),
                                    "close": float(c.get("close", 0)),
                                    "volume": float(c.get("volume", 0))})
                candles.sort(key=lambda x: x["timestamp"])

            # v117: Fallback to GeckoTerminal if DexPaprika returns no data
            if len(candles) < 3:
                try:
                    pool_addr = entry["pool"]
                    gt_r = requests.get(
                        f"https://api.geckoterminal.com/api/v2/networks/solana/pools/{pool_addr}/ohlcv/minute",
                        params={"aggregate": "15", "limit": 100,
                                "before_timestamp": entry["end_ts"]},
                        headers={"Accept": "application/json"},
                        timeout=15,
                    )
                    if gt_r.status_code == 200:
                        gt_data = gt_r.json().get("data", {}).get("attributes", {}).get("ohlcv_list", [])
                        candles = []
                        for row in gt_data:
                            if len(row) >= 6:
                                candles.append({"timestamp": int(row[0]), "open": float(row[1]),
                                                "high": float(row[2]), "low": float(row[3]),
                                                "close": float(row[4]), "volume": float(row[5])})
                        candles.sort(key=lambda x: x["timestamp"])
                        # Filter to our window
                        candles = [c for c in candles if entry["start_ts"] <= c["timestamp"] <= entry["end_ts"]]
                except Exception:
                    pass
                _time_mod.sleep(0.5)  # rate limit GeckoTerminal

            if len(candles) >= 3:
                entry["cache_file"].write_text(json.dumps(candles))
                fetched += 1
            # v117: Don't cache empty results — retry next cycle
        except Exception as e:
            logger.debug("ohlcv_precache: error for %s: %s", entry["pool"][:12], e)

    if fetched > 0:
        logger.info("ohlcv_precache: cached %d/%d OHLCV windows", fetched, len(to_fetch[:10]))


def check_paper_trades(client) -> dict:
    """
    Check all open paper trades against current prices.
    Closes trades that hit TP, SL, trailing stop, or timeout.
    v73: Exit prices include sell slippage simulation.
    v92: Uses _evaluate_trade_exit() helper with trailing stop support.

    SL cascade: when SL triggers, ALL open tranches for the same
    (token_address, strategy, cycle_ts) close at -SL%.

    Moonbag tranches (tp_price=NULL) only close on SL or timeout.

    Returns {"checked": N, "closed": M, "tp": X, "sl": Y, "timeout": Z,
            "pnl_usd": total, "rt_pnl_usd": RT-only}.
    """
    now = datetime.now(timezone.utc)

    try:
        result = client.table("paper_trades").select("*").eq("status", "open").neq("source", "rt_live").execute()
        open_trades = result.data or []
        if _monitoring:
            _estimate_egress("paper_trader", "paper_trades", len(open_trades))
    except Exception as e:
        logger.error("paper_trader: failed to fetch open trades: %s", e)
        return {"checked": 0, "closed": 0, "tp": 0, "sl": 0, "timeout": 0, "pnl_usd": 0, "rt_pnl_usd": 0, "rt_closed": 0}

    if not open_trades:
        return {"checked": 0, "closed": 0, "tp": 0, "sl": 0, "timeout": 0, "pnl_usd": 0, "rt_pnl_usd": 0, "rt_closed": 0}

    # Batch fetch current prices
    addresses = list({t["token_address"] for t in open_trades})
    # v115: Include watchlist tokens in price fetch
    for (waddr, _) in _dip_watchlist:
        if waddr not in addresses:
            addresses.append(waddr)
    prices = _fetch_prices_batch(addresses)
    _log_price_ticks(client, prices, "full")

    # v115: Process dip watchlist on full check too
    if _dip_watchlist:
        _process_dip_watchlist(client, prices, now)

    # v73: Load sell slippage + fee config for exit price simulation
    _sell_slip_bps = SELL_SLIPPAGE_BPS
    _sell_fee_bps = SELL_FEE_BPS
    _active_strats = []
    try:
        _cfg = _load_paper_trade_config(client)
        _sell_slip_bps = int(_cfg.get("sell_slippage_bps", SELL_SLIPPAGE_BPS))
        _sell_fee_bps = int(_cfg.get("sell_fee_bps", SELL_FEE_BPS))
        _active_strats = _cfg.get("active_strategies", [])
    except Exception:
        pass
    _sell_slip_factor = 1 - _sell_slip_bps / 10_000

    counts = {"checked": len(open_trades), "closed": 0, "tp": 0, "sl": 0, "timeout": 0}
    _total_pnl_usd = 0.0
    _rt_pnl_usd = 0.0
    _rt_closed = 0

    # Track SL-triggered groups so we can cascade
    sl_triggered = set()

    # Sort so main/tp tranches come before moonbag (SL detection first)
    sorted_trades = sorted(open_trades, key=lambda t: (t.get("tranche_label", "") == "moonbag"))
    closed_ids = set()

    for trade in sorted_trades:
        if trade["id"] in closed_ids:
            continue

        addr = trade["token_address"]
        current_price = prices.get(addr)
        # v115: DIP_BUY P1/P2 exit independently (no SL cascade between them)
        label = trade.get("tranche_label", "")
        if "dip_p" in label:
            group_key = (addr, trade["strategy"], trade["cycle_ts"], label)
        else:
            group_key = (addr, trade["strategy"], trade["cycle_ts"])

        is_cascade = group_key in sl_triggered
        ev = _evaluate_trade_exit(trade, current_price, now, _sell_slip_factor, sl_cascade=is_cascade, sell_fee_bps=_sell_fee_bps)
        if ev is None:
            continue

        # Always update high_price_seen (even without exit)
        if ev.get("high_price_seen") is not None and ev["high_price_seen"] > float(trade.get("high_price_seen") or 0):
            try:
                client.table("paper_trades").update(
                    {"high_price_seen": ev["high_price_seen"]}
                ).eq("id", trade["id"]).execute()
            except Exception:
                pass

        if "status" not in ev:
            continue  # no exit

        new_status = ev["status"]
        pnl_usd = ev.get("pnl_usd")

        # Track SL cascade
        if new_status == "sl_hit" and not is_cascade:
            sl_triggered.add(group_key)

        update = {k: ev[k] for k in ("status", "exit_price", "exit_at", "pnl_pct", "pnl_usd", "exit_minutes") if k in ev}
        if ev.get("high_price_seen") is not None:
            update["high_price_seen"] = ev["high_price_seen"]

        try:
            # v114: Conditional update — only close if still open (prevents race with fast check)
            res = client.table("paper_trades").update(update).eq("id", trade["id"]).eq("status", "open").execute()
            if not res.data:
                logger.debug("paper_trader: trade %s already closed by another loop, skipping", trade["id"])
                continue
            closed_ids.add(trade["id"])
            counts["closed"] += 1
            _total_pnl_usd += pnl_usd or 0
            if trade.get("source") == "rt" and not trade.get("is_shadow"):
                _rt_pnl_usd += pnl_usd or 0
                _rt_closed += 1
                # v113: Update bankroll BEFORE alert so balance is fresh
                try:
                    from safe_scraper import _rt_update_bankroll, _rt_load_bankroll
                    _rt_update_bankroll(pnl_usd or 0, 1, strategy=trade.get("strategy", ""))
                    _br = _rt_load_bankroll()
                    bal = float(_br.get("current_balance", 0))
                    _strat_bals = _br.get("strategy_bankrolls") or {}
                except Exception:
                    bal = 0
                    _strat_bals = {}
                try:
                    from alerter import alert_trade_closed
                    portfolio = get_open_portfolio(client)
                    alert_trade_closed(
                        symbol=trade["symbol"], strategy=trade["strategy"],
                        exit_reason=new_status,
                        pnl_pct=ev.get("pnl_pct", 0), pnl_usd=pnl_usd or 0,
                        pos_usd=float(trade.get("position_usd") or 0),
                        entry_price=float(trade.get("entry_price") or 0),
                        exit_price=ev.get("exit_price", 0),
                        high_price=ev.get("high_price_seen", 0),
                        minutes=int(ev.get("exit_minutes", 0)),
                        kol=trade.get("kol_group", ""),
                        bankroll=bal,
                        ca=trade.get("token_address", ""),
                        deployed_usd=portfolio["deployed_usd"],
                        open_count=portfolio["open_count"],
                        strategy_bankrolls=_strat_bals,
                        active_strategies=_active_strats,
                    )
                except Exception as e:
                    logger.warning("trade close alert failed: %s", e)
            status_key = new_status.replace("_hit", "").replace("_stop", "")
            counts[status_key] = counts.get(status_key, 0) + 1
            usd_str = f" ${pnl_usd:+.2f}" if pnl_usd is not None else ""
            logger.info(
                "paper_trader: CLOSED %s %s/%s/%s — %s pnl=%.1f%%%s after %dmin",
                trade["symbol"], trade["strategy"], trade.get("tranche_label", "main"),
                addr[:8], new_status, ev.get("pnl_pct", 0) * 100, usd_str, ev.get("exit_minutes", 0),
            )
        except Exception as e:
            logger.error("paper_trader: update failed for trade %s: %s", trade["id"], e)

    # Second pass: close remaining open trades in SL-triggered groups
    for trade in open_trades:
        if trade["id"] in closed_ids:
            continue
        # v115: DIP_BUY P1/P2 exit independently
        label = trade.get("tranche_label", "")
        if "dip_p" in label:
            group_key = (trade["token_address"], trade["strategy"], trade["cycle_ts"], label)
        else:
            group_key = (trade["token_address"], trade["strategy"], trade["cycle_ts"])
        if group_key not in sl_triggered:
            continue

        addr = trade["token_address"]
        current_price = prices.get(addr)

        ev = _evaluate_trade_exit(trade, current_price, now, _sell_slip_factor, sl_cascade=True, sell_fee_bps=_sell_fee_bps)
        if ev is None or "status" not in ev:
            continue

        pnl_usd = ev.get("pnl_usd")
        update = {k: ev[k] for k in ("status", "exit_price", "exit_at", "pnl_pct", "pnl_usd", "exit_minutes") if k in ev}
        if ev.get("high_price_seen") is not None:
            update["high_price_seen"] = ev["high_price_seen"]

        try:
            # v114: Conditional update — only close if still open (prevents race)
            res = client.table("paper_trades").update(update).eq("id", trade["id"]).eq("status", "open").execute()
            if not res.data:
                logger.debug("paper_trader: cascade trade %s already closed, skipping", trade["id"])
                continue
            closed_ids.add(trade["id"])
            counts["closed"] += 1
            _total_pnl_usd += pnl_usd or 0
            if trade.get("source") == "rt" and not trade.get("is_shadow"):
                _rt_pnl_usd += pnl_usd or 0
                _rt_closed += 1
                # v113: Update bankroll BEFORE alert so balance is fresh
                try:
                    from safe_scraper import _rt_update_bankroll, _rt_load_bankroll
                    _rt_update_bankroll(pnl_usd or 0, 1, strategy=trade.get("strategy", ""))
                    _br = _rt_load_bankroll()
                    bal = float(_br.get("current_balance", 0))
                    _strat_bals = _br.get("strategy_bankrolls") or {}
                except Exception:
                    bal = 0
                    _strat_bals = {}
                try:
                    from alerter import alert_trade_closed
                    portfolio = get_open_portfolio(client)
                    alert_trade_closed(
                        symbol=trade["symbol"], strategy=trade["strategy"],
                        exit_reason=ev.get("status", "sl_hit"),
                        pnl_pct=ev.get("pnl_pct", 0), pnl_usd=pnl_usd or 0,
                        pos_usd=float(trade.get("position_usd") or 0),
                        entry_price=float(trade.get("entry_price") or 0),
                        exit_price=ev.get("exit_price", 0),
                        high_price=ev.get("high_price_seen", 0),
                        minutes=int(ev.get("exit_minutes", 0)),
                        kol=trade.get("kol_group", ""),
                        bankroll=bal,
                        ca=trade.get("token_address", ""),
                        deployed_usd=portfolio["deployed_usd"],
                        open_count=portfolio["open_count"],
                        strategy_bankrolls=_strat_bals,
                        active_strategies=_active_strats,
                    )
                except Exception as e:
                    logger.warning("SL cascade trade close alert failed: %s", e)
            counts["sl"] = counts.get("sl", 0) + 1
            usd_str = f" ${pnl_usd:+.2f}" if pnl_usd is not None else ""
            logger.info(
                "paper_trader: CLOSED (SL cascade) %s %s/%s — pnl=%.1f%%%s",
                trade["symbol"], trade["strategy"], trade.get("tranche_label", ""),
                ev.get("pnl_pct", 0) * 100, usd_str,
            )
        except Exception as e:
            logger.error("paper_trader: update failed for trade %s: %s", trade["id"], e)

    if counts["closed"] > 0:
        logger.info(
            "paper_trader: checked %d open, closed %d (TP=%d SL=%d timeout=%d trail=%d)",
            counts["checked"], counts["closed"], counts["tp"], counts["sl"],
            counts["timeout"], counts.get("trail", 0),
        )
        if _monitoring:
            _metrics.record_paper_trade_close(counts["closed"], _total_pnl_usd)

        # v114: Pre-cache OHLCV for closed trades (so sim.py has data ready)
        _precache_ohlcv_for_closed(closed_ids, sorted_trades)

    counts["pnl_usd"] = round(_total_pnl_usd, 2)
    counts["rt_pnl_usd"] = round(_rt_pnl_usd, 2)
    counts["rt_closed"] = _rt_closed
    return counts


def correct_closed_prices(client) -> int:
    """v107: Post-close price correction for recently closed trades.

    Problem: DexScreener may not index new pools during pump.fun→Raydium migration.
    Trades close (timeout) with high_price_seen=entry while the real price is x10+.
    This corrupts ML training data and PnL reporting.

    Fix: for trades closed in the last 6h, re-fetch current price via DexScreener+GeckoTerminal.
    If current price > high_price_seen, update high_price_seen (but NOT status/pnl — the trade
    was already closed, we just fix the tracking data for ML accuracy).

    Runs once per full cycle (~15min), not on every price refresh.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat()

    try:
        result = (
            client.table("paper_trades")
            .select("id, token_address, symbol, entry_price, high_price_seen, status")
            .neq("status", "open")
            .gte("exit_at", cutoff)
            .execute()
        )
        closed_trades = result.data or []
    except Exception as e:
        logger.warning("correct_closed_prices: fetch failed: %s", e)
        return 0

    if not closed_trades:
        return 0

    # Dedup addresses
    addresses = list({t["token_address"] for t in closed_trades})
    prices = _fetch_prices_batch(addresses)

    corrected = 0
    for trade in closed_trades:
        addr = trade["token_address"]
        current = prices.get(addr)
        if current is None or current <= 0:
            continue

        entry = float(trade.get("entry_price") or 0)
        old_high = float(trade.get("high_price_seen") or 0)

        if current > old_high and entry > 0:
            try:
                client.table("paper_trades").update(
                    {"high_price_seen": current}
                ).eq("id", trade["id"]).execute()
                corrected += 1
                if current / entry > 1.5:  # only log significant corrections
                    logger.info(
                        "correct_closed_prices: %s %s high %.1fx→%.1fx",
                        trade["symbol"], addr[:8],
                        old_high / entry if old_high > 0 else 0,
                        current / entry,
                    )
            except Exception:
                pass

    if corrected:
        logger.info("correct_closed_prices: updated %d/%d trades", corrected, len(closed_trades))
    return corrected


def check_paper_trades_fast(client) -> dict:
    """v92: Fast 30s check for recent RT trades only (opened in last 30 min).
    Catches fast spikes that the 3-min full check would miss.
    Only checks SL/TP — no timeout (too young). No SL cascade (single-tranche RT).
    """
    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(minutes=30)).isoformat()

    try:
        result = (
            client.table("paper_trades")
            .select("*")
            .eq("status", "open")
            .gte("created_at", cutoff)
            .execute()
        )
        recent_trades = result.data or []
    except Exception as e:
        logger.warning("paper_fast: fetch failed: %s", e)
        return {"checked": 0, "closed": 0}

    # v115: Rebuild dip watchlist on first run after restart
    _rebuild_dip_watchlist(client)

    if not recent_trades and not _dip_watchlist:
        return {"checked": 0, "closed": 0}

    addresses = list({t["token_address"] for t in recent_trades})
    # v115: Include watchlist tokens in price fetch
    for (waddr, _) in _dip_watchlist:
        if waddr not in addresses:
            addresses.append(waddr)
    prices = _fetch_prices_batch(addresses) if addresses else {}
    _log_price_ticks(client, prices, "fast")

    # v115: Process dip watchlist every 30s
    if _dip_watchlist:
        _process_dip_watchlist(client, prices, now)

    if not recent_trades:
        return {"checked": 0, "closed": 0}

    _sell_slip_bps = SELL_SLIPPAGE_BPS
    _sell_fee_bps = SELL_FEE_BPS
    try:
        _cfg = _load_paper_trade_config(client)
        _sell_slip_bps = int(_cfg.get("sell_slippage_bps", SELL_SLIPPAGE_BPS))
        _sell_fee_bps = int(_cfg.get("sell_fee_bps", SELL_FEE_BPS))
    except Exception:
        pass
    _sell_slip_factor = 1 - _sell_slip_bps / 10_000

    closed = 0
    for trade in recent_trades:
        addr = trade["token_address"]
        current_price = prices.get(addr)

        ev = _evaluate_trade_exit(trade, current_price, now, _sell_slip_factor, sell_fee_bps=_sell_fee_bps)
        if ev is None:
            continue

        # Update high_price_seen even without exit
        if ev.get("high_price_seen") is not None and ev["high_price_seen"] > float(trade.get("high_price_seen") or 0):
            try:
                client.table("paper_trades").update(
                    {"high_price_seen": ev["high_price_seen"]}
                ).eq("id", trade["id"]).execute()
            except Exception:
                pass

        if "status" not in ev:
            continue

        update = {k: ev[k] for k in ("status", "exit_price", "exit_at", "pnl_pct", "pnl_usd", "exit_minutes") if k in ev}
        if ev.get("high_price_seen") is not None:
            update["high_price_seen"] = ev["high_price_seen"]

        try:
            # v114: Conditional update — only close if still open (prevents race with full check)
            res = client.table("paper_trades").update(update).eq("id", trade["id"]).eq("status", "open").execute()
            if not res.data:
                logger.debug("paper_fast: trade %s already closed by another loop, skipping", trade["id"])
                continue
            closed += 1
            pnl_usd = ev.get("pnl_usd")
            usd_str = f" ${pnl_usd:+.2f}" if pnl_usd is not None else ""
            logger.info(
                "paper_fast: CLOSED %s %s/%s — %s pnl=%.1f%%%s after %dmin",
                trade["symbol"], trade["strategy"], addr[:8],
                ev["status"], ev.get("pnl_pct", 0) * 100, usd_str, ev.get("exit_minutes", 0),
            )
            # v113: Update bankroll + alert on main RT trade close
            if trade.get("source") == "rt" and not trade.get("is_shadow"):
                try:
                    from safe_scraper import _rt_update_bankroll, _rt_load_bankroll
                    _rt_update_bankroll(pnl_usd or 0, 1, strategy=trade.get("strategy", ""))
                    _br = _rt_load_bankroll()
                    bal = float(_br.get("current_balance", 0))
                    _strat_bals = _br.get("strategy_bankrolls") or {}
                except Exception:
                    bal = 0
                    _strat_bals = {}
                try:
                    from alerter import alert_trade_closed
                    portfolio = get_open_portfolio(client)
                    alert_trade_closed(
                        symbol=trade["symbol"], strategy=trade["strategy"],
                        exit_reason=ev["status"],
                        pnl_pct=ev.get("pnl_pct", 0), pnl_usd=pnl_usd or 0,
                        pos_usd=float(trade.get("position_usd") or 0),
                        entry_price=float(trade.get("entry_price") or 0),
                        exit_price=ev.get("exit_price", 0),
                        high_price=ev.get("high_price_seen", 0),
                        minutes=int(ev.get("exit_minutes", 0)),
                        kol=trade.get("kol_group", ""),
                        bankroll=bal,
                        ca=trade.get("token_address", ""),
                        deployed_usd=portfolio["deployed_usd"],
                        open_count=portfolio["open_count"],
                        strategy_bankrolls=_strat_bals,
                        active_strategies=_active_strats,
                    )
                except Exception as e:
                    logger.warning("paper_fast trade close alert failed: %s", e)
        except Exception as e:
            logger.error("paper_fast: update failed for trade %s: %s", trade["id"], e)

    if closed > 0:
        logger.info("paper_fast: checked %d recent, closed %d", len(recent_trades), closed)
    return {"checked": len(recent_trades), "closed": closed}


def paper_trade_summary(client) -> dict | None:
    """
    Compute summary stats for closed paper trades (last 7 days).
    Per-strategy breakdown with weighted PnL for multi-tranche strategies.
    Returns summary dict or None if no trades.
    """
    try:
        result = (
            client.table("paper_trades")
            .select("*")
            .neq("status", "open")
            .eq("is_shadow", False)
            .gte("created_at", _days_ago_iso(7))
            .execute()
        )
        trades = result.data or []
    except Exception as e:
        logger.error("paper_trader: summary query failed: %s", e)
        return None

    if not trades:
        return None

    # Global stats
    total = len(trades)
    tp_count = sum(1 for t in trades if t["status"] == "tp_hit")
    sl_count = sum(1 for t in trades if t["status"] == "sl_hit")
    timeout_count = sum(1 for t in trades if t["status"] == "timeout")
    trail_count = sum(1 for t in trades if t["status"] == "trail_stop")

    pnls = [float(t["pnl_pct"]) for t in trades if t.get("pnl_pct") is not None]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]

    win_rate = len(winners) / total if total else 0
    avg_pnl = sum(pnls) / len(pnls) if pnls else 0

    # Dollar PnL
    total_invested = sum(float(t.get("position_usd") or 0) for t in trades)
    total_pnl_usd = sum(float(t.get("pnl_usd") or 0) for t in trades)
    roi_pct = round(total_pnl_usd / total_invested * 100, 2) if total_invested else 0

    # Per-strategy breakdown
    strategy_stats = {}
    for strat_name in STRATEGIES:
        strat_trades = [t for t in trades if t.get("strategy") == strat_name]
        if not strat_trades:
            continue

        # Group by (token_address, cycle_ts) for weighted PnL across tranches
        positions = {}
        for t in strat_trades:
            key = (t["token_address"], t["cycle_ts"])
            positions.setdefault(key, []).append(t)

        pos_pnls = []
        pos_wins = 0
        for key, pos_trades in positions.items():
            weighted_pnl = sum(
                float(t.get("pnl_pct", 0) or 0) * float(t.get("tranche_pct", 1.0))
                for t in pos_trades
            )
            pos_pnls.append(weighted_pnl)
            if weighted_pnl > 0:
                pos_wins += 1

        n_positions = len(pos_pnls)
        s_tp = sum(1 for t in strat_trades if t["status"] == "tp_hit")
        s_sl = sum(1 for t in strat_trades if t["status"] == "sl_hit")
        s_to = sum(1 for t in strat_trades if t["status"] == "timeout")
        s_avg_pnl = sum(pos_pnls) / n_positions if n_positions else 0

        s_winners = [p for p in pos_pnls if p > 0]
        s_losers = [p for p in pos_pnls if p < 0]
        s_pf = abs(sum(s_winners) / sum(s_losers)) if s_losers and sum(s_losers) != 0 else float("inf")

        s_invested = sum(float(t.get("position_usd") or 0) for t in strat_trades)
        s_pnl_usd = sum(float(t.get("pnl_usd") or 0) for t in strat_trades)

        # Enriched stats: expectancy, breakeven WR, max consecutive losses
        s_expectancy = sum(pos_pnls) / n_positions if n_positions else 0
        s_avg_win = sum(s_winners) / len(s_winners) if s_winners else 0
        s_avg_loss = abs(sum(s_losers)) / len(s_losers) if s_losers else 0
        s_breakeven_wr = s_avg_loss / (s_avg_win + s_avg_loss) if (s_avg_win + s_avg_loss) > 0 else 0.5
        s_max_consec = 0
        s_consec = 0
        for p in pos_pnls:
            if p < 0:
                s_consec += 1
                s_max_consec = max(s_max_consec, s_consec)
            else:
                s_consec = 0

        strategy_stats[strat_name] = {
            "positions": n_positions,
            "trade_rows": len(strat_trades),
            "tp": s_tp,
            "sl": s_sl,
            "timeout": s_to,
            "win_rate": round(pos_wins / n_positions, 3) if n_positions else 0,
            "avg_pnl": round(s_avg_pnl, 4),
            "expectancy": round(s_expectancy, 4),
            "profit_factor": round(s_pf, 2) if s_pf != float("inf") else "inf",
            "breakeven_wr": round(s_breakeven_wr, 4),
            "max_consecutive_losses": s_max_consec,
            "total_pnl_pct": round(sum(pos_pnls) * 100, 2),
            "invested_usd": round(s_invested, 2),
            "pnl_usd": round(s_pnl_usd, 2),
        }

    # Global enriched stats
    global_expectancy = avg_pnl  # avg_pnl IS expectancy (mean PnL per trade)
    global_avg_win = sum(winners) / len(winners) if winners else 0
    global_avg_loss = abs(sum(losers)) / len(losers) if losers else 0
    global_breakeven_wr = global_avg_loss / (global_avg_win + global_avg_loss) if (global_avg_win + global_avg_loss) > 0 else 0.5
    global_pf = abs(sum(winners) / sum(losers)) if losers and sum(losers) != 0 else float("inf")
    g_max_consec = 0
    g_consec = 0
    for p in pnls:
        if p < 0:
            g_consec += 1
            g_max_consec = max(g_max_consec, g_consec)
        else:
            g_consec = 0

    summary = {
        "total_rows": total,
        "tp": tp_count,
        "sl": sl_count,
        "timeout": timeout_count,
        "win_rate": round(win_rate, 3),
        "avg_pnl": round(avg_pnl, 4),
        "expectancy": round(global_expectancy, 4),
        "profit_factor": round(global_pf, 2) if global_pf != float("inf") else "inf",
        "breakeven_wr": round(global_breakeven_wr, 4),
        "max_consecutive_losses": g_max_consec,
        "total_invested_usd": round(total_invested, 2),
        "total_pnl_usd": round(total_pnl_usd, 2),
        "roi_pct": roi_pct,
        "strategies": strategy_stats,
    }

    # v66: RT vs batch split
    rt_trades = [t for t in trades if t.get("source") == "rt"]
    batch_trades = [t for t in trades if t.get("source") != "rt"]

    def _source_stats(src_trades, label):
        # Only count closed trades (already filtered by neq("status","open") above)
        closed = [t for t in src_trades if t.get("status") in ("tp_hit", "sl_hit", "timeout", "trail_stop")]
        if not closed:
            return None
        n = len(closed)
        w = sum(1 for t in closed if float(t.get("pnl_pct") or 0) > 0)
        inv = sum(float(t.get("position_usd") or 0) for t in closed)
        pnl = sum(float(t.get("pnl_usd") or 0) for t in closed)
        wr = w / n if n else 0
        return {"label": label, "rows": n, "win_rate": round(wr, 3),
                "invested": round(inv, 2), "pnl_usd": round(pnl, 2)}

    rt_stats = _source_stats(rt_trades, "RT")
    batch_stats = _source_stats(batch_trades, "batch")
    summary["rt_stats"] = rt_stats
    summary["batch_stats"] = batch_stats

    logger.info(
        "paper_trader SUMMARY (7d): %d rows, WR=%.1f%%, avgPnL=%.2f%%, E[R]=%.2f%%, PF=%s | "
        "$%.2f invested, $%+.2f PnL (ROI %.1f%%) | TP=%d SL=%d TO=%d | maxConsecL=%d",
        total, win_rate * 100, avg_pnl * 100, global_expectancy * 100,
        round(global_pf, 2) if global_pf != float("inf") else "inf",
        total_invested, total_pnl_usd, roi_pct,
        tp_count, sl_count, timeout_count, g_max_consec,
    )
    # v66: Log RT vs batch breakdown
    for ss in [rt_stats, batch_stats]:
        if ss:
            logger.info(
                "  [%s] %d rows, WR=%.1f%%, $%.2f invested, $%+.2f PnL",
                ss["label"], ss["rows"], ss["win_rate"] * 100, ss["invested"], ss["pnl_usd"],
            )
    for name, s in strategy_stats.items():
        logger.info(
            "  %s: %d pos, WR=%.1f%%, E[R]=%.2f%%, PF=%s, BEwr=%.1f%% | $%.2f→$%+.2f | maxCL=%d",
            name, s["positions"], s["win_rate"] * 100, s["expectancy"] * 100,
            s["profit_factor"], s["breakeven_wr"] * 100, s["invested_usd"], s["pnl_usd"],
            s["max_consecutive_losses"],
        )
    return summary


def _days_ago_iso(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def kol_attribution(client, days: int = 7) -> dict:
    """
    v74: Aggregate paper trade outcomes by KOL.
    Returns {kol_group: {total, wins, wr, pnl_usd, avg_pnl_pct, best_trade, worst_trade}}.
    Feeds back into KOL whitelist optimization.
    """
    try:
        result = (
            client.table("paper_trades")
            .select("kol_group, kol_tier, pnl_pct, pnl_usd, status, strategy, symbol")
            .neq("status", "open")
            .eq("is_shadow", False)
            .not_.is_("kol_group", "null")
            .gte("created_at", _days_ago_iso(days))
            .execute()
        )
        trades = result.data or []
    except Exception as e:
        logger.error("kol_attribution: query failed: %s", e)
        return {}

    if not trades:
        return {}

    # Group by KOL
    by_kol: dict[str, list[dict]] = {}
    for t in trades:
        kol = t.get("kol_group")
        if kol:
            by_kol.setdefault(kol, []).append(t)

    attribution = {}
    for kol, kol_trades in by_kol.items():
        total = len(kol_trades)
        pnls = [float(t.get("pnl_pct") or 0) for t in kol_trades]
        wins = sum(1 for p in pnls if p > 0)
        pnl_usd = sum(float(t.get("pnl_usd") or 0) for t in kol_trades)

        best = max(kol_trades, key=lambda t: float(t.get("pnl_pct") or 0))
        worst = min(kol_trades, key=lambda t: float(t.get("pnl_pct") or 0))

        attribution[kol] = {
            "tier": kol_trades[0].get("kol_tier", "?"),
            "total": total,
            "wins": wins,
            "wr": round(wins / total, 3) if total else 0,
            "pnl_usd": round(pnl_usd, 2),
            "avg_pnl_pct": round(sum(pnls) / total * 100, 2) if total else 0,
            "best_trade": f"{best.get('symbol')} +{float(best.get('pnl_pct') or 0)*100:.1f}%",
            "worst_trade": f"{worst.get('symbol')} {float(worst.get('pnl_pct') or 0)*100:.1f}%",
        }

    # Log top and bottom 5 KOLs
    sorted_kols = sorted(attribution.items(), key=lambda x: x[1]["pnl_usd"], reverse=True)
    logger.info("KOL Attribution (%dd, %d KOLs, %d trades):", days, len(attribution), sum(v["total"] for v in attribution.values()))
    for kol, stats in sorted_kols[:5]:
        logger.info("  TOP: %s (%s) — %d trades, WR=%.0f%%, PnL=$%+.2f",
                     kol, stats["tier"], stats["total"], stats["wr"]*100, stats["pnl_usd"])
    for kol, stats in sorted_kols[-3:]:
        if stats["pnl_usd"] < 0:
            logger.info("  BOT: %s (%s) — %d trades, WR=%.0f%%, PnL=$%+.2f",
                         kol, stats["tier"], stats["total"], stats["wr"]*100, stats["pnl_usd"])

    # v82 P2-2: Closed-loop — feed attribution back into kol_rt_whitelist
    # v103: Disabled auto-update — whitelist is now manually managed
    # _update_whitelist_from_attribution(client, attribution, by_kol, days)

    return attribution


def _update_whitelist_from_attribution(client, attribution: dict, by_kol: dict, days: int):
    """v82: Merge kol_attribution results into scoring_config.kol_rt_whitelist.
    Adds per-strategy breakdown and best_strategy for adaptive trading."""
    deprecated = {"MOONBAG", "WIDE_RUNNER", "SCALE_OUT", "TP100_SL30"}
    min_calls = 3
    wr_threshold = 0.40

    try:
        # Read current whitelist
        r = client.table("scoring_config").select("kol_rt_whitelist").eq("id", 1).execute()
        whitelist = {}
        if r.data and r.data[0].get("kol_rt_whitelist"):
            cached = r.data[0]["kol_rt_whitelist"]
            if isinstance(cached, str):
                import json
                cached = json.loads(cached)
            whitelist = cached

        # Build per-strategy breakdown from raw trades
        for kol, kol_trades in by_kol.items():
            active_trades = [t for t in kol_trades if t.get("strategy") not in deprecated]
            if not active_trades:
                continue

            attr = attribution.get(kol, {})
            total = len(active_trades)
            wins = sum(1 for t in active_trades if float(t.get("pnl_usd") or 0) > 0)
            pnl = sum(float(t.get("pnl_usd") or 0) for t in active_trades)
            wr = wins / total if total > 0 else 0

            # Per-strategy
            strat_map = {}
            from collections import defaultdict
            strat_agg = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0})
            for t in active_trades:
                s = t.get("strategy", "?")
                p = float(t.get("pnl_usd") or 0)
                strat_agg[s]["n"] += 1
                strat_agg[s]["pnl"] += p
                if p > 0:
                    strat_agg[s]["wins"] += 1

            best_strat = None
            best_pnl = -999
            for s, ss in strat_agg.items():
                swr = ss["wins"] / ss["n"] if ss["n"] > 0 else 0
                strat_map[s] = {"n": ss["n"], "wr": round(swr, 4), "pnl": round(ss["pnl"], 2)}
                if ss["pnl"] > best_pnl:
                    best_pnl = ss["pnl"]
                    best_strat = s

            approved = wr >= wr_threshold and total >= min_calls
            whitelist[kol] = {
                "wr": round(wr, 4),
                "total": total,
                "hits": wins,
                "pnl": round(pnl, 2),
                "approved": approved,
                "best_strategy": best_strat,
                "strategies": strat_map,
            }

        # Fallback: relax to 30% if < 3 approved
        approved_count = sum(1 for v in whitelist.values() if v.get("approved"))
        if approved_count < 3:
            for kol, wl in whitelist.items():
                if not wl.get("approved") and wl.get("total", 0) >= min_calls:
                    if wl.get("wr", 0) >= 0.30:
                        wl["approved"] = True

        # Write back
        from datetime import datetime, timezone
        client.table("scoring_config").update({
            "kol_rt_whitelist": whitelist,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", 1).execute()

        approved_count = sum(1 for v in whitelist.values() if v.get("approved"))
        logger.info("kol_attribution → whitelist updated: %d/%d approved", approved_count, len(whitelist))

    except Exception as e:
        logger.warning("kol_attribution → whitelist update failed: %s", e)
