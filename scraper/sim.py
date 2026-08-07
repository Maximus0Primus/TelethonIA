"""
Unified Strategy Simulator — replaces 16 separate sim scripts.

Tests 650+ strategy configurations on OHLCV candle data with:
- Correct first-call dedup (no 3-min batch duplicates)
- Candle-by-candle simulation with slippage
- Kelly-sized compound bankroll
- Monte Carlo risk analysis
- Runner capture analysis

Usage:
    python scraper/sim.py                           # full run
    python scraper/sim.py --cache-only              # cache only
    python scraper/sim.py --dry-run                 # show counts
    python scraper/sim.py --max-fetch 100           # limit API calls
    python scraper/sim.py --since 2026-03-15        # date filter
    python scraper/sim.py --strategies dynamic      # only dynamic trail
    python scraper/sim.py --strategies DTRAIL,FIXED # only these types
    python scraper/sim.py --top 30                  # top N results
    python scraper/sim.py --runner-analysis         # runner capture %
    python scraper/sim.py --mc-sims 5000            # Monte Carlo sims
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time as time_mod
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

from sim_engines import simulate, resample_to_live_checks, compute_buy_slippage

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRAPER_DIR = Path(__file__).resolve().parent
load_dotenv(SCRAPER_DIR / ".env")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
SB_HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
}

CACHE_DIR = SCRAPER_DIR / "ohlcv_cache"
CACHE_DIR.mkdir(exist_ok=True)

# Bankroll
START_BANKROLL = 500.0
KELLY_FRAC = 0.127
MAX_POS = 120.0
MIN_TRADES = 20

# KOL whitelist
KOL_WHITELIST = {
    "degenncabal", "legerlegends", "luca_apes", "marcellcooks",
    "bossmancallsofficial", "alstein_gemclub", "donniesdegen", "darkocalls",
    "degenscabal", "eunicalls", "levisaplha", "powsgemcalls",
    "gubbinscalls", "lollycalls", "archerrgambles", "spidersjournal",
}

# v113: KOL blacklist — excluded from sim (not scraped in live or bad data)
KOL_BLACKLIST = {
    "bat_gamble",    # removed from scraping v108 — can't reproduce these trades
    "veigarcalls",   # ROCKET pool bug + only 1-2 trades, skews everything
}

# v113: Token address blacklist — known bad trades (wrong pool, fake PnL)
TOKEN_BLACKLIST = {
    "4YiLHDR4B4pE4R5GUMA8HG8YunyeLwcobtEtvwMupump",  # $ROCKET — pump.fun pool bug
}

# OHLCV window (6h + buffer)
MAX_WINDOW_MIN = 365


# ---------------------------------------------------------------------------
# Strategy grid generation
# ---------------------------------------------------------------------------

def build_strategy_grid(strategy_filter: str | None = None) -> list[dict]:
    """Generate all strategy configurations.
    v116: Expanded grid — more param ranges for exhaustive search."""
    configs = []

    # 1. FIXED: TP x SL x timeout (expanded)
    for tp in [30, 40, 50, 60, 70, 80, 90, 100, 120, 150]:
        for sl in [20, 30, 40, 50, 60, 70, 80]:
            for timeout in [30, 45, 60, 90, 120, 180, 240, 360, 480]:
                configs.append({
                    "name": f"TP{tp}_SL{sl}_{timeout}m",
                    "type": "FIXED",
                    "tp_mult": 1 + tp / 100,
                    "sl_mult": 1 - sl / 100,
                    "horizon_min": timeout,
                })

    # 2. DTRAIL: trail x act x SL x timeout (expanded)
    for trail in [3, 5, 8, 10, 15]:
        for act in [5, 10, 15, 20, 25, 30, 35, 40]:
            for sl in [40, 50, 60, 70, 80]:
                for timeout in [60, 120, 240, 360, 480]:
                    configs.append({
                        "name": f"DTRAIL{trail}_ACT{act}_SL{sl}_{timeout}m",
                        "type": "DTRAIL",
                        "trail_pct": trail / 100,
                        "activation_pct": act / 100,
                        "sl_mult": 1 - sl / 100,
                        "horizon_min": timeout,
                    })

    # 3. TRAIL: trail x TP x SL x timeout (expanded)
    for trail in [5, 10, 15, 20, 25]:
        for tp in [30, 50, 70, 100, 150]:
            for sl in [40, 50, 60, 70]:
                for timeout in [60, 120, 240, 360]:
                    configs.append({
                        "name": f"TRAIL{trail}_TP{tp}_SL{sl}_{timeout}m",
                        "type": "TRAIL",
                        "trail_pct": trail / 100,
                        "activation_pct": trail / 100,
                        "tp_mult": 1 + tp / 100,
                        "sl_mult": 1 - sl / 100,
                        "horizon_min": timeout,
                    })

    # 4. BE: activation x TP x SL x timeout (expanded)
    for be_act in [10, 15, 20, 25, 30]:
        for tp in [30, 50, 70, 100, 150]:
            for sl in [40, 50, 60, 70]:
                for timeout in [60, 120, 240, 360]:
                    configs.append({
                        "name": f"BE{be_act}_TP{tp}_SL{sl}_{timeout}m",
                        "type": "BE",
                        "be_activation": be_act / 100,
                        "tp_mult": 1 + tp / 100,
                        "sl_mult": 1 - sl / 100,
                        "horizon_min": timeout,
                    })

    # 5. SCALP: TP x SL x timeout (expanded)
    for tp in [5, 10, 15, 20, 25, 30]:
        for sl in [5, 10, 15, 20, 25]:
            for timeout in [15, 30, 45, 60]:
                configs.append({
                    "name": f"SCALP_TP{tp}_SL{sl}_{timeout}m",
                    "type": "SCALP",
                    "tp_mult": 1 + tp / 100,
                    "sl_mult": 1 - sl / 100,
                    "horizon_min": timeout,
                })

    # 6. DECAY: tp_start/end x SL x timeout (expanded)
    for tp_s, tp_e, label_tp in [
        (2.0, 1.20, "TP100_E20"), (2.0, 1.15, "TP100_E15"),
        (2.0, 1.30, "TP100_E30"), (1.70, 1.15, "TP70_E15"),
        (2.0, 1.10, "TP100_E10"), (1.50, 1.10, "TP50_E10"),
        (1.50, 1.20, "TP50_E20"), (2.50, 1.20, "TP150_E20"),
        (2.50, 1.30, "TP150_E30"), (3.0, 1.30, "TP200_E30"),
    ]:
        for sl in [40, 50, 60, 70]:
            for timeout in [60, 120, 240, 360]:
                configs.append({
                    "name": f"DECAY_{label_tp}_SL{sl}_{timeout}m", "type": "DECAY",
                    "tp_start": tp_s, "tp_end": tp_e, "sl_mult": 1 - sl / 100,
                    "horizon_min": timeout,
                })

    # 7. SPLIT: variants x SL x timeout (expanded)
    for label, t1, t2, t2t in [
        ("SPLIT_50_100", 1.50, 2.0, None),
        ("SPLIT_50_TRAIL", 1.50, None, 0.20),
        ("SPLIT_30_100", 1.30, 2.0, None),
        ("SPLIT_50_150", 1.50, 2.50, None),
        ("SPLIT_30_TRAIL", 1.30, None, 0.15),
    ]:
        for sl in [40, 50, 60, 70]:
            for timeout in [60, 120, 240, 360]:
                configs.append({
                    "name": f"{label}_SL{sl}_{timeout}m", "type": "SPLIT",
                    "t1_tp": t1, "t2_tp": t2, "sl_mult": 1 - sl / 100, "t2_trail": t2t,
                    "horizon_min": timeout,
                })

    # 8. DYNAMIC_TRAIL strategies (always full mode now)
    configs.extend(_build_dynamic_trail_grid(full=True))

    # 9. CONTEXTUAL: trail adapts to mcap segment
    configs.extend(_build_contextual_grid())

    # 10. SCALE_OUT: progressive exit in tranches
    configs.extend(_build_scale_out_grid())

    # 11. DIP_BUY: re-enter after dump + bounce
    configs.extend(_build_dip_buy_grid())

    # Apply filter
    if strategy_filter:
        filters = [s.strip().upper() for s in strategy_filter.split(",")]
        if "DYNAMIC_FULL" in filters:
            filters.remove("DYNAMIC_FULL")
        if "DYNAMIC" in filters:
            filters.remove("DYNAMIC")
            filters.append("DYNAMIC_TRAIL")
        if filters:
            configs = [c for c in configs if c["type"] in filters]

    return configs


def _build_dynamic_trail_grid(full: bool = False) -> list[dict]:
    """Build dynamic trail strategy configs. (expanded v116 — always full)"""
    configs = []

    # --- TIME DECAY (expanded) ---
    # Pruned: only combos where start and end differ by ≥3 (skip near-identical pairs)
    td_combos = [(s, e) for s in [3, 5, 8, 10, 12, 15, 20, 25]
                  for e in [3, 5, 8, 10, 15, 20] if s != e and abs(s - e) >= 3]
    td_timeouts = [60, 120, 240, 360, 480]
    td_acts = [10, 15, 20, 25, 30]
    td_sls = [40, 50, 60, 70]

    for start, end in td_combos:
        direction = "W2T" if start > end else "T2W"
        for timeout in td_timeouts:
            for act in td_acts:
                for sl in td_sls:
                    configs.append({
                        "name": f"TD_{direction}_{start}to{end}_ACT{act}_SL{sl}_{timeout}m",
                        "type": "DYNAMIC_TRAIL", "mode": "time_decay",
                        "trail_start": start, "trail_end": end,
                        "activation_pct": act / 100, "sl_mult": 1 - sl / 100,
                        "horizon_min": timeout,
                    })

    # --- GAIN ADAPTIVE (expanded) ---
    profiles = [
        ("STD", [30, 100, 300], [0.05, 0.10, 0.15, 0.20]),
        ("AGG", [20, 50, 150], [0.05, 0.08, 0.12, 0.18]),
        ("PAT", [50, 150, 400], [0.05, 0.10, 0.15, 0.25]),
        ("TIGHT", [15, 40, 100], [0.03, 0.05, 0.08, 0.12]),
        ("WIDE", [50, 200, 500], [0.08, 0.15, 0.20, 0.30]),
    ]
    for label, thresholds, trails in profiles:
        for timeout in [120, 240, 360, 480]:
            for act in [10, 15, 20, 25]:
                for sl in [50, 60, 70]:
                    configs.append({
                        "name": f"GADAPT_{label}_ACT{act}_SL{sl}_{timeout}m",
                        "type": "DYNAMIC_TRAIL", "mode": "gain_adaptive",
                        "gain_thresholds": thresholds, "gain_trails": trails,
                        "activation_pct": act / 100, "sl_mult": 1 - sl / 100,
                        "horizon_min": timeout,
                    })

    # --- GAIN-TIME HYBRID (expanded) ---
    for timeout in [60, 120, 240, 360]:
        for act in [10, 15, 20, 25]:
            for sl in [50, 60, 70]:
                configs.append({
                    "name": f"GTHYBRID_ACT{act}_SL{sl}_{timeout}m",
                    "type": "DYNAMIC_TRAIL", "mode": "gain_time_hybrid",
                    "activation_pct": act / 100, "sl_mult": 1 - sl / 100,
                    "horizon_min": timeout,
                })

    # --- RATCHET TRAIL (expanded) ---
    milestone_sets = [
        ("STD", [(30, 10, 5), (50, 25, 7), (100, 50, 10), (200, 120, 15), (400, 250, 20)]),
        ("AGG", [(20, 5, 5), (40, 15, 8), (80, 35, 12), (150, 80, 15)]),
        ("PAT", [(50, 20, 5), (100, 50, 8), (200, 100, 12), (500, 300, 18)]),
        ("TIGHT", [(15, 5, 3), (30, 10, 5), (60, 25, 8), (100, 50, 10)]),
    ]
    for ms_label, milestones in milestone_sets:
        for timeout in [120, 240, 360, 480]:
            for act in [10, 15, 20, 25]:
                for sl in [50, 60, 70]:
                    configs.append({
                        "name": f"RATCHET_{ms_label}_ACT{act}_SL{sl}_{timeout}m",
                        "type": "DYNAMIC_TRAIL", "mode": "ratchet_trail",
                        "milestones": milestones, "trail_base": 5,
                        "activation_pct": act / 100, "sl_mult": 1 - sl / 100,
                        "horizon_min": timeout,
                    })

    # --- TIME-GAIN RATCHET (expanded) ---
    tg_milestone_sets = [
        ("STD", [(30, 10, 8), (50, 25, 10), (100, 50, 12), (200, 120, 15)]),
        ("AGG", [(20, 5, 5), (40, 15, 8), (80, 30, 10), (150, 70, 12)]),
        ("PAT", [(50, 20, 8), (100, 50, 10), (200, 100, 15), (400, 200, 20)]),
    ]
    for tg_label, milestones in tg_milestone_sets:
        for timeout in [120, 240, 360, 480]:
            for act in [10, 15, 20, 25]:
                for sl in [50, 60, 70]:
                    configs.append({
                        "name": f"TGRATCHET_{tg_label}_ACT{act}_SL{sl}_{timeout}m",
                        "type": "DYNAMIC_TRAIL", "mode": "time_gain_ratchet",
                        "milestones": milestones, "trail_base": 5,
                        "activation_pct": act / 100, "sl_mult": 1 - sl / 100,
                        "horizon_min": timeout,
                    })

    return configs


def _build_contextual_grid() -> list[dict]:
    """Build CONTEXTUAL strategies — trail/timeout vary by mcap segment. (expanded v116)"""
    configs = []
    segmentation_sets = [
        # 2 segments: micro vs rest
        ("2SEG_100K", [100000],
         [[15, 5], [12, 5], [20, 8], [10, 3], [8, 5]],
         [[360, 120], [360, 240], [480, 120], [240, 120]],
         [[25, 15], [20, 15], [30, 15], [20, 10]]),
        # 3 segments: micro / small / mid+
        ("3SEG", [100000, 1000000],
         [[15, 10, 5], [20, 10, 5], [12, 8, 5], [20, 12, 3], [10, 5, 3]],
         [[360, 240, 120], [360, 360, 240], [480, 240, 120]],
         [[25, 20, 15], [30, 20, 15], [20, 15, 10]]),
        # 2 segments: small vs big
        ("2SEG_500K", [500000],
         [[12, 5], [15, 5], [10, 5], [20, 5], [8, 3]],
         [[360, 120], [360, 240], [480, 120], [240, 60]],
         [[20, 15], [25, 15], [30, 15], [15, 10]]),
        # 2 segments: tiny (< 50K mcap) vs rest
        ("2SEG_50K", [50000],
         [[20, 8], [15, 5], [25, 10]],
         [[480, 240], [360, 120]],
         [[30, 15], [25, 20]]),
    ]
    for seg_label, bps, trail_combos, to_combos, act_combos in segmentation_sets:
        for ti, trails in enumerate(trail_combos):
            for toi, timeouts in enumerate(to_combos):
                for ai, acts in enumerate(act_combos):
                    for sl in [50, 60, 70]:
                        name = f"CTX_{seg_label}_t{ti}_to{toi}_a{ai}_SL{sl}"
                        configs.append({
                            "name": name, "type": "CONTEXTUAL",
                            "mcap_breakpoints": bps,
                            "trail_per_segment": trails,
                            "timeout_per_segment": timeouts,
                            "act_per_segment": acts,
                            "sl_mult": 1 - sl / 100,
                            "horizon_min": max(timeouts),
                        })
    return configs


def _build_scale_out_grid() -> list[dict]:
    """Build SCALE_OUT strategies — progressive exit in tranches. (expanded v116)"""
    configs = []
    tranche_configs = [
        ("SO_30_60_100", [(30, 0.25), (60, 0.25), (100, 0.25)]),  # 25% runner
        ("SO_25_50_100", [(25, 0.25), (50, 0.25), (100, 0.25)]),
        ("SO_30_60", [(30, 0.33), (60, 0.33)]),  # 33% runner
        ("SO_50_100", [(50, 0.33), (100, 0.33)]),  # 33% runner, patient
        ("SO_20_40_80", [(20, 0.25), (40, 0.25), (80, 0.25)]),  # aggressive
        ("SO_50_150", [(50, 0.33), (150, 0.33)]),  # very patient
        ("SO_20_50", [(20, 0.33), (50, 0.33)]),  # quick scalp out
    ]
    for label, tranches in tranche_configs:
        for runner_trail in [5, 10, 15, 20, 25]:
            for runner_act in [20, 30, 40, 50, 70]:
                for sl in [50, 60, 70]:
                    for timeout in [120, 240, 360]:
                        configs.append({
                            "name": f"{label}_RT{runner_trail}_RA{runner_act}_SL{sl}_{timeout}m",
                            "type": "SCALE_OUT",
                            "tranches": tranches,
                            "runner_trail": runner_trail,
                            "runner_act": runner_act,
                            "sl_mult": 1 - sl / 100,
                            "horizon_min": timeout,
                        })
    return configs


def _build_dip_buy_grid() -> list[dict]:
    """Build DIP_BUY strategies — re-enter after dump + bounce.

    v116: Two tiers of DIP_BUY configs:
    1. Shared-param (original): P1 and P2 use same trail/act/sl — full dip/bounce/timeout grid.
    2. Split-param (new): P1 and P2 have independent trail/act/sl — focused on best dip/bounce.
    """
    configs = []

    # ---- Tier 1: Shared-param DIP_BUY (original grid) ----
    for dip in [20, 30, 40]:
        for bounce in [0, 5, 10]:
            for trail in [5, 10]:
                for act in [15, 20, 30]:
                    for timeout in [240, 360]:
                        for sl in [60, 70]:
                            b_label = f"B{bounce}" if bounce > 0 else "DIR"
                            configs.append({
                                "name": f"DIP{dip}_{b_label}_T{trail}_A{act}_SL{sl}_{timeout}m",
                                "type": "DIP_BUY",
                                "dip_threshold": -dip,
                                "bounce_threshold": bounce,
                                "dip_size_mult": 1.0,
                                "trail": trail, "act": act, "sl": sl,
                                "sl_mult": 1 - sl / 100,
                                "horizon_min": timeout,
                            })

    # ---- Tier 2: Split-param DIP_BUY (P1 ≠ P2) ----
    # Rationale: P1 enters at KOL call (top of pump) → needs tighter SL, wider trail
    #            P2 enters after dip+bounce (proven floor) → can use wider SL, tighter trail
    # Full dip/bounce/timeout grid × independent P1/P2 trail/act/sl
    _P1_TRAILS = [5, 10, 15]      # P1 wider trail: let pump run
    _P1_ACTS   = [10, 15, 20]     # P1 activation thresholds
    _P1_SLS    = [50, 60, 70]     # P1 SL: tighter to cut rugs fast
    _P2_TRAILS = [5, 10]          # P2 tighter trail: lock in bounce profit
    _P2_ACTS   = [10, 15]         # P2 lower activation: bounce already confirmed
    _P2_SLS    = [60, 70]         # P2 wider SL: dip already cleaned weak hands

    for dip in [20, 30, 40]:
        for bounce in [0, 5, 10]:
            for timeout in [240, 360]:
                b_label = f"B{bounce}" if bounce > 0 else "DIR"
                for p1_trail in _P1_TRAILS:
                    for p1_act in _P1_ACTS:
                        for p1_sl in _P1_SLS:
                            for p2_trail in _P2_TRAILS:
                                for p2_act in _P2_ACTS:
                                    for p2_sl in _P2_SLS:
                                        # Skip if P1 == P2 (already covered by tier 1)
                                        if (p1_trail, p1_act, p1_sl) == (p2_trail, p2_act, p2_sl):
                                            continue
                                        configs.append({
                                            "name": (f"DIP{dip}_{b_label}"
                                                     f"_P1T{p1_trail}A{p1_act}S{p1_sl}"
                                                     f"_P2T{p2_trail}A{p2_act}S{p2_sl}"
                                                     f"_{timeout}m"),
                                            "type": "DIP_BUY",
                                            "dip_threshold": -dip,
                                            "bounce_threshold": bounce,
                                            "dip_size_mult": 1.0,
                                            # Shared fallbacks (used by engine if p1_*/p2_* missing)
                                            "trail": p1_trail, "act": p1_act, "sl": p1_sl,
                                            "sl_mult": 1 - p1_sl / 100,
                                            "horizon_min": timeout,
                                            # P1-specific
                                            "p1_trail": p1_trail, "p1_act": p1_act, "p1_sl": p1_sl,
                                            # P2-specific
                                            "p2_trail": p2_trail, "p2_act": p2_act, "p2_sl": p2_sl,
                                        })

    # ---- DIP + SCALE_OUT (unchanged) ----
    for dip in [20, 30]:
        for bounce in [0, 5]:
            for tranche_label, tranches in [
                ("SO30_60", [(30, 0.33), (60, 0.33)]),
                ("SO25_50_100", [(25, 0.25), (50, 0.25), (100, 0.25)]),
                ("SO50_100", [(50, 0.33), (100, 0.33)]),
            ]:
                for runner_trail in [10, 15]:
                    for runner_act in [30, 50]:
                        for sl in [60, 70]:
                            b_label = f"B{bounce}" if bounce > 0 else "DIR"
                            configs.append({
                                "name": f"DIP{dip}_{b_label}_{tranche_label}_RT{runner_trail}_RA{runner_act}_SL{sl}",
                                "type": "DIP_SCALE_OUT",
                                "dip_threshold": -dip,
                                "bounce_threshold": bounce,
                                "dip_size_mult": 1.0,
                                "tranches": tranches,
                                "runner_trail": runner_trail,
                                "runner_act": runner_act,
                                "trail": 5, "act": 20, "sl": sl,
                                "sl_mult": 1 - sl / 100,
                                "horizon_min": 360,
                            })

    return configs


# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------

# v14: chain filter applied to every paper_trades + price_ticks query.
# Set once by main() from args.chain before any fetch runs. Default 'solana'
# preserves legacy behavior — every pre-v14 sim run implicitly assumed solana.
# Set to 'all' to disable the filter (mix solana + ethereum; only useful for
# raw DB audits, never for PnL-based ranking since fee models differ).
_SIM_CHAIN: str = "solana"


def _chain_params() -> list[tuple]:
    """Return the chain filter tuples to append to sb_get params.
    Empty list when chain=='all' (= no filter)."""
    if _SIM_CHAIN == "all":
        return []
    return [("chain", f"eq.{_SIM_CHAIN}")]


def set_sim_chain(chain: str) -> None:
    """Set the module-level chain filter. Called from main() after argparse."""
    global _SIM_CHAIN
    _SIM_CHAIN = chain


def sb_get(table: str, params: list[tuple]) -> list[dict]:
    """Paginate Supabase REST GET. Auto-selects keyset pagination on
    `created_at` when SELECT includes `id` + `created_at` and ORDER targets
    `created_at`; otherwise falls back to OFFSET pagination.

    v14e.59: keyset switch. The mega-sweep extended query pulls paper_trades
    grouped per shard; the SOL universe crossed ~240k rows and OFFSET 238000
    on Supabase (60s statement_timeout, sequential scan from row 0) returns
    HTTP 500 (code 57014). Keyset on the indexed `created_at` is O(log N) per
    page and survives unbounded growth. Dedup by `id` because consecutive
    pages overlap on identical timestamps (millisecond precision; rare but
    happens with batch inserts).
    """
    select_str = next((v for k, v in params if k == "select"), "")
    select_cols = {c.strip() for c in select_str.split(",")}
    order_vals = [v for k, v in params if k == "order"]
    uses_created_at_order = any("created_at" in o for o in order_vals)
    use_keyset = (
        uses_created_at_order
        and "id" in select_cols
        and "created_at" in select_cols
    )

    if not use_keyset:
        all_rows = []
        limit = 1000
        offset = 0
        while True:
            p = params + [("limit", str(limit)), ("offset", str(offset))]
            r = requests.get(f"{SUPABASE_URL}/rest/v1/{table}", headers=SB_HEADERS,
                             params=p, timeout=30)
            if r.status_code != 200:
                print(f"  Supabase error {r.status_code}: {r.text[:300]}")
                r.raise_for_status()
            rows = r.json()
            all_rows.extend(rows)
            if len(rows) < limit:
                break
            offset += len(rows)
        return all_rows

    # Keyset pagination on created_at. Preserve all caller-supplied filters
    # on first request, then on every subsequent request replace any
    # `created_at` lower-bound filter (gte./gt.) with a fresh cursor.
    all_rows: list[dict] = []
    seen_ids: set = set()
    limit = 1000
    cursor: str | None = None
    base_no_cursor = [
        (k, v) for k, v in params
        if not (k == "created_at" and isinstance(v, str)
                and (v.startswith("gte.") or v.startswith("gt.")))
    ]
    while True:
        if cursor is None:
            p = list(params) + [("limit", str(limit))]
        else:
            p = base_no_cursor + [("created_at", f"gte.{cursor}"), ("limit", str(limit))]
        r = requests.get(f"{SUPABASE_URL}/rest/v1/{table}", headers=SB_HEADERS,
                         params=p, timeout=30)
        if r.status_code != 200:
            print(f"  Supabase error {r.status_code}: {r.text[:300]}")
            r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        new_rows = [row for row in rows if row.get("id") not in seen_ids]
        for row in new_rows:
            seen_ids.add(row.get("id"))
        all_rows.extend(new_rows)
        if len(rows) < limit:
            break
        next_cursor = rows[-1].get("created_at")
        if next_cursor is None or next_cursor == cursor:
            # Safety: tie on the entire last page (every row has cursor's ts)
            # would loop forever. Break and let caller see what we have.
            print(f"  WARN: keyset stalled at created_at={cursor}; "
                  f"returning {len(all_rows)} rows so far")
            break
        cursor = next_cursor
    return all_rows


def fetch_all_trades_by_strategy(since: str) -> list[dict]:
    """Fetch ALL closed paper trades (including shadows) grouped by strategy.
    Used by --from-trades mode: real PnL instead of OHLCV simulation.
    v14: filtered by _SIM_CHAIN (default 'solana')."""
    params = [
        ("select", "token_address,strategy,pnl_pct,status,created_at,kol_group,"
                   "position_usd,exit_minutes,high_price_seen,entry_price,exit_price,"
                   "rt_liquidity_usd,rt_token_age_hours,is_shadow,chain"),
        ("status", "in.(trail_stop,sl_hit,timeout,tp_hit)"),
        ("source", "eq.rt"),
        ("created_at", f"gte.{since}T00:00:00Z"),
        ("order", "created_at.asc"),
    ] + _chain_params()
    trades = sb_get("paper_trades", params)
    print(f"Fetched {len(trades)} closed trades (chain={_SIM_CHAIN}, all strategies, incl shadows) since {since}")
    return trades


def fetch_paper_trades(since: str) -> list[dict]:
    """v14: filtered by _SIM_CHAIN (default 'solana')."""
    params = [
        ("select", "id,token_address,pair_address,strategy,entry_price,exit_price,"
                   "status,pnl_pct,created_at,kol_group,source,high_price_seen,position_usd,"
                   "entry_mcap,rt_liquidity_usd,rt_volume_24h,rt_token_age_hours,"
                   "rt_is_pump_fun,n_kol_confirmations,chain"),
        ("status", "in.(trail_stop,sl_hit,timeout,tp_hit)"),
        ("source", "eq.rt"),  # v113: RT only — batch has no age/enrichment data
        ("created_at", f"gte.{since}T00:00:00Z"),
        ("order", "created_at.asc"),
    ] + _chain_params()
    trades = sb_get("paper_trades", params)
    print(f"Fetched {len(trades)} closed paper trades (chain={_SIM_CHAIN}) since {since}")
    return trades


# ---------------------------------------------------------------------------
# Dedup: first call only per token within 24h
# ---------------------------------------------------------------------------

def dedup_first_call(trades: list[dict]) -> list[dict]:
    """Keep only the first call per unique token within 24h windows."""
    sorted_trades = sorted(trades, key=lambda t: t["created_at"])
    seen: dict[str, datetime] = {}  # token_address -> last entry time
    result = []
    for t in sorted_trades:
        token = t["token_address"]
        dt = datetime.fromisoformat(t["created_at"].replace("Z", "+00:00"))
        last = seen.get(token)
        if last and (dt - last).total_seconds() < 86400:
            continue
        seen[token] = dt
        result.append(t)
    return result


# ---------------------------------------------------------------------------
# OHLCV: pair resolution + candle fetch + caching
# ---------------------------------------------------------------------------

PAIR_CACHE_FILE = CACHE_DIR / "_pair_address_cache.json"


def _load_pair_cache() -> dict[str, str | None]:
    if PAIR_CACHE_FILE.exists():
        try:
            return json.loads(PAIR_CACHE_FILE.read_text())
        except (json.JSONDecodeError, KeyError):
            pass
    return {}


def _save_pair_cache(cache: dict):
    PAIR_CACHE_FILE.write_text(json.dumps(cache))


_pair_cache: dict[str, str | None] = _load_pair_cache()


def resolve_pair_address(token_address: str) -> str | None:
    if token_address in _pair_cache:
        return _pair_cache[token_address]
    try:
        r = requests.get(
            f"https://api.dexscreener.com/latest/dex/tokens/{token_address}",
            timeout=15)
        if r.status_code == 429:
            time_mod.sleep(3)
            r = requests.get(
                f"https://api.dexscreener.com/latest/dex/tokens/{token_address}",
                timeout=15)
        r.raise_for_status()
        pairs = r.json().get("pairs") or []
        sol_pairs = [p for p in pairs if p.get("chainId") == "solana"]
        if not sol_pairs:
            _pair_cache[token_address] = None
            return None
        sol_pairs.sort(key=lambda p: float(p.get("liquidity", {}).get("usd", 0) or 0),
                       reverse=True)
        addr = sol_pairs[0].get("pairAddress")
        _pair_cache[token_address] = addr
        return addr
    except Exception:
        _pair_cache[token_address] = None
        return None


def resolve_pairs_batch(tokens: list[str]):
    need = [t for t in tokens if t not in _pair_cache]
    if not need:
        return
    print(f"  Resolving {len(need)} pair addresses via DexScreener...")
    for i, token in enumerate(need):
        if (i + 1) % 20 == 0:
            print(f"    [{i+1}/{len(need)}] resolved...", flush=True)
            _save_pair_cache(_pair_cache)
        resolve_pair_address(token)
        time_mod.sleep(0.35)
    _save_pair_cache(_pair_cache)
    resolved = sum(1 for t in need if _pair_cache.get(t))
    print(f"  Resolved {resolved}/{len(need)} pair addresses")


def _cache_key(pool: str, start_ts: int, window: int = MAX_WINDOW_MIN) -> str:
    h = hashlib.md5(f"{pool}_{start_ts}_{window}".encode()).hexdigest()[:12]
    return f"{pool[:12]}_{start_ts}_{window}_{h}.json"


def _load_cache(pool: str, start_ts: int, window: int = MAX_WINDOW_MIN) -> list[dict] | None:
    path = CACHE_DIR / _cache_key(pool, start_ts, window)
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, list) and len(data) > 0:
                return data
        except (json.JSONDecodeError, KeyError):
            pass
    return None


def _load_legacy_cache(pool: str, start_ts: int) -> list[dict] | None:
    """Try old cache format (no window_min in key)."""
    h = hashlib.md5(f"{pool}_{start_ts}".encode()).hexdigest()[:12]
    path = CACHE_DIR / f"{pool[:12]}_{start_ts}_{h}.json"
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, list) and len(data) > 0:
                return data
        except (json.JSONDecodeError, KeyError):
            pass
    return None


def _save_cache(pool: str, start_ts: int, candles: list[dict], window: int = MAX_WINDOW_MIN):
    path = CACHE_DIR / _cache_key(pool, start_ts, window)
    path.write_text(json.dumps(candles))


def fetch_ohlcv_dexpaprika(pool: str, start_ts: int, end_ts: int) -> list[dict] | None:
    start_iso = datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_iso = datetime.fromtimestamp(end_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    url = f"https://api.dexpaprika.com/networks/solana/pools/{pool}/ohlcv"
    try:
        r = requests.get(url, params={"start": start_iso, "end": end_iso, "interval": "15m"},
                         timeout=20)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        candles = []
        for c in data:
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
        return candles if candles else None
    except Exception as e:
        print(f"  DexPaprika error for {pool[:12]}: {e}")
        return None


def fetch_ohlcv_gecko(pool: str, start_ts: int, end_ts: int) -> list[dict] | None:
    url = f"https://api.geckoterminal.com/api/v2/networks/solana/pools/{pool}/ohlcv/minute"
    try:
        r = requests.get(url, params={"aggregate": "1", "limit": "1000",
                         "before_timestamp": str(end_ts + 60)}, timeout=20)
        if r.status_code == 404:
            return None
        if r.status_code == 429:
            time_mod.sleep(10)
            r = requests.get(url, params={"aggregate": "1", "limit": "1000",
                             "before_timestamp": str(end_ts + 60)}, timeout=20)
            if r.status_code == 429:
                return None
        r.raise_for_status()
        ohlcv_list = r.json().get("data", {}).get("attributes", {}).get("ohlcv_list", [])
        if not ohlcv_list:
            return None
        candles = []
        for c in ohlcv_list:
            ts = int(c[0])
            if ts < start_ts - 60 or ts > end_ts + 120:
                continue
            candles.append({"timestamp": ts, "open": float(c[1]), "high": float(c[2]),
                            "low": float(c[3]), "close": float(c[4]),
                            "volume": float(c[5]) if len(c) > 5 else 0})
        candles.sort(key=lambda x: x["timestamp"])
        return candles if candles else None
    except Exception as e:
        print(f"  GeckoTerminal error for {pool[:12]}: {e}")
        return None


BIRDEYE_API_KEY = os.environ.get("BIRDEYE_API_KEY")


def fetch_ohlcv_birdeye(token_mint: str, start_ts: int, end_ts: int) -> list[dict] | None:
    """Fetch OHLCV from Birdeye using token mint (not pool address). 30 CU per call."""
    if not BIRDEYE_API_KEY:
        return None
    try:
        r = requests.get(
            "https://public-api.birdeye.so/defi/ohlcv",
            params={"address": token_mint, "type": "15m",
                    "time_from": start_ts, "time_to": end_ts},
            headers={"X-API-KEY": BIRDEYE_API_KEY, "x-chain": "solana"},
            timeout=15,
        )
        if r.status_code == 429:
            time_mod.sleep(3)
            r = requests.get(
                "https://public-api.birdeye.so/defi/ohlcv",
                params={"address": token_mint, "type": "15m",
                        "time_from": start_ts, "time_to": end_ts},
                headers={"X-API-KEY": BIRDEYE_API_KEY, "x-chain": "solana"},
                timeout=15,
            )
            if r.status_code == 429:
                return None
        if r.status_code != 200:
            return None
        items = r.json().get("data", {}).get("items", [])
        if not items:
            return None
        candles = []
        for c in items:
            h, l, o, cl, v = c.get("h", 0), c.get("l", 0), c.get("o", 0), c.get("c", 0), c.get("v", 0)
            ts = int(c.get("unixTime", 0))
            if h > 0 and l > 0 and ts >= start_ts - 60:
                candles.append({"timestamp": ts, "open": float(o), "high": float(h),
                                "low": float(l), "close": float(cl), "volume": float(v)})
        candles.sort(key=lambda x: x["timestamp"])
        return candles if candles else None
    except Exception as e:
        print(f"  Birdeye error for {token_mint[:12]}: {e}")
        return None


def fetch_candles_for_trade(trade: dict, cache_only: bool = False) -> tuple[list[dict] | None, bool]:
    pool = trade.get("pair_address")
    token = trade.get("token_address")
    dt = datetime.fromisoformat(trade["created_at"].replace("Z", "+00:00"))
    start_ts = int(dt.timestamp())
    end_ts = start_ts + MAX_WINDOW_MIN * 60

    if not pool:
        pool = _pair_cache.get(token) or (resolve_pair_address(token) if not cache_only else None)
        if not pool:
            # v114: Even without pool, try Birdeye (uses token mint)
            if not cache_only and token:
                candles = fetch_ohlcv_birdeye(token, start_ts, end_ts)
                if candles and len(candles) >= 3:
                    return candles, True
            return None, False

    cached = _load_cache(pool, start_ts, MAX_WINDOW_MIN)
    if cached is not None:
        return (cached if cached else None), False

    legacy = _load_legacy_cache(pool, start_ts)
    if legacy is not None:
        if cache_only:
            return (legacy if legacy else None), False

    if cache_only:
        return None, False

    candles = fetch_ohlcv_dexpaprika(pool, start_ts, end_ts)
    if candles and len(candles) >= 3:
        _save_cache(pool, start_ts, candles, MAX_WINDOW_MIN)
        return candles, True

    time_mod.sleep(2.5)
    candles = fetch_ohlcv_gecko(pool, start_ts, end_ts)
    if candles and len(candles) >= 5:
        _save_cache(pool, start_ts, candles, MAX_WINDOW_MIN)
        return candles, True

    # v114: 3rd fallback — Birdeye (uses token mint, works for deindexed tokens)
    time_mod.sleep(0.5)
    candles = fetch_ohlcv_birdeye(token, start_ts, end_ts)
    if candles and len(candles) >= 3:
        _save_cache(pool, start_ts, candles, MAX_WINDOW_MIN)
        return candles, True

    _save_cache(pool, start_ts, [], MAX_WINDOW_MIN)
    return None, True


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pnl_list: list[float], n_days: int) -> dict:
    n = len(pnl_list)
    if n == 0:
        return {}
    wins = sum(1 for p in pnl_list if p > 0)
    avg_pnl = statistics.mean(pnl_list)
    med_pnl = statistics.median(pnl_list)
    std_pnl = statistics.stdev(pnl_list) if n > 1 else 1.0

    tpd = n / max(n_days, 1)
    sharpe = (avg_pnl / std_pnl * math.sqrt(tpd)) if std_pnl > 0 else 0.0

    avg_win = statistics.mean([p for p in pnl_list if p > 0]) if wins > 0 else 0
    losses = [p for p in pnl_list if p <= 0]
    avg_loss = abs(statistics.mean(losses)) if losses else 1.0
    wr = wins / n
    rr = avg_win / avg_loss if avg_loss > 0 else 0
    kelly = wr - (1 - wr) / rr if rr > 0 else 0

    return {
        "n_trades": n, "wr_pct": wr * 100, "avg_pnl_pct": avg_pnl * 100,
        "median_pnl_pct": med_pnl * 100, "sharpe": sharpe, "kelly": kelly,
    }


FLAT_POS_SIZE = 0  # 0 = use Kelly, >0 = fixed $ per trade (set by --flat-sizing)
USE_UNIFIED_SIM = True  # v125: use production _evaluate_trade_exit() for OHLCV sim


def simulate_bankroll(trade_results: list[dict]) -> dict:
    bankroll = START_BANKROLL
    peak_bankroll = bankroll
    max_dd = 0.0
    n_trades = 0
    seen_tokens: dict[str, str] = {}

    for t in trade_results:
        pnl_pct = t["pnl_pct"]
        token = t["token_address"]
        day = t["created_at"][:10]

        last_day = seen_tokens.get(token)
        if last_day:
            try:
                if (datetime.strptime(day, "%Y-%m-%d") -
                        datetime.strptime(last_day, "%Y-%m-%d")).days < 1:
                    continue
            except ValueError:
                pass
        seen_tokens[token] = day

        if FLAT_POS_SIZE > 0:
            pos_size = min(FLAT_POS_SIZE, bankroll * 0.5)  # never risk > 50% of bankroll
        else:
            pos_size = min(bankroll * KELLY_FRAC, MAX_POS)
        if pos_size < 1.0:
            continue

        bankroll += pos_size * pnl_pct
        n_trades += 1

        if bankroll > peak_bankroll:
            peak_bankroll = bankroll
        dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
        if dd > max_dd:
            max_dd = dd

    return {
        "final_bankroll": bankroll, "max_dd_pct": max_dd * 100,
        "n_trades": n_trades, "total_pnl_usd": bankroll - START_BANKROLL,
    }


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def monte_carlo(pnl_list: list[float], n_sims: int = 1000, n_trades: int = 200) -> dict:
    if len(pnl_list) < MIN_TRADES:
        return {}

    finals = []
    max_dds = []
    ruins = 0

    for _ in range(n_sims):
        b = START_BANKROLL
        peak = b
        max_dd = 0.0
        for _ in range(n_trades):
            pnl = random.choice(pnl_list)
            pos = min(b * KELLY_FRAC, MAX_POS)
            if pos < 1.0:
                ruins += 1
                break
            b += pos * pnl
            if b > peak:
                peak = b
            dd = (peak - b) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        finals.append(b)
        max_dds.append(max_dd)

    finals.sort()
    return {
        "median": finals[len(finals) // 2],
        "p5": finals[int(len(finals) * 0.05)],
        "p25": finals[int(len(finals) * 0.25)],
        "p75": finals[int(len(finals) * 0.75)],
        "p95": finals[int(len(finals) * 0.95)],
        "ror_pct": ruins / n_sims * 100,
        "mean_max_dd": statistics.mean(max_dds) * 100,
    }


# ---------------------------------------------------------------------------
# Runner capture analysis
# ---------------------------------------------------------------------------

def runner_analysis(trade_entries: list[dict], candle_store: dict,
                    grid: list[dict], all_results: dict) -> list[dict]:
    """Compute capture % on x2+ runners for each strategy."""
    # Find runners
    runners = []
    for te in trade_entries:
        key = te["candles_key"]
        candles = candle_store.get(key)
        if not candles:
            continue
        max_price = max(c["high"] for c in candles)
        max_gain = max_price / te["entry_price"] - 1
        if max_gain >= 1.0:
            runners.append(te)

    if not runners:
        return []

    runner_keys = {r["candles_key"] for r in runners}

    results = []
    for cfg in grid:
        name = cfg["name"]
        if name not in all_results:
            continue
        strat_results = all_results[name]
        runner_pnls = []
        runner_maxes = []
        for te in runners:
            key = te["candles_key"]
            if key in strat_results:
                runner_pnls.append(strat_results[key]["pnl_pct"])
                candles = candle_store[key]
                max_gain = max(c["high"] for c in candles) / te["entry_price"] - 1
                runner_maxes.append(max_gain)

        if not runner_pnls:
            continue

        avg_capture = statistics.mean(runner_pnls)
        avg_max = statistics.mean(runner_maxes)
        capture_rate = avg_capture / avg_max * 100 if avg_max > 0 else 0

        results.append({
            "name": name, "type": cfg["type"],
            "n_runners": len(runner_pnls),
            "avg_capture_pct": avg_capture * 100,
            "avg_max_pct": avg_max * 100,
            "capture_rate": capture_rate,
            "best_capture_pct": max(runner_pnls) * 100,
        })

    results.sort(key=lambda x: -x["capture_rate"])
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Dual-Wallet Simulation
# ---------------------------------------------------------------------------

CANDLE_INTERVAL_MIN = 15  # DexPaprika/Birdeye default


def get_price_at_delta(candles: list[dict], base_ts: int, delta_min: float) -> float | None:
    """Get interpolated price at base_ts + delta_min from OHLCV candles."""
    if delta_min == 0:
        return None  # caller should use entry_price directly
    target_ts = base_ts + delta_min * 60
    for c in candles:
        candle_end = c["timestamp"] + CANDLE_INTERVAL_MIN * 60
        if c["timestamp"] <= target_ts < candle_end:
            frac = (target_ts - c["timestamp"]) / (CANDLE_INTERVAL_MIN * 60)
            return c["open"] + (c["close"] - c["open"]) * frac
    return None


def simulate_dual_wallet(sim_candles: list[dict], raw_candles: list[dict],
                         entry_price: float, cfg: dict,
                         context: dict | None, delta_min: float,
                         position_usd: float) -> dict | None:
    """
    Run simulate() for two independent wallets:
      Wallet A: buys at entry_price (t=0)
      Wallet B: buys at t+delta_min (OHLCV price) or t=0 if delta=0
    Both use liquidity-aware slippage.

    sim_candles: candles used for simulation (may be resampled for --realistic)
    raw_candles: original 15min OHLCV candles (used for price lookup at delta)
    """
    liq = (context or {}).get("liq", 0)
    # v14e.28: chain dispatch — EVM trades get gas-folded slip via sim_engines
    chain = (context or {}).get("chain") or "solana"
    base_ts = raw_candles[0]["timestamp"] if raw_candles else 0

    simultaneous = (delta_min == 0)
    n_sim = 2 if simultaneous else 1

    # Wallet A: always buys at signal time
    wa_slip = compute_buy_slippage(position_usd, liq, n_sim, chain=chain)
    wa_entry = entry_price * (1 + wa_slip)
    wa_result = simulate(sim_candles, wa_entry, cfg, context)

    # Wallet B
    if simultaneous:
        wb_entry = wa_entry
        wb_sim_candles = sim_candles
    else:
        # Use raw candles for price interpolation (not resampled)
        wb_raw_price = get_price_at_delta(raw_candles, base_ts, delta_min)
        if wb_raw_price is None:
            return None
        wb_slip = compute_buy_slippage(position_usd, liq, 1, chain=chain)
        wb_entry = wb_raw_price * (1 + wb_slip)
        # Trim sim candles from Wallet B's entry time
        target_ts = base_ts + delta_min * 60
        wb_sim_candles = [c for c in sim_candles if c["timestamp"] >= target_ts]
        if len(wb_sim_candles) < 2:
            return None

    wb_result = simulate(wb_sim_candles, wb_entry, cfg, context)

    # Single wallet baseline (original flat slippage for fair comparison)
    single_result = simulate(sim_candles, entry_price, cfg, context)

    return {
        "wa": wa_result,
        "wb": wb_result,
        "single": single_result,
        "combined_pnl_pct": (wa_result["pnl_pct"] + wb_result["pnl_pct"]) / 2,
        "combined_total_pnl_pct": wa_result["pnl_pct"] + wb_result["pnl_pct"],
        "delta_min": delta_min,
        "wa_slip": wa_slip,
        "wb_slip": compute_buy_slippage(position_usd, liq, 1, chain=chain) if not simultaneous else wa_slip,
    }


def simulate_bankroll_dual(trade_results_a: list[dict], trade_results_b: list[dict]) -> dict:
    """Two independent bankrolls, each starting at START_BANKROLL."""
    br_a = simulate_bankroll(trade_results_a)
    br_b = simulate_bankroll(trade_results_b)
    return {
        "wa_final": br_a["final_bankroll"],
        "wb_final": br_b["final_bankroll"],
        "combined_final": br_a["final_bankroll"] + br_b["final_bankroll"],
        "single_2x": br_a["final_bankroll"],  # what 2x single bankroll would be
    }


# ---------------------------------------------------------------------------
# v124: FROM-TICKS — tick-level replay simulation
# ---------------------------------------------------------------------------

# Tick data start date (price_ticks table deployed v118 Apr 6)
TICK_DATA_START = "2026-04-06"


def _fetch_tick_trades(since: str, include_shadows: bool = False) -> list[dict]:
    """Fetch closed RT paper trades that have tick coverage.
    include_shadows=True keeps is_shadow rows (for grid sweeps needing volume)."""
    params = [
        ("select", "id,token_address,symbol,entry_price,sl_price,tp_price,"
                   "strategy,horizon_minutes,tranche_label,tranche_pct,"
                   "position_usd,status,created_at,exit_at,pnl_pct,pnl_usd,"
                   "exit_minutes,high_price_seen,kol_group,rt_liquidity_usd,"
                   "dex_spot_price_at_entry,source,exit_price,entry_mcap,"
                   "snapshot_id,entry_score,rt_token_age_hours,rt_is_pump_fun,is_shadow,chain"),
        ("status", "in.(trail_stop,sl_hit,timeout,tp_hit)"),
        ("source", "eq.rt"),
        ("created_at", f"gte.{since}T00:00:00Z"),
        ("order", "created_at.asc"),
    ] + _chain_params()
    if not include_shadows:
        params.insert(3, ("is_shadow", "eq.false"))
    trades = sb_get("paper_trades", params)
    label = "RT+shadow" if include_shadows else "RT (non-shadow)"
    print(f"Fetched {len(trades)} closed {label} trades (chain={_SIM_CHAIN}) since {since}")
    return trades


def _fetch_ticks_for_tokens(token_ranges: dict[str, tuple]) -> dict[str, list[dict]]:
    """Fetch price ticks for multiple tokens in bulk.
    token_ranges: {token_address: (start_iso, end_iso)}
    Returns: {token_address: [sorted ticks]}"""
    ticks_by_token: dict[str, list[dict]] = {}

    for addr, (t_start, t_end) in token_ranges.items():
        params = [
            ("select", "price_usd,fetched_at,source,volume_usd,liquidity_usd,chain"),
            ("token_address", f"eq.{addr}"),
            ("fetched_at", f"gte.{t_start}"),
            ("fetched_at", f"lte.{t_end}"),
            ("order", "fetched_at.asc"),
        ] + _chain_params()
        rows = sb_get("price_ticks", params)
        if rows:
            ticks_by_token[addr] = rows

    total_ticks = sum(len(v) for v in ticks_by_token.values())
    print(f"Fetched {total_ticks} ticks for {len(ticks_by_token)} tokens")
    return ticks_by_token


# v131 Gap #2: Subsample tick replay to match paper polling cadence (30s).
# price_ticks table logs at ~15s (live) / ~30s (fast/full). Replaying every
# tick means sim sees 2× the exit opportunities paper would catch → DTRAIL
# and tight trails are over-evaluated. Subsample to ≥30s spacing to mirror
# paper's check_paper_trades_fast cadence exactly.
PAPER_POLL_INTERVAL_SEC = 30


def _subsample_ticks(ticks: list[dict], interval_sec: int = PAPER_POLL_INTERVAL_SEC) -> list[dict]:
    """Keep first tick + next tick ≥ interval_sec after the last kept one."""
    if not ticks or interval_sec <= 0:
        return ticks
    kept = []
    last_ts = None
    for t in ticks:
        try:
            ts = datetime.fromisoformat(t["fetched_at"].replace("Z", "+00:00"))
        except Exception:
            continue
        if last_ts is None or (ts - last_ts).total_seconds() >= interval_sec:
            kept.append(t)
            last_ts = ts
    return kept


def _filter_ticks_by_source(ticks: list[dict], price_source: str,
                            subsample_sec: int = PAPER_POLL_INTERVAL_SEC) -> list[dict]:
    """Filter/merge ticks by source preference, then subsample to paper cadence."""
    if price_source == "jupiter":
        jup = [t for t in ticks if t["source"] == "jupiter"]
        if jup:
            return _subsample_ticks(jup, subsample_sec)
        return _subsample_ticks(
            [t for t in ticks if t["source"] in ("fast", "full", "live")],
            subsample_sec,
        )

    elif price_source == "dexscreener":
        return _subsample_ticks(
            [t for t in ticks if t["source"] in ("fast", "full", "live")],
            subsample_sec,
        )

    else:  # "both" — merge, prefer Jupiter at each timestamp
        dex_ticks = {t["fetched_at"]: t for t in ticks if t["source"] in ("fast", "full", "live")}
        jup_ticks = {t["fetched_at"]: t for t in ticks if t["source"] == "jupiter"}
        merged = {**dex_ticks, **jup_ticks}
        merged_sorted = sorted(merged.values(), key=lambda t: t["fetched_at"])
        return _subsample_ticks(merged_sorted, subsample_sec)


def _build_fake_trade(trade: dict, strategy_override: str = None,
                      sl_mult: float = None, horizon_min: int = None,
                      sim_live_entry: bool = False) -> dict:
    """Build a trade dict compatible with _evaluate_trade_exit().

    sim_live_entry: if True, simulate Jupiter fill slippage on entry price.
    Live fills are typically 3-5% worse than Jupiter quote on memecoins.
    This makes the sim match live PnL more accurately."""
    strategy = strategy_override or trade["strategy"]
    entry_price = float(trade["entry_price"])

    if sim_live_entry:
        # v14e.24: use the production constant (strategies.BUY_SLIPPAGE_BPS = 225)
        # for a single source of truth across paper / sim / shadow / live.
        # Previously a piecewise liq curve (2-10%) — empirical fit R²=5% showed
        # liq doesn't predict slip; the constant median is the honest baseline.
        from strategies import BUY_SLIPPAGE_BPS as _PT_BUY_SLIP_BPS
        slip_pct = _PT_BUY_SLIP_BPS / 10_000
        fill_price = entry_price * (1 + slip_pct)  # fill is HIGHER (you pay more)
        # dex_spot = market price at entry (for trail activation reference)
        dex_spot = entry_price
        entry_price = fill_price
    else:
        dex_spot = float(trade.get("dex_spot_price_at_entry") or 0)

    # Compute SL price from override or original
    if sl_mult is not None:
        computed_sl = entry_price * sl_mult
    else:
        computed_sl = float(trade["sl_price"])

    # Compute TP price — DTRAIL/TRAIL strategies have no TP (None)
    tp_price = float(trade["tp_price"]) if trade.get("tp_price") else None

    return {
        "id": trade["id"],
        "entry_price": entry_price,
        "sl_price": computed_sl,
        "tp_price": tp_price,
        "position_usd": float(trade.get("position_usd") or 10.0),
        "strategy": strategy,
        "tranche_label": trade.get("tranche_label", "main"),
        "horizon_minutes": horizon_min or trade.get("horizon_minutes", 120),
        "created_at": trade["created_at"],
        "high_price_seen": dex_spot if dex_spot > 0 else entry_price,
        "rt_liquidity_usd": trade.get("rt_liquidity_usd"),
        "dex_spot_price_at_entry": dex_spot,
    }


def _load_live_strategy_overrides() -> dict:
    """v132: Pull production strategy_overrides from scoring_config.rt_trade_config.
    Returns {strategy_name: {polling_sec, price_source, ema_window}} or {} on failure.
    """
    try:
        rows = sb_get("scoring_config", [("id", "eq.1"), ("select", "rt_trade_config")])
        if rows and rows[0].get("rt_trade_config"):
            cfg = rows[0]["rt_trade_config"]
            if isinstance(cfg, str):
                import json
                cfg = json.loads(cfg)
            return cfg.get("strategy_overrides", {}) or {}
    except Exception as e:
        print(f"[warn] failed to load live strategy_overrides: {e}")
    return {}


# v137: realistic polling — replaces the v131 "next-tick-after-gap" subsample
# with deterministic 30s grid + cache look-back. Mimics paper_trader's actual
# unified_check_loop (30s) + _should_poll_trade(polling_sec) + _jupiter_prices_cache
# semantics.
#
# Tried v137.1 (filter jupiter to paper-logged only, drop live_trader ticks): MAE
# got WORSE (17% -> 30%) because sparser ticks meant look-back picked up stale
# prices between paper's 60s-throttled logs, whereas live_trader's 15s ticks were
# actually giving the sim better visibility into real price action (real paper
# fetched every 30s via Jupiter API — cache was fresh — even though it only
# logged every 60s). Including live_trader ticks is the better approximation.
#
# Residual ~+4pp bias on trail_stop PnL is STRUCTURAL and comes from: price_ticks
# undersamples what real paper's cache actually saw (60s throttle pre-v137 vs
# 30s real fetch cadence). Throttle is now 30s post-v137 deploy, so NEW data will
# converge. For DTRAIL decisions on historical data, use --from-trades.
LOOP_SEC = 30  # paper_trader unified_check_loop interval


def _latest_tick_at_or_before(sorted_ticks: list[dict], t_iso: str) -> tuple[float | None, str | None]:
    """Look up cache-style: latest tick whose fetched_at <= t_iso.
    sorted_ticks must be pre-sorted ascending by fetched_at."""
    last = (None, None)
    for tk in sorted_ticks:
        if tk["fetched_at"] <= t_iso:
            p = float(tk["price_usd"])
            if p > 0:
                last = (p, tk["fetched_at"])
        else:
            break
    return last


def _replay_trade_orchestrated(fake_trade: dict, ds_ticks: list[dict],
                               jup_ticks: list[dict], orchestration: str,
                               poll_sec: int = 0, ema_window: int = 3) -> dict | None:
    """v132: orchestrated tick replay. Supports decision vs exec price separation.
    v137: realistic polling cadence — deterministic 30s grid + cache look-back
          (was: tick-driven subsample, which over-estimated tight-trail strategies).
    orchestration: jupiter | ds | hybrid | confirm | ema
    """
    from paper_trader import _evaluate_trade_exit, _last_eval_ts

    entry_time = datetime.fromisoformat(fake_trade["created_at"].replace("Z", "+00:00"))
    trade_id = str(fake_trade["id"])
    _last_eval_ts.pop(trade_id, None)

    # Sort streams once for look-back. We keep live_trader-logged jupiter ticks
    # in the stream (v137.1 tested filtering them out: MAE went 17% -> 30%).
    # Real paper_trader's cache was refreshed every 30s (not limited by log
    # throttle), so live_trader's 15s-cadence jupiter logs are actually a
    # decent proxy for what real paper's cache held between its own logs.
    jup_sorted = sorted(jup_ticks, key=lambda t: t["fetched_at"]) if jup_ticks else []
    ds_sorted = sorted(ds_ticks, key=lambda t: t["fetched_at"]) if ds_ticks else []

    if not jup_sorted and not ds_sorted:
        return None

    horizon_min = int(fake_trade.get("horizon_minutes", 120) or 120)
    horizon_sec = horizon_min * 60
    effective_poll_sec = poll_sec if poll_sec > 0 else LOOP_SEC

    # v137: build deterministic poll schedule on a 30s grid from entry, with
    # _should_poll_trade(poll_sec) filter (matches paper_trader exactly).
    poll_offsets = []
    last_check = -10**9
    t = LOOP_SEC
    while t <= horizon_sec:
        if (t - last_check) >= effective_poll_sec:
            poll_offsets.append(t)
            last_check = t
        t += LOOP_SEC

    sell_slip = 1 - 10 / 10_000  # 10bps base; _evaluate_trade_exit applies dynamic on top
    ema_val = None
    alpha = 2 / (ema_window + 1)
    last_dec_ts_iso = None
    last_exec_p = None

    for offset_sec in poll_offsets:
        poll_time = entry_time + timedelta(seconds=offset_sec)
        poll_time_iso = poll_time.isoformat().replace("+00:00", "Z")

        jp, jp_ts = _latest_tick_at_or_before(jup_sorted, poll_time_iso)
        ds, ds_ts = _latest_tick_at_or_before(ds_sorted, poll_time_iso)

        # Per-orchestration decision/exec selection
        if orchestration == "ds":
            decision_p = ds if ds is not None else jp
            exec_p = jp if jp is not None else ds
        elif orchestration == "hybrid":
            decision_p = ds if ds is not None else jp
            exec_p = jp if jp is not None else ds
        elif orchestration == "confirm":
            if jp is not None and ds is not None:
                decision_p = (jp + ds) / 2
            else:
                decision_p = jp if jp is not None else ds
            exec_p = jp if jp is not None else ds
        elif orchestration == "ema":
            if jp is None:
                continue
            ema_val = jp if ema_val is None else (alpha * jp + (1 - alpha) * ema_val)
            decision_p = ema_val
            exec_p = jp
        # v142 C — jp_sampled_Ns : freeze decision price between bar boundaries.
        # Simple sampling, NOT a faithful OHLCV port (no intra-bar wicks emitted).
        elif orchestration in ("jp_sampled_60s", "jp_sampled_180s"):
            if jp is None:
                continue
            bar_sec = 60 if orchestration == "jp_sampled_60s" else 180
            if not hasattr(_replay_trade_orchestrated, "_sampled_state"):
                _replay_trade_orchestrated._sampled_state = {}
            key = trade_id
            st = _replay_trade_orchestrated._sampled_state.setdefault(key, {"ts": 0, "close": jp})
            cur_bar = (offset_sec // bar_sec) * bar_sec
            if cur_bar > st["ts"]:
                st["close"] = jp
                st["ts"] = cur_bar
            decision_p = st["close"]
            exec_p = jp
        # v142 D — ohlc_burst_60s : at each bar close, emit [O, L/H, H/L, C]
        # sequence as 4 consecutive evals. Port of sim_engines OHLCV tick
        # synthesis. Best-effort reconstruction from our polled ticks (still
        # misses exchange-level micro-wicks between polls).
        elif orchestration == "ohlc_burst_60s":
            if jp is None:
                continue
            if not hasattr(_replay_trade_orchestrated, "_burst_state"):
                _replay_trade_orchestrated._burst_state = {}
            bst = _replay_trade_orchestrated._burst_state.setdefault(
                trade_id, {"buf": [], "last_emitted_bar": 0})
            # Accumulate tick in bar buffer
            bst["buf"].append((offset_sec, jp))
            # Check if we crossed a 60s boundary since last emit
            cur_bar_start = (offset_sec // 60) * 60
            prev_bar_start = cur_bar_start - 60
            if prev_bar_start > bst["last_emitted_bar"]:
                bar_ticks = [p for t, p in bst["buf"] if prev_bar_start <= t < cur_bar_start]
                if len(bar_ticks) >= 2:
                    o, c = bar_ticks[0], bar_ticks[-1]
                    h, lo = max(bar_ticks), min(bar_ticks)
                    burst = [o, lo, h, c] if c >= o else [o, h, lo, c]
                    bst["last_emitted_bar"] = prev_bar_start
                    # Emit the 4 synthetic prices SEQUENTIALLY, letting eval
                    # trigger on the first match.
                    burst_triggered = False
                    for b_price in burst:
                        ev = _evaluate_trade_exit(fake_trade, b_price, poll_time, sell_slip,
                                                  sell_fee_bps=0, decision_price=b_price)
                        if ev and ev.get("high_price_seen") is not None:
                            new_h = ev["high_price_seen"]
                            if new_h > float(fake_trade.get("high_price_seen") or 0):
                                fake_trade["high_price_seen"] = new_h
                        if ev and "status" in ev and ev["status"]:
                            return {
                                "exit_reason": ev["status"],
                                "exit_price": ev.get("exit_price", 0),
                                "pnl_pct": ev.get("pnl_pct", 0),
                                "pnl_usd": ev.get("pnl_usd", 0),
                                "exit_minutes": ev.get("exit_minutes", 0),
                                "high_price_seen": fake_trade.get("high_price_seen"),
                            }
                # Purge old ticks to keep buffer small
                bst["buf"] = [x for x in bst["buf"] if x[0] >= prev_bar_start]
            decision_p = jp
            exec_p = jp
        # v142 C — Twin-source confirmation. Both jp and ds must agree
        # before letting SL/TP trigger. Single-source breach suppressed.
        elif orchestration == "twin_confirm":
            if jp is None:
                continue
            sl_price = float(fake_trade.get("sl_price") or 0)
            tp_price = float(fake_trade.get("tp_price") or 0) or None
            if ds is None:
                decision_p = jp
            else:
                dec = jp
                if sl_price:
                    jp_brk = jp <= sl_price
                    ds_brk = ds <= sl_price
                    if jp_brk != ds_brk:
                        dec = max(jp, ds)  # suppress single-source SL breach
                if tp_price:
                    jp_brk = jp >= tp_price
                    ds_brk = ds >= tp_price
                    if jp_brk != ds_brk:
                        dec = min(jp, ds)  # suppress single-source TP breach
                decision_p = dec
            exec_p = jp
        else:  # "jupiter" default
            decision_p = jp
            exec_p = jp if jp is not None else ds

        if decision_p is None or exec_p is None:
            continue
        last_exec_p = exec_p

        ev = _evaluate_trade_exit(fake_trade, exec_p, poll_time, sell_slip,
                                  sell_fee_bps=0, decision_price=decision_p)
        if ev is None:
            continue
        if ev.get("high_price_seen") is not None:
            new_high = ev["high_price_seen"]
            if new_high > float(fake_trade.get("high_price_seen") or 0):
                fake_trade["high_price_seen"] = new_high
        if "status" in ev and ev["status"]:
            return {
                "exit_reason": ev["status"],
                "exit_price": ev.get("exit_price", 0),
                "pnl_pct": ev.get("pnl_pct", 0),
                "pnl_usd": ev.get("pnl_usd", 0),
                "exit_minutes": ev.get("exit_minutes", 0),
                "high_price_seen": fake_trade.get("high_price_seen"),
            }

    # End of horizon — timeout at last seen exec price
    if last_exec_p is None:
        return None
    entry_price = float(fake_trade["entry_price"])
    pnl_pct = round((last_exec_p / entry_price) - 1, 4) if entry_price > 0 else 0
    return {
        "exit_reason": "timeout_eod",
        "exit_price": last_exec_p,
        "pnl_pct": pnl_pct,
        "pnl_usd": round(float(fake_trade.get("position_usd") or 10.0) * pnl_pct, 2),
        "exit_minutes": horizon_min,
        "high_price_seen": fake_trade.get("high_price_seen"),
    }


# ---------------------------------------------------------------------------
# v138: Replay from persisted eval_history (perfect alignment, 0% bias)
# ---------------------------------------------------------------------------
def _replay_from_eval_history(fake_trade: dict, eval_history: list[dict]) -> dict | None:
    """v138 B: replay using the EXACT (decision, exec) pairs paper_trader logged.

    Each eval_history entry: {"t": iso, "d": decision_p, "e": exec_p, "h": high_at_poll}.
    Re-runs them through _evaluate_trade_exit with no reconstruction guesswork.
    Result is mathematically identical to what real paper saw at trade close,
    modulo strategy override (allows what-if "what would strat X have done on
    the same price stream?")
    """
    from paper_trader import _evaluate_trade_exit, _last_eval_ts
    if not eval_history:
        return None
    trade_id = str(fake_trade["id"])
    _last_eval_ts.pop(trade_id, None)

    sell_slip = 1 - 10 / 10_000  # base bps; _evaluate adds dynamic on top
    last_exec = None
    for poll in eval_history:
        try:
            t_iso = poll["t"]
            dec_p = poll.get("d")
            exec_p = poll.get("e")
            if dec_p is None or exec_p is None:
                continue
            t = datetime.fromisoformat(t_iso.replace("Z", "+00:00"))
        except Exception:
            continue
        last_exec = exec_p
        ev = _evaluate_trade_exit(fake_trade, exec_p, t, sell_slip,
                                  sell_fee_bps=0, decision_price=dec_p)
        if ev is None:
            continue
        if ev.get("high_price_seen") is not None:
            new_high = ev["high_price_seen"]
            if new_high > float(fake_trade.get("high_price_seen") or 0):
                fake_trade["high_price_seen"] = new_high
        if "status" in ev and ev["status"]:
            return {
                "exit_reason": ev["status"],
                "exit_price": ev.get("exit_price", 0),
                "pnl_pct": ev.get("pnl_pct", 0),
                "pnl_usd": ev.get("pnl_usd", 0),
                "exit_minutes": ev.get("exit_minutes", 0),
                "high_price_seen": fake_trade.get("high_price_seen"),
            }
    if last_exec is None:
        return None
    entry_price = float(fake_trade["entry_price"])
    pnl_pct = round((last_exec / entry_price) - 1, 4) if entry_price > 0 else 0
    return {
        "exit_reason": "timeout_eod", "exit_price": last_exec,
        "pnl_pct": pnl_pct,
        "pnl_usd": round(float(fake_trade.get("position_usd") or 10.0) * pnl_pct, 2),
        "exit_minutes": int(fake_trade.get("horizon_minutes", 120) or 120),
        "high_price_seen": fake_trade.get("high_price_seen"),
    }


# ---------------------------------------------------------------------------
# v138 D: cache_snapshots loader — alternative to price_ticks reconstruction
# ---------------------------------------------------------------------------
def _fetch_cache_snapshots(token_addr: str, t_start: str, t_end: str
                           ) -> list[tuple[datetime, float]]:
    """Return [(snapshot_at, jp_price)] for token_addr in window from cache_snapshots.
    Snapshot rows store the FULL paper_trader cache state at each loop tick;
    extracting one token gives the price the cache held at that moment.
    """
    rows = sb_get("cache_snapshots", [
        ("select", "snapshot_at,jp_prices"),
        ("snapshot_at", f"gte.{t_start}"),
        ("snapshot_at", f"lte.{t_end}"),
        ("order", "snapshot_at.asc"),
    ])
    out = []
    for r in rows:
        jp = r.get("jp_prices") or {}
        if isinstance(jp, str):
            import json
            jp = json.loads(jp)
        p = jp.get(token_addr)
        if p is None or float(p) <= 0:
            continue
        ts = datetime.fromisoformat(r["snapshot_at"].replace("Z", "+00:00"))
        out.append((ts, float(p)))
    return out


def _replay_trade_on_ticks(fake_trade: dict, ticks: list[dict],
                           disable_lazy: bool = False) -> dict | None:
    """Replay one trade through price ticks using production exit logic.
    Returns sim result dict or None if no ticks."""
    from paper_trader import _evaluate_trade_exit, _last_eval_ts, LAZY_STRATEGIES

    if not ticks:
        return None

    # Reset LAZY state for this trade
    trade_id = str(fake_trade["id"])
    _last_eval_ts.pop(trade_id, None)

    # Temporarily disable LAZY for grid search
    saved_lazy = None
    if disable_lazy:
        saved_lazy = set(LAZY_STRATEGIES)
        LAZY_STRATEGIES.clear()

    sell_slip = 1 - 10 / 10_000  # 10 bps base (Jupiter Ultra)
    entry_time = datetime.fromisoformat(
        fake_trade["created_at"].replace("Z", "+00:00"))

    try:
        for tick in ticks:
            tick_time = datetime.fromisoformat(
                tick["fetched_at"].replace("Z", "+00:00"))

            # Skip ticks before trade entry
            if tick_time < entry_time:
                continue

            tick_price = float(tick["price_usd"])
            if tick_price <= 0:
                continue

            ev = _evaluate_trade_exit(fake_trade, tick_price, tick_time, sell_slip)

            if ev is None:
                continue

            # Always update high_price_seen
            if ev.get("high_price_seen") is not None:
                new_high = ev["high_price_seen"]
                if new_high > float(fake_trade.get("high_price_seen") or 0):
                    fake_trade["high_price_seen"] = new_high

            # Exit triggered
            if "status" in ev and ev["status"] not in (None,):
                return {
                    "exit_reason": ev["status"],
                    "exit_price": ev.get("exit_price", 0),
                    "pnl_pct": ev.get("pnl_pct", 0),
                    "pnl_usd": ev.get("pnl_usd", 0),
                    "exit_minutes": ev.get("exit_minutes", 0),
                    "high_price_seen": fake_trade["high_price_seen"],
                }

        # No exit — timeout at last tick
        last_tick = ticks[-1]
        last_time = datetime.fromisoformat(
            last_tick["fetched_at"].replace("Z", "+00:00"))
        last_price = float(last_tick["price_usd"])
        entry_price = float(fake_trade["entry_price"])
        pnl_pct = round((last_price / entry_price) - 1, 4) if entry_price > 0 else 0
        pos_usd = float(fake_trade.get("position_usd") or 10.0)

        return {
            "exit_reason": "timeout_eod",
            "exit_price": last_price,
            "pnl_pct": pnl_pct,
            "pnl_usd": round(pos_usd * pnl_pct, 2),
            "exit_minutes": int((last_time - entry_time).total_seconds() / 60),
            "high_price_seen": fake_trade["high_price_seen"],
        }
    finally:
        # Restore LAZY state
        if saved_lazy is not None:
            LAZY_STRATEGIES.clear()
            LAZY_STRATEGIES.update(saved_lazy)
        _last_eval_ts.pop(trade_id, None)


def _build_tick_grid() -> list[dict]:
    """Build exhaustive strategy grid for tick sim.
    Tests all types supported by _evaluate_trade_exit(): FIXED, DTRAIL, TRAIL, BE, SCALP.
    Each config tested with 8 check interval profiles.
    Includes exit style variants (trail-only, grace-period trail)."""
    configs = []

    # --- Check interval profiles (8 levels) ---
    # (label, fast_sec, fast_window_sec, slow_sec)
    INTERVAL_PROFILES = [
        ("CURRENT", 0, 0, 0),           # check every tick (~30s)
        ("FAST_15", 15, 60, 60),         # 15s fast, 1min window, 1min slow
        ("FAST_30", 30, 120, 120),       # 30s fast, 2min window, 2min slow
        ("LAZY_FAST", 60, 120, 180),     # 1min fast, 2min window, 3min slow
        ("LAZY_MED", 120, 300, 360),     # 2min fast, 5min window, 6min slow
        ("LAZY_STD", 180, 300, 600),     # 3min fast, 5min window, 10min slow
        ("LAZY_SLOW", 300, 600, 900),    # 5min fast, 10min window, 15min slow
        ("LAZY_XSLOW", 600, 900, 1200),  # 10min fast, 15min window, 20min slow
    ]

    # --- 1. DTRAIL: trail x act x SL x horizon ---
    for trail in [3, 5, 8, 10, 15, 20]:
        for act in [5, 10, 15, 20, 30]:
            for sl in [40, 50, 60, 70]:
                for horizon in [60, 120, 180, 240, 360]:
                    configs.append({
                        "name": f"DTRAIL{trail}_ACT{act}_SL{sl}",
                        "type": "DTRAIL",
                        "tp_mult": None,
                        "sl_mult": 1 - sl / 100,
                        "horizon": horizon,
                    })

    # --- 1b. DTRAIL trail-only: SL at -80% (unreachable) → pure trail exit ---
    for trail in [3, 5, 8, 10, 15]:
        for act in [5, 10, 15, 20]:
            for horizon in [120, 240, 360]:
                configs.append({
                    "name": f"DTRAIL{trail}_ACT{act}_TRAILONLY",
                    "type": "DTRAIL",
                    "tp_mult": None,
                    "sl_mult": 0.20,  # -80% SL = effectively trail-only
                    "horizon": horizon,
                })

    # --- 1c. DTRAIL grace period: trail only activates after X min ---
    # Implemented by setting high activation_pct → trail won't trigger early
    # (price must pump > act% before trail even starts watching)
    for trail in [5, 8, 10]:
        for act in [30, 40, 50]:  # high activation = grace period proxy
            for sl in [40, 50]:
                for horizon in [120, 240]:
                    configs.append({
                        "name": f"DTRAIL{trail}_ACT{act}_SL{sl}",
                        "type": "DTRAIL",
                        "tp_mult": None,
                        "sl_mult": 1 - sl / 100,
                        "horizon": horizon,
                    })

    # --- 2. FIXED: TP x SL x horizon ---
    for tp in [30, 40, 50, 60, 80, 100, 150]:
        for sl in [30, 40, 50, 70]:
            for horizon in [60, 120, 240, 360]:
                configs.append({
                    "name": f"TP{tp}_SL{sl}",
                    "type": "FIXED",
                    "tp_mult": 1 + tp / 100,
                    "sl_mult": 1 - sl / 100,
                    "horizon": horizon,
                })

    # --- 3. TRAIL: trail x TP x SL x horizon ---
    for trail in [5, 10, 15, 20]:
        for tp in [50, 70, 100, 150]:
            for sl in [50, 60, 70]:
                for horizon in [120, 240, 360]:
                    configs.append({
                        "name": f"TRAIL{trail}_TP{tp}_SL{sl}",
                        "type": "TRAIL",
                        "tp_mult": 1 + tp / 100,
                        "sl_mult": 1 - sl / 100,
                        "horizon": horizon,
                    })

    # --- 4. BE: activation x TP x SL x horizon ---
    for be_act in [10, 15, 20, 30]:
        for tp in [50, 70, 100]:
            for sl in [50, 60, 70]:
                for horizon in [120, 240, 360]:
                    configs.append({
                        "name": f"BE{be_act}_TP{tp}_SL{sl}",
                        "type": "BE",
                        "tp_mult": 1 + tp / 100,
                        "sl_mult": 1 - sl / 100,
                        "horizon": horizon,
                    })

    # --- 5. SCALP: tight TP x SL x short horizon ---
    for tp in [10, 15, 20, 25, 30]:
        for sl in [10, 15, 20, 30]:
            for horizon in [15, 30, 60]:
                configs.append({
                    "name": f"SCALP_TP{tp}_SL{sl}",
                    "type": "SCALP",
                    "tp_mult": 1 + tp / 100,
                    "sl_mult": 1 - sl / 100,
                    "horizon": horizon,
                })

    # --- 6. DIP strategies — full grid (DIP_BUY tiers 1+2 + DIP_SCALE_OUT) ---
    # Reuses the OHLCV sim's DIP builder so tick sim grades the exact same configs.
    # Both routed through tick-native replayers (see _replay_dip_buy_with_intervals
    # and _replay_dip_scale_out_with_intervals).
    for dip_cfg in _build_dip_buy_grid():
        configs.append({
            **dip_cfg,
            "horizon": dip_cfg["horizon_min"],  # tick grid uses "horizon" key
        })

    # Deduplicate (grace period configs may overlap with base DTRAIL)
    seen = set()
    deduped = []
    for cfg in configs:
        key = (cfg["name"], cfg.get("tp_mult"), cfg["sl_mult"], cfg["horizon"])
        if key not in seen:
            seen.add(key)
            deduped.append(cfg)

    # Cross each config with each interval profile
    full_grid = []
    for cfg in deduped:
        for prof_label, fast_s, fast_w, slow_s in INTERVAL_PROFILES:
            full_grid.append({
                **cfg,
                "interval_profile": prof_label,
                "lazy_fast_sec": fast_s,
                "lazy_fast_window": fast_w,
                "lazy_slow_sec": slow_s,
            })

    return full_grid


def _replay_dip_scale_out_with_intervals(fake_cfg: dict, ticks: list[dict],
                                          entry_price: float, entry_time_iso: str,
                                          lazy_fast_sec: int, lazy_fast_window: int,
                                          lazy_slow_sec: int,
                                          liq_usd: float = 50_000) -> dict | None:
    """Tick-native DIP_SCALE_OUT replay. Mirrors simulate_dip_scale_out
    (sim_engines.py) but on real ticks with interval throttling.

    Both P1 (original entry) and P2 (dip re-entry) use scale-out tranches:
    sell sell_frac at each gain_pct TP, remainder rides the runner_trail.
    """
    from sim_engines import _exit, BUY_SLIPPAGE as _BUY_SLIP, SLIPPAGE_TRAIL as _SLIP_TRAIL
    from sim_engines import _sim_liquidity_usd
    import sim_engines as _se
    _se._sim_liquidity_usd = liq_usd  # let _exit see the right liq for slippage

    if not ticks:
        return None

    entry_time = datetime.fromisoformat(entry_time_iso.replace("Z", "+00:00"))

    sl_pct = fake_cfg["sl"] / 100
    trail_pct = fake_cfg.get("trail", 5) / 100
    act_pct = fake_cfg.get("act", 20) / 100
    horizon = fake_cfg["horizon_min"]
    dip_threshold = abs(fake_cfg["dip_threshold"])
    if dip_threshold > 1:
        dip_threshold = dip_threshold / 100
    bounce_threshold = fake_cfg.get("bounce_threshold", 0)
    if bounce_threshold > 1:
        bounce_threshold = bounce_threshold / 100
    dip_size_mult = fake_cfg.get("dip_size_mult", 1.0)
    tranches = fake_cfg["tranches"]
    runner_trail = fake_cfg["runner_trail"] / 100
    runner_act = fake_cfg.get("runner_act", 50) / 100

    p1_weight = 1.0 / (1.0 + dip_size_mult)
    p2_weight = dip_size_mult / (1.0 + dip_size_mult)

    # Position 1 state
    p1_entry = entry_price
    p1_sl = p1_entry * (1 - sl_pct)
    p1_remaining = 1.0
    p1_pnl = 0.0
    p1_tranche_sold = [False] * len(tranches)
    p1_high = p1_entry
    p1_runner_active = False
    p1_closed = False

    # Position 2 state
    p2_opened = False
    p2_entry = 0.0
    p2_sl = 0.0
    p2_remaining = 1.0
    p2_pnl = 0.0
    p2_tranche_sold = [False] * len(tranches)
    p2_high = 0.0
    p2_runner_active = False
    p2_closed = False

    low_since_entry = entry_price
    dip_triggered = False
    reentry_done = False

    is_lazy = lazy_fast_sec > 0
    last_check_ts = 0.0
    last_tick_price = entry_price
    last_mins = 0

    for tick in ticks:
        try:
            tick_time = datetime.fromisoformat(tick["fetched_at"].replace("Z", "+00:00"))
        except Exception:
            continue
        if tick_time < entry_time:
            continue
        tick_price = float(tick["price_usd"])
        if tick_price <= 0:
            continue

        mins = (tick_time - entry_time).total_seconds() / 60.0
        last_tick_price = tick_price
        last_mins = mins

        # Dip tracking runs every tick (no throttle)
        if tick_price < low_since_entry:
            low_since_entry = tick_price

        # Running peak for both positions (always updated)
        if tick_price > p1_high:
            p1_high = tick_price
        if p2_opened and tick_price > p2_high:
            p2_high = tick_price

        # Throttle exit evaluation (but not dip detection / peak tracking)
        if is_lazy:
            now_ts = tick_time.timestamp()
            age_sec = (tick_time - entry_time).total_seconds()
            interval = lazy_fast_sec if age_sec < lazy_fast_window else lazy_slow_sec
            if last_check_ts > 0 and (now_ts - last_check_ts) < interval:
                continue
            last_check_ts = now_ts

        # === Position 1 ===
        if not p1_closed:
            if tick_price <= p1_sl:
                sl_res = _exit("sl_hit", p1_sl, p1_entry, mins, is_sl=True)
                p1_pnl += sl_res["pnl_pct"] * p1_remaining
                p1_closed = True
            else:
                for i, (gain_pct, sell_frac) in enumerate(tranches):
                    if p1_tranche_sold[i]:
                        continue
                    tp = p1_entry * (1 + gain_pct / 100)
                    if tick_price >= tp:
                        tp_res = _exit("tp_hit", tp, p1_entry, mins)
                        actual = min(sell_frac, p1_remaining)
                        p1_pnl += tp_res["pnl_pct"] * actual
                        p1_remaining -= actual
                        p1_tranche_sold[i] = True
                if p1_remaining > 0.001:
                    if not p1_runner_active and p1_high >= p1_entry * (1 + runner_act):
                        p1_runner_active = True
                    if p1_runner_active:
                        trigger = p1_high * (1 - runner_trail)
                        if trigger > p1_entry and tick_price <= trigger:
                            tr_res = _exit("trail_stop", trigger, p1_entry, mins)
                            p1_pnl += tr_res["pnl_pct"] * p1_remaining
                            p1_remaining = 0
                            p1_closed = True
                elif p1_remaining <= 0.001:
                    p1_closed = True

        # === Dip detection + P2 entry ===
        if not reentry_done and not p1_closed:
            dip_level = entry_price * (1 - dip_threshold)
            if low_since_entry <= dip_level:
                dip_triggered = True
            if dip_triggered:
                if bounce_threshold <= 0:
                    p2_opened = True
                    reentry_done = True
                    p2_entry = dip_level * (1 + _BUY_SLIP)
                    p2_sl = p2_entry * (1 - sl_pct)
                    p2_high = tick_price
                else:
                    if low_since_entry > 0 and (tick_price / low_since_entry - 1) >= bounce_threshold:
                        p2_opened = True
                        reentry_done = True
                        p2_entry = low_since_entry * (1 + bounce_threshold) * (1 + _BUY_SLIP)
                        p2_sl = p2_entry * (1 - sl_pct)
                        p2_high = tick_price

        # === Position 2 ===
        if p2_opened and not p2_closed:
            if tick_price <= p2_sl:
                sl_res = _exit("sl_hit", p2_sl, p2_entry, mins, is_sl=True)
                p2_pnl += sl_res["pnl_pct"] * p2_remaining
                p2_closed = True
            else:
                for i, (gain_pct, sell_frac) in enumerate(tranches):
                    if p2_tranche_sold[i]:
                        continue
                    tp = p2_entry * (1 + gain_pct / 100)
                    if tick_price >= tp:
                        tp_res = _exit("tp_hit", tp, p2_entry, mins)
                        actual = min(sell_frac, p2_remaining)
                        p2_pnl += tp_res["pnl_pct"] * actual
                        p2_remaining -= actual
                        p2_tranche_sold[i] = True
                if p2_remaining > 0.001:
                    if not p2_runner_active and p2_high >= p2_entry * (1 + runner_act):
                        p2_runner_active = True
                    if p2_runner_active:
                        trigger = p2_high * (1 - runner_trail)
                        if trigger > p2_entry and tick_price <= trigger:
                            tr_res = _exit("trail_stop", trigger, p2_entry, mins)
                            p2_pnl += tr_res["pnl_pct"] * p2_remaining
                            p2_remaining = 0
                            p2_closed = True
                elif p2_remaining <= 0.001:
                    p2_closed = True

        # Early exit when both done
        if p1_closed and (p2_closed or not p2_opened):
            combined = p1_pnl * p1_weight + p2_pnl * p2_weight if p2_opened else p1_pnl
            return {
                "exit_reason": "trail_stop",
                "exit_price": 0,
                "pnl_pct": round(combined, 4),
                "pnl_usd": round(10.0 * combined, 2),
                "exit_minutes": int(mins),
                "high_price_seen": max(p1_high, p2_high),
                "peak_from_entry": round(p1_high / p1_entry - 1, 4),
                "peak_to_exit_drop": 0.0,
                "time_to_peak_min": 0,
                "exit_in_first_5min": mins < 5,
                "p2_opened": p2_opened,
            }

        # Horizon timeout
        if mins >= horizon:
            if not p1_closed and p1_remaining > 0:
                to_res = _exit("timeout", tick_price, p1_entry, mins)
                p1_pnl += to_res["pnl_pct"] * p1_remaining
            if p2_opened and not p2_closed and p2_remaining > 0:
                to_res = _exit("timeout", tick_price, p2_entry, mins)
                p2_pnl += to_res["pnl_pct"] * p2_remaining
            combined = p1_pnl * p1_weight + p2_pnl * p2_weight if p2_opened else p1_pnl
            return {
                "exit_reason": "timeout",
                "exit_price": 0,
                "pnl_pct": round(combined, 4),
                "pnl_usd": round(10.0 * combined, 2),
                "exit_minutes": int(mins),
                "high_price_seen": max(p1_high, p2_high),
                "peak_from_entry": round(p1_high / p1_entry - 1, 4),
                "peak_to_exit_drop": 0.0,
                "time_to_peak_min": 0,
                "exit_in_first_5min": False,
                "p2_opened": p2_opened,
            }

    # Data ended before horizon
    if not p1_closed and p1_remaining > 0:
        p1_pnl += (last_tick_price * (1 - _SLIP_TRAIL) / p1_entry - 1) * p1_remaining
    if p2_opened and not p2_closed and p2_remaining > 0:
        p2_pnl += (last_tick_price * (1 - _SLIP_TRAIL) / p2_entry - 1) * p2_remaining
    combined = p1_pnl * p1_weight + p2_pnl * p2_weight if p2_opened else p1_pnl
    return {
        "exit_reason": "timeout_eod",
        "exit_price": 0,
        "pnl_pct": round(combined, 4),
        "pnl_usd": round(10.0 * combined, 2),
        "exit_minutes": int(last_mins),
        "high_price_seen": max(p1_high, p2_high),
        "peak_from_entry": round(p1_high / p1_entry - 1, 4),
        "peak_to_exit_drop": 0.0,
        "time_to_peak_min": 0,
        "exit_in_first_5min": False,
        "p2_opened": p2_opened,
    }


def _replay_dip_buy_with_intervals(fake_cfg: dict, ticks: list[dict],
                                    entry_price: float, entry_time_iso: str,
                                    lazy_fast_sec: int, lazy_fast_window: int,
                                    lazy_slow_sec: int) -> dict | None:
    """Tick-native DIP_BUY replay. Mirrors simulate_unified_dip_buy but uses
    real ticks + interval throttling (like _replay_with_intervals).

    P1 opens at entry. Watch for dip (price ≤ entry*(1-dip)) then bounce
    (price ≥ low*(1+bounce)) to open P2 at bounce level. Both positions
    evaluated independently via _evaluate_trade_exit tick-by-tick.

    Supports split-param P1/P2 via cfg["p1_*"] / cfg["p2_*"] keys.
    """
    from paper_trader import _evaluate_trade_exit, _last_eval_ts
    from strategies import sim_cfg_to_fake_trade

    if not ticks:
        return None

    sell_slip = 1 - 10 / 10_000
    entry_time = datetime.fromisoformat(entry_time_iso.replace("Z", "+00:00"))

    # DIP params
    dip_threshold = abs(cfg_val := fake_cfg.get("dip_threshold", -0.30))
    if dip_threshold > 1:  # expressed as -30 (pct) rather than -0.30
        dip_threshold = dip_threshold / 100.0
    bounce_threshold = fake_cfg.get("bounce_threshold", 0) or 0
    if bounce_threshold > 1:
        bounce_threshold = bounce_threshold / 100.0
    dip_size_mult = fake_cfg.get("dip_size_mult", 1.0)

    # P1 trade — trail/act/sl resolved by _get_trail_config from strategy NAME,
    # so we keep fake_cfg["name"] intact. tranche_label="dip_p1" (not "dip_p2")
    # ensures split-param regex returns P1 params.
    p1_cfg = dict(fake_cfg)
    p1_cfg["tranche_label"] = "dip_p1"
    p1_trade = sim_cfg_to_fake_trade(p1_cfg, entry_price, entry_time_iso,
                                      trade_id=f"tick_p1_{id(fake_cfg)}")
    _last_eval_ts.pop(p1_trade["id"], None)

    # P2 state
    p2_trade = None
    p2_open = False
    low_since_entry = entry_price
    dip_triggered = False
    p1_result = None
    p2_result = None

    p1_weight = 1.0 / (1.0 + dip_size_mult)
    p2_weight = dip_size_mult / (1.0 + dip_size_mult)

    is_lazy = lazy_fast_sec > 0
    last_check_ts = 0.0

    try:
        for tick in ticks:
            try:
                tick_time = datetime.fromisoformat(
                    tick["fetched_at"].replace("Z", "+00:00"))
            except Exception:
                continue
            if tick_time < entry_time:
                continue
            tick_price = float(tick["price_usd"])
            if tick_price <= 0:
                continue

            # Track low for dip detection (runs every tick, not throttled)
            if tick_price < low_since_entry:
                low_since_entry = tick_price
            if not p2_open and not dip_triggered:
                if low_since_entry <= entry_price * (1 - dip_threshold):
                    dip_triggered = True
            if dip_triggered and not p2_open:
                if bounce_threshold <= 0:
                    p2_entry = low_since_entry * 1.015  # BUY_SLIPPAGE
                    p2_cfg = dict(fake_cfg)
                    p2_cfg["tranche_label"] = "dip_p2"  # split regex returns P2 params
                    # SL for P2 (split-param): use p2_sl if provided, else shared
                    if "p2_sl" in fake_cfg:
                        p2_cfg["sl_mult"] = 1 - fake_cfg["p2_sl"] / 100
                    p2_trade = sim_cfg_to_fake_trade(p2_cfg, p2_entry,
                                                      tick["fetched_at"],
                                                      trade_id=f"tick_p2_{id(fake_cfg)}")
                    _last_eval_ts.pop(p2_trade["id"], None)
                    p2_open = True
                elif tick_price / low_since_entry - 1 >= bounce_threshold:
                    p2_entry = low_since_entry * (1 + bounce_threshold) * 1.015
                    p2_cfg = dict(fake_cfg)
                    p2_cfg["tranche_label"] = "dip_p2"  # split regex returns P2 params
                    # SL for P2 (split-param): use p2_sl if provided, else shared
                    if "p2_sl" in fake_cfg:
                        p2_cfg["sl_mult"] = 1 - fake_cfg["p2_sl"] / 100
                    p2_trade = sim_cfg_to_fake_trade(p2_cfg, p2_entry,
                                                      tick["fetched_at"],
                                                      trade_id=f"tick_p2_{id(fake_cfg)}")
                    _last_eval_ts.pop(p2_trade["id"], None)
                    p2_open = True

            # Interval throttling applies to exit evaluation only
            if is_lazy:
                now_ts = tick_time.timestamp()
                age_sec = (tick_time - entry_time).total_seconds()
                interval = lazy_fast_sec if age_sec < lazy_fast_window else lazy_slow_sec
                if last_check_ts > 0 and (now_ts - last_check_ts) < interval:
                    continue
                last_check_ts = now_ts

            # Evaluate P1
            if p1_result is None:
                ev = _evaluate_trade_exit(p1_trade, tick_price, tick_time, sell_slip)
                if ev is not None:
                    if ev.get("high_price_seen") is not None:
                        nh = ev["high_price_seen"]
                        if nh > float(p1_trade.get("high_price_seen") or 0):
                            p1_trade["high_price_seen"] = nh
                    if ev.get("status") is not None:
                        p1_result = {
                            "exit_reason": ev["status"],
                            "pnl_pct": ev.get("pnl_pct", 0),
                            "exit_minutes": ev.get("exit_minutes", 0),
                        }

            # Evaluate P2
            if p2_open and p2_result is None and p2_trade is not None:
                ev = _evaluate_trade_exit(p2_trade, tick_price, tick_time, sell_slip)
                if ev is not None:
                    if ev.get("high_price_seen") is not None:
                        nh = ev["high_price_seen"]
                        if nh > float(p2_trade.get("high_price_seen") or 0):
                            p2_trade["high_price_seen"] = nh
                    if ev.get("status") is not None:
                        p2_result = {
                            "exit_reason": ev["status"],
                            "pnl_pct": ev.get("pnl_pct", 0),
                            "exit_minutes": ev.get("exit_minutes", 0),
                        }

            if p1_result is not None and (p2_result is not None or not p2_open):
                break

        # Compose weighted result
        total_pnl = 0.0
        last_reason = "timeout_eod"
        last_elapsed = 0

        if p1_result:
            total_pnl += p1_result["pnl_pct"] * (p1_weight if p2_open else 1.0)
            last_reason = p1_result["exit_reason"]
            last_elapsed = p1_result["exit_minutes"]
        else:
            last_price = float(ticks[-1]["price_usd"])
            total_pnl += (last_price / entry_price - 1) * (p1_weight if p2_open else 1.0)

        if p2_open:
            if p2_result:
                total_pnl += p2_result["pnl_pct"] * p2_weight
                last_elapsed = max(last_elapsed, p2_result["exit_minutes"])
            elif p2_trade:
                last_price = float(ticks[-1]["price_usd"])
                total_pnl += (last_price / float(p2_trade["entry_price"]) - 1) * p2_weight

        pos_usd = 10.0
        return {
            "exit_reason": last_reason,
            "exit_price": 0,  # composite; not meaningful for analytics
            "pnl_pct": round(total_pnl, 4),
            "pnl_usd": round(pos_usd * total_pnl, 2),
            "exit_minutes": last_elapsed,
            "high_price_seen": max(
                float(p1_trade.get("high_price_seen") or entry_price),
                float((p2_trade or {}).get("high_price_seen") or 0),
            ),
            "peak_from_entry": round(
                max(
                    float(p1_trade.get("high_price_seen") or entry_price) / entry_price - 1,
                    0.0,
                ),
                4,
            ),
            "peak_to_exit_drop": 0.0,
            "time_to_peak_min": 0,
            "exit_in_first_5min": last_elapsed < 5,
            "p2_opened": p2_open,
        }
    finally:
        _last_eval_ts.pop(p1_trade["id"], None)
        if p2_trade:
            _last_eval_ts.pop(p2_trade["id"], None)


# --- On-chain feature filters ---

FEATURE_FILTERS = [
    ("ALL", lambda feat: True),
    # Liquidity bands
    ("LIQ>20K", lambda feat: (feat.get("liquidity_usd") or 0) >= 20_000),
    ("LIQ>10K", lambda feat: (feat.get("liquidity_usd") or 0) >= 10_000),
    ("LIQ>5K", lambda feat: (feat.get("liquidity_usd") or 0) >= 5_000),
    ("LIQ<5K", lambda feat: 0 < (feat.get("liquidity_usd") or 0) < 5_000),
    ("LIQ<10K", lambda feat: 0 < (feat.get("liquidity_usd") or 0) < 10_000),
    # Mcap bands
    ("MCAP>500K", lambda feat: (feat.get("market_cap") or 0) >= 500_000),
    ("MCAP>100K", lambda feat: (feat.get("market_cap") or 0) >= 100_000),
    ("MCAP>50K", lambda feat: (feat.get("market_cap") or 0) >= 50_000),
    ("MCAP>25K", lambda feat: (feat.get("market_cap") or 0) >= 25_000),
    ("MCAP<25K", lambda feat: (feat.get("market_cap") or 0) < 25_000),
    ("MCAP<50K", lambda feat: (feat.get("market_cap") or 0) < 50_000),
    ("MCAP_25-100K", lambda feat: 25_000 <= (feat.get("market_cap") or 0) < 100_000),
    ("MCAP_100-500K", lambda feat: 100_000 <= (feat.get("market_cap") or 0) < 500_000),
    # Age bands
    ("AGE>6h", lambda feat: (feat.get("token_age_hours") or 0) >= 6),
    ("AGE>2h", lambda feat: (feat.get("token_age_hours") or 0) >= 2),
    ("AGE>1h", lambda feat: (feat.get("token_age_hours") or 0) >= 1),
    ("AGE>30m", lambda feat: (feat.get("token_age_hours") or 0) >= 0.5),
    ("AGE<30m", lambda feat: (feat.get("token_age_hours") or 99) < 0.5),
    ("AGE<1h", lambda feat: (feat.get("token_age_hours") or 99) < 1),
    ("AGE<2h", lambda feat: (feat.get("token_age_hours") or 99) < 2),
    ("AGE_1-6h", lambda feat: 1 <= (feat.get("token_age_hours") or 0) < 6),
    # Pump.fun
    ("PUMP_FUN", lambda feat: feat.get("is_pump_fun") is True),
    ("NOT_PUMP", lambda feat: feat.get("is_pump_fun") is not True),
    # Score bands
    ("SCORE>60", lambda feat: (feat.get("score_at_snapshot") or 0) >= 60),
    ("SCORE>50", lambda feat: (feat.get("score_at_snapshot") or 0) >= 50),
    ("SCORE>40", lambda feat: (feat.get("score_at_snapshot") or 0) >= 40),
    ("SCORE<40", lambda feat: 0 < (feat.get("score_at_snapshot") or 0) < 40),
    ("SCORE<50", lambda feat: 0 < (feat.get("score_at_snapshot") or 0) < 50),
    # On-chain quality
    ("GINI<0.8", lambda feat: (feat.get("helius_gini") or 1) < 0.8),
    ("WHALE>0", lambda feat: (feat.get("whale_new_entries") or 0) > 0),
    ("BSR>1", lambda feat: (feat.get("buy_sell_ratio_24h") or 0) > 1),
    ("BSR>1.5", lambda feat: (feat.get("buy_sell_ratio_24h") or 0) > 1.5),
    ("HOLD>100", lambda feat: (feat.get("holder_count") or 0) > 100),
    ("IMPACT<5%", lambda feat: (feat.get("jup_price_impact_1k") or 99) < 5),
    # Combo filters
    ("QUALITY", lambda feat: (
        (feat.get("market_cap") or 0) >= 25_000
        and (feat.get("token_age_hours") or 0) >= 1
        and (feat.get("liquidity_usd") or 0) >= 5_000
    )),
    ("PREMIUM", lambda feat: (
        (feat.get("market_cap") or 0) >= 50_000
        and (feat.get("token_age_hours") or 0) >= 2
        and (feat.get("liquidity_usd") or 0) >= 10_000
    )),
    ("ULTRA", lambda feat: (
        (feat.get("market_cap") or 0) >= 100_000
        and (feat.get("token_age_hours") or 0) >= 2
        and (feat.get("liquidity_usd") or 0) >= 10_000
        and (feat.get("holder_count") or 0) >= 100
    )),
]


def _fetch_snapshot_features_for_trades(trades: list[dict]) -> dict[int, dict]:
    """Fetch on-chain features from token_snapshots, matched by token_address + nearest time.
    Returns: {trade_id: {features...}} for each trade."""

    # Group trades by token_address
    by_token: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        if t.get("token_address"):
            by_token[t["token_address"]].append(t)

    features_by_trade_id: dict[int, dict] = {}
    fetched = 0

    for addr, token_trades in by_token.items():
        # Get time range for this token
        times = [t["created_at"] for t in token_trades]
        min_t = min(times)
        max_t = max(times)

        params = [
            ("select", "id,snapshot_at,liquidity_usd,market_cap,token_age_hours,is_pump_fun,"
                       "helius_gini,whale_new_entries,score_at_snapshot,jup_price_impact_1k,"
                       "buy_sell_ratio_24h,holder_count"),
            ("token_address", f"eq.{addr}"),
            ("snapshot_at", f"gte.{min_t[:10]}T00:00:00Z"),
            ("order", "snapshot_at.desc"),
            ("limit", "10"),
        ]
        rows = sb_get("token_snapshots", params)
        if not rows:
            continue

        # For each trade, find closest snapshot (within 60min)
        for t in token_trades:
            trade_dt = datetime.fromisoformat(t["created_at"].replace("Z", "+00:00"))
            best_snap = None
            best_delta = 9999999
            for r in rows:
                snap_dt = datetime.fromisoformat(r["snapshot_at"].replace("Z", "+00:00"))
                delta = abs((snap_dt - trade_dt).total_seconds())
                if delta < best_delta and delta < 3600:  # within 1h
                    best_delta = delta
                    best_snap = r
            if best_snap:
                features_by_trade_id[t["id"]] = best_snap
                fetched += 1

    print(f"Matched snapshot features for {fetched}/{len(trades)} trades")
    return features_by_trade_id


def _replay_with_trigger(fake_trade: dict, ticks: list[dict]) -> dict | None:
    """Simulate Jupiter Trigger V2: on-chain stop-loss/trail that fills instantly.

    How Trigger V2 works in reality:
    - At trade open: place a limit sell order at SL price on Jupiter
    - When trail activates: PATCH the order to trail_trigger price
    - Fill is INSTANT when price crosses the order (no polling delay)
    - Trade-off: the trigger price is set discretely (PATCH every ~30s),
      so it can't react to sub-30s spikes

    This sim compares trigger (instant fill at order price) vs polling
    (detect at next check, execute with slippage)."""
    from paper_trader import _get_trail_config

    if not ticks:
        return None

    entry_price = float(fake_trade["entry_price"])
    entry_time = datetime.fromisoformat(
        fake_trade["created_at"].replace("Z", "+00:00"))
    sl_price = float(fake_trade["sl_price"])
    horizon = fake_trade.get("horizon_minutes", 120)
    trail_pct, act_pct = _get_trail_config(fake_trade)

    # State
    high_seen = entry_price
    trigger_price = sl_price  # initial trigger = SL
    trail_activated = False
    last_patch_ts = 0.0  # last time we "patched" the trigger order

    for tick in ticks:
        tick_time = datetime.fromisoformat(
            tick["fetched_at"].replace("Z", "+00:00"))
        if tick_time < entry_time:
            continue

        tick_price = float(tick["price_usd"])
        if tick_price <= 0:
            continue

        elapsed_min = (tick_time - entry_time).total_seconds() / 60

        # Update peak
        if tick_price > high_seen:
            high_seen = tick_price

        # Check timeout
        if elapsed_min >= horizon:
            pnl = round((tick_price / entry_price) - 1, 4)
            return {
                "exit_reason": "timeout",
                "exit_price": tick_price,
                "pnl_pct": pnl,
                "pnl_usd": round(float(fake_trade.get("position_usd") or 10) * pnl, 2),
                "exit_minutes": int(elapsed_min),
                "high_price_seen": high_seen,
                "peak_from_entry": round(high_seen / entry_price - 1, 4),
                "peak_to_exit_drop": round(1 - tick_price / high_seen, 4) if high_seen > 0 else 0,
                "time_to_peak_min": 0,
                "exit_in_first_5min": False,
            }

        # Trail activation check (every tick — Trigger V2 PATCHes at ~30s)
        if trail_pct is not None and not trail_activated:
            act_price = entry_price * (1 + act_pct)
            if high_seen >= act_price:
                trail_activated = True

        # PATCH trigger price upward (every 30s, simulating PATCH latency)
        now_ts = tick_time.timestamp()
        if trail_activated and (now_ts - last_patch_ts) >= 30:
            new_trigger = high_seen * (1 - trail_pct)
            if new_trigger > trigger_price:
                trigger_price = new_trigger
            last_patch_ts = now_ts

        # INSTANT FILL: if price crosses trigger (Trigger V2 fills on-chain)
        if tick_price <= trigger_price:
            # Fill at trigger_price (not tick_price — limit order fills at order price)
            fill_price = trigger_price
            pnl = round((fill_price / entry_price) - 1, 4)
            exit_reason = "trail_stop" if trail_activated else "sl_hit"
            return {
                "exit_reason": exit_reason,
                "exit_price": fill_price,
                "pnl_pct": pnl,
                "pnl_usd": round(float(fake_trade.get("position_usd") or 10) * pnl, 2),
                "exit_minutes": int(elapsed_min),
                "high_price_seen": high_seen,
                "peak_from_entry": round(high_seen / entry_price - 1, 4),
                "peak_to_exit_drop": round(1 - fill_price / high_seen, 4) if high_seen > 0 else 0,
                "time_to_peak_min": 0,
                "exit_in_first_5min": elapsed_min < 5,
            }

    # No exit → timeout at last tick
    last_tick = ticks[-1]
    last_price = float(last_tick["price_usd"])
    pnl = round((last_price / entry_price) - 1, 4) if entry_price > 0 else 0
    return {
        "exit_reason": "timeout_eod",
        "exit_price": last_price,
        "pnl_pct": pnl,
        "pnl_usd": round(float(fake_trade.get("position_usd") or 10) * pnl, 2),
        "exit_minutes": int((datetime.fromisoformat(
            last_tick["fetched_at"].replace("Z", "+00:00")) - entry_time).total_seconds() / 60),
        "high_price_seen": high_seen,
        "peak_from_entry": round(high_seen / entry_price - 1, 4),
        "peak_to_exit_drop": round(1 - last_price / high_seen, 4) if high_seen > 0 else 0,
        "time_to_peak_min": 0,
        "exit_in_first_5min": False,
    }


def _replay_with_trigger_sl_only(fake_trade: dict, ticks: list[dict],
                                 lazy_fast_sec: int, lazy_fast_window: int,
                                 lazy_slow_sec: int) -> dict | None:
    """Hybrid: SL via on-chain trigger (instant fill), trail via polling.
    Best of both worlds: instant SL protection + polling flexibility for trail."""
    from paper_trader import _evaluate_trade_exit, _last_eval_ts, _get_trail_config

    if not ticks:
        return None

    trade_id = str(fake_trade["id"])
    _last_eval_ts.pop(trade_id, None)
    entry_price = float(fake_trade["entry_price"])
    entry_time = datetime.fromisoformat(fake_trade["created_at"].replace("Z", "+00:00"))
    sl_price = float(fake_trade["sl_price"])
    sell_slip = 1 - 10 / 10_000
    is_lazy = lazy_fast_sec > 0
    last_check_ts = 0.0
    max_price_seen = entry_price

    try:
        for tick in ticks:
            tick_time = datetime.fromisoformat(tick["fetched_at"].replace("Z", "+00:00"))
            if tick_time < entry_time:
                continue
            tick_price = float(tick["price_usd"])
            if tick_price <= 0:
                continue

            if tick_price > max_price_seen:
                max_price_seen = tick_price

            # INSTANT SL check (on-chain trigger fills immediately)
            if tick_price <= sl_price:
                elapsed = (tick_time - entry_time).total_seconds() / 60
                pnl = round((sl_price / entry_price) - 1, 4)  # fill at SL price
                return {
                    "exit_reason": "sl_hit",
                    "exit_price": sl_price,
                    "pnl_pct": pnl,
                    "pnl_usd": round(float(fake_trade.get("position_usd") or 10) * pnl, 2),
                    "exit_minutes": int(elapsed),
                    "high_price_seen": max_price_seen,
                    "peak_from_entry": round(max_price_seen / entry_price - 1, 4),
                    "peak_to_exit_drop": round(1 - sl_price / max_price_seen, 4) if max_price_seen > 0 else 0,
                    "time_to_peak_min": 0,
                    "exit_in_first_5min": elapsed < 5,
                }

            # POLLING for trail/TP/timeout (with interval throttle)
            if is_lazy:
                now_ts = tick_time.timestamp()
                age_sec = (tick_time - entry_time).total_seconds()
                interval = lazy_fast_sec if age_sec < lazy_fast_window else lazy_slow_sec
                if last_check_ts > 0 and (now_ts - last_check_ts) < interval:
                    if tick_price > float(fake_trade.get("high_price_seen") or 0):
                        fake_trade["high_price_seen"] = tick_price
                    continue
                last_check_ts = now_ts

            # Temporarily set SL very low so _evaluate_trade_exit only handles trail/TP/timeout
            saved_sl = fake_trade["sl_price"]
            fake_trade["sl_price"] = entry_price * 0.01  # effectively disable SL in eval
            ev = _evaluate_trade_exit(fake_trade, tick_price, tick_time, sell_slip, sell_fee_bps=0)
            fake_trade["sl_price"] = saved_sl  # restore

            if ev is None:
                continue
            if ev.get("high_price_seen") is not None:
                new_high = ev["high_price_seen"]
                if new_high > float(fake_trade.get("high_price_seen") or 0):
                    fake_trade["high_price_seen"] = new_high
            if "status" in ev and ev["status"] is not None and ev["status"] != "sl_hit":
                exit_price = ev.get("exit_price", 0)
                elapsed = ev.get("exit_minutes", 0)
                return {
                    "exit_reason": ev["status"],
                    "exit_price": exit_price,
                    "pnl_pct": ev.get("pnl_pct", 0),
                    "pnl_usd": ev.get("pnl_usd", 0),
                    "exit_minutes": elapsed,
                    "high_price_seen": fake_trade["high_price_seen"],
                    "peak_from_entry": round(max_price_seen / entry_price - 1, 4),
                    "peak_to_exit_drop": round(1 - exit_price / max_price_seen, 4) if max_price_seen > 0 else 0,
                    "time_to_peak_min": 0,
                    "exit_in_first_5min": elapsed < 5,
                }

        # Timeout
        last_tick = ticks[-1]
        last_price = float(last_tick["price_usd"])
        pnl = round((last_price / entry_price) - 1, 4) if entry_price > 0 else 0
        return {
            "exit_reason": "timeout_eod", "exit_price": last_price,
            "pnl_pct": pnl, "pnl_usd": round(float(fake_trade.get("position_usd") or 10) * pnl, 2),
            "exit_minutes": int((datetime.fromisoformat(last_tick["fetched_at"].replace("Z","+00:00")) - entry_time).total_seconds() / 60),
            "high_price_seen": max_price_seen,
            "peak_from_entry": round(max_price_seen / entry_price - 1, 4),
            "peak_to_exit_drop": 0, "time_to_peak_min": 0, "exit_in_first_5min": False,
        }
    finally:
        _last_eval_ts.pop(trade_id, None)


# ---------------------------------------------------------------------------
# Smoothing strategies for decision price (v133)
# ---------------------------------------------------------------------------
# Each mode transforms a raw tick_price into a "decision_price" used for
# SL/TP/trail trigger evaluation. The exit execution price stays at the raw
# tick_price (mirrors live behavior where Jupiter fills at instant price).
#
# Supported modes:
#   raw           — no smoothing (baseline)
#   median_3      — median of last 3 Jupiter ticks (kills spikes, ~30-60s lag)
#   median_5      — median of last 5 ticks (smoother, ~60-120s lag)
#   winsor_p95    — clip delta vs prev to ±18% (kills flash wicks)
#   dual_confirm  — trigger requires 2 consecutive ticks on same side of SL/TP
#   ema_fast      — EMA window=2 (reactive smoothing)
#   ema_slow      — EMA window=8 (wide-trail smoothing)
#   hysteresis    — once trigger crossed, needs 2% retrace before re-arming
#   volume_gated  — skip ticks with volume_usd < 500 (ghost liquidity filter)
#   max_hybrid    — decision = tick_price (proxy; full version needs DS stream)
#
# State is held per-trade in a dict. Caller resets it on new trade.
SMOOTHING_MODES = [
    "raw", "median_3", "median_5", "winsor_p95", "dual_confirm",
    "ema_fast", "ema_slow", "hysteresis", "volume_gated",
    # v143.2 — single-stream ports from paper_trader._decision_price
    "jp_sampled_60s", "jp_sampled_180s", "vwap_5min", "ohlc_burst_60s",
    # v143.3 — dual-stream ports (JP + DS). Caller must pass dex_ticks to
    # _replay_with_intervals so the tick stream is enriched with ds_price
    # before dispatch to _smooth_decision.
    "confirm", "twin_confirm", "hybrid",
]

DUAL_STREAM_MODES = {"confirm", "twin_confirm", "hybrid"}


def _smooth_decision(tick: dict, state: dict, mode: str,
                     entry_price: float, sl_price: float,
                     tp_price: float | None) -> float | None:
    """Return smoothed decision price, or None to skip this tick entirely
    (volume_gated). Mutates `state`.

    v143.3 — dual-stream modes (confirm / twin_confirm / hybrid) require the
    caller to enrich each primary tick with tick['ds_price'] before calling
    this. That is done in _replay_with_intervals when `dex_ticks` is provided.
    If ds_price is missing on a dual mode, the function falls back to raw.
    """
    p = float(tick["price_usd"])
    if mode == "raw":
        return p

    # v143.3 — dual-stream consensus modes
    if mode in ("confirm", "twin_confirm", "hybrid"):
        ds = tick.get("ds_price")
        if ds is None or ds <= 0:
            return p  # no paired DS price yet → pass through (warm-up)
        ds = float(ds)
        if mode == "confirm":
            return (p + ds) / 2.0
        if mode == "hybrid":
            # Decision on DS, exit still at Jupiter. _evaluate_trade_exit
            # receives tick_price as exit_ref, so returning `ds` routes the
            # trigger comparisons through DS while execution stays Jupiter.
            return ds
        # twin_confirm — require both sources to agree on breaching SL/TP.
        # If they disagree, serve the non-breaching side so the trigger is
        # suppressed this tick.
        if sl_price:
            jp_breach = p <= sl_price
            ds_breach = ds <= sl_price
            if jp_breach != ds_breach:
                return max(p, ds)
        if tp_price:
            jp_breach = p >= tp_price
            ds_breach = ds >= tp_price
            if jp_breach != ds_breach:
                return min(p, ds)
        return p

    if mode == "median_3" or mode == "median_5":
        window = 3 if mode == "median_3" else 5
        hist = state.setdefault("hist", [])
        hist.append(p)
        if len(hist) > window:
            hist.pop(0)
        if len(hist) < window:
            return p  # warm-up: pass through
        return sorted(hist)[len(hist) // 2]

    if mode == "winsor_p95":
        prev = state.get("prev_p", p)
        delta = p - prev
        cap = prev * 0.18  # p95 tick-to-tick
        if delta > cap:
            p_cap = prev + cap
        elif delta < -cap:
            p_cap = prev - cap
        else:
            p_cap = p
        state["prev_p"] = p_cap
        return p_cap

    if mode == "dual_confirm":
        # Gate: only pass price through if it's "stable" vs prev (triggers
        # will fire naturally in _evaluate_trade_exit). Here we require that
        # 2 consecutive ticks agree on being below SL or above TP.
        prev = state.get("prev_p", p)
        was_breach = state.get("was_breach", None)
        breach = None
        if sl_price and p <= sl_price and prev <= sl_price:
            breach = "sl"
        elif tp_price and p >= tp_price and prev >= tp_price:
            breach = "tp"
        state["prev_p"] = p
        state["was_breach"] = breach
        # If single-tick breach not confirmed, return a safe price that doesn't trigger
        if (sl_price and p <= sl_price and prev > sl_price) or \
           (tp_price and p >= tp_price and prev < tp_price):
            # Return prev (last non-breach) to skip the trigger this round
            return prev
        return p

    if mode == "ema_fast" or mode == "ema_slow":
        window = 2 if mode == "ema_fast" else 8
        alpha = 2.0 / (window + 1)
        prev_ema = state.get("ema")
        if prev_ema is None:
            state["ema"] = p
            return p
        new_ema = alpha * p + (1 - alpha) * prev_ema
        state["ema"] = new_ema
        return new_ema

    if mode == "hysteresis":
        # Once SL/TP crossed, require 2% retrace before re-checking.
        prev_p = state.get("prev_p", p)
        armed_sl = state.setdefault("armed_sl", True)
        armed_tp = state.setdefault("armed_tp", True)
        if not armed_sl:
            # re-arm after 2% bounce up
            if sl_price and p >= sl_price * 1.02:
                state["armed_sl"] = True
        elif sl_price and p <= sl_price:
            state["armed_sl"] = False
        if not armed_tp:
            if tp_price and p <= tp_price * 0.98:
                state["armed_tp"] = True
        elif tp_price and p >= tp_price:
            state["armed_tp"] = False
        state["prev_p"] = p
        # Serve a price that respects current armed state
        if not state["armed_sl"] and sl_price and p <= sl_price:
            return sl_price * 1.001  # slightly above SL to avoid re-trigger
        if not state["armed_tp"] and tp_price and p >= tp_price:
            return tp_price * 0.999
        return p

    if mode == "volume_gated":
        vol = tick.get("volume_usd")
        if vol is not None and vol < 500:
            return None  # skip tick entirely
        return p

    # v143.2 — jp_sampled_60s / _180s: freeze decision between bar boundaries.
    # Mirrors paper_trader._decision_price. Suppresses intra-bar triggers.
    if mode == "jp_sampled_60s" or mode == "jp_sampled_180s":
        bar_sec = 60 if mode == "jp_sampled_60s" else 180
        try:
            tick_ts = datetime.fromisoformat(
                tick["fetched_at"].replace("Z", "+00:00")).timestamp()
        except Exception:
            return p
        last_close_ts = state.get("ohlcv_ts", 0)
        cur_bar = int(tick_ts // bar_sec) * bar_sec
        if cur_bar > last_close_ts:
            state["ohlcv_close"] = p
            state["ohlcv_ts"] = cur_bar
        return state.get("ohlcv_close", p)

    # v143.2 — VWAP 5min over sliding window. Needs price_ticks.volume_usd,
    # which is the rolling total; dv = delta between consecutive ticks.
    if mode == "vwap_5min":
        try:
            tick_ts = datetime.fromisoformat(
                tick["fetched_at"].replace("Z", "+00:00")).timestamp()
        except Exception:
            return p
        buf = state.setdefault("vwap_buf", [])  # [(ts, price, dv)]
        prev_total = state.get("vwap_last_vol")
        cur_vol = float(tick.get("volume_usd") or 0)
        dv = max(0.0, cur_vol - prev_total) if prev_total is not None else 0.0
        state["vwap_last_vol"] = cur_vol
        if dv > 0:
            buf.append((tick_ts, p, dv))
        cutoff = tick_ts - 300
        state["vwap_buf"] = [x for x in buf if x[0] >= cutoff]
        total_v = sum(v for _, _, v in state["vwap_buf"])
        if total_v <= 0:
            return p  # warm-up / zero volume
        return sum(px * v for _, px, v in state["vwap_buf"]) / total_v

    # v143.2 — ohlc_burst_60s: accumulate ticks per 60s bar, emit synthetic
    # O/L/H/C sequence at bar close. Caller must consume the emitted list,
    # so we return the raw price here; burst emission is driven by state.
    # sim_engines.candles_to_synthetic_ticks() already handles this when the
    # sim is fed candles directly. For tick-replay, we approximate by
    # returning the running OHLC close once per bar.
    if mode == "ohlc_burst_60s":
        try:
            tick_ts = datetime.fromisoformat(
                tick["fetched_at"].replace("Z", "+00:00")).timestamp()
        except Exception:
            return p
        bar_sec = 60
        cur_bar = int(tick_ts // bar_sec) * bar_sec
        buf = state.setdefault("ohlc_buf", [])
        last_bar = state.get("ohlc_last_bar", cur_bar)
        if cur_bar > last_bar and buf:
            # Emit synthetic bar close = last tick of prior bar
            state["ohlc_close"] = buf[-1]
            state["ohlc_buf"] = []
            state["ohlc_last_bar"] = cur_bar
        buf.append(p)
        state["ohlc_buf"] = buf[-300:]  # cap
        return state.get("ohlc_close", p)

    return p


def _replay_with_intervals(fake_trade: dict, ticks: list[dict],
                           lazy_fast_sec: int, lazy_fast_window: int,
                           lazy_slow_sec: int,
                           smoothing: str = "raw",
                           dex_ticks: list[dict] | None = None) -> dict | None:
    """Replay trade with custom check interval throttling + exit analytics.
    If all intervals are 0, checks every tick (CURRENT mode).
    Otherwise simulates LAZY-style throttle at given intervals.

    v143.3 — `dex_ticks` (optional) is a parallel DexScreener stream used by
    dual-stream smoothing modes (confirm / twin_confirm / hybrid). If passed,
    each primary tick is enriched with tick['ds_price'] = nearest-prior DS
    price before smoothing. Required when smoothing in DUAL_STREAM_MODES.

    Returns enhanced result with peak/timing analytics for exit style analysis."""
    from paper_trader import _evaluate_trade_exit, _last_eval_ts

    if not ticks:
        return None

    trade_id = str(fake_trade["id"])
    _last_eval_ts.pop(trade_id, None)

    sell_slip = 1 - 10 / 10_000
    entry_time = datetime.fromisoformat(
        fake_trade["created_at"].replace("Z", "+00:00"))
    entry_price = float(fake_trade["entry_price"])

    is_lazy = lazy_fast_sec > 0
    last_check_ts = 0.0

    # Exit analytics tracking
    max_price_seen = entry_price
    time_to_peak_min = 0
    tick_count = 0

    # Smoothing state (per-trade)
    smooth_state: dict = {}
    sl_price_trade = float(fake_trade.get("sl_price") or 0)
    tp_price_trade = float(fake_trade.get("tp_price") or 0) or None

    # v143.3 — build sorted DS-by-timestamp list for nearest-prior lookup
    # when a dual-stream smoothing mode is requested. O(n) precompute, then
    # O(1) amortized per primary tick using a cursor.
    ds_sorted: list[tuple[str, float]] = []
    if dex_ticks:
        for dt in dex_ticks:
            dp = float(dt.get("price_usd") or 0)
            if dp > 0 and dt.get("fetched_at"):
                ds_sorted.append((dt["fetched_at"], dp))
        ds_sorted.sort(key=lambda x: x[0])
    ds_cursor = 0  # advances as primary ticks step forward in time
    last_ds_price: float | None = None

    try:
        for tick in ticks:
            tick_time = datetime.fromisoformat(
                tick["fetched_at"].replace("Z", "+00:00"))
            if tick_time < entry_time:
                continue

            tick_price = float(tick["price_usd"])
            if tick_price <= 0:
                continue

            tick_count += 1

            # Track absolute peak (even during lazy skips)
            if tick_price > max_price_seen:
                max_price_seen = tick_price
                time_to_peak_min = int((tick_time - entry_time).total_seconds() / 60)

            # Custom interval throttle
            if is_lazy:
                now_ts = tick_time.timestamp()
                age_sec = (tick_time - entry_time).total_seconds()
                interval = lazy_fast_sec if age_sec < lazy_fast_window else lazy_slow_sec
                if last_check_ts > 0 and (now_ts - last_check_ts) < interval:
                    if tick_price > float(fake_trade.get("high_price_seen") or 0):
                        fake_trade["high_price_seen"] = tick_price
                    continue
                last_check_ts = now_ts

            # v143.3 — for dual-stream modes, advance DS cursor and attach
            # nearest-prior DS price to the tick before smoothing.
            if ds_sorted and smoothing in DUAL_STREAM_MODES:
                tick_ts_str = tick["fetched_at"]
                while ds_cursor < len(ds_sorted) and ds_sorted[ds_cursor][0] <= tick_ts_str:
                    last_ds_price = ds_sorted[ds_cursor][1]
                    ds_cursor += 1
                if last_ds_price is not None:
                    tick = dict(tick)
                    tick["ds_price"] = last_ds_price

            # Apply smoothing → decision_price (exit still at raw tick_price)
            decision_p = _smooth_decision(tick, smooth_state, smoothing,
                                          entry_price, sl_price_trade,
                                          tp_price_trade)
            if decision_p is None:  # volume_gated skip
                continue

            ev = _evaluate_trade_exit(fake_trade, tick_price, tick_time,
                                      sell_slip, sell_fee_bps=0,
                                      decision_price=decision_p)
            if ev is None:
                continue

            if ev.get("high_price_seen") is not None:
                new_high = ev["high_price_seen"]
                if new_high > float(fake_trade.get("high_price_seen") or 0):
                    fake_trade["high_price_seen"] = new_high

            if "status" in ev and ev["status"] is not None:
                exit_price = ev.get("exit_price", 0)
                peak_from_entry = (max_price_seen / entry_price - 1) if entry_price > 0 else 0
                exit_min = ev.get("exit_minutes", 0)
                return {
                    "exit_reason": ev["status"],
                    "exit_price": exit_price,
                    "pnl_pct": ev.get("pnl_pct", 0),
                    "pnl_usd": ev.get("pnl_usd", 0),
                    "exit_minutes": exit_min,
                    "high_price_seen": fake_trade["high_price_seen"],
                    # Exit analytics
                    "peak_from_entry": round(peak_from_entry, 4),
                    "peak_to_exit_drop": round(1 - exit_price / max_price_seen, 4) if max_price_seen > 0 else 0,
                    "time_to_peak_min": time_to_peak_min,
                    "exit_in_first_5min": exit_min < 5,
                }

        # No exit → timeout at last tick
        last_tick = ticks[-1]
        last_time = datetime.fromisoformat(
            last_tick["fetched_at"].replace("Z", "+00:00"))
        last_price = float(last_tick["price_usd"])
        pnl_pct = round((last_price / entry_price) - 1, 4) if entry_price > 0 else 0
        pos_usd = float(fake_trade.get("position_usd") or 10.0)
        peak_from_entry = (max_price_seen / entry_price - 1) if entry_price > 0 else 0

        return {
            "exit_reason": "timeout_eod",
            "exit_price": last_price,
            "pnl_pct": pnl_pct,
            "pnl_usd": round(pos_usd * pnl_pct, 2),
            "exit_minutes": int((last_time - entry_time).total_seconds() / 60),
            "high_price_seen": fake_trade["high_price_seen"],
            "peak_from_entry": round(peak_from_entry, 4),
            "peak_to_exit_drop": round(1 - last_price / max_price_seen, 4) if max_price_seen > 0 else 0,
            "time_to_peak_min": time_to_peak_min,
            "exit_in_first_5min": False,
        }
    finally:
        _last_eval_ts.pop(trade_id, None)


def _tick_grid_search(trades: list[dict], ticks_by_token: dict[str, list[dict]],
                      price_source: str, mc_sims: int = 500,
                      features_by_snapshot: dict[int, dict] | None = None,
                      feature_filter_name: str = "ALL") -> list[dict]:
    """Exhaustive grid search: all strategy types x all interval profiles on tick data.
    If features_by_snapshot provided, filters trades by on-chain features."""

    grid = _build_tick_grid()

    # Apply feature filter to trades (features keyed by trade_id)
    if feature_filter_name != "ALL" and features_by_snapshot:
        filter_fn = None
        for fname, fn in FEATURE_FILTERS:
            if fname == feature_filter_name:
                filter_fn = fn
                break
        if filter_fn:
            filtered_trades = []
            for t in trades:
                feat = features_by_snapshot.get(t["id"], {})
                if filter_fn(feat):
                    filtered_trades.append(t)
            print(f"Feature filter '{feature_filter_name}': {len(filtered_trades)}/{len(trades)} trades pass")
            trades = filtered_trades

    if len(trades) < 5:
        print(f"Not enough trades after filter. Skipping.")
        return []

    print(f"\nGrid search: {len(grid):,} configs x {len(trades)} trades "
          f"(filter={feature_filter_name})")

    # Pre-filter ticks per token once
    filtered_ticks: dict[str, list[dict]] = {}
    for addr, raw in ticks_by_token.items():
        ft = _filter_ticks_by_source(raw, price_source)
        if ft:
            filtered_ticks[addr] = ft

    n_days = max(1, (datetime.fromisoformat(trades[-1]["created_at"].replace("Z", "+00:00")) -
                     datetime.fromisoformat(trades[0]["created_at"].replace("Z", "+00:00"))).days)

    results = []
    for i, cfg in enumerate(grid):
        strat_name = cfg["name"]
        tp_mult = cfg.get("tp_mult")
        sl_mult = cfg["sl_mult"]
        horizon = cfg["horizon"]
        profile = cfg["interval_profile"]

        pnl_list = []
        trade_results = []
        exit_analytics = []  # for exit style analysis

        for trade in trades:
            addr = trade["token_address"]
            ticks = filtered_ticks.get(addr)
            if not ticks:
                continue
            # v144: always expose the raw DS stream so dual-stream smoothing modes
            # work even when primary source=jupiter (no-op for raw smoothing).
            _raw = ticks_by_token.get(addr) or []
            _ds_stream = [t for t in _raw if t.get("source") in ("fast", "full", "live")]

            entry_price = float(trade["entry_price"])

            # Route DIP strategies through specialized replayers
            if cfg["type"] == "DIP_BUY":
                sim = _replay_dip_buy_with_intervals(
                    cfg, ticks,
                    entry_price=entry_price,
                    entry_time_iso=trade["created_at"],
                    lazy_fast_sec=cfg["lazy_fast_sec"],
                    lazy_fast_window=cfg["lazy_fast_window"],
                    lazy_slow_sec=cfg["lazy_slow_sec"],
                )
                if sim is None:
                    continue
            elif cfg["type"] == "DIP_SCALE_OUT":
                sim = _replay_dip_scale_out_with_intervals(
                    cfg, ticks,
                    entry_price=entry_price,
                    entry_time_iso=trade["created_at"],
                    lazy_fast_sec=cfg["lazy_fast_sec"],
                    lazy_fast_window=cfg["lazy_fast_window"],
                    lazy_slow_sec=cfg["lazy_slow_sec"],
                    liq_usd=trade.get("rt_liquidity_usd") or 50_000,
                )
                if sim is None:
                    continue
            else:
                fake = {
                    "id": trade["id"],
                    "entry_price": entry_price,
                    "sl_price": entry_price * sl_mult,
                    "tp_price": entry_price * tp_mult if tp_mult else None,
                    "position_usd": float(trade.get("position_usd") or 10.0),
                    "strategy": strat_name,
                    "tranche_label": "main",
                    "horizon_minutes": horizon,
                    "created_at": trade["created_at"],
                    "high_price_seen": entry_price,
                    "rt_liquidity_usd": trade.get("rt_liquidity_usd"),
                    "dex_spot_price_at_entry": float(trade.get("dex_spot_price_at_entry") or 0),
                }

                sim = _replay_with_intervals(
                    fake, ticks,
                    lazy_fast_sec=cfg["lazy_fast_sec"],
                    lazy_fast_window=cfg["lazy_fast_window"],
                    lazy_slow_sec=cfg["lazy_slow_sec"],
                    dex_ticks=_ds_stream,
                )
                if sim is None:
                    continue

            pnl_list.append(sim["pnl_pct"])
            trade_results.append({
                "pnl_pct": sim["pnl_pct"],
                "token_address": addr,
                "created_at": trade["created_at"],
            })
            exit_analytics.append(sim)

        if len(pnl_list) < 5:
            continue

        metrics = compute_metrics(pnl_list, n_days)
        br = simulate_bankroll(sorted(trade_results, key=lambda x: x["created_at"]))

        # Exit style analytics
        n_trail = sum(1 for e in exit_analytics if e["exit_reason"] == "trail_stop")
        n_sl = sum(1 for e in exit_analytics if e["exit_reason"] == "sl_hit")
        n_tp = sum(1 for e in exit_analytics if e["exit_reason"] == "tp_hit")
        n_tmo = sum(1 for e in exit_analytics if "timeout" in e["exit_reason"])
        n_early = sum(1 for e in exit_analytics if e.get("exit_in_first_5min"))
        avg_peak = statistics.mean([e["peak_from_entry"] for e in exit_analytics]) if exit_analytics else 0
        avg_peak_drop = statistics.mean([e["peak_to_exit_drop"] for e in exit_analytics]) if exit_analytics else 0

        results.append({
            "strategy": strat_name,
            "type": cfg["type"],
            "mode": profile,
            "filter": feature_filter_name,
            "horizon": horizon,
            "trail_n": n_trail, "sl_n": n_sl, "tp_n": n_tp, "tmo_n": n_tmo,
            "early_exit_n": n_early,
            "avg_peak": round(avg_peak * 100, 1),
            "avg_peak_drop": round(avg_peak_drop * 100, 1),
            **metrics, **br,
        })

        if (i + 1) % 500 == 0:
            print(f"  ... {i + 1:,}/{len(grid):,} configs done "
                  f"({len(results)} viable)")

    results.sort(key=lambda x: -x.get("final_bankroll", 0))
    print(f"Grid search complete: {len(results):,} configs with enough trades")
    return results


def _tick_validation(trades: list[dict], sim_results: dict[int, dict]) -> dict:
    """Compare tick sim vs actual paper results."""
    divergent = []
    pnl_errors = []
    exit_mismatches = 0
    sign_flips = 0

    for trade in trades:
        tid = trade["id"]
        sim = sim_results.get(tid)
        if sim is None:
            continue

        actual_pnl = float(trade.get("pnl_pct") or 0)
        sim_pnl = sim["pnl_pct"]
        delta = abs(sim_pnl - actual_pnl)
        pnl_errors.append(delta)

        actual_exit = trade.get("status", "")
        sim_exit = sim["exit_reason"]
        if actual_exit != sim_exit:
            exit_mismatches += 1

        if (actual_pnl > 0) != (sim_pnl > 0) and abs(actual_pnl) > 0.01:
            sign_flips += 1

        if delta > 0.05:  # >5% divergence
            divergent.append({
                "id": tid,
                "symbol": trade.get("symbol", "?"),
                "actual_pnl": round(actual_pnl, 4),
                "sim_pnl": round(sim_pnl, 4),
                "delta": round(delta, 4),
                "actual_exit": actual_exit,
                "sim_exit": sim_exit,
                "actual_min": trade.get("exit_minutes"),
                "sim_min": sim.get("exit_minutes"),
            })

    n = len(pnl_errors)
    mae = statistics.mean(pnl_errors) if pnl_errors else 0

    return {
        "n_compared": n,
        "mae": mae,
        "exit_mismatch_rate": exit_mismatches / n if n else 0,
        "sign_flips": sign_flips,
        "n_divergent": len(divergent),
        "divergent_trades": divergent,
    }


def _synthetic_strategy_sweep(args):
    """v134: Test NEW synthetic strategies (not in shadow DB) on post-v132 ticks.

    Takes a universe of post-Apr 13 token-calls (any existing strategy, 24h-dedup
    per token) and replays user-defined synthetic strategies — with smoothing +
    source + polling variants. Lets us grade TP70_SL30+median_5, BE20_TP70_SL30,
    FAST_TP60_SL20, etc. without waiting for shadow data to accumulate.

    Specs parsed from --synthetic-strats, each like:
      'NAME:tp=70,sl=30,horizon=30,be_act=20'
    Default horizon=120, be_act=0 (no BE).
    """
    import csv, re as _re

    SOURCES = ["jupiter", "dexscreener", "both"]
    POLL_INTERVALS = [60, 120]  # keep grid compact; 0/30 covered elsewhere
    MODES = SMOOTHING_MODES

    print("=" * 100)
    print("SYNTHETIC STRATEGY SWEEP (v134)")
    print("=" * 100)

    # Parse synthetic strategy specs
    specs: list[dict] = []
    for raw in (args.synthetic_strats or "").split(";"):
        raw = raw.strip()
        if not raw:
            continue
        if ":" not in raw:
            print(f"[warn] bad spec (expected NAME:tp=x,sl=y): {raw}")
            continue
        name, params = raw.split(":", 1)
        cfg = {"name": name.strip(), "tp": None, "sl": None, "horizon": 120, "be_act": 0}
        for kv in params.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                k, v = k.strip(), v.strip()
                if k in ("tp", "sl", "horizon", "be_act"):
                    try:
                        cfg[k] = int(v)
                    except ValueError:
                        pass
        if cfg["tp"] is None or cfg["sl"] is None:
            print(f"[warn] missing tp/sl in spec: {raw}")
            continue
        specs.append(cfg)

    if not specs:
        print("No valid synthetic strategies. Use --synthetic-strats 'NAME:tp=70,sl=30,horizon=30[;...]'")
        return

    print(f"Synthetic strategies ({len(specs)}):")
    for s in specs:
        be = f" be_act={s['be_act']}" if s['be_act'] else ""
        print(f"  {s['name']:<30s} tp={s['tp']}% sl={s['sl']}% horizon={s['horizon']}min{be}")
    print(f"Modes: {len(MODES)}  Sources: {SOURCES}  Polls: {POLL_INTERVALS}")
    print(f"-> {len(specs) * len(MODES) * len(SOURCES) * len(POLL_INTERVALS)} configs total")

    since = max(args.since, TICK_DATA_START)

    # Pull the universe of post-v132 token-calls (any strategy — we'll override).
    # FAST_TP50_SL30 shadows cover every KOL call on every token post-Apr 13.
    trades = _fetch_tick_trades(since, include_shadows=True)
    trades = [t for t in trades if t.get("strategy") == "FAST_TP50_SL30"]
    # Apply --until upper bound if set
    until = getattr(args, "until", "") or ""
    if until:
        trades = [t for t in trades if t["created_at"][:10] < until]
        print(f"Window: {since} to {until} (exclusive) = {len(trades)} raw trades")
    # Dedup per token (24h window)
    sorted_t = sorted(trades, key=lambda t: t["created_at"])
    seen: dict[str, datetime] = {}
    universe = []
    for t in sorted_t:
        addr = t["token_address"]
        dt = datetime.fromisoformat(t["created_at"].replace("Z", "+00:00"))
        last = seen.get(addr)
        if last and (dt - last).total_seconds() < 86400:
            continue
        seen[addr] = dt
        universe.append(t)
    print(f"Universe of token-calls (post-{since}, dedup 24h): {len(universe)}")

    # Fetch ticks
    token_ranges: dict[str, tuple] = {}
    for t in universe:
        addr = t["token_address"]
        entry = t["created_at"]
        horizon = max(s["horizon"] for s in specs)
        entry_dt = datetime.fromisoformat(entry.replace("Z", "+00:00"))
        end_iso = (entry_dt + timedelta(minutes=horizon + 30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        if addr not in token_ranges:
            token_ranges[addr] = (entry, end_iso)
        else:
            lo, hi = token_ranges[addr]
            token_ranges[addr] = (min(lo, entry), max(hi, end_iso))
    raw_ticks_by_token = _fetch_ticks_for_tokens(token_ranges)

    streams_by_token: dict[str, dict[str, list[dict]]] = {}
    for addr, raw in raw_ticks_by_token.items():
        streams_by_token[addr] = {src: _filter_ticks_by_source(raw, src) for src in SOURCES}

    # Run grid
    all_rows: list[dict] = []
    pnls_by_combo: dict[tuple, list[float]] = {}
    for spec in specs:
        # Synthetic strategy name must preserve BE regex match if be_act>0
        strat_name = spec["name"]
        print(f"\n=== {strat_name}  tp={spec['tp']}% sl={spec['sl']}% horizon={spec['horizon']}m"
              f"{' be_act='+str(spec['be_act']) if spec['be_act'] else ''} ===")
        for source in SOURCES:
            for poll in POLL_INTERVALS:
                for mode in MODES:
                    pnls, wins, n = [], 0, 0
                    for t in universe:
                        ticks = streams_by_token.get(t["token_address"], {}).get(source) or []
                        if not ticks:
                            continue
                        # v144: ds_stream feeds dual-stream smoothing modes (confirm/
                        # twin_confirm/hybrid) that require a parallel DexScreener tick stream.
                        _ds_stream = streams_by_token.get(t["token_address"], {}).get("dexscreener") or []
                        entry_price = float(t["entry_price"])
                        tp_price = entry_price * (1 + spec["tp"] / 100)
                        sl_price = entry_price * (1 - spec["sl"] / 100)
                        # v14e.31: position_usd chain-aware. Match shadow/paper/live
                        # ($50 for EVM via _evm_min_position_usd) so sim slip kernel
                        # (gas-as-bps formula) matches reality. SOL keeps $10 — Jupiter
                        # Ultra is near-zero slip independent of position anyway.
                        _pos_usd = 50.0 if _SIM_CHAIN in ("ethereum", "bsc", "base") else 10.0
                        fake = {
                            "id": f"{strat_name}_{t['id']}",
                            "entry_price": entry_price,
                            "sl_price": sl_price,
                            "tp_price": tp_price,
                            "position_usd": _pos_usd,
                            "strategy": strat_name,
                            "tranche_label": "main",
                            "horizon_minutes": spec["horizon"],
                            "created_at": t["created_at"],
                            "high_price_seen": entry_price,
                            "rt_liquidity_usd": t.get("rt_liquidity_usd"),
                            "dex_spot_price_at_entry": entry_price,
                        }
                        sim = _replay_with_intervals(
                            fake, ticks,
                            lazy_fast_sec=poll,
                            lazy_fast_window=poll * 10,
                            lazy_slow_sec=poll,
                            smoothing=mode,
                            dex_ticks=_ds_stream,
                        )
                        if sim is None:
                            continue
                        n += 1
                        pnls.append(sim["pnl_pct"])
                        if sim["pnl_pct"] > 0:
                            wins += 1
                    if n == 0:
                        continue
                    pnls_by_combo[(strat_name, source, poll, mode)] = pnls
                    all_rows.append({
                        "strategy": strat_name,
                        "source": source,
                        "poll_sec": poll,
                        "mode": mode,
                        "n": n,
                        "wr_pct": round(wins / n * 100, 1),
                        "avg_pnl_pct": round(statistics.mean(pnls) * 100, 2),
                        "median_pnl_pct": round(statistics.median(pnls) * 100, 2),
                        "sum_pnl_10usd": round(sum(pnls) * 10.0, 2),
                    })

    # Write CSV
    out_path = "scraper/synthetic_strategy_results.csv"
    try:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            if all_rows:
                w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
                w.writeheader()
                w.writerows(all_rows)
        print(f"\nWrote {out_path} ({len(all_rows)} rows)")
    except Exception as e:
        print(f"[warn] CSV write failed: {e}")

    # Ranking — top per synthetic strategy by avg_pnl_pct (fair, size-independent)
    print("\n" + "=" * 100)
    print("RANKED by avg_pnl_pct — top 3 combos per synthetic strategy")
    print("=" * 100)
    by_strat: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        by_strat[r["strategy"]].append(r)
    for strat, rows in by_strat.items():
        rows.sort(key=lambda r: -r["avg_pnl_pct"])
        print(f"\n[{strat}]")
        print(f"  {'src':<12s} {'poll':>4s} {'mode':<14s} {'n':>4s} {'wr%':>6s} {'avg%':>7s} {'med%':>7s}")
        for r in rows[:3]:
            print(f"  {r['source']:<12s} {r['poll_sec']:>4d}  {r['mode']:<14s} {r['n']:>4d} "
                  f"{r['wr_pct']:>5.1f}% {r['avg_pnl_pct']:>6.2f}% {r['median_pnl_pct']:>6.2f}%")

    # Overall top-10 combos (all strategies mixed) by avg_pnl_pct
    print("\n" + "=" * 100)
    print("GLOBAL TOP 10 — all synthetic strategies × configs")
    print("=" * 100)
    all_rows.sort(key=lambda r: -r["avg_pnl_pct"])
    print(f"{'strategy':<25s} {'src':<12s} {'poll':>4s} {'mode':<14s} {'n':>4s} {'wr%':>6s} {'avg%':>7s} {'med%':>7s}")
    for r in all_rows[:10]:
        print(f"{r['strategy']:<25s} {r['source']:<12s} {r['poll_sec']:>4d}  "
              f"{r['mode']:<14s} {r['n']:>4d} {r['wr_pct']:>5.1f}% "
              f"{r['avg_pnl_pct']:>6.2f}% {r['median_pnl_pct']:>6.2f}%")

    # Quick sizing note on top-1
    if all_rows:
        top = all_rows[0]
        pnls = pnls_by_combo.get((top["strategy"], top["source"], top["poll_sec"], top["mode"]), [])
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        aw = statistics.mean(wins) if wins else 0
        al = statistics.mean(losses) if losses else 0
        kelly = 0.0
        if al < 0:
            b = aw / abs(al) if abs(al) > 0 else 0
            if b > 0:
                kelly = (b * (len(wins)/len(pnls)) - (1 - len(wins)/len(pnls))) / b
        kelly = max(0.0, min(kelly, 1.0))
        print(f"\nTop combo sizing: Kelly full={kelly*100:.1f}%  half={kelly*50:.1f}%  "
              f"avg_win={aw*100:.1f}%  avg_loss={al*100:.1f}%")


def _smoothing_sweep(args):
    """v133: Grid sweep — smoothing × tick source × polling × strategy.
    Outputs ranked CSV + best-combo-per-strategy summary, including the
    current-prod config as a baseline row.
    """
    import csv
    POLL_INTERVALS = [0, 30, 60, 120]   # 0 = every tick (~15-30s)
    SOURCES = ["jupiter", "dexscreener", "both"]
    PROD_CONFIG = {
        "FAST_TP50_SL30": ("dexscreener", 60, "raw"),     # hybrid = DS decide
        "DTRAIL10_ACT15_SL70": ("dexscreener", 120, "raw"),
        "DTRAIL3_ACT5_SL60": ("jupiter", 120, "raw"),
        "DIP30_B5_T5_A20_SL70_240m": ("jupiter", 30, "raw"),
    }

    print("=" * 100)
    print("SMOOTHING GRID SWEEP (v133)")
    print(f"Smoothing modes ({len(SMOOTHING_MODES)}): {', '.join(SMOOTHING_MODES)}")
    print(f"Sources: {SOURCES}")
    print(f"Poll intervals: {POLL_INTERVALS}")
    print(f"Strats: {args.smoothing_strats}")
    total = len(SMOOTHING_MODES) * len(SOURCES) * len(POLL_INTERVALS)
    print(f"-> {total} configs per strategy")
    print("=" * 100)

    since = max(args.since, TICK_DATA_START)
    target_strats = [s.strip() for s in args.smoothing_strats.split(",") if s.strip()]

    # Include shadow trades — they expand N (e.g. FAST 14 → ~900) without
    # changing test fidelity since the sweep re-simulates exits tick-by-tick.
    # Natural filter: tokens without ticks get dropped downstream.
    trades = _fetch_tick_trades(since, include_shadows=True)
    if not trades:
        print("No trades with tick coverage. Exiting.")
        return
    trades = [t for t in trades if t.get("strategy") in target_strats]
    n_shadow = sum(1 for t in trades if t.get("is_shadow"))
    print(f"Trades for target strats: {len(trades)} ({n_shadow} shadow)")

    # Per-(strategy, token) 24h dedup — parallel strats on same token are independent
    sorted_t = sorted(trades, key=lambda t: t["created_at"])
    seen: dict[tuple, datetime] = {}
    deduped = []
    for t in sorted_t:
        key = (t["strategy"], t["token_address"])
        dt = datetime.fromisoformat(t["created_at"].replace("Z", "+00:00"))
        last = seen.get(key)
        if last and (dt - last).total_seconds() < 86400:
            continue
        seen[key] = dt
        deduped.append(t)
    trades = deduped
    print(f"Trades (per-strat dedup): {len(trades)}")
    if not trades:
        return

    # Token time ranges
    token_ranges: dict[str, tuple] = {}
    for t in trades:
        addr = t["token_address"]
        entry = t["created_at"]
        horizon = t.get("horizon_minutes", 240)
        entry_dt = datetime.fromisoformat(entry.replace("Z", "+00:00"))
        end_iso = (entry_dt + timedelta(minutes=horizon + 30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        if addr not in token_ranges:
            token_ranges[addr] = (entry, end_iso)
        else:
            lo, hi = token_ranges[addr]
            token_ranges[addr] = (min(lo, entry), max(hi, end_iso))

    raw_ticks_by_token = _fetch_ticks_for_tokens(token_ranges)
    # Pre-compute streams per source for every token (avoid re-filtering in loop)
    streams_by_token: dict[str, dict[str, list[dict]]] = {}
    for addr, raw in raw_ticks_by_token.items():
        streams_by_token[addr] = {
            src: _filter_ticks_by_source(raw, src) for src in SOURCES
        }

    # Group trades by strategy
    by_strat: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        if t["token_address"] in streams_by_token:
            by_strat[t["strategy"]].append(t)

    all_rows: list[dict] = []
    pnls_by_combo: dict[tuple, list[float]] = {}  # (strat, src, poll, mode) -> pnl_pct list
    days_span: dict[str, int] = {}
    for strat in target_strats:
        strat_trades = by_strat.get(strat, [])
        if not strat_trades:
            print(f"[{strat}] no trades with ticks, skipped")
            continue
        print(f"\n=== {strat} ({len(strat_trades)} trades) ===")
        # Compute date span for per-day aggregates
        dates = [t["created_at"][:10] for t in strat_trades]
        try:
            d0 = datetime.strptime(min(dates), "%Y-%m-%d")
            d1 = datetime.strptime(max(dates), "%Y-%m-%d")
            days_span[strat] = max(1, (d1 - d0).days + 1)
        except Exception:
            days_span[strat] = 1

        for source in SOURCES:
            for poll in POLL_INTERVALS:
                for mode in SMOOTHING_MODES:
                    pnls, pnl_usds, wins, n = [], [], 0, 0
                    for t in strat_trades:
                        ticks = streams_by_token[t["token_address"]].get(source) or []
                        if not ticks:
                            continue
                        # v143.4 — supply the DS stream for dual-stream smoothing
                        # (confirm / twin_confirm / hybrid). _replay_with_intervals
                        # ignores this arg for single-stream modes, so it is safe
                        # to pass unconditionally.
                        ds_stream = streams_by_token[t["token_address"]].get("dexscreener")
                        fake = _build_fake_trade(t)
                        sim = _replay_with_intervals(
                            fake, ticks,
                            lazy_fast_sec=poll,
                            lazy_fast_window=poll * 10 if poll > 0 else 0,
                            lazy_slow_sec=poll,
                            smoothing=mode,
                            dex_ticks=ds_stream,
                        )
                        if sim is None:
                            continue
                        n += 1
                        pnls.append(sim["pnl_pct"])
                        pnl_usds.append(sim.get("pnl_usd", 0))
                        if sim["pnl_pct"] > 0:
                            wins += 1
                    if n == 0:
                        continue
                    pnls_by_combo[(strat, source, poll, mode)] = pnls
                    all_rows.append({
                        "strategy": strat,
                        "source": source,
                        "poll_sec": poll,
                        "mode": mode,
                        "n": n,
                        "wr_pct": round(wins / n * 100, 1),
                        "avg_pnl_pct": round(statistics.mean(pnls) * 100, 2),
                        "median_pnl_pct": round(statistics.median(pnls) * 100, 2),
                        "sum_pnl_usd": round(sum(pnl_usds), 2),
                    })

    # Write CSV
    out_path = "scraper/smoothing_sweep_results.csv"
    try:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            if all_rows:
                w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
                w.writeheader()
                w.writerows(all_rows)
        print(f"\nWrote {out_path} ({len(all_rows)} rows)")
    except Exception as e:
        print(f"[warn] could not write CSV: {e}")

    # Summary: top-5 + prod baseline per strategy
    print("\n" + "=" * 100)
    print("RANKED — top 5 per strategy + prod baseline")
    print("=" * 100)
    by_strat_rows: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        by_strat_rows[r["strategy"]].append(r)

    for strat, rows in by_strat_rows.items():
        rows.sort(key=lambda r: -r["sum_pnl_usd"])
        prod_src, prod_poll, prod_mode = PROD_CONFIG.get(strat, ("jupiter", 0, "raw"))
        prod_row = next(
            (r for r in rows if r["source"] == prod_src and r["poll_sec"] == prod_poll
             and r["mode"] == prod_mode),
            None,
        )
        prod_pnl = prod_row["sum_pnl_usd"] if prod_row else 0.0
        print(f"\n[{strat}]  prod: src={prod_src} poll={prod_poll}s mode={prod_mode}  ->  ${prod_pnl:.2f}")
        print(f"  {'src':<12s} {'poll':>4s}s {'mode':<14s} {'n':>4s} {'wr%':>6s} {'avg%':>7s} {'sumPnL$':>10s}  {'d_vs_prod':>10s}")
        for r in rows[:5]:
            delta = r["sum_pnl_usd"] - prod_pnl
            sign = "+" if delta >= 0 else ""
            print(f"  {r['source']:<12s} {r['poll_sec']:>4d}  {r['mode']:<14s} {r['n']:>4d} "
                  f"{r['wr_pct']:>5.1f}% {r['avg_pnl_pct']:>6.2f}% {r['sum_pnl_usd']:>9.2f}$  {sign}{delta:>8.2f}")

    # ---------- Position sizing / Kelly / Monte Carlo on the top-1 per strat ---
    print("\n" + "=" * 100)
    print("SIZING ANALYSIS — top-1 combo per strategy")
    print("=" * 100)
    POS_SIZES = [3, 10, 25, 50, 100, 200]

    for strat, rows in by_strat_rows.items():
        rows.sort(key=lambda r: -r["sum_pnl_usd"])
        top = rows[0]
        key = (strat, top["source"], top["poll_sec"], top["mode"])
        pnls = pnls_by_combo.get(key, [])
        if not pnls:
            continue
        n = len(pnls)
        n_days = days_span.get(strat, 1)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        wr = len(wins) / n if n else 0
        avg_win = statistics.mean(wins) if wins else 0
        avg_loss = statistics.mean(losses) if losses else 0
        # Kelly fraction: f = (b*p - q) / b, where b=avg_win_abs / avg_loss_abs, p=wr, q=1-wr
        kelly = 0.0
        if avg_loss < 0:
            b = avg_win / abs(avg_loss) if abs(avg_loss) > 0 else 0
            if b > 0:
                kelly = (b * wr - (1 - wr)) / b
        kelly = max(0.0, min(kelly, 1.0))
        kelly_half = kelly / 2  # half-kelly (industry standard for noisy edges)

        print(f"\n[{strat}]  best: src={top['source']} poll={top['poll_sec']}s mode={top['mode']}")
        print(f"  N={n}  span={n_days}d  WR={wr*100:.1f}%  avg_win=+{avg_win*100:.2f}%  avg_loss={avg_loss*100:.2f}%")
        print(f"  Kelly full={kelly*100:.1f}%   Kelly half={kelly_half*100:.1f}%")

        # Daily gain at fixed position sizes (actual 6-day results scaled)
        total_pnl_pct = sum(pnls)
        print(f"  {'pos_usd':>8s}  {'total$':>10s}  {'$/day':>10s}  {'$/trade':>10s}")
        for pos in POS_SIZES:
            tot = pos * total_pnl_pct
            per_day = tot / n_days
            per_trade = tot / n
            print(f"  ${pos:>6d}  {tot:>10.2f}  {per_day:>10.2f}  {per_trade:>10.2f}")

        # Monte Carlo (bootstrap): simulate 200 trades, compound at position = Kelly-half * bankroll (cap $200)
        if n >= 10:
            import random as _r
            n_sims = 1000
            finals = []
            ruins = 0
            max_dds = []
            bankroll0 = 1000.0
            cap = 200.0
            for _ in range(n_sims):
                b = bankroll0
                peak = b
                dd = 0.0
                bust = False
                for _ in range(200):
                    pnl = _r.choice(pnls)
                    pos = min(b * kelly_half, cap)
                    if pos < 1.0:
                        bust = True
                        break
                    b += pos * pnl
                    if b > peak:
                        peak = b
                    d = (peak - b) / peak
                    if d > dd:
                        dd = d
                if bust:
                    ruins += 1
                finals.append(b)
                max_dds.append(dd)
            finals.sort()
            p5 = finals[int(n_sims * 0.05)]
            p50 = finals[n_sims // 2]
            p95 = finals[int(n_sims * 0.95)]
            print(f"  Monte Carlo (200 trades, half-Kelly, start=${bankroll0:.0f}): "
                  f"p5=${p5:.0f}  p50=${p50:.0f}  p95=${p95:.0f}  "
                  f"avg_max_dd={statistics.mean(max_dds)*100:.1f}%  ruin_rate={ruins/n_sims*100:.1f}%")


def _eval_history_simulation(args):
    """v138 B: replay each trade from its persisted eval_history.
    Perfect 0% sim/real alignment. Auditing tool: replays the EXACT polls
    paper_trader did. Useful to validate bug fixes in _evaluate_trade_exit
    against historical reality without any reconstruction."""
    print("=" * 90)
    print("FROM-EVAL-HISTORY MODE (v138 B): perfect replay from logged polls")
    print("=" * 90)

    rows = sb_get("paper_trades", [
        ("select", "id,token_address,symbol,strategy,tranche_label,entry_price,"
                   "sl_price,tp_price,horizon_minutes,position_usd,"
                   "rt_liquidity_usd,dex_spot_price_at_entry,created_at,"
                   "status,pnl_pct,eval_history,chain"),
        ("status", "in.(trail_stop,sl_hit,timeout,tp_hit)"),
        ("source", "eq.rt"),
        ("created_at", f"gte.{args.since}T00:00:00Z"),
        ("eval_history", "not.is.null"),
        ("order", "created_at.asc"),
    ] + _chain_params())
    print(f"Fetched {len(rows)} closed trades with eval_history (chain={_SIM_CHAIN}) since {args.since}")
    if not rows:
        print("No trades have eval_history yet — only trades closed after v138 deploy "
              "carry the field. Wait 24-48h for the first batch.")
        return

    if args.strategies:
        wanted = {s.strip() for s in args.strategies.split(",") if s.strip()}
        rows = [r for r in rows if r.get("strategy") in wanted]
        print(f"After --strategies filter: {len(rows)} trades")

    from collections import defaultdict
    deltas_by_strat = defaultdict(list)
    sims_by_strat = defaultdict(list)
    for r in rows:
        eh = r.get("eval_history") or []
        if isinstance(eh, str):
            import json
            eh = json.loads(eh)
        if not eh:
            continue
        fake = {
            "id": r["id"], "entry_price": float(r["entry_price"]),
            "sl_price": float(r["sl_price"]),
            "tp_price": float(r["tp_price"]) if r.get("tp_price") else None,
            "position_usd": float(r.get("position_usd") or 10),
            "strategy": r["strategy"],
            "tranche_label": r.get("tranche_label", "main"),
            "horizon_minutes": r.get("horizon_minutes", 120),
            "created_at": r["created_at"],
            "high_price_seen": float(r["entry_price"]),
            "rt_liquidity_usd": r.get("rt_liquidity_usd"),
            "dex_spot_price_at_entry": float(r.get("dex_spot_price_at_entry")
                                              or r["entry_price"]),
        }
        sim = _replay_from_eval_history(fake, eh)
        if sim is None:
            continue
        real_pct = float(r["pnl_pct"]) * 100 if r.get("pnl_pct") is not None else 0
        sim_pct = sim["pnl_pct"] * 100
        deltas_by_strat[r["strategy"]].append(sim_pct - real_pct)
        sims_by_strat[r["strategy"]].append(sim_pct)

    print(f"\n{'Strategy':<28}{'N':>4}{'sim_avg':>10}{'bias_vs_real':>14}{'MAE':>10}")
    for strat in sorted(sims_by_strat):
        n = len(sims_by_strat[strat])
        if n < 3:
            continue
        sa = statistics.mean(sims_by_strat[strat])
        bm = statistics.mean(deltas_by_strat[strat])
        mae = statistics.mean(abs(d) for d in deltas_by_strat[strat])
        print(f"{strat:<28}{n:>4}{sa:>+9.2f}%{bm:>+13.2f}%{mae:>9.2f}%")
    print("\nNote: bias should be 0.00% for trades closed AFTER v138 deploy "
          "(eval_history captures every poll perfectly). Non-zero bias on a "
          "trade indicates a bug in _evaluate_trade_exit changes since the "
          "trade closed.")


def _tick_based_simulation(args):
    """FROM-TICKS: replay price ticks through _evaluate_trade_exit."""
    print("=" * 90)
    print("FROM-TICKS MODE: Tick-level replay simulation (30s resolution)")
    print(f"Price source: {args.price_source}")
    if getattr(args, "from_live_config", False):
        print("  [v132] --from-live-config: loading strategy_overrides from DB")
    if getattr(args, "priority_fee_sol", 0) > 0:
        print(f"  [v132] priority-fee-sol: {args.priority_fee_sol} SOL / round-trip "
              f"(~${args.priority_fee_sol*150:.2f})")
    print("=" * 90)

    # v132: Load live overrides once if requested
    _live_overrides = _load_live_strategy_overrides() if getattr(args, "from_live_config", False) else {}
    if _live_overrides:
        for s, ov in _live_overrides.items():
            print(f"    {s:30s} poll={ov.get('polling_sec','-'):>3}s  source={ov.get('price_source','-')}")

    since = max(args.since, TICK_DATA_START)

    # 1. Fetch trades
    trades = _fetch_tick_trades(since)
    if not trades:
        print("No trades with tick coverage. Exiting.")
        return

    # v126: Align dedup with paper/live cooldown (24h sliding window per token).
    # v125 used a naive "first trade per token" which removed legitimate re-calls
    # >24h apart. Now matches paper_trader dedup_cooldown_h=24 exactly.
    deduped = dedup_first_call(trades)
    print(f"After 24h cooldown dedup: {len(deduped)} unique token-calls (was {len(trades)})")

    sim_trades = deduped

    # 2. Build token time ranges
    token_ranges: dict[str, tuple] = {}
    for t in sim_trades:
        addr = t["token_address"]
        entry = t["created_at"]
        # Buffer: horizon + 30 min after exit
        horizon = t.get("horizon_minutes", 120)
        entry_dt = datetime.fromisoformat(entry.replace("Z", "+00:00"))
        end_dt = entry_dt + timedelta(minutes=horizon + 30)
        end_iso = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        if addr not in token_ranges:
            token_ranges[addr] = (entry, end_iso)
        else:
            old_start, old_end = token_ranges[addr]
            token_ranges[addr] = (min(old_start, entry), max(old_end, end_iso))

    # 3. Fetch ticks
    ticks_by_token = _fetch_ticks_for_tokens(token_ranges)

    # 4. Replay each trade
    print(f"\nReplaying {len(sim_trades)} trades on ticks...")
    sim_results: dict[int, dict] = {}  # trade_id -> sim result
    by_strategy: dict[str, list[float]] = defaultdict(list)
    by_strategy_trades: dict[str, list[dict]] = defaultdict(list)
    skipped = 0

    for trade in sim_trades:
        addr = trade["token_address"]
        raw_ticks = ticks_by_token.get(addr)
        if not raw_ticks:
            skipped += 1
            continue

        ticks = _filter_ticks_by_source(raw_ticks, args.price_source)
        if not ticks:
            skipped += 1
            continue

        fake = _build_fake_trade(trade, sim_live_entry=args.sim_live_entry)

        # v132: --from-live-config — per-trade orchestration from prod DB
        strat = trade.get("strategy", "")
        if getattr(args, "from_live_config", False):
            override = _live_overrides.get(strat, {}) if _live_overrides else {}
            orch_mode = override.get("price_source")
            poll_sec = int(override.get("polling_sec", 0) or 0)
            ema_w = int(override.get("ema_window", 3) or 3)
        else:
            orch_mode = getattr(args, "orchestration", None)
            poll_sec = int(getattr(args, "poll_sec", 0) or 0)
            ema_w = int(getattr(args, "ema_window", 3) or 3)

        if orch_mode:
            # v137.1: pass RAW ticks (not pre-subsampled) so _replay_trade_orchestrated
            # can do its own grid-based polling + paper-only jupiter filter. The old
            # _filter_ticks_by_source subsample clashed with realistic polling and
            # dropped the source attribution needed for paper/live separation.
            ds_ticks = [t for t in raw_ticks if t.get("source") in ("fast", "full", "live")]
            jp_ticks = [t for t in raw_ticks if t.get("source") == "jupiter"]
            sim = _replay_trade_orchestrated(
                fake, ds_ticks, jp_ticks,
                orchestration=orch_mode,
                poll_sec=poll_sec,
                ema_window=ema_w,
            )
        else:
            sim = _replay_trade_on_ticks(fake, ticks)
        if sim is None:
            skipped += 1
            continue

        # v132: Priority fee cost model (applied to closed trades only)
        if getattr(args, "priority_fee_sol", 0) > 0:
            sol_px = 150.0  # approx
            fee_usd = float(args.priority_fee_sol) * sol_px
            pos = float(fake.get("position_usd") or 10.0)
            if pos > 0:
                sim["pnl_pct"] = sim.get("pnl_pct", 0) - fee_usd / pos
                sim["pnl_usd"] = sim.get("pnl_usd", 0) - fee_usd

        sim_results[trade["id"]] = sim
        strat = trade["strategy"]
        by_strategy[strat].append(sim["pnl_pct"])
        by_strategy_trades[strat].append({
            "pnl_pct": sim["pnl_pct"],
            "token_address": addr,
            "created_at": trade["created_at"],
        })

    print(f"Simulated: {len(sim_results)} trades  |  Skipped (no ticks): {skipped}")

    # 5. Report — per strategy
    dates = [t["created_at"][:10] for t in sim_trades if t["id"] in sim_results]
    n_days = max(1, (datetime.strptime(max(dates), "%Y-%m-%d") -
                     datetime.strptime(min(dates), "%Y-%m-%d")).days + 1) if dates else 1

    ranked = []
    for strat_name, pnl_list in by_strategy.items():
        if len(pnl_list) < 3:
            continue
        metrics = compute_metrics(pnl_list, n_days)
        br_trades = by_strategy_trades[strat_name]
        br = simulate_bankroll(sorted(br_trades, key=lambda x: x["created_at"]))
        ranked.append({"name": strat_name, **metrics, **br})

    ranked.sort(key=lambda x: -x.get("final_bankroll", 0))

    print(f"\n{'=' * 100}")
    print(f"TICK SIM RESULTS — {args.price_source.upper()} price source, {len(sim_results)} trades, {n_days} days")
    print(f"{'=' * 100}")
    header = (f"{'Rank':>4s}  {'Strategy':40s} {'N':>5s} {'WR%':>5s} "
              f"{'AvgPnL%':>8s} {'Sharpe':>7s} {'MaxDD%':>7s} {'Final$':>9s}")
    print(header)
    print("-" * len(header))
    for i, r in enumerate(ranked):
        print(f"{i+1:4d}  {r['name']:40s} {r['n_trades']:5d} {r['wr_pct']:4.0f}% "
              f"{r['avg_pnl_pct']:+7.1f}% {r['sharpe']:7.2f} {r['max_dd_pct']:6.1f}% "
              f"$ {r['final_bankroll']:8.0f}")

    # Monte Carlo on top 3
    print(f"\n{'=' * 100}")
    print(f"MONTE CARLO (top 3, {args.mc_sims} sims)")
    print(f"{'=' * 100}")
    mc_header = f"{'Strategy':45s} {'Median$':>8s} {'P5$':>8s} {'P25$':>8s} {'P75$':>8s} {'P95$':>8s}"
    print(mc_header)
    print("-" * len(mc_header))
    for r in ranked[:3]:
        pnl_list = by_strategy.get(r["name"], [])
        mc = monte_carlo(pnl_list, args.mc_sims, min(args.mc_trades, len(pnl_list)))
        if mc:
            print(f"{r['name']:45s} $ {mc['median']:6.0f} $ {mc['p5']:6.0f} "
                  f"$ {mc['p25']:6.0f} $ {mc['p75']:6.0f} $ {mc['p95']:6.0f}")

    # 6. Validation mode
    if args.validate_ticks:
        print(f"\n{'=' * 100}")
        print("VALIDATION: Tick Sim vs Actual Paper Results")
        print(f"{'=' * 100}")
        val = _tick_validation(sim_trades, sim_results)
        print(f"Trades compared: {val['n_compared']}")
        print(f"MAE (pnl_pct):   {val['mae']:.4f} ({val['mae']*100:.1f}%)")
        print(f"Exit mismatch:   {val['exit_mismatch_rate']*100:.1f}%")
        print(f"Sign flips:      {val['sign_flips']}")
        print(f"Divergent (>5%): {val['n_divergent']}")
        if val["divergent_trades"]:
            print(f"\n{'Symbol':12s} {'Actual':>8s} {'Sim':>8s} {'Delta':>7s} "
                  f"{'ActExit':12s} {'SimExit':12s} {'ActMin':>7s} {'SimMin':>7s}")
            print("-" * 80)
            for d in val["divergent_trades"][:20]:
                print(f"{d['symbol']:12s} {d['actual_pnl']:+7.3f} {d['sim_pnl']:+7.3f} "
                      f"{d['delta']:6.3f}  {d['actual_exit']:12s} {d['sim_exit']:12s} "
                      f"{str(d.get('actual_min','')):>7s} {str(d.get('sim_min','')):>7s}")

    # 7. Grid search
    if args.grid_ticks:
        eligible_trades = [t for t in sim_trades if t["id"] in sim_results]

        # Fetch on-chain features via token_address + time match
        features_by_trade = _fetch_snapshot_features_for_trades(eligible_trades)

        # Run grid for ALL trades first (main grid)
        print(f"\n{'=' * 120}")
        print("GRID SEARCH — All strategies x intervals on tick data")
        print(f"{'=' * 120}")
        grid_results = _tick_grid_search(
            eligible_trades, ticks_by_token, args.price_source, args.mc_sims,
            features_by_trade, "ALL")

        # Print top 40 with exit analytics
        print(f"\n{'Rank':>4s}  {'Strategy':28s} {'Type':7s} {'Mode':10s} {'H':>4s} {'N':>4s} {'WR%':>5s} "
              f"{'AvgPnL%':>8s} {'Shrp':>5s} {'DD%':>5s} {'Final$':>8s} "
              f"{'TRL':>3s} {'SL':>3s} {'TMO':>3s} {'Erly':>4s} {'Peak%':>5s}")
        print("-" * 130)
        for i, r in enumerate(grid_results[:40]):
            print(f"{i+1:4d}  {r['strategy']:28s} {r.get('type','?'):7s} {r.get('mode','?'):10s} "
                  f"{r['horizon']:4d} {r['n_trades']:4d} "
                  f"{r['wr_pct']:4.0f}% {r['avg_pnl_pct']:+7.1f}% {r['sharpe']:5.2f} "
                  f"{r['max_dd_pct']:4.0f}% ${r['final_bankroll']:7.0f} "
                  f"{r.get('trail_n',0):3d} {r.get('sl_n',0):3d} {r.get('tmo_n',0):3d} "
                  f"{r.get('early_exit_n',0):4d} {r.get('avg_peak',0):4.0f}%")

        # Best per type
        print(f"\n{'=' * 120}")
        print("BEST PER STRATEGY TYPE")
        print(f"{'=' * 120}")
        seen_types = set()
        for r in grid_results:
            key = r.get("type", "?")
            if key not in seen_types:
                seen_types.add(key)
                print(f"  {key:12s}  {r['strategy']:30s} {r.get('mode',''):10s} H={r['horizon']:3d} "
                      f"WR={r['wr_pct']:.0f}% AvgPnL={r['avg_pnl_pct']:+.1f}% "
                      f"Sharpe={r['sharpe']:.2f} Final=${r['final_bankroll']:.0f}")

        # =================================================================
        # PASS 2: Cross top 50 strategies × features × trigger modes
        # =================================================================
        _interval_map = {
            "CURRENT": (0,0,0), "FAST_15": (15,60,60), "FAST_30": (30,120,120),
            "LAZY_FAST": (60,120,180), "LAZY_MED": (120,300,360),
            "LAZY_STD": (180,300,600), "LAZY_SLOW": (300,600,900),
            "LAZY_XSLOW": (600,900,1200),
        }
        _re_mod = __import__("re")

        top_n_cross = 50
        top_configs = grid_results[:top_n_cross]

        # --- Helper: replay a config on a set of trades ---
        def _run_config_on_trades(cfg_r, trade_list, source, trigger_mode="polling"):
            """Run one strategy config on a list of trades.
            trigger_mode: 'polling' | 'trigger_sl' | 'trigger_trail' | 'trigger_sl_only'"""
            _sl_m = _re_mod.search(r"SL(\d+)", cfg_r["strategy"])
            _sl_v = int(_sl_m.group(1)) if _sl_m else 40
            fs, fw, ss = _interval_map.get(cfg_r.get("mode", "CURRENT"), (0,0,0))
            pnl_list = []
            tr_list = []
            for t in trade_list:
                addr = t["token_address"]
                raw = ticks_by_token.get(addr)
                if not raw:
                    continue
                tks = _filter_ticks_by_source(raw, source)
                if not tks:
                    continue
                # v144: ds_stream for dual-stream smoothing (confirm/twin_confirm/hybrid)
                _ds_stream = [t2 for t2 in raw if t2.get("source") in ("fast", "full", "live")]
                ep = float(t["entry_price"])
                fake = {
                    "id": t["id"], "entry_price": ep,
                    "sl_price": ep * (1 - _sl_v / 100),
                    "tp_price": None,
                    "position_usd": float(t.get("position_usd") or 10.0),
                    "strategy": cfg_r["strategy"],
                    "tranche_label": "main",
                    "horizon_minutes": cfg_r["horizon"],
                    "created_at": t["created_at"],
                    "high_price_seen": ep,
                    "rt_liquidity_usd": t.get("rt_liquidity_usd"),
                    "dex_spot_price_at_entry": float(t.get("dex_spot_price_at_entry") or 0),
                }
                if trigger_mode == "polling":
                    sim = _replay_with_intervals(fake, tks, fs, fw, ss, dex_ticks=_ds_stream)
                elif trigger_mode == "trigger_trail":
                    sim = _replay_with_trigger(fake, tks)
                elif trigger_mode == "trigger_sl_only":
                    # Trigger for SL only (no trail PATCH), polling for trail
                    sim = _replay_with_trigger_sl_only(fake, tks, fs, fw, ss)
                else:
                    sim = _replay_with_intervals(fake, tks, fs, fw, ss, dex_ticks=_ds_stream)
                if sim:
                    pnl_list.append(sim["pnl_pct"])
                    tr_list.append({"pnl_pct": sim["pnl_pct"], "token_address": addr,
                                    "created_at": t["created_at"]})
            if len(pnl_list) < 3:
                return None
            wr = sum(1 for p in pnl_list if p > 0) / len(pnl_list) * 100
            avg = statistics.mean(pnl_list) * 100
            br = simulate_bankroll(sorted(tr_list, key=lambda x: x["created_at"]))
            return {"n": len(pnl_list), "wr": wr, "avg_pnl": avg,
                    "final": br["final_bankroll"], "max_dd": br["max_dd_pct"]}

        print(f"\n{'=' * 130}")
        print(f"PASS 2: Top {top_n_cross} strategies × 12 filters × 4 trigger modes × 2 price sources")
        print(f"{'=' * 130}")

        # --- A. Feature × Strategy cross matrix ---
        if features_by_trade:
            print(f"\n--- A. STRATEGY × FEATURE FILTER CROSS ---")
            print(f"\n{'Rank':>4s}  {'Strategy':28s} {'Mode':10s} {'Filter':15s} {'N':>4s} {'WR%':>5s} "
                  f"{'AvgPnL%':>8s} {'Final$':>8s} {'DD%':>5s}")
            print("-" * 110)

            cross_results = []
            for cfg_r in top_configs[:20]:  # top 20 × 12 filters = 240 combos
                for fname, ffn in FEATURE_FILTERS:
                    if fname == "ALL":
                        continue  # already have this from pass 1
                    ftrades = [t for t in eligible_trades
                               if ffn(features_by_trade.get(t["id"], {}))]
                    r = _run_config_on_trades(cfg_r, ftrades, args.price_source)
                    if r:
                        cross_results.append({
                            "strategy": cfg_r["strategy"], "mode": cfg_r["mode"],
                            "horizon": cfg_r["horizon"], "filter": fname, **r})

            cross_results.sort(key=lambda x: -x["final"])
            for i, r in enumerate(cross_results[:30]):
                print(f"{i+1:4d}  {r['strategy']:28s} {r['mode']:10s} {r['filter']:15s} "
                      f"{r['n']:4d} {r['wr']:4.0f}% {r['avg_pnl']:+7.1f}% "
                      f"${r['final']:7.0f} {r['max_dd']:4.0f}%")

        # --- B. Jupiter vs DexScreener on top 10 ---
        print(f"\n--- B. PRICE SOURCE: Jupiter vs DexScreener (top 10) ---")
        print(f"\n  {'Strategy':28s} {'Mode':10s} {'Jup$':>8s} {'Dex$':>8s} {'Delta':>7s} {'JupWR':>5s} {'DexWR':>5s}")
        print(f"  {'-'*80}")
        for cfg_r in top_configs[:10]:
            rj = _run_config_on_trades(cfg_r, eligible_trades, "jupiter")
            rd = _run_config_on_trades(cfg_r, eligible_trades, "dexscreener")
            if rj and rd:
                delta = rj["final"] - rd["final"]
                print(f"  {cfg_r['strategy']:28s} {cfg_r['mode']:10s} "
                      f"${rj['final']:7.0f} ${rd['final']:7.0f} "
                      f"{'+'if delta>=0 else ''}{delta:6.0f} {rj['wr']:4.0f}% {rd['wr']:4.0f}%")

        # --- C. Trigger V2 modes on top 10 DTRAIL ---
        print(f"\n--- C. TRIGGER V2 MODES (top 10 DTRAIL) ---")
        print(f"  Modes: polling (check+sell) | trigger_trail (SL+PATCH trail on-chain) | trigger_sl_only (SL on-chain, trail via polling)")
        print(f"\n  {'Strategy':28s} {'Mode':10s} {'Polling$':>8s} {'TrigTrail$':>9s} {'TrigSL$':>9s} {'Best':>10s}")
        print(f"  {'-'*90}")
        top_dtrails = [r for r in grid_results if r["type"] == "DTRAIL"][:10]
        for cfg_r in top_dtrails:
            rp = _run_config_on_trades(cfg_r, eligible_trades, args.price_source, "polling")
            rt = _run_config_on_trades(cfg_r, eligible_trades, args.price_source, "trigger_trail")
            rs = _run_config_on_trades(cfg_r, eligible_trades, args.price_source, "trigger_sl_only")
            if rp and rt and rs:
                vals = {"polling": rp["final"], "trig_trail": rt["final"], "trig_sl": rs["final"]}
                best_mode = max(vals, key=vals.get)
                print(f"  {cfg_r['strategy']:28s} {cfg_r['mode']:10s} "
                      f"${rp['final']:7.0f} ${rt['final']:8.0f} ${rs['final']:8.0f} "
                      f"  {best_mode}")
            elif rp and rt:
                print(f"  {cfg_r['strategy']:28s} {cfg_r['mode']:10s} "
                      f"${rp['final']:7.0f} ${rt['final']:8.0f}       —")

        # Save grid CSV
        csv_path = SCRAPER_DIR / "grid_search_ticks.csv"
        with open(csv_path, "w", newline="") as f:
            if grid_results:
                writer = csv.DictWriter(f, fieldnames=list(grid_results[0].keys()))
                writer.writeheader()
                writer.writerows(grid_results)
        print(f"\nGrid CSV saved to: {csv_path}")

    # 8. Per-trade CSV export
    if args.tick_csv:
        csv_path = Path(args.tick_csv)
        rows = []
        for trade in sim_trades:
            sim = sim_results.get(trade["id"])
            if sim is None:
                continue
            rows.append({
                "id": trade["id"],
                "symbol": trade.get("symbol", ""),
                "strategy": trade["strategy"],
                "entry_price": trade["entry_price"],
                "actual_pnl": trade.get("pnl_pct"),
                "sim_pnl": sim["pnl_pct"],
                "actual_exit": trade.get("status"),
                "sim_exit": sim["exit_reason"],
                "actual_minutes": trade.get("exit_minutes"),
                "sim_minutes": sim["exit_minutes"],
                "sim_high": sim["high_price_seen"],
                "actual_high": trade.get("high_price_seen"),
                "created_at": trade["created_at"],
            })
        if rows:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"\nPer-trade CSV saved to: {csv_path}")

    print(f"\n{'=' * 100}")
    print(f"Tick sim complete. {len(sim_results)} trades simulated, {skipped} skipped.")


# ---------------------------------------------------------------------------
# v140: MEGA SWEEP — strategies × filters × sources × smoothings × polling.
# Uses multiprocessing to scale 134K configs in ~30-45min.
# Per-worker loads ticks from temp JSON file, runs _evaluate_trade_exit
# with v138.5-calibrated dynamic slippage.
# ---------------------------------------------------------------------------
_MEGA_TOP_KOLS = {"FrenzGems", "jadendegens", "gubbinscalls", "Archerrgambles",
                  "ChadleyGambles123", "zcallz"}

# NEW v139/v140 strategy variants beyond STRATEGIES dict (TP200/TP300/TP500)
_MEGA_NEW_STRATS = {
    "TP150_SL40_2H":      (2.50, 0.60, 120, None),
    "TP150_SL40_4H":      (2.50, 0.60, 240, None),
    "TP200_SL30_2H":      (3.00, 0.70, 120, None),
    "TP200_SL30_4H":      (3.00, 0.70, 240, None),
    "TP200_SL40_2H":      (3.00, 0.60, 120, None),
    "TP200_SL40_4H_v":    (3.00, 0.60, 240, None),
    "TP200_SL50_4H":      (3.00, 0.50, 240, None),
    "TP300_SL40_4H":      (4.00, 0.60, 240, None),
    "TP300_SL50_4H":      (4.00, 0.50, 240, None),
    "TP500_SL50_4H":      (6.00, 0.50, 240, None),
    "FAST_TP200_SL40":    (3.00, 0.60, 60,  None),
    "BE15_TP200_SL40_4H": (3.00, 0.60, 240, 0.15),
    "BE25_TP200_SL40_4H": (3.00, 0.60, 240, 0.25),
    "BE50_TP200_SL30_4H": (3.00, 0.70, 240, 0.50),
    "BE15_TP300_SL50_4H": (4.00, 0.50, 240, 0.15),
}

_MEGA_SOURCES = ["jupiter", "dexscreener"]
_MEGA_SMOOTHINGS = ["raw", "ema_fast", "ema_slow", "median_3", "median_5",
                    "winsor_p95", "dual_confirm", "hysteresis"]
_MEGA_POLLING_MODES = ["fast", "static_60", "static_120", "static_240", "lazy"]
_MEGA_FILTERS = ["NONE", "NOZEROLIQ", "SCORE30", "SCORE40", "MCAP_MID",
                 "TOPKOL", "NOZEROLIQ_SCORE30"]

# --- v144 EXTENDED MEGA SWEEP ---
# Fires via --mega-sweep-extended. Meant to run several hours (half a day).
# Adds: source=both (merged streams), 4 LAZY cadence variants, volume_gated
# smoothing, 5 new filters (finer score thresholds + filter combos).
# NOT YET added (need dual-stream tick-window logic, deferred patch):
# confirm, twin_confirm, hybrid, vwap_5min, ohlc_burst_60s, jp_sampled_{60,180}s.
_MEGA_EXT_SOURCES = ["jupiter", "dexscreener", "both"]
_MEGA_EXT_SMOOTHINGS = ["raw", "ema_fast", "ema_slow", "median_3", "median_5",
                        "winsor_p95", "dual_confirm", "hysteresis", "volume_gated"]
_MEGA_EXT_POLLING_MODES = ["fast", "static_30", "static_60", "static_120", "static_240",
                           "lazy_fast", "lazy_med", "lazy", "lazy_slow", "lazy_xslow"]
_MEGA_EXT_FILTERS = ["NONE", "NOZEROLIQ", "SCORE30", "SCORE35", "SCORE40",
                     "SCORE45", "SCORE50", "MCAP_MID", "TOPKOL",
                     "NOZEROLIQ_SCORE30", "NOZEROLIQ_SCORE40", "MCAP_MID_SCORE40",
                     # v14e.43 — BSR gates. ⚠ BSR seul perd $/d (validation
                     # empirique post-add). Gardés pour confirmation out-of-
                     # sample au sweep level. À retirer après prochain sweep.
                     "BSR52", "BSR55", "NOZEROLIQ_BSR52", "NOZEROLIQ_BSR55",
                     # v14e.43b — KW (kol_win_rate) gates, validated walk-
                     # forward sur target $/d : SOL train +$777 → test +$222,
                     # ETH train +$41 → test +$440. Cross-chain robuste.
                     "KW34", "KW26", "NOZEROLIQ_KW34", "NOZEROLIQ_KW26",
                     # v14e.43b — BSR_MCAP combo: BSR>=0.53 AND mcap>=$45K
                     # est le top combo per-strat sur SOL losers (+$20-26/d)
                     "BSR_MCAP",
                     # v14e.72 — sentiment band (inverted U, so a BAND not a
                     # threshold) + KOL call cadence. Both permutation-validated.
                     "SENT30_70", "SENT45_65", "SENT50_60", "SENT_NOHYPE",
                     "GAP24", "NOBURST", "SENT30_70_GAP24",
                     ]

# v14e.27 — token-age dimension. Default scrape gate is 12h (safe_scraper +
# pipeline). The age sweep tests whether relaxing to 24h or 48h opens an edge:
#   "ALL"   — no age cap (use whatever rt_token_age_hours the trade carried)
#   "AGE12" — token <= 12h at entry (current default — equivalent to no filter
#             for the historical universe, since the global gate already capped
#             at 12h on most days; included for forward compat when the global
#             gate is relaxed).
#   "AGE24" — disjoint band 12-24h
#   "AGE48" — disjoint band 24-48h
# Disjoint bands match strategies.py AGE24_/AGE48_ filter convention so the
# sweep result maps 1:1 onto the deployed shadows.
_MEGA_EXT_AGE_BANDS = ["ALL", "AGE12", "AGE24", "AGE48"]
_MEGA_AGE_BANDS = ["ALL"]   # base sweep stays 1-band for backwards compat

_MEGA_LOOP_SEC = 30
_MEGA_LAZY_FAST_SEC = 180
_MEGA_LAZY_FAST_WINDOW = 300
_MEGA_LAZY_SLOW_SEC = 600

# Worker-global ticks cache (loaded by _mega_init_worker from JSON)
_MEGA_TICKS = None
_MEGA_DAY_REGIME = {}  # v14e.26: {date_str: "active"/"quiet"/"dead"}
_MEGA_TEST_DAYS = set()  # v14e.26: walk-forward test set (last 3 days)
_MEGA_ROLLING_WINDOWS = {}  # v14e.49: {window_label: set(date_str) for last 14d/7d/3d}


def _mega_init_worker(ticks_path, day_regime_path=None, sim_chain="solana"):
    """multiprocessing initializer — loads ticks JSON once per worker process.

    v14e.31: also propagates _SIM_CHAIN so workers spawned via 'spawn' method
    (Windows) get the right chain. On 'fork' (Linux, GH Actions) state is
    inherited from parent so this is redundant — but cheap to be explicit.
    """
    import json
    global _MEGA_TICKS, _MEGA_DAY_REGIME, _MEGA_TEST_DAYS, _MEGA_ROLLING_WINDOWS, _SIM_CHAIN
    with open(ticks_path) as f:
        _MEGA_TICKS = json.load(f)
    if day_regime_path:
        with open(day_regime_path) as f:
            _meta = json.load(f)
        _MEGA_DAY_REGIME = _meta.get("day_regime", {})
        _MEGA_TEST_DAYS = set(_meta.get("test_days", []))
        # v14e.49: rolling time-window buckets {label: set(date)} — answers
        # "did this strat keep working in the last 3d?" alongside the regime
        # split which answers "does it work on bad-vibe days?".
        _MEGA_ROLLING_WINDOWS = {
            k: set(v) for k, v in _meta.get("rolling_windows", {}).items()
        }
    _SIM_CHAIN = sim_chain


def _compute_day_regime(universe, ticks, peak_window_min=120, pump_threshold_pct=50,
                        active_rate=0.30, quiet_rate=0.15):
    """v14e.26: classify each calendar day in the universe as active/quiet/dead.

    Definition: pump_rate(day) = fraction of tokens entered on `day` whose
    intra-window peak (entry_price → max within `peak_window_min`) exceeds
    `pump_threshold_pct`%. Thresholds:
        active: pump_rate >= 30%
        quiet:  15% <= pump_rate < 30%
        dead:   pump_rate < 15%

    Returns: ({day_str: regime}, pump_rates_by_day) for inspection.
    Days with <5 tokens are tagged "unknown" (excluded from regime stats).
    """
    from collections import defaultdict
    by_day = defaultdict(list)
    for u in universe:
        addr = u["token_address"]
        td = ticks.get(addr)
        if not td:
            continue
        try:
            entry_p = float(u["entry_price"])
            if entry_p <= 0:
                continue
        except (TypeError, ValueError):
            continue
        try:
            entry_dt = datetime.fromisoformat(u["created_at"].replace("Z", "+00:00"))
        except (TypeError, ValueError):
            continue
        end_iso = (entry_dt + timedelta(minutes=peak_window_min)).isoformat().replace("+00:00", "Z")
        all_ticks = []
        for t in td.get("jp", []) + td.get("ds", []):
            ft = t.get("fetched_at")
            if ft and u["created_at"] <= ft <= end_iso:
                try:
                    p = float(t["price_usd"])
                    if p > 0:
                        all_ticks.append(p)
                except (TypeError, ValueError):
                    pass
        if not all_ticks:
            continue
        peak_pct = (max(all_ticks) / entry_p - 1) * 100
        day = u["created_at"][:10]
        by_day[day].append(peak_pct >= pump_threshold_pct)

    day_regime = {}
    pump_rates = {}
    for day, hits in by_day.items():
        if len(hits) < 5:
            day_regime[day] = "unknown"
            continue
        rate = sum(hits) / len(hits)
        pump_rates[day] = rate
        if rate >= active_rate:
            day_regime[day] = "active"
        elif rate >= quiet_rate:
            day_regime[day] = "quiet"
        else:
            day_regime[day] = "dead"
    return day_regime, pump_rates


def _compute_rolling_windows(universe, windows=(3, 7, 14)):
    """v14e.49: build {window_label: set(day_str)} for rolling time slices.

    Anchor = max(created_at) in the universe (so a sweep over a static historical
    range still gets meaningful 14d/7d/3d cuts relative to its OWN end-date,
    not wall-clock now). Returns labels "3d", "7d", "14d".
    """
    if not universe:
        return {}
    days = sorted({u["created_at"][:10] for u in universe if u.get("created_at")})
    if not days:
        return {}
    end = days[-1]
    end_dt = datetime.fromisoformat(end)
    out = {}
    for w in windows:
        cutoff = (end_dt - timedelta(days=w - 1)).date().isoformat()
        out[f"{w}d"] = {d for d in days if d >= cutoff}
    return out


def _mega_poll_offsets(polling_mode, horizon_sec):
    LOOP = _MEGA_LOOP_SEC
    if polling_mode == "fast":
        return list(range(LOOP, horizon_sec + 1, LOOP))
    if polling_mode.startswith("static_"):
        poll_sec = int(polling_mode.split("_")[1])
        out, last, t = [], -10**9, LOOP
        while t <= horizon_sec:
            if (t - last) >= poll_sec:
                out.append(t); last = t
            t += LOOP
        return out
    # v144 ext: per-variant LAZY cadences (fast/window/slow). Original "lazy"
    # maps to LAZY_STD. Others match the strategies.py reference profiles.
    LAZY_PROFILES = {
        "lazy":        (_MEGA_LAZY_FAST_SEC, _MEGA_LAZY_FAST_WINDOW, _MEGA_LAZY_SLOW_SEC),  # 180/300/600
        "lazy_fast":   (60,  120,  180),
        "lazy_med":    (120, 300,  360),
        "lazy_slow":   (300, 600,  900),
        "lazy_xslow":  (600, 900, 1200),
    }
    if polling_mode in LAZY_PROFILES:
        fast_sec, fast_win, slow_sec = LAZY_PROFILES[polling_mode]
        out, last, t = [], -10**9, LOOP
        while t <= horizon_sec:
            interval = fast_sec if t < fast_win else slow_sec
            if (t - last) >= interval:
                out.append(t); last = t
            t += LOOP
        return out
    return []


class _MegaSmState:
    __slots__ = ("ema", "hist", "prev_p", "armed_sl", "armed_tp", "last_vol")
    def __init__(self):
        self.ema = None; self.hist = []; self.prev_p = None
        self.armed_sl = True; self.armed_tp = True
        self.last_vol = None


def _mega_smooth(st, p, mode, sl_price, tp_price):
    if mode == "raw": return p
    if mode == "ema_fast":
        a = 2/3
        st.ema = p if st.ema is None else a*p + (1-a)*st.ema
        return st.ema
    if mode == "ema_slow":
        a = 2/9
        st.ema = p if st.ema is None else a*p + (1-a)*st.ema
        return st.ema
    if mode == "median_3":
        st.hist.append(p)
        if len(st.hist) > 3: st.hist.pop(0)
        return sorted(st.hist)[len(st.hist)//2] if len(st.hist) >= 3 else p
    if mode == "median_5":
        st.hist.append(p)
        if len(st.hist) > 5: st.hist.pop(0)
        return sorted(st.hist)[len(st.hist)//2] if len(st.hist) >= 5 else p
    if mode == "winsor_p95":
        if st.prev_p is None: st.prev_p = p; return p
        cap = st.prev_p * 0.18
        delta = p - st.prev_p
        out = st.prev_p + max(-cap, min(cap, delta))
        st.prev_p = out; return out
    if mode == "dual_confirm":
        if st.prev_p is None: st.prev_p = p; return p
        prev = st.prev_p; st.prev_p = p
        if sl_price and p <= sl_price and prev > sl_price: return prev
        if tp_price and p >= tp_price and prev < tp_price: return prev
        return p
    if mode == "hysteresis":
        if not st.armed_sl and sl_price and p >= sl_price * 1.02: st.armed_sl = True
        elif st.armed_sl and sl_price and p <= sl_price: st.armed_sl = False
        if not st.armed_tp and tp_price and p <= tp_price * 0.98: st.armed_tp = True
        elif st.armed_tp and tp_price and p >= tp_price: st.armed_tp = False
        if not st.armed_sl and sl_price and p <= sl_price: return sl_price * 1.001
        if not st.armed_tp and tp_price and p >= tp_price: return tp_price * 0.999
        return p
    if mode == "volume_gated":
        # Only accept price updates when volume is present; else return previous.
        # Volume is not passed to _mega_smooth in current architecture, so we fall
        # back to dual_confirm semantics (require prior tick confirmation). Proper
        # volume_gated needs tick-object access — TODO in a follow-up patch.
        if st.prev_p is None: st.prev_p = p; return p
        prev = st.prev_p; st.prev_p = p
        if sl_price and p <= sl_price and prev > sl_price: return prev
        if tp_price and p >= tp_price and prev < tp_price: return prev
        return p
    return p


def _mega_latest_at_or_before(sorted_ticks, t_iso):
    last = None
    for tk in sorted_ticks:
        if tk["fetched_at"] <= t_iso:
            p = float(tk["price_usd"])
            if p > 0: last = p
        else: break
    return last


def _mega_apply_filter(u, fname):
    if fname == "NONE": return True
    if fname == "NOZEROLIQ": return (u.get("rt_liquidity_usd") or 0) > 0
    if fname == "SCORE30": return (u.get("rt_score") or 0) >= 30
    if fname == "SCORE35": return (u.get("rt_score") or 0) >= 35
    if fname == "SCORE40": return (u.get("rt_score") or 0) >= 40
    if fname == "SCORE45": return (u.get("rt_score") or 0) >= 45
    if fname == "SCORE50": return (u.get("rt_score") or 0) >= 50
    if fname == "MCAP_MID": return 30_000 <= (u.get("entry_mcap") or 0) <= 500_000
    if fname == "TOPKOL": return (u.get("kol_group") or "") in _MEGA_TOP_KOLS
    # v14e.43 — BSR gates (rt_buy_sell_ratio) from reverse-engineer findings
    if fname == "BSR52": return (u.get("rt_buy_sell_ratio") or 0) >= 0.52
    if fname == "BSR55": return (u.get("rt_buy_sell_ratio") or 0) >= 0.55
    if fname == "NOZEROLIQ_BSR52":
        return (u.get("rt_liquidity_usd") or 0) > 0 and (u.get("rt_buy_sell_ratio") or 0) >= 0.52
    if fname == "NOZEROLIQ_BSR55":
        return (u.get("rt_liquidity_usd") or 0) > 0 and (u.get("rt_buy_sell_ratio") or 0) >= 0.55
    # v14e.43b — KW (kol_win_rate) + combo gates
    if fname == "KW34": return (u.get("kol_win_rate") or 0) >= 0.34
    if fname == "KW26": return (u.get("kol_win_rate") or 0) >= 0.26
    if fname == "NOZEROLIQ_KW34":
        return (u.get("rt_liquidity_usd") or 0) > 0 and (u.get("kol_win_rate") or 0) >= 0.34
    if fname == "NOZEROLIQ_KW26":
        return (u.get("rt_liquidity_usd") or 0) > 0 and (u.get("kol_win_rate") or 0) >= 0.26
    if fname == "BSR_MCAP":
        return (u.get("rt_buy_sell_ratio") or 0) >= 0.53 and (u.get("entry_mcap") or 0) >= 45000
    # --- v14e.72: message-sentiment bands + KOL call cadence -----------------
    # Both validated 2026-08-05 against a permutation null on 120d of shadow.
    # sentiment vs outcome is an INVERTED U (<0.3 = -0.57%/trade, 0.5-0.6 =
    # +7.97%, >=0.7 = -11.71%), so the axis is a BAND, not a threshold — a
    # `>= x` gate cannot express it, which is why these are new names.
    # Band WIDTH is the open question: excluding both tails (SENT30_70) keeps
    # 4x the volume of SENT50_60 for nearly the same total $. The sweep is
    # exactly the right place to settle that across every exit.
    # `sentiment` is None when no mention joins (2.8% of rows, avg -28.3%/trade)
    # → excluded rather than passed through.
    if fname.startswith("SENT"):
        s = u.get("sentiment")
        if s is None:
            return False
        if fname == "SENT30_70": return 0.30 <= s < 0.70
        if fname == "SENT45_65": return 0.45 <= s < 0.65
        if fname == "SENT50_60": return 0.50 <= s < 0.60
        if fname == "SENT_NOHYPE": return s < 0.70          # drop the hype tail only
        if fname == "SENT30_70_GAP24":
            return 0.30 <= s < 0.70 and (u.get("kol_gap_h") or 0) >= 24
    # Hours since the SAME KOL's previous call. Monotone dose-response on ~600k
    # rows: burst <1h = -5.43%/trade → 24-72h = +2.22%, holds after removing
    # olympeqg (57% of the burst bucket). None = first ever call for that KOL.
    if fname == "GAP24":
        return (u.get("kol_gap_h") or 0) >= 24
    if fname == "NOBURST":
        return (u.get("kol_gap_h") or 0) >= 1


def _mega_enrich_universe(universe, since):
    """v14e.72: attach `kol_gap_h` and `sentiment` to every universe row.

    Both feed filter arms added in v14e.72. Called once per sweep run, before
    the config matrix fans out, so the cost is one pass + one paginated fetch
    rather than anything per-config.

    `kol_gap_h` is derived from the universe itself — it is already one row per
    token carrying kol_group + created_at, so the KOL's previous call is just
    the previous row once sorted per KOL. No extra query.

    `sentiment` needs kol_mentions, which paper_trades does not carry. Matched
    on (kol_group, resolved_ca) taking the FIRST message, mirroring the RT path
    which fires on the first call.
    """
    by_kol = {}
    for u in universe:
        by_kol.setdefault(u.get("kol_group") or "", []).append(u)
    for rows in by_kol.values():
        rows.sort(key=lambda r: r["created_at"])
        prev = None
        for r in rows:
            if prev is None:
                r["kol_gap_h"] = None
            else:
                r["kol_gap_h"] = (
                    datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
                    - datetime.fromisoformat(prev.replace("Z", "+00:00"))
                ).total_seconds() / 3600.0
            prev = r["created_at"]

    sent = {}
    try:
        rows = sb_get("kol_mentions", [
            ("select", "kol_group,resolved_ca,sentiment,message_date"),
            ("resolved_ca", "not.is.null"),
            ("message_date", f"gte.{since}"),
            ("order", "message_date"),
        ] + _chain_params())
        for r in rows:
            k = (r.get("kol_group"), r.get("resolved_ca"))
            if k not in sent and r.get("sentiment") is not None:
                sent[k] = float(r["sentiment"])
    except Exception as e:
        print(f"[mega-enrich] sentiment fetch failed ({e}) — SENT* arms will match nothing")

    hit = 0
    for u in universe:
        v = sent.get((u.get("kol_group"), u.get("token_address")))
        u["sentiment"] = v
        hit += v is not None
    print(f"[mega-enrich] kol_gap_h on {len(universe)} rows | "
          f"sentiment matched {hit}/{len(universe)} ({100*hit/max(len(universe),1):.1f}%)")
    return universe


def _mega_apply_age_band(u, age_band):
    """v14e.27: token-age band filter, mirrors STRATEGY_FILTERS min/max_age_hours.
    Bands are disjoint to match strategies.py AGE24/AGE48 convention."""
    if age_band == "ALL":
        return True
    age_h = u.get("rt_token_age_hours")
    if age_h is None:
        # No age data persisted on this trade — exclude from age-banded runs
        # rather than silently bucketing into "ALL".
        return False
    age_h = float(age_h)
    if age_band == "AGE12":
        return age_h <= 12.0
    if age_band == "AGE24":
        return 12.0 < age_h <= 24.0
    if age_band == "AGE48":
        return 24.0 < age_h <= 48.0
    return True
    if fname == "NOZEROLIQ_SCORE30":
        return (u.get("rt_liquidity_usd") or 0) > 0 and (u.get("rt_score") or 0) >= 30
    if fname == "NOZEROLIQ_SCORE40":
        return (u.get("rt_liquidity_usd") or 0) > 0 and (u.get("rt_score") or 0) >= 40
    if fname == "MCAP_MID_SCORE40":
        return 30_000 <= (u.get("entry_mcap") or 0) <= 500_000 and (u.get("rt_score") or 0) >= 40
    return True


_MEGA_SELL_SLIP_BASE = 1 - 10/10_000

# v144.13 — per-family slip calibration (Sprint #2 option b).
# Memory Apr 20 (dtrail_shadow_artifact_apr20.md): "Paper models 200 bps sell
# slip; live actual = 9429 bps (47×)". DTRAIL / TRAIL / DIP / SPLIT / SCALP
# all exit on dynamic triggers (trail stops, dip rebuys, scalp tight TPs) where
# Jupiter Ultra fill diverges from the modeled tick price far more than for
# static TP/SL strats. Mega_sweep applies uniform 10bps slip, which:
#   - OVERESTIMATES trail/dtrail/dip rankings by 5-10× (44% of sweep universe)
#   - Matches fine for FAST/BE/TP* static exits.
# Multipliers are conservative first-pass (not the full 47× which would zero
# out all trail P&L). Revisit with live calibration once N≥30 per family.
_MEGA_FAMILY_SLIP_MULT = {
    # Ordered by descending multiplier — first match wins (= worst slip wins),
    # so a hybrid like SPLIT_50_TRAIL_SL50 inherits TRAIL's 8× (conservative).
    "DTRAIL":  10.0,   # 100 bps effective — narrow trails on memecoins = catastrophic slip
    "TRAIL":    8.0,   #  80 bps — wide trails a bit less bad
    "DIP":      6.0,   #  60 bps — staged re-entries
    "SCALP":    5.0,   #  50 bps — tiny TPs amplify any slip
    "SPLIT":    4.0,   #  40 bps — partial exits compound slippage
}

# v14e.19: families excluded by default from --mega-sweep / --mega-sweep-eval-history.
# These are documented sim artefacts (`docs/known_issues.md §2`, `dtrail_shadow_artifact_apr20.md`):
# sim over-estimates by 47× the live slip for trail-style exits, and `position_reconciler`
# closes 50-65% of trades before the trail can fire. Result: every mega sweep ranks them
# at the top of the leaderboard, polluting reads. Use --include-trail-families to opt-in.
import re as _trail_re
_MEGA_EXCLUDE_TRAIL_RE = _trail_re.compile(
    r"^(DTRAIL|TRAIL|DIP|PTRAIL|SPLIT|SCALE_OUT|MOONBAG|WIDE_RUNNER)",
    _trail_re.IGNORECASE,
)


def _filter_trail_families(pool: dict, include_trails: bool) -> dict:
    """Drop trail/dip/split/etc strats unless --include-trail-families set."""
    if include_trails:
        return pool
    kept = {k: v for k, v in pool.items() if not _MEGA_EXCLUDE_TRAIL_RE.match(k)}
    dropped = len(pool) - len(kept)
    if dropped:
        print(f"v14e.19: excluded {dropped} trail/dip/split strats (use --include-trail-families to keep)")
    return kept


# v14e.20: standalone _HYST strats are confirmed sim artefacts on real paper data.
# Paired test on N=55-90 (Apr 17→25 SOL paper main, exclu bat_gamble):
#   FAST_TP50_SL30_HYST   = +3.4% vs base FAST_TP50_SL30 +12.2% → −8.9pp
#   FAST_TP80_SL25_HYST   = +0.4% vs base FAST_TP80_SL25 +10.0% → −9.6pp
#   BE25_TP80_SL30_HYST   = −5.7% (no base recovery)
#   FAST_TP100_SL20_HYST  = −8.0% (no base recovery)
# HYST + filter SCORE30 still passes (BE25_TP80_SL30_S30_HYST +1.5pp paired-test),
# so we keep "_S\d+_HYST" / "_NZ.*_HYST" which already gate via filter.
_MEGA_EXCLUDE_HYST_STANDALONE = _trail_re.compile(
    r"_HYST$"
)
_MEGA_KEEP_HYST_WITH_FILTER = _trail_re.compile(
    r"_(S\d+|NZ\w*?)_HYST$", _trail_re.IGNORECASE
)


def _filter_hyst_standalone(pool: dict, include_hyst: bool) -> dict:
    """Drop standalone `_HYST` strats. Keep `_S30_HYST` / `_NZS30_HYST` (filter-gated)."""
    if include_hyst:
        return pool
    def keep(name: str) -> bool:
        if not _MEGA_EXCLUDE_HYST_STANDALONE.search(name):
            return True
        # Has _HYST suffix — only keep if combined with a quality filter.
        return bool(_MEGA_KEEP_HYST_WITH_FILTER.search(name))
    kept = {k: v for k, v in pool.items() if keep(k)}
    dropped = len(pool) - len(kept)
    if dropped:
        print(f"v14e.20: excluded {dropped} standalone _HYST strats (use --include-hyst to keep)")
    return kept


def _filter_smoothings_default(smoothings: list, include_smoothing_artefacts: bool) -> list:
    """Drop documented sim-artefact smoothings unless opted-in.

    `hysteresis` and `winsor_p95` consistently dominate mega-sweep tops because the
    kernel hides whipsaw that hits real paper execution. `dual_confirm` and the
    median variants are kept (they have legitimate use cases on dual-stream sources).
    """
    if include_smoothing_artefacts:
        return smoothings
    artefact = {"hysteresis", "winsor_p95"}
    kept = [s for s in smoothings if s not in artefact]
    dropped = len(smoothings) - len(kept)
    if dropped:
        print(f"v14e.20: excluded {dropped} artefact smoothings ({sorted(artefact & set(smoothings))})")
    return kept


def _mega_family_slip_mult(strat_name: str) -> float:
    """Return slip multiplier for strategy family. 1.0 if static TP/SL strat.

    Scans the ordered _MEGA_FAMILY_SLIP_MULT dict and returns the first match
    by prefix or embedded suffix (`_FAMILY`). Dict is ordered by descending
    multiplier so hybrid strategies take the worst slip — e.g. SPLIT_TRAIL
    gets TRAIL's 8× rather than SPLIT's 4×.
    """
    if not strat_name:
        return 1.0
    up = strat_name.upper()
    for prefix, mult in _MEGA_FAMILY_SLIP_MULT.items():
        if up.startswith(prefix) or f"_{prefix}" in up:
            return mult
    return 1.0


def _mega_gap_through(ev, exec_p, entry_price, sell_slip):
    """v14e.84 — le sweep bookait ses stops au niveau THEORIQUE du stop.

    C'est l'origine principale d'un ecart mesure de +11.71 pp entre le sweep et
    les trades reellement enregistres (104 paires rejouees sur les memes tokens,
    meme entree, meme strategie). Decomposition par type de sortie:

        sl_hit   n=41   reel -51.21 %   sweep -32.68 %
        be_stop  n=11   reel -24.16 %   sweep  -1.23 %

    Les DECISIONS de sortie concordent (sl_hit -> sl_hit dans 85 % des cas);
    c'est le PRIX booke pour la meme sortie qui diverge. Mesure cote production
    sur 608 sl_hit de BE25_TP80_SL30:

        stop theorique          -30.00 %
        sortie reellement booke -49.09 %   soit -27.27 % SOUS le stop
        416/608 sortent a plus de 10 % sous le stop

    C'est le gap-through, et il n'a rien d'un artefact: un memecoin qui declenche
    un stop a -30 % est deja en train de s'effondrer quand la vente passe. La
    production modelise ca (et se tient a -1.90 pp du live sur 145 paires
    sim<->live, donc elle est calibree); `_evaluate_trade_exit` renvoie, lui, un
    exit_price ancre sur `sl_price` + quelques bps, ce qui suppose une sortie
    AU niveau du stop -- une hypothese qui ne tient que sur un marche liquide.

    Correctif: quand le prix OBSERVE au moment du declenchement est sous le prix
    booke, on rebooke a ce prix observe. On ne peut pas vendre mieux que le
    marche. Aucun effet sur les sorties TP (le prix observe est au-dessus) ni
    sur les timeouts (deja bookes au prix courant).
    """
    pnl = ev.get("pnl_pct", 0)
    if ev.get("status") not in ("sl_hit", "be_stop") or not entry_price or not exec_p:
        return pnl
    reel = (exec_p * sell_slip) / entry_price - 1
    return round(min(pnl, reel), 4)


def _mega_replay_one(tp_mult, sl_mult, horizon_min, be_act,
                    jp_sorted, ds_sorted, entry_price, entry_time_iso,
                    source, smoothing, polling_mode, rt_liq_usd,
                    strat_name=None):
    # v142: accept the real strategy name so _evaluate_trade_exit's name-based
    # parsing (DTRAIL_RE, BE_RE, DIP_RE, DECAY_RE) + STRATEGIES tranche lookup
    # (trail_tiers, tp_schedule, time_be_minute) all engage correctly. Without
    # this, DTRAIL/DIP/TRAIL strats were running as plain TP/SL + horizon only,
    # producing misleading mega-sweep rankings.
    from paper_trader import _evaluate_trade_exit, _last_eval_ts
    entry_time = datetime.fromisoformat(entry_time_iso.replace("Z", "+00:00"))
    trade_id = f"mega_{id(jp_sorted)}"
    _last_eval_ts.pop(trade_id, None)
    horizon_sec = horizon_min * 60
    poll_offsets = _mega_poll_offsets(polling_mode, horizon_sec)
    if not poll_offsets: return None
    sl_price = entry_price * sl_mult
    tp_price = entry_price * tp_mult if tp_mult else None
    # v144.13: apply per-family slip correction. Base is 10bps (1 - 10/10_000);
    # trail/dtrail/dip/split/scalp get multiplier 4-10× to match live reality.
    _family_mult = _mega_family_slip_mult(strat_name)
    _sell_slip = 1 - (10 * _family_mult) / 10_000 if _family_mult != 1.0 else _MEGA_SELL_SLIP_BASE
    # v14e.31: position_usd chain-aware so the EVM gas-as-bps slip in
    # _evaluate_trade_exit (via _evm_slip_bps_with_gas) matches shadow/paper/live.
    # ETH paper/live position cap = $50 (matches _evm_min_position_usd). At $10
    # the sim was over-pessimistic (slip 1600 bps vs 400 bps real). SOL keeps
    # $10 — Jupiter Ultra near-zero slip is position-independent anyway.
    _pos_usd = 50.0 if _SIM_CHAIN in ("ethereum", "bsc", "base") else 10.0
    fake_trade = {
        "id": trade_id, "entry_price": entry_price,
        "sl_price": sl_price, "tp_price": tp_price,
        "position_usd": _pos_usd,
        "strategy": strat_name or (f"BE{int(be_act*100)}_TP80_SL30" if be_act else "TP80_SL30"),
        "tranche_label": "main", "horizon_minutes": horizon_min,
        "created_at": entry_time_iso,
        "high_price_seen": entry_price,
        "rt_liquidity_usd": rt_liq_usd,
        "dex_spot_price_at_entry": entry_price,
    }
    st = _MegaSmState()
    last_exec = None
    for offset in poll_offsets:
        poll_time = entry_time + timedelta(seconds=offset)
        poll_iso = poll_time.isoformat().replace("+00:00", "Z")
        jp = _mega_latest_at_or_before(jp_sorted, poll_iso)
        ds = _mega_latest_at_or_before(ds_sorted, poll_iso)
        if source == "jupiter":
            base = jp; exec_p = jp
        elif source == "both":
            # Merge: prefer Jupiter for decision (RFQ accuracy) but use it for
            # execution too. DS is fallback when Jupiter not yet available.
            base = jp if jp is not None else ds
            exec_p = jp if jp is not None else ds
        else:
            base = ds; exec_p = jp if jp is not None else ds
        if base is None or exec_p is None: continue
        last_exec = exec_p
        dec_p = _mega_smooth(st, base, smoothing, sl_price, tp_price)
        ev = _evaluate_trade_exit(fake_trade, exec_p, poll_time, _sell_slip,
                                  sell_fee_bps=0, decision_price=dec_p)
        if ev is None: continue
        if ev.get("high_price_seen") is not None:
            h = ev["high_price_seen"]
            if h > float(fake_trade.get("high_price_seen") or 0):
                fake_trade["high_price_seen"] = h
        if "status" in ev and ev["status"]:
            return _mega_gap_through(ev, exec_p, entry_price, _sell_slip)
    if last_exec is None: return None
    return round((last_exec / entry_price) - 1, 4) if entry_price else 0


def _mega_process_config(args):
    import numpy as np
    import json as _json
    from collections import defaultdict
    # v14e.27: tuple grew to include age_band. Backwards-compat — older callers
    # passing the 10-arg form get age_band="ALL" so existing entrypoints
    # (eval_history mode + tests) keep their behaviour.
    if len(args) == 10:
        (strat_name, tp_mult, sl_mult, horizon_min, be_act,
         fname, source, smoothing, polling_mode, universe) = args
        age_band = "ALL"
    else:
        (strat_name, tp_mult, sl_mult, horizon_min, be_act,
         fname, age_band, source, smoothing, polling_mode, universe) = args
    pnls = []
    pnls_by_day = defaultdict(list)  # v14e.26: track per-day pnls
    for u in universe:
        if not _mega_apply_filter(u, fname): continue
        if not _mega_apply_age_band(u, age_band): continue
        addr = u["token_address"]
        td = _MEGA_TICKS.get(addr)
        if not td: continue
        entry_ts = datetime.fromisoformat(u["created_at"].replace("Z", "+00:00"))
        t_end = (entry_ts + timedelta(minutes=horizon_min)).isoformat().replace("+00:00", "Z")
        jp = [t for t in td["jp"] if u["created_at"] <= t["fetched_at"] <= t_end]
        ds = [t for t in td["ds"] if u["created_at"] <= t["fetched_at"] <= t_end]
        if len(jp) < 3 and len(ds) < 3: continue
        pnl = _mega_replay_one(tp_mult, sl_mult, horizon_min, be_act,
                              jp, ds, float(u["entry_price"]), u["created_at"],
                              source, smoothing, polling_mode,
                              u.get("rt_liquidity_usd"),
                              strat_name=strat_name)
        if pnl is not None:
            pnls.append(pnl)
            pnls_by_day[u["created_at"][:10]].append(pnl)
    n = len(pnls)
    if n < 10: return None
    arr = np.array(pnls)
    wr = float((arr > 0).mean()) * 100
    avg = float(arr.mean()) * 100
    med = float(np.median(arr)) * 100
    # v14e.45: arr est une fraction (pnl_pct stocké en frac dans paper_trades),
    # *100 = percent. Le median ne peut donc pas être < -100% (un trade ne perd
    # pas plus de 100% du capital). Garde-fou contre une régression double-*100.
    assert med >= -100.0 - 1e-6, f"impossible median_pnl_pct={med} (arr min={arr.min()}, max={arr.max()})"
    std = float(arr.std(ddof=1)) * 100 if n > 1 else 0
    sharpe = (avg / std) if std > 0 else 0
    eq = np.cumprod(1 + arr)
    peaks = np.maximum.accumulate(eq)
    dd = float(((eq - peaks) / peaks).min()) * 100
    n_pass = sum(1 for u in universe
                 if _mega_apply_filter(u, fname) and _mega_apply_age_band(u, age_band))
    trade_rate = n_pass / max(1, len(universe)) * 18
    dollars_day = 50 * (avg / 100) * trade_rate

    # v14e.26 — Feature 1+2: Per-regime breakdown
    pnl_by_regime = defaultdict(list)
    for day, day_pnls in pnls_by_day.items():
        regime = _MEGA_DAY_REGIME.get(day, "unknown")
        if regime == "unknown":
            continue
        pnl_by_regime[regime].extend(day_pnls)
    pnl_active = float(np.mean(pnl_by_regime["active"]) * 100) if pnl_by_regime.get("active") else None
    pnl_quiet = float(np.mean(pnl_by_regime["quiet"]) * 100) if pnl_by_regime.get("quiet") else None
    pnl_dead = float(np.mean(pnl_by_regime["dead"]) * 100) if pnl_by_regime.get("dead") else None
    n_active = len(pnl_by_regime["active"])
    n_quiet = len(pnl_by_regime["quiet"])
    n_dead = len(pnl_by_regime["dead"])

    # v14e.26 — Feature 3: Walk-forward train/test split (last 3 days = test)
    train_pnls = []
    test_pnls = []
    for day, day_pnls in pnls_by_day.items():
        (test_pnls if day in _MEGA_TEST_DAYS else train_pnls).extend(day_pnls)
    wf_train_pnl = float(np.mean(train_pnls) * 100) if train_pnls else None
    wf_test_pnl = float(np.mean(test_pnls) * 100) if test_pnls else None
    wf_consistent = False
    if wf_train_pnl is not None and wf_test_pnl is not None:
        same_sign = (wf_train_pnl > 0) == (wf_test_pnl > 0)
        denom = max(abs(wf_train_pnl), 0.5)
        magnitude_ok = abs(wf_test_pnl - wf_train_pnl) / denom < 0.6
        wf_consistent = bool(same_sign and magnitude_ok)

    # v14e.26 — daily_pnl_json: needed for Feature 4 (bootstrap rank stability) in analyze step
    daily_pnl = {day: round(float(np.mean(p) * 100), 3) for day, p in pnls_by_day.items()}

    # v14e.49 — Rolling time-window robustness (3d / 7d / 14d).
    # Same data, sliced by recency. Answers "is the strat still alive?" while
    # the regime split (active/quiet/dead) answers "is it weather-resistant?".
    # $/day formula matches the all-data path (avg_pct * trade_rate * $50),
    # but trade_rate is recomputed per window so a strat that stops firing
    # is correctly penalised (not averaged out by older active days).
    rolling_metrics = {}
    for w_label, w_days in _MEGA_ROLLING_WINDOWS.items():
        if not w_days:
            continue
        w_pnls = []
        for day, day_pnls in pnls_by_day.items():
            if day in w_days:
                w_pnls.extend(day_pnls)
        n_w = len(w_pnls)
        if n_w == 0:
            rolling_metrics[w_label] = {"n": 0, "avg": None, "wr": None, "dpd": None}
            continue
        w_arr = np.array(w_pnls)
        w_avg = float(w_arr.mean()) * 100
        w_wr = float((w_arr > 0).mean()) * 100
        w_days_obs = max(1, len(w_days))
        n_pass_w = sum(
            1 for u in universe
            if u.get("created_at", "")[:10] in w_days
            and _mega_apply_filter(u, fname)
            and _mega_apply_age_band(u, age_band)
        )
        # universe size in this window (for trade_rate denominator)
        universe_w = sum(1 for u in universe if u.get("created_at", "")[:10] in w_days) or 1
        trade_rate_w = n_pass_w / universe_w * 18
        dpd_w = 50 * (w_avg / 100) * trade_rate_w
        rolling_metrics[w_label] = {
            "n": n_w,
            "avg": round(w_avg, 3),
            "wr": round(w_wr, 2),
            "dpd": round(dpd_w, 2),
            "days_obs": w_days_obs,
        }

    def _rm(label, key, default=None):
        return rolling_metrics.get(label, {}).get(key, default)

    # Robustness flags — gated on N_3d ≥ 10 to avoid noise verdicts.
    dpd_3d = _rm("3d", "dpd")
    dpd_14d = _rm("14d", "dpd")
    n_3d = _rm("3d", "n", 0) or 0
    fragile_recent = False
    regime_change = False
    if dpd_14d is not None and dpd_3d is not None and n_3d >= 10 and abs(dpd_14d) > 0.5:
        ratio = dpd_3d / dpd_14d if dpd_14d != 0 else None
        if ratio is not None:
            # Fragile: 3d earns < 50% of 14d baseline (or sign-flipped)
            if ratio < 0.5:
                fragile_recent = True
            # Regime change: 3d > 1.5× baseline — could be overfit-prone luck
            if ratio > 1.5:
                regime_change = True

    return {
        "strategy": strat_name, "filter": fname, "age_band": age_band,
        "source": source, "smoothing": smoothing, "polling_mode": polling_mode,
        "n_pass": n_pass, "n": n, "wr_pct": round(wr, 2),
        "avg_pnl_pct": round(avg, 3), "median_pnl_pct": round(med, 3),
        "sharpe": round(sharpe, 4), "max_dd_pct": round(dd, 2),
        "dollars_per_day": round(dollars_day, 2), "horizon_min": horizon_min,
        # v14e.26 regime + walk-forward columns
        "n_active": n_active, "n_quiet": n_quiet, "n_dead": n_dead,
        "pnl_active_pct": round(pnl_active, 3) if pnl_active is not None else None,
        "pnl_quiet_pct": round(pnl_quiet, 3) if pnl_quiet is not None else None,
        "pnl_dead_pct": round(pnl_dead, 3) if pnl_dead is not None else None,
        "wf_train_pnl_pct": round(wf_train_pnl, 3) if wf_train_pnl is not None else None,
        "wf_test_pnl_pct": round(wf_test_pnl, 3) if wf_test_pnl is not None else None,
        "wf_consistent": wf_consistent,
        # v14e.49 rolling-window robustness columns
        "n_14d": _rm("14d", "n", 0), "avg_pnl_pct_14d": _rm("14d", "avg"),
        "wr_pct_14d": _rm("14d", "wr"), "dollars_per_day_14d": _rm("14d", "dpd"),
        "n_7d": _rm("7d", "n", 0), "avg_pnl_pct_7d": _rm("7d", "avg"),
        "wr_pct_7d": _rm("7d", "wr"), "dollars_per_day_7d": _rm("7d", "dpd"),
        "n_3d": _rm("3d", "n", 0), "avg_pnl_pct_3d": _rm("3d", "avg"),
        "wr_pct_3d": _rm("3d", "wr"), "dollars_per_day_3d": _rm("3d", "dpd"),
        "fragile_recent": fragile_recent,
        "regime_change_3d": regime_change,
        "daily_pnl_json": _json.dumps(daily_pnl, separators=(",", ":")),
    }


def _mega_sweep_run_eh(args):
    """v144.9: mega-sweep driven by paper_trades.eval_history (ground truth).

    Universe = all closed paper trades with eval_history (live + paper, dedup'd
    per token keeping the earliest entry). Each eval_history entry provides
    (t, decision, exec, high) at true 30s cadence — identical to what the live
    bot saw. Strategy dimension is tested by rebuilding fake_trade rows with
    the candidate (tp, sl, horizon, be) and replaying via _mega_replay_one,
    treating exec_price as the Jupiter feed. DS dimension unavailable (the
    archive only persists the collapsed decision+exec pair) so source={jupiter}
    only.
    """
    import multiprocessing as mp
    import json as _json
    import time as _time
    import pandas as pd
    import re as _re

    csv_out = args.mega_csv_out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "_mega_sweep_eh.csv")
    n_workers = args.mega_workers or min(12, mp.cpu_count() - 2)
    since = args.mega_since

    # Eval_history mode uses the same dimension grid except source (jupiter-only)
    sources = ["jupiter"]
    smoothings = _MEGA_EXT_SMOOTHINGS if getattr(args, "mega_sweep_extended", False) else _MEGA_SMOOTHINGS
    polling_modes = _MEGA_EXT_POLLING_MODES if getattr(args, "mega_sweep_extended", False) else _MEGA_POLLING_MODES
    filters = _MEGA_EXT_FILTERS if getattr(args, "mega_sweep_extended", False) else _MEGA_FILTERS
    smoothings = _filter_smoothings_default(smoothings, getattr(args, "include_smoothing_artefacts", False))
    print(f"\n*** MEGA SWEEP — EVAL_HISTORY MODE (v144.9) ***")
    print(f"  sources={len(sources)} smoothings={len(smoothings)} polling={len(polling_modes)} filters={len(filters)}")
    print(f"  per-strat configs: {len(sources)*len(smoothings)*len(polling_modes)*len(filters)}")
    t0 = _time.time()

    from strategies import STRATEGIES as _STRATS
    full_pool = {}
    for name, tranches in _STRATS.items():
        tr0 = tranches[0]
        tp = tr0.get("tp_mult")
        sl = tr0.get("sl_mult", 0.50)
        h = tr0.get("horizon_min", 120) or 120
        be_m = _re.match(r"^BE(\d+)_TP", name)
        be_act = int(be_m.group(1)) / 100 if be_m else None
        full_pool[name] = (tp, sl, h, be_act)
    full_pool.update(_MEGA_NEW_STRATS)
    full_pool = _filter_trail_families(full_pool, getattr(args, "include_trail_families", False))
    full_pool = _filter_hyst_standalone(full_pool, getattr(args, "include_hyst", False))
    print(f"Strategies: {len(full_pool)}")

    # Fetch closed trades with eval_history — both paper and live contribute.
    # Dedup per token keeping the earliest entry so the sweep doesn't double-
    # count the same token via its live copy.
    params = [
        ("select", "id,token_address,created_at,entry_price,rt_liquidity_usd,"
                   "rt_score,kol_group,entry_mcap,source,eval_history,chain,"
                   "rt_buy_sell_ratio,kol_win_rate,rt_token_age_hours"),
        ("created_at", f"gte.{since}"),
        ("status", "in.(tp_hit,sl_hit,be_stop,trail_stop,timeout)"),
        ("eval_history", "not.is.null"),
        ("order", "created_at"),
    ] + _chain_params()
    rows = sb_get("paper_trades", params)
    print(f"Raw: {len(rows)} trades with eval_history (chain={_SIM_CHAIN}) since {since}")

    by_token = {}
    for r in rows:
        eh = r.get("eval_history")
        if not isinstance(eh, list) or len(eh) < 3:
            continue
        tok = r["token_address"]
        if tok not in by_token or r["created_at"] < by_token[tok]["created_at"]:
            by_token[tok] = r
    universe = list(by_token.values())
    universe = _mega_enrich_universe(universe, since)
    print(f"Universe: {len(universe)} unique tokens (dedup'd, ≥3 polls each)")

    # Build fake "ticks" dict — eval_history's exec field maps to Jupiter ticks.
    # DS bucket stays empty; sources != jupiter will skip these tokens in
    # _mega_replay_one (base = ds = None → continue). That's the expected
    # behaviour: eval_history doesn't have a separate DS stream.
    ticks = {}
    for u in universe:
        eh = u["eval_history"]
        jp_ticks = []
        for p in eh:
            try:
                e = p.get("e")
                t_iso = p.get("t")
                if e is None or t_iso is None or float(e) <= 0:
                    continue
                jp_ticks.append({
                    "fetched_at": t_iso.replace("+00:00", "Z") if "+" in t_iso else t_iso,
                    "price_usd": float(e),
                    "source": "jupiter",
                })
            except Exception:
                continue
        if jp_ticks:
            ticks[u["token_address"]] = {"jp": jp_ticks, "ds": []}
    print(f"  {len(ticks)} tokens with usable jp stream")

    if ticks:
        jp_counts = [len(v["jp"]) for v in ticks.values()]
        import statistics as _stx
        print(f"  Jupiter coverage (eval_history): median={_stx.median(jp_counts):.0f} ticks/token  "
              f"min={min(jp_counts)}  max={max(jp_counts)}")
        print(f"  (No DS bucket — eval_history persists decision+exec only)")

    ticks_path = os.path.join(os.path.dirname(csv_out), "_mega_ticks_eh_tmp.json")
    with open(ticks_path, "w") as f:
        _json.dump(ticks, f)
    print(f"  ticks JSON: {ticks_path} ({os.path.getsize(ticks_path)/1e6:.1f} MB)")

    # v14e.49 — Rolling windows in eval_history mode too (no regime data here,
    # but recency cuts are independent of regime classification).
    rolling_windows_eh = _compute_rolling_windows(universe, windows=(3, 7, 14))
    regime_path_eh = None
    if rolling_windows_eh:
        regime_path_eh = os.path.join(os.path.dirname(csv_out), "_mega_regime_eh_tmp.json")
        with open(regime_path_eh, "w") as f:
            _json.dump({"day_regime": {}, "test_days": [],
                        "rolling_windows": {k: sorted(v) for k, v in rolling_windows_eh.items()}}, f)
        anchor = max((u["created_at"][:10] for u in universe if u.get("created_at")),
                     default=None)
        print(f"\nRolling windows (anchor={anchor}):")
        for label in ("3d", "7d", "14d"):
            wd = rolling_windows_eh.get(label, set())
            if wd:
                print(f"    {label:<4} {len(wd)} days  ({min(wd)} -> {max(wd)})")

    jobs = []
    for strat_name, (tp, sl, h, be) in full_pool.items():
        for fname in filters:
            for src in sources:
                for smooth in smoothings:
                    for poll in polling_modes:
                        jobs.append((strat_name, tp, sl, h, be, fname, src, smooth, poll, universe))
    total = len(jobs)
    print(f"\nTotal configs: {total}")
    print(f"Launching {n_workers} workers...\n")
    results = []
    t_start = _time.time()
    with mp.Pool(n_workers, initializer=_mega_init_worker,
                 initargs=(ticks_path, regime_path_eh, _SIM_CHAIN)) as pool:
        for i, r in enumerate(pool.imap_unordered(_mega_process_config, jobs, chunksize=50)):
            if r is not None:
                results.append(r)
            if (i+1) % 2000 == 0:
                pct = 100 * (i+1) / total
                el = _time.time() - t_start
                eta = el / (i+1) * (total - i - 1)
                print(f"  {i+1}/{total} ({pct:.1f}%) in {el:.0f}s, ETA {eta:.0f}s", flush=True)

    try:
        os.remove(ticks_path)
    except Exception:
        pass
    if regime_path_eh:
        try:
            os.remove(regime_path_eh)
        except Exception:
            pass

    df = pd.DataFrame(results)
    df.to_csv(csv_out, index=False)
    print(f"\n{len(df)} valid rows / {total} configs -> {csv_out}")
    print(f"Total time: {_time.time()-t0:.0f}s")

    if not df.empty:
        df = df.sort_values("dollars_per_day", ascending=False)
        print("\n" + "=" * 120)
        print("TOP 40 BY $/DAY (EVAL_HISTORY MODE — A/B vs price_ticks mega sweep)")
        print("=" * 120)
        print(df.head(40)[["strategy", "filter", "source", "smoothing", "polling_mode",
                           "n", "wr_pct", "avg_pnl_pct", "median_pnl_pct", "dollars_per_day"]].to_string(index=False))
        print("\nDONE. Compare rankings with _mega_sweep_full.csv to catch "
              "price_ticks-induced bias (diff >= 5 ranks = suspect).")


def _mega_sweep_run(args):
    """v140: full mega sweep entry point. Multiprocessing grid sweep.

    v144: --mega-sweep-extended swaps dimension lists for the full matrix
    (9 polling modes × 12 filters × 3 sources × 9 smoothings = 2916 configs
    per strategy, ~800K total). Runtime several hours. More ticks = slower.

    v144.9: --mega-sweep-eval-history routes universe+ticks through
    paper_trades.eval_history instead of price_ticks. Universe becomes the set
    of tokens actually traded live/paper (ground truth, 30s polling resolution).
    Use A/B vs the price_ticks-based mega sweep to catch ranking bias induced
    by Jupiter tick sparsity in the archive logger.
    """
    if getattr(args, "mega_sweep_eval_history", False):
        _mega_sweep_run_eh(args)
        return
    import multiprocessing as mp
    import json as _json
    import time as _time
    import pandas as pd
    import re as _re

    extended = getattr(args, "mega_sweep_extended", False)
    default_csv = "_mega_sweep_extended.csv" if extended else "_mega_sweep_full.csv"
    csv_out = args.mega_csv_out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), default_csv)
    n_workers = args.mega_workers or min(12, mp.cpu_count() - 2)
    since = args.mega_since

    # Select dimension lists based on mode
    if extended:
        sources = _MEGA_EXT_SOURCES
        smoothings = _MEGA_EXT_SMOOTHINGS
        polling_modes = _MEGA_EXT_POLLING_MODES
        filters = _MEGA_EXT_FILTERS
        age_bands = _MEGA_EXT_AGE_BANDS
        print(f"\n*** EXTENDED MEGA SWEEP — FULL MATRIX ***")
    else:
        sources = _MEGA_SOURCES
        smoothings = _MEGA_SMOOTHINGS
        polling_modes = _MEGA_POLLING_MODES
        filters = _MEGA_FILTERS
        age_bands = _MEGA_AGE_BANDS

    # v14e.43: optional source-shard filter so CI can split into 3 parallel jobs
    # (each fits under 6h GH hard cap). 'all' keeps every source (local/dev).
    shard = getattr(args, "mega_source_shard", "all")
    if shard != "all":
        if shard not in sources:
            print(f"[mega-source-shard] '{shard}' not in current sources={sources} — running empty shard.")
            sources = []
        else:
            sources = [shard]
            print(f"[mega-source-shard] restricted to source='{shard}' (1/{len(_MEGA_EXT_SOURCES) if extended else len(_MEGA_SOURCES)} shards)")
    smoothings = _filter_smoothings_default(smoothings, getattr(args, "include_smoothing_artefacts", False))

    # -----------------------------------------------------------------
    # v14e.77 — OU PASSE LE CALCUL: 98.6 % sur des dimensions d'artefact
    # -----------------------------------------------------------------
    # `smoothing` et `polling_mode` ne sont pas des choix de strategie: ce sont
    # des hypotheses sur la FACON DONT ON LIT le prix. 7 x 10 = 70 combinaisons
    # par config reelle, soit 98.6 % du budget de calcul, pour une seule qui
    # corresponde a la prod (raw / cadence 3 min).
    #
    # Elles coutent deux fois. En calcul: elles interdisaient d'elargir la
    # fenetre de ticks a 4 mois sous le cap GH de 6h. En lecture: elles
    # dupliquent chaque config reelle en 70 lignes quasi identiques, ce qui
    # gonfle artificiellement les tests apparies — BSR_MCAP sortait n°1 sur
    # 251k cellules "toutes sources" et n°11 sur les cellules independantes.
    #
    # --mega-lean-grid garde de quoi tester la robustesse (2 lissages, 3
    # cadences) et rend le reste du budget a la PROFONDEUR D'HISTORIQUE, qui
    # est la seule dimension ou un portefeuille type E30 peut apparaitre.
    # Sans le flag, comportement d'avant a l'identique.
    if getattr(args, "mega_lean_grid", False):
        _av = len(smoothings) * len(polling_modes)
        smoothings = [s for s in ("raw", "ema_fast") if s in smoothings] or smoothings[:1]
        polling_modes = [p for p in ("fast", "lazy_fast", "lazy") if p in polling_modes] or polling_modes[:1]
        _ap = len(smoothings) * len(polling_modes)
        print(f"[mega-lean-grid] lissage x cadence: {_av} -> {_ap} combinaisons "
              f"({_av/max(_ap,1):.1f}x de calcul rendu a la profondeur d'historique)")

    print(f"  sources={len(sources)} smoothings={len(smoothings)} polling={len(polling_modes)} "
          f"filters={len(filters)} age_bands={len(age_bands)} ({age_bands})")
    print(f"  per-strat configs: {len(sources)*len(smoothings)*len(polling_modes)*len(filters)*len(age_bands)}")

    print(f"\n{'#'*90}\n# v140 MEGA SWEEP {datetime.now().isoformat()[:19]}\n{'#'*90}\n")
    t0 = _time.time()

    from strategies import STRATEGIES as _STRATS
    full_pool = {}
    for name, tranches in _STRATS.items():
        tr0 = tranches[0]
        tp = tr0.get("tp_mult")
        sl = tr0.get("sl_mult", 0.50)
        h = tr0.get("horizon_min", 120) or 120
        be_m = _re.match(r"^BE(\d+)_TP", name)
        be_act = int(be_m.group(1)) / 100 if be_m else None
        full_pool[name] = (tp, sl, h, be_act)
    full_pool.update(_MEGA_NEW_STRATS)
    full_pool = _filter_trail_families(full_pool, getattr(args, "include_trail_families", False))
    full_pool = _filter_hyst_standalone(full_pool, getattr(args, "include_hyst", False))
    # v14e.56: optional strat-shard filter — allows splitting the strategy pool across
    # additional CI parallel jobs when --mega-source-shard alone hits the GH 6h cap.
    # Format: "X/N" where X in [1..N]. Strats are split deterministically by sorted name
    # hash (modulo) so the same strat always lands in the same shard across runs.
    # Default "1/1" = no split (behaviour identical to pre-v14e.56). Empty/missing
    # arg keeps the full pool — zero regression for local/dev runs.
    strat_shard_raw = getattr(args, "mega_strat_shard", "1/1") or "1/1"
    try:
        _ssx, _ssn = strat_shard_raw.split("/")
        _ssx, _ssn = int(_ssx), int(_ssn)
        assert 1 <= _ssx <= _ssn, "shard index out of range"
    except Exception:
        print(f"[mega-strat-shard] invalid '{strat_shard_raw}' — falling back to 1/1 (no split)")
        _ssx, _ssn = 1, 1
    if _ssn > 1:
        # Deterministic split by sorted name index — same strat always in same shard
        # regardless of dict iteration order. Hash-mod gives even distribution.
        _names_sorted = sorted(full_pool.keys())
        _kept = {n: full_pool[n] for i, n in enumerate(_names_sorted) if (i % _ssn) == (_ssx - 1)}
        print(f"[mega-strat-shard] {_ssx}/{_ssn} kept {len(_kept)}/{len(full_pool)} strategies")
        full_pool = _kept
    print(f"Strategies: {len(full_pool)} (incl. {len(_MEGA_NEW_STRATS)} new TP200+ variants)")

    # v142: sb_get() paginates internally via offset+limit; the previous manual
    # range_lo/range_hi loop called it with kwargs that don't exist -> TypeError.
    # v14e.27: rt_token_age_hours included for age-band sweep dimension.
    params = [
        # v14e.72: rt_buy_sell_ratio + kol_win_rate were referenced by
        # _mega_apply_filter since v14e.43 but never selected, so BSR52/BSR55/
        # KW34/KW26/NOZEROLIQ_BSR*/BSR_MCAP evaluated `(None or 0) >= thr` =
        # always False. 7 of 21 filter arms matched zero trades and burned
        # sweep compute for nothing. Selecting them makes those arms real.
        ("select", "id,token_address,created_at,entry_price,rt_liquidity_usd,"
                   "rt_score,kol_group,entry_mcap,rt_token_age_hours,chain,"
                   "rt_buy_sell_ratio,kol_win_rate"),
        ("source", "eq.rt"),
        ("created_at", f"gte.{since}"),
        ("order", "created_at"),
    ] + _chain_params()
    rows = sb_get("paper_trades", params)
    by_token = {}
    for r in rows:
        if r["token_address"] not in by_token:
            by_token[r["token_address"]] = r
    universe = list(by_token.values())
    universe = _mega_enrich_universe(universe, since)
    print(f"Universe: {len(universe)} unique tokens since {since}")

    # -----------------------------------------------------------------
    # v14e.77 — LA FENETRE DE TICKS SUIVAIT L'HORLOGE, PAS LES TRADES
    # -----------------------------------------------------------------
    # Avant: `start = now - 8 jours` en dur, `end = now + 1h`, pour TOUS les
    # tokens. L'univers etait bien construit sur --mega-since (4 mois, 2717
    # tokens) mais un token appele en juin n'avait aucun tick dans [now-8j,
    # now+1h], donc `rs` etait vide, donc il etait absent de `ticks`, et le
    # replay le sautait en silence (`if addr in streams_by_token`).
    #
    # Effet mesure sur le run 31040338036: "Universe: 2717" puis "240 with
    # ticks" — 91 % de l'univers jete, et 9 jours de profondeur au lieu de 4
    # mois. Un sweep qui ne voit que 9 jours ne peut PAS trouver un
    # portefeuille type E30, dont tout l'interet est qu'une strategie porte
    # mai pendant que l'autre porte juin. Le classement n'etait pas trop
    # severe: il etait aveugle.
    #
    # Correctif: chaque token est interroge sur SA propre fenetre,
    # [entree - 5 min, entree + horizon max + 30 min]. La requete est plus
    # etroite qu'avant pour les tokens recents (donc plus rapide), et non vide
    # pour les anciens.
    # full_pool[name] = (tp_mult, sl_mult, horizon_min, be_activation)
    _horizons = [float(v[2]) for v in full_pool.values()
                 if isinstance(v, (tuple, list)) and len(v) > 2 and v[2]]
    _max_h = max(_horizons) if _horizons else 240.0
    print(f"Fetching ticks... (fenetre par token: entree -5min -> +{_max_h + 30:.0f}min)")
    ticks = {}
    _sans_entree = 0
    for i, u in enumerate(universe):
        addr = u["token_address"]
        try:
            _e = datetime.fromisoformat(str(u["created_at"]).replace("Z", "+00:00"))
        except Exception:
            _sans_entree += 1
            continue
        params = [
            ("select", "price_usd,fetched_at,source,chain"),
            ("token_address", f"eq.{addr}"),
            ("fetched_at", f"gte.{(_e - timedelta(minutes=5)).isoformat()}"),
            ("fetched_at", f"lte.{(_e + timedelta(minutes=_max_h + 30)).isoformat()}"),
            ("order", "fetched_at"),
        ] + _chain_params()
        rs = sb_get("price_ticks", params)
        if rs:
            jp = sorted([t for t in rs if t["source"] == "jupiter"], key=lambda t: t["fetched_at"])
            ds = sorted([t for t in rs if t["source"] in ("fast", "full", "live")], key=lambda t: t["fetched_at"])
            ticks[addr] = {"jp": jp, "ds": ds}
        if (i+1) % 20 == 0: print(f"  {i+1}/{len(universe)}", flush=True)
    print(f"  {len(ticks)} with ticks ({_time.time()-t0:.0f}s)")
    if _sans_entree:
        print(f"  [WARN] {_sans_entree} tokens sans created_at exploitable")
    # Garde-fou: si on reperd la profondeur, on veut le voir dans le log, pas
    # le decouvrir trois runs plus tard en relisant daily_pnl_json.
    _couverture = len(ticks) / max(len(universe), 1)
    if _couverture < 0.5:
        print(f"  [WARN] seulement {100*_couverture:.0f}% de l'univers a des ticks — "
              f"le sweep tourne sur un sous-ensemble, ne pas lire son classement "
              f"comme s'il portait sur {len(universe)} tokens")

    # v144.9: Jupiter coverage stats — warn when price_ticks is sparse on Jupiter.
    # price_ticks logs Jupiter at 3-min batch cadence vs live 30s polling; tokens
    # that drop out of the active rotation end up with 0-2 jup ticks in window,
    # forcing the replay to fall back to DS-only. Heavy DS-fallback skews rankings
    # towards strats tolerant to DS noise. Prefer --mega-sweep-eval-history for
    # clean ranking, or interpret this run accordingly.
    if ticks:
        jp_counts = [len(v["jp"]) for v in ticks.values()]
        ds_counts = [len(v["ds"]) for v in ticks.values()]
        n_zero_jp = sum(1 for c in jp_counts if c == 0)
        n_low_jp = sum(1 for c in jp_counts if c < 10)
        import statistics as _stx
        print(f"\n  Jupiter coverage: median={_stx.median(jp_counts):.0f} ticks/token  "
              f"zero_jup={n_zero_jp}/{len(ticks)} ({100*n_zero_jp/len(ticks):.1f}%)  "
              f"<10_jup={n_low_jp}/{len(ticks)} ({100*n_low_jp/len(ticks):.1f}%)")
        print(f"  DS coverage:      median={_stx.median(ds_counts):.0f} ticks/token")
        if n_zero_jp / len(ticks) > 0.15:
            print(f"  [WARN] >15% tokens have 0 Jupiter ticks — jup-source results biased "
                  f"towards DS fallback. Cross-check with --mega-sweep-eval-history.")

    ticks_path = os.path.join(os.path.dirname(csv_out), "_mega_ticks_tmp.json")
    with open(ticks_path, "w") as f:
        _json.dump(ticks, f)
    print(f"  ticks JSON: {ticks_path} ({os.path.getsize(ticks_path)/1e6:.1f} MB)")

    # v14e.26 — Per-day regime classification (active/quiet/dead) + walk-forward split
    print("\nClassifying day regimes (peak_window=120min, pump_threshold=+50%)...")
    day_regime, pump_rates = _compute_day_regime(universe, ticks)
    sorted_days = sorted(day_regime.keys())
    test_days = set(sorted_days[-3:])  # last 3 days = walk-forward test set
    counts = {"active": 0, "quiet": 0, "dead": 0, "unknown": 0}
    for d, r in day_regime.items():
        counts[r] = counts.get(r, 0) + 1
    print(f"  Days classified: {len(day_regime)}")
    for r, c in sorted(counts.items()):
        print(f"    {r:<8} {c}")
    print(f"  Walk-forward test days (last 3): {sorted(test_days)}")
    if pump_rates:
        for d in sorted(pump_rates):
            r = day_regime[d]
            mark = " <- TEST" if d in test_days else ""
            print(f"    {d}  pump_rate={pump_rates[d]*100:>5.1f}%  regime={r:<8}{mark}")

    # v14e.49 — Rolling time-window buckets (3d / 7d / 14d) for recency cuts
    rolling_windows = _compute_rolling_windows(universe, windows=(3, 7, 14))
    if rolling_windows:
        anchor = max((u["created_at"][:10] for u in universe if u.get("created_at")),
                     default=None)
        print(f"\nRolling windows (anchor={anchor}):")
        for label in ("3d", "7d", "14d"):
            wd = rolling_windows.get(label, set())
            if wd:
                print(f"    {label:<4} {len(wd)} days  ({min(wd)} -> {max(wd)})")

    # Persist regime metadata for workers
    regime_path = os.path.join(os.path.dirname(csv_out), "_mega_regime_tmp.json")
    with open(regime_path, "w") as f:
        _json.dump({"day_regime": day_regime, "test_days": sorted(test_days),
                    "pump_rates": pump_rates,
                    "rolling_windows": {k: sorted(v) for k, v in rolling_windows.items()}}, f)

    # v14e.26 BONUS — `--exclude-dead-days` filter universe before sweep
    if getattr(args, "exclude_dead_days", False):
        before = len(universe)
        universe = [u for u in universe if day_regime.get(u["created_at"][:10]) != "dead"]
        print(f"\n[--exclude-dead-days] Universe filtered: {before} -> {len(universe)} (dropped {before - len(universe)} dead-day trades)")

    jobs = []
    for strat_name, (tp, sl, h, be) in full_pool.items():
        for fname in filters:
            for age_band in age_bands:
                for src in sources:
                    for smooth in smoothings:
                        for poll in polling_modes:
                            jobs.append((strat_name, tp, sl, h, be, fname, age_band,
                                         src, smooth, poll, universe))
    total = len(jobs)
    print(f"\nTotal configs: {total}")
    print(f"Launching {n_workers} workers...\n")
    results = []
    t_start = _time.time()
    with mp.Pool(n_workers, initializer=_mega_init_worker,
                 initargs=(ticks_path, regime_path, _SIM_CHAIN)) as pool:
        for i, r in enumerate(pool.imap_unordered(_mega_process_config, jobs, chunksize=50)):
            if r is not None: results.append(r)
            if (i+1) % 2000 == 0:
                pct = 100 * (i+1) / total
                el = _time.time() - t_start
                eta = el / (i+1) * (total - i - 1)
                print(f"  {i+1}/{total} ({pct:.1f}%) in {el:.0f}s, ETA {eta:.0f}s", flush=True)

    try: os.remove(ticks_path)
    except Exception: pass
    try: os.remove(regime_path)
    except Exception: pass

    df = pd.DataFrame(results)
    df.to_csv(csv_out, index=False)
    print(f"\n{len(df)} valid rows / {total} configs -> {csv_out}")
    print(f"Total time: {_time.time()-t0:.0f}s")

    df = df.sort_values("dollars_per_day", ascending=False)
    print("\n" + "=" * 120)
    print("TOP 40 BY $/DAY")
    print("=" * 120)
    print(df.head(40)[["strategy","filter","source","smoothing","polling_mode",
                       "n","wr_pct","avg_pnl_pct","median_pnl_pct","dollars_per_day"]].to_string(index=False))
    print("\n" + "=" * 120)
    print("BEST PER STRATEGY (any filter) — top 30")
    print("=" * 120)
    bs = df.drop_duplicates(subset=["strategy"], keep="first").head(30)
    print(bs[["strategy","filter","source","smoothing","polling_mode",
              "n","avg_pnl_pct","dollars_per_day"]].to_string(index=False))
    print("\n" + "=" * 120)
    print("BEST PER FILTER")
    print("=" * 120)
    bf = df.drop_duplicates(subset=["filter"], keep="first")
    print(bf[["filter","strategy","source","smoothing","polling_mode",
              "n","avg_pnl_pct","dollars_per_day"]].to_string(index=False))
    print("\nDONE.")


def main():
    parser = argparse.ArgumentParser(description="Unified strategy simulator")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--max-fetch", type=int, default=0)
    parser.add_argument("--since", type=str, default="2026-03-01")
    parser.add_argument("--strategies", type=str, default=None)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--runner-analysis", action="store_true")
    parser.add_argument("--mc-sims", type=int, default=1000)
    parser.add_argument("--mc-trades", type=int, default=200)
    parser.add_argument("--realistic", action="store_true",
                        help="Resample candles to 3-min spot checks (like live)")
    parser.add_argument("--check-interval", type=int, default=3,
                        help="Slow check interval in minutes for --realistic mode")
    parser.add_argument("--fast-interval", type=int, default=30,
                        help="Fast check interval in seconds (default: 30)")
    parser.add_argument("--fast-window", type=int, default=30,
                        help="Fast check window in minutes (default: 30)")
    parser.add_argument("--max-age", type=float, default=12.0,
                        help="Max token age in hours (live default: 12h). 0=no filter")
    parser.add_argument("--check-grid", action="store_true",
                        help="Grid search check intervals (fast_sec x slow_min x fast_window)")
    parser.add_argument("--check-grid-top", type=int, default=10,
                        help="Number of top strategies to test in check-grid")
    parser.add_argument("--interval-cross", action="store_true",
                        help="Full cross-grid: all strategies x 5 interval profiles")
    parser.add_argument("--flat-sizing", type=float, default=0,
                        help="Fixed position size in USD (default: 0 = Kelly sizing). Use ~99 to match paper trader.")
    parser.add_argument("--chain", type=str, default="solana",
                        choices=["solana", "ethereum", "all"],
                        help="v14: filter trade universe + price_ticks by chain. "
                             "'solana' (default) = legacy behavior. 'ethereum' = ETH only. "
                             "'all' = mix (only meaningful when fee models don't matter).")
    parser.add_argument("--from-trades", action="store_true",
                        help="Use real paper trade PnL instead of OHLCV simulation (ground truth mode)")
    parser.add_argument("--from-ticks", action="store_true",
                        help="Tick-replay simulation: replay real 30s price ticks through _evaluate_trade_exit")
    parser.add_argument("--from-eval-history", action="store_true",
                        help="v138 B: replay each trade from its persisted eval_history "
                             "(perfect 0% sim/real divergence, ground truth for poll-level audit)")
    parser.add_argument("--from-cache-snapshots", action="store_true",
                        help="v138 D: rebuild price stream from cache_snapshots table "
                             "(captures the cache state at every loop tick, no throttle gaps)")
    parser.add_argument("--price-source", type=str, default="jupiter",
                        choices=["jupiter", "dexscreener", "both"],
                        help="Price source for --from-ticks (default: jupiter)")
    parser.add_argument("--orchestration", type=str, default=None,
                        choices=["jupiter", "ds", "hybrid", "confirm", "ema",
                                 "median_3", "median_5", "winsor_p95", "dual_confirm",
                                 "hysteresis", "ema_fast", "ema_slow",
                                 "jp_sampled_60s", "jp_sampled_180s",
                                 "ohlc_burst_60s", "vwap_5min", "twin_confirm"],
                        help="v132: Orchestration mode (overrides --price-source for tick replay)")
    parser.add_argument("--poll-sec", type=int, default=0,
                        help="v132: Subsample ticks to N-second polling interval (0=no subsampling)")
    parser.add_argument("--ema-window", type=int, default=3,
                        help="v132: EMA window for --orchestration ema")
    parser.add_argument("--from-live-config", action="store_true",
                        help="v132: Auto-load strategy_overrides from scoring_config.rt_trade_config "
                             "so each trade replays with its production polling + price_source. "
                             "Overrides --orchestration and --poll-sec when set.")
    parser.add_argument("--priority-fee-sol", type=float, default=0.0,
                        help="v132: Deduct priority fee per round-trip (default 0.0, typical Jupiter auto ~0.0005)")
    parser.add_argument("--grid-ticks", action="store_true",
                        help="Grid search DTRAIL params on tick data")
    parser.add_argument("--smoothing-sweep", action="store_true",
                        help="v133: Run tick sim once per smoothing mode (raw, median_3, "
                             "median_5, winsor_p95, dual_confirm, ema_fast, ema_slow, "
                             "hysteresis, volume_gated) on a focused list of live strategies.")
    parser.add_argument("--smoothing-strats", type=str,
                        default="FAST_TP50_SL30,DTRAIL10_ACT15_SL70,DTRAIL3_ACT5_SL60,DIP30_B5_T5_A20_SL70_240m",
                        help="Comma-separated strategies to sweep in --smoothing-sweep")
    parser.add_argument("--synthetic-sweep", action="store_true",
                        help="v134: Test synthetic strategies (tp/sl/be_act/horizon specs) on post-v132 ticks")
    parser.add_argument("--synthetic-strats", type=str, default="",
                        help="Semicolon-separated specs, e.g. 'TP70_SL30:tp=70,sl=30,horizon=30;"
                             "BE20_TP70_SL30:tp=70,sl=30,horizon=30,be_act=20'")
    parser.add_argument("--until", type=str, default="",
                        help="ISO date upper bound for trade fetch (e.g. 2026-04-13) — used for "
                             "windowed robustness tests")
    parser.add_argument("--validate-ticks", action="store_true",
                        help="Compare tick sim results vs actual paper PnL")
    parser.add_argument("--tick-csv", type=str, default=None,
                        help="Export per-trade tick sim results to CSV")
    parser.add_argument("--sim-live-entry", action="store_true",
                        help="Simulate live entry slippage (3-5%% worse entry price) in tick sim")
    parser.add_argument("--dual-wallet", action="store_true",
                        help="Run dual-wallet analysis on top strategies")
    parser.add_argument("--dual-deltas", type=str, default="0,5,10,15,30",
                        help="Comma-separated delta minutes for Wallet B (default: 0,5,10,15,30)")
    parser.add_argument("--dual-top", type=int, default=30,
                        help="Number of top strategies to test with dual-wallet (default: 30)")
    parser.add_argument("--dual-position", type=float, default=0,
                        help="Position size USD for slippage calc (default: Kelly-sized)")
    parser.add_argument("--legacy-sim", action="store_true",
                        help="Force legacy sim engines (flat 2.5%% slippage, duplicated exit logic)")
    parser.add_argument("--no-ticks", action="store_true",
                        help="Force OHLCV mode even when tick data is available")
    parser.add_argument("--divergence-report", action="store_true",
                        help="Show paper vs live exit price divergence summary from recent trades")
    # v140: full mega sweep — strategies × filters × sources × smoothings × polling
    parser.add_argument("--mega-sweep", action="store_true",
                        help="v140: Full mega sweep — all STRATEGIES + new TP200 variants × 7 filters × "
                             "2 sources × 8 smoothings × 5 polling modes. Multiprocessing. ~30-45min.")
    parser.add_argument("--exclude-dead-days", action="store_true",
                        help="v14e.26 BONUS: drop dead-day trades from the universe before the "
                             "mega sweep. Dead = pump_rate < 15%% (fraction of tokens reaching peak >= +50%% "
                             "in 2h). Useful when the recent SOL market regime is non-representative "
                             "(e.g. 3+ days of mort) and would distort the ranking. Cross-check vs full run.")
    parser.add_argument("--mega-sweep-extended", action="store_true",
                        help="v144: EXTENDED mega sweep — 12 filters × 3 sources × 9 smoothings × "
                             "10 polling modes (adds source=both, 4 LAZY cadence variants, 5 finer "
                             "filters, volume_gated smoothing). ~5× the configs vs --mega-sweep. "
                             "Runtime several hours (meant to run afternoon/overnight). Output: "
                             "_mega_sweep_extended.csv.")
    parser.add_argument("--mega-sweep-eval-history", action="store_true",
                        help="v144.9: mega sweep driven by paper_trades.eval_history instead of "
                             "price_ticks. Universe = tokens actually traded (live/paper). "
                             "Source forced to jupiter (eval_history has no DS stream). "
                             "Resolution 30s (ground truth) vs 3-min batch in price_ticks. "
                             "Use A/B vs --mega-sweep to detect ranking bias from Jupiter "
                             "tick sparsity in the archive. Output: _mega_sweep_eh.csv.")
    parser.add_argument("--mega-workers", type=int, default=0,
                        help="Workers for --mega-sweep (default: min(12, cpu_count-2))")
    parser.add_argument("--mega-csv-out", type=str, default=None,
                        help="Output CSV for --mega-sweep (default: scraper/_mega_sweep_full.csv)")
    parser.add_argument("--mega-since", type=str, default="2026-04-13T20:00:00Z",
                        help="Universe cutoff for --mega-sweep (default: post-v132)")
    parser.add_argument("--mega-source-shard", type=str, default="all",
                        choices=["all", "jupiter", "dexscreener", "both"],
                        help="v14e.43: shard the extended sweep by source so each shard "
                             "fits under the GitHub-hosted runner 6h hard cap. "
                             "'all' (default) runs every source — kept for local/dev. "
                             "Pick one of jupiter/dexscreener/both in CI matrix; the merge "
                             "job concatenates the per-shard CSVs into the canonical one.")
    parser.add_argument("--mega-strat-shard", type=str, default="1/1",
                        help="v14e.56: secondary shard on the strategy pool (X/N format). "
                             "Used together with --mega-source-shard to split the sweep "
                             "into more parallel CI jobs (3 sources × 2 strat halves = 6 "
                             "shards) when source-only sharding hits the 6h cap. Default "
                             "'1/1' = no split (full pool, pre-v14e.56 behaviour). The "
                             "split is deterministic by sorted strat name index modulo N.")
    parser.add_argument("--mega-lean-grid", action="store_true",
                        help="v14e.77: reduit lissage x cadence de 70 a 6 combinaisons "
                             "(raw/ema_fast x fast/lazy_fast/lazy). Ce sont des hypotheses "
                             "de LECTURE du prix, pas des choix de strategie: elles "
                             "consommaient 98.6%% du budget de calcul et dupliquaient chaque "
                             "config reelle en 70 lignes quasi identiques. Le calcul rendu "
                             "finance la profondeur d'historique (--mega-since), seule "
                             "dimension ou un portefeuille multi-regimes peut apparaitre.")
    parser.add_argument("--include-trail-families", action="store_true",
                        help="v14e.19: opt-in to keep DTRAIL/TRAIL/DIP/PTRAIL/SPLIT/"
                             "SCALE_OUT/MOONBAG/WIDE_RUNNER families in mega-sweep. "
                             "Default excluded (documented sim artefact: live slip 47× "
                             # v14e.77: %% obligatoire — argparse applique `help % params`,
                             # donc un '%' nu casse --help pour TOUT le parser
                             # ("unsupported format character 'b'" sur "65% before").
                             "paper, position_reconciler closes 50-65%% before trail fires). "
                             "Cuts sweep size by ~60%% and cleans rankings.")
    parser.add_argument("--include-hyst", action="store_true",
                        help="v14e.20: opt-in to keep standalone `_HYST` strats. Default "
                             "excluded — paired-test on N>=55 paper main shows −8.9 to "
                             "−9.6pp vs base on FAST_TP50/TP80, and BE25_HYST/FAST_TP100_HYST "
                             "are net negative ($-171/$-220). `_S30_HYST` / `_NZS30_HYST` "
                             "(filter-gated) are kept regardless.")
    parser.add_argument("--include-smoothing-artefacts", action="store_true",
                        help="v14e.20: opt-in to keep `hysteresis` and `winsor_p95` smoothings "
                             "in the sweep dimension. Default excluded — both kernels hide "
                             "whipsaw that hits real paper execution and dominate sweep tops.")
    args = parser.parse_args()

    # v14: lock the chain filter for this run before any fetch fires.
    set_sim_chain(args.chain)
    print(f"[sim] chain filter: {args.chain}")

    global FLAT_POS_SIZE, USE_UNIFIED_SIM
    if args.flat_sizing > 0:
        FLAT_POS_SIZE = args.flat_sizing
    if args.legacy_sim:
        USE_UNIFIED_SIM = False
        print("LEGACY SIM: using old sim engines (flat 2.5% slippage, duplicated exit logic)")

    # v125: --divergence-report: show paper vs live exit price divergence
    if args.divergence_report:
        print("=" * 80)
        print("DIVERGENCE REPORT: Paper vs Live exit prices")
        print("=" * 80)
        params = [
            ("select", "symbol,strategy,status,paper_exit_price,exit_price,price_divergence_pct,exit_at"),
            ("source", "eq.rt_live"),
            ("price_divergence_pct", "not.is.null"),
            ("order", "exit_at.desc"),
            ("limit", "500"),
        ]
        trades = sb_get("paper_trades", params)
        if not trades:
            print("No live trades with divergence data found.")
            return

        all_divs = [abs(float(t["price_divergence_pct"])) for t in trades if t.get("price_divergence_pct")]
        by_status = {}
        by_strategy = {}
        for t in trades:
            div = abs(float(t["price_divergence_pct"]))
            status = t.get("status", "unknown")
            strat = t.get("strategy", "unknown")
            by_status.setdefault(status, []).append(div)
            by_strategy.setdefault(strat, []).append(div)

        print(f"\nTotal trades with divergence data: {len(all_divs)}")
        if all_divs:
            print(f"Mean absolute divergence: {statistics.mean(all_divs):.2%}")
            print(f"Median: {statistics.median(all_divs):.2%}")
            print(f"Max: {max(all_divs):.2%}")

        print(f"\n{'Exit Type':<15} {'N':>4} {'Mean Abs':>10} {'Median':>10} {'Max':>10}")
        print("-" * 55)
        for status, divs in sorted(by_status.items()):
            print(f"{status:<15} {len(divs):>4} {statistics.mean(divs):>10.2%} "
                  f"{statistics.median(divs):>10.2%} {max(divs):>10.2%}")

        print(f"\n{'Strategy':<30} {'N':>4} {'Mean Abs':>10}")
        print("-" * 50)
        for strat, divs in sorted(by_strategy.items(), key=lambda x: -statistics.mean(x[1])):
            print(f"{strat:<30} {len(divs):>4} {statistics.mean(divs):>10.2%}")

        # Flag large divergences
        large = [t for t in trades if abs(float(t.get("price_divergence_pct") or 0)) > 0.05]
        if large:
            print(f"\nTrades with >5% divergence: {len(large)}")
            for t in large[:10]:
                print(f"  {t['symbol']} {t['strategy']} {t['status']} "
                      f"paper=${float(t.get('paper_exit_price') or 0):.8f} "
                      f"actual=${float(t.get('exit_price') or 0):.8f} "
                      f"div={float(t['price_divergence_pct']):.2%}")
        return

    # v125: Auto-switch to --from-ticks when >14 days of tick data available
    if not args.from_ticks and not args.from_trades and not args.no_ticks and not args.grid_ticks:
        try:
            params = [
                ("select", "fetched_at"),
                ("order", "fetched_at.asc"),
                ("limit", "1"),
            ]
            earliest = sb_get("price_ticks", params)
            if earliest:
                import datetime as _dt_mod
                first_tick = _dt_mod.datetime.fromisoformat(
                    earliest[0]["fetched_at"].replace("Z", "+00:00"))
                days_of_data = (_dt_mod.datetime.now(_dt_mod.timezone.utc) - first_tick).days
                if days_of_data >= 14:
                    args.from_ticks = True
                    print(f"Tick data available ({days_of_data} days). "
                          f"Auto-switching to --from-ticks mode. "
                          f"Use --no-ticks to force OHLCV.")
        except Exception as e:
            print(f"Tick coverage check failed ({e}), using OHLCV mode")

    # =====================================================================
    # FROM-TRADES MODE: use real paper trade PnL, skip OHLCV entirely
    # =====================================================================
    if args.from_trades:
        print("=" * 80)
        print("FROM-TRADES MODE: Real paper trade results (ground truth)")
        print("=" * 80)

        all_trades = fetch_all_trades_by_strategy(args.since)
        if not all_trades:
            print("No trades found. Exiting.")
            return

        # Filter by token age
        if args.max_age > 0:
            all_trades = [t for t in all_trades
                          if t.get("rt_token_age_hours") is not None
                          and float(t["rt_token_age_hours"]) <= args.max_age]
            print(f"After age filter (<= {args.max_age}h): {len(all_trades)} trades")

        # Dedup: first trade per (token_address, strategy) within 24h
        sorted_trades = sorted(all_trades, key=lambda t: t["created_at"])
        seen_ts: dict[str, datetime] = {}  # (token, strategy) -> last time
        deduped = []
        for t in sorted_trades:
            key = f"{t['token_address']}_{t['strategy']}"
            dt = datetime.fromisoformat(t["created_at"].replace("Z", "+00:00"))
            last = seen_ts.get(key)
            if last and (dt - last).total_seconds() < 86400:
                continue
            seen_ts[key] = dt
            deduped.append(t)
        print(f"After dedup (first per token x strategy, 24h): {len(deduped)} trades")

        # Group by strategy
        by_strategy: dict[str, list[dict]] = defaultdict(list)
        for t in deduped:
            by_strategy[t["strategy"]].append(t)

        # Date range
        dates = [t["created_at"][:10] for t in deduped]
        date_range = (datetime.strptime(max(dates), "%Y-%m-%d") -
                      datetime.strptime(min(dates), "%Y-%m-%d")).days + 1
        print(f"Period: {min(dates)} -> {max(dates)} ({date_range} days)")
        print(f"Strategies with data: {len(by_strategy)}")

        # Compute metrics per strategy
        ranked = []
        for strat_name, trades in by_strategy.items():
            for wl_flag in [False, True]:
                if wl_flag:
                    strat_trades = [t for t in trades
                                    if (t.get("kol_group") or "").lower() in KOL_WHITELIST]
                else:
                    strat_trades = trades

                pnl_list = [float(t["pnl_pct"]) for t in strat_trades]
                if len(pnl_list) < MIN_TRADES:
                    continue

                n_tokens = len(set(t["token_address"] for t in strat_trades))
                metrics = compute_metrics(pnl_list, date_range)

                # Bankroll simulation
                br_trades = [{
                    "pnl_pct": float(t["pnl_pct"]),
                    "token_address": t["token_address"],
                    "created_at": t["created_at"],
                } for t in strat_trades]
                br = simulate_bankroll(sorted(br_trades, key=lambda x: x["created_at"]))

                # Detect strategy type from name
                stype = "UNKNOWN"
                for prefix in ["DIP_SCALE_OUT", "DIP", "DTRAIL", "TRAIL", "SCALP",
                                "DECAY", "SPLIT", "BE", "DYNAMIC_TRAIL", "CONTEXTUAL",
                                "SCALE_OUT", "FIXED", "TP"]:
                    if strat_name.startswith(prefix) or (prefix == "FIXED" and strat_name.startswith("TP")):
                        stype = prefix if prefix != "TP" else "FIXED"
                        break

                ranked.append({
                    "name": strat_name, "type": stype,
                    "whitelist": "YES" if wl_flag else "NO",
                    "n_tokens": n_tokens,
                    **metrics, **br,
                })

        ranked.sort(key=lambda x: -x["final_bankroll"])

        # Output
        top_n = args.top
        print(f"\n{'=' * 110}")
        print(f"TOP {top_n} STRATEGIES — FROM REAL PAPER TRADES (ground truth)")
        print(f"{'=' * 110}")
        header = (f"{'Rank':>4s}  {'Strategy':40s} {'WL?':4s} {'Type':12s} "
                  f"{'N':>5s} {'Tok':>4s} {'WR%':>5s} {'AvgPnL%':>8s} {'Sharpe':>7s} "
                  f"{'MaxDD%':>7s} {'Final$':>9s}")
        print(header)
        print("-" * len(header))
        for i, r in enumerate(ranked[:top_n]):
            print(f"{i+1:4d}  {r['name']:40s} {r['whitelist']:4s} {r['type']:12s} "
                  f"{r['n_trades']:5d} {r.get('n_tokens', 0):4d} {r['wr_pct']:4.0f}% "
                  f"{r['avg_pnl_pct']:+7.1f}% {r['sharpe']:7.2f} {r['max_dd_pct']:6.1f}% "
                  f"$ {r['final_bankroll']:8.0f}")

        # Best per type
        print(f"\n{'=' * 110}")
        print("BEST PER STRATEGY TYPE — FROM REAL TRADES")
        print(f"{'=' * 110}")
        for stype in ["FIXED", "DTRAIL", "TRAIL", "BE", "SCALP", "DECAY",
                       "DYNAMIC_TRAIL", "CONTEXTUAL", "SCALE_OUT", "DIP", "DIP_SCALE_OUT"]:
            all_r = [r for r in ranked if r["type"] == stype and r["whitelist"] == "NO"]
            wl_r = [r for r in ranked if r["type"] == stype and r["whitelist"] == "YES"]
            if all_r or wl_r:
                best_all = all_r[0] if all_r else None
                best_wl = wl_r[0] if wl_r else None
                print(f"  {stype:15s} "
                      f"All: {best_all['name']:40s} ${best_all['final_bankroll']:7.0f}" if best_all else f"  {stype:15s} All: {'—':40s}        ",
                      end="")
                if best_wl:
                    print(f"  WL: {best_wl['name']:40s} ${best_wl['final_bankroll']:7.0f}")
                else:
                    print()

        # Monte Carlo on top 5
        print(f"\n{'=' * 110}")
        print(f"MONTE CARLO (top 5, {args.mc_sims} sims x {args.mc_trades} trades)")
        print(f"{'=' * 110}")
        mc_header = (f"{'Strategy':45s} {'WL?':4s} {'Median$':>8s} {'P5$':>8s} "
                     f"{'P25$':>8s} {'P75$':>8s} {'P95$':>8s}")
        print(mc_header)
        print("-" * len(mc_header))
        for r in ranked[:5]:
            strat_trades = by_strategy.get(r["name"], [])
            if r["whitelist"] == "YES":
                strat_trades = [t for t in strat_trades
                                if (t.get("kol_group") or "").lower() in KOL_WHITELIST]
            pnl_list = [float(t["pnl_pct"]) for t in strat_trades]
            mc = monte_carlo(pnl_list, args.mc_sims, args.mc_trades)
            if mc:
                print(f"{r['name']:45s} {r['whitelist']:4s} $ {mc['median']:6.0f} "
                      f"$ {mc['p5']:6.0f} $ {mc['p25']:6.0f} $ {mc['p75']:6.0f} "
                      f"$ {mc['p95']:6.0f}")

        print(f"\nTotal ranked strategies: {len(ranked)}")
        return  # Skip OHLCV simulation path

    # =====================================================================
    # SYNTHETIC STRATEGY SWEEP (v134): test NEW strategies not in shadow DB
    # =====================================================================
    if getattr(args, "synthetic_sweep", False):
        _synthetic_strategy_sweep(args)
        return

    # =====================================================================
    # SMOOTHING SWEEP (v133): grade 8 decision-price smoothing modes
    # =====================================================================
    if getattr(args, "smoothing_sweep", False):
        _smoothing_sweep(args)
        return

    # =====================================================================
    # MEGA SWEEP (v140): full strategies × filters × sources × smoothings × polling
    # multiprocessing grid sweep. Uses _evaluate_trade_exit (v138.5 slip).
    # =====================================================================
    if (getattr(args, "mega_sweep", False)
            or getattr(args, "mega_sweep_extended", False)
            or getattr(args, "mega_sweep_eval_history", False)):
        _mega_sweep_run(args)
        return

    # =====================================================================
    # FROM-EVAL-HISTORY MODE (v138 B): replay from persisted eval_history.
    # This is mathematically perfect alignment for trades that recorded their
    # poll history (everything closed after v138 deploy).
    # =====================================================================
    if getattr(args, "from_eval_history", False):
        _eval_history_simulation(args)
        return

    # =====================================================================
    # FROM-TICKS MODE: tick-level replay through _evaluate_trade_exit
    # =====================================================================
    if args.from_ticks:
        _tick_based_simulation(args)
        return

    # --- Grid ---
    grid = build_strategy_grid(args.strategies)
    type_counts = defaultdict(int)
    for cfg in grid:
        type_counts[cfg["type"]] += 1

    print("=" * 80)
    print("UNIFIED STRATEGY SIMULATOR")
    print("=" * 80)
    print(f"\nStrategy grid: {len(grid)} configs x 2 (whitelist) = {len(grid) * 2} total")
    for t in ["FIXED", "DTRAIL", "TRAIL", "BE", "SCALP", "DECAY", "SPLIT", "DYNAMIC_TRAIL", "CONTEXTUAL", "SCALE_OUT", "DIP_BUY", "DIP_SCALE_OUT"]:
        if type_counts.get(t, 0) > 0:
            print(f"  {t}: {type_counts[t]}")

    max_horizon = max(cfg["horizon_min"] for cfg in grid)
    print(f"\nMax timeout: {max_horizon}min ({max_horizon/60:.1f}h)")

    # --- Fetch trades ---
    print("\n" + "-" * 80)
    all_trades = fetch_paper_trades(args.since)
    if not all_trades:
        print("No trades found. Exiting.")
        return

    # --- Dedup: first call only ---
    unique_tokens = dedup_first_call(all_trades)
    print(f"After first-call dedup: {len(unique_tokens)} unique entries "
          f"(removed {len(all_trades) - len(unique_tokens)} duplicates)")
    print(f"Unique token addresses: {len(set(t['token_address'] for t in unique_tokens))}")

    # --- Cache stats ---
    no_pair = sum(1 for t in unique_tokens if not t.get("pair_address"))
    print(f"Missing pair_address: {no_pair}")

    cache_hits = 0
    cache_misses = 0
    for t in unique_tokens:
        pool = t.get("pair_address") or _pair_cache.get(t["token_address"]) or "unknown"
        dt = datetime.fromisoformat(t["created_at"].replace("Z", "+00:00"))
        start_ts = int(dt.timestamp())
        if _load_cache(pool, start_ts, MAX_WINDOW_MIN) is not None:
            cache_hits += 1
        elif _load_legacy_cache(pool, start_ts) is not None:
            cache_hits += 1
        else:
            cache_misses += 1

    print(f"OHLCV cache: {cache_hits} hits, {cache_misses} misses")

    if args.dry_run:
        print(f"\n[DRY RUN] Would fetch OHLCV for up to {cache_misses} tokens. Exiting.")
        return

    # --- Resolve pairs ---
    if not args.cache_only:
        print("\n" + "-" * 80)
        print("RESOLVING PAIR ADDRESSES")
        missing_tokens = list(set(
            t["token_address"] for t in unique_tokens if not t.get("pair_address")
        ))
        resolve_pairs_batch(missing_tokens)

    # --- Fetch OHLCV ---
    print("\n" + "-" * 80)
    print("FETCHING OHLCV CANDLES" + (" (CACHE ONLY)" if args.cache_only else ""))

    candle_store: dict[str, list[dict]] = {}
    no_data_keys = set()
    api_calls = 0
    max_fetch = args.max_fetch if args.max_fetch > 0 else float("inf")

    for i, t in enumerate(unique_tokens):
        dt = datetime.fromisoformat(t["created_at"].replace("Z", "+00:00"))
        key = f"{t['token_address']}_{int(dt.timestamp())}"

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(unique_tokens)}] api={api_calls}, "
                  f"with_data={len(candle_store)}", flush=True)

        if not t.get("pair_address"):
            t["pair_address"] = _pair_cache.get(t["token_address"])

        if args.cache_only or api_calls >= max_fetch:
            pool = t.get("pair_address")
            if pool:
                start_ts = int(dt.timestamp())
                cached = _load_cache(pool, start_ts, MAX_WINDOW_MIN)
                if cached:
                    candle_store[key] = cached
                    continue
                legacy = _load_legacy_cache(pool, start_ts)
                if legacy:
                    candle_store[key] = legacy
                    continue
            no_data_keys.add(key)
            continue

        try:
            candles, called_api = fetch_candles_for_trade(t, cache_only=False)
        except Exception as e:
            print(f"  ERROR fetching {t['token_address'][:12]}: {e}")
            no_data_keys.add(key)
            continue
        if called_api:
            api_calls += 1
            time_mod.sleep(0.15)
        if candles and len(candles) >= 3:
            candle_store[key] = candles
        else:
            no_data_keys.add(key)

    _save_pair_cache(_pair_cache)
    print(f"\nOHLCV: {len(candle_store)} with data, {len(no_data_keys)} no data")

    if not candle_store:
        print("No OHLCV data available. Exiting.")
        return

    # --- Build trade entries ---
    trade_entries = []
    skipped_age = 0
    for t in unique_tokens:
        dt = datetime.fromisoformat(t["created_at"].replace("Z", "+00:00"))
        key = f"{t['token_address']}_{int(dt.timestamp())}"
        entry_price = float(t.get("entry_price") or 0)
        if entry_price <= 0 or key not in candle_store:
            continue
        # v113: Token age filter — match live behavior (default 12h)
        # NULL age = unknown = exclude (batch trades have no age data)
        raw_age = t.get("rt_token_age_hours")
        age_h = float(raw_age) if raw_age is not None else None
        if args.max_age > 0 and (age_h is None or age_h > args.max_age):
            skipped_age += 1
            continue
        # v113: Blacklist — bad KOLs and known bad tokens
        kol = (t.get("kol_group") or "unknown").lower()
        if kol in KOL_BLACKLIST or t["token_address"] in TOKEN_BLACKLIST:
            skipped_age += 1  # reuse counter for simplicity
            continue
        trade_entries.append({
            "token_address": t["token_address"],
            "created_at": t["created_at"],
            "kol_group": (t.get("kol_group") or "unknown").lower(),
            "entry_price": entry_price,
            "candles_key": key,
            "context": {
                "mcap": float(t.get("entry_mcap") or 0),
                "liq": float(t.get("rt_liquidity_usd") or 0),
                "vol24": float(t.get("rt_volume_24h") or 0),
                "age_h": float(t.get("rt_token_age_hours") or 0),
                "is_pump": int(t.get("rt_is_pump_fun") or 0),
                "n_kols": int(t.get("n_kol_confirmations") or 1),
                # v14e.28: chain + position propagate to sim_engines for chain-aware
                # buy/sell slip (EVM gets gas-folded slip, SOL keeps Jupiter Ultra median).
                "chain": t.get("chain") or "solana",
                "position_usd": float(t.get("position_usd") or 0),
            },
        })

    print(f"\n" + "-" * 80)
    print("RUNNING SIMULATIONS")
    print(f"Trade entries with OHLCV: {len(trade_entries)}")
    if skipped_age > 0:
        print(f"Skipped {skipped_age} tokens older than {args.max_age}h (--max-age)")
    if args.realistic:
        print(f"REALISTIC MODE: fast={args.fast_interval}s/{args.fast_window}min, slow={args.check_interval}min")
    print(f"Total simulations: {len(grid)} strategies x {len(trade_entries)} trades "
          f"= {len(grid) * len(trade_entries):,}")

    # Precompute resampled candles if realistic mode
    if args.realistic:
        live_candle_store = {}
        for key, candles in candle_store.items():
            live_candle_store[key] = resample_to_live_checks(
                candles, args.check_interval,
                fast_interval_sec=args.fast_interval,
                fast_window_min=args.fast_window)
        sim_store = live_candle_store
    else:
        sim_store = candle_store

    # --- Simulate ---
    t_start = time_mod.time()
    all_results: dict[str, dict[str, dict]] = {}  # name -> {candles_key -> result}
    all_pnl_lists: dict[str, list[float]] = {}

    for cfg in grid:
        name = cfg["name"]
        results = {}
        pnl_list = []
        for te in trade_entries:
            candles = sim_store[te["candles_key"]]
            res = simulate(candles, te["entry_price"], cfg, context=te.get("context"),
                          unified=USE_UNIFIED_SIM)
            res["token_address"] = te["token_address"]
            res["created_at"] = te["created_at"]
            res["kol_group"] = te["kol_group"]
            results[te["candles_key"]] = res
            pnl_list.append(res["pnl_pct"])
        all_results[name] = results
        all_pnl_lists[name] = pnl_list

    elapsed = time_mod.time() - t_start
    total_sims = len(grid) * len(trade_entries)
    print(f"Simulation complete in {elapsed:.1f}s ({total_sims / elapsed:,.0f} sims/sec)")

    # --- Compute date range ---
    dates = [te["created_at"][:10] for te in trade_entries]
    date_range = (datetime.strptime(max(dates), "%Y-%m-%d") -
                  datetime.strptime(min(dates), "%Y-%m-%d")).days + 1

    # --- Rank strategies (all KOLs + whitelist) ---
    ranked = []
    for cfg in grid:
        name = cfg["name"]
        for wl_flag in [False, True]:
            if wl_flag:
                pnl_list = [all_results[name][te["candles_key"]]["pnl_pct"]
                            for te in trade_entries
                            if te["kol_group"] in KOL_WHITELIST
                            and te["candles_key"] in all_results[name]]
                trade_sub = [te for te in trade_entries
                             if te["kol_group"] in KOL_WHITELIST]
            else:
                pnl_list = all_pnl_lists[name]
                trade_sub = trade_entries

            if len(pnl_list) < MIN_TRADES:
                continue

            metrics = compute_metrics(pnl_list, date_range)

            br_trades = []
            for te in trade_sub:
                if te["candles_key"] not in all_results[name]:
                    continue
                res = all_results[name][te["candles_key"]]
                br_trades.append({
                    "pnl_pct": res["pnl_pct"],
                    "token_address": te["token_address"],
                    "created_at": te["created_at"],
                })
            br = simulate_bankroll(sorted(br_trades, key=lambda x: x["created_at"]))

            ranked.append({
                "name": name, "type": cfg["type"],
                "whitelist": "YES" if wl_flag else "NO",
                **metrics, **br,
            })

    ranked.sort(key=lambda x: -x["final_bankroll"])

    # --- Output ---
    print(f"\n" + "-" * 80)
    print(f"COMPUTING METRICS")
    print(f"Ranked strategies: {len(ranked)} (min {MIN_TRADES} trades)")

    # TOP N
    top_n = args.top
    print(f"\n{'=' * 100}")
    print(f"TOP {top_n} STRATEGIES (by final bankroll, min {MIN_TRADES} trades)")
    print(f"{'=' * 100}")
    header = (f"{'Rank':>4s}  {'Strategy':35s} {'WL?':4s} {'Type':12s} "
              f"{'N':>4s} {'WR%':>5s} {'AvgPnL%':>8s} {'Sharpe':>7s} "
              f"{'MaxDD%':>7s} {'Kelly':>6s} {'Final$':>9s}")
    print(header)
    print("-" * len(header))
    for i, r in enumerate(ranked[:top_n]):
        print(f"{i+1:4d}  {r['name']:35s} {r['whitelist']:4s} {r['type']:12s} "
              f"{r['n_trades']:4d} {r['wr_pct']:4.0f}% {r['avg_pnl_pct']:+7.1f}% "
              f"{r['sharpe']:7.2f} {r['max_dd_pct']:6.1f}% {r['kelly']:6.3f} "
              f"$ {r['final_bankroll']:8.0f}")

    # BEST PER TYPE
    print(f"\n{'=' * 100}")
    print("BEST PER STRATEGY TYPE")
    print(f"{'=' * 100}")
    types_seen = set()
    print(f"{'Type':14s} {'Best (all KOLs)':35s} {'Final$':>8s}  "
          f"{'Best (whitelist)':35s} {'Final$':>8s}")
    print("-" * 110)
    for t in ["FIXED", "DTRAIL", "TRAIL", "BE", "SCALP", "DECAY", "SPLIT", "DYNAMIC_TRAIL", "CONTEXTUAL", "SCALE_OUT", "DIP_BUY", "DIP_SCALE_OUT"]:
        all_kol = [r for r in ranked if r["type"] == t and r["whitelist"] == "NO"]
        wl_kol = [r for r in ranked if r["type"] == t and r["whitelist"] == "YES"]
        best_all = all_kol[0]["name"] if all_kol else "-"
        best_all_v = all_kol[0]["final_bankroll"] if all_kol else 0
        best_wl = wl_kol[0]["name"] if wl_kol else "-"
        best_wl_v = wl_kol[0]["final_bankroll"] if wl_kol else 0
        if all_kol or wl_kol:
            print(f"{t:14s} {best_all:35s} ${best_all_v:7.0f}  "
                  f"{best_wl:35s} ${best_wl_v:7.0f}")

    # TIMEOUT ANALYSIS
    print(f"\n{'=' * 100}")
    print("TIMEOUT ANALYSIS (best final$ per type per timeout, all KOLs)")
    print(f"{'=' * 100}")
    timeouts = sorted(set(cfg["horizon_min"] for cfg in grid))
    types_for_to = ["FIXED", "DTRAIL", "DYNAMIC_TRAIL", "CONTEXTUAL", "SCALE_OUT", "DIP_BUY", "DIP_SCALE_OUT"]
    print(f"{'Timeout':>7s}  ", end="")
    for t in types_for_to:
        print(f"{t:>12s}", end="")
    print()
    print("-" * (8 + 12 * len(types_for_to)))
    for to in timeouts:
        print(f"{to:5d}min  ", end="")
        for t in types_for_to:
            matches = [r for r in ranked if r["type"] == t and r["whitelist"] == "NO"
                       and any(c["horizon_min"] == to and c["name"] == r["name"]
                               for c in grid)]
            if matches:
                print(f"$ {matches[0]['final_bankroll']:8.0f}  ", end="")
            else:
                print(f"{'':>12s}", end="")
        print()

    # WHITELIST IMPACT
    print(f"\n{'=' * 100}")
    print("WHITELIST IMPACT (top 10 strategies by delta)")
    print(f"{'=' * 100}")
    wl_impact = []
    for cfg in grid:
        name = cfg["name"]
        all_r = next((r for r in ranked if r["name"] == name and r["whitelist"] == "NO"), None)
        wl_r = next((r for r in ranked if r["name"] == name and r["whitelist"] == "YES"), None)
        if all_r and wl_r:
            wl_impact.append({
                "name": name,
                "all": all_r["final_bankroll"], "wl": wl_r["final_bankroll"],
                "delta": wl_r["final_bankroll"] - all_r["final_bankroll"],
                "wl_wr": wl_r["wr_pct"], "all_wr": all_r["wr_pct"],
            })
    wl_impact.sort(key=lambda x: -x["delta"])
    print(f"{'Strategy':35s} {'All KOLs':>10s} {'Whitelist':>10s} {'Delta':>10s}  "
          f"{'WL WR%':>6s} {'All WR%':>7s}")
    print("-" * 85)
    for w in wl_impact[:10]:
        print(f"{w['name']:35s} $ {w['all']:8.0f} $ {w['wl']:8.0f} $ {w['delta']:+7.0f}  "
              f"{w['wl_wr']:5.0f}% {w['all_wr']:6.0f}%")

    # KOL BREAKDOWN for #1
    if ranked:
        best = ranked[0]
        print(f"\n{'=' * 100}")
        print(f"KOL BREAKDOWN: #1 {best['name']} (WL={best['whitelist']})")
        print(f"{'=' * 100}")
        kol_stats = defaultdict(lambda: {"n": 0, "pnl_sum": 0, "wins": 0})
        for te in trade_entries:
            if te["candles_key"] not in all_results[best["name"]]:
                continue
            if best["whitelist"] == "YES" and te["kol_group"] not in KOL_WHITELIST:
                continue
            res = all_results[best["name"]][te["candles_key"]]
            kol = te["kol_group"]
            kol_stats[kol]["n"] += 1
            kol_stats[kol]["pnl_sum"] += res["pnl_pct"]
            if res["pnl_pct"] > 0:
                kol_stats[kol]["wins"] += 1

        kol_list = []
        for kol, s in kol_stats.items():
            avg_pnl = s["pnl_sum"] / s["n"]
            est_usd = s["pnl_sum"] * min(START_BANKROLL * KELLY_FRAC, MAX_POS)
            kol_list.append({"kol": kol, "n": s["n"], "wr": s["wins"] / s["n"] * 100,
                             "avg_pnl": avg_pnl * 100, "est_usd": est_usd})
        kol_list.sort(key=lambda x: -x["est_usd"])

        print(f"{'KOL':30s} {'N':>4s} {'WR%':>5s} {'Avg PnL%':>9s} {'~Total$':>9s}")
        print("-" * 60)
        for k in kol_list:
            print(f"{k['kol']:30s} {k['n']:4d} {k['wr']:4.0f}% {k['avg_pnl']:+8.1f}% "
                  f"$ {k['est_usd']:+7.0f}")

    # RUNNER ANALYSIS
    if args.runner_analysis:
        print(f"\n{'=' * 100}")
        print("RUNNER CAPTURE ANALYSIS (x2+ tokens)")
        print(f"{'=' * 100}")
        ra = runner_analysis(trade_entries, candle_store, grid, all_results)
        if ra:
            n_runners = ra[0]["n_runners"]
            print(f"Runners found: {n_runners}")
            print(f"\n{'Strategy':40s} {'CaptRate':>8s} {'AvgCapt%':>9s} "
                  f"{'AvgMax%':>8s} {'BestCapt%':>10s}")
            print("-" * 80)
            for r in ra[:30]:
                print(f"{r['name']:40s} {r['capture_rate']:7.0f}% "
                      f"{r['avg_capture_pct']:+8.1f}% {r['avg_max_pct']:+7.0f}% "
                      f"{r['best_capture_pct']:+9.1f}%")
        else:
            print("No x2+ runners found in dataset.")

    # MONTE CARLO
    print(f"\n{'=' * 100}")
    print(f"MONTE CARLO (top 5 strategies, {args.mc_sims} sims x {args.mc_trades} trades)")
    print(f"{'=' * 100}")
    mc_strats = ranked[:5]
    print(f"{'Strategy':40s} {'WL?':4s} {'Median$':>8s} {'P5$':>8s} {'P25$':>8s} "
          f"{'P75$':>8s} {'P95$':>8s} {'RoR%':>5s} {'Sharpe':>7s}")
    print("-" * 105)
    for r in mc_strats:
        pnl_list = all_pnl_lists.get(r["name"], [])
        if r["whitelist"] == "YES":
            pnl_list = [all_results[r["name"]][te["candles_key"]]["pnl_pct"]
                        for te in trade_entries
                        if te["kol_group"] in KOL_WHITELIST
                        and te["candles_key"] in all_results[r["name"]]]
        mc = monte_carlo(pnl_list, args.mc_sims, args.mc_trades)
        if mc:
            print(f"{r['name']:40s} {r['whitelist']:4s} $ {mc['median']:6.0f} "
                  f"$ {mc['p5']:6.0f} $ {mc['p25']:6.0f} $ {mc['p75']:6.0f} "
                  f"$ {mc['p95']:6.0f} {mc['ror_pct']:4.1f}% {r['sharpe']:7.2f}")

    # CHECK INTERVAL GRID SEARCH (old, per-strategy)
    if args.check_grid and ranked:
        n_cg = min(args.check_grid_top, len(ranked))
        cg_strats = ranked[:n_cg]

        # Grid: fast_interval_sec x slow_interval_min x fast_window_min
        fast_secs = [15, 30, 45, 60, 120, 180]      # how often to check in fast phase
        slow_mins = [1, 2, 3, 5, 10]                 # how often to check after fast phase
        fast_windows = [5, 15, 30, 60]               # how long the fast phase lasts (minutes)

        print(f"\n{'=' * 130}")
        print(f"CHECK INTERVAL GRID SEARCH (top {n_cg} strategies)")
        print(f"Fast intervals: {fast_secs}s | Slow intervals: {slow_mins}min | Fast windows: {fast_windows}min")
        print(f"{'=' * 130}")

        for r in cg_strats:
            name = r["name"]
            is_wl = r["whitelist"] == "YES"
            cfg = next(c for c in grid if c["name"] == name)

            strat_trades = [te for te in trade_entries
                           if (not is_wl or te["kol_group"] in KOL_WHITELIST)
                           and te["candles_key"] in candle_store]

            if len(strat_trades) < MIN_TRADES:
                continue

            print(f"\n  {name} (WL={r['whitelist']}, N={len(strat_trades)})")

            # Collect results for all combos
            combo_results = []
            for fw in fast_windows:
                for fs in fast_secs:
                    for sm in slow_mins:
                        pnl_list = []
                        for te in strat_trades:
                            raw = candle_store[te["candles_key"]]
                            resampled = resample_to_live_checks(
                                raw, interval_min=sm,
                                fast_interval_sec=fs,
                                fast_window_min=fw)
                            res = simulate(resampled, te["entry_price"], cfg,
                                          context=te.get("context"),
                                          unified=USE_UNIFIED_SIM)
                            pnl_list.append(res["pnl_pct"])

                        avg_pnl = statistics.mean(pnl_list) * 100
                        wr = sum(1 for p in pnl_list if p > 0) / len(pnl_list) * 100
                        br = simulate_bankroll([{
                            "pnl_pct": pnl_list[i],
                            "token_address": strat_trades[i]["token_address"],
                            "created_at": strat_trades[i]["created_at"],
                        } for i in range(len(pnl_list))])

                        combo_results.append({
                            "fast_sec": fs, "slow_min": sm, "fast_win": fw,
                            "avg_pnl": avg_pnl, "wr": wr,
                            "final_br": br["final_bankroll"],
                            "label": f"fast={fs}s/{fw}min slow={sm}min",
                        })

            # Sort by bankroll
            combo_results.sort(key=lambda x: -x["final_br"])

            # Print top 15 and bottom 5
            header = (f"    {'Rank':>4s} {'Fast Check':>10s} {'Window':>7s} {'Slow Check':>10s} "
                      f"{'WR%':>5s} {'AvgPnL%':>8s} {'Final$':>8s}")
            print(header)
            print(f"    {'-' * (len(header) - 4)}")
            for i, cr in enumerate(combo_results[:15]):
                marker = " <-- BEST" if i == 0 else ""
                marker = " <-- CURRENT" if cr["fast_sec"] == 30 and cr["slow_min"] == 3 and cr["fast_win"] == 30 else marker
                print(f"    {i+1:4d} {cr['fast_sec']:>7d}s  {cr['fast_win']:>5d}min {cr['slow_min']:>7d}min "
                      f"{cr['wr']:4.0f}% {cr['avg_pnl']:+7.1f}% ${cr['final_br']:7.0f}{marker}")
            if len(combo_results) > 15:
                print(f"    {'...':>4s}")
                for cr in combo_results[-3:]:
                    marker = " <-- CURRENT" if cr["fast_sec"] == 30 and cr["slow_min"] == 3 and cr["fast_win"] == 30 else ""
                    rank = combo_results.index(cr) + 1
                    print(f"    {rank:4d} {cr['fast_sec']:>7d}s  {cr['fast_win']:>5d}min {cr['slow_min']:>7d}min "
                          f"{cr['wr']:4.0f}% {cr['avg_pnl']:+7.1f}% ${cr['final_br']:7.0f}{marker}")

    # STRATEGY × INTERVAL CROSS-GRID
    if args.interval_cross:
        PROFILES = [
            ("AGGRESSIVE", 15, 30, 1),    # fast=15s, window=30min, slow=1min
            ("CURRENT",    30, 30, 3),     # what prod does now
            ("MODERATE",   60, 15, 3),     # slightly less frequent
            ("RELAXED",   120,  5, 5),     # calibrated to match DTRAIL10 reality
            ("LAZY",      180,  5, 10),    # infrequent checks
        ]

        print(f"\n{'=' * 140}")
        print(f"STRATEGY × INTERVAL CROSS-GRID ({len(grid)} strategies × {len(PROFILES)} profiles × {len(trade_entries)} trades)")
        print(f"{'=' * 140}")

        # Pre-compute resampled candles for each profile
        profile_stores = {}
        for pname, fs, fw, sm in PROFILES:
            store = {}
            for key, candles in candle_store.items():
                store[key] = resample_to_live_checks(candles, interval_min=sm,
                                                      fast_interval_sec=fs,
                                                      fast_window_min=fw)
            profile_stores[pname] = store
            print(f"  Resampled {pname}: fast={fs}s/{fw}min, slow={sm}min")

        # Run all strategies × all profiles
        cross_results = {}  # (strategy_name, profile) -> {metrics}
        t_cross = time_mod.time()
        total_cross = len(grid) * len(PROFILES) * len(trade_entries)

        for pi, (pname, fs, fw, sm) in enumerate(PROFILES):
            p_store = profile_stores[pname]
            for cfg in grid:
                name = cfg["name"]
                pnl_list = []
                for te in trade_entries:
                    candles = p_store[te["candles_key"]]
                    res = simulate(candles, te["entry_price"], cfg, context=te.get("context"),
                                  unified=USE_UNIFIED_SIM)
                    pnl_list.append(res["pnl_pct"])

                if len(pnl_list) < MIN_TRADES:
                    continue

                wr = sum(1 for p in pnl_list if p > 0) / len(pnl_list) * 100
                avg_pnl = statistics.mean(pnl_list) * 100
                br_trades = [{"pnl_pct": pnl_list[i],
                              "token_address": trade_entries[i]["token_address"],
                              "created_at": trade_entries[i]["created_at"]}
                             for i in range(len(pnl_list))]
                br = simulate_bankroll(sorted(br_trades, key=lambda x: x["created_at"]))

                cross_results[(name, pname)] = {
                    "name": name, "type": cfg["type"], "profile": pname,
                    "wr": wr, "avg_pnl": avg_pnl, "final_br": br["final_bankroll"],
                    "n": len(pnl_list),
                }

            print(f"  Profile {pi+1}/{len(PROFILES)} {pname} done")

        elapsed_cross = time_mod.time() - t_cross
        print(f"  Cross-grid complete: {total_cross:,} sims in {elapsed_cross:.1f}s "
              f"({total_cross / elapsed_cross:,.0f} sims/sec)")

        # --- Report 1: Best (strategy, interval) combos overall ---
        all_combos = sorted(cross_results.values(), key=lambda x: -x["final_br"])

        print(f"\n  {'-' * 120}")
        print(f"  TOP 30 (STRATEGY × INTERVAL) COMBOS")
        print(f"  {'-' * 120}")
        header = f"  {'Rank':>4s} {'Strategy':40s} {'Profile':>12s} {'WR%':>5s} {'AvgPnL%':>8s} {'Final$':>8s}"
        print(header)
        print(f"  {'-' * (len(header) - 2)}")
        for i, c in enumerate(all_combos[:30]):
            print(f"  {i+1:4d} {c['name']:40s} {c['profile']:>12s} {c['wr']:4.0f}% "
                  f"{c['avg_pnl']:+7.1f}% ${c['final_br']:7.0f}")

        # --- Report 2: Best interval per strategy TYPE ---
        print(f"\n  {'-' * 120}")
        print(f"  BEST INTERVAL PER STRATEGY TYPE")
        print(f"  {'-' * 120}")
        strat_types = ["FIXED", "DTRAIL", "TRAIL", "BE", "SCALP", "DECAY",
                       "DYNAMIC_TRAIL", "CONTEXTUAL", "SCALE_OUT", "DIP_BUY", "DIP_SCALE_OUT"]
        header2 = f"  {'Type':15s}"
        for pname, _, _, _ in PROFILES:
            header2 += f" {pname:>14s}"
        print(header2)
        print(f"  {'-' * (15 + 15 * len(PROFILES))}")

        for stype in strat_types:
            line = f"  {stype:15s}"
            for pname, _, _, _ in PROFILES:
                matches = [v for v in cross_results.values()
                           if v["type"] == stype and v["profile"] == pname]
                if matches:
                    best = max(matches, key=lambda x: x["final_br"])
                    line += f" ${best['final_br']:>12.0f}"
                else:
                    line += f" {'—':>14s}"
            print(line)

        # --- Report 3: For each strategy type, best strategy + best interval ---
        print(f"\n  {'-' * 120}")
        print(f"  OPTIMAL STRATEGY + INTERVAL PER TYPE")
        print(f"  {'-' * 120}")
        print(f"  {'Type':15s} {'Best Strategy':40s} {'Best Interval':>14s} {'WR%':>5s} {'AvgPnL%':>8s} {'Final$':>8s}")
        print(f"  {'-' * 95}")

        for stype in strat_types:
            matches = [v for v in cross_results.values() if v["type"] == stype]
            if matches:
                best = max(matches, key=lambda x: x["final_br"])
                print(f"  {stype:15s} {best['name']:40s} {best['profile']:>14s} "
                      f"{best['wr']:4.0f}% {best['avg_pnl']:+7.1f}% ${best['final_br']:7.0f}")

        # --- Report 4: Strategies that CHANGE RANK significantly across profiles ---
        print(f"\n  {'-' * 120}")
        print(f"  INTERVAL-SENSITIVE STRATEGIES (biggest rank change CURRENT vs RELAXED)")
        print(f"  {'-' * 120}")

        # Rank within CURRENT and RELAXED profiles
        current_ranked = sorted([v for v in cross_results.values() if v["profile"] == "CURRENT"],
                                key=lambda x: -x["final_br"])
        relaxed_ranked = sorted([v for v in cross_results.values() if v["profile"] == "RELAXED"],
                                key=lambda x: -x["final_br"])
        current_rank = {v["name"]: i+1 for i, v in enumerate(current_ranked)}
        relaxed_rank = {v["name"]: i+1 for i, v in enumerate(relaxed_ranked)}

        deltas = []
        for name in current_rank:
            if name in relaxed_rank:
                cr = current_rank[name]
                rr = relaxed_rank[name]
                cur_br = next(v["final_br"] for v in current_ranked if v["name"] == name)
                rel_br = next(v["final_br"] for v in relaxed_ranked if v["name"] == name)
                deltas.append({"name": name, "cur_rank": cr, "rel_rank": rr,
                               "rank_delta": cr - rr, "cur_br": cur_br, "rel_br": rel_br,
                               "br_delta": rel_br - cur_br})

        # Show strategies that improve most with RELAXED
        deltas.sort(key=lambda x: -x["br_delta"])
        print(f"  Strategies that IMPROVE most with RELAXED interval:")
        print(f"  {'Strategy':40s} {'CURRENT$':>9s} {'RELAXED$':>9s} {'Delta$':>8s}  {'Cur Rank':>8s} {'Rel Rank':>8s}")
        print(f"  {'-' * 80}")
        for d in deltas[:10]:
            print(f"  {d['name']:40s} ${d['cur_br']:8.0f} ${d['rel_br']:8.0f} "
                  f"${d['br_delta']:+7.0f}  #{d['cur_rank']:>6d} #{d['rel_rank']:>6d}")

        # Show strategies that get WORSE with RELAXED
        print(f"\n  Strategies that get WORSE with RELAXED interval:")
        print(f"  {'Strategy':40s} {'CURRENT$':>9s} {'RELAXED$':>9s} {'Delta$':>8s}  {'Cur Rank':>8s} {'Rel Rank':>8s}")
        print(f"  {'-' * 80}")
        for d in deltas[-10:]:
            print(f"  {d['name']:40s} ${d['cur_br']:8.0f} ${d['rel_br']:8.0f} "
                  f"${d['br_delta']:+7.0f}  #{d['cur_rank']:>6d} #{d['rel_rank']:>6d}")

        # --- Report 5: DTRAIL3 specifically across all profiles ---
        print(f"\n  {'-' * 120}")
        print(f"  DTRAIL3 variants across ALL profiles (the strategy you questioned)")
        print(f"  {'-' * 120}")
        dtrail3_strats = sorted(set(v["name"] for v in cross_results.values()
                                    if "DTRAIL3" in v["name"]),
                                key=lambda n: -max(v["final_br"] for v in cross_results.values()
                                                   if v["name"] == n))[:5]
        for strat in dtrail3_strats:
            print(f"\n  {strat}:")
            print(f"  {'Profile':>14s} {'WR%':>5s} {'AvgPnL%':>8s} {'Final$':>8s}")
            for pname, _, _, _ in PROFILES:
                key = (strat, pname)
                if key in cross_results:
                    v = cross_results[key]
                    marker = " <-- PROD" if pname == "CURRENT" else ""
                    print(f"  {pname:>14s} {v['wr']:4.0f}% {v['avg_pnl']:+7.1f}% ${v['final_br']:7.0f}{marker}")

    # DUAL-WALLET ANALYSIS
    if args.dual_wallet and ranked:
        deltas = [float(d) for d in args.dual_deltas.split(",")]
        n_dual = min(args.dual_top, len(ranked))
        dual_strats = ranked[:n_dual]

        print(f"\n{'=' * 120}")
        print(f"DUAL-WALLET ANALYSIS (top {n_dual} strategies, deltas: {deltas})")
        print(f"2 independent wallets, each full position. Liquidity-aware slippage.")
        print(f"{'=' * 120}")

        for r in dual_strats:
            name = r["name"]
            is_wl = r["whitelist"] == "YES"
            cfg = next(c for c in grid if c["name"] == name)

            # Collect trades for this strategy
            strat_trades = []
            for te in trade_entries:
                if is_wl and te["kol_group"] not in KOL_WHITELIST:
                    continue
                if te["candles_key"] not in sim_store:
                    continue
                strat_trades.append(te)

            if len(strat_trades) < MIN_TRADES:
                continue

            print(f"\n  {name} (WL={r['whitelist']}, N={len(strat_trades)})")
            header = (f"    {'Delta':>6s} | {'WA PnL%':>8s} {'WB PnL%':>8s} {'Combined':>9s} "
                      f"{'2x Total':>9s} | {'Single%':>8s} {'vs Sngl':>8s} | "
                      f"{'WA Slip':>7s} {'WB Slip':>7s} | "
                      f"{'WA BR$':>8s} {'WB BR$':>8s} {'Total$':>8s} {'1x BR$':>8s}")
            print(header)
            print(f"    {'-' * (len(header) - 4)}")

            for delta in deltas:
                wa_pnls = []
                wb_pnls = []
                single_pnls = []
                wa_slips = []
                wb_slips = []
                wa_trades_br = []
                wb_trades_br = []
                skipped = 0

                for te in strat_trades:
                    candles = sim_store[te["candles_key"]]
                    pos = args.dual_position if args.dual_position > 0 else min(
                        START_BANKROLL * KELLY_FRAC, MAX_POS)

                    raw_candles = candle_store[te["candles_key"]]
                    dw = simulate_dual_wallet(
                        candles, raw_candles, te["entry_price"], cfg,
                        context=te.get("context"),
                        delta_min=delta,
                        position_usd=pos,
                    )
                    if dw is None:
                        skipped += 1
                        continue

                    wa_pnls.append(dw["wa"]["pnl_pct"])
                    wb_pnls.append(dw["wb"]["pnl_pct"])
                    single_pnls.append(dw["single"]["pnl_pct"])
                    wa_slips.append(dw["wa_slip"])
                    wb_slips.append(dw["wb_slip"])
                    wa_trades_br.append({
                        "pnl_pct": dw["wa"]["pnl_pct"],
                        "token_address": te["token_address"],
                        "created_at": te["created_at"],
                    })
                    wb_trades_br.append({
                        "pnl_pct": dw["wb"]["pnl_pct"],
                        "token_address": te["token_address"],
                        "created_at": te["created_at"],
                    })

                if len(wa_pnls) < MIN_TRADES:
                    continue

                avg_wa = statistics.mean(wa_pnls) * 100
                avg_wb = statistics.mean(wb_pnls) * 100
                avg_single = statistics.mean(single_pnls) * 100
                avg_combined = (avg_wa + avg_wb) / 2
                total_2x = avg_wa + avg_wb
                vs_single = avg_combined - avg_single
                avg_wa_slip = statistics.mean(wa_slips) * 100
                avg_wb_slip = statistics.mean(wb_slips) * 100

                # Bankroll sim
                br_wa = simulate_bankroll(sorted(wa_trades_br, key=lambda x: x["created_at"]))
                br_wb = simulate_bankroll(sorted(wb_trades_br, key=lambda x: x["created_at"]))

                delta_label = f"{int(delta)}min" if delta > 0 else "0(sim)"
                print(f"    {delta_label:>6s} | {avg_wa:+7.1f}% {avg_wb:+7.1f}% {avg_combined:+8.1f}% "
                      f"{total_2x:+8.1f}% | {avg_single:+7.1f}% {vs_single:+7.1f}% | "
                      f"{avg_wa_slip:6.2f}% {avg_wb_slip:6.2f}% | "
                      f"${br_wa['final_bankroll']:7.0f} ${br_wb['final_bankroll']:7.0f} "
                      f"${br_wa['final_bankroll'] + br_wb['final_bankroll']:7.0f} "
                      f"${r['final_bankroll']:7.0f}")
                if skipped > 0:
                    print(f"           (skipped {skipped} trades — no OHLCV at delta)")

    # CSV
    csv_path = SCRAPER_DIR / "grid_search_results.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "rank", "name", "whitelist", "type", "n_trades", "wr_pct",
            "avg_pnl_pct", "median_pnl_pct", "sharpe", "max_dd_pct",
            "kelly", "final_bankroll", "total_pnl_usd",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, r in enumerate(ranked):
            writer.writerow({
                "rank": i + 1,
                **{k: r.get(k) for k in fieldnames if k != "rank"},
            })
    print(f"\nCSV saved to: {csv_path}")
    print(f"Total ranked strategies: {len(ranked)}")


if __name__ == "__main__":
    # v140: needed for multiprocessing on Windows (--mega-sweep mode)
    import multiprocessing as _mp
    _mp.freeze_support()
    main()
