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

from sim_engines import simulate, resample_to_live_checks

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
    """Generate all strategy configurations."""
    configs = []

    # 1. FIXED: TP{40..100} x SL{30..70} x timeout
    for tp in [40, 50, 60, 70, 80, 90, 100]:
        for sl in [30, 40, 50, 60, 70]:
            for timeout in [30, 45, 60, 120, 240, 360]:
                configs.append({
                    "name": f"TP{tp}_SL{sl}_{timeout}m",
                    "type": "FIXED",
                    "tp_mult": 1 + tp / 100,
                    "sl_mult": 1 - sl / 100,
                    "horizon_min": timeout,
                })

    # 2. DTRAIL: trail{5,10} x act{10..30} x SL{50..70} x timeout
    for trail in [5, 10]:
        for act in [10, 15, 20, 25, 30]:
            for sl in [50, 60, 70]:
                for timeout in [120, 240, 360]:
                    configs.append({
                        "name": f"DTRAIL{trail}_ACT{act}_SL{sl}_{timeout}m",
                        "type": "DTRAIL",
                        "trail_pct": trail / 100,
                        "activation_pct": act / 100,
                        "sl_mult": 1 - sl / 100,
                        "horizon_min": timeout,
                    })

    # 3. TRAIL: trail x TP x SL x timeout
    for trail in [10, 15]:
        for tp in [50, 100]:
            for sl in [50, 60, 70]:
                for timeout in [120, 240, 360]:
                    configs.append({
                        "name": f"TRAIL{trail}_TP{tp}_SL{sl}_{timeout}m",
                        "type": "TRAIL",
                        "trail_pct": trail / 100,
                        "activation_pct": trail / 100,
                        "tp_mult": 1 + tp / 100,
                        "sl_mult": 1 - sl / 100,
                        "horizon_min": timeout,
                    })

    # 4. BE: activation x TP x SL x timeout
    for be_act in [15, 20, 30]:
        for tp in [50, 100]:
            for sl in [50, 60, 70]:
                for timeout in [60, 120, 240]:
                    configs.append({
                        "name": f"BE{be_act}_TP{tp}_SL{sl}_{timeout}m",
                        "type": "BE",
                        "be_activation": be_act / 100,
                        "tp_mult": 1 + tp / 100,
                        "sl_mult": 1 - sl / 100,
                        "horizon_min": timeout,
                    })

    # 5. SCALP: TP{10..20} x SL{10..20} x timeout{30,60}
    for tp in [10, 15, 20]:
        for sl in [10, 15, 20]:
            for timeout in [30, 60]:
                configs.append({
                    "name": f"SCALP_TP{tp}_SL{sl}_{timeout}m",
                    "type": "SCALP",
                    "tp_mult": 1 + tp / 100,
                    "sl_mult": 1 - sl / 100,
                    "horizon_min": timeout,
                })

    # 6. DECAY: tp_start/end x SL x timeout
    for tp_s, tp_e, label_tp in [(2.0, 1.20, "TP100_E20"), (2.0, 1.15, "TP100_E15"),
                                  (2.0, 1.30, "TP100_E30"), (1.70, 1.15, "TP70_E15")]:
        for sl in [50, 60, 70]:
            for timeout in [120, 240, 360]:
                configs.append({
                    "name": f"DECAY_{label_tp}_SL{sl}_{timeout}m", "type": "DECAY",
                    "tp_start": tp_s, "tp_end": tp_e, "sl_mult": 1 - sl / 100,
                    "horizon_min": timeout,
                })

    # 7. SPLIT: variants x SL x timeout
    for label, t1, t2, t2t in [
        ("SPLIT_50_100", 1.50, 2.0, None),
        ("SPLIT_50_TRAIL", 1.50, None, 0.20),
    ]:
        for sl in [50, 60, 70]:
            for timeout in [120, 240, 360]:
                configs.append({
                    "name": f"{label}_SL{sl}_{timeout}m", "type": "SPLIT",
                    "t1_tp": t1, "t2_tp": t2, "sl_mult": 1 - sl / 100, "t2_trail": t2t,
                    "horizon_min": timeout,
                })

    # 8. DYNAMIC_TRAIL strategies
    configs.extend(_build_dynamic_trail_grid(
        full=strategy_filter and "dynamic_full" in strategy_filter))

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
    """Build dynamic trail strategy configs."""
    configs = []

    # --- TIME DECAY ---
    if full:
        td_combos = [(s, e) for s in [5, 8, 10, 12, 15, 20]
                      for e in [3, 4, 5, 8, 10, 15] if s != e]
        td_timeouts = [120, 240, 360]
        td_acts = [15, 20, 25]
        td_sls = [50, 60, 70]
    else:
        td_combos = [
            (15, 3), (12, 4), (20, 5), (10, 3),  # wide -> tight
            (3, 15), (4, 12),                       # tight -> wide (NEW)
        ]
        td_timeouts = [240, 360]
        td_acts = [15, 20]
        td_sls = [60, 70]

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

    # --- GAIN ADAPTIVE ---
    profiles = [
        ("STD", [30, 100, 300], [0.05, 0.10, 0.15, 0.20]),
        ("AGG", [20, 50, 150], [0.05, 0.08, 0.12, 0.18]),
        ("PAT", [50, 150, 400], [0.05, 0.10, 0.15, 0.25]),
    ]
    ga_timeouts = [240, 360] if not full else [120, 240, 360]
    ga_sls = [60, 70] if not full else [50, 60, 70]

    for label, thresholds, trails in profiles:
        for timeout in ga_timeouts:
            for sl in ga_sls:
                configs.append({
                    "name": f"GADAPT_{label}_SL{sl}_{timeout}m",
                    "type": "DYNAMIC_TRAIL", "mode": "gain_adaptive",
                    "gain_thresholds": thresholds, "gain_trails": trails,
                    "activation_pct": 0.20, "sl_mult": 1 - sl / 100,
                    "horizon_min": timeout,
                })

    # --- GAIN-TIME HYBRID ---
    for timeout in [120, 240, 360]:
        for sl in [50, 60, 70]:
            configs.append({
                "name": f"GTHYBRID_SL{sl}_{timeout}m",
                "type": "DYNAMIC_TRAIL", "mode": "gain_time_hybrid",
                "activation_pct": 0.20, "sl_mult": 1 - sl / 100,
                "horizon_min": timeout,
            })

    # --- RATCHET TRAIL ---
    milestone_sets = [
        ("STD", [(30, 10, 5), (50, 25, 7), (100, 50, 10), (200, 120, 15), (400, 250, 20)]),
        ("AGG", [(20, 5, 5), (40, 15, 8), (80, 35, 12), (150, 80, 15)]),
    ]
    for ms_label, milestones in milestone_sets:
        for timeout in [240, 360]:
            for act in [15, 20]:
                for sl in [60, 70]:
                    configs.append({
                        "name": f"RATCHET_{ms_label}_ACT{act}_SL{sl}_{timeout}m",
                        "type": "DYNAMIC_TRAIL", "mode": "ratchet_trail",
                        "milestones": milestones, "trail_base": 5,
                        "activation_pct": act / 100, "sl_mult": 1 - sl / 100,
                        "horizon_min": timeout,
                    })

    # --- TIME-GAIN RATCHET ---
    for timeout in [240, 360]:
        for act in [15, 20]:
            for sl in [60, 70]:
                configs.append({
                    "name": f"TGRATCHET_ACT{act}_SL{sl}_{timeout}m",
                    "type": "DYNAMIC_TRAIL", "mode": "time_gain_ratchet",
                    "milestones": [(30, 10, 8), (50, 25, 10), (100, 50, 12), (200, 120, 15)],
                    "trail_base": 5,
                    "activation_pct": act / 100, "sl_mult": 1 - sl / 100,
                    "horizon_min": timeout,
                })

    return configs


def _build_contextual_grid() -> list[dict]:
    """Build CONTEXTUAL strategies — trail/timeout vary by mcap segment."""
    configs = []
    # Each set: (label, breakpoints, trails_per_seg, timeouts_per_seg, acts_per_seg)
    segmentation_sets = [
        # 2 segments: micro vs rest
        ("2SEG_100K", [100000],
         [[15, 5], [12, 5], [20, 8]],
         [[360, 120], [360, 240]],
         [[25, 15], [20, 15]]),
        # 3 segments: micro / small / mid+
        ("3SEG", [100000, 1000000],
         [[15, 10, 5], [20, 10, 5], [12, 8, 5]],
         [[360, 240, 120], [360, 360, 240]],
         [[25, 20, 15], [30, 20, 15]]),
        # 2 segments: small vs big
        ("2SEG_500K", [500000],
         [[12, 5], [15, 5], [10, 5]],
         [[360, 120], [360, 240]],
         [[20, 15], [25, 15]]),
    ]
    for seg_label, bps, trail_combos, to_combos, act_combos in segmentation_sets:
        for ti, trails in enumerate(trail_combos):
            for toi, timeouts in enumerate(to_combos):
                for ai, acts in enumerate(act_combos):
                    for sl in [60, 70]:
                        name = f"CTX_{seg_label}_t{ti}_to{toi}_a{ai}_SL{sl}"
                        configs.append({
                            "name": name, "type": "CONTEXTUAL",
                            "mcap_breakpoints": bps,
                            "trail_per_segment": trails,
                            "timeout_per_segment": timeouts,
                            "act_per_segment": acts,
                            "sl_mult": 1 - sl / 100,
                            "horizon_min": max(timeouts),  # for grid stats
                        })
    return configs


def _build_scale_out_grid() -> list[dict]:
    """Build SCALE_OUT strategies — progressive exit in tranches."""
    configs = []
    tranche_configs = [
        ("SO_30_60_100", [(30, 0.25), (60, 0.25), (100, 0.25)]),  # 25% runner
        ("SO_25_50_100", [(25, 0.25), (50, 0.25), (100, 0.25)]),
        ("SO_30_60", [(30, 0.33), (60, 0.33)]),  # 33% runner
        ("SO_50_100", [(50, 0.33), (100, 0.33)]),  # 33% runner, patient
        ("SO_20_40_80", [(20, 0.25), (40, 0.25), (80, 0.25)]),  # aggressive
    ]
    for label, tranches in tranche_configs:
        for runner_trail in [10, 15, 20]:
            for runner_act in [30, 50]:
                for sl in [60, 70]:
                    for timeout in [240, 360]:
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
    """Build DIP_BUY strategies — re-enter after dump + bounce."""
    configs = []
    # Full DIP grid: ALL params variable
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

    # DIP + SCALE_OUT: ALL params variable
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

    # SCALE_OUT: also vary runner_act
    return configs


# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------

def sb_get(table: str, params: list[tuple]) -> list[dict]:
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


def fetch_paper_trades(since: str) -> list[dict]:
    params = [
        ("select", "id,token_address,pair_address,strategy,entry_price,exit_price,"
                   "status,pnl_pct,created_at,kol_group,source,high_price_seen,position_usd,"
                   "entry_mcap,rt_liquidity_usd,rt_volume_24h,rt_token_age_hours,"
                   "rt_is_pump_fun,n_kol_confirmations"),
        ("status", "in.(trail_stop,sl_hit,timeout,tp_hit)"),
        ("created_at", f"gte.{since}T00:00:00Z"),
        ("order", "created_at.asc"),
    ]
    trades = sb_get("paper_trades", params)
    print(f"Fetched {len(trades)} closed paper trades since {since}")
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


def fetch_candles_for_trade(trade: dict, cache_only: bool = False) -> tuple[list[dict] | None, bool]:
    pool = trade.get("pair_address")
    token = trade.get("token_address")
    dt = datetime.fromisoformat(trade["created_at"].replace("Z", "+00:00"))
    start_ts = int(dt.timestamp())
    end_ts = start_ts + MAX_WINDOW_MIN * 60

    if not pool:
        pool = _pair_cache.get(token) or (resolve_pair_address(token) if not cache_only else None)
        if not pool:
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
                        help="Check interval in minutes for --realistic mode")
    parser.add_argument("--max-age", type=float, default=12.0,
                        help="Max token age in hours (live default: 12h). 0=no filter")
    args = parser.parse_args()

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
        age_h = float(t.get("rt_token_age_hours") or 0)
        if args.max_age > 0 and age_h > args.max_age:
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
            },
        })

    print(f"\n" + "-" * 80)
    print("RUNNING SIMULATIONS")
    print(f"Trade entries with OHLCV: {len(trade_entries)}")
    if skipped_age > 0:
        print(f"Skipped {skipped_age} tokens older than {args.max_age}h (--max-age)")
    if args.realistic:
        print(f"REALISTIC MODE: resampling to {args.check_interval}min spot checks")
    print(f"Total simulations: {len(grid)} strategies x {len(trade_entries)} trades "
          f"= {len(grid) * len(trade_entries):,}")

    # Precompute resampled candles if realistic mode
    if args.realistic:
        live_candle_store = {}
        for key, candles in candle_store.items():
            live_candle_store[key] = resample_to_live_checks(candles, args.check_interval)
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
            res = simulate(candles, te["entry_price"], cfg, context=te.get("context"))
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
    main()
