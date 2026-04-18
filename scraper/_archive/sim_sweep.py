"""
v142 — Coarse grid sweep + walk-forward validation + slippage haircut
for the 3 data-validated strategies (TIME_DECAY_V2, BOND_FAST, PEAK_TRAIL_V2).

Stages:
  1. Load closed paper trades with eval_history (v138+) from Supabase.
  2. For each strategy, generate ~200-650 config variants.
  3. Replay every (trade, config) via tick-by-tick eval.
  4. Split 14d into 4 weekly folds (walk-forward).
  5. Compute fold-mean PnL + fold-min + stability (min/mean).
  6. Apply per-exit-type slippage haircut (based on v141 audit calibration).
  7. Output top 5 per strategy + full CSV.

Usage:
    python scraper/sim_sweep.py                    # default 14d, 4 folds, haircut ON
    python scraper/sim_sweep.py --days 14 --folds 4
    python scraper/sim_sweep.py --no-haircut       # raw replay results
    python scraper/sim_sweep.py --strat td2        # only TIME_DECAY_V2
"""

import argparse
import csv
import itertools
import json
import os
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client


SCRAPER_DIR = Path(__file__).resolve().parent
load_dotenv(SCRAPER_DIR / ".env")


# ---- Slippage haircut (per-exit-type, bps) --------------------------------
# Calibrated on v141 audit findings (Apr 17 2026):
#   - sl_hit bonding: +1165 bps observed; sim unslipped = 430 bps gap -> 430 haircut
#   - trail_crash:   800 bps (memory: trail_crash when crash_ratio<0.70)
#   - tp_hit:         near-zero on Jupiter Ultra RFQ (10-30 bps)
#   - timeout:       120 bps (normal exit, thin books on stale tokens)
# All values are BASIS POINTS subtracted from the raw sim exit_return.
HAIRCUT_BPS = {
    "tp_hit":      30,    # Ultra RFQ tight
    "tp_late":     80,    # late exit, thinner
    "sl_hit":      300,   # base; +bonding_extra below
    "be_stop":     200,   # breakeven stop (active mgmt, exec at ~entry)
    "trail_stop":  250,
    "trail_crash": 800,
    "timeout":     120,
    "noexit":        0,   # no trade booked
}
BONDING_EXTRA_BPS = 400  # extra penalty for bonding SL/trail_crash (drain + high spread)


# ---- Replay primitives ----------------------------------------------------

def _parse_ts(s):
    if isinstance(s, datetime):
        return s
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _preload_ticks(trade):
    """Decode eval_history into a sorted list of (minutes, price, high). Done once
    per trade; replay functions iterate this list for each config variant."""
    eh = trade.get("eval_history") or []
    if not eh:
        return []
    try:
        created = _parse_ts(trade["created_at"])
    except Exception:
        return []
    entry = float(trade.get("entry_price") or 0)
    if entry <= 0:
        return []
    ticks = []
    for poll in eh:
        try:
            t = _parse_ts(poll["t"])
            mins = (t - created).total_seconds() / 60.0
            d = float(poll.get("d") or 0) or None
            e = float(poll.get("e") or 0) or None
            h = float(poll.get("h") or 0) or None
            price = e if (e and e > 0) else d
            if price is None or price <= 0:
                continue
            ticks.append((mins, price, h if (h and h > 0) else price))
        except (KeyError, TypeError, ValueError):
            continue
    ticks.sort(key=lambda x: x[0])
    return ticks


def _resample_ticks(ticks, polling_sec):
    """Downsample ticks to simulate a coarser polling interval. Keeps the first
    tick at or after each n*polling_sec boundary. Required for polling sensitivity."""
    if polling_sec <= 30 or not ticks:
        return ticks
    step_min = polling_sec / 60.0
    out = []
    next_thresh = 0.0
    for t in ticks:
        if t[0] >= next_thresh:
            out.append(t)
            next_thresh = t[0] + step_min
    return out


# ---- Strategy replays (parametrized) --------------------------------------

def replay_time_decay(ticks, entry, cfg):
    """TIME_DECAY_V2 replay.

    cfg keys: be_minute, tp_start, tp_mid, tp_late, sl_start, timeout_min
    TP schedule is piecewise-linear: (0, tp_start) -> (be_minute, tp_mid) ->
    (be_minute+10, tp_late) -> (timeout, tp_late).
    SL: sl_start until be_minute, then max(sl_start, entry).
    """
    if not ticks:
        return {"taken": False}
    be_minute = cfg["be_minute"]
    tp_start = cfg["tp_start"]
    tp_mid = cfg["tp_mid"]
    tp_late = cfg["tp_late"]
    sl_start = entry * cfg["sl_start"]
    timeout = cfg["timeout_min"]

    decay_bps = [(0, tp_start), (be_minute, tp_mid), (be_minute + 10, tp_late), (timeout, tp_late)]

    def _tp_mult(mins):
        for i in range(len(decay_bps) - 1):
            m1, v1 = decay_bps[i]
            m2, v2 = decay_bps[i + 1]
            if m1 <= mins <= m2:
                if m2 == m1:
                    return v2
                return v1 + (v2 - v1) * (mins - m1) / (m2 - m1)
        return decay_bps[-1][1]

    for mins, price, high in ticks:
        if mins >= timeout:
            return {"taken": True, "status": "timeout", "pnl_pct": (price / entry) - 1, "exit_min": mins}
        effective_sl = max(sl_start, entry if mins >= be_minute else 0.0)
        if effective_sl > 0 and price <= effective_sl:
            status = "sl_hit" if effective_sl == sl_start else "be_stop"
            return {"taken": True, "status": status, "pnl_pct": (effective_sl / entry) - 1, "exit_min": mins}
        tp_m = _tp_mult(mins)
        if tp_m > 1.0:
            tp_px = entry * tp_m
            if price >= tp_px:
                return {"taken": True, "status": "tp_hit", "pnl_pct": (tp_px / entry) - 1, "exit_min": mins}
        elif tp_m <= 1.0 and mins >= be_minute + 10 and price >= entry:
            return {"taken": True, "status": "tp_late", "pnl_pct": (price / entry) - 1, "exit_min": mins}
    return {"taken": True, "status": "noexit", "pnl_pct": 0.0, "exit_min": 0}


def replay_bond_fast(ticks, entry, cfg, liq_usd, is_bonding):
    """BOND_FAST replay with entry filter on liquidity.

    cfg keys: max_liq, tp_mult, sl_mult, trail_pct, trail_act, timeout_min
    """
    if not is_bonding and liq_usd >= cfg["max_liq"]:
        return {"taken": False, "reason": "liq_too_high"}
    if not ticks:
        return {"taken": False}
    tp = entry * cfg["tp_mult"]
    sl = entry * cfg["sl_mult"]
    trail_pct = cfg["trail_pct"]
    trail_act_p = entry * (1 + cfg["trail_act"])
    timeout = cfg["timeout_min"]

    high = entry
    trail_armed = False
    for mins, price, h in ticks:
        high = max(high, h, price)
        if mins >= timeout:
            return {"taken": True, "status": "timeout", "pnl_pct": (price / entry) - 1, "exit_min": mins}
        if price <= sl:
            return {"taken": True, "status": "sl_hit", "pnl_pct": (sl / entry) - 1, "exit_min": mins}
        if price >= tp:
            return {"taken": True, "status": "tp_hit", "pnl_pct": (tp / entry) - 1, "exit_min": mins}
        if trail_pct and high >= trail_act_p:
            trail_armed = True
        if trail_armed and trail_pct:
            trig = high * (1 - trail_pct)
            if price <= trig and trig > entry:
                return {"taken": True, "status": "trail_stop", "pnl_pct": (trig / entry) - 1, "exit_min": mins}
    return {"taken": True, "status": "noexit", "pnl_pct": 0.0, "exit_min": 0}


def replay_standard(ticks, entry, cfg):
    """Generic replay for standard STRATEGIES dict entries (single-tranche).
    Supports: tp_mult, sl_mult, horizon_min, be_activation, trail_pct,
    trail_activation_pct, tp_decay_end, trail_tiers, time_be_minute, tp_schedule.

    Order of checks (matches paper_trader._evaluate_trade_exit):
      1. SL (with BE override from be_activation or time_be_minute)
      2. TP (scalar OR tp_schedule OR tp_decay_end)
      3. Trail stop (trail_pct OR trail_tiers)
      4. Timeout
    """
    if not ticks:
        return {"taken": False}
    sl_mult = cfg.get("sl_mult", 0.5)
    tp_mult = cfg.get("tp_mult")
    horizon = cfg.get("horizon_min", 120)
    be_act = cfg.get("be_activation")
    time_be = cfg.get("time_be_minute")
    trail_pct = cfg.get("trail_pct")
    trail_act = cfg.get("trail_activation_pct") or (trail_pct or 0)
    trail_tiers = cfg.get("trail_tiers")
    tp_schedule = cfg.get("tp_schedule")
    tp_decay_end = cfg.get("tp_decay_end")

    sl_base = entry * sl_mult
    tp_px = entry * tp_mult if tp_mult else None
    high = entry

    for mins, price, h in ticks:
        high = max(high, h, price)
        if mins >= horizon:
            return {"taken": True, "status": "timeout", "pnl_pct": (price / entry) - 1, "exit_min": mins}

        # Effective SL: peak-based BE OR time-based BE
        effective_sl = sl_base
        if be_act is not None and high >= entry * (1 + be_act):
            effective_sl = max(effective_sl, entry)
        if time_be is not None and mins >= float(time_be):
            effective_sl = max(effective_sl, entry)

        if price <= effective_sl:
            status = "be_stop" if effective_sl > sl_base else "sl_hit"
            return {"taken": True, "status": status, "pnl_pct": (effective_sl / entry) - 1, "exit_min": mins}

        # TP via tp_schedule (piecewise-linear)
        if tp_schedule:
            tp_m = None
            for i in range(len(tp_schedule) - 1):
                m1, v1 = tp_schedule[i]; m2, v2 = tp_schedule[i + 1]
                if m1 <= mins <= m2:
                    tp_m = v2 if m2 == m1 else v1 + (v2 - v1) * (mins - m1) / (m2 - m1)
                    break
            if tp_m is None:
                tp_m = tp_schedule[-1][1] if mins >= tp_schedule[-1][0] else tp_schedule[0][1]
            if tp_m > 1.0 and price >= entry * tp_m:
                return {"taken": True, "status": "tp_hit", "pnl_pct": tp_m - 1, "exit_min": mins}
            elif tp_m <= 1.0 and price >= entry:
                return {"taken": True, "status": "tp_late", "pnl_pct": (price / entry) - 1, "exit_min": mins}
        # TP scalar
        elif tp_px and price >= tp_px:
            return {"taken": True, "status": "tp_hit", "pnl_pct": tp_mult - 1, "exit_min": mins}
        # TP decay (legacy)
        elif tp_px and tp_decay_end is not None:
            activation_frac = 0.5
            if mins > horizon * activation_frac:
                progress = min((mins - horizon * activation_frac) / (horizon * (1 - activation_frac)), 1.0)
                decayed_mult = tp_mult - (tp_mult - tp_decay_end) * progress
                if price >= entry * decayed_mult:
                    return {"taken": True, "status": "tp_hit", "pnl_pct": decayed_mult - 1, "exit_min": mins}

        # Trail (tiered OR scalar)
        if trail_tiers and high > entry:
            ratio = high / entry
            eff_trail = sorted(trail_tiers)[0][1]
            for pm, pp in sorted(trail_tiers):
                if ratio >= pm:
                    eff_trail = pp
                else:
                    break
            if high >= entry * (1 + trail_act):
                trig = high * (1 - eff_trail)
                if price <= trig and trig > entry:
                    status = "trail_crash" if (price / trig) < 0.70 else "trail_stop"
                    return {"taken": True, "status": status, "pnl_pct": (trig / entry) - 1, "exit_min": mins}
        elif trail_pct and high >= entry * (1 + trail_act):
            trig = high * (1 - trail_pct)
            if price <= trig and trig > entry:
                status = "trail_crash" if (price / trig) < 0.70 else "trail_stop"
                return {"taken": True, "status": status, "pnl_pct": (trig / entry) - 1, "exit_min": mins}
    return {"taken": True, "status": "noexit", "pnl_pct": 0.0, "exit_min": 0}


def replay_peak_trail(ticks, entry, cfg):
    """PEAK_TRAIL_V2 replay with tiered trail.

    cfg keys: tiers (list of [peak_mult, trail_pct]), trail_act, sl_mult, timeout_min
    """
    if not ticks:
        return {"taken": False}
    tiers = sorted(cfg["tiers"])
    sl = entry * cfg["sl_mult"]
    timeout = cfg["timeout_min"]
    trail_act_p = entry * (1 + cfg["trail_act"])

    def _trail_pct(peak):
        ratio = peak / entry if entry > 0 else 1.0
        best = tiers[0][1]
        for mult, pct in tiers:
            if ratio >= mult:
                best = pct
            else:
                break
        return best

    high = entry
    trail_armed = False
    for mins, price, h in ticks:
        high = max(high, h, price)
        if mins >= timeout:
            return {"taken": True, "status": "timeout", "pnl_pct": (price / entry) - 1, "exit_min": mins}
        if price <= sl:
            return {"taken": True, "status": "sl_hit", "pnl_pct": (sl / entry) - 1, "exit_min": mins}
        if high >= trail_act_p:
            trail_armed = True
        if trail_armed:
            trig = high * (1 - _trail_pct(high))
            if price <= trig and trig > entry:
                status = "trail_crash" if (price / trig) < 0.70 else "trail_stop"
                return {"taken": True, "status": status, "pnl_pct": (trig / entry) - 1, "exit_min": mins}
    return {"taken": True, "status": "noexit", "pnl_pct": 0.0, "exit_min": 0}


# ---- Haircut --------------------------------------------------------------

def apply_haircut(result, is_bonding):
    """Subtract per-exit-type slippage (bps) from pnl_pct."""
    if not result.get("taken"):
        return result
    status = result.get("status", "noexit")
    bps = HAIRCUT_BPS.get(status, 100)
    if is_bonding and status in ("sl_hit", "trail_crash"):
        bps += BONDING_EXTRA_BPS
    new_pct = result["pnl_pct"] - bps / 10_000
    return {**result, "pnl_pct": new_pct, "haircut_bps": bps}


# ---- Grid definitions -----------------------------------------------------

def grid_time_decay():
    grid = []
    for be in (3, 5, 7, 10):
        for tp_start in (1.50, 1.80, 2.00):
            for tp_mid in (1.20, 1.40):
                if tp_mid >= tp_start:
                    continue
                for sl_start in (0.65, 0.70, 0.75):
                    for timeout in (20, 30, 45):
                        grid.append({
                            "be_minute": be, "tp_start": tp_start,
                            "tp_mid": tp_mid, "tp_late": 1.00,
                            "sl_start": sl_start, "timeout_min": timeout,
                        })
    return grid


def grid_time_decay_fine():
    """Fine grid around the coarse-sweep top 5: be=5, tp_start=2.00, tp_mid~1.40,
    sl_start=0.65, timeout=30."""
    grid = []
    for be in (3, 4, 5, 6, 7):
        for tp_start in (1.70, 1.80, 1.90, 2.00, 2.20, 2.50):
            for tp_mid in (1.25, 1.30, 1.35, 1.40, 1.45, 1.50):
                if tp_mid >= tp_start:
                    continue
                for tp_late in (1.00, 1.05, 1.10, 1.15):
                    if tp_late >= tp_mid:
                        continue
                    for sl_start in (0.55, 0.60, 0.65, 0.70):
                        for timeout in (20, 25, 30, 35, 45):
                            grid.append({
                                "be_minute": be, "tp_start": tp_start,
                                "tp_mid": tp_mid, "tp_late": tp_late,
                                "sl_start": sl_start, "timeout_min": timeout,
                            })
    return grid


def grid_bond_fast():
    grid = []
    for max_liq in (1_000, 3_000, 10_000):
        for tp_mult in (1.30, 1.40, 1.50, 1.80):
            for sl_mult in (0.70, 0.75, 0.80):
                for trail_pct in (0.0, 0.10, 0.15, 0.20):  # 0 = no trail
                    for trail_act in (0.15, 0.25):
                        for timeout in (15, 20, 30):
                            grid.append({
                                "max_liq": max_liq, "tp_mult": tp_mult,
                                "sl_mult": sl_mult, "trail_pct": trail_pct,
                                "trail_act": trail_act, "timeout_min": timeout,
                            })
    return grid


def grid_peak_trail():
    grid = []
    # Compact: vary each tier's trail_pct independently but keep peaks fixed
    for t1 in (0.10, 0.15):
        for t2 in (0.18, 0.25):
            for t3 in (0.30, 0.40):
                for t4 in (0.45, 0.55):
                    for trail_act in (0.15, 0.20, 0.25):
                        for sl_mult in (0.50, 0.60, 0.70):
                            for timeout in (30, 60, 90):
                                grid.append({
                                    "tiers": [(1.30, t1), (1.80, t2), (3.00, t3), (6.00, t4)],
                                    "trail_act": trail_act, "sl_mult": sl_mult,
                                    "timeout_min": timeout,
                                })
    return grid


# ---- Walk-forward + scoring -----------------------------------------------

def split_folds(trades, k):
    """Chronological split: oldest -> newest, equal-sized folds."""
    sorted_trades = sorted(trades, key=lambda t: t.get("created_at") or "")
    n = len(sorted_trades)
    fold_size = max(1, n // k)
    folds = [sorted_trades[i * fold_size:(i + 1) * fold_size] for i in range(k - 1)]
    folds.append(sorted_trades[(k - 1) * fold_size:])
    return folds


def score_strategy_name(strat_name, tranche, folds, apply_haircut_flag, filt=None):
    """Replay an existing STRATEGIES-dict entry (via replay_standard) on each
    fold. tranche: single-tranche dict from STRATEGIES. filt: optional filter
    dict with min_liquidity_usd / max_liquidity_usd / etc.
    Position sizing: fixed $50 per trade for fair cross-strat comparison.
    """
    POS_USD = 50.0
    fold_pnl_usd = []
    fold_wr = []
    fold_n = []

    for fold_trades in folds:
        pnl_usd = 0.0
        taken_count = 0
        winners = 0
        for trade in fold_trades:
            # Apply entry filter if strategy has one
            if filt:
                liq = float(trade.get("rt_liquidity_usd") or 0)
                if filt.get("min_liquidity_usd") and liq < filt["min_liquidity_usd"]:
                    continue
                if filt.get("max_liquidity_usd") and liq > filt["max_liquidity_usd"]:
                    continue

            entry = float(trade.get("entry_price") or 0)
            if entry <= 0:
                continue

            res = replay_standard(trade["_ticks"], entry, tranche)
            if not res.get("taken"):
                continue
            if apply_haircut_flag:
                res = apply_haircut(res, bool(trade.get("rt_is_pump_fun")))
            pnl_usd += res["pnl_pct"] * POS_USD
            taken_count += 1
            if res["pnl_pct"] > 0:
                winners += 1

        fold_pnl_usd.append(round(pnl_usd, 2))
        fold_wr.append(round(winners / taken_count, 3) if taken_count else 0)
        fold_n.append(taken_count)

    mean_pnl = statistics.mean(fold_pnl_usd) if fold_pnl_usd else 0
    min_pnl = min(fold_pnl_usd) if fold_pnl_usd else 0
    pos_folds = sum(1 for x in fold_pnl_usd if x > 0)
    stability = pos_folds / len(fold_pnl_usd) if fold_pnl_usd else 0
    total_taken = sum(fold_n)

    return {
        "strat_name": strat_name,
        "fold_pnl_usd": fold_pnl_usd,
        "fold_wr": fold_wr,
        "fold_n": fold_n,
        "mean_pnl_usd": round(mean_pnl, 2),
        "min_pnl_usd": round(min_pnl, 2),
        "total_pnl_usd": round(sum(fold_pnl_usd), 2),
        "total_taken": total_taken,
        "stability": stability,
    }


def run_mega_sweep(trades, folds, apply_haircut_flag):
    """Mega sweep: iterate every single-tranche entry in STRATEGIES dict +
    test them on the same trade set. Excludes multi-tranche (MOONBAG/SCALE_OUT/
    SPLIT/WIDE_RUNNER/DIP_split) for clean cross-comparison.
    """
    from strategies import STRATEGIES, STRATEGY_FILTERS
    results = []
    skipped_multi = 0
    for strat_name, tranches in STRATEGIES.items():
        if len(tranches) > 1:
            skipped_multi += 1
            continue
        tranche = tranches[0]
        filt = STRATEGY_FILTERS.get(strat_name)
        score = score_strategy_name(strat_name, tranche, folds, apply_haircut_flag, filt)
        score["cfg"] = tranche
        score["filter"] = filt
        results.append(score)
    return results, skipped_multi


def score_config(strat_name, cfg, folds, apply_haircut_flag):
    """Replay cfg on each fold; return per-fold aggregate stats.

    Returns dict with keys: fold_pnl_usd (list), fold_wr (list), fold_n_taken (list),
    mean_pnl_usd, min_pnl_usd, pnl_stability (min/mean clamped), total_n_taken.
    """
    fold_pnl_usd = []
    fold_wr = []
    fold_n = []
    total_taken = 0
    total_pnl_usd = 0.0

    for fold_trades in folds:
        pnl_usd = 0.0
        taken_count = 0
        winners = 0
        for trade in fold_trades:
            ticks = trade["_ticks"]
            entry = float(trade.get("entry_price") or 0)
            if entry <= 0:
                continue

            if strat_name == "td2":
                res = replay_time_decay(ticks, entry, cfg)
            elif strat_name == "bond":
                liq = float(trade.get("rt_liquidity_usd") or 0)
                bond = bool(trade.get("rt_is_pump_fun"))
                res = replay_bond_fast(ticks, entry, cfg, liq, bond)
            elif strat_name == "ptrail":
                res = replay_peak_trail(ticks, entry, cfg)
            else:
                continue

            if not res.get("taken"):
                continue
            if apply_haircut_flag:
                res = apply_haircut(res, bool(trade.get("rt_is_pump_fun")))
            pnl_usd += res["pnl_pct"] * float(trade.get("position_usd") or 0)
            taken_count += 1
            if res["pnl_pct"] > 0:
                winners += 1

        fold_pnl_usd.append(round(pnl_usd, 2))
        fold_wr.append(round(winners / taken_count, 3) if taken_count else 0)
        fold_n.append(taken_count)
        total_taken += taken_count
        total_pnl_usd += pnl_usd

    mean_pnl = statistics.mean(fold_pnl_usd) if fold_pnl_usd else 0
    min_pnl = min(fold_pnl_usd) if fold_pnl_usd else 0
    pos_folds = sum(1 for x in fold_pnl_usd if x > 0)
    # Stability: fraction of folds positive (1.0 = always positive)
    stability = pos_folds / len(fold_pnl_usd) if fold_pnl_usd else 0

    return {
        "fold_pnl_usd": fold_pnl_usd,
        "fold_wr": fold_wr,
        "fold_n": fold_n,
        "mean_pnl_usd": round(mean_pnl, 2),
        "min_pnl_usd": round(min_pnl, 2),
        "total_pnl_usd": round(total_pnl_usd, 2),
        "total_taken": total_taken,
        "stability": stability,
    }


# ---- Main driver ----------------------------------------------------------

def fetch_trades(client, since_iso, min_N_history=3):
    resp = (
        client.table("paper_trades")
        .select("id,symbol,token_address,strategy,source,status,pnl_pct,pnl_usd,"
                "entry_price,exit_price,position_usd,rt_liquidity_usd,rt_is_pump_fun,"
                "created_at,exit_at,eval_history")
        .neq("status", "open")
        .neq("status", "closing")
        .eq("is_shadow", False)
        .gte("created_at", since_iso)
        .not_.is_("eval_history", "null")
        .limit(5000)
        .execute()
    )
    rows = resp.data or []
    trades = []
    for r in rows:
        eh = r.get("eval_history")
        if not isinstance(eh, list) or len(eh) < min_N_history:
            continue
        r["_ticks"] = _preload_ticks(r)
        if len(r["_ticks"]) >= min_N_history:
            trades.append(r)
    return trades


def top_k(results, k=5):
    """Rank by (mean_pnl_usd × stability) — rewards high+consistent."""
    scored = [(r["mean_pnl_usd"] * max(r["stability"], 0.25), r) for r in results]
    scored.sort(key=lambda x: -x[0])
    return scored[:k]


def format_cfg(strat_name, cfg):
    if strat_name == "td2":
        return (f"be{cfg['be_minute']}m tp{int((cfg['tp_start']-1)*100)}->"
                f"{int((cfg['tp_mid']-1)*100)}->late sl{int((1-cfg['sl_start'])*100)} "
                f"t{cfg['timeout_min']}m")
    if strat_name == "bond":
        trail = f"trail{int(cfg['trail_pct']*100)}@{int(cfg['trail_act']*100)}" if cfg['trail_pct'] > 0 else "notrail"
        return (f"liq<${cfg['max_liq']//1000}K tp{int((cfg['tp_mult']-1)*100)} "
                f"sl{int((1-cfg['sl_mult'])*100)} {trail} t{cfg['timeout_min']}m")
    if strat_name == "ptrail":
        tiers = cfg["tiers"]
        pcts = "/".join(str(int(p*100)) for _, p in tiers)
        return (f"tiers{pcts} act{int(cfg['trail_act']*100)} "
                f"sl{int((1-cfg['sl_mult'])*100)} t{cfg['timeout_min']}m")
    return str(cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--strat", type=str, default="all",
                        choices=["all", "td2", "bond", "ptrail"])
    parser.add_argument("--fine-td2", action="store_true",
                        help="Use fine TD2 grid (~3K configs) instead of coarse (216)")
    parser.add_argument("--mega", action="store_true",
                        help="Iterate EVERY single-tranche STRATEGIES entry via replay_standard")
    parser.add_argument("--no-haircut", action="store_true",
                        help="Disable slippage haircut (raw replay, optimistic)")
    parser.add_argument("--out", type=str, default=str(SCRAPER_DIR / "sim_sweep.csv"))
    parser.add_argument("--top-json", type=str, default=str(SCRAPER_DIR / "sim_sweep_top.json"))
    parser.add_argument("--min_N_history", type=int, default=3)
    args = parser.parse_args()

    apply_haircut_flag = not args.no_haircut

    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    client = create_client(url, key)

    since = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()
    print(f"=== sim_sweep — {args.days}d, {args.folds} folds, haircut={'ON' if apply_haircut_flag else 'OFF'} ===\n")
    print(f"Loading trades since {since}...")
    trades = fetch_trades(client, since, min_N_history=args.min_N_history)
    print(f"[COUNTS] loaded {len(trades)} trades with eval_history>={args.min_N_history}\n")
    if not trades:
        print("No trades. Aborting.")
        return

    folds = split_folds(trades, args.folds)
    print(f"[FOLDS] sizes: {[len(f) for f in folds]}\n")

    strat_grids = {
        "td2": grid_time_decay_fine() if args.fine_td2 else grid_time_decay(),
        "bond": grid_bond_fast(),
        "ptrail": grid_peak_trail(),
    }
    strat_labels = {
        "td2": "TIME_DECAY_V2",
        "bond": "BOND_FAST",
        "ptrail": "PEAK_TRAIL_V2",
    }

    selected = [args.strat] if args.strat != "all" else ["td2", "bond", "ptrail"]

    all_rows = []
    top_results = {}

    # Mega-sweep branch: iterate ALL STRATEGIES dict entries (single-tranche)
    if args.mega:
        print(f"[MEGA] iterating every single-tranche entry in STRATEGIES...")
        mega_results, skipped = run_mega_sweep(trades, folds, apply_haircut_flag)
        print(f"[MEGA] scored {len(mega_results)} strategies (skipped {skipped} multi-tranche)\n")

        # Rank
        def _rank_key(r):
            return r["mean_pnl_usd"] * max(r["stability"], 0.25)
        mega_results.sort(key=_rank_key, reverse=True)

        print(f"{'Strategy':<45} {'mean$':>8} {'min$':>8} {'total$':>9} {'stab':>5} {'N':>5} folds_pnl")
        print("-" * 130)
        for r in mega_results[:25]:
            print(f"{r['strat_name']:<45} {r['mean_pnl_usd']:>+8.2f} {r['min_pnl_usd']:>+8.2f} "
                  f"{r['total_pnl_usd']:>+9.2f} {r['stability']:>5.2f} {r['total_taken']:>5} {r['fold_pnl_usd']}")
        print("\n--- BOTTOM 10 (worst) ---")
        for r in mega_results[-10:]:
            print(f"{r['strat_name']:<45} {r['mean_pnl_usd']:>+8.2f} {r['min_pnl_usd']:>+8.2f} "
                  f"{r['total_pnl_usd']:>+9.2f} {r['stability']:>5.2f} {r['total_taken']:>5}")

        # Save full CSV
        import csv as _csv
        mega_csv = Path(args.out).with_name("sim_mega_sweep.csv")
        with open(mega_csv, "w", newline="", encoding="utf-8") as f:
            writer = _csv.writer(f)
            writer.writerow(["strat_name", "mean_pnl_usd", "min_pnl_usd", "total_pnl_usd",
                             "total_taken", "stability", "fold_pnl_usd", "fold_wr", "fold_n",
                             "cfg_json", "filter_json"])
            for r in mega_results:
                writer.writerow([
                    r["strat_name"], r["mean_pnl_usd"], r["min_pnl_usd"], r["total_pnl_usd"],
                    r["total_taken"], round(r["stability"], 3),
                    json.dumps(r["fold_pnl_usd"]), json.dumps(r["fold_wr"]),
                    json.dumps(r["fold_n"]), json.dumps(r["cfg"], default=str),
                    json.dumps(r.get("filter") or {}, default=str),
                ])
        print(f"\n[OK] Mega CSV ({len(mega_results)} strats) written to {mega_csv}")

        mega_json = Path(args.top_json).with_name("sim_mega_sweep_top.json")
        mega_json.write_text(json.dumps({
            "days": args.days, "folds": args.folds, "haircut": apply_haircut_flag,
            "n_trades": len(trades),
            "top_30": mega_results[:30],
            "bottom_10": mega_results[-10:],
        }, indent=2, default=str))
        print(f"[OK] Mega top-30 JSON written to {mega_json}")
        return

    for strat_name in selected:
        grid = strat_grids[strat_name]
        print(f"[{strat_labels[strat_name]}] sweeping {len(grid)} configs × {args.folds} folds × {len(trades)} trades...")
        results = []
        for cfg in grid:
            score = score_config(strat_name, cfg, folds, apply_haircut_flag)
            score["cfg"] = cfg
            score["strat"] = strat_labels[strat_name]
            score["desc"] = format_cfg(strat_name, cfg)
            results.append(score)
            all_rows.append({
                "strat": strat_labels[strat_name],
                "desc": score["desc"],
                "mean_pnl_usd": score["mean_pnl_usd"],
                "min_pnl_usd": score["min_pnl_usd"],
                "total_pnl_usd": score["total_pnl_usd"],
                "total_taken": score["total_taken"],
                "stability": round(score["stability"], 3),
                "fold_pnl_usd": json.dumps(score["fold_pnl_usd"]),
                "fold_wr": json.dumps(score["fold_wr"]),
                "cfg_json": json.dumps(cfg, default=str),
            })

        top5 = top_k(results, 5)
        top_results[strat_labels[strat_name]] = [{
            "desc": r["desc"], "cfg": r["cfg"],
            "mean_pnl_usd": r["mean_pnl_usd"], "min_pnl_usd": r["min_pnl_usd"],
            "total_pnl_usd": r["total_pnl_usd"], "total_taken": r["total_taken"],
            "stability": r["stability"], "fold_pnl_usd": r["fold_pnl_usd"],
            "fold_wr": r["fold_wr"],
        } for _, r in top5]

        print(f"\n--- {strat_labels[strat_name]} TOP 5 ---")
        print(f"{'Config':<50} {'mean$':>8} {'min$':>8} {'total$':>9} {'stab':>5} {'N':>4} folds_pnl")
        for score_val, r in top5:
            print(f"{r['desc']:<50} {r['mean_pnl_usd']:>+8.2f} {r['min_pnl_usd']:>+8.2f} "
                  f"{r['total_pnl_usd']:>+9.2f} {r['stability']:>5.2f} {r['total_taken']:>4} {r['fold_pnl_usd']}")
        print()

    # Write CSV
    out_path = Path(args.out)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        if all_rows:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
    print(f"[OK] CSV ({len(all_rows)} rows) written to {out_path}")

    # Write top JSON
    top_json_path = Path(args.top_json)
    top_json_path.write_text(json.dumps({
        "days": args.days, "folds": args.folds, "haircut": apply_haircut_flag,
        "n_trades": len(trades),
        "tops": top_results,
    }, indent=2, default=str))
    print(f"[OK] Top-5 JSON written to {top_json_path}")


if __name__ == "__main__":
    main()
