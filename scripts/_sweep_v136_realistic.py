"""v136 realistic polling sweep — replaces v135's "next-tick-at/after-gap"
with "deterministic 30s grid, look-back to latest tick at/before poll-time".

This mimics paper_trader's actual cadence:
  - unified_check_loop runs every 30s
  - _should_poll_trade(polling_sec) filter gates per-trade evaluation
  - At each poll, _jupiter_prices_cache holds the latest fetch (look-back)

KNOWN limitation: price_ticks for paper-only tokens are throttled to 60s
(_log_price_ticks line 369). Real paper saw fresher prices than the sim
can reconstruct. Sim will still under-estimate peak detection for fast
movers; for tight trails this residual bias is ~3-5%.

Same outputs as v135: ranking + portfolio analysis.
"""
from __future__ import annotations

import os
import re
import sys
import time
from datetime import datetime, timedelta
from itertools import combinations

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SCRAPER = os.path.join(ROOT, "scraper")
sys.path.insert(0, SCRAPER)
sys.path.insert(0, HERE)

load_dotenv(os.path.join(SCRAPER, ".env"))
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

from strategies import (  # noqa: E402
    STRATEGIES, STRATEGY_FILTERS, _get_decay_end, _get_trail_config, _BE_RE,
)
from _sweep_v135_full import (  # noqa: E402
    _build_streams, _fetch_paper_trades_universe, _fetch_ticks_for_token,
    _family, _kelly_f, _compute_metrics, MIN_N, VIRTUAL_POS, POST_V132,
    PRICE_SOURCES, POLL_SECS, MULTI_TRANCHE,
)

CSV_OUT = os.path.join(SCRAPER, "_sweep_v136_realistic.csv")
LOOP_SEC = 30  # unified_check_loop interval

# Tighter mode list — dropped exotic modes (winsor/dual_confirm/hysteresis)
# that didn't outperform raw/ema_fast in v135 anyway. Saves 50% of compute.
SMOOTH_MODES = ["raw", "ema_fast", "median_5"]

# Dynamic slippage approximation: mirrors paper_trader._dynamic_sell_slip_factor
# heuristic. Values calibrated from v133-D live data.
SLIP_BPS = {
    "tp_hit": 10,
    "sl_hit": 50,
    "trail_stop": 30,
    "trail_crash": 250,
    "timeout": 30,
}


def _slip_factor(exit_type: str) -> float:
    return 1 - SLIP_BPS.get(exit_type, 30) / 10_000


def _latest_tick_at_or_before(stream: list[tuple[datetime, float]],
                              t: datetime) -> tuple[float | None, datetime | None]:
    """Return (price, tick_time) of latest tick whose ts <= t. None if none."""
    last = (None, None)
    for ts, p in stream:
        if ts <= t:
            last = (p, ts)
        else:
            break
    return last


def _evaluate_realistic(strat: str, tp_mult: float | None, sl_mult: float,
                        horizon_min: int, entry_price: float,
                        entry_time: datetime,
                        jp_stream: list[tuple[datetime, float]],
                        ds_stream: list[tuple[datetime, float]],
                        source: str, poll_sec: int, mode: str) -> dict:
    """Realistic polling sim with deterministic 30s grid + cache look-back."""
    market_ref = entry_price
    sl_price = entry_price * sl_mult
    tp_price = entry_price * tp_mult if tp_mult else None
    trail_pct, activation_pct = _get_trail_config({"strategy": strat,
                                                     "tranche_label": "main"})
    decay_end = _get_decay_end(strat)
    be_match = _BE_RE.match(strat)
    horizon_sec = horizon_min * 60

    high_seen = entry_price

    # Poll schedule: 30s grid from entry, _should_poll(poll_sec) filter
    poll_offsets = []
    last_check = -10**9
    t = LOOP_SEC
    while t <= horizon_sec:
        if (t - last_check) >= poll_sec:
            poll_offsets.append(t)
            last_check = t
        t += LOOP_SEC

    ema_state = None
    median_hist: list[float] = []
    last_exec_p = entry_price

    for offset_sec in poll_offsets:
        poll_time = entry_time + timedelta(seconds=offset_sec)

        jp, _ = _latest_tick_at_or_before(jp_stream, poll_time)
        ds, _ = _latest_tick_at_or_before(ds_stream, poll_time)

        if source == "jupiter":
            dec_p_raw = jp
            exec_p = jp
        elif source == "dexscreener":
            dec_p_raw = ds
            exec_p = jp if jp is not None else ds
        else:  # both
            dec_p_raw = ds if ds is not None else jp
            exec_p = jp if jp is not None else ds

        if dec_p_raw is None or exec_p is None:
            continue

        # Smoothing
        if mode == "ema_fast":
            alpha = 2 / 3
            ema_state = dec_p_raw if ema_state is None else \
                alpha * dec_p_raw + (1 - alpha) * ema_state
            dec_p = ema_state
        elif mode == "median_5":
            median_hist.append(dec_p_raw)
            if len(median_hist) > 5:
                median_hist.pop(0)
            dec_p = sorted(median_hist)[len(median_hist) // 2] \
                if len(median_hist) >= 5 else dec_p_raw
        else:  # raw
            dec_p = dec_p_raw

        last_exec_p = exec_p
        if dec_p > high_seen:
            high_seen = dec_p

        # BE override
        effective_sl = sl_price
        if be_match and high_seen > 0:
            be_act = int(be_match.group(1)) / 100
            if high_seen >= market_ref * (1 + be_act):
                effective_sl = entry_price

        # SL on raw exec (paper_trader.py:1228)
        if exec_p <= effective_sl:
            exit_p = effective_sl * _slip_factor("sl_hit")
            return {"status": "sl_hit", "pnl_pct": exit_p / entry_price - 1,
                    "exit_minutes": int(offset_sec / 60)}

        # TP
        if tp_price is not None and dec_p >= tp_price:
            exit_p = tp_price * _slip_factor("tp_hit")
            return {"status": "tp_hit", "pnl_pct": exit_p / entry_price - 1,
                    "exit_minutes": int(offset_sec / 60)}

        # Decay TP
        if tp_price is not None and decay_end is not None:
            elapsed = offset_sec / 60
            tp_mult_cur = tp_price / entry_price if entry_price > 0 else 1
            if elapsed > horizon_min * 0.5:
                prog = min((elapsed - horizon_min * 0.5) / (horizon_min * 0.5), 1.0)
                decayed_mult = tp_mult_cur - (tp_mult_cur - decay_end) * prog
                decayed_price = entry_price * decayed_mult
                if dec_p >= decayed_price:
                    exit_p = decayed_price * _slip_factor("tp_hit")
                    return {"status": "tp_hit", "pnl_pct": exit_p / entry_price - 1,
                            "exit_minutes": int(elapsed)}

        # Trail (uses dynamic slip)
        if trail_pct is not None and high_seen > 0 and market_ref > 0:
            activation_price = market_ref * (1 + activation_pct)
            if high_seen >= activation_price:
                trigger = high_seen * (1 - trail_pct)
                if dec_p <= trigger and trigger > entry_price:
                    crash_ratio = exec_p / trigger if trigger > 0 else 1.0
                    exit_type = "trail_crash" if crash_ratio < 0.70 else "trail_stop"
                    exit_p = exec_p * _slip_factor(exit_type)
                    return {"status": exit_type,
                            "pnl_pct": exit_p / entry_price - 1,
                            "exit_minutes": int(offset_sec / 60)}

    # Timeout
    if last_exec_p is None:
        return {"status": "no_data", "pnl_pct": 0.0, "exit_minutes": 0}
    exit_p = last_exec_p * _slip_factor("timeout")
    return {"status": "timeout", "pnl_pct": exit_p / entry_price - 1,
            "exit_minutes": horizon_min}


def main() -> None:
    t0 = time.time()
    print("=" * 90)
    print("v136 REALISTIC SWEEP — deterministic 30s grid + cache look-back + dyn slip")
    print("=" * 90)
    uni = _fetch_paper_trades_universe()
    print(f"  {len(uni)} unique tokens (post-v132 RT calls dedup'd)")

    MAX_HORIZON_MIN = max(
        max((tr.get("horizon_min", 120) or 120) for tr in trs)
        for trs in STRATEGIES.values()
    )
    print(f"  max horizon: {MAX_HORIZON_MIN} min")

    print("Fetching ticks per token...")
    token_streams: dict[str, dict[str, list[tuple[datetime, float]]]] = {}
    entry_info: dict[str, tuple[float, datetime]] = {}
    for i, row in uni.iterrows():
        addr = row["token_address"]
        entry_ts = row["created_at_ts"]
        entry_price = float(row["entry_price"])
        t_start = entry_ts.isoformat().replace("+00:00", "Z")
        t_end = (entry_ts + timedelta(minutes=MAX_HORIZON_MIN)
                 ).isoformat().replace("+00:00", "Z")
        try:
            raw_ticks = _fetch_ticks_for_token(addr, t_start, t_end)
        except Exception:
            continue
        if not raw_ticks:
            continue
        streams = _build_streams(raw_ticks)
        streams = {k: v for k, v in streams.items() if len(v) >= 3}
        if not streams:
            continue
        token_streams[addr] = streams
        entry_info[addr] = (entry_price, entry_ts.to_pydatetime())
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(uni)} -> {len(token_streams)} usable")
    print(f"  kept {len(token_streams)} tokens")

    results: list[dict] = []
    total = len(STRATEGIES) * len(PRICE_SOURCES) * len(POLL_SECS) * len(SMOOTH_MODES)
    print(f"\nSweep: {len(STRATEGIES)} strats × "
          f"{len(PRICE_SOURCES)}src × {len(POLL_SECS)}poll × "
          f"{len(SMOOTH_MODES)}mode = {total} configs")

    done = 0
    for strat, tranches in STRATEGIES.items():
        tr0 = tranches[0]
        tp_mult = tr0.get("tp_mult")
        sl_mult = tr0.get("sl_mult", 0.50)
        horizon_min = tr0.get("horizon_min", 120) or 120

        for src in PRICE_SOURCES:
            for poll in POLL_SECS:
                for mode in SMOOTH_MODES:
                    pnls = []
                    for addr, (entry_price, entry_ts) in entry_info.items():
                        streams = token_streams[addr]
                        jp_stream = streams.get("jupiter", [])
                        ds_stream = streams.get("dexscreener", [])
                        if not jp_stream and not ds_stream:
                            continue
                        r = _evaluate_realistic(strat, tp_mult, sl_mult,
                                                horizon_min, entry_price,
                                                entry_ts, jp_stream, ds_stream,
                                                src, poll, mode)
                        if r["status"] == "no_data":
                            continue
                        pnls.append(r["pnl_pct"])
                    pnl_arr = np.array(pnls)
                    m = _compute_metrics(pnl_arr)
                    if not m:
                        continue
                    results.append({
                        "strategy": strat, "family": _family(strat),
                        "source": src, "poll_sec": poll, "mode": mode,
                        "horizon_min": horizon_min,
                        "multi_tranche": strat in MULTI_TRANCHE,
                        "has_filter": strat in STRATEGY_FILTERS,
                        **m,
                    })
                    done += 1
        if done > 0 and len(results) % (len(SMOOTH_MODES) * 10) < len(SMOOTH_MODES):
            elapsed = time.time() - t0
            pct = 100.0 * done / total
            print(f"  progress: {done}/{total} ({pct:.1f}%) in {elapsed:.0f}s")

    print(f"\nSweep done in {time.time() - t0:.0f}s — {len(results)} rows")
    df = pd.DataFrame(results)
    cols = ["strategy", "family", "source", "poll_sec", "mode", "n", "wr_pct",
            "avg_pnl_pct", "median_pnl_pct", "stdev_pnl_pct", "sharpe",
            "kelly_pct", "max_dd_pct", "horizon_min", "total_pnl_usd",
            "multi_tranche", "has_filter"]
    df = df[cols]
    df.to_csv(CSV_OUT, index=False)
    print(f"CSV: {CSV_OUT}  ({len(df)} rows)")

    # ---- Composite rank ----
    print("\n" + "=" * 90)
    print(f"TOP 30 BY COMPOSITE RANK (n>={MIN_N})")
    print("=" * 90)
    ranked = df[df["n"] >= MIN_N].copy()
    if not ranked.empty:
        ranked["rk_kelly"] = ranked["kelly_pct"].rank(ascending=False, method="min")
        ranked["rk_median"] = ranked["median_pnl_pct"].rank(ascending=False, method="min")
        ranked["rk_sharpe"] = ranked["sharpe"].rank(ascending=False, method="min")
        ranked["rk_sum"] = ranked[["rk_kelly", "rk_median", "rk_sharpe"]].sum(axis=1)
        ranked = ranked.sort_values("rk_sum")
        top = ranked.head(30)[[
            "strategy", "family", "source", "poll_sec", "mode", "n", "wr_pct",
            "avg_pnl_pct", "median_pnl_pct", "sharpe", "kelly_pct", "max_dd_pct",
        ]]
        print(top.to_string(index=False))

    # ---- Per-family winners ----
    print("\n" + "=" * 90)
    print("BEST PER FAMILY")
    print("=" * 90)
    winners = []
    for strat, g in df.groupby("strategy"):
        qual = g[g["n"] >= MIN_N]
        pool = qual if not qual.empty else g
        best = pool.sort_values(["kelly_pct", "avg_pnl_pct", "sharpe"],
                                ascending=False).iloc[0]
        winners.append({
            "strategy": strat, "family": _family(strat),
            "best_src": best["source"], "best_poll": int(best["poll_sec"]),
            "best_mode": best["mode"], "best_n": int(best["n"]),
            "best_avg_pct": float(best["avg_pnl_pct"]),
            "best_kelly_pct": float(best["kelly_pct"]),
        })
    win_df = pd.DataFrame(winners).sort_values("best_kelly_pct", ascending=False)
    fam_win = (win_df.sort_values("best_kelly_pct", ascending=False)
                     .groupby("family").head(1)
                     .sort_values("best_kelly_pct", ascending=False))
    print(fam_win[["family", "strategy", "best_src", "best_poll", "best_mode",
                   "best_n", "best_avg_pct", "best_kelly_pct"]].to_string(index=False))

    print("\nTop 20 strategies overall:")
    print(win_df.head(20)[["strategy", "family", "best_src", "best_poll",
                           "best_mode", "best_n", "best_avg_pct",
                           "best_kelly_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
