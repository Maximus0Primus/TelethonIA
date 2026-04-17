"""v135 full strategy sweep on post-v132 price_ticks.

Ephemeral read-only analysis. For every strategy in strategies.STRATEGIES,
replay each call in the post-v132 token universe across:
  * price_source: jupiter, dexscreener, both
  * poll_sec:    60, 120
  * smoothing:   raw, median_3, median_5, winsor_p95, dual_confirm,
                 ema_fast (w=2), ema_slow (w=8), hysteresis

Writes scraper/_sweep_v135_full.csv and prints the top-30 composite ranking,
per-family winners, divergence flags, and a recommended 2-strategy portfolio.

USAGE:
    python scripts/_sweep_v135_full.py
"""

from __future__ import annotations

import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import combinations

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

# ---------------------------------------------------------------------------
# Paths + Supabase
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SCRAPER = os.path.join(ROOT, "scraper")
sys.path.insert(0, SCRAPER)

load_dotenv(os.path.join(SCRAPER, ".env"))
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------------------------------------------------------
# Strategy universe
# ---------------------------------------------------------------------------
from strategies import (  # noqa: E402
    STRATEGIES, STRATEGY_FILTERS, _get_decay_end, _get_trail_config,
    _BE_RE,
)

# Decision: only the first "main"-equivalent tranche for each strategy.
# Multi-tranche strategies (SCALE_OUT, MOONBAG, WIDE_RUNNER, SPLIT_*) are
# approximated with their first tranche's exit logic on the full position.
# Flagged as "approx_multi" in the output.
MULTI_TRANCHE = {n for n, trs in STRATEGIES.items() if len(trs) > 1}

POST_V132 = "2026-04-13T20:00:00Z"
CSV_OUT = os.path.join(SCRAPER, "_sweep_v135_full.csv")

# Compute budget — configs per strategy
PRICE_SOURCES = ["jupiter", "dexscreener", "both"]
POLL_SECS = [60, 120]
SMOOTH_MODES = [
    "raw", "median_3", "median_5", "winsor_p95",
    "dual_confirm", "ema_fast", "ema_slow", "hysteresis",
]

MIN_N = 30  # composite rank threshold
VIRTUAL_POS = 10.0  # matches paper_trader shadow sim default

# ---------------------------------------------------------------------------
# Data fetch
# ---------------------------------------------------------------------------


def _fetch_paper_trades_universe() -> pd.DataFrame:
    """RT paper_trades post-v132 → dedup per token_address keeping earliest entry."""
    rows: list[dict] = []
    page = 1000
    off = 0
    cols = ("token_address,created_at,entry_price,dex_spot_price_at_entry,"
            "rt_liquidity_usd,symbol,source")
    while True:
        resp = (
            sb.table("paper_trades")
              .select(cols)
              .eq("source", "rt")
              .gte("created_at", POST_V132)
              .order("created_at")
              .range(off, off + page - 1)
              .execute()
        )
        b = resp.data or []
        rows.extend(b)
        if len(b) < page:
            break
        off += page
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")
    df = df.dropna(subset=["entry_price", "token_address", "created_at"])
    df["created_at_ts"] = pd.to_datetime(df["created_at"], utc=True)
    df = (df.sort_values("created_at_ts")
            .drop_duplicates(subset=["token_address"], keep="first")
            .reset_index(drop=True))
    return df


def _fetch_ticks_for_token(addr: str, t_start: str, t_end: str) -> list[dict]:
    """Fetch price_ticks for addr in [t_start, t_end]."""
    rows: list[dict] = []
    page = 1000
    off = 0
    while True:
        resp = (
            sb.table("price_ticks")
              .select("price_usd,fetched_at,source")
              .eq("token_address", addr)
              .gte("fetched_at", t_start)
              .lte("fetched_at", t_end)
              .order("fetched_at")
              .range(off, off + page - 1)
              .execute()
        )
        b = resp.data or []
        rows.extend(b)
        if len(b) < page:
            break
        off += page
    return rows


# ---------------------------------------------------------------------------
# Tick preparation — per (token, source)
# ---------------------------------------------------------------------------
def _build_streams(raw_ticks: list[dict]) -> dict[str, list[tuple[datetime, float]]]:
    """Split raw ticks into {jupiter, dexscreener, both} streams.
    Each stream is a list of (datetime, price) sorted by time."""
    jup = []
    dex = []
    for r in raw_ticks:
        try:
            ts = datetime.fromisoformat(r["fetched_at"].replace("Z", "+00:00"))
            p = float(r["price_usd"])
        except Exception:
            continue
        if p <= 0:
            continue
        if r["source"] == "jupiter":
            jup.append((ts, p))
        elif r["source"] in ("fast", "full", "live", "check"):
            dex.append((ts, p))
    jup.sort(key=lambda x: x[0])
    dex.sort(key=lambda x: x[0])
    # both = jupiter preferred at each timestamp
    both = sorted(dex + jup, key=lambda x: x[0])
    return {"jupiter": jup, "dexscreener": dex, "both": both}


def _subsample(stream: list[tuple[datetime, float]], poll_sec: int
               ) -> list[tuple[datetime, float]]:
    if poll_sec <= 0 or not stream:
        return stream
    out = [stream[0]]
    last_t = stream[0][0]
    for ts, p in stream[1:]:
        if (ts - last_t).total_seconds() >= poll_sec:
            out.append((ts, p))
            last_t = ts
    return out


# ---------------------------------------------------------------------------
# Smoothing — stateless helpers applied to a stream
# ---------------------------------------------------------------------------
def _smooth_stream(stream: list[tuple[datetime, float]], mode: str,
                   sl_price: float, tp_price: float | None
                   ) -> list[tuple[datetime, float]]:
    """Return (ts, decision_price) stream based on smoothing mode.
    exit_ref is always the raw price at the same index; we return only the
    decision price and the caller re-pairs with raw later."""
    n = len(stream)
    if n == 0 or mode == "raw":
        return stream

    if mode == "ema_fast" or mode == "ema_slow":
        window = 2 if mode == "ema_fast" else 8
        alpha = 2 / (window + 1)
        out = []
        ema = None
        for ts, p in stream:
            ema = p if ema is None else alpha * p + (1 - alpha) * ema
            out.append((ts, ema))
        return out

    if mode == "median_3" or mode == "median_5":
        w = 3 if mode == "median_3" else 5
        out = []
        hist: list[float] = []
        for ts, p in stream:
            hist.append(p)
            if len(hist) > w:
                hist.pop(0)
            if len(hist) < w:
                out.append((ts, p))  # warm-up pass-through
            else:
                out.append((ts, sorted(hist)[w // 2]))
        return out

    if mode == "winsor_p95":
        out = []
        prev = stream[0][1]
        for ts, p in stream:
            delta = p - prev
            cap = prev * 0.18
            if delta > cap:
                q = prev + cap
            elif delta < -cap:
                q = prev - cap
            else:
                q = p
            out.append((ts, q))
            prev = q
        return out

    if mode == "dual_confirm":
        out = []
        prev = stream[0][1]
        for ts, p in stream:
            decision = p
            if sl_price and p <= sl_price and prev > sl_price:
                decision = prev  # suppress 1-tick breach
            elif tp_price and p >= tp_price and prev < tp_price:
                decision = prev
            out.append((ts, decision))
            prev = p
        return out

    if mode == "hysteresis":
        out = []
        armed_sl = True
        armed_tp = True
        for ts, p in stream:
            # re-arm when price retraces 2% past trigger
            if not armed_sl and sl_price and p >= sl_price * 1.02:
                armed_sl = True
            elif armed_sl and sl_price and p <= sl_price:
                armed_sl = False
            if not armed_tp and tp_price and p <= tp_price * 0.98:
                armed_tp = True
            elif armed_tp and tp_price and p >= tp_price:
                armed_tp = False
            # if disarmed, serve price that doesn't retrigger
            if not armed_sl and sl_price and p <= sl_price:
                out.append((ts, sl_price * 1.001))
            elif not armed_tp and tp_price and p >= tp_price:
                out.append((ts, tp_price * 0.999))
            else:
                out.append((ts, p))
        return out

    return stream  # unknown → raw


# ---------------------------------------------------------------------------
# Exit evaluator — reimplemented to be stateless & fast (avoids paper_trader
# globals). Mirrors _evaluate_trade_exit semantics for the strategy types
# we care about (FIXED/TP-SL, DECAY, BE, TRAIL, DTRAIL). DIP_BUY watchlist
# dynamics (dip+bounce P2 entry) are not replayed — we treat DIP strategies
# as their P1 tranche only (trail+SL) which is what the shadow pipeline does.
# ---------------------------------------------------------------------------
SELL_SLIP = 1 - 10 / 10_000  # 10bps Ultra RFQ
TRAIL_CACHE: dict[str, tuple[float | None, float | None]] = {}


def _cached_trail_config(strat: str) -> tuple[float | None, float | None]:
    c = TRAIL_CACHE.get(strat)
    if c is not None:
        return c
    c = _get_trail_config({"strategy": strat, "tranche_label": "main"})
    TRAIL_CACHE[strat] = c
    return c


def _evaluate_call(strat: str, tp_mult: float | None, sl_mult: float,
                   horizon_min: int, entry_price: float, entry_time: datetime,
                   dec_stream: list[tuple[datetime, float]],
                   raw_stream: list[tuple[datetime, float]]) -> dict:
    """Replay one (strategy × token call) through decision stream.
    raw_stream used for exit_price booking (Jupiter exec ref).
    Returns {status, pnl_pct, pnl_usd, exit_minutes}."""
    sl_price = entry_price * sl_mult
    tp_price = entry_price * tp_mult if tp_mult else None
    market_ref = entry_price  # approximate (no dex_spot at entry on shadows)

    trail_pct, activation_pct = _cached_trail_config(strat)
    decay_end = _get_decay_end(strat)
    be_match = _BE_RE.match(strat)

    horizon_delta = timedelta(minutes=horizon_min)
    end_time = entry_time + horizon_delta
    high_seen = entry_price

    # Index into raw stream for lookup of exec price at decision_ts
    raw_idx = 0
    raw_len = len(raw_stream)

    for ts, dec_p in dec_stream:
        if ts < entry_time:
            continue
        if ts > end_time:
            break
        # advance raw_idx to nearest >= ts
        while raw_idx < raw_len and raw_stream[raw_idx][0] < ts:
            raw_idx += 1
        if raw_idx < raw_len:
            exec_p = raw_stream[raw_idx][1]
        elif raw_len > 0:
            exec_p = raw_stream[-1][1]
        else:
            exec_p = dec_p

        if dec_p > high_seen:
            high_seen = dec_p

        # BE stop override
        effective_sl = sl_price
        if be_match and high_seen > 0:
            be_act = int(be_match.group(1)) / 100
            if high_seen >= market_ref * (1 + be_act):
                effective_sl = entry_price

        # SL check on exec_p (v133)
        if exec_p <= effective_sl:
            exit_p = effective_sl * SELL_SLIP
            pnl_pct = exit_p / entry_price - 1
            return {"status": "sl_hit", "pnl_pct": pnl_pct,
                    "exit_minutes": int((ts - entry_time).total_seconds() / 60)}

        # TP check on decision
        if tp_price is not None and dec_p >= tp_price:
            exit_p = tp_price * SELL_SLIP
            pnl_pct = exit_p / entry_price - 1
            return {"status": "tp_hit", "pnl_pct": pnl_pct,
                    "exit_minutes": int((ts - entry_time).total_seconds() / 60)}

        # Decay TP
        if tp_price is not None and decay_end is not None:
            elapsed = (ts - entry_time).total_seconds() / 60
            tp_mult_cur = tp_price / entry_price if entry_price > 0 else 1
            act_frac = 0.5
            if elapsed > horizon_min * act_frac:
                prog = min((elapsed - horizon_min * act_frac) /
                           (horizon_min * (1 - act_frac)), 1.0)
                decayed_mult = tp_mult_cur - (tp_mult_cur - decay_end) * prog
                decayed_price = entry_price * decayed_mult
                if dec_p >= decayed_price:
                    exit_p = decayed_price * SELL_SLIP
                    pnl_pct = exit_p / entry_price - 1
                    return {"status": "tp_hit", "pnl_pct": pnl_pct,
                            "exit_minutes": int(elapsed)}

        # Trail
        if trail_pct is not None and high_seen > 0 and market_ref > 0:
            activation_price = market_ref * (1 + activation_pct)
            if high_seen >= activation_price:
                trigger = high_seen * (1 - trail_pct)
                if dec_p <= trigger and trigger > entry_price:
                    exit_p = exec_p * SELL_SLIP
                    pnl_pct = exit_p / entry_price - 1
                    return {"status": "trail_stop", "pnl_pct": pnl_pct,
                            "exit_minutes": int((ts - entry_time).total_seconds() / 60)}

    # Timeout — use last raw price in window (or dec if raw empty)
    last_p = None
    for ts, p in reversed(raw_stream):
        if ts <= end_time:
            last_p = p
            break
    if last_p is None:
        for ts, p in reversed(dec_stream):
            if ts <= end_time:
                last_p = p
                break
    if last_p is None:
        return {"status": "no_data", "pnl_pct": 0.0, "exit_minutes": 0}
    exit_p = last_p * SELL_SLIP
    pnl_pct = exit_p / entry_price - 1
    return {"status": "timeout", "pnl_pct": pnl_pct, "exit_minutes": horizon_min}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _kelly_f(p: float, w: float, l: float) -> float:
    if l <= 0 or w <= 0:
        return 0.0
    b = w / l
    return max(0.0, (p * (b + 1) - 1) / b)


def _compute_metrics(pnl: np.ndarray) -> dict:
    n = len(pnl)
    if n == 0:
        return {}
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    wr = len(wins) / n
    avg = float(pnl.mean())
    med = float(np.median(pnl))
    std = float(pnl.std(ddof=1)) if n > 1 else 0.0
    sharpe = (avg / std) if std > 0 else 0.0
    avg_w = float(wins.mean()) if len(wins) else 0.0
    avg_l = float(-losses.mean()) if len(losses) else 0.0
    kelly = _kelly_f(wr, avg_w, avg_l) if avg_l > 0 else 0.0
    # max drawdown of equity curve (cumulative compounded on pnl_pct)
    equity = np.cumprod(1 + pnl)
    peaks = np.maximum.accumulate(equity)
    dd = (equity - peaks) / peaks
    max_dd = float(dd.min()) if dd.size else 0.0
    return {
        "n": n, "wr_pct": round(wr * 100, 2),
        "avg_pnl_pct": round(avg * 100, 3),
        "median_pnl_pct": round(med * 100, 3),
        "stdev_pnl_pct": round(std * 100, 3),
        "sharpe": round(sharpe, 4),
        "kelly_pct": round(kelly * 100, 3),
        "max_dd_pct": round(max_dd * 100, 2),
        "total_pnl_usd": round(float((pnl * VIRTUAL_POS).sum()), 2),
    }


# ---------------------------------------------------------------------------
# Family classifier
# ---------------------------------------------------------------------------
FAM_PATTERNS = [
    ("FAST",    re.compile(r"^FAST_TP")),
    ("FAST45",  re.compile(r"^FAST45_TP")),
    ("FAST60",  re.compile(r"^FAST60_TP")),
    ("SLOW4H",  re.compile(r"^SLOW4H_TP")),
    ("SLOW6H",  re.compile(r"^SLOW6H_TP")),
    ("BE",      re.compile(r"^BE\d+_TP")),
    ("DTRAIL",  re.compile(r"^DTRAIL\d+_")),
    ("TRAIL",   re.compile(r"^TRAIL\d+_")),
    ("DIP",     re.compile(r"^DIP")),
    ("SPLIT",   re.compile(r"^SPLIT_")),
    ("DECAY",   re.compile(r"^DECAY_")),
    ("SCALP",   re.compile(r"^SCALP_")),
    ("TP",      re.compile(r"^TP\d+_")),
]


def _family(name: str) -> str:
    for fam, pat in FAM_PATTERNS:
        if pat.match(name):
            return fam
    return "OTHER"


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    print("=" * 90)
    print("v135 FULL SWEEP — 224 strategies × 48 configs × post-v132 ticks")
    print("=" * 90)
    print("Loading paper_trades universe...")
    uni = _fetch_paper_trades_universe()
    print(f"  {len(uni)} unique tokens (post-v132 RT calls dedup'd)")

    # Compute horizon we need per token → use max strategy horizon (360 min for SLOW6H)
    MAX_HORIZON_MIN = max(
        max((tr.get("horizon_min", 120) or 120) for tr in trs)
        for trs in STRATEGIES.values()
    )
    print(f"  max horizon across strategies: {MAX_HORIZON_MIN} min")

    # Fetch ticks per token
    print("Fetching ticks per token (this dominates wall time)...")
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
        except Exception as e:
            print(f"  [warn] tick fetch failed for {addr[:10]}: {e}")
            continue
        if not raw_ticks:
            continue
        streams = _build_streams(raw_ticks)
        # Keep only streams with ≥3 ticks in coverage window
        streams = {k: v for k, v in streams.items() if len(v) >= 3}
        if not streams:
            continue
        token_streams[addr] = streams
        entry_info[addr] = (entry_price, entry_ts.to_pydatetime())
        if (i + 1) % 10 == 0:
            print(f"  fetched {i + 1}/{len(uni)} tokens, {len(token_streams)} usable")
    print(f"  kept {len(token_streams)} tokens with usable ticks")

    # ---- Sweep ----
    results: list[dict] = []
    total_configs = len(STRATEGIES) * len(PRICE_SOURCES) * len(POLL_SECS) * len(SMOOTH_MODES)
    print(f"\nRunning sweep: {len(STRATEGIES)} strats × "
          f"{len(PRICE_SOURCES)}src × {len(POLL_SECS)}poll × "
          f"{len(SMOOTH_MODES)}mode = {total_configs} configs")

    # Pre-subsample streams per (source, poll). Cache.
    subsample_cache: dict[tuple[str, str, int], list[tuple[datetime, float]]] = {}
    for addr, streams in token_streams.items():
        for src in PRICE_SOURCES:
            stream = streams.get(src, [])
            for poll in POLL_SECS:
                subsample_cache[(addr, src, poll)] = _subsample(stream, poll)

    done = 0
    for strat, tranches in STRATEGIES.items():
        tr0 = tranches[0]  # first tranche — approx for multi-tranche
        tp_mult = tr0.get("tp_mult")
        sl_mult = tr0.get("sl_mult", 0.50)
        horizon_min = tr0.get("horizon_min", 120) or 120

        # Entry-gated strategies: note but don't skip. STRATEGY_FILTERS is
        # evaluated at entry time on token features, not on ticks — most
        # post-v132 shadow trades IGNORE filters (shadow = bypass). We
        # simulate the exit regardless; the filter note is reported at end.
        for src in PRICE_SOURCES:
            for poll in POLL_SECS:
                for mode in SMOOTH_MODES:
                    pnls: list[float] = []
                    statuses: list[str] = []
                    exit_mins: list[int] = []
                    for addr, (entry_price, entry_ts) in entry_info.items():
                        dec_stream_raw = subsample_cache.get((addr, src, poll), [])
                        if not dec_stream_raw:
                            continue
                        # Raw exec stream = jupiter preferred, fallback to "both"
                        exec_stream = token_streams[addr].get("jupiter") or \
                            token_streams[addr].get("both") or dec_stream_raw
                        if not exec_stream:
                            continue
                        # Smooth decision stream for this strategy's SL/TP
                        sl_p = entry_price * sl_mult
                        tp_p = entry_price * tp_mult if tp_mult else None
                        dec_stream = _smooth_stream(
                            dec_stream_raw, mode, sl_p, tp_p
                        )
                        r = _evaluate_call(
                            strat, tp_mult, sl_mult, horizon_min,
                            entry_price, entry_ts, dec_stream, exec_stream,
                        )
                        if r["status"] == "no_data":
                            continue
                        pnls.append(r["pnl_pct"])
                        statuses.append(r["status"])
                        exit_mins.append(r["exit_minutes"])
                    pnl_arr = np.array(pnls)
                    m = _compute_metrics(pnl_arr)
                    if not m:
                        continue
                    results.append({
                        "strategy": strat,
                        "family": _family(strat),
                        "source": src,
                        "poll_sec": poll,
                        "mode": mode,
                        "horizon_min": horizon_min,
                        "multi_tranche": strat in MULTI_TRANCHE,
                        "has_filter": strat in STRATEGY_FILTERS,
                        **m,
                    })
                    done += 1
        if (len(results) % (48 * 10)) < 48 and done > 0:
            elapsed = time.time() - t0
            pct = 100.0 * done / total_configs
            print(f"  progress: {done}/{total_configs} ({pct:.1f}%) "
                  f"in {elapsed:.0f}s")

    print(f"\nSweep complete in {time.time() - t0:.0f}s — {len(results)} rows")

    df = pd.DataFrame(results)
    col_order = ["strategy", "family", "source", "poll_sec", "mode",
                 "n", "wr_pct", "avg_pnl_pct", "median_pnl_pct",
                 "stdev_pnl_pct", "sharpe", "kelly_pct", "max_dd_pct",
                 "horizon_min", "total_pnl_usd", "multi_tranche", "has_filter"]
    df = df[col_order]
    df.to_csv(CSV_OUT, index=False)
    print(f"CSV written: {CSV_OUT}  ({len(df)} rows)")

    # ---- Composite rank (MIN_N filter) ----
    print("\n" + "=" * 90)
    print(f"TOP 30 BY COMPOSITE RANK (n >= {MIN_N}, rank=kelly+median+sharpe)")
    print("=" * 90)
    ranked = df[df["n"] >= MIN_N].copy()
    if not ranked.empty:
        ranked["rk_kelly"] = ranked["kelly_pct"].rank(ascending=False, method="min")
        ranked["rk_median"] = ranked["median_pnl_pct"].rank(ascending=False, method="min")
        ranked["rk_sharpe"] = ranked["sharpe"].rank(ascending=False, method="min")
        ranked["rk_sum"] = ranked[["rk_kelly", "rk_median", "rk_sharpe"]].sum(axis=1)
        ranked = ranked.sort_values("rk_sum")
        top = ranked.head(30)[[
            "strategy", "family", "source", "poll_sec", "mode",
            "n", "wr_pct", "avg_pnl_pct", "median_pnl_pct", "sharpe",
            "kelly_pct", "max_dd_pct", "rk_sum",
        ]]
        print(top.to_string(index=False))
    else:
        print("(no configs with n>=30 — relax MIN_N)")

    # ---- Per-strategy winner table ----
    print("\n" + "=" * 90)
    print("PER-STRATEGY WINNERS (best config per strategy, n>=30 preferred)")
    print("=" * 90)
    winners = []
    for strat, g in df.groupby("strategy"):
        qual = g[g["n"] >= MIN_N]
        pool = qual if not qual.empty else g
        # pick by avg_pnl_pct as primary tiebreak
        best = pool.sort_values(
            ["kelly_pct", "avg_pnl_pct", "sharpe"], ascending=False
        ).iloc[0]
        raw = g[(g["source"] == "jupiter") & (g["poll_sec"] == 60) &
                (g["mode"] == "raw")]
        raw_avg = float(raw["avg_pnl_pct"].iloc[0]) if not raw.empty else None
        config_sensitive = (
            raw_avg is not None and
            abs(float(best["avg_pnl_pct"]) - raw_avg) > 5.0
        )
        exotic = best["mode"] in ("ema_slow", "winsor_p95", "hysteresis")
        winners.append({
            "strategy": strat,
            "family": _family(strat),
            "best_src": best["source"],
            "best_poll": int(best["poll_sec"]),
            "best_mode": best["mode"],
            "best_n": int(best["n"]),
            "best_avg_pct": float(best["avg_pnl_pct"]),
            "best_kelly_pct": float(best["kelly_pct"]),
            "raw_avg_pct": raw_avg,
            "config_sensitive": config_sensitive,
            "exotic_only": exotic and (raw_avg is not None and
                                        raw_avg < 0 and best["avg_pnl_pct"] > 0),
        })
    win_df = pd.DataFrame(winners).sort_values("best_kelly_pct", ascending=False)
    # print family winners (top per family)
    fam_win = (win_df.sort_values("best_kelly_pct", ascending=False)
                     .groupby("family")
                     .head(1)
                     .sort_values("best_kelly_pct", ascending=False))
    print("Best strategy per family:")
    print(fam_win[["family", "strategy", "best_src", "best_poll", "best_mode",
                   "best_n", "best_avg_pct", "best_kelly_pct"]].to_string(index=False))
    print()
    print("Top-20 strategies overall (by best-config Kelly):")
    print(win_df.head(20)[["strategy", "family", "best_src", "best_poll",
                           "best_mode", "best_n", "best_avg_pct",
                           "best_kelly_pct"]].to_string(index=False))

    # ---- Divergence flags ----
    print("\n" + "=" * 90)
    print("CONFIG-SENSITIVE STRATEGIES (>5pp delta vs raw/jupiter/60s)")
    print("=" * 90)
    sens = win_df[win_df["config_sensitive"] == True].head(20)  # noqa: E712
    if not sens.empty:
        print(sens[["strategy", "best_src", "best_poll", "best_mode",
                    "best_avg_pct", "raw_avg_pct"]].to_string(index=False))
    else:
        print("  (none)")

    print("\n" + "=" * 90)
    print("EXOTIC-SMOOTHING-ONLY STRATEGIES (unprofitable raw, profitable smoothed)")
    print("=" * 90)
    exo = win_df[win_df["exotic_only"] == True].head(20)  # noqa: E712
    if not exo.empty:
        print(exo[["strategy", "best_mode", "best_avg_pct",
                   "raw_avg_pct"]].to_string(index=False))
    else:
        print("  (none)")

    # ---- Multi-tranche approx note ----
    multi_in_top = win_df[win_df["strategy"].isin(MULTI_TRANCHE)].head(10)
    if not multi_in_top.empty:
        print("\n" + "=" * 90)
        print("MULTI-TRANCHE APPROX NOTE (only first tranche replayed)")
        print("=" * 90)
        print(multi_in_top[["strategy", "family", "best_avg_pct",
                            "best_kelly_pct"]].to_string(index=False))

    # ---- Portfolio recommendation ----
    print("\n" + "=" * 90)
    print("RECOMMENDED 2-STRATEGY PORTFOLIO (lowest correlation, highest combined Kelly)")
    print("=" * 90)
    cand = (ranked.sort_values("rk_sum")
                   .drop_duplicates(subset=["strategy"], keep="first")
                   .head(15))
    if cand.empty or len(cand) < 2:
        print("  (not enough qualifying configs for pair analysis)")
        return
    # Build per-token pnl series per (strategy_config_key)
    # Re-run top 15 to capture per-token pnl for correlation
    print("Computing per-token pnl for top-15 for correlation...")
    # We still have the necessary context — per-token pnl recomputation:
    top15_keys = list(cand[["strategy", "source", "poll_sec", "mode"]]
                      .itertuples(index=False, name=None))
    per_token_pnl: dict[tuple, dict[str, float]] = {}
    for key in top15_keys:
        strat, src, poll, mode = key
        tr0 = STRATEGIES[strat][0]
        tp_mult = tr0.get("tp_mult")
        sl_mult = tr0.get("sl_mult", 0.50)
        horizon_min = tr0.get("horizon_min", 120) or 120
        pnl_by_tok: dict[str, float] = {}
        for addr, (entry_price, entry_ts) in entry_info.items():
            dec_stream_raw = subsample_cache.get((addr, src, poll), [])
            if not dec_stream_raw:
                continue
            exec_stream = token_streams[addr].get("jupiter") or \
                token_streams[addr].get("both") or dec_stream_raw
            if not exec_stream:
                continue
            sl_p = entry_price * sl_mult
            tp_p = entry_price * tp_mult if tp_mult else None
            dec_stream = _smooth_stream(dec_stream_raw, mode, sl_p, tp_p)
            r = _evaluate_call(strat, tp_mult, sl_mult, horizon_min,
                               entry_price, entry_ts, dec_stream, exec_stream)
            if r["status"] != "no_data":
                pnl_by_tok[addr] = r["pnl_pct"]
        per_token_pnl[key] = pnl_by_tok

    pair_rows = []
    for a, b in combinations(top15_keys, 2):
        pa = per_token_pnl[a]
        pb = per_token_pnl[b]
        common = list(set(pa.keys()) & set(pb.keys()))
        if len(common) < 20:
            continue
        sa = np.array([pa[t] for t in common])
        sb2 = np.array([pb[t] for t in common])
        corr = float(np.corrcoef(sa, sb2)[0, 1])
        port = 0.5 * sa + 0.5 * sb2
        n = len(port)
        wins = port[port > 0]
        losses = port[port <= 0]
        wr = len(wins) / n
        avg_w = float(wins.mean()) if len(wins) else 0.0
        avg_l = float(-losses.mean()) if len(losses) else 0.0
        kelly = _kelly_f(wr, avg_w, avg_l)
        pair_rows.append({
            "a": f"{a[0]}[{a[1][0]}/{a[2]}/{a[3]}]",
            "b": f"{b[0]}[{b[1][0]}/{b[2]}/{b[3]}]",
            "corr": round(corr, 3),
            "n_joint": n,
            "port_avg_pct": round(float(port.mean()) * 100, 3),
            "port_kelly_pct": round(kelly * 100, 3),
        })
    if pair_rows:
        pair_df = pd.DataFrame(pair_rows)
        pair_df["score"] = pair_df["port_kelly_pct"] - pair_df["corr"] * 50
        best_pair = pair_df.sort_values("score", ascending=False).head(5)
        print("Top 5 pairs (score = port_kelly_pct - 50 * corr):")
        print(best_pair.to_string(index=False))
    else:
        print("  (not enough joint tokens to correlate)")

    # ---- Summary footer ----
    print("\n" + "=" * 90)
    print("SWEEP SUMMARY")
    print("=" * 90)
    print(f"Total (strategy, source, poll, mode) rows: {len(df)}")
    print(f"Rows with n>=30: {(df['n'] >= MIN_N).sum()}")
    print(f"Strategies w/ entry filters (exit-sim only): {list(STRATEGY_FILTERS.keys())}")
    print(f"Multi-tranche strategies (approx'd): "
          f"{sorted(MULTI_TRANCHE)}")
    print(f"Tokens with ticks: {len(entry_info)}")
    print(f"Total wall time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
