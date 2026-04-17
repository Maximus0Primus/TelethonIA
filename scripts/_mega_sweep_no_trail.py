"""TRUE mega sweep — non-trail strategies × full orchestration grid.

Cross-product dimensions:
  Strategies: 113 non-trail (FIXED, BE, FAST, FAST45, FAST60, SLOW4H, SLOW6H,
              SCALP, DECAY, QUICK_SCALP, FRESH_MICRO)
  Source: jupiter, dexscreener (2)
  Smoothing: raw, ema_fast (w=2), ema_slow (w=8), median_3, median_5,
             winsor_p95, dual_confirm, hysteresis (8)
  Polling mode: fast (30s every tick), static_60, static_120, static_240,
                lazy (180s for first 5min then 600s) (5)

Total: 113 × 2 × 8 × 5 = 9040 configs × ~65 tokens = ~590K replays.

Outputs: scraper/_mega_sweep_no_trail.{csv,log}
"""
from __future__ import annotations
import os
import re
import sys
import time
import statistics
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SCRAPER = os.path.join(ROOT, "scraper")
sys.path.insert(0, SCRAPER)

load_dotenv(os.path.join(SCRAPER, ".env"))
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

import sim  # noqa
from strategies import STRATEGIES, _get_trail_config, _get_decay_end, _BE_RE  # noqa
from paper_trader import _evaluate_trade_exit, _last_eval_ts  # noqa

POST_V132 = "2026-04-13T20:00:00Z"
CSV_OUT = os.path.join(SCRAPER, "_mega_sweep_no_trail.csv")

ACTIVE_PAPER = ["BE25_TP80_SL30", "BE25_TP80_SL30_DS", "FAST_TP100_SL20",
                "TP50_SL15", "BE15_TP100_SL50"]
ACTIVE_LIVE = ["BE25_TP80_SL30", "BE15_TP100_SL50"]
PAPER_POS_USD = 50.0
LIVE_POS_USD = 1.70

TRAIL_PATTERNS = (re.compile(r"^DTRAIL\d+_"),
                  re.compile(r"^TRAIL\d+_"),
                  re.compile(r"^DIP\d+_"),
                  re.compile(r"^SPLIT_.*TRAIL"))


def _is_trail_strat(name: str) -> bool:
    return any(p.match(name) for p in TRAIL_PATTERNS)


SOURCES = ["jupiter", "dexscreener"]
SMOOTHINGS = ["raw", "ema_fast", "ema_slow", "median_3", "median_5",
              "winsor_p95", "dual_confirm", "hysteresis"]
POLLING_MODES = ["fast", "static_60", "static_120", "static_240", "lazy"]
NON_TRAIL_STRATS = sorted(s for s in STRATEGIES if not _is_trail_strat(s))

LOOP_SEC = 30
LAZY_FAST_SEC = 180
LAZY_FAST_WINDOW = 300
LAZY_SLOW_SEC = 600


# =============================================================================
# Polling schedule generator — covers the 5 modes
# =============================================================================
def _poll_offsets(polling_mode: str, horizon_sec: int) -> list[int]:
    """Return list of poll offsets (seconds since entry) for the given mode."""
    if polling_mode == "fast":
        return list(range(LOOP_SEC, horizon_sec + 1, LOOP_SEC))
    if polling_mode.startswith("static_"):
        poll_sec = int(polling_mode.split("_")[1])
        out = []
        last = -10**9
        t = LOOP_SEC
        while t <= horizon_sec:
            if (t - last) >= poll_sec:
                out.append(t)
                last = t
            t += LOOP_SEC
        return out
    if polling_mode == "lazy":
        out = []
        last = -10**9
        t = LOOP_SEC
        while t <= horizon_sec:
            interval = LAZY_FAST_SEC if t < LAZY_FAST_WINDOW else LAZY_SLOW_SEC
            if (t - last) >= interval:
                out.append(t)
                last = t
            t += LOOP_SEC
        return out
    return []


# =============================================================================
# Smoothing — applied to the source-base price stream (jupiter or ds) per poll
# =============================================================================
class _SmoothingState:
    __slots__ = ("ema", "hist", "prev_p", "armed_sl", "armed_tp")

    def __init__(self):
        self.ema = None
        self.hist = []
        self.prev_p = None
        self.armed_sl = True
        self.armed_tp = True


def _apply_smoothing(state: _SmoothingState, base_p: float, mode: str,
                     sl_price: float, tp_price: float | None) -> float:
    """Mirror paper_trader._decision_price smoothing modes."""
    if mode == "raw":
        return base_p
    if mode == "ema_fast":
        alpha = 2 / (2 + 1)
        state.ema = base_p if state.ema is None else alpha * base_p + (1 - alpha) * state.ema
        return state.ema
    if mode == "ema_slow":
        alpha = 2 / (8 + 1)
        state.ema = base_p if state.ema is None else alpha * base_p + (1 - alpha) * state.ema
        return state.ema
    if mode == "median_3":
        state.hist.append(base_p)
        if len(state.hist) > 3:
            state.hist.pop(0)
        if len(state.hist) < 3:
            return base_p
        return sorted(state.hist)[len(state.hist) // 2]
    if mode == "median_5":
        state.hist.append(base_p)
        if len(state.hist) > 5:
            state.hist.pop(0)
        if len(state.hist) < 5:
            return base_p
        return sorted(state.hist)[len(state.hist) // 2]
    if mode == "winsor_p95":
        if state.prev_p is None:
            state.prev_p = base_p
            return base_p
        cap = state.prev_p * 0.18
        delta = base_p - state.prev_p
        out = state.prev_p + max(-cap, min(cap, delta))
        state.prev_p = out
        return out
    if mode == "dual_confirm":
        if state.prev_p is None:
            state.prev_p = base_p
            return base_p
        prev = state.prev_p
        state.prev_p = base_p
        if sl_price and base_p <= sl_price and prev > sl_price:
            return prev
        if tp_price and base_p >= tp_price and prev < tp_price:
            return prev
        return base_p
    if mode == "hysteresis":
        if not state.armed_sl and sl_price and base_p >= sl_price * 1.02:
            state.armed_sl = True
        elif state.armed_sl and sl_price and base_p <= sl_price:
            state.armed_sl = False
        if not state.armed_tp and tp_price and base_p <= tp_price * 0.98:
            state.armed_tp = True
        elif state.armed_tp and tp_price and base_p >= tp_price:
            state.armed_tp = False
        if not state.armed_sl and sl_price and base_p <= sl_price:
            return sl_price * 1.001
        if not state.armed_tp and tp_price and base_p >= tp_price:
            return tp_price * 0.999
        return base_p
    return base_p


# =============================================================================
# Look-back (cache-style) — latest tick at/before poll_time
# =============================================================================
def _latest_at_or_before(sorted_ticks: list[dict], t_iso: str) -> float | None:
    last = None
    for tk in sorted_ticks:
        if tk["fetched_at"] <= t_iso:
            p = float(tk["price_usd"])
            if p > 0:
                last = p
        else:
            break
    return last


# =============================================================================
# Single-trade replay
# =============================================================================
SELL_SLIP = 1 - 10 / 10_000


def _replay(fake_trade: dict, jp_sorted: list[dict], ds_sorted: list[dict],
            source: str, smoothing: str, polling_mode: str,
            horizon_min: int) -> dict | None:
    entry_time = datetime.fromisoformat(fake_trade["created_at"].replace("Z", "+00:00"))
    trade_id = str(fake_trade["id"])
    _last_eval_ts.pop(trade_id, None)

    horizon_sec = horizon_min * 60
    poll_offsets = _poll_offsets(polling_mode, horizon_sec)
    if not poll_offsets:
        return None

    sl_price = float(fake_trade["sl_price"])
    tp_price = float(fake_trade["tp_price"]) if fake_trade.get("tp_price") else None
    state = _SmoothingState()
    last_exec = None

    for offset in poll_offsets:
        poll_time = entry_time + timedelta(seconds=offset)
        poll_iso = poll_time.isoformat().replace("+00:00", "Z")
        jp = _latest_at_or_before(jp_sorted, poll_iso)
        ds = _latest_at_or_before(ds_sorted, poll_iso)

        if source == "jupiter":
            base = jp
            exec_p = jp
        else:  # dexscreener
            base = ds
            exec_p = jp if jp is not None else ds

        if base is None or exec_p is None:
            continue
        last_exec = exec_p

        dec_p = _apply_smoothing(state, base, smoothing, sl_price, tp_price)

        ev = _evaluate_trade_exit(fake_trade, exec_p, poll_time, SELL_SLIP,
                                  sell_fee_bps=0, decision_price=dec_p)
        if ev is None:
            continue
        if ev.get("high_price_seen") is not None:
            new_high = ev["high_price_seen"]
            if new_high > float(fake_trade.get("high_price_seen") or 0):
                fake_trade["high_price_seen"] = new_high
        if "status" in ev and ev["status"]:
            return {
                "status": ev["status"],
                "pnl_pct": ev.get("pnl_pct", 0),
            }

    if last_exec is None:
        return None
    entry_price = float(fake_trade["entry_price"])
    return {
        "status": "timeout_eod",
        "pnl_pct": round((last_exec / entry_price) - 1, 4) if entry_price > 0 else 0,
    }


# =============================================================================
# Universe + tick fetch
# =============================================================================
def _fetch_universe() -> list[dict]:
    rows = []
    page = 1000
    off = 0
    while True:
        r = (sb.table("paper_trades")
               .select("token_address,created_at,entry_price,rt_liquidity_usd,"
                       "dex_spot_price_at_entry")
               .eq("source", "rt").gte("created_at", POST_V132)
               .order("created_at").range(off, off + page - 1).execute().data)
        if not r:
            break
        rows.extend(r)
        if len(r) < page:
            break
        off += page
    by_token = {}
    for r in rows:
        if r["token_address"] not in by_token:
            by_token[r["token_address"]] = r
    return list(by_token.values())


def _fetch_ticks_bulk(addrs: list[str]) -> dict[str, dict]:
    out = {}
    end = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    start = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    for i, addr in enumerate(addrs):
        rows = (sb.table("price_ticks")
                  .select("price_usd,fetched_at,source")
                  .eq("token_address", addr)
                  .gte("fetched_at", start).lte("fetched_at", end)
                  .order("fetched_at").execute().data)
        if rows:
            jp = sorted([t for t in rows if t["source"] == "jupiter"],
                         key=lambda t: t["fetched_at"])
            ds = sorted([t for t in rows if t["source"] in ("fast", "full", "live")],
                         key=lambda t: t["fetched_at"])
            out[addr] = {"jp": jp, "ds": ds}
        if (i + 1) % 10 == 0:
            print(f"  ticks: {i + 1}/{len(addrs)}", flush=True)
    return out


# =============================================================================
# Ground truth $/day
# =============================================================================
def ground_truth_daily(strats: list[str], pos_usd: float, label: str) -> None:
    print("=" * 90)
    print(f"GROUND TRUTH — {label} ($/day from real --from-trades)")
    print("=" * 90)
    print(f"{'Strategy':<22}{'N':>5}{'WR%':>6}{'avg%':>8}{'days':>7}"
          f"{'trd/day':>9}{'$/trade':>10}{'$/day':>10}")
    now = datetime.now(timezone.utc)
    total = 0.0
    for s in strats:
        rows = (sb.table("paper_trades")
                  .select("pnl_pct,pnl_usd,created_at,status")
                  .eq("strategy", s).eq("source", "rt")
                  .gte("created_at", POST_V132).execute().data)
        closed = [r for r in rows if r.get("status") in
                  ("tp_hit", "sl_hit", "timeout", "trail_stop", "trail_crash")]
        if not closed:
            continue
        first_dt = datetime.fromisoformat(min(r["created_at"] for r in closed
                                               ).replace("Z", "+00:00"))
        days = max(0.5, (now - first_dt).total_seconds() / 86400)
        pnls = [float(r["pnl_pct"]) for r in closed if r.get("pnl_pct") is not None]
        wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        avg = statistics.mean(pnls) * 100
        per_day = len(closed) / days
        usd_t = pos_usd * (avg / 100)
        usd_d = usd_t * per_day
        total += usd_d
        print(f"{s:<22}{len(closed):>5}{wr:>5.0f}%{avg:>+7.2f}%{days:>7.2f}"
              f"{per_day:>9.1f}{usd_t:>+9.2f}${usd_d:>+9.2f}$")
    print(f"{'-'*90}")
    print(f"{'TOTAL projected $/day':<22}{'':<35}{total:>+19.2f}$")
    print()


# =============================================================================
# Main
# =============================================================================
def main():
    print(f"\n{'#'*90}\n# MEGA SWEEP TRUE GRID {datetime.now().isoformat()[:19]}\n{'#'*90}\n")

    # 1) Ground truth
    ground_truth_daily(ACTIVE_PAPER, PAPER_POS_USD, "PAPER (active 5)")
    ground_truth_daily(ACTIVE_LIVE, LIVE_POS_USD, "LIVE (active 2)")

    # 2) Mega sweep
    print("=" * 90)
    print(f"GRID: {len(NON_TRAIL_STRATS)} strats × {len(SOURCES)} src × "
          f"{len(SMOOTHINGS)} smooth × {len(POLLING_MODES)} poll-mode = "
          f"{len(NON_TRAIL_STRATS)*len(SOURCES)*len(SMOOTHINGS)*len(POLLING_MODES)} configs")
    print("=" * 90)

    universe = _fetch_universe()
    print(f"Universe: {len(universe)} unique tokens")
    addrs = [u["token_address"] for u in universe]
    print(f"Fetching ticks for {len(addrs)} tokens...")
    t0 = time.time()
    ticks = _fetch_ticks_bulk(addrs)
    print(f"  done in {time.time()-t0:.0f}s, {len(ticks)} tokens with ticks")

    results = []
    done = 0
    total_configs = (len(NON_TRAIL_STRATS) * len(SOURCES) *
                     len(SMOOTHINGS) * len(POLLING_MODES))
    t0 = time.time()
    for strat in NON_TRAIL_STRATS:
        tr0 = STRATEGIES[strat][0]
        tp_mult = tr0.get("tp_mult")
        sl_mult = tr0.get("sl_mult", 0.50)
        horizon_min = tr0.get("horizon_min", 120) or 120

        for source in SOURCES:
            for smoothing in SMOOTHINGS:
                for poll_mode in POLLING_MODES:
                    pnls = []
                    for u in universe:
                        addr = u["token_address"]
                        td = ticks.get(addr)
                        if not td:
                            continue
                        # Filter to trade window
                        entry_ts = datetime.fromisoformat(u["created_at"].replace("Z", "+00:00"))
                        t_end_iso = (entry_ts + timedelta(minutes=horizon_min)
                                     ).isoformat().replace("+00:00", "Z")
                        jp_win = [t for t in td["jp"]
                                   if u["created_at"] <= t["fetched_at"] <= t_end_iso]
                        ds_win = [t for t in td["ds"]
                                   if u["created_at"] <= t["fetched_at"] <= t_end_iso]
                        if len(jp_win) < 3 and len(ds_win) < 3:
                            continue
                        entry_price = float(u["entry_price"])
                        sl_p = entry_price * sl_mult
                        tp_p = entry_price * tp_mult if tp_mult else None
                        fake = {
                            "id": f"sim_{addr[:8]}",
                            "entry_price": entry_price,
                            "sl_price": sl_p, "tp_price": tp_p,
                            "position_usd": 10.0,
                            "strategy": strat, "tranche_label": "main",
                            "horizon_minutes": horizon_min,
                            "created_at": u["created_at"],
                            "high_price_seen": entry_price,
                            "rt_liquidity_usd": u.get("rt_liquidity_usd"),
                            "dex_spot_price_at_entry": float(u.get("dex_spot_price_at_entry") or entry_price),
                        }
                        res = _replay(fake, jp_win, ds_win, source, smoothing,
                                      poll_mode, horizon_min)
                        if res is None:
                            continue
                        pnls.append(res["pnl_pct"])
                    done += 1
                    n = len(pnls)
                    if n < 10:
                        continue
                    arr = np.array(pnls)
                    wr = float((arr > 0).mean()) * 100
                    avg = float(arr.mean()) * 100
                    med = float(np.median(arr)) * 100
                    std = float(arr.std(ddof=1)) * 100 if n > 1 else 0
                    sharpe = (avg / std) if std > 0 else 0
                    wins = arr[arr > 0]
                    losses = arr[arr <= 0]
                    aw = float(wins.mean()) if len(wins) else 0
                    al = float(-losses.mean()) if len(losses) else 0
                    kelly = (max(0, (wr/100 * (aw/al + 1) - 1) / (aw/al)) * 100
                             if al > 0 and aw > 0 else 0)
                    eq = np.cumprod(1 + arr)
                    peaks = np.maximum.accumulate(eq)
                    dd = float(((eq - peaks) / peaks).min()) * 100
                    results.append({
                        "strategy": strat, "source": source, "smoothing": smoothing,
                        "polling_mode": poll_mode, "n": n,
                        "wr_pct": round(wr, 2), "avg_pnl_pct": round(avg, 3),
                        "median_pnl_pct": round(med, 3), "stdev_pct": round(std, 3),
                        "sharpe": round(sharpe, 4), "kelly_pct": round(kelly, 3),
                        "max_dd_pct": round(dd, 2), "horizon_min": horizon_min,
                    })
                    if done % 200 == 0:
                        elapsed = time.time() - t0
                        pct = 100 * done / total_configs
                        print(f"  progress: {done}/{total_configs} ({pct:.1f}%) "
                              f"in {elapsed:.0f}s", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(CSV_OUT, index=False)
    print(f"\nCSV: {CSV_OUT}  ({len(df)} rows in {time.time()-t0:.0f}s)")

    # Reports
    df["rk_kelly"] = df["kelly_pct"].rank(ascending=False, method="min")
    df["rk_med"] = df["median_pnl_pct"].rank(ascending=False, method="min")
    df["rk_sh"] = df["sharpe"].rank(ascending=False, method="min")
    df["rk_sum"] = df[["rk_kelly", "rk_med", "rk_sh"]].sum(axis=1)
    df = df.sort_values("rk_sum")

    print("\n" + "=" * 90)
    print("TOP 30 OVERALL (composite kelly + median + sharpe)")
    print("=" * 90)
    print(df.head(30)[["strategy", "source", "smoothing", "polling_mode",
                        "n", "wr_pct", "avg_pnl_pct", "median_pnl_pct",
                        "sharpe", "kelly_pct", "max_dd_pct"]].to_string(index=False))

    print("\n" + "=" * 90)
    print("BEST CONFIG PER STRATEGY (top 25)")
    print("=" * 90)
    bps = (df.sort_values("rk_sum")
             .drop_duplicates(subset=["strategy"], keep="first").head(25))
    print(bps[["strategy", "source", "smoothing", "polling_mode",
                "n", "wr_pct", "avg_pnl_pct", "kelly_pct"]].to_string(index=False))

    # Per family
    fam_pat = [
        ("BE", re.compile(r"^BE\d+_TP")),
        ("FAST", re.compile(r"^FAST_TP")),
        ("FAST60", re.compile(r"^FAST60_TP")),
        ("FAST45", re.compile(r"^FAST45_TP")),
        ("SLOW4H", re.compile(r"^SLOW4H_TP")),
        ("SLOW6H", re.compile(r"^SLOW6H_TP")),
        ("DECAY", re.compile(r"^DECAY_")),
        ("SCALP", re.compile(r"^SCALP_")),
        ("FIXED", re.compile(r"^TP\d+_")),
        ("OTHER", re.compile(r".*")),
    ]

    def _fam(name):
        for f, p in fam_pat:
            if p.match(name):
                return f
        return "OTHER"
    df["family"] = df["strategy"].map(_fam)
    fw = (df.sort_values("rk_sum").drop_duplicates(subset=["family"], keep="first"))
    print("\n" + "=" * 90)
    print("BEST PER FAMILY")
    print("=" * 90)
    print(fw[["family", "strategy", "source", "smoothing", "polling_mode",
                "n", "wr_pct", "avg_pnl_pct", "kelly_pct"]].to_string(index=False))

    # Per polling mode (best across all strats)
    print("\n" + "=" * 90)
    print("BEST CONFIG PER POLLING MODE")
    print("=" * 90)
    pm = (df.sort_values("rk_sum")
            .drop_duplicates(subset=["polling_mode"], keep="first"))
    print(pm[["polling_mode", "strategy", "source", "smoothing", "n",
                "wr_pct", "avg_pnl_pct", "kelly_pct"]].to_string(index=False))

    # Per smoothing mode
    print("\n" + "=" * 90)
    print("BEST CONFIG PER SMOOTHING MODE")
    print("=" * 90)
    sm = (df.sort_values("rk_sum")
            .drop_duplicates(subset=["smoothing"], keep="first"))
    print(sm[["smoothing", "strategy", "source", "polling_mode", "n",
                "wr_pct", "avg_pnl_pct", "kelly_pct"]].to_string(index=False))

    print("\nDONE.")


if __name__ == "__main__":
    main()
