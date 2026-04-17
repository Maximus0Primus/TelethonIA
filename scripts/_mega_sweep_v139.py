"""v139 MEGA SWEEP — strategies × FILTERS × sources × smoothings × polling modes.

Added vs v138.2 mega:
  1. Filter dimension (NONE / NOZEROLIQ / SCORE30 / SCORE40 / MCAP_MID /
                       TOPKOL / combos) — tested in _test_new_strategies.py
  2. Extended strategy pool with TP150-TP500 variants (asymmetric payoff)
     at horizon 120-240min. Not in STRATEGIES dict — defined inline.
  3. Focus on promising families only (FIXED/BE/FAST/TP200) — skip DTRAIL,
     SCALP, SLOW4/6H, DECAY (mega v138.2 showed them mediocre).

Estimated matrix: ~35 strats × 7 filters × 2 src × 4 smooth × 5 poll
                = ~9800 configs × ~65 tokens = ~637K replays (~10-15 min).

Outputs: scraper/_mega_sweep_v139.{csv,log}
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

from paper_trader import _evaluate_trade_exit, _last_eval_ts  # noqa
from strategies import _BE_RE, _get_trail_config, _get_decay_end  # noqa

POST_V132 = "2026-04-13T20:00:00Z"
CSV_OUT = os.path.join(SCRAPER, "_mega_sweep_v139.csv")

TOP_KOLS = {"FrenzGems", "jadendegens", "gubbinscalls", "Archerrgambles",
            "ChadleyGambles123", "zcallz"}

# Strategy spec = (tp_mult, sl_mult, horizon_min, be_activation_or_None)
STRATEGY_POOL = {
    # FIXED family — TP/SL grid
    "TP50_SL15":   (1.50, 0.85, 120, None),
    "TP50_SL30":   (1.50, 0.70, 120, None),
    "TP80_SL30":   (1.80, 0.70, 120, None),
    "TP100_SL50":  (2.00, 0.50, 120, None),
    "TP100_SL30":  (2.00, 0.70, 120, None),
    # FAST family — short horizon
    "FAST_TP50_SL30":  (1.50, 0.70, 30, None),
    "FAST_TP80_SL25":  (1.80, 0.75, 30, None),
    "FAST_TP100_SL20": (2.00, 0.80, 30, None),
    "FAST_TP100_SL50": (2.00, 0.50, 30, None),
    # BE family — breakeven protection
    "BE15_TP100_SL50": (2.00, 0.50, 120, 0.15),
    "BE25_TP80_SL30":  (1.80, 0.70, 30,  0.25),
    "BE25_TP100_SL50": (2.00, 0.50, 120, 0.25),
    "BE30_TP100_SL50": (2.00, 0.50, 120, 0.30),
    # NEW v139 — TP200+ asymmetric payoff
    "TP150_SL40_2H":   (2.50, 0.60, 120, None),
    "TP150_SL40_4H":   (2.50, 0.60, 240, None),
    "TP200_SL30_4H":   (3.00, 0.70, 240, None),
    "TP200_SL40_4H":   (3.00, 0.60, 240, None),
    "TP200_SL50_4H":   (3.00, 0.50, 240, None),
    "TP300_SL40_4H":   (4.00, 0.60, 240, None),
    "TP300_SL50_4H":   (4.00, 0.50, 240, None),
    "TP500_SL50_4H":   (6.00, 0.50, 240, None),
    # NEW v139 — BE + TP200 combo
    "BE15_TP200_SL40_4H": (3.00, 0.60, 240, 0.15),
    "BE25_TP200_SL40_4H": (3.00, 0.60, 240, 0.25),
    "BE50_TP200_SL30_4H": (3.00, 0.70, 240, 0.50),  # BE activation at +50%
    # NEW v139 — TP200 short horizon
    "FAST_TP200_SL40": (3.00, 0.60, 60, None),
}

# Entry filters
FILTERS = {
    "NONE":              lambda t: True,
    "NOZEROLIQ":         lambda t: (t.get("rt_liquidity_usd") or 0) > 0,
    "SCORE30":           lambda t: (t.get("rt_score") or 0) >= 30,
    "SCORE40":           lambda t: (t.get("rt_score") or 0) >= 40,
    "MCAP_MID":          lambda t: 30_000 <= (t.get("entry_mcap") or 0) <= 500_000,
    "TOPKOL":            lambda t: (t.get("kol_group") or "") in TOP_KOLS,
    "NOZEROLIQ_SCORE30": lambda t: (t.get("rt_liquidity_usd") or 0) > 0 and (t.get("rt_score") or 0) >= 30,
}

SOURCES = ["jupiter", "dexscreener"]
SMOOTHINGS = ["raw", "ema_fast", "median_5", "median_3"]
POLLING_MODES = ["fast", "static_60", "static_120", "static_240", "lazy"]

LOOP_SEC = 30
LAZY_FAST_SEC = 180
LAZY_FAST_WINDOW = 300
LAZY_SLOW_SEC = 600


# ===== Poll schedule =====
def _poll_offsets(polling_mode: str, horizon_sec: int) -> list[int]:
    if polling_mode == "fast":
        return list(range(LOOP_SEC, horizon_sec + 1, LOOP_SEC))
    if polling_mode.startswith("static_"):
        poll_sec = int(polling_mode.split("_")[1])
        out = []; last = -10**9; t = LOOP_SEC
        while t <= horizon_sec:
            if (t - last) >= poll_sec:
                out.append(t); last = t
            t += LOOP_SEC
        return out
    if polling_mode == "lazy":
        out = []; last = -10**9; t = LOOP_SEC
        while t <= horizon_sec:
            interval = LAZY_FAST_SEC if t < LAZY_FAST_WINDOW else LAZY_SLOW_SEC
            if (t - last) >= interval:
                out.append(t); last = t
            t += LOOP_SEC
        return out
    return []


# ===== Smoothing =====
class _SmState:
    __slots__ = ("ema", "hist")
    def __init__(self): self.ema = None; self.hist = []


def _smooth(st: _SmState, p: float, mode: str) -> float:
    if mode == "raw": return p
    if mode == "ema_fast":
        alpha = 2/3
        st.ema = p if st.ema is None else alpha*p + (1-alpha)*st.ema
        return st.ema
    if mode == "median_3":
        st.hist.append(p)
        if len(st.hist) > 3: st.hist.pop(0)
        return sorted(st.hist)[len(st.hist)//2] if len(st.hist) >= 3 else p
    if mode == "median_5":
        st.hist.append(p)
        if len(st.hist) > 5: st.hist.pop(0)
        return sorted(st.hist)[len(st.hist)//2] if len(st.hist) >= 5 else p
    return p


def _latest_at_or_before(sorted_ticks, t_iso):
    last = None
    for tk in sorted_ticks:
        if tk["fetched_at"] <= t_iso:
            p = float(tk["price_usd"])
            if p > 0: last = p
        else: break
    return last


# ===== Replay =====
SELL_SLIP = 1 - 10/10_000


def _replay(tp_mult, sl_mult, horizon_min, be_act,
            jp_sorted, ds_sorted, entry_price, entry_time,
            source, smoothing, polling_mode, rt_liq_usd):
    trade_id = f"sweep_{id(jp_sorted)}"
    _last_eval_ts.pop(trade_id, None)
    horizon_sec = horizon_min * 60
    poll_offsets = _poll_offsets(polling_mode, horizon_sec)
    if not poll_offsets: return None

    sl_price = entry_price * sl_mult
    tp_price = entry_price * tp_mult if tp_mult else None

    fake_trade = {
        "id": trade_id, "entry_price": entry_price,
        "sl_price": sl_price, "tp_price": tp_price,
        "position_usd": 10.0,
        "strategy": f"BE{int(be_act*100)}_TP80_SL30" if be_act else "TP80_SL30",  # hint for BE detection
        "tranche_label": "main", "horizon_minutes": horizon_min,
        "created_at": entry_time.isoformat().replace("+00:00", "Z"),
        "high_price_seen": entry_price,
        "rt_liquidity_usd": rt_liq_usd,
        "dex_spot_price_at_entry": entry_price,
    }

    st = _SmState()
    last_exec = None

    for offset in poll_offsets:
        poll_time = entry_time + timedelta(seconds=offset)
        poll_iso = poll_time.isoformat().replace("+00:00", "Z")
        jp = _latest_at_or_before(jp_sorted, poll_iso)
        ds = _latest_at_or_before(ds_sorted, poll_iso)

        if source == "jupiter":
            base = jp; exec_p = jp
        else:
            base = ds; exec_p = jp if jp is not None else ds

        if base is None or exec_p is None: continue
        last_exec = exec_p
        dec_p = _smooth(st, base, smoothing)

        ev = _evaluate_trade_exit(fake_trade, exec_p, poll_time, SELL_SLIP,
                                  sell_fee_bps=0, decision_price=dec_p)
        if ev is None: continue
        if ev.get("high_price_seen") is not None:
            h = ev["high_price_seen"]
            if h > float(fake_trade.get("high_price_seen") or 0):
                fake_trade["high_price_seen"] = h
        if "status" in ev and ev["status"]:
            return {"status": ev["status"], "pnl_pct": ev.get("pnl_pct", 0)}

    if last_exec is None: return None
    return {"status": "timeout_eod",
            "pnl_pct": round((last_exec / entry_price) - 1, 4) if entry_price else 0}


# ===== Universe + ticks =====
def load_universe():
    rows = []; off = 0
    while True:
        r = (sb.table("paper_trades")
               .select("id,token_address,created_at,entry_price,rt_liquidity_usd,"
                       "rt_score,kol_group,entry_mcap,rt_is_pump_fun,"
                       "n_kol_confirmations,rt_token_age_hours")
               .eq("source", "rt").gte("created_at", POST_V132)
               .order("created_at").range(off, off + 999).execute().data)
        if not r: break
        rows.extend(r)
        if len(r) < 1000: break
        off += 1000
    by_token = {}
    for r in rows:
        if r["token_address"] not in by_token:
            by_token[r["token_address"]] = r
    return list(by_token.values())


def fetch_ticks(addrs):
    end = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    start = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    out = {}
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
            print(f"  ticks: {i+1}/{len(addrs)}", flush=True)
    return out


# ===== Main =====
def main():
    print(f"{'#'*90}\n# v139 MEGA SWEEP {datetime.now().isoformat()[:19]}\n{'#'*90}\n")

    print("Loading universe + ticks...")
    t0 = time.time()
    universe = load_universe()
    print(f"Universe: {len(universe)} unique tokens")
    ticks = fetch_ticks([u["token_address"] for u in universe])
    print(f"  {len(ticks)} tokens with ticks in {time.time()-t0:.0f}s")

    total_configs = (len(STRATEGY_POOL) * len(FILTERS) * len(SOURCES)
                     * len(SMOOTHINGS) * len(POLLING_MODES))
    print(f"\nGrid: {len(STRATEGY_POOL)} strats × {len(FILTERS)} filters × "
          f"{len(SOURCES)} src × {len(SMOOTHINGS)} smooth × {len(POLLING_MODES)} poll "
          f"= {total_configs} configs")
    print(f"Est. replays: ~{total_configs * len(universe)}")

    # Pre-filter tokens per filter
    filtered = {fname: [u for u in universe if ff(u)] for fname, ff in FILTERS.items()}
    for fname, ts in filtered.items():
        print(f"  filter {fname:<22}: {len(ts)}/{len(universe)} pass")

    results = []
    done = 0
    t0 = time.time()

    for strat_name, (tp_m, sl_m, horizon, be_act) in STRATEGY_POOL.items():
        for fname, ftokens in filtered.items():
            if len(ftokens) < 10:
                done += len(SOURCES) * len(SMOOTHINGS) * len(POLLING_MODES)
                continue  # skip filter combos with <10 tokens

            for source in SOURCES:
                for smoothing in SMOOTHINGS:
                    for poll_mode in POLLING_MODES:
                        pnls = []
                        for u in ftokens:
                            addr = u["token_address"]
                            td = ticks.get(addr)
                            if not td: continue
                            entry_ts = datetime.fromisoformat(u["created_at"].replace("Z", "+00:00"))
                            t_end = (entry_ts + timedelta(minutes=horizon)
                                     ).isoformat().replace("+00:00", "Z")
                            jp = [t for t in td["jp"]
                                  if u["created_at"] <= t["fetched_at"] <= t_end]
                            ds = [t for t in td["ds"]
                                  if u["created_at"] <= t["fetched_at"] <= t_end]
                            if len(jp) < 3 and len(ds) < 3: continue

                            res = _replay(tp_m, sl_m, horizon, be_act,
                                          jp, ds, float(u["entry_price"]),
                                          entry_ts, source, smoothing, poll_mode,
                                          u.get("rt_liquidity_usd"))
                            if res is None: continue
                            pnls.append(res["pnl_pct"])

                        done += 1
                        n = len(pnls)
                        if n < 10: continue
                        arr = np.array(pnls)
                        wr = float((arr > 0).mean()) * 100
                        avg = float(arr.mean()) * 100
                        med = float(np.median(arr)) * 100
                        # $/jour at $50/trade, trade_rate scaled by filter selectivity
                        trade_rate = len(ftokens) / max(1, len(universe)) * 18
                        dollars_day = 50 * (avg / 100) * trade_rate
                        # Sharpe
                        std = float(arr.std(ddof=1)) * 100 if n > 1 else 0
                        sharpe = (avg / std) if std > 0 else 0
                        # max DD
                        eq = np.cumprod(1 + arr)
                        peaks = np.maximum.accumulate(eq)
                        dd = float(((eq - peaks) / peaks).min()) * 100

                        results.append({
                            "strategy": strat_name, "filter": fname,
                            "source": source, "smoothing": smoothing,
                            "polling_mode": poll_mode,
                            "n_pass": len(ftokens), "n": n,
                            "wr_pct": round(wr, 2),
                            "avg_pnl_pct": round(avg, 3),
                            "median_pnl_pct": round(med, 3),
                            "sharpe": round(sharpe, 4),
                            "max_dd_pct": round(dd, 2),
                            "dollars_per_day": round(dollars_day, 2),
                            "horizon_min": horizon,
                        })

                        if done % 500 == 0:
                            print(f"  progress: {done}/{total_configs} "
                                  f"({100*done/total_configs:.1f}%) in {time.time()-t0:.0f}s",
                                  flush=True)

    df = pd.DataFrame(results)
    df.to_csv(CSV_OUT, index=False)
    print(f"\n{len(df)} rows in {time.time()-t0:.0f}s → {CSV_OUT}")

    # Rank by $/day (the metric that matters for fixed-position paper)
    df = df.sort_values("dollars_per_day", ascending=False)

    print("\n" + "=" * 110)
    print("TOP 30 BY $/DAY (fixed $50/trade × filtered trade rate)")
    print("=" * 110)
    print(df.head(30)[["strategy", "filter", "source", "smoothing", "polling_mode",
                        "n", "wr_pct", "avg_pnl_pct", "median_pnl_pct",
                        "dollars_per_day"]].to_string(index=False))

    # Best per (strategy, filter) combo — collapses source/smoothing/polling noise
    print("\n" + "=" * 110)
    print("BEST CONFIG PER (STRATEGY × FILTER) — top 30")
    print("=" * 110)
    bc = (df.sort_values("dollars_per_day", ascending=False)
            .drop_duplicates(subset=["strategy", "filter"], keep="first").head(30))
    print(bc[["strategy", "filter", "source", "smoothing", "polling_mode",
               "n", "avg_pnl_pct", "dollars_per_day"]].to_string(index=False))

    # Best per strategy (any filter)
    print("\n" + "=" * 110)
    print("BEST CONFIG PER STRATEGY (any filter) — top 25")
    print("=" * 110)
    bs = (df.sort_values("dollars_per_day", ascending=False)
            .drop_duplicates(subset=["strategy"], keep="first").head(25))
    print(bs[["strategy", "filter", "source", "smoothing", "polling_mode",
               "n", "avg_pnl_pct", "dollars_per_day"]].to_string(index=False))

    # Best per filter (any strategy)
    print("\n" + "=" * 110)
    print("BEST CONFIG PER FILTER")
    print("=" * 110)
    bf = (df.sort_values("dollars_per_day", ascending=False)
            .drop_duplicates(subset=["filter"], keep="first"))
    print(bf[["filter", "strategy", "source", "smoothing", "polling_mode",
               "n", "avg_pnl_pct", "dollars_per_day"]].to_string(index=False))

    print("\nDONE.")


if __name__ == "__main__":
    main()
