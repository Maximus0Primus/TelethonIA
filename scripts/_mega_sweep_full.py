"""v139 FULL MEGA SWEEP — ALL strategies × filters × sources × smoothings × polling.

Uses multiprocessing Pool to complete in ~30-45 min instead of 5h sequential.
Ticks are shared via temp JSON file (not pickle).
"""
from __future__ import annotations
import os
import sys
import time
import json
import tempfile
import multiprocessing as mp
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

POST_V132 = "2026-04-13T20:00:00Z"
CSV_OUT = os.path.join(SCRAPER, "_mega_sweep_full.csv")
TICKS_TMP = os.path.join(SCRAPER, "_mega_ticks_tmp.json")

TOP_KOLS = {"FrenzGems", "jadendegens", "gubbinscalls", "Archerrgambles",
            "ChadleyGambles123", "zcallz"}

NEW_V139_STRATS = {
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

SOURCES = ["jupiter", "dexscreener"]
SMOOTHINGS = ["raw", "ema_fast", "ema_slow", "median_3", "median_5",
              "winsor_p95", "dual_confirm", "hysteresis"]
POLLING_MODES = ["fast", "static_60", "static_120", "static_240", "lazy"]
FILTERS = ["NONE", "NOZEROLIQ", "SCORE30", "SCORE40", "MCAP_MID", "TOPKOL",
           "NOZEROLIQ_SCORE30"]

LOOP_SEC = 30
LAZY_FAST_SEC = 180
LAZY_FAST_WINDOW = 300
LAZY_SLOW_SEC = 600

_TICKS = None


def _init_worker(ticks_path):
    global _TICKS
    with open(ticks_path) as f:
        _TICKS = json.load(f)


def _poll_offsets(polling_mode, horizon_sec):
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


class _SmState:
    __slots__ = ("ema", "hist", "prev_p", "armed_sl", "armed_tp")
    def __init__(self):
        self.ema = None; self.hist = []; self.prev_p = None
        self.armed_sl = True; self.armed_tp = True


def _smooth(st, p, mode, sl_price, tp_price):
    if mode == "raw": return p
    if mode == "ema_fast":
        alpha = 2/3
        st.ema = p if st.ema is None else alpha*p + (1-alpha)*st.ema
        return st.ema
    if mode == "ema_slow":
        alpha = 2/9
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
    return p


def _latest_at_or_before(sorted_ticks, t_iso):
    last = None
    for tk in sorted_ticks:
        if tk["fetched_at"] <= t_iso:
            p = float(tk["price_usd"])
            if p > 0: last = p
        else: break
    return last


SELL_SLIP_BASE = 1 - 10/10_000


def _replay_one(tp_mult, sl_mult, horizon_min, be_act,
                jp_sorted, ds_sorted, entry_price, entry_time_iso,
                source, smoothing, polling_mode, rt_liq_usd):
    from paper_trader import _evaluate_trade_exit, _last_eval_ts
    entry_time = datetime.fromisoformat(entry_time_iso.replace("Z", "+00:00"))
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
        "strategy": f"BE{int(be_act*100)}_TP80_SL30" if be_act else "TP80_SL30",
        "tranche_label": "main", "horizon_minutes": horizon_min,
        "created_at": entry_time_iso,
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
        dec_p = _smooth(st, base, smoothing, sl_price, tp_price)
        ev = _evaluate_trade_exit(fake_trade, exec_p, poll_time, SELL_SLIP_BASE,
                                  sell_fee_bps=0, decision_price=dec_p)
        if ev is None: continue
        if ev.get("high_price_seen") is not None:
            h = ev["high_price_seen"]
            if h > float(fake_trade.get("high_price_seen") or 0):
                fake_trade["high_price_seen"] = h
        if "status" in ev and ev["status"]:
            return ev.get("pnl_pct", 0)
    if last_exec is None: return None
    return round((last_exec / entry_price) - 1, 4) if entry_price else 0


def _apply_filter(u, fname):
    if fname == "NONE": return True
    if fname == "NOZEROLIQ": return (u.get("rt_liquidity_usd") or 0) > 0
    if fname == "SCORE30": return (u.get("rt_score") or 0) >= 30
    if fname == "SCORE40": return (u.get("rt_score") or 0) >= 40
    if fname == "MCAP_MID": return 30_000 <= (u.get("entry_mcap") or 0) <= 500_000
    if fname == "TOPKOL": return (u.get("kol_group") or "") in TOP_KOLS
    if fname == "NOZEROLIQ_SCORE30":
        return (u.get("rt_liquidity_usd") or 0) > 0 and (u.get("rt_score") or 0) >= 30
    return True


def _process_config(args):
    (strat_name, tp_mult, sl_mult, horizon_min, be_act,
     fname, source, smoothing, polling_mode, universe) = args
    pnls = []
    for u in universe:
        if not _apply_filter(u, fname): continue
        addr = u["token_address"]
        td = _TICKS.get(addr)
        if not td: continue
        entry_ts = datetime.fromisoformat(u["created_at"].replace("Z", "+00:00"))
        t_end = (entry_ts + timedelta(minutes=horizon_min)).isoformat().replace("+00:00", "Z")
        jp = [t for t in td["jp"] if u["created_at"] <= t["fetched_at"] <= t_end]
        ds = [t for t in td["ds"] if u["created_at"] <= t["fetched_at"] <= t_end]
        if len(jp) < 3 and len(ds) < 3: continue
        pnl = _replay_one(tp_mult, sl_mult, horizon_min, be_act,
                          jp, ds, float(u["entry_price"]), u["created_at"],
                          source, smoothing, polling_mode,
                          u.get("rt_liquidity_usd"))
        if pnl is not None: pnls.append(pnl)
    n = len(pnls)
    if n < 10: return None
    arr = np.array(pnls)
    wr = float((arr > 0).mean()) * 100
    avg = float(arr.mean()) * 100
    med = float(np.median(arr)) * 100
    std = float(arr.std(ddof=1)) * 100 if n > 1 else 0
    sharpe = (avg / std) if std > 0 else 0
    eq = np.cumprod(1 + arr)
    peaks = np.maximum.accumulate(eq)
    dd = float(((eq - peaks) / peaks).min()) * 100
    n_pass = sum(1 for u in universe if _apply_filter(u, fname))
    trade_rate = n_pass / max(1, len(universe)) * 18
    dollars_day = 50 * (avg / 100) * trade_rate
    return {
        "strategy": strat_name, "filter": fname, "source": source,
        "smoothing": smoothing, "polling_mode": polling_mode,
        "n_pass": n_pass, "n": n, "wr_pct": round(wr, 2),
        "avg_pnl_pct": round(avg, 3), "median_pnl_pct": round(med, 3),
        "sharpe": round(sharpe, 4), "max_dd_pct": round(dd, 2),
        "dollars_per_day": round(dollars_day, 2), "horizon_min": horizon_min,
    }


def main():
    print(f"{'#'*90}\n# v139 FULL MEGA SWEEP {datetime.now().isoformat()[:19]}\n{'#'*90}\n")
    t0 = time.time()

    from strategies import STRATEGIES as _STRATS
    full_pool = {}
    import re as _re
    for name, tranches in _STRATS.items():
        tr0 = tranches[0]
        tp = tr0.get("tp_mult")
        sl = tr0.get("sl_mult", 0.50)
        h = tr0.get("horizon_min", 120) or 120
        be_m = _re.match(r"^BE(\d+)_TP", name)
        be_act = int(be_m.group(1)) / 100 if be_m else None
        full_pool[name] = (tp, sl, h, be_act)
    full_pool.update(NEW_V139_STRATS)
    print(f"Strategies: {len(full_pool)}")

    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    rows = []; off = 0
    while True:
        r = (sb.table("paper_trades")
               .select("id,token_address,created_at,entry_price,rt_liquidity_usd,"
                       "rt_score,kol_group,entry_mcap")
               .eq("source", "rt").gte("created_at", POST_V132)
               .order("created_at").range(off, off+999).execute().data)
        if not r: break
        rows.extend(r)
        if len(r) < 1000: break
        off += 1000
    by_token = {}
    for r in rows:
        if r["token_address"] not in by_token:
            by_token[r["token_address"]] = r
    universe = list(by_token.values())
    print(f"Universe: {len(universe)} unique tokens")

    print("Fetching ticks...")
    ticks = {}
    end = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    start = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    for i, u in enumerate(universe):
        addr = u["token_address"]
        rows = (sb.table("price_ticks")
                  .select("price_usd,fetched_at,source")
                  .eq("token_address", addr)
                  .gte("fetched_at", start).lte("fetched_at", end)
                  .order("fetched_at").execute().data)
        if rows:
            jp = sorted([t for t in rows if t["source"] == "jupiter"], key=lambda t: t["fetched_at"])
            ds = sorted([t for t in rows if t["source"] in ("fast","full","live")], key=lambda t: t["fetched_at"])
            ticks[addr] = {"jp": jp, "ds": ds}
        if (i+1) % 20 == 0: print(f"  {i+1}/{len(universe)}", flush=True)
    print(f"  {len(ticks)} with ticks in {time.time()-t0:.0f}s")

    # Persist ticks to temp JSON for worker-side loading (no pickle)
    with open(TICKS_TMP, "w") as f:
        json.dump(ticks, f)
    print(f"  ticks saved to {TICKS_TMP}: {os.path.getsize(TICKS_TMP)/1e6:.1f} MB")

    jobs = []
    for strat_name, (tp, sl, h, be) in full_pool.items():
        for fname in FILTERS:
            for source in SOURCES:
                for smoothing in SMOOTHINGS:
                    for poll_mode in POLLING_MODES:
                        jobs.append((strat_name, tp, sl, h, be, fname, source, smoothing, poll_mode, universe))
    total = len(jobs)
    print(f"\nTotal configs: {total}")

    n_workers = min(12, mp.cpu_count() - 2)
    print(f"Launching {n_workers} workers...\n")
    results = []
    t_start = time.time()
    with mp.Pool(n_workers, initializer=_init_worker, initargs=(TICKS_TMP,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_process_config, jobs, chunksize=50)):
            if r is not None: results.append(r)
            if (i+1) % 2000 == 0:
                pct = 100 * (i+1) / total
                elapsed = time.time() - t_start
                eta = elapsed / (i+1) * (total - i - 1)
                print(f"  {i+1}/{total} ({pct:.1f}%) in {elapsed:.0f}s, ETA {eta:.0f}s", flush=True)

    try: os.remove(TICKS_TMP)
    except Exception: pass

    df = pd.DataFrame(results)
    df.to_csv(CSV_OUT, index=False)
    print(f"\n{len(df)} valid rows / {total} → {CSV_OUT}")
    print(f"Total: {time.time()-t0:.0f}s")

    df = df.sort_values("dollars_per_day", ascending=False)
    print("\n" + "=" * 120)
    print("TOP 40 BY $/DAY")
    print("=" * 120)
    print(df.head(40)[["strategy","filter","source","smoothing","polling_mode",
                        "n","wr_pct","avg_pnl_pct","median_pnl_pct","dollars_per_day"]].to_string(index=False))

    print("\n" + "=" * 120)
    print("BEST PER STRATEGY (any filter) — top 40")
    print("=" * 120)
    bs = df.drop_duplicates(subset=["strategy"], keep="first").head(40)
    print(bs[["strategy","filter","source","smoothing","polling_mode",
               "n","avg_pnl_pct","dollars_per_day"]].to_string(index=False))

    print("\n" + "=" * 120)
    print("BEST PER FILTER")
    print("=" * 120)
    bf = df.drop_duplicates(subset=["filter"], keep="first")
    print(bf[["filter","strategy","source","smoothing","polling_mode",
               "n","avg_pnl_pct","dollars_per_day"]].to_string(index=False))

    print("\nDONE.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
