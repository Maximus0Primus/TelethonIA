"""Measure cadence bias of v135 sim vs v136 realistic vs real paper.

For each closed real paper trade in 6 active strategies:
  1. Replay with v135 _evaluate_call (current sim, fixed slip)
  2. Replay with v136 _evaluate_realistic (30s grid + look-back + dynamic slip)
  3. Compare both deltas vs real PnL

Prints per-strategy mean/median delta and MAE so we can quantify the fix.
"""
from __future__ import annotations
import os
import sys
import statistics
from datetime import datetime, timedelta

from dotenv import load_dotenv
from supabase import create_client

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SCRAPER = os.path.join(ROOT, "scraper")
sys.path.insert(0, SCRAPER)
sys.path.insert(0, HERE)

load_dotenv(os.path.join(SCRAPER, ".env"))
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

from strategies import STRATEGIES  # noqa: E402
from _sweep_v135_full import (  # noqa: E402
    _build_streams, _subsample, _smooth_stream, _evaluate_call,
    _fetch_ticks_for_token,
)
from _sweep_v136_realistic import _evaluate_realistic  # noqa: E402

POST_V132 = "2026-04-13T20:00:00Z"

PROBE = {
    "DTRAIL3_ACT10_SL70":  ("jupiter", 120, "raw"),
    "DTRAIL10_ACT10_SL70": ("jupiter", 120, "raw"),
    "DTRAIL5_ACT10_SL60":  ("jupiter", 120, "raw"),
    "DTRAIL3_ACT20_SL50":  ("jupiter", 120, "raw"),
    "BE25_TP80_SL30":      ("jupiter", 120, "ema_fast"),
    "FAST_TP100_SL20":     ("dexscreener", 120, "raw"),
}


def main():
    print("=" * 100)
    print("CADENCE BIAS PROBE — v135 sim vs v136 realistic vs real paper")
    print("=" * 100)
    print(f"{'Strategy':<22}{'N':>4}{'real_avg':>10}{'v135_avg':>10}{'v135_d':>9}"
          f"{'v135_MAE':>10}{'v136_avg':>10}{'v136_d':>9}{'v136_MAE':>10}"
          f"{'sm_v135':>9}{'sm_v136':>9}")

    for strat, (src, poll, mode) in PROBE.items():
        rows = (sb.table("paper_trades")
                  .select("token_address,created_at,entry_price,status,pnl_pct")
                  .eq("strategy", strat).eq("source", "rt")
                  .gte("created_at", POST_V132).execute().data)
        closed = [r for r in rows if r.get("status") in
                  ("tp_hit", "sl_hit", "timeout", "trail_stop", "trail_crash")]
        if not closed:
            continue

        tr0 = STRATEGIES[strat][0]
        tp_mult = tr0.get("tp_mult")
        sl_mult = tr0.get("sl_mult", 0.50)
        horizon_min = tr0.get("horizon_min", 120) or 120

        v135_d, v136_d, v135_p, v136_p, real_p = [], [], [], [], []
        sm_135 = sm_136 = 0
        for r in closed:
            addr = r["token_address"]
            entry_ts = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
            entry_price = float(r["entry_price"])
            t_end = (entry_ts + timedelta(minutes=horizon_min)
                     ).isoformat().replace("+00:00", "Z")
            try:
                ticks = _fetch_ticks_for_token(addr, r["created_at"], t_end)
            except Exception:
                continue
            streams = _build_streams(ticks)
            streams = {k: v for k, v in streams.items() if len(v) >= 3}
            if "jupiter" not in streams:
                continue
            jp = streams.get("jupiter", [])
            ds = streams.get("dexscreener", [])

            # v135
            dec_raw = _subsample(streams.get(src, []), poll)
            exec_str = streams.get("jupiter") or streams.get("both") or dec_raw
            sl_p = entry_price * sl_mult
            tp_p = entry_price * tp_mult if tp_mult else None
            dec_smooth = _smooth_stream(dec_raw, mode, sl_p, tp_p)
            v135 = _evaluate_call(strat, tp_mult, sl_mult, horizon_min,
                                  entry_price, entry_ts, dec_smooth, exec_str)
            # v136
            v136 = _evaluate_realistic(strat, tp_mult, sl_mult, horizon_min,
                                       entry_price, entry_ts, jp, ds,
                                       src, poll, mode)
            real_pct = float(r["pnl_pct"]) * 100 if r.get("pnl_pct") is not None else 0
            v135_p.append(v135["pnl_pct"] * 100)
            v136_p.append(v136["pnl_pct"] * 100)
            real_p.append(real_pct)
            v135_d.append(v135["pnl_pct"] * 100 - real_pct)
            v136_d.append(v136["pnl_pct"] * 100 - real_pct)
            if v135["status"] == r["status"]:
                sm_135 += 1
            if v136["status"] == r["status"] or \
               (v136["status"] == "trail_crash" and r["status"] == "trail_stop"):
                sm_136 += 1

        n = len(v135_d)
        if not n:
            continue
        ra = statistics.mean(real_p)
        v135_a = statistics.mean(v135_p)
        v136_a = statistics.mean(v136_p)
        v135_dm = statistics.mean(v135_d)
        v136_dm = statistics.mean(v136_d)
        v135_mae = statistics.mean(abs(x) for x in v135_d)
        v136_mae = statistics.mean(abs(x) for x in v136_d)
        print(f"{strat:<22}{n:>4}{ra:>+9.2f}%{v135_a:>+9.2f}%{v135_dm:>+8.2f}%"
              f"{v135_mae:>9.2f}%{v136_a:>+9.2f}%{v136_dm:>+8.2f}%{v136_mae:>9.2f}%"
              f"{sm_135:>5}/{n:<3}{sm_136:>5}/{n:<3}")


if __name__ == "__main__":
    main()
