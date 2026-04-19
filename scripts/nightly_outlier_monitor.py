"""Nightly monitor — alert on paper↔live outliers where sync=True.

Why the sync=True filter: v142E (entry) + v143.5 (exit) should close the gap
for every new trade. An outlier with sync=False is historic or live-swap-fail
and not actionable. An outlier with sync=True is a NEW logic bug — that's what
we want to catch early.

Exits with code 2 when any sync=True outlier found (so the GH Actions job fails
and alerts Telegram). Also prints summary stats from diverge_report logic.
"""
import os, sys, statistics as st, json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

# Last 48h rolling window so we always see enough N without old sync=False noise
SINCE = os.environ.get("OUTLIER_SINCE",
    (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat())
THRESHOLD_PP = float(os.environ.get("OUTLIER_THRESHOLD_PP", "10"))


def fetch_all(tbl, sel, **f):
    out, step, off = [], 1000, 0
    while True:
        q = sb.table(tbl).select(sel)
        for k, v in f.items():
            if k.startswith("gte_"): q = q.gte(k[4:], v)
            elif k.startswith("eq_"): q = q.eq(k[3:], v)
        r = q.range(off, off+step-1).execute()
        if not r.data: break
        out.extend(r.data)
        if len(r.data) < step: break
        off += step
    return out


def main():
    print(f"Nightly outlier monitor — window since {SINCE}")
    rows = fetch_all("paper_trades",
        "id,symbol,strategy,status,source,pnl_pct,entry_price,exit_price,entry_source,"
        "created_at,exit_at,rt_is_pump_fun,token_address,paper_sim_pnl_pct",
        gte_created_at=SINCE)
    closed = [r for r in rows if r.get("status") in ("sl_hit","trail_stop","tp_hit","timeout","be_stop")
              and not str(r.get("strategy","")).startswith("DTRAIL")]
    print(f"  closed trades: {len(closed)}")

    pairs = defaultdict(dict)
    for r in closed:
        src = "live" if r.get("source") == "rt_live" else "paper"
        pairs[(r["token_address"], r["strategy"])][src] = r

    matched = [(lv, pp) for v in pairs.values() for lv, pp in [(v.get("live"), v.get("paper"))] if lv and pp]
    print(f"  matched pairs: {len(matched)}")

    sim_diffs = []
    for lv, _ in matched:
        if lv.get("paper_sim_pnl_pct") is not None:
            d = (float(lv.get("pnl_pct") or 0) - float(lv["paper_sim_pnl_pct"])) * 100
            sim_diffs.append(d)
    if sim_diffs:
        print(f"  sim-live diff median: {st.median(sim_diffs):+.2f}pp  "
              f"mean: {st.mean(sim_diffs):+.2f}pp  max|{max(abs(d) for d in sim_diffs):.2f}|pp")

    outliers_synced = []
    outliers_unsynced = 0
    for lv, pp in matched:
        dpp = (float(lv.get("pnl_pct") or 0) - float(pp.get("pnl_pct") or 0)) * 100
        if abs(dpp) <= THRESHOLD_PP:
            continue
        synced = (lv.get("entry_source") == "live_sync") or (pp.get("entry_source") == "live_sync")
        if synced:
            outliers_synced.append({
                "symbol": lv["symbol"], "strategy": lv["strategy"],
                "pnl_live": round(float(lv.get("pnl_pct") or 0)*100, 2),
                "pnl_paper": round(float(pp.get("pnl_pct") or 0)*100, 2),
                "delta_pp": round(dpp, 2),
                "status_live": lv["status"], "status_paper": pp["status"],
                "exit_at": lv.get("exit_at"),
                "is_pump": lv.get("rt_is_pump_fun"),
            })
        else:
            outliers_unsynced += 1

    print(f"\n  outliers |L-P|>{THRESHOLD_PP}pp : {len(outliers_synced)+outliers_unsynced} total")
    print(f"    sync=False (expected, historic): {outliers_unsynced}")
    print(f"    sync=True  (BUG SIGNAL):         {len(outliers_synced)}")

    # Emit JSON for CI to parse + upload
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "nightly_outliers.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "window_since": SINCE,
            "threshold_pp": THRESHOLD_PP,
            "n_pairs": len(matched),
            "sim_live_median_pp": st.median(sim_diffs) if sim_diffs else None,
            "outliers_synced": outliers_synced,
            "outliers_unsynced_count": outliers_unsynced,
        }, f, indent=2, default=str)
    print(f"  saved -> {out_path}")

    # GH Actions outputs
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as gh:
            gh.write(f"sync_true_count={len(outliers_synced)}\n")
            gh.write(f"sync_false_count={outliers_unsynced}\n")
            gh.write(f"pairs={len(matched)}\n")
            if outliers_synced:
                first = outliers_synced[0]
                gh.write(f"first_symbol={first['symbol']}\n")
                gh.write(f"first_strategy={first['strategy']}\n")
                gh.write(f"first_delta={first['delta_pp']}\n")

    for o in outliers_synced:
        print(f"    [SYNC=TRUE] {o['symbol']} {o['strategy']} L={o['pnl_live']:+.1f}% "
              f"P={o['pnl_paper']:+.1f}% Δ={o['delta_pp']:+.1f}pp "
              f"statL={o['status_live']} statP={o['status_paper']}")

    if outliers_synced:
        print(f"\n::error::Found {len(outliers_synced)} sync=True outlier(s) — post-v143.5 logic bug signal")
        sys.exit(2)

    print("\nNo sync=True outliers. Alignment holding.")


if __name__ == "__main__":
    main()
