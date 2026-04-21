"""
Preuve d'alignement : rejoue les trades live reels via paper_trades.eval_history
(polls persistes par live_trader + paper_trader depuis v138), et compare au
paper_sim_pnl_pct (ce que paper_trader AURAIT book avec le meme input, stocke
par live_trader depuis v141).

v144.7: switched from price_ticks reconstruction to eval_history replay.
price_ticks logs Jupiter at 3-min batch cadence, too coarse vs live 30s polling.
eval_history stores every (decision_price, exec_price) pair the live bot saw,
so replay input is mathematically identical — pure logic check.

v144.8: compare replay vs paper_sim_pnl_pct (not live pnl_pct). live pnl_pct
reflects the REAL Jupiter Ultra fill price which includes positive/negative
slippage the sim cannot model (e.g. CHUCHU: TP=+50%, Ultra fill landed at
+120% on a spike). paper_sim_pnl_pct is what paper_trader would have booked
in pure simulation at the same moment — the apples-to-apples reference for
"is our sim logic correct?".
"""
import os
import sys
from collections import defaultdict
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
from sim import _replay_from_eval_history

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

SINCE = os.environ.get("ALIGN_SINCE", "2026-04-15 00:00:00+00:00")  # post-v142E shadow-sync


def fetch_all(table, **f):
    out, step, offset = [], 1000, 0
    while True:
        q = sb.table(table).select("*")
        for k, v in f.items():
            if k.startswith("gte_"): q = q.gte(k[4:], v)
            elif k.startswith("eq_"): q = q.eq(k[3:], v)
        q = q.range(offset, offset+step-1).order("created_at")
        r = q.execute()
        if not r.data: break
        out.extend(r.data)
        if len(r.data) < step: break
        offset += step
    return out


def main():
    print("Loading closed live trades with eval_history...")
    live_trades = fetch_all("paper_trades", gte_created_at=SINCE)
    live_trades = [t for t in live_trades
                   if t.get("source") == "rt_live"
                   and t.get("tx_signature")
                   and t.get("status") in ("sl_hit", "trail_stop", "tp_hit", "timeout", "be_stop")]
    print(f"  {len(live_trades)} closed live trades in window")

    # v144.11: relax eval_history filter from ≥2 to ≥1 — FAST strategies often
    # exit on the first poll (timeout/SL) and would otherwise be dropped, which
    # is exactly the population where economic drift concentrates. paper_sim_pnl_pct
    # remains the hard gate since it's our apples-to-apples anchor.
    with_history = [t for t in live_trades
                    if isinstance(t.get("eval_history"), list) and len(t["eval_history"]) >= 1
                    and t.get("paper_sim_pnl_pct") is not None]
    print(f"  {len(with_history)} have eval_history (≥1 poll) AND paper_sim_pnl_pct")
    skipped_no_hist = len(live_trades) - len(with_history)

    if not with_history:
        print("No live trades with eval_history + paper_sim_pnl_pct to compare.")
        return

    print(f"\n{'Symbol':<14}{'Strategy':<25}{'PaperSim':>10}{'Replay':>10}"
          f"{'Diff (pp)':>12}{'Live PnL':>10}{'Jup slip':>11}{'Status':>12}{'N':>5}")
    print("-" * 115)

    diffs = []
    per_strat = defaultdict(list)
    per_strat_slip = defaultdict(list)  # v144.11: economic drift (live fill vs paper_sim)
    for tr in with_history:
        paper_sim_pnl = float(tr.get("paper_sim_pnl_pct") or 0) * 100
        pnl_live = float(tr.get("pnl_pct") or 0) * 100
        status_live = tr.get("status")
        fake = dict(tr)
        fake["high_price_seen"] = float(tr["entry_price"])
        result = _replay_from_eval_history(fake, tr["eval_history"])
        if result is None:
            continue
        sim_pnl_pct = float(result["pnl_pct"]) * 100
        sim_status = result["exit_reason"]
        # v144.8: apples-to-apples — replay (pure sim) vs paper_sim_pnl_pct (sim-at-live-time).
        # Both are pure simulation values, their diff measures sim LOGIC drift.
        diff_pp = sim_pnl_pct - paper_sim_pnl
        # Jupiter fill slippage = real live pnl - what pure sim would have booked
        jup_slip_pp = pnl_live - paper_sim_pnl
        n_polls = len(tr["eval_history"])
        diffs.append(diff_pp)
        per_strat[tr["strategy"]].append(diff_pp)
        per_strat_slip[tr["strategy"]].append(jup_slip_pp)
        print(f"{tr['symbol']:<14}{tr['strategy']:<25}{paper_sim_pnl:>9.2f}%{sim_pnl_pct:>9.2f}%"
              f"{diff_pp:>11.2f}pp{pnl_live:>9.2f}%{jup_slip_pp:>10.2f}pp"
              f"{(status_live + '/' + sim_status):>12}{n_polls:>5}")

    if diffs:
        print("-" * 115)
        avg_diff = sum(diffs) / len(diffs)
        max_diff = max(abs(d) for d in diffs)
        print(f"\nReplay vs paper_sim on {len(diffs)} trades (skipped {skipped_no_hist}). "
              f"Avg diff = {avg_diff:+.2f}pp | Max |diff| = {max_diff:.2f}pp")
        print("(Jup slip column = real fill vs pure sim — NOT a gate metric, purely informative)")
        if max_diff < 3:
            print("[OK] SIM LOGIC VERIFIED - replay matches paper_sim within 3pp (logic identical)")
        elif max_diff < 10:
            print("[WARN] PARTIAL ALIGNMENT - some diffs up to 10pp (possible logic drift)")
        else:
            print("[FAIL] LOGIC DRIFT - replay vs paper_sim diffs >10pp, real sim bug")

        import statistics as _st
        print("\n=== Per-strategy LOGIC drift breakdown (replay vs paper_sim) ===")
        for strat, ds in sorted(per_strat.items(), key=lambda x: -len(x[1])):
            in1 = sum(1 for d in ds if abs(d) <= 1)
            in3 = sum(1 for d in ds if abs(d) <= 3)
            in10 = sum(1 for d in ds if abs(d) <= 10)
            print(f"  {strat}: N={len(ds)}, median={_st.median(ds):+.2f}pp, "
                  f"mean={_st.mean(ds):+.2f}pp, maxabs={max(ds, key=abs):+.2f}pp, "
                  f"within_1pp={in1}, within_3pp={in3}, within_10pp={in10}")

        # v144.11: second gate — ECONOMIC drift = real Jupiter fill vs paper_sim.
        # Logic can be perfect (replay == paper_sim) yet a strategy collapses
        # because its exits land in a regime where Jupiter Ultra fills asymmetrically
        # worse than paper's slip model (MEV, routing, thin liquidity). FAST_TP50_SL30
        # with 30min timeout is the classic victim.
        print("\n=== Per-strategy ECONOMIC drift (live fill vs paper_sim) ===")
        collapsed = []
        MIN_N = 5
        MEAN_FAIL = -3.0   # live pnl systematically below paper_sim by >3pp on average
        MEDIAN_FAIL = 5.0  # |median| slip > 5pp
        for strat, ds in sorted(per_strat_slip.items(), key=lambda x: -len(x[1])):
            n = len(ds)
            med = _st.median(ds)
            mean = _st.mean(ds)
            maxabs = max(ds, key=abs)
            flag = ""
            if n >= MIN_N and (mean < MEAN_FAIL or abs(med) > MEDIAN_FAIL):
                flag = "  [COLLAPSE]"
                collapsed.append(strat)
            print(f"  {strat}: N={n}, median={med:+.2f}pp, mean={mean:+.2f}pp, "
                  f"maxabs={maxabs:+.2f}pp{flag}")

        if collapsed:
            print(f"\n[ALERT] ECONOMIC DRIFT — live underperforms paper_sim on {', '.join(collapsed)}")
            print("        Root cause is execution (Jupiter fill vs modeled slip), NOT a loop bug.")
            print(f"        Gate: mean < {MEAN_FAIL}pp OR |median| > {MEDIAN_FAIL}pp (N >= {MIN_N}).")
            sys.exit(2)


if __name__ == "__main__":
    main()
