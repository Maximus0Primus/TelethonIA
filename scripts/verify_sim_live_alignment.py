"""
Preuve d'alignement : rejoue les trades live reels via paper_trades.eval_history
(polls persistes par live_trader + paper_trader depuis v138), et compare le PnL
simule au PnL reel on-chain.

v144.7: switched from price_ticks reconstruction to eval_history replay.
price_ticks logs Jupiter at 3-min batch cadence, which is too coarse vs the
30s live polling. eval_history stores every (decision_price, exec_price) pair
the live bot actually saw, so replay is mathematically identical input —
divergence is now a pure logic check rather than a tick-sampling artifact.
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

    with_history = [t for t in live_trades
                    if isinstance(t.get("eval_history"), list) and len(t["eval_history"]) >= 2]
    print(f"  {len(with_history)} have eval_history (≥2 polls)")
    skipped_no_hist = len(live_trades) - len(with_history)

    if not with_history:
        print("No live trades with eval_history to compare.")
        return

    print(f"\n{'Symbol':<14}{'Strategy':<25}{'Live PnL':>10}{'Sim PnL':>10}"
          f"{'Diff (pp)':>12}{'Status live':>14}{'Status sim':>14}{'N_polls':>10}")
    print("-" * 110)

    diffs = []
    per_strat = defaultdict(list)
    for tr in with_history:
        pnl_live = float(tr.get("pnl_pct") or 0) * 100
        status_live = tr.get("status")
        fake = dict(tr)
        # Reset high_price_seen so replay tracks from entry, not from the persisted peak
        fake["high_price_seen"] = float(tr["entry_price"])
        result = _replay_from_eval_history(fake, tr["eval_history"])
        if result is None:
            continue
        sim_pnl_pct = float(result["pnl_pct"]) * 100
        sim_status = result["exit_reason"]
        diff_pp = sim_pnl_pct - pnl_live
        n_polls = len(tr["eval_history"])
        # eval_history replay never hits insufficient-data since it's the exact
        # poll stream the live bot saw. timeout_eod only surfaces when the live
        # trade closed before any status triggered in the replay — meaning the
        # strategy logic itself missed an exit the live bot took. That IS a bug.
        diffs.append(diff_pp)
        per_strat[tr["strategy"]].append(diff_pp)
        print(f"{tr['symbol']:<14}{tr['strategy']:<25}{pnl_live:>9.2f}%{sim_pnl_pct:>9.2f}%"
              f"{diff_pp:>11.2f}pp{status_live:>14}{sim_status:>14}{n_polls:>10}")

    if diffs:
        print("-" * 110)
        avg_diff = sum(diffs) / len(diffs)
        max_diff = max(abs(d) for d in diffs)
        print(f"\nAligned on {len(diffs)} trades (skipped {skipped_no_hist} without eval_history). "
              f"Avg diff = {avg_diff:+.2f}pp | Max |diff| = {max_diff:.2f}pp")
        if max_diff < 3:
            print("[OK] ALIGNMENT VERIFIED - all sim vs live diffs under 3pp (logic identical)")
        elif max_diff < 10:
            print("[WARN] PARTIAL ALIGNMENT - some diffs up to 10pp (possible logic drift)")
        else:
            print("[FAIL] MISALIGNMENT DETECTED - diffs >10pp, real sim vs live logic bug")

        import statistics as _st
        print("\n=== Per-strategy diff breakdown ===")
        for strat, ds in sorted(per_strat.items(), key=lambda x: -len(x[1])):
            in1 = sum(1 for d in ds if abs(d) <= 1)
            in3 = sum(1 for d in ds if abs(d) <= 3)
            in10 = sum(1 for d in ds if abs(d) <= 10)
            print(f"  {strat}: N={len(ds)}, median={_st.median(ds):+.2f}pp, "
                  f"mean={_st.mean(ds):+.2f}pp, maxabs={max(ds, key=abs):+.2f}pp, "
                  f"within_1pp={in1}, within_3pp={in3}, within_10pp={in10}")


if __name__ == "__main__":
    main()
