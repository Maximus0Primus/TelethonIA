"""
Preuve d'alignement : rejoue les trades live réels dans le sim avec --from-live-config,
et compare le PnL simulé au PnL réel on-chain.
"""
import os
import sys
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
import paper_trader as _pt
from paper_trader import (
    _strategy_orchestration, _decision_price, _evaluate_trade_exit,
    _jupiter_prices_cache, _dex_prices_cache, _ema_state, _smooth_state,
)

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

SINCE = os.environ.get("ALIGN_SINCE", "2026-04-15 00:00:00+00:00")  # post-v142E shadow-sync


def fetch_all(table, **f):
    out, step, offset = [], 1000, 0
    while True:
        q = sb.table(table).select("*")
        for k, v in f.items():
            if k.startswith("gte_"): q = q.gte(k[4:], v)
            elif k.startswith("eq_"): q = q.eq(k[3:], v)
            elif k == "in_token": q = q.in_("token_address", v)
        q = q.range(offset, offset+step-1).order("fetched_at" if "tick" in table else "created_at")
        r = q.execute()
        if not r.data: break
        out.extend(r.data)
        if len(r.data) < step: break
        offset += step
    return out


def fetch_live_cfg():
    r = sb.table("scoring_config").select("rt_trade_config").eq("id", 1).execute()
    cfg = r.data[0]["rt_trade_config"]
    if isinstance(cfg, str):
        import json; cfg = json.loads(cfg)
    return cfg


def replay_trade(trade, ticks, live_cfg):
    """Replay a trade through historical ticks using production orchestration config.

    v143: feeds the real `_decision_price` (all smoothing modes supported:
    median_3/5, hysteresis, winsor_p95, ema_fast/slow, twin_confirm, vwap_5min,
    jp_sampled_*, dual_confirm, confirm). Ticks are merged chronologically and
    populate the per-addr Jupiter/DS caches exactly as the live scraper does;
    eval fires at `polling_sec` cadence and uses the smoothed decision price.
    """
    strategy = trade["strategy"]
    orch = _strategy_orchestration(strategy, live_cfg)
    polling_sec = int(orch.get("polling_sec", 30))
    addr = trade["token_address"]

    # Merge ticks chronologically with source tag
    merged = []
    for t in ticks:
        p = float(t.get("price_usd") or 0)
        if p <= 0:
            continue
        ts = t["fetched_at"]
        if ts < trade["created_at"]:
            continue
        src = "jp" if t["source"] == "jupiter" else ("ds" if t["source"] in ("fast","full","live") else None)
        if src:
            merged.append((ts, src, p))
    merged.sort(key=lambda x: x[0])
    if not merged:
        return None, "no_ticks"

    # Isolate per-trade smoother state (caches are per-addr, must clear entry)
    trade_id = f"replay_{trade['id']}"
    _ema_state.pop(trade_id, None)
    _smooth_state.pop(trade_id, None)
    _jupiter_prices_cache.pop(addr, None)
    _dex_prices_cache.pop(addr, None)

    fake = dict(trade)
    fake["id"] = trade_id

    last_eval_t = None
    entry = float(trade["entry_price"])

    for ts, src, p in merged:
        # Update the cache that the live scraper would have populated
        if src == "jp":
            _jupiter_prices_cache[addr] = p
        else:
            _dex_prices_cache[addr] = p

        tt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if last_eval_t is not None and (tt - last_eval_t).total_seconds() < polling_sec:
            continue  # cache updated, but don't eval yet

        dec_price, exit_ref = _decision_price(addr, strategy, trade_id, orch, trade=fake)
        if dec_price is None:
            continue  # waiting on the required source
        if exit_ref is None:
            exit_ref = dec_price

        last_eval_t = tt
        ev = _evaluate_trade_exit(fake, exit_ref, tt, sell_slip_factor=1.0,
                                   sell_fee_bps=0, decision_price=dec_price)
        if ev is None:
            continue
        if ev.get("high_price_seen"):
            new_high = ev["high_price_seen"]
            if new_high > float(fake.get("high_price_seen") or 0):
                fake["high_price_seen"] = new_high
        if ev.get("status"):
            # Cleanup per-trade state before returning
            _ema_state.pop(trade_id, None)
            _smooth_state.pop(trade_id, None)
            return ev["pnl_pct"], ev["status"]

    # Fallback: timeout at last merged tick
    _ema_state.pop(trade_id, None)
    _smooth_state.pop(trade_id, None)
    _, _, last_p = merged[-1]
    return round((last_p / entry) - 1, 4), "timeout_eod"


def main():
    print("Loading live trades post-v131...")
    live_trades = fetch_all("paper_trades", gte_created_at=SINCE)
    live_trades = [t for t in live_trades if t.get("source") == "rt_live" and t.get("tx_signature") and t.get("status") in ("sl_hit","trail_stop","tp_hit","timeout","be_stop")]
    print(f"  {len(live_trades)} closed live trades")

    if not live_trades:
        print("No closed live trades to compare.")
        return

    tokens = list({t["token_address"] for t in live_trades})
    print(f"  Loading ticks for {len(tokens)} tokens...")
    ticks = fetch_all("price_ticks", gte_fetched_at=SINCE, in_token=tokens)
    print(f"  {len(ticks)} ticks")

    by_token = defaultdict(list)
    for tk in ticks:
        by_token[tk["token_address"]].append(tk)

    print("\nLoading live orchestration config from DB...")
    rt_cfg = fetch_live_cfg()
    overrides = rt_cfg.get("strategy_overrides", {})
    for s, ov in overrides.items():
        print(f"    {s:30s}  poll={ov.get('polling_sec','-')}s  src={ov.get('price_source','-')}")

    print(f"\n{'Symbol':<14}{'Strategy':<25}{'Live PnL':>10}{'Sim PnL':>10}{'Diff (pp)':>12}{'Status live':>14}{'Status sim':>14}")
    print("-"*100)

    diffs = []
    per_strat = defaultdict(list)
    for tr in live_trades:
        pnl_live = float(tr.get("pnl_pct") or 0) * 100
        status_live = tr.get("status")
        sim_pnl, sim_status = replay_trade(tr, by_token[tr["token_address"]], rt_cfg)
        if sim_pnl is None:
            print(f"{tr['symbol']:<14}{tr['strategy']:<25}{pnl_live:>9.2f}%{'no_data':>10}")
            continue
        sim_pnl_pct = sim_pnl * 100
        diff_pp = sim_pnl_pct - pnl_live
        diffs.append(diff_pp)
        per_strat[tr["strategy"]].append(diff_pp)
        print(f"{tr['symbol']:<14}{tr['strategy']:<25}{pnl_live:>9.2f}%{sim_pnl_pct:>9.2f}%{diff_pp:>11.2f}pp{status_live:>14}{sim_status:>14}")

    if diffs:
        print("-"*100)
        avg_diff = sum(diffs)/len(diffs)
        max_diff = max(abs(d) for d in diffs)
        print(f"\nAligned on {len(diffs)} trades. Avg diff = {avg_diff:+.2f}pp | Max |diff| = {max_diff:.2f}pp")
        if max_diff < 3:
            print("[OK] ALIGNMENT VERIFIED - all sim vs live diffs under 3pp (within noise)")
        elif max_diff < 10:
            print("[WARN] PARTIAL ALIGNMENT - some diffs up to 10pp (tick sampling gaps likely)")
        else:
            print("[FAIL] MISALIGNMENT DETECTED - diffs >10pp, investigate orchestration logic")

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
