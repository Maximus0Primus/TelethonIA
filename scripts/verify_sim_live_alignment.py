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
from paper_trader import _strategy_orchestration, _decision_price, _evaluate_trade_exit

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

SINCE = "2026-04-13 07:00:00+00:00"  # post-v131


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
    """Replay a trade through historical ticks using production orchestration config."""
    strategy = trade["strategy"]
    orch = _strategy_orchestration(strategy, live_cfg)
    polling_sec = int(orch.get("polling_sec", 30))
    source = orch.get("price_source", "jupiter")

    ds_ticks = [(t["fetched_at"], float(t["price_usd"])) for t in ticks if t["source"] in ("fast","full","live") and float(t["price_usd"]) > 0]
    jp_ticks = [(t["fetched_at"], float(t["price_usd"])) for t in ticks if t["source"] == "jupiter" and float(t["price_usd"]) > 0]
    ds_ticks.sort(); jp_ticks.sort()

    # Pick decision stream per source
    if source == "ds":
        dec_stream = ds_ticks
    elif source == "hybrid":
        dec_stream = ds_ticks
    else:
        dec_stream = jp_ticks

    # Subsample at polling_sec
    sampled = []
    last_t = None
    for t, p in dec_stream:
        if t < trade["created_at"]:
            continue
        tt = datetime.fromisoformat(t.replace("Z", "+00:00"))
        if last_t is None or (tt - last_t).total_seconds() >= polling_sec:
            sampled.append((t, p))
            last_t = tt

    if not sampled:
        return None, "no_ticks"

    # Jupiter-sorted for exit nearest-after
    jp_sorted = sorted(jp_ticks)
    entry = float(trade["entry_price"])

    # Mutable fake for _evaluate_trade_exit
    fake = dict(trade)
    fake["id"] = str(trade["id"])

    def nearest_jp(ts):
        for tt, pp in jp_sorted:
            if tt >= ts:
                return pp
        return jp_sorted[-1][1] if jp_sorted else None

    for t, dec_price in sampled:
        tick_time = datetime.fromisoformat(t.replace("Z", "+00:00"))
        exec_price = nearest_jp(t) if source in ("ds","hybrid") else dec_price
        if exec_price is None:
            exec_price = dec_price

        ev = _evaluate_trade_exit(fake, exec_price, tick_time, sell_slip_factor=1.0,
                                   sell_fee_bps=0, decision_price=dec_price)
        if ev is None:
            continue
        if ev.get("high_price_seen"):
            new_high = ev["high_price_seen"]
            if new_high > float(fake.get("high_price_seen") or 0):
                fake["high_price_seen"] = new_high
        if "status" in ev and ev["status"]:
            return ev["pnl_pct"], ev["status"]

    # Fallback: timeout at last tick
    _, last_p = sampled[-1]
    pnl = (last_p / entry) - 1
    return round(pnl, 4), "timeout_eod"


def main():
    print("Loading live trades post-v131...")
    live_trades = fetch_all("paper_trades", gte_created_at=SINCE)
    live_trades = [t for t in live_trades if t.get("tx_signature") and t.get("status") in ("sl_hit","trail_stop","tp_hit","timeout")]
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
        print(f"{tr['symbol']:<14}{tr['strategy']:<25}{pnl_live:>9.2f}%{sim_pnl_pct:>9.2f}%{diff_pp:>11.2f}pp{status_live:>14}{sim_status:>14}")

    if diffs:
        print("-"*100)
        avg_diff = sum(diffs)/len(diffs)
        max_diff = max(abs(d) for d in diffs)
        print(f"\nAligned on {len(diffs)} trades. Avg diff = {avg_diff:+.2f}pp | Max |diff| = {max_diff:.2f}pp")
        if max_diff < 3:
            print("✅ ALIGNMENT VERIFIED — all sim vs live diffs under 3pp (within noise)")
        elif max_diff < 10:
            print("🟡 PARTIAL ALIGNMENT — some diffs up to 10pp (tick sampling gaps likely)")
        else:
            print("🔴 MISALIGNMENT DETECTED — diffs >10pp, investigate orchestration logic")


if __name__ == "__main__":
    main()
