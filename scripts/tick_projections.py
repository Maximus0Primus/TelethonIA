"""Tick sim projections: DTRAIL5_ACT10_SL40 with mcap filters."""
import sys; sys.path.insert(0, "/opt/TelethonIA/scraper")
from sim import (_fetch_tick_trades, _fetch_ticks_for_tokens,
                 _filter_ticks_by_source, _replay_with_intervals,
                 simulate_bankroll, monte_carlo)
import statistics
from datetime import datetime, timezone, timedelta
from collections import defaultdict

trades = _fetch_tick_trades("2026-04-06")
eligible = [t for t in trades if t.get("token_address")]
token_ranges = {}
for t in eligible:
    addr = t["token_address"]
    entry_dt = datetime.fromisoformat(t["created_at"].replace("Z", "+00:00"))
    end_dt = entry_dt + timedelta(minutes=int(t.get("horizon_minutes", 120)) + 30)
    end_iso = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    if addr not in token_ranges:
        token_ranges[addr] = (t["created_at"], end_iso)
    else:
        s, e = token_ranges[addr]
        token_ranges[addr] = (min(s, t["created_at"]), max(e, end_iso))
ticks_by_token = _fetch_ticks_for_tokens(token_ranges)

for label, mcap_min in [("ALL", 0), ("MCAP>25K", 25000), ("MCAP>50K", 50000), ("MCAP>100K", 100000)]:
    print(f"\n--- Processing {label} (mcap_min={mcap_min}) ---")
    pnl_list, tr_list = [], []
    by_day = defaultdict(list)
    n_pass_mcap = 0
    n_has_ticks = 0
    n_has_jup = 0
    EXCLUDE_TOKENS = {"$PEEPA", "$SPERM"}
    for t in eligible:
        if t.get("symbol") in EXCLUDE_TOKENS:
            continue
        mcap_raw = t.get("entry_mcap")
        mcap = float(mcap_raw) if mcap_raw is not None else None
        if mcap is None and mcap_min > 0:
            continue
        if (mcap or 0) < mcap_min:
            continue
        n_pass_mcap += 1
        addr = t["token_address"]
        raw = ticks_by_token.get(addr)
        if not raw:
            continue
        n_has_ticks += 1
        tks = _filter_ticks_by_source(raw, "jupiter")
        if not tks:
            continue
        n_has_jup += 1
        ep = float(t["entry_price"])
        fake = {
            "id": t["id"], "entry_price": ep, "sl_price": ep * 0.60, "tp_price": None,
            "position_usd": float(t.get("position_usd") or 10),
            "strategy": "DTRAIL5_ACT10_SL40", "tranche_label": "main", "horizon_minutes": 120,
            "created_at": t["created_at"], "high_price_seen": ep,
            "rt_liquidity_usd": t.get("rt_liquidity_usd"),
            "dex_spot_price_at_entry": float(t.get("dex_spot_price_at_entry") or 0),
        }
        sim = _replay_with_intervals(fake, tks, 60, 120, 180)
        if sim:
            pnl_list.append(sim["pnl_pct"])
            pos = float(t.get("position_usd") or 10)
            by_day[t["created_at"][:10]].append(sim["pnl_pct"] * pos)
            tr_list.append({"pnl_pct": sim["pnl_pct"], "token_address": addr, "created_at": t["created_at"]})

    if not pnl_list:
        print(f"  {label}: pass_mcap={n_pass_mcap}, has_ticks={n_has_ticks}, has_jup={n_has_jup}, sim={len(pnl_list)}")
        continue
    n = len(pnl_list)
    wr = sum(1 for p in pnl_list if p > 0) / n * 100
    avg_pnl = statistics.mean(pnl_list) * 100
    n_days = len(by_day)
    daily_pnls = [sum(v) for v in by_day.values()]
    avg_daily = statistics.mean(daily_pnls)
    br = simulate_bankroll(sorted(tr_list, key=lambda x: x["created_at"]))
    mc = monte_carlo(pnl_list, 1000, min(200, n))

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Trades: {n} over {n_days} days ({n / n_days:.1f}/day)")
    print(f"  WR: {wr:.0f}%  AvgPnL: {avg_pnl:+.1f}%")
    print(f"  Daily PnL:")
    for d in sorted(by_day):
        dpnl = sum(by_day[d])
        nt = len(by_day[d])
        print(f"    {d}: ${dpnl:+8.0f}  ({nt} trades)")
    print(f"  Avg $/day: ${avg_daily:.0f}  -> $/month: ${avg_daily * 30:.0f}")
    bfin = br["final_bankroll"]
    bdd = br["max_dd_pct"]
    print(f"  Bankroll: ${bfin:.0f} (start $500, DD {bdd:.0f}%)")
    if mc:
        print(f"  Monte Carlo: median ${mc['median']:.0f}  P5 ${mc['p5']:.0f}  P95 ${mc['p95']:.0f}")
