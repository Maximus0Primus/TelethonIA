"""Detailed dual-wallet analysis for top DIP_BUY vs DTRAIL strategies."""

import statistics
import sys
from datetime import datetime
from collections import defaultdict

import sim as _sim_mod
from sim import (
    SUPABASE_URL, SB_HEADERS, SCRAPER_DIR,
    START_BANKROLL, KELLY_FRAC, MAX_POS, MIN_TRADES, MAX_WINDOW_MIN,
    KOL_BLACKLIST, TOKEN_BLACKLIST, KOL_WHITELIST, CACHE_DIR,
    build_strategy_grid, simulate_bankroll, simulate_dual_wallet,
    fetch_paper_trades, dedup_first_call,
)
_load_cache = _sim_mod._load_cache
from sim_engines import simulate, resample_to_live_checks


def main():
    grid = build_strategy_grid()
    TARGETS = [
        "DIP30_B5_P1T5A20S70_P2T5A10S70_240m",  # Best DIP
        "DTRAIL3_ACT5_SL60_120m",                 # Best DTRAIL
        "DTRAIL10_ACT15_SL70_120m",               # Current live
    ]
    target_cfgs = {c["name"]: c for c in grid if c["name"] in TARGETS}
    print(f"Strategies: {list(target_cfgs.keys())}\n")

    # Fetch trades using sim.py's own function
    trades_raw = fetch_paper_trades("2026-03-01")
    unique = dedup_first_call(trades_raw)
    print(f"Unique tokens (deduped): {len(unique)}")

    # Load pair cache
    _load_pair_cache()

    # Build entries with OHLCV
    trade_entries = []
    for t in unique:
        dt = datetime.fromisoformat(t["created_at"].replace("Z", "+00:00"))
        entry_price = float(t.get("entry_price") or 0)
        if entry_price <= 0:
            continue
        raw_age = t.get("rt_token_age_hours")
        age_h = float(raw_age) if raw_age is not None else None
        if age_h is None or age_h > 12:
            continue
        kol = (t.get("kol_group") or "unknown").lower()
        if kol in KOL_BLACKLIST or t["token_address"] in TOKEN_BLACKLIST:
            continue

        pair = t.get("pair_address") or ""
        pool_key = pair if pair else t["token_address"][:12]
        candles = _load_cache(pool_key, int(dt.timestamp()), MAX_WINDOW_MIN)
        if not candles or len(candles) < 3:
            continue

        trade_entries.append({
            "token_address": t["token_address"],
            "created_at": t["created_at"],
            "kol_group": kol,
            "entry_price": entry_price,
            "candles": candles,
            "context": {
                "mcap": float(t.get("entry_mcap") or 0),
                "liq": float(t.get("rt_liquidity_usd") or 0),
                "vol24": float(t.get("rt_volume_24h") or 0),
                "age_h": float(t.get("rt_token_age_hours") or 0),
                "is_pump": int(t.get("rt_is_pump_fun") or 0),
                "n_kols": int(t.get("n_kol_confirmations") or 1),
            },
        })

    print(f"Trade entries with OHLCV: {len(trade_entries)}")
    if not trade_entries:
        print("No entries!")
        return

    dates = [te["created_at"][:10] for te in trade_entries]
    min_date, max_date = min(dates), max(dates)
    n_days = (datetime.strptime(max_date, "%Y-%m-%d") -
              datetime.strptime(min_date, "%Y-%m-%d")).days + 1
    print(f"Period: {min_date} -> {max_date} ({n_days} days)\n")

    # Analyze each strategy
    for name in TARGETS:
        cfg = target_cfgs.get(name)
        if not cfg:
            continue

        print(f"\n{'=' * 85}")
        print(f"  {name}")
        print(f"{'=' * 85}")

        pnls = []
        daily_pnls = defaultdict(list)

        for te in trade_entries:
            candles = resample_to_live_checks(te["candles"], 3)
            res = simulate(candles, te["entry_price"], cfg, context=te.get("context"))
            pnl = res["pnl_pct"]
            pnls.append(pnl)
            day = te["created_at"][:10]
            daily_pnls[day].append(pnl)

        n = len(pnls)
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / n * 100
        avg_pnl = statistics.mean(pnls) * 100
        med_pnl = statistics.median(pnls) * 100
        std_pnl = statistics.stdev(pnls) * 100 if n > 1 else 0

        # Bankroll simulation
        sorted_te = sorted(trade_entries, key=lambda x: x["created_at"])
        bankroll = START_BANKROLL
        peak = bankroll
        max_dd = 0
        seen_tk = {}
        n_actual = 0
        for te in sorted_te:
            token = te["token_address"]
            day = te["created_at"][:10]
            last_day = seen_tk.get(token)
            if last_day:
                try:
                    if (datetime.strptime(day, "%Y-%m-%d") -
                            datetime.strptime(last_day, "%Y-%m-%d")).days < 1:
                        continue
                except Exception:
                    pass
            seen_tk[token] = day
            candles = resample_to_live_checks(te["candles"], 3)
            res = simulate(candles, te["entry_price"], cfg, context=te.get("context"))
            pos = min(bankroll * KELLY_FRAC, MAX_POS)
            if pos < 1:
                continue
            bankroll += pos * res["pnl_pct"]
            n_actual += 1
            if bankroll > peak:
                peak = bankroll
            dd = (peak - bankroll) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        total_profit = bankroll - START_BANKROLL
        print(f"\n  OVERVIEW")
        print(f"  {'Starting capital:':<25s} ${START_BANKROLL:,.0f}")
        print(f"  {'Final bankroll:':<25s} ${bankroll:,.0f}")
        print(f"  {'Total profit:':<25s} ${total_profit:+,.0f}")
        print(f"  {'Period:':<25s} {n_days} days ({min_date} -> {max_date})")
        print(f"  {'Trades (unique tokens):':<25s} {n}")
        print(f"  {'Trades (bankroll dedup):':<25s} {n_actual}")
        print(f"  {'Avg trades/day:':<25s} {n / n_days:.1f}")
        print(f"  {'Avg profit/day:':<25s} ${total_profit / n_days:+,.1f}")
        print(f"  {'Win rate:':<25s} {wr:.1f}%")
        print(f"  {'Avg PnL/trade:':<25s} {avg_pnl:+.1f}%")
        print(f"  {'Median PnL/trade:':<25s} {med_pnl:+.1f}%")
        print(f"  {'Std dev PnL:':<25s} {std_pnl:.1f}%")
        print(f"  {'Sharpe ratio:':<25s} {avg_pnl / std_pnl:.2f}" if std_pnl > 0 else "")
        print(f"  {'Max drawdown:':<25s} {max_dd * 100:.1f}%")
        print(f"  {'Kelly fraction:':<25s} {KELLY_FRAC:.1%}")
        print(f"  {'Max position:':<25s} ${MAX_POS:.0f}")

        # Daily breakdown
        print(f"\n  DAILY BREAKDOWN")
        print(f"  {'Date':<12s} {'N':>4s} {'WR%':>5s} {'AvgPnL%':>8s} {'DayP&L$':>9s} {'Bankroll$':>10s}")
        print(f"  {'-' * 52}")
        all_days = sorted(daily_pnls.keys())
        running_br = START_BANKROLL
        daily_profits = []
        for day in all_days:
            dp = daily_pnls[day]
            day_n = len(dp)
            day_wr = sum(1 for p in dp if p > 0) / day_n * 100
            day_avg = statistics.mean(dp) * 100
            day_profit = sum(min(running_br * KELLY_FRAC, MAX_POS) * p for p in dp)
            running_br += day_profit
            daily_profits.append(day_profit)
            print(f"  {day:<12s} {day_n:4d} {day_wr:4.0f}% {day_avg:+7.1f}% "
                  f"${day_profit:+8.0f} ${running_br:9.0f}")

        if len(daily_profits) > 1:
            print(f"\n  DAILY STATS")
            print(f"  {'Mean profit/day:':<25s} ${statistics.mean(daily_profits):+,.0f}")
            print(f"  {'Median profit/day:':<25s} ${statistics.median(daily_profits):+,.0f}")
            print(f"  {'Std dev/day:':<25s} ${statistics.stdev(daily_profits):,.0f}")
            print(f"  {'Best day:':<25s} ${max(daily_profits):+,.0f}")
            print(f"  {'Worst day:':<25s} ${min(daily_profits):+,.0f}")
            wd = sum(1 for p in daily_profits if p > 0)
            print(f"  {'Winning days:':<25s} {wd}/{len(daily_profits)} ({wd / len(daily_profits) * 100:.0f}%)")

        # Dual-wallet delta=0
        print(f"\n  DUAL-WALLET (2 wallets simultaneous)")
        dw_results = []
        for te in trade_entries:
            raw_c = te["candles"]
            sim_c = resample_to_live_checks(raw_c, 3)
            pos = min(START_BANKROLL * KELLY_FRAC, MAX_POS)
            dw = simulate_dual_wallet(sim_c, raw_c, te["entry_price"], cfg,
                                      context=te.get("context"), delta_min=0,
                                      position_usd=pos)
            if dw:
                dw_results.append(dw)

        if dw_results:
            avg_single = statistics.mean([d["single"]["pnl_pct"] for d in dw_results]) * 100
            avg_dual = statistics.mean([d["wa"]["pnl_pct"] for d in dw_results]) * 100
            avg_slip = statistics.mean([d["wa_slip"] for d in dw_results]) * 100

            # 2x bankroll sim
            wa_trades = [{"pnl_pct": d["wa"]["pnl_pct"],
                          "token_address": trade_entries[i]["token_address"],
                          "created_at": trade_entries[i]["created_at"]}
                         for i, d in enumerate(dw_results)]
            br_dual = simulate_bankroll(sorted(wa_trades, key=lambda x: x["created_at"]))
            dual_total = br_dual["final_bankroll"] * 2
            dual_profit = dual_total - 1000

            print(f"  {'1 wallet ($500 start):':<30s} ${bankroll:,.0f}  (profit ${total_profit:+,.0f})")
            print(f"  {'2 wallets ($1000 start):':<30s} ${dual_total:,.0f}  (profit ${dual_profit:+,.0f})")
            print(f"  {'Extra slippage/trade:':<30s} {avg_single - avg_dual:.2f}%")
            print(f"  {'Avg slippage (dual):':<30s} {avg_slip:.2f}%")
            print(f"  {'Scaling efficiency:':<30s} {dual_profit / (total_profit * 2) * 100:.1f}% of perfect 2x")


def bankroll_curve():
    """Trade-by-trade bankroll progression — how fast to Kelly cap."""
    grid = build_strategy_grid()
    TARGETS = {
        "DIP30_B5_P1T5A20S70_P2T5A10S70_240m": "DIP30 (best)",
        "DTRAIL3_ACT5_SL60_120m": "DTRAIL3 (best)",
        "DTRAIL10_ACT15_SL70_120m": "DTRAIL10 (live)",
    }
    cfgs = {n: c for c in grid for n in TARGETS if c["name"] == n}

    trades_raw = fetch_paper_trades("2026-03-01")
    unique = dedup_first_call(trades_raw)

    entries = []
    for t in unique:
        dt = datetime.fromisoformat(t["created_at"].replace("Z", "+00:00"))
        ep = float(t.get("entry_price") or 0)
        if ep <= 0:
            continue
        raw_age = t.get("rt_token_age_hours")
        age_h = float(raw_age) if raw_age is not None else None
        if age_h is None or age_h > 12:
            continue
        kol = (t.get("kol_group") or "unknown").lower()
        if kol in KOL_BLACKLIST or t["token_address"] in TOKEN_BLACKLIST:
            continue
        pair = t.get("pair_address") or ""
        pool_key = pair if pair else t["token_address"][:12]
        candles = _load_cache(pool_key, int(dt.timestamp()), MAX_WINDOW_MIN)
        if not candles or len(candles) < 3:
            continue
        entries.append({
            "token_address": t["token_address"],
            "created_at": t["created_at"],
            "entry_price": ep,
            "candles": candles,
            "context": {
                "mcap": float(t.get("entry_mcap") or 0),
                "liq": float(t.get("rt_liquidity_usd") or 0),
            },
        })

    entries.sort(key=lambda x: x["created_at"])
    print(f"Trades: {len(entries)}, {entries[0]['created_at'][:10]} -> {entries[-1]['created_at'][:10]}")
    print(f"Kelly cap ($120/trade) hit when bankroll reaches ${MAX_POS / KELLY_FRAC:,.0f}\n")

    CAP_BR = MAX_POS / KELLY_FRAC

    for name, label in TARGETS.items():
        cfg = cfgs[name]
        br = START_BANKROLL
        peak = br
        seen = {}
        tn = 0
        cap_t = None
        cap_day = None

        print(f"{'=' * 85}")
        print(f"  {label} ({name})")
        print(f"{'=' * 85}")
        print(f"{'#':>4s} {'Date':>12s} {'PnL%':>7s} {'Pos$':>6s} {'P&L$':>7s} {'BR$':>8s} {'DD%':>5s}  Note")
        print(f"{'-' * 70}")

        for te in entries:
            tok = te["token_address"]
            day = te["created_at"][:10]
            last = seen.get(tok)
            if last:
                try:
                    if (datetime.strptime(day, "%Y-%m-%d") -
                            datetime.strptime(last, "%Y-%m-%d")).days < 1:
                        continue
                except Exception:
                    pass
            seen[tok] = day

            c = resample_to_live_checks(te["candles"], 3)
            res = simulate(c, te["entry_price"], cfg, context=te.get("context"))

            pos = min(br * KELLY_FRAC, MAX_POS)
            if pos < 1:
                continue

            tn += 1
            pnl = pos * res["pnl_pct"]
            br += pnl
            if br > peak:
                peak = br
            dd = (peak - br) / peak * 100 if peak > 0 else 0

            note = ""
            if cap_t is None and br >= CAP_BR:
                cap_t = tn
                cap_day = day
                note = "<-- KELLY CAP $120"

            # Show first 15, every 20th, cap trade, and last
            show = tn <= 15 or tn % 20 == 0 or note
            if show:
                capped = "*" if pos >= MAX_POS else " "
                print(f"{tn:4d} {day:>12s} {res['pnl_pct'] * 100:+6.1f}% "
                      f"${pos:5.0f}{capped} {pnl:+6.0f}$ {br:7.0f}$ {dd:4.1f}%  {note}")

        print(f"{'-' * 70}")
        print(f"  Final: ${br:,.0f} | {tn} trades over "
              f"{entries[0]['created_at'][:10]} -> {entries[-1]['created_at'][:10]}")
        if cap_t:
            print(f"  Kelly cap hit at trade #{cap_t} ({cap_day})")
            print(f"  Trades before cap: {cap_t} | Trades after cap (at $120): {tn - cap_t}")
        else:
            print(f"  Kelly cap NEVER reached (peak ${peak:,.0f})")
        print()


if __name__ == "__main__":
    import sys
    if "--curve" in sys.argv:
        bankroll_curve()
    else:
        main()
