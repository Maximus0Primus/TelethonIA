"""
Check-grid polling: pour chaque intervalle de polling (ex: 10s, 30s, 60s, 120s), rejouer les trades
en ne gardant qu'un tick tous les N secondes. Mesurer l'impact sur les stratégies trail.
"""
import os
import sys
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

STRATS = {
    "DTRAIL10_ACT15_SL70": {"trail": 0.10, "act": 0.15, "sl": 0.70},
    "DTRAIL3_ACT5_SL60":  {"trail": 0.03, "act": 0.05, "sl": 0.60},
    "DTRAIL20_ACT30_SL70": {"trail": 0.20, "act": 0.30, "sl": 0.70},  # lazy trail
}
POLL_SECONDS = [10, 20, 30, 60, 120, 300]  # polling intervals to test
SINCE = "2026-04-08T00:00:00+00:00"


def fetch_all(table, **filters):
    out, step, offset = [], 1000, 0
    while True:
        q = sb.table(table).select("*")
        for k, v in filters.items():
            if k.startswith("gte_"):
                q = q.gte(k[4:], v)
            elif k.startswith("eq_"):
                q = q.eq(k[3:], v)
            elif k == "in_token":
                q = q.in_("token_address", v)
        q = q.range(offset, offset + step - 1).order("fetched_at" if "tick" in table else "created_at")
        res = q.execute()
        if not res.data:
            break
        out.extend(res.data)
        if len(res.data) < step:
            break
        offset += step
    return out


def subsample_ticks(ticks, interval_sec):
    """Keep only ticks separated by at least interval_sec."""
    if not ticks:
        return []
    out = [ticks[0]]
    last_t = _parse_ts(ticks[0][0])
    for t, p in ticks[1:]:
        tt = _parse_ts(t)
        if (tt - last_t).total_seconds() >= interval_sec:
            out.append((t, p))
            last_t = tt
    return out


def _parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def run_trail(entry, ticks, cfg):
    if not ticks:
        return 0.0, "no_data"
    high = entry
    act_p = entry * (1 + cfg["act"])
    sl_p = entry * (1 - cfg["sl"])
    for t, price in ticks:
        if price <= 0:
            continue
        high = max(high, price)
        if price <= sl_p:
            return (price / entry) - 1, "sl_hit"
        if high >= act_p:
            trig = high * (1 - cfg["trail"])
            if price <= trig and trig > entry:
                return (price / entry) - 1, "trail_stop"
    last_p = ticks[-1][1]
    return (last_p / entry) - 1, "timeout"


def main():
    print("Loading DTRAIL trades since Apr 8...")
    trades = []
    for strat in STRATS:
        t = fetch_all("paper_trades", gte_created_at=SINCE, eq_is_shadow=False, eq_strategy=strat)
        trades.extend(t)
    print(f"  {len(trades)} trades across {len(STRATS)} strategies")

    tokens = list({t["token_address"] for t in trades if t.get("token_address")})
    print(f"  {len(tokens)} unique tokens")

    print("Loading ticks...")
    ticks = fetch_all("price_ticks", gte_fetched_at=SINCE, in_token=tokens)
    print(f"  {len(ticks)} ticks")

    # Jupiter ticks only for consistency (since live uses Jupiter)
    by_token = defaultdict(list)
    for tk in ticks:
        if tk["source"] == "jupiter":
            by_token[tk["token_address"]].append((tk["fetched_at"], float(tk["price_usd"])))
    for ca in by_token:
        by_token[ca].sort()

    print(f"\n{'Strategy':<25}{'Poll (s)':>10}{'N':>6}{'AvgPnL':>10}{'WR':>8}{'Total$':>10}")
    print("-" * 72)

    for strat_name, cfg in STRATS.items():
        strat_trades = [t for t in trades if t["strategy"] == strat_name]
        for poll_sec in POLL_SECONDS:
            pnls = []
            for trade in strat_trades:
                ca = trade.get("token_address")
                entry = float(trade.get("entry_price") or 0)
                opened = trade["created_at"]
                if entry <= 0 or ca not in by_token:
                    continue
                token_ticks = [(t, p) for t, p in by_token[ca] if t >= opened]
                if len(token_ticks) < 3:
                    continue
                sampled = subsample_ticks(token_ticks, poll_sec)
                if len(sampled) < 2:
                    continue
                pnl, _ = run_trail(entry, sampled, cfg)
                pnls.append(pnl)
            if not pnls:
                continue
            avg = sum(pnls) / len(pnls) * 100
            wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
            tot = sum(p * 10 for p in pnls)
            print(f"{strat_name:<25}{poll_sec:>10}{len(pnls):>6}{avg:>9.2f}%{wr:>7.1f}%{tot:>9.2f}$")
        print()


if __name__ == "__main__":
    main()
