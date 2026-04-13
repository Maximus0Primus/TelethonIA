"""
Backtest hybride: trail trigger basé sur DexScreener ticks, exit price = Jupiter tick nearest after trigger.
Compare 3 modes: DS-only, Jupiter-only, Hybrid (DS-decision + Jupiter-exec).
"""
import os
import sys
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

# Strategy: DTRAIL10_ACT15_SL70
TRAIL_PCT = 0.10
ACT_PCT = 0.15
SL_PCT = 0.70
SINCE = "2026-04-08T00:00:00+00:00"


def fetch_all(table, **filters):
    out = []
    step = 1000
    offset = 0
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


def run_trail(entry_price, ticks_for_decision, ticks_for_exec):
    """
    ticks_for_decision: list of (timestamp, price) for trail trigger detection
    ticks_for_exec: list of (timestamp, price) for exit price lookup (may be same list)
    Returns: (exit_reason, exit_price, pnl_pct, exit_time)
    """
    if not ticks_for_decision:
        return ("no_data", None, 0.0, None)
    high_seen = entry_price
    activation_price = entry_price * (1 + ACT_PCT)
    sl_price = entry_price * (1 - SL_PCT)
    # sort exec ticks by time for nearest-after lookup
    exec_sorted = sorted(ticks_for_exec, key=lambda x: x[0])
    for t, price in ticks_for_decision:
        if price <= 0:
            continue
        high_seen = max(high_seen, price)
        # SL check
        if price <= sl_price:
            exit_p = _nearest_after(exec_sorted, t) or price
            return ("sl_hit", exit_p, (exit_p / entry_price) - 1, t)
        # Trail check
        if high_seen >= activation_price:
            trail_trigger = high_seen * (1 - TRAIL_PCT)
            if price <= trail_trigger and trail_trigger > entry_price:
                exit_p = _nearest_after(exec_sorted, t) or price
                return ("trail_stop", exit_p, (exit_p / entry_price) - 1, t)
    # No exit → timeout at last known price
    last_t, last_p = ticks_for_decision[-1]
    exit_p = _nearest_after(exec_sorted, last_t) or last_p
    return ("timeout", exit_p, (exit_p / entry_price) - 1, last_t)


def _nearest_after(exec_ticks, ts):
    """Find first exec tick with timestamp >= ts."""
    for t, p in exec_ticks:
        if t >= ts and p > 0:
            return p
    # fallback: last tick
    return exec_ticks[-1][1] if exec_ticks else None


def main():
    print("Loading live-eligible trades since Apr 8...")
    trades = fetch_all("paper_trades", gte_created_at=SINCE, eq_is_shadow=False, eq_strategy="DTRAIL10_ACT15_SL70")
    print(f"  {len(trades)} DTRAIL10 trades (non-shadow)")

    tokens = list({t["token_address"] for t in trades if t.get("token_address")})
    print(f"  {len(tokens)} unique tokens")

    print("Loading ticks for these tokens...")
    ticks = fetch_all("price_ticks", gte_fetched_at=SINCE, in_token=tokens)
    print(f"  {len(ticks)} ticks")

    # Index ticks by token and source
    by_token = defaultdict(lambda: {"ds": [], "jupiter": []})
    for tk in ticks:
        ca = tk["token_address"]
        ts = tk["fetched_at"]
        price = float(tk["price_usd"])
        src = tk["source"]
        if src in ("fast", "full", "live"):  # DS-like
            by_token[ca]["ds"].append((ts, price))
        elif src == "jupiter":
            by_token[ca]["jupiter"].append((ts, price))

    results = {"ds_only": [], "jupiter_only": [], "hybrid": [], "paper_booked": []}

    for trade in trades:
        ca = trade.get("token_address")
        entry = float(trade.get("entry_price") or 0)
        opened = trade["created_at"]
        if entry <= 0 or ca not in by_token:
            continue
        ds_ticks = [(t, p) for t, p in by_token[ca]["ds"] if t >= opened]
        jp_ticks = [(t, p) for t, p in by_token[ca]["jupiter"] if t >= opened]
        if len(ds_ticks) < 3 or len(jp_ticks) < 3:
            continue

        # Mode 1: DS-only (decision + exec on DS)
        _, _, pnl_ds, _ = run_trail(entry, ds_ticks, ds_ticks)
        # Mode 2: Jupiter-only
        _, _, pnl_jp, _ = run_trail(entry, jp_ticks, jp_ticks)
        # Mode 3: Hybrid (DS decide, Jupiter exec)
        _, _, pnl_hy, _ = run_trail(entry, ds_ticks, jp_ticks)

        results["ds_only"].append(pnl_ds)
        results["jupiter_only"].append(pnl_jp)
        results["hybrid"].append(pnl_hy)
        if trade.get("pnl_pct") is not None:
            results["paper_booked"].append(float(trade["pnl_pct"]))

    print("\n" + "=" * 70)
    print(f"RESULTS — DTRAIL10_ACT15_SL70 on {len(results['ds_only'])} trades with both tick sources")
    print("=" * 70)
    print(f"{'Mode':<18}{'N':>4}{'Avg PnL':>12}{'WR':>8}{'Sum $':>10}{'Worst':>10}{'Best':>10}")
    for mode in ("ds_only", "jupiter_only", "hybrid", "paper_booked"):
        pnls = results[mode]
        if not pnls:
            print(f"{mode:<18}  (no data)")
            continue
        avg = sum(pnls) / len(pnls) * 100
        wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        total_usd = sum(p * 10 for p in pnls)  # $10/trade
        worst = min(pnls) * 100
        best = max(pnls) * 100
        print(f"{mode:<18}{len(pnls):>4}{avg:>11.2f}%{wr:>7.1f}%{total_usd:>9.2f}$ {worst:>9.1f}% {best:>9.1f}%")

    # Paired delta
    if results["hybrid"] and results["jupiter_only"]:
        deltas = [h - j for h, j in zip(results["hybrid"], results["jupiter_only"])]
        print(f"\nHybrid vs Jupiter paired delta: mean={sum(deltas)/len(deltas)*100:.2f}pp  "
              f"wins={sum(1 for d in deltas if d > 0)}/{len(deltas)}")


if __name__ == "__main__":
    main()
