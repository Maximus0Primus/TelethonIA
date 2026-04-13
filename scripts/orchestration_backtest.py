"""
Backtest complet : pour chaque stratégie x mode d'orchestration DS/Jupiter x polling interval,
mesurer avg PnL et WR. Inspiré de la question : le pre-Apr7 "marchait" par biais de lissage —
peut-on reproduire ce lissage intentionnellement avec DS et Jupiter combinés ?

Modes:
  jupiter_only     : ticks Jupiter seuls (baseline live)
  ds_only          : ticks DexScreener seuls
  hybrid_ds_jup    : décision sur DS, exit_price Jupiter
  avg_both         : prix = moyenne(DS, Jupiter) nearest en temps
  confirm_both     : trigger nécessite DS ET Jupiter sous le seuil
  ema_jupiter_3    : EMA 3-points sur Jupiter (lissage pur)
  ema_jupiter_5    : EMA 5-points sur Jupiter
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
    # DTRAIL family
    "DTRAIL10_ACT15_SL70": {"kind": "dtrail", "trail": 0.10, "act": 0.15, "sl": 0.70, "horizon": 120},
    "DTRAIL3_ACT5_SL60":   {"kind": "dtrail", "trail": 0.03, "act": 0.05, "sl": 0.60, "horizon": 120},
    # FAST family (fixed TP/SL + timeout)
    "FAST45_TP50_SL30":    {"kind": "fixed", "tp": 0.50, "sl": 0.30, "horizon": 45},
    "FAST_TP50_SL30":      {"kind": "fixed", "tp": 0.50, "sl": 0.30, "horizon": 30},
    "FAST_TP70_SL50":      {"kind": "fixed", "tp": 0.70, "sl": 0.50, "horizon": 30},
    "FAST45_TP40_SL30":    {"kind": "fixed", "tp": 0.40, "sl": 0.30, "horizon": 45},
    # SCALP family
    "SCALP_TP15_SL10":     {"kind": "fixed", "tp": 0.15, "sl": 0.10, "horizon": 15},
    "SCALP_TP20_SL10":     {"kind": "fixed", "tp": 0.20, "sl": 0.10, "horizon": 15},
    # Fixed TP/SL, 2h timeout
    "TP50_SL30":           {"kind": "fixed", "tp": 0.50, "sl": 0.30, "horizon": 120},
    "TP100_SL50":          {"kind": "fixed", "tp": 1.00, "sl": 0.50, "horizon": 120},
    "TP70_SL30":           {"kind": "fixed", "tp": 0.70, "sl": 0.30, "horizon": 120},
    # BE (breakeven) family
    "BE15_TP50_SL30":      {"kind": "be", "be_act": 0.15, "tp": 0.50, "sl": 0.30, "horizon": 120},
}
POLL_SECONDS = [30, 60, 120]
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


def _parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def subsample(ticks, interval_sec):
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


def nearest_after(ticks, ts):
    for t, p in ticks:
        if t >= ts and p > 0:
            return p
    return ticks[-1][1] if ticks else None


def interleave_avg(ds_ticks, jp_ticks):
    """For each Jupiter tick, blend with nearest DS tick in time."""
    if not jp_ticks:
        return []
    out = []
    for t, jp in jp_ticks:
        # find nearest DS tick
        closest = None
        best_dt = 999999
        for dt, dp in ds_ticks:
            d = abs((_parse_ts(dt) - _parse_ts(t)).total_seconds())
            if d < best_dt:
                best_dt = d
                closest = dp
            if d > 300:
                continue
        if closest and best_dt < 120:
            out.append((t, (jp + closest) / 2))
        else:
            out.append((t, jp))
    return out


def ema(ticks, window):
    if not ticks:
        return []
    alpha = 2 / (window + 1)
    out = []
    ema_val = ticks[0][1]
    for t, p in ticks:
        ema_val = alpha * p + (1 - alpha) * ema_val
        out.append((t, ema_val))
    return out


def simulate(entry, ticks_dec, ticks_exec, cfg, horizon_min, opened_ts, confirm_ticks=None):
    """
    ticks_dec : series used for decision (trail / TP / SL trigger)
    ticks_exec: series to pull exit price from (nearest after trigger time)
    confirm_ticks: if set, trigger requires BOTH ticks_dec and confirm_ticks under threshold
    """
    if not ticks_dec:
        return 0.0, "no_data"
    kind = cfg["kind"]
    exec_sorted = sorted(ticks_exec, key=lambda x: x[0])
    high = entry
    be_active = False
    t_open = _parse_ts(opened_ts)
    for t, price in ticks_dec:
        if price <= 0:
            continue
        # horizon timeout
        elapsed = (_parse_ts(t) - t_open).total_seconds() / 60
        if elapsed >= horizon_min:
            exit_p = nearest_after(exec_sorted, t) or price
            return (exit_p / entry) - 1, "timeout"
        high = max(high, price)

        if kind == "fixed":
            tp_p = entry * (1 + cfg["tp"])
            sl_p = entry * (1 - cfg["sl"])
            if price >= tp_p:
                exit_p = nearest_after(exec_sorted, t) or price
                return (exit_p / entry) - 1, "tp_hit"
            if price <= sl_p:
                exit_p = nearest_after(exec_sorted, t) or price
                return (exit_p / entry) - 1, "sl_hit"
        elif kind == "be":
            tp_p = entry * (1 + cfg["tp"])
            be_trigger = entry * (1 + cfg["be_act"])
            sl_p = entry if be_active else entry * (1 - cfg["sl"])
            if high >= be_trigger:
                be_active = True
            if price >= tp_p:
                exit_p = nearest_after(exec_sorted, t) or price
                return (exit_p / entry) - 1, "tp_hit"
            if price <= sl_p:
                exit_p = nearest_after(exec_sorted, t) or price
                return (exit_p / entry) - 1, "sl_hit" if not be_active else "be_stop"
        elif kind == "dtrail":
            act_p = entry * (1 + cfg["act"])
            sl_p = entry * (1 - cfg["sl"])
            if price <= sl_p:
                exit_p = nearest_after(exec_sorted, t) or price
                return (exit_p / entry) - 1, "sl_hit"
            if high >= act_p:
                trig = high * (1 - cfg["trail"])
                if price <= trig and trig > entry:
                    # Double confirmation check
                    if confirm_ticks is not None:
                        conf_p = nearest_after(confirm_ticks, t)
                        if conf_p is None or conf_p > trig:
                            continue  # not confirmed, skip trigger
                    exit_p = nearest_after(exec_sorted, t) or price
                    return (exit_p / entry) - 1, "trail_stop"
    # end of ticks — treat as timeout
    last_t, last_p = ticks_dec[-1]
    exit_p = nearest_after(exec_sorted, last_t) or last_p
    return (exit_p / entry) - 1, "end_of_data"


def main():
    print("Loading trades since Apr 8...")
    trades = fetch_all("paper_trades", gte_created_at=SINCE)
    trades = [t for t in trades if t.get("strategy") in STRATS]
    print(f"  {len(trades)} trades across {len(STRATS)} strategies")

    tokens = list({t["token_address"] for t in trades if t.get("token_address")})
    print(f"  {len(tokens)} unique tokens")

    print("Loading ticks...")
    ticks = fetch_all("price_ticks", gte_fetched_at=SINCE, in_token=tokens)
    print(f"  {len(ticks)} ticks")

    by_token = defaultdict(lambda: {"ds": [], "jupiter": []})
    for tk in ticks:
        ca = tk["token_address"]
        t = tk["fetched_at"]
        p = float(tk["price_usd"])
        if p <= 0:
            continue
        if tk["source"] in ("fast", "full", "live"):
            by_token[ca]["ds"].append((t, p))
        elif tk["source"] == "jupiter":
            by_token[ca]["jupiter"].append((t, p))
    for ca in by_token:
        by_token[ca]["ds"].sort()
        by_token[ca]["jupiter"].sort()

    print(f"\n{'Strategy':<25}{'Mode':<18}{'Poll':>6}{'N':>5}{'Avg%':>9}{'WR%':>7}{'Tot$':>9}")
    print("-" * 80)

    for strat_name, cfg in STRATS.items():
        strat_trades = [t for t in trades if t["strategy"] == strat_name]
        if not strat_trades:
            continue

        modes = []
        for poll in POLL_SECONDS:
            modes.append(("jupiter", "jupiter_only", poll))
            modes.append(("ds", "ds_only", poll))
            modes.append(("hybrid", "hybrid_ds_jup", poll))
            modes.append(("avg", "avg_both", poll))
            modes.append(("confirm", "confirm_both", poll))
            modes.append(("ema3", "ema_jupiter_3", poll))

        for variant, label, poll in modes:
            pnls = []
            for trade in strat_trades:
                ca = trade.get("token_address")
                entry = float(trade.get("entry_price") or 0)
                opened = trade["created_at"]
                if entry <= 0 or ca not in by_token:
                    continue
                ds_raw = [(t, p) for t, p in by_token[ca]["ds"] if t >= opened]
                jp_raw = [(t, p) for t, p in by_token[ca]["jupiter"] if t >= opened]
                if len(jp_raw) < 3:
                    continue

                if variant == "jupiter":
                    dec = subsample(jp_raw, poll)
                    exe = jp_raw
                    conf = None
                elif variant == "ds":
                    if len(ds_raw) < 3:
                        continue
                    dec = subsample(ds_raw, poll)
                    exe = ds_raw
                    conf = None
                elif variant == "hybrid":
                    if len(ds_raw) < 3:
                        continue
                    dec = subsample(ds_raw, poll)
                    exe = jp_raw
                    conf = None
                elif variant == "avg":
                    if len(ds_raw) < 3:
                        continue
                    merged = interleave_avg(ds_raw, jp_raw)
                    dec = subsample(merged, poll)
                    exe = jp_raw
                    conf = None
                elif variant == "confirm":
                    if len(ds_raw) < 3:
                        continue
                    dec = subsample(jp_raw, poll)
                    exe = jp_raw
                    conf = ds_raw
                elif variant == "ema3":
                    smoothed = ema(jp_raw, 3)
                    dec = subsample(smoothed, poll)
                    exe = jp_raw
                    conf = None
                else:
                    continue

                if len(dec) < 2:
                    continue
                pnl, _ = simulate(entry, dec, exe, cfg, cfg["horizon"], opened,
                                  confirm_ticks=conf)
                pnls.append(pnl)

            if len(pnls) < 10:
                continue
            avg = sum(pnls) / len(pnls) * 100
            wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
            tot = sum(p * 10 for p in pnls)
            print(f"{strat_name:<25}{label:<18}{poll:>6}{len(pnls):>5}{avg:>8.2f}%{wr:>6.1f}%{tot:>8.2f}$")
        print()


if __name__ == "__main__":
    main()
