"""
Entry-timing backtest — does delaying the entry beat entering at the call?

Motivation (Aug 5 2026): among SOL tokens that eventually gain >+100%, the share
that first dips below -40% went 10-27% (April) -> 74-82% (late July). Every SL in
the 30-40% band gets taken out during that dip and misses the runup. So the lever
is the ENTRY, not another TP/SL sweep.

Method
------
- Universe: every SOL token with an RT paper trade in the price_ticks window.
- t0 = first RT paper_trade.created_at for that token (the call).
- Price path = price_ticks (3-min batch cadence, ~30s in the fast window).
- ENTRY rules produce (entry_ts, entry_price) or None (rule never triggered).
- EXIT reuses sim_engines.simulate_fixed so slippage matches production exactly
  (_dynamic_sell_slip_factor). Buy slip = strategies.BUY_SLIPPAGE_BPS.
- Ranking metric is the ARITHMETIC mean (fixed sizing) plus the GEOMETRIC mean
  (compounding). Median is reported only as a shape diagnostic, never to rank.
- Comparisons are PAIRED on the intersection of tokens: a delayed rule that only
  fires on 40% of tokens must be compared against the baseline on those same
  tokens, never against the baseline's full universe.

Usage
-----
    python scripts/_entry_timing_backtest.py            # uses cache if present
    python scripts/_entry_timing_backtest.py --refresh  # re-pull ticks
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scraper"))

import sim_engines  # noqa: E402
from strategies import BUY_SLIPPAGE_BPS  # noqa: E402

CACHE = Path(os.environ.get("ENTRY_BT_CACHE", Path(__file__).parent / "_entry_bt_cache.json"))
WINDOW_DAYS = 30
PROBE_STRATEGY = "FAST_TP50_SL30_LAZYFAST"   # 644/646 tokens, 745 rows
MAX_HOLD_MIN = 360          # hard cap on how far we replay a token
ENTRY_SEARCH_MIN = 120      # a delayed rule must trigger within this window or it is a no-trade
BUY_SLIP = BUY_SLIPPAGE_BPS / 10_000


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _client():
    from supabase import create_client
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])


def _keyset_all(query_fn, key, page=1000):
    """Keyset pagination on a timestamp column. OFFSET paging on these tables
    hits the statement timeout once the offset gets deep — keyset stays flat."""
    out, cursor = [], None
    while True:
        rows = query_fn(cursor, page)
        if not rows:
            break
        out.extend(rows)
        if len(rows) < page:
            break
        nxt = rows[-1][key]
        if nxt == cursor:      # whole page shares one timestamp: bail out
            break
        cursor = nxt
    return out


def fetch_data() -> dict:
    sb = _client()
    since = (datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)).isoformat()

    blacklist = set()
    cfg = sb.table("scoring_config").select("paper_trade_config").eq("id", 1).execute().data
    if cfg:
        ptc = cfg[0].get("paper_trade_config") or {}
        blacklist = set((ptc.get("kol_chain_blacklist") or {}).get("solana") or [])

    # One row per token: the first RT call, its KOL, its liquidity.
    # PROBE_STRATEGY is a shadow strategy that fires once per token and covers
    # 644/646 of them — reading it instead of the whole grid cuts the pull 100x.
    def _trades(cursor, lim):
        q = (sb.table("paper_trades")
             .select("token_address,symbol,created_at,kol_group,rt_liquidity_usd")
             .eq("source", "rt").eq("chain", "solana")
             .eq("strategy", PROBE_STRATEGY)
             .gte("created_at", cursor or since)
             .order("created_at").limit(lim))
        return q.execute().data

    trades = _keyset_all(_trades, "created_at")

    calls: dict[str, dict] = {}
    for t in trades:
        a = t["token_address"]
        if a not in calls:
            calls[a] = {
                "symbol": t.get("symbol"),
                "t0": t["created_at"],
                "kol": t.get("kol_group"),
                "liq": float(t.get("rt_liquidity_usd") or 0),
            }
    calls = {a: c for a, c in calls.items() if c["kol"] not in blacklist}

    def _ticks(cursor, lim):
        return (sb.table("price_ticks")
                .select("token_address,price_usd,fetched_at,source")
                .eq("chain", "solana")
                .gte("fetched_at", cursor or since)
                .order("fetched_at").limit(lim).execute().data)

    ticks = _keyset_all(_ticks, "fetched_at")

    # CRITICAL: price_ticks is a multi-source LOG (jupiter + DexScreener
    # fast/full interleaved every 11-20s), not a price series. Mixing them
    # fabricates -85%/+640% "moves" that are just quote disagreement, and any
    # rule that triggers on a low print systematically selects the low-quoting
    # source. Production never mixes either: paper_trader._decision_price
    # resolves ONE source per strategy. So we keep the source tag and replay
    # one coherent stream at a time.
    series: dict[str, list] = defaultdict(list)
    for r in ticks:
        p = r.get("price_usd")
        if p is None or float(p) <= 0:
            continue
        series[r["token_address"]].append([_ts(r["fetched_at"]), float(p), r.get("source")])
    for a in series:
        series[a].sort()

    return {"calls": calls, "series": {a: s for a, s in series.items() if a in calls}}


def project_source(series: dict, source: str) -> dict:
    """Keep only one coherent price stream. 'ds' = DexScreener (fast+full,
    same provider, so mutually consistent); 'jupiter' = executable quotes."""
    keep = {"fast", "full"} if source == "ds" else {source}
    out = {}
    for a, rows in series.items():
        s = [[ts, p] for ts, p, src in rows if src in keep]
        if len(s) >= 5:
            out[a] = s
    return out


def _ts(iso: str) -> int:
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())


def load(refresh: bool) -> dict:
    if CACHE.exists() and not refresh:
        return json.loads(CACHE.read_text())
    data = fetch_data()
    CACHE.write_text(json.dumps(data))
    return data


# ---------------------------------------------------------------------------
# Entry rules -> (entry_ts, entry_price) or None
# ---------------------------------------------------------------------------

def entry_immediate(path, t0):
    for ts, p in path:
        if ts >= t0:
            return ts, p
    return None


def make_entry_wait(minutes):
    def rule(path, t0):
        target = t0 + minutes * 60
        for ts, p in path:
            if ts >= target:
                return ts, p
        return None
    rule.__name__ = f"wait{minutes}m"
    return rule


def make_entry_dip(drop_pct, fill_lag=0):
    """Enter on the first print at or below call_price * (1 - drop).

    fill_lag=0 fills AT the trigger print, which assumes the dip tick is
    actually reachable. price_ticks are 3-min batch prints, so that tick may be
    a wick nobody could hit. fill_lag=1 fills at the NEXT print instead — the
    honest version of "I saw the dip, I bought right after". If the edge only
    survives at fill_lag=0 it is a measurement artefact, not a strategy.
    """
    def rule(path, t0):
        ref = None
        limit = t0 + ENTRY_SEARCH_MIN * 60
        for i, (ts, p) in enumerate(path):
            if ts < t0:
                continue
            if ref is None:
                ref = p
                continue
            if ts > limit:
                return None
            if p <= ref * (1 - drop_pct):
                j = i + fill_lag
                if j >= len(path):
                    return None
                return path[j][0], path[j][1]
        return None
    rule.__name__ = f"dip{int(drop_pct*100)}" + (f"_lag{fill_lag}" if fill_lag else "")
    return rule


def make_entry_reclaim(drop_pct, reclaim_pct):
    """Wait for a dip of `drop`, then enter once price climbs back `reclaim`
    off the observed trough. Buys strength after the flush instead of the knife."""
    def rule(path, t0):
        ref = None
        dipped = False
        trough = None
        limit = t0 + ENTRY_SEARCH_MIN * 60
        for ts, p in path:
            if ts < t0:
                continue
            if ref is None:
                ref = p
                continue
            if ts > limit:
                return None
            if not dipped:
                if p <= ref * (1 - drop_pct):
                    dipped = True
                    trough = p
            else:
                trough = min(trough, p)
                if p >= trough * (1 + reclaim_pct):
                    return ts, p
        return None
    rule.__name__ = f"dip{int(drop_pct*100)}_rec{int(reclaim_pct*100)}"
    return rule


ENTRY_RULES = [
    ("immediate", entry_immediate),
    ("wait15m", make_entry_wait(15)),
    ("wait30m", make_entry_wait(30)),
    ("wait60m", make_entry_wait(60)),
    ("dip20", make_entry_dip(0.20)),
    ("dip30", make_entry_dip(0.30)),
    ("dip40", make_entry_dip(0.40)),
    ("dip50", make_entry_dip(0.50)),
    # Conservative fills: buy one / two prints AFTER the dip trigger.
    ("dip30_lag1", make_entry_dip(0.30, fill_lag=1)),
    ("dip40_lag1", make_entry_dip(0.40, fill_lag=1)),
    ("dip50_lag1", make_entry_dip(0.50, fill_lag=1)),
    ("dip40_lag2", make_entry_dip(0.40, fill_lag=2)),
    ("dip50_lag2", make_entry_dip(0.50, fill_lag=2)),
    ("dip30_rec15", make_entry_reclaim(0.30, 0.15)),
    ("dip40_rec15", make_entry_reclaim(0.40, 0.15)),
    ("dip40_rec25", make_entry_reclaim(0.40, 0.25)),
    ("dip50_rec25", make_entry_reclaim(0.50, 0.25)),
]

EXIT_CFGS = [
    ("TP50_SL30_2H",  {"tp_mult": 1.50, "sl_mult": 0.70, "horizon_min": 120}),
    ("TP50_SL50_2H",  {"tp_mult": 1.50, "sl_mult": 0.50, "horizon_min": 120}),
    ("TP100_SL50_4H", {"tp_mult": 2.00, "sl_mult": 0.50, "horizon_min": 240}),
    ("TP100_SL70_4H", {"tp_mult": 2.00, "sl_mult": 0.30, "horizon_min": 240}),
    ("TP200_SL50_4H", {"tp_mult": 3.00, "sl_mult": 0.50, "horizon_min": 240}),
    ("TP30_SL30_1H",  {"tp_mult": 1.30, "sl_mult": 0.70, "horizon_min": 60}),
    ("TP20_SL80_2H",  {"tp_mult": 1.20, "sl_mult": 0.20, "horizon_min": 120}),
]


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def to_candles(path, from_ts, max_min):
    """price_ticks are point prints: open=high=low=close. No fabricated wicks."""
    end = from_ts + max_min * 60
    return [{"timestamp": ts, "open": p, "high": p, "low": p, "close": p, "volume": 0}
            for ts, p in path if from_ts <= ts <= end]


def run(data):
    calls, series = data["calls"], data["series"]
    results = defaultdict(list)   # (entry, exit) -> [(token, week, pnl)]

    for addr, meta in calls.items():
        path = series.get(addr)
        if not path or len(path) < 5:
            continue
        t0 = _ts(meta["t0"])
        if not any(ts >= t0 for ts, _ in path):
            continue

        sim_engines._sim_liquidity_usd = meta["liq"] or 50_000
        sim_engines._sim_chain = "solana"
        sim_engines._sim_position_usd = 10

        week = datetime.fromtimestamp(t0, timezone.utc).strftime("%Y-W%V")

        for ename, erule in ENTRY_RULES:
            got = erule(path, t0)
            if not got:
                continue
            ets, eprice = got
            eprice *= (1 + BUY_SLIP)          # pay the spread on the way in
            candles = to_candles(path, ets, MAX_HOLD_MIN)
            if len(candles) < 3:
                continue
            for xname, cfg in EXIT_CFGS:
                r = sim_engines.simulate_fixed(candles, eprice, cfg)
                results[(ename, xname)].append((addr, week, r["pnl_pct"]))
    return results


# ---------------------------------------------------------------------------
# Stats — mean first, median only as a shape diagnostic
# ---------------------------------------------------------------------------

def stats(rows):
    pnls = [p for _, _, p in rows]
    n = len(pnls)
    if n == 0:
        return None
    mean = sum(pnls) / n
    s = sorted(pnls)
    med = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    geo = math.exp(sum(math.log(max(1 + p, 1e-6)) for p in pnls) / n) - 1
    ex_best = (sum(pnls) - max(pnls))
    byweek = defaultdict(list)
    for _, w, p in rows:
        byweek[w].append(p)
    weeks = {w: sum(v) / len(v) for w, v in byweek.items() if len(v) >= 8}
    pos_weeks = sum(1 for m in weeks.values() if m > 0)
    return {"n": n, "mean": mean, "med": med, "geo": geo, "sum": sum(pnls),
            "ex_best": ex_best, "weeks": len(weeks), "pos_weeks": pos_weeks}


def paired(rows_a, rows_b):
    """Mean delta on the intersection of tokens only."""
    a = {t: p for t, _, p in rows_a}
    b = {t: p for t, _, p in rows_b}
    common = a.keys() & b.keys()
    if len(common) < 10:
        return None
    da = sum(a[t] for t in common) / len(common)
    db = sum(b[t] for t in common) / len(common)
    return {"n": len(common), "a": da, "b": db, "delta": da - db}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--source", default="jupiter", choices=["jupiter", "ds"],
                    help="which coherent price stream to replay (never mix)")
    args = ap.parse_args()

    data = load(args.refresh)
    data = {"calls": data["calls"], "series": project_source(data["series"], args.source)}
    print(f"SOURCE={args.source}  tokens={len(data['calls'])}  avec_ticks={len(data['series'])}")

    results = run(data)
    rank = []
    for key, rows in results.items():
        st = stats(rows)
        if st and st["n"] >= 25:
            rank.append((key, st))
    rank.sort(key=lambda x: -x[1]["mean"])

    print(f"\n{'entree':<14}{'exit':<15}{'n':>5}{'moy%':>8}{'geo%':>8}{'med%':>8}"
          f"{'sansMax':>9}{'sem+':>7}")
    print("-" * 76)
    for (e, x), s in rank[:25]:
        print(f"{e:<14}{x:<15}{s['n']:>5}{100*s['mean']:>8.1f}{100*s['geo']:>8.1f}"
              f"{100*s['med']:>8.1f}{100*s['ex_best']:>9.0f}"
              f"{s['pos_weeks']:>4}/{s['weeks']}")

    print("\n=== TEST APPARIE vs 'immediate' (meme exit, memes tokens) ===")
    print(f"{'entree':<14}{'exit':<15}{'n_comm':>7}{'retarde%':>10}{'immed%':>9}{'delta':>8}")
    print("-" * 65)
    for (e, x), _ in rank[:25]:
        if e == "immediate":
            continue
        base = results.get(("immediate", x))
        if not base:
            continue
        pr = paired(results[(e, x)], base)
        if pr:
            print(f"{e:<14}{x:<15}{pr['n']:>7}{100*pr['a']:>10.1f}"
                  f"{100*pr['b']:>9.1f}{100*pr['delta']:>+8.1f}")

    # Decisive check: how much of the dip edge is just an unfillable wick?
    print("\n=== REALISME DU FILL: dip au tick declencheur vs 1 et 2 ticks apres ===")
    print(f"{'exit':<15}{'variante':<14}{'n':>5}{'moy%':>8}{'geo%':>8}{'sem+':>7}")
    print("-" * 57)
    for x, _ in EXIT_CFGS:
        for depth in (30, 40, 50):
            for suffix in ("", "_lag1", "_lag2"):
                rows = results.get((f"dip{depth}{suffix}", x))
                if not rows:
                    continue
                s = stats(rows)
                if s and s["n"] >= 25:
                    print(f"{x:<15}{'dip'+str(depth)+suffix:<14}{s['n']:>5}"
                          f"{100*s['mean']:>8.1f}{100*s['geo']:>8.1f}"
                          f"{s['pos_weeks']:>4}/{s['weeks']}")
        print()


if __name__ == "__main__":
    main()
