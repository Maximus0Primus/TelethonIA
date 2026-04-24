"""Backtest: would raising the age_hours window have made money?

Extracts all "token too old" skips from the last 48h VPS logs, buckets them
by age-at-skip, fetches the OHLCV path starting at the KOL call time, and
simulates 3 live strategies (TP80_SL25, TP50_SL30, BE25_TP80_SL30) with the
same fee model as live Jupiter. Aggregates per age band.

Usage:
    # Make sure ./kol_48h.log exists (pulled from VPS earlier)
    python scripts/_sim_age_window.py
"""
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))

from supabase import create_client

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

LOG_PATH = "./kol_48h.log"
POSITION_USD = 1.70  # live size currently
SELL_FEE_BPS = 10    # Jupiter Ultra RFQ baseline
BUY_FEE_BPS = 10

STRATS = [
    {"name": "TP80_SL25", "tp": 0.80, "sl": 0.25, "h_min": 30, "be": None},
    {"name": "TP50_SL30", "tp": 0.50, "sl": 0.30, "h_min": 30, "be": None},
    {"name": "BE25_TP80_SL30", "tp": 0.80, "sl": 0.30, "h_min": 120, "be": 0.25},
]


def parse_skip_rows(path):
    """yield (ts_iso, symbol, age_hours) per too-old skip."""
    pat = re.compile(
        r"^(\w{3} \d+ \d+:\d+:\d+).*RT SKIP: (\$\S+) — token too old \((\d+)h"
    )
    out = []
    year = datetime.now(timezone.utc).year
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            # Approximate timestamp (journalctl uses syslog format w/o year)
            try:
                ts = datetime.strptime(f"{year} {m.group(1)}", "%Y %b %d %H:%M:%S")
                ts = ts.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            out.append((ts, m.group(2), int(m.group(3))))
    return out


def resolve_cas(symbols_with_ts):
    """Find resolved_ca per (symbol, ts) via kol_mentions closest message."""
    out = {}
    for ts, sym, age in symbols_with_ts:
        win_lo = (ts - timedelta(minutes=30)).isoformat()
        win_hi = (ts + timedelta(minutes=30)).isoformat()
        try:
            r = (sb.table("kol_mentions")
                 .select("resolved_ca,chain,message_date")
                 .eq("symbol", sym)
                 .gte("message_date", win_lo)
                 .lte("message_date", win_hi)
                 .not_.is_("resolved_ca", "null")
                 .limit(1).execute())
            if r.data:
                out[(ts, sym, age)] = (r.data[0]["resolved_ca"], r.data[0].get("chain") or "solana")
        except Exception as e:
            print(f"  resolve fail {sym}: {e}")
    return out


def fetch_ohlcv_dexpaprika(pool_or_token, chain, start_iso, end_iso):
    """Simple DexPaprika OHLCV fetch. Returns list of (ts, open, close, low, high, vol)."""
    import urllib.request
    import urllib.parse
    import json as _json
    if chain != "solana":
        return []
    try:
        # Try token endpoint (direct CA)
        url = (f"https://api.dexpaprika.com/networks/solana/tokens/{pool_or_token}/ohlcv"
               f"?start={urllib.parse.quote(start_iso)}&interval=5m&limit=48")
        with urllib.request.urlopen(url, timeout=10) as r:
            data = _json.loads(r.read())
        if not data or "data" not in data:
            return []
        out = []
        for c in data["data"]:
            try:
                out.append((c["time_open"], float(c["open"]), float(c["close"]),
                            float(c["low"]), float(c["high"]), float(c.get("volume_usd") or 0)))
            except Exception:
                continue
        return out
    except Exception:
        return []


def simulate(ohlcv, entry_price, strat):
    """Simulate one strat on the candles. Returns (exit_reason, pnl_pct_net)."""
    if not ohlcv or entry_price <= 0:
        return None
    sl_price = entry_price * (1 - strat["sl"])
    tp_price = entry_price * (1 + strat["tp"])
    start_ts = ohlcv[0][0]
    high_seen = entry_price
    # Parse start iso
    if isinstance(start_ts, str):
        start_dt = datetime.fromisoformat(start_ts.replace("Z", "+00:00"))
    else:
        start_dt = datetime.fromtimestamp(start_ts / 1000, tz=timezone.utc)

    for ts, o, c, low, high, v in ohlcv:
        try:
            t = (datetime.fromisoformat(ts.replace("Z", "+00:00"))
                 if isinstance(ts, str)
                 else datetime.fromtimestamp(ts / 1000, tz=timezone.utc))
        except Exception:
            continue
        elapsed_min = (t - start_dt).total_seconds() / 60
        if elapsed_min > strat["h_min"]:
            exit_px = c
            gross = (exit_px / entry_price - 1)
            net = gross - (BUY_FEE_BPS + SELL_FEE_BPS) / 10_000
            return "timeout", net
        high_seen = max(high_seen, high)
        effective_sl = sl_price
        if strat["be"] is not None and high_seen >= entry_price * (1 + strat["be"]):
            effective_sl = max(sl_price, entry_price)
        # SL hit (wick low touched)
        if low <= effective_sl:
            exit_px = effective_sl
            gross = (exit_px / entry_price - 1)
            net = gross - (BUY_FEE_BPS + SELL_FEE_BPS) / 10_000
            return ("sl_hit" if effective_sl == sl_price else "be_stop"), net
        # TP hit (wick high touched)
        if high >= tp_price:
            exit_px = tp_price
            gross = (exit_px / entry_price - 1)
            net = gross - (BUY_FEE_BPS + SELL_FEE_BPS) / 10_000
            return "tp_hit", net

    # Candles exhausted before horizon
    exit_px = ohlcv[-1][2]
    gross = (exit_px / entry_price - 1)
    net = gross - (BUY_FEE_BPS + SELL_FEE_BPS) / 10_000
    return "no_data", net


def main():
    if not os.path.exists(LOG_PATH):
        print(f"[err] {LOG_PATH} missing — run: ssh vps \"journalctl -u kol-scraper --since '48h ago' --no-pager\" > kol_48h.log")
        sys.exit(1)

    rows = parse_skip_rows(LOG_PATH)
    print(f"Parsed {len(rows)} too-old skips from logs")
    # Dedup (symbol, ts quantized to minute) to avoid the same call polluting multiple strats
    seen = set()
    unique = []
    for ts, sym, age in rows:
        key = (sym, ts.replace(second=0, microsecond=0))
        if key in seen:
            continue
        seen.add(key)
        unique.append((ts, sym, age))
    print(f"Unique (symbol, minute) skips: {len(unique)}")

    # Age buckets
    buckets = [("12-24h", 12, 24), ("24-48h", 24, 48), ("48-72h", 48, 72),
               ("72-168h (1w)", 72, 168), ("168-720h (1m)", 168, 720), (">720h", 720, 99999)]

    print(f"\nResolving CAs via kol_mentions (this hits Supabase, ~30s)...")
    resolved = resolve_cas(unique)
    print(f"  Resolved {len(resolved)}/{len(unique)} ({len(resolved)/len(unique)*100:.0f}%)")

    # Per-bucket aggregation
    agg = defaultdict(lambda: {"n": 0, "per_strat": defaultdict(lambda: {"n": 0, "pnl_usd": 0, "wins": 0})})
    ohlcv_fetched = 0
    ohlcv_with_data = 0

    print(f"\nFetching DexPaprika OHLCV for {len(resolved)} CAs...")
    for i, ((ts, sym, age), (ca, chain)) in enumerate(resolved.items()):
        if chain != "solana":
            continue
        start_iso = ts.isoformat()
        end_iso = (ts + timedelta(hours=3)).isoformat()
        ohlcv = fetch_ohlcv_dexpaprika(ca, chain, start_iso, end_iso)
        ohlcv_fetched += 1
        if not ohlcv or len(ohlcv) < 2:
            continue
        ohlcv_with_data += 1
        # Use first candle open as entry
        entry = ohlcv[0][1]
        if entry <= 0:
            continue
        # Bucket
        bucket_name = next((n for n, lo, hi in buckets if lo <= age < hi), ">720h")
        agg[bucket_name]["n"] += 1
        for strat in STRATS:
            r = simulate(ohlcv, entry, strat)
            if r is None:
                continue
            reason, net_pct = r
            pnl_usd = POSITION_USD * net_pct
            agg[bucket_name]["per_strat"][strat["name"]]["n"] += 1
            agg[bucket_name]["per_strat"][strat["name"]]["pnl_usd"] += pnl_usd
            if net_pct > 0:
                agg[bucket_name]["per_strat"][strat["name"]]["wins"] += 1
        if (i + 1) % 10 == 0:
            print(f"  progress {i+1}/{len(resolved)}... ({ohlcv_with_data} with OHLCV)")
        time.sleep(0.15)  # be nice to DexPaprika

    print(f"\nOHLCV fetched: {ohlcv_fetched}, with usable candles: {ohlcv_with_data}")

    print("\n=== PnL per age bucket (hypothetical if we'd traded) ===")
    print(f"{'Bucket':<18}{'N tokens':>10}{'Strat':<18}{'N sim':>7}{'WR%':>6}{'PnL $':>10}{'$/trade':>9}")
    grand = defaultdict(lambda: {"n": 0, "pnl": 0})
    for bname, _, _ in buckets:
        d = agg.get(bname)
        if not d or d["n"] == 0:
            continue
        for sname in [s["name"] for s in STRATS]:
            ps = d["per_strat"].get(sname, {"n": 0, "pnl_usd": 0, "wins": 0})
            if ps["n"] == 0:
                continue
            wr = ps["wins"] / ps["n"] * 100
            pt = ps["pnl_usd"] / ps["n"] if ps["n"] else 0
            print(f"{bname:<18}{d['n']:>10}{sname:<18}{ps['n']:>7}{wr:>5.0f}%{ps['pnl_usd']:>+9.2f}{pt:>+8.3f}")
            grand[sname]["n"] += ps["n"]
            grand[sname]["pnl"] += ps["pnl_usd"]

    print(f"\n=== Grand total (all age bands) ===")
    for sname, d in grand.items():
        print(f"  {sname:<18} N={d['n']:<4} PnL=${d['pnl']:+.2f}  avg=${d['pnl']/d['n']:+.3f}/trade" if d['n'] else f"  {sname}: no data")

    print(f"\n=== Key question: is 12h->48h+ a money gain or loss? ===")
    # Tokens 12-48h
    n_12_48 = sum(agg[b]["n"] for b in ("12-24h", "24-48h"))
    n_48_plus = sum(agg[b]["n"] for b in ("48-72h", "72-168h (1w)", "168-720h (1m)", ">720h"))
    print(f"Tokens 12-48h (relaxing to 48h):  {n_12_48}")
    print(f"Tokens >48h (stay blocked):       {n_48_plus}")
    for sname in [s["name"] for s in STRATS]:
        pnl_12_48 = sum(agg[b]["per_strat"].get(sname, {"pnl_usd": 0})["pnl_usd"]
                        for b in ("12-24h", "24-48h"))
        n_12_48_sim = sum(agg[b]["per_strat"].get(sname, {"n": 0})["n"]
                          for b in ("12-24h", "24-48h"))
        if n_12_48_sim:
            print(f"  {sname:<18} 12-48h band: N={n_12_48_sim}  PnL=${pnl_12_48:+.2f}  "
                  f"avg=${pnl_12_48/n_12_48_sim:+.3f}/trade")


if __name__ == "__main__":
    main()
