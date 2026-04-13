"""
Convertit les ticks Jupiter en bougies 1-min compatibles avec ohlcv_cache/, puis
exécute l'OHLCV-sim sur ces bougies pour comparer aux résultats tick-level.

Usage:
  python scripts/jupiter_ticks_to_candles.py [--bucket-sec 60] [--since 2026-04-08] [--compare]

--compare : pour chaque token avec ticks Jupiter, compare la version Jupiter-candles
            à la version OHLCV-cache existante (DexScreener-derived). Montre si
            les stats de strat changent quand on remplace la source OHLCV.
"""
import os
import sys
import json
import argparse
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
SCRAPER_DIR = SCRIPT_DIR.parent / "scraper"
sys.path.insert(0, str(SCRAPER_DIR))
load_dotenv(SCRAPER_DIR / ".env")

from supabase import create_client

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

OUT_DIR = SCRAPER_DIR / "jupiter_candles_cache"
OUT_DIR.mkdir(exist_ok=True)


def fetch_all(table, **f):
    out, step, offset = [], 1000, 0
    while True:
        q = sb.table(table).select("*")
        for k, v in f.items():
            if k.startswith("gte_"): q = q.gte(k[4:], v)
            elif k.startswith("eq_"): q = q.eq(k[3:], v)
            elif k == "in_token": q = q.in_("token_address", v)
        q = q.range(offset, offset + step - 1).order("fetched_at" if "tick" in table else "created_at")
        r = q.execute()
        if not r.data: break
        out.extend(r.data)
        if len(r.data) < step: break
        offset += step
    return out


def bucket_ticks(ticks: list[dict], bucket_sec: int = 60) -> list[dict]:
    """Bucket (fetched_at, price_usd) ticks into OHLCV candles."""
    if not ticks:
        return []
    parsed = []
    for tk in ticks:
        price = float(tk["price_usd"])
        if price <= 0:
            continue
        ts = datetime.fromisoformat(tk["fetched_at"].replace("Z", "+00:00"))
        parsed.append((int(ts.timestamp()), price, float(tk.get("volume_usd") or 0)))
    if not parsed:
        return []
    parsed.sort()

    buckets: dict[int, list] = defaultdict(list)
    for ts, price, vol in parsed:
        bucket = (ts // bucket_sec) * bucket_sec
        buckets[bucket].append((price, vol))

    candles = []
    prev_close = parsed[0][1]
    for bucket in sorted(buckets.keys()):
        entries = buckets[bucket]
        prices = [p for p, _ in entries]
        vols = [v for _, v in entries]
        candles.append({
            "timestamp": bucket,
            "open": prices[0],
            "high": max(prices),
            "low": min(prices),
            "close": prices[-1],
            "volume": sum(vols) if vols else 0.0,
        })
        prev_close = prices[-1]
    return candles


def save_candles(token_address: str, candles: list[dict], since_ts: int):
    """Mimics OHLCV cache filename format: {addr12}_{start_ts}_{days}_{hash12}.json"""
    if not candles:
        return None
    days = max(1, int((candles[-1]["timestamp"] - candles[0]["timestamp"]) / 86400) + 1)
    raw = json.dumps(candles, sort_keys=True).encode()
    h = hashlib.md5(raw).hexdigest()[:12]
    fname = f"{token_address[:12]}_{since_ts}_{days}_{h}.json"
    path = OUT_DIR / fname
    path.write_text(json.dumps(candles))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-04-08")
    ap.add_argument("--bucket-sec", type=int, default=60)
    ap.add_argument("--compare", action="store_true",
                    help="Compare Jupiter-candles stats to existing OHLCV cache per token")
    args = ap.parse_args()

    since_iso = f"{args.since}T00:00:00+00:00"
    since_ts = int(datetime.fromisoformat(since_iso).timestamp())

    print(f"Loading trades since {args.since}...")
    trades = fetch_all("paper_trades", gte_created_at=since_iso)
    tokens = list({t["token_address"] for t in trades if t.get("token_address")})
    print(f"  {len(tokens)} unique tokens across {len(trades)} trades")

    print("Loading Jupiter ticks...")
    ticks = fetch_all("price_ticks", gte_fetched_at=since_iso, in_token=tokens)
    jup_ticks = [t for t in ticks if t["source"] == "jupiter"]
    print(f"  {len(jup_ticks)} Jupiter ticks out of {len(ticks)} total")

    by_token = defaultdict(list)
    for t in jup_ticks:
        by_token[t["token_address"]].append(t)

    candles_per_token = {}
    for addr, tks in by_token.items():
        candles = bucket_ticks(tks, args.bucket_sec)
        if candles:
            candles_per_token[addr] = candles
            save_candles(addr, candles, since_ts)

    print(f"\nGenerated candles for {len(candles_per_token)} tokens ({args.bucket_sec}s buckets)")
    print(f"Written to {OUT_DIR}")

    # Summary stats
    if candles_per_token:
        ncandles = [len(c) for c in candles_per_token.values()]
        print(f"Candles/token: min={min(ncandles)} median={sorted(ncandles)[len(ncandles)//2]} max={max(ncandles)}")

    if args.compare:
        # For each token, compare a simple FIXED TP50/SL30 outcome on Jupiter-candles vs ticks
        print("\n" + "="*80)
        print("COMPARE: TP50/SL30 outcome on Jupiter-candles vs raw Jupiter-ticks")
        print("="*80)
        print(f"{'Token':<14}{'Candle exit':>14}{'Tick exit':>14}{'Diff':>12}")
        total_candle_pnl = 0
        total_tick_pnl = 0
        n_compared = 0

        for trade in trades:
            addr = trade["token_address"]
            if addr not in candles_per_token or addr not in by_token:
                continue
            entry = float(trade.get("entry_price") or 0)
            if entry <= 0:
                continue
            tp = entry * 1.5
            sl = entry * 0.7

            opened_ts = int(datetime.fromisoformat(trade["created_at"].replace("Z", "+00:00")).timestamp())

            # Candle-based eval (only checks close prices = OHLCV bias)
            c_pnl = None
            for c in candles_per_token[addr]:
                if c["timestamp"] < opened_ts:
                    continue
                if c["timestamp"] - opened_ts > 30 * 60:  # 30min horizon
                    c_pnl = (c["close"] / entry) - 1
                    break
                if c["close"] >= tp:
                    c_pnl = (tp / entry) - 1
                    break
                if c["close"] <= sl:
                    c_pnl = (sl / entry) - 1
                    break

            # Tick-based eval (sees every tick)
            t_pnl = None
            for tk in by_token[addr]:
                tts = int(datetime.fromisoformat(tk["fetched_at"].replace("Z", "+00:00")).timestamp())
                if tts < opened_ts:
                    continue
                if tts - opened_ts > 30 * 60:
                    t_pnl = (float(tk["price_usd"]) / entry) - 1
                    break
                p = float(tk["price_usd"])
                if p >= tp:
                    t_pnl = (tp / entry) - 1
                    break
                if p <= sl:
                    t_pnl = (sl / entry) - 1
                    break

            if c_pnl is None or t_pnl is None:
                continue
            total_candle_pnl += c_pnl
            total_tick_pnl += t_pnl
            n_compared += 1
            if abs(c_pnl - t_pnl) > 0.05:
                sym = trade.get("symbol", addr[:8])[:13]
                print(f"{sym:<14}{c_pnl*100:>12.1f}%{t_pnl*100:>12.1f}%{(c_pnl-t_pnl)*100:>10.1f}pp")

        if n_compared:
            print(f"\nN compared: {n_compared}")
            print(f"Candle avg PnL: {total_candle_pnl/n_compared*100:+.2f}%")
            print(f"Tick   avg PnL: {total_tick_pnl/n_compared*100:+.2f}%")
            print(f"Bias (candle vs tick): {(total_candle_pnl-total_tick_pnl)/n_compared*100:+.2f}pp")
            print("\nPositive bias = OHLCV hides losses (candle close missed the SL hit)")
            print("Negative bias = OHLCV hides gains (candle close missed the TP hit)")


if __name__ == "__main__":
    main()
