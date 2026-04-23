"""KOL leaderboard — price-tick quality, strategy-agnostic.

For every (kol_group, token) first call we fetch the raw price_ticks series
after entry and compute:
  - max_gain_pct       = (max(price) - entry) / entry   in [entry, entry+2h]
  - max_drawdown_pct   = (min(price) - entry) / entry   in [entry, entry+2h]
  - ret_{30m,1h,2h}    = price(~t) / entry - 1
  - positive           = max_gain_pct >= 0.10 (default +10% threshold)
  - negative           = max_drawdown_pct <= -0.20 (default -20%)

Then we aggregate per KOL across unique first-call tokens:
  - N              : # unique tokens called
  - win_rate       : % tokens that pumped >= +10% before -20% drawdown
  - median_max_up  : median of max_gain_pct
  - median_max_dn  : median of max_drawdown_pct
  - pos_neg_ratio  : wins / losses (wins: +10% pump, losses: -20% dump)
  - median_ret_1h  : median price at 1h vs entry

Why independent of strategy: TP/SL/timeout all sample the same underlying
price tape. A KOL whose tokens reliably push past +10% before -20% is a
good caller regardless of whether we trade FAST or BE25. We rank by
win_rate first (conviction), then pos_neg_ratio (asymmetry), then N.

Usage:
    python scripts/kol_tick_quality.py              # last 30d
    WINDOW_DAYS=14 python scripts/kol_tick_quality.py
    MIN_N=5 python scripts/kol_tick_quality.py
"""
import os
import sys
import csv
import statistics as st
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

WINDOW_DAYS   = int(os.environ.get("WINDOW_DAYS", "30"))
HORIZON_HOURS = float(os.environ.get("HORIZON_HOURS", "2"))
MIN_N         = int(os.environ.get("MIN_N", "3"))
PUMP_THR      = float(os.environ.get("PUMP_THR", "0.10"))   # +10% = "win"
DUMP_THR      = float(os.environ.get("DUMP_THR", "-0.20"))  # -20% = "loss"

since_iso = (datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)).isoformat()
print(f"=== KOL tick-quality leaderboard ===")
print(f"window: last {WINDOW_DAYS}d since {since_iso}")
print(f"horizon: {HORIZON_HOURS}h post-entry   pump_thr=+{PUMP_THR*100:.0f}%   "
      f"dump_thr={DUMP_THR*100:.0f}%   min_N={MIN_N}\n")


def fetch_all(tbl, sel, order_col=None, **f):
    out, step, off = [], 1000, 0
    while True:
        q = sb.table(tbl).select(sel)
        for k, v in f.items():
            if k.startswith("gte_"): q = q.gte(k[4:], v)
            elif k.startswith("lte_"): q = q.lte(k[4:], v)
            elif k.startswith("eq_"): q = q.eq(k[3:], v)
            elif k.startswith("neq_"): q = q.neq(k[4:], v)
        if order_col:
            q = q.order(order_col)
        r = q.range(off, off+step-1).execute()
        if not r.data:
            break
        out.extend(r.data)
        if len(r.data) < step:
            break
        off += step
    return out


print("1/3 Loading paper_trades (any source, need kol_group + entry_price)...")
trades = fetch_all(
    "paper_trades",
    "id,kol_group,token_address,symbol,entry_price,created_at,source",
    gte_created_at=since_iso,
)
trades = [t for t in trades if t.get("kol_group") and t.get("token_address")
          and float(t.get("entry_price") or 0) > 0]
print(f"  {len(trades):,} trades with kol attribution")

# Dedup: one row per (kol, token) — earliest entry. This prevents a KOL's
# rank from being inflated when the same token appears many times in shadows.
first_call = {}  # (kol, addr) -> (entry_ts, entry_price, symbol)
for t in trades:
    key = (t["kol_group"], t["token_address"])
    ts = t["created_at"]
    if key not in first_call or ts < first_call[key][0]:
        first_call[key] = (ts, float(t["entry_price"]), t.get("symbol") or "?")
print(f"  {len(first_call):,} unique (kol, token) first calls\n")

# Group by token so we pull price_ticks once per token (not per KOL-pair).
by_token = defaultdict(list)  # token_address -> [(kol, entry_ts, entry_price, symbol)]
for (kol, addr), (ts, ep, sym) in first_call.items():
    by_token[addr].append((kol, ts, ep, sym))
print(f"2/3 Fetching price_ticks for {len(by_token):,} tokens "
      f"(horizon {HORIZON_HOURS}h, may take a minute)...")


def fetch_ticks(addr: str, entry_iso: str, end_iso: str) -> list[dict]:
    return fetch_all(
        "price_ticks",
        "fetched_at,price_usd",
        order_col="fetched_at",
        eq_token_address=addr,
        gte_fetched_at=entry_iso,
        lte_fetched_at=end_iso,
    )


per_kol = defaultdict(list)   # kol -> [metric dict per token]
skipped_no_ticks = 0
processed = 0
for addr, calls in by_token.items():
    # Use earliest entry across KOLs to fetch once; we'll re-slice per KOL below.
    earliest_ts = min(c[1] for c in calls)
    try:
        earliest_dt = datetime.fromisoformat(earliest_ts.replace("Z", "+00:00"))
    except Exception:
        continue
    end_dt = earliest_dt + timedelta(hours=HORIZON_HOURS + 0.1)
    ticks = fetch_ticks(addr, earliest_dt.isoformat(), end_dt.isoformat())
    if not ticks:
        skipped_no_ticks += 1
        processed += 1
        if processed % 100 == 0:
            print(f"  {processed}/{len(by_token)} tokens processed "
                  f"(skipped {skipped_no_ticks} no-ticks)")
        continue
    # Parse tick timestamps once
    parsed = []
    for r in ticks:
        try:
            dt = datetime.fromisoformat(r["fetched_at"].replace("Z", "+00:00"))
            p = float(r["price_usd"]) if r.get("price_usd") is not None else None
            if p and p > 0:
                parsed.append((dt, p))
        except Exception:
            pass
    if not parsed:
        skipped_no_ticks += 1
        processed += 1
        continue
    for kol, entry_iso, entry_price, symbol in calls:
        try:
            entry_dt = datetime.fromisoformat(entry_iso.replace("Z", "+00:00"))
        except Exception:
            continue
        horizon_end = entry_dt + timedelta(hours=HORIZON_HOURS)
        window = [(dt, p) for dt, p in parsed if entry_dt <= dt <= horizon_end]
        if len(window) < 2:
            continue
        prices = [p for _, p in window]
        max_up = (max(prices) - entry_price) / entry_price
        max_dn = (min(prices) - entry_price) / entry_price

        def ret_at(target_min: float) -> float | None:
            target = entry_dt + timedelta(minutes=target_min)
            # last tick at or before target
            bucket = [p for dt, p in window if dt <= target]
            if not bucket:
                return None
            return bucket[-1] / entry_price - 1

        # Determine win/loss: first to breach PUMP_THR vs DUMP_THR (path-dependent).
        hit_pump = False
        hit_dump = False
        for dt, p in window:
            chg = p / entry_price - 1
            if chg >= PUMP_THR and not hit_dump:
                hit_pump = True
                break
            if chg <= DUMP_THR and not hit_pump:
                hit_dump = True
                break

        per_kol[kol].append({
            "token": symbol,
            "addr": addr,
            "max_up": max_up,
            "max_dn": max_dn,
            "ret_30m": ret_at(30),
            "ret_1h":  ret_at(60),
            "ret_2h":  ret_at(120),
            "hit_pump": hit_pump,
            "hit_dump": hit_dump,
            "n_ticks": len(window),
        })
    processed += 1
    if processed % 100 == 0:
        print(f"  {processed}/{len(by_token)} tokens processed "
              f"(skipped {skipped_no_ticks} no-ticks)")

print(f"  done: {processed} tokens, {skipped_no_ticks} skipped (no ticks)\n")

print("3/3 Aggregating per KOL...")


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


rows = []
for kol, metrics in per_kol.items():
    n = len(metrics)
    if n < MIN_N:
        continue
    wins   = sum(1 for m in metrics if m["hit_pump"])
    losses = sum(1 for m in metrics if m["hit_dump"])
    neutral = n - wins - losses
    win_rate   = 100 * wins / n
    loss_rate  = 100 * losses / n
    pos_neg    = (wins / losses) if losses else (wins if wins else 0.0)
    rows.append({
        "kol": kol,
        "N": n,
        "win_pct":    round(win_rate, 1),
        "loss_pct":   round(loss_rate, 1),
        "neutral":    neutral,
        "pos_neg":    round(pos_neg, 2),
        "med_max_up": round(100 * (med([m["max_up"] for m in metrics]) or 0), 1),
        "med_max_dn": round(100 * (med([m["max_dn"] for m in metrics]) or 0), 1),
        "med_ret_30m": round(100 * (med([m["ret_30m"] for m in metrics]) or 0), 2),
        "med_ret_1h":  round(100 * (med([m["ret_1h"]  for m in metrics]) or 0), 2),
        "med_ret_2h":  round(100 * (med([m["ret_2h"]  for m in metrics]) or 0), 2),
    })

# Ranking: composite — win_pct desc, then pos_neg desc, then N desc.
rows.sort(key=lambda r: (-r["win_pct"], -r["pos_neg"], -r["N"]))

print(f"\n{'Rank':<5}{'KOL':<25}{'N':>5}{'Win%':>7}{'Loss%':>7}{'Neut':>6}"
      f"{'P/N':>7}{'MaxUp%':>9}{'MaxDn%':>9}{'R30m%':>8}{'R1h%':>8}{'R2h%':>8}")
print("-" * 112)
for i, r in enumerate(rows, 1):
    print(f"{i:<5}{r['kol']:<25}{r['N']:>5}{r['win_pct']:>7.1f}{r['loss_pct']:>7.1f}"
          f"{r['neutral']:>6}{r['pos_neg']:>7.2f}{r['med_max_up']:>+9.1f}"
          f"{r['med_max_dn']:>+9.1f}{r['med_ret_30m']:>+8.2f}"
          f"{r['med_ret_1h']:>+8.2f}{r['med_ret_2h']:>+8.2f}")

# Save CSV for further work
out = os.path.join(os.path.dirname(__file__), "..", "data", "kol_tick_quality.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["kol"])
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"\nSaved -> {out}  ({len(rows)} KOLs, min_N={MIN_N})")
