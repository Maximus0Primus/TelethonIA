"""Best strategy per chain (SOL/ETH) over multiple time windows.

Use paper RT shadow trades (source='rt') as the proxy for what live would have done
on the SAME token universe. All 127 shadow strategies run on every RT token, so
ranking them is equivalent to asking 'which strategy should I have run live'.

Periods: all-time, 7d, 3d, last 10h.
Chains:  solana, ethereum.

Metric ranking: $/day at paper position size (typically $50/trade) — the dollar
P&L is what matters, not avg%. We also report avg%, WR, N for context.
"""
import os, sys, statistics as st
from collections import defaultdict
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

NOW = datetime.now(timezone.utc)
PERIODS = [
    ("ALL",      None,                          None),
    ("7d",       NOW - timedelta(days=7),       7.0),
    ("3d",       NOW - timedelta(days=3),       3.0),
    ("10h",      NOW - timedelta(hours=10),     10/24),
]
CLOSED = ("sl_hit", "trail_stop", "tp_hit", "timeout", "be_stop")
TOP_N = 10
MIN_N_DISPLAY = 3  # don't display strats with less than 3 trades to avoid noise


def fetch_all(tbl, sel, since_iso=None, **filters):
    out, step, off = [], 1000, 0
    while True:
        q = sb.table(tbl).select(sel)
        if since_iso:
            q = q.gte("created_at", since_iso)
        for k, v in filters.items():
            q = q.eq(k, v)
        r = q.range(off, off + step - 1).execute()
        if not r.data:
            break
        out.extend(r.data)
        if len(r.data) < step:
            break
        off += step
    return out


def rank(rows, period_days):
    """Group by strategy, compute stats, rank by total $."""
    by_strat = defaultdict(list)
    for r in rows:
        if r.get("status") not in CLOSED:
            continue
        by_strat[r["strategy"]].append(r)

    out = []
    for s, xs in by_strat.items():
        n = len(xs)
        if n < MIN_N_DISPLAY:
            continue
        pnls = [float(x.get("pnl_pct") or 0) * 100 for x in xs]
        usd = sum(float(x.get("pnl_usd") or 0) for x in xs)
        wr = 100 * sum(1 for p in pnls if p > 0) / n
        avg = st.mean(pnls)
        per_day = usd / period_days if period_days else None
        out.append({
            "strat": s, "n": n, "wr": wr, "avg": avg,
            "usd_total": usd, "per_day": per_day,
        })
    # Rank by total $ (or per-day if available)
    out.sort(key=lambda r: (r["per_day"] if r["per_day"] is not None else r["usd_total"]),
             reverse=True)
    return out


def print_ranking(title, ranking, period_days):
    print()
    print("=" * 92)
    print(title)
    print("=" * 92)
    if not ranking:
        print("  (no closed trades)")
        return
    if period_days:
        hdr = f"{'#':>3}  {'Strategy':<34}{'N':>6}{'WR%':>7}{'avg%':>9}{'$ tot':>11}{'$/jour':>10}"
    else:
        hdr = f"{'#':>3}  {'Strategy':<34}{'N':>6}{'WR%':>7}{'avg%':>9}{'$ tot':>11}"
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(ranking[:TOP_N], 1):
        if period_days:
            print(f"{i:>3}  {r['strat']:<34}{r['n']:>6}{r['wr']:>6.1f}%"
                  f"{r['avg']:>+8.2f}%{r['usd_total']:>+10.2f}{r['per_day']:>+9.2f}")
        else:
            print(f"{i:>3}  {r['strat']:<34}{r['n']:>6}{r['wr']:>6.1f}%"
                  f"{r['avg']:>+8.2f}%{r['usd_total']:>+10.2f}")
    if len(ranking) > TOP_N:
        print(f"  ... ({len(ranking) - TOP_N} more strats with N>={MIN_N_DISPLAY})")


def main():
    # Fetch absolute earliest period needed
    earliest = min((p[1] for p in PERIODS if p[1] is not None), default=None)
    earliest_iso = earliest.isoformat() if earliest else None

    print(f"Fetching paper RT shadow trades since {earliest_iso or 'BEGINNING'} ...")
    # We fetch all chains in one query, filter in Python
    all_rows = fetch_all(
        "paper_trades",
        "strategy,status,source,pnl_pct,pnl_usd,chain,created_at",
        since_iso=earliest_iso,
        source="rt",
    )
    # For "ALL" we need everything — re-fetch separately if we limited by date
    if earliest_iso:
        all_time_rows = fetch_all(
            "paper_trades",
            "strategy,status,source,pnl_pct,pnl_usd,chain,created_at",
            since_iso=None,
            source="rt",
        )
    else:
        all_time_rows = all_rows

    print(f"  total RT shadow rows fetched (recent window): {len(all_rows)}")
    print(f"  total RT shadow rows fetched (all-time):       {len(all_time_rows)}")

    chain_norms = {"solana": "SOL", "ethereum": "ETH"}

    for chain_db, chain_label in chain_norms.items():
        for label, since_dt, days in PERIODS:
            src = all_time_rows if since_dt is None else all_rows
            sub = [r for r in src if r.get("chain") == chain_db]
            if since_dt is not None:
                since_iso = since_dt.isoformat()
                sub = [r for r in sub if r.get("created_at") and r["created_at"] >= since_iso]
            ranking = rank(sub, days)
            title = f"{chain_label} — {label}  (paper RT shadows = proxy live, $50/trade)"
            print_ranking(title, ranking, days)


if __name__ == "__main__":
    main()
