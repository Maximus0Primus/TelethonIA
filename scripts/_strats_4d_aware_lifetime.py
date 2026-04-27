"""4-day strategy ranking that respects each strat's actual activation date.

For each strategy, we:
  1. Find its first observed trade (within or before the 4d window)
  2. Compute $/day on its actual active window inside the 4d cutoff
  3. Show: total $, $/day pro-rated, N, family

This avoids penalizing LOCK / AGE24 / v14e.32 cluster strats that were activated
only hours ago — they get judged on their lifetime, not on 4 calendar days.
"""
import os, sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

NOW = datetime.now(timezone.utc)
SINCE_4D = NOW - timedelta(days=4)
SINCE_4D_ISO = SINCE_4D.isoformat()
CLOSED = ("sl_hit", "trail_stop", "tp_hit", "timeout", "be_stop")


def fetch_all(tbl, sel, **filters):
    out, off, step = [], 0, 1000
    while True:
        q = sb.table(tbl).select(sel).gte("created_at", SINCE_4D_ISO)
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


def family(strat):
    if strat.startswith("ETH_"):
        s = strat[4:]
    else:
        s = strat
    if "AGE24" in s or s.startswith("AGE24"): return "AGE24"
    if "AGE48" in s: return "AGE48"
    if "AGE72" in s: return "AGE72"
    if s.startswith("SCALP"): return "SCALP"
    if s.startswith("DECAY"): return "DECAY"
    if "LOCK" in s: return "LOCK"
    if s.startswith("DTRAIL"): return "DTRAIL"
    if s.startswith("TRAIL"): return "TRAIL"
    if s.startswith("DIP"): return "DIP"
    if s.startswith("BE"): return "BE"
    if s.startswith("FAST"): return "FAST"
    if s.startswith("SLOW"): return "SLOW"
    if s.startswith("BOND"): return "BOND"
    return "OTHER"


def run(chain_db, chain_label):
    rows = fetch_all(
        "paper_trades",
        "strategy,status,pnl_pct,pnl_usd,created_at",
        source="rt", chain=chain_db,
    )
    # Per strat: first_seen, last_seen, closed trades
    by_strat = defaultdict(list)
    first_seen = {}
    for r in rows:
        s = r["strategy"]
        ts = r["created_at"]
        if s not in first_seen or ts < first_seen[s]:
            first_seen[s] = ts
        if r.get("status") in CLOSED:
            by_strat[s].append(r)

    rows_out = []
    for s, xs in by_strat.items():
        n = len(xs)
        if n < 2:
            continue
        usd = sum(float(x.get("pnl_usd") or 0) for x in xs)
        avg_pct = sum(float(x.get("pnl_pct") or 0) * 100 for x in xs) / n
        wr = 100 * sum(1 for x in xs if float(x.get("pnl_pct") or 0) > 0) / n
        # Lifetime in 4d window: from first_seen (or window start) → NOW
        fs_dt = datetime.fromisoformat(first_seen[s].replace("Z", "+00:00"))
        if fs_dt < SINCE_4D:
            fs_dt = SINCE_4D
        lifetime_h = max((NOW - fs_dt).total_seconds() / 3600, 0.5)
        per_day = usd * 24 / lifetime_h
        rows_out.append({
            "strat": s, "fam": family(s), "n": n, "wr": wr, "avg": avg_pct,
            "usd": usd, "lifetime_h": lifetime_h, "per_day": per_day,
            "first_seen": first_seen[s][:19],
        })
    rows_out.sort(key=lambda r: -r["per_day"])

    print()
    print("=" * 110)
    print(f"{chain_label} — last 4d, lifetime-aware $/jour (N>=2)")
    print("=" * 110)
    print(f"{'#':>3}  {'strat':<42}{'fam':<8}{'N':>4}{'avg%':>9}{'WR':>5}"
          f"{'$ tot':>10}{'live h':>8}{'$/jour':>10}  {'first seen UTC':<20}")
    print("-" * 110)
    for i, r in enumerate(rows_out[:25], 1):
        print(f"{i:>3}  {r['strat']:<42}{r['fam']:<8}{r['n']:>4}"
              f"{r['avg']:>+8.2f}%{r['wr']:>4.0f}%{r['usd']:>+9.2f}"
              f"{r['lifetime_h']:>7.1f}h{r['per_day']:>+9.2f}  {r['first_seen']:<20}")

    # Family aggregate
    print()
    print(f"--- {chain_label} family aggregate (lifetime-weighted $/jour) ---")
    fam_agg = defaultdict(lambda: {"usd": 0, "h": 0, "n": 0, "best": None, "best_pd": -1e9})
    for r in rows_out:
        f = fam_agg[r["fam"]]
        f["usd"] += r["usd"]
        f["h"] = max(f["h"], r["lifetime_h"])  # family lifetime = oldest variant
        f["n"] += r["n"]
        if r["per_day"] > f["best_pd"]:
            f["best_pd"] = r["per_day"]
            f["best"] = r["strat"]
    print(f"  {'family':<10}{'N tot':>7}{'$ tot':>10}{'h max':>8}{'$/j famille':>14}  {'best':<35}{'best $/j':>11}")
    for fam, d in sorted(fam_agg.items(), key=lambda kv: -(kv[1]['usd']*24/max(kv[1]['h'],0.5))):
        per_day_fam = d["usd"] * 24 / max(d["h"], 0.5)
        print(f"  {fam:<10}{d['n']:>7}{d['usd']:>+9.2f}{d['h']:>7.1f}h{per_day_fam:>+13.2f}"
              f"  {d['best']:<35}{d['best_pd']:>+10.2f}")


run("solana", "SOL")
run("ethereum", "ETH")
