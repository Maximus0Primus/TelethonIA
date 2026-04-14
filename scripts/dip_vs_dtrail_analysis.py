"""
Compare DIP30 vs DTRAIL strategies from live paper trading data.
"""
import os, sys, statistics
from collections import defaultdict

# Load env
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))

from supabase import create_client

url = os.environ["SUPABASE_URL"]
key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
sb = create_client(url, key)

# Fetch all closed trades for DIP and DTRAIL strategies
# Status is the exit reason: sl_hit, tp_hit, trail_stop, timeout
# "open" means still running - we want everything that's NOT open
def fetch_trades(pattern: str):
    """Fetch closed trades matching a strategy LIKE pattern."""
    all_rows = []
    offset = 0
    page = 1000
    while True:
        resp = (
            sb.table("paper_trades")
            .select("id, strategy, tranche_label, kol_group, status, pnl_pct, pnl_usd, entry_price, exit_price, exit_minutes, symbol, position_usd, created_at")
            .neq("status", "open")
            .like("strategy", pattern)
            .range(offset, offset + page - 1)
            .execute()
        )
        rows = resp.data or []
        all_rows.extend(rows)
        if len(rows) < page:
            break
        offset += page
    return all_rows

print("Fetching DIP30 trades...")
dip_trades = fetch_trades("DIP30%")
print(f"  -> {len(dip_trades)} rows")

print("Fetching DTRAIL trades...")
dtrail_trades = fetch_trades("DTRAIL%")
print(f"  -> {len(dtrail_trades)} rows")

if not dip_trades and not dtrail_trades:
    print("\nNo closed trades found for either strategy group. Exiting.")
    sys.exit(0)

# ── Helper ──────────────────────────────────────────────────────────────
def compute_stats(trades):
    if not trades:
        return None
    pnls = [t["pnl_pct"] for t in trades if t.get("pnl_pct") is not None]
    pnl_usds = [t["pnl_usd"] for t in trades if t.get("pnl_usd") is not None]
    if not pnls:
        return None
    wins = [p for p in pnls if p > 0]
    return {
        "n": len(pnls),
        "wr": f"{len(wins)/len(pnls)*100:.1f}%",
        "avg_pnl": f"{statistics.mean(pnls)*100:.2f}%",
        "med_pnl": f"{statistics.median(pnls)*100:.2f}%",
        "total_usd": f"${sum(pnl_usds):.2f}" if pnl_usds else "N/A",
        "max_win": f"{max(pnls)*100:.1f}%" if pnls else "N/A",
        "max_loss": f"{min(pnls)*100:.1f}%" if pnls else "N/A",
        "avg_exit_min": f"{statistics.mean([t['exit_minutes'] for t in trades if t.get('exit_minutes') is not None]):.0f}m" if any(t.get('exit_minutes') for t in trades) else "N/A",
    }

def print_table(title, headers, rows, col_widths=None):
    if not col_widths:
        col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) + 2 for i, h in enumerate(headers)]
    print(f"\n{'='*sum(col_widths)}")
    print(f"  {title}")
    print(f"{'='*sum(col_widths)}")
    header_line = "".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * sum(col_widths))
    for row in rows:
        print("".join(str(row[i]).ljust(w) for i, w in enumerate(col_widths)))

# ── 1. Strategy-level comparison ────────────────────────────────────────
print("\n")
by_strategy = defaultdict(list)
for t in dip_trades + dtrail_trades:
    by_strategy[t["strategy"]].append(t)

rows_1 = []
for strat in sorted(by_strategy.keys()):
    s = compute_stats(by_strategy[strat])
    if s:
        rows_1.append([strat, s["n"], s["wr"], s["avg_pnl"], s["med_pnl"], s["total_usd"], s["max_win"], s["max_loss"], s["avg_exit_min"]])

headers_1 = ["Strategy", "N", "WR", "Avg PnL%", "Med PnL%", "Total $", "Max Win", "Max Loss", "Avg Exit"]
print_table("STRATEGY COMPARISON: DIP30 vs DTRAIL", headers_1, rows_1)

# ── 2. Strategy GROUP comparison (DIP vs DTRAIL aggregated) ─────────────
print("\n")
group_rows = []
for label, trades in [("DIP30 (all)", dip_trades), ("DTRAIL (all)", dtrail_trades)]:
    s = compute_stats(trades)
    if s:
        group_rows.append([label, s["n"], s["wr"], s["avg_pnl"], s["med_pnl"], s["total_usd"], s["max_win"], s["max_loss"], s["avg_exit_min"]])

print_table("AGGREGATE: DIP30 vs DTRAIL groups", headers_1, group_rows)

# ── 3. DIP tranche breakdown (P1 vs P2) ────────────────────────────────
if dip_trades:
    by_tranche = defaultdict(list)
    for t in dip_trades:
        label = t.get("tranche_label") or "unknown"
        by_tranche[label].append(t)

    tranche_rows = []
    for lbl in sorted(by_tranche.keys()):
        s = compute_stats(by_tranche[lbl])
        if s:
            tranche_rows.append([lbl, s["n"], s["wr"], s["avg_pnl"], s["med_pnl"], s["total_usd"], s["max_win"], s["max_loss"]])

    headers_t = ["Tranche", "N", "WR", "Avg PnL%", "Med PnL%", "Total $", "Max Win", "Max Loss"]
    print_table("DIP30 TRANCHE BREAKDOWN (P1 vs P2)", headers_t, tranche_rows)

# ── 4. DIP per-KOL breakdown ───────────────────────────────────────────
if dip_trades:
    by_kol = defaultdict(list)
    for t in dip_trades:
        kol = t.get("kol_group") or "unknown"
        by_kol[kol].append(t)

    kol_rows = []
    for kol in sorted(by_kol.keys()):
        s = compute_stats(by_kol[kol])
        if s:
            kol_rows.append([kol, s["n"], s["wr"], s["avg_pnl"], s["med_pnl"], s["total_usd"], s["max_win"], s["max_loss"]])

    # Sort by total USD descending
    kol_rows.sort(key=lambda r: float(r[5].replace("$", "").replace(",", "")), reverse=True)

    headers_k = ["KOL", "N", "WR", "Avg PnL%", "Med PnL%", "Total $", "Max Win", "Max Loss"]
    print_table("DIP30 PER-KOL BREAKDOWN", headers_k, kol_rows)

# ── 5. Exit reason distribution ────────────────────────────────────────
print("\n")
for label, trades in [("DIP30", dip_trades), ("DTRAIL", dtrail_trades)]:
    if not trades:
        continue
    exit_counts = defaultdict(int)
    for t in trades:
        exit_counts[t.get("status", "unknown")] += 1
    total = sum(exit_counts.values())
    print(f"\n  EXIT REASONS — {label} ({total} trades)")
    print(f"  {'-'*45}")
    for reason in sorted(exit_counts.keys(), key=lambda k: -exit_counts[k]):
        cnt = exit_counts[reason]
        pct = cnt / total * 100
        # Also compute avg PnL for this exit reason
        subset = [t for t in trades if t.get("status") == reason]
        avg_p = statistics.mean([t["pnl_pct"] for t in subset if t.get("pnl_pct") is not None]) * 100 if subset else 0
        print(f"  {reason:<16} {cnt:>4} ({pct:5.1f}%)   avg pnl: {avg_p:+.1f}%")

# ── 6. DTRAIL per-KOL breakdown (top 15) ──────────────────────────────
if dtrail_trades:
    by_kol_dt = defaultdict(list)
    for t in dtrail_trades:
        kol = t.get("kol_group") or "unknown"
        by_kol_dt[kol].append(t)

    kol_dt_rows = []
    for kol in sorted(by_kol_dt.keys()):
        s = compute_stats(by_kol_dt[kol])
        if s:
            kol_dt_rows.append([kol, s["n"], s["wr"], s["avg_pnl"], s["med_pnl"], s["total_usd"], s["max_win"], s["max_loss"]])

    kol_dt_rows.sort(key=lambda r: float(r[5].replace("$", "").replace(",", "")), reverse=True)

    headers_k2 = ["KOL", "N", "WR", "Avg PnL%", "Med PnL%", "Total $", "Max Win", "Max Loss"]
    print_table("DTRAIL PER-KOL BREAKDOWN (top 15 by total $)", headers_k2, kol_dt_rows[:15])

print("\n\nDone.")
