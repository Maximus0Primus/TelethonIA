"""Reconcile rt_bankroll against paper_trades ground truth.

For each (chain, strategy) entry in `rt_bankroll.strategy_bankrolls_per_chain`,
recompute SUM(pnl_usd) + COUNT(*) from `paper_trades` where
  source='rt' AND is_shadow=false AND status IN ('tp_hit','sl_hit','timeout','trail_stop','be_stop')
Compare to the claimed bankroll values, flag drift, and (optionally) patch the DB.

Usage:
  python scripts/_reconcile_bankrolls.py                # dry-run, prints table
  python scripts/_reconcile_bankrolls.py --apply        # write fixes to DB (with backup)
  python scripts/_reconcile_bankrolls.py --alert        # send Telegram if any drift > $100

Backup auto-created at `data/rt_bankroll_pre_reconcile_<TS>.json` before any write.

Baseline note: paper main P&L is summed from the start of the strategy's bankroll lifecycle.
If a strategy was reset (bankroll wiped + re-seeded), pnl_usd from BEFORE the reset must be
excluded. We approximate this by using `starting_balance` as the anchor and reading the
`since_at` field if present (added by reset scripts since v138.3). For ETH, the bankroll
dict uses keys `balance`/`pnl`/`trades`, not `current_balance`/`total_pnl`/`total_trades`.

Reference: tasks/dedup_rules.md — paper main rows are NEVER re-deduped (paper_trader.py
already blocks re-entry while a position is open). We just sum them.
"""
from __future__ import annotations
import os, sys, io, json, argparse
from pathlib import Path
from datetime import datetime, timezone

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / "scraper" / ".env")
except ImportError:
    pass

import requests
from supabase import create_client

SB_URL = os.environ.get("SUPABASE_URL")
SB_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
if not SB_URL or not SB_KEY:
    print("FATAL: SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY missing")
    sys.exit(1)

TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("MONITOR_BOT_TOKEN")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID") or os.environ.get("MONITOR_CHAT_ID")

sb = create_client(SB_URL, SB_KEY)
NOW = datetime.now(timezone.utc)
DRIFT_THRESHOLD = 1.0       # flag any |drift| > $1
ALERT_THRESHOLD = 100.0     # telegram alert when |drift| > $100


# ──────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true", help="Write fixes to DB (default: dry-run)")
    p.add_argument("--alert", action="store_true", help="Send Telegram alert if drift > $100")
    p.add_argument("--since", default=None, help="ISO ts to start summing pnl_usd (default: use starting balance baseline)")
    return p.parse_args()


def pull_paginated(query_builder):
    rows = []
    offset = 0
    while True:
        res = query_builder.range(offset, offset + 999).execute()
        b = res.data or []
        rows.extend(b)
        if len(b) < 1000:
            break
        offset += 1000
    return rows


def get_bankroll_metrics(d):
    """Handle both bankroll dict conventions (SOL vs ETH)."""
    if not isinstance(d, dict):
        return None, None, None, None
    # SOL convention
    if "current_balance" in d or "total_pnl" in d:
        balance = d.get("current_balance")
        pnl = d.get("total_pnl")
        trades = d.get("total_trades")
        start = d.get("starting_balance")
        return balance, pnl, trades, start
    # ETH convention
    if "balance" in d or "pnl" in d:
        balance = d.get("balance")
        pnl = d.get("pnl")
        trades = d.get("trades")
        start = d.get("starting_balance")
        return balance, pnl, trades, start
    return None, None, None, None


def write_bankroll_metrics(d, pnl, trades, balance):
    """Write back metrics into bankroll dict preserving its convention."""
    if "current_balance" in d or "total_pnl" in d:
        d["total_pnl"] = pnl
        d["total_trades"] = trades
        d["current_balance"] = balance
    else:
        d["pnl"] = pnl
        d["trades"] = trades
        d["balance"] = balance
    d["last_updated_at"] = NOW.isoformat()
    return d


def send_telegram(text):
    if not TG_TOKEN or not TG_CHAT:
        print("[telegram] creds missing — skipping")
        return False
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    if len(text) > 3900:
        text = text[:3900] + "\n...[truncated]"
    try:
        r = requests.post(url, data={
            "chat_id": TG_CHAT,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }, timeout=10)
        if r.status_code == 200:
            print("[telegram] sent OK")
            return True
        print(f"[telegram] HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"[telegram] error: {e}")
    return False


# ──────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    print(f"=== Bankroll Reconcile — {NOW.isoformat()} ===")
    print(f"Mode: {'APPLY (writes DB)' if args.apply else 'DRY-RUN (no writes)'}")

    # 1. Pull current bankroll
    br_row = sb.table("rt_bankroll").select("*").execute().data
    if not br_row:
        print("FATAL: rt_bankroll table empty")
        sys.exit(1)
    br = br_row[0]
    per_chain = br.get("strategy_bankrolls_per_chain", {})

    # 2. Backup
    backup_dir = Path(__file__).resolve().parent.parent / "data"
    backup_dir.mkdir(exist_ok=True)
    ts = NOW.strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_dir / f"rt_bankroll_pre_reconcile_{ts}.json"
    backup_path.write_text(json.dumps(br, indent=2, default=str))
    print(f"Backup → {backup_path}")

    # 3. For each (chain, strategy), recompute
    rows_for_table = []
    drift_total_abs = 0.0
    max_drift = ("", "", 0.0)

    for chain, strategies in per_chain.items():
        if not isinstance(strategies, dict):
            continue
        for strat, bk in strategies.items():
            claimed_balance, claimed_pnl, claimed_trades, starting = get_bankroll_metrics(bk)
            if claimed_pnl is None:
                continue
            # Pull all closed paper_main trades
            q = sb.table("paper_trades").select(
                "id,pnl_usd,status,created_at"
            ).eq("chain", chain).eq("strategy", strat).eq("source", "rt").eq("is_shadow", False).in_(
                "status", ["tp_hit", "sl_hit", "timeout", "trail_stop", "be_stop"]
            ).order("created_at")
            if args.since:
                q = q.gte("created_at", args.since)
            trades = pull_paginated(q)
            actual_pnl = sum(float(t.get("pnl_usd") or 0) for t in trades)
            actual_count = len(trades)
            actual_balance = (starting or 0) + actual_pnl

            drift_pnl = actual_pnl - float(claimed_pnl or 0)
            drift_balance = actual_balance - float(claimed_balance or 0) if claimed_balance is not None else 0

            status = "OK"
            if abs(drift_pnl) > DRIFT_THRESHOLD:
                status = "DRIFT"
                drift_total_abs += abs(drift_pnl)
                if abs(drift_pnl) > abs(max_drift[2]):
                    max_drift = (chain, strat, drift_pnl)

            rows_for_table.append((
                chain, strat,
                float(claimed_pnl or 0), actual_pnl, drift_pnl,
                int(claimed_trades or 0), actual_count,
                status, bk,
            ))

    # 4. Print table
    print(f"\n{'chain':<10} {'strategy':<40} {'claimed_pnl':>12} {'actual_pnl':>12} {'drift':>10} {'cl_N':>5} {'ac_N':>5} {'status':<8}")
    print("-" * 110)
    drift_rows = [r for r in rows_for_table if r[7] == "DRIFT"]
    for chain, strat, cp, ap, dr, cn, an, st, _bk in sorted(rows_for_table, key=lambda x: -abs(x[4])):
        print(f"{chain:<10} {strat:<40} {cp:>+12.2f} {ap:>+12.2f} {dr:>+10.2f} {cn:>5} {an:>5} {st:<8}")

    n_drift = len(drift_rows)
    print(f"\nSummary: {n_drift}/{len(rows_for_table)} entries with drift > ${DRIFT_THRESHOLD:.0f}")
    print(f"Total abs drift: ${drift_total_abs:.2f}")
    if max_drift[0]:
        print(f"Max single drift: {max_drift[0]}/{max_drift[1]} = ${max_drift[2]:+.2f}")

    # 5. Apply (with backup) if requested
    if args.apply and n_drift > 0:
        print(f"\n--- Applying fixes ({n_drift} entries) ---")
        for chain, strat, cp, ap, dr, cn, an, st, bk in drift_rows:
            # Mutate the dict in-place (sits inside per_chain)
            new_pnl = ap
            new_trades = an
            starting = bk.get("starting_balance") or 0
            new_balance = starting + new_pnl
            write_bankroll_metrics(bk, new_pnl, new_trades, new_balance)
            print(f"  patched {chain}/{strat}: pnl {cp:+.2f} → {new_pnl:+.2f}  trades {cn} → {new_trades}  bal → {new_balance:.2f}")
        # Single write
        sb.table("rt_bankroll").update({
            "strategy_bankrolls_per_chain": per_chain,
            "last_updated_at": NOW.isoformat(),
        }).eq("id", br.get("id", 1)).execute()
        print(f"\n✓ rt_bankroll updated. Backup at {backup_path}")
    elif args.apply:
        print("\nNo drift > threshold — nothing to apply.")
    else:
        print("\n(dry-run — pass --apply to write fixes)")

    # 6. Alert
    if args.alert and drift_total_abs > ALERT_THRESHOLD:
        msg_lines = [
            f"*Bankroll Drift Alert — {NOW.date().isoformat()}*",
            f"Total abs drift: ${drift_total_abs:.2f} ({n_drift} entries)",
            f"Max: {max_drift[0]}/{max_drift[1]} = ${max_drift[2]:+.2f}",
            "",
            "Top offenders:",
        ]
        for chain, strat, cp, ap, dr, cn, an, st, _bk in sorted(drift_rows, key=lambda x: -abs(x[4]))[:5]:
            msg_lines.append(f"- {chain}/{strat}: claim ${cp:+.2f} → actual ${ap:+.2f} (drift ${dr:+.2f})")
        msg_lines.append("")
        msg_lines.append("Run `python scripts/_reconcile_bankrolls.py --apply` to fix")
        send_telegram("\n".join(msg_lines))

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
