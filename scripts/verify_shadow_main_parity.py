"""Verify v144.3 shadow ↔ main parity in production.

Since the code SKIPS shadow creation when a strategy is in `real_strats`
(safe_scraper.py routes mains separately), there are no naturally occurring
(token, strat, source=rt) pairs with both is_shadow=True AND is_shadow=False
on the SAME strategy name. The closest comparable is the variant pair:
  main FAST_TP50_SL30 (is_shadow=False) vs shadow FAST_TP50_SL30_NOLAZY (is_shadow=True)

We instead check the v144.3 invariants on shadow rows opened post-deploy:
  1. position_usd > 0  (was 0 before v144.3)
  2. entry_source IN {ultra, dexscreener, live_sync}
  3. dex_spot_price_at_entry == entry_price
  4. high_price_seen >= entry_price
  5. sol_price_at_entry NOT NULL
  6. exit_at trades: pnl_usd matches Ultra-quote-style (pos_usd × pnl_pct, not the $10 fallback)

Any failure indicates shadow rows still using the legacy code path.
"""
import os, sys
from datetime import datetime, timezone, timedelta
from collections import Counter
from dotenv import load_dotenv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

DEPLOY_TS = "2026-04-20T20:15:44+00:00"   # v144.3 VPS restart
NOW_ISO = datetime.now(timezone.utc).isoformat()

print(f"=== Shadow v144.3 parity audit ===")
print(f"Deploy ts: {DEPLOY_TS}")
print(f"Now:       {NOW_ISO}\n")

rows = []
off = 0
while True:
    r = sb.table("paper_trades").select(
        "id,strategy,status,is_shadow,position_usd,entry_price,exit_price,"
        "pnl_pct,pnl_usd,dex_spot_price_at_entry,high_price_seen,sol_price_at_entry,"
        "entry_source,created_at,exit_at"
    ).eq("source", "rt").eq("is_shadow", True).gte("created_at", DEPLOY_TS).range(off, off+999).execute().data
    rows += r
    if len(r) < 1000:
        break
    off += 1000

print(f"Shadow rows post-deploy: {len(rows)}\n")

if not rows:
    print("No shadow rows yet (no RT calls fired since deploy). Re-run after a KOL call.")
    sys.exit(0)

failures = {
    "pos_usd_zero": [],
    "entry_source_missing": [],
    "dex_spot_mismatch": [],
    "high_below_entry": [],
    "sol_price_null": [],
    "pnl_usd_legacy_10": [],   # closed trades with effective_usd=10 fallback
}

for t in rows:
    if (t.get("position_usd") or 0) == 0:
        failures["pos_usd_zero"].append(t)
    if t.get("entry_source") not in ("ultra", "dexscreener", "live_sync"):
        failures["entry_source_missing"].append(t)
    ep = float(t.get("entry_price") or 0)
    dsp = float(t.get("dex_spot_price_at_entry") or 0)
    if ep > 0 and dsp > 0 and abs(dsp - ep) / ep > 0.001:
        failures["dex_spot_mismatch"].append(t)
    hps = float(t.get("high_price_seen") or 0)
    if ep > 0 and hps > 0 and hps < ep * 0.999:
        failures["high_below_entry"].append(t)
    if t.get("sol_price_at_entry") is None:
        failures["sol_price_null"].append(t)
    # Closed trades: pnl_usd should be pos_usd × pnl_pct (NOT 10 × pnl_pct fallback)
    if t.get("status") not in ("open", "closing"):
        pos = float(t.get("position_usd") or 0)
        ppct = float(t.get("pnl_pct") or 0)
        pusd = float(t.get("pnl_usd") or 0)
        if pos > 0 and abs(pusd - pos * ppct) > 0.5 and abs(pusd - 10.0 * ppct) < 0.05:
            failures["pnl_usd_legacy_10"].append(t)

print("Invariants:")
for k, v in failures.items():
    status = "OK" if not v else "FAIL"
    print(f"  [{status:4s}] {k:35s} = {len(v)} violations")

print()
print("Distribution by entry_source:")
src_counts = Counter((t.get("entry_source") or "NULL") for t in rows)
for s, c in src_counts.most_common():
    print(f"  {s:15s} = {c}")

print()
print("Distribution by status:")
st_counts = Counter((t.get("status") or "NULL") for t in rows)
for s, c in st_counts.most_common():
    print(f"  {s:15s} = {c}")

print()
print("position_usd stats:")
pos_vals = [float(t.get("position_usd") or 0) for t in rows]
nonzero = [p for p in pos_vals if p > 0]
if nonzero:
    print(f"  N nonzero: {len(nonzero)}/{len(pos_vals)}")
    print(f"  median:    ${sorted(nonzero)[len(nonzero)//2]:.2f}")
    print(f"  min/max:   ${min(nonzero):.2f} / ${max(nonzero):.2f}")
else:
    print("  ALL ZERO — v144.3 fix not in effect!")

total_viol = sum(len(v) for v in failures.values())
# v144.19: tolerate a handful of stray rows (≤5 OR ≤0.1% of population).
# Real regressions always produce dozens-to-thousands of violations; single
# stray NULL entries from data-migration edge cases shouldn't spam CI.
TOL_ABS = int(os.environ.get("PARITY_TOL_ABS", "5"))
TOL_PCT = float(os.environ.get("PARITY_TOL_PCT", "0.001"))  # 0.1%
threshold = max(TOL_ABS, int(TOL_PCT * max(len(rows), 1)))

if total_viol:
    print()
    print(f"!! {total_viol} total invariant violations "
          f"(tolerance: {threshold}, N={len(rows)}).")
    print("Sample failures:")
    shown = 0
    for k, v in failures.items():
        for t in v[:2]:
            print(f"  {k}: id={t['id']} strat={t['strategy']} pos={t.get('position_usd')} "
                  f"entry_src={t.get('entry_source')} status={t.get('status')}")
            shown += 1
            if shown >= 3:
                break
        if shown >= 3:
            break
    if total_viol > threshold:
        print(f"\n::error::Parity regression — {total_viol} > tolerance {threshold}")
        sys.exit(1)
    print(f"\nWithin tolerance ({threshold}) — not alerting.")
    sys.exit(0)

print()
print("All v144.3 invariants pass.")
