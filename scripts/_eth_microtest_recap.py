"""Slippage / paired-test recap of ETH live microtest.

After 5-10 closed ETH live trades, this gives the empirical answer:
  - What's the real slippage on KOL-called memecoins (vs Apr 26 PEPE smoke)?
  - What's the paper-vs-live drift in PAIRED test (apples-to-apples)?
  - Should we recalibrate ETH_BUY_SLIPPAGE_BPS / ETH_SELL_SLIPPAGE_BPS?
  - Should we scale to Phase 2 ($100/trade) or stay/abort?

Reads paper_trades chain=ethereum source=rt_live, joined with the matching
paper-main row (same token, same strategy, same cycle) for paired comparison.
"""
import os, sys, json, statistics as st
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

CLOSED = ("sl_hit", "trail_stop", "tp_hit", "timeout", "be_stop")


def fetch(src):
    out, off, step = [], 0, 1000
    while True:
        r = sb.table("paper_trades").select(
            "id,strategy,status,source,pnl_pct,pnl_usd,kol_group,token_address,symbol,"
            "created_at,is_shadow,chain,entry_price,execution_price,exit_price,"
            "buy_slippage_bps,sell_slippage_bps,buy_fee_bps,gas_usd_buy,gas_usd_sell,"
            "quote_slip_bps_buy,quote_slip_bps_sell,paper_sim_pnl_pct,position_usd,"
            "buy_exec_ms,sell_exec_ms,block_number_buy,block_number_sell"
        ).eq("source", src).eq("chain", "ethereum").range(off, off+step-1).execute()
        if not r.data:
            break
        out.extend(r.data)
        if len(r.data) < step:
            break
        off += step
    return out


live = [r for r in fetch("rt_live") if r.get("status") in CLOSED]
print("=" * 92)
print(f"ETH LIVE MICROTEST RECAP — {len(live)} closed trades")
print("=" * 92)
if not live:
    print("  no closed live trades yet. Wait for the first cycle to close.")
    sys.exit(0)

# 1. Per-trade slippage breakdown
print("\n[1] Per-trade slippage + cost breakdown")
print(f"  {'symbol':<12}{'strat':<22}{'pos$':>6}{'gasBuy':>7}{'gasSel':>7}{'qsBuy':>7}{'qsSel':>7}"
      f"{'dsBuy':>7}{'dsSel':>7}{'paperPNL':>9}{'livePNL':>9}{'diff':>7}")
for r in sorted(live, key=lambda x: x["created_at"]):
    sym = (r.get("symbol") or "?")[:11]
    strat = (r.get("strategy") or "?")[:21]
    pos = float(r.get("position_usd") or 0)
    gb = float(r.get("gas_usd_buy") or 0)
    gs = float(r.get("gas_usd_sell") or 0)
    qb = int(r.get("quote_slip_bps_buy") or 0)
    qs = int(r.get("quote_slip_bps_sell") or 0)
    dsb = int(r.get("buy_slippage_bps") or 0)
    dss = int(r.get("sell_slippage_bps") or 0)
    pp = float(r.get("paper_sim_pnl_pct") or 0) * 100
    lp = float(r.get("pnl_pct") or 0) * 100
    diff = lp - pp
    print(f"  {sym:<12}{strat:<22}{pos:>6.0f}{gb:>+7.2f}{gs:>+7.2f}{qb:>+7}{qs:>+7}"
          f"{dsb:>+7}{dss:>+7}{pp:>+8.2f}%{lp:>+8.2f}%{diff:>+6.1f}")

# 2. Aggregate slippage stats
print("\n[2] Slippage aggregates (median preferred — heavy-tailed)")
gb_list = [float(r.get("gas_usd_buy") or 0) for r in live if r.get("gas_usd_buy")]
gs_list = [float(r.get("gas_usd_sell") or 0) for r in live if r.get("gas_usd_sell")]
qb_list = [int(r.get("quote_slip_bps_buy") or 0) for r in live]
qs_list = [int(r.get("quote_slip_bps_sell") or 0) for r in live]
dsb_list = [int(r.get("buy_slippage_bps") or 0) for r in live]
dss_list = [int(r.get("sell_slippage_bps") or 0) for r in live]
def s(label, xs):
    if not xs: return f"  {label:<32} no data"
    return f"  {label:<32} median={st.median(xs):>+8.2f}  mean={st.mean(xs):>+8.2f}  N={len(xs)}"
print(s("gas_usd_buy",        gb_list))
print(s("gas_usd_sell",       gs_list))
print(s("quote_slip_bps_buy",  qb_list))
print(s("quote_slip_bps_sell", qs_list))
print(s("dexscr_slip_bps_buy", dsb_list))
print(s("dexscr_slip_bps_sel", dss_list))

# 3. Paired-test drift
print("\n[3] Paired-test drift (apples-to-apples on intersection paper-vs-live)")
pairs = []
for r in live:
    pp = r.get("paper_sim_pnl_pct")
    if pp is None: continue
    lp = float(r.get("pnl_pct") or 0)
    pairs.append({
        "symbol": r.get("symbol"),
        "paper_pct": float(pp) * 100,
        "live_pct":  lp * 100,
        "drift_pp":  (lp - float(pp)) * 100,
    })
if not pairs:
    print("  no paired observations yet (paper_sim_pnl_pct missing) — needs paper run alongside")
else:
    drifts = [p["drift_pp"] for p in pairs]
    print(f"  N pairs = {len(pairs)}")
    print(f"  paired drift mean   = {st.mean(drifts):+.2f}pp")
    print(f"  paired drift median = {st.median(drifts):+.2f}pp")
    if len(drifts) >= 3:
        print(f"  paired drift std    = {st.pstdev(drifts):.2f}pp")

# 4. Verdict + scale-up recommendation
print("\n[4] Verdict")
n = len(live)
if n < 5:
    print("  insufficient N — keep collecting (target 10 closed trades)")
elif pairs:
    md = st.median([p["drift_pp"] for p in pairs])
    if md > -3:
        print(f"  drift median {md:+.2f}pp = ACCEPTABLE — consider Phase 2 ($100/trade) at N>=15")
    elif md > -7:
        print(f"  drift median {md:+.2f}pp = MARGINAL — collect more N before scaling")
    else:
        print(f"  drift median {md:+.2f}pp = BAD — abort, recalibrate ETH slip constants empirically")
        print("  Suggested: set ETH_BUY_SLIPPAGE_BPS = median(quote_slip_bps_buy) + 50 buffer")
        if qb_list:
            print(f"  Empirical median quote_slip_buy = {st.median(qb_list):.0f}bps -> calibrate to ~{int(st.median(qb_list))+50}")
