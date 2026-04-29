"""v14e.43 ETH slippage empirical calibration.

Mirror of `_calibrate_buy_slip.py` (SOL) but for ETH live trades. Reads
paper_trades chain=ethereum source=rt_live, dumps per-trade cost breakdown
(gas + Uniswap router quote slip + DexScreener slip), aggregates the
distribution, and proposes new ETH_BUY_SLIPPAGE_BPS / ETH_SELL_SLIPPAGE_BPS
constants for `scraper/strategies.py`.

Why: paper sim uses 100/100 bps base (defensive, calibrated on PEPE Apr 26
smoke). After N=8 real KOL memecoin trades, the empirical adverse median
is ~9× higher (~880 buy / ~900 sell). Paper sim is therefore ranking ETH
strategies too optimistically.

Usage:
    python scripts/_calibrate_eth_slip.py            # report only
    python scripts/_calibrate_eth_slip.py --apply    # also writes JSONB
                                                     # paper_trade_config
                                                     # eth_buy_slippage_bps /
                                                     # eth_sell_slippage_bps

The strategies.py constants are kept in sync via a manual edit (printed at
the bottom of the report) — committed to the repo so paper sim and
sim_engines.py both pick up the new value at next reload.
"""
import os, sys, json, statistics as st, argparse
from datetime import datetime, timezone
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
CLOSED = ("sl_hit", "trail_stop", "tp_hit", "timeout", "be_stop")

# Conservative shrinkage factor — at small N the empirical median can be
# inflated by 1-2 outliers. We don't want paper sim to over-pessimize either,
# so cap the bumped value at SHRINK × empirical median.
SHRINK = 0.65


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write proposed values to JSONB paper_trade_config")
    args = ap.parse_args()

    rows = (sb.table("paper_trades").select(
        "id,symbol,strategy,status,position_usd,pnl_pct,"
        "gas_usd_buy,gas_usd_sell,quote_slip_bps_buy,quote_slip_bps_sell,"
        "buy_slippage_bps,sell_slippage_bps,paper_sim_pnl_pct"
    ).eq("source", "rt_live").eq("chain", "ethereum")
       .in_("status", list(CLOSED))
       .execute().data)
    print(f"=" * 96)
    print(f"ETH SLIP CALIBRATION — N={len(rows)} closed live trades")
    print(f"=" * 96)
    if not rows:
        print("No data — abort.")
        return

    # Per-trade cost breakdown
    print(f"\n[1] Per-trade cost breakdown (round-trip)")
    print(f"  {'sym':<11} {'strat':<22} {'PnL%':>7} "
          f"{'gas$':>6} {'qBuy':>5} {'qSell':>5} {'tot$':>6} {'tot%':>6} {'gross%':>7}")
    costs_pct = []
    qsb_all = []
    qss_all = []
    gas_buys = []
    gas_sells = []
    for r in rows:
        pos = float(r.get("position_usd") or 0)
        if pos <= 0: continue
        gb = float(r.get("gas_usd_buy") or 0)
        gs = float(r.get("gas_usd_sell") or 0)
        qb = float(r.get("quote_slip_bps_buy") or 0)
        qs = float(r.get("quote_slip_bps_sell") or 0)
        slip_cost_usd = pos * max(0, -qb) / 10000 + pos * max(0, -qs) / 10000
        tot_cost = gb + gs + slip_cost_usd
        tot_pct = tot_cost / pos * 100
        pnl = (r.get("pnl_pct") or 0) * 100
        gross = pnl + tot_pct
        costs_pct.append(tot_pct)
        qsb_all.append(qb); qss_all.append(qs)
        gas_buys.append(gb); gas_sells.append(gs)
        print(f"  {r.get('symbol') or '?':<11} {r['strategy']:<22} {pnl:>+6.1f}% "
              f"{gb+gs:>+5.2f} {qb:>+5.0f} {qs:>+5.0f} {tot_cost:>+5.2f} "
              f"{tot_pct:>+5.1f}% {gross:>+6.1f}%")

    print(f"\n[2] Distribution (signed, all trades)")
    print(f"  gas_usd_buy   median=${st.median(gas_buys):.2f}  mean=${st.mean(gas_buys):.2f}")
    print(f"  gas_usd_sell  median=${st.median(gas_sells):.2f}  mean=${st.mean(gas_sells):.2f}")
    print(f"  quote_slip_bps_buy   median={st.median(qsb_all):+.0f}  mean={st.mean(qsb_all):+.0f}")
    print(f"  quote_slip_bps_sell  median={st.median(qss_all):+.0f}  mean={st.mean(qss_all):+.0f}")
    print(f"  total cost%  median={st.median(costs_pct):.1f}%  mean={st.mean(costs_pct):.1f}%")

    # Adverse-only — what paper sim should model as expected drag
    neg_qsb = sorted([-q for q in qsb_all if q < 0])
    neg_qss = sorted([-q for q in qss_all if q < 0])
    print(f"\n[3] Adverse-side distribution (only trades where slip was negative)")
    print(f"  buy : N={len(neg_qsb)}/{len(qsb_all)}  median={st.median(neg_qsb) if neg_qsb else 0:.0f} bps  "
          f"p75={neg_qsb[int(len(neg_qsb)*0.75)] if neg_qsb else 0:.0f}  max={max(neg_qsb) if neg_qsb else 0:.0f}")
    print(f"  sell: N={len(neg_qss)}/{len(qss_all)}  median={st.median(neg_qss) if neg_qss else 0:.0f} bps  "
          f"p75={neg_qss[int(len(neg_qss)*0.75)] if neg_qss else 0:.0f}  max={max(neg_qss) if neg_qss else 0:.0f}")

    # Proposal: shrunk-median
    cur_buy = 100   # ETH_BUY_SLIPPAGE_BPS in strategies.py
    cur_sell = 100  # ETH_SELL_SLIPPAGE_BPS

    # Use EXPECTED value of slip (signed mean) as the calibration target — that's
    # what paper sim should subtract on average. Floor at current to avoid going
    # backwards if the empirical sample is favorable.
    expected_buy = max(int(round(-st.mean(qsb_all))), 0)   # bumped only if mean is adverse
    expected_sell = max(int(round(-st.mean(qss_all))), 0)
    proposed_buy = max(cur_buy, int(round(expected_buy * SHRINK / 50.0) * 50))   # round to 50
    proposed_sell = max(cur_sell, int(round(expected_sell * SHRINK / 50.0) * 50))

    print(f"\n[4] Proposed paper-sim slip constants (shrinkage={SHRINK} on N={len(rows)})")
    print(f"  ETH_BUY_SLIPPAGE_BPS  : {cur_buy} -> {proposed_buy}  "
          f"(expected adverse {expected_buy} bps, shrunk {int(expected_buy*SHRINK)})")
    print(f"  ETH_SELL_SLIPPAGE_BPS : {cur_sell} -> {proposed_sell}  "
          f"(expected adverse {expected_sell} bps, shrunk {int(expected_sell*SHRINK)})")
    print(f"\n  Implied paper-sim ETH cost drag at $20 pos:")
    print(f"    before: 200 bps slip + ~$3 gas = 200 + 1500 = 1700 bps = 17%")
    print(f"    after : {proposed_buy+proposed_sell} bps slip + ~$3 gas = "
          f"{proposed_buy+proposed_sell+1500} bps = {(proposed_buy+proposed_sell+1500)/100:.1f}%")
    print(f"    empirical (median total cost) : {st.median(costs_pct):.1f}%")

    if not args.apply:
        print(f"\n[5] To apply: re-run with --apply (writes JSONB paper_trade_config)")
        print(f"     Also edit scraper/strategies.py:")
        print(f"        ETH_BUY_SLIPPAGE_BPS = {proposed_buy}")
        print(f"        ETH_SELL_SLIPPAGE_BPS = {proposed_sell}")
        print(f"     Commit + push for paper sim and sim_engines to pick up the new value.")
        return

    # --apply: persist new values in JSONB so live_trader_eth picks them up
    cfg_row = sb.table("scoring_config").select("paper_trade_config").eq("id", 1).execute().data[0]
    cfg = cfg_row["paper_trade_config"]
    if isinstance(cfg, str):
        cfg = json.loads(cfg)
    cfg.setdefault("eth_buy_slippage_bps", cur_buy)
    cfg.setdefault("eth_sell_slippage_bps", cur_sell)
    cfg["eth_buy_slippage_bps"] = proposed_buy
    cfg["eth_sell_slippage_bps"] = proposed_sell

    backup_path = os.path.join(os.path.dirname(__file__), "..", "data",
                                f"paper_trade_config_pre_eth_slip_calib_"
                                f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json")
    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"\n  backup -> {backup_path}")
    sb.table("scoring_config").update({"paper_trade_config": cfg}).eq("id", 1).execute()
    print(f"  JSONB updated. paper_trade_config.eth_buy_slippage_bps={proposed_buy}, "
          f"eth_sell_slippage_bps={proposed_sell}")
    print(f"\n  Now also edit scraper/strategies.py manually:")
    print(f"      ETH_BUY_SLIPPAGE_BPS = {proposed_buy}")
    print(f"      ETH_SELL_SLIPPAGE_BPS = {proposed_sell}")
    print(f"  Commit + push so paper sim picks up the same values at next reload.")


if __name__ == "__main__":
    main()
