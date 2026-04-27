"""Phase 1 ETH live microtest enabler — Option C.

Configures a tightly-bounded ETH live test to gather slippage data on actual
KOL-called memecoins (Apr 26 smoke test was on PEPE, NOT representative).

Configuration applied:
  live_trading.eth_live_enabled       = true
  live_trading.eth_allocations        = {"ETH_TP80_SL40_T2H": 1.0}  # most-N strat
  live_trading.eth_max_position_usd   = 50    # matches paper cost model
  live_trading.eth_max_open_positions = 1     # serial only
  live_trading.eth_buy_slippage_bps   = 500   # defensive vs shallow pools
  live_trading.eth_sell_slippage_bps  = 600   # defensive — exits worse than entries
  live_trading.eth_daily_loss_limit_usd = 40

Pre-flight checks BEFORE enabling:
  1. ETH_PRIVATE_KEY in env
  2. Wallet balance >= $80 (1 trade $50 + $5 gas + $25 buffer)
  3. Instrumentation columns present (gas_usd_buy etc.)
  4. live_trader_eth open + close paths still wired in safe_scraper

Usage:
  python scripts/_enable_eth_phase1_microtest.py            # check + dry-run
  python scripts/_enable_eth_phase1_microtest.py --apply    # actually enable

After applying, you MUST restart the VPS service for config to load.
Disable later with: python scripts/_disable_eth_live.py --apply
"""
import os, sys, json, copy, argparse
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

DESIRED = {
    "eth_live_enabled": True,
    "eth_allocations": {"ETH_TP80_SL40_T2H": 1.0},
    "eth_max_position_usd": 20,    # ultra-safe microtest — wallet $43, gives
                                    # 2 trades buffer + gas room. Paper sim min
                                    # is $50 so cost-drag bias ~+12pp on live;
                                    # slippage observed here is a LOWER BOUND
                                    # (less price impact than $50 in same pool).
    "eth_max_open_positions": 1,
    "eth_buy_slippage_bps": 500,
    "eth_sell_slippage_bps": 600,
    "eth_daily_loss_limit_usd": 30,  # ~3 SL hits ($10 each) buffer
}
MIN_WALLET_USD = 60   # 1 trade $25 + ~$5 gas + ~$30 buffer for 2-3 trades window


def preflight(local: bool = False) -> tuple[bool, list[str]]:
    """Return (ok, list_of_errors). local=True relaxes wallet checks (key lives
    on VPS only — verify there with scripts/_vps_eth_wallet_check.py)."""
    errors = []
    # 1. ETH_PRIVATE_KEY — mandatory on VPS, optional locally
    pk = os.environ.get("ETH_PRIVATE_KEY")
    if not pk:
        if local:
            print("  ETH_PRIVATE_KEY absent locally (expected — lives on VPS). "
                  "RUN scripts/_vps_eth_wallet_check.py ON THE VPS before restart.")
        else:
            errors.append("ETH_PRIVATE_KEY missing in scraper/.env")
    # 2. wallet balance (only if key available)
    if pk:
        try:
            from live_trader_eth import _client, _eth_usd_price
            w3r, _, acct = _client()
            bal_eth = w3r.eth.get_balance(acct.address) / 1e18
            usd = bal_eth * _eth_usd_price(w3r)
            if usd < MIN_WALLET_USD:
                errors.append(f"Wallet balance ${usd:.2f} < ${MIN_WALLET_USD} required")
            else:
                print(f"  wallet OK: {bal_eth:.6f} ETH (~${usd:.2f})")
        except Exception as e:
            errors.append(f"wallet check failed: {e}")
    # 3. Instrumentation columns
    r = sb.table("paper_trades").select("*").limit(1).execute()
    cols = set(r.data[0].keys()) if r.data else set()
    needed = {"gas_usd_buy", "gas_usd_sell", "quote_slip_bps_buy",
              "quote_slip_bps_sell", "block_number_buy", "block_number_sell"}
    missing = needed - cols
    if missing:
        errors.append(f"DB migration not applied — missing cols: {sorted(missing)}. "
                      f"Run scripts/_eth_microtest_migration.sql in Supabase Studio.")
    else:
        print("  instrumentation OK: all 6 cols present")
    # 4. code paths
    ss_path = os.path.join(os.path.dirname(__file__), "..", "scraper", "safe_scraper.py")
    ss = open(ss_path, encoding="utf-8", errors="ignore").read()
    if "open_live_trade as open_live_trade_eth" not in ss:
        errors.append("safe_scraper.py: ETH OPEN dispatcher path missing")
    if "check_live_trades_eth" not in ss:
        errors.append("safe_scraper.py: ETH CLOSE dispatcher path missing")
    return len(errors) == 0, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--local", action="store_true",
                    help="Run from local dev machine — relaxes wallet checks "
                         "since ETH_PRIVATE_KEY is only on VPS. Verify wallet "
                         "on VPS via scripts/_vps_eth_wallet_check.py first.")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="DANGEROUS — skip ALL preflight checks")
    args = ap.parse_args()

    print("=" * 92)
    print("PHASE 1 ETH MICROTEST — pre-flight checks")
    print("=" * 92)
    ok, errors = preflight(local=args.local)
    if not ok:
        print("\nPRE-FLIGHT FAILED:")
        for e in errors:
            print(f"  - {e}")
        if not args.skip_preflight:
            print("\nFix the issues above, then re-run. Or pass --skip-preflight (dangerous).")
            sys.exit(1)
        print("\n--skip-preflight set — proceeding anyway (DANGEROUS)")

    # Read current config
    cfg = sb.table("scoring_config").select("rt_trade_config").eq("id", 1).execute().data[0]["rt_trade_config"]
    if isinstance(cfg, str):
        cfg = json.loads(cfg)
    new_cfg = copy.deepcopy(cfg)
    live = new_cfg.setdefault("live_trading", {})

    print()
    print("=" * 92)
    print("CONFIG DIFF")
    print("=" * 92)
    for k, v in DESIRED.items():
        old = live.get(k, "<NOT SET>")
        print(f"  {k:<32} {str(old):<20} -> {v}")
        live[k] = v

    if not args.apply:
        print("\n[DRY RUN] pass --apply to write config to DB")
        print("After applying, restart VPS: systemctl restart kol-scraper.service (or /deploy)")
        return

    # Backup
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bk_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(bk_dir, exist_ok=True)
    bk = os.path.join(bk_dir, f"rt_trade_config_pre_eth_phase1_{ts}.json")
    with open(bk, "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    print(f"\nbackup -> {bk}")

    sb.table("scoring_config").update({"rt_trade_config": new_cfg}).eq("id", 1).execute()
    print("\nDB updated. NEXT STEPS:")
    print("  1) Restart VPS service: systemctl restart kol-scraper.service")
    print("  2) Tail logs for first ETH live trade: journalctl -u kol-scraper -f | grep 'ETH LIVE'")
    print("  3) After 10 closed trades, run: python scripts/_eth_microtest_recap.py")
    print("  4) To kill: python scripts/_disable_eth_live.py --apply")


if __name__ == "__main__":
    main()
