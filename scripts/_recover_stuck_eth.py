"""v14e.42 one-shot recovery for the closing_retry sentinel-leak bug.

Two parts:
  1) Fix 6 ETH live rows that successfully sold on-chain but kept status='closing_retry'.
     Derive the real terminal status from pnl_pct vs strategy TP/SL thresholds.
  2) Insert a synthetic rt_live row for $INCOME — bot bought it on-chain but the
     post-buy DB insert silently failed, leaving an orphan token in the wallet
     until the user manually sold via Rabby.

Run once after deploying the code patches in live_trader{.py, _eth.py}.
"""
import os, sys, json
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
from paper_trader import STRATEGIES

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
WALLET = "0xc5c92e3ac207f686d09686fe1de79a302d9410e9"
RPCS = [
    "https://ethereum-rpc.publicnode.com",
    "https://rpc.ankr.com/eth",
    "https://eth.merkle.io",
]

def rpc(method, params):
    last = None
    for url in RPCS:
        try:
            req = Request(url,
                data=json.dumps({"jsonrpc": "2.0", "id": 1,
                                 "method": method, "params": params}).encode(),
                headers={"Content-Type": "application/json",
                         "User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=15) as r:
                resp = json.loads(r.read())
                if "result" in resp:
                    return resp["result"]
                last = (url, resp.get("error"))
        except Exception as e:
            last = (url, str(e)[:80])
    raise RuntimeError(f"all RPCs failed: {last}")


def derive_terminal_status(strategy: str, pnl_pct: float) -> str:
    tr = (STRATEGIES.get(strategy) or [{}])[0]
    tp_mult = tr.get("tp_mult")
    sl_mult = tr.get("sl_mult", 0.0)
    if tp_mult and pnl_pct >= (tp_mult - 1):
        return "tp_hit"
    if sl_mult and pnl_pct <= (sl_mult - 1):
        return "sl_hit"
    return "timeout"


def fix_closing_retry_rows():
    print("=" * 60)
    print("Part 1 — fix closing_retry rows")
    print("=" * 60)
    res = (sb.table("paper_trades")
            .select("id,symbol,strategy,status,pnl_pct,pnl_usd,exit_at,"
                    "tx_signature,tx_signature_exit,token_address,position_usd,chain")
            .eq("status", "closing_retry")
            .eq("source", "rt_live")
            .execute())
    rows = res.data or []
    print(f"Found {len(rows)} rows in closing_retry")
    if not rows:
        return

    for row in rows:
        pnl = float(row.get("pnl_pct") or 0)
        new_status = derive_terminal_status(row["strategy"], pnl)
        print(f"  {row['symbol']:<11} {row['strategy']:<28} pnl={pnl*100:+6.2f}% "
              f"-> {new_status:<10} sell_tx=0x{(row.get('tx_signature_exit') or '')[:10]}")
        sb.table("paper_trades").update({"status": new_status}).eq("id", row["id"]).execute()
    print(f"Updated {len(rows)} rows.")


def insert_income_orphan():
    print()
    print("=" * 60)
    print("Part 2 — insert INCOME orphan rt_live row")
    print("=" * 60)
    INCOME_CA = "0xe5e066480a0b579432d55af4080f84f33b50cd0a"
    BUY_TX = "0xb435ab7e9f8eb87a9fcd8e3d0afb4754208a8e184a9da6a29f0b64f345f491eb"
    SELL_TX = "0x1e4746935abeab4091055ac64770c999ab296141b7be296db127860ab4129071"

    # Idempotency
    existing = (sb.table("paper_trades").select("id")
                .eq("token_address", INCOME_CA)
                .eq("source", "rt_live").execute())
    if existing.data:
        print(f"  rt_live row already exists for INCOME (id={existing.data[0]['id']}), skipping")
        return

    # Pull buy + sell tx data on-chain
    buy_tx = rpc("eth_getTransactionByHash", [BUY_TX])
    buy_rec = rpc("eth_getTransactionReceipt", [BUY_TX])
    buy_blk = rpc("eth_getBlockByNumber", [buy_tx["blockNumber"], False])
    sell_blk = rpc("eth_getBlockByNumber",
                    [rpc("eth_getTransactionByHash", [SELL_TX])["blockNumber"], False])

    buy_ts = datetime.fromtimestamp(int(buy_blk["timestamp"], 16), timezone.utc)
    sell_ts = datetime.fromtimestamp(int(sell_blk["timestamp"], 16), timezone.utc)
    eth_spent_wei = int(buy_tx["value"], 16)
    eth_spent = eth_spent_wei / 1e18

    # Compute tokens received from Transfer logs
    TRANSFER = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
    addr_topic = "0x" + WALLET.lower().replace("0x", "").rjust(64, "0")
    tokens_received = 0
    for log in buy_rec.get("logs", []):
        if (log.get("address", "").lower() == INCOME_CA.lower()
                and log.get("topics") and log["topics"][0] == TRANSFER
                and len(log["topics"]) >= 3 and log["topics"][2].lower() == addr_topic):
            tokens_received = int(log["data"], 16)
            break

    # decimals + decimal-adjusted human amount
    dec_raw = rpc("eth_call", [{"to": INCOME_CA, "data": "0x313ce567"}, "latest"])
    dec = int(dec_raw, 16) if dec_raw and dec_raw != "0x" else 18

    # Find paper twin to copy KOL metadata + estimate position_usd from sibling shadow
    shadow_q = (sb.table("paper_trades").select("*")
                .eq("token_address", INCOME_CA)
                .eq("source", "rt").limit(1).execute())
    shadow = (shadow_q.data or [{}])[0]
    entry_price = float(shadow.get("entry_price") or 0)
    pos_usd = 20.0  # eth_max_position_usd cap (matches every other ETH live trade)

    # Best-effort exit_price from sell tx: ETH received / tokens_sold ratio
    sell_rec = rpc("eth_getTransactionReceipt", [SELL_TX])
    eth_received_wei = 0
    for log in sell_rec.get("logs", []):
        # WETH withdraw events to wallet end up here too; pick the largest ETH-side
        if (log.get("topics") and len(log["topics"]) >= 3
                and log["topics"][0] == TRANSFER
                and log["topics"][2].lower() == addr_topic):
            v = int(log["data"], 16)
            if v > eth_received_wei:
                eth_received_wei = v
    eth_received = eth_received_wei / 1e18 if eth_received_wei else 0
    # rough: assume ETH price ~= price at buy block
    eth_usd_at_buy = pos_usd / eth_spent if eth_spent > 0 else 0
    usd_received = eth_received * eth_usd_at_buy if eth_received else 0

    if entry_price > 0 and pos_usd > 0 and usd_received > 0:
        exit_price = entry_price * (usd_received / pos_usd)
        pnl_pct = round((exit_price / entry_price) - 1, 4)
    else:
        exit_price = entry_price
        pnl_pct = 0
    pnl_usd = round(pos_usd * pnl_pct, 2)

    new_row = {
        "cycle_ts": buy_ts.isoformat(),
        "symbol": "$INCOME",
        "token_address": INCOME_CA,
        "chain": "ethereum",
        "rank_in_cycle": 0,
        "entry_price": entry_price or None,
        "entry_score": int(shadow.get("entry_score") or 0),
        "entry_mcap": shadow.get("entry_mcap"),
        "status": "manual_recovered",
        "strategy": "ETH_TP80_SL40_T2H",  # representative paper-main
        "tp_price": (entry_price * 1.80) if entry_price else None,
        "sl_price": (entry_price * 0.60) if entry_price else None,
        "horizon_minutes": 120,
        "tranche_pct": 1.0,
        "tranche_label": "main",
        "position_usd": pos_usd,
        "source": "rt_live",
        "tx_signature": BUY_TX[2:],            # store stripped (matches other rows)
        "tx_signature_exit": SELL_TX[2:],
        "execution_price": entry_price or None,
        "kol_group": shadow.get("kol_group"),
        "kol_tier": shadow.get("kol_tier"),
        "kol_score": shadow.get("kol_score"),
        "kol_win_rate": shadow.get("kol_win_rate"),
        "rt_score": shadow.get("rt_score"),
        "rt_liquidity_usd": shadow.get("rt_liquidity_usd"),
        "rt_volume_24h": shadow.get("rt_volume_24h"),
        "rt_buy_sell_ratio": shadow.get("rt_buy_sell_ratio"),
        "rt_token_age_hours": shadow.get("rt_token_age_hours"),
        "rt_is_pump_fun": 0,
        "buy_slippage_bps": 0,
        "buy_fee_bps": 300,
        "sol_price_at_entry": eth_usd_at_buy,
        "position_sol": round(eth_spent, 6),
        "buy_input_lamports": eth_spent_wei,
        "buy_output_tokens": tokens_received,
        "dex_spot_price_at_entry": entry_price or None,
        "high_price_seen": entry_price or None,
        "entry_source": "manual_recover",
        "exit_price": exit_price or None,
        "exit_at": sell_ts.isoformat(),
        "pnl_pct": pnl_pct,
        "pnl_usd": pnl_usd,
        "exit_minutes": int((sell_ts - buy_ts).total_seconds() / 60),
        "sell_output_lamports": eth_received_wei,
        "sell_sol_received": round(eth_received, 6) if eth_received else None,
        "sol_price_at_exit": eth_usd_at_buy,
        "created_at": buy_ts.isoformat(),
        "sell_attempts": 0,
    }
    # drop None/optional cols defensively
    new_row = {k: v for k, v in new_row.items() if v is not None}
    print(f"  buy_ts={buy_ts}  sell_ts={sell_ts}  pnl={pnl_pct*100:+.2f}% (${pnl_usd})")
    print(f"  status='manual_recovered' (distinct from real terminals so it's "
          f"excluded from outlier monitor + livestats — flag for audit)")

    # buy_output_tokens raw amount can exceed bigint (2^63) for low-price tokens.
    # Drop pre-emptively if it overflows — the buy tx hash is already stored, so
    # the amount can be re-derived from chain if ever needed.
    if new_row.get("buy_output_tokens", 0) > 9_000_000_000_000_000_000:
        new_row.pop("buy_output_tokens", None)
    if new_row.get("sell_output_lamports", 0) > 9_000_000_000_000_000_000:
        new_row.pop("sell_output_lamports", None)
    if new_row.get("buy_input_lamports", 0) > 9_000_000_000_000_000_000:
        new_row.pop("buy_input_lamports", None)

    OPTIONAL = ("rt_score", "rt_liquidity_usd", "rt_volume_24h", "rt_buy_sell_ratio",
                "rt_token_age_hours", "buy_output_tokens", "sell_output_lamports",
                "buy_input_lamports", "entry_mcap")
    for attempt in range(8):
        try:
            sb.table("paper_trades").insert(new_row).execute()
            print("  -> inserted OK")
            return
        except Exception as e:
            err = str(e)
            stripped = False
            # match by column name OR by overflow value
            for c in OPTIONAL:
                if (c in err or "out of range" in err) and c in new_row:
                    new_row.pop(c, None); stripped = True; break
            if not stripped:
                print(f"  -> insert failed (no more optional cols to strip): {err[:200]}")
                return
    print("  -> insert exhausted retries")


if __name__ == "__main__":
    fix_closing_retry_rows()
    insert_income_orphan()
    print()
    print("Done. Verify with:")
    print("  SELECT status, COUNT(*) FROM paper_trades "
          "WHERE source='rt_live' AND chain='ethereum' GROUP BY 1;")
