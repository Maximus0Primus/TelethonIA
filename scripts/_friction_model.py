"""Real friction decomposition (on-chain) + position-size viability model for the 3 live SOL strats."""
import os, json, time, requests
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from supabase import create_client
from dotenv import load_dotenv

load_dotenv("scraper/.env")
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
KEY = os.environ["HELIUS_API_KEY"]
URL = f"https://mainnet.helius-rpc.com/?api-key={KEY}"
NOW = datetime.now(timezone.utc)
EXIT = ["tp_hit", "sl_hit", "timeout", "trail_stop", "be_stop"]
RENT = 0.00203928  # SPL token account rent (SOL)

cfg = sb.table("scoring_config").select("paper_trade_config,rt_trade_config").eq("id", 1).execute().data[0]
SOL_BL = set(cfg["paper_trade_config"].get("kol_chain_blacklist", {}).get("solana", []))
ALLOCS = cfg["rt_trade_config"].get("hybrid_strategy", {}).get("allocations", {})
SOL_STRATS = [k for k in ALLOCS if not k.startswith(("ETH_", "BSC_", "BASE_"))]

# ---------- PART 1: real friction from on-chain (29 live trades) ----------
SINCE = "2026-05-17T00:00:00Z"
lr = []
off = 0
while True:
    b = sb.table("paper_trades").select(
        "strategy,status,pnl_pct,pnl_usd,position_sol,sol_price_at_entry,buy_input_lamports,sell_sol_received,tx_signature,tx_signature_exit"
    ).eq("source", "rt_live").eq("is_shadow", False).gte("created_at", SINCE).order("id").range(off, off + 499).execute().data or []
    lr += b
    if len(b) < 500:
        break
    off += 500
lr = [r for r in lr if r["status"] in EXIT and r.get("strategy")]
print(f"Live trades reconciled: {len(lr)}")


def gtx(sig):
    for _ in range(3):
        try:
            r = requests.post(URL, json={"jsonrpc": "2.0", "id": 1, "method": "getTransaction",
                                         "params": [sig, {"maxSupportedTransactionVersion": 0, "encoding": "jsonParsed"}]}, timeout=25).json()
            return r.get("result")
        except Exception:
            time.sleep(1)
    return None


tot_swap_out = tot_swap_in = tot_fee = tot_rent_dep = 0.0
n_ata = 0
gross_sol = 0.0
for r in lr:
    pos = float(r["position_sol"] or 0.012)
    tb = gtx(r["tx_signature"]) if r.get("tx_signature") else None
    ts = gtx(r["tx_signature_exit"]) if r.get("tx_signature_exit") else None
    if not tb or not ts:
        continue
    fb = (tb["meta"].get("fee", 0)) / 1e9
    fs = (ts["meta"].get("fee", 0)) / 1e9
    db = (tb["meta"]["postBalances"][0] - tb["meta"]["preBalances"][0]) / 1e9  # negative
    ds = (ts["meta"]["postBalances"][0] - ts["meta"]["preBalances"][0]) / 1e9  # positive
    swap_out = pos                       # SOL into the swap
    rent_dep = max(0.0, (-db) - swap_out - fb)  # extra over swap+fee = ATA rent
    swap_in = ds + fs                    # SOL out of the sell swap
    tot_swap_out += swap_out
    tot_swap_in += swap_in
    tot_fee += fb + fs
    tot_rent_dep += rent_dep
    if rent_dep > 0.0015:
        n_ata += 1
    gross_sol += swap_in - swap_out
    time.sleep(0.02)

SOLP = 84.5
n = len(lr)
# rent recovered: from getSignaturesForAddress "other" bucket measured earlier ~ count of closes * RENT
# Approximate net rent loss = deposited - recovered. We measured total wallet ~ -0.05 SOL.
print(f"\n=== PART 1 — REAL FRICTION ({n} live round-trips @ ~$1) ===")
print(f"Gross price PnL (lamport flow, incl slippage): {gross_sol:+.5f} SOL = ${gross_sol*SOLP:+.2f}")
print(f"Network+priority fees:                          {tot_fee:.5f} SOL = ${tot_fee*SOLP:.2f}  (${tot_fee*SOLP/n:.3f}/trade)")
print(f"ATA rent DEPOSITED ({n_ata} new accounts):       {tot_rent_dep:.5f} SOL = ${tot_rent_dep*SOLP:.2f}  (recoverable via close_ata)")
print(f"  -> rent if NEVER recovered: ${tot_rent_dep*SOLP/n:.3f}/trade | if fully recovered: $0/trade")
# Measured real wallet delta = -0.0499 SOL (full history incl close_ata + fails)
REAL_WALLET = -0.0499
fixed_total = REAL_WALLET - gross_sol   # everything that's not price movement
print(f"\nMeasured real wallet delta (all txns):          {REAL_WALLET:+.5f} SOL = ${REAL_WALLET*SOLP:+.2f}")
print(f"=> TOTAL non-price friction:                     {-(fixed_total):.5f} SOL = ${-fixed_total*SOLP:.2f}")
FIXED_PER_TRADE_USD = -fixed_total * SOLP / n
print(f"=> FIXED cost per trade (fees + net rent churn): ${FIXED_PER_TRADE_USD:.3f}/trade")
print(f"   (of which fees ${tot_fee*SOLP/n:.3f}, rest is unrecovered rent / churn ${FIXED_PER_TRADE_USD - tot_fee*SOLP/n:.3f})")

# best-case fixed if close_ata were perfect = fees only
FIXED_BEST = tot_fee * SOLP / n
print(f"   BEST-CASE fixed (if close_ata perfect) = fees only = ${FIXED_BEST:.3f}/trade")

# ---------- PART 2: size viability model for 3 strats ----------
print(f"\n=== PART 2 — VIABILITY BY POSITION SIZE (3 live strats, 14d shadow) ===")
SINCE14 = (NOW - timedelta(days=14)).isoformat().replace("+00:00", "Z")


def dedup(rows):
    rows = sorted(rows, key=lambda r: r["created_at"])
    seen = {}
    out = []
    for r in rows:
        if r["kol_group"] in SOL_BL:
            continue
        t = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
        k = (r["token_address"],)
        p = seen.get(k)
        if p and (t - p) < timedelta(hours=24):
            continue
        seen[k] = t
        out.append(r)
    return out


SIZES = [1, 2, 5, 10, 20, 50]
for strat in SOL_STRATS:
    rows = []
    off = 0
    while True:
        b = sb.table("paper_trades").select("kol_group,token_address,pnl_pct,status,created_at").eq("chain", "solana").eq("is_shadow", True).eq("strategy", strat).in_("status", EXIT).gte("created_at", SINCE14).order("created_at").range(off, off + 999).execute().data or []
        rows += b
        if len(b) < 1000:
            break
        off += 1000
    d = dedup(rows)
    if not d:
        print(f"\n{strat}: no shadow rows")
        continue
    pnls = [float(r["pnl_pct"]) for r in d]
    N = len(pnls)
    avg = sum(pnls) / N
    days = (datetime.fromisoformat(d[-1]["created_at"].replace("Z","+00:00")) - datetime.fromisoformat(d[0]["created_at"].replace("Z","+00:00"))).days or 1
    trades_per_day = N / days
    print(f"\n## {strat}  (N={N} dedup, avg gross pnl/trade={avg*100:+.2f}%, ~{trades_per_day:.1f} trades/day)")
    print(f"   {'pos$':>5} | {'gross$/trade':>12} | {'-fixed':>7} | {'NET$/trade':>10} | {'NET$/day':>9} | viable?")
    for s in SIZES:
        gross_t = s * avg
        net_t = gross_t - FIXED_PER_TRADE_USD
        net_day = net_t * trades_per_day
        print(f"   {s:>5} | {gross_t:>+12.3f} | {-FIXED_PER_TRADE_USD:>7.3f} | {net_t:>+10.3f} | {net_day:>+9.2f} | {'YES' if net_t>0 else 'no'}")
    # breakeven position
    if avg > 0:
        be = FIXED_PER_TRADE_USD / avg
        be_best = FIXED_BEST / avg
        print(f"   >> BREAKEVEN position: ${be:.2f} (current fixed) | ${be_best:.2f} (if close_ata fixed)")
    else:
        print(f"   >> avg gross pnl NEGATIVE — unviable at ANY position size")
