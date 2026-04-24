"""Replay ETH tokens through a grid of strategies to find what would have won.

Universe: the 4 ETH tokens with paper trades (ASTEROID, GENZ, CHINESEASTEROID, EIB).
Ticks: price_ticks rows for these addresses (65-110 per token).
Fee model: ETH ($15 gas round-trip + 200 bps MEV, same as live_trader config).

Grid: all (TP%, SL%, horizon) combos + BE variants. Flat-size $200/trade.
"""
import os
import sys
from datetime import datetime, timezone
from collections import defaultdict
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))

from supabase import create_client
from paper_trader import _dynamic_sell_slip_factor

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

ETH_GAS_USD_PER_SIDE = 7.50  # $15 round-trip
ETH_MEV_BPS = 200
POSITION_USD = 200.0


def fetch_token_data(addr: str) -> dict:
    """Fetch trade entry/created_at + ticks chronologically."""
    tr = sb.table("paper_trades").select(
        "symbol,entry_price,created_at,token_address"
    ).eq("token_address", addr).eq("chain", "ethereum").limit(1).execute().data
    if not tr:
        return None
    entry = float(tr[0]["entry_price"])
    created = datetime.fromisoformat(tr[0]["created_at"].replace("Z", "+00:00"))

    ticks = sb.table("price_ticks").select(
        "fetched_at,price_usd,liquidity_usd"
    ).eq("token_address", addr).eq("chain", "ethereum").order(
        "fetched_at"
    ).execute().data
    # Keep only ticks at or after entry
    parsed = []
    for t in ticks:
        ft = datetime.fromisoformat(t["fetched_at"].replace("Z", "+00:00"))
        if ft < created:
            continue
        parsed.append({
            "t": ft,
            "p": float(t["price_usd"]),
            "liq": float(t.get("liquidity_usd") or 0),
        })
    return {
        "symbol": tr[0]["symbol"],
        "entry": entry,
        "created": created,
        "ticks": parsed,
    }


def simulate(token_data: dict, tp_pct: float, sl_pct: float,
             horizon_min: int, be_act_pct: float | None = None) -> dict:
    """Simulate one strategy on one token's ticks. Returns exit dict.

    tp_pct / sl_pct are fractions (0.8 = +80% TP, 0.5 = -50% SL).
    be_act_pct: if peak >= entry * (1 + be_act_pct), SL moves to entry.
    """
    entry = token_data["entry"]
    created = token_data["created"]
    ticks = token_data["ticks"]

    sl_price = entry * (1 - sl_pct)
    tp_price = entry * (1 + tp_pct)

    high_seen = entry
    exit_reason = None
    exit_price = None
    exit_minutes = None

    for tk in ticks:
        elapsed = (tk["t"] - created).total_seconds() / 60
        if elapsed > horizon_min:
            exit_reason = "timeout"
            exit_price = tk["p"]
            exit_minutes = int(elapsed)
            break
        high_seen = max(high_seen, tk["p"])
        effective_sl = sl_price
        if be_act_pct is not None and high_seen >= entry * (1 + be_act_pct):
            effective_sl = max(effective_sl, entry)
        # Priority: SL hit → TP hit → continue
        if tk["p"] <= effective_sl:
            exit_reason = "sl_hit" if effective_sl <= sl_price else "be_stop"
            # ETH slip: MEV bps + dynamic slip (fn of liquidity)
            slip_factor = (1 - ETH_MEV_BPS / 10_000)
            # Use the production dynamic slip too (branches on chain)
            # Build a minimal trade dict for the helper
            fake_trade = {
                "chain": "ethereum",
                "rt_liquidity_usd": tk["liq"],
                "sl_price": sl_price,
            }
            try:
                dyn = _dynamic_sell_slip_factor(fake_trade, exit_reason,
                                                base_bps=ETH_MEV_BPS, sell_fee_bps=0)
                exit_price = effective_sl * dyn
            except Exception:
                exit_price = effective_sl * slip_factor
            exit_minutes = int(elapsed)
            break
        if tk["p"] >= tp_price:
            exit_reason = "tp_hit"
            slip_factor = (1 - ETH_MEV_BPS / 10_000)
            fake_trade = {"chain": "ethereum", "rt_liquidity_usd": tk["liq"]}
            try:
                dyn = _dynamic_sell_slip_factor(fake_trade, "tp_hit",
                                                base_bps=ETH_MEV_BPS, sell_fee_bps=0)
                exit_price = tp_price * dyn
            except Exception:
                exit_price = tp_price * slip_factor
            exit_minutes = int(elapsed)
            break

    # Ticks exhausted without exit — treat as timeout at last tick
    if exit_reason is None:
        if ticks:
            last = ticks[-1]
            exit_reason = "no_data"
            exit_price = last["p"]
            exit_minutes = int((last["t"] - created).total_seconds() / 60)
        else:
            return None

    # PnL: gross = (exit/entry - 1) * POSITION_USD
    #      net  = gross - 2 * ETH_GAS_USD_PER_SIDE
    gross_pct = (exit_price / entry) - 1
    gross_usd = gross_pct * POSITION_USD
    net_usd = gross_usd - 2 * ETH_GAS_USD_PER_SIDE
    net_pct = net_usd / POSITION_USD
    return {
        "exit_reason": exit_reason,
        "exit_price": exit_price,
        "gross_pct": gross_pct,
        "net_pct": net_pct,
        "net_usd": net_usd,
        "exit_minutes": exit_minutes,
    }


def main():
    tokens = [
        "0xa29B1250e55489707aF48b2Ddeda249465861c9f",
        "0x7a1943613dceb741676e2f34a5ae1160f9361110",
        "0x72b734e6046304A78734937Da869638e7e5b51d0",
        "0x57ac094217e0e3887ab135de205b0a2bb170e87a",
    ]
    data = {addr: fetch_token_data(addr) for addr in tokens}
    data = {a: d for a, d in data.items() if d and d["ticks"]}

    print(f"Loaded {len(data)} tokens with ticks:")
    for addr, d in data.items():
        print(f"  {d['symbol']:<20} entry={d['entry']:.4e} ticks={len(d['ticks'])} "
              f"span={int((d['ticks'][-1]['t']-d['ticks'][0]['t']).total_seconds()/60)}min")

    # Grid: TP × SL × horizon × (no_BE, BE activations)
    tp_grid = [0.30, 0.50, 0.80, 1.00, 1.50, 2.00, 3.00, 5.00]
    sl_grid = [0.20, 0.30, 0.40, 0.50, 0.70]
    h_grid = [60, 120, 240, 480]
    be_grid = [None, 0.20, 0.30, 0.50, 1.00]  # None = no BE

    print("\n=== GRID SIM RESULTS (sorted by total net $ across 4 tokens) ===")
    results = []
    for tp in tp_grid:
        for sl in sl_grid:
            for h in h_grid:
                for be in be_grid:
                    total_net = 0
                    n_win = 0
                    n_tp = 0
                    n_sl = 0
                    n_to = 0
                    per_tok = []
                    for addr, d in data.items():
                        r = simulate(d, tp, sl, h, be)
                        if r is None:
                            continue
                        total_net += r["net_usd"]
                        if r["net_pct"] > 0:
                            n_win += 1
                        if r["exit_reason"] == "tp_hit":
                            n_tp += 1
                        elif r["exit_reason"] in ("sl_hit", "be_stop"):
                            n_sl += 1
                        else:
                            n_to += 1
                        per_tok.append(f"{d['symbol'].replace('$','')}={r['net_usd']:+.0f}")
                    label_be = f"BE{int(be*100)}" if be else "noBE"
                    strat_name = f"{label_be}_TP{int(tp*100)}_SL{int(sl*100)}_H{h}"
                    results.append({
                        "name": strat_name,
                        "tp": tp, "sl": sl, "h": h, "be": be,
                        "total_net": total_net,
                        "wr": n_win / len(data) if data else 0,
                        "tp_hits": n_tp, "sl_hits": n_sl, "to": n_to,
                        "per_tok": per_tok,
                    })

    # Sort by total net
    results.sort(key=lambda r: -r["total_net"])

    print(f"\nTop 20 strats (on 4 tokens, $200/trade, ETH fee model):")
    print(f"{'Strat':<30}{'Net$':>9}{'WR':>6}{'TP':>4}{'SL':>4}{'TO':>4}   Per-token")
    print("-" * 120)
    for r in results[:20]:
        print(f"{r['name']:<30}{r['total_net']:>+9.2f}{int(r['wr']*100):>5}%{r['tp_hits']:>4}"
              f"{r['sl_hits']:>4}{r['to']:>4}   {', '.join(r['per_tok'])}")

    print(f"\nBottom 10 (worst losers):")
    for r in results[-10:]:
        print(f"{r['name']:<30}{r['total_net']:>+9.2f}{int(r['wr']*100):>5}%{r['tp_hits']:>4}"
              f"{r['sl_hits']:>4}{r['to']:>4}   {', '.join(r['per_tok'])}")

    # What the 3 live ETH strats would look like
    print(f"\n=== 3 live ETH strats (same grid) ===")
    prod_strats = [
        ("ETH_TP100_SL50", 1.00, 0.50, 240, None),
        ("ETH_TP80_SL40_T2H", 0.80, 0.40, 120, None),
        ("ETH_BE50_TP150_SL50", 1.50, 0.50, 240, 0.50),
    ]
    print(f"{'Strat':<25}{'Net$':>9}{'WR':>6}   Per-token")
    for name, tp, sl, h, be in prod_strats:
        total = 0
        wins = 0
        per = []
        for addr, d in data.items():
            r = simulate(d, tp, sl, h, be)
            if r is None: continue
            total += r["net_usd"]
            if r["net_pct"] > 0: wins += 1
            per.append(f"{d['symbol'].replace('$','')}={r['net_usd']:+.0f}")
        print(f"{name:<25}{total:>+9.2f}{int(wins/len(data)*100):>5}%   {', '.join(per)}")


if __name__ == "__main__":
    main()
