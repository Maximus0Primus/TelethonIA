"""Unified sim/paper/live divergence report since v143.5 (Apr 18 21:37 UTC).

Joins live rows (source=rt_live) with their same-token/same-strat paper twins
(source=rt), plus the sim replay produced by verify_sim_live_alignment.py.

Persists the per-trade table as JSON for later inspection.
"""
import os, sys, json, statistics as st
from collections import defaultdict
from datetime import datetime
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
import paper_trader as _pt
from paper_trader import (_strategy_orchestration, _decision_price, _evaluate_trade_exit,
                          _jupiter_prices_cache, _dex_prices_cache, _ema_state, _smooth_state,
                          _dynamic_sell_slip_factor, SELL_FEE_BPS)

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
SINCE = os.environ.get("SINCE", "2026-04-15T00:00:00+00:00")
OUT_JSON = os.path.join(os.path.dirname(__file__), "..", "data", "diverge_apr19.json")
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

COLS = ("id,strategy,symbol,status,pnl_pct,pnl_usd,paper_sim_pnl_pct,eval_history,"
        "entry_price,exit_price,paper_exit_price,execution_price,dex_spot_price_at_entry,"
        "high_price_seen,slippage_actual_bps,buy_slippage_bps,sell_slippage_bps,"
        "tp_price,sl_price,horizon_minutes,"
        "position_usd,created_at,exit_at,source,token_address,rt_is_pump_fun,rt_liquidity_usd")


def fetch_all(table, sel="*", **f):
    out, step, off = [], 1000, 0
    while True:
        q = sb.table(table).select(sel)
        for k, v in f.items():
            if k.startswith("gte_"): q = q.gte(k[4:], v)
            elif k.startswith("eq_"): q = q.eq(k[3:], v)
            elif k.startswith("in_"): q = q.in_(k[3:], v)
        r = q.range(off, off+step-1).execute()
        if not r.data: break
        out.extend(r.data)
        if len(r.data) < step: break
        off += step
    return out


def fetch_live_cfg():
    r = sb.table("scoring_config").select("rt_trade_config").eq("id", 1).execute()
    cfg = r.data[0]["rt_trade_config"]
    if isinstance(cfg, str): cfg = json.loads(cfg)
    return cfg


def replay_sim(trade, ticks, live_cfg):
    """Mirror verify_sim_live_alignment.replay_trade."""
    strategy = trade["strategy"]
    orch = _strategy_orchestration(strategy, live_cfg)
    polling_sec = int(orch.get("polling_sec", 30))
    addr = trade["token_address"]
    merged = []
    for t in ticks:
        p = float(t.get("price_usd") or 0)
        if p <= 0: continue
        ts = t["fetched_at"]
        if ts < trade["created_at"]: continue
        src = "jp" if t["source"] == "jupiter" else ("ds" if t["source"] in ("fast","full","live") else None)
        if src: merged.append((ts, src, p))
    merged.sort(key=lambda x: x[0])
    if not merged: return None
    trade_id = f"replay_{trade['id']}"
    _ema_state.pop(trade_id, None); _smooth_state.pop(trade_id, None)
    _jupiter_prices_cache.pop(addr, None); _dex_prices_cache.pop(addr, None)
    fake = dict(trade); fake["id"] = trade_id
    fake["high_price_seen"] = float(trade["entry_price"])
    last_eval_t = None
    for ts, src, p in merged:
        if src == "jp": _jupiter_prices_cache[addr] = p
        else: _dex_prices_cache[addr] = p
        tt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if last_eval_t is not None and (tt - last_eval_t).total_seconds() < polling_sec: continue
        dec_price, exit_ref = _decision_price(addr, strategy, trade_id, orch, trade=fake)
        if dec_price is None: continue
        if exit_ref is None: exit_ref = dec_price
        last_eval_t = tt
        ev = _evaluate_trade_exit(fake, exit_ref, tt, sell_slip_factor=1.0, sell_fee_bps=0, decision_price=dec_price)
        if ev is None: continue
        if ev.get("high_price_seen"):
            nh = ev["high_price_seen"]
            if nh > float(fake.get("high_price_seen") or 0): fake["high_price_seen"] = nh
        if ev.get("status"):
            _ema_state.pop(trade_id, None); _smooth_state.pop(trade_id, None)
            return ev["pnl_pct"]
    _ema_state.pop(trade_id, None); _smooth_state.pop(trade_id, None)
    return None


print(f"Loading closed trades since {SINCE}...")
all_closed = fetch_all("paper_trades", sel=COLS, gte_created_at=SINCE)
all_closed = [t for t in all_closed if t.get("status") in ("sl_hit","trail_stop","tp_hit","timeout","be_stop")]
# Drop legacy DTRAIL* strats (user: "dtrail on s'en fout")
all_closed = [t for t in all_closed if not str(t.get("strategy","")).startswith("DTRAIL")]
print(f"  {len(all_closed)} closed total (DTRAIL* filtered)")

by_key = defaultdict(dict)
for t in all_closed:
    src = "live" if t.get("source") == "rt_live" else "paper"
    by_key[(t["token_address"], t["strategy"])][src] = t

pairs = [(k, v["live"], v["paper"]) for k, v in by_key.items() if "live" in v and "paper" in v]
print(f"  {len(pairs)} matched (live, paper) pairs")

live_cfg = fetch_live_cfg()
tokens = list({lv["token_address"] for _, lv, _ in pairs})
print(f"  Loading ticks for {len(tokens)} tokens...")
ticks = fetch_all("price_ticks", sel="token_address,fetched_at,source,price_usd",
                  gte_fetched_at=SINCE, in_token_address=tokens)
by_tok = defaultdict(list)
for tk in ticks: by_tok[tk["token_address"]].append(tk)
print(f"  {len(ticks)} ticks")

rows = []
# v144.8: prefer eval_history replay over price_ticks reconstruction.
# eval_history is the exact (decision, exec) stream the live bot saw at 30s;
# price_ticks samples Jupiter at 3-min batch (lossy). paper_sim_pnl_pct remains
# the top preference — it's the sim value computed at live-close time.
from sim import _replay_from_eval_history as _eh_replay
for (tok, strat), lv, pp in pairs:
    sim_pnl = lv.get("paper_sim_pnl_pct")
    if sim_pnl is None:
        eh = lv.get("eval_history") or []
        if isinstance(eh, list) and len(eh) >= 2:
            fake_lv = dict(lv); fake_lv["high_price_seen"] = float(lv["entry_price"])
            res = _eh_replay(fake_lv, eh)
            sim_pnl = res["pnl_pct"] if res else None
        if sim_pnl is None:
            sim_pnl = replay_sim(lv, by_tok.get(tok, []), live_cfg)
    sim_pct = float(sim_pnl) * 100 if sim_pnl is not None else None
    live_pct = float(lv.get("pnl_pct") or 0) * 100
    paper_pct = float(pp.get("pnl_pct") or 0) * 100
    pos = float(lv.get("position_usd") or 0)
    rows.append({
        "symbol": lv["symbol"], "strategy": strat, "token": tok,
        "is_pump": bool(lv.get("rt_is_pump_fun") or pp.get("rt_is_pump_fun")),
        "liq_usd": lv.get("rt_liquidity_usd"),
        "entry_live": lv["entry_price"], "entry_paper": pp["entry_price"],
        "exit_live": lv.get("exit_price"), "exit_paper": pp.get("exit_price"),
        "pnl_live_pct": live_pct, "pnl_paper_pct": paper_pct, "pnl_sim_pct": sim_pct,
        "pnl_live_usd": float(lv.get("pnl_usd") or 0),
        "pnl_paper_usd": float(pp.get("pnl_usd") or 0),
        "pnl_sim_usd": (sim_pct/100*pos) if sim_pct is not None else None,
        # Real applied paper slip: (1 - _dynamic_sell_slip_factor) × 10_000
        "slip_live_bps": lv.get("slippage_actual_bps"),
        "slip_paper_effective_bps": round(
            (1 - _dynamic_sell_slip_factor(pp, pp["status"], fee_bps=SELL_FEE_BPS)) * 10_000
        ),
        "slip_live_applied_bps": round(
            (1 - _dynamic_sell_slip_factor(lv, lv["status"], fee_bps=SELL_FEE_BPS)) * 10_000
        ),
        "status_live": lv["status"], "status_paper": pp["status"],
        "exit_at_live": lv.get("exit_at"),
    })

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2, default=str)
print(f"\nSaved {len(rows)} rows -> {OUT_JSON}")


def summarize(tag, sub):
    if not sub: print(f"\n== {tag}: (empty) =="); return
    live_pct = [r["pnl_live_pct"] for r in sub]
    paper_pct = [r["pnl_paper_pct"] for r in sub]
    sim_pct = [r["pnl_sim_pct"] for r in sub if r["pnl_sim_pct"] is not None]
    live_usd = [r["pnl_live_usd"] for r in sub]
    paper_usd = [r["pnl_paper_usd"] for r in sub]
    sim_usd = [r["pnl_sim_usd"] for r in sub if r["pnl_sim_usd"] is not None]
    # divergences
    lp = [r["pnl_live_pct"] - r["pnl_paper_pct"] for r in sub]
    ls = [r["pnl_live_pct"] - r["pnl_sim_pct"] for r in sub if r["pnl_sim_pct"] is not None]
    sp = [r["pnl_sim_pct"] - r["pnl_paper_pct"] for r in sub if r["pnl_sim_pct"] is not None]
    entry_lp = []
    exit_lp = []
    for r in sub:
        el, ep = float(r["entry_live"] or 0), float(r["entry_paper"] or 0)
        xl, xp = float(r["exit_live"] or 0), float(r["exit_paper"] or 0)
        if el > 0 and ep > 0: entry_lp.append((el/ep - 1) * 100)
        if xl > 0 and xp > 0: exit_lp.append((xl/xp - 1) * 100)
    slip_l = [float(r["slip_live_bps"]) for r in sub if r["slip_live_bps"] is not None]
    slip_p_eff = [float(r["slip_paper_effective_bps"]) for r in sub if r["slip_paper_effective_bps"] is not None]
    slip_l_mod = [float(r["slip_live_applied_bps"]) for r in sub if r["slip_live_applied_bps"] is not None]
    slip_delta = [rl - rp for rl, rp in zip(slip_l, slip_p_eff)] if slip_l and slip_p_eff and len(slip_l)==len(slip_p_eff) else []

    def m(x, fmt="{:+.2f}"):
        return f"med={fmt.format(st.median(x))} mean={fmt.format(st.mean(x))}" if x else "n/a"

    print(f"\n== {tag}  N={len(sub)} ==")
    print(f"  PnL%   sim : {m(sim_pct)}   paper : {m(paper_pct)}   live : {m(live_pct)}")
    print(f"  PnL$   sim : sum={sum(sim_usd):+.2f}  paper : sum={sum(paper_usd):+.2f}  live : sum={sum(live_usd):+.2f}")
    print(f"  div   live-paper pp : {m(lp)}   live-sim pp : {m(ls)}   sim-paper pp : {m(sp)}")
    print(f"  entry live-paper % : {m(entry_lp)}")
    print(f"  exit  live-paper % : {m(exit_lp)}")
    print(f"  slip  live ACTUAL bps        : {m(slip_l, '{:.0f}')}")
    print(f"  slip  sim-model applied bps  : {m(slip_p_eff, '{:.0f}')}  (paper), {m(slip_l_mod, '{:.0f}')}  (if sim-modelled on live)")
    print(f"  slip  live_actual - sim_model: {m(slip_delta, '{:.0f}')}")


summarize("ALL",             rows)
summarize("non-pump.fun",    [r for r in rows if not r["is_pump"]])
summarize("pump.fun only",   [r for r in rows if r["is_pump"]])

# Per-strategy breakdown
per_strat = defaultdict(list)
for r in rows: per_strat[r["strategy"]].append(r)
print("\n== Per-strategy (all incl. pump) ==")
hdr = f"{'Strategy':<26} {'N':>3} {'sim%':>8} {'paper%':>8} {'live%':>8} {'L-P pp':>8} {'L-S pp':>8} {'$ live':>9} {'$ paper':>9} {'$ sim':>9}"
print(hdr); print("-"*len(hdr))
for s, sub in sorted(per_strat.items(), key=lambda x: -len(x[1])):
    if len(sub) < 3: continue
    live_pct = [r["pnl_live_pct"] for r in sub]
    paper_pct = [r["pnl_paper_pct"] for r in sub]
    sim_pct = [r["pnl_sim_pct"] for r in sub if r["pnl_sim_pct"] is not None]
    lp = [r["pnl_live_pct"] - r["pnl_paper_pct"] for r in sub]
    ls = [r["pnl_live_pct"] - r["pnl_sim_pct"] for r in sub if r["pnl_sim_pct"] is not None]
    live_usd = sum(r["pnl_live_usd"] for r in sub)
    paper_usd = sum(r["pnl_paper_usd"] for r in sub)
    sim_usd = sum(r["pnl_sim_usd"] for r in sub if r["pnl_sim_usd"] is not None)
    sim_med = st.median(sim_pct) if sim_pct else float('nan')
    ls_med = st.median(ls) if ls else float('nan')
    print(f"{s:<26} {len(sub):>3} {sim_med:>+7.2f}% {st.median(paper_pct):>+7.2f}% {st.median(live_pct):>+7.2f}% "
          f"{st.median(lp):>+7.2f} {ls_med:>+7.2f} {live_usd:>+8.2f} {paper_usd:>+8.2f} {sim_usd:>+8.2f}")
