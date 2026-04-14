"""
Backtest 5 novel overlays on top of FAST_TP50_SL30 and DTRAIL10_ACT15_SL70 bases.
Overlays:
  jup_ds_div        — exit if |Jupiter/DS - 1| > 15% AND Jupiter < DS
  liq_breaker       — exit if liq_change_pct < -25% in 1 tick
  median3_trail     — decision_price = median of last 3 ticks
  vol_exhaust       — tighten trail if volume stagnant 2 consecutive ticks
  slippage_tp       — shift TP up by slippage_bps when Jup-DS delta high
Base strategies:
  FAST  : TP=+50%, SL=-30%, 30min timeout
  DTRAIL: trail=10%, act=15%, SL=-70%, 120min timeout
"""
import os, sys
from datetime import datetime
from collections import defaultdict
from statistics import median
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

SINCE = "2026-04-08T00:00:00+00:00"

# Overlay tuning knobs
JUP_DS_DIV_THRESH = 0.15          # exit if Jupiter < DS by 15%
LIQ_DROP_THRESH = -25.0           # exit if liq drops 25% in 1 tick
MEDIAN_WINDOW = 3
VOL_STAGNATION_TICKS = 2          # 2 consecutive non-increasing volume ticks
VOL_TIGHTEN_MULT = 0.5            # halve trail when volume exhausted
SLIPPAGE_BUMP_BPS = 10000         # full bump = 100% of delta shifted into TP


def fetch_all(table, **f):
    out, step, offset = [], 1000, 0
    while True:
        q = sb.table(table).select("*")
        for k, v in f.items():
            if k.startswith("gte_"): q = q.gte(k[4:], v)
            elif k.startswith("eq_"): q = q.eq(k[3:], v)
            elif k == "in_token": q = q.in_("token_address", v)
        q = q.range(offset, offset+step-1).order("fetched_at" if "tick" in table else "created_at")
        r = q.execute()
        if not r.data: break
        out.extend(r.data)
        if len(r.data) < step: break
        offset += step
    return out


def merge_streams(ticks):
    """Merge DS+Jupiter ticks by nearest time. Returns list of
    {t, jp, ds, vol, liq, liq_change} — one row per Jupiter tick."""
    ds = []
    jp = []
    for tk in ticks:
        t = tk["fetched_at"]
        p = float(tk["price_usd"])
        if p <= 0: continue
        if tk["source"] == "jupiter":
            jp.append((t, p))
        elif tk["source"] in ("fast", "full", "live"):
            ds.append({
                "t": t, "p": p,
                "vol": float(tk.get("volume_usd") or 0),
                "liq": float(tk.get("liquidity_usd") or 0),
                "liq_change": float(tk.get("liq_change_pct") or 0),
            })
    jp.sort(); ds.sort(key=lambda x: x["t"])

    merged = []
    i_ds = 0
    for tj, pj in jp:
        # find DS tick with time closest to tj (prefer same-or-before)
        while i_ds + 1 < len(ds) and ds[i_ds + 1]["t"] <= tj:
            i_ds += 1
        ds_ref = ds[i_ds] if ds else None
        if ds_ref is None:
            continue
        merged.append({
            "t": tj, "jp": pj, "ds": ds_ref["p"],
            "vol": ds_ref["vol"], "liq": ds_ref["liq"], "liq_change": ds_ref["liq_change"],
        })
    return merged


def run_strategy(merged, entry, base, overlays=None, horizon_min=120,
                 trail_pct=0.10, act_pct=0.15, sl_pct=0.70,
                 tp_pct=0.50, fsl_pct=0.30):
    """Run a single strategy over merged tick stream. overlays is a set/list.
    Returns (pnl_pct, reason)."""
    if not merged:
        return None, "no_data"
    if overlays is None:
        overlays = set()
    if isinstance(overlays, str):
        overlays = {overlays}
    else:
        overlays = set(overlays)

    entry_time = datetime.fromisoformat(merged[0]["t"].replace("Z", "+00:00"))
    high = entry
    price_history = []
    vol_history = []
    vol_stagnant_count = 0
    active_trail_mult = 1.0

    for row in merged:
        t = datetime.fromisoformat(row["t"].replace("Z", "+00:00"))
        elapsed = (t - entry_time).total_seconds() / 60
        if elapsed > horizon_min:
            pnl = (row["jp"] / entry) - 1
            return pnl, "timeout"

        # Decision price: median3 overlay smooths
        if "median3_trail" in overlays:
            price_history.append(row["jp"])
            if len(price_history) > MEDIAN_WINDOW:
                price_history = price_history[-MEDIAN_WINDOW:]
            dec_price = median(price_history)
        else:
            dec_price = row["jp"]

        exec_price = row["jp"]

        # ===== OVERLAY DEFENSIVE EXITS =====
        if "jup_ds_div" in overlays:
            if row["ds"] > 0 and row["jp"] / row["ds"] - 1 < -JUP_DS_DIV_THRESH:
                return ((exec_price / entry) - 1), "jup_ds_div"

        if "liq_breaker" in overlays:
            if row["liq_change"] < LIQ_DROP_THRESH:
                return ((exec_price / entry) - 1), "liq_breaker"

        if "vol_exhaust" in overlays:
            vol_history.append(row["vol"])
            if len(vol_history) >= 3:
                delta1 = vol_history[-1] - vol_history[-2]
                delta2 = vol_history[-2] - vol_history[-3]
                if delta1 <= 0 and delta2 <= 0:
                    vol_stagnant_count += 1
                else:
                    vol_stagnant_count = 0
            active_trail_mult = VOL_TIGHTEN_MULT if vol_stagnant_count >= VOL_STAGNATION_TICKS else 1.0

        # ===== BASE STRATEGY LOGIC =====
        high = max(high, dec_price)

        if base == "FAST":
            tp_thresh = 1 + tp_pct
            if "slippage_tp" in overlays and row["ds"] > 0:
                slip = abs(row["jp"] / row["ds"] - 1)
                tp_thresh = 1 + tp_pct + slip
            tp = entry * tp_thresh
            sl = entry * (1 - fsl_pct)
            if dec_price >= tp:
                return ((tp / entry) - 1), "tp_hit"
            if dec_price <= sl:
                return ((sl / entry) - 1), "sl_hit"

        elif base == "DTRAIL":
            activation = entry * (1 + act_pct)
            sl = entry * (1 - sl_pct)
            if dec_price <= sl:
                return ((sl / entry) - 1), "sl_hit"
            if high >= activation:
                effective_trail = trail_pct * active_trail_mult
                trigger = high * (1 - effective_trail)
                if dec_price <= trigger and trigger > entry:
                    return ((exec_price / entry) - 1), "trail_stop"

    last_p = merged[-1]["jp"]
    return ((last_p / entry) - 1), "end_of_data"


def main():
    print("Loading trades since Apr 8...")
    trades = fetch_all("paper_trades", gte_created_at=SINCE)
    # Focus on non-shadow rt trades for relevance
    trades = [t for t in trades if t.get("source") == "rt"]
    print(f"  {len(trades)} rt trades")

    tokens = list({t["token_address"] for t in trades if t.get("token_address")})
    print(f"  {len(tokens)} unique tokens")

    print("Loading ticks...")
    ticks = fetch_all("price_ticks", gte_fetched_at=SINCE, in_token=tokens)
    print(f"  {len(ticks)} ticks")

    by_token = defaultdict(list)
    for t in ticks:
        by_token[t["token_address"]].append(t)

    scenarios = [
        ("FAST",   "baseline",                       ()),
        ("FAST",   "median3",                        ("median3_trail",)),
        ("FAST",   "slippage_tp",                    ("slippage_tp",)),
        ("FAST",   "median3+slippage_tp",            ("median3_trail", "slippage_tp")),
        ("FAST",   "median3+slippage+div",           ("median3_trail", "slippage_tp", "jup_ds_div")),
        ("FAST",   "median3+slippage+liq",           ("median3_trail", "slippage_tp", "liq_breaker")),
        ("FAST",   "all_overlays",                   ("median3_trail", "slippage_tp", "jup_ds_div", "liq_breaker", "vol_exhaust")),
        ("DTRAIL", "baseline",                       ()),
        ("DTRAIL", "jup_ds_div",                     ("jup_ds_div",)),
        ("DTRAIL", "jup_ds_div+liq",                 ("jup_ds_div", "liq_breaker")),
        ("DTRAIL", "jup_ds_div+median3",             ("jup_ds_div", "median3_trail")),
        ("DTRAIL", "jup_ds_div+vol_exhaust",         ("jup_ds_div", "vol_exhaust")),
        ("DTRAIL", "jup_ds_div+liq+vol_exhaust",     ("jup_ds_div", "liq_breaker", "vol_exhaust")),
        ("DTRAIL", "all_overlays",                   ("jup_ds_div", "liq_breaker", "median3_trail", "vol_exhaust")),
    ]
    scenario_keys = [(b, lbl) for b, lbl, _ in scenarios]
    results = {k: [] for k in scenario_keys}
    exit_reasons = {k: defaultdict(int) for k in scenario_keys}

    first_calls = {}
    for t in trades:
        key = (t["token_address"], t["strategy"])
        if key not in first_calls or t["created_at"] < first_calls[key]["created_at"]:
            first_calls[key] = t
    dedup_trades = list(first_calls.values())
    print(f"  After dedup: {len(dedup_trades)} unique (token, strategy) first calls")

    for trade in dedup_trades:
        addr = trade["token_address"]
        if addr not in by_token:
            continue
        entry = float(trade.get("entry_price") or 0)
        if entry <= 0:
            continue
        # Keep only ticks after trade open
        tks_after = [tk for tk in by_token[addr] if tk["fetched_at"] >= trade["created_at"]]
        merged = merge_streams(tks_after)
        if len(merged) < 3:
            continue

        for base, label, overlays in scenarios:
            horizon = 30 if base == "FAST" else 120
            pnl, reason = run_strategy(merged, entry, base, overlays, horizon_min=horizon)
            if pnl is None:
                continue
            results[(base, label)].append(pnl)
            exit_reasons[(base, label)][reason] += 1

    print(f"\n{'Base':<8}{'Overlays':<30}{'N':>6}{'Avg%':>9}{'WR%':>7}{'MaxDD$':>10}{'Total$':>10}{'vs Base':>10}")
    print("-" * 92)
    for base in ("FAST", "DTRAIL"):
        baseline_avg = None
        for b, lbl, _ovs in scenarios:
            if b != base: continue
            pnls = results[(b, lbl)]
            if not pnls: continue
            avg = sum(pnls) / len(pnls) * 100
            wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
            cumul = 0; peak = 0; max_dd = 0
            for p in pnls:
                cumul += p * 10
                peak = max(peak, cumul)
                max_dd = max(max_dd, peak - cumul)
            tot = sum(p * 10 for p in pnls)
            if lbl == "baseline":
                baseline_avg = avg
                delta_str = ""
            else:
                delta = avg - (baseline_avg or 0)
                delta_str = f"{delta:+.2f}pp"
            print(f"{b:<8}{lbl:<30}{len(pnls):>6}{avg:>8.2f}%{wr:>6.1f}%{max_dd:>9.0f}${tot:>9.0f}${delta_str:>10}")
        print()

    # Show exit reason mix for biggest movers
    print("\nExit reason breakdown (selected):")
    for b, lbl, _ovs in scenarios:
        if lbl == "baseline":
            continue
        reasons = exit_reasons[(b, lbl)]
        total = sum(reasons.values())
        if total == 0:
            continue
        top = sorted(reasons.items(), key=lambda x: -x[1])[:4]
        summary = ", ".join(f"{r}={n}" for r, n in top)
        print(f"  {b:<8}{lbl:<30} {summary}")


if __name__ == "__main__":
    main()
