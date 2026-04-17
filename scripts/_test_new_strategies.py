"""Test 7 candidate strategy ideas on real post-v132 shadow data.

Strategies tested (entry filter + exit logic combos):
  1. BASELINE_BE25       — BE25_TP80_SL30 (sanity check vs ground truth)
  2. CONFIRM2_BE25       — only enter if n_kol_confirmations >= 2
  3. HIGHLIQ_BE25        — only enter if rt_liquidity_usd >= 20000
  4. NOPUMP_BE25         — skip pump.fun bonding tokens
  5. EARLY_DUMP_BE25     — early exit if price < entry × 0.80 in first 5min
  6. MOONBAG_30_70       — 30% TP50 + 70% TP500 (or BE-protected timeout)
  7. CONFIRM2_HIGHLIQ_BE25 — combined filters (most selective)

Plus variations of TP200 / SL30 for asymmetric payoff testing.

Replays via sim._replay_trade_orchestrated using shadow data ground truth.
"""
from __future__ import annotations
import os
import sys
import statistics
from datetime import datetime, timedelta, timezone
from collections import defaultdict

from dotenv import load_dotenv
from supabase import create_client

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SCRAPER = os.path.join(ROOT, "scraper")
sys.path.insert(0, SCRAPER)

load_dotenv(os.path.join(SCRAPER, ".env"))
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

import sim  # noqa
from paper_trader import _evaluate_trade_exit, _last_eval_ts  # noqa

POST_V132 = "2026-04-13T20:00:00Z"
POSITION_USD = 50.0


# ----- Universe loader (one trade per token, with full metadata) -----
def load_universe():
    """Get unique-per-token RT trades with all metadata needed for filters."""
    print("Loading universe...")
    cols = ("id,token_address,created_at,entry_price,sl_price,tp_price,"
            "horizon_minutes,position_usd,rt_liquidity_usd,"
            "dex_spot_price_at_entry,strategy,tranche_label,kol_group,"
            "rt_score,kol_score,n_kol_confirmations,rt_is_pump_fun,"
            "entry_mcap,rt_volume_24h,rt_buy_sell_ratio,is_shadow")
    rows = []
    page = 1000
    off = 0
    while True:
        r = (sb.table("paper_trades").select(cols)
               .eq("source", "rt").gte("created_at", POST_V132)
               .order("created_at").range(off, off + page - 1).execute().data)
        if not r:
            break
        rows.extend(r)
        if len(r) < page:
            break
        off += page

    by_token = {}
    for r in rows:
        addr = r["token_address"]
        if addr not in by_token:
            by_token[addr] = r  # earliest entry per token
    return list(by_token.values())


# ----- Tick fetcher with cache -----
_TICK_CACHE = {}


def fetch_ticks_cached(addr: str, t_start: str, t_end: str):
    key = f"{addr}:{t_start}"
    if key in _TICK_CACHE:
        return _TICK_CACHE[key]
    rows = (sb.table("price_ticks")
              .select("price_usd,fetched_at,source")
              .eq("token_address", addr)
              .gte("fetched_at", t_start).lte("fetched_at", t_end)
              .order("fetched_at").execute().data)
    _TICK_CACHE[key] = rows
    return rows


# ----- Entry filters -----
TOP_KOLS = {"FrenzGems", "jadendegens", "gubbinscalls", "Archerrgambles",
            "ChadleyGambles123", "zcallz"}


def make_filter(min_kol_confs=0, min_liq=0, max_liq=None, skip_pumpfun=False,
                min_score=0, max_score=100,
                min_mcap=0, max_mcap=None,
                min_age=0, max_age=None,
                kol_whitelist=None, skip_zero_liq=False):
    def fn(t):
        if (t.get("n_kol_confirmations") or 0) < min_kol_confs:
            return False
        liq = float(t.get("rt_liquidity_usd") or 0)
        if liq < min_liq:
            return False
        if max_liq is not None and liq > max_liq:
            return False
        if skip_zero_liq and liq <= 0:
            return False
        if skip_pumpfun and (t.get("rt_is_pump_fun") or 0) > 0:
            return False
        sc = float(t.get("rt_score") or 0)
        if sc < min_score or sc > max_score:
            return False
        mc = float(t.get("entry_mcap") or 0)
        if mc < min_mcap:
            return False
        if max_mcap is not None and mc > max_mcap:
            return False
        age = float(t.get("rt_token_age_hours") or 0)
        if age < min_age:
            return False
        if max_age is not None and age > max_age:
            return False
        if kol_whitelist and (t.get("kol_group") or "") not in kol_whitelist:
            return False
        return True
    return fn


# ----- Custom exit logic: EARLY_DUMP_CUT -----
def replay_early_dump_cut(fake, jp_ticks, ds_ticks, src, poll_sec, dump_threshold=-0.20,
                          window_min=5):
    """Standard replay BUT exit immediately if price < entry × (1 + dump_threshold)
    within first window_min minutes. Catches early-dump rugs faster than SL30/50."""
    entry_time = datetime.fromisoformat(fake["created_at"].replace("Z", "+00:00"))
    entry_price = float(fake["entry_price"])
    cutoff_price = entry_price * (1 + dump_threshold)
    window_end = entry_time + timedelta(minutes=window_min)

    # Check ticks in window for early dump
    all_ticks = sorted(jp_ticks + ds_ticks, key=lambda t: t["fetched_at"])
    for tk in all_ticks:
        ts = datetime.fromisoformat(tk["fetched_at"].replace("Z", "+00:00"))
        if ts < entry_time:
            continue
        if ts > window_end:
            break
        p = float(tk["price_usd"])
        if p > 0 and p <= cutoff_price:
            # Early dump triggered — exit at this price with slippage
            sell_slip = 1 - 250/10_000  # heavy slip on early dump (rug)
            exit_p = p * sell_slip
            pnl_pct = exit_p / entry_price - 1
            return {"status": "early_dump_cut",
                    "exit_reason": "early_dump_cut",
                    "pnl_pct": pnl_pct,
                    "exit_minutes": int((ts - entry_time).total_seconds() / 60)}
    # No early dump → fall through to standard sim
    return sim._replay_trade_orchestrated(fake, ds_ticks, jp_ticks, src, poll_sec=poll_sec)


# ----- Custom: MOONBAG_30_70 -----
def replay_moonbag(fake, jp_ticks, ds_ticks, src, poll_sec):
    """30% sells at TP50, 70% rides with SL=entry until TP500 or timeout 4h.
    Avg pnl = 0.3 × tranche1_pnl + 0.7 × tranche2_pnl."""
    entry_price = float(fake["entry_price"])

    # Tranche 1: TP50, SL30, horizon 2h
    fake1 = dict(fake)
    fake1["tp_price"] = entry_price * 1.50
    fake1["sl_price"] = entry_price * 0.70
    fake1["horizon_minutes"] = 120
    res1 = sim._replay_trade_orchestrated(fake1, ds_ticks, jp_ticks, src, poll_sec=poll_sec)
    pnl1 = res1["pnl_pct"] if res1 else 0

    # Tranche 2: TP500 (5x), SL=entry (BE), horizon 4h
    fake2 = dict(fake)
    fake2["tp_price"] = entry_price * 5.00
    fake2["sl_price"] = entry_price * 1.00  # BE
    fake2["horizon_minutes"] = 240
    res2 = sim._replay_trade_orchestrated(fake2, ds_ticks, jp_ticks, src, poll_sec=poll_sec)
    pnl2 = res2["pnl_pct"] if res2 else 0

    pnl_combined = 0.3 * pnl1 + 0.7 * pnl2
    return {"status": "moonbag",
            "exit_reason": "moonbag",
            "pnl_pct": pnl_combined,
            "exit_minutes": 240}


# ----- Strategy specs -----
STRATEGIES_TO_TEST = [
    # === BASELINES (sanity) ===
    ("BASELINE_BE25",            make_filter(),                                   "BE25_TP80_SL30", None, "ema"),
    ("BASELINE_TP200_SL40",      make_filter(),                                   "TP200_SL40",     None, "jupiter"),
    ("MOONBAG_30_70",            make_filter(),                                   "MOONBAG",        "moonbag", "jupiter"),

    # === SCORE-BASED ===
    ("HIGHSCORE_BE25_score40",   make_filter(min_score=40),                       "BE25_TP80_SL30", None, "ema"),
    ("HIGHSCORE_TP200_score40",  make_filter(min_score=40),                       "TP200_SL40",     None, "jupiter"),
    ("HIGHSCORE_TP200_score30",  make_filter(min_score=30),                       "TP200_SL40",     None, "jupiter"),

    # === MCAP-BASED ===
    ("MCAP_30_100K_TP200",       make_filter(min_mcap=30000, max_mcap=100000),    "TP200_SL40",     None, "jupiter"),
    ("MCAP_30_100K_BE25",        make_filter(min_mcap=30000, max_mcap=100000),    "BE25_TP80_SL30", None, "ema"),
    ("MCAP_LT30K_TP200",         make_filter(max_mcap=30000),                     "TP200_SL40",     None, "jupiter"),
    ("MCAP_100_500K_BE25",       make_filter(min_mcap=100000, max_mcap=500000),   "BE25_TP80_SL30", None, "ema"),

    # === AGE-BASED ===
    ("FRESH_LT1H_TP200",         make_filter(max_age=1),                          "TP200_SL40",     None, "jupiter"),
    ("OLDER_GT4H_BE25",          make_filter(min_age=4),                          "BE25_TP80_SL30", None, "ema"),

    # === KOL WHITELIST ===
    ("TOPKOLS_BE25",             make_filter(kol_whitelist=TOP_KOLS),             "BE25_TP80_SL30", None, "ema"),
    ("TOPKOLS_TP200",            make_filter(kol_whitelist=TOP_KOLS),             "TP200_SL40",     None, "jupiter"),

    # === LIQUIDITY ===
    ("NOZEROLIQ_BE25",           make_filter(skip_zero_liq=True),                 "BE25_TP80_SL30", None, "ema"),
    ("NOZEROLIQ_TP200",          make_filter(skip_zero_liq=True),                 "TP200_SL40",     None, "jupiter"),

    # === COMBOS ===
    ("MCAP30_100K_TOPKOLS_TP200",make_filter(min_mcap=30000, max_mcap=100000, kol_whitelist=TOP_KOLS), "TP200_SL40", None, "jupiter"),
    ("HIGHSCORE_MCAP30_TP200",   make_filter(min_score=30, min_mcap=30000),       "TP200_SL40",     None, "jupiter"),
    ("OLDER_NOZERO_TP200",       make_filter(min_age=4, skip_zero_liq=True),      "TP200_SL40",     None, "jupiter"),
]

# Map strategy name → (tp_mult, sl_mult, horizon_min)
EXIT_PARAMS = {
    "BE25_TP80_SL30": (1.80, 0.70, 30),
    "TP200_SL40":     (3.00, 0.60, 240),
    "MOONBAG":        (1.50, 0.70, 120),  # ignored, custom replay handles it
}


# ----- Main -----
def run():
    universe = load_universe()
    print(f"Universe: {len(universe)} unique tokens")

    print("\n" + "=" * 100)
    print(f"{'Strategy':<25}{'N_pass':>8}{'N_eval':>8}{'WR%':>6}{'avg%':>9}{'med%':>9}"
          f"{'$/day(50)':>11}  notes")
    print("=" * 100)

    results = []
    for name, filt_fn, base_strat, custom, orch in STRATEGIES_TO_TEST:
        tp_mult, sl_mult, horizon_min = EXIT_PARAMS[base_strat]
        passed = [t for t in universe if filt_fn(t)]
        n_pass = len(passed)
        if n_pass < 5:
            print(f"{name:<25}{n_pass:>8}  --- too few tokens ---")
            continue

        pnls = []
        for u in passed:
            addr = u["token_address"]
            entry_ts = datetime.fromisoformat(u["created_at"].replace("Z", "+00:00"))
            t_start = u["created_at"]
            t_end = (entry_ts + timedelta(minutes=horizon_min)).isoformat().replace("+00:00", "Z")
            try:
                ticks = fetch_ticks_cached(addr, t_start, t_end)
            except Exception:
                continue
            if len(ticks) < 3:
                continue
            ds_t = [t for t in ticks if t.get("source") in ("fast", "full", "live")]
            jp_t = [t for t in ticks if t.get("source") == "jupiter"]
            entry_price = float(u["entry_price"])
            fake = {
                "id": f"sim_{u['id']}",
                "entry_price": entry_price,
                "sl_price": entry_price * sl_mult,
                "tp_price": entry_price * tp_mult if tp_mult else None,
                "position_usd": 10.0,
                "strategy": base_strat if base_strat != "MOONBAG" else "BE25_TP80_SL30",
                "tranche_label": "main",
                "horizon_minutes": horizon_min,
                "created_at": u["created_at"],
                "high_price_seen": entry_price,
                "rt_liquidity_usd": u.get("rt_liquidity_usd"),
                "dex_spot_price_at_entry": entry_price,
            }
            if custom == "early_dump":
                res = replay_early_dump_cut(fake, jp_t, ds_t, orch, 30)
            elif custom == "moonbag":
                res = replay_moonbag(fake, jp_t, ds_t, orch, 30)
            else:
                res = sim._replay_trade_orchestrated(fake, ds_t, jp_t, orch, poll_sec=30)
            if res is None:
                continue
            pnls.append(res["pnl_pct"])

        n_eval = len(pnls)
        if n_eval < 5:
            print(f"{name:<25}{n_pass:>8}{n_eval:>8}  --- insufficient ticks ---")
            continue
        wr = sum(1 for p in pnls if p > 0) / n_eval * 100
        avg = statistics.mean(pnls) * 100
        med = statistics.median(pnls) * 100

        # $/jour at $50/trade — assume same trade_per_day rate as BASELINE
        # (filters reduce universe → fewer trades/day proportionally)
        trade_rate = n_pass / max(1, len(universe)) * 18  # 18 trades/day baseline
        dollars_per_day = 50 * (avg / 100) * trade_rate

        notes = ""
        if "CONFIRM2" in name:
            notes = f"({n_pass}/{len(universe)}={n_pass*100//len(universe)}% selectivity)"
        elif "HIGHLIQ" in name:
            notes = f"liq>=20K, {n_pass}/{len(universe)}"
        elif "NOPUMP" in name:
            notes = f"no pump.fun, {n_pass}/{len(universe)}"
        elif "HIGHSCORE" in name:
            notes = f"score>=60, {n_pass}/{len(universe)}"

        print(f"{name:<25}{n_pass:>8}{n_eval:>8}{wr:>5.0f}%{avg:>+8.2f}%{med:>+8.2f}%{dollars_per_day:>+10.2f}$  {notes}")
        results.append((name, n_pass, n_eval, wr, avg, med, dollars_per_day))

    print("\n" + "=" * 100)
    print("Sorted by $/day:")
    print("=" * 100)
    for r in sorted(results, key=lambda x: -x[6]):
        print(f"  {r[0]:<25} avg={r[4]:+.2f}% wr={r[3]:.0f}% n={r[2]:>3} → {r[6]:+.2f}$/jour")


if __name__ == "__main__":
    run()
