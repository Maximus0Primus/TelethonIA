"""
Paper Trading System v6 — Multi-strategy with tranche support + portfolio allocation.

Strategies run in parallel per token:
- TP30_SL10:    100% at 1.3x, -10% SL, 12h horizon (v86: replaces TP30_SL50, 3:1 R:R)
- TP50_SL30:    100% at 1.5x, -30% SL, 24h horizon (RT best 7d: +8% ROI, 50% WR)
- TP100_SL50:   100% at 2x,   -50% SL, 48h horizon (v6: wide SL to survive memecoin volatility)
- FRESH_MICRO:  100% at 1.3x, -70% SL, 24h (score 10-49, fresh KOL, micro-cap)
- QUICK_SCALP:  100% at 1.5x, ~no SL, 6h timeout (score 10-49, momentum)
- TP50_SL15:    100% at 1.5x, -15% SL, 24h horizon (tight SL, best EV on KOL-filtered tokens)
- TP30_SL30:    100% at 1.3x, -30% SL, 12h horizon (symmetric risk/reward, cuts losses early)
- TP50_SL50:    100% at 1.5x, -50% SL, 24h horizon (symmetric risk/reward, room to recover)

Deprecated (data shows negative EV — kept in code for backtesting, removed from active_strategies):
- TP30_SL50:   -2.5% ROI, 49% WR. Asymmetric R:R — wins too small vs losses.
- TP100_SL30:  -14% ROI live, 12% WR. Tight SL (-30%) too small for 2x target.
- MOONBAG:     -45.6% ROI live, 8% WR.
- WIDE_RUNNER: -70.4% ROI live, 0% WR in RT.
- SCALE_OUT:   -29% ROI live, 6% WR.
- QUICK_SCALP: v92 deprecated — 33% WR, negative PnL. 6h too short for TP30.
- TP30_SL10:   v92 deprecated — tight SL (-10%) stops out too fast.
- TP50_SL15:   v92 deprecated — tight SL (-15%) kills trades that would recover.
- TP30_SL30:   v92 deprecated — TP30 too small to overcome slippage.

Each tranche = 1 row in paper_trades. SL triggers close ALL open tranches
for the same token+strategy. Moonbag tranches (tp_price=NULL) only close
on SL or timeout.

v3: Score-weighted portfolio allocation. $50 budget per cycle split
proportionally by token score. Tracks position_usd and pnl_usd.
v4: Data-driven strategies with entry filters (STRATEGY_FILTERS).
v5: +TP30_SL50, dedup cooldown 24h default, MOONBAG/WIDE_RUNNER deprecated.
"""

import logging
from datetime import datetime, timezone, timedelta

import requests
import time as _time_mod

logger = logging.getLogger(__name__)

# v67: Monitoring — conditional import
try:
    from monitor import metrics as _metrics, estimate_egress as _estimate_egress
    _monitoring = True
except ImportError:
    _monitoring = False

DEXSCREENER_BATCH_URL = "https://api.dexscreener.com/tokens/v1/solana/{addresses}"
BATCH_SIZE = 30

# v122: Track which addresses got Jupiter price override (for alert annotation + tick logging)
_jupiter_overridden: set[str] = set()
_jupiter_prices_cache: dict[str, float] = {}  # v122: Jupiter prices for tick logging
_dex_prices_cache: dict[str, float] = {}  # v123: DexScreener prices preserved for tick logging

# v138: per-trade eval history accumulator. Maps trade_id -> [{"t","d","e","h"}, ...]
# Persisted to paper_trades.eval_history on close → guarantees 0% sim/real divergence.
_eval_history: dict[str, list[dict]] = {}
_EVAL_HISTORY_MAX_POLLS = 500  # cap memory usage on long-lived trades

# v88: Bot ML predictions — precomputed in GH Actions, read from Supabase
_BOT_PREDICTIONS: dict = {}  # {(token_address, strategy): gate_mult}

# --- Defaults (overridden by scoring_config.paper_trade_config) ---
TOP_N = 5
PORTFOLIO_BUDGET = 200.0  # v94: USD per cycle, score-weighted across top N (was 50)
DEDUP_COOLDOWN_HOURS = 24  # v5: was 0 — re-trading same token across cycles was the #1 PnL killer
CA_FILTER = True

# v125: Strategy definitions, fee constants, trail config, LAZY mode centralized in strategies.py
from strategies import (
    _DEFAULT_DEPRECATED, SHADOW_STRATEGIES,
    BUY_SLIPPAGE_BPS, SELL_SLIPPAGE_BPS, BUY_FEE_BPS, SELL_FEE_BPS,
    STRATEGIES, STRATEGY_FILTERS,
    _DECAY_RE, _TRAIL_RE, _DTRAIL_RE, _DIP_RE, _DIP_SPLIT_RE, _BE_RE,
    _get_decay_end, _get_trail_config,
    LAZY_STRATEGIES, LAZY_FAST_SEC, LAZY_FAST_WINDOW, LAZY_SLOW_SEC,
)

# v115: DIP_BUY in-memory watchlist — tracks tokens waiting for dip+bounce to open P2
# Key: (token_address, strategy_name) → tracking state
_dip_watchlist: dict[tuple, dict] = {}
_dip_watchlist_rebuilt = False


def _load_bot_predictions(client) -> None:
    """v88: Load precomputed bot ML predictions from Supabase (one query per cycle)."""
    global _BOT_PREDICTIONS
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        result = (
            client.table("bot_ml_predictions")
            .select("token_address, strategy, gate_mult")
            .gte("predicted_at", cutoff)
            .execute()
        )
        _BOT_PREDICTIONS = {
            (r["token_address"], r["strategy"]): float(r["gate_mult"])
            for r in (result.data or [])
        }
        if _BOT_PREDICTIONS:
            logger.info("bot_ml: loaded %d predictions", len(_BOT_PREDICTIONS))
    except Exception as e:
        logger.warning("bot_ml: load failed: %s", e)
        _BOT_PREDICTIONS = {}


def _bot_ml_gate(token: dict, strategy_name: str, config: dict | None = None) -> float:
    """v90: ML gate with inversion support. Model is anti-predictive (7d data N=189):
    blocked=60% WR, full=35% WR. Modes: disabled/normal/inverted."""
    mode = (config or {}).get("ml_gate_mode", "disabled")
    if mode == "disabled":
        return 1.0

    raw = _BOT_PREDICTIONS.get(
        (token.get("token_address"), strategy_name), 1.0
    )

    if mode == "inverted":
        # Anti-predictive model: what it blocks performs best. Flip the signal.
        if raw <= 0.0:
            return 1.5   # model said SKIP → BOOST 50%
        elif raw < 1.0:
            return 1.3   # model said HALF → BOOST 30%
        else:
            return 0.7   # model said FULL → REDUCE 30%

    return raw  # normal mode


def _passes_strategy_filter(token: dict, strategy_name: str) -> bool:
    """Check if a token passes the entry filter for a given strategy."""
    filt = STRATEGY_FILTERS.get(strategy_name)
    if not filt:
        return True  # no filter = always pass

    score = token.get("score", 0)
    if score < filt.get("min_score", 0) or score > filt.get("max_score", 100):
        return False
    mcap = float(token.get("market_cap") or 0)
    if filt.get("max_mcap") and mcap > filt["max_mcap"]:
        return False
    if filt.get("min_mcap") and mcap < filt["min_mcap"]:
        return False
    kf = float(token.get("kol_freshness") or 0)
    if kf < filt.get("min_kol_freshness", 0):
        return False
    mm = float(token.get("momentum_mult") or 1.0)
    if mm < filt.get("min_momentum_mult", 0):
        return False
    # v139: liquidity + rt_score gates (RT-side fields, set in safe_scraper)
    liq = float(token.get("_rt_liquidity_usd") or token.get("rt_liquidity_usd")
                or token.get("liquidity_usd") or 0)
    if liq < filt.get("min_liquidity_usd", 0):
        return False
    # v142: upper bound — for bonding-only / low-liq strats (BOND_FAST)
    if filt.get("max_liquidity_usd") and liq > filt["max_liquidity_usd"]:
        return False
    rt_score = float(token.get("_rt_score") or token.get("rt_score") or 0)
    if rt_score < filt.get("min_rt_score", 0):
        return False
    return True


def _fetch_price_fallback(address: str) -> float | None:
    """Fallback: fetch price from GeckoTerminal (no API key needed, on-chain data).
    Catches pool migrations and tokens DexScreener hasn't indexed yet."""
    try:
        resp = requests.get(
            f"https://api.geckoterminal.com/api/v2/networks/solana/tokens/{address}",
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        price = resp.json().get("data", {}).get("attributes", {}).get("price_usd")
        if price:
            return float(price)
    except Exception:
        pass
    return None


# v118: Price tick logger — records DexScreener spot prices for future backtesting
_last_tick_log: dict[str, float] = {}  # token_address -> last log timestamp
_last_dex_extra: dict[str, dict] = {}  # v121: token_address -> {volume_usd, liquidity_usd}
_last_tick_liq: dict[str, float] = {}  # v121: token_address -> last liquidity_usd (for rug detection)

# v132: Per-trade polling throttle (trade_id -> last_check_ts). In-memory only; reset on restart.
_last_check_ts: dict[int, float] = {}
# v132: EMA state per trade (trade_id -> ema_value) for ema_jupiter_N modes.
_ema_state: dict[int, float] = {}
# v134: Smoothing state buffers per trade (trade_id -> opaque state dict).
# Modes: median_3/5 (rolling buffer), winsor_p95 (prev price),
#        dual_confirm (prev price + last breach), hysteresis (armed flags).
_smooth_state: dict[int, dict] = {}

# v121: Cached SOL price for paper trade USD context
_cached_sol_price: float = 0.0
_cached_sol_price_ts: float = 0.0


# v132: Orchestration helpers
_DEFAULT_ORCH = {"polling_sec": 30, "price_source": "jupiter", "ema_window": 3}

def _strategy_orchestration(strategy: str, rt_config: dict | None) -> dict:
    """Return orchestration config for a strategy. Falls back to defaults.
    Config shape:  rt_trade_config.strategy_overrides.<STRAT> = {polling_sec, price_source, ema_window}
    price_source: jupiter | ds | hybrid | confirm | ema
    """
    if not rt_config:
        return dict(_DEFAULT_ORCH)
    overrides = rt_config.get("strategy_overrides", {}) or {}
    cfg = dict(_DEFAULT_ORCH)
    cfg.update(overrides.get(strategy, {}) or {})
    return cfg


def _should_poll_trade(trade_id: int, polling_sec: int) -> bool:
    """True if enough time elapsed since last check (v132 per-strategy polling)."""
    now = _time_mod.time()
    last = _last_check_ts.get(trade_id, 0)
    if (now - last) >= polling_sec:
        _last_check_ts[trade_id] = now
        return True
    return False


def _record_eval_poll(trade_id, now: datetime, decision_p: float | None,
                      exec_p: float | None, high_seen: float | None) -> None:
    """v138: append one poll to per-trade eval history for perfect sim replay."""
    if trade_id is None or decision_p is None or exec_p is None:
        return
    key = str(trade_id)
    hist = _eval_history.setdefault(key, [])
    if len(hist) >= _EVAL_HISTORY_MAX_POLLS:
        return  # hard cap to bound memory on stuck trades
    hist.append({
        "t": now.isoformat().replace("+00:00", "Z"),
        "d": float(decision_p),
        "e": float(exec_p),
        "h": float(high_seen) if high_seen is not None else None,
    })


def _flush_eval_history(trade_id) -> list[dict] | None:
    """v138: pop and return the accumulated eval history for a closing trade."""
    if trade_id is None:
        return None
    return _eval_history.pop(str(trade_id), None) or None


def _log_cache_snapshot(client) -> None:
    """v138: dump current paper_trader caches to cache_snapshots.
    One row per call. Caller invokes this once per loop tick after _fetch_prices_batch.
    Sim can replay from snapshots for ANY token (covers tokens without live trades)."""
    if not client or not _jupiter_prices_cache:
        return
    try:
        client.table("cache_snapshots").insert({
            "jp_prices": dict(_jupiter_prices_cache),
            "ds_prices": dict(_dex_prices_cache) if _dex_prices_cache else None,
            "n_tokens": len(_jupiter_prices_cache),
        }).execute()
    except Exception as e:
        logger.debug("cache_snapshot insert failed: %s", e)


def _decision_price(addr: str, strategy: str, trade_id: int, orch: dict,
                    trade: dict | None = None) -> tuple[float | None, float | None]:
    """Return (decision_price, exit_price_ref) based on strategy's price_source.
    exit_price_ref is always Jupiter (or fallback current) since live exec = Jupiter.

    v134: Added smoothing modes ported from sim.py._smooth_decision:
      median_3/5, winsor_p95, dual_confirm, ema_fast (w=2), ema_slow (w=8),
      hysteresis. volume_gated NOT ported (prod cache lacks per-tick volume).
    These modes all consume the Jupiter stream as input and return a smoothed
    decision price; exit_ref stays Jupiter (matches live Ultra fill).
    """
    jp = _jupiter_prices_cache.get(addr)
    ds = _dex_prices_cache.get(addr)
    src = orch.get("price_source", "jupiter")

    if src == "jupiter":
        return jp, jp
    if src == "ds":
        return ds, (jp if jp else ds)
    if src == "hybrid":
        # decision on DS, exit at Jupiter
        return (ds if ds else jp), (jp if jp else ds)
    if src == "confirm":
        if jp and ds:
            return ((jp + ds) / 2), jp
        return (jp or ds), (jp or ds)
    if src == "ema":
        if not jp:
            return None, (jp or ds)
        window = int(orch.get("ema_window", 3))
        alpha = 2 / (window + 1)
        prev = _ema_state.get(trade_id, jp)
        ema_val = alpha * jp + (1 - alpha) * prev
        _ema_state[trade_id] = ema_val
        return ema_val, jp

    # --- v134 smoothing modes (Jupiter stream with per-trade state) ---
    # All return (smoothed_decision, jp_exit_ref). Fall back to jp if stream empty.
    if not jp:
        return None, (jp or ds)

    state = _smooth_state.setdefault(trade_id, {})

    if src == "ema_fast" or src == "ema_slow":
        window = 2 if src == "ema_fast" else 8
        alpha = 2 / (window + 1)
        prev = _ema_state.get(trade_id, jp)
        ema_val = alpha * jp + (1 - alpha) * prev
        _ema_state[trade_id] = ema_val
        return ema_val, jp

    if src == "median_3" or src == "median_5":
        window = 3 if src == "median_3" else 5
        hist = state.setdefault("hist", [])
        hist.append(jp)
        if len(hist) > window:
            hist.pop(0)
        if len(hist) < window:
            return jp, jp  # warm-up: pass through
        return sorted(hist)[len(hist) // 2], jp

    if src == "winsor_p95":
        prev = state.get("prev_p", jp)
        delta = jp - prev
        cap = prev * 0.18  # p95 tick-to-tick move
        if delta > cap:
            p = prev + cap
        elif delta < -cap:
            p = prev - cap
        else:
            p = jp
        state["prev_p"] = p
        return p, jp

    if src == "dual_confirm":
        # Require 2 consecutive ticks on the same side of SL/TP before triggering.
        # Needs sl_price / tp_price from the trade dict.
        if trade is None:
            return jp, jp  # fall back to raw if trade context missing
        sl_price = float(trade.get("sl_price") or 0)
        tp_price = float(trade.get("tp_price") or 0) or None
        prev = state.get("prev_p", jp)
        state["prev_p"] = jp
        # If only current tick breaches (prev did not) -> return prev to suppress trigger
        if sl_price and jp <= sl_price and prev > sl_price:
            return prev, jp
        if tp_price and jp >= tp_price and prev < tp_price:
            return prev, jp
        return jp, jp

    if src == "hysteresis":
        if trade is None:
            return jp, jp
        sl_price = float(trade.get("sl_price") or 0)
        tp_price = float(trade.get("tp_price") or 0) or None
        armed_sl = state.setdefault("armed_sl", True)
        armed_tp = state.setdefault("armed_tp", True)
        # Re-arm after 2% retrace past the trigger
        if not armed_sl and sl_price and jp >= sl_price * 1.02:
            state["armed_sl"] = True
            armed_sl = True
        elif armed_sl and sl_price and jp <= sl_price:
            state["armed_sl"] = False
        if not armed_tp and tp_price and jp <= tp_price * 0.98:
            state["armed_tp"] = True
            armed_tp = True
        elif armed_tp and tp_price and jp >= tp_price:
            state["armed_tp"] = False
        # If disarmed, serve a price that doesn't retrigger
        if not state["armed_sl"] and sl_price and jp <= sl_price:
            return sl_price * 1.001, jp
        if not state["armed_tp"] and tp_price and jp >= tp_price:
            return tp_price * 0.999, jp
        return jp, jp

    # v142 C — jp_sampled_60s / _180s : sample `jp` at each bar boundary,
    # return that close throughout the bar. NOT a port of the OHLCV sim
    # (which emits O/L/H/C as 4 synthetic ticks per bar). This mode simply
    # freezes the decision price between 60s/180s boundaries, suppressing
    # intra-bar triggers. Utile pour tester "décision slow" vs "décision tick".
    # Pour le vrai port OHLCV avec émission 4-ticks, voir ohlc_burst_60s
    # dans check_paper_trades (emits a burst of 4 synthetic prices at each
    # bar boundary through `_decision_prices_burst`).
    if src == "jp_sampled_60s" or src == "jp_sampled_180s":
        bar_sec = 60 if src == "jp_sampled_60s" else 180
        now_ts = _time_mod.time()
        last_close_ts = state.get("ohlcv_ts", 0)
        cur_bar = int(now_ts // bar_sec) * bar_sec
        if cur_bar > last_close_ts:
            state["ohlcv_close"] = jp
            state["ohlcv_ts"] = cur_bar
        close = state.get("ohlcv_close", jp)
        return close, jp

    # v142 D — ohlc_burst_60s : accumule les ticks sur 60s, au bar close
    # renvoie une LIST de 4 prix synthétiques (O, L, H, C ordered bull/bear)
    # que le caller itère comme 4 evals successifs. Port littéral de
    # sim_engines.candles_to_synthetic_ticks() sur les ticks que nous
    # pollons réellement. LIMITATION : nos ticks ne contiennent que 2-4
    # snapshots par 60s, pas les milliers de trades d'exchange. Les wicks
    # entre 2 polls sont inaccessibles. Best-effort, pas un vrai port.
    if src == "ohlc_burst_60s":
        # Return jp as normal decision. The burst logic is handled in
        # check_paper_trades via _decision_prices_burst (separate helper).
        return jp, jp

    # v142 C — VWAP 5min. Volume-weighted avg of prices over sliding 5min
    # window. Volume per tick = delta of DexScreener rolling volume_usd
    # between polls. Weights heavy trading periods above light ones, better
    # reflecting actual execution-weighted market level than plain median/ema.
    if src == "vwap_5min":
        now_ts = _time_mod.time()
        buf = state.setdefault("vwap_buf", [])  # list of (ts, price, dv)
        prev_total_vol = state.get("vwap_last_vol")
        dex_extra = _last_dex_extra.get(addr, {})
        cur_vol = float(dex_extra.get("volume_usd") or 0)
        dv = max(0.0, cur_vol - prev_total_vol) if prev_total_vol is not None else 0.0
        state["vwap_last_vol"] = cur_vol
        if dv > 0:
            buf.append((now_ts, jp, dv))
        # Keep only last 5 min
        cutoff = now_ts - 300
        state["vwap_buf"] = [x for x in buf if x[0] >= cutoff]
        total_v = sum(v for _, _, v in state["vwap_buf"])
        if total_v <= 0:
            return jp, jp  # warm-up or zero volume
        vwap = sum(p * v for _, p, v in state["vwap_buf"]) / total_v
        return vwap, jp

    # v142 C — Twin-source confirmation. Requires BOTH Jupiter and DS to
    # breach the same threshold (SL or TP) before letting the trigger through.
    # Single-source breach → returns the non-breaching source to suppress the
    # false signal. Eliminates divergence-induced phantom triggers (common
    # 2-5% gap between DS spot and Jupiter quote on illiquid tokens).
    if src == "twin_confirm":
        if trade is None or ds is None or jp is None:
            return (jp or ds), (jp or ds)
        sl_price = float(trade.get("sl_price") or 0)
        tp_price = float(trade.get("tp_price") or 0) or None
        if sl_price:
            jp_breach = jp <= sl_price
            ds_breach = ds <= sl_price
            if jp_breach != ds_breach:
                # Only one source breaches → suppress (use the higher one)
                return max(jp, ds), jp
        if tp_price:
            jp_breach = jp >= tp_price
            ds_breach = ds >= tp_price
            if jp_breach != ds_breach:
                return min(jp, ds), jp
        return jp, jp

    # Unknown mode: fall back to jupiter
    return jp, jp


# ---------------------------------------------------------------------------
# v142 D — OHLC burst emission (port of sim_engines.candles_to_synthetic_ticks)
# ---------------------------------------------------------------------------
# Per-trade tick buffer for ohlc_burst_60s mode. Accumulates (ts, price) pairs
# over each 60s window. At bar boundary, the caller queries _decision_prices_burst
# which returns the OHLC-ordered synthetic tick sequence.
_ohlc_buffer: dict[int, list[tuple[float, float]]] = {}  # trade_id -> [(ts, price), ...]
_ohlc_last_bar: dict[int, int] = {}  # trade_id -> last bar_start ts processed


def _record_ohlc_tick(trade_id: int, price: float) -> None:
    """Append a tick to per-trade OHLC buffer. Called from check_paper_trades
    at each poll to accumulate tick history for burst emission."""
    if trade_id is None or price is None or price <= 0:
        return
    now_ts = _time_mod.time()
    buf = _ohlc_buffer.setdefault(trade_id, [])
    buf.append((now_ts, float(price)))
    # Cap buffer to prevent memory leak on stuck trades
    if len(buf) > 300:  # ~5min at 1s resolution
        _ohlc_buffer[trade_id] = buf[-300:]


def _decision_prices_burst(trade_id: int, bar_sec: int = 60) -> list[float] | None:
    """Returns OHLC-ordered synthetic tick sequence for the MOST RECENTLY
    CLOSED bar, or None if no bar has closed since last call.

    Port of sim_engines.candles_to_synthetic_ticks():
      - Compute O (first tick in bar), H (max), L (min), C (last tick)
      - Bullish (close >= open): [O, L, H, C]
      - Bearish (close < open):  [O, H, L, C]

    The caller iterates this list, calling _evaluate_trade_exit once per
    synthetic price. Mimics the exchange-aggregated intra-bar wick behavior
    of DexScreener 15-min OHLCV as closely as our tick polling allows
    (2-4 ticks per 60s — still misses exchange-level micro-wicks).

    Returns None when: buffer empty, no bar boundary crossed, or already
    emitted for the most recent bar.
    """
    buf = _ohlc_buffer.get(trade_id)
    if not buf:
        return None
    now_ts = _time_mod.time()
    cur_bar_start = int(now_ts // bar_sec) * bar_sec
    prev_bar_start = cur_bar_start - bar_sec
    last_emitted_bar = _ohlc_last_bar.get(trade_id, 0)
    if prev_bar_start <= last_emitted_bar:
        return None  # already emitted for this bar
    # Find ticks in the previous bar window
    bar_ticks = [p for t, p in buf if prev_bar_start <= t < cur_bar_start]
    if len(bar_ticks) < 2:
        return None  # need at least 2 ticks to form a meaningful bar
    o = bar_ticks[0]
    c = bar_ticks[-1]
    h = max(bar_ticks)
    lo = min(bar_ticks)
    _ohlc_last_bar[trade_id] = prev_bar_start
    # Bullish order: O → L → H → C. Bearish: O → H → L → C.
    if c >= o:
        return [o, lo, h, c]
    return [o, h, lo, c]


def _clear_ohlc_state(trade_id: int) -> None:
    """Cleanup buffers on trade close to prevent memory leaks."""
    _ohlc_buffer.pop(trade_id, None)
    _ohlc_last_bar.pop(trade_id, None)


def _get_sol_price() -> float:
    """Get SOL/USD price, cached for 60s. For paper trade sol_price_at_entry/exit."""
    global _cached_sol_price, _cached_sol_price_ts
    now = _time_mod.time()
    if _cached_sol_price > 0 and now - _cached_sol_price_ts < 60:
        return _cached_sol_price
    try:
        resp = requests.get(
            "https://api.dexscreener.com/tokens/v1/solana/So11111111111111111111111111111111111111112",
            timeout=10,
        )
        if resp.status_code == 200:
            pairs = resp.json()
            if isinstance(pairs, list) and pairs:
                price = float(pairs[0].get("priceUsd", 0))
                if price > 0:
                    _cached_sol_price = price
                    _cached_sol_price_ts = now
                    return price
    except Exception:
        pass
    return _cached_sol_price or 80.0  # fallback


def _log_price_ticks(client, prices: dict[str, float], source: str = "check",
                     live_tokens: set | None = None):
    """Log DexScreener spot prices to price_ticks table.
    v121: Includes volume/liquidity. Throttle: 15s for live trades, 60s for paper.
    v122: Also logs Jupiter price ticks for pump.fun bonding curve tokens (source='jupiter')."""
    if not prices or not client:
        return
    now = _time_mod.time()
    rows = []
    for addr, price in prices.items():
        if price <= 0:
            continue
        last = _last_tick_log.get(addr, 0)
        # v137: paper-only throttle 60->30s to align price_ticks resolution with
        # the actual unified_check_loop cadence (30s). Closes the cadence gap that
        # made the v135 sim over-estimate DTRAIL by 5-10pp.
        throttle = 15 if (live_tokens and addr in live_tokens) else 30
        if now - last < throttle:
            continue
        _last_tick_log[addr] = now
        # v123: Log DexScreener price (not Jupiter-overridden) as primary tick.
        # Sim needs both DexScreener and Jupiter price series separately.
        dex_price = _dex_prices_cache.get(addr, price)
        row = {"token_address": addr, "price_usd": dex_price, "source": source}
        # v121: Enrich with volume/liquidity from DexScreener (same API call)
        extra = _last_dex_extra.get(addr)
        if extra:
            cur_liq = extra.get("liquidity_usd")
            row["volume_usd"] = extra.get("volume_usd")
            row["liquidity_usd"] = cur_liq
            # v121: Detect liquidity changes (rug pull = sudden drop)
            prev_liq = _last_tick_liq.get(addr)
            if prev_liq and prev_liq > 0 and cur_liq is not None:
                liq_change = round((cur_liq / prev_liq - 1) * 100, 2)
                row["liq_change_pct"] = liq_change
            if cur_liq and cur_liq > 0:
                _last_tick_liq[addr] = cur_liq
        rows.append(row)

        # v122: Log Jupiter price as separate tick for ALL tokens with Jupiter pricing
        jup_price = _jupiter_prices_cache.get(addr)
        if jup_price and jup_price > 0:
            rows.append({
                "token_address": addr,
                "price_usd": jup_price,
                "source": "jupiter",
            })

    if not rows:
        return
    try:
        client.table("price_ticks").insert(rows).execute()
    except Exception as e:
        logger.debug("price_ticks insert failed (%d rows): %s", len(rows), e)


def _fetch_prices_batch(addresses: list[str]) -> dict[str, float]:
    """Batch fetch current USD prices. DexScreener primary, Jupiter fallback.
    v107: Jupiter fallback catches pool migrations that DexScreener misses.
    v143.6: short DS cache TTL (5s) mirroring the 14s Jupiter cooldown, so
    paper_fast and live_trader — called back-to-back in the same loop tick —
    read the same DS snapshot instead of two fresh fetches seconds apart.
    Prevents micro-divergence on ds / hybrid / twin_confirm / confirm strategies.
    """
    if not addresses:
        return {}
    # v143.6 — DS cache TTL: if the last fetch covered these addresses and is
    # fresh enough, reuse _dex_prices_cache + _jupiter_prices_cache instead of
    # re-fetching. Ensures paper + live share the same snapshot in one loop tick.
    now_ts = _time_mod.time()
    _last_ds_ts = getattr(_fetch_prices_batch, "_last_ds_ts", 0)
    _last_ds_addrs = getattr(_fetch_prices_batch, "_last_ds_addrs", set())
    _ds_ttl_sec = 5
    if (now_ts - _last_ds_ts) < _ds_ttl_sec and _last_ds_addrs.issuperset(addresses):
        # Cache hit: synthesize result from module caches, applying Jupiter
        # override as the normal path would.
        cached = {}
        for addr in addresses:
            jp = _jupiter_prices_cache.get(addr)
            ds = _dex_prices_cache.get(addr)
            if jp and jp > 0:
                cached[addr] = jp
            elif ds and ds > 0:
                cached[addr] = ds
        return cached

    prices = {}
    for i in range(0, len(addresses), BATCH_SIZE):
        chunk = addresses[i:i + BATCH_SIZE]
        addr_str = ",".join(chunk)
        try:
            resp = requests.get(
                DEXSCREENER_BATCH_URL.format(addresses=addr_str),
                timeout=15,
            )
            if resp.status_code != 200:
                logger.warning("paper_trader: DexScreener batch %d", resp.status_code)
                continue
            data = resp.json()
            pairs = data if isinstance(data, list) else data.get("pairs", [])
            if not isinstance(pairs, list):
                continue
            # Pick highest-volume pair per base token address
            by_addr: dict[str, list] = {}
            for p in pairs:
                addr = p.get("baseToken", {}).get("address", "")
                if addr:
                    by_addr.setdefault(addr, []).append(p)
            for addr, token_pairs in by_addr.items():
                best = max(token_pairs, key=lambda p: float(p.get("volume", {}).get("h24", 0) or 0))
                price = best.get("priceUsd")
                if price:
                    try:
                        prices[addr] = float(price)
                        # v121: Cache volume/liquidity for price tick enrichment
                        _last_dex_extra[addr] = {
                            "volume_usd": float(best.get("volume", {}).get("h24", 0) or 0),
                            "liquidity_usd": float(best.get("liquidity", {}).get("usd", 0) or 0),
                        }
                    except (ValueError, TypeError):
                        pass
        except requests.RequestException as e:
            logger.warning("paper_trader: DexScreener batch error: %s", e)

    # v107: GeckoTerminal fallback for tokens DexScreener missed (pool migration, deindexing)
    missing = [a for a in addresses if a not in prices]
    if missing:
        gecko_recovered = 0
        for addr in missing:
            fallback_price = _fetch_price_fallback(addr)
            if fallback_price and fallback_price > 0:
                prices[addr] = fallback_price
                gecko_recovered += 1
        if gecko_recovered:
            logger.info("paper_trader: GeckoTerminal fallback recovered %d/%d missing prices", gecko_recovered, len(missing))

    # v122→v123: Jupiter Price as PRIMARY for ALL tokens (paper/live alignment).
    # Fetch Jupiter prices (1 batch call) with 14s cooldown, then override DexScreener.
    # When cooldown is active, reuse cached Jupiter prices (don't fall back to DexScreener).
    _jupiter_overridden.clear()
    now_ts = _time_mod.time()
    _jup_fetch_cooldown = getattr(_fetch_prices_batch, "_last_jup_ts", 0)
    _skip_jup = (now_ts - _jup_fetch_cooldown) < 14  # 14s cooldown — matches live tick resolution (15s)
    if addresses and not _skip_jup:
        _jupiter_prices_cache.clear()
        try:
            from enrich_jupiter import _fetch_jupiter_prices
            jup_prices = _fetch_jupiter_prices(addresses)
            _fetch_prices_batch._last_jup_ts = now_ts  # rate limit cooldown
        except Exception as e:
            logger.debug("paper_trader: Jupiter price fetch failed: %s", e)
            jup_prices = {}
        # Cache ALL Jupiter prices
        for addr, jup_price in jup_prices.items():
            if jup_price and jup_price > 0:
                _jupiter_prices_cache[addr] = jup_price

    # v123: Snapshot DexScreener prices BEFORE Jupiter override (for tick logging).
    # Sim needs both DexScreener and Jupiter price series.
    _dex_prices_cache.clear()
    _dex_prices_cache.update(prices)

    # v123: ALWAYS apply Jupiter cache as primary (even during cooldown).
    # This ensures paper_fast, live_monitor, and price_refresh all use Jupiter
    # regardless of which loop fetched it.
    jup_overrides = 0
    for addr in addresses:
        jup_price = _jupiter_prices_cache.get(addr)
        if jup_price and jup_price > 0:
            prices[addr] = jup_price
            _jupiter_overridden.add(addr)
            jup_overrides += 1
    if jup_overrides:
        _src = "fresh" if not _skip_jup else "cached"
        logger.info(
            "paper_trader: Jupiter price primary for %d/%d tokens (%s)",
            jup_overrides, len(addresses), _src,
        )

    # v143.6 — stamp DS TTL so the next caller within 5s reuses this snapshot
    _fetch_prices_batch._last_ds_ts = now_ts
    _fetch_prices_batch._last_ds_addrs = set(addresses)

    return prices


def _load_paper_trade_config(client) -> dict:
    """
    Load paper_trade_config from scoring_config table.
    Returns config dict with keys: top_n, budget_usd, active_strategies,
    dedup_cooldown_hours, ca_filter. Falls back to module defaults on error.
    """
    defaults = {
        "top_n": TOP_N,
        "budget_usd": PORTFOLIO_BUDGET,
        "active_strategies": list(STRATEGIES.keys()),
        "dedup_cooldown_hours": DEDUP_COOLDOWN_HOURS,
        "ca_filter": CA_FILTER,
        "buy_slippage_bps": BUY_SLIPPAGE_BPS,
        "sell_slippage_bps": SELL_SLIPPAGE_BPS,
        "buy_fee_bps": BUY_FEE_BPS,
        "sell_fee_bps": SELL_FEE_BPS,
        "ml_gate_mode": "disabled",  # v90: "disabled" | "normal" | "inverted"
        "experiment_id": None,       # v92: A/B testing
        "variant_id": None,          # v92: A/B testing
        "deprecated_strategies": list(_DEFAULT_DEPRECATED),  # v92: dynamic deprecated
    }
    try:
        result = client.table("scoring_config").select("paper_trade_config").eq("id", 1).execute()
        if result.data and result.data[0].get("paper_trade_config"):
            raw = result.data[0]["paper_trade_config"]
            if isinstance(raw, str):
                import json
                raw = json.loads(raw)
            # Merge with defaults (unknown keys ignored, missing keys use default)
            config = {k: raw.get(k, v) for k, v in defaults.items()}
            # Type safety: JSONB stores numbers as float, but top_n must be int
            config["top_n"] = int(config["top_n"])
            config["budget_usd"] = float(config["budget_usd"])
            config["dedup_cooldown_hours"] = int(config.get("dedup_cooldown_hours", DEDUP_COOLDOWN_HOURS))
            # Validate active_strategies against known strategies
            config["active_strategies"] = [
                s for s in config["active_strategies"] if s in STRATEGIES
            ]
            if not config["active_strategies"]:
                config["active_strategies"] = defaults["active_strategies"]
            # v92: pass through A/B testing keys from raw config
            for extra_key in ("experiment_id", "variant_id"):
                if raw.get(extra_key):
                    config[extra_key] = raw[extra_key]
            logger.info("paper_trader: loaded config from DB: top_n=%d, budget=$%.0f, strategies=%s, dedup=%dh, ca_filter=%s",
                        config["top_n"], config["budget_usd"], config["active_strategies"],
                        config["dedup_cooldown_hours"], config["ca_filter"])
            return config
    except Exception as e:
        logger.warning("paper_trader: failed to load config from scoring_config: %s", e)
    return defaults


def get_deprecated_strategies(client=None, config=None) -> set:
    """v92: Load deprecated_strategies from paper_trade_config. Fallback to hardcoded."""
    if config and config.get("deprecated_strategies"):
        return set(config["deprecated_strategies"])
    if client:
        try:
            r = client.table("scoring_config").select("paper_trade_config").eq("id", 1).execute()
            if r.data and r.data[0].get("paper_trade_config"):
                ptc = r.data[0]["paper_trade_config"]
                if isinstance(ptc, str):
                    import json
                    ptc = json.loads(ptc)
                return set(ptc.get("deprecated_strategies", _DEFAULT_DEPRECATED))
        except Exception:
            pass
    return set(_DEFAULT_DEPRECATED)


def get_open_portfolio(client) -> dict:
    """Return snapshot of open main (non-shadow) positions for alert context.
    Returns {open_count, deployed_usd}. Lightweight: single aggregate query."""
    try:
        result = (
            client.table("paper_trades")
            .select("position_usd")
            .eq("status", "open")
            .eq("is_shadow", False)
            .execute()
        )
        rows = result.data or []
        total = sum(float(r.get("position_usd") or 0) for r in rows)
        return {"open_count": len(rows), "deployed_usd": round(total, 2)}
    except Exception as e:
        logger.warning("get_open_portfolio failed: %s", e)
        return {"open_count": 0, "deployed_usd": 0}


def open_paper_trades(client, ranking: list[dict], cycle_ts: datetime, config: dict | None = None) -> int:
    """
    Open paper trades for top N tokens across configured strategies.
    Each strategy may have multiple tranches (e.g. SCALE_OUT has 4 rows per token).
    Dedup: skip if token_address + strategy already has an open trade.
    Cooldown dedup: skip if same (token, strategy) closed within dedup_cooldown_hours.
    Returns number of new trade rows opened.
    """
    # v88: Load precomputed bot ML predictions (one query per cycle)
    # v109: Skip loading — ML gate is disabled (ml_gate_mode=disabled, ml_threshold=99).
    # Saves one Supabase query per cycle. Re-enable when ML is useful.
    # _load_bot_predictions(client)

    if config is None:
        config = {
            "top_n": TOP_N,
            "budget_usd": PORTFOLIO_BUDGET,
            "active_strategies": list(STRATEGIES.keys()),
            "dedup_cooldown_hours": DEDUP_COOLDOWN_HOURS,
            "ca_filter": CA_FILTER,
        }

    top_n = config["top_n"]
    budget_usd = config["budget_usd"]
    deprecated = set(config.get("deprecated_strategies", _DEFAULT_DEPRECATED))
    active_strategies = [s for s in config["active_strategies"] if s in STRATEGIES and s not in deprecated]
    dedup_cooldown_h = config.get("dedup_cooldown_hours", DEDUP_COOLDOWN_HOURS)
    ca_filter = config.get("ca_filter", True)
    buy_slip_bps_base = int(config.get("buy_slippage_bps", BUY_SLIPPAGE_BPS))
    sell_slip_bps = int(config.get("sell_slippage_bps", SELL_SLIPPAGE_BPS))
    buy_fee_bps = int(config.get("buy_fee_bps", BUY_FEE_BPS))

    # v130: Quote Ultra at live's actual position size so paper's entry_price
    # reflects the exact route/fill live will execute against. One quote per
    # token per cycle (caller loop iterates tokens once), shared across all
    # tranches/strategies of the same token.
    _live_cfg = config.get("live_trading", {}) if isinstance(config.get("live_trading"), dict) else {}
    _ultra_quote_sol = float(_live_cfg.get("max_position_sol", 0.15))
    ultra_quote_lamports = int(_ultra_quote_sol * 1_000_000_000)

    # Filter candidates
    base_filter = [
        t for t in ranking
        if t.get("score", 0) > 0
        and t.get("token_address")
        and t.get("price_usd") and float(t["price_usd"]) > 0
    ]
    if ca_filter:
        base_filter = [
            t for t in base_filter
            if (t.get("ca_mention_count", 0) or 0) > 0 or (t.get("url_mention_count", 0) or 0) > 0
        ]
    candidates = base_filter[:top_n]

    if not candidates:
        return 0

    # Score-weighted portfolio allocation
    scores = [max(t.get("score", 1), 1) for t in candidates]
    total_score = sum(scores)
    for i, token in enumerate(candidates):
        token["_alloc_usd"] = round(budget_usd * scores[i] / total_score, 2)

    # Check which (token_address, strategy) combos already have open trades
    addrs = [t["token_address"] for t in candidates]
    try:
        existing = (
            client.table("paper_trades")
            .select("token_address, strategy")
            .eq("status", "open")
            .in_("token_address", addrs)
            .execute()
        )
        open_combos = {
            (r["token_address"], r["strategy"]) for r in (existing.data or [])
        }
    except Exception as e:
        logger.error("paper_trader: failed to check open trades: %s", e)
        open_combos = set()

    # Cooldown dedup: check recently closed trades (main + shadow)
    # v105: Apply cooldown to ALL trades (not just main). Shadow re-entries on the same
    # token pollute data — a KOL re-calling a dead token generates 47 losing shadow trades.
    cooldown_combos = set()
    if dedup_cooldown_h > 0:
        cooldown_since = (cycle_ts - timedelta(hours=dedup_cooldown_h)).isoformat()
        try:
            recent = (
                client.table("paper_trades")
                .select("token_address, strategy")
                .neq("status", "open")
                .gte("exit_at", cooldown_since)
                .in_("token_address", addrs)
                .execute()
            )
            cooldown_combos = {
                (r["token_address"], r["strategy"]) for r in (recent.data or [])
            }
        except Exception as e:
            logger.warning("paper_trader: cooldown dedup query failed: %s", e)

    # Lookup snapshot_id for candidates missing it (batch query)
    need_snap = [t for t in candidates if not t.get("snapshot_id") and t.get("token_address")]
    if need_snap:
        snap_addrs = [t["token_address"] for t in need_snap]
        try:
            snap_res = (
                client.table("token_snapshots")
                .select("id, token_address, snapshot_at")
                .in_("token_address", snap_addrs)
                .gte("snapshot_at", (cycle_ts - timedelta(minutes=10)).isoformat())
                .lte("snapshot_at", (cycle_ts + timedelta(minutes=10)).isoformat())
                .order("snapshot_at", desc=True)
                .execute()
            )
            # Keep closest snapshot per token_address
            snap_map = {}
            for s in (snap_res.data or []):
                addr = s["token_address"]
                if addr not in snap_map:
                    snap_map[addr] = s["id"]
            for t in need_snap:
                sid = snap_map.get(t["token_address"])
                if sid:
                    t["snapshot_id"] = sid
        except Exception as e:
            logger.debug("paper_trader: snapshot_id lookup failed: %s", e)

    opened = 0
    for rank_idx, token in enumerate(candidates, 1):
        addr = token["token_address"]
        raw_price = float(token["price_usd"])
        # v74: Dynamic slippage — scale with liquidity depth score
        # liquidity_depth_score: 1.0 = deep liquidity, 0.1 = shallow
        lds = float(token.get("liquidity_depth_score") or token.get("_rt_liquidity_depth_score") or 0)
        # v94: RT fallback — derive LDS proxy from rt_liquidity_usd when Jupiter LDS unavailable
        if not lds and token.get("_rt_liquidity_usd"):
            rt_liq = float(token["_rt_liquidity_usd"])
            # $50K+ = 1.0, $10K = 0.5, $5K = 0.25, $1K = 0.05
            lds = min(1.0, rt_liq / 50_000) if rt_liq > 0 else 0.1
        lds = max(0.1, min(1.0, lds)) if lds else 1.0
        # Shallow liquidity → up to 3x base slippage; deep → 1x
        slip_mult = 1.0 + 2.0 * (1.0 - lds)  # 1.0 for lds=1.0, 3.0 for lds=0.0
        buy_slip_bps = int(buy_slip_bps_base * slip_mult)
        # v142 E: shadow-sync — if live already opened this trade and passed its
        # actual Jupiter fill price via token["_rt_force_entry_price"], use it
        # directly. Fixes P1 (TP/SL status inversion) and P4 (entry_price ±9%
        # divergence) caused by paper + live doing independent DS/Ultra quote
        # fetches 4-27s apart. When not present (no live, live failed, or
        # non-hybrid flow), paper falls back to its own Ultra quote.
        forced_entry = token.get("_rt_force_entry_price")
        # v130: Quote Jupiter Ultra /order at live's position size — same route/fill
        # as the live swap. Single source of truth for entry_price, market_ref, and
        # high_price_seen (no DexScreener/Ultra mixing like v127 had).
        sol_price_entry = _get_sol_price()
        ultra_price = None
        if forced_entry and float(forced_entry) > 0:
            ultra_price = float(forced_entry)
            entry_source = "live_sync"
        elif ultra_quote_lamports > 0 and sol_price_entry > 0:
            try:
                from enrich_jupiter import fetch_ultra_quote_price
                ultra_price = fetch_ultra_quote_price(addr, ultra_quote_lamports, sol_price_entry)
            except Exception as e:
                logger.debug("paper_trader: ultra quote failed for %s: %s", addr[:8], e)
        if ultra_price and ultra_price > 0:
            entry_price = ultra_price
            if not forced_entry:
                entry_source = "ultra"
            _jupiter_prices_cache[addr] = ultra_price  # warm for tracking symmetry
        else:
            # Fallback when Ultra API unavailable (fresh token, Helius rate limit,
            # Jupiter 5xx). Tagged so --from-trades can filter on entry_source='ultra'.
            logger.info(
                "paper_trader: Ultra quote unavailable for %s (%s) — DexScreener fallback",
                token.get("symbol", "???"), addr[:8],
            )
            entry_price = raw_price
            entry_source = "dexscreener"
        alloc_usd = token.get("_alloc_usd", budget_usd / top_n)

        # Common fields for all tranches of this token
        base_row = {
            "cycle_ts": cycle_ts.isoformat(),
            "symbol": token.get("symbol", "???"),
            "token_address": addr,
            "rank_in_cycle": rank_idx,
            "entry_price": entry_price,
            "entry_score": int(token.get("score", 0)),
            "entry_mcap": float(token["market_cap"]) if token.get("market_cap") else None,
            "status": "open",
            "unique_kols": token.get("unique_kols"),
            "whale_new_entries": token.get("whale_new_entries"),
            "momentum_mult": float(token["momentum_mult"]) if token.get("momentum_mult") else None,
            "portfolio_budget": budget_usd,
            "sol_price_at_entry": sol_price_entry,  # v121
            # v130: ALL price refs anchored to entry_price (Ultra quote or raw fallback).
            # Single source = no mixing bugs in DTRAIL/BE gates. entry_source tags
            # which source was used so analysis can filter.
            "dex_spot_price_at_entry": entry_price,
            "high_price_seen": entry_price,
            "entry_source": entry_source,
        }
        # v96: Batch KOL attribution — propagate top_kol as kol_group + source="batch"
        if not token.get("_rt_source"):
            top_kols = token.get("top_kols") or []
            if isinstance(top_kols, list) and top_kols:
                base_row["kol_group"] = top_kols[0]
            base_row["source"] = "batch"
        # v92: A/B testing — thread experiment/variant into trade rows
        if config.get("experiment_id"):
            base_row["experiment_id"] = config["experiment_id"]
        if config.get("variant_id"):
            base_row["variant_id"] = config["variant_id"]
        if token.get("snapshot_id"):
            base_row["snapshot_id"] = int(token["snapshot_id"])

        # v66: RT metadata propagation (keys prefixed _rt_ in token dict → DB columns)
        _rt_col_map = {
            "_rt_source": "source",
            "_rt_kol_group": "kol_group",
            "_rt_kol_tier": "kol_tier",
            "_rt_kol_score": "kol_score",
            "_rt_kol_win_rate": "kol_win_rate",
            "_rt_score": "rt_score",
            "_rt_liquidity_usd": "rt_liquidity_usd",
            "_rt_volume_24h": "rt_volume_24h",
            "_rt_buy_sell_ratio": "rt_buy_sell_ratio",
            "_rt_token_age_hours": "rt_token_age_hours",
            "_rt_is_pump_fun": "rt_is_pump_fun",
            "_rt_pair_address": "pair_address",  # v109: track pool for migration debug
            "_rt_ml_pred": "ml_pred",        # v77: ML predicted avg PnL — enables A/B analysis
            "_rt_kol_ml_pred": "kol_ml_pred", # v78: KOL ML predicted return — enables KCO A/B
            "_rt_n_kol_confirmations": "n_kol_confirmations",  # v80: multi-KOL confirmation count
            "_rt_experiment_id": "experiment_id",    # v92: A/B testing
            "_rt_variant_id": "variant_id",          # v92: A/B testing
            "_rt_message_ts": "message_ts",          # v121: Telegram message timestamp
            "_rt_price_at_message": "price_at_message",  # v121: DexScreener price at call time
        }
        for src_key, db_col in _rt_col_map.items():
            val = token.get(src_key)
            if val is not None:
                base_row[db_col] = val

        for strat_name in active_strategies:
            if not _passes_strategy_filter(token, strat_name):
                continue  # token doesn't qualify for this strategy
            tranches = STRATEGIES[strat_name]

            if (addr, strat_name) in open_combos:
                continue
            if (addr, strat_name) in cooldown_combos:
                continue

            # v108: Defensive re-check right before insert to catch race conditions
            # (e.g. batch + RT overlap, or two RT calls that slipped past in-flight lock)
            try:
                recheck = (
                    client.table("paper_trades")
                    .select("id", count="exact")
                    .eq("token_address", addr)
                    .eq("strategy", strat_name)
                    .eq("status", "open")
                    .execute()
                )
                if recheck.count and recheck.count > 0:
                    open_combos.add((addr, strat_name))
                    continue
            except Exception:
                pass  # proceed with insert on query failure

            # v87: Bot ML gate — position sizing or skip (v89: disabled by default)
            bot_ml_mult = _bot_ml_gate(token, strat_name, config)
            if bot_ml_mult <= 0.0:
                logger.info("bot_ml_gate: SKIP %s/%s (win_prob < 0.30)", token.get("symbol"), strat_name)
                continue
            if bot_ml_mult < 1.0:
                logger.info("bot_ml_gate: HALF %s/%s (mult=%.1f)", token.get("symbol"), strat_name, bot_ml_mult)

            for tranche in tranches:
                tp_price = entry_price * tranche["tp_mult"] if tranche["tp_mult"] else None
                sl_price = entry_price * tranche["sl_mult"]

                row = {
                    **base_row,
                    "strategy": strat_name,
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                    "horizon_minutes": tranche["horizon_min"],
                    "tranche_pct": tranche["pct"],
                    "tranche_label": tranche["label"],
                    "position_usd": round(alloc_usd * tranche["pct"] * bot_ml_mult, 2),
                }

                try:
                    client.table("paper_trades").insert(row).execute()
                    opened += 1
                    # v115: DIP_BUY — add to watchlist for P2 monitoring
                    if tranche["label"] == "dip_p1":
                        # v116: Try split regex first, then shared
                        ms = _DIP_SPLIT_RE.match(strat_name)
                        m = _DIP_RE.match(strat_name) if not ms else None
                        if ms:
                            _dip_watchlist[(addr, strat_name)] = {
                                "entry_price": entry_price,
                                "low_seen": entry_price,
                                "dip_pct": int(ms.group(1)) / 100,
                                "bounce_pct": int(ms.group(2)) / 100,
                                "sl_pct": int(ms.group(8)) / 100,  # P2 SL
                                "horizon_min": int(ms.group(9)),
                                "alloc_usd": round(alloc_usd * tranche["pct"] * bot_ml_mult, 2),
                                "base_row": {k: v for k, v in base_row.items()},
                                "created_at": datetime.now(timezone.utc),
                            }
                            logger.info("dip_watch: watching %s/%s for -%.0f%% dip (split P2: T%sA%sS%s)",
                                        token.get("symbol"), strat_name, int(ms.group(1)),
                                        ms.group(6), ms.group(7), ms.group(8))
                        elif m:
                            _dip_watchlist[(addr, strat_name)] = {
                                "entry_price": entry_price,
                                "low_seen": entry_price,
                                "dip_pct": int(m.group(1)) / 100,
                                "bounce_pct": int(m.group(2)) / 100,
                                "sl_pct": int(m.group(5)) / 100,
                                "horizon_min": int(m.group(6)),
                                "alloc_usd": round(alloc_usd * tranche["pct"] * bot_ml_mult, 2),
                                "base_row": {k: v for k, v in base_row.items()},
                                "created_at": datetime.now(timezone.utc),
                            }
                            logger.info("dip_watch: watching %s/%s for -%.0f%% dip",
                                        token.get("symbol"), strat_name, int(m.group(1)))
                except Exception as e:
                    logger.error(
                        "paper_trader: insert failed for %s/%s/%s: %s",
                        token.get("symbol"), strat_name, tranche["label"], e,
                    )
            # v108: Mark as opened so subsequent iterations in this call don't re-insert
            open_combos.add((addr, strat_name))

    # ── Shadow trades: open $0 trades for ALL other single-tranche strategies ──
    # This lets us compare strategies on the SAME tokens (apples-to-apples).
    shadow_enabled = config.get("shadow_enabled", True)
    shadow_opened = 0
    if shadow_enabled:
        for rank_idx, token in enumerate(candidates, 1):
            addr = token["token_address"]
            raw_price = float(token["price_usd"])
            lds = float(token.get("liquidity_depth_score") or token.get("_rt_liquidity_depth_score") or 0)
            if not lds and token.get("_rt_liquidity_usd"):
                rt_liq = float(token["_rt_liquidity_usd"])
                lds = min(1.0, rt_liq / 50_000) if rt_liq > 0 else 0.1
            lds = max(0.1, min(1.0, lds)) if lds else 1.0
            slip_mult = 1.0 + 2.0 * (1.0 - lds)
            buy_slip_bps = int(buy_slip_bps_base * slip_mult)
            # v123: Jupiter price as entry (same as main trades)
            jup_entry = _jupiter_prices_cache.get(addr)
            if jup_entry and jup_entry > 0:
                entry_price = jup_entry
            else:
                entry_price = raw_price  # v123: no slippage markup (match live)

            shadow_base = {
                "cycle_ts": cycle_ts.isoformat(),
                "symbol": token.get("symbol", "???"),
                "token_address": addr,
                "rank_in_cycle": rank_idx,
                "entry_price": entry_price,
                "entry_score": int(token.get("score", 0)),
                "entry_mcap": float(token["market_cap"]) if token.get("market_cap") else None,
                "status": "open",
                "is_shadow": True,
                "position_usd": 0,
                "portfolio_budget": budget_usd,
            }
            # v92: A/B testing — thread experiment/variant into shadow rows
            if config.get("experiment_id"):
                shadow_base["experiment_id"] = config["experiment_id"]
            if config.get("variant_id"):
                shadow_base["variant_id"] = config["variant_id"]
            if token.get("snapshot_id"):
                shadow_base["snapshot_id"] = int(token["snapshot_id"])
            # Propagate RT metadata to shadows too (for KOL attribution)
            for src_key, db_col in _rt_col_map.items():
                val = token.get(src_key)
                if val is not None:
                    shadow_base[db_col] = val

            # all_real_strategies: full list of strategies opened as real (across all calls)
            # In RT hybrid mode, this includes all hybrid allocations (not just this call's active_strategies)
            real_strats = set(config.get("all_real_strategies", active_strategies))

            # v138.4: BATCH shadow inserts. Was: 217 sequential inserts × ~50ms HTTP RT
            # = 10-20s of ds→pre_buy latency that delayed live trade execution. Now:
            # 1 HTTP roundtrip per call. Cuts measured ds→pre_buy from ~15s to ~3s.
            shadow_rows: list[dict] = []
            for strat_name in SHADOW_STRATEGIES:
                if strat_name in real_strats:
                    continue  # opened as real trade (this call or sibling call)
                if not _passes_strategy_filter(token, strat_name):
                    continue
                if (addr, strat_name) in open_combos:
                    continue
                if (addr, strat_name) in cooldown_combos:
                    continue

                tranches = STRATEGIES[strat_name]
                for tranche in tranches:
                    tp_price = entry_price * tranche["tp_mult"] if tranche.get("tp_mult") else None
                    sl_price = entry_price * tranche["sl_mult"]
                    shadow_rows.append({
                        **shadow_base,
                        "strategy": strat_name,
                        "tp_price": tp_price,
                        "sl_price": sl_price,
                        "horizon_minutes": tranche["horizon_min"],
                        "tranche_pct": tranche["pct"],
                        "tranche_label": tranche["label"],
                    })
                # v108: Mark as opened so subsequent tokens in this call don't re-insert
                open_combos.add((addr, strat_name))

            # Single batch insert (Supabase supports list payload). Falls back to
            # per-row inserts if the batch fails so partial success is possible.
            if shadow_rows:
                try:
                    client.table("paper_trades").insert(shadow_rows).execute()
                    shadow_opened = len(shadow_rows)
                except Exception as e:
                    logger.warning("paper_trader: batch shadow insert failed (%d rows): %s — falling back to per-row",
                                   len(shadow_rows), e)
                    for r in shadow_rows:
                        try:
                            client.table("paper_trades").insert(r).execute()
                            shadow_opened += 1
                        except Exception as e2:
                            logger.error("paper_trader: shadow row insert failed %s/%s: %s",
                                         token.get("symbol"), r.get("strategy"), e2)

    allocs = [f"{t.get('symbol','?')}=${t.get('_alloc_usd',0):.1f}" for t in candidates]
    logger.info(
        "paper_trader: opened %d rows + %d shadow, $%.0f budget → %s (%d strategies, dedup=%dh)",
        opened, shadow_opened, budget_usd, ", ".join(allocs), len(active_strategies), dedup_cooldown_h,
    )
    if _monitoring and opened > 0:
        _metrics.record_paper_trade_open(opened)
    return opened


def _dynamic_sell_slip_factor(trade: dict, exit_type: str, base_bps: int = 10,
                              fee_bps: int = SELL_FEE_BPS) -> float:
    """v138.5: recalibrated against 132 live trades (Apr 13-17 post-v132).

    Measured medians (paper_exit vs real_exit divergence on rt_live):
      sl_hit       N=34  median -4.35%  → ~435 bps real slip (was ~30-120 sim)
      trail_stop   N=52  median -2.48%  → ~250 bps  (was ~15-60 sim)
      trail_crash  ~outliers in trail_stop, median -10%+ → 1000+ bps
      tp_hit       N=15  median +7.74%  → POSITIVE slip (Jupiter trigger overshoots
                                          tp_price, fill HIGHER than target = bonus)
      timeout      N=31  median -1.22%  → ~100 bps (was ~30)

    Slip does NOT scale strongly with liquidity in $5K-100K range — flat baseline
    per exit type, with low-liq amplifier only below $20K.

    base_bps param kept for backward compat but ignored (slip now per exit type).
    Conservative tp_hit: median was +7.7% but N=15 only → use +300 bps (+3%) until
    more samples accumulate.
    """
    liq_usd = float(trade.get("rt_liquidity_usd") or 50_000)
    # Low-liq amplifier (kicks in below $20K)
    if liq_usd < 5_000:
        liq_mult = 2.0
    elif liq_usd < 20_000:
        liq_mult = 1.3
    else:
        liq_mult = 1.0

    # Per-exit-type baseline bps (negative = positive slippage / overshoot)
    if exit_type == "trail_crash":
        type_bps = 1000        # was 50-200; real outliers show -10% to -29%
    elif exit_type == "sl_hit":
        type_bps = 435         # was 30-120; real median -4.35%
    elif exit_type == "trail_stop":
        type_bps = 250         # was 15-60; real median -2.48%
    elif exit_type == "tp_hit":
        type_bps = -300        # POSITIVE slip — Jupiter trigger fills above target
    elif exit_type == "timeout":
        type_bps = 120         # was 30; real median -1.22%
    elif exit_type == "be_stop":
        # v142: breakeven stop — active close at ~entry price. Less dump pressure
        # than sl_hit (we exit on BE violation, usually in a normal pullback, not
        # crash). Mid-range between tp_hit and sl_hit.
        type_bps = 200
    elif exit_type == "tp_late":
        # v142: late-phase take-any-profit. Thinner book than fresh TP.
        type_bps = 80
    else:
        type_bps = 100

    adjusted_bps = int(type_bps * liq_mult) + fee_bps
    # Caps prevent runaway on edge cases
    if exit_type == "trail_crash":
        adjusted_bps = max(-1000, min(2500, adjusted_bps))
    else:
        adjusted_bps = max(-1000, min(1500, adjusted_bps))

    return 1 - adjusted_bps / 10_000


# v125: Regex parsers, _get_trail_config, _get_decay_end, LAZY mode now imported from strategies.py
_last_eval_ts: dict[str, float] = {}  # trade_id -> last evaluation timestamp


def _should_evaluate_exit(trade: dict, now: datetime) -> bool:
    """v118: Check if this trade should be evaluated for exit.
    LAZY strategies are throttled to check less frequently.
    Only applies to hybrid trades (position_usd > 0), NOT shadows.
    Shadows keep CURRENT interval = control group for A/B test."""
    strat = trade.get("strategy", "")
    if strat not in LAZY_STRATEGIES:
        return True
    # Shadows (position_usd=0) always use CURRENT interval (control group)
    if float(trade.get("position_usd") or 0) == 0:
        return True

    trade_id = str(trade.get("id", ""))
    now_ts = now.timestamp()

    # How old is the trade?
    try:
        created = datetime.fromisoformat(trade["created_at"].replace("Z", "+00:00"))
        age_sec = (now - created).total_seconds()
    except Exception:
        return True

    # Determine throttle interval
    if age_sec < LAZY_FAST_WINDOW:
        interval = LAZY_FAST_SEC
    else:
        interval = LAZY_SLOW_SEC

    last = _last_eval_ts.get(trade_id, 0)
    if now_ts - last < interval:
        return False  # skip this evaluation

    _last_eval_ts[trade_id] = now_ts
    return True


def _override_exit_with_ultra_quote(client, trade: dict, ev: dict) -> dict:
    """
    v131 Gap #1: Replace simulated exit_price with real Jupiter Ultra SELL quote.

    Called after _evaluate_trade_exit returns a closing status. Quotes Ultra
    at the actual token amount the paper trade would be selling. Recomputes
    pnl_pct + pnl_usd. Falls back to the formula-based exit_price if the
    Ultra quote fails (fresh token, Jupiter 5xx, decimals unavailable).

    Returns the (possibly modified) ev dict.
    """
    if not ev or ev.get("status") is None:
        return ev
    # Shadow trades (pos_usd=0) skip Ultra quote — they're analysis-only
    pos_usd = float(trade.get("position_usd") or 0)
    entry_price = float(trade.get("entry_price") or 0)
    if pos_usd <= 0 or entry_price <= 0:
        return ev
    # Only apply to Ultra-entered trades — DexScreener fallback entries use formula
    if trade.get("entry_source") != "ultra":
        return ev

    addr = trade.get("token_address")
    if not addr:
        return ev

    try:
        from enrich_jupiter import fetch_ultra_sell_quote_price, get_token_decimals
        decimals = get_token_decimals(addr)
        if decimals is None:
            return ev
        token_amount = (pos_usd / entry_price) * (10 ** decimals)
        sol_price = _get_sol_price()
        ultra_exit = fetch_ultra_sell_quote_price(addr, int(token_amount), sol_price)
        if ultra_exit and ultra_exit > 0:
            ev["exit_price"] = ultra_exit
            ev["pnl_pct"] = round((ultra_exit / entry_price) - 1, 4)
            ev["pnl_usd"] = round(pos_usd * ev["pnl_pct"], 2)
    except Exception as e:
        logger.debug("paper ultra exit quote failed for %s: %s", addr[:8], e)

    return ev


def _evaluate_trade_exit(trade: dict, current_price: float | None,
                         now: datetime, sell_slip_factor: float,
                         sl_cascade: bool = False,
                         sell_fee_bps: int = SELL_FEE_BPS,
                         decision_price: float | None = None) -> dict | None:
    """v94: Shared exit logic for check_paper_trades + check_paper_trades_fast.

    Checks in order: SL → TP → timeout.
    Updates high_price_seen on every call (even when no exit).
    sell_slip_factor is used as base_bps source; dynamic slippage + fee applied per exit type.

    v118: LAZY mode — returns high_price_seen update only (no exit eval)
    when _should_evaluate_exit() says to skip.

    v132: Orchestration support — decision_price (optional) overrides current_price for
    TP/SL/trail trigger comparisons. current_price is still used for exit_price booking.
    Enables price_source={ds,hybrid,confirm,ema3} without changing execution semantics.

    Returns dict with keys {status, exit_price, pnl_pct, pnl_usd, exit_minutes,
    high_price_seen} or None if no action. Caller handles DB update.
    """
    # v132: If decision_price provided, use it for TP/SL/trail evaluation.
    # current_price still used for exit_price (matches live Jupiter execution).
    eval_price = decision_price if decision_price is not None else current_price
    entry_price = float(trade["entry_price"])
    sl_price = float(trade["sl_price"])
    tp_price = float(trade["tp_price"]) if trade.get("tp_price") is not None else None
    pos_usd = float(trade.get("position_usd") or 0)

    # v124: For live trades, trail/BE activation should reference the market price at entry
    # (dex_spot_price_at_entry), not the fill price (entry_price). The fill is always lower
    # due to on-chain slippage (5-15% on memecoins), which would make activation too easy.
    # Paper trades don't have this field, so they fall back to entry_price (which ≈ market).
    market_ref_price = float(trade.get("dex_spot_price_at_entry") or 0) or entry_price

    # Derive base_bps from the flat sell_slip_factor passed by caller
    base_bps = int(round((1 - sell_slip_factor) * 10_000))

    created_str = trade["created_at"]
    try:
        created_at = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
    except Exception:
        return None

    elapsed_minutes = (now - created_at).total_seconds() / 60
    horizon = trade.get("horizon_minutes", 120)

    # Track high_price_seen (always, even in LAZY skip mode)
    # v132: use eval_price (decision) for peak tracking so trail activation uses
    # the same smoothed/DS source as trigger detection.
    high_seen = float(trade.get("high_price_seen") or 0)
    if eval_price is not None and eval_price > high_seen:
        high_seen = eval_price

    # v118: LAZY mode — update high_price_seen but skip exit evaluation
    if not _should_evaluate_exit(trade, now):
        return {"high_price_seen": high_seen}  # caller updates DB but no exit

    new_status = None
    exit_price = None

    # 1) SL cascade from sibling tranche
    if sl_cascade:
        new_status = "sl_hit"
        exit_price = sl_price * _dynamic_sell_slip_factor(trade, "sl_hit", base_bps, sell_fee_bps)

    elif eval_price is not None:
        # v142: resolve tranche config once for time_be_minute / tp_schedule /
        # trail_tiers extensions. Cheap dict lookup. None for unknown strats.
        from strategies import _find_tranche_config as _pt_find_tranche
        _v142_tranche = _pt_find_tranche(
            trade.get("strategy", ""), trade.get("tranche_label", "main") or "main"
        ) or {}

        # 2) SL check — with breakeven stop override
        #    BE strategies: once peak exceeded entry*(1+be_act), SL moves to entry price
        effective_sl = sl_price
        be_match = _BE_RE.match(trade.get("strategy", ""))
        if be_match and market_ref_price > 0 and high_seen > 0:
            be_act = int(be_match.group(1)) / 100  # e.g., BE20 → 0.20
            if high_seen >= market_ref_price * (1 + be_act):
                # Breakeven activated — SL is now entry price
                effective_sl = entry_price

        # v142: TIME-based BE — after N minutes elapsed, SL moves to entry
        # regardless of peak reached. Distinct from peak-based BE_RE logic above.
        # Config key: `time_be_minute` in tranche dict.
        _time_be = _v142_tranche.get("time_be_minute")
        if _time_be and elapsed_minutes >= float(_time_be):
            effective_sl = max(effective_sl, entry_price)

        # v133: SL check against exit_ref (Jupiter quote — what the sell would actually
        # fill at) rather than decision_price. Fixes hybrid-mode divergence where DS
        # noise dips trigger phantom SLs in paper while live Jupiter quote stays above.
        # TP/trail still use eval_price (DS catches fast pumps faster).
        sl_eval = current_price if current_price is not None else eval_price

        if sl_eval <= effective_sl:
            new_status = "sl_hit" if effective_sl <= sl_price else "be_stop"
            exit_price = effective_sl * _dynamic_sell_slip_factor(
                trade, "sl_hit" if new_status == "sl_hit" else "be_stop",
                base_bps, sell_fee_bps,
            )
        # 3) TP check (only tranches with TP target)
        elif tp_price is not None and eval_price >= tp_price:
            new_status = "tp_hit"
            exit_price = tp_price * _dynamic_sell_slip_factor(trade, "tp_hit", base_bps, sell_fee_bps)
        # 3a-v142) TP_SCHEDULE — piecewise-linear TP mult over time.
        # Config: `tp_schedule` = [(minute, tp_mult), ...] sorted by minute.
        # Overrides tp_decay_end when present. Allows TP to rise or fall over time.
        elif _v142_tranche.get("tp_schedule"):
            _sched = _v142_tranche["tp_schedule"]
            # interpolate tp_mult at current elapsed_minutes
            _tp_m = None
            for i in range(len(_sched) - 1):
                m1, v1 = _sched[i]
                m2, v2 = _sched[i + 1]
                if m1 <= elapsed_minutes <= m2:
                    _tp_m = v2 if m2 == m1 else v1 + (v2 - v1) * (elapsed_minutes - m1) / (m2 - m1)
                    break
            if _tp_m is None and _sched:
                _tp_m = _sched[-1][1] if elapsed_minutes >= _sched[-1][0] else _sched[0][1]
            if _tp_m is not None and _tp_m > 1.0:
                _sched_price = entry_price * _tp_m
                if eval_price >= _sched_price:
                    new_status = "tp_hit"
                    exit_price = _sched_price * _dynamic_sell_slip_factor(trade, "tp_hit", base_bps, sell_fee_bps)
            elif _tp_m is not None and _tp_m <= 1.0 and eval_price >= entry_price:
                # Late-phase catch-any-profit: TP mult ≤ 1 means take breakeven+
                new_status = "tp_late"
                exit_price = eval_price * _dynamic_sell_slip_factor(trade, "tp_late", base_bps, sell_fee_bps)
        # 3b) v106: Time-decay TP — threshold decreases in second half of horizon
        #     Derive tp_decay_end from strategy name (DECAY_TPxx_SLyy_Ezz → 1 + zz/100)
        elif tp_price is not None and _get_decay_end(trade.get("strategy", "")) is not None:
            tp_decay_end = _get_decay_end(trade["strategy"])
            tp_mult = tp_price / entry_price if entry_price > 0 else 1
            activation_frac = 0.5  # decay starts at 50% of horizon
            if elapsed_minutes > horizon * activation_frac:
                decay_progress = (elapsed_minutes - horizon * activation_frac) / (horizon * (1 - activation_frac))
                decay_progress = min(decay_progress, 1.0)
                # Linear interpolation: tp_mult → tp_decay_end
                decayed_mult = tp_mult - (tp_mult - tp_decay_end) * decay_progress
                decayed_price = entry_price * decayed_mult
                if eval_price >= decayed_price:
                    new_status = "tp_hit"
                    # Exit at the decayed threshold, not the peak
                    exit_price = decayed_price * _dynamic_sell_slip_factor(trade, "tp_hit", base_bps, sell_fee_bps)
    # 3c) v106/v110: Trailing stop — exit when price drops trail_pct% from peak.
    #     v106 TRAIL: activates once peak > entry * (1 + trail_pct).
    #     v110 DTRAIL: activates once peak > entry * (1 + activation_pct).
    if new_status is None and eval_price is not None:
        # v142: _v142_tranche was defined in the SL/TP elif block above (same
        # condition eval_price is not None). Reuse it; guard with try/except in
        # the unlikely case of control-flow change.
        try:
            _tranche_for_trail = _v142_tranche  # noqa: F821
        except NameError:
            from strategies import _find_tranche_config as _pt_find_tranche
            _tranche_for_trail = _pt_find_tranche(
                trade.get("strategy", ""), trade.get("tranche_label", "main") or "main"
            ) or {}
        trail_pct, activation_pct = _get_trail_config(trade)
        # v142: trail_tiers override — scalar trail_pct varies with peak ratio.
        _tiers = _tranche_for_trail.get("trail_tiers")
        if _tiers and high_seen > 0 and entry_price > 0:
            _ratio = high_seen / entry_price
            _sorted_tiers = sorted(_tiers)
            _tier_pct = _sorted_tiers[0][1]
            for _pm, _pp in _sorted_tiers:
                if _ratio >= _pm:
                    _tier_pct = _pp
                else:
                    break
            trail_pct = _tier_pct
            if activation_pct is None:
                activation_pct = float(_tranche_for_trail.get("trail_activation_pct") or 0.15)
        if trail_pct is not None and high_seen > 0 and market_ref_price > 0:
            activation_price = market_ref_price * (1 + activation_pct)
            if high_seen >= activation_price:
                trail_trigger = high_seen * (1 - trail_pct)
                if eval_price <= trail_trigger and trail_trigger > entry_price:
                    new_status = "trail_stop"
                    # v124: Detect true crash vs normal pullback.
                    # With 30s check intervals, memecoins routinely swing 5-15% between
                    # checks — that's normal volatility, not a crash. Only classify as
                    # crash when price is >30% below trigger (liquidity rug / true dump).
                    # v119 used 0.95 threshold which classified 58% of exits as "crash".
                    crash_ratio = current_price / trail_trigger if trail_trigger > 0 else 1.0
                    if crash_ratio < 0.70:
                        # True crash/rug: price >30% below trigger — liquidity gone
                        exit_price = current_price * _dynamic_sell_slip_factor(trade, "trail_crash", base_bps, sell_fee_bps)
                    else:
                        # Normal trail exit: Jupiter RFQ fills tight
                        exit_price = current_price * _dynamic_sell_slip_factor(trade, "trail_stop", base_bps, sell_fee_bps)
    # 4) Timeout
    if new_status is None and elapsed_minutes >= horizon:
        new_status = "timeout"
        exit_price = (current_price if current_price else entry_price) * _dynamic_sell_slip_factor(trade, "timeout", base_bps, sell_fee_bps)

    # Always return high_price_seen update (even without exit)
    result = {"high_price_seen": high_seen}

    if new_status is None:
        return result  # no exit, but may need to update high_price_seen

    pnl_pct = round((exit_price / entry_price) - 1, 4) if exit_price and entry_price else 0
    # v99: Shadow trades (pos_usd=0) get simulated $10 pnl_usd so stats aren't NULL
    effective_usd = pos_usd if pos_usd else 10.0
    pnl_usd = round(effective_usd * pnl_pct, 2)

    result.update({
        "status": new_status,
        "exit_price": exit_price,
        "exit_at": now.isoformat(),
        "pnl_pct": pnl_pct,
        "pnl_usd": pnl_usd,
        "exit_minutes": int(elapsed_minutes),
        "sol_price_at_exit": _get_sol_price(),  # v121: SOL context at exit
    })
    return result


# ---------------------------------------------------------------------------
# v115: DIP_BUY watchlist — monitor for dip+bounce to open P2
# ---------------------------------------------------------------------------

def _rebuild_dip_watchlist(client):
    """Rebuild dip watchlist from open P1 trades on startup."""
    global _dip_watchlist, _dip_watchlist_rebuilt
    if _dip_watchlist_rebuilt:
        return
    _dip_watchlist_rebuilt = True

    try:
        result = (
            client.table("paper_trades")
            .select("*")
            .eq("status", "open")
            .eq("tranche_label", "dip_p1")
            .execute()
        )
        for trade in (result.data or []):
            addr = trade["token_address"]
            strat = trade["strategy"]
            key = (addr, strat)

            # Skip if P2 already opened
            p2 = (
                client.table("paper_trades")
                .select("id", count="exact")
                .eq("token_address", addr)
                .eq("strategy", strat)
                .eq("tranche_label", "dip_p2")
                .neq("status", "open")  # if P2 exists and closed, also skip
                .execute()
            )
            p2_open = (
                client.table("paper_trades")
                .select("id", count="exact")
                .eq("token_address", addr)
                .eq("strategy", strat)
                .eq("tranche_label", "dip_p2")
                .eq("status", "open")
                .execute()
            )
            if (p2.count and p2.count > 0) or (p2_open.count and p2_open.count > 0):
                continue

            # v116: Try split regex first, then shared
            ms = _DIP_SPLIT_RE.match(strat)
            m = _DIP_RE.match(strat) if not ms else None
            if not ms and not m:
                continue

            created = datetime.fromisoformat(trade["created_at"].replace("Z", "+00:00"))
            if ms:
                dip_pct = int(ms.group(1)) / 100
                bounce_pct = int(ms.group(2)) / 100
                sl_pct = int(ms.group(8)) / 100  # P2 SL
                horizon_min = int(ms.group(9))
            else:
                dip_pct = int(m.group(1)) / 100
                bounce_pct = int(m.group(2)) / 100
                sl_pct = int(m.group(5)) / 100
                horizon_min = int(m.group(6))

            _dip_watchlist[key] = {
                "entry_price": float(trade["entry_price"]),
                "low_seen": float(trade["entry_price"]),  # conservative reset
                "dip_pct": dip_pct,
                "bounce_pct": bounce_pct,
                "sl_pct": sl_pct,
                "horizon_min": horizon_min,
                "alloc_usd": float(trade["position_usd"]),  # P1 pos = P2 pos (50/50)
                "base_row": {
                    "cycle_ts": trade.get("cycle_ts"),
                    "symbol": trade.get("symbol"),
                    "token_address": addr,
                    "pair_address": trade.get("pair_address"),
                    "rank_in_cycle": trade.get("rank_in_cycle"),
                    "entry_score": trade.get("entry_score"),
                    "entry_mcap": trade.get("entry_mcap"),
                    "source": trade.get("source"),
                    "kol_group": trade.get("kol_group"),
                    "is_shadow": trade.get("is_shadow", False),
                },
                "created_at": created,
            }

        if _dip_watchlist:
            logger.info("dip_watch: rebuilt watchlist with %d entries", len(_dip_watchlist))
    except Exception as e:
        logger.warning("dip_watch: rebuild failed: %s", e)


def _process_dip_watchlist(client, prices: dict, now: datetime):
    """Check dip+bounce conditions for DIP_BUY P2 entries."""
    to_remove = []

    for key, watch in _dip_watchlist.items():
        addr, strat_name = key
        current_price = prices.get(addr)
        if current_price is None:
            continue

        # Update low_seen
        if current_price < watch["low_seen"]:
            watch["low_seen"] = current_price

        # Check timeout — watchlist shouldn't outlive the strategy horizon
        elapsed = (now - watch["created_at"]).total_seconds() / 60
        if elapsed >= watch["horizon_min"]:
            to_remove.append(key)
            logger.debug("dip_watch: timeout for %s/%s after %dmin", addr[:8], strat_name, int(elapsed))
            continue

        # Check if P1 is still open
        try:
            p1_check = (
                client.table("paper_trades")
                .select("id", count="exact")
                .eq("token_address", addr)
                .eq("strategy", strat_name)
                .eq("tranche_label", "dip_p1")
                .eq("status", "open")
                .execute()
            )
            if not p1_check.count or p1_check.count == 0:
                to_remove.append(key)
                logger.info("dip_watch: P1 closed for %s/%s, cancelling P2 watch", addr[:8], strat_name)
                continue
        except Exception:
            continue

        # Check dip condition: price dropped dip_pct from P1 entry
        dip_level = watch["entry_price"] * (1 - watch["dip_pct"])
        if watch["low_seen"] <= dip_level:
            # Dip triggered — check bounce from low
            if watch["low_seen"] > 0:
                bounce_from_low = current_price / watch["low_seen"] - 1
            else:
                bounce_from_low = 0
            if bounce_from_low >= watch["bounce_pct"]:
                # BOUNCE CONFIRMED — open P2
                _open_dip_p2(client, key, watch, current_price, now)
                to_remove.append(key)

    for key in to_remove:
        _dip_watchlist.pop(key, None)


def _open_dip_p2(client, key: tuple, watch: dict, current_price: float, now: datetime):
    """Open DIP_BUY P2 trade on confirmed bounce."""
    addr, strat_name = key

    # P2 entry = low * (1 + bounce) * (1 + slippage + fee)
    buy_slip = BUY_SLIPPAGE_BPS / 10_000 + BUY_FEE_BPS / 10_000
    p2_entry = watch["low_seen"] * (1 + watch["bounce_pct"]) * (1 + buy_slip)
    p2_sl = p2_entry * (1 - watch["sl_pct"])

    # P2 timeout = remaining time from P1 entry
    elapsed_since_p1 = (now - watch["created_at"]).total_seconds() / 60
    remaining_horizon = max(int(watch["horizon_min"] - elapsed_since_p1), 10)  # min 10min

    row = {
        **watch["base_row"],
        "strategy": strat_name,
        "entry_price": p2_entry,
        "sl_price": p2_sl,
        "tp_price": None,
        "horizon_minutes": remaining_horizon,
        "tranche_pct": 0.5,
        "tranche_label": "dip_p2",
        "position_usd": round(watch["alloc_usd"], 2),
        "status": "open",
        "high_price_seen": current_price,
    }

    try:
        client.table("paper_trades").insert(row).execute()
        logger.info(
            "dip_watch: OPENED P2 %s/%s — entry=$%.8f (low=$%.8f, bounce=+%.0f%%), alloc=$%.0f, horizon=%dmin",
            watch["base_row"].get("symbol", "?"), strat_name,
            p2_entry, watch["low_seen"], watch["bounce_pct"] * 100,
            watch["alloc_usd"], remaining_horizon,
        )
        # Alert for P2 open
        if not watch["base_row"].get("is_shadow"):
            try:
                from alerter import _send as _alert_send
                dip_pct = (1 - watch["low_seen"] / watch["entry_price"]) * 100
                msg = (
                    f"🔄 DIP BUY P2\n\n"
                    f"${watch['base_row'].get('symbol', '?')} dipped <b>-{dip_pct:.0f}%</b> → bounced +{watch['bounce_pct']*100:.0f}%\n"
                    f"💰 P2 entry: ${p2_entry:.8f} | ${watch['alloc_usd']:.0f}\n"
                    f"⏱ Remaining: {remaining_horizon}min"
                )
                _alert_send(msg, "dip_buy_p2")
            except Exception:
                pass
    except Exception as e:
        logger.error("dip_watch: P2 insert failed for %s/%s: %s", addr[:8], strat_name, e)


def _precache_ohlcv_for_closed(closed_ids: set, trades: list[dict]):
    """v114: Pre-cache OHLCV for recently closed trades so sim.py has data ready.

    Uses sim.py's cache format (ohlcv_cache/{pool}_{ts}_{window}_{hash}.json).
    Only fetches for trades with pair_address that don't already have cached data.
    Non-blocking: errors are silently logged and don't affect trade processing.
    """
    import hashlib
    import json
    import os
    from pathlib import Path

    cache_dir = Path(__file__).resolve().parent / "ohlcv_cache"
    cache_dir.mkdir(exist_ok=True)
    window = 365  # match sim.py MAX_WINDOW_MIN

    closed_trades = [t for t in trades if t["id"] in closed_ids and t.get("pair_address")]
    # Dedup by (pair_address, created_at) — multiple strategies for same token
    seen = set()
    to_fetch = []
    for t in closed_trades:
        pool = t["pair_address"]
        created = t["created_at"]
        key = (pool, created)
        if key in seen:
            continue
        seen.add(key)

        dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
        start_ts = int(dt.timestamp())
        h = hashlib.md5(f"{pool}_{start_ts}_{window}".encode()).hexdigest()[:12]
        cache_file = cache_dir / f"{pool[:12]}_{start_ts}_{window}_{h}.json"
        if cache_file.exists():
            continue  # already cached
        to_fetch.append({"pool": pool, "token": t["token_address"], "start_ts": start_ts,
                         "end_ts": start_ts + window * 60, "cache_file": cache_file})

    if not to_fetch:
        return

    logger.info("ohlcv_precache: fetching %d OHLCV windows for closed trades", len(to_fetch))
    fetched = 0
    for entry in to_fetch[:10]:  # limit to 10 per cycle to avoid rate limits
        try:
            # Try DexPaprika first (free, uses pool address)
            start_iso = datetime.fromtimestamp(entry["start_ts"], tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            end_iso = datetime.fromtimestamp(entry["end_ts"], tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            r = requests.get(
                f"https://api.dexpaprika.com/networks/solana/pools/{entry['pool']}/ohlcv",
                params={"start": start_iso, "end": end_iso, "interval": "15m"},
                timeout=15,
            )
            candles = []
            if r.status_code == 200:
                for c in r.json():
                    ts_str = c.get("time_open") or c.get("time_close") or c.get("timestamp")
                    if not ts_str:
                        continue
                    ts = int(datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()) \
                        if isinstance(ts_str, str) else int(ts_str)
                    candles.append({"timestamp": ts, "open": float(c.get("open", 0)),
                                    "high": float(c.get("high", 0)), "low": float(c.get("low", 0)),
                                    "close": float(c.get("close", 0)),
                                    "volume": float(c.get("volume", 0))})
                candles.sort(key=lambda x: x["timestamp"])

            # v117: Fallback to GeckoTerminal if DexPaprika returns no data
            if len(candles) < 3:
                try:
                    pool_addr = entry["pool"]
                    gt_r = requests.get(
                        f"https://api.geckoterminal.com/api/v2/networks/solana/pools/{pool_addr}/ohlcv/minute",
                        params={"aggregate": "15", "limit": 100,
                                "before_timestamp": entry["end_ts"]},
                        headers={"Accept": "application/json"},
                        timeout=15,
                    )
                    if gt_r.status_code == 200:
                        gt_data = gt_r.json().get("data", {}).get("attributes", {}).get("ohlcv_list", [])
                        candles = []
                        for row in gt_data:
                            if len(row) >= 6:
                                candles.append({"timestamp": int(row[0]), "open": float(row[1]),
                                                "high": float(row[2]), "low": float(row[3]),
                                                "close": float(row[4]), "volume": float(row[5])})
                        candles.sort(key=lambda x: x["timestamp"])
                        # Filter to our window
                        candles = [c for c in candles if entry["start_ts"] <= c["timestamp"] <= entry["end_ts"]]
                except Exception:
                    pass
                _time_mod.sleep(0.5)  # rate limit GeckoTerminal

            if len(candles) >= 3:
                entry["cache_file"].write_text(json.dumps(candles))
                fetched += 1
            # v117: Don't cache empty results — retry next cycle
        except Exception as e:
            logger.debug("ohlcv_precache: error for %s: %s", entry["pool"][:12], e)

    if fetched > 0:
        logger.info("ohlcv_precache: cached %d/%d OHLCV windows", fetched, len(to_fetch[:10]))


def check_paper_trades(client) -> dict:
    """
    Check all open paper trades against current prices.
    Closes trades that hit TP, SL, trailing stop, or timeout.
    v73: Exit prices include sell slippage simulation.
    v92: Uses _evaluate_trade_exit() helper with trailing stop support.

    SL cascade: when SL triggers, ALL open tranches for the same
    (token_address, strategy, cycle_ts) close at -SL%.

    Moonbag tranches (tp_price=NULL) only close on SL or timeout.

    Returns {"checked": N, "closed": M, "tp": X, "sl": Y, "timeout": Z,
            "pnl_usd": total, "rt_pnl_usd": RT-only}.
    """
    now = datetime.now(timezone.utc)

    try:
        result = client.table("paper_trades").select("*").eq("status", "open").neq("source", "rt_live").execute()
        open_trades = result.data or []
        if _monitoring:
            _estimate_egress("paper_trader", "paper_trades", len(open_trades))
    except Exception as e:
        logger.error("paper_trader: failed to fetch open trades: %s", e)
        return {"checked": 0, "closed": 0, "tp": 0, "sl": 0, "timeout": 0, "pnl_usd": 0, "rt_pnl_usd": 0, "rt_closed": 0}

    if not open_trades:
        return {"checked": 0, "closed": 0, "tp": 0, "sl": 0, "timeout": 0, "pnl_usd": 0, "rt_pnl_usd": 0, "rt_closed": 0}

    # Batch fetch current prices
    addresses = list({t["token_address"] for t in open_trades})
    # v115: Include watchlist tokens in price fetch
    for (waddr, _) in _dip_watchlist:
        if waddr not in addresses:
            addresses.append(waddr)
    prices = _fetch_prices_batch(addresses)
    _log_price_ticks(client, prices, "full")
    _log_cache_snapshot(client)  # v138 D: snapshot full cache state

    # v115: Process dip watchlist on full check too
    if _dip_watchlist:
        _process_dip_watchlist(client, prices, now)

    # v73: Load sell slippage + fee config for exit price simulation
    # v142: also load buy slip so we can persist both bps on close (observability
    # gap: paper_trades rows had NULL buy/sell_slippage_bps, blocking divergence
    # analysis against live's real execution slippage).
    _sell_slip_bps = SELL_SLIPPAGE_BPS
    _sell_fee_bps = SELL_FEE_BPS
    _buy_slip_bps = BUY_SLIPPAGE_BPS
    _active_strats = []
    try:
        _cfg = _load_paper_trade_config(client)
        _sell_slip_bps = int(_cfg.get("sell_slippage_bps", SELL_SLIPPAGE_BPS))
        _sell_fee_bps = int(_cfg.get("sell_fee_bps", SELL_FEE_BPS))
        _buy_slip_bps = int(_cfg.get("buy_slippage_bps", BUY_SLIPPAGE_BPS))
        _active_strats = _cfg.get("active_strategies", [])
    except Exception:
        pass
    _sell_slip_factor = 1 - _sell_slip_bps / 10_000

    counts = {"checked": len(open_trades), "closed": 0, "tp": 0, "sl": 0, "timeout": 0}
    _total_pnl_usd = 0.0
    _rt_pnl_usd = 0.0
    _rt_closed = 0

    # Track SL-triggered groups so we can cascade
    sl_triggered = set()

    # v132: Load orchestration config for per-strategy polling + price source
    _rt_cfg_orch = {}
    try:
        from safe_scraper import _rt_load_config as _rt_load
        _rt_cfg_orch = _rt_load() or {}
    except Exception:
        pass

    # Sort so main/tp tranches come before moonbag (SL detection first)
    sorted_trades = sorted(open_trades, key=lambda t: (t.get("tranche_label", "") == "moonbag"))
    closed_ids = set()

    for trade in sorted_trades:
        if trade["id"] in closed_ids:
            continue

        addr = trade["token_address"]
        strategy = trade.get("strategy", "")
        trade_id = trade.get("id")
        orch = _strategy_orchestration(strategy, _rt_cfg_orch)

        # v132: Per-strategy polling skip (full-cycle check, ~3-15min interval)
        # For the slow check we're far less aggressive about skipping since it
        # already runs infrequently. Only skip if we polled very recently via fast check.
        if not _should_poll_trade(trade_id, int(orch.get("polling_sec", 30))):
            continue

        current_price = prices.get(addr)
        # v132: Orchestration — decision_price from configured source, exec stays Jupiter
        decision_price, exit_ref = _decision_price(addr, strategy, trade_id, orch, trade=trade)
        if exit_ref is not None:
            current_price = exit_ref

        # v115: DIP_BUY P1/P2 exit independently (no SL cascade between them)
        label = trade.get("tranche_label", "")
        if "dip_p" in label:
            group_key = (addr, strategy, trade["cycle_ts"], label)
        else:
            group_key = (addr, strategy, trade["cycle_ts"])

        is_cascade = group_key in sl_triggered
        # v138: record this poll BEFORE eval (captures every decision, even no-op)
        _record_eval_poll(trade_id, now, decision_price, current_price,
                          float(trade.get("high_price_seen") or 0))
        # v142 D: accumulate tick into OHLC buffer for ohlc_burst_60s mode.
        # Cheap no-op for other modes.
        _record_ohlc_tick(trade_id, current_price)

        # v142 D: ohlc_burst_60s — at each bar close, emit 4 synthetic OHLC
        # ticks to `_evaluate_trade_exit` in sequence before the normal eval.
        # Port of sim_engines.candles_to_synthetic_ticks() so intra-bar wicks
        # captured by our polling (high/low of last 60s) can trigger SL/TP/trail
        # in the correct chronological order (bullish: O→L→H→C, bearish: O→H→L→C).
        if orch.get("price_source") == "ohlc_burst_60s":
            burst_prices = _decision_prices_burst(trade_id, bar_sec=60)
            if burst_prices:
                burst_exit = False
                ev = None
                for i, burst_p in enumerate(burst_prices):
                    ev = _evaluate_trade_exit(trade, burst_p, now, 1.0,
                                              sl_cascade=is_cascade,
                                              sell_fee_bps=0,
                                              decision_price=burst_p)
                    if ev and ev.get("status"):
                        burst_exit = True
                        break
                if burst_exit:
                    ev = _override_exit_with_ultra_quote(client, trade, ev)
                    # Fall through to the existing close-handling code below
                else:
                    ev = None  # reset so the normal eval runs

        if orch.get("price_source") != "ohlc_burst_60s" or (ev is None):
            # Normal eval path (or burst didn't trigger → continue with current tick)
            # v123: sell_slip_factor=1.0 to match live (Jupiter Ultra RFQ = near-zero slippage)
            ev = _evaluate_trade_exit(trade, current_price, now, 1.0, sl_cascade=is_cascade,
                                       sell_fee_bps=0, decision_price=decision_price)
        if ev is None:
            continue
        ev = _override_exit_with_ultra_quote(client, trade, ev)

        # Always update high_price_seen (even without exit)
        if ev.get("high_price_seen") is not None and ev["high_price_seen"] > float(trade.get("high_price_seen") or 0):
            try:
                client.table("paper_trades").update(
                    {"high_price_seen": ev["high_price_seen"]}
                ).eq("id", trade["id"]).execute()
            except Exception:
                pass

        if "status" not in ev:
            continue  # no exit

        new_status = ev["status"]
        pnl_usd = ev.get("pnl_usd")

        # Track SL cascade
        if new_status == "sl_hit" and not is_cascade:
            sl_triggered.add(group_key)

        update = {k: ev[k] for k in ("status", "exit_price", "exit_at", "pnl_pct", "pnl_usd", "exit_minutes", "sol_price_at_exit") if k in ev}
        if ev.get("high_price_seen") is not None:
            update["high_price_seen"] = ev["high_price_seen"]
        # v142: persist slip bps assumed by the close so divergence vs live's real
        # fill is measurable. Previously NULL on every paper row — blocking audit.
        update["buy_slippage_bps"] = _buy_slip_bps
        update["sell_slippage_bps"] = _sell_slip_bps
        # v138: persist accumulated poll history alongside close fields
        hist = _flush_eval_history(trade["id"])
        if hist:
            update["eval_history"] = hist

        try:
            # v114: Conditional update — only close if still open (prevents race with fast check)
            res = client.table("paper_trades").update(update).eq("id", trade["id"]).eq("status", "open").execute()
            if not res.data:
                logger.debug("paper_trader: trade %s already closed by another loop, skipping", trade["id"])
                continue
            closed_ids.add(trade["id"])
            counts["closed"] += 1
            _total_pnl_usd += pnl_usd or 0
            if trade.get("source") == "rt" and not trade.get("is_shadow"):
                _rt_pnl_usd += pnl_usd or 0
                _rt_closed += 1
                # v113: Update bankroll BEFORE alert so balance is fresh
                try:
                    from safe_scraper import _rt_update_bankroll, _rt_load_bankroll
                    _rt_update_bankroll(pnl_usd or 0, 1, strategy=trade.get("strategy", ""))
                    _br = _rt_load_bankroll()
                    bal = float(_br.get("current_balance", 0))
                    _strat_bals = _br.get("strategy_bankrolls") or {}
                except Exception:
                    bal = 0
                    _strat_bals = {}
                try:
                    from alerter import alert_trade_closed
                    portfolio = get_open_portfolio(client)
                    alert_trade_closed(
                        symbol=trade["symbol"], strategy=trade["strategy"],
                        exit_reason=new_status,
                        pnl_pct=ev.get("pnl_pct", 0), pnl_usd=pnl_usd or 0,
                        pos_usd=float(trade.get("position_usd") or 0),
                        entry_price=float(trade.get("entry_price") or 0),
                        exit_price=ev.get("exit_price", 0),
                        high_price=ev.get("high_price_seen", 0),
                        minutes=int(ev.get("exit_minutes", 0)),
                        kol=trade.get("kol_group", ""),
                        bankroll=bal,
                        ca=trade.get("token_address", ""),
                        deployed_usd=portfolio["deployed_usd"],
                        open_count=portfolio["open_count"],
                        strategy_bankrolls=_strat_bals,
                        active_strategies=_active_strats,
                        price_source="jupiter" if addr in _jupiter_overridden else "",
                    )
                except Exception as e:
                    logger.warning("trade close alert failed: %s", e)
            status_key = new_status.replace("_hit", "").replace("_stop", "")
            counts[status_key] = counts.get(status_key, 0) + 1
            usd_str = f" ${pnl_usd:+.2f}" if pnl_usd is not None else ""
            logger.info(
                "paper_trader: CLOSED %s %s/%s/%s — %s pnl=%.1f%%%s after %dmin",
                trade["symbol"], trade["strategy"], trade.get("tranche_label", "main"),
                addr[:8], new_status, ev.get("pnl_pct", 0) * 100, usd_str, ev.get("exit_minutes", 0),
            )
        except Exception as e:
            logger.error("paper_trader: update failed for trade %s: %s", trade["id"], e)

    # Second pass: close remaining open trades in SL-triggered groups
    for trade in open_trades:
        if trade["id"] in closed_ids:
            continue
        # v115: DIP_BUY P1/P2 exit independently
        label = trade.get("tranche_label", "")
        if "dip_p" in label:
            group_key = (trade["token_address"], trade["strategy"], trade["cycle_ts"], label)
        else:
            group_key = (trade["token_address"], trade["strategy"], trade["cycle_ts"])
        if group_key not in sl_triggered:
            continue

        addr = trade["token_address"]
        current_price = prices.get(addr)

        ev = _evaluate_trade_exit(trade, current_price, now, _sell_slip_factor, sl_cascade=True, sell_fee_bps=_sell_fee_bps)
        if ev is None or "status" not in ev:
            continue
        ev = _override_exit_with_ultra_quote(client, trade, ev)

        pnl_usd = ev.get("pnl_usd")
        update = {k: ev[k] for k in ("status", "exit_price", "exit_at", "pnl_pct", "pnl_usd", "exit_minutes", "sol_price_at_exit") if k in ev}
        if ev.get("high_price_seen") is not None:
            update["high_price_seen"] = ev["high_price_seen"]
        # v142: persist slip bps on cascade close too (see main pass above)
        update["buy_slippage_bps"] = _buy_slip_bps
        update["sell_slippage_bps"] = _sell_slip_bps

        try:
            # v114: Conditional update — only close if still open (prevents race)
            res = client.table("paper_trades").update(update).eq("id", trade["id"]).eq("status", "open").execute()
            if not res.data:
                logger.debug("paper_trader: cascade trade %s already closed, skipping", trade["id"])
                continue
            closed_ids.add(trade["id"])
            counts["closed"] += 1
            _total_pnl_usd += pnl_usd or 0
            if trade.get("source") == "rt" and not trade.get("is_shadow"):
                _rt_pnl_usd += pnl_usd or 0
                _rt_closed += 1
                # v113: Update bankroll BEFORE alert so balance is fresh
                try:
                    from safe_scraper import _rt_update_bankroll, _rt_load_bankroll
                    _rt_update_bankroll(pnl_usd or 0, 1, strategy=trade.get("strategy", ""))
                    _br = _rt_load_bankroll()
                    bal = float(_br.get("current_balance", 0))
                    _strat_bals = _br.get("strategy_bankrolls") or {}
                except Exception:
                    bal = 0
                    _strat_bals = {}
                try:
                    from alerter import alert_trade_closed
                    portfolio = get_open_portfolio(client)
                    alert_trade_closed(
                        symbol=trade["symbol"], strategy=trade["strategy"],
                        exit_reason=ev.get("status", "sl_hit"),
                        pnl_pct=ev.get("pnl_pct", 0), pnl_usd=pnl_usd or 0,
                        pos_usd=float(trade.get("position_usd") or 0),
                        entry_price=float(trade.get("entry_price") or 0),
                        exit_price=ev.get("exit_price", 0),
                        high_price=ev.get("high_price_seen", 0),
                        minutes=int(ev.get("exit_minutes", 0)),
                        kol=trade.get("kol_group", ""),
                        bankroll=bal,
                        ca=trade.get("token_address", ""),
                        deployed_usd=portfolio["deployed_usd"],
                        open_count=portfolio["open_count"],
                        strategy_bankrolls=_strat_bals,
                        active_strategies=_active_strats,
                    )
                except Exception as e:
                    logger.warning("SL cascade trade close alert failed: %s", e)
            counts["sl"] = counts.get("sl", 0) + 1
            usd_str = f" ${pnl_usd:+.2f}" if pnl_usd is not None else ""
            logger.info(
                "paper_trader: CLOSED (SL cascade) %s %s/%s — pnl=%.1f%%%s",
                trade["symbol"], trade["strategy"], trade.get("tranche_label", ""),
                ev.get("pnl_pct", 0) * 100, usd_str,
            )
        except Exception as e:
            logger.error("paper_trader: update failed for trade %s: %s", trade["id"], e)

    if counts["closed"] > 0:
        logger.info(
            "paper_trader: checked %d open, closed %d (TP=%d SL=%d timeout=%d trail=%d)",
            counts["checked"], counts["closed"], counts["tp"], counts["sl"],
            counts["timeout"], counts.get("trail", 0),
        )
        if _monitoring:
            _metrics.record_paper_trade_close(counts["closed"], _total_pnl_usd)

        # v114: Pre-cache OHLCV for closed trades (so sim.py has data ready)
        _precache_ohlcv_for_closed(closed_ids, sorted_trades)

    counts["pnl_usd"] = round(_total_pnl_usd, 2)
    counts["rt_pnl_usd"] = round(_rt_pnl_usd, 2)
    counts["rt_closed"] = _rt_closed
    return counts


def correct_closed_prices(client) -> int:
    """v107: Post-close price correction for recently closed trades.

    Problem: DexScreener may not index new pools during pump.fun→Raydium migration.
    Trades close (timeout) with high_price_seen=entry while the real price is x10+.
    This corrupts ML training data and PnL reporting.

    Fix: for trades closed in the last 6h, re-fetch current price via DexScreener+GeckoTerminal.
    If current price > high_price_seen, update high_price_seen (but NOT status/pnl — the trade
    was already closed, we just fix the tracking data for ML accuracy).

    Runs once per full cycle (~15min), not on every price refresh.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat()

    try:
        result = (
            client.table("paper_trades")
            .select("id, token_address, symbol, entry_price, high_price_seen, status")
            .neq("status", "open")
            .gte("exit_at", cutoff)
            .execute()
        )
        closed_trades = result.data or []
    except Exception as e:
        logger.warning("correct_closed_prices: fetch failed: %s", e)
        return 0

    if not closed_trades:
        return 0

    # Dedup addresses
    addresses = list({t["token_address"] for t in closed_trades})
    prices = _fetch_prices_batch(addresses)

    corrected = 0
    for trade in closed_trades:
        addr = trade["token_address"]
        current = prices.get(addr)
        if current is None or current <= 0:
            continue

        entry = float(trade.get("entry_price") or 0)
        old_high = float(trade.get("high_price_seen") or 0)

        if current > old_high and entry > 0:
            try:
                client.table("paper_trades").update(
                    {"high_price_seen": current}
                ).eq("id", trade["id"]).execute()
                corrected += 1
                if current / entry > 1.5:  # only log significant corrections
                    logger.info(
                        "correct_closed_prices: %s %s high %.1fx→%.1fx",
                        trade["symbol"], addr[:8],
                        old_high / entry if old_high > 0 else 0,
                        current / entry,
                    )
            except Exception:
                pass

    if corrected:
        logger.info("correct_closed_prices: updated %d/%d trades", corrected, len(closed_trades))
    return corrected


def check_paper_trades_fast(client) -> dict:
    """v92: Fast 30s check for recent RT trades only (opened in last 30 min).
    Catches fast spikes that the 3-min full check would miss.
    Only checks SL/TP — no timeout (too young). No SL cascade (single-tranche RT).
    """
    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(minutes=30)).isoformat()

    try:
        result = (
            client.table("paper_trades")
            .select("*")
            .eq("status", "open")
            .neq("source", "rt_live")  # v120: rt_live trades handled by live_trade_monitor (needs Jupiter sell)
            .gte("created_at", cutoff)
            .execute()
        )
        recent_trades = result.data or []
    except Exception as e:
        logger.warning("paper_fast: fetch failed: %s", e)
        return {"checked": 0, "closed": 0}

    # v115: Rebuild dip watchlist on first run after restart
    _rebuild_dip_watchlist(client)

    if not recent_trades and not _dip_watchlist:
        return {"checked": 0, "closed": 0}

    addresses = list({t["token_address"] for t in recent_trades})
    # v115: Include watchlist tokens in price fetch
    for (waddr, _) in _dip_watchlist:
        if waddr not in addresses:
            addresses.append(waddr)
    prices = _fetch_prices_batch(addresses) if addresses else {}
    _log_price_ticks(client, prices, "fast")
    _log_cache_snapshot(client)  # v138 D: snapshot full cache state

    # v115: Process dip watchlist every 30s
    if _dip_watchlist:
        _process_dip_watchlist(client, prices, now)

    if not recent_trades:
        return {"checked": 0, "closed": 0}

    _sell_slip_bps = SELL_SLIPPAGE_BPS
    _sell_fee_bps = SELL_FEE_BPS
    _buy_slip_bps = BUY_SLIPPAGE_BPS
    _active_strats = []
    try:
        _cfg = _load_paper_trade_config(client)
        _sell_slip_bps = int(_cfg.get("sell_slippage_bps", SELL_SLIPPAGE_BPS))
        _sell_fee_bps = int(_cfg.get("sell_fee_bps", SELL_FEE_BPS))
        _buy_slip_bps = int(_cfg.get("buy_slippage_bps", BUY_SLIPPAGE_BPS))
        _active_strats = _cfg.get("active_strategies", [])
    except Exception:
        pass
    _sell_slip_factor = 1 - _sell_slip_bps / 10_000

    # v132: Load rt_trade_config once for orchestration lookups
    _rt_cfg_orch = {}
    try:
        from safe_scraper import _rt_load_config as _rt_load
        _rt_cfg_orch = _rt_load() or {}
    except Exception:
        pass

    closed = 0
    for trade in recent_trades:
        addr = trade["token_address"]
        strategy = trade.get("strategy", "")
        trade_id = trade.get("id")
        orch = _strategy_orchestration(strategy, _rt_cfg_orch)

        # v132: Skip if polling interval not elapsed for this strategy
        if not _should_poll_trade(trade_id, int(orch.get("polling_sec", 30))):
            continue

        current_price = prices.get(addr)
        # v132: Compute decision_price based on strategy's price_source.
        decision_price, exit_ref = _decision_price(addr, strategy, trade_id, orch, trade=trade)
        # exit_ref (Jupiter) preferred as current_price for exit booking
        if exit_ref is not None:
            current_price = exit_ref

        # v138: record this poll BEFORE eval (captures every decision, even no-op)
        _record_eval_poll(trade_id, now, decision_price, current_price,
                          float(trade.get("high_price_seen") or 0))

        # v123: sell_slip_factor=1.0 to match live (Jupiter Ultra RFQ = near-zero slippage)
        ev = _evaluate_trade_exit(trade, current_price, now, 1.0, sell_fee_bps=0,
                                  decision_price=decision_price)
        if ev is None:
            continue
        ev = _override_exit_with_ultra_quote(client, trade, ev)

        # Update high_price_seen even without exit
        if ev.get("high_price_seen") is not None and ev["high_price_seen"] > float(trade.get("high_price_seen") or 0):
            try:
                client.table("paper_trades").update(
                    {"high_price_seen": ev["high_price_seen"]}
                ).eq("id", trade["id"]).execute()
            except Exception:
                pass

        if "status" not in ev:
            continue

        update = {k: ev[k] for k in ("status", "exit_price", "exit_at", "pnl_pct", "pnl_usd", "exit_minutes", "sol_price_at_exit") if k in ev}
        if ev.get("high_price_seen") is not None:
            update["high_price_seen"] = ev["high_price_seen"]
        # v142: persist slip bps (see check_paper_trades for rationale)
        update["buy_slippage_bps"] = _buy_slip_bps
        update["sell_slippage_bps"] = _sell_slip_bps
        # v138: persist accumulated poll history alongside close fields
        hist = _flush_eval_history(trade["id"])
        if hist:
            update["eval_history"] = hist

        try:
            # v114: Conditional update — only close if still open (prevents race with full check)
            res = client.table("paper_trades").update(update).eq("id", trade["id"]).eq("status", "open").execute()
            if not res.data:
                logger.debug("paper_fast: trade %s already closed by another loop, skipping", trade["id"])
                continue
            closed += 1
            pnl_usd = ev.get("pnl_usd")
            usd_str = f" ${pnl_usd:+.2f}" if pnl_usd is not None else ""
            logger.info(
                "paper_fast: CLOSED %s %s/%s — %s pnl=%.1f%%%s after %dmin",
                trade["symbol"], trade["strategy"], addr[:8],
                ev["status"], ev.get("pnl_pct", 0) * 100, usd_str, ev.get("exit_minutes", 0),
            )
            # v113: Update bankroll + alert on main RT trade close
            if trade.get("source") == "rt" and not trade.get("is_shadow"):
                try:
                    from safe_scraper import _rt_update_bankroll, _rt_load_bankroll
                    _rt_update_bankroll(pnl_usd or 0, 1, strategy=trade.get("strategy", ""))
                    _br = _rt_load_bankroll()
                    bal = float(_br.get("current_balance", 0))
                    _strat_bals = _br.get("strategy_bankrolls") or {}
                except Exception:
                    bal = 0
                    _strat_bals = {}
                try:
                    from alerter import alert_trade_closed
                    portfolio = get_open_portfolio(client)
                    alert_trade_closed(
                        symbol=trade["symbol"], strategy=trade["strategy"],
                        exit_reason=ev["status"],
                        pnl_pct=ev.get("pnl_pct", 0), pnl_usd=pnl_usd or 0,
                        pos_usd=float(trade.get("position_usd") or 0),
                        entry_price=float(trade.get("entry_price") or 0),
                        exit_price=ev.get("exit_price", 0),
                        high_price=ev.get("high_price_seen", 0),
                        minutes=int(ev.get("exit_minutes", 0)),
                        kol=trade.get("kol_group", ""),
                        bankroll=bal,
                        ca=trade.get("token_address", ""),
                        deployed_usd=portfolio["deployed_usd"],
                        open_count=portfolio["open_count"],
                        strategy_bankrolls=_strat_bals,
                        active_strategies=_active_strats,
                    )
                except Exception as e:
                    logger.warning("paper_fast trade close alert failed: %s", e)
        except Exception as e:
            logger.error("paper_fast: update failed for trade %s: %s", trade["id"], e)

    if closed > 0:
        logger.info("paper_fast: checked %d recent, closed %d", len(recent_trades), closed)
    return {"checked": len(recent_trades), "closed": closed}


def paper_trade_summary(client) -> dict | None:
    """
    Compute summary stats for closed paper trades (last 7 days).
    Per-strategy breakdown with weighted PnL for multi-tranche strategies.
    Returns summary dict or None if no trades.
    """
    try:
        result = (
            client.table("paper_trades")
            .select("*")
            .neq("status", "open")
            .eq("is_shadow", False)
            .gte("created_at", _days_ago_iso(7))
            .execute()
        )
        trades = result.data or []
    except Exception as e:
        logger.error("paper_trader: summary query failed: %s", e)
        return None

    if not trades:
        return None

    # Global stats
    total = len(trades)
    tp_count = sum(1 for t in trades if t["status"] == "tp_hit")
    sl_count = sum(1 for t in trades if t["status"] == "sl_hit")
    timeout_count = sum(1 for t in trades if t["status"] == "timeout")
    trail_count = sum(1 for t in trades if t["status"] == "trail_stop")

    pnls = [float(t["pnl_pct"]) for t in trades if t.get("pnl_pct") is not None]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]

    win_rate = len(winners) / total if total else 0
    avg_pnl = sum(pnls) / len(pnls) if pnls else 0

    # Dollar PnL
    total_invested = sum(float(t.get("position_usd") or 0) for t in trades)
    total_pnl_usd = sum(float(t.get("pnl_usd") or 0) for t in trades)
    roi_pct = round(total_pnl_usd / total_invested * 100, 2) if total_invested else 0

    # Per-strategy breakdown
    strategy_stats = {}
    for strat_name in STRATEGIES:
        strat_trades = [t for t in trades if t.get("strategy") == strat_name]
        if not strat_trades:
            continue

        # Group by (token_address, cycle_ts) for weighted PnL across tranches
        positions = {}
        for t in strat_trades:
            key = (t["token_address"], t["cycle_ts"])
            positions.setdefault(key, []).append(t)

        pos_pnls = []
        pos_wins = 0
        for key, pos_trades in positions.items():
            weighted_pnl = sum(
                float(t.get("pnl_pct", 0) or 0) * float(t.get("tranche_pct", 1.0))
                for t in pos_trades
            )
            pos_pnls.append(weighted_pnl)
            if weighted_pnl > 0:
                pos_wins += 1

        n_positions = len(pos_pnls)
        s_tp = sum(1 for t in strat_trades if t["status"] == "tp_hit")
        s_sl = sum(1 for t in strat_trades if t["status"] == "sl_hit")
        s_to = sum(1 for t in strat_trades if t["status"] == "timeout")
        s_avg_pnl = sum(pos_pnls) / n_positions if n_positions else 0

        s_winners = [p for p in pos_pnls if p > 0]
        s_losers = [p for p in pos_pnls if p < 0]
        s_pf = abs(sum(s_winners) / sum(s_losers)) if s_losers and sum(s_losers) != 0 else float("inf")

        s_invested = sum(float(t.get("position_usd") or 0) for t in strat_trades)
        s_pnl_usd = sum(float(t.get("pnl_usd") or 0) for t in strat_trades)

        # Enriched stats: expectancy, breakeven WR, max consecutive losses
        s_expectancy = sum(pos_pnls) / n_positions if n_positions else 0
        s_avg_win = sum(s_winners) / len(s_winners) if s_winners else 0
        s_avg_loss = abs(sum(s_losers)) / len(s_losers) if s_losers else 0
        s_breakeven_wr = s_avg_loss / (s_avg_win + s_avg_loss) if (s_avg_win + s_avg_loss) > 0 else 0.5
        s_max_consec = 0
        s_consec = 0
        for p in pos_pnls:
            if p < 0:
                s_consec += 1
                s_max_consec = max(s_max_consec, s_consec)
            else:
                s_consec = 0

        strategy_stats[strat_name] = {
            "positions": n_positions,
            "trade_rows": len(strat_trades),
            "tp": s_tp,
            "sl": s_sl,
            "timeout": s_to,
            "win_rate": round(pos_wins / n_positions, 3) if n_positions else 0,
            "avg_pnl": round(s_avg_pnl, 4),
            "expectancy": round(s_expectancy, 4),
            "profit_factor": round(s_pf, 2) if s_pf != float("inf") else "inf",
            "breakeven_wr": round(s_breakeven_wr, 4),
            "max_consecutive_losses": s_max_consec,
            "total_pnl_pct": round(sum(pos_pnls) * 100, 2),
            "invested_usd": round(s_invested, 2),
            "pnl_usd": round(s_pnl_usd, 2),
        }

    # Global enriched stats
    global_expectancy = avg_pnl  # avg_pnl IS expectancy (mean PnL per trade)
    global_avg_win = sum(winners) / len(winners) if winners else 0
    global_avg_loss = abs(sum(losers)) / len(losers) if losers else 0
    global_breakeven_wr = global_avg_loss / (global_avg_win + global_avg_loss) if (global_avg_win + global_avg_loss) > 0 else 0.5
    global_pf = abs(sum(winners) / sum(losers)) if losers and sum(losers) != 0 else float("inf")
    g_max_consec = 0
    g_consec = 0
    for p in pnls:
        if p < 0:
            g_consec += 1
            g_max_consec = max(g_max_consec, g_consec)
        else:
            g_consec = 0

    summary = {
        "total_rows": total,
        "tp": tp_count,
        "sl": sl_count,
        "timeout": timeout_count,
        "win_rate": round(win_rate, 3),
        "avg_pnl": round(avg_pnl, 4),
        "expectancy": round(global_expectancy, 4),
        "profit_factor": round(global_pf, 2) if global_pf != float("inf") else "inf",
        "breakeven_wr": round(global_breakeven_wr, 4),
        "max_consecutive_losses": g_max_consec,
        "total_invested_usd": round(total_invested, 2),
        "total_pnl_usd": round(total_pnl_usd, 2),
        "roi_pct": roi_pct,
        "strategies": strategy_stats,
    }

    # v66: RT vs batch split
    rt_trades = [t for t in trades if t.get("source") == "rt"]
    batch_trades = [t for t in trades if t.get("source") != "rt"]

    def _source_stats(src_trades, label):
        # Only count closed trades (already filtered by neq("status","open") above)
        closed = [t for t in src_trades if t.get("status") in ("tp_hit", "sl_hit", "timeout", "trail_stop")]
        if not closed:
            return None
        n = len(closed)
        w = sum(1 for t in closed if float(t.get("pnl_pct") or 0) > 0)
        inv = sum(float(t.get("position_usd") or 0) for t in closed)
        pnl = sum(float(t.get("pnl_usd") or 0) for t in closed)
        wr = w / n if n else 0
        return {"label": label, "rows": n, "win_rate": round(wr, 3),
                "invested": round(inv, 2), "pnl_usd": round(pnl, 2)}

    rt_stats = _source_stats(rt_trades, "RT")
    batch_stats = _source_stats(batch_trades, "batch")
    summary["rt_stats"] = rt_stats
    summary["batch_stats"] = batch_stats

    logger.info(
        "paper_trader SUMMARY (7d): %d rows, WR=%.1f%%, avgPnL=%.2f%%, E[R]=%.2f%%, PF=%s | "
        "$%.2f invested, $%+.2f PnL (ROI %.1f%%) | TP=%d SL=%d TO=%d | maxConsecL=%d",
        total, win_rate * 100, avg_pnl * 100, global_expectancy * 100,
        round(global_pf, 2) if global_pf != float("inf") else "inf",
        total_invested, total_pnl_usd, roi_pct,
        tp_count, sl_count, timeout_count, g_max_consec,
    )
    # v66: Log RT vs batch breakdown
    for ss in [rt_stats, batch_stats]:
        if ss:
            logger.info(
                "  [%s] %d rows, WR=%.1f%%, $%.2f invested, $%+.2f PnL",
                ss["label"], ss["rows"], ss["win_rate"] * 100, ss["invested"], ss["pnl_usd"],
            )
    for name, s in strategy_stats.items():
        logger.info(
            "  %s: %d pos, WR=%.1f%%, E[R]=%.2f%%, PF=%s, BEwr=%.1f%% | $%.2f→$%+.2f | maxCL=%d",
            name, s["positions"], s["win_rate"] * 100, s["expectancy"] * 100,
            s["profit_factor"], s["breakeven_wr"] * 100, s["invested_usd"], s["pnl_usd"],
            s["max_consecutive_losses"],
        )
    return summary


def _days_ago_iso(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def kol_attribution(client, days: int = 7) -> dict:
    """
    v74: Aggregate paper trade outcomes by KOL.
    Returns {kol_group: {total, wins, wr, pnl_usd, avg_pnl_pct, best_trade, worst_trade}}.
    Feeds back into KOL whitelist optimization.
    """
    try:
        result = (
            client.table("paper_trades")
            .select("kol_group, kol_tier, pnl_pct, pnl_usd, status, strategy, symbol")
            .neq("status", "open")
            .eq("is_shadow", False)
            .not_.is_("kol_group", "null")
            .gte("created_at", _days_ago_iso(days))
            .execute()
        )
        trades = result.data or []
    except Exception as e:
        logger.error("kol_attribution: query failed: %s", e)
        return {}

    if not trades:
        return {}

    # Group by KOL
    by_kol: dict[str, list[dict]] = {}
    for t in trades:
        kol = t.get("kol_group")
        if kol:
            by_kol.setdefault(kol, []).append(t)

    attribution = {}
    for kol, kol_trades in by_kol.items():
        total = len(kol_trades)
        pnls = [float(t.get("pnl_pct") or 0) for t in kol_trades]
        wins = sum(1 for p in pnls if p > 0)
        pnl_usd = sum(float(t.get("pnl_usd") or 0) for t in kol_trades)

        best = max(kol_trades, key=lambda t: float(t.get("pnl_pct") or 0))
        worst = min(kol_trades, key=lambda t: float(t.get("pnl_pct") or 0))

        attribution[kol] = {
            "tier": kol_trades[0].get("kol_tier", "?"),
            "total": total,
            "wins": wins,
            "wr": round(wins / total, 3) if total else 0,
            "pnl_usd": round(pnl_usd, 2),
            "avg_pnl_pct": round(sum(pnls) / total * 100, 2) if total else 0,
            "best_trade": f"{best.get('symbol')} +{float(best.get('pnl_pct') or 0)*100:.1f}%",
            "worst_trade": f"{worst.get('symbol')} {float(worst.get('pnl_pct') or 0)*100:.1f}%",
        }

    # Log top and bottom 5 KOLs
    sorted_kols = sorted(attribution.items(), key=lambda x: x[1]["pnl_usd"], reverse=True)
    logger.info("KOL Attribution (%dd, %d KOLs, %d trades):", days, len(attribution), sum(v["total"] for v in attribution.values()))
    for kol, stats in sorted_kols[:5]:
        logger.info("  TOP: %s (%s) — %d trades, WR=%.0f%%, PnL=$%+.2f",
                     kol, stats["tier"], stats["total"], stats["wr"]*100, stats["pnl_usd"])
    for kol, stats in sorted_kols[-3:]:
        if stats["pnl_usd"] < 0:
            logger.info("  BOT: %s (%s) — %d trades, WR=%.0f%%, PnL=$%+.2f",
                         kol, stats["tier"], stats["total"], stats["wr"]*100, stats["pnl_usd"])

    # v82 P2-2: Closed-loop — feed attribution back into kol_rt_whitelist
    # v103: Disabled auto-update — whitelist is now manually managed
    # _update_whitelist_from_attribution(client, attribution, by_kol, days)

    return attribution


def _update_whitelist_from_attribution(client, attribution: dict, by_kol: dict, days: int):
    """v82: Merge kol_attribution results into scoring_config.kol_rt_whitelist.
    Adds per-strategy breakdown and best_strategy for adaptive trading."""
    deprecated = {"MOONBAG", "WIDE_RUNNER", "SCALE_OUT", "TP100_SL30"}
    min_calls = 3
    wr_threshold = 0.40

    try:
        # Read current whitelist
        r = client.table("scoring_config").select("kol_rt_whitelist").eq("id", 1).execute()
        whitelist = {}
        if r.data and r.data[0].get("kol_rt_whitelist"):
            cached = r.data[0]["kol_rt_whitelist"]
            if isinstance(cached, str):
                import json
                cached = json.loads(cached)
            whitelist = cached

        # Build per-strategy breakdown from raw trades
        for kol, kol_trades in by_kol.items():
            active_trades = [t for t in kol_trades if t.get("strategy") not in deprecated]
            if not active_trades:
                continue

            attr = attribution.get(kol, {})
            total = len(active_trades)
            wins = sum(1 for t in active_trades if float(t.get("pnl_usd") or 0) > 0)
            pnl = sum(float(t.get("pnl_usd") or 0) for t in active_trades)
            wr = wins / total if total > 0 else 0

            # Per-strategy
            strat_map = {}
            from collections import defaultdict
            strat_agg = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0})
            for t in active_trades:
                s = t.get("strategy", "?")
                p = float(t.get("pnl_usd") or 0)
                strat_agg[s]["n"] += 1
                strat_agg[s]["pnl"] += p
                if p > 0:
                    strat_agg[s]["wins"] += 1

            best_strat = None
            best_pnl = -999
            for s, ss in strat_agg.items():
                swr = ss["wins"] / ss["n"] if ss["n"] > 0 else 0
                strat_map[s] = {"n": ss["n"], "wr": round(swr, 4), "pnl": round(ss["pnl"], 2)}
                if ss["pnl"] > best_pnl:
                    best_pnl = ss["pnl"]
                    best_strat = s

            approved = wr >= wr_threshold and total >= min_calls
            whitelist[kol] = {
                "wr": round(wr, 4),
                "total": total,
                "hits": wins,
                "pnl": round(pnl, 2),
                "approved": approved,
                "best_strategy": best_strat,
                "strategies": strat_map,
            }

        # Fallback: relax to 30% if < 3 approved
        approved_count = sum(1 for v in whitelist.values() if v.get("approved"))
        if approved_count < 3:
            for kol, wl in whitelist.items():
                if not wl.get("approved") and wl.get("total", 0) >= min_calls:
                    if wl.get("wr", 0) >= 0.30:
                        wl["approved"] = True

        # Write back
        from datetime import datetime, timezone
        client.table("scoring_config").update({
            "kol_rt_whitelist": whitelist,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", 1).execute()

        approved_count = sum(1 for v in whitelist.values() if v.get("approved"))
        logger.info("kol_attribution → whitelist updated: %d/%d approved", approved_count, len(whitelist))

    except Exception as e:
        logger.warning("kol_attribution → whitelist update failed: %s", e)
