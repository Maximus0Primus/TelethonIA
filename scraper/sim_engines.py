"""
Strategy simulation engines — pure functions, zero external dependencies.

Each engine takes (candles, entry_price, cfg) and returns:
    {"exit_reason": str, "pnl_pct": float, "elapsed_min": float}

Slippage constants are applied at exit.
"""

from strategies import BUY_SLIPPAGE_BPS as _PT_BUY_SLIPPAGE_BPS

# ---------------------------------------------------------------------------
# Slippage constants
# ---------------------------------------------------------------------------
# v126: Legacy fallback only. _exit() now calls production _dynamic_sell_slip_factor
# from paper_trader, matching live behavior (10bps base + liq/exit multipliers).
# These constants remain for data-end exits and the DIP_BUY buy-side.
# v14e.24 (Apr 25): BUY_SLIPPAGE / compute_buy_slippage now mirror the
# empirical median (strategies.BUY_SLIPPAGE_BPS = 225) — single source of truth
# across paper / sim / shadow. AMM impact term dropped: empirical R²=5%, the
# liquidity feature does not predict slip, so the previous theoretical
# `BASE_FEE + position/(liq+position)` was over-engineering.
# v14e.28 (Apr 26): added chain dispatch — Solana keeps the 225 bps median,
# EVM chains delegate to paper_trader._evm_slip_bps_with_gas which folds gas
# (flat USD → bps via position size) on top of per-chain base slip.
SLIPPAGE_TRAIL = 0.025   # legacy fallback — 2.5% for trail/timeout/TP (Solana)
SLIPPAGE_SL = 0.025      # legacy fallback — 2.5% for SL (Solana)
BUY_SLIPPAGE = _PT_BUY_SLIPPAGE_BPS / 10_000  # Solana — mirrors strategies.BUY_SLIPPAGE_BPS


def compute_buy_slippage(position_usd: float, liquidity_usd: float,
                         n_simultaneous: int = 1, chain: str = "solana") -> float:
    """v14e.28: chain-aware buy slip used by sim and paper. Returns slip as a
    fraction (e.g. 0.0225 = 225 bps).

    - Solana: returns the empirical median (paper_trader matches, single source
      of truth from strategies.BUY_SLIPPAGE_BPS).
    - EVM (ethereum/bsc/base): delegates to paper_trader._evm_slip_bps_with_gas
      which adds gas-as-bps on top of the per-chain base slip. Position-size-
      sensitive (gas dominates at small positions).

    Backward compat: chain defaults to 'solana' so existing callers (Solana sim
    paths) get bit-for-bit identical behavior.
    """
    if chain in ("ethereum", "bsc", "base"):
        try:
            from paper_trader import _evm_slip_bps_with_gas
            return _evm_slip_bps_with_gas(position_usd, chain, "buy") / 10_000
        except Exception:
            # Import cycle / test isolation — fall back to ETH constants directly
            from strategies import (
                ETH_GAS_COST_USD_PER_SIDE, ETH_BUY_SLIPPAGE_BPS,
                BSC_GAS_COST_USD_PER_SIDE, BSC_BUY_SLIPPAGE_BPS,
                BASE_GAS_COST_USD_PER_SIDE, BASE_BUY_SLIPPAGE_BPS,
            )
            params = {
                "ethereum": (ETH_GAS_COST_USD_PER_SIDE, ETH_BUY_SLIPPAGE_BPS),
                "bsc":      (BSC_GAS_COST_USD_PER_SIDE, BSC_BUY_SLIPPAGE_BPS),
                "base":     (BASE_GAS_COST_USD_PER_SIDE, BASE_BUY_SLIPPAGE_BPS),
            }[chain]
            gas_usd, base_slip = params
            pos = max(float(position_usd or 0), 1.0)
            gas_bps = int((gas_usd / pos) * 10_000)
            return max(50, min(2000, base_slip + gas_bps)) / 10_000
    return BUY_SLIPPAGE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dynamic_sell_slippage(liquidity_usd: float, is_sl: bool = False,
                            chain: str = "solana", position_usd: float = 0) -> float:
    """Sim-side legacy fallback for sell slippage. Production code goes through
    paper_trader._dynamic_sell_slip_factor (called via _exit below). This is
    used only when production fn unavailable (import cycles / standalone tests).

    v14e.28: chain dispatch added. SOL keeps liq-aware Solana model. EVM uses
    flat per-chain base + gas. Backward compat: chain defaults to solana so all
    existing callers behave identically.
    """
    if chain in ("ethereum", "bsc", "base"):
        try:
            from paper_trader import _evm_slip_bps_with_gas
            return _evm_slip_bps_with_gas(position_usd, chain, "sell") / 10_000
        except Exception:
            return 0.015  # 1.5% safe fallback for EVM (memecoin shallow pool)
    base_bps = 200
    fee_bps = 50
    liq = liquidity_usd if liquidity_usd > 0 else 13_000  # median from real trades
    liq_mult = max(1.0, min(4.0, 50_000 / max(liq, 1_000)))
    exit_mult = 1.5 if is_sl else 1.0
    adjusted_bps = int(base_bps * liq_mult * exit_mult) + fee_bps
    return adjusted_bps / 10_000


_sim_liquidity_usd = 0  # Set per-simulation by simulate() from context
_sim_chain = "solana"   # Set per-simulation by simulate() from context (v14e.28)
_sim_position_usd = 0   # Set per-simulation; needed for EVM gas-as-bps math


def _exit(reason: str, exit_price: float, entry_price: float,
          elapsed_min: float, is_sl: bool = False) -> dict:
    """v126: Use production _dynamic_sell_slip_factor (paper_trader) for legacy engines.
    Matches live slippage exactly: SOL = 10bps base + liq/exit multipliers,
    EVM = per-chain base + gas-as-bps via _evm_slip_bps_with_gas.

    v14e.28: pass chain + position into the production fn so EVM trades get
    the right (gas-dominated) slip. Falls back to flat 2.5% if prod fn fails."""
    try:
        from paper_trader import _dynamic_sell_slip_factor
        # v14e.28: include chain + position so paper_trader._dynamic_sell_slip_factor
        # routes EVM trades through _evm_slip_bps_with_gas (its own internal dispatch
        # at paper_trader.py:1632-1636). For SOL, position_usd is read from trade dict
        # by the production fn, so passing 0 is safe (matches default).
        fake_trade = {
            "rt_liquidity_usd": _sim_liquidity_usd or 50_000,
            "chain": _sim_chain,
            "position_usd": _sim_position_usd or 0,
        }
        exit_type = reason if reason in ("sl_hit", "tp_hit", "trail_stop", "trail_crash", "timeout") else "timeout"
        slip_factor = _dynamic_sell_slip_factor(fake_trade, exit_type)
        net_price = exit_price * slip_factor
    except Exception:
        slip = SLIPPAGE_SL if is_sl else SLIPPAGE_TRAIL
        net_price = exit_price * (1 - slip)
    return {
        "exit_reason": reason,
        "pnl_pct": net_price / entry_price - 1,
        "elapsed_min": elapsed_min,
    }


def _elapsed(candle: dict, base_ts: int) -> float:
    return (candle["timestamp"] - base_ts) / 60.0


def resample_to_live_checks(candles: list[dict], interval_min: int = 3,
                            fast_interval_sec: int = 30,
                            fast_window_min: int = 30,
                            hl_blend: float = 0.0) -> list[dict]:
    """
    Resample candles to simulate live price checks.
    - First `fast_window_min` min: check every `fast_interval_sec` seconds
    - After that: check every `interval_min` minutes

    high/low are blended between close-only and real extremes:
      blended_high = close + hl_blend * (real_high - close)
      blended_low  = close + hl_blend * (close - real_low) [towards low]

    hl_blend=0.0: close-only (old behavior, ignores intra-candle moves)
    hl_blend=1.0: full high/low (too aggressive, SLs trigger on noise)
    hl_blend=0.5: halfway — DexScreener sees ~50% of intra-candle range
    """
    if not candles:
        return candles
    base_ts = candles[0]["timestamp"]
    fast_cutoff = base_ts + fast_window_min * 60
    fast_interval = fast_interval_sec
    normal_interval = interval_min * 60

    sampled = []
    next_check_ts = base_ts
    # Accumulate high/low between checks
    pending_high = None
    pending_low = None

    for c in candles:
        h = c["high"]
        lo = c["low"]
        if pending_high is None:
            pending_high = h
            pending_low = lo
        else:
            pending_high = max(pending_high, h)
            pending_low = min(pending_low, lo)

        if c["timestamp"] >= next_check_ts:
            close = c["close"]
            # Blend high/low towards close based on hl_blend
            blended_high = close + hl_blend * (pending_high - close)
            blended_low = close - hl_blend * (close - pending_low)
            sampled.append({
                "timestamp": c["timestamp"],
                "open": close,
                "high": blended_high,
                "low": blended_low,
                "close": close,
                "volume": c.get("volume", 0),
            })
            pending_high = None
            pending_low = None
            if c["timestamp"] < fast_cutoff:
                next_check_ts = c["timestamp"] + fast_interval
            else:
                next_check_ts = c["timestamp"] + normal_interval

    if not sampled:
        c = candles[0]
        cl = c["close"]
        sampled.append({
            "timestamp": c["timestamp"], "open": cl,
            "high": cl + hl_blend * (c["high"] - cl),
            "low": cl - hl_blend * (cl - c["low"]),
            "close": cl, "volume": c.get("volume", 0),
        })

    return sampled


def _timeout_exit(candle: dict, entry_price: float, base_ts: int) -> dict:
    return _exit("timeout", candle["close"], entry_price, _elapsed(candle, base_ts))


def _data_end_exit(candles: list[dict], entry_price: float, base_ts: int) -> dict:
    last = candles[-1]
    return _exit("timeout_data", last["close"], entry_price, _elapsed(last, base_ts))


# ---------------------------------------------------------------------------
# FIXED / SCALP: static TP + SL + timeout
# ---------------------------------------------------------------------------

def simulate_fixed(candles: list[dict], entry_price: float, cfg: dict) -> dict:
    tp_price = entry_price * cfg["tp_mult"]
    sl_price = entry_price * cfg["sl_mult"]
    horizon = cfg["horizon_min"]
    base_ts = candles[0]["timestamp"]

    for c in candles:
        mins = _elapsed(c, base_ts)
        if c["low"] <= sl_price:
            return _exit("sl_hit", sl_price, entry_price, mins, is_sl=True)
        if c["high"] >= tp_price:
            return _exit("tp_hit", tp_price, entry_price, mins)
        if mins >= horizon:
            return _timeout_exit(c, entry_price, base_ts)

    return _data_end_exit(candles, entry_price, base_ts)


# ---------------------------------------------------------------------------
# DTRAIL: dynamic trailing stop, NO TP cap
# ---------------------------------------------------------------------------

def simulate_dtrail(candles: list[dict], entry_price: float, cfg: dict) -> dict:
    high_seen = entry_price
    activated = False
    sl_price = entry_price * cfg["sl_mult"]
    trail_pct = cfg["trail_pct"]
    act_pct = cfg["activation_pct"]
    horizon = cfg["horizon_min"]
    base_ts = candles[0]["timestamp"]

    for c in candles:
        mins = _elapsed(c, base_ts)
        if c["high"] > high_seen:
            high_seen = c["high"]
        if c["low"] <= sl_price:
            return _exit("sl_hit", sl_price, entry_price, mins, is_sl=True)
        if not activated and high_seen >= entry_price * (1 + act_pct):
            activated = True
        if activated:
            trail_trigger = high_seen * (1 - trail_pct)
            if trail_trigger > entry_price and c["low"] <= trail_trigger:
                return _exit("trail_stop", trail_trigger, entry_price, mins)
        if mins >= horizon:
            return _timeout_exit(c, entry_price, base_ts)

    return _data_end_exit(candles, entry_price, base_ts)


# ---------------------------------------------------------------------------
# TRAIL: trailing stop + TP cap
# ---------------------------------------------------------------------------

def simulate_trail_tp(candles: list[dict], entry_price: float, cfg: dict) -> dict:
    high_seen = entry_price
    activated = False
    tp_price = entry_price * cfg["tp_mult"]
    sl_price = entry_price * cfg["sl_mult"]
    trail_pct = cfg["trail_pct"]
    act_pct = cfg["activation_pct"]
    horizon = cfg["horizon_min"]
    base_ts = candles[0]["timestamp"]

    for c in candles:
        mins = _elapsed(c, base_ts)
        if c["high"] > high_seen:
            high_seen = c["high"]
        if c["low"] <= sl_price:
            return _exit("sl_hit", sl_price, entry_price, mins, is_sl=True)
        if c["high"] >= tp_price:
            return _exit("tp_hit", tp_price, entry_price, mins)
        if not activated and high_seen >= entry_price * (1 + act_pct):
            activated = True
        if activated:
            trail_trigger = high_seen * (1 - trail_pct)
            if trail_trigger > entry_price and c["low"] <= trail_trigger:
                return _exit("trail_stop", trail_trigger, entry_price, mins)
        if mins >= horizon:
            return _timeout_exit(c, entry_price, base_ts)

    return _data_end_exit(candles, entry_price, base_ts)


# ---------------------------------------------------------------------------
# BREAKEVEN: SL moves to entry after activation
# ---------------------------------------------------------------------------

def simulate_breakeven(candles: list[dict], entry_price: float, cfg: dict) -> dict:
    tp_price = entry_price * cfg["tp_mult"]
    original_sl = entry_price * cfg["sl_mult"]
    be_act_price = entry_price * (1 + cfg["be_activation"])
    horizon = cfg["horizon_min"]
    base_ts = candles[0]["timestamp"]
    be_activated = False

    for c in candles:
        mins = _elapsed(c, base_ts)
        if c["high"] >= be_act_price:
            be_activated = True
        effective_sl = entry_price if be_activated else original_sl
        if c["low"] <= effective_sl:
            reason = "be_hit" if be_activated else "sl_hit"
            return _exit(reason, effective_sl, entry_price, mins, is_sl=True)
        if c["high"] >= tp_price:
            return _exit("tp_hit", tp_price, entry_price, mins)
        if mins >= horizon:
            return _timeout_exit(c, entry_price, base_ts)

    return _data_end_exit(candles, entry_price, base_ts)


# ---------------------------------------------------------------------------
# DECAY: TP decays over time (first half static, second half linear)
# ---------------------------------------------------------------------------

def simulate_decay(candles: list[dict], entry_price: float, cfg: dict) -> dict:
    tp_start = cfg["tp_start"]
    tp_end = cfg["tp_end"]
    sl_price = entry_price * cfg["sl_mult"]
    horizon = cfg["horizon_min"]
    base_ts = candles[0]["timestamp"]
    half = horizon / 2.0

    for c in candles:
        mins = _elapsed(c, base_ts)
        if mins <= half:
            current_tp = entry_price * tp_start
        else:
            progress = min((mins - half) / half, 1.0)
            current_tp = entry_price * (tp_start + (tp_end - tp_start) * progress)
        if c["low"] <= sl_price:
            return _exit("sl_hit", sl_price, entry_price, mins, is_sl=True)
        if c["high"] >= current_tp:
            return _exit("tp_hit", current_tp, entry_price, mins)
        if mins >= horizon:
            return _timeout_exit(c, entry_price, base_ts)

    return _data_end_exit(candles, entry_price, base_ts)


# ---------------------------------------------------------------------------
# SPLIT: 2 tranches (50/50), SL kills both
# ---------------------------------------------------------------------------

def simulate_split(candles: list[dict], entry_price: float, cfg: dict) -> dict:
    t1_tp_price = entry_price * cfg["t1_tp"]
    t2_tp_price = entry_price * cfg["t2_tp"] if cfg.get("t2_tp") else None
    sl_price = entry_price * cfg["sl_mult"]
    t2_trail_pct = cfg.get("t2_trail")
    horizon = cfg["horizon_min"]
    base_ts = candles[0]["timestamp"]

    t1_closed = False
    t1_pnl = 0.0
    t2_closed = False
    t2_pnl = 0.0
    high_seen = entry_price
    t2_trail_active = False

    for c in candles:
        mins = _elapsed(c, base_ts)
        if c["high"] > high_seen:
            high_seen = c["high"]

        # SL kills both
        if c["low"] <= sl_price:
            sl_res = _exit("sl_hit", sl_price, entry_price, mins, is_sl=True)
            pnl_sl = sl_res["pnl_pct"]
            if not t1_closed:
                t1_pnl = pnl_sl
            if not t2_closed:
                t2_pnl = pnl_sl
            return {"exit_reason": "sl_hit", "pnl_pct": 0.5 * t1_pnl + 0.5 * t2_pnl,
                    "elapsed_min": mins}

        # T1: fixed TP
        if not t1_closed and c["high"] >= t1_tp_price:
            t1_pnl = _exit("tp_hit", t1_tp_price, entry_price, mins)["pnl_pct"]
            t1_closed = True

        # T2: fixed TP or trail
        if not t2_closed:
            if t2_tp_price and c["high"] >= t2_tp_price:
                t2_pnl = _exit("tp_hit", t2_tp_price, entry_price, mins)["pnl_pct"]
                t2_closed = True
            elif t2_trail_pct:
                if t1_closed and not t2_trail_active:
                    t2_trail_active = True
                if t2_trail_active:
                    trail_trigger = high_seen * (1 - t2_trail_pct)
                    if trail_trigger > entry_price and c["low"] <= trail_trigger:
                        t2_pnl = _exit("trail_stop", trail_trigger, entry_price, mins)["pnl_pct"]
                        t2_closed = True

        if t1_closed and t2_closed:
            return {"exit_reason": "tp_hit", "pnl_pct": 0.5 * t1_pnl + 0.5 * t2_pnl,
                    "elapsed_min": mins}

        # Timeout
        if mins >= horizon:
            timeout_pnl = _exit("timeout", c["close"], entry_price, mins)["pnl_pct"]
            if not t1_closed:
                t1_pnl = timeout_pnl
            if not t2_closed:
                t2_pnl = timeout_pnl
            return {"exit_reason": "timeout", "pnl_pct": 0.5 * t1_pnl + 0.5 * t2_pnl,
                    "elapsed_min": mins}

    # Data ends
    last_pnl = _data_end_exit(candles, entry_price, base_ts)["pnl_pct"]
    if not t1_closed:
        t1_pnl = last_pnl
    if not t2_closed:
        t2_pnl = last_pnl
    return {"exit_reason": "timeout_data", "pnl_pct": 0.5 * t1_pnl + 0.5 * t2_pnl,
            "elapsed_min": _elapsed(candles[-1], base_ts)}


# ---------------------------------------------------------------------------
# DYNAMIC_TRAIL: adaptive trailing with 5 sub-modes
# ---------------------------------------------------------------------------

def simulate_dynamic_trail(candles: list[dict], entry_price: float, cfg: dict) -> dict:
    """
    Dynamic trail with configurable behavior via cfg["mode"]:
      - time_decay:       trail interpolates between trail_start and trail_end
                          over the timeout window. Works BOTH directions.
      - gain_adaptive:    trail widens at gain thresholds (configurable)
      - gain_time_hybrid: trail based on gain + tightens near timeout
      - ratchet_trail:    SL locks at milestones + trail from there
      - time_gain_ratchet: ratchet + time-based tightening
    """
    base_ts = candles[0]["timestamp"]
    high_seen = entry_price
    activated = False
    sl_price = entry_price * cfg["sl_mult"]
    horizon = cfg["horizon_min"]
    act_pct = cfg.get("activation_pct", 0.20)
    mode = cfg["mode"]
    base_trail = cfg.get("trail_base", 5) / 100

    for c in candles:
        mins = _elapsed(c, base_ts)
        if c["high"] > high_seen:
            high_seen = c["high"]

        current_gain = (high_seen / entry_price - 1) * 100  # in %
        time_pct = min(mins / horizon, 1.0)  # 0 to 1

        # --- Compute dynamic trail % ---
        if mode == "time_decay":
            # Linear interpolation: trail_start -> trail_end over full horizon
            # Both directions work: (15,3) = wide→tight, (3,15) = tight→wide
            t_start = cfg["trail_start"] / 100
            t_end = cfg["trail_end"] / 100
            trail = t_start + (t_end - t_start) * time_pct

        elif mode == "gain_adaptive":
            # Trail widens at configurable gain thresholds
            thresholds = cfg.get("gain_thresholds", [30, 100, 300])
            trails = cfg.get("gain_trails", [0.05, 0.10, 0.15, 0.20])
            trail = trails[0]
            for i, threshold in enumerate(thresholds):
                if current_gain >= threshold:
                    trail = trails[min(i + 1, len(trails) - 1)]

        elif mode == "gain_time_hybrid":
            # Trail based on gain level, tightened near timeout
            gain_trail = 0.05 + min(current_gain / 100, 3) * 0.05
            time_factor = 1.0 - time_pct * 0.5  # tighten by 50% at timeout
            trail = max(gain_trail * time_factor, 0.03)

        elif mode == "ratchet_trail":
            # SL locks at milestones; trail from each milestone level
            milestones = cfg.get("milestones", [
                (30, 10, 5), (50, 25, 7), (100, 50, 10),
                (200, 120, 15), (400, 250, 20),
            ])
            trail = base_trail
            for gain_level, lock_pct, new_trail_pct in milestones:
                if current_gain >= gain_level:
                    lock_price = entry_price * (1 + lock_pct / 100)
                    if lock_price > sl_price:
                        sl_price = lock_price
                    trail = new_trail_pct / 100

        elif mode == "time_gain_ratchet":
            # Ratchet + tighten trail in last 30% of timeout
            milestones = cfg.get("milestones", [
                (30, 10, 8), (50, 25, 10), (100, 50, 12), (200, 120, 15),
            ])
            trail = base_trail
            for gain_level, lock_pct, new_trail_pct in milestones:
                if current_gain >= gain_level:
                    lock_price = entry_price * (1 + lock_pct / 100)
                    if lock_price > sl_price:
                        sl_price = lock_price
                    trail = new_trail_pct / 100
            # Tighten in last 30%
            if time_pct > 0.7:
                tighten = (time_pct - 0.7) / 0.3
                trail = max(trail * (1 - tighten * 0.5), 0.03)

        else:
            trail = base_trail

        # --- SL check ---
        if c["low"] <= sl_price:
            return _exit("sl_hit", sl_price, entry_price, mins, is_sl=True)

        # --- Trail activation ---
        if not activated and high_seen >= entry_price * (1 + act_pct):
            activated = True

        # --- Trail trigger ---
        if activated:
            trail_trigger = high_seen * (1 - trail)
            if trail_trigger > sl_price and c["low"] <= trail_trigger:
                return _exit("trail_stop", trail_trigger, entry_price, mins)

        # --- Timeout ---
        if mins >= horizon:
            return _timeout_exit(c, entry_price, base_ts)

    return _data_end_exit(candles, entry_price, base_ts)


# ---------------------------------------------------------------------------
# CONTEXTUAL: trail/timeout/activation adapt to token characteristics
# ---------------------------------------------------------------------------

def _get_segment(value: float, breakpoints: list[float]) -> int:
    """Return segment index (0 = below first breakpoint)."""
    for i, bp in enumerate(breakpoints):
        if value < bp:
            return i
    return len(breakpoints)


def simulate_contextual(candles: list[dict], entry_price: float, cfg: dict,
                         context: dict | None = None) -> dict:
    """
    DTRAIL-style but trail/timeout/activation vary by token segment.
    Falls back to middle segment if context is missing.
    """
    ctx = context or {}
    mcap = ctx.get("mcap", 500000)

    # Determine segment from mcap
    breakpoints = cfg["mcap_breakpoints"]
    seg = _get_segment(mcap, breakpoints)
    n_seg = len(breakpoints) + 1

    trail_pct = cfg["trail_per_segment"][min(seg, n_seg - 1)] / 100
    horizon = cfg["timeout_per_segment"][min(seg, n_seg - 1)]
    act_pct = cfg["act_per_segment"][min(seg, n_seg - 1)] / 100
    sl_price = entry_price * cfg["sl_mult"]

    base_ts = candles[0]["timestamp"]
    high_seen = entry_price
    activated = False

    for c in candles:
        mins = _elapsed(c, base_ts)
        if c["high"] > high_seen:
            high_seen = c["high"]
        if c["low"] <= sl_price:
            return _exit("sl_hit", sl_price, entry_price, mins, is_sl=True)
        if not activated and high_seen >= entry_price * (1 + act_pct):
            activated = True
        if activated:
            trail_trigger = high_seen * (1 - trail_pct)
            if trail_trigger > entry_price and c["low"] <= trail_trigger:
                return _exit("trail_stop", trail_trigger, entry_price, mins)
        if mins >= horizon:
            return _timeout_exit(c, entry_price, base_ts)

    return _data_end_exit(candles, entry_price, base_ts)


# ---------------------------------------------------------------------------
# SCALE_OUT: progressive exit in multiple tranches
# ---------------------------------------------------------------------------

def simulate_scale_out(candles: list[dict], entry_price: float, cfg: dict,
                        context: dict | None = None) -> dict:
    """
    Sell in tranches at gain milestones, trail the remainder.
    cfg["tranches"] = [(gain_pct, sell_fraction), ...] sorted ascending
    cfg["runner_trail"] = trail % on remaining after all tranches
    cfg["runner_act"] = activation % for the runner trail
    """
    sl_price = entry_price * cfg["sl_mult"]
    horizon = cfg["horizon_min"]
    tranches = cfg["tranches"]  # [(30, 0.25), (60, 0.25), (100, 0.25)]
    runner_trail = cfg["runner_trail"] / 100
    runner_act = cfg.get("runner_act", 50) / 100
    base_ts = candles[0]["timestamp"]

    remaining = 1.0
    total_pnl = 0.0
    tranche_sold = [False] * len(tranches)
    high_seen = entry_price
    runner_activated = False

    for c in candles:
        mins = _elapsed(c, base_ts)
        if c["high"] > high_seen:
            high_seen = c["high"]

        # SL kills everything remaining
        if c["low"] <= sl_price:
            sl_pnl = _exit("sl_hit", sl_price, entry_price, mins, is_sl=True)["pnl_pct"]
            total_pnl += sl_pnl * remaining
            return {"exit_reason": "sl_hit", "pnl_pct": total_pnl, "elapsed_min": mins}

        # Check tranches
        for i, (gain_pct, sell_frac) in enumerate(tranches):
            if tranche_sold[i]:
                continue
            tp_price = entry_price * (1 + gain_pct / 100)
            if c["high"] >= tp_price:
                tp_pnl = _exit("tp_hit", tp_price, entry_price, mins)["pnl_pct"]
                actual_sell = min(sell_frac, remaining)
                total_pnl += tp_pnl * actual_sell
                remaining -= actual_sell
                tranche_sold[i] = True

        # Runner trail on remaining
        if remaining > 0.001:
            if not runner_activated and high_seen >= entry_price * (1 + runner_act):
                runner_activated = True
            if runner_activated:
                trail_trigger = high_seen * (1 - runner_trail)
                if trail_trigger > entry_price and c["low"] <= trail_trigger:
                    trail_pnl = _exit("trail_stop", trail_trigger, entry_price, mins)["pnl_pct"]
                    total_pnl += trail_pnl * remaining
                    return {"exit_reason": "trail_stop", "pnl_pct": total_pnl,
                            "elapsed_min": mins}
        elif remaining <= 0.001:
            return {"exit_reason": "tp_hit", "pnl_pct": total_pnl, "elapsed_min": mins}

        # Timeout
        if mins >= horizon:
            timeout_pnl = _exit("timeout", c["close"], entry_price, mins)["pnl_pct"]
            total_pnl += timeout_pnl * remaining
            return {"exit_reason": "timeout", "pnl_pct": total_pnl, "elapsed_min": mins}

    # Data ends
    end_pnl = _data_end_exit(candles, entry_price, base_ts)["pnl_pct"]
    total_pnl += end_pnl * remaining
    return {"exit_reason": "timeout_data", "pnl_pct": total_pnl,
            "elapsed_min": _elapsed(candles[-1], base_ts)}


# ---------------------------------------------------------------------------
# DIP_BUY: re-enter after dump + bounce
# ---------------------------------------------------------------------------

def simulate_dip_buy(candles: list[dict], entry_price: float, cfg: dict,
                      context: dict | None = None) -> dict:
    """
    Base DTRAIL strategy + re-entry if price dips then bounces.
    Two positions tracked independently, weighted by size.

    v116: P1 and P2 can have independent trail/act/sl params.
    If p2_trail/p2_act/p2_sl are not set, falls back to shared trail/act/sl.
    """
    # Shared params
    horizon = cfg["horizon_min"]
    dip_threshold = cfg["dip_threshold"] / 100    # e.g. -0.30
    bounce_threshold = cfg["bounce_threshold"] / 100  # e.g. 0.10
    dip_size_mult = cfg["dip_size_mult"]  # e.g. 0.5
    base_ts = candles[0]["timestamp"]

    # P1 params (fall back to shared)
    p1_sl_pct = cfg.get("p1_sl", cfg["sl"])
    p1_trail_pct = cfg.get("p1_trail", cfg["trail"]) / 100
    p1_act_pct = cfg.get("p1_act", cfg["act"]) / 100

    # P2 params (fall back to shared)
    p2_sl_pct = cfg.get("p2_sl", cfg["sl"])
    p2_trail_pct = cfg.get("p2_trail", cfg["trail"]) / 100
    p2_act_pct = cfg.get("p2_act", cfg["act"]) / 100

    # Position 1: original
    p1_entry = entry_price
    p1_sl = p1_entry * (1 - p1_sl_pct / 100)
    p1_high = p1_entry
    p1_activated = False
    p1_weight = 1.0 / (1.0 + dip_size_mult)  # normalize weights
    p1_closed = False
    p1_pnl = 0.0

    # Position 2: dip buy (not yet opened)
    p2_opened = False
    p2_entry = 0.0
    p2_sl = 0.0
    p2_high = 0.0
    p2_activated = False
    p2_weight = dip_size_mult / (1.0 + dip_size_mult)
    p2_closed = False
    p2_pnl = 0.0

    # Dip tracking
    low_since_entry = entry_price
    dip_triggered = False
    reentry_done = False

    for c in candles:
        mins = _elapsed(c, base_ts)

        # Track lows for dip detection
        if c["low"] < low_since_entry:
            low_since_entry = c["low"]

        # --- Position 1 logic ---
        if not p1_closed:
            if c["high"] > p1_high:
                p1_high = c["high"]
            if c["low"] <= p1_sl:
                p1_pnl = _exit("sl_hit", p1_sl, p1_entry, mins, is_sl=True)["pnl_pct"]
                p1_closed = True
            elif not p1_activated and p1_high >= p1_entry * (1 + p1_act_pct):
                p1_activated = True
            if not p1_closed and p1_activated:
                trail = p1_high * (1 - p1_trail_pct)
                if trail > p1_entry and c["low"] <= trail:
                    p1_pnl = _exit("trail_stop", trail, p1_entry, mins)["pnl_pct"]
                    p1_closed = True

        # --- Dip buy detection ---
        if not reentry_done and not p1_closed:
            dip_level = entry_price * (1 + dip_threshold)  # e.g. entry * 0.70
            if low_since_entry <= dip_level:
                dip_triggered = True
            if dip_triggered:
                if bounce_threshold <= 0:
                    # DIRECT dip buy: enter at the dip level immediately
                    p2_opened = True
                    reentry_done = True
                    p2_entry = dip_level * (1 + BUY_SLIPPAGE)  # buy slippage (1.5%)
                    p2_sl = p2_entry * (1 - p2_sl_pct / 100)
                    p2_high = c["high"]
                else:
                    bounce_from_low = (c["high"] / low_since_entry - 1)
                    if bounce_from_low >= bounce_threshold:
                        # Open position 2 on confirmed bounce
                        p2_opened = True
                        reentry_done = True
                        p2_entry = low_since_entry * (1 + bounce_threshold) * (1 + BUY_SLIPPAGE)
                        p2_sl = p2_entry * (1 - p2_sl_pct / 100)
                        p2_high = c["high"]

        # --- Position 2 logic ---
        if p2_opened and not p2_closed:
            if c["high"] > p2_high:
                p2_high = c["high"]
            if c["low"] <= p2_sl:
                p2_pnl = _exit("sl_hit", p2_sl, p2_entry, mins, is_sl=True)["pnl_pct"]
                p2_closed = True
            elif not p2_activated and p2_high >= p2_entry * (1 + p2_act_pct):
                p2_activated = True
            if not p2_closed and p2_activated:
                trail = p2_high * (1 - p2_trail_pct)
                if trail > p2_entry and c["low"] <= trail:
                    p2_pnl = _exit("trail_stop", trail, p2_entry, mins)["pnl_pct"]
                    p2_closed = True

        # --- Both closed? ---
        both_done = p1_closed and (p2_closed or not p2_opened)
        if both_done:
            combined = p1_pnl * p1_weight
            if p2_opened:
                combined += p2_pnl * p2_weight
            else:
                combined = p1_pnl  # no p2, full weight to p1
            return {"exit_reason": "trail_stop", "pnl_pct": combined, "elapsed_min": mins}

        # --- Timeout ---
        if mins >= horizon:
            timeout_pnl_1 = _exit("timeout", c["close"], p1_entry, mins)["pnl_pct"] \
                if not p1_closed else p1_pnl
            if p2_opened:
                timeout_pnl_2 = _exit("timeout", c["close"], p2_entry, mins)["pnl_pct"] \
                    if not p2_closed else p2_pnl
                combined = timeout_pnl_1 * p1_weight + timeout_pnl_2 * p2_weight
            else:
                combined = timeout_pnl_1
            return {"exit_reason": "timeout", "pnl_pct": combined, "elapsed_min": mins}

    # Data ends
    last_close = candles[-1]["close"]
    last_mins = _elapsed(candles[-1], base_ts)
    end_pnl_1 = (last_close * (1 - SLIPPAGE_TRAIL) / p1_entry - 1) if not p1_closed else p1_pnl
    if p2_opened:
        end_pnl_2 = (last_close * (1 - SLIPPAGE_TRAIL) / p2_entry - 1) if not p2_closed else p2_pnl
        combined = end_pnl_1 * p1_weight + end_pnl_2 * p2_weight
    else:
        combined = end_pnl_1
    return {"exit_reason": "timeout_data", "pnl_pct": combined, "elapsed_min": last_mins}


# ---------------------------------------------------------------------------
# DIP_SCALE_OUT: re-enter on dip + sell in tranches
# ---------------------------------------------------------------------------

def simulate_dip_scale_out(candles: list[dict], entry_price: float, cfg: dict,
                            context: dict | None = None) -> dict:
    """
    Hybrid: DIP_BUY logic for positions + SCALE_OUT logic for exits.
    Both positions (original + dip re-entry) use scale-out tranches.
    """
    sl_pct = cfg["sl"]
    trail_pct = cfg["trail"] / 100
    act_pct = cfg["act"] / 100
    horizon = cfg["horizon_min"]
    dip_threshold = cfg["dip_threshold"] / 100
    bounce_threshold = cfg["bounce_threshold"] / 100
    dip_size_mult = cfg["dip_size_mult"]
    tranches = cfg["tranches"]
    runner_trail = cfg["runner_trail"] / 100
    runner_act = cfg.get("runner_act", 50) / 100
    base_ts = candles[0]["timestamp"]

    # Weights
    p1_weight = 1.0 / (1.0 + dip_size_mult)
    p2_weight = dip_size_mult / (1.0 + dip_size_mult)

    # Position 1: scale-out exit
    p1_entry = entry_price
    p1_sl = p1_entry * (1 - sl_pct / 100)
    p1_remaining = 1.0
    p1_pnl = 0.0
    p1_tranche_sold = [False] * len(tranches)
    p1_high = p1_entry
    p1_runner_active = False
    p1_closed = False

    # Position 2: dip buy (not opened yet)
    p2_opened = False
    p2_entry = 0.0
    p2_sl = 0.0
    p2_remaining = 1.0
    p2_pnl = 0.0
    p2_tranche_sold = [False] * len(tranches)
    p2_high = 0.0
    p2_runner_active = False
    p2_closed = False

    low_since_entry = entry_price
    dip_triggered = False
    reentry_done = False

    for c in candles:
        mins = _elapsed(c, base_ts)
        if c["low"] < low_since_entry:
            low_since_entry = c["low"]

        # === Position 1 ===
        if not p1_closed:
            if c["high"] > p1_high:
                p1_high = c["high"]
            # SL kills remaining
            if c["low"] <= p1_sl:
                sl_pnl = _exit("sl_hit", p1_sl, p1_entry, mins, is_sl=True)["pnl_pct"]
                p1_pnl += sl_pnl * p1_remaining
                p1_closed = True
            else:
                # Tranches
                for i, (gain_pct, sell_frac) in enumerate(tranches):
                    if p1_tranche_sold[i]:
                        continue
                    tp = p1_entry * (1 + gain_pct / 100)
                    if c["high"] >= tp:
                        tp_pnl = _exit("tp_hit", tp, p1_entry, mins)["pnl_pct"]
                        actual = min(sell_frac, p1_remaining)
                        p1_pnl += tp_pnl * actual
                        p1_remaining -= actual
                        p1_tranche_sold[i] = True
                # Runner trail
                if p1_remaining > 0.001:
                    if not p1_runner_active and p1_high >= p1_entry * (1 + runner_act):
                        p1_runner_active = True
                    if p1_runner_active:
                        trigger = p1_high * (1 - runner_trail)
                        if trigger > p1_entry and c["low"] <= trigger:
                            trail_pnl_val = _exit("trail_stop", trigger, p1_entry, mins)["pnl_pct"]
                            p1_pnl += trail_pnl_val * p1_remaining
                            p1_remaining = 0
                            p1_closed = True
                elif p1_remaining <= 0.001:
                    p1_closed = True

        # === Dip detection ===
        if not reentry_done and not p1_closed:
            dip_level = entry_price * (1 + dip_threshold)
            if low_since_entry <= dip_level:
                dip_triggered = True
            if dip_triggered:
                if bounce_threshold <= 0:
                    p2_opened = True
                    reentry_done = True
                    p2_entry = dip_level * (1 + BUY_SLIPPAGE)  # buy slippage (1.5%)
                    p2_sl = p2_entry * (1 - sl_pct / 100)
                    p2_high = c["high"]
                else:
                    bounce = (c["high"] / low_since_entry - 1)
                    if bounce >= bounce_threshold:
                        p2_opened = True
                        reentry_done = True
                        p2_entry = low_since_entry * (1 + bounce_threshold) * (1 + BUY_SLIPPAGE)
                        p2_sl = p2_entry * (1 - sl_pct / 100)
                        p2_high = c["high"]

        # === Position 2 (scale-out exit) ===
        if p2_opened and not p2_closed:
            if c["high"] > p2_high:
                p2_high = c["high"]
            if c["low"] <= p2_sl:
                sl_pnl = _exit("sl_hit", p2_sl, p2_entry, mins, is_sl=True)["pnl_pct"]
                p2_pnl += sl_pnl * p2_remaining
                p2_closed = True
            else:
                for i, (gain_pct, sell_frac) in enumerate(tranches):
                    if p2_tranche_sold[i]:
                        continue
                    tp = p2_entry * (1 + gain_pct / 100)
                    if c["high"] >= tp:
                        tp_pnl = _exit("tp_hit", tp, p2_entry, mins)["pnl_pct"]
                        actual = min(sell_frac, p2_remaining)
                        p2_pnl += tp_pnl * actual
                        p2_remaining -= actual
                        p2_tranche_sold[i] = True
                if p2_remaining > 0.001:
                    if not p2_runner_active and p2_high >= p2_entry * (1 + runner_act):
                        p2_runner_active = True
                    if p2_runner_active:
                        trigger = p2_high * (1 - runner_trail)
                        if trigger > p2_entry and c["low"] <= trigger:
                            trail_pnl_val = _exit("trail_stop", trigger, p2_entry, mins)["pnl_pct"]
                            p2_pnl += trail_pnl_val * p2_remaining
                            p2_remaining = 0
                            p2_closed = True
                elif p2_remaining <= 0.001:
                    p2_closed = True

        # Both done
        if p1_closed and (p2_closed or not p2_opened):
            if p2_opened:
                combined = p1_pnl * p1_weight + p2_pnl * p2_weight
            else:
                combined = p1_pnl
            return {"exit_reason": "trail_stop", "pnl_pct": combined, "elapsed_min": mins}

        # Timeout
        if mins >= horizon:
            if not p1_closed and p1_remaining > 0:
                to_pnl = _exit("timeout", c["close"], p1_entry, mins)["pnl_pct"]
                p1_pnl += to_pnl * p1_remaining
            if p2_opened and not p2_closed and p2_remaining > 0:
                to_pnl = _exit("timeout", c["close"], p2_entry, mins)["pnl_pct"]
                p2_pnl += to_pnl * p2_remaining
            if p2_opened:
                combined = p1_pnl * p1_weight + p2_pnl * p2_weight
            else:
                combined = p1_pnl
            return {"exit_reason": "timeout", "pnl_pct": combined, "elapsed_min": mins}

    # Data ends
    last = candles[-1]
    last_mins = _elapsed(last, base_ts)
    if not p1_closed and p1_remaining > 0:
        p1_pnl += (last["close"] * (1 - SLIPPAGE_TRAIL) / p1_entry - 1) * p1_remaining
    if p2_opened and not p2_closed and p2_remaining > 0:
        p2_pnl += (last["close"] * (1 - SLIPPAGE_TRAIL) / p2_entry - 1) * p2_remaining
    if p2_opened:
        combined = p1_pnl * p1_weight + p2_pnl * p2_weight
    else:
        combined = p1_pnl
    return {"exit_reason": "timeout_data", "pnl_pct": combined, "elapsed_min": last_mins}


# ---------------------------------------------------------------------------
# v125: Unified simulation — uses production _evaluate_trade_exit()
# ---------------------------------------------------------------------------

def candles_to_synthetic_ticks(candles: list[dict], base_ts: int) -> list[dict]:
    """Convert 15-min OHLCV candles into synthetic ticks for _evaluate_trade_exit().

    For each candle, generates 4 price points:
      - If close >= open (bullish): open → low → high → close
      - If close < open (bearish): open → high → low → close
    Timestamps spaced 225s apart within each candle's 15-min interval.

    Returns list of dicts: {"price_usd": float, "fetched_at": str (ISO)}
    """
    from datetime import datetime, timezone
    ticks = []
    for c in candles:
        ts = c["timestamp"]
        o, h, l, cl = c["open"], c["high"], c["low"], c["close"]
        if cl >= o:
            prices = [o, l, h, cl]
        else:
            prices = [o, h, l, cl]
        for i, price in enumerate(prices):
            tick_ts = ts + i * 225  # 4 ticks × 225s = 900s = 15min
            dt = datetime.fromtimestamp(tick_ts, tz=timezone.utc)
            ticks.append({
                "price_usd": price,
                "fetched_at": dt.isoformat(),
            })
    return ticks


def simulate_unified(candles: list[dict], entry_price: float, cfg: dict,
                     context: dict | None = None) -> dict:
    """OHLCV simulation using production _evaluate_trade_exit().

    Converts candles to synthetic ticks, builds a fake trade dict from cfg,
    then replays through _evaluate_trade_exit() — identical logic to tick sim
    and live trading.

    Slippage: 10bps base (Jupiter Ultra) + dynamic per exit type/liquidity.
    Crash detection: automatic (trail_crash if >30% below trigger).
    """
    from datetime import datetime, timezone
    from strategies import sim_cfg_to_fake_trade
    from paper_trader import _evaluate_trade_exit

    if not candles:
        return {"exit_reason": "no_data", "pnl_pct": 0, "elapsed_min": 0}

    base_ts = candles[0]["timestamp"]
    ticks = candles_to_synthetic_ticks(candles, base_ts)

    # Build fake trade from sim config
    liq = (context or {}).get("liq", 50_000)
    created_dt = datetime.fromtimestamp(base_ts, tz=timezone.utc)
    created_at = created_dt.isoformat()

    fake_trade = sim_cfg_to_fake_trade(cfg, entry_price, created_at,
                                       liquidity_usd=liq)

    # 10bps base slippage (Jupiter Ultra RFQ)
    sell_slip = 1 - 10 / 10_000

    for tick in ticks:
        tick_time = datetime.fromisoformat(
            tick["fetched_at"].replace("Z", "+00:00"))
        tick_price = tick["price_usd"]
        if tick_price <= 0:
            continue

        ev = _evaluate_trade_exit(fake_trade, tick_price, tick_time, sell_slip)
        if ev is None:
            continue

        # Always update high_price_seen
        if ev.get("high_price_seen") is not None:
            new_high = ev["high_price_seen"]
            if new_high > float(fake_trade.get("high_price_seen") or 0):
                fake_trade["high_price_seen"] = new_high

        # Exit triggered
        if "status" in ev and ev["status"] is not None:
            return {
                "exit_reason": ev["status"],
                "pnl_pct": ev.get("pnl_pct", 0),
                "elapsed_min": ev.get("exit_minutes", 0),
            }

    # No exit — timeout at last candle
    last_ts = candles[-1]["timestamp"]
    elapsed = (last_ts - base_ts) / 60.0
    last_price = candles[-1]["close"]
    slip = SLIPPAGE_TRAIL  # fallback for data-end exit
    net_price = last_price * (1 - slip)
    return {
        "exit_reason": "data_end",
        "pnl_pct": net_price / entry_price - 1,
        "elapsed_min": elapsed,
    }


def simulate_unified_multi(candles: list[dict], entry_price: float, cfg: dict,
                           context: dict | None = None) -> dict:
    """Unified sim for SPLIT strategies (2 tranches).

    Creates 2 fake trades, replays both through _evaluate_trade_exit().
    SL cascade: if one tranche SL hits, force-close the other.
    """
    from datetime import datetime, timezone
    from strategies import sim_cfg_to_fake_trade, STRATEGIES
    from paper_trader import _evaluate_trade_exit

    if not candles:
        return {"exit_reason": "no_data", "pnl_pct": 0, "elapsed_min": 0}

    strat_name = cfg.get("name", "")
    tranches = STRATEGIES.get(strat_name, [])
    if len(tranches) < 2:
        return simulate_unified(candles, entry_price, cfg, context)

    base_ts = candles[0]["timestamp"]
    ticks = candles_to_synthetic_ticks(candles, base_ts)
    liq = (context or {}).get("liq", 50_000)
    created_dt = datetime.fromtimestamp(base_ts, tz=timezone.utc)
    created_at = created_dt.isoformat()
    sell_slip = 1 - 10 / 10_000

    # Build fake trades per tranche
    fake_trades = []
    for i, tr in enumerate(tranches):
        tr_cfg = {
            "name": strat_name,
            "tp_mult": tr.get("tp_mult"),
            "sl_mult": tr.get("sl_mult", 0.50),
            "horizon": tr.get("horizon_min", 120),
            "tranche_label": tr.get("label", f"t{i}"),
        }
        ft = sim_cfg_to_fake_trade(tr_cfg, entry_price, created_at,
                                    liquidity_usd=liq,
                                    trade_id=f"sim_{i}")
        fake_trades.append((ft, tr.get("pct", 0.5), False))  # (trade, weight, closed)

    results = [None] * len(tranches)

    for tick in ticks:
        tick_time = datetime.fromisoformat(
            tick["fetched_at"].replace("Z", "+00:00"))
        tick_price = tick["price_usd"]
        if tick_price <= 0:
            continue

        # Check SL cascade from previous tick
        sl_cascade = any(
            results[j] is not None and results[j]["exit_reason"] == "sl_hit"
            for j in range(len(tranches))
        )

        for i, (ft, weight, closed) in enumerate(fake_trades):
            if closed:
                continue

            ev = _evaluate_trade_exit(ft, tick_price, tick_time, sell_slip,
                                      sl_cascade=sl_cascade)
            if ev is None:
                continue

            if ev.get("high_price_seen") is not None:
                new_high = ev["high_price_seen"]
                if new_high > float(ft.get("high_price_seen") or 0):
                    ft["high_price_seen"] = new_high

            if "status" in ev and ev["status"] is not None:
                results[i] = {
                    "exit_reason": ev["status"],
                    "pnl_pct": ev.get("pnl_pct", 0),
                    "elapsed_min": ev.get("exit_minutes", 0),
                }
                fake_trades[i] = (ft, weight, True)

        if all(closed for _, _, closed in fake_trades):
            break

    # Compute weighted PnL
    total_pnl = 0
    total_weight = 0
    last_reason = "data_end"
    last_elapsed = 0
    for i, (ft, weight, closed) in enumerate(fake_trades):
        if results[i] is not None:
            total_pnl += results[i]["pnl_pct"] * weight
            total_weight += weight
            last_reason = results[i]["exit_reason"]
            last_elapsed = max(last_elapsed, results[i]["elapsed_min"])
        else:
            # Unclosed tranche — timeout at data end
            last_price = candles[-1]["close"]
            pnl = last_price / entry_price - 1
            total_pnl += pnl * weight
            total_weight += weight
            last_elapsed = max(last_elapsed, (candles[-1]["timestamp"] - base_ts) / 60.0)

    return {
        "exit_reason": last_reason,
        "pnl_pct": total_pnl,
        "elapsed_min": last_elapsed,
    }


def simulate_unified_dip_buy(candles: list[dict], entry_price: float, cfg: dict,
                              context: dict | None = None) -> dict:
    """Unified sim for DIP_BUY strategies.

    P1 opens immediately. Watches for dip + bounce to open P2.
    Both positions evaluated independently through _evaluate_trade_exit().
    """
    from datetime import datetime, timezone
    from strategies import sim_cfg_to_fake_trade
    from paper_trader import _evaluate_trade_exit

    if not candles:
        return {"exit_reason": "no_data", "pnl_pct": 0, "elapsed_min": 0}

    base_ts = candles[0]["timestamp"]
    ticks = candles_to_synthetic_ticks(candles, base_ts)
    liq = (context or {}).get("liq", 50_000)
    created_dt = datetime.fromtimestamp(base_ts, tz=timezone.utc)
    created_at = created_dt.isoformat()
    sell_slip = 1 - 10 / 10_000

    # DIP params from cfg
    dip_pct = cfg.get("dip_pct", 0.30)
    bounce_pct = cfg.get("bounce_pct", 0.05)
    dip_size_mult = cfg.get("dip_size_mult", 1.0)

    # P1 opens immediately
    p1_cfg = dict(cfg)
    p1_cfg["tranche_label"] = "dip_p1"
    p1_trade = sim_cfg_to_fake_trade(p1_cfg, entry_price, created_at,
                                      liquidity_usd=liq, trade_id="sim_p1")

    # P2 state
    p2_trade = None
    p2_open = False
    low_since_entry = entry_price
    dip_triggered = False

    p1_result = None
    p2_result = None
    p1_weight = 1.0 / (1.0 + dip_size_mult)
    p2_weight = dip_size_mult / (1.0 + dip_size_mult)

    for tick in ticks:
        tick_time = datetime.fromisoformat(
            tick["fetched_at"].replace("Z", "+00:00"))
        tick_price = tick["price_usd"]
        if tick_price <= 0:
            continue

        # Track low for dip detection
        if tick_price < low_since_entry:
            low_since_entry = tick_price

        # Check dip + bounce for P2 entry
        if not p2_open and not dip_triggered:
            if low_since_entry <= entry_price * (1 - dip_pct):
                dip_triggered = True

        if dip_triggered and not p2_open:
            if bounce_pct <= 0:
                # Direct re-entry at dip level
                p2_entry = low_since_entry * (1 + BUY_SLIPPAGE)
                p2_cfg = dict(cfg)
                p2_cfg["tranche_label"] = "dip_p2"
                p2_trade = sim_cfg_to_fake_trade(p2_cfg, p2_entry,
                                                  tick["fetched_at"],
                                                  liquidity_usd=liq,
                                                  trade_id="sim_p2")
                p2_open = True
            elif tick_price / low_since_entry - 1 >= bounce_pct:
                p2_entry = low_since_entry * (1 + bounce_pct) * (1 + BUY_SLIPPAGE)
                p2_cfg = dict(cfg)
                p2_cfg["tranche_label"] = "dip_p2"
                p2_trade = sim_cfg_to_fake_trade(p2_cfg, p2_entry,
                                                  tick["fetched_at"],
                                                  liquidity_usd=liq,
                                                  trade_id="sim_p2")
                p2_open = True

        # Evaluate P1
        if p1_result is None:
            ev = _evaluate_trade_exit(p1_trade, tick_price, tick_time, sell_slip)
            if ev is not None:
                if ev.get("high_price_seen") is not None:
                    new_h = ev["high_price_seen"]
                    if new_h > float(p1_trade.get("high_price_seen") or 0):
                        p1_trade["high_price_seen"] = new_h
                if "status" in ev and ev["status"] is not None:
                    p1_result = {"exit_reason": ev["status"],
                                 "pnl_pct": ev.get("pnl_pct", 0),
                                 "elapsed_min": ev.get("exit_minutes", 0)}

        # Evaluate P2
        if p2_open and p2_result is None and p2_trade is not None:
            ev = _evaluate_trade_exit(p2_trade, tick_price, tick_time, sell_slip)
            if ev is not None:
                if ev.get("high_price_seen") is not None:
                    new_h = ev["high_price_seen"]
                    if new_h > float(p2_trade.get("high_price_seen") or 0):
                        p2_trade["high_price_seen"] = new_h
                if "status" in ev and ev["status"] is not None:
                    p2_result = {"exit_reason": ev["status"],
                                 "pnl_pct": ev.get("pnl_pct", 0),
                                 "elapsed_min": ev.get("exit_minutes", 0)}

        # Both done
        if p1_result is not None and (p2_result is not None or not p2_open):
            break

    # Compute weighted result
    total_pnl = 0
    last_reason = "data_end"
    last_elapsed = 0

    if p1_result:
        total_pnl += p1_result["pnl_pct"] * p1_weight
        last_reason = p1_result["exit_reason"]
        last_elapsed = p1_result["elapsed_min"]
    else:
        last_price = candles[-1]["close"]
        total_pnl += (last_price / entry_price - 1) * p1_weight
        last_elapsed = (candles[-1]["timestamp"] - base_ts) / 60.0

    if p2_open and p2_result:
        total_pnl += p2_result["pnl_pct"] * p2_weight
        last_elapsed = max(last_elapsed, p2_result["elapsed_min"])
    elif p2_open and p2_trade:
        last_price = candles[-1]["close"]
        p2_entry = float(p2_trade["entry_price"])
        total_pnl += (last_price / p2_entry - 1) * p2_weight
    else:
        # P2 never opened — P1 gets full weight
        if p1_result:
            total_pnl = p1_result["pnl_pct"]
        else:
            last_price = candles[-1]["close"]
            total_pnl = last_price / entry_price - 1

    return {
        "exit_reason": last_reason,
        "pnl_pct": total_pnl,
        "elapsed_min": last_elapsed,
    }


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

# Types that use unified path (production exit logic) by default
_UNIFIED_TYPES = {"FIXED", "SCALP", "DTRAIL", "TRAIL", "BE", "DECAY"}

def simulate(candles: list[dict], entry_price: float, cfg: dict,
             context: dict | None = None, unified: bool = True) -> dict:
    """Route to the correct simulation engine based on cfg['type'].

    unified=True (default): use production _evaluate_trade_exit() for types
    deployed in paper/live trading. unified=False: use legacy engines (for comparison).
    """
    global _sim_liquidity_usd, _sim_chain, _sim_position_usd
    ctx = context or {}
    _sim_liquidity_usd = ctx.get("liq", 0)
    # v14e.28: chain + position propagate from sim caller → _exit → paper_trader
    # _dynamic_sell_slip_factor so EVM trades get gas-dominated slip in sim too.
    _sim_chain = ctx.get("chain") or "solana"
    _sim_position_usd = float(ctx.get("position_usd") or 0)
    t = cfg["type"]

    # Unified path for production strategy types
    if unified:
        if t in _UNIFIED_TYPES:
            return simulate_unified(candles, entry_price, cfg, context)
        if t == "SPLIT":
            return simulate_unified_multi(candles, entry_price, cfg, context)
        if t == "DIP_BUY":
            return simulate_unified_dip_buy(candles, entry_price, cfg, context)

    # Legacy path (sim-only experimental types, or --legacy-sim mode)
    if t in ("FIXED", "SCALP"):
        return simulate_fixed(candles, entry_price, cfg)
    elif t == "DTRAIL":
        return simulate_dtrail(candles, entry_price, cfg)
    elif t == "TRAIL":
        return simulate_trail_tp(candles, entry_price, cfg)
    elif t == "BE":
        return simulate_breakeven(candles, entry_price, cfg)
    elif t == "DECAY":
        return simulate_decay(candles, entry_price, cfg)
    elif t == "SPLIT":
        return simulate_split(candles, entry_price, cfg)
    elif t == "DYNAMIC_TRAIL":
        return simulate_dynamic_trail(candles, entry_price, cfg)
    elif t == "CONTEXTUAL":
        return simulate_contextual(candles, entry_price, cfg, context)
    elif t == "SCALE_OUT":
        return simulate_scale_out(candles, entry_price, cfg, context)
    elif t == "DIP_BUY":
        return simulate_dip_buy(candles, entry_price, cfg, context)
    elif t == "DIP_SCALE_OUT":
        return simulate_dip_scale_out(candles, entry_price, cfg, context)
    else:
        raise ValueError(f"Unknown strategy type: {t}")
