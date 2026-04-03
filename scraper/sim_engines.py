"""
Strategy simulation engines — pure functions, zero external dependencies.

Each engine takes (candles, entry_price, cfg) and returns:
    {"exit_reason": str, "pnl_pct": float, "elapsed_min": float}

Slippage constants are applied at exit.
"""

# ---------------------------------------------------------------------------
# Slippage constants
# ---------------------------------------------------------------------------
SLIPPAGE_TRAIL = 0.025   # 2.5% for trail/timeout/TP exits (matches live: 2% sell + 0.5% fee)
SLIPPAGE_SL = 0.025      # 2.5% for SL exits (v113: was 3.5% — paper_trader uses same slippage for all exits)
BUY_SLIPPAGE = 0.015     # 1.5% for DIP_BUY re-entries (matches live: 1% buy + 0.5% fee)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exit(reason: str, exit_price: float, entry_price: float,
          elapsed_min: float, is_sl: bool = False) -> dict:
    slip = SLIPPAGE_SL if is_sl else SLIPPAGE_TRAIL
    net_price = exit_price * (1 - slip)
    return {
        "exit_reason": reason,
        "pnl_pct": net_price / entry_price - 1,
        "elapsed_min": elapsed_min,
    }


def _elapsed(candle: dict, base_ts: int) -> float:
    return (candle["timestamp"] - base_ts) / 60.0


def resample_to_live_checks(candles: list[dict], interval_min: int = 3) -> list[dict]:
    """
    Resample candles to simulate live price checks exactly like paper_trader.py:
    - First 30 min: check every 30 seconds (check_paper_trades_fast)
    - After 30 min: check every `interval_min` minutes (check_paper_trades)

    Uses CLOSE price only (like live uses current_price from DexScreener).
    Sets high=low=close to mimic spot-only checks.
    """
    if not candles:
        return candles
    base_ts = candles[0]["timestamp"]
    fast_cutoff = base_ts + 30 * 60  # 30 min fast check window
    fast_interval = 30  # 30 seconds
    normal_interval = interval_min * 60

    sampled = []
    next_check_ts = base_ts

    for c in candles:
        if c["timestamp"] >= next_check_ts:
            sampled.append({
                "timestamp": c["timestamp"],
                "open": c["close"],
                "high": c["close"],
                "low": c["close"],
                "close": c["close"],
                "volume": c.get("volume", 0),
            })
            # Fast checks for first 30 min, normal after
            if c["timestamp"] < fast_cutoff:
                next_check_ts = c["timestamp"] + fast_interval
            else:
                next_check_ts = c["timestamp"] + normal_interval

    if not sampled:
        c = candles[0]
        sampled.append({
            "timestamp": c["timestamp"], "open": c["close"],
            "high": c["close"], "low": c["close"], "close": c["close"],
            "volume": c.get("volume", 0),
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
    """
    sl_pct = cfg["sl"]
    trail_pct = cfg["trail"] / 100
    act_pct = cfg["act"] / 100
    horizon = cfg["horizon_min"]
    dip_threshold = cfg["dip_threshold"] / 100    # e.g. -0.30
    bounce_threshold = cfg["bounce_threshold"] / 100  # e.g. 0.10
    dip_size_mult = cfg["dip_size_mult"]  # e.g. 0.5
    base_ts = candles[0]["timestamp"]

    # Position 1: original
    p1_entry = entry_price
    p1_sl = p1_entry * (1 - sl_pct / 100)
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
            elif not p1_activated and p1_high >= p1_entry * (1 + act_pct):
                p1_activated = True
            if not p1_closed and p1_activated:
                trail = p1_high * (1 - trail_pct)
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
                    p2_sl = p2_entry * (1 - sl_pct / 100)
                    p2_high = c["high"]
                else:
                    bounce_from_low = (c["high"] / low_since_entry - 1)
                    if bounce_from_low >= bounce_threshold:
                        # Open position 2 on confirmed bounce
                        p2_opened = True
                        reentry_done = True
                        p2_entry = low_since_entry * (1 + bounce_threshold) * (1 + BUY_SLIPPAGE)
                        p2_sl = p2_entry * (1 - sl_pct / 100)
                        p2_high = c["high"]

        # --- Position 2 logic ---
        if p2_opened and not p2_closed:
            if c["high"] > p2_high:
                p2_high = c["high"]
            if c["low"] <= p2_sl:
                p2_pnl = _exit("sl_hit", p2_sl, p2_entry, mins, is_sl=True)["pnl_pct"]
                p2_closed = True
            elif not p2_activated and p2_high >= p2_entry * (1 + act_pct):
                p2_activated = True
            if not p2_closed and p2_activated:
                trail = p2_high * (1 - trail_pct)
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
# Router
# ---------------------------------------------------------------------------

def simulate(candles: list[dict], entry_price: float, cfg: dict,
             context: dict | None = None) -> dict:
    """Route to the correct simulation engine based on cfg['type']."""
    t = cfg["type"]
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
