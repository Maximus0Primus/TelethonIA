"""
Price Action Analysis — compute price-based multipliers from OHLCV + DexScreener data.

Core principle from MemecoinGuide: "Buy the dip, NOT the pump"
- 2.5-3x below ATH with volume = ideal setup
- Never chase a pump
- Don't catch a falling knife

Uses pandas-ta (RSI, MACD, BBands, OBV) when available, with per-submultiplier
fallback to raw DexScreener heuristics when OHLCV data or pandas-ta is missing.

Returns a price_action_multiplier in [pa_norm_floor, pa_norm_cap] applied to the final score.
Default bounds: [0.4, 1.3]. Overridable via pa_norm_floor/pa_norm_cap kwargs.

v46: All thresholds/factors configurable via pa_config JSONB from scoring_config.
"""

import logging

import pandas as pd

from enrich_birdeye_ohlcv import count_support_touches

logger = logging.getLogger(__name__)

# Graceful pandas-ta import
try:
    import pandas_ta as ta
    _HAS_PANDAS_TA = True
except ImportError:
    _HAS_PANDAS_TA = False
    logger.info("pandas-ta not installed — using raw heuristic fallbacks for price action")


# v46: Default PA config — matches all pre-v46 hardcoded behavior exactly.
_DEFAULT_PA_CONFIG = {
    # Position vs ATH (ascending thresholds for _tier_lookup)
    "pos_thresholds": [0.10, 0.20, 0.50, 0.70, 0.90],
    "pos_factors": [0.4, 0.8, 1.3, 0.9, 0.6, 0.4],

    # RSI-based direction classification
    "rsi_hard_pump": 80, "rsi_pump": 70,
    "rsi_freefall": 20, "rsi_dying": 30,
    "rsi_bounce_upper": 40, "rsi_plateau_lower": 40, "rsi_plateau_upper": 60,
    "rsi_strong_bounce_slope": 3, "rsi_plateau_slope_max": 3,

    # Direction multipliers (v27: all penalties set to 1.0, now configurable via Optuna)
    "dir_hard_pump_mult": 1.0,
    "dir_pump_mult": 1.0,
    "dir_freefall_mult": 1.0,
    "dir_dying_mult": 1.0,
    "dir_strong_bounce_mult": 1.4,
    "dir_bounce_mult": 1.3,
    "dir_plateau_mult": 1.1,
    "dir_macd_bonus": 0.1,
    "dir_cap": 1.5,

    # Fallback direction thresholds (DexScreener heuristics)
    "fb_hard_pump_1h": 50, "fb_hard_pump_5m": 25, "fb_hard_pump_mult": 1.0,
    "fb_pump_1h": 20, "fb_pump_5m": 10, "fb_pump_mult": 1.0,
    "fb_dying_1h": -10, "fb_dying_6h": -30, "fb_dying_mult": 1.0,
    "fb_freefall_1h": -20, "fb_freefall_5m": -10, "fb_freefall_mult": 1.0,
    "fb_plateau_1h": 5, "fb_plateau_6h": 10, "fb_plateau_mult": 1.1,
    "fb_bounce_6h": -15, "fb_bounce_1h": 5, "fb_bounce_mult": 1.3,
    "fb_strong_bounce_6h": -30, "fb_strong_bounce_1h": 10, "fb_strong_bounce_mult": 1.4,
    "fb_bsr_bounce_bonus": 0.1, "fb_bsr_bounce_threshold": 0.6,
    "fb_bsr_dead_cat_penalty": 0.2, "fb_bsr_dead_cat_threshold": 0.4,

    # pc24h fallback direction
    "pc24_freefall": -60, "pc24_freefall_mult": 1.0,
    "pc24_dying": -40, "pc24_dying_mult": 1.0,
    "pc24_bleeding": -20, "pc24_bleeding_mult": 1.0,
    "pc24_pumping_hard": 100, "pc24_pumping_hard_mult": 1.0,
    "pc24_pumping": 50, "pc24_pumping_mult": 1.0,

    # pc24h position fallback
    "pc24_pos_crash70_mult": 0.4, "pc24_pos_crash50_mult": 0.6, "pc24_pos_crash30_mult": 0.8,
    "pc24_pos_pump200_mult": 0.5, "pc24_pos_pump100_mult": 0.6,

    # Volume confirmation (OBV path)
    "obv_strong_accum": 2.0, "obv_strong_mult": 1.3,
    "obv_mod_accum": 0.5, "obv_mod_mult": 1.2,
    "obv_dead_cat": -0.5, "obv_dead_cat_mult": 0.6,
    "obv_pump_exhaust": -0.5, "obv_pump_exhaust_mult": 0.7,
    "obv_bottom_accum": 1.0, "obv_bottom_mult": 1.2,

    # Volume confirmation fallback
    "ush_high": 2.0, "ush_high_mult": 1.3,
    "ush_mid": 1.2, "ush_mid_mult": 1.1,
    "ush_low": 0.3, "ush_low_mult": 0.6,
    "vol_conc_high": 0.3, "vol_conc_high_mult": 1.2,
    "vol_conc_mid": 0.1, "vol_conc_mid_mult": 1.05,
    "vol_conc_dead": 0.02, "vol_conc_dead_mult": 0.6,

    # Support detection
    "bb_strong_pctb": 0.1, "bb_strong_mult": 1.2,
    "bb_near_pctb": 0.2, "bb_near_mult": 1.15,
    "candle_strong_count": 3, "candle_strong_mult": 1.15,
    "candle_bounce_count": 2, "candle_bounce_mult": 1.2,
    "support_tolerance": 0.03,
}


def _pa_tier_lookup(value: float, thresholds: list, factors: list) -> float:
    """Generic tier lookup: walk ascending thresholds, return matching factor."""
    for i, t in enumerate(thresholds):
        if value < t:
            return factors[i]
    return factors[-1]


def _candles_to_dataframe(candle_data: list[dict] | None) -> pd.DataFrame | None:
    """Convert candle list to pandas DataFrame sorted by timestamp."""
    if not candle_data or len(candle_data) < 5:
        return None
    df = pd.DataFrame(candle_data)
    required = {"open", "high", "low", "close", "volume", "timestamp"}
    if not required.issubset(df.columns):
        return None
    df = df.sort_values("timestamp").reset_index(drop=True)
    # Ensure numeric types
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    if len(df) < 5:
        return None
    return df


def _compute_ta_indicators(df: pd.DataFrame) -> dict:
    """
    Compute RSI(14), MACD(12,26,9), BBands(20,2), OBV from DataFrame.
    Returns dict of indicator values (latest row). Empty dict if pandas-ta unavailable.
    """
    if not _HAS_PANDAS_TA or df is None or len(df) < 15:
        return {}

    indicators = {}
    try:
        # RSI(14) — needs 15+ candles
        rsi = ta.rsi(df["close"], length=14)
        if rsi is not None and len(rsi) > 0:
            rsi_val = rsi.iloc[-1]
            if pd.notna(rsi_val):
                indicators["rsi_14"] = float(rsi_val)
                # RSI slope over last 3 periods for trend direction
                if len(rsi) >= 4:
                    recent = rsi.dropna().tail(4)
                    if len(recent) >= 4:
                        indicators["rsi_slope"] = float(recent.iloc[-1] - recent.iloc[-4])

        # MACD(12,26,9) — needs 35+ candles
        if len(df) >= 35:
            macd_df = ta.macd(df["close"], fast=12, slow=26, signal=9)
            if macd_df is not None and len(macd_df) > 0:
                hist_col = [c for c in macd_df.columns if "h" in c.lower() or "hist" in c.lower()]
                if hist_col:
                    hist_val = macd_df[hist_col[0]].iloc[-1]
                    if pd.notna(hist_val):
                        indicators["macd_histogram"] = float(hist_val)

        # BBands(20,2) — needs 20+ candles
        if len(df) >= 20:
            bb = ta.bbands(df["close"], length=20, std=2)
            if bb is not None and len(bb) > 0:
                # Column names: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
                bbl_col = [c for c in bb.columns if c.startswith("BBL")]
                bbu_col = [c for c in bb.columns if c.startswith("BBU")]
                bbp_col = [c for c in bb.columns if c.startswith("BBP")]
                bbm_col = [c for c in bb.columns if c.startswith("BBM")]

                if bbp_col:
                    pct_b = bb[bbp_col[0]].iloc[-1]
                    if pd.notna(pct_b):
                        indicators["bb_pct_b"] = float(pct_b)

                if bbl_col and bbu_col and bbm_col:
                    bbl = bb[bbl_col[0]].iloc[-1]
                    bbu = bb[bbu_col[0]].iloc[-1]
                    bbm = bb[bbm_col[0]].iloc[-1]
                    if pd.notna(bbl) and pd.notna(bbu) and bbm > 0:
                        indicators["bb_width"] = float((bbu - bbl) / bbm)

        # OBV — needs 5+ candles
        if len(df) >= 5:
            obv = ta.obv(df["close"], df["volume"])
            if obv is not None and len(obv) >= 5:
                recent_obv = obv.dropna().tail(5)
                if len(recent_obv) >= 5:
                    avg_vol = df["volume"].tail(10).mean()
                    if avg_vol > 0:
                        obv_change = float(recent_obv.iloc[-1] - recent_obv.iloc[0])
                        indicators["obv_slope_norm"] = round(obv_change / avg_vol, 4)

    except Exception as e:
        logger.debug("pandas-ta indicator computation error: %s", e)

    return indicators


def compute_price_action_score(
    token: dict,
    *,
    pa_norm_floor: float = 0.4,
    pa_norm_cap: float = 1.3,
    pa_config: dict | None = None,
) -> dict:
    """
    Analyze price action from OHLCV candles and DexScreener price changes.

    Dual-path: uses pandas-ta indicators when candle data is available,
    falls back to raw DexScreener heuristics per submultiplier.

    v46: All thresholds read from pa_config. Defaults match pre-v46 behavior.

    Returns dict with:
        price_action_mult: float [pa_norm_floor, pa_norm_cap] — multiplier for final score
        position_mult, direction_mult, vol_confirm, support_mult
        momentum_direction: str — human-readable label
        price_action_score: float — normalized [0, 1] for base score component
        rsi_14, macd_histogram, bb_width, bb_pct_b, obv_slope_norm — ML features
    """
    # v46: Merge config with defaults (config overrides defaults)
    cfg = _DEFAULT_PA_CONFIG.copy()
    if pa_config:
        cfg.update(pa_config)

    # Compute TA indicators from OHLCV candles (if available)
    candle_data = token.get("candle_data")
    df = _candles_to_dataframe(candle_data)
    indicators = _compute_ta_indicators(df)

    # Store ML feature fields on the result (will be merged into token dict)
    ml_features = {
        "rsi_14": indicators.get("rsi_14"),
        "macd_histogram": indicators.get("macd_histogram"),
        "bb_width": indicators.get("bb_width"),
        "bb_pct_b": indicators.get("bb_pct_b"),
        "obv_slope_norm": indicators.get("obv_slope_norm"),
    }

    # ===================================================================
    # 1. Position vs ATH — v46: thresholds from config
    # ===================================================================
    ath_ratio = token.get("ath_ratio")
    position_mult = 1.0

    if ath_ratio is not None:
        position_mult = _pa_tier_lookup(
            ath_ratio,
            cfg["pos_thresholds"],
            cfg["pos_factors"],
        )

    # ===================================================================
    # 2. Momentum Direction — RSI + MACD if available, else raw thresholds
    # v46: all multipliers from config (direction penalties now tunable)
    # ===================================================================
    pc_5m = token.get("price_change_5m") or 0
    pc_1h = token.get("price_change_1h") or 0
    pc_6h = token.get("price_change_6h") or 0

    direction_mult = 1.0
    momentum_direction = "neutral"

    rsi = indicators.get("rsi_14")
    rsi_slope = indicators.get("rsi_slope", 0)
    macd_hist = indicators.get("macd_histogram")

    if rsi is not None:
        # RSI-based momentum classification
        if rsi > cfg["rsi_hard_pump"]:
            direction_mult = cfg["dir_hard_pump_mult"]
            momentum_direction = "hard_pumping"
        elif rsi > cfg["rsi_pump"]:
            direction_mult = cfg["dir_pump_mult"]
            momentum_direction = "pumping"
        elif rsi < cfg["rsi_freefall"]:
            direction_mult = cfg["dir_freefall_mult"]
            momentum_direction = "freefall"
        elif rsi < cfg["rsi_dying"] and rsi_slope <= 0:
            direction_mult = cfg["dir_dying_mult"]
            momentum_direction = "dying"
        elif (rsi < cfg["rsi_bounce_upper"]
              and rsi_slope > cfg["rsi_strong_bounce_slope"]
              and macd_hist is not None and macd_hist > 0):
            direction_mult = cfg["dir_strong_bounce_mult"]
            momentum_direction = "strong_bounce"
        elif cfg["rsi_dying"] <= rsi < cfg["rsi_bounce_upper"] and rsi_slope > 0:
            direction_mult = cfg["dir_bounce_mult"]
            momentum_direction = "bouncing"
        elif (cfg["rsi_plateau_lower"] <= rsi <= cfg["rsi_plateau_upper"]
              and abs(rsi_slope) < cfg["rsi_plateau_slope_max"]):
            direction_mult = cfg["dir_plateau_mult"]
            momentum_direction = "plateau"

        # MACD confirmation adjustment
        if macd_hist is not None:
            if momentum_direction in ("bouncing", "strong_bounce") and macd_hist > 0:
                direction_mult = min(cfg["dir_cap"], direction_mult + cfg["dir_macd_bonus"])
    else:
        # Fallback: multi-signal DexScreener heuristics (no OHLCV candles)
        bsr_5m = token.get("buy_sell_ratio_5m") or 0.5
        bsr_1h = token.get("buy_sell_ratio_1h") or 0.5

        is_hard_pumping = pc_1h > cfg["fb_hard_pump_1h"] or pc_5m > cfg["fb_hard_pump_5m"]
        is_pumping = pc_1h > cfg["fb_pump_1h"] or pc_5m > cfg["fb_pump_5m"]
        is_dying = pc_1h < cfg["fb_dying_1h"] and pc_6h < cfg["fb_dying_6h"]
        is_freefall = pc_1h < cfg["fb_freefall_1h"] and pc_5m < cfg["fb_freefall_5m"]
        is_plateau = abs(pc_1h) < cfg["fb_plateau_1h"] and abs(pc_6h) < cfg["fb_plateau_6h"]
        is_bouncing_fb = pc_6h < cfg["fb_bounce_6h"] and pc_1h > cfg["fb_bounce_1h"]
        is_strong_bounce = pc_6h < cfg["fb_strong_bounce_6h"] and pc_1h > cfg["fb_strong_bounce_1h"]

        if is_hard_pumping:
            direction_mult = cfg["fb_hard_pump_mult"]
            momentum_direction = "hard_pumping"
        elif is_pumping:
            direction_mult = cfg["fb_pump_mult"]
            momentum_direction = "pumping"
        elif is_freefall:
            direction_mult = cfg["fb_freefall_mult"]
            momentum_direction = "freefall"
        elif is_dying:
            direction_mult = cfg["fb_dying_mult"]
            momentum_direction = "dying"
        elif is_strong_bounce:
            direction_mult = cfg["fb_strong_bounce_mult"]
            momentum_direction = "strong_bounce"
        elif is_bouncing_fb:
            direction_mult = cfg["fb_bounce_mult"]
            momentum_direction = "bouncing"
        elif is_plateau:
            direction_mult = cfg["fb_plateau_mult"]
            momentum_direction = "plateau"

        # Buy/sell ratio confirmation: adjust direction_mult for bounce bonuses
        if momentum_direction in ("bouncing", "strong_bounce") and bsr_1h > cfg["fb_bsr_bounce_threshold"]:
            direction_mult = min(cfg["dir_cap"], direction_mult + cfg["fb_bsr_bounce_bonus"])
        elif momentum_direction in ("bouncing", "strong_bounce") and bsr_1h < cfg["fb_bsr_dead_cat_threshold"]:
            direction_mult = max(1.0, direction_mult - cfg["fb_bsr_dead_cat_penalty"])

    # v9: When fallback leaves momentum "neutral", use pc24h as tiebreaker
    pc_24h = token.get("price_change_24h") or 0
    if rsi is None and momentum_direction == "neutral" and pc_24h:
        if pc_24h < cfg["pc24_freefall"]:
            direction_mult = cfg["pc24_freefall_mult"]
            momentum_direction = "freefall"
        elif pc_24h < cfg["pc24_dying"]:
            direction_mult = cfg["pc24_dying_mult"]
            momentum_direction = "dying"
        elif pc_24h < cfg["pc24_bleeding"]:
            direction_mult = cfg["pc24_bleeding_mult"]
            momentum_direction = "bleeding"
        elif pc_24h > cfg["pc24_pumping_hard"]:
            direction_mult = cfg["pc24_pumping_hard_mult"]
            momentum_direction = "pumping"
        elif pc_24h > cfg["pc24_pumping"]:
            direction_mult = cfg["pc24_pumping_mult"]
            momentum_direction = "pumping"

    # v9: When ath_ratio is missing, use pc24h as proxy for position
    if ath_ratio is None and pc_24h:
        if pc_24h < -70:
            position_mult = cfg["pc24_pos_crash70_mult"]
        elif pc_24h < -50:
            position_mult = cfg["pc24_pos_crash50_mult"]
        elif pc_24h < -30:
            position_mult = cfg["pc24_pos_crash30_mult"]
        elif pc_24h > 200:
            position_mult = cfg["pc24_pos_pump200_mult"]
        elif pc_24h > 100:
            position_mult = cfg["pc24_pos_pump100_mult"]

    # Derive boolean flags from momentum_direction for use in vol_confirm/support
    is_bouncing = momentum_direction in ("bouncing", "strong_bounce")
    is_pumping = momentum_direction in ("pumping", "hard_pumping")
    is_deep_dip = ath_ratio is not None and ath_ratio < 0.25

    # ===================================================================
    # 3. Volume Confirmation — OBV if available, else ultra_short_heat
    # v46: all thresholds from config
    # ===================================================================
    obv_slope = indicators.get("obv_slope_norm")
    vol_confirm = 1.0

    if obv_slope is not None:
        # OBV-based volume confirmation
        if is_bouncing and obv_slope > cfg["obv_strong_accum"]:
            vol_confirm = cfg["obv_strong_mult"]
        elif is_bouncing and obv_slope > cfg["obv_mod_accum"]:
            vol_confirm = cfg["obv_mod_mult"]
        elif is_bouncing and obv_slope < cfg["obv_dead_cat"]:
            vol_confirm = cfg["obv_dead_cat_mult"]
        elif is_pumping and obv_slope < cfg["obv_pump_exhaust"]:
            vol_confirm = cfg["obv_pump_exhaust_mult"]
        elif is_deep_dip and obv_slope > cfg["obv_bottom_accum"]:
            vol_confirm = cfg["obv_bottom_mult"]
    else:
        # Fallback: multi-signal volume confirmation from DexScreener
        vol_5m = token.get("volume_5m") or 0
        vol_1h = token.get("volume_1h") or 0
        vol_6h = token.get("volume_6h") or 0
        vol_24h = token.get("volume_24h") or 0
        bsr_1h_vol = token.get("buy_sell_ratio_1h") or 0.5

        # Ultra-short heat: is volume accelerating right now?
        ultra_short_heat = token.get("ultra_short_heat") or 0
        if ultra_short_heat == 0 and vol_1h > 0:
            ultra_short_heat = (vol_5m * 12) / vol_1h

        # Volume concentration: what % of 24h volume happened in last hour?
        vol_concentration = (vol_1h / vol_24h) if vol_24h > 0 else 0

        # Composite volume signal: heat + concentration + buy pressure
        vol_signals = []

        # Signal 1: Ultra-short heat (is volume accelerating?)
        if ultra_short_heat > cfg["ush_high"]:
            vol_signals.append(cfg["ush_high_mult"])
        elif ultra_short_heat > cfg["ush_mid"]:
            vol_signals.append(cfg["ush_mid_mult"])
        elif ultra_short_heat < cfg["ush_low"]:
            vol_signals.append(cfg["ush_low_mult"])
        else:
            vol_signals.append(1.0)

        # Signal 2: Volume concentration (is activity happening NOW?)
        if vol_concentration > cfg["vol_conc_high"]:
            vol_signals.append(cfg["vol_conc_high_mult"])
        elif vol_concentration > cfg["vol_conc_mid"]:
            vol_signals.append(cfg["vol_conc_mid_mult"])
        elif vol_concentration < cfg["vol_conc_dead"] and vol_24h > 0:
            vol_signals.append(cfg["vol_conc_dead_mult"])
        else:
            vol_signals.append(1.0)

        # Signal 3: Buy pressure alignment with momentum
        if is_bouncing and bsr_1h_vol > 0.6:
            vol_signals.append(1.2)   # buyers confirm bounce
        elif is_bouncing and bsr_1h_vol < 0.4:
            vol_signals.append(0.6)   # dead cat bounce
        elif is_pumping and bsr_1h_vol < 0.45:
            vol_signals.append(0.7)   # pump exhaustion
        elif is_deep_dip and bsr_1h_vol > 0.6:
            vol_signals.append(1.2)   # accumulation at bottom
        else:
            vol_signals.append(1.0)

        vol_confirm = sum(vol_signals) / len(vol_signals)

    # ===================================================================
    # 4. Support Detection — BBands %B + candle clustering
    # v46: thresholds from config
    # ===================================================================
    support_mult = 1.0
    support_bounces = 0
    bb_pct_b = indicators.get("bb_pct_b")

    if candle_data:
        support_bounces = count_support_touches(candle_data, tolerance=cfg["support_tolerance"])

    if bb_pct_b is not None:
        # BBands-enhanced support detection
        if bb_pct_b <= cfg["bb_strong_pctb"] and is_bouncing:
            support_mult = cfg["bb_strong_mult"]
        elif bb_pct_b <= cfg["bb_near_pctb"] and support_bounces >= cfg["candle_bounce_count"]:
            support_mult = cfg["bb_near_mult"]
        elif support_bounces >= cfg["candle_strong_count"]:
            support_mult = cfg["candle_strong_mult"]
    else:
        # Fallback: candle support only
        if candle_data:
            if support_bounces >= cfg["candle_bounce_count"] and is_bouncing:
                support_mult = cfg["candle_bounce_mult"]
            elif support_bounces >= cfg["candle_strong_count"]:
                support_mult = cfg["candle_strong_mult"]

    # ===================================================================
    # Final: average of all 4 sub-multipliers, clamped to [pa_norm_floor, pa_norm_cap]
    # v20: bounds passed as params from pipeline (dynamic via SCORING_PARAMS)
    # ===================================================================
    price_action_mult = (position_mult + direction_mult + vol_confirm + support_mult) / 4.0
    price_action_mult = max(pa_norm_floor, min(pa_norm_cap, price_action_mult))

    # Normalized score [0, 1] — neutral (all sub-mults=1.0) maps to 0.5
    # v10: piecewise normalization so neutral=0.5
    below_range = 1.0 - pa_norm_floor  # 0.6 at default
    above_range = pa_norm_cap - 1.0    # 0.3 at default
    if price_action_mult <= 1.0:
        price_action_score = (price_action_mult - pa_norm_floor) / (below_range * 2) if below_range > 0 else 0.5
    else:
        price_action_score = 0.5 + (price_action_mult - 1.0) / (above_range * 2) if above_range > 0 else 0.5

    result = {
        "price_action_mult": round(price_action_mult, 3),
        "price_action_score": round(price_action_score, 3),
        "position_mult": round(position_mult, 3),
        "direction_mult": round(direction_mult, 3),
        "vol_confirm": round(vol_confirm, 3),
        "support_mult": round(support_mult, 3),
        "momentum_direction": momentum_direction,
        "support_bounces": support_bounces,
    }
    # Merge ML feature fields
    result.update(ml_features)

    return result
