"""
Price Action Analysis — compute price-based multipliers from OHLCV + DexScreener data.

Core principle from MemecoinGuide: "Buy the dip, NOT the pump"
- 2.5-3x below ATH with volume = ideal setup
- Never chase a pump
- Don't catch a falling knife

Uses pandas-ta (RSI, MACD, BBands, OBV) when available, with per-submultiplier
fallback to raw DexScreener heuristics when OHLCV data or pandas-ta is missing.

Returns a price_action_multiplier in [0.4, 1.3] applied to the final score.
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


def compute_price_action_score(token: dict) -> dict:
    """
    Analyze price action from OHLCV candles and DexScreener price changes.

    Dual-path: uses pandas-ta indicators when candle data is available,
    falls back to raw DexScreener heuristics per submultiplier.

    Returns dict with:
        price_action_mult: float [0.4, 1.3] — multiplier for final score
        position_mult, direction_mult, vol_confirm, support_mult
        momentum_direction: str — human-readable label
        price_action_score: float — normalized [0, 1] for base score component
        rsi_14, macd_histogram, bb_width, bb_pct_b, obv_slope_norm — ML features
    """
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
    # 1. Position vs ATH (unchanged — ath_ratio buckets work well)
    # ===================================================================
    ath_ratio = token.get("ath_ratio")
    position_mult = 1.0

    if ath_ratio is not None:
        if ath_ratio > 0.90:
            position_mult = 0.4
        elif ath_ratio > 0.70:
            position_mult = 0.6
        elif ath_ratio > 0.50:
            position_mult = 0.9
        elif ath_ratio >= 0.20:
            position_mult = 1.3    # Sweet spot: 50-80% dip from ATH
        elif ath_ratio >= 0.10:
            position_mult = 0.8
        else:
            position_mult = 0.4

    # ===================================================================
    # 2. Momentum Direction — RSI + MACD if available, else raw thresholds
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
        if rsi > 80:
            direction_mult = 0.3
            momentum_direction = "hard_pumping"
        elif rsi > 70:
            direction_mult = 0.5
            momentum_direction = "pumping"
        elif rsi < 20:
            direction_mult = 0.3
            momentum_direction = "freefall"
        elif rsi < 30 and rsi_slope <= 0:
            direction_mult = 0.5
            momentum_direction = "dying"
        elif rsi < 40 and rsi_slope > 3 and macd_hist is not None and macd_hist > 0:
            direction_mult = 1.4
            momentum_direction = "strong_bounce"
        elif 30 <= rsi < 40 and rsi_slope > 0:
            direction_mult = 1.3
            momentum_direction = "bouncing"
        elif 40 <= rsi <= 60 and abs(rsi_slope) < 3:
            direction_mult = 1.1
            momentum_direction = "plateau"

        # MACD confirmation adjustment
        if macd_hist is not None:
            if momentum_direction in ("bouncing", "strong_bounce") and macd_hist > 0:
                direction_mult = min(1.5, direction_mult + 0.1)
            elif momentum_direction in ("pumping", "hard_pumping") and macd_hist < 0:
                direction_mult = max(0.2, direction_mult - 0.1)  # Divergence warning
    else:
        # Fallback: multi-signal DexScreener heuristics (no OHLCV candles)
        # Uses price changes, buy/sell ratios, and volume concentration
        bsr_5m = token.get("buy_sell_ratio_5m") or 0.5
        bsr_1h = token.get("buy_sell_ratio_1h") or 0.5

        is_hard_pumping = pc_1h > 50 or pc_5m > 25
        is_pumping = pc_1h > 20 or pc_5m > 10
        is_dying = pc_1h < -10 and pc_6h < -30
        is_freefall = pc_1h < -20 and pc_5m < -10
        is_plateau = abs(pc_1h) < 5 and abs(pc_6h) < 10
        is_bouncing = pc_6h < -15 and pc_1h > 5
        is_strong_bounce = pc_6h < -30 and pc_1h > 10

        if is_hard_pumping:
            direction_mult = 0.3
            momentum_direction = "hard_pumping"
        elif is_pumping:
            direction_mult = 0.5
            momentum_direction = "pumping"
        elif is_freefall:
            direction_mult = 0.3
            momentum_direction = "freefall"
        elif is_dying:
            direction_mult = 0.5
            momentum_direction = "dying"
        elif is_strong_bounce:
            direction_mult = 1.4
            momentum_direction = "strong_bounce"
        elif is_bouncing:
            direction_mult = 1.3
            momentum_direction = "bouncing"
        elif is_plateau:
            direction_mult = 1.1
            momentum_direction = "plateau"

        # Buy/sell ratio confirmation: adjust direction_mult based on order flow
        if momentum_direction in ("bouncing", "strong_bounce") and bsr_1h > 0.6:
            direction_mult = min(1.5, direction_mult + 0.1)  # buyers confirm bounce
        elif momentum_direction in ("bouncing", "strong_bounce") and bsr_1h < 0.4:
            direction_mult = max(0.5, direction_mult - 0.2)  # sellers = dead cat bounce
        elif momentum_direction in ("pumping", "hard_pumping") and bsr_5m < 0.45:
            direction_mult = max(0.2, direction_mult - 0.1)  # pump losing steam

    # v9: When fallback leaves momentum "neutral", use pc24h as tiebreaker
    pc_24h = token.get("price_change_24h") or 0
    if rsi is None and momentum_direction == "neutral" and pc_24h:
        if pc_24h < -60:
            direction_mult = 0.3
            momentum_direction = "freefall"
        elif pc_24h < -40:
            direction_mult = 0.5
            momentum_direction = "dying"
        elif pc_24h < -20:
            direction_mult = 0.7
            momentum_direction = "bleeding"
        elif pc_24h > 100:
            direction_mult = 0.5
            momentum_direction = "pumping"
        elif pc_24h > 50:
            direction_mult = 0.7
            momentum_direction = "pumping"

    # v9: When ath_ratio is missing, use pc24h as proxy for position
    if ath_ratio is None and pc_24h:
        if pc_24h < -70:
            position_mult = 0.4
        elif pc_24h < -50:
            position_mult = 0.6
        elif pc_24h < -30:
            position_mult = 0.8
        elif pc_24h > 200:
            position_mult = 0.5
        elif pc_24h > 100:
            position_mult = 0.6

    # Derive boolean flags from momentum_direction for use in vol_confirm/support
    is_bouncing = momentum_direction in ("bouncing", "strong_bounce")
    is_pumping = momentum_direction in ("pumping", "hard_pumping")
    is_deep_dip = ath_ratio is not None and ath_ratio < 0.25

    # ===================================================================
    # 3. Volume Confirmation — OBV if available, else ultra_short_heat
    # ===================================================================
    obv_slope = indicators.get("obv_slope_norm")
    vol_confirm = 1.0

    if obv_slope is not None:
        # OBV-based volume confirmation
        if is_bouncing and obv_slope > 2.0:
            vol_confirm = 1.3    # Strong accumulation
        elif is_bouncing and obv_slope > 0.5:
            vol_confirm = 1.2    # Moderate accumulation
        elif is_bouncing and obv_slope < -0.5:
            vol_confirm = 0.6    # Dead cat bounce
        elif is_pumping and obv_slope < -0.5:
            vol_confirm = 0.7    # Pump losing steam
        elif is_deep_dip and obv_slope > 1.0:
            vol_confirm = 1.2    # Accumulation at bottom
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
        if ultra_short_heat > 2.0:
            vol_signals.append(1.3)
        elif ultra_short_heat > 1.2:
            vol_signals.append(1.1)
        elif ultra_short_heat < 0.3:
            vol_signals.append(0.6)
        else:
            vol_signals.append(1.0)

        # Signal 2: Volume concentration (is activity happening NOW?)
        if vol_concentration > 0.3:    # >30% of daily vol in last hour
            vol_signals.append(1.2)
        elif vol_concentration > 0.1:
            vol_signals.append(1.05)
        elif vol_concentration < 0.02 and vol_24h > 0:  # <2% = dead
            vol_signals.append(0.6)
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
    # ===================================================================
    support_mult = 1.0
    support_bounces = 0
    bb_pct_b = indicators.get("bb_pct_b")

    if candle_data:
        support_bounces = count_support_touches(candle_data, tolerance=0.03)

    if bb_pct_b is not None:
        # BBands-enhanced support detection
        if bb_pct_b <= 0.1 and is_bouncing:
            support_mult = 1.2    # At lower band + bouncing = strong support
        elif bb_pct_b <= 0.2 and support_bounces >= 2:
            support_mult = 1.15   # Near lower band + candle support
        elif support_bounces >= 3:
            support_mult = 1.15   # Strong candle support regardless of BBands
    else:
        # Fallback: candle support only
        if candle_data:
            if support_bounces >= 2 and is_bouncing:
                support_mult = 1.2
            elif support_bounces >= 3:
                support_mult = 1.15

    # ===================================================================
    # Final: average of all 4 sub-multipliers, clamped to [0.4, 1.3]
    # ===================================================================
    price_action_mult = (position_mult + direction_mult + vol_confirm + support_mult) / 4.0
    price_action_mult = max(0.4, min(1.3, price_action_mult))

    # Normalized score [0, 1] — neutral (all sub-mults=1.0) maps to 0.5
    # v10: piecewise normalization so neutral=0.5 (was 0.667 with linear)
    if price_action_mult <= 1.0:
        price_action_score = (price_action_mult - 0.4) / 1.2   # [0.4,1.0] → [0,0.5]
    else:
        price_action_score = 0.5 + (price_action_mult - 1.0) / 0.6  # [1.0,1.3] → [0.5,1.0]

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
