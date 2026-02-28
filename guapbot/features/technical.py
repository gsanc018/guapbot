"""
guapbot/features/technical.py

Computes ~100 technical indicators from a raw OHLCV DataFrame using pandas-ta.

Design rules (CRITICAL — no look-ahead bias):
  - Every indicator uses only data available at bar open time.
  - All pandas-ta calls use append=False and are assigned explicitly.
  - No forward-fill across NaN boundaries at the start of a series.
  - NaN rows at the head (warm-up period) are preserved — pipeline drops them.

Input DataFrame contract:
  - Index: pd.DatetimeIndex, UTC, name="timestamp"
  - Columns: open, high, low, close, volume, trades

Output:
  - Same index as input
  - All indicator columns appended (original OHLCV columns dropped)
  - Column naming: snake_case, descriptive (e.g. rsi_14, macd_signal_12_26_9)

Indicator categories:
  1.  Trend          — SMA, EMA, WMA, DEMA, TEMA, HMA, VWMA
  2.  MACD family    — MACD line, signal, histogram
  3.  ADX family     — ADX, DI+, DI-
  4.  Ichimoku       — tenkan, kijun, senkou_a, senkou_b, chikou
  5.  Parabolic SAR  — psar, psar_long, psar_short
  6.  Momentum       — RSI, Stochastic, CCI, Williams %R, ROC, Momentum
  7.  Oscillators    — Awesome Oscillator, TRIX, Ultimate Oscillator
  8.  Volatility     — ATR, Bollinger Bands, Keltner Channel, Donchian
  9.  Volume         — OBV, VWAP, CMF, MFI, AD, ADOSC, EOM
  10. Cross-asset    — ETHBTC ratio (injected by pipeline, not computed here)
  11. Price derived  — log returns, range, body, upper/lower wicks, gap
  12. Rolling stats  — rolling mean/std of returns, skew, realized vol
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
    _HAS_PANDAS_TA = True
except ImportError:  # pragma: no cover
    _HAS_PANDAS_TA = False
    warnings.warn(
        "pandas-ta not installed. Install with: pip install pandas-ta. "
        "Only price-derived features will be available.",
        stacklevel=2,
    )

from guapbot.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Column name constants — single source of truth
# ---------------------------------------------------------------------------

# Trend
SMA_10 = "sma_10"
SMA_20 = "sma_20"
SMA_50 = "sma_50"
SMA_200 = "sma_200"
EMA_9 = "ema_9"
EMA_21 = "ema_21"
EMA_50 = "ema_50"
EMA_200 = "ema_200"
WMA_20 = "wma_20"
DEMA_20 = "dema_20"
TEMA_20 = "tema_20"
HMA_20 = "hma_20"
VWMA_20 = "vwma_20"

# EMA crossover signals
EMA_9_21_CROSS = "ema_9_21_cross"      # +1 bullish cross, -1 bearish, 0 flat
EMA_50_200_CROSS = "ema_50_200_cross"  # golden/death cross

# Price vs moving averages (distance, normalised by price)
PRICE_VS_SMA20 = "price_vs_sma20"
PRICE_VS_SMA50 = "price_vs_sma50"
PRICE_VS_EMA21 = "price_vs_ema21"
PRICE_VS_EMA200 = "price_vs_ema200"

# MACD (12, 26, 9)
MACD_LINE = "macd_12_26_9"
MACD_SIGNAL = "macd_signal_12_26_9"
MACD_HIST = "macd_hist_12_26_9"
MACD_HIST_CHANGE = "macd_hist_change"  # first derivative of histogram

# ADX (14)
ADX_14 = "adx_14"
DMP_14 = "dmp_14"   # DI+
DMN_14 = "dmn_14"   # DI-
DI_DIFF = "di_diff" # DI+ minus DI-

# Ichimoku (9, 26, 52)
ICHIMOKU_TENKAN = "ichimoku_tenkan"
ICHIMOKU_KIJUN = "ichimoku_kijun"
ICHIMOKU_SENKOU_A = "ichimoku_senkou_a"
ICHIMOKU_SENKOU_B = "ichimoku_senkou_b"
ICHIMOKU_CHIKOU = "ichimoku_chikou"
PRICE_VS_CLOUD = "price_vs_cloud"  # +1 above, -1 below, 0 inside

# Parabolic SAR
PSAR_LONG = "psar_long"
PSAR_SHORT = "psar_short"
PSAR_SIGNAL = "psar_signal"  # +1 bullish, -1 bearish

# RSI
RSI_14 = "rsi_14"
RSI_7 = "rsi_7"
RSI_21 = "rsi_21"
RSI_14_CHANGE = "rsi_14_change"         # first derivative
RSI_14_DIVERGENCE = "rsi_14_div"        # price up but RSI down (rolling 5)

# Stochastic
STOCH_K = "stoch_k_14_3_3"
STOCH_D = "stoch_d_14_3_3"
STOCH_CROSS = "stoch_cross"

# CCI, Williams %R, ROC, Momentum
CCI_20 = "cci_20"
WILLR_14 = "willr_14"
ROC_10 = "roc_10"
ROC_20 = "roc_20"
MOM_10 = "mom_10"

# Awesome Oscillator, TRIX, Ultimate Oscillator
AO = "ao"
TRIX_18 = "trix_18"
UO = "uo_7_14_28"

# ATR
ATR_14 = "atr_14"
ATR_14_PCT = "atr_14_pct"  # ATR as % of close
NATR_14 = "natr_14"        # Normalized ATR

# Bollinger Bands (20, 2)
BB_UPPER = "bb_upper_20_2"
BB_MID = "bb_mid_20_2"
BB_LOWER = "bb_lower_20_2"
BB_WIDTH = "bb_width_20_2"       # (upper - lower) / mid
BB_PCT = "bb_pct_20_2"           # (close - lower) / (upper - lower)
BB_SIGNAL = "bb_signal_20_2"     # +1 above upper, -1 below lower, 0 inside

# Keltner Channel (20, 2)
KC_UPPER = "kc_upper_20_2"
KC_MID = "kc_mid_20_2"
KC_LOWER = "kc_lower_20_2"
KC_WIDTH = "kc_width_20_2"

# Donchian Channel (20)
DC_UPPER = "dc_upper_20"
DC_MID = "dc_mid_20"
DC_LOWER = "dc_lower_20"
DC_WIDTH = "dc_width_20"
DC_PCT = "dc_pct_20"   # position within channel

# Squeeze (BB inside KC)
SQUEEZE = "squeeze"    # 1 = BB inside KC (low vol compression), 0 = expanded

# Volume indicators
OBV = "obv"
OBV_EMA = "obv_ema_20"         # OBV smoothed
OBV_TREND = "obv_trend"        # sign of OBV vs its EMA
CMF_20 = "cmf_20"
MFI_14 = "mfi_14"
AD = "ad"                       # Chaikin A/D Line
ADOSC = "adosc"                 # Chaikin A/D Oscillator
EOM_14 = "eom_14"              # Ease of Movement
VWAP = "vwap"
PRICE_VS_VWAP = "price_vs_vwap"

# Volume momentum
VOLUME_SMA_20 = "volume_sma_20"
VOLUME_RATIO = "volume_ratio"   # current volume / 20-bar SMA
VOLUME_TREND = "volume_trend"   # sign of volume change

# Price-derived features
LOG_RETURN = "log_return"
LOG_RETURN_2 = "log_return_2"   # 2-bar log return
LOG_RETURN_4 = "log_return_4"
LOG_RETURN_8 = "log_return_8"
LOG_RETURN_24 = "log_return_24"
HIGH_LOW_RANGE = "high_low_range"       # (high - low) / close
BODY = "body"                           # |close - open| / close
UPPER_WICK = "upper_wick"              # (high - max(open,close)) / close
LOWER_WICK = "lower_wick"              # (min(open,close) - low) / close
CANDLE_DIR = "candle_dir"              # +1 bullish, -1 bearish
GAP = "gap"                            # (open - prev_close) / prev_close

# Rolling statistics (on log returns)
ROLLING_MEAN_8 = "rolling_mean_8"
ROLLING_MEAN_24 = "rolling_mean_24"
ROLLING_STD_8 = "rolling_std_8"
ROLLING_STD_24 = "rolling_std_24"
ROLLING_SKEW_24 = "rolling_skew_24"
REALIZED_VOL_24 = "realized_vol_24"   # annualized realized vol
REALIZED_VOL_168 = "realized_vol_168" # 7-day realized vol


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators for a single timeframe OHLCV DataFrame.

    Args:
        df: OHLCV DataFrame with UTC DatetimeIndex.
            Must have columns: open, high, low, close, volume

    Returns:
        DataFrame with only indicator columns (OHLCV dropped).
        Same index as input. NaN rows at head are preserved.
    """
    if df.empty:
        raise ValueError("Cannot compute indicators on empty DataFrame")

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]
    v = df["volume"]

    out = pd.DataFrame(index=df.index)

    # ------------------------------------------------------------------
    # 1. Price-derived (no external library needed — compute first)
    # ------------------------------------------------------------------
    out[LOG_RETURN] = np.log(c / c.shift(1))
    out[LOG_RETURN_2] = np.log(c / c.shift(2))
    out[LOG_RETURN_4] = np.log(c / c.shift(4))
    out[LOG_RETURN_8] = np.log(c / c.shift(8))
    out[LOG_RETURN_24] = np.log(c / c.shift(24))

    out[HIGH_LOW_RANGE] = (h - l) / c
    out[BODY] = (c - o).abs() / c
    out[UPPER_WICK] = (h - pd.concat([o, c], axis=1).max(axis=1)) / c
    out[LOWER_WICK] = (pd.concat([o, c], axis=1).min(axis=1) - l) / c
    out[CANDLE_DIR] = np.sign(c - o)
    out[GAP] = (o - c.shift(1)) / c.shift(1)

    # ------------------------------------------------------------------
    # 2. Rolling statistics
    # ------------------------------------------------------------------
    ret = out[LOG_RETURN]
    out[ROLLING_MEAN_8] = ret.rolling(8).mean()
    out[ROLLING_MEAN_24] = ret.rolling(24).mean()
    out[ROLLING_STD_8] = ret.rolling(8).std()
    out[ROLLING_STD_24] = ret.rolling(24).std()
    out[ROLLING_SKEW_24] = ret.rolling(24).skew()
    # Annualised realized vol: std of hourly returns * sqrt(8760)
    out[REALIZED_VOL_24] = ret.rolling(24).std() * np.sqrt(8760)
    out[REALIZED_VOL_168] = ret.rolling(168).std() * np.sqrt(8760)

    # Volume stats (always available)
    vol_sma = v.rolling(20).mean()
    out[VOLUME_SMA_20] = vol_sma
    out[VOLUME_RATIO] = v / vol_sma.replace(0, np.nan)
    out[VOLUME_TREND] = np.sign(v - v.shift(1))

    if not _HAS_PANDAS_TA:
        logger.warning("pandas-ta not available — returning price-derived features only")
        return out

    # ------------------------------------------------------------------
    # 3. Trend indicators
    # ------------------------------------------------------------------
    out[SMA_10] = ta.sma(c, length=10)
    out[SMA_20] = ta.sma(c, length=20)
    out[SMA_50] = ta.sma(c, length=50)
    out[SMA_200] = ta.sma(c, length=200)
    out[EMA_9] = ta.ema(c, length=9)
    out[EMA_21] = ta.ema(c, length=21)
    out[EMA_50] = ta.ema(c, length=50)
    out[EMA_200] = ta.ema(c, length=200)
    out[WMA_20] = ta.wma(c, length=20)
    out[DEMA_20] = ta.dema(c, length=20)
    out[TEMA_20] = ta.tema(c, length=20)
    out[HMA_20] = ta.hma(c, length=20)

    # VWMA needs volume
    vwma = ta.vwma(c, v, length=20)
    out[VWMA_20] = vwma

    # Price vs MA (normalised distance)
    out[PRICE_VS_SMA20] = (c - out[SMA_20]) / out[SMA_20]
    out[PRICE_VS_SMA50] = (c - out[SMA_50]) / out[SMA_50]
    out[PRICE_VS_EMA21] = (c - out[EMA_21]) / out[EMA_21]
    out[PRICE_VS_EMA200] = (c - out[EMA_200]) / out[EMA_200]

    # EMA crossover signals
    out[EMA_9_21_CROSS] = _crossover_signal(out[EMA_9], out[EMA_21])
    out[EMA_50_200_CROSS] = _crossover_signal(out[EMA_50], out[EMA_200])

    # ------------------------------------------------------------------
    # 4. MACD (12, 26, 9)
    # ------------------------------------------------------------------
    macd_df = ta.macd(c, fast=12, slow=26, signal=9)
    if macd_df is not None and not macd_df.empty:
        cols = macd_df.columns.tolist()
        # pandas-ta names: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        out[MACD_LINE] = macd_df.iloc[:, 0]
        out[MACD_HIST] = macd_df.iloc[:, 1]
        out[MACD_SIGNAL] = macd_df.iloc[:, 2]
        out[MACD_HIST_CHANGE] = out[MACD_HIST].diff(1)

    # ------------------------------------------------------------------
    # 5. ADX (14)
    # ------------------------------------------------------------------
    adx_df = ta.adx(h, l, c, length=14)
    if adx_df is not None and not adx_df.empty:
        out[ADX_14] = adx_df.iloc[:, 0]   # ADX_14
        out[DMP_14] = adx_df.iloc[:, 1]   # DMP_14
        out[DMN_14] = adx_df.iloc[:, 2]   # DMN_14
        out[DI_DIFF] = out[DMP_14] - out[DMN_14]

    # ------------------------------------------------------------------
    # 6. Ichimoku (9, 26, 52)
    # ------------------------------------------------------------------
    _compute_ichimoku(h, l, c, out)

    # ------------------------------------------------------------------
    # 7. Parabolic SAR
    # ------------------------------------------------------------------
    psar_df = ta.psar(h, l, c)
    if psar_df is not None and not psar_df.empty:
        # pandas-ta returns: PSARl, PSARs, PSARaf, PSARr columns
        psar_cols = psar_df.columns.tolist()
        long_col = next((x for x in psar_cols if "PSARl" in x), None)
        short_col = next((x for x in psar_cols if "PSARs" in x), None)
        if long_col:
            out[PSAR_LONG] = psar_df[long_col]
        if short_col:
            out[PSAR_SHORT] = psar_df[short_col]
        # Signal: if long psar is not NaN, we're in an uptrend
        if long_col and short_col:
            out[PSAR_SIGNAL] = np.where(
                psar_df[long_col].notna(), 1.0,
                np.where(psar_df[short_col].notna(), -1.0, 0.0)
            )

    # ------------------------------------------------------------------
    # 8. RSI (7, 14, 21)
    # ------------------------------------------------------------------
    out[RSI_7] = ta.rsi(c, length=7)
    out[RSI_14] = ta.rsi(c, length=14)
    out[RSI_21] = ta.rsi(c, length=21)
    out[RSI_14_CHANGE] = out[RSI_14].diff(1)

    # RSI divergence: price making new high but RSI lower (5-bar rolling)
    price_up = c.rolling(5).apply(lambda x: x[-1] > x[0], raw=True)
    rsi_down = out[RSI_14].rolling(5).apply(lambda x: x[-1] < x[0], raw=True)
    out[RSI_14_DIVERGENCE] = (price_up == 1) & (rsi_down == 1)
    out[RSI_14_DIVERGENCE] = out[RSI_14_DIVERGENCE].astype(float)

    # ------------------------------------------------------------------
    # 9. Stochastic (14, 3, 3)
    # ------------------------------------------------------------------
    stoch_df = ta.stoch(h, l, c, k=14, d=3, smooth_k=3)
    if stoch_df is not None and not stoch_df.empty:
        out[STOCH_K] = stoch_df.iloc[:, 0]
        out[STOCH_D] = stoch_df.iloc[:, 1]
        out[STOCH_CROSS] = _crossover_signal(out[STOCH_K], out[STOCH_D])

    # ------------------------------------------------------------------
    # 10. CCI, Williams %R, ROC, Momentum
    # ------------------------------------------------------------------
    out[CCI_20] = ta.cci(h, l, c, length=20)
    out[WILLR_14] = ta.willr(h, l, c, length=14)
    out[ROC_10] = ta.roc(c, length=10)
    out[ROC_20] = ta.roc(c, length=20)
    out[MOM_10] = ta.mom(c, length=10)

    # ------------------------------------------------------------------
    # 11. Oscillators
    # ------------------------------------------------------------------
    out[AO] = ta.ao(h, l)
    trix = ta.trix(c, length=18)
    if trix is not None and not trix.empty:
        out[TRIX_18] = trix.iloc[:, 0]
    uo = ta.uo(h, l, c, fast=7, medium=14, slow=28)
    if uo is not None:
        out[UO] = uo

    # ------------------------------------------------------------------
    # 12. ATR
    # ------------------------------------------------------------------
    out[ATR_14] = ta.atr(h, l, c, length=14)
    out[ATR_14_PCT] = out[ATR_14] / c
    out[NATR_14] = ta.natr(h, l, c, length=14)

    # ------------------------------------------------------------------
    # 13. Bollinger Bands (20, 2)
    # ------------------------------------------------------------------
    bb_df = ta.bbands(c, length=20, std=2)
    if bb_df is not None and not bb_df.empty:
        out[BB_LOWER] = bb_df.iloc[:, 0]
        out[BB_MID] = bb_df.iloc[:, 1]
        out[BB_UPPER] = bb_df.iloc[:, 2]
        bw = bb_df.iloc[:, 3]   # BBB (bandwidth)
        bp = bb_df.iloc[:, 4]   # BBP (percent)
        out[BB_WIDTH] = (out[BB_UPPER] - out[BB_LOWER]) / out[BB_MID]
        out[BB_PCT] = bp
        out[BB_SIGNAL] = np.where(
            c > out[BB_UPPER], 1.0,
            np.where(c < out[BB_LOWER], -1.0, 0.0)
        )

    # ------------------------------------------------------------------
    # 14. Keltner Channel (20, 2)
    # ------------------------------------------------------------------
    kc_df = ta.kc(h, l, c, length=20, scalar=2)
    if kc_df is not None and not kc_df.empty:
        out[KC_LOWER] = kc_df.iloc[:, 0]
        out[KC_MID] = kc_df.iloc[:, 1]
        out[KC_UPPER] = kc_df.iloc[:, 2]
        out[KC_WIDTH] = (out[KC_UPPER] - out[KC_LOWER]) / out[KC_MID]

    # Squeeze: BB inside KC
    if BB_UPPER in out.columns and KC_UPPER in out.columns:
        squeeze = (out[BB_UPPER] < out[KC_UPPER]) & (out[BB_LOWER] > out[KC_LOWER])
        out[SQUEEZE] = squeeze.astype(float)

    # ------------------------------------------------------------------
    # 15. Donchian Channel (20)
    # ------------------------------------------------------------------
    dc_df = ta.donchian(h, l, length=20)
    if dc_df is not None and not dc_df.empty:
        out[DC_LOWER] = dc_df.iloc[:, 0]
        out[DC_MID] = dc_df.iloc[:, 1]
        out[DC_UPPER] = dc_df.iloc[:, 2]
        ch_range = out[DC_UPPER] - out[DC_LOWER]
        out[DC_WIDTH] = ch_range / c
        out[DC_PCT] = (c - out[DC_LOWER]) / ch_range.replace(0, np.nan)

    # ------------------------------------------------------------------
    # 16. Volume indicators
    # ------------------------------------------------------------------
    out[OBV] = ta.obv(c, v)
    obv_ema = ta.ema(out[OBV], length=20)
    out[OBV_EMA] = obv_ema
    out[OBV_TREND] = np.sign(out[OBV] - out[OBV_EMA])

    out[CMF_20] = ta.cmf(h, l, c, v, length=20)
    out[MFI_14] = ta.mfi(h, l, c, v, length=14)

    ad = ta.ad(h, l, c, v)
    out[AD] = ad
    adosc = ta.adosc(h, l, c, v, fast=3, slow=10)
    out[ADOSC] = adosc

    eom = ta.eom(h, l, c, v, length=14)
    if eom is not None:
        out[EOM_14] = eom

    # VWAP — pandas-ta's VWAP needs a DatetimeIndex
    try:
        vwap_result = ta.vwap(h, l, c, v)
        out[VWAP] = vwap_result
        out[PRICE_VS_VWAP] = (c - out[VWAP]) / out[VWAP]
    except Exception:
        # VWAP can fail on non-intraday data — compute manually
        typical = (h + l + c) / 3
        cum_tv = (typical * v).cumsum()
        cum_v = v.cumsum()
        out[VWAP] = cum_tv / cum_v.replace(0, np.nan)
        out[PRICE_VS_VWAP] = (c - out[VWAP]) / out[VWAP]

    logger.debug(f"Computed {len(out.columns)} indicators on {len(out)} bars")
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _crossover_signal(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """
    Return +1 on the bar where fast crosses above slow,
    -1 where fast crosses below slow, 0 otherwise.
    No look-ahead: uses current and prior bar only.
    """
    # pandas-ta can emit object-dtype series with None on sparse/short windows.
    # Coerce to numeric so comparisons are well-defined and never raise.
    fast_num = pd.to_numeric(fast, errors="coerce")
    slow_num = pd.to_numeric(slow, errors="coerce")

    above = (fast_num > slow_num).fillna(False).astype(bool)
    prev_above = above.shift(1, fill_value=False)
    cross_up = above & (~prev_above)
    cross_dn = (~above) & prev_above
    signal = pd.Series(0.0, index=fast.index)
    signal[cross_up] = 1.0
    signal[cross_dn] = -1.0
    return signal


def _compute_ichimoku(
    h: pd.Series,
    l: pd.Series,
    c: pd.Series,
    out: pd.DataFrame,
) -> None:
    """
    Compute Ichimoku components and add to out in-place.

    Uses standard settings (9, 26, 52).
    All components use only past data — Senkou span A/B are NOT shifted forward
    (that would introduce look-ahead). We use the un-shifted values for ML features.
    """
    # Tenkan-sen (conversion line): (9-period high + low) / 2
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    # Kijun-sen (base line): (26-period high + low) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    # Senkou Span A: (tenkan + kijun) / 2  — NOT shifted (avoids look-ahead)
    senkou_a = (tenkan + kijun) / 2
    # Senkou Span B: (52-period high + low) / 2 — NOT shifted
    senkou_b = (h.rolling(52).max() + l.rolling(52).min()) / 2
    # Chikou Span: close shifted back 26 periods (uses past close, no look-ahead)
    chikou = c.shift(26)

    out[ICHIMOKU_TENKAN] = tenkan
    out[ICHIMOKU_KIJUN] = kijun
    out[ICHIMOKU_SENKOU_A] = senkou_a
    out[ICHIMOKU_SENKOU_B] = senkou_b
    out[ICHIMOKU_CHIKOU] = chikou

    # Price vs cloud: +1 above, -1 below, 0 inside
    cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    cloud_bot = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)
    out[PRICE_VS_CLOUD] = np.where(
        c > cloud_top, 1.0,
        np.where(c < cloud_bot, -1.0, 0.0)
    )


def get_feature_names() -> list[str]:
    """Return all feature column names in the order they're computed."""
    return [
        # Price-derived
        LOG_RETURN, LOG_RETURN_2, LOG_RETURN_4, LOG_RETURN_8, LOG_RETURN_24,
        HIGH_LOW_RANGE, BODY, UPPER_WICK, LOWER_WICK, CANDLE_DIR, GAP,
        # Rolling stats
        ROLLING_MEAN_8, ROLLING_MEAN_24, ROLLING_STD_8, ROLLING_STD_24,
        ROLLING_SKEW_24, REALIZED_VOL_24, REALIZED_VOL_168,
        # Volume
        VOLUME_SMA_20, VOLUME_RATIO, VOLUME_TREND,
        # Trend
        SMA_10, SMA_20, SMA_50, SMA_200,
        EMA_9, EMA_21, EMA_50, EMA_200,
        WMA_20, DEMA_20, TEMA_20, HMA_20, VWMA_20,
        PRICE_VS_SMA20, PRICE_VS_SMA50, PRICE_VS_EMA21, PRICE_VS_EMA200,
        EMA_9_21_CROSS, EMA_50_200_CROSS,
        # MACD
        MACD_LINE, MACD_SIGNAL, MACD_HIST, MACD_HIST_CHANGE,
        # ADX
        ADX_14, DMP_14, DMN_14, DI_DIFF,
        # Ichimoku
        ICHIMOKU_TENKAN, ICHIMOKU_KIJUN, ICHIMOKU_SENKOU_A,
        ICHIMOKU_SENKOU_B, ICHIMOKU_CHIKOU, PRICE_VS_CLOUD,
        # PSAR
        PSAR_LONG, PSAR_SHORT, PSAR_SIGNAL,
        # RSI
        RSI_7, RSI_14, RSI_21, RSI_14_CHANGE, RSI_14_DIVERGENCE,
        # Stochastic
        STOCH_K, STOCH_D, STOCH_CROSS,
        # Momentum
        CCI_20, WILLR_14, ROC_10, ROC_20, MOM_10,
        # Oscillators
        AO, TRIX_18, UO,
        # ATR
        ATR_14, ATR_14_PCT, NATR_14,
        # Bollinger
        BB_UPPER, BB_MID, BB_LOWER, BB_WIDTH, BB_PCT, BB_SIGNAL,
        # Keltner
        KC_UPPER, KC_MID, KC_LOWER, KC_WIDTH,
        # Squeeze
        SQUEEZE,
        # Donchian
        DC_UPPER, DC_MID, DC_LOWER, DC_WIDTH, DC_PCT,
        # Volume indicators
        OBV, OBV_EMA, OBV_TREND, CMF_20, MFI_14, AD, ADOSC, EOM_14,
        VWAP, PRICE_VS_VWAP,
    ]
