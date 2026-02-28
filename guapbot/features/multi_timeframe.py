"""
guapbot/features/multi_timeframe.py

Multi-timeframe feature aggregation.

Strategy:
  - 1h bars are the primary action timeframe (already in cache).
  - 4h and 1d bars are RESAMPLED from the 1h base cache.
    This avoids dependency on a separate cached 4h/1d file and eliminates
    the ETHBTC 4h data gap (only available from Q4 2024 in Session 2).
  - All timeframe features are aligned back to the 1h index using
    forward-fill — the 4h/1d value computed at bar N is propagated
    until the next 4h/1d bar opens. No look-ahead.

Output:
  - One flat DataFrame with all features prefixed by timeframe:
      1h_rsi_14, 4h_rsi_14, 1d_rsi_14
  - Index = 1h DatetimeIndex (UTC)
  - Shape: (n_1h_bars, n_features_per_tf * n_timeframes)

Resampling rules:
  OHLCV aggregation:
    open   → first
    high   → max
    low    → min
    close  → last
    volume → sum
    trades → sum (if present)

  Feature computation happens AFTER resampling, on the coarser bars.
  This means 4h features are computed on 4h candles, not on 1h features
  averaged up — which is correct and avoids distortion.

No look-ahead guarantee:
  - pd.resample(..., closed='left', label='left') labels each bar with
    its OPEN time. The 4h bar starting at 08:00 uses data from
    08:00–11:59 and is labelled 08:00. It's only available at 12:00.
  - After computing features, we shift each higher-timeframe column by
    one bar (on the coarser index) before ffill, so that a 4h bar's
    features only appear in the 1h index AFTER that bar closes.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from guapbot.features.technical import compute_indicators
from guapbot.utils.logging import get_logger

logger = get_logger(__name__)

# Timeframe labels and their pandas resample rules
TIMEFRAMES = {
    "1h": None,      # base — no resampling
    "4h": "4h",
    "1d": "1D",
}

# Resample aggregation for OHLCV
OHLCV_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def build_multi_timeframe_features(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Build features for all three timeframes from a 1h OHLCV DataFrame.

    Args:
        df_1h: 1h OHLCV DataFrame with UTC DatetimeIndex.

    Returns:
        Flat DataFrame with prefixed columns (1h_*, 4h_*, 1d_*),
        aligned to the 1h index. No look-ahead.
    """
    if df_1h.empty:
        raise ValueError("df_1h is empty — cannot build multi-timeframe features")

    result_frames: list[pd.DataFrame] = []

    for tf_label, resample_rule in TIMEFRAMES.items():
        logger.debug(f"Building {tf_label} features...")

        if resample_rule is None:
            # 1h: compute directly
            df_tf = df_1h.copy()
        else:
            df_tf = _resample_ohlcv(df_1h, resample_rule)

        if df_tf.empty:
            logger.warning(f"Empty DataFrame after resampling to {tf_label} — skipping")
            continue

        # Compute indicators on the resampled bars
        try:
            features_tf = compute_indicators(df_tf)
        except Exception as exc:
            logger.error(f"Failed to compute {tf_label} indicators: {exc}")
            raise

        # Shift by 1 bar so features are only available AFTER the bar closes
        # This is the key no-look-ahead step for coarser timeframes
        if resample_rule is not None:
            features_tf = features_tf.shift(1)

        # Prefix all columns with timeframe label
        features_tf.columns = [f"{tf_label}_{col}" for col in features_tf.columns]

        if resample_rule is not None:
            # Reindex to 1h timestamps using forward-fill
            # Only ffill — never bfill (that would introduce look-ahead)
            features_tf = features_tf.reindex(df_1h.index, method="ffill")

        result_frames.append(features_tf)
        logger.debug(
            f"  {tf_label}: {len(features_tf.columns)} features on "
            f"{len(df_tf)} bars → aligned to {len(df_1h)} 1h bars"
        )

    if not result_frames:
        raise RuntimeError("No timeframe features computed — check input data")

    combined = pd.concat(result_frames, axis=1)
    combined = combined.sort_index()

    logger.info(
        f"Multi-timeframe features: {len(combined.columns)} total columns "
        f"across {len(TIMEFRAMES)} timeframes, {len(combined)} bars"
    )
    return combined


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample a 1h OHLCV DataFrame to a coarser timeframe.

    Uses closed='left', label='left' so each bar is labelled by its
    open time. The bar is only complete when the next bar opens.

    Args:
        df:   1h OHLCV DataFrame
        rule: pandas resample rule ('4h', '1D', etc.)

    Returns:
        Resampled OHLCV DataFrame. Drops incomplete trailing bar.
    """
    # Only resample columns that exist in the DataFrame
    agg_rules = {col: agg for col, agg in OHLCV_AGG.items() if col in df.columns}
    if "trades" in df.columns:
        agg_rules["trades"] = "sum"

    resampled = (
        df[list(agg_rules.keys())]
        .resample(rule, closed="left", label="left")
        .agg(agg_rules)
        .dropna(subset=["close"])  # drop empty bars
    )

    # Drop the last (potentially incomplete) bar — it may not have closed yet
    if len(resampled) > 1:
        resampled = resampled.iloc[:-1]

    return resampled


def get_timeframe_column_names(base_columns: list[str]) -> dict[str, list[str]]:
    """
    Return a dict mapping timeframe label → expected prefixed column names.
    Useful for downstream validation.
    """
    return {
        tf: [f"{tf}_{col}" for col in base_columns]
        for tf in TIMEFRAMES
    }
