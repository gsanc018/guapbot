"""
guapbot/features/pipeline.py

FeaturePipeline — master feature construction pipeline.

This is the ONLY entry point for the feature layer. Every other layer
(regime, models, ensemble) calls this. Never import technical.py or
multi_timeframe.py directly from outside the features package.

Responsibilities:
  1. Load raw OHLCV data via DataManager.fetch()
  2. Inject cross-asset features (ETHBTC ratio)
  3. Build multi-timeframe features via build_multi_timeframe_features()
  4. Normalise: rolling z-score (200-bar window, clipped [-5, 5])
  5. Drop NaN warm-up rows
  6. Save normalised features to Parquet cache
  7. Serve cached features on subsequent calls

Cache layout:
  data/cache/features_{pair}_norm.parquet   ← normalised (what models consume)
  data/cache/features_{pair}_raw.parquet    ← raw (for inspection / debugging)

Normalisation rules (CRITICAL — no look-ahead bias):
  - Rolling mean and std computed with min_periods=50
  - z = (x - rolling_mean) / rolling_std
  - Clipped to [-5, 5] to suppress outlier spikes
  - Binary/signal columns (values in {-1, 0, 1}) are NOT normalised
  - Volume columns are log-transformed before z-scoring (right-skewed)
  - NaN values after normalisation are forward-filled then zero-filled

Public API:
    pipe = FeaturePipeline()
    df   = pipe.fit_transform("XBTUSD")       # build everything, save, return
    df   = pipe.transform("XBTUSD")           # load from cache (must exist)
    df   = pipe.get_observation("XBTUSD")     # latest single row as dict
    info = pipe.cache_info()                   # summary of cached features
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from guapbot.data.manager import DataManager
from guapbot.features.multi_timeframe import build_multi_timeframe_features
from guapbot.utils.config import settings
from guapbot.utils.logging import get_logger

logger = get_logger(__name__)

# Pairs that get a full feature set (traded + signal-only)
FEATURE_PAIRS = ["XBTUSD", "ETHUSD", "ETHBTC"]

# Traded pairs — ETHBTC features are injected as cross-asset into these
TRADED_PAIRS = ["XBTUSD", "ETHUSD"]

# Normalisation settings
ZSCORE_WINDOW = 200     # rolling window for mean/std
ZSCORE_MIN_PERIODS = 50 # minimum bars before normalising
ZSCORE_CLIP = 5.0       # clip z-scores to [-5, 5]

# Column name patterns that should NOT be z-score normalised
# These are already bounded/binary signals
_NO_NORM_PATTERNS = [
    "_cross",       # crossover signals: {-1, 0, 1}
    "_signal",      # PSAR signal, BB signal: {-1, 0, 1}
    "candle_dir",   # {-1, +1}
    "squeeze",      # {0, 1}
    "obv_trend",    # {-1, 0, 1}
    "volume_trend", # {-1, 0, 1}
    "_div",         # RSI divergence: {0, 1}
    "psar_long",    # NaN or float (level, not a signal)
    "psar_short",
]

# Volume-related columns that benefit from log transform before z-scoring
_LOG_TRANSFORM_PATTERNS = [
    "volume_sma",
    "volume_ratio",
    "obv",
    "obv_ema",
    "ad",           # A/D line can be very large
]


class FeaturePipeline:
    """
    End-to-end feature construction pipeline for GuapBot.

    Args:
        manager:   DataManager instance. Injected for testability.
        cache_dir: Override the default cache directory.
    """

    def __init__(
        self,
        manager: Optional[DataManager] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self._manager = manager or DataManager()
        self._cache_dir = Path(cache_dir or settings.data_cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, pair: str) -> pd.DataFrame:
        """
        Build complete normalised features for a pair from scratch.

        Steps:
          1. Load raw OHLCV from cache (run backfill_trades() first for recent bars)
          2. Build multi-timeframe indicators
          3. Inject ETHBTC cross-asset features (for traded pairs)
          4. Normalise (rolling z-score)
          5. Drop warm-up NaN rows
          6. Save raw + normalised to Parquet
          7. Return normalised DataFrame

        Args:
            pair: One of XBTUSD, ETHUSD, ETHBTC

        Returns:
            Normalised feature DataFrame, index=UTC DatetimeIndex
        """
        self._validate_pair(pair)
        logger.info(f"FeaturePipeline.fit_transform({pair})")

        # Step 1: Load OHLCV
        df_1h = self._manager.fetch(pair, "1h")
        if df_1h.empty:
            raise RuntimeError(f"No 1h data available for {pair}. Run: guapbot data download")

        logger.info(f"  Loaded {len(df_1h)} 1h bars for {pair}")

        # Step 2: Multi-timeframe features
        raw_features = build_multi_timeframe_features(df_1h)

        # Step 3: Cross-asset features (inject ETHBTC into traded pairs)
        if pair in TRADED_PAIRS:
            raw_features = self._inject_cross_asset(raw_features, pair)

        # Step 4: Normalise
        norm_features = self._normalise(raw_features)

        # Step 5: Drop warm-up rows (first ZSCORE_WINDOW rows are unreliable)
        norm_features = norm_features.iloc[ZSCORE_WINDOW:]
        raw_features = raw_features.iloc[ZSCORE_WINDOW:]

        if norm_features.empty:
            raise RuntimeError(
                f"No data remaining after dropping {ZSCORE_WINDOW} warm-up bars. "
                f"Need at least {ZSCORE_WINDOW + 1} bars of 1h data."
            )

        # Step 6: Save
        self._save(pair, raw_features, norm_features)

        logger.info(
            f"  Pipeline complete: {len(norm_features)} bars, "
            f"{len(norm_features.columns)} features "
            f"({norm_features.index[0].date()} → {norm_features.index[-1].date()})"
        )
        return norm_features

    def transform(self, pair: str) -> pd.DataFrame:
        """
        Load normalised features from cache.

        Use this during live trading / inference — avoids recomputing.
        Raises FileNotFoundError if cache doesn't exist (run fit_transform first).

        Args:
            pair: One of XBTUSD, ETHUSD, ETHBTC

        Returns:
            Cached normalised feature DataFrame
        """
        self._validate_pair(pair)
        path = self._norm_path(pair)
        if not path.exists():
            raise FileNotFoundError(
                f"No feature cache for {pair}. Run: guapbot features build {pair}"
            )
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        elif df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        logger.debug(f"Loaded {len(df)} feature rows for {pair} from cache")
        return df.sort_index()

    def get_observation(self, pair: str) -> dict:
        """
        Return the latest single observation as a flat dict.

        This is what the slow clock calls every 1h bar to get the
        current feature vector for model.predict(obs).

        Args:
            pair: Traded pair

        Returns:
            Dict of {feature_name: float}, latest bar only.
            Also includes 'timestamp' key.
        """
        df = self.transform(pair)
        latest = df.iloc[-1]
        obs = latest.to_dict()
        obs["timestamp"] = latest.name.isoformat()
        return obs

    def cache_info(self) -> dict:
        """Return summary of cached feature files."""
        info: dict = {}
        for pair in FEATURE_PAIRS:
            norm_path = self._norm_path(pair)
            raw_path = self._raw_path(pair)
            entry: dict = {}

            if norm_path.exists():
                try:
                    df_norm = pd.read_parquet(norm_path)
                    entry["bars"] = len(df_norm)
                    entry["features"] = len(df_norm.columns)
                    entry["first"] = str(df_norm.index[0])[:10]
                    entry["last"] = str(df_norm.index[-1])[:10]
                    entry["norm_size_kb"] = round(norm_path.stat().st_size / 1024, 1)
                except Exception as exc:
                    entry["error"] = str(exc)
            else:
                entry["status"] = "not built"

            if raw_path.exists():
                entry["raw_size_kb"] = round(raw_path.stat().st_size / 1024, 1)

            info[pair] = entry
        return info

    # ------------------------------------------------------------------
    # Internal — normalisation
    # ------------------------------------------------------------------

    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply rolling z-score normalisation to all feature columns.

        Rules:
          - Binary/signal columns: pass through unchanged
          - Log-skewed volume columns: log1p transform, then z-score
          - All others: z-score directly
          - Clip to [-ZSCORE_CLIP, +ZSCORE_CLIP]
          - Forward-fill remaining NaNs, then zero-fill
        """
        out = df.copy()

        for col in df.columns:
            series = df[col]

            if self._is_no_norm(col):
                # Pass through — already bounded
                continue

            if self._is_log_transform(col):
                # Log-transform first (volumes are right-skewed)
                series = np.log1p(series.clip(lower=0))

            rolling_mean = series.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_MIN_PERIODS).mean()
            rolling_std = series.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_MIN_PERIODS).std()

            # Avoid division by zero (constant series)
            rolling_std = rolling_std.replace(0, np.nan)

            z = (series - rolling_mean) / rolling_std
            out[col] = z.clip(-ZSCORE_CLIP, ZSCORE_CLIP)

        # Clean up residual NaNs
        out = out.ffill().fillna(0.0)
        return out

    @staticmethod
    def _is_no_norm(col: str) -> bool:
        return any(pattern in col for pattern in _NO_NORM_PATTERNS)

    @staticmethod
    def _is_log_transform(col: str) -> bool:
        return any(pattern in col for pattern in _LOG_TRANSFORM_PATTERNS)

    # ------------------------------------------------------------------
    # Internal — cross-asset features
    # ------------------------------------------------------------------

    def _inject_cross_asset(self, features: pd.DataFrame, pair: str) -> pd.DataFrame:
        """
        Inject ETHBTC cross-asset features into a traded pair's feature DataFrame.

        ETHBTC is a signal-only asset — never traded, but its trend and
        ratio drive the capital split between money_printer and sat_stacker.

        Injected features (1h only — sufficient for signal):
          - ethbtc_close     : raw ETHBTC close price
          - ethbtc_log_ret   : ETHBTC log return
          - ethbtc_rsi_14    : ETHBTC RSI (momentum signal)
          - ethbtc_ema_21    : ETHBTC EMA-21 trend
          - ethbtc_vs_ema21  : (close - ema21) / ema21
          - ethbtc_trend_dir : +1 close > EMA21, -1 below
        """
        try:
            df_ethbtc = self._manager.fetch("ETHBTC", "1h")
        except Exception as exc:
            logger.warning(f"Could not fetch ETHBTC for cross-asset features: {exc}")
            return features

        if df_ethbtc.empty:
            logger.warning("ETHBTC cache empty — skipping cross-asset features")
            return features

        c = df_ethbtc["close"]
        cross = pd.DataFrame(index=df_ethbtc.index)
        cross["ethbtc_close"] = c
        cross["ethbtc_log_ret"] = np.log(c / c.shift(1))

        # Simple indicators (no pandas-ta dependency for cross-asset)
        cross["ethbtc_ema_21"] = c.ewm(span=21, adjust=False).mean()
        cross["ethbtc_vs_ema21"] = (c - cross["ethbtc_ema_21"]) / cross["ethbtc_ema_21"]
        cross["ethbtc_trend_dir"] = np.where(c > cross["ethbtc_ema_21"], 1.0, -1.0)

        # RSI (14) — manual implementation (no look-ahead)
        delta = c.diff(1)
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        cross["ethbtc_rsi_14"] = 100 - (100 / (1 + rs))

        # Align to the feature DataFrame's index using ffill (no look-ahead)
        cross_aligned = cross.reindex(features.index, method="ffill")

        # Concatenate
        result = pd.concat([features, cross_aligned], axis=1)
        logger.debug(f"Injected {len(cross.columns)} ETHBTC cross-asset features into {pair}")
        return result

    # ------------------------------------------------------------------
    # Internal — I/O
    # ------------------------------------------------------------------

    def _save(
        self, pair: str, raw: pd.DataFrame, norm: pd.DataFrame
    ) -> None:
        raw_path = self._raw_path(pair)
        norm_path = self._norm_path(pair)
        raw.sort_index().to_parquet(raw_path, compression="snappy")
        norm.sort_index().to_parquet(norm_path, compression="snappy")
        logger.info(
            f"  Saved: {norm_path.name} "
            f"({round(norm_path.stat().st_size / 1024, 1)} KB normalised, "
            f"{round(raw_path.stat().st_size / 1024, 1)} KB raw)"
        )

    def _norm_path(self, pair: str) -> Path:
        return self._cache_dir / f"features_{pair}_norm.parquet"

    def _raw_path(self, pair: str) -> Path:
        return self._cache_dir / f"features_{pair}_raw.parquet"

    @staticmethod
    def _validate_pair(pair: str) -> None:
        if pair not in FEATURE_PAIRS:
            raise ValueError(
                f"Unknown pair '{pair}'. "
                f"Valid pairs for feature pipeline: {FEATURE_PAIRS}"
            )
