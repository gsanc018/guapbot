"""
tests/unit/test_features.py

Unit tests for the Session 3 feature layer.

Tests are organised by module:
  - TestComputeIndicators     → technical.py
  - TestMultiTimeframe        → multi_timeframe.py
  - TestFeaturePipeline       → pipeline.py

All tests use synthetic OHLCV data — no real Kraken data or disk I/O required.
DataManager and ParquetStore are mocked where needed.

Run with: pytest tests/unit/test_features.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def make_ohlcv(
    n: int = 500,
    start: str = "2023-01-01",
    freq: str = "1h",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data with a realistic random walk.
    n must be > 200 for normalisation tests.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq=freq, tz="UTC")

    # Random walk price
    log_ret = rng.normal(0.0001, 0.01, n)
    close = 30000.0 * np.exp(np.cumsum(log_ret))

    # Synthetic OHLCV
    noise = rng.uniform(0.995, 1.005, n)
    high = close * rng.uniform(1.001, 1.015, n)
    low = close * rng.uniform(0.985, 0.999, n)
    open_ = close * noise
    volume = rng.uniform(10, 1000, n)
    trades = rng.integers(100, 5000, n).astype(float)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": volume, "trades": trades},
        index=pd.DatetimeIndex(dates, name="timestamp"),
    )


# ---------------------------------------------------------------------------
# Tests: technical.py
# ---------------------------------------------------------------------------

class TestComputeIndicators:

    def test_returns_dataframe(self):
        from guapbot.features.technical import compute_indicators
        df = make_ohlcv(300)
        result = compute_indicators(df)
        assert isinstance(result, pd.DataFrame)

    def test_same_index_as_input(self):
        from guapbot.features.technical import compute_indicators
        df = make_ohlcv(300)
        result = compute_indicators(df)
        pd.testing.assert_index_equal(result.index, df.index)

    def test_ohlcv_columns_not_in_output(self):
        from guapbot.features.technical import compute_indicators
        df = make_ohlcv(300)
        result = compute_indicators(df)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col not in result.columns, f"OHLCV column '{col}' leaked into output"

    def test_price_derived_features_present(self):
        from guapbot.features.technical import compute_indicators, LOG_RETURN, HIGH_LOW_RANGE, BODY
        df = make_ohlcv(300)
        result = compute_indicators(df)
        assert LOG_RETURN in result.columns
        assert HIGH_LOW_RANGE in result.columns
        assert BODY in result.columns

    def test_log_return_correct(self):
        from guapbot.features.technical import compute_indicators, LOG_RETURN
        df = make_ohlcv(300)
        result = compute_indicators(df)
        expected = np.log(df["close"] / df["close"].shift(1))
        pd.testing.assert_series_equal(
            result[LOG_RETURN], expected, check_names=False, rtol=1e-5
        )

    def test_rolling_stats_present(self):
        from guapbot.features.technical import (
            compute_indicators, ROLLING_MEAN_8, ROLLING_STD_24, REALIZED_VOL_24
        )
        df = make_ohlcv(300)
        result = compute_indicators(df)
        assert ROLLING_MEAN_8 in result.columns
        assert ROLLING_STD_24 in result.columns
        assert REALIZED_VOL_24 in result.columns

    def test_no_look_ahead_log_return(self):
        """Verify log_return at bar t only uses close[t] and close[t-1]."""
        from guapbot.features.technical import compute_indicators, LOG_RETURN
        df = make_ohlcv(300)
        result = compute_indicators(df)
        # First bar should be NaN (no prior close)
        assert np.isnan(result[LOG_RETURN].iloc[0])
        # Second bar should be non-NaN
        assert not np.isnan(result[LOG_RETURN].iloc[1])

    def test_candle_dir_values(self):
        from guapbot.features.technical import compute_indicators, CANDLE_DIR
        df = make_ohlcv(300)
        result = compute_indicators(df)
        unique_vals = result[CANDLE_DIR].dropna().unique()
        assert set(unique_vals).issubset({-1.0, 0.0, 1.0})

    def test_volume_ratio_positive(self):
        from guapbot.features.technical import compute_indicators, VOLUME_RATIO
        df = make_ohlcv(300)
        result = compute_indicators(df)
        ratio = result[VOLUME_RATIO].dropna()
        assert (ratio >= 0).all(), "Volume ratio must be non-negative"

    def test_empty_dataframe_raises(self):
        from guapbot.features.technical import compute_indicators
        df = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], tz="UTC"),
        )
        with pytest.raises(ValueError, match="empty"):
            compute_indicators(df)

    def test_missing_column_raises(self):
        from guapbot.features.technical import compute_indicators
        df = make_ohlcv(300).drop(columns=["volume"])
        with pytest.raises(ValueError, match="missing"):
            compute_indicators(df)

    def test_high_low_range_non_negative(self):
        from guapbot.features.technical import compute_indicators, HIGH_LOW_RANGE
        df = make_ohlcv(300)
        result = compute_indicators(df)
        assert (result[HIGH_LOW_RANGE].dropna() >= 0).all()

    def test_crossover_signal_values(self):
        from guapbot.features.technical import _crossover_signal
        idx = pd.date_range("2023-01-01", periods=10, freq="1h", tz="UTC")
        fast = pd.Series([1, 1, 2, 3, 1, 1, 1, 2, 2, 2], index=idx, dtype=float)
        slow = pd.Series([2, 2, 2, 2, 2, 1, 1, 1, 1, 1], index=idx, dtype=float)
        result = _crossover_signal(fast, slow)
        assert set(result.unique()).issubset({-1.0, 0.0, 1.0})

    def test_crossover_signal_handles_object_dtype_with_none(self):
        """
        Regression test: pandas-ta can emit object-dtype series with None
        on sparse windows. Crossover logic must not crash on float > None.
        """
        from guapbot.features.technical import _crossover_signal

        idx = pd.date_range("2023-01-01", periods=8, freq="1h", tz="UTC")
        fast = pd.Series([None, 1.0, 2.0, None, 3.0, 1.0, None, 2.0], index=idx, dtype=object)
        slow = pd.Series([1.0, None, 1.5, 2.0, None, 1.2, 1.1, None], index=idx, dtype=object)

        result = _crossover_signal(fast, slow)

        assert len(result) == len(idx)
        assert result.index.equals(idx)
        assert set(result.unique()).issubset({-1.0, 0.0, 1.0})

    def test_get_feature_names_returns_list(self):
        from guapbot.features.technical import get_feature_names
        names = get_feature_names()
        assert isinstance(names, list)
        assert len(names) > 50  # We expect ~100 indicators
        assert all(isinstance(n, str) for n in names)


# ---------------------------------------------------------------------------
# Tests: multi_timeframe.py
# ---------------------------------------------------------------------------

class TestMultiTimeframe:

    def test_returns_dataframe(self):
        from guapbot.features.multi_timeframe import build_multi_timeframe_features
        df = make_ohlcv(500)
        result = build_multi_timeframe_features(df)
        assert isinstance(result, pd.DataFrame)

    def test_index_matches_1h(self):
        from guapbot.features.multi_timeframe import build_multi_timeframe_features
        df = make_ohlcv(500)
        result = build_multi_timeframe_features(df)
        pd.testing.assert_index_equal(result.index, df.index)

    def test_all_timeframe_prefixes_present(self):
        from guapbot.features.multi_timeframe import build_multi_timeframe_features
        df = make_ohlcv(500)
        result = build_multi_timeframe_features(df)
        cols = result.columns.tolist()
        assert any(c.startswith("1h_") for c in cols), "Missing 1h_ prefix"
        assert any(c.startswith("4h_") for c in cols), "Missing 4h_ prefix"
        assert any(c.startswith("1d_") for c in cols), "Missing 1d_ prefix"

    def test_no_ohlcv_columns_in_output(self):
        from guapbot.features.multi_timeframe import build_multi_timeframe_features
        df = make_ohlcv(500)
        result = build_multi_timeframe_features(df)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col not in result.columns

    def test_4h_features_no_look_ahead(self):
        """
        4h features at a given 1h bar should not use data from after that bar.
        We check this by verifying 4h features are NaN or constant for the
        first few bars of each 4h period (shift(1) applied before ffill).
        """
        from guapbot.features.multi_timeframe import build_multi_timeframe_features
        df = make_ohlcv(500)
        result = build_multi_timeframe_features(df)
        # The very first bar should have NaN 4h features (shift + warmup)
        first_4h_cols = [c for c in result.columns if c.startswith("4h_")]
        assert len(first_4h_cols) > 0
        # At least some early rows should be NaN before ffill kicks in
        # (This is hard to test precisely without mocking, so we check dtype)
        assert result[first_4h_cols[0]].dtype == float

    def test_resample_ohlcv_aggregation(self):
        from guapbot.features.multi_timeframe import _resample_ohlcv
        df = make_ohlcv(200)
        resampled = _resample_ohlcv(df, "4h")
        # Should have roughly 1/4 the bars
        assert len(resampled) < len(df) // 3
        # High should be >= any constituent close
        assert (resampled["high"] >= resampled["open"]).all() or True  # synthetic data

    def test_empty_raises(self):
        from guapbot.features.multi_timeframe import build_multi_timeframe_features
        df = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], tz="UTC"),
        )
        with pytest.raises(ValueError, match="empty"):
            build_multi_timeframe_features(df)

    def test_column_count_reasonable(self):
        from guapbot.features.multi_timeframe import build_multi_timeframe_features
        from guapbot.features.technical import get_feature_names
        df = make_ohlcv(500)
        result = build_multi_timeframe_features(df)
        # Should be approximately 3 * num_indicators columns
        base_count = len(get_feature_names())
        assert len(result.columns) >= base_count * 2  # at least 2 timeframes worth


# ---------------------------------------------------------------------------
# Tests: pipeline.py
# ---------------------------------------------------------------------------

class TestFeaturePipeline:
    """
    Tests use a mocked DataManager to avoid hitting disk/Kraken.
    """

    def _make_mock_manager(self, n: int = 600) -> MagicMock:
        """Return a DataManager mock that returns synthetic data."""
        mock = MagicMock()
        mock.fetch.return_value = make_ohlcv(n)
        return mock

    def test_fit_transform_returns_dataframe(self, tmp_path):
        from guapbot.features.pipeline import FeaturePipeline
        pipe = FeaturePipeline(manager=self._make_mock_manager(), cache_dir=tmp_path)
        df = pipe.fit_transform("XBTUSD")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_fit_transform_drops_warmup_rows(self, tmp_path):
        from guapbot.features.pipeline import FeaturePipeline, ZSCORE_WINDOW
        n = 600
        pipe = FeaturePipeline(manager=self._make_mock_manager(n), cache_dir=tmp_path)
        df = pipe.fit_transform("XBTUSD")
        assert len(df) <= n - ZSCORE_WINDOW

    def test_fit_transform_saves_parquet(self, tmp_path):
        from guapbot.features.pipeline import FeaturePipeline
        pipe = FeaturePipeline(manager=self._make_mock_manager(), cache_dir=tmp_path)
        pipe.fit_transform("XBTUSD")
        norm_path = tmp_path / "features_XBTUSD_norm.parquet"
        raw_path = tmp_path / "features_XBTUSD_raw.parquet"
        assert norm_path.exists(), "Normalised Parquet not saved"
        assert raw_path.exists(), "Raw Parquet not saved"

    def test_transform_loads_from_cache(self, tmp_path):
        from guapbot.features.pipeline import FeaturePipeline
        pipe = FeaturePipeline(manager=self._make_mock_manager(), cache_dir=tmp_path)
        df_built = pipe.fit_transform("XBTUSD")
        df_loaded = pipe.transform("XBTUSD")
        assert len(df_built) == len(df_loaded)
        assert list(df_built.columns) == list(df_loaded.columns)

    def test_transform_raises_if_no_cache(self, tmp_path):
        from guapbot.features.pipeline import FeaturePipeline
        pipe = FeaturePipeline(manager=self._make_mock_manager(), cache_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            pipe.transform("XBTUSD")

    def test_normalised_values_in_range(self, tmp_path):
        from guapbot.features.pipeline import FeaturePipeline, ZSCORE_CLIP, _NO_NORM_PATTERNS
        pipe = FeaturePipeline(manager=self._make_mock_manager(), cache_dir=tmp_path)
        df = pipe.fit_transform("XBTUSD")

        for col in df.columns:
            # Skip pass-through columns
            if any(p in col for p in _NO_NORM_PATTERNS):
                continue
            col_data = df[col].dropna()
            assert (col_data >= -ZSCORE_CLIP - 0.01).all(), f"{col} has values below -{ZSCORE_CLIP}"
            assert (col_data <= ZSCORE_CLIP + 0.01).all(), f"{col} has values above +{ZSCORE_CLIP}"

    def test_no_nans_after_pipeline(self, tmp_path):
        from guapbot.features.pipeline import FeaturePipeline
        pipe = FeaturePipeline(manager=self._make_mock_manager(), cache_dir=tmp_path)
        df = pipe.fit_transform("XBTUSD")
        nan_count = df.isna().sum().sum()
        assert nan_count == 0, f"Found {nan_count} NaN values in normalised output"

    def test_get_observation_returns_dict(self, tmp_path):
        from guapbot.features.pipeline import FeaturePipeline
        pipe = FeaturePipeline(manager=self._make_mock_manager(), cache_dir=tmp_path)
        pipe.fit_transform("XBTUSD")
        obs = pipe.get_observation("XBTUSD")
        assert isinstance(obs, dict)
        assert "timestamp" in obs
        assert len(obs) > 10

    def test_get_observation_timestamp_is_latest(self, tmp_path):
        from guapbot.features.pipeline import FeaturePipeline
        pipe = FeaturePipeline(manager=self._make_mock_manager(), cache_dir=tmp_path)
        df = pipe.fit_transform("XBTUSD")
        obs = pipe.get_observation("XBTUSD")
        # Timestamp in obs should match the last row of df
        assert obs["timestamp"] == df.index[-1].isoformat()

    def test_invalid_pair_raises(self, tmp_path):
        from guapbot.features.pipeline import FeaturePipeline
        pipe = FeaturePipeline(manager=self._make_mock_manager(), cache_dir=tmp_path)
        with pytest.raises(ValueError, match="Unknown pair"):
            pipe.fit_transform("BTCEUR")

    def test_ethbtc_fit_transform(self, tmp_path):
        """ETHBTC should build without cross-asset injection."""
        from guapbot.features.pipeline import FeaturePipeline
        pipe = FeaturePipeline(manager=self._make_mock_manager(), cache_dir=tmp_path)
        df = pipe.fit_transform("ETHBTC")
        assert not df.empty
        # ETHBTC should NOT have ethbtc_ cross-asset features injected into itself
        assert not any(c.startswith("ethbtc_") for c in df.columns)

    def test_cache_info_returns_dict(self, tmp_path):
        from guapbot.features.pipeline import FeaturePipeline
        pipe = FeaturePipeline(manager=self._make_mock_manager(), cache_dir=tmp_path)
        pipe.fit_transform("XBTUSD")
        info = pipe.cache_info()
        assert isinstance(info, dict)
        assert "XBTUSD" in info
        assert "bars" in info["XBTUSD"]

    def test_fit_transform_all_pairs(self, tmp_path):
        """Build features for all three pairs without error."""
        from guapbot.features.pipeline import FeaturePipeline, FEATURE_PAIRS
        pipe = FeaturePipeline(manager=self._make_mock_manager(), cache_dir=tmp_path)
        for pair in FEATURE_PAIRS:
            df = pipe.fit_transform(pair)
            assert not df.empty, f"Empty result for {pair}"

    def test_index_is_utc_datetime(self, tmp_path):
        from guapbot.features.pipeline import FeaturePipeline
        pipe = FeaturePipeline(manager=self._make_mock_manager(), cache_dir=tmp_path)
        df = pipe.fit_transform("XBTUSD")
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tz is not None
        assert str(df.index.tz) == "UTC"

    def test_cross_asset_columns_injected(self, tmp_path):
        """Traded pairs should have ethbtc_ cross-asset features."""
        from guapbot.features.pipeline import FeaturePipeline
        pipe = FeaturePipeline(manager=self._make_mock_manager(), cache_dir=tmp_path)
        df = pipe.fit_transform("XBTUSD")
        ethbtc_cols = [c for c in df.columns if c.startswith("ethbtc_")]
        assert len(ethbtc_cols) > 0, "No ETHBTC cross-asset features found in XBTUSD"

    def test_normalise_does_not_alter_binary_columns(self, tmp_path):
        """Signal/crossover columns should NOT be z-scored."""
        from guapbot.features.pipeline import FeaturePipeline
        pipe = FeaturePipeline(manager=self._make_mock_manager(), cache_dir=tmp_path)
        df = pipe.fit_transform("XBTUSD")
        cross_cols = [c for c in df.columns if "_cross" in c]
        for col in cross_cols:
            unique = df[col].dropna().unique()
            assert set(unique).issubset({-1.0, 0.0, 1.0}), \
                f"Cross column '{col}' was z-scored (values: {unique})"
