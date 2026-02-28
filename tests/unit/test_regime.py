"""
tests/unit/test_regime.py

Unit tests for Session 4: Regime Layer.
No real API calls — LLM labeler is mocked.
HMM training uses tiny synthetic datasets (fast, < 1s).

Run with: pytest tests/unit/test_regime.py -v
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from guapbot.regime.base import RegimeDetector, RegimeResult
from guapbot.regime.detector import Detector, build_regime_detectors
from guapbot.regime.rule_labeler import RuleLabeler
from guapbot.regime.statistical import REGIME_FEATURES, HMMDetector


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_labeled_df(
    n: int = 100,
    timeframe: str = "1h",
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic feature DataFrame with a valid 'label' column."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")

    labels = (
        ["trending", "ranging", "volatile"]
        if timeframe != "daily"
        else ["bullish", "bearish", "neutral"]
    )
    label_col = [labels[i % 3] for i in range(n)]

    data = {feat: rng.uniform(0, 1, n) for feat in REGIME_FEATURES}
    data["label"] = label_col
    # Add a 'close' column for LLM prompt building
    data["close"] = rng.uniform(40_000, 60_000, n)

    return pd.DataFrame(data, index=pd.DatetimeIndex(idx, name="timestamp"))


def make_obs(seed: int = 0) -> dict:
    """Synthetic single-bar observation dict."""
    rng = np.random.default_rng(seed)
    return {feat: float(rng.uniform(0, 1)) for feat in REGIME_FEATURES}


# ------------------------------------------------------------------
# TestHMMDetector
# ------------------------------------------------------------------

class TestHMMDetector:

    def test_construction_valid_timeframes(self) -> None:
        for tf in ("1h", "4h", "daily"):
            d = HMMDetector(timeframe=tf)
            assert d.timeframe == tf
            assert not d._fitted

    def test_construction_invalid_timeframe_raises(self) -> None:
        with pytest.raises(ValueError, match="timeframe"):
            HMMDetector(timeframe="5m")

    def test_fit_sets_fitted_flag(self) -> None:
        df = make_labeled_df(100, "1h")
        d = HMMDetector(timeframe="1h")
        result = d.fit(df)
        assert d._fitted
        assert result is d  # returns self

    def test_fit_resolves_state_labels(self) -> None:
        df = make_labeled_df(100, "1h")
        d = HMMDetector(timeframe="1h")
        d.fit(df)
        assert len(d._state_to_label) == 3
        valid = {"trending", "ranging", "volatile"}
        assert set(d._state_to_label.values()) <= valid

    def test_fit_daily_uses_daily_labels(self) -> None:
        df = make_labeled_df(100, "daily")
        d = HMMDetector(timeframe="daily")
        d.fit(df)
        valid = {"bullish", "bearish", "neutral"}
        assert set(d._state_to_label.values()) <= valid

    def test_fit_raises_missing_feature_columns(self) -> None:
        df = make_labeled_df(100, "1h")
        df = df.drop(columns=["adx_14"])
        d = HMMDetector(timeframe="1h")
        with pytest.raises(ValueError, match="adx_14"):
            d.fit(df)

    def test_fit_raises_missing_label_column(self) -> None:
        df = make_labeled_df(100, "1h").drop(columns=["label"])
        d = HMMDetector(timeframe="1h")
        with pytest.raises(ValueError, match="'label'"):
            d.fit(df)

    def test_fit_raises_invalid_labels(self) -> None:
        df = make_labeled_df(100, "1h")
        df["label"] = "unknown_regime"  # inject bad labels
        d = HMMDetector(timeframe="1h")
        with pytest.raises(ValueError, match="Unknown labels"):
            d.fit(df)

    def test_fit_raises_too_few_rows(self) -> None:
        df = make_labeled_df(10, "1h")
        d = HMMDetector(timeframe="1h")
        with pytest.raises(ValueError, match="50"):
            d.fit(df)

    def test_detect_returns_regime_result(self) -> None:
        df = make_labeled_df(100, "1h")
        d = HMMDetector(timeframe="1h").fit(df)
        result = d.detect(make_obs())
        assert isinstance(result, RegimeResult)
        assert result.timeframe == "1h"
        assert result.label in {"trending", "ranging", "volatile"}
        assert 0.0 <= result.confidence <= 1.0

    def test_detect_with_series_obs(self) -> None:
        df = make_labeled_df(100, "1h")
        d = HMMDetector(timeframe="1h").fit(df)
        obs_series = pd.Series(make_obs())
        result = d.detect(obs_series)
        assert isinstance(result, RegimeResult)

    def test_detect_unfitted_raises(self) -> None:
        d = HMMDetector(timeframe="1h")
        with pytest.raises(RuntimeError, match="fit()"):
            d.detect(make_obs())

    def test_detect_batch_via_base_class(self) -> None:
        df = make_labeled_df(100, "1h")
        d = HMMDetector(timeframe="1h").fit(df)
        results = d.detect_batch(df[REGIME_FEATURES])
        assert len(results) == 100
        assert all(isinstance(r, RegimeResult) for r in results)

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        df = make_labeled_df(100, "1h")
        d = HMMDetector(timeframe="1h").fit(df)
        path = tmp_path / "test_hmm.pkl"
        d.save(path)
        assert path.exists()

        d2 = HMMDetector(timeframe="1h")
        d2.load(path)
        assert d2._fitted
        assert d2._state_to_label == d._state_to_label

        # Both should agree on the same observation
        obs = make_obs()
        r1 = d.detect(obs)
        r2 = d2.detect(obs)
        assert r1.label == r2.label
        assert abs(r1.confidence - r2.confidence) < 1e-6

    def test_save_unfitted_raises(self, tmp_path: Path) -> None:
        d = HMMDetector(timeframe="1h")
        with pytest.raises(RuntimeError, match="unfitted"):
            d.save(tmp_path / "bad.pkl")

    def test_repr_shows_fitted_status(self) -> None:
        d = HMMDetector(timeframe="4h")
        assert "unfitted" in repr(d)
        d.fit(make_labeled_df(100, "4h"))
        assert "fitted" in repr(d)


# ------------------------------------------------------------------
# TestDetector (wrapper)
# ------------------------------------------------------------------

class TestDetector:

    def test_construction(self) -> None:
        d = Detector(timeframe="1h")
        assert d.timeframe == "1h"
        assert not d._fitted

    def test_detect_unfitted_returns_fallback(self) -> None:
        """Unfitted detector must NOT raise — it returns low-confidence fallback."""
        d = Detector(timeframe="1h")
        result = d.detect(make_obs())
        assert isinstance(result, RegimeResult)
        assert result.label == "ranging"
        assert result.confidence == 0.0
        assert result.timeframe == "1h"

    def test_detect_unfitted_daily_fallback(self) -> None:
        d = Detector(timeframe="daily")
        result = d.detect(make_obs())
        assert result.label == "neutral"
        assert result.confidence == 0.0

    def test_fit_then_detect_works(self) -> None:
        df = make_labeled_df(100, "1h")
        d = Detector(timeframe="1h")
        d.fit(df)
        assert d._fitted
        result = d.detect(make_obs())
        assert isinstance(result, RegimeResult)
        assert result.confidence > 0.0  # real prediction, not fallback

    def test_save_and_load(self, tmp_path: Path) -> None:
        df = make_labeled_df(100, "1h")
        d = Detector(timeframe="1h", model_dir=tmp_path)
        d.fit(df)
        d.save("XBTUSD")

        model_file = tmp_path / "XBTUSD_1h.pkl"
        assert model_file.exists()

        d2 = Detector(timeframe="1h", model_dir=tmp_path)
        d2.load("XBTUSD")
        assert d2._fitted

    def test_load_raises_when_no_file(self, tmp_path: Path) -> None:
        d = Detector(timeframe="1h", model_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            d.load("XBTUSD")


# ------------------------------------------------------------------
# TestBuildRegimeDetectors
# ------------------------------------------------------------------

class TestBuildRegimeDetectors:

    def test_returns_three_detectors(self, tmp_path: Path) -> None:
        detectors = build_regime_detectors("XBTUSD", model_dir=tmp_path, auto_load=False)
        assert len(detectors) == 3

    def test_timeframes_are_1h_4h_daily(self, tmp_path: Path) -> None:
        detectors = build_regime_detectors("XBTUSD", model_dir=tmp_path, auto_load=False)
        assert [d.timeframe for d in detectors] == ["1h", "4h", "daily"]

    def test_auto_load_false_returns_unfitted(self, tmp_path: Path) -> None:
        detectors = build_regime_detectors("XBTUSD", model_dir=tmp_path, auto_load=False)
        assert all(not d._fitted for d in detectors)

    def test_auto_load_true_no_files_returns_unfitted(self, tmp_path: Path) -> None:
        """Should not crash when no saved models exist."""
        detectors = build_regime_detectors("XBTUSD", model_dir=tmp_path, auto_load=True)
        assert all(not d._fitted for d in detectors)

    def test_auto_load_true_loads_saved_models(self, tmp_path: Path) -> None:
        # Save all three
        for tf in ("1h", "4h", "daily"):
            df = make_labeled_df(100, tf)
            d = Detector(timeframe=tf, model_dir=tmp_path)
            d.fit(df)
            d.save("XBTUSD")

        detectors = build_regime_detectors("XBTUSD", model_dir=tmp_path, auto_load=True)
        assert all(d._fitted for d in detectors)

    def test_detect_all_returns_regime_vector(self, tmp_path: Path) -> None:
        detectors = build_regime_detectors("XBTUSD", model_dir=tmp_path, auto_load=False)
        obs = make_obs()
        regime_vector = [d.detect(obs) for d in detectors]
        assert len(regime_vector) == 3
        assert all(isinstance(r, RegimeResult) for r in regime_vector)
        assert [r.timeframe for r in regime_vector] == ["1h", "4h", "daily"]


# ------------------------------------------------------------------
# TestRuleLabeler
# ------------------------------------------------------------------

def make_feature_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Synthetic feature DataFrame with realistic indicator values."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "adx_14": rng.uniform(10, 40, n),
            "atr_14_pct": rng.uniform(0.005, 0.04, n),
            "bb_width_20_2": rng.uniform(0.02, 0.12, n),
            "rsi_14": rng.uniform(30, 70, n),
            "price_vs_ema21": rng.uniform(-0.05, 0.05, n),
            "volume_ratio": rng.uniform(0.5, 2.0, n),
            "log_ret_1": rng.uniform(-0.03, 0.03, n),
        },
        index=pd.DatetimeIndex(idx, name="timestamp"),
    )


class TestRuleLabeler:

    def test_construction_valid_timeframes(self) -> None:
        for tf in ("1h", "4h", "daily"):
            labeler = RuleLabeler(timeframe=tf)
            assert labeler.timeframe == tf

    def test_construction_invalid_timeframe_raises(self) -> None:
        with pytest.raises(ValueError, match="timeframe"):
            RuleLabeler(timeframe="5m")

    def test_label_dataframe_adds_label_column(self) -> None:
        labeler = RuleLabeler(timeframe="1h")
        df = make_feature_df(100)
        result = labeler.label_dataframe(df)
        assert "label" in result.columns
        assert len(result) == 100

    def test_label_dataframe_no_nans(self) -> None:
        labeler = RuleLabeler(timeframe="1h")
        result = labeler.label_dataframe(make_feature_df(100))
        assert result["label"].notna().all()

    def test_intraday_labels_are_valid(self) -> None:
        for tf in ("1h", "4h"):
            labeler = RuleLabeler(timeframe=tf)
            result = labeler.label_dataframe(make_feature_df(100))
            assert set(result["label"].unique()) <= {"trending", "ranging", "volatile"}

    def test_daily_labels_are_valid(self) -> None:
        labeler = RuleLabeler(timeframe="daily")
        result = labeler.label_dataframe(make_feature_df(100))
        assert set(result["label"].unique()) <= {"bullish", "bearish", "neutral"}

    def test_high_atr_labelled_volatile(self) -> None:
        """Bars with ATR% well above threshold should all be volatile."""
        df = make_feature_df(50)
        df["atr_14_pct"] = 0.05          # well above _VOLATILE_ATR_PCT=0.025
        df["bb_width_20_2"] = 0.02       # below bb threshold
        df["adx_14"] = 10.0              # low ADX
        labeler = RuleLabeler(timeframe="1h")
        result = labeler.label_dataframe(df)
        assert (result["label"] == "volatile").all()

    def test_high_adx_low_atr_labelled_trending(self) -> None:
        """Bars with strong ADX and low volatility should be trending."""
        df = make_feature_df(50)
        df["adx_14"] = 35.0              # above _TRENDING_ADX=25
        df["atr_14_pct"] = 0.010         # below volatile threshold
        df["bb_width_20_2"] = 0.03       # below volatile threshold
        labeler = RuleLabeler(timeframe="1h")
        result = labeler.label_dataframe(df)
        assert (result["label"] == "trending").all()

    def test_low_adx_low_atr_labelled_ranging(self) -> None:
        """Bars with weak trend and low volatility should be ranging."""
        df = make_feature_df(50)
        df["adx_14"] = 12.0              # below trending threshold
        df["atr_14_pct"] = 0.008         # below volatile threshold
        df["bb_width_20_2"] = 0.03       # below volatile threshold
        labeler = RuleLabeler(timeframe="1h")
        result = labeler.label_dataframe(df)
        assert (result["label"] == "ranging").all()

    def test_volatile_beats_trending(self) -> None:
        """High ATR + high ADX → volatile wins (not trending)."""
        df = make_feature_df(20)
        df["adx_14"] = 35.0
        df["atr_14_pct"] = 0.05          # volatile threshold hit
        labeler = RuleLabeler(timeframe="1h")
        result = labeler.label_dataframe(df)
        assert (result["label"] == "volatile").all()

    def test_daily_bullish_on_high_rsi_above_ema(self) -> None:
        df = make_feature_df(50)
        df["rsi_14"] = 62.0
        df["price_vs_ema21"] = 0.03      # above EMA
        labeler = RuleLabeler(timeframe="daily")
        result = labeler.label_dataframe(df)
        assert (result["label"] == "bullish").all()

    def test_daily_bearish_on_low_rsi_below_ema(self) -> None:
        df = make_feature_df(50)
        df["rsi_14"] = 38.0
        df["price_vs_ema21"] = -0.03     # below EMA
        labeler = RuleLabeler(timeframe="daily")
        result = labeler.label_dataframe(df)
        assert (result["label"] == "bearish").all()

    def test_daily_neutral_without_ema_col(self) -> None:
        """Without price_vs_ema21, RSI alone controls the label."""
        df = make_feature_df(50)
        df["rsi_14"] = 50.0              # neutral RSI
        df = df.drop(columns=["price_vs_ema21"])
        labeler = RuleLabeler(timeframe="daily")
        result = labeler.label_dataframe(df)
        assert (result["label"] == "neutral").all()

    def test_missing_atr_col_still_runs(self) -> None:
        """Missing optional columns should not crash — just skip that rule."""
        df = make_feature_df(50).drop(columns=["atr_14_pct"])
        labeler = RuleLabeler(timeframe="1h")
        result = labeler.label_dataframe(df)
        assert "label" in result.columns

    def test_output_does_not_modify_input(self) -> None:
        df = make_feature_df(50)
        original_cols = set(df.columns)
        labeler = RuleLabeler(timeframe="1h")
        labeler.label_dataframe(df)
        assert set(df.columns) == original_cols  # input unchanged
