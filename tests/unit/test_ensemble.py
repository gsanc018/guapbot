"""
tests/unit/test_ensemble.py

Unit tests for EnsembleLearner (ensemble_lightgbm.py).

All tests use synthetic data — no real feature cache or API calls.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# LightGBM requires libomp on macOS; skip gracefully if unavailable
try:
    from guapbot.models.ensemble_lightgbm import (
        EnsembleLearner,
        _FEATURE_COLS,
        _MODEL_NAMES,
        _ONLINE_BUFFER_SZ,
        _LGB_AVAILABLE,
    )
except Exception:
    _LGB_AVAILABLE = False
    EnsembleLearner = None  # type: ignore[assignment,misc]
    _FEATURE_COLS = []
    _MODEL_NAMES = []
    _ONLINE_BUFFER_SZ = 24

lgb_skip = pytest.mark.skipif(
    not _LGB_AVAILABLE,
    reason="LightGBM unavailable (brew install libomp on macOS)",
)

from guapbot.models.ensemble import EnsembleInput, ModelSignal, TradeOutcome
from guapbot.regime.base import RegimeResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_BARS = 200


@pytest.fixture
def signal_history_df() -> pd.DataFrame:
    """Synthetic signal_history DataFrame matching EnsembleLearner._FEATURE_COLS + target."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2022-01-01", periods=N_BARS, freq="1h", tz="UTC")
    data = {col: rng.uniform(-1, 1, N_BARS) for col in _FEATURE_COLS}
    # Confidence and rolling_sharpe columns should be positive
    for m in _MODEL_NAMES:
        data[f"{m}_confidence"]    = rng.uniform(0, 1,  N_BARS)
        data[f"{m}_rolling_sharpe"] = rng.uniform(-2, 2, N_BARS)
    # Regime label columns are ints 0-2
    for tf in ("1h", "4h", "daily"):
        data[f"regime_{tf}_label"]      = rng.integers(0, 3, N_BARS).astype(float)
        data[f"regime_{tf}_confidence"] = rng.uniform(0, 1, N_BARS)
    # Binary target
    data["target"] = rng.choice([0, 1], size=N_BARS).astype(float)
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def fitted_learner(signal_history_df) -> EnsembleLearner:
    """Pre-fitted EnsembleLearner on synthetic data."""
    learner = EnsembleLearner("XBTUSD")
    learner.fit(signal_history_df)
    return learner


@pytest.fixture
def ensemble_input() -> EnsembleInput:
    """Minimal EnsembleInput with all 5 model signals at neutral values."""
    signals = [
        ModelSignal(model_name=m, pair="XBTUSD", signal=0.1, confidence=0.5)
        for m in _MODEL_NAMES
    ]
    regimes = [
        RegimeResult(label="trending", confidence=0.6, timeframe="1h"),
        RegimeResult(label="ranging",  confidence=0.5, timeframe="4h"),
        RegimeResult(label="bullish",  confidence=0.7, timeframe="daily"),
    ]
    return EnsembleInput(signals=signals, regimes=regimes, pair="XBTUSD")


def _make_outcome(signal: float = 0.1, ret: float = 0.005) -> TradeOutcome:
    return TradeOutcome(
        bar_time="2025-01-01T00:00:00",
        pair="XBTUSD",
        final_signal=signal,
        realised_return=ret,
        regime_labels={"1h": "trending", "4h": "ranging", "daily": "bullish"},
        model_signals={m: signal for m in _MODEL_NAMES},
    )


# ---------------------------------------------------------------------------
# TestEnsembleLearner
# ---------------------------------------------------------------------------

@lgb_skip
class TestEnsembleLearner:

    def test_interface_compliance(self, signal_history_df, ensemble_input):
        """Unfitted → RuntimeError on combine(); fit → _fitted=True."""
        learner = EnsembleLearner("XBTUSD")
        assert not learner._fitted

        with pytest.raises(RuntimeError):
            learner.combine(ensemble_input)

        learner.fit(signal_history_df)
        assert learner._fitted

    def test_fit_returns_self(self, signal_history_df):
        learner = EnsembleLearner("XBTUSD")
        result  = learner.fit(signal_history_df)
        assert result is learner

    def test_combine_in_range(self, fitted_learner, ensemble_input):
        """Output of combine() must be in [-1, 1]."""
        signal = fitted_learner.combine(ensemble_input)
        assert isinstance(signal, float)
        assert -1.0 <= signal <= 1.0

    def test_combine_missing_model_graceful(self, fitted_learner):
        """combine() with only 2 of 5 models should still return valid signal."""
        partial_input = EnsembleInput(
            signals=[
                ModelSignal("trend_following", "XBTUSD", 0.5, 0.7),
                ModelSignal("lstm",            "XBTUSD", -0.3, 0.6),
            ],
            regimes=[
                RegimeResult("trending", 0.5, "1h"),
                RegimeResult("trending", 0.5, "4h"),
                RegimeResult("bullish",  0.5, "daily"),
            ],
            pair="XBTUSD",
        )
        signal = fitted_learner.combine(partial_input)
        assert -1.0 <= signal <= 1.0

    def test_update_accumulates_buffer(self, fitted_learner):
        """23 outcomes → buffer stays; 24th outcome → buffer is flushed."""
        for _ in range(_ONLINE_BUFFER_SZ - 1):
            fitted_learner.update(_make_outcome())
        assert len(fitted_learner._outcome_buffer) == _ONLINE_BUFFER_SZ - 1

        fitted_learner.update(_make_outcome())
        assert len(fitted_learner._outcome_buffer) == 0  # flushed

    def test_update_does_not_break_combine(self, fitted_learner, ensemble_input):
        """After an online update, combine() should still work."""
        for _ in range(_ONLINE_BUFFER_SZ):
            fitted_learner.update(_make_outcome())

        signal = fitted_learner.combine(ensemble_input)
        assert -1.0 <= signal <= 1.0

    def test_save_load_round_trip(self, fitted_learner, ensemble_input, tmp_path):
        """Predictions must match between original and loaded model."""
        path = str(tmp_path / "ensemble.pkl")
        sig_before = fitted_learner.combine(ensemble_input)

        fitted_learner.save(path)

        loaded = EnsembleLearner("XBTUSD")
        loaded.load(path)
        sig_after = loaded.combine(ensemble_input)

        assert sig_after == pytest.approx(sig_before, abs=1e-5)

    def test_save_unfitted_raises(self, tmp_path):
        learner = EnsembleLearner("XBTUSD")
        with pytest.raises(RuntimeError):
            learner.save(str(tmp_path / "ensemble.pkl"))

    def test_fit_raises_without_target(self, signal_history_df):
        learner = EnsembleLearner("XBTUSD")
        df_no_target = signal_history_df.drop(columns=["target"])
        with pytest.raises(ValueError):
            learner.fit(df_no_target)

    def test_repr(self, signal_history_df):
        learner = EnsembleLearner("XBTUSD")
        assert "unfitted" in repr(learner)
        learner.fit(signal_history_df)
        assert "fitted" in repr(learner)
