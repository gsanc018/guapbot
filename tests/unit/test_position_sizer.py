"""
tests/unit/test_position_sizer.py

Unit tests for DefaultPositionSizer — all 5 sizing layers.

All tests use synthetic inputs — no real data or API calls.
"""
from __future__ import annotations

import pytest

from guapbot.execution.position_sizer import AlertState, DefaultPositionSizer, SizingResult
from guapbot.regime.base import RegimeResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sizer() -> DefaultPositionSizer:
    return DefaultPositionSizer(max_long=0.25, max_short=0.15)


def _no_alert() -> AlertState:
    return AlertState(active=False)


def _active_alert(max_pos: float = 0.05) -> AlertState:
    return AlertState(active=True, severity=0.8, max_position=max_pos)


def _regimes(conf: float = 1.0) -> list[RegimeResult]:
    return [
        RegimeResult(label="trending", confidence=conf, timeframe="1h"),
        RegimeResult(label="trending", confidence=conf, timeframe="4h"),
        RegimeResult(label="bullish",  confidence=conf, timeframe="daily"),
    ]


# ---------------------------------------------------------------------------
# TestDefaultPositionSizer
# ---------------------------------------------------------------------------

class TestDefaultPositionSizer:

    def test_returns_sizing_result(self, sizer):
        result = sizer.size(0.5, _regimes(), _no_alert())
        assert isinstance(result, SizingResult)

    def test_zero_signal_zero_position(self, sizer):
        result = sizer.size(0.0, _regimes(), _no_alert())
        assert result.position == pytest.approx(0.0, abs=1e-9)

    def test_full_long_signal_caps_at_max_long(self, sizer):
        """signal=1.0, perfect Kelly, low ATR, high regime conf → max_long."""
        result = sizer.size(
            1.0,
            _regimes(conf=1.0),
            _no_alert(),
            atr=0.02,       # exactly at ATR target → no ATR scaling
            win_rate=0.6,
            avg_win=0.05,
            avg_loss=0.02,
        )
        assert result.position <= sizer.max_long
        assert result.position > 0.0

    def test_full_short_signal_negative(self, sizer):
        """signal=-1.0 should produce a negative position."""
        result = sizer.size(-1.0, _regimes(), _no_alert(), atr=0.02)
        assert result.position < 0.0
        assert result.position >= -sizer.max_short

    def test_hard_cap_long_not_exceeded(self, sizer):
        """Even with perfect conditions, position must not exceed max_long."""
        result = sizer.size(
            1.0, _regimes(conf=1.0), _no_alert(),
            atr=0.001,      # extremely low ATR → would scale up without cap
            win_rate=0.9, avg_win=0.1, avg_loss=0.001,
        )
        assert result.position <= sizer.max_long

    def test_hard_cap_short_not_exceeded(self, sizer):
        result = sizer.size(
            -1.0, _regimes(conf=1.0), _no_alert(),
            atr=0.001,
            win_rate=0.9, avg_win=0.1, avg_loss=0.001,
        )
        assert result.position >= -sizer.max_short

    def test_low_regime_confidence_reduces_position(self, sizer):
        """Low regime confidence should reduce position vs high confidence."""
        high_conf = sizer.size(0.8, _regimes(conf=1.0), _no_alert(), atr=0.02)
        low_conf  = sizer.size(0.8, _regimes(conf=0.2), _no_alert(), atr=0.02)
        assert low_conf.position < high_conf.position

    def test_alert_active_caps_position(self, sizer):
        """Active alert should cap position at alert.max_position."""
        cap = 0.03
        result = sizer.size(
            1.0, _regimes(), _active_alert(max_pos=cap),
            atr=0.02,
        )
        assert abs(result.position) <= cap + 1e-9

    def test_alert_active_short_caps_position(self, sizer):
        cap = 0.03
        result = sizer.size(
            -1.0, _regimes(), _active_alert(max_pos=cap),
            atr=0.02,
        )
        assert abs(result.position) <= cap + 1e-9

    def test_high_atr_reduces_position(self, sizer):
        """High ATR (2× target) should produce smaller position than normal ATR."""
        normal = sizer.size(0.8, _regimes(), _no_alert(), atr=0.02)
        high   = sizer.size(0.8, _regimes(), _no_alert(), atr=0.10)
        assert high.position < normal.position

    def test_negative_kelly_becomes_zero(self, sizer):
        """Kelly formula can return negative when edge is negative; should clamp to 0."""
        result = sizer.size(
            0.8, _regimes(), _no_alert(),
            atr=0.02,
            win_rate=0.1,    # terrible win rate
            avg_win=0.001,
            avg_loss=0.1,
        )
        # Position should be 0 because half-Kelly clamps negative Kelly to 0
        assert result.position == pytest.approx(0.0, abs=1e-9)

    def test_audit_trail_populated(self, sizer):
        """SizingResult should have all audit fields filled."""
        result = sizer.size(0.5, _regimes(), _no_alert(), atr=0.02)
        assert result.base_signal != 0.0
        assert result.after_kelly != 0.0
        assert result.after_caps  == result.position

    def test_side_set_correctly(self, sizer):
        long_r  = sizer.size( 0.5, _regimes(), _no_alert(), atr=0.02)
        short_r = sizer.size(-0.5, _regimes(), _no_alert(), atr=0.02)
        flat_r  = sizer.size( 0.0, _regimes(), _no_alert(), atr=0.02)
        assert long_r.side  == "long"
        assert short_r.side == "short"
        assert flat_r.side  == "flat"

    def test_invalid_signal_raises(self, sizer):
        with pytest.raises(ValueError):
            from guapbot.execution.position_sizer import SizingContext
            SizingContext(signal=1.5)   # > 1.0 is invalid

    def test_alert_state_severity_out_of_range_raises(self):
        with pytest.raises(ValueError):
            AlertState(active=True, severity=1.5)

    def test_zero_max_long_disables_longs(self):
        """max_long=0.0 must be respected — not replaced by settings default."""
        sizer = DefaultPositionSizer(max_long=0.0, max_short=0.15)
        result = sizer.size(1.0, _regimes(), _no_alert(), atr=0.02)
        assert result.position == pytest.approx(0.0, abs=1e-9)

    def test_zero_max_short_disables_shorts(self):
        """max_short=0.0 must be respected — not replaced by settings default."""
        sizer = DefaultPositionSizer(max_long=0.25, max_short=0.0)
        result = sizer.size(-1.0, _regimes(), _no_alert(), atr=0.02)
        assert result.position == pytest.approx(0.0, abs=1e-9)
