"""
tests/unit/test_backtest.py

Unit tests for BacktestEngine and BacktestReport (report metrics).

All tests use synthetic data — no real feature cache or API calls.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from guapbot.backtest.engine import BacktestEngine, BacktestResult, _compute_metrics
from guapbot.backtest.report import print_report

# ---------------------------------------------------------------------------
# Helpers / stub model
# ---------------------------------------------------------------------------

class _ConstantModel:
    """Stub model that always returns a fixed signal."""

    def __init__(self, signal: float = 0.0, confidence: float = 0.5):
        self._signal    = signal
        self._confidence = confidence
        self._fitted    = True

    def predict(self, obs) -> float:
        return self._signal

    def confidence(self, obs) -> float:
        return self._confidence

    def __repr__(self) -> str:
        return f"ConstantModel({self._signal})"


N_BARS = 300


@pytest.fixture
def flat_df() -> pd.DataFrame:
    """Feature DataFrame with zero log returns (no PnL expected)."""
    rng = np.random.default_rng(1)
    idx = pd.date_range("2024-01-01", periods=N_BARS, freq="1h", tz="UTC")
    data = {
        "feature_a":     rng.normal(0, 1, N_BARS),
        "feature_b":     rng.normal(0, 1, N_BARS),
        "1h_log_return": np.zeros(N_BARS, dtype=float),
    }
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def trending_up_df() -> pd.DataFrame:
    """Feature DataFrame with consistent positive log returns."""
    idx = pd.date_range("2024-01-01", periods=N_BARS, freq="1h", tz="UTC")
    data = {
        "feature_a":     np.zeros(N_BARS),
        "1h_log_return": np.full(N_BARS, 0.001),  # +0.1% every bar
    }
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def trending_down_df() -> pd.DataFrame:
    """Feature DataFrame with consistent negative log returns."""
    idx = pd.date_range("2024-01-01", periods=N_BARS, freq="1h", tz="UTC")
    data = {
        "feature_a":     np.zeros(N_BARS),
        "1h_log_return": np.full(N_BARS, -0.001),
    }
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# TestBacktestEngine
# ---------------------------------------------------------------------------

class TestBacktestEngine:

    def test_run_returns_result(self, flat_df):
        engine = BacktestEngine(models=[_ConstantModel(0.0)])
        result = engine.run(flat_df)
        assert isinstance(result, BacktestResult)
        assert isinstance(result.equity_curve,    pd.Series)
        assert isinstance(result.position_series, pd.Series)
        assert isinstance(result.trade_log,       pd.DataFrame)
        assert isinstance(result.metrics,         dict)

    def test_equity_curve_length(self, flat_df):
        engine = BacktestEngine(models=[_ConstantModel(0.0)])
        result = engine.run(flat_df)
        assert len(result.equity_curve) == N_BARS

    def test_equity_starts_at_initial_capital(self, flat_df):
        capital = 12_345.0
        engine  = BacktestEngine(models=[_ConstantModel(0.0)], initial_capital=capital)
        result  = engine.run(flat_df)
        # First bar: position=0, no PnL, no fee → equity unchanged
        assert result.equity_curve.iloc[0] == pytest.approx(capital, rel=1e-4)

    def test_flat_signal_zero_pnl(self, trending_up_df):
        """All-zero signals → zero position → equity stays constant."""
        engine = BacktestEngine(
            models=[_ConstantModel(0.0)],
            initial_capital=10_000.0,
            fee_rate=0.0,
        )
        result = engine.run(trending_up_df)
        # Equity should remain within floating-point noise of initial capital
        assert result.equity_curve.min() == pytest.approx(10_000.0, rel=1e-6)
        assert result.equity_curve.max() == pytest.approx(10_000.0, rel=1e-6)

    def test_long_signal_profits_in_up_trend(self, trending_up_df):
        """Full long position in a rising market should grow equity."""
        engine = BacktestEngine(
            models=[_ConstantModel(1.0)],
            initial_capital=10_000.0,
            fee_rate=0.0,
            max_long=0.25,
        )
        result = engine.run(trending_up_df)
        assert result.equity_curve.iloc[-1] > 10_000.0

    def test_fees_reduce_equity_vs_no_fees(self, trending_up_df):
        """Engine with fees should end with less equity than fee-free."""
        models = [_ConstantModel(0.5)]

        result_free = BacktestEngine(models=models, fee_rate=0.0).run(trending_up_df)
        result_paid = BacktestEngine(models=models, fee_rate=0.0026).run(trending_up_df)

        assert result_paid.equity_curve.iloc[-1] < result_free.equity_curve.iloc[-1]

    def test_no_trade_log_when_signal_constant(self, flat_df):
        """Constant signal → position never changes after bar 0 → only one trade entry."""
        engine = BacktestEngine(models=[_ConstantModel(0.5)], fee_rate=0.0026)
        result = engine.run(flat_df)
        # Position changes once (from 0 to 0.5*max_long) — only 1 trade row
        assert len(result.trade_log) == 1

    def test_empty_df_raises(self):
        engine = BacktestEngine(models=[_ConstantModel(0.0)])
        with pytest.raises(ValueError):
            engine.run(pd.DataFrame())

    def test_no_models_raises(self):
        with pytest.raises(ValueError):
            BacktestEngine(models=[])

    def test_position_respects_max_long(self, flat_df):
        """Signal=1.0 should cap position at max_long."""
        max_long = 0.25
        engine   = BacktestEngine(models=[_ConstantModel(1.0)], max_long=max_long, fee_rate=0.0)
        result   = engine.run(flat_df)
        assert result.position_series.max() == pytest.approx(max_long, rel=1e-6)

    def test_position_respects_max_short(self, flat_df):
        """Signal=-1.0 should cap absolute position at max_short."""
        max_short = 0.15
        engine    = BacktestEngine(models=[_ConstantModel(-1.0)], max_short=max_short, fee_rate=0.0)
        result    = engine.run(flat_df)
        assert result.position_series.min() == pytest.approx(-max_short, rel=1e-6)

    def test_target_column_dropped(self, trending_up_df):
        """DataFrames with a 'target' column should not crash."""
        df_with_target = trending_up_df.copy()
        df_with_target["target"] = 1
        engine = BacktestEngine(models=[_ConstantModel(0.0)])
        result = engine.run(df_with_target)
        assert len(result.equity_curve) == N_BARS


# ---------------------------------------------------------------------------
# TestBacktestReport (metrics only — no Plotly required)
# ---------------------------------------------------------------------------

class TestBacktestReport:

    def _make_result(self, equity_values: list[float]) -> BacktestResult:
        idx = pd.date_range("2024-01-01", periods=len(equity_values), freq="1h", tz="UTC")
        equity = pd.Series(equity_values, index=idx, name="equity")
        trades = pd.DataFrame(columns=["time", "signal", "position", "pnl_pct", "fee_pct"])
        metrics = _compute_metrics(equity, trades)
        return BacktestResult(
            equity_curve=equity,
            position_series=pd.Series(np.zeros(len(equity_values)), index=idx),
            trade_log=trades,
            metrics=metrics,
        )

    def test_sharpe_positive_for_positive_drift(self):
        # Positive drift 0.1%/bar + noise: mean>0, std>0 → Sharpe > 0
        rng = np.random.default_rng(7)
        returns = rng.normal(0.001, 0.003, 200)
        equity = [10_000.0]
        for r in returns:
            equity.append(equity[-1] * (1 + r))
        result = self._make_result(equity)
        assert result.metrics["sharpe"] > 0.0

    def test_sharpe_negative_for_negative_drift(self):
        # Negative drift: mean<0, std>0 → Sharpe < 0
        rng = np.random.default_rng(7)
        returns = rng.normal(-0.001, 0.003, 200)
        equity = [10_000.0]
        for r in returns:
            equity.append(equity[-1] * (1 + r))
        result = self._make_result(equity)
        assert result.metrics["sharpe"] < 0.0

    def test_max_drawdown_nonpositive(self):
        import random
        rng = random.Random(42)
        equity = [10_000.0]
        for _ in range(199):
            equity.append(equity[-1] * (1 + rng.uniform(-0.01, 0.008)))
        result = self._make_result(equity)
        assert result.metrics["max_drawdown"] <= 0.0

    def test_max_drawdown_zero_for_monotone_up(self):
        equity = [10_000 + i for i in range(200)]
        result = self._make_result(equity)
        assert result.metrics["max_drawdown"] == pytest.approx(0.0, abs=1e-8)

    def test_total_return_matches_equity(self):
        # Use enough bars that ann_factor doesn't overflow
        equity = [10_000.0 * (1.001 ** i) for i in range(500)]
        result = self._make_result(equity)
        expected = equity[-1] / equity[0] - 1
        assert result.metrics["total_return"] == pytest.approx(expected, rel=1e-5)

    def test_print_report_does_not_raise(self, capsys):
        """print_report() should produce output without errors."""
        equity = [10_000 * (1.0005 ** i) for i in range(100)]
        result = self._make_result(equity)
        print_report(result, title="Test")
        captured = capsys.readouterr()
        assert "Sharpe" in captured.out

    def test_save_html_raises_without_plotly(self, tmp_path, monkeypatch):
        """save_html raises ImportError when plotly is unavailable."""
        import sys
        import importlib
        from guapbot.backtest import report as report_mod

        # Temporarily hide plotly
        original = sys.modules.get("plotly")
        sys.modules["plotly"] = None  # type: ignore[assignment]
        sys.modules["plotly.graph_objects"] = None  # type: ignore[assignment]
        sys.modules["plotly.subplots"] = None  # type: ignore[assignment]

        equity = [10_000.0] * 50
        result = self._make_result(equity)

        try:
            with pytest.raises((ImportError, TypeError)):
                report_mod.save_html(result, str(tmp_path / "r.html"))
        finally:
            # Restore
            if original is None:
                sys.modules.pop("plotly", None)
                sys.modules.pop("plotly.graph_objects", None)
                sys.modules.pop("plotly.subplots", None)
            else:
                sys.modules["plotly"] = original
