"""
tests/unit/test_paper_trader.py

Unit tests for OrderManager, OrderExecutor, and PaperTrader.

All tests use synthetic data — no real feature cache, Redis, or API calls.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from guapbot.execution.order_manager import Order, OrderManager
from guapbot.execution.order_executor import OrderExecutor
from guapbot.execution.paper_trader import BarStats, PaperTrader


# ---------------------------------------------------------------------------
# Helpers / stub model
# ---------------------------------------------------------------------------

class _ConstantModel:
    """Stub model that always returns a fixed signal."""

    def __init__(self, signal: float = 0.0, confidence: float = 0.5):
        self._signal     = signal
        self._confidence = confidence
        self._fitted     = True

    def predict(self, obs) -> float:
        return self._signal

    def confidence(self, obs) -> float:
        return self._confidence


N_BARS = 100


@pytest.fixture
def flat_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=N_BARS, freq="1h", tz="UTC")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "feature_a":     rng.normal(0, 1, N_BARS),
            "1h_log_return": np.zeros(N_BARS),
        },
        index=idx,
    )


@pytest.fixture
def trending_up_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=N_BARS, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "feature_a":     np.zeros(N_BARS),
            "1h_log_return": np.full(N_BARS, 0.001),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# TestOrderManager
# ---------------------------------------------------------------------------

class TestOrderManager:

    def test_submit_creates_open_order(self):
        om = OrderManager()
        order = Order(pair="XBTUSD", side="buy", qty=0.01)
        om.submit(order)
        assert len(om.get_open()) == 1
        assert om.get_open()[0].status == "open"

    def test_fill_closes_order(self):
        om = OrderManager()
        order = Order(pair="XBTUSD", side="buy", qty=0.01)
        om.submit(order)
        filled = om.fill(order.order_id, fill_price=50_000.0)
        assert filled.status == "filled"
        assert filled.fill_price == pytest.approx(50_000.0)
        assert len(om.get_open()) == 0
        assert len(om.get_history()) == 1

    def test_cancel_removes_open_order(self):
        om = OrderManager()
        order = Order(pair="XBTUSD", side="sell", qty=0.01)
        om.submit(order)
        result = om.cancel(order.order_id)
        assert result is True
        assert len(om.get_open()) == 0
        history = om.get_history()
        assert history[0].status == "cancelled"

    def test_fill_unknown_order_raises(self):
        om = OrderManager()
        with pytest.raises(KeyError):
            om.fill("nonexistent-id", fill_price=1.0)

    def test_cancel_unknown_returns_false(self):
        om = OrderManager()
        assert om.cancel("nonexistent-id") is False

    def test_n_trades_counts_fills_only(self):
        om = OrderManager()
        o1 = Order(pair="XBTUSD", side="buy", qty=0.01)
        o2 = Order(pair="XBTUSD", side="sell", qty=0.01)
        om.submit(o1); om.submit(o2)
        om.fill(o1.order_id, 50_000.0)
        om.cancel(o2.order_id)
        assert om.n_trades() == 1

    def test_cancel_all_open(self):
        om = OrderManager()
        for _ in range(5):
            om.submit(Order(pair="XBTUSD", side="buy", qty=0.001))
        cancelled = om.cancel_all_open()
        assert cancelled == 5
        assert len(om.get_open()) == 0

    def test_invalid_side_raises(self):
        with pytest.raises(ValueError):
            Order(pair="XBTUSD", side="hold", qty=0.01)

    def test_invalid_qty_raises(self):
        with pytest.raises(ValueError):
            Order(pair="XBTUSD", side="buy", qty=0.0)


# ---------------------------------------------------------------------------
# TestOrderExecutor
# ---------------------------------------------------------------------------

class TestOrderExecutor:

    def test_execute_returns_order_on_significant_delta(self):
        om  = OrderManager()
        exc = OrderExecutor(om, slippage=0.0, min_delta=0.001)
        order = exc.execute(
            pair="XBTUSD",
            target_position=0.10,
            current_position=0.0,
            mid_price=50_000.0,
            equity=10_000.0,
        )
        assert order is not None
        assert order.is_filled
        assert order.side == "buy"

    def test_execute_returns_none_below_threshold(self):
        om  = OrderManager()
        exc = OrderExecutor(om, min_delta=0.01)
        order = exc.execute(
            pair="XBTUSD",
            target_position=0.001,   # tiny delta
            current_position=0.0,
            mid_price=50_000.0,
            equity=10_000.0,
        )
        assert order is None

    def test_sell_order_for_negative_delta(self):
        om  = OrderManager()
        exc = OrderExecutor(om, slippage=0.0, min_delta=0.001)
        order = exc.execute(
            pair="XBTUSD",
            target_position=0.0,
            current_position=0.10,
            mid_price=50_000.0,
            equity=10_000.0,
        )
        assert order is not None
        assert order.side == "sell"

    def test_slippage_increases_buy_price(self):
        om  = OrderManager()
        exc = OrderExecutor(om, slippage=0.001, min_delta=0.001)
        order = exc.execute(
            pair="XBTUSD",
            target_position=0.10,
            current_position=0.0,
            mid_price=50_000.0,
            equity=10_000.0,
        )
        assert order.fill_price > 50_000.0

    def test_zero_price_returns_none(self):
        om  = OrderManager()
        exc = OrderExecutor(om)
        order = exc.execute("XBTUSD", 0.1, 0.0, mid_price=0.0, equity=10_000.0)
        assert order is None

    def test_zero_equity_returns_none(self):
        """Zero or negative equity must return None, not raise ValueError from Order(qty=0)."""
        om  = OrderManager()
        exc = OrderExecutor(om, slippage=0.0, min_delta=0.001)
        order = exc.execute("XBTUSD", 0.1, 0.0, mid_price=50_000.0, equity=0.0)
        assert order is None


# ---------------------------------------------------------------------------
# TestPaperTrader
# ---------------------------------------------------------------------------

class TestPaperTrader:

    def test_step_returns_bar_stats(self, flat_df):
        trader = PaperTrader(models=[_ConstantModel(0.0)], initial_capital=10_000.0)
        obs    = flat_df.iloc[0].to_dict()
        stats  = trader.step(obs, log_return=0.0)
        assert isinstance(stats, BarStats)

    def test_zero_signal_zero_pnl(self, flat_df):
        """Zero signal → zero position → zero PnL on any return."""
        trader = PaperTrader(
            models=[_ConstantModel(0.0)],
            initial_capital=10_000.0,
            fee_rate=0.0,
        )
        for t in range(N_BARS):
            obs    = flat_df.iloc[t].to_dict()
            lr     = float(flat_df["1h_log_return"].iloc[t])
            stats  = trader.step(obs, log_return=lr)
        assert trader.equity == pytest.approx(10_000.0, rel=1e-6)

    def test_long_signal_profits_in_uptrend(self, trending_up_df):
        trader = PaperTrader(
            models=[_ConstantModel(1.0)],
            initial_capital=10_000.0,
            fee_rate=0.0,
        )
        for t in range(N_BARS):
            obs = trending_up_df.iloc[t].to_dict()
            lr  = float(trending_up_df["1h_log_return"].iloc[t])
            trader.step(obs, log_return=lr)
        assert trader.equity > 10_000.0

    def test_fees_reduce_equity(self, trending_up_df):
        """Equity with fees should be less than equity without fees."""
        def run(fee):
            trader = PaperTrader(
                models=[_ConstantModel(0.5)],
                initial_capital=10_000.0,
                fee_rate=fee,
            )
            for t in range(N_BARS):
                obs = trending_up_df.iloc[t].to_dict()
                lr  = float(trending_up_df["1h_log_return"].iloc[t])
                trader.step(obs, log_return=lr)
            return trader.equity

        assert run(0.0026) < run(0.0)

    def test_no_models_raises(self):
        with pytest.raises(ValueError):
            PaperTrader(models=[])

    def test_kill_switch_total_dd(self):
        """Any non-zero loss fires the kill switch when the limit is effectively zero."""
        trader = PaperTrader(
            models=[_ConstantModel(1.0)],
            initial_capital=10_000.0,
            fee_rate=0.0,
            total_dd_limit=-1e-6,   # fires on any non-zero loss
        )
        obs = {"feature_a": 0.0, "1h_log_return": 0.0}
        # Large log_return magnitude ensures the loss exceeds the near-zero limit
        # regardless of exact Kelly/regime-scaling of position size
        stats = trader.step(obs, log_return=-0.10)
        assert stats.kill_switch != ""

    def test_summary_returns_result(self, flat_df):
        trader = PaperTrader(models=[_ConstantModel(0.0)], initial_capital=10_000.0)
        for t in range(10):
            trader.step(flat_df.iloc[t].to_dict(), log_return=0.0)
        result = trader.summary()
        assert result.bars_processed == 10
        assert result.final_equity == pytest.approx(10_000.0, rel=1e-6)
        assert result.total_return == pytest.approx(0.0, abs=1e-6)

    def test_trade_logged_on_first_bar(self, flat_df):
        """Position opens on bar 0 → at least 1 trade in order manager."""
        trader = PaperTrader(
            models=[_ConstantModel(0.5)],
            initial_capital=10_000.0,
            fee_rate=0.0026,
        )
        trader.step(flat_df.iloc[0].to_dict(), log_return=0.0)
        assert trader.order_manager.n_trades() >= 1
