"""
tests/unit/test_portfolio.py

Unit tests for the portfolio layer:
    CrossSignals, CorrelationTracker, CapitalAllocator,
    BaseCurrencyAccounting, Benchmark, PortfolioRiskManager,
    PortfolioRunner.

All tests use synthetic data — no real feature cache, Redis, or API calls.
"""
from __future__ import annotations

import pytest

from guapbot.portfolio.cross_signals import CrossSignalResult, CrossSignals
from guapbot.portfolio.correlation import CorrelationTracker
from guapbot.portfolio.allocator import AllocationResult, CapitalAllocator
from guapbot.portfolio.base_currency import BaseCurrencyAccounting, PortfolioSnapshot
from guapbot.portfolio.benchmark import Benchmark
from guapbot.portfolio.risk import PortfolioRiskManager
from guapbot.portfolio import PortfolioRunner, PortfolioBarResult


# ---------------------------------------------------------------------------
# Stub model (same pattern as test_paper_trader.py)
# ---------------------------------------------------------------------------

class _ConstantModel:
    """Always returns a fixed signal."""

    def __init__(self, signal: float = 0.0):
        self._signal = signal
        self._fitted = True

    def predict(self, obs) -> float:
        return self._signal

    def confidence(self, obs) -> float:
        return 0.5


# ---------------------------------------------------------------------------
# CrossSignals
# ---------------------------------------------------------------------------

class TestCrossSignals:

    def test_neutral_before_warmup(self):
        """Fewer than 50 bars → neutral result (EMA not warmed up)."""
        cs = CrossSignals()
        for i in range(49):
            result = cs.compute({"close": 50_000.0}, {"close": 3_000.0})
        assert result.trend == "neutral"
        assert result.ethbtc_signal == pytest.approx(0.0)
        assert result.confidence == pytest.approx(0.0)

    def test_neutral_on_zero_price(self):
        """Zero price → neutral result."""
        cs = CrossSignals()
        result = cs.compute({"close": 0.0}, {"close": 3_000.0})
        assert result.trend == "neutral"

    def test_btc_leading_signal(self):
        """
        Declining ETHBTC ratio (ETH price falling relative to BTC) after warmup
        → BTC is leading → positive signal.
        """
        cs = CrossSignals()
        # Warm up with a stable ratio
        for _ in range(50):
            cs.compute({"close": 50_000.0}, {"close": 2_500.0})  # ratio=0.05

        # Now ETH drops relative to BTC (ratio falls → BTC outperforms → btc_leading)
        for _ in range(20):
            result = cs.compute({"close": 50_000.0}, {"close": 2_000.0})  # ratio=0.04

        # After the shift, fast EMA < slow EMA → divergence < 0 → signal > 0
        assert result.ethbtc_signal > 0.0, f"Expected positive signal, got {result.ethbtc_signal}"

    def test_eth_leading_signal(self):
        """
        Rising ETHBTC ratio (ETH outperforming BTC) → negative signal.
        """
        cs = CrossSignals()
        for _ in range(50):
            cs.compute({"close": 50_000.0}, {"close": 2_500.0})  # ratio=0.05

        # ETH rises relative to BTC (ratio rises → eth_leading)
        for _ in range(20):
            result = cs.compute({"close": 50_000.0}, {"close": 4_000.0})  # ratio=0.08

        assert result.ethbtc_signal < 0.0, f"Expected negative signal, got {result.ethbtc_signal}"

    def test_signal_clipped_to_range(self):
        """Signal must always be in [-1, +1]."""
        cs = CrossSignals()
        for _ in range(50):
            cs.compute({"close": 50_000.0}, {"close": 2_500.0})
        for _ in range(30):
            result = cs.compute({"close": 100_000.0}, {"close": 1.0})  # extreme ratio drop
        assert -1.0 <= result.ethbtc_signal <= 1.0

    def test_falls_back_to_price_columns(self):
        """Should accept 'price' and 'mid' in addition to 'close'."""
        cs = CrossSignals()
        result = cs.compute({"price": 50_000.0}, {"mid": 3_000.0})
        assert isinstance(result, CrossSignalResult)


# ---------------------------------------------------------------------------
# CorrelationTracker
# ---------------------------------------------------------------------------

class TestCorrelationTracker:

    def test_returns_zero_before_min_bars(self):
        """Fewer than 30 bars → correlation() == 0.0."""
        ct = CorrelationTracker(window=720)
        for _ in range(29):
            ct.update(0.01, 0.01)
        assert ct.correlation() == pytest.approx(0.0)

    def test_identical_returns_high_correlation(self):
        """Identical return series → Pearson r ≈ +1 → exposure_factor ≈ 0.5."""
        ct = CorrelationTracker(window=100)
        for i in range(100):
            ct.update(float(i) * 0.001, float(i) * 0.001)
        r = ct.correlation()
        assert r > 0.99, f"Expected r≈1.0, got {r}"
        factor = ct.exposure_factor()
        assert factor == pytest.approx(0.5, abs=0.02)

    def test_uncorrelated_returns_factor_near_one(self):
        """Alternating +/- returns → low correlation → exposure_factor ≈ 1.0."""
        ct = CorrelationTracker(window=100)
        for i in range(100):
            ct.update(0.01 if i % 2 == 0 else -0.01,
                      -0.01 if i % 2 == 0 else 0.01)
        r = ct.correlation()
        assert r < 0.0, f"Expected negative r for anti-correlated series, got {r}"
        factor = ct.exposure_factor()
        assert factor == pytest.approx(1.0), f"Expected factor=1.0 for r<0, got {factor}"

    def test_exposure_factor_range(self):
        """exposure_factor must always be in [0.5, 1.0]."""
        ct = CorrelationTracker(window=50)
        import random
        rng = random.Random(42)
        for _ in range(200):
            ct.update(rng.gauss(0, 0.01), rng.gauss(0, 0.01))
            f = ct.exposure_factor()
            assert 0.5 <= f <= 1.0, f"exposure_factor out of range: {f}"


# ---------------------------------------------------------------------------
# CapitalAllocator
# ---------------------------------------------------------------------------

class TestCapitalAllocator:

    def test_neutral_signal_returns_base_split(self):
        """Signal = 0.0 → exact base split."""
        alloc = CapitalAllocator(base_split=0.60)
        neutral = CrossSignalResult(ethbtc_signal=0.0, trend="neutral", confidence=0.0)
        result = alloc.allocate(neutral)
        assert result.money_printer_fraction == pytest.approx(0.60)
        assert result.sat_stacker_fraction == pytest.approx(0.40)
        assert result.money_printer_fraction + result.sat_stacker_fraction == pytest.approx(1.0)

    def test_btc_leading_shifts_to_sat_stacker(self):
        """Positive signal (BTC leading) → money_printer gets less."""
        alloc = CapitalAllocator(base_split=0.60, ethbtc_sensitivity=0.20)
        btc_lead = CrossSignalResult(ethbtc_signal=1.0, trend="btc_leading", confidence=1.0)
        result = alloc.allocate(btc_lead)
        assert result.money_printer_fraction == pytest.approx(0.40)  # 0.60 - 1.0*0.20
        assert result.sat_stacker_fraction == pytest.approx(0.60)

    def test_eth_leading_shifts_to_money_printer(self):
        """Negative signal (ETH leading) → money_printer gets more."""
        alloc = CapitalAllocator(base_split=0.60, ethbtc_sensitivity=0.20)
        eth_lead = CrossSignalResult(ethbtc_signal=-1.0, trend="eth_leading", confidence=1.0)
        result = alloc.allocate(eth_lead)
        assert result.money_printer_fraction == pytest.approx(0.80)  # 0.60 + 0.20

    def test_clips_to_min_split(self):
        """Extreme positive signal clips to min_split."""
        alloc = CapitalAllocator(base_split=0.60, min_split=0.30, ethbtc_sensitivity=0.20)
        extreme = CrossSignalResult(ethbtc_signal=5.0, trend="btc_leading", confidence=1.0)
        result = alloc.allocate(extreme)
        assert result.money_printer_fraction == pytest.approx(0.30)

    def test_clips_to_max_split(self):
        """Extreme negative signal clips to max_split."""
        alloc = CapitalAllocator(base_split=0.60, max_split=0.80, ethbtc_sensitivity=0.20)
        extreme = CrossSignalResult(ethbtc_signal=-5.0, trend="eth_leading", confidence=1.0)
        result = alloc.allocate(extreme)
        assert result.money_printer_fraction == pytest.approx(0.80)

    def test_fractions_always_sum_to_one(self):
        """Regardless of signal, mp + ss must equal 1.0."""
        alloc = CapitalAllocator()
        for sig in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            cr = CrossSignalResult(ethbtc_signal=sig, trend="neutral", confidence=abs(sig))
            r = alloc.allocate(cr)
            assert r.money_printer_fraction + r.sat_stacker_fraction == pytest.approx(1.0)

    def test_invalid_split_bounds_raises(self):
        with pytest.raises(ValueError):
            CapitalAllocator(min_split=0.80, max_split=0.30)  # inverted

    def test_base_split_outside_bounds_raises(self):
        with pytest.raises(ValueError):
            CapitalAllocator(base_split=0.95, min_split=0.30, max_split=0.80)


# ---------------------------------------------------------------------------
# BaseCurrencyAccounting
# ---------------------------------------------------------------------------

class TestBaseCurrencyAccounting:

    def test_sat_conversion(self):
        """ss_equity=10000 at btc_price=50000 → 0.2 BTC → 20M sats."""
        acc = BaseCurrencyAccounting(
            initial_mp_usd=6_000.0,
            initial_ss_usd=4_000.0,
            initial_btc_price=50_000.0,
        )
        snap = acc.update(mp_equity=6_000.0, ss_equity=10_000.0,
                          btc_price=50_000.0, bars=1)
        assert snap.ss_equity_btc == pytest.approx(0.2)
        assert snap.ss_sats_accumulated == 20_000_000

    def test_total_equity_sum(self):
        """total_equity_usd = mp_equity + ss_equity."""
        acc = BaseCurrencyAccounting(5_000.0, 5_000.0, 40_000.0)
        snap = acc.update(mp_equity=5_500.0, ss_equity=4_800.0,
                          btc_price=40_000.0, bars=10)
        assert snap.total_equity_usd == pytest.approx(10_300.0)

    def test_total_return_on_flat(self):
        """Flat equity → total_return_pct == 0."""
        acc = BaseCurrencyAccounting(5_000.0, 5_000.0, 40_000.0)
        snap = acc.update(5_000.0, 5_000.0, 40_000.0, 1)
        assert snap.total_return_pct == pytest.approx(0.0, abs=1e-9)

    def test_btc_price_zero_does_not_raise(self):
        """Zero btc_price should clamp to 1.0, not divide by zero."""
        acc = BaseCurrencyAccounting(5_000.0, 5_000.0, 40_000.0)
        snap = acc.update(5_000.0, 5_000.0, btc_price=0.0, bars=1)
        assert snap.ss_equity_btc > 0


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

class TestBenchmark:

    def test_flat_market_zero_returns(self):
        """Unchanged prices → all returns == 0."""
        b = Benchmark(50_000.0, 3_000.0, 10_000.0)
        snap = b.update(50_000.0, 3_000.0, 10_000.0)
        assert snap.btc_bah_return == pytest.approx(0.0, abs=1e-9)
        assert snap.eth_bah_return == pytest.approx(0.0, abs=1e-9)
        assert snap.portfolio_return == pytest.approx(0.0, abs=1e-9)

    def test_btc_doubles(self):
        """BTC price doubles → btc_bah_return ≈ +1.0 (100%)."""
        b = Benchmark(50_000.0, 3_000.0, 10_000.0)
        snap = b.update(100_000.0, 3_000.0, 10_000.0)
        assert snap.btc_bah_return == pytest.approx(1.0)

    def test_alpha_positive_when_outperforming(self):
        """Portfolio grows while market is flat → positive alpha."""
        b = Benchmark(50_000.0, 3_000.0, 10_000.0)
        snap = b.update(50_000.0, 3_000.0, 11_000.0)
        assert snap.alpha_vs_btc > 0.0
        assert snap.alpha_vs_combined > 0.0


# ---------------------------------------------------------------------------
# PortfolioRiskManager
# ---------------------------------------------------------------------------

class TestPortfolioRiskManager:

    def test_all_clear_on_flat_equity(self):
        """Equity at peak → empty kill string."""
        rm = PortfolioRiskManager(daily_dd_limit=-0.05, total_dd_limit=-0.15)
        result = rm.update(10_000.0)
        assert result == ""
        result = rm.update(10_000.0)
        assert result == ""

    def test_total_dd_fires(self):
        """Drop below total_dd_limit → non-empty kill reason."""
        rm = PortfolioRiskManager(total_dd_limit=-0.10)
        rm.update(10_000.0)   # sets peak
        result = rm.update(8_999.0)  # -10.01% drawdown
        assert result != "", f"Expected kill switch to fire, got: {result!r}"
        assert "PORTFOLIO KILL" in result

    def test_daily_dd_fires(self):
        """Intra-day drop below daily_dd_limit → non-empty kill reason."""
        rm = PortfolioRiskManager(daily_dd_limit=-0.05, total_dd_limit=-0.20)
        rm.update(10_000.0)   # sets peak + day_start
        result = rm.update(9_499.0)  # -5.01% daily drawdown
        assert result != "", f"Expected daily kill switch to fire, got: {result!r}"

    def test_no_fire_within_limits(self):
        """Small daily dip that doesn't cross threshold → no kill."""
        rm = PortfolioRiskManager(daily_dd_limit=-0.05, total_dd_limit=-0.15)
        rm.update(10_000.0)
        result = rm.update(9_600.0)  # -4% daily, within -5% limit
        assert result == ""


# ---------------------------------------------------------------------------
# PortfolioRunner (integration)
# ---------------------------------------------------------------------------

class TestPortfolioRunner:

    def _make_runner(self, mp_cap=6_000.0, ss_cap=4_000.0):
        """Build a minimal PortfolioRunner with stub models."""
        from guapbot.execution.paper_trader import PaperTrader
        from guapbot.portfolio import (
            CrossSignals, CorrelationTracker, CapitalAllocator,
            BaseCurrencyAccounting, Benchmark, PortfolioRiskManager,
            PortfolioRunner,
        )

        mp_trader = PaperTrader(
            models=[_ConstantModel(0.5)],
            initial_capital=mp_cap,
            fee_rate=0.0,
            pair="XBTUSD",
        )
        ss_trader = PaperTrader(
            models=[_ConstantModel(0.3)],
            initial_capital=ss_cap,
            fee_rate=0.0,
            max_long=0.20,
            max_short=0.0,
            pair="ETHUSD",
        )
        runner = PortfolioRunner(
            mp_trader=mp_trader,
            ss_trader=ss_trader,
            cross=CrossSignals(),
            correlation=CorrelationTracker(window=100),
            allocator=CapitalAllocator(base_split=0.60),
            accounting=BaseCurrencyAccounting(mp_cap, ss_cap, 50_000.0),
            risk=PortfolioRiskManager(),
            benchmark=Benchmark(50_000.0, 3_000.0, mp_cap + ss_cap),
        )
        return runner

    def test_step_returns_portfolio_bar_result(self):
        """step() returns a PortfolioBarResult with correct types."""
        runner = self._make_runner()
        obs = {"close": 50_000.0, "feature_a": 0.1, "1h_log_return": 0.001}
        result = runner.step(obs, 0.001, {"close": 3_000.0, "feature_a": 0.1}, 0.0005)
        assert isinstance(result, PortfolioBarResult)
        assert result.t == 0
        assert isinstance(result.snapshot.total_equity_usd, float)
        assert result.snapshot.total_equity_usd > 0

    def test_bars_increment_correctly(self):
        """bars_processed increments with each step."""
        runner = self._make_runner()
        obs_xbt = {"close": 50_000.0}
        obs_eth = {"close": 3_000.0}
        for i in range(5):
            runner.step(obs_xbt, 0.001, obs_eth, 0.0005)
        assert runner.bars_processed == 5

    def test_kill_switch_propagates(self):
        """A kill switch in either PaperTrader surfaces in PortfolioBarResult."""
        from guapbot.execution.paper_trader import PaperTrader
        from guapbot.portfolio import (
            CrossSignals, CorrelationTracker, CapitalAllocator,
            BaseCurrencyAccounting, PortfolioRiskManager, PortfolioRunner,
        )

        # mp_trader with hair-trigger kill switch
        mp_trader = PaperTrader(
            models=[_ConstantModel(1.0)],
            initial_capital=6_000.0,
            fee_rate=0.0,
            total_dd_limit=-1e-6,   # fires on any loss
            pair="XBTUSD",
        )
        ss_trader = PaperTrader(
            models=[_ConstantModel(0.3)],
            initial_capital=4_000.0,
            fee_rate=0.0,
            max_long=0.20,
            max_short=0.0,
            pair="ETHUSD",
        )
        runner = PortfolioRunner(
            mp_trader=mp_trader,
            ss_trader=ss_trader,
            cross=CrossSignals(),
            correlation=CorrelationTracker(window=100),
            allocator=CapitalAllocator(),
            accounting=BaseCurrencyAccounting(6_000.0, 4_000.0, 50_000.0),
            risk=PortfolioRiskManager(),
        )
        obs = {"close": 50_000.0}
        result = runner.step(obs, -0.10, {"close": 3_000.0}, 0.0)
        assert result.kill_switch != "", "Expected kill switch to fire"

    def test_no_kill_switch_on_flat(self):
        """Zero returns → no kill switch."""
        runner = self._make_runner()
        obs_xbt = {"close": 50_000.0}
        obs_eth = {"close": 3_000.0}
        for _ in range(10):
            result = runner.step(obs_xbt, 0.0, obs_eth, 0.0)
        assert result.kill_switch == ""
