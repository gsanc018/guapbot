"""
guapbot/portfolio/__init__.py

Portfolio Layer — dual-strategy coordination for GuapBot.

Manages money_printer (USD-base, XBTUSD) and sat_stacker (BTC-base,
ETHUSD) simultaneously. At each bar:
  1. CrossSignals computes the ETHBTC trend
  2. CorrelationTracker updates rolling BTC/ETH correlation
  3. CapitalAllocator derives the current split; fraction → regime confidence → position scale
  4. Both PaperTraders step independently
  5. BaseCurrencyAccounting reports dual-base equity + sats
  6. Benchmark tracks buy-and-hold alpha
  7. PortfolioRiskManager checks combined kill switches

Public API:
    from guapbot.portfolio import (
        PortfolioRunner, PortfolioBarResult,
        CrossSignals, CrossSignalResult,
        CorrelationTracker,
        CapitalAllocator, AllocationResult,
        BaseCurrencyAccounting, PortfolioSnapshot,
        Benchmark, BenchmarkSnapshot,
        PortfolioRiskManager,
    )
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from guapbot.execution.paper_trader import BarStats, PaperTrader
from guapbot.regime.base import RegimeResult
from guapbot.portfolio.allocator import AllocationResult, CapitalAllocator
from guapbot.portfolio.base_currency import BaseCurrencyAccounting, PortfolioSnapshot
from guapbot.portfolio.benchmark import Benchmark, BenchmarkSnapshot
from guapbot.portfolio.correlation import CorrelationTracker
from guapbot.portfolio.cross_signals import CrossSignalResult, CrossSignals
from guapbot.portfolio.risk import PortfolioRiskManager
from guapbot.utils.logging import get_logger

log = get_logger(__name__)

# Price column search order — same as paper_trader.py
_PRICE_COLS = ["close", "price", "mid"]


def _allocation_regimes(fraction: float) -> list[RegimeResult]:
    """
    Build a 3-timeframe regime vector whose confidence equals the allocation fraction.

    DefaultPositionSizer scales positions by mean(regime.confidence), so passing
    fraction=0.60 for money_printer and fraction=0.40 for sat_stacker causes
    money_printer to hold proportionally larger positions — making allocation real.
    """
    conf = float(max(0.0, min(1.0, fraction)))
    return [
        RegimeResult(label="trending", confidence=conf, timeframe="1h"),
        RegimeResult(label="trending", confidence=conf, timeframe="4h"),
        RegimeResult(label="bullish",  confidence=conf, timeframe="daily"),
    ]


@dataclass
class PortfolioBarResult:
    """Per-bar output from PortfolioRunner.step()."""

    t: int
    mp_stats: BarStats           # money_printer execution stats
    ss_stats: BarStats           # sat_stacker execution stats
    allocation: AllocationResult
    snapshot: PortfolioSnapshot
    benchmark: Optional[BenchmarkSnapshot] = None
    kill_switch: str = ""        # non-empty = at least one kill switch fired


class PortfolioRunner:
    """
    Dual-strategy bar-by-bar orchestrator.

    Connects CrossSignals → CorrelationTracker → CapitalAllocator →
    both PaperTraders → BaseCurrencyAccounting → Benchmark →
    PortfolioRiskManager on every bar.

    Args:
        mp_trader:    PaperTrader for money_printer (XBTUSD)
        ss_trader:    PaperTrader for sat_stacker (ETHUSD)
        cross:        CrossSignals instance
        correlation:  CorrelationTracker instance
        allocator:    CapitalAllocator instance
        accounting:   BaseCurrencyAccounting instance
        risk:         PortfolioRiskManager instance
        benchmark:    Benchmark instance (optional)
    """

    def __init__(
        self,
        mp_trader: PaperTrader,
        ss_trader: PaperTrader,
        cross: CrossSignals,
        correlation: CorrelationTracker,
        allocator: CapitalAllocator,
        accounting: BaseCurrencyAccounting,
        risk: PortfolioRiskManager,
        benchmark: Optional[Benchmark] = None,
    ) -> None:
        self._mp = mp_trader
        self._ss = ss_trader
        self._cross = cross
        self._corr = correlation
        self._allocator = allocator
        self._accounting = accounting
        self._benchmark = benchmark
        self._risk = risk
        self._bars = 0

    def step(
        self,
        mp_obs: dict,
        mp_log_return: float,
        ss_obs: dict,
        ss_log_return: float,
    ) -> PortfolioBarResult:
        """
        Process one bar through the full portfolio pipeline.

        Args:
            mp_obs:         feature dict for money_printer at bar t
            mp_log_return:  next-bar log return for money_printer
            ss_obs:         feature dict for sat_stacker at bar t
            ss_log_return:  next-bar log return for sat_stacker

        Returns:
            PortfolioBarResult with all per-bar state.
        """
        t = self._bars

        # 1. ETHBTC cross-asset signal
        cross = self._cross.compute(mp_obs, ss_obs)

        # 2. Rolling correlation update
        self._corr.update(mp_log_return, ss_log_return)

        # 3. Capital allocation — drives position sizing for both strategies.
        # The allocation fraction feeds into regime confidence, which DefaultPositionSizer
        # uses to scale down positions (higher allocation → higher confidence → larger size).
        alloc = self._allocator.allocate(cross, self._corr.exposure_factor())

        # Build regime vectors whose confidence encodes the allocation signal.
        # This is the mechanism by which the ETHBTC trend actually shifts position sizes:
        # mp_fraction=0.60, ss_fraction=0.40 → mp has more capital influence than ss.
        mp_regimes = _allocation_regimes(alloc.money_printer_fraction)
        ss_regimes = _allocation_regimes(alloc.sat_stacker_fraction)

        # 4. Step both PaperTraders with allocation-derived regimes
        mp_stats = self._mp.step(mp_obs, mp_log_return, regimes=mp_regimes)
        ss_stats = self._ss.step(ss_obs, ss_log_return, regimes=ss_regimes)

        # 5. Dual-base accounting
        btc_price = self._resolve_price(mp_obs)
        eth_price = self._resolve_price(ss_obs)
        snapshot = self._accounting.update(
            mp_equity=self._mp.equity,
            ss_equity=self._ss.equity,
            btc_price=btc_price,
            bars=t + 1,
        )

        # 6. Benchmark
        bench = None
        if self._benchmark is not None:
            bench = self._benchmark.update(btc_price, eth_price, snapshot.total_equity_usd)

        # 7. Portfolio-level risk check
        port_kill = self._risk.update(snapshot.total_equity_usd)

        # Combine all kill signals
        kill = mp_stats.kill_switch or ss_stats.kill_switch or port_kill

        self._bars += 1

        return PortfolioBarResult(
            t=t,
            mp_stats=mp_stats,
            ss_stats=ss_stats,
            allocation=alloc,
            snapshot=snapshot,
            benchmark=bench,
            kill_switch=kill,
        )

    @property
    def mp_equity(self) -> float:
        return self._mp.equity

    @property
    def ss_equity(self) -> float:
        return self._ss.equity

    @property
    def total_equity(self) -> float:
        return self._mp.equity + self._ss.equity

    @property
    def bars_processed(self) -> int:
        return self._bars

    @staticmethod
    def _resolve_price(obs: dict) -> float:
        for col in _PRICE_COLS:
            val = obs.get(col, 0.0)
            if val and float(val) > 0:
                return float(val)
        return 1.0

    def __repr__(self) -> str:
        return (
            f"PortfolioRunner("
            f"mp_equity={self._mp.equity:.2f}, "
            f"ss_equity={self._ss.equity:.2f}, "
            f"bars={self._bars})"
        )


__all__ = [
    "PortfolioRunner",
    "PortfolioBarResult",
    "CrossSignals",
    "CrossSignalResult",
    "CorrelationTracker",
    "CapitalAllocator",
    "AllocationResult",
    "BaseCurrencyAccounting",
    "PortfolioSnapshot",
    "Benchmark",
    "BenchmarkSnapshot",
    "PortfolioRiskManager",
]
