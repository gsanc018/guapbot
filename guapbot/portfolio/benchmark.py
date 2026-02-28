"""
guapbot/portfolio/benchmark.py

Buy-and-hold benchmark comparisons.

Tracks how the portfolio performs vs simple buy-and-hold of BTC and ETH.
At each bar, computes:
  - BTC buy-and-hold return (held since portfolio start)
  - ETH buy-and-hold return (held since portfolio start)
  - 50/50 BTC+ETH buy-and-hold return
  - Portfolio return vs that combined benchmark (alpha)

No positions are taken — this is purely a passive reference tracker.
"""
from __future__ import annotations

from dataclasses import dataclass

from guapbot.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class BenchmarkSnapshot:
    """Per-bar benchmark comparison."""

    btc_bah_return: float      # BTC buy-and-hold return since start
    eth_bah_return: float      # ETH buy-and-hold return since start
    combined_bah_return: float # 50/50 BTC+ETH buy-and-hold return
    portfolio_return: float    # actual portfolio total return since start
    alpha_vs_btc: float        # portfolio_return - btc_bah_return
    alpha_vs_combined: float   # portfolio_return - combined_bah_return


class Benchmark:
    """
    Passive buy-and-hold tracker for portfolio alpha computation.

    Args:
        initial_btc_price:     XBTUSD price at strategy start
        initial_eth_price:     ETHUSD price at strategy start
        initial_portfolio_usd: total portfolio capital at start
    """

    def __init__(
        self,
        initial_btc_price: float,
        initial_eth_price: float,
        initial_portfolio_usd: float,
    ) -> None:
        self._btc0 = max(initial_btc_price, 1.0)
        self._eth0 = max(initial_eth_price, 1.0)
        self._portfolio0 = max(initial_portfolio_usd, 1.0)
        # Hold half in BTC, half in ETH
        half = initial_portfolio_usd / 2.0
        self._btc_units = half / self._btc0
        self._eth_units = half / self._eth0

    def update(
        self,
        btc_price: float,
        eth_price: float,
        portfolio_equity_usd: float,
    ) -> BenchmarkSnapshot:
        """
        Compute benchmark snapshot for one bar.

        Args:
            btc_price:            current XBTUSD price
            eth_price:            current ETHUSD price
            portfolio_equity_usd: current total portfolio equity in USD

        Returns:
            BenchmarkSnapshot with all return and alpha fields.
        """
        btc_price = max(btc_price, 1.0)
        eth_price = max(eth_price, 1.0)

        btc_bah_return = (btc_price / self._btc0) - 1.0
        eth_bah_return = (eth_price / self._eth0) - 1.0

        combined_bah_usd = self._btc_units * btc_price + self._eth_units * eth_price
        combined_bah_return = (combined_bah_usd / self._portfolio0) - 1.0

        portfolio_return = (portfolio_equity_usd / self._portfolio0) - 1.0

        alpha_vs_btc = portfolio_return - btc_bah_return
        alpha_vs_combined = portfolio_return - combined_bah_return

        log.debug(
            "Benchmark: btc_bah=%.4f eth_bah=%.4f combined=%.4f portfolio=%.4f alpha=%.4f",
            btc_bah_return, eth_bah_return, combined_bah_return, portfolio_return, alpha_vs_combined,
        )

        return BenchmarkSnapshot(
            btc_bah_return=btc_bah_return,
            eth_bah_return=eth_bah_return,
            combined_bah_return=combined_bah_return,
            portfolio_return=portfolio_return,
            alpha_vs_btc=alpha_vs_btc,
            alpha_vs_combined=alpha_vs_combined,
        )

    def __repr__(self) -> str:
        return (
            f"Benchmark("
            f"btc0={self._btc0:.2f}, "
            f"eth0={self._eth0:.2f}, "
            f"portfolio0={self._portfolio0:.2f})"
        )
