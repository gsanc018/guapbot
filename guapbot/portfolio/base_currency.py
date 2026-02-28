"""
guapbot/portfolio/base_currency.py

Dual-base portfolio accounting.

GuapBot tracks performance in two currencies simultaneously:
  - money_printer: USD-base (grow dollars)
  - sat_stacker:   BTC-base (accumulate satoshis)

At each bar, BaseCurrencyAccounting converts sat_stacker's USD equity
into BTC equivalent using the current XBTUSD price, then reports:
  - dollar equity for both strategies
  - BTC and satoshi counts for sat_stacker
  - combined total return in USD terms
"""
from __future__ import annotations

from dataclasses import dataclass

from guapbot.utils.logging import get_logger

log = get_logger(__name__)

_SATS_PER_BTC = 100_000_000


@dataclass
class PortfolioSnapshot:
    """Per-bar combined portfolio state."""

    # Money Printer (USD base)
    mp_equity_usd: float
    mp_total_return: float       # fraction vs initial mp capital

    # Sat Stacker (BTC base)
    ss_equity_usd: float         # current USD value of sat_stacker
    ss_equity_btc: float         # ss_equity_usd / btc_price
    ss_sats_accumulated: int     # int(ss_equity_btc * 1e8)
    ss_total_return_btc: float   # fraction vs initial ss BTC equivalent

    # Combined
    total_equity_usd: float
    total_return_pct: float      # fraction vs initial total capital
    bars_processed: int


class BaseCurrencyAccounting:
    """
    Tracks dual-base PnL across both strategies.

    Args:
        initial_mp_usd:   starting USD capital allocated to money_printer
        initial_ss_usd:   starting USD capital allocated to sat_stacker
        initial_btc_price: XBTUSD price at strategy start (for BTC baseline)
    """

    def __init__(
        self,
        initial_mp_usd: float,
        initial_ss_usd: float,
        initial_btc_price: float,
    ) -> None:
        self._initial_mp_usd = initial_mp_usd
        self._initial_ss_usd = initial_ss_usd
        self._initial_btc_price = max(initial_btc_price, 1.0)
        # BTC equivalent at inception for sat_stacker baseline
        self._initial_ss_btc = initial_ss_usd / self._initial_btc_price
        self._initial_total_usd = initial_mp_usd + initial_ss_usd

    def update(
        self,
        mp_equity: float,
        ss_equity: float,
        btc_price: float,
        bars: int,
    ) -> PortfolioSnapshot:
        """
        Compute a portfolio snapshot for one bar.

        Args:
            mp_equity:  current equity of the money_printer PaperTrader (USD)
            ss_equity:  current equity of the sat_stacker PaperTrader (USD)
            btc_price:  current XBTUSD mid-price for BTC conversion
            bars:       current bar count

        Returns:
            PortfolioSnapshot with all accounting fields populated.
        """
        btc_price = max(btc_price, 1.0)

        # Money printer returns
        mp_total_return = (mp_equity / self._initial_mp_usd) - 1.0 if self._initial_mp_usd > 0 else 0.0

        # Sat stacker BTC accounting
        ss_equity_btc = ss_equity / btc_price
        ss_sats = int(ss_equity_btc * _SATS_PER_BTC)
        ss_total_return_btc = (
            (ss_equity_btc / self._initial_ss_btc) - 1.0
            if self._initial_ss_btc > 0 else 0.0
        )

        total_equity_usd = mp_equity + ss_equity
        total_return_pct = (
            (total_equity_usd / self._initial_total_usd) - 1.0
            if self._initial_total_usd > 0 else 0.0
        )

        log.debug(
            "Portfolio accounting: mp_usd=%.2f ss_usd=%.2f ss_btc=%.6f sats=%d total=%.2f",
            mp_equity, ss_equity, ss_equity_btc, ss_sats, total_equity_usd,
        )

        return PortfolioSnapshot(
            mp_equity_usd=mp_equity,
            mp_total_return=mp_total_return,
            ss_equity_usd=ss_equity,
            ss_equity_btc=ss_equity_btc,
            ss_sats_accumulated=ss_sats,
            ss_total_return_btc=ss_total_return_btc,
            total_equity_usd=total_equity_usd,
            total_return_pct=total_return_pct,
            bars_processed=bars,
        )

    def __repr__(self) -> str:
        return (
            f"BaseCurrencyAccounting("
            f"mp_initial={self._initial_mp_usd:.2f}, "
            f"ss_initial={self._initial_ss_usd:.2f}, "
            f"btc_price_initial={self._initial_btc_price:.2f})"
        )
