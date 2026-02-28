"""
guapbot/portfolio/risk.py

Portfolio-level kill switches.

Operates on the COMBINED portfolio equity (money_printer + sat_stacker),
independently of the per-strategy kill switches inside each PaperTrader.

Kill switch hierarchy:
  1. Per-strategy PaperTrader fires first  → halts that strategy only
  2. PortfolioRiskManager fires            → halts BOTH strategies

Both can fire on the same bar. The CLI loop checks all three kill_switch
fields (mp_stats.kill_switch, ss_stats.kill_switch, and portfolio risk)
and halts on the first non-empty reason.
"""
from __future__ import annotations

from guapbot.utils.logging import get_logger

log = get_logger(__name__)


class PortfolioRiskManager:
    """
    Portfolio-level kill switches on combined equity.

    Args:
        daily_dd_limit:  daily drawdown threshold (default -0.05 = -5%)
        total_dd_limit:  total drawdown threshold (default -0.15 = -15%)
    """

    def __init__(
        self,
        daily_dd_limit: float = -0.05,
        total_dd_limit: float = -0.15,
    ) -> None:
        self._daily_dd_limit = daily_dd_limit
        self._total_dd_limit = total_dd_limit

        self._peak_equity: float | None = None
        self._day_start_equity: float | None = None
        self._bars = 0

    def update(self, total_equity: float) -> str:
        """
        Check kill switches for the current bar's combined equity.

        Args:
            total_equity: combined USD equity of both strategies this bar

        Returns:
            Empty string if all clear, or a descriptive reason string
            if a kill switch fires.
        """
        self._bars += 1

        # Initialise on first bar
        if self._peak_equity is None:
            self._peak_equity = total_equity
            self._day_start_equity = total_equity
            return ""

        # Update peak
        if total_equity > self._peak_equity:
            self._peak_equity = total_equity

        # Reset daily baseline every ~24 bars
        if self._bars % 24 == 0:
            self._day_start_equity = total_equity

        # --- Total drawdown ---
        total_dd = (total_equity - self._peak_equity) / self._peak_equity
        if total_dd <= self._total_dd_limit:
            reason = (
                f"PORTFOLIO KILL — total drawdown {total_dd:.1%} "
                f"exceeded limit {self._total_dd_limit:.1%}"
            )
            log.warning("%s (bar=%d equity=%.2f)", reason, self._bars, total_equity)
            return reason

        # --- Daily drawdown ---
        if self._day_start_equity and self._day_start_equity > 0:
            daily_dd = (total_equity - self._day_start_equity) / self._day_start_equity
            if daily_dd <= self._daily_dd_limit:
                reason = (
                    f"PORTFOLIO KILL — daily drawdown {daily_dd:.1%} "
                    f"exceeded limit {self._daily_dd_limit:.1%}"
                )
                log.warning("%s (bar=%d equity=%.2f)", reason, self._bars, total_equity)
                return reason

        return ""

    @property
    def peak_equity(self) -> float | None:
        return self._peak_equity

    def __repr__(self) -> str:
        return (
            f"PortfolioRiskManager("
            f"daily_limit={self._daily_dd_limit:.1%}, "
            f"total_limit={self._total_dd_limit:.1%}, "
            f"peak={self._peak_equity})"
        )
