"""
guapbot/portfolio/allocator.py

Dynamic capital allocator for the dual-strategy portfolio.

Decides how much of total capital goes to money_printer (XBTUSD) vs
sat_stacker (ETHUSD) based on:
  1. ETHBTC cross-asset signal (BTC leading → more sat_stacker; ETH leading → more money_printer)
  2. Rolling BTC/ETH correlation (high correlation → reduce both exposures via exposure_factor)

The output fractions always sum to 1.0. They inform how much starting
capital each PaperTrader receives, and are tracked per-bar for analytics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from guapbot.portfolio.cross_signals import CrossSignalResult
from guapbot.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class AllocationResult:
    """Capital split output from CapitalAllocator.allocate()."""

    money_printer_fraction: float   # fraction of total capital → money_printer [0, 1]
    sat_stacker_fraction: float     # fraction of total capital → sat_stacker  [0, 1]
    exposure_factor: float          # correlation scaling  [0.5, 1.0]
    reason: str                     # human-readable explanation


class CapitalAllocator:
    """
    Computes the per-bar capital split between the two strategies.

    Args:
        base_split:          default fraction for money_printer (default 0.60)
        min_split:           floor for money_printer fraction (default 0.30)
        max_split:           ceiling for money_printer fraction (default 0.80)
        ethbtc_sensitivity:  how much the ETHBTC signal shifts the split (default 0.20)
    """

    def __init__(
        self,
        base_split: float = 0.60,
        min_split: float = 0.30,
        max_split: float = 0.80,
        ethbtc_sensitivity: float = 0.20,
    ) -> None:
        if not (0 < min_split < max_split < 1):
            raise ValueError(
                f"Require 0 < min_split < max_split < 1, "
                f"got min={min_split} max={max_split}"
            )
        if not (min_split <= base_split <= max_split):
            raise ValueError(
                f"base_split {base_split} must be in [min_split, max_split]"
            )
        self._base_split = base_split
        self._min_split = min_split
        self._max_split = max_split
        self._sensitivity = ethbtc_sensitivity

    def allocate(
        self,
        cross: CrossSignalResult,
        correlation_factor: float = 1.0,
    ) -> AllocationResult:
        """
        Compute the capital split for one bar.

        Args:
            cross:              CrossSignalResult from CrossSignals.compute()
            correlation_factor: exposure_factor from CorrelationTracker (default 1.0)

        Returns:
            AllocationResult with money_printer_fraction + sat_stacker_fraction + metadata.

        Logic:
            Positive ETHBTC signal (BTC leading) → shift capital toward sat_stacker
              mp_fraction = base_split - signal * sensitivity
            The fraction is clipped to [min_split, max_split].
            sat_stacker_fraction = 1 - money_printer_fraction.
        """
        # Shift based on ETHBTC signal
        mp_fraction = float(np.clip(
            self._base_split - cross.ethbtc_signal * self._sensitivity,
            self._min_split,
            self._max_split,
        ))
        ss_fraction = 1.0 - mp_fraction

        reason = (
            f"ethbtc={cross.trend} "
            f"signal={cross.ethbtc_signal:+.3f} "
            f"mp={mp_fraction:.2%} "
            f"ss={ss_fraction:.2%} "
            f"corr_factor={correlation_factor:.3f}"
        )

        log.debug("CapitalAllocator: %s", reason)
        return AllocationResult(
            money_printer_fraction=mp_fraction,
            sat_stacker_fraction=ss_fraction,
            exposure_factor=float(correlation_factor),
            reason=reason,
        )

    def __repr__(self) -> str:
        return (
            f"CapitalAllocator("
            f"base={self._base_split:.0%}, "
            f"range=[{self._min_split:.0%}, {self._max_split:.0%}], "
            f"sensitivity={self._sensitivity})"
        )
