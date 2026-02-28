"""
guapbot/portfolio/correlation.py

Rolling Pearson correlation between XBTUSD and ETHUSD log returns.

When BTC and ETH move together (high correlation), holding both strategies
at full size means the portfolio is less diversified than it appears. The
exposure_factor() output reduces both strategies' effective size when
correlation is high.

Correlation window: 720 bars (30 days × 24h).
"""
from __future__ import annotations

from collections import deque

import numpy as np

from guapbot.utils.logging import get_logger

log = get_logger(__name__)


class CorrelationTracker:
    """
    Tracks rolling Pearson correlation between two return series.

    The correlation is computed over the last `window` bars. Until the
    window is full, correlation() returns 0.0 (no adjustment).

    Args:
        window: number of bars in the rolling window (default 720 = 30 days)
    """

    def __init__(self, window: int = 720) -> None:
        self._window = window
        self._xbt_buf: deque[float] = deque(maxlen=window)
        self._eth_buf: deque[float] = deque(maxlen=window)

    def update(self, xbt_log_return: float, eth_log_return: float) -> None:
        """Append one bar's log returns for both assets."""
        self._xbt_buf.append(float(xbt_log_return))
        self._eth_buf.append(float(eth_log_return))

    def correlation(self) -> float:
        """
        Rolling Pearson r between XBTUSD and ETHUSD returns.

        Returns:
            Pearson r ∈ [-1, +1]. Returns 0.0 if fewer than 30 bars
            are available (too few to be meaningful).
        """
        n = len(self._xbt_buf)
        if n < 30:
            return 0.0

        x = np.array(self._xbt_buf, dtype=float)
        y = np.array(self._eth_buf, dtype=float)

        # Pearson r via np.corrcoef
        try:
            r = float(np.corrcoef(x, y)[0, 1])
        except Exception:
            return 0.0

        if np.isnan(r):
            return 0.0
        return float(np.clip(r, -1.0, 1.0))

    def exposure_factor(self) -> float:
        """
        Scalar in [0.5, 1.0] that scales down joint exposure when correlated.

        Formula: 1.0 - 0.5 * max(0, r)
            r = +1.0 → factor = 0.5  (fully correlated → halve exposure)
            r = 0.0  → factor = 1.0  (uncorrelated → full exposure)
            r < 0    → factor = 1.0  (negatively correlated → no reduction)

        Returns:
            exposure_factor ∈ [0.5, 1.0]
        """
        r = self.correlation()
        factor = float(1.0 - 0.5 * max(0.0, r))
        log.debug("CorrelationTracker: r=%.4f exposure_factor=%.4f", r, factor)
        return factor

    def __repr__(self) -> str:
        return (
            f"CorrelationTracker(window={self._window}, "
            f"bars={len(self._xbt_buf)}, "
            f"r={self.correlation():.4f})"
        )
