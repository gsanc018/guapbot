"""
guapbot/portfolio/cross_signals.py

ETHBTC cross-asset trend signal.

Derives the ETHBTC trend from the BTC and ETH close prices in each
strategy's obs dict. A positive signal means BTC is outperforming ETH
(sat_stacker benefits); a negative signal means ETH is outperforming
(money_printer benefits).

The signal is computed from a 20-bar vs 50-bar EMA of the ETHBTC ratio
(eth_price / btc_price). The result is used by CapitalAllocator to
shift the capital split between the two strategies.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from guapbot.utils.logging import get_logger

log = get_logger(__name__)

_EMA_FAST = 20
_EMA_SLOW = 50

# Price column search order (mirrors paper_trader.py _PRICE_COLS)
_PRICE_COLS = ["close", "price", "mid"]


@dataclass
class CrossSignalResult:
    """Output of CrossSignals.compute()."""

    ethbtc_signal: float   # [-1, +1]; +1 = BTC leading, -1 = ETH leading
    trend: str             # "btc_leading" | "eth_leading" | "neutral"
    confidence: float      # [0, 1]


class CrossSignals:
    """
    Derives the ETHBTC trend signal from BTC and ETH obs dicts.

    No external data source needed — prices are read directly from the
    feature obs vectors supplied at each bar.

    Maintains internal EMA state; do NOT share one instance across
    multiple independent runs.
    """

    def __init__(self) -> None:
        self._ema_fast: float | None = None
        self._ema_slow: float | None = None
        self._alpha_fast = 2.0 / (_EMA_FAST + 1)
        self._alpha_slow = 2.0 / (_EMA_SLOW + 1)
        self._bars = 0

    def compute(self, xbt_obs: dict, eth_obs: dict) -> CrossSignalResult:
        """
        Compute the ETHBTC trend signal for one bar.

        Args:
            xbt_obs: feature dict from the XBTUSD bar (must contain a price column)
            eth_obs: feature dict from the ETHUSD bar (must contain a price column)

        Returns:
            CrossSignalResult with signal, trend label, and confidence.
            Returns neutral before the 50-bar slow EMA warms up.
        """
        btc_price = self._resolve_price(xbt_obs)
        eth_price = self._resolve_price(eth_obs)

        if btc_price <= 0 or eth_price <= 0:
            log.debug("CrossSignals: invalid prices btc=%.4f eth=%.4f", btc_price, eth_price)
            return CrossSignalResult(0.0, "neutral", 0.0)

        ratio = eth_price / btc_price
        self._bars += 1

        # Update EMAs
        if self._ema_fast is None:
            self._ema_fast = ratio
            self._ema_slow = ratio
        else:
            self._ema_fast = self._alpha_fast * ratio + (1 - self._alpha_fast) * self._ema_fast
            self._ema_slow = self._alpha_slow * ratio + (1 - self._alpha_slow) * self._ema_slow

        # Require slow EMA to warm up
        if self._bars < _EMA_SLOW:
            return CrossSignalResult(0.0, "neutral", 0.0)

        # Positive divergence → ETH outperforming → negative signal (eth_leading)
        # Negative divergence → BTC outperforming → positive signal (btc_leading)
        if self._ema_slow <= 0:
            return CrossSignalResult(0.0, "neutral", 0.0)

        divergence = (self._ema_fast - self._ema_slow) / self._ema_slow
        # Invert: eth-leading divergence → negative signal
        raw_signal = float(np.clip(-divergence * 10.0, -1.0, 1.0))
        confidence = float(min(1.0, abs(raw_signal)))

        if raw_signal > 0.05:
            trend = "btc_leading"
        elif raw_signal < -0.05:
            trend = "eth_leading"
        else:
            trend = "neutral"

        log.debug(
            "CrossSignals: ratio=%.6f ema20=%.6f ema50=%.6f signal=%.4f trend=%s",
            ratio, self._ema_fast, self._ema_slow, raw_signal, trend,
        )
        return CrossSignalResult(raw_signal, trend, confidence)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_price(obs: dict) -> float:
        for col in _PRICE_COLS:
            val = obs.get(col, 0.0)
            if val and float(val) > 0:
                return float(val)
        return 0.0

    def __repr__(self) -> str:
        ema_fast = f"{self._ema_fast:.6f}" if self._ema_fast is not None else "None"
        ema_slow = f"{self._ema_slow:.6f}" if self._ema_slow is not None else "None"
        return f"CrossSignals(bars={self._bars}, ema_fast={ema_fast}, ema_slow={ema_slow})"
