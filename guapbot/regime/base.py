"""
guapbot.regime.base
-------------------
Base interface for all regime detectors.

One detector is trained and deployed per timeframe (1h, 4h, daily).
The regime VECTOR — not a single label — is the output. All three
timeframes run simultaneously and their outputs are passed as a dict
to the ensemble.

Interface contract:
    detector.fit(df)         → train on labelled historical data
    detector.detect(obs)     → (label: str, confidence: float)

Label values:
    1h / 4h:  'trending' | 'ranging' | 'volatile'
    daily:    'bullish'  | 'bearish' | 'neutral'
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from guapbot.utils.logging import get_logger

log = get_logger(__name__)

# Allowed regime labels per timeframe type
IntradayLabel = Literal["trending", "ranging", "volatile"]
DailyLabel = Literal["bullish", "bearish", "neutral"]
RegimeLabel = str   # union of both — validated at runtime


@dataclass(frozen=True)
class RegimeResult:
    """
    Output of a single detector for a single bar.

    Attributes:
        label:      regime class string
        confidence: [0.0, 1.0] — how sure the detector is
        timeframe:  '1h', '4h', or 'daily'
    """

    label: RegimeLabel
    confidence: float
    timeframe: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.timeframe not in ("1h", "4h", "daily"):
            raise ValueError(f"timeframe must be '1h', '4h', or 'daily', got {self.timeframe}")

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "timeframe": self.timeframe,
        }


class RegimeDetector(ABC):
    """
    Abstract base for all regime detectors.

    The regime layer sits between the feature layer and the ensemble.
    It does NOT know about models or position sizing — it only answers:
    "What regime are we in right now, and how confident are you?"

    Subclasses must implement:
        fit(df)      — train / warm-start on labelled history
        detect(obs)  — classify a single observation

    The fitted state (HMM parameters, scaler state, etc.) should be
    persisted by the subclass via save() / load() if needed.
    """

    def __init__(self, timeframe: str) -> None:
        """
        Args:
            timeframe: '1h', '4h', or 'daily'
        """
        if timeframe not in ("1h", "4h", "daily"):
            raise ValueError(f"timeframe must be '1h', '4h', or 'daily', got {timeframe}")
        self.timeframe = timeframe
        self._fitted = False

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement both
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "RegimeDetector":
        """
        Train the detector on labelled historical data.

        Args:
            df: DataFrame with at minimum a 'label' column (produced by
                scripts/label_regimes.py) plus the feature columns the
                detector needs.

        Returns:
            self (for chaining)
        """
        ...

    @abstractmethod
    def detect(self, obs: pd.Series | dict) -> RegimeResult:
        """
        Classify the current market observation.

        Args:
            obs: a single-bar observation — either a named pd.Series or
                 a plain dict with the expected feature keys.

        Returns:
            RegimeResult with label, confidence, and timeframe.
        """
        ...

    # ------------------------------------------------------------------
    # Convenience helpers (may be overridden)
    # ------------------------------------------------------------------

    def detect_batch(self, df: pd.DataFrame) -> list[RegimeResult]:
        """
        Run detect() on every row of *df*. Useful for backtesting.

        Args:
            df: DataFrame where each row is one bar observation.

        Returns:
            List of RegimeResult, one per row, in order.
        """
        if not self._fitted:
            raise RuntimeError("Detector must be fit() before calling detect_batch()")
        return [self.detect(row) for _, row in df.iterrows()]

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return f"{self.__class__.__name__}(timeframe={self.timeframe!r}, {status})"
