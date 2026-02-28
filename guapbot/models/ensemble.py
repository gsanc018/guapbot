"""
guapbot.models.ensemble
-----------------------
Ensemble meta-learner interface.

The ensemble is the brain of GuapBot. It takes the raw signals from
every sub-model, the current regime vector, and each model's rolling
performance — and combines them into a single final trading signal.

Implementation: trained LightGBM meta-learner (walk-forward stacking).
Online daily updates via LightGBM continue_training — never a full retrain.

Interface contract:
    ensemble.fit(signal_history)         → walk-forward stacking train
    ensemble.combine(signals, regimes)   → final signal float
    ensemble.update(outcome)             → online learning step
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from guapbot.regime.base import RegimeResult
from guapbot.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ModelSignal:
    """
    Signal + confidence from a single sub-model for one bar.

    Attributes:
        model_name:  identifier, e.g. 'rl_agent', 'lstm', 'trend_following'
        pair:        'XBTUSD' or 'ETHUSD'
        signal:      [-1.0, +1.0]
        confidence:  [0.0, 1.0]
        rolling_sharpe: model's rolling 30d Sharpe — ensemble uses this
                        to weight models dynamically
    """

    model_name: str
    pair: str
    signal: float
    confidence: float
    rolling_sharpe: float = 0.0

    def __post_init__(self) -> None:
        if not -1.0 <= self.signal <= 1.0:
            raise ValueError(f"signal must be in [-1, 1], got {self.signal}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass
class EnsembleInput:
    """
    Everything the ensemble needs to produce a final signal.

    Attributes:
        signals:  list of ModelSignal from each sub-model
        regimes:  list of RegimeResult — one per timeframe (1h, 4h, daily)
        pair:     the asset being traded
    """

    signals: list[ModelSignal]
    regimes: list[RegimeResult]
    pair: str

    @property
    def regime_dict(self) -> dict[str, RegimeResult]:
        """Index regimes by timeframe for easy lookup."""
        return {r.timeframe: r for r in self.regimes}


@dataclass
class TradeOutcome:
    """
    Outcome of a single bar's trade — fed back for online learning.

    Attributes:
        bar_time:       ISO timestamp of the bar
        pair:           'XBTUSD' or 'ETHUSD'
        final_signal:   what the ensemble output
        realised_return: actual PnL as a fraction
        regime_labels:  regime labels active at bar time (for feature reconstruction)
        model_signals:  sub-model signals at bar time
    """

    bar_time: str
    pair: str
    final_signal: float
    realised_return: float
    regime_labels: dict[str, str] = field(default_factory=dict)
    model_signals: dict[str, float] = field(default_factory=dict)


class BaseEnsemble(ABC):
    """
    Abstract base for the GuapBot ensemble meta-learner.

    The concrete implementation in models/ensemble.py trains a LightGBM
    model using walk-forward stacking, then updates it daily via online
    learning. This base class defines the interface that the execution
    layer and tests depend on.
    """

    def __init__(self, pair: str) -> None:
        """
        Args:
            pair: 'XBTUSD' or 'ETHUSD'
        """
        self.pair = pair
        self._fitted = False

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, signal_history: pd.DataFrame) -> "BaseEnsemble":
        """
        Train the meta-learner via walk-forward stacking.

        Args:
            signal_history: DataFrame with columns:
                - one column per model signal  (e.g. 'rl_signal')
                - one column per model confidence
                - regime columns for each timeframe
                - rolling_sharpe_<model> columns
                - 'target' column: actual return or direction label

        Returns:
            self
        """
        ...

    @abstractmethod
    def combine(self, inputs: EnsembleInput) -> float:
        """
        Combine sub-model signals into a single final signal.

        Args:
            inputs: EnsembleInput with signals and regime vector.

        Returns:
            float in [-1.0, +1.0] — the final trading signal.
        """
        ...

    @abstractmethod
    def update(self, outcome: TradeOutcome) -> None:
        """
        Online learning step. Called after each bar's outcome is known.

        Uses LightGBM continue_training to incrementally update the
        meta-learner with fresh data without a full retrain.

        Args:
            outcome: TradeOutcome from the previous bar.
        """
        ...

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return f"{self.__class__.__name__}(pair={self.pair!r}, {status})"
