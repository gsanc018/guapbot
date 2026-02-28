"""
guapbot.models.base
-------------------
Base interface for every trading model in Layer 4.

All five models (RL, LSTM, TrendFollowing, MeanReversion, GradientBoost)
implement this interface. The ensemble meta-learner treats them as a
homogeneous list — it does not care what is inside.

Interface contract:
    model.fit(df)         → train on feature DataFrame
    model.predict(obs)    → signal in [-1.0, +1.0]
    model.confidence(obs) → float in [0.0, 1.0]

One model instance is trained per asset. XBTUSD and ETHUSD each get
their own separate fitted model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from guapbot.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ModelMetadata:
    """
    Immutable record of a model's identity and training provenance.

    Stored alongside the model weights for registry and lineage tracking.
    """

    model_class: str            # e.g. 'RLAgent', 'LSTMModel'
    pair: str                   # e.g. 'XBTUSD'
    strategy: str               # 'money_printer' or 'sat_stacker'
    version: str = "0.1.0"
    trained_at: datetime = field(default_factory=datetime.utcnow)
    train_start: str = ""       # ISO date string
    train_end: str = ""
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "model_class": self.model_class,
            "pair": self.pair,
            "strategy": self.strategy,
            "version": self.version,
            "trained_at": self.trained_at.isoformat(),
            "train_start": self.train_start,
            "train_end": self.train_end,
            "notes": self.notes,
        }


class BaseModel(ABC):
    """
    Abstract base for all GuapBot trading models.

    Key design choices:
    - predict() returns a float in [-1, +1], not a discrete {-1, 0, 1}
    - confidence() is a SEPARATE call — callers can use signal without it
    - fit() receives a full feature DataFrame — models own their feature selection
    - Models do not know about regime, position sizing, or the ensemble

    Subclasses must implement fit(), predict(), and confidence().
    save() and load() should be overridden for models with learnable weights.
    """

    def __init__(self, pair: str, strategy: str) -> None:
        """
        Args:
            pair:     trading pair this model is trained for, e.g. 'XBTUSD'
            strategy: 'money_printer' or 'sat_stacker'
        """
        if strategy not in ("money_printer", "sat_stacker"):
            raise ValueError(f"strategy must be 'money_printer' or 'sat_stacker', got {strategy!r}")
        self.pair = pair
        self.strategy = strategy
        self._fitted = False
        self.metadata: ModelMetadata | None = None

    # ------------------------------------------------------------------
    # Abstract interface — all three must be implemented
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseModel":
        """
        Train the model on historical feature data.

        Args:
            df: DataFrame of features produced by the feature pipeline.
                Must include a target column (direction or return).
                The exact columns required depend on the model type.

        Returns:
            self (for chaining)
        """
        ...

    @abstractmethod
    def predict(self, obs: pd.Series | dict) -> float:
        """
        Generate a directional signal for the current bar.

        Args:
            obs: a single observation — either a named pd.Series or a
                 plain dict with the required feature keys.

        Returns:
            float in [-1.0, +1.0]
                -1.0 = maximum bearish (max short)
                 0.0 = flat / no position
                +1.0 = maximum bullish (max long)
        """
        ...

    @abstractmethod
    def confidence(self, obs: pd.Series | dict) -> float:
        """
        Return the model's confidence in the current prediction.

        Args:
            obs: same observation passed to predict()

        Returns:
            float in [0.0, 1.0]
                0.0 = no confidence (ensemble will discount heavily)
                1.0 = maximum confidence
        """
        ...

    # ------------------------------------------------------------------
    # Optional — override for models with serialisable weights
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist model weights and metadata to *path*. Override in subclasses."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement save()")

    def load(self, path: str) -> "BaseModel":
        """Load weights and metadata from *path*. Override in subclasses."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement load()")

    # ------------------------------------------------------------------
    # Helpers (do not override)
    # ------------------------------------------------------------------

    def _validate_signal(self, signal: float) -> float:
        """Clip signal to [-1, +1] and warn if it was out of range."""
        if not -1.0 <= signal <= 1.0:
            log.warning(
                "%s.predict() returned %.4f for %s, clipping to [-1, 1]",
                self.__class__.__name__, signal, self.pair,
            )
            return max(-1.0, min(1.0, signal))
        return signal

    def _validate_confidence(self, conf: float) -> float:
        """Clip confidence to [0, 1] and warn if out of range."""
        if not 0.0 <= conf <= 1.0:
            log.warning(
                "%s.confidence() returned %.4f for %s, clipping to [0, 1]",
                self.__class__.__name__, conf, self.pair,
            )
            return max(0.0, min(1.0, conf))
        return conf

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return f"{self.__class__.__name__}(pair={self.pair!r}, strategy={self.strategy!r}, {status})"
