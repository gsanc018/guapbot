"""GuapBot models layer — Layer 4."""

from guapbot.models.base import BaseModel, ModelMetadata
from guapbot.models.ensemble import BaseEnsemble, EnsembleInput, ModelSignal, TradeOutcome
try:
    from guapbot.models.ensemble_lightgbm import EnsembleLearner
except Exception:
    EnsembleLearner = None  # type: ignore[assignment,misc]
from guapbot.models.gradient_boost import GradientBoost
from guapbot.models.lstm import LSTMModel
from guapbot.models.mean_reversion import MeanReversion
from guapbot.models.rl_agent import RLAgent
from guapbot.models.trend_following import TrendFollowing

__all__ = [
    "BaseModel",
    "ModelMetadata",
    "BaseEnsemble",
    "EnsembleInput",
    "ModelSignal",
    "TradeOutcome",
    "EnsembleLearner",
    "TrendFollowing",
    "MeanReversion",
    "GradientBoost",
    "LSTMModel",
    "RLAgent",
]
