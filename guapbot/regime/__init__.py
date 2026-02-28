"""GuapBot regime layer."""

from guapbot.regime.base import RegimeDetector, RegimeResult
from guapbot.regime.statistical import HMMDetector, REGIME_FEATURES
from guapbot.regime.rule_labeler import RuleLabeler
from guapbot.regime.detector import Detector, build_regime_detectors

__all__ = [
    "RegimeDetector",
    "RegimeResult",
    "HMMDetector",
    "REGIME_FEATURES",
    "RuleLabeler",
    "Detector",
    "build_regime_detectors",
]
