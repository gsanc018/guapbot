"""
guapbot.regime.detector
-----------------------
Production-ready regime detector for runtime use.

Wraps HMMDetector with:
  - Graceful fallback (low-confidence neutral result) when not yet fitted,
    so the rest of the system can run in degraded mode without crashing.
  - Disk-based persistence via models/regime/{pair}_{timeframe}.pkl.
  - build_regime_detectors() factory for constructing all 3 timeframe
    detectors in one call.

Usage:
    # First time — after running scripts/label_regimes.py:
    detector = Detector(timeframe="1h")
    detector.fit(labeled_df)
    detector.save("XBTUSD")

    # Every subsequent run:
    detectors = build_regime_detectors("XBTUSD")   # loads from disk

    # Slow-clock loop (once per 1h bar):
    obs = pipeline.get_observation("XBTUSD")
    regime_vector = [d.detect(obs) for d in detectors]
    # → [RegimeResult('trending', 0.87, '1h'),
    #    RegimeResult('ranging',  0.71, '4h'),
    #    RegimeResult('bullish',  0.64, 'daily')]
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from guapbot.regime.base import RegimeDetector, RegimeResult
from guapbot.regime.statistical import HMMDetector
from guapbot.utils.logging import get_logger

log = get_logger(__name__)

_DEFAULT_LABEL: dict[str, str] = {
    "1h": "ranging",
    "4h": "ranging",
    "daily": "neutral",
}

_DEFAULT_MODEL_DIR = Path("models/regime")


class Detector(RegimeDetector):
    """
    Production regime detector (one instance per timeframe per pair).

    Wraps HMMDetector and adds graceful fallback, transparent persistence,
    and a clean public API for the rest of the system to consume.
    """

    def __init__(self, timeframe: str, model_dir: Path | None = None) -> None:
        """
        Args:
            timeframe:  '1h', '4h', or 'daily'
            model_dir:  root directory for saved models
                        (default: models/regime/)
        """
        super().__init__(timeframe)
        self._model_dir = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR
        self._hmm = HMMDetector(timeframe)

    # ------------------------------------------------------------------
    # RegimeDetector interface
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "Detector":
        """
        Train the underlying HMM.

        Args:
            df: DataFrame with REGIME_FEATURES columns + 'label' column.
                Produced by scripts/label_regimes.py.
        """
        self._hmm.fit(df)
        self._fitted = True
        return self

    def detect(self, obs: pd.Series | dict) -> RegimeResult:
        """
        Classify the current bar.

        If the detector has not been fitted yet (no model on disk),
        returns a low-confidence fallback so the rest of the system
        continues running without crashing.
        """
        if not self._fitted:
            log.warning(
                "Regime detector not fitted — returning fallback result | timeframe=%s",
                self.timeframe,
            )
            return RegimeResult(
                label=_DEFAULT_LABEL[self.timeframe],
                confidence=0.0,
                timeframe=self.timeframe,
            )
        return self._hmm.detect(obs)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, pair: str) -> Path:
        """
        Save the fitted HMM to models/regime/{pair}_{timeframe}.pkl

        Args:
            pair: trading pair string, e.g. 'XBTUSD'

        Returns:
            Path where the model was written.
        """
        path = self._model_path(pair)
        self._hmm.save(path)
        log.info("Detector saved | pair=%s timeframe=%s path=%s", pair, self.timeframe, path)
        return path

    def load(self, pair: str) -> "Detector":
        """
        Load a previously saved HMM from disk.

        Args:
            pair: trading pair string, e.g. 'XBTUSD'

        Returns:
            self (for chaining)

        Raises:
            FileNotFoundError: if no saved model exists for this pair/timeframe.
        """
        path = self._model_path(pair)
        if not path.exists():
            raise FileNotFoundError(
                f"No saved regime model at {path}. "
                f"Run scripts/label_regimes.py then 'guapbot regime fit' first."
            )
        self._hmm.load(path)
        self._fitted = True
        log.info("Detector loaded | pair=%s timeframe=%s", pair, self.timeframe)
        return self

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _model_path(self, pair: str) -> Path:
        return self._model_dir / f"{pair}_{self.timeframe}.pkl"


def build_regime_detectors(
    pair: str,
    model_dir: Path | None = None,
    *,
    auto_load: bool = True,
) -> list[Detector]:
    """
    Build all three regime detectors (1h, 4h, daily) for a trading pair.

    If saved models exist on disk and auto_load=True, they are loaded
    automatically.  Otherwise, detectors are returned unfitted — they
    will return low-confidence fallback results until fit() is called.

    Args:
        pair:       trading pair, e.g. 'XBTUSD'
        model_dir:  override the default models/regime/ directory
        auto_load:  attempt to load saved models (default True)

    Returns:
        [1h_detector, 4h_detector, daily_detector]
    """
    detectors: list[Detector] = []
    for tf in ("1h", "4h", "daily"):
        d = Detector(timeframe=tf, model_dir=model_dir)
        if auto_load:
            try:
                d.load(pair)
            except FileNotFoundError:
                log.info(
                    "No saved regime model — detector is unfitted | pair=%s timeframe=%s",
                    pair, tf,
                )
        detectors.append(d)
    return detectors
