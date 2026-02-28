"""
guapbot/models/gradient_boost.py

GradientBoost — supervised ML model using XGBoost + LightGBM ensemble.

Trains two gradient boosting classifiers (XGBoost and LightGBM) on the
full normalized feature set, then averages their probability outputs for
a final directional signal.

Task: binary direction classification.
  Target: +1 if next-bar log return > 0, -1 otherwise.
  Encoded as {0, 1} for classifier training (0 = down, 1 = up).

Training:
  - Receives normalized feature DataFrame with 'target' column injected by CLI.
  - Drops non-feature columns before training (target, any string columns).
  - Both XGBoost and LightGBM trained with same hyperparams for consistency.
  - Predictions averaged across both models (ensemble of two boosts).

Inference:
  - predict(obs) → (prob_up - prob_down) → signal in [-1, 1]
    E.g. prob_up=0.7, prob_down=0.3 → 0.7 - 0.3 = 0.4 → signal = +0.4
  - confidence(obs) → max(prob_up, prob_down) → [0.5, 1.0] mapped to [0, 1]

Persistence:
  - joblib.dump / joblib.load (compact, fast, works with both xgb and lgb)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from guapbot.models.base import BaseModel
from guapbot.utils.logging import get_logger

log = get_logger(__name__)

# Columns to drop before training — not features
_DROP_COLS = {"target", "timestamp"}

# Default hyperparameters — conservative to avoid overfitting
_N_ESTIMATORS = 500
_MAX_DEPTH = 4
_LEARNING_RATE = 0.05
_SUBSAMPLE = 0.8
_COL_SAMPLE = 0.8


class GradientBoost(BaseModel):
    """
    Gradient boosting ensemble (XGBoost + LightGBM) for binary direction
    classification. Trains one classifier per library, averages predictions.

    One instance per asset. Requires 'target' column in fit() DataFrame.
    """

    def __init__(self, pair: str, strategy: str) -> None:
        super().__init__(pair, strategy)
        self._xgb = None
        self._lgb = None
        self._feature_cols: list[str] = []

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "GradientBoost":
        """
        Train XGBoost + LightGBM classifiers on normalized features.

        Args:
            df: Normalized feature DataFrame with a 'target' column.
                target = +1 (next bar up) or -1 (next bar down).
                NaN rows and last row (no future target) should be dropped
                by the caller before passing.

        Returns:
            self
        """
        try:
            import xgboost as xgb
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError(
                "xgboost and lightgbm are required. "
                "Install with: pip install xgboost lightgbm"
            ) from exc

        if "target" not in df.columns:
            raise ValueError(
                "GradientBoost.fit() requires a 'target' column in df. "
                "Expected values: +1 (up) or -1 (down)."
            )

        # Determine feature columns
        self._feature_cols = [
            c for c in df.columns
            if c not in _DROP_COLS and df[c].dtype in (np.float64, np.float32, float)
        ]
        if not self._feature_cols:
            raise ValueError("No numeric feature columns found in df")

        X = df[self._feature_cols].to_numpy(dtype=np.float32)
        # Encode target: -1 → 0 (down), +1 → 1 (up)
        y = ((df["target"] > 0).astype(int)).to_numpy()

        n_samples = len(X)
        log.info(
            "GradientBoost(%s): training on %d samples, %d features",
            self.pair, n_samples, len(self._feature_cols),
        )

        # Train XGBoost
        self._xgb = xgb.XGBClassifier(
            n_estimators=_N_ESTIMATORS,
            max_depth=_MAX_DEPTH,
            learning_rate=_LEARNING_RATE,
            subsample=_SUBSAMPLE,
            colsample_bytree=_COL_SAMPLE,
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
            n_jobs=-1,
        )
        self._xgb.fit(X, y)
        log.info("GradientBoost(%s): XGBoost trained", self.pair)

        # Train LightGBM
        self._lgb = lgb.LGBMClassifier(
            n_estimators=_N_ESTIMATORS,
            max_depth=_MAX_DEPTH,
            learning_rate=_LEARNING_RATE,
            subsample=_SUBSAMPLE,
            colsample_bytree=_COL_SAMPLE,
            verbosity=-1,
            n_jobs=-1,
        )
        self._lgb.fit(X, y)
        log.info("GradientBoost(%s): LightGBM trained", self.pair)

        self._fitted = True
        return self

    def predict(self, obs: Union[pd.Series, dict]) -> float:
        """
        Return signal = (prob_up - prob_down) averaged across both models.

        Args:
            obs: Single-bar observation (pd.Series or dict).

        Returns:
            float in [-1.0, +1.0]
        """
        if not self._fitted:
            raise RuntimeError(
                f"GradientBoost({self.pair}) must be fit() before predict()"
            )

        X = self._obs_to_array(obs)
        avg_prob_up = self._avg_prob_up(X)
        signal = avg_prob_up * 2.0 - 1.0  # [0,1] → [-1, +1]
        return self._validate_signal(signal)

    def confidence(self, obs: Union[pd.Series, dict]) -> float:
        """
        Return confidence = distance of prediction from 0.5 probability.

        max(prob_up, prob_down) scaled to [0.5, 1.0] then mapped to [0, 1].
        """
        if not self._fitted:
            raise RuntimeError(
                f"GradientBoost({self.pair}) must be fit() before confidence()"
            )

        X = self._obs_to_array(obs)
        avg_prob_up = self._avg_prob_up(X)
        # Distance from 0.5 doubles the confidence signal
        conf = abs(avg_prob_up - 0.5) * 2.0
        return self._validate_confidence(conf)

    def save(self, path: str) -> None:
        """
        Persist both models and feature column list to a single .pkl file.

        Args:
            path: File path, e.g. 'models/money_printer/gradient_boost/XBTUSD/model.pkl'
        """
        if not self._fitted:
            raise RuntimeError("Model must be fit() before save()")

        try:
            import joblib
        except ImportError as exc:
            raise ImportError("joblib is required for save(). Install: pip install joblib") from exc

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "xgb": self._xgb,
            "lgb": self._lgb,
            "feature_cols": self._feature_cols,
            "pair": self.pair,
            "strategy": self.strategy,
        }
        joblib.dump(payload, path)
        log.info("GradientBoost(%s): saved to %s", self.pair, path)

    def load(self, path: str) -> "GradientBoost":
        """
        Load both models and feature column list from a .pkl file.

        Args:
            path: File path previously passed to save().

        Returns:
            self (fitted)
        """
        try:
            import joblib
        except ImportError as exc:
            raise ImportError("joblib is required for load(). Install: pip install joblib") from exc

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        payload = joblib.load(path)
        self._xgb = payload["xgb"]
        self._lgb = payload["lgb"]
        self._feature_cols = payload["feature_cols"]
        self._fitted = True
        log.info("GradientBoost(%s): loaded from %s", self.pair, path)
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _obs_to_array(self, obs: Union[pd.Series, dict]) -> np.ndarray:
        """Convert an observation to a 2D float32 array for sklearn-style predict."""
        obs_map = obs if isinstance(obs, dict) else obs.to_dict()
        values = [float(obs_map.get(col, 0.0)) for col in self._feature_cols]
        return np.array(values, dtype=np.float32).reshape(1, -1)

    def _avg_prob_up(self, X: np.ndarray) -> float:
        """Average probability of upward movement across both models."""
        prob_xgb = float(self._xgb.predict_proba(X)[0, 1])
        prob_lgb = float(self._lgb.predict_proba(X)[0, 1])
        return (prob_xgb + prob_lgb) / 2.0
