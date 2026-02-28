"""
guapbot.regime.statistical
--------------------------
HMM-based regime detector.

Uses a Gaussian Hidden Markov Model (hmmlearn.hmm.GaussianHMM) trained
on a subset of technical features.  After fitting, each HMM state is
resolved to a human-readable regime label via majority-vote against
LLM-generated ground-truth labels (produced by scripts/label_regimes.py).

Regime features used:
    adx_14        — directional trend strength (ADX)
    atr_14_pct    — normalised volatility (ATR / close)
    bb_width_20_2 — Bollinger bandwidth  (volatility proxy)
    rsi_14        — momentum state
    volume_ratio  — volume anomaly (current / 20-bar SMA)
    log_ret_1     — 1-bar log return (price direction)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from guapbot.regime.base import RegimeDetector, RegimeResult
from guapbot.utils.logging import get_logger

log = get_logger(__name__)

# Features the HMM is trained on.  Order matters — must be consistent
# between fit() and detect().
REGIME_FEATURES: list[str] = [
    "adx_14",
    "atr_14_pct",
    "bb_width_20_2",
    "rsi_14",
    "volume_ratio",
    "log_ret_1",
]

_INTRADAY_LABELS: list[str] = ["trending", "ranging", "volatile"]
_DAILY_LABELS: list[str] = ["bullish", "bearish", "neutral"]


def _valid_labels_for(timeframe: str) -> list[str]:
    return _DAILY_LABELS if timeframe == "daily" else _INTRADAY_LABELS


class HMMDetector(RegimeDetector):
    """
    Gaussian HMM regime detector.

    Training workflow:
        1. Run scripts/label_regimes.py → saves CSV with 'label' column.
        2. Load that CSV, call detector.fit(df).
        3. Call detector.save(path) to persist.
        4. At runtime, detector.load(path) then detector.detect(obs).

    Confidence is the posterior probability of the predicted state:
        high confidence → HMM is certain which state we're in.
        low confidence  → posterior is spread across states.
    """

    def __init__(self, timeframe: str, n_components: int = 3) -> None:
        """
        Args:
            timeframe:    '1h', '4h', or 'daily'
            n_components: number of HMM hidden states (default 3 matches
                          the 3 regime labels per timeframe)
        """
        super().__init__(timeframe)
        self.n_components = n_components
        self._hmm: Any = None           # GaussianHMM (imported lazily)
        self._scaler: StandardScaler = StandardScaler()
        self._state_to_label: dict[int, str] = {}
        self._valid_labels = _valid_labels_for(timeframe)

    # ------------------------------------------------------------------
    # RegimeDetector interface
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "HMMDetector":
        """
        Train the HMM on labelled feature data.

        Args:
            df: DataFrame with columns REGIME_FEATURES + 'label'.
                Produced by scripts/label_regimes.py.

        Returns:
            self
        """
        self._validate_fit_df(df)
        X = df[REGIME_FEATURES].values.astype(float)

        # Drop rows with NaN (warm-up period at start of history)
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        labels = df["label"].iloc[mask.nonzero()[0]]

        X_scaled = self._scaler.fit_transform(X)

        from hmmlearn.hmm import GaussianHMM  # lazy import
        self._hmm = GaussianHMM(
            n_components=self.n_components,
            covariance_type="full",
            n_iter=200,
            random_state=42,
            verbose=False,
        )
        self._hmm.fit(X_scaled)
        self._resolve_state_labels(X_scaled, labels)
        self._fitted = True

        log.info(
            "HMMDetector fitted | timeframe=%s n_bars=%d state_map=%s",
            self.timeframe, len(X), self._state_to_label,
        )
        return self

    def detect(self, obs: pd.Series | dict) -> RegimeResult:
        """
        Classify a single bar observation.

        Args:
            obs: named Series or dict with at least REGIME_FEATURES keys.

        Returns:
            RegimeResult with label, confidence, timeframe.
        """
        if not self._fitted:
            raise RuntimeError(
                "HMMDetector must be fit() before calling detect(). "
                "Run scripts/label_regimes.py then call fit()."
            )
        X = self._extract_features(obs)
        X_scaled = self._scaler.transform(X)

        state = int(self._hmm.predict(X_scaled)[0])
        posteriors = self._hmm.predict_proba(X_scaled)[0]
        confidence = float(posteriors[state])

        label = self._state_to_label.get(state, self._valid_labels[0])
        return RegimeResult(label=label, confidence=confidence, timeframe=self.timeframe)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """
        Persist the fitted detector to disk.

        Saves: HMM parameters, scaler state, and state→label mapping.

        Args:
            path: file path (e.g. models/regime/XBTUSD_1h.pkl)
        """
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted detector.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "hmm": self._hmm,
            "scaler": self._scaler,
            "state_to_label": self._state_to_label,
            "timeframe": self.timeframe,
            "n_components": self.n_components,
        }
        try:
            import joblib
            joblib.dump(payload, path)
        except ImportError:
            with open(path, "wb") as f:
                pickle.dump(payload, f)

        log.info("HMMDetector saved | path=%s", path)

    def load(self, path: Path) -> "HMMDetector":
        """
        Load a previously saved detector from disk.

        Args:
            path: file path written by save()

        Returns:
            self (for chaining)
        """
        path = Path(path)
        try:
            import joblib
            payload = joblib.load(path)
        except ImportError:
            with open(path, "rb") as f:
                payload = pickle.load(f)

        self._hmm = payload["hmm"]
        self._scaler = payload["scaler"]
        self._state_to_label = payload["state_to_label"]
        self.n_components = payload["n_components"]
        self._fitted = True

        log.info("HMMDetector loaded | path=%s timeframe=%s", path, self.timeframe)
        return self

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_features(self, obs: pd.Series | dict) -> np.ndarray:
        """Extract REGIME_FEATURES from obs in correct order."""
        if isinstance(obs, dict):
            values = [float(obs.get(f, 0.0)) for f in REGIME_FEATURES]
        else:
            values = [
                float(obs[f]) if f in obs.index else 0.0
                for f in REGIME_FEATURES
            ]
        return np.array(values, dtype=float).reshape(1, -1)

    def _resolve_state_labels(self, X_scaled: np.ndarray, labels: pd.Series) -> None:
        """
        Map each HMM state index to a regime label via majority vote.

        For each HMM state, find which LLM label appears most often
        among the bars the HMM assigns to that state.
        """
        states = self._hmm.predict(X_scaled)
        state_to_label: dict[int, str] = {}

        for state in range(self.n_components):
            mask = states == state
            if mask.sum() == 0:
                # State never visited — positional fallback
                fallback = self._valid_labels[state % len(self._valid_labels)]
                state_to_label[state] = fallback
                log.warning(
                    "HMM state %d never visited, using fallback label '%s'",
                    state, fallback,
                )
                continue
            counts = labels.iloc[mask].value_counts()
            state_to_label[state] = str(counts.index[0])

        self._state_to_label = state_to_label

    def _validate_fit_df(self, df: pd.DataFrame) -> None:
        """Raise informative errors for bad input."""
        missing = [f for f in REGIME_FEATURES if f not in df.columns]
        if missing:
            raise ValueError(
                f"fit() DataFrame is missing regime features: {missing}\n"
                f"Expected columns: {REGIME_FEATURES}"
            )
        if "label" not in df.columns:
            raise ValueError(
                "fit() DataFrame must have a 'label' column. "
                "Run scripts/label_regimes.py first."
            )
        valid = _valid_labels_for(self.timeframe)
        bad = set(df["label"].dropna().unique()) - set(valid)
        if bad:
            raise ValueError(
                f"Unknown labels for timeframe '{self.timeframe}': {bad}. "
                f"Valid: {valid}"
            )
        if len(df) < 50:
            raise ValueError(
                f"Need at least 50 labelled bars to train HMM, got {len(df)}."
            )
