"""
guapbot/models/trend_following.py

TrendFollowing — rule-based trend-following trading model.

No machine learning. Signal derived entirely from pre-computed indicator
columns in the normalized feature DataFrame. No training required beyond
verifying the expected columns exist.

Strategy logic:
  1. Primary signal: EMA crossover signals (already computed by pipeline).
     1h_ema_9_21_cross  — short-term momentum
     4h_ema_9_21_cross  — medium-term momentum
     1h_ema_50_200_cross — long-term trend (golden/death cross)
  2. Signal strength multiplier: ADX (z-scored) via sigmoid.
     High ADX → strong trend → higher confidence / larger signal.
  3. Donchian position: 1h_dc_pct_20 — position within Donchian channel.
     Near upper → bullish; near lower → bearish. Scaled contribution.
  4. PSAR confirmation: 1h_psar_signal (+1/-1) adds small weight.

Final signal = weighted average of all available components, clipped to [-1, 1].

Column names follow the pipeline convention: {timeframe}_{indicator}.
The pipeline never z-scores _cross and _signal columns (they are in
_NO_NORM_PATTERNS), so they remain {-1, 0, +1}. Continuous columns
(adx, dc_pct) are z-scored and carry statistical meaning.
"""
from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

from guapbot.models.base import BaseModel
from guapbot.utils.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Column preferences (ordered: most preferred first)
# ---------------------------------------------------------------------------

# Binary crossover signals — not z-scored by pipeline
_CROSS_COLS = [
    ("1h_ema_9_21_cross", 0.35),   # short-term trend, highest weight
    ("4h_ema_9_21_cross", 0.30),   # medium-term trend
    ("1h_ema_50_200_cross", 0.15), # long-term golden/death cross
    ("4h_ema_50_200_cross", 0.10), # long-term on 4h
    ("1h_psar_signal", 0.10),      # parabolic SAR confirmation
]

# Continuous z-scored columns used for signal strength and confirmation
_ADX_COLS = ["1h_adx_14", "4h_adx_14", "adx_14"]
_DC_PCT_COLS = ["1h_dc_pct_20", "dc_pct_20"]

# Weight given to Donchian channel position (rest comes from cross signals)
_DC_WEIGHT = 0.10

# ADX sigmoid scale factor: how strongly ADX amplifies the base signal.
# At ADX z-score=0 (average trend strength), multiplier ≈ 0.73.
# At ADX z-score=2 (strong trend), multiplier ≈ 0.95.
_ADX_SIGMOID_SCALE = 1.5


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


class TrendFollowing(BaseModel):
    """
    Rule-based trend-following model using EMA crossovers, ADX, and
    Donchian channel position. No training required.

    One instance per asset (e.g. XBTUSD). fit() validates that the
    expected feature columns are present and notes which are available.
    """

    def __init__(self, pair: str, strategy: str) -> None:
        super().__init__(pair, strategy)
        self._available_cross: list[tuple[str, float]] = []
        self._adx_col: str | None = None
        self._dc_col: str | None = None

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "TrendFollowing":
        """
        Validate feature columns and note which signals are available.

        Args:
            df: Normalized feature DataFrame from FeaturePipeline.
                May include a 'target' column — it is ignored.

        Returns:
            self
        """
        cols = set(df.columns)

        # Resolve available crossover columns
        self._available_cross = [
            (col, w) for col, w in _CROSS_COLS if col in cols
        ]
        if not self._available_cross:
            log.warning(
                "TrendFollowing(%s): no EMA cross columns found. "
                "Model will return 0.0 signals. Expected: %s",
                self.pair, [c for c, _ in _CROSS_COLS],
            )

        # Resolve ADX column
        self._adx_col = next((c for c in _ADX_COLS if c in cols), None)
        if not self._adx_col:
            log.warning(
                "TrendFollowing(%s): no ADX column found — signal strength "
                "will not be scaled. Expected one of: %s", self.pair, _ADX_COLS
            )

        # Resolve Donchian channel column
        self._dc_col = next((c for c in _DC_PCT_COLS if c in cols), None)

        self._fitted = True
        log.info(
            "TrendFollowing(%s) fitted: %d cross cols, adx=%s, dc=%s",
            self.pair, len(self._available_cross), self._adx_col, self._dc_col,
        )
        return self

    def predict(self, obs: Union[pd.Series, dict]) -> float:
        """
        Return a directional signal in [-1.0, +1.0].

        Args:
            obs: Single-bar observation (pd.Series or dict).

        Returns:
            float in [-1.0, +1.0]
        """
        if not self._fitted:
            raise RuntimeError(
                f"TrendFollowing({self.pair}) must be fit() before predict()"
            )

        obs_map = obs if isinstance(obs, dict) else obs.to_dict()

        # Step 1: weighted sum of available crossover signals
        total_weight = 0.0
        weighted_signal = 0.0

        for col, w in self._available_cross:
            val = float(obs_map.get(col, 0.0))
            if not np.isnan(val):
                weighted_signal += val * w
                total_weight += w

        # Normalize cross signal to [-1, 1]
        if total_weight > 0:
            cross_signal = weighted_signal / total_weight
        else:
            cross_signal = 0.0

        # Step 2: Donchian position — map [0, 1] range to [-1, +1]
        dc_signal = 0.0
        if self._dc_col:
            dc_raw = float(obs_map.get(self._dc_col, np.nan))
            if not np.isnan(dc_raw):
                # dc_pct_20 is z-scored — convert to approximately [-1, 1]
                # by using its sign and clipped magnitude
                dc_signal = float(np.clip(dc_raw / 2.0, -1.0, 1.0))

        # Blend cross signal and Donchian
        if dc_signal != 0.0:
            base_signal = (1.0 - _DC_WEIGHT) * cross_signal + _DC_WEIGHT * dc_signal
        else:
            base_signal = cross_signal

        # Step 3: ADX multiplier — amplify signal in strong trends
        if self._adx_col:
            adx_z = float(obs_map.get(self._adx_col, 0.0))
            if not np.isnan(adx_z):
                adx_multiplier = _sigmoid(adx_z * _ADX_SIGMOID_SCALE)
                base_signal = base_signal * adx_multiplier

        return self._validate_signal(base_signal)

    def confidence(self, obs: Union[pd.Series, dict]) -> float:
        """
        Return confidence in [0.0, 1.0] based on ADX trend strength.

        High ADX z-score → stronger trend → higher confidence.
        Falls back to 0.5 if ADX column not available.
        """
        if not self._fitted:
            raise RuntimeError(
                f"TrendFollowing({self.pair}) must be fit() before confidence()"
            )

        if not self._adx_col:
            return 0.5

        obs_map = obs if isinstance(obs, dict) else obs.to_dict()
        adx_z = float(obs_map.get(self._adx_col, 0.0))

        if np.isnan(adx_z):
            return 0.5

        # Map ADX z-score to [0.3, 0.9] via sigmoid
        raw = _sigmoid(adx_z * _ADX_SIGMOID_SCALE)  # [0, 1]
        conf = 0.3 + raw * 0.6                       # [0.3, 0.9]
        return self._validate_confidence(conf)

    def save(self, path: str) -> None:
        raise NotImplementedError(
            "TrendFollowing has no learnable weights to save. "
            "Re-create and re-fit from the feature DataFrame."
        )

    def load(self, path: str) -> "TrendFollowing":
        raise NotImplementedError(
            "TrendFollowing has no learnable weights to load."
        )
