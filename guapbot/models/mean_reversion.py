"""
guapbot/models/mean_reversion.py

MeanReversion — rule-based mean-reversion trading model.

No machine learning. Fades extremes: short overbought, long oversold.
Opposite philosophy to TrendFollowing — signals agree with TrendFollowing
only when momentum diverges from the mean.

Strategy logic:
  1. Bollinger Band signal: 1h_bb_signal_20_2 ∈ {-1, 0, +1}
     Pipeline convention: +1 = price ABOVE upper band (overbought)
                          -1 = price BELOW lower band (oversold)
     Mean reversion FADES this: overbought → short, oversold → long.
     So we NEGATE the signal.
  2. RSI fade: if RSI z-score > 1.5 → short (overbought fade)
                if RSI z-score < -1.5 → long (oversold fade)
  3. Stochastic: stoch_cross ∈ {-1, 0, +1} — negate for reversion.
  4. Bollinger %B z-score: distance from mid-band as continuous signal.
     Faded: high %B → short, low %B → long.

Final signal = weighted blend, clipped to [-1, 1].
Confidence = how far from equilibrium (larger deviation = higher confidence).

Column names follow the pipeline convention: {timeframe}_{indicator}.
BB signal and stoch_cross are NOT z-scored by the pipeline.
RSI and bb_pct are z-scored.
"""
from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

from guapbot.models.base import BaseModel
from guapbot.utils.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Column preferences
# ---------------------------------------------------------------------------

# BB signal columns (not z-scored: {-1, 0, +1}) — negate for reversion
_BB_SIGNAL_COLS = ["1h_bb_signal_20_2", "bb_signal_20_2"]

# RSI z-scored columns
_RSI_COLS = ["1h_rsi_14", "rsi_14", "1h_rsi_7", "rsi_7"]

# Stochastic crossover (not z-scored: {-1, 0, +1}) — negate for reversion
_STOCH_CROSS_COLS = ["1h_stoch_cross", "stoch_cross"]

# Bollinger %B z-scored — negate for reversion
_BB_PCT_COLS = ["1h_bb_pct_20_2", "bb_pct_20_2"]

# Component weights
_W_BB_SIGNAL = 0.40
_W_RSI = 0.30
_W_STOCH = 0.15
_W_BB_PCT = 0.15

# RSI z-score thresholds: beyond these we generate a fade signal
_RSI_FADE_THRESHOLD = 1.0


class MeanReversion(BaseModel):
    """
    Rule-based mean-reversion model. Fades Bollinger Band extremes,
    RSI overbought/oversold zones, and stochastic extremes.

    One instance per asset. fit() validates and notes available columns.
    """

    def __init__(self, pair: str, strategy: str) -> None:
        super().__init__(pair, strategy)
        self._bb_signal_col: str | None = None
        self._rsi_col: str | None = None
        self._stoch_col: str | None = None
        self._bb_pct_col: str | None = None

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "MeanReversion":
        """
        Validate feature columns and note which signals are available.

        Args:
            df: Normalized feature DataFrame. 'target' column is ignored.

        Returns:
            self
        """
        cols = set(df.columns)

        self._bb_signal_col = next((c for c in _BB_SIGNAL_COLS if c in cols), None)
        self._rsi_col = next((c for c in _RSI_COLS if c in cols), None)
        self._stoch_col = next((c for c in _STOCH_CROSS_COLS if c in cols), None)
        self._bb_pct_col = next((c for c in _BB_PCT_COLS if c in cols), None)

        n_available = sum(
            1 for c in [self._bb_signal_col, self._rsi_col, self._stoch_col, self._bb_pct_col]
            if c is not None
        )

        if n_available == 0:
            log.warning(
                "MeanReversion(%s): no mean-reversion signal columns found. "
                "Model will return 0.0. Expected: %s",
                self.pair, _BB_SIGNAL_COLS + _RSI_COLS + _STOCH_CROSS_COLS,
            )

        self._fitted = True
        log.info(
            "MeanReversion(%s) fitted: bb=%s, rsi=%s, stoch=%s, bb_pct=%s",
            self.pair, self._bb_signal_col, self._rsi_col,
            self._stoch_col, self._bb_pct_col,
        )
        return self

    def predict(self, obs: Union[pd.Series, dict]) -> float:
        """
        Return a mean-reversion directional signal in [-1.0, +1.0].

        Positive values indicate an oversold bounce (long); negative
        values indicate an overbought fade (short).

        Args:
            obs: Single-bar observation (pd.Series or dict).

        Returns:
            float in [-1.0, +1.0]
        """
        if not self._fitted:
            raise RuntimeError(
                f"MeanReversion({self.pair}) must be fit() before predict()"
            )

        obs_map = obs if isinstance(obs, dict) else obs.to_dict()
        components: list[tuple[float, float]] = []

        # BB signal: pipeline +1 = overbought → we SHORT → negate
        if self._bb_signal_col:
            val = float(obs_map.get(self._bb_signal_col, 0.0))
            if not np.isnan(val):
                components.append((-val, _W_BB_SIGNAL))

        # RSI fade: convert z-score to fade signal
        if self._rsi_col:
            rsi_z = float(obs_map.get(self._rsi_col, 0.0))
            if not np.isnan(rsi_z):
                # Soft fade: linear beyond threshold, capped at ±1
                if abs(rsi_z) > _RSI_FADE_THRESHOLD:
                    rsi_signal = -np.clip(
                        (rsi_z - np.sign(rsi_z) * _RSI_FADE_THRESHOLD) / 2.0,
                        -1.0, 1.0
                    )
                else:
                    rsi_signal = 0.0
                components.append((float(rsi_signal), _W_RSI))

        # Stochastic: pipeline convention — negate for reversion
        if self._stoch_col:
            val = float(obs_map.get(self._stoch_col, 0.0))
            if not np.isnan(val):
                components.append((-val, _W_STOCH))

        # BB %B z-scored: positive = price near upper band → negate to fade
        if self._bb_pct_col:
            bb_z = float(obs_map.get(self._bb_pct_col, 0.0))
            if not np.isnan(bb_z):
                bb_signal = -np.clip(bb_z / 2.0, -1.0, 1.0)
                components.append((float(bb_signal), _W_BB_PCT))

        if not components:
            return 0.0

        # Weighted average across available components
        total_w = sum(w for _, w in components)
        signal = sum(s * w for s, w in components) / total_w
        return self._validate_signal(signal)

    def confidence(self, obs: Union[pd.Series, dict]) -> float:
        """
        Return confidence in [0.0, 1.0].

        Confidence is proportional to how far price has deviated from
        its mean. A strong Bollinger Band excursion → high confidence.
        """
        if not self._fitted:
            raise RuntimeError(
                f"MeanReversion({self.pair}) must be fit() before confidence()"
            )

        obs_map = obs if isinstance(obs, dict) else obs.to_dict()
        deviations: list[float] = []

        # BB signal magnitude (0 = inside band, 1 = outside)
        if self._bb_signal_col:
            val = abs(float(obs_map.get(self._bb_signal_col, 0.0)))
            if not np.isnan(val):
                deviations.append(val)

        # RSI deviation beyond threshold
        if self._rsi_col:
            rsi_z = abs(float(obs_map.get(self._rsi_col, 0.0)))
            if not np.isnan(rsi_z) and rsi_z > _RSI_FADE_THRESHOLD:
                # Map [threshold, threshold+3] → [0, 1]
                dev = np.clip((rsi_z - _RSI_FADE_THRESHOLD) / 3.0, 0.0, 1.0)
                deviations.append(float(dev))

        # BB %B z-score: larger z = more extreme = more confident
        if self._bb_pct_col:
            bb_z = abs(float(obs_map.get(self._bb_pct_col, 0.0)))
            if not np.isnan(bb_z):
                dev = np.clip(bb_z / 3.0, 0.0, 1.0)
                deviations.append(float(dev))

        if not deviations:
            return 0.4  # low default when no signals available

        # Scale to [0.3, 0.85] — never fully confident on mean reversion
        raw = float(np.mean(deviations))
        conf = 0.3 + raw * 0.55
        return self._validate_confidence(conf)

    def save(self, path: str) -> None:
        raise NotImplementedError(
            "MeanReversion has no learnable weights to save. "
            "Re-create and re-fit from the feature DataFrame."
        )

    def load(self, path: str) -> "MeanReversion":
        raise NotImplementedError(
            "MeanReversion has no learnable weights to load."
        )
