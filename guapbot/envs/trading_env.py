"""
guapbot/envs/trading_env.py

BitcoinTradingEnv — Gymnasium environment for RL model training.

Single-asset, continuous action space. One instance per asset.
Used exclusively by RLAgent.fit() — not used at inference time.

Observation space:
    All normalized feature columns from FeaturePipeline, stacked as
    a float32 vector. Clipped to [-5, 5] (matching pipeline normalisation).

Action space:
    Box(-1, 1, shape=(1,)) — continuous position fraction.
    -1.0 = maximum short, 0.0 = flat, +1.0 = maximum long.

Reward:
    action * realized_log_return_next_bar
    Positive when position direction matches actual price movement.
    No transaction costs at this stage (Session 7 adds realistic costs).

Episode:
    One pass through the entire training DataFrame, step by step.
    reset() starts at bar 0. done=True when the last bar is reached.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

import gymnasium
from gymnasium import spaces

from guapbot.utils.logging import get_logger

logger = get_logger(__name__)

# Column to use for log returns in the reward function.
# The pipeline produces this for every timeframe; 1h is the action timeframe.
_LOG_RETURN_COL = "1h_log_return"
_LOG_RETURN_FALLBACKS = ["log_return", "1h_log_ret_1", "log_ret_1"]


class BitcoinTradingEnv(gymnasium.Env):
    """
    Gymnasium environment for training the GuapBot RL agent.

    One environment instance covers one asset (e.g. XBTUSD). Provide a
    normalized feature DataFrame from FeaturePipeline — the environment
    treats every column as part of the observation vector.

    Args:
        df:              Normalized feature DataFrame. Every column becomes
                         part of the observation. Should NOT include a
                         'target' column.
        log_return_col:  Column name to use as the per-step reward signal.
                         Defaults to '1h_log_return' with fallbacks.

    Usage:
        env = BitcoinTradingEnv(df)
        obs, _ = env.reset()
        obs, reward, done, truncated, info = env.step(np.array([0.5]))
    """

    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        log_return_col: Optional[str] = None,
    ) -> None:
        if df.empty:
            raise ValueError("df must not be empty")

        self._df = df.reset_index(drop=True)
        self._n_features = len(df.columns)
        self._feature_cols = list(df.columns)

        # Locate the log-return column for reward computation
        self._log_return_col = self._resolve_log_return_col(df, log_return_col)

        # Pre-compute log returns array for fast stepping
        if self._log_return_col:
            self._log_returns = self._df[self._log_return_col].fillna(0.0).to_numpy(dtype=np.float32)
        else:
            logger.warning(
                "No log-return column found — rewards will be zero. "
                "Expected one of: %s", [_LOG_RETURN_COL] + _LOG_RETURN_FALLBACKS
            )
            self._log_returns = np.zeros(len(self._df), dtype=np.float32)

        # Pre-compute observation matrix for fast stepping
        self._obs_matrix = self._df.to_numpy(dtype=np.float32)

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(self._n_features,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        self._current_step: int = 0
        logger.debug(
            "BitcoinTradingEnv: %d bars, %d features, reward col=%r",
            len(self._df), self._n_features, self._log_return_col,
        )

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Reset the environment to the first bar.

        Returns:
            (observation, info_dict)
        """
        if seed is not None:
            super().reset(seed=seed)  # type: ignore[misc]

        self._current_step = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Advance one bar.

        Args:
            action: numpy array of shape (1,) with value in [-1.0, +1.0].

        Returns:
            (obs, reward, terminated, truncated, info)
            - terminated: True on last bar of episode
            - truncated:  always False (no time limit)
            - info:       dict with 'step', 'log_return', 'position'
        """
        position = float(np.clip(action[0], -1.0, 1.0))

        # Reward: position * next-bar log return.
        # obs[t] contains log_return[t] (already realized — no future info).
        # The position is held into bar t+1, so reward uses log_return[t+1].
        next_idx = min(self._current_step + 1, len(self._log_returns) - 1)
        log_ret = float(self._log_returns[next_idx])
        reward = position * log_ret

        self._current_step += 1
        terminated = self._current_step >= len(self._df) - 1
        truncated = False

        obs = self._get_obs()
        info = {
            "step": self._current_step,
            "log_return": log_ret,
            "position": position,
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """No-op — rendering not implemented."""

    def close(self) -> None:
        """Clean up resources (no-op for this env)."""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def feature_cols(self) -> list[str]:
        """Ordered list of feature column names in the observation vector."""
        return self._feature_cols.copy()

    @property
    def n_steps(self) -> int:
        """Total number of steps (bars) in the environment."""
        return len(self._df)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Return the observation vector at the current step."""
        idx = min(self._current_step, len(self._obs_matrix) - 1)
        return self._obs_matrix[idx].copy()

    @staticmethod
    def _resolve_log_return_col(df: pd.DataFrame, override: Optional[str]) -> Optional[str]:
        """Find the log-return column to use for rewards."""
        if override is not None:
            if override in df.columns:
                return override
            logger.warning("Specified log_return_col %r not found in DataFrame", override)

        for col in [_LOG_RETURN_COL] + _LOG_RETURN_FALLBACKS:
            if col in df.columns:
                return col

        return None
