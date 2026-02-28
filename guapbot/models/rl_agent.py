"""
guapbot/models/rl_agent.py

RLAgent — Reinforcement Learning agent via Stable-Baselines3.

Default algorithm: SAC (Soft Actor-Critic) — best balance of sample
efficiency and stability for continuous action spaces.

Architecture:
  - Uses BitcoinTradingEnv from guapbot/envs/trading_env.py
  - SB3 SAC with MlpPolicy (default: 2×256 hidden layers)
  - Action space: continuous [-1, 1] (maps directly to position fraction)

Training:
  - Creates BitcoinTradingEnv from the normalized feature DataFrame.
  - Default total_timesteps = len(df) * 10 (10 passes through history).
  - Algos supported: sac (default), ppo, td3
    ppo: good for longer episodes, on-policy
    td3: deterministic policy, good for stable training

Inference:
  - predict(obs): convert obs dict/Series → numpy array → SB3 predict()
  - confidence: fixed 0.6 (RL agents don't expose calibrated probability)
    Use a higher value for live trading once you've validated agent performance.

Persistence:
  - SB3 native .zip format (model.save() / SACModel.load())
  - Feature column order is also stored alongside the model
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Union

import numpy as np
import pandas as pd

from guapbot.models.base import BaseModel
from guapbot.utils.logging import get_logger

log = get_logger(__name__)

# Supported RL algorithms
AlgoType = Literal["sac", "ppo", "td3"]

# Columns to exclude from the observation fed to the RL agent
_DROP_COLS = {"target", "timestamp"}

# Default training budget: 10 passes through training data
_TIMESTEPS_MULTIPLIER = 10

# Fixed confidence for RL agent (no calibrated probability output)
_RL_CONFIDENCE = 0.6


def _load_algo(algo: str):
    """Lazy-import the SB3 algorithm class."""
    try:
        if algo == "sac":
            from stable_baselines3 import SAC
            return SAC
        elif algo == "ppo":
            from stable_baselines3 import PPO
            return PPO
        elif algo == "td3":
            from stable_baselines3 import TD3
            return TD3
        else:
            raise ValueError(f"Unsupported algo: {algo!r}. Choose from: sac, ppo, td3")
    except ImportError as exc:
        raise ImportError(
            "stable-baselines3 is required. "
            "Install with: pip install stable-baselines3"
        ) from exc


class RLAgent(BaseModel):
    """
    Reinforcement Learning trading agent using Stable-Baselines3.

    One instance per asset. Uses continuous action space ∈ [-1, 1].
    SAC is the default algorithm — switch to PPO or TD3 via the `algo`
    parameter or `guapbot train rl --algo ppo`.
    """

    def __init__(self, pair: str, strategy: str, algo: AlgoType = "sac") -> None:
        """
        Args:
            pair:     Trading pair, e.g. 'XBTUSD'.
            strategy: 'money_printer' or 'sat_stacker'.
            algo:     SB3 algorithm: 'sac' (default), 'ppo', 'td3'.
        """
        super().__init__(pair, strategy)
        if algo not in ("sac", "ppo", "td3"):
            raise ValueError(f"algo must be one of sac/ppo/td3, got {algo!r}")
        self.algo = algo
        self._model = None
        self._feature_cols: list[str] = []

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "RLAgent":
        """
        Train the RL agent on normalized features via BitcoinTradingEnv.

        Args:
            df: Normalized feature DataFrame. May include 'target' column
                (it is dropped before passing to the environment).
                Should be the training split only (no future data leakage).

        Returns:
            self
        """
        from guapbot.envs.trading_env import BitcoinTradingEnv

        # Determine feature columns (drop target if present)
        self._feature_cols = [
            c for c in df.columns
            if c not in _DROP_COLS and df[c].dtype in (np.float64, np.float32, float)
        ]
        if not self._feature_cols:
            raise ValueError("No numeric feature columns found in df")

        env_df = df[self._feature_cols].copy()
        env = BitcoinTradingEnv(env_df)

        AlgoCls = _load_algo(self.algo)

        total_timesteps = len(df) * _TIMESTEPS_MULTIPLIER
        log.info(
            "RLAgent(%s, %s): training for %d timesteps on %d bars",
            self.pair, self.algo, total_timesteps, len(df),
        )

        self._model = AlgoCls(
            policy="MlpPolicy",
            env=env,
            verbose=0,
        )
        self._model.learn(total_timesteps=total_timesteps)

        self._fitted = True
        log.info("RLAgent(%s, %s): training complete", self.pair, self.algo)
        return self

    def predict(self, obs: Union[pd.Series, dict]) -> float:
        """
        Return signal in [-1.0, +1.0] from the trained RL policy.

        Args:
            obs: Single-bar observation (pd.Series or dict).

        Returns:
            float in [-1.0, +1.0]
        """
        if not self._fitted:
            raise RuntimeError(
                f"RLAgent({self.pair}) must be fit() before predict()"
            )

        obs_arr = self._obs_to_array(obs)
        action, _ = self._model.predict(obs_arr, deterministic=True)
        signal = float(np.clip(action[0], -1.0, 1.0))
        return self._validate_signal(signal)

    def confidence(self, obs: Union[pd.Series, dict]) -> float:
        """
        Return fixed confidence of 0.6.

        RL agents don't expose calibrated probabilities without
        additional critic evaluation overhead. Use a fixed conservative
        value until live performance validates the agent.
        """
        if not self._fitted:
            raise RuntimeError(
                f"RLAgent({self.pair}) must be fit() before confidence()"
            )
        return _RL_CONFIDENCE

    def save(self, path: str) -> None:
        """
        Save the trained SB3 model and feature column list.

        SB3 saves to a .zip file at `path`. Feature columns are stored
        separately at `path + '.cols.npy'`.

        Args:
            path: File path without extension, e.g.
                  'models/money_printer/rl_agent/XBTUSD/model'
        """
        if not self._fitted:
            raise RuntimeError("Model must be fit() before save()")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._model.save(path)
        np.save(path + ".cols.npy", np.array(self._feature_cols))
        log.info("RLAgent(%s, %s): saved to %s", self.pair, self.algo, path)

    def load(self, path: str) -> "RLAgent":
        """
        Load the trained SB3 model and feature column list.

        Args:
            path: File path previously passed to save() (without extension).

        Returns:
            self (fitted)
        """
        AlgoCls = _load_algo(self.algo)

        zip_path = path if path.endswith(".zip") else path + ".zip"
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Model file not found: {zip_path}")

        self._model = AlgoCls.load(zip_path)

        cols_path = path.removesuffix(".zip") + ".cols.npy"
        if os.path.exists(cols_path):
            self._feature_cols = list(np.load(cols_path, allow_pickle=True))
        else:
            log.warning("Feature column list not found at %s — obs→array may fail", cols_path)
            self._feature_cols = []

        self._fitted = True
        log.info("RLAgent(%s, %s): loaded from %s", self.pair, self.algo, path)
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _obs_to_array(self, obs: Union[pd.Series, dict]) -> np.ndarray:
        """Convert observation to 1D float32 array in feature_cols order."""
        obs_map = obs if isinstance(obs, dict) else obs.to_dict()
        values = [float(obs_map.get(col, 0.0)) for col in self._feature_cols]
        return np.array(values, dtype=np.float32)
