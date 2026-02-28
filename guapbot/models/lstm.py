"""
guapbot/models/lstm.py

LSTMModel — PyTorch LSTM sequence model for directional prediction.

Architecture:
  Input: normalized feature vector (n_features,) per bar
  Sequence length: 60 bars (the last 60 1h bars)
  Network: LSTM(hidden=128, layers=2, dropout=0.2) → Linear(128, 64)
           → ReLU → Linear(64, 1) → Tanh
  Output: signal in [-1, +1] (tanh forces bounded output)

Training:
  - Build overlapping 60-bar windows from the normalized feature DataFrame.
  - Target = sign of next-bar log return (from 'target' column).
  - Adam optimizer, lr=1e-3, 20 epochs, batch_size=64.
  - After training, seeds internal deque with last 60 rows of training data
    so the model is immediately ready for inference without warm-up calls.

Inference:
  - predict(obs) appends obs to internal deque(maxlen=60).
    Returns 0.0 (flat) until buffer reaches 60 observations.
    On each call after that: runs a forward pass through the LSTM.
  - confidence(obs) = |predict(obs)| — tanh magnitude as proxy.

Persistence:
  - torch.save: state_dict + feature_cols list + serialized buffer
  - torch.load: restores all of the above
"""
from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from guapbot.models.base import BaseModel
from guapbot.utils.logging import get_logger

log = get_logger(__name__)

# Architecture constants
SEQ_LEN = 60
HIDDEN_SIZE = 128
N_LAYERS = 2
DROPOUT = 0.2
FC_HIDDEN = 64

# Training constants
N_EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# Columns to exclude from the feature set
_DROP_COLS = {"target", "timestamp"}


def _build_net(n_features: int):
    """Build the LSTM network (import torch here to allow graceful failure)."""
    import torch
    import torch.nn as nn

    class _LSTMNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=HIDDEN_SIZE,
                num_layers=N_LAYERS,
                dropout=DROPOUT if N_LAYERS > 1 else 0.0,
                batch_first=True,
            )
            self.fc1 = nn.Linear(HIDDEN_SIZE, FC_HIDDEN)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(FC_HIDDEN, 1)
            self.tanh = nn.Tanh()

        def forward(self, x):
            # x: (batch, seq_len, n_features)
            _, (h_n, _) = self.lstm(x)
            out = h_n[-1]             # last layer's hidden state: (batch, hidden)
            out = self.relu(self.fc1(out))
            out = self.tanh(self.fc2(out))
            return out.squeeze(-1)    # (batch,)

    return _LSTMNet()


class LSTMModel(BaseModel):
    """
    2-layer LSTM model for sequential directional prediction.

    Maintains an internal rolling buffer of the last 60 observations.
    Returns 0.0 (flat) until buffer has 60 entries — no warm-up required
    after fit() because the buffer is seeded with training data tail.

    One instance per asset.
    """

    def __init__(self, pair: str, strategy: str) -> None:
        super().__init__(pair, strategy)
        self._net = None
        self._feature_cols: list[str] = []
        self._buffer: deque = deque(maxlen=SEQ_LEN)
        self._last_signal: float = 0.0

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "LSTMModel":
        """
        Train the LSTM on overlapping 60-bar sequences.

        Args:
            df: Normalized feature DataFrame with 'target' column.
                target = +1 (up) or -1 (down) for the next bar.

        Returns:
            self
        """
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required. Install: pip install torch"
            ) from exc

        if "target" not in df.columns:
            raise ValueError(
                "LSTMModel.fit() requires a 'target' column in df. "
                "Expected values: +1 (up) or -1 (down)."
            )

        self._feature_cols = [
            c for c in df.columns
            if c not in _DROP_COLS and df[c].dtype in (np.float64, np.float32, float)
        ]
        n_features = len(self._feature_cols)
        if n_features == 0:
            raise ValueError("No numeric feature columns found in df")

        # Build overlapping sequences
        X_arr = df[self._feature_cols].to_numpy(dtype=np.float32)
        y_arr = df["target"].to_numpy(dtype=np.float32)

        n = len(X_arr)
        if n <= SEQ_LEN:
            raise ValueError(
                f"Need more than {SEQ_LEN} bars to build sequences, got {n}"
            )

        # Shape: (n_sequences, SEQ_LEN, n_features)
        sequences = np.stack([X_arr[i:i + SEQ_LEN] for i in range(n - SEQ_LEN)])
        targets = y_arr[SEQ_LEN:]  # next-bar direction for each sequence

        X_t = torch.FloatTensor(sequences)
        y_t = torch.FloatTensor(targets)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        # Build network
        self._net = _build_net(n_features)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()  # regress directly to +1 / -1 target

        log.info(
            "LSTMModel(%s): training on %d sequences, %d features, %d epochs",
            self.pair, len(sequences), n_features, N_EPOCHS,
        )

        self._net.train()
        for epoch in range(N_EPOCHS):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = self._net(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                log.debug(
                    "LSTMModel(%s): epoch %d/%d loss=%.4f",
                    self.pair, epoch + 1, N_EPOCHS, epoch_loss / len(loader),
                )

        self._net.eval()

        # Seed buffer with last SEQ_LEN rows of training data
        self._buffer = deque(maxlen=SEQ_LEN)
        for i in range(max(0, n - SEQ_LEN), n):
            self._buffer.append(X_arr[i])

        self._fitted = True
        log.info("LSTMModel(%s): training complete", self.pair)
        return self

    def predict(self, obs: Union[pd.Series, dict]) -> float:
        """
        Return signal in [-1.0, +1.0] using rolling 60-bar window.

        Appends obs to internal buffer. Returns 0.0 (flat) if buffer
        has fewer than SEQ_LEN entries. Otherwise runs a forward pass.

        Args:
            obs: Single-bar observation (pd.Series or dict).

        Returns:
            float in [-1.0, +1.0]
        """
        if not self._fitted:
            raise RuntimeError(
                f"LSTMModel({self.pair}) must be fit() before predict()"
            )

        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch required for LSTMModel") from exc

        obs_map = obs if isinstance(obs, dict) else obs.to_dict()
        feature_vec = np.array(
            [float(obs_map.get(col, 0.0)) for col in self._feature_cols],
            dtype=np.float32,
        )
        self._buffer.append(feature_vec)

        if len(self._buffer) < SEQ_LEN:
            return 0.0  # still warming up

        seq = np.stack(list(self._buffer))  # (SEQ_LEN, n_features)
        X = torch.FloatTensor(seq).unsqueeze(0)  # (1, SEQ_LEN, n_features)

        with torch.no_grad():
            signal = float(self._net(X).item())

        self._last_signal = signal
        return self._validate_signal(signal)

    def confidence(self, obs: Union[pd.Series, dict]) -> float:
        """
        Return confidence in [0.0, 1.0].

        Uses the tanh output magnitude of the last predict() call.
        Calls predict() internally if needed to get fresh signal.
        """
        if not self._fitted:
            raise RuntimeError(
                f"LSTMModel({self.pair}) must be fit() before confidence()"
            )

        if len(self._buffer) < SEQ_LEN:
            return 0.0

        # Use magnitude of last signal as proxy confidence
        conf = abs(self._last_signal)
        return self._validate_confidence(conf)

    def save(self, path: str) -> None:
        """
        Save model state, feature columns, and buffer contents.

        Args:
            path: File path, e.g. 'models/money_printer/lstm/XBTUSD/model.pt'
        """
        if not self._fitted:
            raise RuntimeError("Model must be fit() before save()")

        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch required for LSTMModel.save()") from exc

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self._net.state_dict(),
            "feature_cols": self._feature_cols,
            "buffer": list(self._buffer),
            "pair": self.pair,
            "strategy": self.strategy,
        }
        torch.save(payload, path)
        log.info("LSTMModel(%s): saved to %s", self.pair, path)

    def load(self, path: str) -> "LSTMModel":
        """
        Load model state, feature columns, and buffer from a .pt file.

        Args:
            path: File path previously passed to save().

        Returns:
            self (fitted)
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch required for LSTMModel.load()") from exc

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        payload = torch.load(path, map_location="cpu", weights_only=False)
        self._feature_cols = payload["feature_cols"]
        n_features = len(self._feature_cols)

        self._net = _build_net(n_features)
        self._net.load_state_dict(payload["state_dict"])
        self._net.eval()

        self._buffer = deque(maxlen=SEQ_LEN)
        for vec in payload["buffer"]:
            self._buffer.append(np.array(vec, dtype=np.float32))

        self._fitted = True
        log.info("LSTMModel(%s): loaded from %s", self.pair, path)
        return self
