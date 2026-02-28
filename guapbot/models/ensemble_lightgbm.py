"""
guapbot/models/ensemble_lightgbm.py

Concrete LightGBM meta-learner implementation of BaseEnsemble.

Training:
    Walk-forward stacking (TimeSeriesSplit, 4 folds). Each fold trains a
    temporary LGBMClassifier and generates out-of-fold (OOF) predictions.
    A final LGBMClassifier is trained on the full signal_history and kept
    as the live model.

Online updates:
    Outcomes are buffered. Every 24 bars (≈ 1 trading day), the buffer is
    flushed: a small LightGBM model is trained on the new data with the
    existing booster as init_model, adding 5 trees without a full retrain.

Feature vector (21 cols, fixed order):
    5 model signals  + 5 model confidences + 5 rolling Sharpes
    + 6 regime features (label_encoded + confidence per timeframe)
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit
    _LGB_AVAILABLE = True
except Exception:
    _LGB_AVAILABLE = False

from guapbot.models.ensemble import (
    BaseEnsemble,
    EnsembleInput,
    ModelSignal,
    TradeOutcome,
)
from guapbot.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_NAMES: list[str] = [
    "trend_following",
    "mean_reversion",
    "gradient_boost",
    "lstm",
    "rl_agent",
]

_FEATURE_COLS: list[str] = (
    [f"{m}_signal"         for m in _MODEL_NAMES]
    + [f"{m}_confidence"   for m in _MODEL_NAMES]
    + [f"{m}_rolling_sharpe" for m in _MODEL_NAMES]
    + [
        "regime_1h_label",    "regime_1h_confidence",
        "regime_4h_label",    "regime_4h_confidence",
        "regime_daily_label", "regime_daily_confidence",
    ]
)  # 21 total

# Encode regime label strings → int
_REGIME_LABEL_MAP: dict[str, int] = {
    "trending": 0, "ranging": 1, "volatile": 2,
    "bullish":  0, "bearish": 1, "neutral":  2,
}

_LGBM_PARAMS: dict = {
    "objective":     "binary",
    "num_leaves":    31,
    "learning_rate": 0.05,
    "n_estimators":  100,
    "verbosity":     -1,
    "random_state":  42,
}

_N_FOLDS           = 4   # walk-forward folds
_ONLINE_BUFFER_SZ  = 24  # flush every 24 outcomes (~1 day of 1h bars)
_ONLINE_ROUNDS     = 5   # trees added per online update


# ---------------------------------------------------------------------------
# EnsembleLearner
# ---------------------------------------------------------------------------

class EnsembleLearner(BaseEnsemble):
    """
    LightGBM meta-learner that combines 5 sub-model signals into a single
    final trading signal, conditioned on the current regime vector.

    Usage:
        learner = EnsembleLearner("XBTUSD")
        learner.fit(signal_history_df)          # walk-forward stacking
        signal = learner.combine(ensemble_input) # float in [-1, +1]
        learner.update(trade_outcome)            # online learning step
    """

    def __init__(self, pair: str) -> None:
        super().__init__(pair)
        self._model: lgb.LGBMClassifier | None = None
        self._outcome_buffer: list[TradeOutcome] = []

    # ------------------------------------------------------------------
    # BaseEnsemble interface
    # ------------------------------------------------------------------

    def fit(self, signal_history: pd.DataFrame) -> "EnsembleLearner":
        """
        Train via walk-forward stacking.

        Args:
            signal_history: DataFrame with columns matching _FEATURE_COLS
                            plus a 'target' column (binary: 1=up, 0=down).

        Returns:
            self
        """
        if not _LGB_AVAILABLE:
            raise ImportError(
                "lightgbm is not available (missing libomp on macOS?). "
                "Install with: brew install libomp"
            )
        X, y = self._build_Xy(signal_history)

        # --- Walk-forward OOF evaluation ---
        tscv = TimeSeriesSplit(n_splits=_N_FOLDS)
        oof_preds = np.zeros(len(X))

        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
            fold_model = lgb.LGBMClassifier(**_LGBM_PARAMS)
            fold_model.fit(X[tr_idx], y[tr_idx])
            oof_preds[val_idx] = fold_model.predict_proba(X[val_idx])[:, 1]
            logger.debug("Ensemble fold %d/%d complete", fold + 1, _N_FOLDS)

        # OOF directional accuracy
        oof_dir = np.sign(oof_preds - 0.5)
        oof_tgt = np.where(y == 1, 1.0, -1.0)
        n_non_flat = (oof_dir != 0).sum()
        if n_non_flat > 0:
            oof_acc = (oof_dir == oof_tgt).sum() / n_non_flat
            logger.info(
                "EnsembleLearner OOF directional accuracy: %.1f%% (%d/%d non-flat)",
                oof_acc * 100, (oof_dir == oof_tgt).sum(), n_non_flat,
            )

        # --- Final model trained on all data ---
        self._model = lgb.LGBMClassifier(**_LGBM_PARAMS)
        self._model.fit(X, y)
        self._fitted = True
        logger.info("EnsembleLearner fitted on %d bars for %s", len(X), self.pair)
        return self

    def combine(self, inputs: EnsembleInput) -> float:
        """
        Combine sub-model signals into a single signal.

        Returns:
            float in [-1.0, +1.0]
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("EnsembleLearner is not fitted. Call fit() first.")

        x = self._input_to_row(inputs)
        prob = float(self._model.predict_proba([x])[0, 1])
        return float(np.clip(prob * 2.0 - 1.0, -1.0, 1.0))

    def update(self, outcome: TradeOutcome) -> None:
        """Buffer outcome; flush to online update every _ONLINE_BUFFER_SZ bars."""
        self._outcome_buffer.append(outcome)
        if len(self._outcome_buffer) >= _ONLINE_BUFFER_SZ:
            self._flush_update()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted EnsembleLearner.")
        joblib.dump(
            {
                "model":   self._model,
                "pair":    self.pair,
                "fitted":  self._fitted,
            },
            path,
        )
        logger.info("EnsembleLearner saved to %s", path)

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self._model   = data["model"]
        self.pair     = data["pair"]
        self._fitted  = data["fitted"]
        logger.info("EnsembleLearner loaded from %s", path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_Xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Extract feature matrix X and binary target y from signal_history."""
        missing = [c for c in _FEATURE_COLS if c not in df.columns]
        if missing:
            logger.warning("Missing feature cols (will fill 0): %s", missing)

        X = df.reindex(columns=_FEATURE_COLS, fill_value=0.0).to_numpy(dtype=np.float32)

        if "target" not in df.columns:
            raise ValueError("signal_history must contain a 'target' column.")
        y = (df["target"] > 0).astype(int).to_numpy()
        return X, y

    @staticmethod
    def _input_to_row(inputs: EnsembleInput) -> np.ndarray:
        """Convert EnsembleInput to a 1-D feature vector matching _FEATURE_COLS."""
        sig_map: dict[str, ModelSignal] = {s.model_name: s for s in inputs.signals}
        regime_map = inputs.regime_dict  # {timeframe: RegimeResult}

        row: list[float] = []

        # signals
        for m in _MODEL_NAMES:
            row.append(sig_map[m].signal if m in sig_map else 0.0)
        # confidences
        for m in _MODEL_NAMES:
            row.append(sig_map[m].confidence if m in sig_map else 0.0)
        # rolling sharpes
        for m in _MODEL_NAMES:
            row.append(sig_map[m].rolling_sharpe if m in sig_map else 0.0)
        # regime features
        for tf in ("1h", "4h", "daily"):
            if tf in regime_map:
                r = regime_map[tf]
                row.append(float(_REGIME_LABEL_MAP.get(r.label, 0)))
                row.append(r.confidence)
            else:
                row.append(0.0)
                row.append(0.5)

        return np.array(row, dtype=np.float32)

    def _flush_update(self) -> None:
        """Online update: add trees to existing model using buffered outcomes."""
        if self._model is None or not self._outcome_buffer:
            self._outcome_buffer.clear()
            return

        rows: list[list[float]] = []
        targets: list[int] = []

        for outcome in self._outcome_buffer:
            # Reconstruct a partial feature row from what TradeOutcome provides
            sig_map = {name: ModelSignal(
                model_name=name,
                pair=outcome.pair,
                signal=float(outcome.model_signals.get(name, 0.0)),
                confidence=float(outcome.model_signals.get(f"{name}_confidence", 0.0)),
            ) for name in _MODEL_NAMES}

            regime_results = []
            from guapbot.regime.base import RegimeResult
            for tf in ("1h", "4h", "daily"):
                label_str = outcome.regime_labels.get(tf, "trending")
                regime_results.append(RegimeResult(label=label_str, confidence=0.5, timeframe=tf))

            ei = EnsembleInput(
                signals=list(sig_map.values()),
                regimes=regime_results,
                pair=outcome.pair,
            )
            rows.append(self._input_to_row(ei).tolist())
            targets.append(1 if outcome.realised_return > 0 else 0)

        X_buf = np.array(rows, dtype=np.float32)
        y_buf = np.array(targets, dtype=int)

        try:
            new_data = lgb.Dataset(X_buf, label=y_buf, free_raw_data=False)
            updated_booster = lgb.train(
                params={k: v for k, v in _LGBM_PARAMS.items() if k != "n_estimators"},
                train_set=new_data,
                num_boost_round=_ONLINE_ROUNDS,
                init_model=self._model.booster_,
                callbacks=[lgb.log_evaluation(period=-1)],
            )
            # Wrap back in sklearn API: replace internal booster
            self._model.booster_ = updated_booster
            logger.debug(
                "EnsembleLearner online update: +%d trees from %d outcomes",
                _ONLINE_ROUNDS, len(self._outcome_buffer),
            )
        except Exception as exc:
            logger.warning("Online update failed: %s — skipping", exc)

        self._outcome_buffer.clear()
