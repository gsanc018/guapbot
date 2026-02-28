"""
guapbot/backtest/engine.py

Event-driven backtester for GuapBot.

Processes a normalised feature DataFrame bar-by-bar, collecting signals
from all available models, combining them via the ensemble (or a simple
average fallback), applying position sizing, and accumulating PnL with
realistic transaction costs.

Costs:
    - Taker fee: 0.26 % per trade (Kraken default)
    - Applied on position *changes* only; no cost for holding

Position sizing (Session 6 level, no Kelly/ATR yet):
    long  signal  → position = signal  * max_long   (e.g. 25 %)
    short signal  → position = signal  * max_short  (e.g. 15 %)
    Position fraction is the fraction of current equity deployed.

Reward / PnL semantics:
    Uses next-bar log return (same semantics as BitcoinTradingEnv) to
    avoid look-ahead bias.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from guapbot.models.base import BaseModel
from guapbot.models.ensemble import BaseEnsemble, EnsembleInput, ModelSignal
from guapbot.regime.base import RegimeResult
from guapbot.utils.logging import get_logger

logger = get_logger(__name__)

# Log-return column search order (mirrors trading_env.py)
_LOG_RETURN_COL       = "1h_log_return"
_LOG_RETURN_FALLBACKS = ["log_return", "1h_log_ret_1", "log_ret_1"]

# Map Python class name (lowercased) → canonical model name used by EnsembleLearner
_CLASS_TO_CANONICAL: dict[str, str] = {
    "trendfollowing": "trend_following",
    "meanreversion":  "mean_reversion",
    "gradientboost":  "gradient_boost",
    "lstmmodel":      "lstm",
    "rlagent":        "rl_agent",
}

# Default regimes used when no detectors are supplied
_DEFAULT_REGIMES = [
    RegimeResult(label="trending", confidence=0.5, timeframe="1h"),
    RegimeResult(label="trending", confidence=0.5, timeframe="4h"),
    RegimeResult(label="bullish",  confidence=0.5, timeframe="daily"),
]


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """
    Output of BacktestEngine.run().

    Attributes:
        equity_curve:    Bar-indexed Series of portfolio equity.
        position_series: Bar-indexed Series of position fractions.
        trade_log:       DataFrame logged on every position change.
        metrics:         Pre-computed performance statistics.
        log_returns:     Raw per-bar log returns (used for buy-and-hold comparison).
    """

    equity_curve:    pd.Series
    position_series: pd.Series
    trade_log:       pd.DataFrame
    metrics:         dict[str, float]
    log_returns:     pd.Series = field(default_factory=pd.Series)


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Event-driven bar-by-bar backtester.

    Args:
        models:          Fitted BaseModel instances (any mix of the 5 types).
        ensemble:        Fitted BaseEnsemble. If None, uses simple average of signals.
        initial_capital: Starting equity in USD.
        fee_rate:        Per-trade taker fee fraction (default 0.0026 = 0.26 %).
        max_long:        Maximum long position as fraction of equity (default 0.25).
        max_short:       Maximum short position as fraction of equity (default 0.15).

    Usage:
        engine = BacktestEngine(models=[trend, meanrev], ensemble=learner)
        result = engine.run(normalised_feature_df)
    """

    def __init__(
        self,
        models: list[BaseModel],
        ensemble: Optional[BaseEnsemble] = None,
        initial_capital: float = 10_000.0,
        fee_rate: float = 0.0026,
        max_long: float = 0.25,
        max_short: float = 0.15,
    ) -> None:
        if not models:
            raise ValueError("At least one model is required.")
        self._models          = models
        self._ensemble        = ensemble
        self._initial_capital = initial_capital
        self._fee_rate        = fee_rate
        self._max_long        = max_long
        self._max_short       = max_short

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> BacktestResult:
        """
        Run the backtest over the supplied DataFrame.

        Args:
            df: Normalised feature DataFrame from FeaturePipeline. Must contain
                a log-return column (1h_log_return or fallback). A 'target'
                column is dropped silently if present.

        Returns:
            BacktestResult with equity curve, position series, trade log, metrics.
        """
        if df.empty:
            raise ValueError("df must not be empty.")

        # Drop target column if present — not a feature
        feature_df = df.drop(columns=["target"], errors="ignore")

        # Resolve log-return column
        log_ret_col = self._resolve_log_return_col(feature_df)
        if log_ret_col:
            log_returns = feature_df[log_ret_col].fillna(0.0).to_numpy(dtype=np.float64)
        else:
            logger.warning(
                "No log-return column found — PnL will be zero. "
                "Expected one of: %s", [_LOG_RETURN_COL] + _LOG_RETURN_FALLBACKS,
            )
            log_returns = np.zeros(len(feature_df), dtype=np.float64)

        n_bars = len(feature_df)
        equity          = self._initial_capital
        prev_position   = 0.0

        equity_vals:   list[float] = []
        position_vals: list[float] = []
        trade_rows:    list[dict]  = []

        for t in range(n_bars):
            obs = feature_df.iloc[t].to_dict()

            # --- Collect model signals ---
            raw_signals: list[float] = []
            model_signals_list: list[ModelSignal] = []

            for model in self._models:
                try:
                    sig  = float(model.predict(obs))
                    conf = float(model.confidence(obs))
                except Exception as exc:
                    logger.debug("Model %s failed at bar %d: %s", model, t, exc)
                    sig, conf = 0.0, 0.0

                raw_signals.append(sig)
                model_signals_list.append(
                    ModelSignal(
                        model_name=_CLASS_TO_CANONICAL.get(
                            type(model).__name__.lower(),
                            type(model).__name__.lower(),
                        ),
                        pair="UNKNOWN",
                        signal=sig,
                        confidence=conf,
                    )
                )

            # --- Combine into final signal ---
            if self._ensemble is not None and self._ensemble._fitted:
                try:
                    inputs = EnsembleInput(
                        signals=model_signals_list,
                        regimes=_DEFAULT_REGIMES,
                        pair="UNKNOWN",
                    )
                    raw_signal = self._ensemble.combine(inputs)
                except Exception as exc:
                    logger.debug("Ensemble combine failed at bar %d: %s", t, exc)
                    raw_signal = float(np.mean(raw_signals)) if raw_signals else 0.0
            else:
                raw_signal = float(np.mean(raw_signals)) if raw_signals else 0.0

            # --- Position sizing ---
            if raw_signal >= 0.0:
                position = raw_signal * self._max_long
            else:
                position = raw_signal * self._max_short  # negative * positive = negative

            # --- Next-bar PnL (no look-ahead) ---
            next_idx = min(t + 1, n_bars - 1)
            log_ret  = log_returns[next_idx]
            pnl_pct  = position * log_ret

            # --- Transaction cost on position change ---
            delta    = abs(position - prev_position)
            fee_pct  = delta * self._fee_rate

            equity  *= (1.0 + pnl_pct - fee_pct)

            equity_vals.append(equity)
            position_vals.append(position)

            # Log trade when position direction flips or opens/closes
            if delta > 1e-6:
                trade_rows.append({
                    "time":     feature_df.index[t],
                    "signal":   raw_signal,
                    "position": position,
                    "pnl_pct":  pnl_pct,
                    "fee_pct":  fee_pct,
                })

            prev_position = position

        idx = feature_df.index
        equity_curve    = pd.Series(equity_vals,   index=idx, name="equity")
        position_series = pd.Series(position_vals, index=idx, name="position")
        trade_log       = pd.DataFrame(trade_rows) if trade_rows else pd.DataFrame(
            columns=["time", "signal", "position", "pnl_pct", "fee_pct"]
        )
        raw_log_returns = pd.Series(log_returns, index=idx, name="log_return")

        metrics = _compute_metrics(equity_curve, trade_log)

        logger.info(
            "Backtest complete: %d bars | Sharpe=%.2f | MaxDD=%.1f%% | Total=%.1f%%",
            n_bars,
            metrics.get("sharpe", 0.0),
            metrics.get("max_drawdown", 0.0) * 100,
            metrics.get("total_return", 0.0) * 100,
        )

        return BacktestResult(
            equity_curve=equity_curve,
            position_series=position_series,
            trade_log=trade_log,
            metrics=metrics,
            log_returns=raw_log_returns,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_log_return_col(df: pd.DataFrame) -> Optional[str]:
        for col in [_LOG_RETURN_COL] + _LOG_RETURN_FALLBACKS:
            if col in df.columns:
                return col
        return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

_PERIODS_PER_YEAR = 8_760  # 1-hour bars


def _compute_metrics(
    equity_curve: pd.Series,
    trade_log: pd.DataFrame,
) -> dict[str, float]:
    """Compute annualised performance statistics from the equity curve."""
    returns = equity_curve.pct_change().dropna()

    if len(returns) < 2:
        return {}

    total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)

    n = len(returns)
    ann_factor = _PERIODS_PER_YEAR / n
    try:
        ann_return = float((1.0 + total_return) ** ann_factor - 1.0)
    except (OverflowError, ValueError):
        ann_return = float("inf") if total_return > 0 else float("-inf")

    std = float(returns.std())
    sharpe = float(returns.mean() / std * np.sqrt(_PERIODS_PER_YEAR)) if std > 1e-12 else 0.0

    downside_returns = returns[returns < 0]
    dstd = float(downside_returns.std()) if len(downside_returns) > 1 else 1e-12
    sortino = float(ann_return / (dstd * np.sqrt(_PERIODS_PER_YEAR))) if dstd > 1e-12 else 0.0

    # Max drawdown
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    max_drawdown = float(drawdown.min())

    calmar = float(ann_return / abs(max_drawdown)) if abs(max_drawdown) > 1e-10 else 0.0

    # Trade-level stats
    n_trades = len(trade_log)
    if n_trades > 0 and "pnl_pct" in trade_log.columns:
        pnls = trade_log["pnl_pct"]
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        win_rate     = float(len(wins) / n_trades)
        avg_win      = float(wins.mean())  if len(wins)   > 0 else 0.0
        avg_loss     = float(losses.mean()) if len(losses) > 0 else 0.0
        profit_factor = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
    else:
        win_rate = avg_win = avg_loss = 0.0
        profit_factor = 0.0

    return {
        "total_return":   total_return,
        "ann_return":     ann_return,
        "sharpe":         sharpe,
        "sortino":        sortino,
        "calmar":         calmar,
        "max_drawdown":   max_drawdown,
        "win_rate":       win_rate,
        "avg_win":        avg_win,
        "avg_loss":       avg_loss,
        "profit_factor":  profit_factor,
        "n_trades":       float(n_trades),
    }
