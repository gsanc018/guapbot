"""
guapbot/execution/paper_trader.py

Simulated (paper) trading engine for GuapBot.

PaperTrader runs the full execution pipeline bar-by-bar over historical or
live-replay feature data:

    Feature obs
        → models → ensemble signal
        → DefaultPositionSizer (Kelly + ATR + regime + alert + caps)
        → OrderExecutor (immediate fill at mid-price + slippage)
        → PnL accounting (log-return × position, fee on delta)
        → MarketState write (position, signal, equity, alert)

Kill switches (both halt the loop and return immediately):
    daily_dd_limit:  if today's drawdown exceeds this, halt (default -5%)
    total_dd_limit:  if peak drawdown exceeds this, halt (default -15%)

The paper trader does NOT know about real money. It uses the feature
DataFrame's log-return column for PnL, and the ticker mid-price for
sizing (from obs dict if available, else falls back to 1.0).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from guapbot.execution.market_state import MarketState
from guapbot.execution.order_executor import OrderExecutor
from guapbot.execution.order_manager import Order, OrderManager
from guapbot.execution.position_sizer import AlertState, DefaultPositionSizer
from guapbot.models.ensemble import BaseEnsemble, EnsembleInput, ModelSignal
from guapbot.regime.base import RegimeResult
from guapbot.utils.logging import get_logger

log = get_logger(__name__)

# Column search order for log-return (mirrors engine.py)
_LOG_RETURN_COL       = "1h_log_return"
_LOG_RETURN_FALLBACKS = ["log_return", "1h_log_ret_1", "log_ret_1"]

# Column search order for price
_PRICE_COLS = ["close", "price", "mid"]

# Default regimes when none are supplied
_DEFAULT_REGIMES = [
    RegimeResult(label="trending", confidence=0.5, timeframe="1h"),
    RegimeResult(label="trending", confidence=0.5, timeframe="4h"),
    RegimeResult(label="bullish",  confidence=0.5, timeframe="daily"),
]

# Map Python class name → canonical model name (same as engine.py)
_CLASS_TO_CANONICAL: dict[str, str] = {
    "trendfollowing": "trend_following",
    "meanreversion":  "mean_reversion",
    "gradientboost":  "gradient_boost",
    "lstmmodel":      "lstm",
    "rlagent":        "rl_agent",
}


@dataclass
class BarStats:
    """Per-bar output from PaperTrader.step()."""

    t: int
    signal: float
    position: float
    log_return: float
    pnl_pct: float
    fee_pct: float
    equity: float
    order: Optional[Order] = None
    kill_switch: str = ""      # non-empty string means loop should stop


@dataclass
class PaperTraderResult:
    """Final summary returned by PaperTrader.run()."""

    bars_processed: int
    final_equity: float
    total_return: float
    n_trades: int
    kill_switch_reason: str = ""
    bar_stats: list[BarStats] = field(default_factory=list)


class PaperTrader:
    """
    Simulates the full GuapBot execution pipeline bar-by-bar.

    Args:
        models:           fitted BaseModel instances
        ensemble:         fitted BaseEnsemble (None = simple mean fallback)
        initial_capital:  starting equity in USD
        fee_rate:         taker fee per trade (default 0.0026 = 0.26%)
        max_long:         max long fraction (default 0.25)
        max_short:        max short fraction (default 0.15)
        slippage:         simulated slippage fraction (default 0.0001)
        daily_dd_limit:   daily drawdown kill switch (default -0.05 = -5%)
        total_dd_limit:   total drawdown kill switch (default -0.15 = -15%)
        state:            optional MarketState; if None, state writes are skipped
        pair:             trading pair label for state keys (default 'UNKNOWN')
    """

    def __init__(
        self,
        models: list,
        ensemble: Optional[BaseEnsemble] = None,
        initial_capital: float = 10_000.0,
        fee_rate: float = 0.0026,
        max_long: float = 0.25,
        max_short: float = 0.15,
        slippage: float = 0.0001,
        daily_dd_limit: float = -0.05,
        total_dd_limit: float = -0.15,
        state: Optional[MarketState] = None,
        pair: str = "UNKNOWN",
    ) -> None:
        if not models:
            raise ValueError("At least one model is required.")

        self._models          = models
        self._ensemble        = ensemble
        self._initial_capital = initial_capital
        self._fee_rate        = fee_rate
        self._daily_dd_limit  = daily_dd_limit
        self._total_dd_limit  = total_dd_limit
        self._state           = state
        self._pair            = pair

        self._sizer    = DefaultPositionSizer(max_long=max_long, max_short=max_short)
        self._orders   = OrderManager()
        self._executor = OrderExecutor(self._orders, slippage=slippage)

        # Runtime state
        self._equity          = initial_capital
        self._peak_equity     = initial_capital
        self._day_start_eq    = initial_capital
        self._current_bar     = 0
        self._position        = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        obs: dict,
        log_return: float,
        regimes: Optional[list[RegimeResult]] = None,
    ) -> BarStats:
        """
        Process one bar through the full execution pipeline.

        Args:
            obs:        feature dict for this bar (from DataFrame.iloc[t].to_dict())
            log_return: next-bar log return (for PnL; avoids look-ahead)
            regimes:    optional regime vector; uses defaults if None

        Returns:
            BarStats for this bar.
        """
        t       = self._current_bar
        active_regimes = regimes or _DEFAULT_REGIMES

        # --- 1. Collect model signals ---
        raw_signals: list[float] = []
        model_signals_list: list[ModelSignal] = []

        for model in self._models:
            try:
                sig  = float(model.predict(obs))
                conf = float(model.confidence(obs))
            except Exception as exc:
                log.debug("Model %s failed at bar %d: %s", model, t, exc)
                sig, conf = 0.0, 0.0

            raw_signals.append(sig)
            model_signals_list.append(
                ModelSignal(
                    model_name=_CLASS_TO_CANONICAL.get(
                        type(model).__name__.lower(),
                        type(model).__name__.lower(),
                    ),
                    pair=self._pair,
                    signal=sig,
                    confidence=conf,
                )
            )

        # --- 2. Ensemble combine (or mean fallback) ---
        if self._ensemble is not None and self._ensemble._fitted:
            try:
                inputs     = EnsembleInput(
                    signals=model_signals_list,
                    regimes=active_regimes,
                    pair=self._pair,
                )
                raw_signal = self._ensemble.combine(inputs)
            except Exception as exc:
                log.debug("Ensemble combine failed at bar %d: %s", t, exc)
                raw_signal = float(np.mean(raw_signals)) if raw_signals else 0.0
        else:
            raw_signal = float(np.mean(raw_signals)) if raw_signals else 0.0

        # --- 3. Read alert state from MarketState ---
        alert = AlertState()
        if self._state is not None:
            alert_data = self._state.get_alert()
            if alert_data:
                alert = AlertState(
                    active=True,
                    severity=float(alert_data.get("severity", 0.5)),
                    source=str(alert_data.get("source", "")),
                    description=str(alert_data.get("description", "")),
                )

        # --- 4. Position sizing ---
        atr = float(obs.get("atr", obs.get("1h_atr_norm", 0.02)))
        sizing = self._sizer.size(
            signal=raw_signal,
            regime=active_regimes,
            alert=alert,
            atr=atr,
        )
        target_position = sizing.position

        # --- 5. Order execution ---
        mid_price = self._resolve_price(obs)
        order = self._executor.execute(
            pair=self._pair,
            target_position=target_position,
            current_position=self._position,
            mid_price=mid_price,
            equity=self._equity,
        )

        # --- 6. PnL accounting ---
        # If no order was placed (delta below dust threshold), the position
        # did not actually change — PnL and fees must reflect the position
        # that was actually held, not the desired target.
        if order is None:
            actual_position = self._position
            fee_pct         = 0.0
        else:
            actual_position  = target_position
            delta            = abs(target_position - self._position)
            fee_pct          = delta * self._fee_rate
            self._position   = target_position

        pnl_pct = actual_position * log_return
        self._equity *= (1.0 + pnl_pct - fee_pct)

        # Update peak for drawdown tracking
        if self._equity > self._peak_equity:
            self._peak_equity = self._equity

        # --- 7. Write to MarketState ---
        if self._state is not None:
            self._state.update(f"signal.{self._pair}", round(raw_signal, 4))
            self._state.update(f"position.{self._pair}", round(actual_position, 4))
            self._state.update(f"equity.{self._pair}", round(self._equity, 2))

        # --- 8. Kill switch checks ---
        kill_switch = self._check_kill_switches(t)

        stats = BarStats(
            t=t,
            signal=raw_signal,
            position=actual_position,   # what was actually held, not the desired target
            log_return=log_return,
            pnl_pct=pnl_pct,
            fee_pct=fee_pct,
            equity=self._equity,
            order=order,
            kill_switch=kill_switch,
        )

        self._current_bar += 1

        # Reset day-start equity every ~24 bars (approximate daily reset)
        if t % 24 == 23:
            self._day_start_eq = self._equity

        return stats

    def summary(self) -> PaperTraderResult:
        """Return a summary of trading so far."""
        total_return = self._equity / self._initial_capital - 1.0
        return PaperTraderResult(
            bars_processed=self._current_bar,
            final_equity=self._equity,
            total_return=total_return,
            n_trades=self._orders.n_trades(),
        )

    @property
    def equity(self) -> float:
        return self._equity

    @property
    def position(self) -> float:
        return self._position

    @property
    def order_manager(self) -> OrderManager:
        return self._orders

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_price(self, obs: dict) -> float:
        """Try to find a price in the obs dict; default to 1.0."""
        for col in _PRICE_COLS:
            if col in obs and obs[col] > 0:
                return float(obs[col])
        return 1.0

    def _check_kill_switches(self, t: int) -> str:
        """
        Check daily and total drawdown kill switches.

        Returns an empty string if all clear, or a descriptive reason
        if a kill switch has fired.
        """
        # Total drawdown
        total_dd = (self._equity - self._peak_equity) / self._peak_equity
        if total_dd <= self._total_dd_limit:
            reason = (
                f"KILL SWITCH — total drawdown {total_dd:.1%} "
                f"exceeded limit {self._total_dd_limit:.1%}"
            )
            log.warning("%s (bar=%d equity=%.2f)", reason, t, round(self._equity, 2))
            return reason

        # Daily drawdown
        if self._day_start_eq > 0:
            daily_dd = (self._equity - self._day_start_eq) / self._day_start_eq
            if daily_dd <= self._daily_dd_limit:
                reason = (
                    f"KILL SWITCH — daily drawdown {daily_dd:.1%} "
                    f"exceeded limit {self._daily_dd_limit:.1%}"
                )
                log.warning("%s (bar=%d equity=%.2f)", reason, t, round(self._equity, 2))
                return reason

        return ""

    def __repr__(self) -> str:
        return (
            f"PaperTrader("
            f"equity={self._equity:.2f}, "
            f"pos={self._position:.4f}, "
            f"bars={self._current_bar}, "
            f"trades={self._orders.n_trades()})"
        )
