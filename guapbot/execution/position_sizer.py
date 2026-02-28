"""
guapbot.execution.position_sizer
---------------------------------
Position sizing interface.

Converts an ensemble signal into a concrete position fraction, applying
five sizing layers in order. Each layer can ONLY reduce — never increase
— the position from the layer above.

Sizing layers (applied in sequence):
    1. Base signal       — ensemble output in [-1, +1]
    2. Kelly adjustment  — scales by current edge estimate (half-Kelly)
    3. ATR adjustment    — scale inversely with volatility
    4. Regime adjustment — scale by regime confidence
    5. Alert override    — hard cap if social alert is active
    6. Hard caps         — never exceed max_long / max_short regardless

Interface contract:
    sizer.size(signal, regime, alert) → position fraction float
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from guapbot.regime.base import RegimeResult
from guapbot.utils.config import settings
from guapbot.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class AlertState:
    """
    Current state of the social alert system.

    Attributes:
        active:         True when a Tier-1 or Tier-2 alert is live
        severity:       0.0 (low) to 1.0 (critical)
        max_position:   hard cap on position size while alert is active
        source:         which stream fired the alert
        description:    human-readable summary
    """

    active: bool = False
    severity: float = 0.0
    max_position: float = 0.05   # default: max 5% of portfolio during alert
    source: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(f"severity must be in [0, 1], got {self.severity}")
        if not 0.0 <= self.max_position <= 1.0:
            raise ValueError(f"max_position must be in [0, 1], got {self.max_position}")


@dataclass
class SizingContext:
    """
    All inputs needed by the position sizer for one bar.

    Attributes:
        signal:          raw ensemble signal [-1, +1]
        atr:             current ATR as fraction of price (e.g. 0.02 = 2%)
        win_rate:        rolling win rate for Kelly [0, 1]
        avg_win:         average winning trade as fraction
        avg_loss:        average losing trade as fraction (positive number)
        regime:          regime vector as list of RegimeResult
        alert:           current alert state
        portfolio_value: total portfolio value in base currency (for logging)
    """

    signal: float
    atr: float = 0.02
    win_rate: float = 0.5
    avg_win: float = 0.02
    avg_loss: float = 0.01
    regime: list[RegimeResult] = field(default_factory=list)
    alert: AlertState = field(default_factory=AlertState)
    portfolio_value: float = 0.0

    def __post_init__(self) -> None:
        if not -1.0 <= self.signal <= 1.0:
            raise ValueError(f"signal must be in [-1, 1], got {self.signal}")
        if self.atr < 0:
            raise ValueError("atr must be non-negative")


@dataclass
class SizingResult:
    """
    Output of the position sizer with full audit trail.

    Each layer's output is stored so the dashboard can show exactly
    how the final position was derived.
    """

    # Final output
    position: float              # fraction of portfolio to deploy

    # Layer-by-layer audit trail
    base_signal: float = 0.0
    after_kelly: float = 0.0
    after_atr: float = 0.0
    after_regime: float = 0.0
    after_alert: float = 0.0
    after_caps: float = 0.0

    # Direction
    side: str = "flat"           # 'long', 'short', or 'flat'

    def __post_init__(self) -> None:
        if self.position > 0:
            self.side = "long"
        elif self.position < 0:
            self.side = "short"
        else:
            self.side = "flat"


class PositionSizer(ABC):
    """
    Abstract base for the GuapBot position sizer.

    The concrete implementation (also in this file) applies all sizing
    layers in the specified order. The interface ensures any replacement
    sizer is a drop-in for the execution layer.

    max_long and max_short come from settings by default but can be
    overridden at construction time for strategy-specific limits.
    """

    def __init__(
        self,
        max_long: float | None = None,
        max_short: float | None = None,
    ) -> None:
        self.max_long = max_long if max_long is not None else settings.max_long_fraction
        self.max_short = max_short if max_short is not None else settings.max_short_fraction

    # ------------------------------------------------------------------
    # Abstract interface — the one method that must be implemented
    # ------------------------------------------------------------------

    @abstractmethod
    def size(
        self,
        signal: float,
        regime: list[RegimeResult],
        alert: AlertState,
        *,
        atr: float = 0.02,
        win_rate: float = 0.5,
        avg_win: float = 0.02,
        avg_loss: float = 0.01,
    ) -> SizingResult:
        """
        Convert an ensemble signal to a position fraction.

        Args:
            signal:    ensemble output in [-1.0, +1.0]
            regime:    list of RegimeResult for all active timeframes
            alert:     current AlertState from the fast clock
            atr:       current ATR as fraction of price
            win_rate:  rolling win rate for Kelly calculation
            avg_win:   average winning trade size
            avg_loss:  average losing trade size

        Returns:
            SizingResult with position fraction and full audit trail.
        """
        ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _apply_hard_caps(self, position: float) -> float:
        """
        Enforce portfolio-level position caps regardless of anything else.

        Never exceeds max_long on the long side or max_short on the short side.
        This is the last and absolute layer — it cannot be bypassed.
        """
        if position > self.max_long:
            log.debug(f"Hard cap: long {position:.4f} → {self.max_long:.4f}")
            return self.max_long
        if position < -self.max_short:
            log.debug(f"Hard cap: short {position:.4f} → {-self.max_short:.4f}")
            return -self.max_short
        return position

    def _half_kelly(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Compute the half-Kelly fraction.

        Standard binary Kelly: f* = p - q/b
            p = win_rate
            q = 1 - win_rate
            b = avg_win / avg_loss  (win/loss odds ratio)

        Expanding: f* = win_rate - (1 - win_rate) * avg_loss / avg_win

        Example with typical inputs (win_rate=0.5, avg_win=0.02, avg_loss=0.01):
            b = 2.0  →  f* = 0.5 - 0.25 = 0.25  →  half_kelly = 0.125  ✓

        Returns a value in [0, 1]; returns 0 if Kelly is negative (no edge).
        """
        if avg_loss <= 0 or avg_win <= 0:
            return 0.5   # default neutral if no history
        kelly = win_rate - (1.0 - win_rate) * avg_loss / avg_win
        half_kelly = kelly / 2.0
        return max(0.0, min(1.0, half_kelly))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(max_long={self.max_long:.0%}, max_short={self.max_short:.0%})"
        )


# ---------------------------------------------------------------------------
# Concrete implementation
# ---------------------------------------------------------------------------

# ATR target: "normal" volatility level. Positions scale inversely around this.
_ATR_TARGET = 0.02   # 2% ATR is considered baseline


class DefaultPositionSizer(PositionSizer):
    """
    Concrete position sizer applying all 5 sizing layers in order.

    Each layer can only reduce (never increase) the position magnitude:
        1. Base signal       → signal × max_long/max_short
        2. Kelly adjustment  → × half_kelly(win_rate, avg_win, avg_loss)
        3. ATR adjustment    → × clip(atr_target / atr, 0.5, 1.0)
        4. Regime adjustment → × mean confidence across all timeframes
        5. Alert override    → cap at alert.max_position if active
        6. Hard caps         → enforce max_long / max_short absolutely

    Args:
        max_long:  maximum long position fraction (default: settings.max_long_fraction)
        max_short: maximum short position fraction (default: settings.max_short_fraction)
        atr_target: ATR considered "normal" (default 0.02 = 2%)
    """

    def __init__(
        self,
        max_long: float | None = None,
        max_short: float | None = None,
        atr_target: float = _ATR_TARGET,
    ) -> None:
        super().__init__(max_long=max_long, max_short=max_short)
        self._atr_target = atr_target

    def size(
        self,
        signal: float,
        regime: list[RegimeResult],
        alert: AlertState,
        *,
        atr: float = 0.02,
        win_rate: float = 0.5,
        avg_win: float = 0.02,
        avg_loss: float = 0.01,
    ) -> SizingResult:
        """Apply all sizing layers and return a full audit-trail SizingResult."""

        # --- Layer 1: Base signal ---
        if signal >= 0.0:
            base = signal * self.max_long
        else:
            base = signal * self.max_short   # negative × positive = negative

        # --- Layer 2: Kelly adjustment ---
        kelly_scale = self._half_kelly(win_rate, avg_win, avg_loss)
        after_kelly = base * kelly_scale

        # --- Layer 3: ATR adjustment ---
        if atr > 1e-8:
            atr_scale = float(min(max(self._atr_target / atr, 0.5), 1.0))
        else:
            atr_scale = 1.0
        after_atr = after_kelly * atr_scale

        # --- Layer 4: Regime adjustment ---
        if regime:
            regime_conf = float(sum(r.confidence for r in regime) / len(regime))
        else:
            regime_conf = 1.0
        after_regime = after_atr * regime_conf

        # --- Layer 5: Alert override ---
        if alert.active:
            cap = alert.max_position
            after_alert = max(-cap, min(cap, after_regime))
            log.debug(
                "Alert active — position capped: cap=%.4f before=%.4f after=%.4f",
                cap, after_regime, after_alert,
            )
        else:
            after_alert = after_regime

        # --- Layer 6: Hard caps ---
        final = self._apply_hard_caps(after_alert)

        result = SizingResult(
            position=final,
            base_signal=base,
            after_kelly=after_kelly,
            after_atr=after_atr,
            after_regime=after_regime,
            after_alert=after_alert,
            after_caps=final,
        )
        log.debug(
            "PositionSizer: signal=%.4f kelly=%.3f atr_scale=%.3f regime_conf=%.3f alert=%s final=%.4f",
            round(signal, 4), round(kelly_scale, 3), round(atr_scale, 3),
            round(regime_conf, 3), alert.active, round(final, 4),
        )
        return result
