"""
guapbot/execution/order_executor.py

Converts a target position fraction into a concrete order.

The executor sits between the position sizer and the order manager:
    PositionSizer → SizingResult → OrderExecutor → Order → OrderManager

Execution modes (Session 7 implements Immediate only):
    Immediate — single market order at current mid-price + slippage
    TWAP      — time-weighted split (Session 14)
    VWAP      — volume-weighted split (Session 14)

Delta threshold:
    If |target - current| < MIN_DELTA the position change is considered dust
    and no order is submitted. This avoids excessive micro-trading.
"""
from __future__ import annotations

from typing import Optional

from guapbot.execution.order_manager import Order, OrderManager
from guapbot.utils.logging import get_logger

log = get_logger(__name__)

# Minimum position change (as fraction of equity) worth trading
_MIN_DELTA = 0.001   # 0.1% — below this we skip the trade

# Default slippage for paper mode: half a spread
_DEFAULT_SLIPPAGE = 0.0001   # 0.01%


class OrderExecutor:
    """
    Converts a target position fraction into an Order and records it.

    In paper mode (default), orders are filled immediately at mid-price
    plus a small slippage. In live mode the same interface is used, but
    the live trader layer replaces simulated fills with real Kraken API
    calls (Session 14).

    Args:
        order_manager: shared OrderManager instance
        slippage:      additional cost per trade as a fraction of price
                       (simulates half-spread cost in paper mode)
        min_delta:     minimum position change worth submitting an order
    """

    def __init__(
        self,
        order_manager: OrderManager,
        slippage: float = _DEFAULT_SLIPPAGE,
        min_delta: float = _MIN_DELTA,
    ) -> None:
        self._order_manager = order_manager
        self._slippage      = slippage
        self._min_delta     = min_delta

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        pair: str,
        target_position: float,
        current_position: float,
        mid_price: float,
        equity: float,
    ) -> Optional[Order]:
        """
        Submit an order to move from current_position to target_position.

        Args:
            pair:             trading pair (e.g. 'XBTUSD')
            target_position:  desired position as fraction of equity [-1, +1]
            current_position: current position as fraction of equity [-1, +1]
            mid_price:        current mid-price in quote currency
            equity:           total portfolio value in quote currency

        Returns:
            Filled Order if an order was placed, or None if delta is below
            the dust threshold or mid_price is zero.
        """
        delta = target_position - current_position

        if abs(delta) < self._min_delta:
            log.debug(
                "OrderExecutor: delta %.5f below threshold %.5f — no order",
                round(delta, 5), self._min_delta,
            )
            return None

        if mid_price <= 0.0:
            log.warning("OrderExecutor: mid_price <= 0 — cannot execute: pair=%s", pair)
            return None

        if equity <= 0.0:
            log.warning("OrderExecutor: equity <= 0 — cannot execute: pair=%s", pair)
            return None

        side = "buy" if delta > 0 else "sell"

        # Notional value of the delta position
        notional = abs(delta) * equity

        # Quantity in base currency
        qty = notional / mid_price

        # Slippage-adjusted fill price
        if side == "buy":
            fill_price = mid_price * (1.0 + self._slippage)
        else:
            fill_price = mid_price * (1.0 - self._slippage)

        order = Order(pair=pair, side=side, qty=qty, price=0.0)  # market order
        self._order_manager.submit(order)
        filled = self._order_manager.fill(order.order_id, fill_price)

        log.debug(
            "OrderExecutor: order filled: pair=%s side=%s qty=%.6f fill_price=%.4f delta=%.4f",
            pair, side, round(qty, 6), round(fill_price, 4), round(delta, 4),
        )
        return filled

    def __repr__(self) -> str:
        return (
            f"OrderExecutor(slippage={self._slippage:.3%}, "
            f"min_delta={self._min_delta:.3%})"
        )
