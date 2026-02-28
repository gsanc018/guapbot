"""
guapbot/execution/order_manager.py

Order lifecycle management for GuapBot.

Tracks the state of every order from submission through fill or cancellation.
In paper mode this is purely in-memory. In live mode the same interface is
used; the live trader layer calls Kraken and then records fills here.

Order lifecycle:
    submit()  → status = 'open'
    fill()    → status = 'filled'
    cancel()  → status = 'cancelled'
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from guapbot.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Order dataclass
# ---------------------------------------------------------------------------

@dataclass
class Order:
    """
    Represents a single order in the system.

    Attributes:
        order_id:   UUID string, auto-generated on construction.
        pair:       Trading pair, e.g. 'XBTUSD'.
        side:       'buy' or 'sell'.
        qty:        Quantity in base currency (e.g. BTC).
        price:      Limit price; 0.0 means market order.
        status:     'open', 'filled', or 'cancelled'.
        created_at: UTC datetime of submission.
        filled_at:  UTC datetime of fill (None until filled).
        fill_price: Actual execution price (0.0 until filled).
    """

    pair: str
    side: str          # 'buy' | 'sell'
    qty: float         # base-currency quantity
    price: float = 0.0  # 0 = market order

    # Set automatically
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "open"
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    filled_at: Optional[datetime] = None
    fill_price: float = 0.0

    def __post_init__(self) -> None:
        if self.side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {self.side!r}")
        if self.qty <= 0:
            raise ValueError(f"qty must be positive, got {self.qty}")

    @property
    def is_open(self) -> bool:
        return self.status == "open"

    @property
    def is_filled(self) -> bool:
        return self.status == "filled"

    @property
    def fill_value(self) -> float:
        """Notional value of the filled order (qty × fill_price)."""
        return self.qty * self.fill_price

    def __repr__(self) -> str:
        return (
            f"Order({self.side.upper()} {self.qty:.6f} {self.pair} "
            f"@ {self.fill_price or self.price:.2f} [{self.status}])"
        )


# ---------------------------------------------------------------------------
# OrderManager
# ---------------------------------------------------------------------------

class OrderManager:
    """
    In-memory order book and lifecycle manager.

    All orders are stored in two dicts:
        _open:   order_id → Order (submitted, not yet filled/cancelled)
        _closed: order_id → Order (filled or cancelled — for audit trail)

    Thread safety: not explicitly thread-safe. The slow clock is single-
    threaded, so this is fine for paper trading. Live trading adds locking
    in Session 14.
    """

    def __init__(self) -> None:
        self._open:   dict[str, Order] = {}
        self._closed: dict[str, Order] = {}

    # ------------------------------------------------------------------
    # Core lifecycle methods
    # ------------------------------------------------------------------

    def submit(self, order: Order) -> Order:
        """
        Record a new order as 'open'.

        Args:
            order: Order dataclass (order_id auto-generated if not set).

        Returns:
            The same Order object with status='open'.
        """
        order.status = "open"
        self._open[order.order_id] = order
        log.debug(
            "Order submitted: id=%s side=%s qty=%.6f pair=%s",
            order.order_id[:8], order.side, round(order.qty, 6), order.pair,
        )
        return order

    def fill(
        self,
        order_id: str,
        fill_price: float,
        fill_time: Optional[datetime] = None,
    ) -> Order:
        """
        Mark an order as filled.

        Args:
            order_id:   UUID of the order to fill.
            fill_price: Actual execution price.
            fill_time:  UTC datetime of fill (defaults to now).

        Returns:
            The updated Order.

        Raises:
            KeyError: if order_id is not in the open book.
        """
        if order_id not in self._open:
            raise KeyError(f"No open order with id {order_id!r}")
        order = self._open.pop(order_id)
        order.status = "filled"
        order.fill_price = fill_price
        order.filled_at = fill_time or datetime.now(tz=timezone.utc)
        self._closed[order_id] = order
        log.debug(
            "Order filled: id=%s side=%s qty=%.6f fill_price=%.4f",
            order_id[:8], order.side, round(order.qty, 6), round(fill_price, 4),
        )
        return order

    def cancel(self, order_id: str) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: UUID of the order to cancel.

        Returns:
            True if cancelled, False if order was not open.
        """
        if order_id not in self._open:
            log.warning("cancel(): order not found or already closed: id=%s", order_id[:8])
            return False
        order = self._open.pop(order_id)
        order.status = "cancelled"
        self._closed[order_id] = order
        log.debug("Order cancelled: id=%s", order_id[:8])
        return True

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_open(self) -> list[Order]:
        """Return all currently open orders."""
        return list(self._open.values())

    def get_history(self, pair: Optional[str] = None) -> list[Order]:
        """
        Return all closed (filled + cancelled) orders, newest first.

        Args:
            pair: optional filter by trading pair.
        """
        orders = list(self._closed.values())
        if pair:
            orders = [o for o in orders if o.pair == pair]
        return sorted(orders, key=lambda o: o.created_at, reverse=True)

    def get_filled(self, pair: Optional[str] = None) -> list[Order]:
        """Return only filled orders, newest first."""
        return [o for o in self.get_history(pair=pair) if o.is_filled]

    # ------------------------------------------------------------------
    # Stats helpers
    # ------------------------------------------------------------------

    def n_trades(self, pair: Optional[str] = None) -> int:
        """Total number of filled orders."""
        return len(self.get_filled(pair=pair))

    def cancel_all_open(self) -> int:
        """Cancel every open order. Returns count of cancelled orders."""
        ids = list(self._open.keys())
        for oid in ids:
            self.cancel(oid)
        return len(ids)

    def __repr__(self) -> str:
        return (
            f"OrderManager(open={len(self._open)}, "
            f"closed={len(self._closed)})"
        )
