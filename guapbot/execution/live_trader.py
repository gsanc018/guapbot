"""
guapbot/execution/live_trader.py

Live trading execution layer — Session 14.

This file is a stub. The full implementation ships in Session 14 once
the full system (portfolio layer, monitoring, Telegram, deployment) is
in place. Until then, all public methods raise NotImplementedError with
a clear message.

Session 14 will implement:
    - Real Kraken REST order placement via KrakenClient
    - Position reconciliation against live Kraken balance
    - Order fill confirmation via polling or WebSocket
    - All kill switch checks against live portfolio equity
    - Telegram notification on every trade and kill-switch event
"""
from __future__ import annotations

from guapbot.utils.logging import get_logger

log = get_logger(__name__)

_NOT_IMPLEMENTED_MSG = (
    "LiveTrader is not implemented until Session 14. "
    "Use PaperTrader for now, or set TRADING_MODE=paper in .env."
)


class LiveTrader:
    """
    Stub for the live Kraken order execution layer.

    Raises NotImplementedError on every method call. Set TRADING_MODE=paper
    to use PaperTrader instead.
    """

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        log.warning("LiveTrader is a stub — not implemented until Session 14.")

    def step(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def summary(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def __repr__(self) -> str:
        return "LiveTrader(STUB — Session 14)"
