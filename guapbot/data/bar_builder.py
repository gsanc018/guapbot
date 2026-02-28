"""
guapbot/data/bar_builder.py

BarBuilder: converts a live KrakenWebSocket trade stream into completed OHLCV bars.

Needed for live trading (Session 7+). Wire it up like this:

    builder = BarBuilder(
        interval="1h",
        on_bar_close=lambda pair, interval, bar: store.append(pair, interval, bar.to_frame().T),
    )
    ws = KrakenWebSocket(pairs=["XBTUSD", "ETHUSD"])
    ws.on("trade", builder.on_trade)
    await ws.start()

The on_bar_close callback fires once per completed bar per pair with:
  pair      — GuapBot pair string, e.g. "XBTUSD"
  interval  — e.g. "1h"
  bar       — pd.Series with index [open, high, low, close, volume, trades]
              and bar.name = bar open timestamp (UTC-aware)

The callback is responsible for:
  1. Appending the bar to ParquetStore (persistence)
  2. Triggering the feature pipeline (signal generation)

TODO: implement before wiring up the live trading loop (Session 7+).
Implementation notes:
  - One BarBuilder instance per interval. Use a dict to track per-pair state:
        self._state: dict[str, _BarState] = {}
  - _BarState holds: open, high, low, close, volume, trades, bar_open_ts
  - _bar_open_time(ts, interval): floor ts to bar boundary
        e.g. ts=14:37 with interval=1h → bar_open=14:00
        Use pd.Timestamp.floor(freq) from INTERVAL_FREQ in aggregator.py
  - on_trade receives Kraken WS trade dict. Relevant keys (v2 format):
        price (float), qty (float), timestamp (ISO8601 string or datetime)
  - When trade.timestamp >= bar_open + bar_duration: finalize bar, emit, reset state
  - Thread safety: asyncio single-threaded, no locks needed
"""
from __future__ import annotations

from typing import Callable

import pandas as pd


class BarBuilder:
    """
    Accumulates live tick events from KrakenWebSocket and emits completed OHLCV bars.

    One BarBuilder per interval. Maintains per-pair in-progress bar state.
    Emits completed bars via the on_bar_close callback when the bar period ends.

    TODO: implement — needed for live trading (Session 7+).
    See module docstring for implementation guidance.
    """

    def __init__(
        self,
        interval: str,
        on_bar_close: Callable[[str, str, pd.Series], None],
    ) -> None:
        """
        Args:
            interval:     '1h', '4h', or '1d'.
            on_bar_close: Async or sync callback invoked when a bar completes.
                          Signature: on_bar_close(pair: str, interval: str, bar: pd.Series)
                          bar.name = bar open timestamp (UTC-aware pd.Timestamp)
                          bar columns: open, high, low, close, volume, trades
        """
        raise NotImplementedError(
            "BarBuilder not yet implemented — needed for live trading (Session 7+). "
            "See module docstring for implementation guidance."
        )

    async def on_trade(self, pair: str, trade: dict) -> None:
        """
        Process one live trade from the KrakenWebSocket trade channel.

        Args:
            pair:  GuapBot pair string (e.g., 'XBTUSD').
            trade: Trade dict from Kraken WS v2 with keys:
                   price (float), qty (float), timestamp (ISO8601 string),
                   side ("buy"/"sell"), ord_type ("market"/"limit"), trade_id (int).

        TODO: accumulate into current bar; call on_bar_close when bar period ends.
        """
        raise NotImplementedError("BarBuilder not yet implemented")
