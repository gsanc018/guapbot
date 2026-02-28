"""
guapbot/data/kraken_ws.py

Kraken WebSocket v2 client.
Streams live ticker and trade data for XBTUSD and ETHUSD.
Writes latest prices into MarketState (Redis) via an injected callback.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Callable, Optional

import websockets
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from guapbot.utils.logging import get_logger

logger = get_logger(__name__)

# Kraken WebSocket v2 endpoint
WS_URL = "wss://ws.kraken.com/v2"

# Map from guapbot pair names to Kraken WS pair strings
WS_PAIRS = {
    "XBTUSD": "XBT/USD",
    "ETHUSD": "ETH/USD",
    "ETHBTC": "ETH/XBT",
}


class KrakenWebSocket:
    """
    Async WebSocket client for live Kraken price data.

    Usage:
        ws = KrakenWebSocket(pairs=["XBTUSD", "ETHUSD"])
        ws.on("ticker", my_callback)   # async callback(pair, data)
        ws.on("trade", my_callback)
        await ws.start()

    The caller is responsible for running this inside an asyncio event loop.
    The fast clock in the main runner starts this as a background task.
    """

    def __init__(self, pairs: list[str] | None = None) -> None:
        self._pairs = pairs or ["XBTUSD", "ETHUSD"]
        self._callbacks: dict[str, list[Callable]] = {"ticker": [], "trade": []}
        self._running = False
        self._ws: Optional[websockets.WebSocketClientProtocol] = None

    # ------------------------------------------------------------------
    # Public API (matches data/social/base.py interface pattern)
    # ------------------------------------------------------------------

    def on(self, event: str, callback: Callable) -> None:
        """Register an async callback for 'ticker' or 'trade' events."""
        if event not in self._callbacks:
            raise ValueError(f"Unknown event '{event}'. Use: ticker, trade")
        self._callbacks[event].append(callback)

    async def start(self) -> None:
        """
        Connect to Kraken WebSocket and begin streaming.
        Reconnects automatically on disconnect.
        Call stop() to exit cleanly.
        """
        self._running = True
        logger.info(f"KrakenWebSocket starting for pairs: {self._pairs}")
        while self._running:
            try:
                await self._connect_and_stream()
            except (ConnectionClosedError, ConnectionClosedOK) as exc:
                if self._running:
                    logger.warning(f"WS disconnected: {exc} — reconnecting in 5s")
                    await asyncio.sleep(5)
            except Exception as exc:
                if self._running:
                    logger.error(f"WS error: {exc} — reconnecting in 10s")
                    await asyncio.sleep(10)
        logger.info("KrakenWebSocket stopped.")

    async def stop(self) -> None:
        """Signal the stream to stop and close the connection."""
        self._running = False
        if self._ws:
            await self._ws.close()

    # ------------------------------------------------------------------
    # Internal: connect, subscribe, listen
    # ------------------------------------------------------------------

    async def _connect_and_stream(self) -> None:
        async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=30) as ws:
            self._ws = ws
            logger.info("WS connected to Kraken")

            # Subscribe to ticker and trade channels
            ws_pairs = [WS_PAIRS[p] for p in self._pairs if p in WS_PAIRS]
            await ws.send(json.dumps({
                "method": "subscribe",
                "params": {
                    "channel": "ticker",
                    "symbol": ws_pairs,
                }
            }))
            await ws.send(json.dumps({
                "method": "subscribe",
                "params": {
                    "channel": "trade",
                    "symbol": ws_pairs,
                }
            }))

            async for raw in ws:
                if not self._running:
                    break
                await self._handle_message(raw)

    async def _handle_message(self, raw: str) -> None:
        """Parse a raw WebSocket message and fire callbacks."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        channel = msg.get("channel")
        if channel not in ("ticker", "trade"):
            # heartbeat, subscriptionStatus, systemStatus — ignore
            return

        data = msg.get("data")
        if not data:
            return

        # Kraken v2: data is a list; each item has 'symbol'
        for item in data:
            symbol = item.get("symbol", "")
            # Convert back to guapbot pair name
            pair = self._ws_symbol_to_pair(symbol)
            if pair is None:
                continue

            callbacks = self._callbacks.get(channel, [])
            for cb in callbacks:
                try:
                    await cb(pair, item)
                except Exception as exc:
                    logger.error(f"WS callback error ({channel}/{pair}): {exc}")

    @staticmethod
    def _ws_symbol_to_pair(symbol: str) -> Optional[str]:
        """Convert WS symbol 'XBT/USD' back to 'XBTUSD'."""
        reverse = {v: k for k, v in WS_PAIRS.items()}
        return reverse.get(symbol)


# ------------------------------------------------------------------
# Convenience: latest price tracker
# ------------------------------------------------------------------

class LivePriceTracker:
    """
    Simple in-process price tracker using KrakenWebSocket.
    Useful for paper trading and dashboard without Redis.
    For production, MarketState (Redis) is the source of truth.
    """

    def __init__(self, pairs: list[str] | None = None) -> None:
        self.prices: dict[str, float] = {}
        self._ws = KrakenWebSocket(pairs=pairs)
        self._ws.on("ticker", self._on_ticker)

    async def _on_ticker(self, pair: str, data: dict) -> None:
        last = data.get("last")
        if last is not None:
            self.prices[pair] = float(last)
            logger.debug(f"Live price {pair}: {last}")

    async def start(self) -> None:
        await self._ws.start()

    async def stop(self) -> None:
        await self._ws.stop()

    def get_price(self, pair: str) -> Optional[float]:
        return self.prices.get(pair)
