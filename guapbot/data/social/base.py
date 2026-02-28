"""
guapbot.data.social.base
------------------------
Base interface for all real-time social data sources.

Every social source (Reddit, Telegram, RSS, Stocktwits) implements this
interface. The alert bus subscribes to all sources via .on('message', cb).

Interface contract:
    source.start()                → begin streaming (async)
    source.stop()                 → clean shutdown (async)
    source.on('message', callback) → register async callback

Callbacks receive a SocialMessage dataclass.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Awaitable, Callable

from guapbot.utils.logging import get_logger

log = get_logger(__name__)


class SourceType(str, Enum):
    reddit = "reddit"
    telegram = "telegram"
    rss = "rss"
    stocktwits = "stocktwits"
    fear_and_greed = "fear_and_greed"


@dataclass
class SocialMessage:
    """
    Normalised message from any social source.

    All sources produce this structure so the alert bus can treat them
    uniformly regardless of origin.
    """

    source: SourceType
    text: str
    timestamp: datetime
    url: str = ""
    author: str = ""
    score: float = 0.0           # upvotes, likes, or raw sentiment score
    raw: dict = field(default_factory=dict)   # original payload

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("SocialMessage.text must not be empty")
        if not isinstance(self.timestamp, datetime):
            raise TypeError("SocialMessage.timestamp must be a datetime")


# Type alias for message callbacks
MessageCallback = Callable[[SocialMessage], Awaitable[None]]


class SocialSource(ABC):
    """
    Abstract base class for all real-time social data sources.

    Subclasses must implement:
        _stream() — the async generator / loop that yields raw messages
        _parse()  — converts raw payload to SocialMessage

    The public API (start / stop / on) is provided here and should not
    be overridden.
    """

    source_type: SourceType  # must be set by each subclass as class var

    def __init__(self) -> None:
        self._callbacks: list[MessageCallback] = []
        self._running = False
        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Public interface (do NOT override)
    # ------------------------------------------------------------------

    def on(self, event: str, callback: MessageCallback) -> None:
        """
        Register an async callback for the given event.

        Currently only 'message' is supported.

        Args:
            event:    must be 'message'
            callback: async function that receives a SocialMessage
        """
        if event != "message":
            raise ValueError(f"Unknown event '{event}'. Only 'message' is supported.")
        if not asyncio.iscoroutinefunction(callback):
            raise TypeError("Callback must be an async function (async def)")
        self._callbacks.append(callback)
        log.debug(f"Callback registered on {self.__class__.__name__}")

    async def start(self) -> None:
        """Begin streaming. Creates a background async task."""
        if self._running:
            log.warning(f"{self.__class__.__name__} already running, ignoring start()")
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name=f"guapbot-{self.source_type}")
        log.info(f"{self.__class__.__name__} started")

    async def stop(self) -> None:
        """Clean shutdown. Cancels the background task and waits for it."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info(f"{self.__class__.__name__} stopped")

    # ------------------------------------------------------------------
    # Internal machinery
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        """Drive the abstract _stream() generator and dispatch messages."""
        try:
            async for message in self._stream():
                if not self._running:
                    break
                await self._dispatch(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.error(f"{self.__class__.__name__} stream error: {exc}")

    async def _dispatch(self, message: SocialMessage) -> None:
        """Fan out a message to all registered callbacks."""
        for cb in self._callbacks:
            try:
                await cb(message)
            except Exception as exc:  # noqa: BLE001
                log.error(f"Callback error in {self.__class__.__name__}: {exc}")

    # ------------------------------------------------------------------
    # Abstract methods — subclasses implement these
    # ------------------------------------------------------------------

    @abstractmethod
    async def _stream(self):
        """
        Async generator that yields SocialMessage objects.

        Implementations should loop until self._running is False.
        Handle reconnection logic internally.

        Yields:
            SocialMessage
        """
        ...
