"""
guapbot.execution.market_state
-------------------------------
Redis-backed shared state object.

MarketState is the communication bus between the slow clock (hourly
ensemble + execution) and the fast clock (real-time social streams).
Every process — trader, streams, dashboard — reads and writes the same
Redis instance.

Interface contract:
    state.update(key, value)     → write to Redis
    state.get(key)               → read from Redis (any process)
    state.inject_alert(alert)    → fast clock writes an alert here

Keys are namespaced under 'guapbot:' to avoid collisions if Redis is
shared with other applications.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

try:
    import redis
    _REDIS_AVAILABLE = True
except ImportError:
    redis = None
    _REDIS_AVAILABLE = False

from guapbot.utils.config import settings
from guapbot.utils.logging import get_logger

log = get_logger(__name__)

# Redis key namespace
_NS = "guapbot:"


def _key(name: str) -> str:
    return f"{_NS}{name}"


class MarketState:
    """
    Redis-backed shared state for all GuapBot processes.

    Values are JSON-serialised so any Python type that is JSON-safe can
    be stored. Timestamps are stored as ISO strings.

    Usage:
        state = MarketState()

        # Slow clock writes the latest signal
        state.update("signal.XBTUSD", 0.72)

        # Fast clock injects a social alert
        state.inject_alert({"source": "reddit", "severity": 0.8})

        # Dashboard reads live position
        pos = state.get("position.XBTUSD")
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        db: int | None = None,
        password: str | None = None,
    ) -> None:
        self._host = host or settings.redis_host
        self._port = port or settings.redis_port
        self._db = db or settings.redis_db
        self._password = password or (settings.redis_password or None)
        self._client: redis.Redis | None = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> "MarketState":
        """Open the Redis connection. Safe to call multiple times."""
        if self._client is not None:
            return self
        self._client = redis.Redis(
            host=self._host,
            port=self._port,
            db=self._db,
            password=self._password,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        try:
            self._client.ping()
            log.info("MarketState connected to Redis", host=self._host, port=self._port)
        except redis.ConnectionError as exc:
            log.warning(f"Redis not available: {exc}. Running in no-op mode.")
            self._client = None
        return self

    def disconnect(self) -> None:
        """Close the Redis connection."""
        if self._client:
            self._client.close()
            self._client = None
            log.info("MarketState disconnected from Redis")

    @property
    def connected(self) -> bool:
        """True if Redis is connected and responding."""
        if self._client is None:
            return False
        try:
            self._client.ping()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def update(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """
        Write *value* to Redis under *key*.

        Args:
            key:   logical key, e.g. 'signal.XBTUSD' (namespace is added automatically)
            value: any JSON-serialisable Python value
            ttl:   optional TTL in seconds; None = no expiry

        Returns:
            True on success, False if Redis is unavailable.
        """
        if self._client is None:
            log.debug(f"MarketState.update({key!r}) skipped — no Redis connection")
            return False
        try:
            serialised = json.dumps(value, default=str)
            redis_key = _key(key)
            if ttl is not None:
                self._client.setex(redis_key, ttl, serialised)
            else:
                self._client.set(redis_key, serialised)
            return True
        except (redis.RedisError, TypeError) as exc:
            log.error(f"MarketState.update({key!r}) failed: {exc}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """
        Read a value from Redis.

        Args:
            key:     logical key (namespace added automatically)
            default: returned if key is missing or Redis is unavailable

        Returns:
            Deserialised Python value, or *default*.
        """
        if self._client is None:
            return default
        try:
            raw = self._client.get(_key(key))
            if raw is None:
                return default
            return json.loads(raw)
        except (redis.RedisError, json.JSONDecodeError) as exc:
            log.error(f"MarketState.get({key!r}) failed: {exc}")
            return default

    def delete(self, key: str) -> bool:
        """Remove a key from Redis."""
        if self._client is None:
            return False
        try:
            self._client.delete(_key(key))
            return True
        except redis.RedisError as exc:
            log.error(f"MarketState.delete({key!r}) failed: {exc}")
            return False

    def inject_alert(self, alert: dict) -> bool:
        """
        Fast clock writes a social alert to shared state.

        The alert dict should contain at minimum:
            source:      which stream (reddit, telegram, rss, stocktwits)
            severity:    0.0 – 1.0
            description: human-readable summary
            fired_at:    ISO timestamp

        The execution layer polls 'alert.active' on every order decision.

        Args:
            alert: dict with alert payload

        Returns:
            True on success.
        """
        if "fired_at" not in alert:
            alert["fired_at"] = datetime.utcnow().isoformat()
        ok = self.update("alert.current", alert)
        if ok:
            self.update("alert.active", True)
            log.warning(
                "Alert injected into MarketState",
                source=alert.get("source"),
                severity=alert.get("severity"),
            )
        return ok

    def clear_alert(self) -> bool:
        """
        Clear the active alert. Called by LLM triage after confirming all-clear.
        """
        self.update("alert.active", False)
        self.delete("alert.current")
        log.info("Alert cleared in MarketState")
        return True

    # ------------------------------------------------------------------
    # Convenience readers
    # ------------------------------------------------------------------

    def get_alert(self) -> dict | None:
        """Return the current alert dict, or None if no alert is active."""
        if not self.get("alert.active", False):
            return None
        return self.get("alert.current")

    def get_signal(self, pair: str) -> float | None:
        """Return the latest ensemble signal for *pair*, or None."""
        return self.get(f"signal.{pair}")

    def get_position(self, pair: str) -> float | None:
        """Return the current position fraction for *pair*, or None."""
        return self.get(f"position.{pair}")

    def set_position(self, pair: str, position: float) -> bool:
        """Write the current position fraction for *pair*."""
        return self.update(f"position.{pair}", position)

    # ------------------------------------------------------------------
    # Debugging / inspection
    # ------------------------------------------------------------------

    def all_keys(self) -> list[str]:
        """Return all GuapBot keys in Redis (strips namespace prefix)."""
        if self._client is None:
            return []
        try:
            raw_keys = self._client.keys(f"{_NS}*")
            return [k.removeprefix(_NS) for k in raw_keys]
        except redis.RedisError:
            return []

    def __repr__(self) -> str:
        status = "connected" if self.connected else "disconnected"
        return f"MarketState(redis={self._host}:{self._port}/{self._db}, {status})"
