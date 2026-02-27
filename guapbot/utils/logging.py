"""
guapbot.utils.logging
---------------------
Structured logging using stdlib logging. Three handlers:

  1. Console  — INFO+, coloured via ANSI
  2. app.log  — DEBUG+, JSON, rotating
  3. trades.log — trade events only

When loguru is installed, this module can be replaced transparently.

Usage:
    from guapbot.utils.logging import get_logger, log_trade

    log = get_logger(__name__)
    log.info("Order placed: %s %s %.4f @ %.2f", pair, side, size, price)
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from pathlib import Path

from guapbot.utils.config import settings

_configured = False


class _JsonFormatter(logging.Formatter):
    """Emit records as single-line JSON for structured log parsing."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "module": record.name,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # Merge any extra fields attached to the record
        for key, val in record.__dict__.items():
            if key not in (
                "args", "asctime", "created", "exc_info", "exc_text",
                "filename", "funcName", "levelname", "levelno", "lineno",
                "message", "module", "msecs", "msg", "name", "pathname",
                "process", "processName", "relativeCreated", "stack_info",
                "thread", "threadName",
            ):
                payload[key] = val
        return json.dumps(payload, default=str)


class _TradeFilter(logging.Filter):
    """Pass only records that carry trade_event=True."""

    def filter(self, record: logging.LogRecord) -> bool:
        return bool(getattr(record, "trade_event", False))


_ANSI = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[35m",
    "RESET": "\033[0m",
}


class _ColorFormatter(logging.Formatter):
    fmt = "%(color)s%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d — %(message)s%(reset)s"

    def format(self, record: logging.LogRecord) -> str:
        record.color = _ANSI.get(record.levelname, "")
        record.reset = _ANSI["RESET"]
        return super().format(record)


def setup_logging(log_dir: Path | None = None, level: str | None = None) -> None:
    """Configure all logging handlers. Idempotent."""
    global _configured
    if _configured:
        return

    _log_dir = log_dir or settings.log_dir
    _level = level or settings.log_level
    _log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("guapbot")
    root.setLevel(logging.DEBUG)

    # Console
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(getattr(logging, _level, logging.INFO))
    ch.setFormatter(_ColorFormatter(datefmt="%Y-%m-%d %H:%M:%S"))
    root.addHandler(ch)

    # app.log — rotating JSON
    fh = logging.handlers.RotatingFileHandler(
        _log_dir / "app.log", maxBytes=50 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_JsonFormatter())
    root.addHandler(fh)

    # trades.log — trade events only
    th = logging.handlers.RotatingFileHandler(
        _log_dir / "trades.log", maxBytes=10 * 1024 * 1024, backupCount=10, encoding="utf-8"
    )
    th.setLevel(logging.INFO)
    th.setFormatter(_JsonFormatter())
    th.addFilter(_TradeFilter())
    root.addHandler(th)

    root.info("Logging initialised", extra={"log_dir": str(_log_dir), "level": _level})
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under guapbot.*name*."""
    setup_logging()
    return logging.getLogger(f"guapbot.{name}")


def log_trade(
    *,
    event: str,
    pair: str,
    side: str,
    size: float,
    price: float,
    strategy: str,
    **kwargs,
) -> None:
    """Write a structured trade event to trades.log."""
    setup_logging()
    log = logging.getLogger("guapbot.trades")
    log.info(
        "[TRADE] %s | %s %s %.6f @ %.2f",
        event, pair, side, size, price,
        extra={
            "trade_event": True,
            "event": event,
            "pair": pair,
            "side": side,
            "size": size,
            "price": price,
            "strategy": strategy,
            **kwargs,
        },
    )
