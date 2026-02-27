"""
guapbot.utils.config
--------------------
Application configuration loaded from environment variables and an
optional .env file. Uses only stdlib + python-dotenv (always available).

Usage:
    from guapbot.utils.config import settings

    key  = settings.kraken_api_key
    mode = settings.trading_mode     # 'paper' or 'live'
    ok   = settings.is_paper         # True
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Load .env file if present — no-op if file doesn't exist
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass


def _float(val: str | None, default: float) -> float:
    try:
        return float(val) if val is not None else default
    except (ValueError, AttributeError, TypeError):
        return default


def _int(val: str | None, default: int) -> int:
    try:
        return int(val) if val is not None else default
    except (ValueError, AttributeError, TypeError):
        return default


@dataclass
class Settings:
    """All GuapBot configuration, sourced from environment / .env file."""

    # Paths 
    data_cache_dir: Path = Path("data/cache")
    models_dir: Path = Path("models")
    logs_dir: Path = Path("logs")

    # Kraken
    kraken_api_key: str = field(default_factory=lambda: os.getenv("KRAKEN_API_KEY", ""))
    kraken_api_secret: str = field(default_factory=lambda: os.getenv("KRAKEN_API_SECRET", ""))

    # Anthropic
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))

    # Redis
    redis_host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    redis_port: int = field(default_factory=lambda: _int(os.getenv("REDIS_PORT"), 6379))
    redis_db: int = field(default_factory=lambda: _int(os.getenv("REDIS_DB"), 0))
    redis_password: str = field(default_factory=lambda: os.getenv("REDIS_PASSWORD", ""))

    # PostgreSQL
    postgres_host: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    postgres_port: int = field(default_factory=lambda: _int(os.getenv("POSTGRES_PORT"), 5432))
    postgres_db: str = field(default_factory=lambda: os.getenv("POSTGRES_DB", "guapbot"))
    postgres_user: str = field(default_factory=lambda: os.getenv("POSTGRES_USER", "guapbot"))
    postgres_password: str = field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", ""))

    # Telegram
    telegram_bot_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))
    telegram_api_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_API_ID", ""))
    telegram_api_hash: str = field(default_factory=lambda: os.getenv("TELEGRAM_API_HASH", ""))

    # Reddit
    reddit_client_id: str = field(default_factory=lambda: os.getenv("REDDIT_CLIENT_ID", ""))
    reddit_client_secret: str = field(default_factory=lambda: os.getenv("REDDIT_CLIENT_SECRET", ""))
    reddit_username: str = field(default_factory=lambda: os.getenv("REDDIT_USERNAME", ""))
    reddit_password: str = field(default_factory=lambda: os.getenv("REDDIT_PASSWORD", ""))

    # Trading mode
    trading_mode: str = field(default_factory=lambda: os.getenv("TRADING_MODE", "paper").lower())

    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper())
    log_dir: Path = field(default_factory=lambda: Path(os.getenv("LOG_DIR", "logs")))

    # Data
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "data/cache")))

    # Risk limits
    daily_drawdown_halt: float = field(
        default_factory=lambda: _float(os.getenv("DAILY_DRAWDOWN_HALT"), -0.05)
    )
    total_drawdown_halt: float = field(
        default_factory=lambda: _float(os.getenv("TOTAL_DRAWDOWN_HALT"), -0.15)
    )
    max_long_fraction: float = field(
        default_factory=lambda: _float(os.getenv("MAX_LONG_FRACTION"), 0.25)
    )
    max_short_fraction: float = field(
        default_factory=lambda: _float(os.getenv("MAX_SHORT_FRACTION"), 0.15)
    )

    # Assets (fixed)
    traded_pairs: list = field(default_factory=lambda: ["XBTUSD", "ETHUSD"])
    signal_only_pairs: list = field(default_factory=lambda: ["ETHBTC"])

    def __post_init__(self) -> None:
        if self.trading_mode not in ("paper", "live"):
            raise ValueError(f"TRADING_MODE must be 'paper' or 'live', got {self.trading_mode!r}")
        if self.daily_drawdown_halt >= 0:
            raise ValueError("DAILY_DRAWDOWN_HALT must be negative")
        if self.total_drawdown_halt >= 0:
            raise ValueError("TOTAL_DRAWDOWN_HALT must be negative")
        if not (0 < self.max_long_fraction <= 1):
            raise ValueError("MAX_LONG_FRACTION must be in (0, 1]")
        if not (0 < self.max_short_fraction <= 1):
            raise ValueError("MAX_SHORT_FRACTION must be in (0, 1]")

    @property
    def is_paper(self) -> bool:
        return self.trading_mode == "paper"

    @property
    def is_live(self) -> bool:
        return self.trading_mode == "live"

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


# Module-level singleton
settings = Settings()
