"""
guapbot/utils/config.py

Pydantic-Settings config. Reads from .env file and environment variables.
This is the single source of truth for all configuration.

Add new settings here. Do NOT hardcode values elsewhere.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Kraken ---
    kraken_api_key: str = ""
    kraken_api_secret: str = ""

    # --- Telegram ---
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # --- Reddit (PRAW) ---
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "GuapBot/1.0"

    # --- Telethon ---
    telethon_api_id: str = ""
    telethon_api_hash: str = ""

    # --- Anthropic ---
    anthropic_api_key: str = ""

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- PostgreSQL ---
    postgres_url: str = "postgresql://guapbot:guapbot@localhost:5432/guapbot"

    # --- Paths ---
    data_cache_dir: Path = Path("data/cache")
    models_dir: Path = Path("models")
    logs_dir: Path = Path("logs")

    # --- Trading ---
    paper_mode: bool = True  # Always start in paper mode — switch to False deliberately
    max_long_pct: float = 0.25   # 25% max long position
    max_short_pct: float = 0.15  # 15% max short position
    kill_switch_daily_drawdown: float = -0.05   # -5% daily halts trading
    kill_switch_total_drawdown: float = -0.15   # -15% total halts trading

    # --- LLM Context ---
    llm_context_refresh_hours: int = 4


# Module-level singleton — import this everywhere
settings = Settings()
