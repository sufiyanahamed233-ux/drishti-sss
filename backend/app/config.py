"""
config.py
---------
Application settings loaded from environment variables (or a .env file).

All configuration is centralised here so every module imports from a single
source of truth.  No hardcoded values live outside this file.

Environment variables
~~~~~~~~~~~~~~~~~~~~~
DATABASE_URL
    Full SQLAlchemy-compatible connection URL, e.g.::

        postgresql+psycopg2://user:pass@localhost:5432/drishti

    Defaults to a local ``drishti`` database for development.

APP_ENV
    One of ``development``, ``testing``, or ``production``.
    Defaults to ``development``.

DEBUG
    Boolean flag.  Defaults to ``False`` in production, ``True`` otherwise.
"""

from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Immutable application settings resolved from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = (
        "postgresql+psycopg2://drishti:drishti@localhost:5432/drishti"
    )

    # ── Application ───────────────────────────────────────────────────────────
    app_env: str = "development"
    debug: bool = False
    app_title: str = "Drishti SSS API"
    app_version: str = "0.3.0"

    @field_validator("app_env")
    @classmethod
    def validate_app_env(cls, v: str) -> str:
        allowed = {"development", "testing", "production"}
        if v not in allowed:
            raise ValueError(f"app_env must be one of {allowed}, got {v!r}")
        return v


# Module-level singleton – import this everywhere.
settings = Settings()
