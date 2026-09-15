"""
src/api/config.py
=================
Phase 15 — API Configuration & Environment Settings.

Manages network, host, port, timeout, and CORS settings for the Production API.
Designed for deployment on Render and consumption by a future Vercel frontend.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]


@dataclass(frozen=True)
class ApiConfig:
    """Configuration settings for the customer agent FastAPI service."""

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = field(default_factory=lambda: list(DEFAULT_CORS_ORIGINS))
    cors_origin_regex: Optional[str] = None
    default_review_mode: str = "deterministic"
    request_timeout_seconds: float = 30.0
    max_message_length: int = 1000

    def safe_metadata(self) -> dict:
        """Returns non-sensitive configuration metadata for diagnostic endpoints."""
        return {
            "host": self.host,
            "port": self.port,
            "cors_origins_count": len(self.cors_origins),
            "cors_regex_active": bool(self.cors_origin_regex),
            "default_review_mode": self.default_review_mode,
            "request_timeout_seconds": self.request_timeout_seconds,
            "max_message_length": self.max_message_length,
        }


def load_api_config() -> ApiConfig:
    """
    Load API configuration from environment variables with sensible production defaults.
    """
    # Port configuration (Render passes PORT as an integer environment variable)
    port_str = os.environ.get("PORT", os.environ.get("API_PORT", "8000")).strip()
    try:
        port = int(port_str)
    except ValueError:
        port = 8000

    host = os.environ.get("API_HOST", "0.0.0.0").strip()

    # CORS origins parsing
    raw_origins = os.environ.get("CORS_ORIGINS", "")
    origins = list(DEFAULT_CORS_ORIGINS)
    if raw_origins:
        for item in raw_origins.split(","):
            cleaned = item.strip()
            if cleaned and cleaned not in origins:
                origins.append(cleaned)

    # Optional origin regex (defaults to None; wildcard regex disabled for production security)
    cors_regex = (
        os.environ.get("CORS_ORIGIN_REGEX", os.environ.get("CORS_VERCEL_REGEX", "")).strip() or None
    )

    # Review mode default
    review_mode = os.environ.get("API_DEFAULT_REVIEW_MODE", "deterministic").strip().lower()
    if review_mode not in ("deterministic", "hybrid"):
        review_mode = "deterministic"

    # Timeout
    timeout_str = os.environ.get("API_REQUEST_TIMEOUT_SECONDS", "30.0").strip()
    try:
        timeout = float(timeout_str)
    except ValueError:
        timeout = 30.0

    # Max message length
    length_str = os.environ.get("API_MAX_MESSAGE_LENGTH", "1000").strip()
    try:
        max_length = int(length_str)
    except ValueError:
        max_length = 1000

    return ApiConfig(
        host=host,
        port=port,
        cors_origins=origins,
        cors_origin_regex=cors_regex,
        default_review_mode=review_mode,
        request_timeout_seconds=timeout,
        max_message_length=max_length,
    )
