"""
src/generation/config.py
========================
Phase 12 — Groq API Configuration and Secure Environment Loader.

Loads GROQ_API_KEY and GROQ_MODEL strictly from environment variables or local .env.
Enforces that no API key is ever printed, logged, or serialized into responses.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env from project root if present
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()


DEFAULT_MODEL = "qwen/qwen3.8-27b"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_TEMPERATURE = 0.2


@dataclass(frozen=True)
class GroqConfig:
    """Secure configuration parameters for Groq API orchestration."""

    api_key: str
    model: str = DEFAULT_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    temperature: float = DEFAULT_TEMPERATURE

    def safe_metadata(self) -> dict[str, str | float | int]:
        """Return safe, non-sensitive metadata for output reporting."""
        return {
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
            "temperature": self.temperature,
            "max_retries": self.max_retries,
            "api_key_configured": bool(self.api_key),
        }

    def __repr__(self) -> str:
        # Prevent accidental printing of the API key
        return (
            f"GroqConfig(model='{self.model}', timeout={self.timeout_seconds}s, "
            f"api_key={'***configured***' if self.api_key else 'None'})"
        )


def load_groq_config(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> GroqConfig:
    """
    Load and validate Groq configuration from environment.

    Raises:
        ValueError: If GROQ_API_KEY or GROQ_MODEL is missing or blank.
    """
    key = api_key or os.getenv("GROQ_API_KEY", "")
    key = key.strip()

    if not key:
        raise ValueError(
            "GROQ_API_KEY is not configured! Please add GROQ_API_KEY to your local .env file."
        )

    model_name = model or os.getenv("GROQ_MODEL", "")
    model_name = model_name.strip()

    if not model_name:
        raise ValueError(
            "GROQ_MODEL is not configured! Please specify a model (e.g. 'qwen/qwen3.8-27b') "
            "in your local .env file."
        )

    return GroqConfig(
        api_key=key,
        model=model_name,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
    )
