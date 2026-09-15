"""
src/generation/groq_client.py
=============================
Phase 12 — Isolated Groq API Client and Structured Response Handler.

Handles communication with GroqCloud endpoints using OpenAI-compatible structured
JSON schema completion. Wraps all network exceptions, timeouts, rate limits, and
malformed outputs into predictable domain exceptions.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from groq import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    Groq,
    RateLimitError,
)

from src.generation.config import GroqConfig, load_groq_config

logger = logging.getLogger(__name__)


class GroqGenerationError(Exception):
    """Domain exception raised when Groq API generation fails."""

    def __init__(
        self,
        message: str,
        error_type: str = "api_error",
        status_code: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.status_code = status_code


class GroqGenerationClient:
    """
    Isolated client wrapper for Groq structured JSON completions.
    """

    def __init__(
        self,
        config: Optional[GroqConfig] = None,
        client: Optional[Any] = None,
    ) -> None:
        self.config = config or load_groq_config()
        # Allow injecting a mock client for unit testing without network requests
        self._client = client or Groq(
            api_key=self.config.api_key,
            timeout=self.config.timeout_seconds,
            max_retries=self.config.max_retries,
        )

    def generate_json_response(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Request a structured JSON completion from Groq.

        Args:
            messages: List of chat message dicts (role, content).
            temperature: Sampling temperature (defaults to config.temperature).

        Returns:
            Parsed JSON response dictionary.

        Raises:
            GroqGenerationError: If the request fails, times out, or returns invalid JSON.
        """
        temp = temperature if temperature is not None else self.config.temperature

        try:
            logger.info("Dispatching Groq chat completion (model: %s)...", self.config.model)
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temp,
                response_format={"type": "json_object"},
            )

            raw_content = response.choices[0].message.content
            if not raw_content:
                raise GroqGenerationError(
                    "Groq returned empty response content",
                    error_type="empty_response",
                )

            try:
                parsed_json = json.loads(raw_content)
                if not isinstance(parsed_json, dict):
                    raise ValueError(f"Expected JSON object, got {type(parsed_json).__name__}")
                return parsed_json
            except (json.JSONDecodeError, ValueError) as json_err:
                logger.error("Failed to parse Groq response as valid JSON: %s", json_err)
                raise GroqGenerationError(
                    f"Malformed JSON returned by model: {json_err}",
                    error_type="malformed_json",
                ) from json_err

        except AuthenticationError as auth_err:
            logger.error("Groq Authentication Error: Invalid API key.")
            raise GroqGenerationError(
                "Authentication failed: Check your GROQ_API_KEY in .env",
                error_type="authentication_error",
                status_code=401,
            ) from auth_err

        except RateLimitError as rate_err:
            logger.warning("Groq Rate Limit Exceeded.")
            raise GroqGenerationError(
                "Rate limit exceeded on Groq API. Please retry shortly.",
                error_type="rate_limit_error",
                status_code=429,
            ) from rate_err

        except APITimeoutError as time_err:
            logger.error("Groq API Timeout after %.1fs", self.config.timeout_seconds)
            raise GroqGenerationError(
                f"Groq request timed out after {self.config.timeout_seconds}s",
                error_type="timeout_error",
                status_code=408,
            ) from time_err

        except APIConnectionError as conn_err:
            logger.error("Groq Network Connection Error: %s", conn_err)
            raise GroqGenerationError(
                "Network connection failed when contacting Groq API",
                error_type="network_error",
            ) from conn_err

        except APIError as api_err:
            logger.error("Groq API Error: %s", api_err)
            raise GroqGenerationError(
                f"Groq API returned an error: {api_err.message}",
                error_type="api_error",
                status_code=getattr(api_err, "status_code", 500),
            ) from api_err
