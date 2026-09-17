"""Shared utilities for LLM clients."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Return True if the exception is a HTTP 429 rate-limit error."""
    return hasattr(exc, "status_code") and getattr(exc, "status_code") == 429


def _retry_decorator():
    """Return a tenacity retry decorator configured for 429 errors."""
    from tenacity import (
        retry,
        retry_if_exception,
        stop_after_attempt,
        wait_random_exponential,
    )

    return retry(
        retry=retry_if_exception(_is_rate_limit_error),
        wait=wait_random_exponential(multiplier=1, max=60),
        stop=stop_after_attempt(6),
        reraise=True,
    )