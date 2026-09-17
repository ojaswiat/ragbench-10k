"""NVIDIA NIM client provider.

Returns a ready-to-use AsyncOpenAI instance pointed at the NIM endpoint
with retry and semaphore applied. Enforces a rate limit of 40 requests per
minute to stay within the free tier limits.
"""

from __future__ import annotations

import asyncio
import logging
import time

from . import config
from .utils import _retry_decorator
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Rate limit: 40 requests per 60 seconds => min interval 1.5 seconds
_MIN_REQUEST_INTERVAL = 60.0 / 40.0
_last_request_time = 0.0
_request_lock = asyncio.Lock()

# A hung call must fail rather than hold its semaphore/lock slot forever.
# 120s sits well above the roughly 10s per-call latency observed live, so
# this only trips for a genuinely stuck request, which then gets logged and
# raised instead of blocking its slot and every other waiter indefinitely.
REQUEST_TIMEOUT_SEC = 120.0


def get_nim_client(model: str) -> AsyncOpenAI:
    """Return an AsyncOpenAI client configured for NIM endpoint.

    The returned object's chat.completions.create method includes retry
    logic, semaphore limiting, and a 40 RPM rate limiter.
    """
    if not config.NIM_API_KEY:
        raise ValueError("NIM_API_KEY must be set in environment")
    _client = AsyncOpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=config.NIM_API_KEY,
    )
    _semaphore = asyncio.Semaphore(getattr(config, "NIM_MAX_CONCURRENCY", 5))

    original_create = _client.chat.completions.create

    @_retry_decorator()
    async def wrapped_create(**kwargs):
        global _last_request_time
        async with _semaphore:
            # The lock gates only when a request may START. Reserving the
            # slot and releasing before the await keeps starts paced at the
            # RPM ceiling while letting up to _semaphore calls run
            # concurrently. Holding it across the call serialised everything
            # and let one hung request block every other waiter.
            wait = 0.0
            async with _request_lock:
                now = time.monotonic()
                wait = max(0.0, _MIN_REQUEST_INTERVAL - (now - _last_request_time))
                _last_request_time = now + wait
            if wait:
                await asyncio.sleep(wait)
            try:
                result = await asyncio.wait_for(
                    original_create(**kwargs), timeout=REQUEST_TIMEOUT_SEC
                )
            except Exception as exc:
                # Logged before re-raising so a retry that eventually
                # succeeds still leaves a record of the failed attempt. This
                # also catches asyncio.TimeoutError from the wait_for above,
                # which is why the exception is rendered with %r: an
                # asyncio.TimeoutError carries no message, so %s logged it as
                # a blank and made a timeout indistinguishable from a
                # successful call with an empty error field.
                logger.warning(
                    "NIMClient call failed: model=%s, error=%r",
                    model,
                    exc,
                )
                raise
            else:
                logger.debug(
                    "NIMClient call succeeded: model=%s, response_id=%s",
                    model,
                    getattr(result, "id", "unknown"),
                )
                return result

    _client.chat.completions.create = wrapped_create  # type: ignore[assignment]
    return _client


__all__ = ["get_nim_client"]