"""Shared plumbing for literature sources: protocol, rate limiting, resilient HTTP."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Protocol

import httpx

from aetheris.core.errors import TransientError
from aetheris.core.models import Paper

USER_AGENT = "Aetheris/0.1 (+https://github.com/anargadirli/aetheris; research engine)"


class Source(Protocol):
    name: str

    async def search(self, query: str, *, limit: int) -> list[Paper]: ...


class RateLimiter:
    """Enforces a minimum interval between request *starts* (shared across tasks)."""

    def __init__(
        self,
        min_interval_s: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._interval = min_interval_s
        self._clock = clock
        self._sleep = sleep
        self._next_slot = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = self._clock()
            wait = self._next_slot - now
            if wait > 0:
                await self._sleep(wait)
                now = self._clock()
            self._next_slot = max(now, self._next_slot) + self._interval


_RETRYABLE = frozenset({429, 500, 502, 503, 504})


async def get_with_backoff(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    limiter: RateLimiter | None = None,
    attempts: int = 3,
    base_delay: float = 1.5,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> httpx.Response:
    """GET with exponential backoff on 429/5xx/network errors; honours ``Retry-After``."""
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        if limiter is not None:
            await limiter.acquire()
        try:
            response = await client.get(url, params=params)
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            last = exc
            delay = base_delay * 2 ** (attempt - 1)
        else:
            if response.status_code not in _RETRYABLE:
                response.raise_for_status()
                return response
            last = TransientError(f"{url} -> HTTP {response.status_code}")
            retry_after = response.headers.get("retry-after", "")
            delay = (
                float(retry_after)
                if retry_after.replace(".", "", 1).isdigit()
                else base_delay * 2 ** (attempt - 1)
            )
        if attempt < attempts:
            await sleep(min(delay, 30.0) * random.uniform(0.9, 1.1))
    raise TransientError(f"GET {url} failed after {attempts} attempts: {last}") from last
