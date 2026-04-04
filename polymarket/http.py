import asyncio
import logging
import time

import httpx

logger = logging.getLogger(__name__)


class RateLimiter:
    """Async token-bucket rate limiter."""

    def __init__(self, max_tokens: int, per_seconds: float):
        self.max_tokens = max_tokens
        self.per_seconds = per_seconds
        self.tokens = float(max_tokens)
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            rate = self.max_tokens / self.per_seconds
            self.tokens = min(self.max_tokens, self.tokens + elapsed * rate)
            self.last_refill = now

            if self.tokens < 1:
                wait = (1 - self.tokens) / rate
                await asyncio.sleep(wait)
                self.tokens = 0
            else:
                self.tokens -= 1


# Conservative limits (85% of actual)
gamma_limiter = RateLimiter(max_tokens=280, per_seconds=10)
clob_limiter = RateLimiter(max_tokens=1400, per_seconds=10)


async def request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    limiter: RateLimiter,
    *,
    max_retries: int = 3,
    **kwargs,
) -> dict | list:
    for attempt in range(max_retries + 1):
        await limiter.acquire()
        try:
            response = await client.request(method, url, **kwargs)
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            if attempt == max_retries:
                raise
            wait = 2**attempt
            logger.warning("Connection error on %s %s (attempt %d): %s, retrying in %ss", method, url, attempt + 1, e, wait)
            await asyncio.sleep(wait)
            continue

        if response.status_code == 429:
            wait = float(response.headers.get("Retry-After", 2**attempt))
            logger.warning("Rate limited on %s %s, waiting %ss", method, url, wait)
            await asyncio.sleep(wait)
            continue

        if response.status_code >= 500:
            if attempt == max_retries:
                response.raise_for_status()
            wait = 2**attempt
            logger.warning("Server error %d on %s %s, retrying in %ss", response.status_code, method, url, wait)
            await asyncio.sleep(wait)
            continue

        response.raise_for_status()
        return response.json()

    raise httpx.HTTPStatusError(
        f"Failed after {max_retries} retries",
        request=response.request,
        response=response,
    )
