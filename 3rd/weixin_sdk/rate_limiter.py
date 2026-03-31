"""
Rate limiting module for Weixin SDK.

Provides token bucket and sliding window rate limiters with both sync and async interfaces.
"""

import asyncio
import functools
import threading
import time
from collections import defaultdict
from typing import Callable, Dict, Optional, Any


class TokenBucket:
    """Token bucket rate limiter implementation."""

    def __init__(self, rate: float, capacity: int):
        """
        Initialize token bucket.

        Args:
            rate: Tokens added per second.
            capacity: Maximum number of tokens in the bucket.
        """
        self.rate = rate
        self.capacity = capacity
        self._buckets: Dict[Optional[str], Dict[str, float]] = defaultdict(
            lambda: {"tokens": float(capacity), "last_update": time.monotonic()}
        )
        self._lock = threading.Lock()

    def _refill(self, bucket: Dict[str, float]) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - bucket["last_update"]
        bucket["tokens"] = min(self.capacity, bucket["tokens"] + elapsed * self.rate)
        bucket["last_update"] = now

    def acquire(self, key: Optional[str] = None) -> None:
        """
        Acquire a token, blocking until available.

        Args:
            key: Optional key for separate rate limit buckets.
        """
        while not self.try_acquire(key):
            time.sleep(0.01)

    def try_acquire(self, key: Optional[str] = None) -> bool:
        """
        Try to acquire a token without blocking.

        Args:
            key: Optional key for separate rate limit buckets.

        Returns:
            True if token acquired, False otherwise.
        """
        with self._lock:
            bucket = self._buckets[key]
            self._refill(bucket)
            if bucket["tokens"] >= 1:
                bucket["tokens"] -= 1
                return True
            return False

    async def async_acquire(self, key: Optional[str] = None) -> None:
        """
        Async acquire a token, blocking until available.

        Args:
            key: Optional key for separate rate limit buckets.
        """
        while not await self.async_try_acquire(key):
            await asyncio.sleep(0.01)

    async def async_try_acquire(self, key: Optional[str] = None) -> bool:
        """
        Async try to acquire a token without blocking.

        Args:
            key: Optional key for separate rate limit buckets.

        Returns:
            True if token acquired, False otherwise.
        """
        bucket = self._buckets[key]
        now = time.monotonic()
        elapsed = now - bucket["last_update"]
        bucket["tokens"] = min(self.capacity, bucket["tokens"] + elapsed * self.rate)
        bucket["last_update"] = now

        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True
        return False


class SlidingWindow:
    """Sliding window rate limiter implementation."""

    def __init__(self, limit: int, window_seconds: float):
        """
        Initialize sliding window rate limiter.

        Args:
            limit: Maximum number of requests allowed in the window.
            window_seconds: Time window in seconds.
        """
        self.limit = limit
        self.window_seconds = window_seconds
        self._windows: Dict[Optional[str], Dict[str, Any]] = defaultdict(
            lambda: {"requests": [], "lock": threading.Lock()}
        )
        self._async_locks: Dict[Optional[str], asyncio.Lock] = defaultdict(asyncio.Lock)

    def _clean_old_requests(self, window: Dict[str, Any]) -> None:
        """Remove requests outside the sliding window."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        window["requests"] = [t for t in window["requests"] if t > cutoff]

    def acquire(self, key: Optional[str] = None) -> None:
        """
        Acquire a slot, blocking until available.

        Args:
            key: Optional key for separate rate limit buckets.
        """
        while not self.try_acquire(key):
            time.sleep(0.01)

    def try_acquire(self, key: Optional[str] = None) -> bool:
        """
        Try to acquire a slot without blocking.

        Args:
            key: Optional key for separate rate limit buckets.

        Returns:
            True if slot acquired, False otherwise.
        """
        window = self._windows[key]
        with window["lock"]:
            self._clean_old_requests(window)
            if len(window["requests"]) < self.limit:
                window["requests"].append(time.monotonic())
                return True
            return False

    async def async_acquire(self, key: Optional[str] = None) -> None:
        """
        Async acquire a slot, blocking until available.

        Args:
            key: Optional key for separate rate limit buckets.
        """
        while not await self.async_try_acquire(key):
            await asyncio.sleep(0.01)

    async def async_try_acquire(self, key: Optional[str] = None) -> bool:
        """
        Async try to acquire a slot without blocking.

        Args:
            key: Optional key for separate rate limit buckets.

        Returns:
            True if slot acquired, False otherwise.
        """
        window = self._windows[key]
        async with self._async_locks[key]:
            now = time.monotonic()
            cutoff = now - self.window_seconds
            window["requests"] = [t for t in window["requests"] if t > cutoff]
            if len(window["requests"]) < self.limit:
                window["requests"].append(now)
                return True
            return False


class RateLimiter:
    """Unified rate limiter interface supporting both TokenBucket and SlidingWindow."""

    def __init__(
        self,
        limit: Optional[int] = None,
        window_seconds: Optional[float] = None,
        rate: Optional[float] = None,
        capacity: Optional[int] = None,
        strategy: str = "token_bucket",
    ):
        """
        Initialize rate limiter.

        Args:
            limit: Maximum requests allowed (for sliding window).
            window_seconds: Time window in seconds (for sliding window).
            rate: Tokens per second (for token bucket).
            capacity: Bucket capacity (for token bucket).
            strategy: "token_bucket" or "sliding_window".
        """
        if strategy == "token_bucket":
            if rate is None or capacity is None:
                raise ValueError("token_bucket requires rate and capacity")
            self._limiter = TokenBucket(rate, capacity)
            self._strategy = "token_bucket"
        elif strategy == "sliding_window":
            if limit is None or window_seconds is None:
                raise ValueError("sliding_window requires limit and window_seconds")
            self._limiter = SlidingWindow(limit, window_seconds)
            self._strategy = "sliding_window"
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def acquire(self, key: Optional[str] = None) -> None:
        """Acquire a slot/token, blocking until available."""
        self._limiter.acquire(key)

    def try_acquire(self, key: Optional[str] = None) -> bool:
        """Try to acquire a slot/token without blocking."""
        return self._limiter.try_acquire(key)

    async def async_acquire(self, key: Optional[str] = None) -> None:
        """Async acquire a slot/token, blocking until available."""
        await self._limiter.async_acquire(key)

    async def async_try_acquire(self, key: Optional[str] = None) -> bool:
        """Async try to acquire a slot/token without blocking."""
        return await self._limiter.async_try_acquire(key)


def rate_limit(
    limit: int,
    window_seconds: float,
    key_func: Optional[Callable[..., Any]] = None,
    strategy: str = "sliding_window",
):
    """
    Decorator to apply rate limiting to a function.

    Args:
        limit: Maximum calls allowed in the window.
        window_seconds: Time window in seconds.
        key_func: Optional function to generate rate limit key from args/kwargs.
        strategy: "token_bucket" or "sliding_window".

    Returns:
        Decorated function with rate limiting applied.
    """
    limiter = RateLimiter(
        limit=limit,
        window_seconds=window_seconds,
        strategy=strategy,
    )

    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                key = key_func(*args, **kwargs) if key_func else None
                await limiter.async_acquire(key)
                return await func(*args, **kwargs)
        else:

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                key = key_func(*args, **kwargs) if key_func else None
                limiter.acquire(key)
                return func(*args, **kwargs)

        return wrapper

    return decorator
