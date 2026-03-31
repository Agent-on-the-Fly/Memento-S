"""
Config Cache with retry logic for Weixin SDK.

Provides intelligent caching for bot configuration with automatic refresh
and exponential backoff retry on failures.

Example usage:
    from weixin_sdk.api.config_cache import ConfigCache

    cache = ConfigCache(
        refresh_interval_seconds=86400,
        min_retry_delay_seconds=2,
        max_retry_delay_seconds=3600,
    )

    # This will return cached config if available and not expired
    # Otherwise fetches fresh config with retry logic
    config = await cache.get_config(lambda: client._fetch_config())
"""

import asyncio
import logging
import random
from dataclasses import dataclass, field
from typing import Callable, Optional, TypeVar, Generic, Awaitable, Coroutine, Any
from datetime import datetime, timedelta

from ..exceptions import WeixinAPIError, WeixinSessionExpiredError

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CachedConfig(Generic[T]):
    """Internal container for cached configuration."""

    data: T
    fetched_at: datetime = field(default_factory=datetime.utcnow)
    fetch_attempts: int = 0


class ConfigCache:
    """
    Thread-safe configuration cache with automatic refresh and retry logic.

    Features:
    - Cache config for configurable duration (default 24 hours)
    - Auto-refresh when expired
    - Exponential backoff on fetch failures (2s → 1 hour)
    - Success resets retry delay to minimum
    - Randomized refresh within window to prevent thundering herd
    - Thread-safe using asyncio.Lock
    - Returns stale cache if fetch fails (if stale cache is available)

    Args:
        refresh_interval_seconds: How long to cache config (default 86400 = 24 hours)
        min_retry_delay_seconds: Initial retry delay on failure (default 2)
        max_retry_delay_seconds: Maximum retry delay cap (default 3600 = 1 hour)
        exponential_base: Base for exponential backoff calculation (default 2.0)
        jitter: Add random jitter to retry delays (default True)
        allow_stale_on_error: Return stale cache if fetch fails (default True)
    """

    def __init__(
        self,
        refresh_interval_seconds: int = 86400,  # 24 hours
        min_retry_delay_seconds: int = 2,
        max_retry_delay_seconds: int = 3600,  # 1 hour
        exponential_base: float = 2.0,
        jitter: bool = True,
        allow_stale_on_error: bool = True,
    ):
        self._refresh_interval = timedelta(seconds=refresh_interval_seconds)
        self._min_retry_delay = min_retry_delay_seconds
        self._max_retry_delay = max_retry_delay_seconds
        self._exponential_base = exponential_base
        self._jitter = jitter
        self._allow_stale_on_error = allow_stale_on_error

        # Cache state
        self._cached_config: Optional[CachedConfig] = None
        self._current_retry_delay = min_retry_delay_seconds
        self._consecutive_failures = 0
        self._last_fetch_attempt: Optional[datetime] = None

        # Thread safety
        self._lock = asyncio.Lock()

    def is_expired(self) -> bool:
        """
        Check if the cached configuration is expired.

        Returns:
            True if cache is expired or no config is cached, False otherwise.
        """
        if self._cached_config is None:
            return True

        age = datetime.utcnow() - self._cached_config.fetched_at

        # Add jitter to prevent thundering herd: refresh up to 10% early
        jitter_factor = random.uniform(0.9, 1.0) if self._jitter else 1.0
        effective_interval = self._refresh_interval * jitter_factor

        return age > effective_interval

    def get_retry_delay(self) -> float:
        """
        Get current retry delay with exponential backoff.

        Returns:
            Current retry delay in seconds.
        """
        delay = self._current_retry_delay

        if self._jitter:
            # Add ±20% jitter
            delay = delay * random.uniform(0.8, 1.2)

        return delay

    def invalidate(self) -> None:
        """Clear the cache. Safe to call from any context."""
        self._cached_config = None
        self._current_retry_delay = self._min_retry_delay
        self._consecutive_failures = 0
        logger.debug("Config cache invalidated")

    async def get_config(self, fetch_func: Callable[[], Coroutine[Any, Any, T]]) -> T:
        """
        Get configuration, using cache if available and not expired.

        If cache is expired or not present, fetches fresh config with
        exponential backoff retry on failures. If fetch fails and stale
        cache is available, returns the stale cache (if configured).

        Args:
            fetch_func: Async callable that fetches fresh configuration.
                       Should return the config data (e.g., GetConfigResp).

        Returns:
            The configuration (cached or freshly fetched).

        Raises:
            WeixinAPIError: If all fetch attempts fail and no stale cache.
            Exception: Re-raises any exception from fetch_func after max retries.
        """
        async with self._lock:
            # Check if we can use cached config
            if self._cached_config is not None and not self.is_expired():
                logger.debug("Returning cached config (not expired)")
                return self._cached_config.data

            # Need to fetch fresh config
            try:
                config = await self._fetch_with_retry(fetch_func)
                self._on_fetch_success(config)
                return config

            except Exception as e:
                # If fetch failed but we have stale cache, return it
                if self._allow_stale_on_error and self._cached_config is not None:
                    logger.warning(f"Fetch failed, returning stale cache. Error: {e}")
                    return self._cached_config.data

                # No stale cache available, raise the error
                logger.error(f"Fetch failed and no stale cache available: {e}")
                raise

    async def refresh_config(
        self, fetch_func: Callable[[], Coroutine[Any, Any, T]]
    ) -> T:
        """
        Force refresh configuration, ignoring cache state.

        Args:
            fetch_func: Async callable that fetches fresh configuration.

        Returns:
            The freshly fetched configuration.

        Raises:
            Exception: Re-raises any exception from fetch_func after max retries.
        """
        async with self._lock:
            try:
                config = await self._fetch_with_retry(fetch_func)
                self._on_fetch_success(config)
                return config

            except Exception as e:
                # Even on forced refresh, try to return stale cache if allowed
                if self._allow_stale_on_error and self._cached_config is not None:
                    logger.warning(
                        f"Force refresh failed, returning stale cache. Error: {e}"
                    )
                    return self._cached_config.data
                raise

    async def _fetch_with_retry(
        self, fetch_func: Callable[[], Coroutine[Any, Any, T]]
    ) -> T:
        """
        Fetch configuration with exponential backoff retry.

        Args:
            fetch_func: Async callable that fetches configuration.

        Returns:
            The fetched configuration.

        Raises:
            Exception: Last exception encountered after all retries.
        """
        self._last_fetch_attempt = datetime.utcnow()

        while True:
            try:
                self._cached_config = None  # Clear while fetching
                config = await fetch_func()
                return config

            except (WeixinAPIError, WeixinSessionExpiredError) as e:
                # Don't retry session errors immediately
                if isinstance(e, WeixinSessionExpiredError):
                    logger.error("Session expired during config fetch")
                    raise

                # Calculate next retry delay
                self._on_fetch_failure()
                delay = self.get_retry_delay()

                logger.warning(
                    f"Config fetch failed (attempt {self._consecutive_failures}): {e}. "
                    f"Retrying in {delay:.2f}s"
                )

                # Check if we've exceeded max delay (indicates we should stop retrying)
                if self._current_retry_delay >= self._max_retry_delay:
                    logger.error(
                        f"Max retry delay reached ({self._max_retry_delay}s), "
                        f"giving up after {self._consecutive_failures} attempts"
                    )
                    raise

                await asyncio.sleep(delay)

            except Exception as e:
                # For other exceptions, also apply backoff but log differently
                self._on_fetch_failure()
                delay = self.get_retry_delay()

                logger.warning(
                    f"Config fetch failed with unexpected error: {e}. "
                    f"Retrying in {delay:.2f}s"
                )

                if self._current_retry_delay >= self._max_retry_delay:
                    logger.error(
                        f"Max retry delay reached ({self._max_retry_delay}s), "
                        f"giving up after {self._consecutive_failures} attempts"
                    )
                    raise

                await asyncio.sleep(delay)

    def _on_fetch_success(self, config: T) -> None:
        """Handle successful config fetch."""
        self._cached_config = CachedConfig(data=config)
        self._current_retry_delay = self._min_retry_delay
        self._consecutive_failures = 0
        logger.debug(
            f"Config fetched successfully, cached at {self._cached_config.fetched_at}"
        )

    def _on_fetch_failure(self) -> None:
        """Handle failed config fetch - apply exponential backoff."""
        self._consecutive_failures += 1

        # Calculate next delay with exponential backoff
        next_delay = self._current_retry_delay * self._exponential_base
        self._current_retry_delay = min(next_delay, self._max_retry_delay)

        logger.debug(
            f"Fetch failure #{self._consecutive_failures}, "
            f"next retry delay: {self._current_retry_delay:.2f}s"
        )

    @property
    def cached_at(self) -> Optional[datetime]:
        """Get timestamp when config was last cached, or None if not cached."""
        if self._cached_config is None:
            return None
        return self._cached_config.fetched_at

    @property
    def is_fresh(self) -> bool:
        """Check if cache exists and is not expired."""
        return self._cached_config is not None and not self.is_expired()

    @property
    def failure_count(self) -> int:
        """Get number of consecutive fetch failures."""
        return self._consecutive_failures
