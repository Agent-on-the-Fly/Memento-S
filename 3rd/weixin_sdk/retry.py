"""
Retry utilities with exponential backoff and circuit breaker pattern.

Provides:
- RetryConfig: Configuration for retry behavior
- retry: Decorator/function for retrying operations
- CircuitBreaker: Circuit breaker pattern implementation
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Set, Type, TypeVar, Any, Optional

from .exceptions import WeixinTimeoutError, WeixinAPIError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests allowed
    OPEN = "open"  # Failing, requests rejected immediately
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RetryConfig:
    """
    Configuration for retry behavior.

    Attributes:
        max_attempts: Maximum number of retry attempts (default: 3)
        base_delay: Base delay in seconds for exponential backoff (default: 1.0)
        max_delay: Maximum delay cap in seconds (default: 30.0)
        retryable_exceptions: Tuple of exception types that should trigger retry
        exponential_base: Base for exponential calculation (default: 2)
        jitter: Add random jitter to delays (default: True)
    """

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    retryable_exceptions: tuple = (WeixinTimeoutError, WeixinAPIError)
    exponential_base: float = 2.0
    jitter: bool = True

    def is_retryable(self, exception: Exception) -> bool:
        """Check if an exception is retryable."""
        # Check if exception is a WeixinAPIError with 5xx status
        if isinstance(exception, WeixinAPIError) and exception.code:
            if 500 <= exception.code < 600:
                return True

        # Check if exception is in retryable exceptions
        return isinstance(exception, self.retryable_exceptions)

    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for the given attempt using exponential backoff.

        Formula: min(base_delay * (exponential_base ** attempt), max_delay)
        With optional jitter: delay * (0.5 + random(0, 1))
        """
        import random

        delay = self.base_delay * (self.exponential_base**attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            delay = delay * (0.5 + random.random())

        return delay


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    Prevents cascading failures by tracking failures and opening the circuit
    after a threshold is reached. The circuit will:
    - OPEN: Reject requests immediately after threshold failures
    - HALF_OPEN: Allow test requests after timeout
    - CLOSE: Resume normal operation on success

    Attributes:
        failure_threshold: Number of failures to open the circuit (default: 5)
        recovery_timeout: Seconds to wait before half-open (default: 60)
        half_open_max_calls: Max test calls in half-open state (default: 3)
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for recovery timeout."""
        if self._state == CircuitState.OPEN:
            if (
                self._last_failure_time
                and (time.monotonic() - self._last_failure_time)
                >= self.recovery_timeout
            ):
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                logger.info("Circuit breaker transitioning to HALF_OPEN")
        return self._state

    def can_execute(self) -> bool:
        """Check if a request can be executed."""
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.OPEN:
            return False
        # HALF_OPEN - allow limited calls
        return self._half_open_calls < self.half_open_max_calls

    def record_success(self):
        """Record a successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1
            if self._half_open_calls >= self.half_open_max_calls:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                logger.info("Circuit breaker CLOSED after successful recovery")
        elif self._state == CircuitState.CLOSED:
            self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self):
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            logger.warning("Circuit breaker OPEN after half-open failure")
        elif self._state == CircuitState.CLOSED:
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker OPEN after {self._failure_count} failures"
                )

    def reset(self):
        """Reset the circuit breaker to initial state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple = (WeixinTimeoutError, WeixinAPIError),
    exponential_base: float = 2.0,
    jitter: bool = True,
    circuit_breaker: Optional[CircuitBreaker] = None,
):
    """
    Decorator that retries a function with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts (default: 3)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay cap in seconds (default: 30.0)
        exceptions: Tuple of exceptions to retry on (default: (WeixinTimeoutError, WeixinAPIError))
        exponential_base: Base for exponential backoff (default: 2.0)
        jitter: Add random jitter to delays (default: True)
        circuit_breaker: Optional circuit breaker to use

    Returns:
        Decorated function with retry logic

    Example:
        @retry(max_attempts=3, base_delay=1.0)
        async def fetch_data():
            return await api_call()
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        retryable_exceptions=exceptions,
        exponential_base=exponential_base,
        jitter=jitter,
    )

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Check circuit breaker
            if circuit_breaker and not circuit_breaker.can_execute():
                raise WeixinAPIError("Circuit breaker is OPEN, request rejected")

            last_exception: Optional[Exception] = None

            for attempt in range(max_attempts):
                try:
                    result = await func(*args, **kwargs)  # type: ignore
                    if circuit_breaker:
                        circuit_breaker.record_success()
                    return result

                except Exception as e:
                    last_exception = e

                    # Check if exception is retryable
                    if not config.is_retryable(e):
                        if circuit_breaker:
                            circuit_breaker.record_failure()
                        raise

                    # Check if we have retries left
                    if attempt < max_attempts - 1:
                        delay = config.calculate_delay(attempt)
                        logger.warning(
                            f"Retryable error in {func.__name__}: {e}. "
                            f"Retrying in {delay:.2f}s (attempt {attempt + 1}/{max_attempts})"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )

                    if circuit_breaker:
                        circuit_breaker.record_failure()

            # Re-raise the last exception if we exhausted retries
            if last_exception:
                raise last_exception
            raise RuntimeError("Retry logic error: no exception but no result")

        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Check circuit breaker
            if circuit_breaker and not circuit_breaker.can_execute():
                raise WeixinAPIError("Circuit breaker is OPEN, request rejected")

            last_exception: Optional[Exception] = None

            for attempt in range(max_attempts):
                try:
                    result = func(*args, **kwargs)
                    if circuit_breaker:
                        circuit_breaker.record_success()
                    return result

                except Exception as e:
                    last_exception = e

                    # Check if exception is retryable
                    if not config.is_retryable(e):
                        if circuit_breaker:
                            circuit_breaker.record_failure()
                        raise

                    # Check if we have retries left
                    if attempt < max_attempts - 1:
                        delay = config.calculate_delay(attempt)
                        logger.warning(
                            f"Retryable error in {func.__name__}: {e}. "
                            f"Retrying in {delay:.2f}s (attempt {attempt + 1}/{max_attempts})"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )

                    if circuit_breaker:
                        circuit_breaker.record_failure()

            if last_exception:
                raise last_exception
            raise RuntimeError("Retry logic error: no exception but no result")

        # Return appropriate wrapper based on whether func is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


async def retry_async(
    func: Callable[..., Any],
    *args: Any,
    config: Optional[RetryConfig] = None,
    circuit_breaker: Optional[CircuitBreaker] = None,
    **kwargs: Any,
) -> Any:
    """
    Retry an async function with the given config.

    Args:
        func: Async function to retry
        *args: Positional arguments for the function
        config: Retry configuration (uses defaults if not provided)
        circuit_breaker: Optional circuit breaker
        **kwargs: Keyword arguments for the function

    Returns:
        Result of the function call
    """
    if config is None:
        config = RetryConfig()

    # Check circuit breaker
    if circuit_breaker and not circuit_breaker.can_execute():
        raise WeixinAPIError("Circuit breaker is OPEN, request rejected")

    last_exception: Optional[Exception] = None

    for attempt in range(config.max_attempts):
        try:
            result = await func(*args, **kwargs)
            if circuit_breaker:
                circuit_breaker.record_success()
            return result

        except Exception as e:
            last_exception = e

            if not config.is_retryable(e):
                if circuit_breaker:
                    circuit_breaker.record_failure()
                raise

            if attempt < config.max_attempts - 1:
                delay = config.calculate_delay(attempt)
                logger.warning(
                    f"Retryable error in {func.__name__}: {e}. "
                    f"Retrying in {delay:.2f}s (attempt {attempt + 1}/{config.max_attempts})"
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"All {config.max_attempts} attempts failed for {func.__name__}: {e}"
                )

            if circuit_breaker:
                circuit_breaker.record_failure()

    if last_exception:
        raise last_exception
    raise RuntimeError("Retry logic error: no exception but no result")
