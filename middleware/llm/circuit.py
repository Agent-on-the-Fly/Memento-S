# SPDX-License-Identifier: Apache-2.0
"""熔断器（Circuit Breaker）子模块。

从 middleware/llm/llm_client.py 中提取的独立组件。
职责：防止 LLM 服务因持续故障被反复调用。

导出：
    CircuitBreaker: 熔断器实现
    CircuitBreakerState: 熔断器状态枚举
    CircuitBreakerConfig: 熔断器配置
"""

from __future__ import annotations

import asyncio
import time
from enum import StrEnum

from utils.logger import get_logger

logger = get_logger()

from .exceptions import LLMConnectionError


class CircuitBreakerState(StrEnum):
    """熔断器状态。"""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half-open"


class CircuitBreakerConfig:
    """熔断器配置。"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls


class CircuitBreaker:
    """简单熔断器实现。

    三状态模型：
    - CLOSED: 正常，允许所有调用
    - OPEN: 故障，拒绝所有调用，等待 recovery_timeout 后进入 HALF_OPEN
    - HALF_OPEN: 恢复试探，允许有限次调用，全部成功则回到 CLOSED
    """

    def __init__(self, config: CircuitBreakerConfig | None = None):
        self.config = config or CircuitBreakerConfig()
        self.failures = 0
        self.last_failure_time: float | None = None
        self.state = CircuitBreakerState.CLOSED
        self.half_open_calls = 0
        self._lock = asyncio.Lock()

    async def call(self, func, *args, **kwargs):
        """在熔断器保护下执行函数。"""
        async with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if (
                    time.time() - (self.last_failure_time or 0)
                    > self.config.recovery_timeout
                ):
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.half_open_calls = 0
                    logger.info("Circuit breaker entering half-open state")
                else:
                    raise LLMConnectionError(
                        "模型服务暂时不可用（已连续多次调用失败）。\n"
                        "请稍后重试，或检查模型配置是否正确。"
                    )

            if (
                self.state == CircuitBreakerState.HALF_OPEN
                and self.half_open_calls >= self.config.half_open_max_calls
            ):
                raise LLMConnectionError("模型服务恢复尝试失败次数过多，请稍后重试。")

            if self.state == CircuitBreakerState.HALF_OPEN:
                self.half_open_calls += 1

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure(e)
            raise

    async def _on_success(self):
        async with self._lock:
            self.failures = 0
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
                self.half_open_calls = 0
                logger.info("Circuit breaker closed")

    async def _on_failure(self, error: Exception | None = None):
        from .exceptions import LLMException

        if error and isinstance(error, LLMException) and not error.retryable:
            return
        async with self._lock:
            self.failures += 1
            self.last_failure_time = time.time()

            if self.failures >= self.config.failure_threshold:
                if self.state != CircuitBreakerState.OPEN:
                    self.state = CircuitBreakerState.OPEN
                    logger.warning(
                        f"Circuit breaker opened after {self.failures} failures"
                    )


__all__ = ["CircuitBreaker", "CircuitBreakerState", "CircuitBreakerConfig"]
