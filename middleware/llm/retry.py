# SPDX-License-Identifier: Apache-2.0
"""重试机制子模块。

从 middleware/llm/llm_client.py 中提取的独立组件。
职责：带指数退避的重试机制 + 异常解析。

导出：
    RetryConfig: 重试配置
    RetryStrategy: 错误重试判断
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from utils.logger import get_logger

logger = get_logger()

from .exceptions import (
    LLMException,
    LLMTimeoutError,
    LLMRateLimitError,
    LLMConnectionError,
    LLMAuthenticationError,
    LLMContentFilterError,
    LLMContextWindowError,
)

import litellm


@dataclass
class RetryConfig:
    """重试配置。"""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    retryable_exceptions: tuple = (
        LLMTimeoutError,
        LLMRateLimitError,
        LLMConnectionError,
    )


def parse_llm_exception(error: Exception, model: str) -> LLMException:
    """将任意异常解析为统一的 LLMException 类型。"""
    if isinstance(error, litellm.Timeout):
        return LLMTimeoutError(
            "模型响应超时，可能是网络问题或模型服务繁忙，请稍后重试", model=model
        )
    if isinstance(error, litellm.RateLimitError):
        retry_after = getattr(error, "retry_after", None)
        return LLMRateLimitError(
            "请求过于频繁，已达到速率限制，请稍后再试",
            model=model,
            retry_after=retry_after,
        )
    if isinstance(error, (litellm.APIConnectionError, ConnectionError, OSError)):
        return LLMConnectionError(
            "网络连接失败，请检查：1)网络连接 2)模型服务地址(base_url)是否正确",
            model=model,
        )
    if isinstance(error, litellm.AuthenticationError):
        return LLMAuthenticationError(
            "API Key 认证失败，请检查 API Key 是否正确", model=model
        )
    if isinstance(error, litellm.ContentPolicyViolationError):
        return LLMContentFilterError(
            "内容包含敏感信息被过滤，请修改输入内容后重试", model=model
        )
    if isinstance(error, litellm.BadRequestError):
        error_msg = str(error)
        if "LLM Provider NOT provided" in error_msg:
            friendly_msg = (
                f"模型配置错误: {model}\n\n"
                "请使用正确的 provider/model 格式，例如:\n"
                "- openai/gpt-4\n"
                "- anthropic/claude-3-opus\n"
                "- openrouter/claude-3.5-sonnet\n"
                "- ollama_chat/llama3\n\n"
                "查看所有支持的 provider: https://docs.litellm.ai/docs/providers"
            )
            return LLMException(friendly_msg, model=model, retryable=False)
        return LLMException(
            f"请求参数错误: {error_msg}", model=model, retryable=False
        )

    error_str = str(error).lower()
    if "timeout" in error_str:
        return LLMTimeoutError(
            "模型响应超时，可能是网络问题或模型服务繁忙，请稍后重试", model=model
        )
    if "rate limit" in error_str or "429" in error_str:
        return LLMRateLimitError(
            "请求过于频繁，已达到速率限制，请稍后再试",
            model=model,
        )
    if "connection" in error_str or "network" in error_str:
        return LLMConnectionError(
            "网络连接失败，请检查：1)网络连接 2)模型服务地址(base_url)是否正确",
            model=model,
        )
    if "authentication" in error_str or "401" in error_str or "403" in error_str:
        return LLMAuthenticationError(
            "API Key 认证失败，请检查 API Key 是否正确", model=model
        )
    if "content filter" in error_str or "moderation" in error_str:
        return LLMContentFilterError(
            "内容包含敏感信息被过滤，请修改输入内容后重试", model=model
        )

    if (
        "context" in error_str
        and ("window" in error_str or "length" in error_str or "limit" in error_str)
    ) or "contextwindowexceeded" in error_str.replace(" ", ""):
        return LLMContextWindowError(
            "对话长度超过模型上下文限制，请新建会话或清除历史对话", model=model
        )
    return LLMException(str(error), model=model, retryable=True)


async def retry_with_backoff(
    func,
    *args,
    config: RetryConfig | None = None,
    model: str = "",
    **kwargs,
) -> Any:
    """带指数退避的重试调用。

    Args:
        func: 要重试的异步函数
        config: 重试配置（None 时使用默认值）
        model: 模型名称（用于错误消息）
        **kwargs: 传递给 func 的额外参数
    """
    config = config or RetryConfig()
    last_exception = None

    for attempt in range(config.max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            llm_error = parse_llm_exception(e, model)
            last_exception = llm_error

            if not llm_error.retryable or attempt >= config.max_retries:
                raise llm_error

            delay = min(
                config.base_delay * (config.exponential_base ** attempt),
                config.max_delay,
            )

            if isinstance(llm_error, LLMRateLimitError) and llm_error.retry_after:
                delay = max(delay, llm_error.retry_after)

            logger.warning(
                f"LLM call failed (attempt {attempt + 1}/{config.max_retries + 1}), "
                f"retrying in {delay:.1f}s: {llm_error.message}"
            )
            await asyncio.sleep(delay)

    raise last_exception


__all__ = ["RetryConfig", "parse_llm_exception", "retry_with_backoff"]
