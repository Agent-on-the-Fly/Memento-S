"""
middleware.llm — 统一 LLM 调用层

基于 litellm 的异步封装，支持：
- 统一配置管理（通过 ConfigManager）
- 自动重试机制
- 超时控制
- 熔断保护
- 流式/非流式调用
"""

from .client import LLMClient
from .schema import LLMResponse, LLMStreamChunk, ToolCall, ContentBlock
from .exceptions import (
    LLMException,
    LLMTimeoutError,
    LLMRateLimitError,
    LLMConnectionError,
)

__all__ = [
    "LLMClient",
    "LLMResponse",
    "LLMStreamChunk",
    "ToolCall",
    "ContentBlock",
    "LLMException",
    "LLMTimeoutError",
    "LLMRateLimitError",
    "LLMConnectionError",
]
