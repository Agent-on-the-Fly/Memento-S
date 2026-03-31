"""
middleware.llm — 统一 LLM 调用层

基于 litellm 的异步封装，支持：
- 统一配置管理（通过 ConfigManager）
- 自动重试机制
- 超时控制
- 熔断保护
- 流式/非流式调用
- Embedding API
"""

from .llm_client import LLMClient, RawTokenConfig, chat_completions, chat_completions_async
from .embedding_client import EmbeddingClient, EmbeddingClientConfig
from .schema import (
    FINISH_CONTENT_FILTER,
    FINISH_LENGTH,
    FINISH_STOP,
    FINISH_TOOL_CALLS,
    LLMResponse,
    LLMStreamChunk,
    ToolCall,
    ContentBlock,
)
from .exceptions import (
    LLMException,
    LLMTimeoutError,
    LLMRateLimitError,
    LLMConnectionError,
)

__all__ = [
    "FINISH_CONTENT_FILTER",
    "FINISH_LENGTH",
    "FINISH_STOP",
    "FINISH_TOOL_CALLS",
    "LLMClient",
    "RawTokenConfig",
    "chat_completions",
    "chat_completions_async",
    "EmbeddingClient",
    "EmbeddingClientConfig",
    "LLMResponse",
    "LLMStreamChunk",
    "ToolCall",
    "ContentBlock",
    "LLMException",
    "LLMTimeoutError",
    "LLMRateLimitError",
    "LLMConnectionError",
]
