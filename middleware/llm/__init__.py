# SPDX-License-Identifier: Apache-2.0
"""
middleware.llm — 统一 LLM 调用层

基于 litellm 的异步封装，支持：
- 统一配置管理（通过 ConfigManager）
- 自动重试机制（retry 模块）
- 超时控制
- 熔断保护（circuit 模块）
- 流式/非流式调用
- Embedding API

子模块结构：
  llm_client.py  — LLMClient 主类（Facade ~700行）
  retry.py       — 重试机制 + 异常解析
  circuit.py     — 熔断器实现
  schema.py      — 类型定义（LLMResponse, ToolCall 等）
  exceptions.py  — 异常定义
  embedding_client.py — Embedding 客户端
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
# 子模块重导出（保持向后兼容）
from .retry import RetryConfig, parse_llm_exception
from .circuit import CircuitBreaker, CircuitBreakerConfig

__all__ = [
    # 主类
    "LLMClient",
    "EmbeddingClient",
    "EmbeddingClientConfig",
    # 配置
    "RetryConfig",
    "CircuitBreakerConfig",
    "RawTokenConfig",
    # 工具函数
    "parse_llm_exception",
    "chat_completions",
    "chat_completions_async",
    # 类型
    "FINISH_CONTENT_FILTER",
    "FINISH_LENGTH",
    "FINISH_STOP",
    "FINISH_TOOL_CALLS",
    "LLMResponse",
    "LLMStreamChunk",
    "ToolCall",
    "ContentBlock",
    # 异常
    "LLMException",
    "LLMTimeoutError",
    "LLMRateLimitError",
    "LLMConnectionError",
    # 子模块（向后兼容）
    "CircuitBreaker",
]
