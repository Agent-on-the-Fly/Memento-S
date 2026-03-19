"""
LLM 相关异常定义。
"""

from __future__ import annotations


class LLMException(Exception):
    """LLM 调用基础异常。"""

    def __init__(
        self, message: str, *, model: str | None = None, retryable: bool = False
    ):
        super().__init__(message)
        self.message = message
        self.model = model
        self.retryable = retryable


class LLMTimeoutError(LLMException):
    """LLM 调用超时。"""

    def __init__(
        self, message: str = "LLM request timeout", *, model: str | None = None
    ):
        super().__init__(message, model=model, retryable=True)


class LLMRateLimitError(LLMException):
    """触发速率限制。"""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        *,
        model: str | None = None,
        retry_after: float | None = None,
    ):
        super().__init__(message, model=model, retryable=True)
        self.retry_after = retry_after


class LLMConnectionError(LLMException):
    """连接错误。"""

    def __init__(self, message: str = "Connection error", *, model: str | None = None):
        super().__init__(message, model=model, retryable=True)


class LLMAuthenticationError(LLMException):
    """认证错误（API Key 无效等）。"""

    def __init__(
        self, message: str = "Authentication failed", *, model: str | None = None
    ):
        super().__init__(message, model=model, retryable=False)


class LLMContentFilterError(LLMException):
    """内容被过滤。"""

    def __init__(self, message: str = "Content filtered", *, model: str | None = None):
        super().__init__(message, model=model, retryable=False)
