"""
LLM 统一数据结构和响应定义。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Literal, Optional


class ContentType(Enum):
    """内容块类型。"""

    TEXT = "text"
    TOOL_CALL = "tool_call"
    IMAGE = "image"


@dataclass
class ContentBlock:
    """统一的内容块。"""

    type: ContentType
    content: str | None = None
    tool_call: Optional["ToolCall"] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    """解析后的单个 tool call。"""

    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式。"""
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
        }


@dataclass
class LLMResponse:
    """
    LLM 返回的结构化响应。

    支持多种内容类型：
    - 纯文本内容
    - Tool calls
    - 混合内容
    """

    content: str | None = None
    content_blocks: list[ContentBlock] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    model: str | None = None
    finish_reason: str | None = None
    raw_response: Any = None

    @property
    def has_tool_calls(self) -> bool:
        """是否包含 tool calls。"""
        return len(self.tool_calls) > 0

    @property
    def text(self) -> str:
        """获取纯文本内容（方便访问）。"""
        return self.content or ""


@dataclass
class LLMStreamChunk:
    """
    流式响应的单个块。

    Attributes:
        delta_content: 增量文本内容
        delta_tool_call: 增量 tool call
        finish_reason: 结束原因（如 "stop", "tool_calls", "length" 等）
        is_finished: 是否结束
        usage: Token 使用情况（通常在最后一个 chunk 中返回）
    """

    delta_content: str | None = None
    delta_tool_call: Optional[ToolCall] = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None

    @property
    def is_finished(self) -> bool:
        """是否已结束。"""
        return self.finish_reason is not None

    @property
    def total_tokens(self) -> int | None:
        """获取总 token 数（如果 usage 可用）。"""
        if self.usage and "total_tokens" in self.usage:
            return self.usage["total_tokens"]
        return None


# 类型别名
LLMResponseGenerator = AsyncGenerator[LLMStreamChunk, None]
MessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    """标准消息格式。"""

    role: MessageRole
    content: str | list[dict[str, Any]]
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为 OpenAI 格式。"""
        msg: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            msg["name"] = self.name
        if self.tool_calls:
            msg["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        return msg
