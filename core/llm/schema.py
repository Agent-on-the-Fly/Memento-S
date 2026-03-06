
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from litellm.types.utils import ModelResponse


@dataclass
class ToolCall:

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: ModelResponse | None = None


@dataclass
class LLMStreamChunk:

    delta_content: str | None = None
    tool_calls_delta: list[Any] | None = None
    finish_reason: str | None = None
