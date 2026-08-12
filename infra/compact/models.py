# SPDX-License-Identifier: Apache-2.0
"""数据类型定义 — 压缩模块的核心数据结构。

包括 CompactTrigger, CompactBudget, CompactContext, CompactResult, MessageGroup
以及 ToolResultReplacementState（跨 infra/context 和 infra/compact 共享）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class CompactTrigger(StrEnum):
    """压缩触发类型 — 标识压缩方式和严重程度。"""

    MICRO = "micro"
    """微压缩 (零 LLM) — 清除旧 tool result 内容。"""

    SM = "sm"
    """SM 压缩 (零 LLM) — 基于外部摘要截断消息。"""

    SINGLE = "single"
    """单消息压缩 (LLM) — 压缩单条超大消息。"""

    EMERGENCY = "emergency"
    """紧急压缩 (LLM) — 全量 9 段摘要。"""

    TRIM = "trim"
    """最后防线 — group_aware_trim，保证不丢消息。"""

    def is_zero_llm(self) -> bool:
        """是否为零 LLM 压缩方式。"""
        return self in (CompactTrigger.MICRO, CompactTrigger.SM, CompactTrigger.TRIM)

    def is_emergency(self) -> bool:
        """是否为紧急压缩。"""
        return self in (CompactTrigger.EMERGENCY, CompactTrigger.TRIM)


@dataclass
class ToolResultReplacementState:
    """跟踪已替换的 tool result，保证幂等替换。

    从 infra/context/providers/compactor.py 迁移到此处统一存放。
    """

    replacements: dict[str, str] = field(default_factory=dict)

    def mark_replaced(self, tool_call_id: str, preview: str) -> None:
        self.replacements[tool_call_id] = preview

    def get_replacement(self, tool_call_id: str) -> str | None:
        return self.replacements.get(tool_call_id)


@dataclass
class CompactBudget:
    """压缩预算 — 定义 token 上限和分布。"""

    total_tokens: int = 0
    """当前消息列表的实际 token 数。"""

    total_limit: int = 0
    """允许的最大 token 数 (input_budget)。"""

    per_message_limit: int = 0
    """单条消息的最大 token 数 (通常为 total_limit // 4)。"""

    @property
    def within_budget(self) -> bool:
        """当前 token 数是否在预算内。"""
        return self.total_tokens <= self.total_limit

    @property
    def overage(self) -> int:
        """超出预算的 token 数。"""
        return max(0, self.total_tokens - self.total_limit)

    def recalculate(
        self,
        total_tokens: int,
        total_limit: int | None = None,
        per_message_limit: int | None = None,
    ) -> "CompactBudget":
        """基于新值重新计算。"""
        return CompactBudget(
            total_tokens=total_tokens,
            total_limit=total_limit if total_limit is not None else self.total_limit,
            per_message_limit=(
                per_message_limit
                if per_message_limit is not None
                else self.per_message_limit
            ),
        )


@dataclass
class CompactResult:
    """压缩结果 — 包含压缩后的消息和元数据。"""

    messages: list[dict[str, Any]]
    """压缩后的消息列表。"""

    trigger: CompactTrigger
    """触发本次压缩的压缩类型。"""

    dropped_count: int = 0
    """丢弃的消息数量。"""

    summary_message: dict[str, Any] | None = None
    """新插入的摘要消息 (如果有)。"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """额外元数据 (saved_tokens, original_count 等)。"""

    @property
    def tokens_before(self) -> int:
        """原始 token 数 (从 metadata 恢复)。"""
        return self.metadata.get("tokens_before", 0)

    @property
    def tokens_after(self) -> int:
        """压缩后 token 数。"""
        return self.metadata.get("tokens_after", 0)

    @property
    def saved_tokens(self) -> int:
        """节省的 token 数。"""
        return self.metadata.get("saved_tokens", 0)


@dataclass
class MessageGroup:
    """API 轮次分组 — 保证 tool_use/tool_result 配对完整。"""

    round_id: int
    """轮次 ID。"""

    messages: list[dict[str, Any]]
    """该轮次内的所有消息。"""

    @property
    def total_tokens(self) -> int:
        """该组的消息总 token 数。"""
        return self.metadata.get("total_tokens", 0)

    @property
    def has_tool_call(self) -> bool:
        """是否包含 tool_call。"""
        return any(msg.get("tool_calls") for msg in self.messages)

    @property
    def metadata(self) -> dict[str, Any]:
        """组级别的元数据。"""
        return self._metadata

    @metadata.setter
    def metadata(self, value: dict[str, Any]) -> None:
        self._metadata = value

    _metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompactContext:
    """压缩执行上下文 — 策略间的数据传递。"""

    messages: list[dict[str, Any]]
    """当前消息列表 (可能被多个策略修改)。"""

    budget: CompactBudget
    """当前预算。"""

    original_count: int = 0
    """原始消息数量。"""

    saved_tokens: int = 0
    """累计节省的 token 数。"""

    state: dict[str, Any] = field(default_factory=dict)
    """策略间共享状态 (如摘要内容等)。"""

    def track_savings(self, saved: int) -> None:
        """记录 token 节省。"""
        self.saved_tokens += saved

    def update_budget(self, tokens: int) -> None:
        """更新预算中的实际 token 数。"""
        self.budget = self.budget.recalculate(tokens)
