# SPDX-License-Identifier: Apache-2.0
"""策略模式基类 — 所有压缩策略的抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from infra.compact.config import CompactConfig
    from infra.compact.models import CompactBudget, CompactContext, CompactResult


class BaseStrategy(ABC):
    """压缩策略基类 — 定义策略接口和通用能力。"""

    name: str = "base"

    def __init__(self, config: "CompactConfig") -> None:
        self._config = config

    @abstractmethod
    async def compact(
        self,
        messages: list[dict[str, Any]],
        budget: "CompactBudget",
        context: "CompactContext | None" = None,
    ) -> "CompactResult":
        """执行压缩策略。

        Args:
            messages: 原始消息列表
            budget: 压缩预算
            context: 压缩执行上下文 (可选，用于策略间数据传递)

        Returns:
            CompactResult 包含压缩结果
        """
        ...

    def _create_result(
        self,
        messages: list[dict[str, Any]],
        trigger_name: str,
        original_count: int,
        summary_message: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "CompactResult":
        """创建标准 CompactResult。"""
        from infra.compact.models import CompactResult, CompactTrigger

        trigger = CompactTrigger(trigger_name)
        return CompactResult(
            messages=messages,
            trigger=trigger,
            dropped_count=original_count - len(messages),
            summary_message=summary_message,
            metadata=metadata or {},
        )

    def _check_within_budget(
        self,
        messages: list[dict[str, Any]],
        budget: "CompactBudget",
    ) -> bool:
        """检查消息列表是否在预算内。"""
        from infra.compact.utils import count_messages_tokens

        total = count_messages_tokens(messages, self._config)
        return total <= budget.total_limit

    async def _llm_summarize(
        self,
        content: str,
        system: str,
        max_tokens: int,
    ) -> str:
        """调用 LLM 总结 (如果有配置 LLM client)。"""
        if self._config.llm_client is None:
            raise RuntimeError("LLM client not configured")

        result = await self._config.llm_client.chat(
            system=system,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            model=self._config.model,
        )
        return result
