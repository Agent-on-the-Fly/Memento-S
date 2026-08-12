# SPDX-License-Identifier: Apache-2.0
"""ArtifactProvider — Tool Result 持久化抽象。

抽象 Tool Result 的动态落盘 + 智能提取能力。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ArtifactProvider(ABC):
    """Tool Result 持久化抽象。

    大 tool result 写入磁盘，上下文用 preview 替代。
    支持智能提取（结构化 → LLM → 截断）以及完整内容回读。
    """

    @abstractmethod
    async def process_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        content: str,
        *,
        remaining_budget_tokens: int = 0,
    ) -> dict[str, Any]:
        """处理并持久化 tool result。

        如果 content 超过动态阈值，写入磁盘，返回 preview。
        如果不超阈值，直接原样返回。

        Returns:
            处理后的 tool result dict（含 role="tool", content, tool_call_id, name）
        """
        ...

    @abstractmethod
    async def read_artifact(self, tool_call_id: str) -> str | None:
        """读取完整 artifact 内容（从磁盘）。"""
        ...

    @abstractmethod
    async def has_artifact(self, tool_call_id: str) -> bool:
        """检查 artifact 是否已落盘。"""
        ...
