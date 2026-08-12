# SPDX-License-Identifier: Apache-2.0
"""CompactorProvider — Pre-API 压缩流水线抽象。

抽象 LLM 调用前的上下文压缩能力：Microcompact → Budget Guard → SM Compact → Emergency。

CompactTrigger 统一从 infra.compact.models 导入（AUTO 语义等价于 SM）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from infra.compact.models import CompactTrigger, ToolResultReplacementState


@dataclass
class CompactorStats:
    """Compactor 统计。"""

    total_calls: int = 0
    total_compacted: int = 0
    consecutive_failures: int = 0
    last_failure_time: float = 0.0

    # 细分
    auto_compactions: int = 0
    emergency_compactions: int = 0


class CompactorProvider(ABC):
    """Pre-API 压缩流水线抽象。

    每次 LLM 调用前执行：
      1. apply_tool_result_budget — 落盘结果替换为 preview
      2. microcompact — 零 LLM 清除旧 tool result 内容
      3. compress_oversized_messages — 单条超大消息 LLM 压缩
      4. budget_guard — 超限 → SM compact → LLM emergency compact → group_aware_trim
    """

    @abstractmethod
    async def prepare_for_api(
        self,
        messages: list[dict[str, Any]],
        *,
        state_messages_ref: list[dict[str, Any]] | None = None,
        force_compact: bool = False,
    ) -> tuple[list[dict[str, Any]], bool, CompactTrigger | None, int, int]:
        """执行预压缩流水线。

        Returns:
            (处理后的消息列表, 是否被压缩, 压缩触发原因, 压缩前 token 数, 压缩后 token 数)
        """
        ...

    @abstractmethod
    def get_stats(self) -> CompactorStats:
        """获取压缩统计。"""
        ...

    @abstractmethod
    async def force_compact_now(
        self,
        history_loader: Any,
        model: str,
        summary_tokens: int,
    ) -> tuple[int, int, str]:
        """立即强制压缩（Agent 外部触发）。

        Returns:
            (压缩前 token 数, 压缩后 token 数, preview 摘要)
        """
        ...
