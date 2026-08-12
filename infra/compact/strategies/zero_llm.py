# SPDX-License-Identifier: Apache-2.0
"""零 LLM 压缩策略 — Microcompact + SM Compact。

Microcompact 委托 infra/shared/compact.py 处理（唯一真实来源）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from infra.compact.models import CompactBudget, CompactContext, CompactResult, CompactTrigger
from infra.compact.strategies.base import BaseStrategy
from infra.compact.utils import (
    adjust_index_to_preserve_invariants,
    deduplicate_and_summarize_tool_results,
    estimate_tokens_fast,
    group_messages_by_round,
)
from infra.shared.compact import microcompact_messages

if TYPE_CHECKING:
    from infra.compact.config import CompactConfig


class ZeroLLMStrategy(BaseStrategy):
    """零 LLM 压缩策略 — Microcompact 和 SM Compact。

    这是最高优先级的压缩策略，因为完全不消耗 LLM 调用。
    """

    name = "zero_llm"

    async def compact(
        self,
        messages: list[dict[str, Any]],
        budget: "CompactBudget",
        context: "CompactContext | None" = None,
    ) -> "CompactResult":
        """执行零 LLM 压缩。

        步骤:
        1. 尝试 Microcompact — 清除旧 tool result 内容
        2. 如果仍超预算，尝试 SM Compact — 基于外部摘要截断
        """
        original_count = len(messages)
        working = list(messages)

        # Step 1: Microcompact
        working, mc_saved = self._microcompact(working)
        tokens_now = self._count_tokens(working)

        if tokens_now <= budget.total_limit:
            return self._create_result(
                working,
                CompactTrigger.MICRO,
                original_count,
                metadata={"saved_tokens": mc_saved},
            )

        # Step 2: SM Compact (如果配置了 summary_reader)
        if self._config.summary_reader is not None:
            sm_result = await self._sm_compact(working, budget)
            if sm_result is not None:
                return sm_result

        # SM Compact 不可用或失败，标记为 MICRO 但实际未成功
        return self._create_result(
            working,
            CompactTrigger.MICRO,
            original_count,
            metadata={"saved_tokens": mc_saved},
        )

    def _microcompact(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int]:
        """清除旧的 compactable tool result 内容。

        先委托 infra/shared/compact.py 的 microcompact_messages 处理，
        再用 deduplicate_and_summarize_tool_results 进一步压缩
        （工具结果去重 + 信息性摘要）。
        """
        keep_recent = self._config.microcompact_keep_recent
        compactable_tools = set(self._config.microcompact_compactable_tools)

        # Step 1: Original microcompact pass
        working, mc_saved = microcompact_messages(messages, keep_recent, compactable_tools)

        # Step 2: Deduplication + informative summaries
        # Use a fraction of keep_recent as token budget for tail protection
        tail_budget = keep_recent * 150  # ~150 chars per message * keep_recent
        working, dedup_pruned = deduplicate_and_summarize_tool_results(
            working,
            protect_tail_tokens=tail_budget,
            protect_tail_count=max(2, keep_recent // 2),
        )

        total_pruned = mc_saved + dedup_pruned
        return working, total_pruned

    async def _sm_compact(
        self,
        messages: list[dict[str, Any]],
        budget: "CompactBudget",
    ) -> "CompactResult | None":
        """SM Compact — 零 LLM 成本 (直接读 summary.md)。

        Uses lastSummarizedSeq for precise message retention boundary.
        """
        from infra.compact.prompts import get_compact_user_summary_message

        sm = self._config.summary_reader
        if sm is None:
            return None

        # 检查 summary 是否存在
        has_content = False
        sm_content = ""

        if hasattr(sm, "get_content"):
            sm_content = sm.get_content()
            has_content = bool(sm_content.strip()) and not sm.is_empty()
        elif hasattr(sm, "has_digests"):
            has_content = sm.has_digests()

        if not has_content:
            return None

        # 获取截断后的摘要
        truncated, success = sm.truncate_for_compact()
        if not success:
            return None

        # 计算要保留的消息
        last_seq = 0
        if hasattr(sm, "last_summarized_seq"):
            last_seq = sm.last_summarized_seq

        keep_idx = self._calculate_keep_index(messages, last_seq)
        keep_idx = adjust_index_to_preserve_invariants(messages, keep_idx)
        kept = messages[keep_idx:]

        # 构建摘要消息
        summary_text = get_compact_user_summary_message(
            truncated,
            recent_preserved=True,
        )
        summary_msg: dict[str, Any] = {
            "role": "user",
            "content": summary_text,
            "_metadata": {"is_compact_summary": True},
        }

        # 保留 system 消息
        system = messages[0] if messages and messages[0].get("role") == "system" else None
        result = ([system] if system else []) + [summary_msg] + kept

        # 检查是否在预算内
        result_tokens = self._count_tokens(result)
        if result_tokens > budget.total_limit:
            return None

        return self._create_result(
            result,
            CompactTrigger.SM,
            len(messages),
            summary_message=summary_msg,
            metadata={
                "saved_tokens": self._count_tokens(messages) - result_tokens,
                "sm_content": truncated[:200],  # 截断用于日志
            },
        )

    def _calculate_keep_index(
        self,
        messages: list[dict[str, Any]],
        last_seq: int = 0,
    ) -> int:
        """Calculate keep index based on SM Compact ratios."""
        from infra.compact.utils import estimate_message_tokens_rough

        sm_min_tokens = max(100, int(budget.total_limit * self._config.sm_compact_min_ratio))
        sm_max_tokens = max(500, int(budget.total_limit * self._config.sm_compact_max_ratio))

        if last_seq > 0:
            # 基于 seq 的精确查找
            start = len(messages)
            for i, msg in enumerate(messages):
                if msg.get("_seq", 0) > last_seq:
                    start = i
                    break

            if start >= len(messages):
                return self._calculate_keep_index_from_tail(messages, sm_min_tokens, sm_max_tokens)

            total_tokens = 0
            for j in range(start, len(messages)):
                total_tokens += estimate_message_tokens_rough([messages[j]])

            if total_tokens < sm_min_tokens:
                while start > 1 and total_tokens < sm_max_tokens:
                    start -= 1
                    total_tokens += estimate_message_tokens_rough([messages[start]])

            return start

        return self._calculate_keep_index_from_tail(messages, sm_min_tokens, sm_max_tokens)

    def _calculate_keep_index_from_tail(
        self,
        messages: list[dict[str, Any]],
        min_tokens: int,
        max_tokens: int,
    ) -> int:
        """Calculate keep index from tail, expanding until min_tokens met."""
        from infra.compact.utils import estimate_message_tokens_rough

        if len(messages) <= 1:
            return 0

        total_tokens = 0
        start_index = len(messages)

        for i in range(len(messages) - 1, 0, -1):
            msg_tokens = estimate_message_tokens_rough([messages[i]])
            total_tokens += msg_tokens
            start_index = i

            if total_tokens >= max_tokens:
                break

            if total_tokens >= min_tokens:
                break

        return start_index

    def _count_tokens(self, messages: list[dict[str, Any]]) -> int:
        """计算消息列表的 token 数。"""
        from infra.compact.utils import count_messages_tokens

        return count_messages_tokens(messages, self._config)
