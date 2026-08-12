# SPDX-License-Identifier: Apache-2.0
"""Fallback 策略 — token-budget-aware group-aware-trim 最后防线。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from infra.compact.models import CompactBudget, CompactContext, CompactResult, CompactTrigger
from infra.compact.strategies.base import BaseStrategy
from infra.compact.utils import (
    find_tail_cut_by_tokens,
    group_messages_by_round,
)

if TYPE_CHECKING:
    from infra.compact.config import CompactConfig


class FallbackStrategy(BaseStrategy):
    """Fallback 策略 — token-budget-aware group-aware-trim 最后防线。

    当所有其他压缩策略都失败时执行此策略。
    按 API 轮次分组后，使用 token budget 从尾部保留完整组，
    保证 tool_use/tool_result 配对完整。
    始终保留 system 消息 (index 0)。
    """

    name = "fallback"

    async def compact(
        self,
        messages: list[dict[str, Any]],
        budget: "CompactBudget",
        context: "CompactContext | None" = None,
    ) -> "CompactResult":
        """执行 token-budget-aware group-aware-trim。"""
        original_count = len(messages)
        keep_tail = self._config.emergency_keep_tail

        if not messages:
            return self._create_result(
                [],
                CompactTrigger.TRIM,
                original_count,
            )

        # Separate system messages
        system_msgs: list[dict[str, Any]] = []
        rest: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "system":
                system_msgs.append(msg)
            else:
                rest.append(msg)

        if not rest:
            return self._create_result(
                system_msgs + rest,
                CompactTrigger.TRIM,
                original_count,
            )

        groups = group_messages_by_round(rest)

        if len(groups) <= keep_tail:
            return self._create_result(
                system_msgs + rest,
                CompactTrigger.TRIM,
                original_count,
                metadata={"kept_groups": len(groups)},
            )

        # Use token budget tail protection for which groups to keep
        # Build message list by group to find cut point
        kept_groups = self._select_groups_by_budget(groups, budget)

        kept_msgs: list[dict[str, Any]] = []
        for g in kept_groups:
            kept_msgs.extend(g)

        return self._create_result(
            system_msgs + kept_msgs,
            CompactTrigger.TRIM,
            original_count,
            metadata={
                "kept_groups": len(kept_groups),
                "total_groups": len(groups),
                "dropped_groups": len(groups) - len(kept_groups),
            },
        )

    def _select_groups_by_budget(
        self,
        groups: list[list[dict[str, Any]]],
        budget: "CompactBudget",
    ) -> list[list[dict[str, Any]]]:
        """Select groups from tail using token budget, keeping at least emergency_keep_tail groups."""
        keep_tail = self._config.emergency_keep_tail
        n = len(groups)

        if n <= keep_tail:
            return list(groups)

        # Use token budget to determine how many groups to keep
        # Protect at least keep_tail groups, try to keep more if budget allows
        tail_groups = list(groups)
        min_keep = max(1, keep_tail)
        max_keep = n

        for candidate_count in range(max_keep, min_keep - 1, -1):
            candidate_groups = tail_groups[-candidate_count:]
            # Estimate tokens
            from infra.compact.utils import estimate_message_tokens_rough
            total_tokens = sum(
                estimate_message_tokens_rough(g) for g in candidate_groups
            )
            if total_tokens <= budget.total_limit:
                return candidate_groups

        # Fallback: return minimum
        return tail_groups[-min_keep:]
