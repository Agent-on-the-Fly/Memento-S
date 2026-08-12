# SPDX-License-Identifier: Apache-2.0
"""LLM 单消息压缩策略 — 压缩单条超大消息。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from infra.compact.models import CompactBudget, CompactContext, CompactResult, CompactTrigger
from infra.compact.strategies.base import BaseStrategy
from infra.compact.utils import (
    extract_key_content,
    estimate_tokens_fast,
)

if TYPE_CHECKING:
    from infra.compact.config import CompactConfig


class LLMSingleStrategy(BaseStrategy):
    """LLM 单消息压缩策略 — 压缩单条超过阈值的消息。

    当单条消息的 token 数超过 total_limit // 4 时触发。
    优先使用 LLM 总结，fallback 到 extract_key_content。
    """

    name = "llm_single"

    async def compact(
        self,
        messages: list[dict[str, Any]],
        budget: "CompactBudget",
        context: "CompactContext | None" = None,
    ) -> "CompactResult":
        """压缩单条超大的消息。"""
        from infra.compact.utils import count_messages_tokens
        from infra.compact.prompts import COMPRESS_TOOL_RESULT_SYSTEM

        original_count = len(messages)
        threshold = budget.per_message_limit
        if threshold <= 0:
            threshold = budget.total_limit // 4

        saved = 0
        result: list[dict[str, Any]] = []
        compressed_any = False

        for msg in messages:
            if msg.get("role") in ("tool", "user", "assistant"):
                content = msg.get("content", "")
                if isinstance(content, str) and content:
                    if estimate_tokens_fast(content) <= threshold:
                        result.append(msg)
                        continue

                    tokens = self._count_text(content)
                    if tokens > threshold:
                        compressed = await self._compress_message(
                            msg, threshold, context
                        )
                        new_content = compressed.get("content", "")
                        new_tokens = self._count_text(new_content) if isinstance(new_content, str) else 0
                        saved += tokens - new_tokens
                        result.append(compressed)
                        compressed_any = True
                        continue
            result.append(msg)

        trigger = CompactTrigger.SINGLE if compressed_any else CompactTrigger.MICRO
        return self._create_result(
            result,
            trigger,
            original_count,
            metadata={
                "saved_tokens": saved,
                "compressed_count": len([m for m in result if m.get("_compressed")] if compressed_any else []),
            },
        )

    async def _compress_message(
        self,
        msg: dict[str, Any],
        max_tokens: int,
        context: "CompactContext | None",
    ) -> dict[str, Any]:
        """压缩单条消息。"""
        from infra.compact.prompts import COMPRESS_TOOL_RESULT_SYSTEM

        content = msg.get("content", "")
        if not isinstance(content, str) or not content:
            return msg

        role = msg.get("role", "user")
        summary_tokens = min(
            self._config.llm_single_summary_tokens,
            max_tokens,
        )

        # 尝试 LLM 总结
        if self._config.llm_client is not None:
            try:
                summary = await self._llm_summarize(
                    content,
                    COMPRESS_TOOL_RESULT_SYSTEM,
                    summary_tokens,
                )
                result = dict(msg)
                result["content"] = f"[compressed from {role}]\n{summary.strip()}"
                result["_compressed"] = True
                return result
            except Exception:
                pass

        # Fallback: extract_key_content
        result = dict(msg)
        result["content"] = extract_key_content(content, max_tokens, model=self._config.model)
        result["_compressed"] = True
        return result

    def _count_text(self, text: str) -> int:
        """计算文本 token 数。"""
        if self._config.token_counter is not None:
            return self._config.token_counter.count_text(text, model=self._config.model)
        return estimate_tokens_fast(text)
