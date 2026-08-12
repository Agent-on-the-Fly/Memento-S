# SPDX-License-Identifier: Apache-2.0
"""LLM 紧急压缩策略 — 全量 13 字段结构化摘要压缩。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from infra.compact.models import CompactBudget, CompactContext, CompactResult, CompactTrigger
from infra.compact.strategies.base import BaseStrategy
from infra.compact.utils import (
    _align_boundary_forward,
    _CHARS_PER_TOKEN,
    count_messages_tokens,
    deduplicate_and_summarize_tool_results,
    extract_summary_from_messages,
    find_tail_cut_by_tokens,
    sanitize_tool_pairs,
    serialize_tool_calls,
    with_summary_prefix,
)
from infra.compact.prompts import (
    COMPACT_SYSTEM_PROMPT,
    build_compression_prompt,
    get_compact_user_summary_message,
)

if TYPE_CHECKING:
    from infra.compact.config import CompactConfig

# Minimum summary token budget
_MIN_SUMMARY_TOKENS = 2000
_SUMMARY_FAILURE_COOLDOWN_SECONDS = 600


def _serialize_for_summary(
    turns: list[dict[str, Any]],
    max_content_chars: int = 6000,
    max_tool_args_chars: int = 1500,
) -> str:
    """Serialize conversation turns into labeled text for the summarizer.

    Includes tool call arguments and result content (up to max_content_chars per message)
    so the summarizer can preserve specific details like file paths, commands, and outputs.
    """
    parts = []
    for msg in turns:
        role = msg.get("role", "unknown")
        content = msg.get("content") or ""

        # Tool results: keep enough content for the summarizer
        if role == "tool":
            tool_id = msg.get("tool_call_id", "")
            if len(content) > max_content_chars:
                content = content[:4000] + "\n...[truncated]...\n" + content[-1500:]
            parts.append(f"[TOOL RESULT {tool_id}]: {content}")
            continue

        # Assistant messages: include tool call names AND arguments
        if role == "assistant":
            if len(content) > max_content_chars:
                content = content[:4000] + "\n...[truncated]...\n" + content[-1500:]
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                tc_parts = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        fn = tc.get("function", {})
                        name = fn.get("name", "?")
                        args = fn.get("arguments", "")
                        if len(args) > max_tool_args_chars:
                            args = args[:1200] + "..."
                        tc_parts.append(f"  {name}({args})")
                content += "\n[Tool calls:\n" + "\n".join(tc_parts) + "\n]"
            parts.append(f"[ASSISTANT]: {content}")
            continue

        # User and other roles
        if len(content) > max_content_chars:
            content = content[:4000] + "\n...[truncated]...\n" + content[-1500:]
        parts.append(f"[{role.upper()}]: {content}")

    return "\n\n".join(parts)


class LLMEmergencyStrategy(BaseStrategy):
    """LLM 紧急压缩策略 — 13 字段结构化摘要。

    当零 LLM 压缩和 SM Compact 都无法满足预算时触发。
    支持迭代压缩（增量更新摘要）和 anti-thrashing。

    算法:
      1. Pre-pass: 工具结果去重 + 信息性摘要
      2. Head 保护: 固定消息数
      3. Tail 保护: token budget 驱动
      4. 13 字段结构化 LLM 摘要
      5. 迭代压缩支持 (previous_summary 增量更新)
      6. Tool pair integrity 校验
    """

    name = "llm_emergency"

    def __init__(self, config: "CompactConfig") -> None:
        super().__init__(config)
        self._previous_summary: str | None = None
        self._last_compression_savings_pct: float = 100.0
        self._ineffective_compression_count: int = 0
        self._summary_failure_cooldown_until: float = 0.0
        self._summary_model_fallen_back: bool = False

    async def compact(
        self,
        messages: list[dict[str, Any]],
        budget: "CompactBudget",
        context: "CompactContext | None" = None,
    ) -> "CompactResult":
        """执行 13 字段结构化摘要压缩。"""
        original_count = len(messages)
        import time
        now = time.monotonic()

        # Anti-thrashing: back off if recent compressions were ineffective
        if self._ineffective_compression_count >= 2 and self._previous_summary is not None:
            if now < self._summary_failure_cooldown_until:
                # Fall back to legacy emergency (no iteration)
                return await self._compact_legacy(messages, budget, original_count)

        # Pre-pass: tool result deduplication + informative summaries
        prune_tail_tokens = self._derive_tail_token_budget()
        deduped, _ = deduplicate_and_summarize_tool_results(
            messages,
            protect_tail_tokens=prune_tail_tokens,
            protect_tail_count=self._config.microcompact_keep_recent,
        )

        # Determine head and tail boundaries
        protect_first_n = self._config.microcompact_keep_recent
        compress_start = _align_boundary_forward(deduped, protect_first_n)
        compress_end = find_tail_cut_by_tokens(
            deduped,
            head_end=compress_start,
            tail_token_budget=prune_tail_tokens,
            min_tail_messages=3,
        )

        if compress_start >= compress_end:
            return self._create_result(list(messages), CompactTrigger.EMERGENCY, original_count)

        turns_to_summarize = deduped[compress_start:compress_end]
        if not turns_to_summarize:
            return self._create_result(list(messages), CompactTrigger.EMERGENCY, original_count)

        # Serialize and generate summary
        summary_budget = self._compute_summary_budget(turns_to_summarize)
        content_to_summarize = _serialize_for_summary(turns_to_summarize)

        prompt = build_compression_prompt(
            content_to_summarize,
            summary_budget,
            previous_summary=self._previous_summary,
        )

        raw_summary = await self._llm_summarize(
            prompt,
            COMPACT_SYSTEM_PROMPT,
            max_tokens=int(summary_budget * 1.3),
        )

        if not raw_summary:
            self._summary_failure_cooldown_until = now + _SUMMARY_FAILURE_COOLDOWN_SECONDS
            return await self._compact_legacy(messages, budget, original_count)

        summary_with_prefix = with_summary_prefix(raw_summary)

        # Build compressed message list
        compressed = self._build_compressed_messages(
            deduped,
            compress_start,
            compress_end,
            summary_with_prefix,
        )

        # Tool pair integrity check
        compressed = sanitize_tool_pairs(compressed)

        # Calculate savings and update state
        tokens_before = count_messages_tokens(messages, self._config)
        tokens_after = count_messages_tokens(compressed, self._config)
        saved = tokens_before - tokens_after
        savings_pct = (saved / tokens_before * 100) if tokens_before > 0 else 0
        self._last_compression_savings_pct = savings_pct

        if savings_pct < 10:
            self._ineffective_compression_count += 1
        else:
            self._ineffective_compression_count = 0
            self._summary_failure_cooldown_until = 0.0

        self._previous_summary = raw_summary
        self._summary_model_fallen_back = False

        summary_msg: dict[str, Any] = {
            "role": "assistant",
            "content": summary_with_prefix + "\n\n" + self._build_handoff_note(compressed, deduped, compress_end),
            "_metadata": {"is_compact_summary": True},
        }

        return self._create_result(
            compressed,
            CompactTrigger.EMERGENCY,
            original_count,
            summary_message=summary_msg,
            metadata={
                "saved_tokens": saved,
                "savings_pct": round(savings_pct, 1),
                "ineffective_count": self._ineffective_compression_count,
            },
        )

    def _derive_tail_token_budget(self) -> int:
        """Derive tail token budget from config (scaled by summary_ratio)."""
        ratio = self._config.summary_ratio if hasattr(self._config, "summary_ratio") else 0.15
        target = int(self._config.summary_tokens * ratio) if hasattr(self._config, "summary_tokens") else 2000
        return max(_MIN_SUMMARY_TOKENS, target)

    def _compute_summary_budget(self, turns_to_summarize: list[dict[str, Any]]) -> int:
        """Scale summary token budget with the amount of content being compressed."""
        content_tokens = sum(
            len(msg.get("content", "") or "") // _CHARS_PER_TOKEN + 10
            for msg in turns_to_summarize
        )
        ratio = getattr(self._config, "summary_ratio", 0.20)
        budget = int(content_tokens * ratio)
        ceiling = int(getattr(self._config, "summary_tokens", 12000) * 0.05)
        return max(_MIN_SUMMARY_TOKENS, min(budget, ceiling))

    def _build_compressed_messages(
        self,
        messages: list[dict[str, Any]],
        compress_start: int,
        compress_end: int,
        summary_with_prefix: str,
    ) -> list[dict[str, Any]]:
        """Assemble the compressed message list: head + summary + tail."""
        compressed = []

        for i in range(compress_start):
            msg = messages[i].copy()
            if i == 0 and msg.get("role") == "system":
                existing = msg.get("content") or ""
                note = "[Note: Some earlier conversation turns have been compacted into a handoff summary to preserve context space. The current session state may still reflect earlier work, so build on that summary and state rather than re-doing work.]"
                if note not in existing:
                    msg["content"] = existing + "\n\n" + note
            compressed.append(msg)

        _merge_into_tail = False
        last_head_role = messages[compress_start - 1].get("role", "user") if compress_start > 0 else "user"
        first_tail_role = messages[compress_end].get("role", "user") if compress_end < len(messages) else "user"

        if last_head_role in ("assistant", "tool"):
            summary_role = "user"
        else:
            summary_role = "assistant"
        if summary_role == first_tail_role:
            flipped = "assistant" if summary_role == "user" else "user"
            if flipped != last_head_role:
                summary_role = flipped
            else:
                _merge_into_tail = True

        if not _merge_into_tail:
            compressed.append({"role": summary_role, "content": summary_with_prefix})

        for i in range(compress_end, len(messages)):
            msg = messages[i].copy()
            if _merge_into_tail and i == compress_end:
                original = msg.get("content") or ""
                msg["content"] = (
                    summary_with_prefix
                    + "\n--- END OF CONTEXT SUMMARY — respond to the message below, not the summary above ---\n"
                    + original
                )
                _merge_into_tail = False
            compressed.append(msg)

        return compressed

    def _build_handoff_note(
        self,
        compressed: list[dict[str, Any]],
        original: list[dict[str, Any]],
        compress_end: int,
    ) -> str:
        """Build the handoff note appended to the summary role message."""
        tail_msgs = len(original) - compress_end
        return (
            f"\n[{len(original)} original turns compressed. "
            f"{tail_msgs} recent turn(s) preserved verbatim.]"
        )

    async def _compact_legacy(
        self,
        messages: list[dict[str, Any]],
        budget: "CompactBudget",
        original_count: int,
    ) -> "CompactResult":
        """Fallback: use legacy emergency prompt (no iterative support)."""
        from infra.compact.prompts import COMPRESS_EMERGENCY_PROMPT, get_compact_user_summary_message
        from infra.compact.utils import messages_to_text, truncate_for_ptl_retry

        system_msg = messages[0] if messages and messages[0].get("role") == "system" else None
        start = 1 if system_msg else 0
        rest = list(messages[start:])

        if not rest:
            return self._create_result(list(messages), CompactTrigger.EMERGENCY, original_count)

        working_rest = list(rest)
        max_retries = self._config.max_ptl_retries
        truncate_ratio = self._config.ptl_truncate_ratio
        summary_tokens = self._derive_tail_token_budget()

        for attempt in range(max_retries + 1):
            context_text = messages_to_text(working_rest)
            full_prompt = COMPRESS_EMERGENCY_PROMPT + "\n\n" + context_text

            try:
                if self._config.llm_client is not None:
                    raw = await self._llm_summarize(
                        full_prompt,
                        COMPACT_SYSTEM_PROMPT,
                        summary_tokens,
                    )
                    summary_text = get_compact_user_summary_message(raw)
                else:
                    from infra.compact.utils import extract_key_content
                    summary_text = extract_key_content(context_text, summary_tokens, model=self._config.model)

                summary_msg: dict[str, Any] = {
                    "role": "user",
                    "content": summary_text,
                    "_metadata": {"is_compact_summary": True},
                }

                result = ([system_msg] if system_msg else []) + [summary_msg]
                result_tokens = count_messages_tokens(result, self._config)

                if result_tokens <= budget.total_limit:
                    return self._create_result(
                        result,
                        CompactTrigger.EMERGENCY,
                        original_count,
                        summary_message=summary_msg,
                        metadata={
                            "saved_tokens": count_messages_tokens(messages, self._config) - result_tokens,
                            "attempt": attempt + 1,
                            "legacy": True,
                        },
                    )

            except Exception as exc:
                exc_str = str(exc).lower()
                is_ptl = "prompt" in exc_str and "long" in exc_str

                if is_ptl and attempt < max_retries:
                    before = len(working_rest)
                    working_rest = truncate_for_ptl_retry(working_rest, truncate_ratio)
                    continue

                return self._create_result(
                    list(messages),
                    CompactTrigger.EMERGENCY,
                    original_count,
                    metadata={"error": str(exc), "attempt": attempt + 1, "legacy": True},
                )

        return self._create_result(
            list(messages),
            CompactTrigger.EMERGENCY,
            original_count,
            metadata={"max_retries_reached": True, "legacy": True},
        )

    def reset_state(self) -> None:
        """Reset per-session compression state (called on /new or /reset)."""
        self._previous_summary = None
        self._last_compression_savings_pct = 100.0
        self._ineffective_compression_count = 0
        self._summary_failure_cooldown_until = 0.0
        self._summary_model_fallen_back = False
