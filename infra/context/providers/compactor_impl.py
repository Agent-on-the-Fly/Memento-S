# SPDX-License-Identifier: Apache-2.0
"""CompactorProvider implementation — TokenBudgetCompactor.

迁移自 core/context/pipeline.py 的 PreApiPipeline。
无 core/ 依赖，LLM 客户端通过参数注入。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from utils.logger import get_logger
from utils.token_utils import count_tokens, count_tokens_messages, estimate_tokens_fast

from infra.compact.models import CompactTrigger, ToolResultReplacementState
from infra.context.providers.artifact import ArtifactProvider
from infra.context.providers.compactor import CompactorProvider, CompactorStats
from infra.shared.compact import microcompact_messages
from infra.context.providers.shared_prompts import (
    COMPACT_SYSTEM_PROMPT,
    COMPRESS_EMERGENCY_PROMPT,
    COMPRESS_SM_COMPACT_PROMPT,
    COMPRESS_TOOL_RESULT_SYSTEM,
    get_compact_user_summary_message,
)
from infra.context.providers.shared_utils import (
    adjust_index_to_preserve_invariants,
    build_digest,
    compress_emergency,
    compress_for_sm_compact,
    compress_message,
    extract_key_content,
    group_aware_trim,
)

logger = get_logger(__name__)

from infra.compact.constants import (
    DEFAULT_EMERGENCY_KEEP_TAIL,
    DEFAULT_FLOOR_TOKENS,
    DEFAULT_MAX_COMPACT_FAILURES,
    DEFAULT_MAX_PTL_RETRIES,
    DEFAULT_MICROCOMPACT_KEEP_RECENT,
    DEFAULT_PTL_TRUNCATE_RATIO,
)


# LLM client type alias
LLMClient = Callable[..., Awaitable[str]]


def _default_llm_client(
    messages: list[dict[str, Any]],
    *,
    system: str = "",
    max_tokens: int = 4000,
) -> Awaitable[str]:
    """Lazy default: tries to import from middleware.llm.llm_client."""
    from middleware.llm.llm_client import chat_completions_async

    return chat_completions_async(
        system=system,
        messages=messages,
        max_tokens=max_tokens,
    )


class TokenBudgetCompactor(CompactorProvider):
    """Pre-API 压缩流水线实现。

    所有 core/ 依赖通过参数注入：
      - artifact_provider: ArtifactProvider（可选）
      - session_memory: SessionMemoryLike（可选）
      - llm_client: 异步 LLM 调用（默认尝试 middleware）
    """

    def __init__(
        self,
        model: str,
        context_budget: int,
        *,
        microcompact_keep_recent: int = DEFAULT_MICROCOMPACT_KEEP_RECENT,
        microcompact_compactable_tools: set[str] | None = None,
        emergency_keep_tail: int = DEFAULT_EMERGENCY_KEEP_TAIL,
        max_compact_failures: int = DEFAULT_MAX_COMPACT_FAILURES,
        sm_compact_min_tokens: int = 800,
        sm_compact_max_tokens: int = 3000,
        artifact_provider: ArtifactProvider | None = None,
        session_memory: Any | None = None,
        breaker_cooldown_s: float = 60.0,
        pipeline_preview_budget: int = 500,
        llm_client: LLMClient | None = None,
    ) -> None:
        self._model = model
        self._context_budget = context_budget
        self._mc_keep_recent = microcompact_keep_recent
        self._mc_tools = microcompact_compactable_tools or {
            "execute_skill",
            "search_skill",
            "read_file",
            "bash",
        }
        self._emergency_keep_tail = emergency_keep_tail
        self._max_failures = max_compact_failures
        self._sm_min_tokens = sm_compact_min_tokens
        self._sm_max_tokens = sm_compact_max_tokens
        self._artifact_provider = artifact_provider
        self._session_memory = session_memory
        self._breaker_cooldown = breaker_cooldown_s
        self._preview_budget = pipeline_preview_budget
        self._llm_client = llm_client or _default_llm_client
        self._consecutive_failures: int = 0
        self._last_failure_time: float = 0.0
        self._replacement_state = ToolResultReplacementState()
        self._stats = CompactorStats()

    def update_budget(self, context_budget: int) -> None:
        """更新 context budget（供 FileContextProvider.init_budget() 调用）。"""
        self._context_budget = context_budget

    async def prepare_for_api(
        self,
        messages: list[dict[str, Any]],
        *,
        state_messages_ref: list[dict[str, Any]] | None = None,
        force_compact: bool = False,
    ) -> tuple[list[dict[str, Any]], bool, CompactTrigger | None, int, int]:
        tokens_before = count_tokens_messages(messages, model=self._model)
        working = list(messages)

        # Step 1: Apply Tool Result Budget
        working = await self._apply_tool_result_budget(working)

        # Step 2: Microcompact (零 LLM 调用)
        working, mc_saved = microcompact_messages(
            working,
            keep_recent=self._mc_keep_recent,
            compactable_tools=self._mc_tools,
        )
        if mc_saved > 0:
            logger.info("TokenBudgetCompactor microcompact freed ~{:.0f} tokens", mc_saved)

        # Step 2.5: Compress oversized individual messages
        working, compress_saved = await self._compress_oversized_messages(working)

        # Step 3: Budget Guard
        tokens_now = tokens_before - mc_saved - compress_saved
        was_compacted = False
        trigger: CompactTrigger | None = None

        if force_compact or tokens_now > self._context_budget:
            tokens_now = count_tokens_messages(working, model=self._model)
            if force_compact or tokens_now > self._context_budget:
                working, was_compacted, trigger = await self._budget_guard(
                    working, state_messages_ref
                )

        tokens_after = (
            count_tokens_messages(working, model=self._model)
            if was_compacted
            else tokens_now
        )

        self._stats.total_calls += 1
        if was_compacted:
            self._stats.total_compacted += 1
            if trigger == CompactTrigger.SM:
                self._stats.auto_compactions += 1
            elif trigger == CompactTrigger.EMERGENCY:
                self._stats.emergency_compactions += 1

        return working, was_compacted, trigger, tokens_before, tokens_after

    async def _apply_tool_result_budget(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """替换已落盘的 tool result 为 preview。"""
        if not self._artifact_provider:
            return messages

        result: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") != "tool":
                result.append(msg)
                continue

            tcid = msg.get("tool_call_id", "")
            if not tcid:
                result.append(msg)
                continue

            cached = self._replacement_state.get_replacement(tcid)
            if cached is not None:
                result.append({**msg, "content": cached})
                continue

            tool_name = msg.get("name", "")

            if await self._artifact_provider.has_artifact(tcid):
                if tool_name == "execute_skill":
                    full_content = await self._artifact_provider.read_artifact(tcid)
                    if full_content is not None:
                        result.append({**msg, "content": full_content})
                        continue

                content = msg.get("content", "")
                preview = extract_key_content(
                    content if isinstance(content, str) else str(content),
                    self._preview_budget,
                    model=self._model,
                )
                replaced = (
                    f"[Output persisted: use recall_context('{tcid}') for full content]\n"
                    f"Preview:\n{preview}"
                )
                self._replacement_state.mark_replaced(tcid, replaced)
                result.append({**msg, "content": replaced})
            else:
                result.append(msg)

        return result

    async def _compress_oversized_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], int]:
        """压缩超过 1/4 context_budget 的单条消息。"""
        threshold = self._context_budget // 4
        if threshold <= 0:
            return messages, 0

        saved = 0
        result: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") in ("tool", "user", "assistant"):
                content = msg.get("content", "")
                if isinstance(content, str) and content:
                    if estimate_tokens_fast(content) <= threshold:
                        result.append(msg)
                        continue
                    tokens = count_tokens(content, model=self._model)
                    if tokens > threshold:
                        compressed = await compress_message(
                            msg,
                            threshold,
                            summary_tokens=min(800, threshold // 2),
                            model=self._model,
                            llm_client=self._llm_client,
                        )
                        new_content = compressed.get("content", "")
                        new_tokens = (
                            count_tokens(new_content, model=self._model)
                            if isinstance(new_content, str)
                            else 0
                        )
                        saved += tokens - new_tokens
                        result.append(compressed)
                        continue
            result.append(msg)
        return result, saved

    async def _budget_guard(
        self,
        messages: list[dict[str, Any]],
        state_ref: list[dict[str, Any]] | None,
    ) -> tuple[list[dict[str, Any]], bool, CompactTrigger | None]:
        """Progressive compression: SM compact → LLM emergency → group_aware_trim。"""
        # Circuit breaker
        if self._consecutive_failures >= self._max_failures:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed < self._breaker_cooldown:
                logger.warning(
                    "Compact circuit breaker active ({} failures, {:.0f}s cooldown remaining)",
                    self._consecutive_failures,
                    self._breaker_cooldown - elapsed,
                )
                result = group_aware_trim(messages, self._emergency_keep_tail)
                return result, True, CompactTrigger.EMERGENCY
            self._consecutive_failures = 0

        # Level 1: SM Compact — zero LLM
        if self._session_memory:
            sm_is_empty = (
                self._session_memory.is_empty()
                if hasattr(self._session_memory, "is_empty")
                else not getattr(self._session_memory, "_digests", [])
            )
            if not sm_is_empty:
                result = await self._try_sm_compact(messages)
                if result is not None:
                    if state_ref is not None:
                        state_ref[:] = result
                    self._consecutive_failures = 0
                    return result, True, CompactTrigger.SM

        # Level 2: LLM Emergency Compact
        try:
            result = await compress_emergency(
                messages,
                model=self._model,
                llm_client=self._llm_client,
            )
            result_tokens = count_tokens_messages(result, model=self._model)
            if result_tokens <= self._context_budget:
                if state_ref is not None:
                    state_ref[:] = result
                self._consecutive_failures = 0
                return result, True, CompactTrigger.EMERGENCY
            logger.warning(
                "Emergency compact result still over budget: {} > {}",
                result_tokens, self._context_budget,
            )
        except Exception:
            logger.opt(exception=True).warning("compress_emergency failed")
            self._consecutive_failures += 1
            self._last_failure_time = time.monotonic()

        # Level 3: Group-Aware Trim
        result = group_aware_trim(messages, self._emergency_keep_tail)
        return result, True, CompactTrigger.EMERGENCY

    async def _try_sm_compact(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]] | None:
        """SM compact — 零 LLM 成本 (直接读 summary.md)。"""
        sm = self._session_memory
        has_content = False
        sm_content = ""

        if hasattr(sm, "get_content"):
            sm_content = sm.get_content()
            has_content = bool(sm_content.strip()) and not sm.is_empty()
        elif hasattr(sm, "has_digests"):
            has_content = sm.has_digests()

        if not has_content:
            return None

        truncated, _ = sm.truncate_for_compact()
        if not truncated:
            return None

        last_seq = sm.last_summarized_seq
        keep_idx = self._calculate_keep_index_from_seq(messages, last_seq)
        keep_idx = adjust_index_to_preserve_invariants(messages, keep_idx)
        kept = messages[keep_idx:]

        summary_text = get_compact_user_summary_message(
            truncated,
            recent_preserved=True,
        )
        summary_msg: dict[str, Any] = {
            "role": "user",
            "content": summary_text,
            "_metadata": {"is_compact_summary": True},
        }
        system = messages[0] if messages and messages[0].get("role") == "system" else None
        result = ([system] if system else []) + [summary_msg] + kept

        result_tokens = count_tokens_messages(result, model=self._model)
        if result_tokens > self._context_budget:
            logger.info(
                "SM compact result over budget: {} > {}, skipping",
                result_tokens, self._context_budget,
            )
            return None

        logger.info(
            "SM compact succeeded: {} -> {} messages, tokens -> {}",
            len(messages), len(result), result_tokens,
        )
        return result

    def _calculate_keep_index_from_seq(
        self, messages: list[dict[str, Any]], last_seq: int
    ) -> int:
        if last_seq <= 0:
            return self._calculate_keep_index(messages)

        start = len(messages)
        for i, msg in enumerate(messages):
            if msg.get("_seq", 0) > last_seq:
                start = i
                break

        if start >= len(messages):
            return self._calculate_keep_index(messages)

        from infra.context.providers.shared_utils import estimate_message_tokens_rough

        total_tokens = 0
        for j in range(start, len(messages)):
            total_tokens += estimate_message_tokens_rough([messages[j]])

        if total_tokens < self._sm_min_tokens:
            while start > 1 and total_tokens < self._sm_max_tokens:
                start -= 1
                total_tokens += estimate_message_tokens_rough([messages[start]])

        return start

    def _calculate_keep_index(self, messages: list[dict[str, Any]]) -> int:
        from infra.context.providers.shared_utils import estimate_message_tokens_rough

        if len(messages) <= 1:
            return 0

        total_tokens = 0
        start_index = len(messages)

        for i in range(len(messages) - 1, 0, -1):
            msg_tokens = estimate_message_tokens_rough([messages[i]])
            total_tokens += msg_tokens
            start_index = i

            if total_tokens >= self._sm_max_tokens:
                break

            if total_tokens >= self._sm_min_tokens:
                break

        return start_index

    def get_stats(self) -> CompactorStats:
        stats = CompactorStats(
            total_calls=self._stats.total_calls,
            total_compacted=self._stats.total_compacted,
            consecutive_failures=self._consecutive_failures,
            last_failure_time=self._last_failure_time,
            auto_compactions=self._stats.auto_compactions,
            emergency_compactions=self._stats.emergency_compactions,
        )
        return stats

    async def force_compact_now(
        self,
        history_loader: Any,
        model: str,
        summary_tokens: int,
    ) -> tuple[int, int, str]:
        """立即强制压缩。"""
        messages = await history_loader() if callable(history_loader) else []
        tokens_before = count_tokens_messages(messages, model=model)

        compacted, was_compacted, _, _, tokens_after = await self.prepare_for_api(
            messages, force_compact=True
        )

        if was_compacted:
            preview = "\n".join(
                m.get("content", "")[:200] for m in compacted if m.get("content")
            )[:500]
        else:
            preview = ""

        return tokens_before, tokens_after, preview
