# SPDX-License-Identifier: Apache-2.0
"""Pre-API Pipeline — 每次 LLM 调用前的上下文处理管道。

对标 CC query.ts L365-L467 的 pre-API 流程。

Pipeline 步骤 (每次 prepare_for_api 调用):
  1. apply_tool_result_budget — 已落盘大结果 → 稳定替换为 preview
  2. infra/compact — 渐进式压缩 (microcompact → SM compact → LLM → trim)

所有压缩逻辑委托给 infra/compact 模块。

层级说明: 放在 core/context/ 而非 core/context/history/，
因为其职责是"消息如何加工后送给 LLM"，而非"历史如何加载和裁剪"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from utils.logger import get_logger
from utils.token_utils import count_tokens, count_tokens_messages

from infra.compact import CompactPipeline, CompactConfig, CompactBudget, compact_trigger_label
from infra.compact.storage import ArtifactStore
from infra.compact.utils import extract_key_content
from infra.compact.models import ToolResultReplacementState

logger = get_logger(__name__)


# 内部用返回类型 (core 层视角，与 infra/compact.PipelineResult 不同)
@dataclass
class PipelineResult:
    """prepare_for_api 的返回结果 — core 层视角。"""

    messages_for_api: list[dict[str, Any]]
    was_compacted: bool = False
    compact_trigger: str | None = None
    tokens_before: int = 0
    tokens_after: int = 0


class _TokenCounterAdapter:
    """将 utils.token_utils 适配为 infra.compact.abc.TokenCounter 协议。"""

    def count_text(self, text: str, *, model: str = "") -> int:
        return count_tokens(text, model=model)

    def count_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str = "",
        tools: list[dict[str, Any]] | None = None,
    ) -> int:
        return count_tokens_messages(messages, model=model, tools=tools)

    def estimate_fast(self, text: str) -> int:
        return len(text) // 3 + 1


class _LLMClientAdapter:
    """将 middleware.llm.llm_client 适配为 infra.compact.abc.LLMClient 协议。"""

    async def chat(
        self,
        system: str,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        model: str = "",
    ) -> str:
        from middleware.llm.llm_client import chat_completions_async

        return await chat_completions_async(
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            model=model,
        )


class PreApiPipeline:
    """每次 LLM 调用前的上下文处理管道。

    压缩逻辑完全委托给 infra.compact.CompactPipeline，
    本类只保留业务相关的 tool result 处理。
    """

    def __init__(
        self,
        model: str,
        context_budget: int,
        microcompact_keep_recent: int,
        microcompact_compactable_tools: set[str],
        emergency_keep_tail: int,
        max_compact_failures: int,
        artifact_store: ArtifactStore | None = None,
        session_memory: Any = None,
        breaker_cooldown_s: float = 60.0,
        pipeline_preview_budget: int = 500,
    ) -> None:
        self._model = model
        self._context_budget = context_budget
        self._mc_keep_recent = microcompact_keep_recent
        self._mc_tools = microcompact_compactable_tools
        self._emergency_keep_tail = emergency_keep_tail
        self._max_failures = max_compact_failures
        self._artifact_store = artifact_store
        self._session_memory = session_memory
        self._breaker_cooldown = breaker_cooldown_s
        self._preview_budget = pipeline_preview_budget
        self._replacement_state = ToolResultReplacementState()

        # 初始化 infra/compact 压缩管道
        self._compact_pipeline = self._init_compact_pipeline()

    def _init_compact_pipeline(self) -> CompactPipeline:
        """初始化 infra.compact 压缩管道。"""
        config = CompactConfig(
            microcompact_keep_recent=self._mc_keep_recent,
            microcompact_compactable_tools=list(self._mc_tools),
            emergency_keep_tail=self._emergency_keep_tail,
            max_compact_failures=self._max_failures,
            breaker_cooldown_s=self._breaker_cooldown,
            sm_compact_min_ratio=0.02,
            sm_compact_max_ratio=0.08,
            model=self._model,
            summary_reader=self._session_memory,
            token_counter=_TokenCounterAdapter(),
            llm_client=_LLMClientAdapter(),
        )

        return CompactPipeline(config=config, observers=[])

    async def prepare_for_api(
        self,
        messages: list[dict[str, Any]],
        *,
        state_messages_ref: list[dict[str, Any]] | None = None,
        force_compact: bool = False,
    ) -> PipelineResult:
        """Pre-API Pipeline 主入口。

        Args:
            messages: 原始消息列表
            state_messages_ref: 可选，通过 in-place 列表修改将结果回写给调用方
            force_compact: 跳过预算检查，强制执行压缩策略链
                          (实际是否压缩由 CompactPipeline 内部策略决定)
        """
        tokens_before = count_tokens_messages(messages, model=self._model)

        # Step 1: Apply Tool Result Budget (业务逻辑，保留)
        working = await self._apply_tool_result_budget(messages)

        # Step 2: 压缩逻辑完全委托给 infra/compact
        budget = CompactBudget(
            total_tokens=tokens_before,
            total_limit=self._context_budget,
            per_message_limit=max(1, self._context_budget // 4),
        )

        compact_result = await self._compact_pipeline.run(
            working,
            budget,
            force_compact=force_compact,
        )

        result_messages = compact_result.messages_for_api
        was_compacted = compact_result.was_compacted
        trigger = compact_trigger_label(compact_result.compact_trigger)

        # 如果 infra/compact 未能在预算内，进行最后防线 fallback
        if not was_compacted or count_tokens_messages(result_messages, model=self._model) > self._context_budget:
            result_messages = self._group_aware_trim_fallback(result_messages)
            was_compacted = True
            trigger = "emergency"
            logger.info("Compact fallback: used group_aware_trim")

        if state_messages_ref is not None:
            state_messages_ref[:] = result_messages

        tokens_after = count_tokens_messages(result_messages, model=self._model)

        return PipelineResult(
            messages_for_api=result_messages,
            was_compacted=was_compacted,
            compact_trigger=trigger,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
        )

    def _group_aware_trim_fallback(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """最后防线: group_aware_trim。

        保留最近 N 轮对话，丢弃更早的轮次。
        未来可考虑统一到 infra/compact 暴露的公开方法。
        """
        from infra.compact.utils import group_messages_by_round

        if not messages:
            return messages

        system_msgs = [m for m in messages if m.get("role") == "system"]
        rest = [m for m in messages if m.get("role") != "system"]

        if not rest:
            return system_msgs

        groups = group_messages_by_round(rest)
        if len(groups) <= self._emergency_keep_tail:
            return system_msgs + rest

        kept_groups = groups[-self._emergency_keep_tail:]
        kept_msgs = []
        for g in kept_groups:
            kept_msgs.extend(g)

        return system_msgs + kept_msgs

    # ------------------------------------------------------------------
    # Step 1: Tool Result Budget
    # ------------------------------------------------------------------

    async def _apply_tool_result_budget(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Check tool results: if persisted in ArtifactStore, replace with preview.

        Uses ToolResultReplacementState for idempotent stable replacements.
        For execute_skill results: auto-recall full content (skip preview replacement)
        so the LLM always sees the complete skill output without needing manual recall.
        """
        if not self._artifact_store:
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

            if await self._artifact_store.has_artifact(tcid):
                if tool_name == "execute_skill":
                    full_content = await self._artifact_store.read_artifact(tcid)
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
