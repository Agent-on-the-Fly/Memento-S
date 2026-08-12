# SPDX-License-Identifier: Apache-2.0
"""Execution helpers — append logic, context token, and pre-API pipeline utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.context import ContextManager
    from ..state import AgentRunState


def _live_ctx_tokens(ctx: ContextManager | None) -> int | None:
    """Return the live token count from ContextManager, or None."""
    if ctx is not None and hasattr(ctx, "total_tokens"):
        return ctx.total_tokens
    return None


async def _append_messages(
    ctx: ContextManager | None,
    messages: list[dict[str, Any]],
    new_msgs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Append messages via ContextManager if available, else plain concat.

    ContextManager.append() 只做纯追加 + token 计数。
    压缩由 prepare_for_api() 在 LLM 调用前处理。
    """
    if ctx is not None:
        return await ctx.append(messages, new_msgs)
    return list(messages) + new_msgs


async def _prepare_messages(
    ctx: ContextManager | None,
    messages: list[dict[str, Any]],
    state: AgentRunState,
    *,
    force_compact: bool = False,
) -> tuple[list[dict[str, Any]], str | None]:
    """Pre-API Pipeline: run prepare_for_api if ContextManager available.

    Returns:
        (processed_messages, compact_trigger) — compact_trigger 为触发压缩的策略名称或 None
    """
    if ctx is not None and hasattr(ctx, "_pipeline") and ctx._pipeline is not None:
        result = await ctx._pipeline.prepare_for_api(
            messages,
            state_messages_ref=state.messages,
            force_compact=force_compact,
        )
        compact_trigger: str | None = None
        if hasattr(result, "compact_trigger") and result.compact_trigger:
            compact_trigger = str(result.compact_trigger)
        if hasattr(result, "was_compacted") and result.was_compacted:
            if hasattr(result, "tokens_after"):
                ctx._total_tokens = result.tokens_after
        return result.messages_for_api, compact_trigger
    return messages, None
