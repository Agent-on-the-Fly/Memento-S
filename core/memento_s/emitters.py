"""AG-UI event emission patterns — the single place for all event output.

This module provides reusable async generators and helpers that emit
AG-UI events.  Business logic (execution, reflection, replan) does NOT
live here — only the "how to emit" layer.

Depends on:
  stream_output  — AGUIEventType, AgentFinishReason, build_event, new_run_id
  middleware.llm  — LLMClient (for streaming)
"""

from __future__ import annotations

from typing import Any, AsyncGenerator

from middleware.llm import LLMClient
from utils.logger import get_logger

from .stream_output import AGUIEventType, AgentFinishReason, build_event, new_run_id

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Streaming finalize (replaces simple_reply + _emit_streaming_finalize)
# ═══════════════════════════════════════════════════════════════════


async def stream_and_finalize(
    *,
    messages: list[dict[str, Any]],
    llm: LLMClient,
    tools: list[dict[str, Any]] | None,
    run_id: str,
    session_id: str,
    step: int,
    step_usage: dict[str, Any] | None = None,
    session_ctx: Any = None,
    session_manager: Any = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Stream an LLM response and emit STEP_FINISHED + TEXT_MESSAGE + RUN_FINISHED.

    This is the shared path for both DIRECT replies (tools=None) and
    post-execution finalize (tools=tool_schemas).
    """
    yield build_event(
        AGUIEventType.STEP_FINISHED, run_id, session_id,
        step=step, status="finalize",
    )

    msg_id = new_run_id()
    yield build_event(
        AGUIEventType.TEXT_MESSAGE_START, run_id, session_id,
        messageId=msg_id, role="assistant",
    )

    text_parts: list[str] = []
    async for chunk in llm.async_stream_chat(messages=messages, tools=tools):
        if chunk.usage:
            step_usage = chunk.usage
        if chunk.delta_content:
            text_parts.append(chunk.delta_content)
            yield build_event(
                AGUIEventType.TEXT_MESSAGE_CONTENT, run_id, session_id,
                messageId=msg_id, delta=chunk.delta_content,
            )

    yield build_event(
        AGUIEventType.TEXT_MESSAGE_END, run_id, session_id,
        messageId=msg_id,
    )

    await persist_session_summary(session_ctx, session_manager, session_id)

    yield build_event(
        AGUIEventType.RUN_FINISHED, run_id, session_id,
        outputText="".join(text_parts).strip(),
        reason=AgentFinishReason.FINAL_ANSWER.value,
        usage=step_usage,
    )


# ═══════════════════════════════════════════════════════════════════
# Non-streaming finalize
# ═══════════════════════════════════════════════════════════════════


async def emit_finalize(
    run_id: str,
    session_id: str,
    step: int,
    step_usage: dict[str, Any] | None,
    reason: AgentFinishReason,
    output_text: str,
    session_ctx: Any = None,
    session_manager: Any = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Emit STEP_FINISHED + RUN_FINISHED without an extra LLM call."""
    yield build_event(
        AGUIEventType.STEP_FINISHED, run_id, session_id,
        step=step, status="finalize",
    )

    await persist_session_summary(session_ctx, session_manager, session_id)

    yield build_event(
        AGUIEventType.RUN_FINISHED, run_id, session_id,
        outputText=output_text, reason=reason.value, usage=step_usage,
    )


# ═══════════════════════════════════════════════════════════════════
# Text message trio
# ═══════════════════════════════════════════════════════════════════


async def emit_text_message(
    run_id: str, session_id: str, text: str,
) -> AsyncGenerator[dict[str, Any], None]:
    """Yield TEXT_MESSAGE_START / CONTENT / END for a complete text block."""
    msg_id = new_run_id()
    yield build_event(
        AGUIEventType.TEXT_MESSAGE_START, run_id, session_id,
        messageId=msg_id, role="assistant",
    )
    yield build_event(
        AGUIEventType.TEXT_MESSAGE_CONTENT, run_id, session_id,
        messageId=msg_id, delta=text,
    )
    yield build_event(
        AGUIEventType.TEXT_MESSAGE_END, run_id, session_id,
        messageId=msg_id,
    )


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def inject_context_tokens(
    event: dict[str, Any], total_tokens: int,
) -> dict[str, Any]:
    """Attach contextTokens to RUN_FINISHED events."""
    if event.get("type") == AGUIEventType.RUN_FINISHED:
        event["contextTokens"] = total_tokens
    return event


async def persist_session_summary(
    session_ctx: Any, session_manager: Any, session_id: str,
) -> None:
    """Best-effort session summary persistence."""
    if not session_ctx or not session_manager:
        return
    try:
        summary = session_ctx.to_summary()
        if summary:
            await session_manager.update_session(session_id, description=summary)
    except Exception as e:
        logger.debug("Session summary persistence failed: {}", e)
