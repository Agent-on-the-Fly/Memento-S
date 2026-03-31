"""Shared finalize helpers — the *Reporter* role.

Responsibilities:
  - Summarize execution results for the user
  - Generate the final reply
  - Persist session summary

Does NOT: execute new actions, modify the plan.
"""

from __future__ import annotations

from typing import Any, AsyncGenerator

from core.protocol import AgentFinishReason, RunEmitter, StepStatus
from middleware.llm import LLMClient
from middleware.llm.utils import (
    looks_like_tool_call_text,
    sanitize_content,
)
from shared.chat import ChatManager
from utils.logger import get_logger

logger = get_logger(__name__)


EMPTY_REPLY_FALLBACK = (
    "（模型未返回有效内容，请重试或换个方式提问。）\n"
    "(The model did not return a valid response. Please try again or rephrase.)"
)

_FINALIZE_RETRY_INSTRUCTION = (
    "Please answer the user's question directly using only the information "
    "available in the conversation above and your own knowledge. "
    "Respond in PLAIN TEXT only — do NOT output any tool calls, function "
    "invocations, XML tags, or control tokens. If you do not have enough "
    "information to answer precisely, say so honestly and suggest what "
    "the user can do."
)


def to_enriched_summary(session_ctx: Any, state: Any = None) -> str:
    """Generate an enriched session summary including plan completion and skills used.

    Falls back to ``session_ctx.to_summary()`` if state is unavailable.
    """
    base = session_ctx.to_summary() if hasattr(session_ctx, "to_summary") else ""
    if state is None:
        return base

    extra_parts: list[str] = []

    task_plan = getattr(state, "task_plan", None)
    if task_plan:
        statuses = getattr(state, "plan_step_statuses", [])
        done = sum(1 for s in statuses if str(s) == "done")
        total = len(task_plan.steps)
        extra_parts.append(f"Plan: {done}/{total} steps completed")

    action_history = getattr(session_ctx, "action_history", [])
    skill_names = sorted({
        r.skill_name for r in action_history
        if getattr(r, "skill_name", "")
    })
    if skill_names:
        extra_parts.append(f"Skills: {', '.join(skill_names[:8])}")

    if extra_parts:
        return f"{base} | {' | '.join(extra_parts)}" if base else " | ".join(extra_parts)
    return base


async def _retry_finalize_plain_text(
    llm: LLMClient,
    messages: list[dict[str, Any]],
) -> str | None:
    """Non-streaming retry when the initial finalize produced only raw tokens.

    Returns cleaned text or None if retry also fails.
    """
    retry_messages = list(messages) + [
        {"role": "system", "content": _FINALIZE_RETRY_INSTRUCTION},
    ]
    try:
        resp = await llm.async_chat(messages=retry_messages, tools=None, max_tokens=512)
        text = (resp.content or "").strip()
        if looks_like_tool_call_text(text):
            text = sanitize_content(text).strip()
        return text or None
    except Exception as exc:
        logger.warning("Finalize plain-text retry failed: {}", exc)
        return None


async def stream_and_finalize(
    *,
    messages: list[dict[str, Any]],
    llm: LLMClient,
    tools: list[dict[str, Any]] | None,
    emitter: RunEmitter,
    step: int,
    step_usage: dict[str, Any] | None = None,
    session_ctx: Any = None,
    context_tokens: int | None = None,
    state: Any = None,
    result_info: dict[str, Any] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Stream an LLM response and emit STEP_FINISHED + TEXT_MESSAGE + RUN_FINISHED.

    Guarantees text_message_end and run_finished are always emitted via try/finally.
    """
    yield emitter.step_finished(step=step, status=StepStatus.FINALIZE)

    msg_id = emitter.new_message_id()
    yield emitter.text_message_start(message_id=msg_id, role="assistant")

    text_parts: list[str] = []
    clean_parts: list[str] = []
    full_text = ""
    got_tool_calls = False
    try:
        async for chunk in llm.async_stream_chat(messages=messages, tools=tools):
            if chunk.usage:
                step_usage = chunk.usage
            if chunk.delta_tool_call:
                got_tool_calls = True
            if chunk.delta_content:
                text_parts.append(chunk.delta_content)
                cleaned_delta = sanitize_content(chunk.delta_content)
                if cleaned_delta:
                    clean_parts.append(cleaned_delta)
                    yield emitter.text_delta(message_id=msg_id, delta=cleaned_delta)

        if got_tool_calls:
            logger.warning(
                "Finalize: got_tool_calls=True, discarding {} streamed text parts "
                "as likely raw tool token contamination",
                len(clean_parts),
            )
            clean_parts.clear()
            text_parts.clear()
            if result_info is not None:
                result_info["got_tool_calls"] = True

        clean_streamed = "".join(clean_parts).strip()
        if clean_streamed:
            if clean_streamed.startswith("Final Answer:"):
                clean_streamed = clean_streamed.removeprefix("Final Answer:").lstrip()
            if looks_like_tool_call_text(clean_streamed):
                logger.warning(
                    "Finalize: clean_streamed still has raw tool tokens after per-chunk sanitize, re-sanitizing"
                )
                clean_streamed = sanitize_content(clean_streamed).strip()
            full_text = clean_streamed
        else:
            raw_text = "".join(text_parts).strip()
            if raw_text:
                logger.warning("Finalize: clean_parts empty but text_parts has content, sanitizing raw")
                if raw_text.startswith("Final Answer:"):
                    raw_text = raw_text.removeprefix("Final Answer:").lstrip()
                full_text = sanitize_content(raw_text).strip()
            else:
                full_text = ""

        if not full_text and got_tool_calls:
            full_text = (
                "这个问题需要访问文件系统或执行操作才能回答，请重新发送您的请求。"
            )
            yield emitter.text_delta(message_id=msg_id, delta=full_text)
        elif not full_text:
            logger.info("Finalize produced empty content after sanitization, retrying with plain-text instruction")
            retry_text = await _retry_finalize_plain_text(llm, messages)
            if retry_text:
                full_text = retry_text
                yield emitter.text_delta(message_id=msg_id, delta=full_text)

        if not full_text:
            full_text = EMPTY_REPLY_FALLBACK
            yield emitter.text_delta(message_id=msg_id, delta=full_text)

    except Exception as e:
        logger.exception("stream_and_finalize streaming error: {}", e)
        full_text = full_text or "".join(text_parts).strip() or EMPTY_REPLY_FALLBACK
    finally:
        yield emitter.text_message_end(message_id=msg_id)

        await persist_session_summary(session_ctx, state=state)

        yield emitter.run_finished(
            output_text=full_text,
            reason=AgentFinishReason.FINAL_ANSWER,
            usage=step_usage,
            context_tokens=context_tokens,
        )


async def persist_session_summary(
    session_ctx: Any,
    state: Any = None,
) -> None:
    """Best-effort session summary persistence with enriched data."""
    if not session_ctx:
        return
    try:
        summary = to_enriched_summary(session_ctx, state=state)
        if summary:
            sid = getattr(session_ctx, "session_id", None)
            if sid:
                await ChatManager.update_session(sid, description=summary)
    except Exception as e:
        logger.error("Session summary persistence failed: {}", e)
