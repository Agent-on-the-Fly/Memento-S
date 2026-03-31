"""Execution helpers — message trimming, plan status rendering, append logic."""

from __future__ import annotations

from typing import Any

from core.prompts.templates import POST_COMPACTION_STATE
from utils.logger import get_logger

from ..state import AgentRunState

logger = get_logger(__name__)


def _trim_messages(
    messages: list[dict[str, Any]], keep_tail: int = 20
) -> list[dict[str, Any]]:
    """保留所有 system 消息 + 最后 keep_tail 条非 system 消息，用于上下文超限时裁剪。"""
    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]
    trimmed = non_system[-keep_tail:] if len(non_system) > keep_tail else non_system
    logger.warning(
        "Context window exceeded, trimming messages: %d → %d non-system messages kept",
        len(non_system),
        len(trimmed),
    )
    return system_msgs + trimmed


def _live_ctx_tokens(ctx: Any) -> int | None:
    """Return the live token count from ContextManager, or None."""
    if ctx is not None and hasattr(ctx, "total_tokens"):
        return ctx.total_tokens
    return None


def _build_plan_status(state: AgentRunState) -> str:
    """Build a structured plan status string from AgentRunState for compaction."""
    if not state.task_plan:
        return ""

    lines: list[str] = []
    for i, ps in enumerate(state.task_plan.steps):
        if i < state.current_plan_step_idx:
            lines.append(f"- Step {ps.step_id}: {ps.action} [DONE]")
        elif i == state.current_plan_step_idx:
            partial = ""
            if state.step_accumulated_results:
                partial = "; ".join(
                    r[:200] for r in state.step_accumulated_results[-3:]
                )
            lines.append(
                f"- Step {ps.step_id}: {ps.action} [IN PROGRESS]"
                + (f" — partial results: {partial}" if partial else "")
            )
        else:
            lines.append(f"- Step {ps.step_id}: {ps.action} [PENDING]")

    return f"Goal: {state.task_plan.goal}\n" + "\n".join(lines)


def _build_post_compaction_msg(state: AgentRunState) -> dict[str, Any] | None:
    """Build a structured post-compaction plan state message."""
    if not state.task_plan:
        return None

    completed_lines: list[str] = []
    current_line = "  (none)"
    remaining_lines: list[str] = []

    for i, ps in enumerate(state.task_plan.steps):
        if i < state.current_plan_step_idx:
            completed_lines.append(f"  - Step {ps.step_id}: {ps.action} [DONE]")
        elif i == state.current_plan_step_idx:
            partial = ""
            if state.step_accumulated_results:
                partial = "; ".join(
                    r[:300] for r in state.step_accumulated_results[-3:]
                )
            current_line = f"  - Step {ps.step_id}: {ps.action} [IN PROGRESS]" + (
                f"\n    Done so far: {partial}" if partial else ""
            )
        else:
            remaining_lines.append(f"  - Step {ps.step_id}: {ps.action} [PENDING]")

    content = POST_COMPACTION_STATE.format(
        goal=state.task_plan.goal,
        completed_steps="\n".join(completed_lines) if completed_lines else "  (none)",
        current_step=current_line,
        remaining_steps="\n".join(remaining_lines) if remaining_lines else "  (none)",
    )
    return {"role": "system", "content": content}


async def _append_messages(
    ctx: Any,
    messages: list[dict[str, Any]],
    new_msgs: list[dict[str, Any]],
    state: AgentRunState | None = None,
) -> list[dict[str, Any]]:
    """Append messages via ContextManager if available, else plain concat.

    When ``state`` is provided, builds plan_status for smarter compaction
    and injects a structured plan state message after compaction occurs.
    """
    if ctx is not None:
        plan_status = _build_plan_status(state) if state else ""
        result = await ctx.append(messages, new_msgs, plan_status=plan_status)

        if state and ctx.consume_compacted_flag():
            post_msg = _build_post_compaction_msg(state)
            if post_msg:
                result = list(result) + [post_msg]
                logger.info("Injected post-compaction plan state into messages")

        return result
    return list(messages) + new_msgs
