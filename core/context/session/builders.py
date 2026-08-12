# SPDX-License-Identifier: Apache-2.0
"""core.context.session.builders — prompt-building helpers for session-level data."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import SessionGoal


def build_session_context_block(session_context: SessionGoal | None, user_content: str) -> str:
    """Build a concise session-context block for the intent prompt.

    Args:
        session_context: Current SessionGoal, or None.
        user_content: The latest user message (used to detect intent shifts).
    """
    if session_context is None:
        return "- No active session context"

    from .types import RECENT_ACTIONS_INTENT

    lines: list[str] = []

    goal = session_context.text
    if goal and goal.strip() != user_content.strip():
        lines.append(f"- Current session goal: {goal[:150]}")

    # 判断是否存在进行中的多步任务：session_goal.text 非空代表有活跃目标
    has_plan = bool(session_context and session_context.text and session_context.text.strip())
    if hasattr(session_context, "plan_step_count"):
        plan_count: int = getattr(session_context, "plan_step_count", 0)
        if plan_count:
            statuses = getattr(session_context, "_plan_statuses", [])
            done = sum(1 for s in statuses if str(s) == "done")
            lines.append(f"- Active task plan: {done}/{plan_count} steps completed")
    lines.append(f"- Multi-step task running: {'YES' if has_plan else 'no'}")

    return "\n".join(lines) if lines else "- No active session context"
