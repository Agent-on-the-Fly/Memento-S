"""Step boundary logic — reflection, replan routing, and inter-step data injection."""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator

from core.manager.session_context import SessionContext
from core.prompts.templates import (
    SKILL_CHECK_HINT_MSG,
    STEP_COMPLETED_MSG,
    STEP_REFLECTION_HINT,
)
from core.protocol import RunEmitter, StepStatus
from middleware.llm import LLMClient
from utils.logger import get_logger

from ..planning import generate_plan
from ..reflection import ReflectionDecision, reflect
from ..state import AgentRunState
from .helpers import _append_messages

logger = get_logger(__name__)


async def run_reflection(
    *,
    state: AgentRunState,
    current_ps: Any,
    step_text: str,
    llm: LLMClient,
    emitter: RunEmitter,
    react_iteration: int = 0,
    max_react_per_step: int = 5,
) -> Any:
    """Run reflection at step boundary and emit result event."""
    react_exhausted = react_iteration >= max_react_per_step
    combined_result = step_text
    if state.step_accumulated_results:
        combined_result += "\n\nTool results:\n" + "\n---\n".join(
            state.step_accumulated_results
        )
    remaining = state.remaining_plan_steps()

    reflection = await reflect(
        plan=state.task_plan,
        current_step=current_ps,
        step_result=combined_result,
        remaining_steps=remaining,
        llm=llm,
        config=state.config,
        context_messages=state.messages,
        react_budget_exhausted=react_exhausted,
        react_iteration=react_iteration,
        max_react_per_step=max_react_per_step,
        replan_count=state.replan_count,
        max_replans=state.config.max_replans,
        reflection_history=state.reflection_history,
    )

    if (
        reflection.decision == ReflectionDecision.REPLAN
        and "skill" in reflection.reason.lower()
    ):
        state.messages.append(
            {
                "role": "system",
                "content": SKILL_CHECK_HINT_MSG.format(reason=reflection.reason),
            }
        )

    return reflection


async def _handle_replan(
    *,
    state: AgentRunState,
    llm: LLMClient,
    session_ctx: SessionContext,
    accumulated_content: str,
    emitter: RunEmitter,
    step: int,
    ctx: Any = None,
    reason: str = "",
) -> AsyncGenerator[dict[str, Any], None]:
    """Generate new plan and reset state."""
    lines: list[str] = []
    for i in range(state.current_plan_step_idx + 1):
        ps = state.task_plan.steps[i]
        tag = "[FAILED]" if i == state.current_plan_step_idx else "[DONE]"
        lines.append(f"- Step {ps.step_id}: {ps.action} {tag}")
    done_summary = "\n".join(lines)

    replan_context = (
        f"Previously attempted steps:\n{done_summary}"
        f"\n\nReason for replan: {reason or 'replanning needed'}"
    )

    new_plan = await generate_plan(
        goal=state.task_plan.goal,
        context=replan_context,
        llm=llm,
    )
    state.reset_for_replan(new_plan)
    state.sync_plan_state(session_ctx)

    yield emitter.plan_generated(**new_plan.to_event_payload(), replan=True)

    if accumulated_content:
        add_msg = {"role": "assistant", "content": accumulated_content}
        state.messages = await _append_messages(
            ctx, state.messages, [add_msg], state=state
        )

    yield emitter.step_finished(step=step, status=StepStatus.CONTINUE)


def _extract_structured_output(raw_results: list[str]) -> str:
    """Extract key fields from raw tool results for inter-step data passing."""
    parts: list[str] = []
    for r in raw_results:
        try:
            parsed = json.loads(r)
            if isinstance(parsed, dict):
                output = parsed.get("output") or parsed.get("summary") or parsed.get("result")
                if output:
                    parts.append(str(output)[:500])
                    continue
        except (ValueError, TypeError):
            pass
        parts.append(r[:500] if len(r) > 500 else r)
    return "\n---\n".join(parts)


async def _inject_step_results(
    state: AgentRunState,
    current_ps: Any,
    reflection: Any,
    ctx: Any,
) -> None:
    """Inject completed-step results into messages for the next step."""
    if state.step_accumulated_results:
        summary = _extract_structured_output(state.step_accumulated_results)
        msg_text = STEP_COMPLETED_MSG.format(
            step_id=current_ps.step_id,
            results=summary,
        )
        if reflection.next_step_hint:
            msg_text += f"\n\nHint for next step: {reflection.next_step_hint}"
        step_msg = {"role": "system", "content": msg_text}
        state.messages = await _append_messages(
            ctx, state.messages, [step_msg], state=state
        )
    elif reflection.next_step_hint:
        hint_msg = {
            "role": "system",
            "content": STEP_REFLECTION_HINT.format(reason=reflection.next_step_hint),
        }
        state.messages = await _append_messages(
            ctx, state.messages, [hint_msg], state=state
        )
