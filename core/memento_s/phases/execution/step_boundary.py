# SPDX-License-Identifier: Apache-2.0
"""Step boundary logic — reflection, replan routing, inter-step data injection, L1 update."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, AsyncGenerator

if TYPE_CHECKING:
    from core.context import ContextManager
    from ..planning import PlanStep
    from ..reflection import ReflectionResult

from infra.context.providers.shared_utils import build_digest
from core.prompts.templates import (
    FINALIZE_INSTRUCTION,
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
from .helpers import _append_messages, _live_ctx_tokens, _prepare_messages
from ...finalize import stream_and_finalize

logger = get_logger(__name__)


async def run_reflection(
    *,
    state: AgentRunState,
    current_ps: PlanStep,
    step_text: str,
    llm: LLMClient,
    emitter: RunEmitter,
    react_iteration: int = 0,
    max_react_per_step: int = 5,
) -> ReflectionResult:
    """Run reflection at step boundary and emit result event."""
    react_exhausted = react_iteration >= max_react_per_step
    combined_result = step_text
    if state.step_accumulated_results:
        combined_result += "\n\nTool results:\n" + "\n---\n".join(
            state.step_accumulated_results
        )
    if state.skill_failure_tracker:
        stats_lines = ["\n\n[Skill Execution Stats]"]
        for sname, errs in state.skill_failure_tracker.items():
            stats_lines.append(f"  {sname}: {len(errs)} consecutive failures ({', '.join(errs[-3:])})")
        combined_result += "\n".join(stats_lines)
    remaining = state.remaining_plan_steps()

    reflection = await reflect(
        plan=state.task_plan,
        current_step=current_ps,
        step_result=combined_result,
        remaining_steps=remaining,
        llm=llm,
        config=state.config,
        context_messages=state.messages,
        react_iteration=react_iteration,
        reflection_history=state.reflection_history,
        state=state,
    )

    if (
        reflection.decision == ReflectionDecision.REPLAN
        and "skill" in reflection.reason.lower()
    ):
        replan_msg = {
            "role": "system",
            "content": SKILL_CHECK_HINT_MSG.format(reason=reflection.reason),
        }
        state.messages = await _append_messages(None, state.messages, [replan_msg])

    # 追加到 reflection_history，限制最多保留最近 10 条
    state.reflection_history.append(reflection.decision.value)
    if len(state.reflection_history) > 10:
        state.reflection_history = state.reflection_history[-10:]

    return reflection


async def _handle_replan(
    *,
    state: AgentRunState,
    llm: LLMClient,
    session_goal_text: str,
    accumulated_content: str,
    emitter: RunEmitter,
    step: int,
    ctx: ContextManager | None = None,
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
        goal=session_goal_text,
        context=replan_context,
        llm=llm,
    )
    state.reset_for_replan(new_plan)

    yield emitter.plan_generated(**new_plan.to_event_payload(), replan=True)

    if accumulated_content:
        add_msg = {"role": "assistant", "content": accumulated_content}
        state.messages = await _append_messages(
            ctx, state.messages, [add_msg]
        )

    yield emitter.step_finished(step=step, status=StepStatus.CONTINUE)


def _extract_structured_output(raw_results: list[str], model: str = "") -> str:
    """Extract key fields from raw tool results for inter-step data passing.

    Uses build_digest for structured extraction — no truncation.
    Size control is delegated to the context pipeline (prepare_for_api).
    """
    parts: list[str] = []
    for r in raw_results:
        try:
            parsed = json.loads(r)
            if isinstance(parsed, dict):
                output_val = parsed.get("output")
                if isinstance(output_val, dict):
                    parts.append(build_digest(parsed, output_val))
                    continue
                fallback = parsed.get("summary") or parsed.get("output") or ""
                parts.append(str(fallback))
            else:
                parts.append(str(parsed))
        except (ValueError, TypeError):
            parts.append(r)
    return "\n---\n".join(parts)


async def _inject_step_results(
    state: AgentRunState,
    current_ps: PlanStep,
    reflection: ReflectionResult,
    ctx: ContextManager | None,
) -> None:
    """Inject completed-step results + trigger L1 SessionMemory LLM update."""
    if ctx is not None and hasattr(ctx, "session_memory"):
        try:
            sm = ctx.session_memory
            if hasattr(sm, "llm_update") and getattr(sm, "_react_since_llm_update", 0) > 0:
                await sm.llm_update(state.messages, str(current_ps.step_id))
        except Exception:
            logger.opt(exception=True).warning("Failed to trigger SM LLM update at step boundary")

    if state.step_accumulated_results:
        _model = getattr(ctx, "_model", "") if ctx is not None else ""
        summary = _extract_structured_output(state.step_accumulated_results, model=_model)
        state.last_step_summary = summary
        msg_text = STEP_COMPLETED_MSG.format(
            step_id=current_ps.step_id,
            results=summary,
        )
        if reflection.next_step_hint:
            msg_text += f"\n\nHint for next step: {reflection.next_step_hint}"
        step_msg = {"role": "system", "content": msg_text}
        state.messages = await _append_messages(
            ctx, state.messages, [step_msg]
        )
    elif reflection.next_step_hint:
        hint_msg = {
            "role": "system",
            "content": STEP_REFLECTION_HINT.format(reason=reflection.next_step_hint),
        }
        state.messages = await _append_messages(
            ctx, state.messages, [hint_msg]
        )

    # 产物注入：从上一步的 execute_skill 结果中提取产物，注入给当前步骤
    # 这是打通产物传递链的关键：让后续步骤知道前面创建了什么文件
    await _inject_previous_artifacts(state, ctx)


async def _inject_previous_artifacts(
    state: AgentRunState,
    ctx: ContextManager | None,
) -> None:
    """Extract and inject artifacts from previous step's execute_skill results.

    This bridges the artifact chain across step boundaries:
    - Step N's skill creates files
    - Files are tracked in execute_skill result's 'artifacts' field
    - Step N+1's prompt gains knowledge of those files via this injection
    """
    all_artifacts: list[str] = []
    seen: set[str] = set()

    # 遍历最近的 tool 消息，收集 artifacts
    # 从最新的往前找，找到一个 user 消息为止
    for msg in reversed(state.messages):
        msg_role = msg.get("role", "")
        if msg_role == "user":
            break
        if msg_role == "tool":
            try:
                content = msg.get("content", "")
                if isinstance(content, str):
                    payload = json.loads(content)
                else:
                    payload = content
                # skill_name 字段存在于 execute_skill 的 JSON 返回结果中
                if payload.get("skill_name") == "execute_skill":
                    artifacts = payload.get("artifacts", []) or []
                    for path in artifacts:
                        if path and path not in seen:
                            seen.add(path)
                            all_artifacts.append(path)
            except (json.JSONDecodeError, TypeError):
                pass

    if not all_artifacts:
        return

    lines = ["## Previous Steps Created Files (available for use)"]
    for path in reversed(all_artifacts):  # 最新的在前
        lines.append(f"- {path}")

    artifact_msg = {
        "role": "system",
        "content": "\n".join(lines),
    }
    await _append_messages(ctx, state.messages, [artifact_msg])


# ── Finalize ──────────────────────────────────────────────────────────────


async def _finalize_run(
    *,
    state: AgentRunState,
    llm: LLMClient,
    ctx: ContextManager | None,
    emitter: RunEmitter,
    step: int,
    step_usage: dict[str, Any] | None = None,
    context_tokens: int | None = None,
    result_info: dict[str, Any] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Inject FINALIZE_INSTRUCTION and stream the final answer."""
    finalize_msg = {"role": "system", "content": FINALIZE_INSTRUCTION}
    if not state.finalize_injected:
        state.messages = await _append_messages(
            ctx, state.messages, [finalize_msg]
        )
        state.finalize_injected = True

    final_messages, _ = await _prepare_messages(ctx, state.messages, state)

    live_tokens = _live_ctx_tokens(ctx) or context_tokens
    async for ev in stream_and_finalize(
        messages=final_messages,
        llm=llm,
        tools=None,
        emitter=emitter,
        step=step,
        step_usage=step_usage,
        context_tokens=live_tokens,
        session_ctx=ctx,
        result_info=result_info,
    ):
        yield ev
