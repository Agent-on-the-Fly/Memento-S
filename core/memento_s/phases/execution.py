"""Phase: Hierarchical execution — outer plan-step loop + inner bounded react loop.

Replaces the flat ``react_loop.py``.  Key differences:
  1. Outer loop iterates over plan steps; inner loop bounded by ``max_react_per_step``.
  2. Reflection happens only at step boundaries, not after every tool call.
  3. All runtime messages use templates (no hardcoded strings).
"""

from __future__ import annotations

import json
import re
from typing import Any, AsyncGenerator

from core.manager.session_context import ActionRecord, SessionContext
from core.prompts.templates import (
    ERROR_POLICY_MSG,
    EXEC_FAILURES_EXCEEDED_MSG,
    FINALIZE_INSTRUCTION,
    MAX_ITERATIONS_MSG,
    SKILL_CHECK_HINT_MSG,
    STEP_COMPLETED_MSG,
    STEP_GOAL_HINT,
    STEP_REFLECTION_HINT,
)
from core.skill.execution.error_policy import ErrorAction, ErrorPolicy
from middleware.llm import LLMClient
from middleware.llm.schema import ToolCall
from middleware.llm.utils import looks_like_tool_call_text
from utils.debug_logger import log_agent_phase
from utils.logger import get_logger

from .state import AgentRunState
from ..emitters import emit_finalize, emit_text_message, stream_and_finalize
from ..stream_output import AGUIEventType, AgentFinishReason, build_event, new_run_id
from ..tools import TOOL_EXECUTE_SKILL, TOOL_SEARCH_SKILL, ToolDispatcher
from ..utils import can_direct_execute_skill, skill_call_to_openai_payload
from .planning import generate_plan
from .reflection import ReflectionDecision, reflect

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════


async def run_plan_execution(
    *,
    state: AgentRunState,
    llm: LLMClient,
    tool_dispatcher: ToolDispatcher,
    tool_schemas: list[dict[str, Any]],
    session_ctx: SessionContext,
    session_id: str,
    run_id: str,
    user_content: str,
    max_iter: int,
    session_manager: Any,
    ctx: Any = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Execute a task plan: outer step loop → inner react loop → reflection.

    ``ctx`` is an optional ``ContextManager`` for persist / compress.
    """
    cfg = state.config
    iteration = 0

    log_agent_phase(
        "EXECUTION_START", session_id,
        f"max_iter={max_iter}, steps={len(state.task_plan.steps) if state.task_plan else 0}",
    )

    while state.current_plan_step() is not None:
        current_ps = state.current_plan_step()
        step_text = ""
        step_usage: dict[str, Any] | None = None

        # ── Inner react loop for this plan step ────────────────────────
        for _react_iter in range(cfg.max_react_per_step):
            iteration += 1
            if iteration > max_iter:
                yield build_event(
                    AGUIEventType.RUN_FINISHED, run_id, session_id,
                    outputText=MAX_ITERATIONS_MSG,
                    reason=AgentFinishReason.MAX_ITERATIONS.value,
                )
                return

            step_hint = {
                "role": "system",
                "content": STEP_GOAL_HINT.format(
                    step_id=current_ps.step_id,
                    action=current_ps.action,
                    expected_output=current_ps.expected_output,
                ),
            }
            react_messages = list(state.messages) + [step_hint]

            yield build_event(
                AGUIEventType.STEP_STARTED, run_id, session_id,
                step=iteration, name=f"step_{current_ps.step_id}_iter_{_react_iter + 1}",
            )

            response = await llm.async_chat(messages=react_messages, tools=tool_schemas)
            accumulated_content = response.content or ""
            collected_tool_calls: list[ToolCall] = response.tool_calls or []
            step_usage = response.usage

            # Retry when LLM emits tool-call-like text but no actual tool_calls
            if (
                response.finish_reason == "length"
                and not collected_tool_calls
                and looks_like_tool_call_text(accumulated_content)
            ):
                retry_resp = await llm.async_chat(messages=react_messages, tools=tool_schemas)
                accumulated_content = retry_resp.content or ""
                collected_tool_calls = retry_resp.tool_calls or []
                step_usage = retry_resp.usage

            # Emit text content
            display = re.sub(r"[ \t]+", " ", accumulated_content.strip()) if accumulated_content else ""
            if display:
                async for ev in emit_text_message(run_id, session_id, display):
                    yield ev
                step_text = display

            # Filter blocked skills
            skill_calls = _filter_blocked(collected_tool_calls, state.blocked_skills)

            # Explicit skill enforcement
            skill_calls = _enforce_explicit_skill(
                skill_calls, state, user_content, tool_dispatcher, session_id,
            )

            # No tool calls → step done
            if not skill_calls:
                yield build_event(
                    AGUIEventType.STEP_FINISHED, run_id, session_id,
                    step=iteration, status="done",
                )
                break

            # Execute tool calls
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": accumulated_content,
                "tool_calls": [skill_call_to_openai_payload(sc) for sc in skill_calls],
            }
            tool_msgs: list[dict[str, Any]] = []

            for sc in skill_calls:
                display_name = sc.arguments.get("skill_name", sc.name) if sc.name == TOOL_EXECUTE_SKILL else sc.name

                yield build_event(
                    AGUIEventType.TOOL_CALL_START, run_id, session_id,
                    step=iteration, toolCallId=sc.id, toolName=display_name, arguments=sc.arguments,
                )

                try:
                    result = await tool_dispatcher.execute(sc.name, sc.arguments)
                    action_success = True
                except Exception as e:
                    result = f"Error: {e}"
                    action_success = False
                    logger.exception("Tool execution failed: step={}, tool={}", iteration, display_name)

                if sc.name == TOOL_EXECUTE_SKILL:
                    async for ev in _check_error_policy(result, run_id, session_id, iteration):
                        yield ev
                        if ev.get("type") == AGUIEventType.RUN_FINISHED:
                            return

                session_ctx.add_action(ActionRecord.from_tool_call(
                    tool_name=sc.name, args=sc.arguments, result=result, success=action_success,
                ))
                state.step_accumulated_results.append(result)

                if sc.name == TOOL_EXECUTE_SKILL:
                    _track_execute_result(state, sc, result)

                yield build_event(
                    AGUIEventType.TOOL_CALL_RESULT, run_id, session_id,
                    step=iteration, toolCallId=sc.id, toolName=display_name, result=result,
                )

                if ctx is not None:
                    tool_msgs.append(ctx.persist_tool_result(sc.id, sc.name, result))
                else:
                    tool_msgs.append({"role": "tool", "tool_call_id": sc.id, "content": result})

            state.messages = await _append_messages(ctx, state.messages, [assistant_msg] + tool_msgs)

            # Failure bail-out
            if state.should_stop_for_failures():
                async for ev in emit_finalize(
                    run_id, session_id, iteration, step_usage,
                    reason=AgentFinishReason.EXEC_FAILURES_EXCEEDED,
                    output_text=EXEC_FAILURES_EXCEEDED_MSG.format(last_error=state.last_execute_error),
                ):
                    yield ev
                return

            yield build_event(
                AGUIEventType.STEP_FINISHED, run_id, session_id,
                step=iteration, status="continue",
            )

        # ── Reflection at step boundary ────────────────────────────────
        combined_result = step_text
        if state.step_accumulated_results:
            combined_result += "\n\nTool results:\n" + "\n---\n".join(state.step_accumulated_results)
        remaining = state.remaining_plan_steps()

        reflection = await reflect(
            plan=state.task_plan,
            current_step=current_ps,
            step_result=combined_result,
            remaining_steps=remaining,
            llm=llm,
            config=cfg,
        )

        # Skill fallback hint
        if reflection.decision == ReflectionDecision.REPLAN and "skill" in reflection.reason.lower():
            state.messages.append({
                "role": "system",
                "content": SKILL_CHECK_HINT_MSG.format(reason=reflection.reason),
            })

        yield build_event(
            AGUIEventType.REFLECTION_RESULT, run_id, session_id,
            decision=reflection.decision, reason=reflection.reason,
            completedStepId=reflection.completed_step_id,
            nextStepHint=reflection.next_step_hint,
        )

        if reflection.decision == ReflectionDecision.FINALIZE:
            session_ctx.mark_step_done(state.current_plan_step_idx)
            async for ev in _finalize_run(
                state=state, llm=llm, ctx=ctx,
                run_id=run_id, session_id=session_id,
                step=iteration, step_usage=step_usage,
                session_ctx=session_ctx, session_manager=session_manager,
            ):
                yield ev
            return

        if reflection.decision == ReflectionDecision.REPLAN:
            if state.can_replan():
                async for ev in _handle_replan(
                    state=state, llm=llm, session_ctx=session_ctx,
                    accumulated_content=step_text, run_id=run_id,
                    session_id=session_id, step=iteration, ctx=ctx,
                    reason=reflection.reason,
                ):
                    yield ev
                continue  # restart outer loop with new plan
            else:
                logger.warning(
                    "Replan exhausted (count={}), forcing continue",
                    state.replan_count,
                )
                reflection.decision = ReflectionDecision.CONTINUE

        if reflection.decision == ReflectionDecision.CONTINUE:
            session_ctx.mark_step_done(state.current_plan_step_idx)
            if not remaining:
                async for ev in _finalize_run(
                    state=state, llm=llm, ctx=ctx,
                    run_id=run_id, session_id=session_id,
                    step=iteration, step_usage=step_usage,
                    session_ctx=session_ctx, session_manager=session_manager,
                ):
                    yield ev
                return

            await _inject_step_results(
                state, current_ps, reflection, ctx,
            )
            state.advance_plan_step()

    # ── All steps completed — streaming finalize ───────────────────────
    async for ev in _finalize_run(
        state=state, llm=llm, ctx=ctx,
        run_id=run_id, session_id=session_id,
        step=iteration, step_usage=step_usage,
        session_ctx=session_ctx, session_manager=session_manager,
    ):
        yield ev


# ═══════════════════════════════════════════════════════════════════
# Replan
# ═══════════════════════════════════════════════════════════════════


async def _handle_replan(
    *,
    state: AgentRunState,
    llm: LLMClient,
    session_ctx: SessionContext,
    accumulated_content: str,
    run_id: str,
    session_id: str,
    step: int,
    ctx: Any = None,
    reason: str = "",
) -> AsyncGenerator[dict[str, Any], None]:
    """Generate new plan and reset state. Messages already contain prior tool results."""
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
    session_ctx.set_plan([f"Step {s.step_id}: {s.action}" for s in new_plan.steps])

    yield build_event(
        AGUIEventType.PLAN_GENERATED, run_id, session_id,
        **new_plan.to_event_payload(), replan=True,
    )

    if accumulated_content:
        add_msg = {"role": "assistant", "content": accumulated_content}
        state.messages = await _append_messages(ctx, state.messages, [add_msg])

    yield build_event(
        AGUIEventType.STEP_FINISHED, run_id, session_id, step=step, status="continue",
    )


# ═══════════════════════════════════════════════════════════════════
# Finalize
# ═══════════════════════════════════════════════════════════════════


async def _finalize_run(
    *,
    state: AgentRunState,
    llm: LLMClient,
    ctx: Any,
    run_id: str,
    session_id: str,
    step: int,
    step_usage: dict[str, Any] | None = None,
    session_ctx: SessionContext,
    session_manager: Any,
) -> AsyncGenerator[dict[str, Any], None]:
    """Inject FINALIZE_INSTRUCTION and stream the final answer."""
    finalize_msg = {"role": "system", "content": FINALIZE_INSTRUCTION}
    state.messages = await _append_messages(ctx, state.messages, [finalize_msg])
    async for ev in stream_and_finalize(
        messages=state.messages, llm=llm, tools=None,
        run_id=run_id, session_id=session_id, step=step,
        step_usage=step_usage,
        session_ctx=session_ctx, session_manager=session_manager,
    ):
        yield ev


# ═══════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════


def _filter_blocked(
    tool_calls: list[ToolCall], blocked: set[str],
) -> list[ToolCall]:
    if not blocked:
        return tool_calls
    return [tc for tc in tool_calls if tc.name not in blocked]


def _enforce_explicit_skill(
    skill_calls: list[ToolCall],
    state: AgentRunState,
    user_content: str,
    tool_dispatcher: ToolDispatcher,
    session_id: str,
) -> list[ToolCall]:
    """Enforce explicit skill intent on first execute_skill call."""
    if not state.explicit_skill_name or state.explicit_skill_retry_done:
        return skill_calls

    has_execute = any(sc.name == TOOL_EXECUTE_SKILL for sc in skill_calls)
    has_search = any(sc.name == TOOL_SEARCH_SKILL for sc in skill_calls)

    if has_execute and not has_search:
        execute_tc = next(sc for sc in skill_calls if sc.name == TOOL_EXECUTE_SKILL)
        if can_direct_execute_skill(user_content, execute_tc.arguments):
            tool_dispatcher.mark_session_searched(session_id)
            state.explicit_skill_retry_done = True
            return skill_calls

        state.explicit_skill_retry_done = True
        return [
            ToolCall(
                id=new_run_id(),
                name=TOOL_SEARCH_SKILL,
                arguments={"query": state.explicit_skill_name, "k": 8},
            )
        ]

    return skill_calls


async def _check_error_policy(
    result: str, run_id: str, session_id: str, step: int,
) -> AsyncGenerator[dict[str, Any], None]:
    """Check error policy on execute_skill result; yield RUN_FINISHED if abort/prompt."""
    try:
        payload = json.loads(result)
        diagnostics = payload.get("diagnostics") if isinstance(payload, dict) else None
        decision = ErrorPolicy.decide_from_diagnostics(
            diagnostics,
            success=bool(payload.get("ok")) if isinstance(payload, dict) else False,
            fallback_error=str(payload.get("summary")) if isinstance(payload, dict) else None,
        )
        if decision and decision.action in {ErrorAction.PROMPT_USER, ErrorAction.ABORT}:
            final_text = ERROR_POLICY_MSG.format(
                action=decision.action.value, reason=decision.reason,
            )
            yield build_event(
                AGUIEventType.STEP_FINISHED, run_id, session_id, step=step, status="finalize",
            )
            yield build_event(
                AGUIEventType.RUN_FINISHED, run_id, session_id,
                outputText=final_text,
                reason=f"execute_skill_{decision.action.value}",
            )
    except Exception:
        pass


def _track_execute_result(state: AgentRunState, sc: ToolCall, result: str) -> None:
    """Track blocked skills and execution failure counts."""
    try:
        payload = json.loads(result)
        summary = str(payload.get("summary", ""))
        output_text = str(payload.get("output", ""))
        if "[NOT_RELEVANT]" in summary or "[NOT_RELEVANT]" in output_text:
            state.blocked_skills.add(sc.arguments.get("skill_name", ""))
        if not payload.get("ok", False):
            state.execute_failures += 1
            state.last_execute_error = summary
        else:
            state.execute_failures = 0
    except Exception:
        state.execute_failures += 1
        state.last_execute_error = result[:200]


async def _inject_step_results(
    state: AgentRunState,
    current_ps: Any,
    reflection: Any,
    ctx: Any,
) -> None:
    """Inject completed-step results into messages for the next step."""
    if state.step_accumulated_results:
        summary = "\n---\n".join(state.step_accumulated_results)
        msg_text = STEP_COMPLETED_MSG.format(
            step_id=current_ps.step_id, results=summary,
        )
        if reflection.next_step_hint:
            msg_text += f"\n\nHint for next step: {reflection.next_step_hint}"
        step_msg = {"role": "system", "content": msg_text}
        state.messages = await _append_messages(ctx, state.messages, [step_msg])
    elif reflection.next_step_hint:
        hint_msg = {
            "role": "system",
            "content": STEP_REFLECTION_HINT.format(reason=reflection.next_step_hint),
        }
        state.messages = await _append_messages(ctx, state.messages, [hint_msg])


async def _append_messages(
    ctx: Any,
    messages: list[dict[str, Any]],
    new_msgs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Append messages via ContextManager if available, else plain concat."""
    if ctx is not None:
        return await ctx.append(messages, new_msgs)
    return list(messages) + new_msgs


