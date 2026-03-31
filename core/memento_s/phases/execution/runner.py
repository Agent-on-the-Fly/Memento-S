"""Plan execution main loop — outer step loop + inner bounded react loop."""

from __future__ import annotations

import asyncio
import re
from typing import Any, AsyncGenerator

from core.manager.session_context import ActionRecord, SessionContext
from core.prompts.templates import (
    EXEC_FAILURES_EXCEEDED_MSG,
    FINALIZE_INSTRUCTION,
    MAX_ITERATIONS_MSG,
    NO_TOOL_NO_FINAL_ANSWER_MSG,
    STEP_GOAL_HINT,
)
from core.protocol import AgentFinishReason, RunEmitter, StepStatus
from middleware.llm import LLMClient
from middleware.llm.exceptions import LLMContextWindowError
from middleware.llm.schema import ToolCall
from middleware.llm.utils import (
    looks_like_tool_call_text,
    sanitize_content,
)
from core.context.block import make_event
from utils.debug_logger import log_agent_phase
from utils.logger import get_logger

from ...finalize import persist_session_summary, stream_and_finalize
from ...tools import TOOL_ASK_USER, TOOL_EXECUTE_SKILL, ToolDispatcher
from ...utils import skill_call_to_openai_payload
from ..reflection import ReflectionDecision
from ..state import AgentRunState
from .helpers import _append_messages, _live_ctx_tokens, _trim_messages
from .step_boundary import _handle_replan, _inject_step_results, run_reflection
from .tool_handler import (
    _check_error_policy,
    _enforce_explicit_skill,
    _filter_blocked,
    _track_execute_result,
)

logger = get_logger(__name__)


def _build_input_summary(state: AgentRunState, current_ps: Any) -> str:
    """Build a summary of outputs from steps referenced by ``input_from``."""
    if not hasattr(current_ps, "input_from") or not current_ps.input_from:
        return ""
    if not state.task_plan:
        return ""

    parts: list[str] = []
    step_results_by_id: dict[int, list[str]] = {}

    for i, step in enumerate(state.task_plan.steps):
        if i < state.current_plan_step_idx:
            step_results_by_id[step.step_id] = []

    for ref_id in current_ps.input_from:
        if ref_id in step_results_by_id:
            parts.append(f"[Step {ref_id} output available in context]")
        else:
            parts.append(f"[Step {ref_id} not yet completed]")

    return "; ".join(parts) if parts else ""


async def run_plan_execution(
    *,
    state: AgentRunState,
    llm: LLMClient,
    tool_dispatcher: ToolDispatcher,
    tool_schemas: list[dict[str, Any]],
    session_ctx: SessionContext,
    emitter: RunEmitter,
    user_content: str,
    max_iter: int,
    ctx: Any = None,
    context_tokens: int | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Execute a task plan: outer step loop -> inner react loop -> reflection."""
    cfg = state.config
    iteration = 0

    log_agent_phase(
        "EXECUTION_START",
        getattr(session_ctx, "session_id", ""),
        f"max_iter={max_iter}, steps={len(state.task_plan.steps) if state.task_plan else 0}",
    )

    while state.current_plan_step() is not None:
        current_ps = state.current_plan_step()
        step_text = ""
        step_usage: dict[str, Any] | None = None

        for _react_iter in range(cfg.max_react_per_step):
            iteration += 1
            if iteration > max_iter:
                yield emitter.run_finished(
                    output_text=MAX_ITERATIONS_MSG,
                    reason=AgentFinishReason.MAX_ITERATIONS,
                    context_tokens=_live_ctx_tokens(ctx) or context_tokens,
                )
                return

            input_summary = _build_input_summary(state, current_ps)
            step_hint = {
                "role": "system",
                "content": STEP_GOAL_HINT.format(
                    step_id=current_ps.step_id,
                    action=current_ps.action,
                    expected_output=current_ps.expected_output,
                    skill_name=current_ps.skill_name or "decide based on available skills",
                    skill_request=current_ps.skill_request or "(agent decides)",
                    input_summary=input_summary or "none",
                ),
            }
            react_messages = list(state.messages) + [step_hint]

            yield emitter.step_started(
                step=iteration,
                name=f"step_{current_ps.step_id}_iter_{_react_iter + 1}",
            )

            try:
                response = await llm.async_chat(
                    messages=react_messages, tools=tool_schemas
                )
            except LLMContextWindowError:
                for keep_tail in (20, 10, 4):
                    state.messages = _trim_messages(state.messages, keep_tail=keep_tail)
                    react_messages = list(state.messages) + [step_hint]
                    try:
                        response = await llm.async_chat(
                            messages=react_messages, tools=tool_schemas
                        )
                        break
                    except LLMContextWindowError:
                        if keep_tail == 4:
                            raise
            accumulated_content = response.content or ""
            collected_tool_calls: list[ToolCall] = response.tool_calls or []
            step_usage = response.usage

            display = (
                ""
                if looks_like_tool_call_text(accumulated_content)
                else sanitize_content(
                    re.sub(r"[ \t]+", " ", accumulated_content.strip())
                ).strip()
                if accumulated_content
                else ""
            )
            # Strip "Final Answer:" prefix for user-facing display
            if display and display.lstrip().startswith("Final Answer:"):
                display = display.lstrip().removeprefix("Final Answer:").lstrip()
            if display:
                msg_id = emitter.new_message_id()
                yield emitter.text_message_start(message_id=msg_id, role="assistant")
                yield emitter.text_delta(message_id=msg_id, delta=display)
                yield emitter.text_message_end(message_id=msg_id)
                step_text = display

            skill_calls = _filter_blocked(collected_tool_calls, state.blocked_skills)

            skill_calls = _enforce_explicit_skill(
                skill_calls,
                state,
                user_content,
                tool_dispatcher,
            )

            if not skill_calls:
                # Check if LLM signaled completion with "Final Answer:" prefix
                is_final = accumulated_content.strip().startswith("Final Answer:")
                if is_final:
                    yield emitter.step_finished(step=iteration, status=StepStatus.DONE)
                    break

                # No tool calls AND no "Final Answer:" — nudge LLM to continue
                logger.warning(
                    "LLM returned text without tool calls or Final Answer prefix "
                    "(iter=%d/%d): %.80s",
                    _react_iter + 1, cfg.max_react_per_step, accumulated_content,
                )
                nudge_msg = {"role": "system", "content": NO_TOOL_NO_FINAL_ANSWER_MSG}
                state.messages = await _append_messages(
                    ctx, state.messages,
                    [{"role": "assistant", "content": accumulated_content}, nudge_msg],
                    state=state,
                )
                yield emitter.step_finished(step=iteration, status=StepStatus.CONTINUE)
                continue

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": accumulated_content,
                "tool_calls": [skill_call_to_openai_payload(sc) for sc in skill_calls],
            }
            tool_msgs: list[dict[str, Any]] = []

            # Pre-filter: ask_user, duplicates
            executable_calls: list[tuple[ToolCall, str]] = []  # (call, display_name)
            for sc in skill_calls:
                if sc.name == TOOL_ASK_USER:
                    question = sc.arguments.get("question", "Could you provide more information?")
                    state.pending_ask_user_call_id = sc.id
                    yield emitter.user_input_requested(question=question)
                    msg_id = emitter.new_message_id()
                    yield emitter.text_message_start(message_id=msg_id, role="assistant")
                    yield emitter.text_delta(message_id=msg_id, delta=question)
                    yield emitter.text_message_end(message_id=msg_id)
                    yield emitter.run_finished(
                        output_text=question,
                        reason=AgentFinishReason.FINAL_ANSWER,
                        context_tokens=_live_ctx_tokens(ctx) or context_tokens,
                    )
                    return

                display_name = (
                    sc.arguments.get("skill_name", sc.name)
                    if sc.name == TOOL_EXECUTE_SKILL
                    else sc.name
                )

                dup_count, last_success = state.check_duplicate_call(
                    sc.name, sc.arguments
                )
                if dup_count > cfg.max_duplicate_tool_calls and last_success:
                    logger.warning(
                        "Duplicate tool call detected: {}({}) repeated {} times, skipping",
                        sc.name, display_name, dup_count,
                    )
                    tool_msgs.append({
                        "role": "tool",
                        "tool_call_id": sc.id,
                        "content": f"Blocked: identical tool call succeeded {dup_count} times with same result. "
                        "Change parameters, try a different skill, or proceed to the next step.",
                    })
                    continue

                executable_calls.append((sc, display_name))

            # Emit tool_call_start for all + record block events
            for sc, display_name in executable_calls:
                yield emitter.tool_call_start(
                    step=iteration, call_id=sc.id,
                    name=display_name, args=sc.arguments,
                )
                # Record tool_call event in block
                if ctx is not None:
                    args_str = str(sc.arguments)
                    if len(args_str) > 200:
                        args_str = args_str[:197] + "..."
                    ctx.append_block_event(make_event(
                        "tool_call",
                        tool_name=sc.name,
                        args_summary=args_str,
                        extra={"tool_call_id": sc.id},
                    ))

            # Execute: parallel if multiple independent calls, else sequential
            async def _exec_one(sc: ToolCall) -> tuple[str, bool]:
                try:
                    r = await tool_dispatcher.execute(sc.name, sc.arguments)
                    return r, True
                except Exception as exc:
                    logger.exception("Tool execution failed: tool=%s", sc.name)
                    return f"Error: {exc}", False

            if len(executable_calls) > 1:
                results_list = await asyncio.gather(
                    *[_exec_one(sc) for sc, _ in executable_calls]
                )
            else:
                results_list = [await _exec_one(sc) for sc, _ in executable_calls]

            # Process results sequentially
            abort_requested = False
            for (sc, display_name), (result, action_success) in zip(
                executable_calls, results_list
            ):
                state.record_tool_result(sc.name, sc.arguments, action_success, result)
                session_ctx.add_action(
                    ActionRecord.from_tool_call(
                        tool_name=sc.name, args=sc.arguments,
                        result=result, success=action_success,
                    )
                )
                state.step_accumulated_results.append(result)

                if sc.name == TOOL_EXECUTE_SKILL:
                    _track_execute_result(state, sc, result)

                yield emitter.tool_call_result(
                    step=iteration, call_id=sc.id,
                    name=display_name, result=result,
                )

                if sc.name == TOOL_EXECUTE_SKILL:
                    should_abort, abort_events = _check_error_policy(
                        result, emitter, iteration,
                        context_tokens=_live_ctx_tokens(ctx) or context_tokens,
                    )
                    for ev in abort_events:
                        yield ev
                    if should_abort:
                        abort_requested = True
                        break

                if ctx is not None:
                    # persist_tool_result handles fold + block event recording in one step
                    tool_msg = ctx.persist_tool_result(
                        sc.id, sc.name, result,
                        status="effective" if action_success else "ineffective",
                    )
                    tool_msgs.append(tool_msg)
                else:
                    tool_msgs.append(
                        {"role": "tool", "tool_call_id": sc.id, "content": result}
                    )

            if abort_requested:
                return

            state.messages = await _append_messages(
                ctx, state.messages, [assistant_msg] + tool_msgs, state=state
            )

            # ── Block event compaction (bounded mode) ──
            if ctx is not None and hasattr(ctx, "compact_active_block_if_needed"):
                ctx.compact_active_block_if_needed()

            if state.should_stop_for_failures():
                yield emitter.step_finished(step=iteration, status=StepStatus.FINALIZE)
                fail_text = EXEC_FAILURES_EXCEEDED_MSG.format(
                    last_error=state.last_execute_error
                )
                fail_msg_id = emitter.new_message_id()
                yield emitter.text_message_start(
                    message_id=fail_msg_id, role="assistant"
                )
                yield emitter.text_delta(message_id=fail_msg_id, delta=fail_text)
                yield emitter.text_message_end(message_id=fail_msg_id)
                await persist_session_summary(session_ctx)
                yield emitter.run_finished(
                    output_text=fail_text,
                    reason=AgentFinishReason.EXEC_FAILURES_EXCEEDED,
                    usage=step_usage,
                    context_tokens=_live_ctx_tokens(ctx) or context_tokens,
                )
                return

            yield emitter.step_finished(step=iteration, status=StepStatus.CONTINUE)

        # ── Reflection at step boundary ────────────────────────────────
        react_used = _react_iter + 1
        reflection = await run_reflection(
            state=state,
            current_ps=current_ps,
            step_text=step_text,
            llm=llm,
            emitter=emitter,
            react_iteration=react_used,
            max_react_per_step=cfg.max_react_per_step,
        )

        yield emitter.reflection_result(
            decision=reflection.decision,
            reason=reflection.reason,
            completed_step_id=reflection.completed_step_id,
            next_step_hint=reflection.next_step_hint,
        )

        if reflection.decision == ReflectionDecision.IN_PROGRESS:
            logger.info("Reflection: in_progress — stay on current step")
            state.step_accumulated_results = []
            continue

        if reflection.decision == ReflectionDecision.FINALIZE:
            session_ctx.mark_step_done(state.current_plan_step_idx)
            async for ev in _finalize_run(
                state=state,
                llm=llm,
                ctx=ctx,
                emitter=emitter,
                step=iteration,
                step_usage=step_usage,
                session_ctx=session_ctx,
                context_tokens=context_tokens,
            ):
                yield ev
            return

        if reflection.decision == ReflectionDecision.REPLAN:
            if state.can_replan():
                async for ev in _handle_replan(
                    state=state,
                    llm=llm,
                    session_ctx=session_ctx,
                    accumulated_content=step_text,
                    emitter=emitter,
                    step=iteration,
                    ctx=ctx,
                    reason=reflection.reason,
                ):
                    yield ev
                continue
            else:
                logger.warning(
                    "Replan exhausted (count={}), forcing continue",
                    state.replan_count,
                )
                reflection.decision = ReflectionDecision.CONTINUE

        if reflection.decision == ReflectionDecision.CONTINUE:
            session_ctx.mark_step_done(state.current_plan_step_idx)
            remaining = state.remaining_plan_steps()
            if not remaining:
                async for ev in _finalize_run(
                    state=state,
                    llm=llm,
                    ctx=ctx,
                    emitter=emitter,
                    step=iteration,
                    step_usage=step_usage,
                    session_ctx=session_ctx,
                    context_tokens=context_tokens,
                ):
                    yield ev
                return

            await _inject_step_results(
                state,
                current_ps,
                reflection,
                ctx,
            )
            state.advance_plan_step()

    # ── All steps completed — streaming finalize ───────────────────────
    async for ev in _finalize_run(
        state=state,
        llm=llm,
        ctx=ctx,
        emitter=emitter,
        step=iteration,
        step_usage=step_usage,
        session_ctx=session_ctx,
        context_tokens=context_tokens,
    ):
        yield ev


async def _finalize_run(
    *,
    state: AgentRunState,
    llm: LLMClient,
    ctx: Any,
    emitter: RunEmitter,
    step: int,
    step_usage: dict[str, Any] | None = None,
    session_ctx: SessionContext,
    context_tokens: int | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Inject FINALIZE_INSTRUCTION and stream the final answer."""
    finalize_msg = {"role": "system", "content": FINALIZE_INSTRUCTION}
    state.messages = await _append_messages(
        ctx, state.messages, [finalize_msg], state=state
    )
    live_tokens = _live_ctx_tokens(ctx) or context_tokens
    async for ev in stream_and_finalize(
        messages=state.messages,
        llm=llm,
        tools=None,
        emitter=emitter,
        step=step,
        step_usage=step_usage,
        session_ctx=session_ctx,
        context_tokens=live_tokens,
    ):
        yield ev
