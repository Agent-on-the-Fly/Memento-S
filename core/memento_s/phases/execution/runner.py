# SPDX-License-Identifier: Apache-2.0
"""Plan execution main loop — outer step loop + inner bounded react loop."""

from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING, Any, AsyncGenerator

if TYPE_CHECKING:
    from core.context import ContextManager

from core.prompts.templates import (
    EXEC_FAILURES_EXCEEDED_MSG,
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
from utils.debug_logger import log_agent_phase
from utils.logger import get_logger

from infra.context.providers.shared_utils import build_digest
from ...skill_dispatch import TOOL_ASK_USER, TOOL_EXECUTE_SKILL, SkillDispatcher
from ...utils import skill_call_to_openai_payload
from ..reflection import ReflectionDecision
from ..planning import PlanStep
from ..state import AgentRunState
from .helpers import _append_messages, _live_ctx_tokens, _prepare_messages
from .step_boundary import _finalize_run, _handle_replan, _inject_step_results, run_reflection
from .tool_handler import (
    _check_error_policy,
    _enforce_explicit_skill,
    _filter_blocked,
    _track_execute_result,
)

logger = get_logger(__name__)


def _build_input_summary(state: AgentRunState, current_ps: PlanStep) -> str:
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
    tool_dispatcher: SkillDispatcher,
    tool_schemas: list[dict[str, Any]],
    session_goal_text: str,
    emitter: RunEmitter,
    user_content: str,
    max_iter: int,
    ctx: ContextManager | None = None,
    context_tokens: int | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Execute a task plan: outer step loop -> inner react loop -> reflection."""
    cfg = state.config
    iteration = 0

    log_agent_phase(
        "EXECUTION_START",
        "",
        f"max_iter={max_iter}, steps={len(state.task_plan.steps) if state.task_plan else 0}",
    )

    while state.current_plan_step() is not None:
        current_ps = state.current_plan_step()
        import json as _json
        step_dump = {
            "step_id": current_ps.step_id,
            "action": current_ps.action,
            "expected_output": current_ps.expected_output,
            "skill_name": current_ps.skill_name,
            "skill_request": current_ps.skill_request,
            "input_from": current_ps.input_from,
        }
        logger.info(
            "=== EXECUTION_STEP_START: step_id={}, skill={}, total_steps={} ===\n"
            "action={!r}\n"
            "expected_output={!r}\n"
            "skill_request={!r}\n"
            "step_json={}",
            current_ps.step_id,
            current_ps.skill_name,
            len(state.task_plan.steps),
            current_ps.action,
            current_ps.expected_output,
            current_ps.skill_request,
            _json.dumps(step_dump, indent=2, ensure_ascii=False),
        )
        step_text = ""
        step_usage: dict[str, Any] | None = None

        # Fresh skill catalog — fetched at step start so newly downloaded skills are visible
        _step_skill_catalog = ""
        try:
            _manifests = await tool_dispatcher._gateway.discover()
            if _manifests:
                _lines = [f"- **{m.name}**: {m.description or 'no description'}" for m in _manifests]
                _step_skill_catalog = "\n\n## Available Local Skills\n" + "\n".join(_lines)
        except Exception:
            pass

        for _react_iter in range(cfg.max_react_per_step):
            iteration += 1
            pending_messages: list[dict[str, Any]] = []
            if iteration > max_iter:
                yield emitter.run_finished(
                    output_text=MAX_ITERATIONS_MSG,
                    reason=AgentFinishReason.MAX_ITERATIONS,
                    context_tokens=_live_ctx_tokens(ctx) or context_tokens,
                )
                return

            logger.info(
                "[Runner] REPL iter={}/{}, plan_step={}, messages={}, "
                "execute_failures={}, skill_failure_tracker={}, replan_count={}",
                iteration,
                cfg.max_react_per_step,
                current_ps.step_id,
                len(state.messages),
                state.execute_failures,
                dict(state.skill_failure_tracker),
                state.replan_count,
            )

            input_summary = _build_input_summary(state, current_ps)
            step_hint = {
                "role": "system",
                "content": STEP_GOAL_HINT.format(
                    skill_catalog=_step_skill_catalog,
                    step_id=current_ps.step_id,
                    action=current_ps.action,
                    expected_output=current_ps.expected_output,
                    skill_name=current_ps.skill_name or "decide based on available skills",
                    skill_request=current_ps.skill_request or "(agent decides)",
                    input_summary=input_summary or "none",
                ),
            }
            react_messages = list(state.messages) + [step_hint]

            # Pre-API Pipeline
            react_messages, _compact_trigger = await _prepare_messages(ctx, react_messages, state)

            yield emitter.step_started(
                step=iteration,
                name=f"step_{current_ps.step_id}_iter_{_react_iter + 1}",
            )

            try:
                response = await llm.async_chat(
                    messages=react_messages, tools=tool_schemas
                )
            except LLMContextWindowError:
                # Force compact via pipeline then retry
                react_messages = list(state.messages) + [step_hint]
                react_messages, _ = await _prepare_messages(
                    ctx, react_messages, state, force_compact=True
                )
                try:
                    response = await llm.async_chat(
                        messages=react_messages, tools=tool_schemas
                    )
                except LLMContextWindowError:
                    logger.error("Context window error persists after force compact")
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
                is_final = accumulated_content.strip().startswith("Final Answer:")
                if is_final:
                    if pending_messages:
                        state.messages = await _append_messages(ctx, state.messages, pending_messages)
                    yield emitter.step_finished(step=iteration, status=StepStatus.DONE)
                    break

                logger.warning(
                    "LLM returned text without tool calls or Final Answer prefix "
                    "(iter={}/{}): {:.80s}",
                    _react_iter + 1, cfg.max_react_per_step, accumulated_content,
                )
                nudge_msg = {"role": "system", "content": NO_TOOL_NO_FINAL_ANSWER_MSG}
                pending_messages.extend(
                    [{"role": "assistant", "content": accumulated_content}, nudge_msg]
                )
                # Worklog append: assistant text response
                if ctx is not None and hasattr(ctx, "session_memory") and hasattr(ctx.session_memory, "append_worklog_entry"):
                    topic = accumulated_content.strip()[:50].replace("\n", " ")
                    ctx.session_memory.append_worklog_entry(f"responded: {topic}")
                yield emitter.step_finished(step=iteration, status=StepStatus.CONTINUE)
                if pending_messages:
                    state.messages = await _append_messages(ctx, state.messages, pending_messages)
                continue

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": accumulated_content,
                "tool_calls": [skill_call_to_openai_payload(sc) for sc in skill_calls],
            }
            tool_msgs: list[dict[str, Any]] = []

            executable_calls: list[tuple[ToolCall, str]] = []
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
                        "Duplicate tool call detected: {}({}) repeated {:} times, skipping",
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

            for sc, display_name in executable_calls:
                yield emitter.tool_call_start(
                    step=iteration, call_id=sc.id,
                    name=display_name, args=sc.arguments,
                )

            async def _exec_one(sc: ToolCall) -> tuple[str, bool]:
                try:
                    r = await tool_dispatcher.execute(sc.name, sc.arguments)
                    return r, True
                except Exception as exc:
                    logger.exception("Tool execution failed: tool={}", str(sc.name))
                    return f"Error: {exc}", False

            if len(executable_calls) > 1:
                results_list = await asyncio.gather(
                    *[_exec_one(sc) for sc, _ in executable_calls]
                )
            else:
                results_list = [await _exec_one(sc) for sc, _ in executable_calls]

            abort_requested = False
            for (sc, display_name), (result, action_success) in zip(
                executable_calls, results_list
            ):
                state.record_tool_result(sc.name, sc.arguments, action_success, result)
                state.step_accumulated_results.append(result)

                if sc.name == TOOL_EXECUTE_SKILL:
                    _track_execute_result(state, sc, result)
                    try:
                        payload = json.loads(result)
                        logger.info(
                            "[Runner] execute_skill result: ok={}, status={}, summary='{}', output_type={}",
                            payload.get("ok"),
                            payload.get("status"),
                            str(payload.get("summary", ""))[:150],
                            type(payload.get("output")).__name__,
                        )
                    except Exception:
                        logger.warning(
                            "[Runner] execute_skill result parse failed: '{}'",
                            result[:200],
                        )

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
                    tool_msg = await ctx.persist_tool_result(sc.id, sc.name, result)
                    tool_msgs.append(tool_msg)

                    if hasattr(ctx, "session_memory") and hasattr(ctx.session_memory, "append_worklog_entry"):
                        if sc.name == TOOL_EXECUTE_SKILL:
                            try:
                                parsed = json.loads(result)
                                output_val = parsed.get("output", {})
                                if isinstance(output_val, dict):
                                    digest = build_digest(parsed, output_val)
                                else:
                                    digest = str(parsed.get("summary", ""))[:100]
                            except Exception:
                                digest = result[:100]
                            ctx.session_memory.append_worklog_entry(digest)
                        else:
                            brief_args = str(sc.arguments)[:60]
                            brief_status = "OK" if action_success else "FAIL"
                            ctx.session_memory.append_worklog_entry(
                                f"{sc.name}({brief_args}) → {brief_status}"
                            )
                else:
                    tool_msgs.append(
                        {"role": "tool", "tool_call_id": sc.id, "content": result}
                    )

            if abort_requested:
                if pending_messages:
                    state.messages = await _append_messages(ctx, state.messages, pending_messages)
                yield emitter.run_finished(
                    output_text=state.last_execute_error or "",
                    reason=AgentFinishReason.ERROR_POLICY_ABORT,
                    usage=step_usage,
                    context_tokens=_live_ctx_tokens(ctx) or context_tokens,
                )
                return

            pending_messages.extend([assistant_msg] + tool_msgs)

            for _sname, _errs in state.skill_failure_tracker.items():
                if len(_errs) >= 2:
                    hint_content = (
                        f"[SYSTEM WARNING] Skill '{_sname}' has failed {len(_errs)} "
                        f"consecutive times (errors: {', '.join(_errs[-3:])}). "
                        f"Consider: use different parameters, try a different skill, "
                        f"or check if the skill's prerequisites are met."
                    )
                    pending_messages.append({"role": "system", "content": hint_content})
                    break

            # Check if SM should trigger LLM update (N-react threshold)
            if ctx is not None and hasattr(ctx, "session_memory"):
                sm = ctx.session_memory
                if hasattr(sm, "should_llm_update") and sm.should_llm_update():
                    try:
                        await sm.llm_update(state.messages, str(current_ps.step_id))
                    except Exception:
                        logger.opt(exception=True).warning("SM LLM update failed in react loop")

            if state.should_stop_for_failures():
                yield emitter.step_finished(step=iteration, status=StepStatus.FINALIZE)
                fail_text = EXEC_FAILURES_EXCEEDED_MSG.format(
                    last_error=state.last_execute_error
                )
                from ...finalize import persist_session_summary

                await persist_session_summary(session_ctx=ctx)
                live_ctx_tokens = _live_ctx_tokens(ctx) or context_tokens
                yield emitter.run_error(
                    message=fail_text,
                    reason=AgentFinishReason.EXEC_FAILURES_EXCEEDED,
                    context_tokens=live_ctx_tokens,
                )
                yield emitter.run_finished(
                    output_text=fail_text,
                    reason=AgentFinishReason.EXEC_FAILURES_EXCEEDED,
                    usage=step_usage,
                    context_tokens=live_ctx_tokens,
                )
                return

            yield emitter.step_finished(step=iteration, status=StepStatus.CONTINUE)

            # ── Batch append all pending messages ──────────────────────────
            if pending_messages:
                state.messages = await _append_messages(
                    ctx, state.messages, pending_messages
                )

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

        logger.info(
            "[Runner] Reflection: decision={}, reason='{}', "
            "completed_step={}, next_hint='{}'",
            reflection.decision,
            str(reflection.reason)[:100],
            reflection.completed_step_id,
            str(reflection.next_step_hint)[:80] if reflection.next_step_hint else "",
        )

        if reflection.decision == ReflectionDecision.IN_PROGRESS:
            logger.info("Reflection: in_progress — stay on current step")
            state.step_accumulated_results = []
            continue

        if reflection.decision == ReflectionDecision.FINALIZE:
            state.advance_plan_step()
            result_info: dict[str, Any] = {}
            async for ev in _finalize_run(
                state=state,
                llm=llm,
                ctx=ctx,
                emitter=emitter,
                step=iteration,
                step_usage=step_usage,
                context_tokens=context_tokens,
                result_info=result_info,
            ):
                yield ev
            return

        if reflection.decision == ReflectionDecision.REPLAN:
            if state.can_replan():
                async for ev in _handle_replan(
                    state=state,
                    llm=llm,
                    session_goal_text=session_goal_text,
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
                    "Replan exhausted (count={:}), forcing continue",
                    state.replan_count,
                )
                reflection.decision = ReflectionDecision.CONTINUE

        if reflection.decision == ReflectionDecision.CONTINUE:
            state.advance_plan_step()
            remaining = state.remaining_plan_steps()
            logger.info(
                "[Runner] Step done: step_id={}, next_remaining={}, "
                "execute_failures={}, messages={}",
                current_ps.step_id,
                len(remaining),
                state.execute_failures,
                len(state.messages),
            )
            if not remaining:
                result_info: dict[str, Any] = {}
                async for ev in _finalize_run(
                    state=state,
                    llm=llm,
                    ctx=ctx,
                    emitter=emitter,
                    step=iteration,
                    step_usage=step_usage,
                    context_tokens=context_tokens,
                    result_info=result_info,
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
    result_info: dict[str, Any] = {}
    async for ev in _finalize_run(
        state=state,
        llm=llm,
        ctx=ctx,
        emitter=emitter,
        step=iteration,
        step_usage=step_usage,
        context_tokens=context_tokens,
        result_info=result_info,
    ):
        yield ev
