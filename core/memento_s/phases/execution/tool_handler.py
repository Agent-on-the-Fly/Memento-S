# SPDX-License-Identifier: Apache-2.0
"""Tool call filtering, enforcement, error policy, and result tracking."""

from __future__ import annotations

import json
from typing import Any

from core.protocol import AgentFinishReason, RunEmitter, StepStatus
from core.skill.execution.policy.recovery import RecoveryPolicy
from core.skill.execution.policy.types import RecoveryAction
from middleware.llm.schema import ToolCall
from utils.logger import get_logger

from core.prompts.templates import ERROR_POLICY_MSG
from ..state import AgentRunState
from ...skill_dispatch import TOOL_EXECUTE_SKILL, TOOL_SEARCH_SKILL, SkillDispatcher
from ...utils import can_direct_execute_skill

logger = get_logger(__name__)


def _filter_blocked(
    tool_calls: list[ToolCall],
    blocked: set[str],
) -> list[ToolCall]:
    if not blocked:
        return tool_calls
    return [tc for tc in tool_calls if tc.name not in blocked]


def _enforce_explicit_skill(
    skill_calls: list[ToolCall],
    state: AgentRunState,
    user_content: str,
    tool_dispatcher: SkillDispatcher,
) -> list[ToolCall]:
    """Enforce explicit skill intent on first execute_skill call."""
    if not state.explicit_skill_name or state.explicit_skill_retry_done:
        return skill_calls

    has_execute = any(sc.name == TOOL_EXECUTE_SKILL for sc in skill_calls)
    has_search = any(sc.name == TOOL_SEARCH_SKILL for sc in skill_calls)

    if has_execute and not has_search:
        execute_tc = next(sc for sc in skill_calls if sc.name == TOOL_EXECUTE_SKILL)
        if can_direct_execute_skill(user_content, execute_tc.arguments):
            state.explicit_skill_retry_done = True

    return skill_calls


def _check_error_policy(
    result: str,
    emitter: RunEmitter,
    step: int,
    *,
    context_tokens: int | None = None,
) -> tuple[bool, list[dict[str, Any]]]:
    """Check error policy on execute_skill result.

    Returns ``(should_abort, events)`` so the caller never inspects wire-format types.
    """
    try:
        payload = json.loads(result)
        diagnostics = payload.get("diagnostics") if isinstance(payload, dict) else None
        decision = RecoveryPolicy.decide_from_diagnostics(
            diagnostics,
            success=bool(payload.get("ok")) if isinstance(payload, dict) else False,
            fallback_error=str(payload.get("summary"))
            if isinstance(payload, dict)
            else None,
        )
        if decision and decision.action in {
            RecoveryAction.PROMPT_USER,
            RecoveryAction.ABORT,
        }:
            error_text = ERROR_POLICY_MSG.format(
                action=decision.action.value,
                reason=decision.reason,
            )
            events = [
                emitter.step_finished(step=step, status=StepStatus.FINALIZE),
                # 错误时使用 run_error 事件，type 为 RUN_ERROR
                emitter.run_error(
                    message=error_text,
                    reason=AgentFinishReason.ERROR_POLICY_ABORT
                    if decision.action == RecoveryAction.ABORT
                    else AgentFinishReason.ERROR_POLICY_PROMPT,
                    context_tokens=context_tokens,
                ),
            ]
            return True, events

        # 产物存在时的 CONTINUE 动作：有产物但未完全成功，允许继续到下一步
        # 不中止执行，让 step boundary 处理继续
        if decision and decision.action == RecoveryAction.CONTINUE:
            return False, []  # should_abort=False, 不发送事件，继续执行
    except Exception:
        pass
    return False, []


def _track_execute_result(state: AgentRunState, sc: ToolCall, result: str) -> None:
    """Track blocked skills, execution failure counts, and per-skill failure patterns."""
    skill_name = sc.arguments.get("skill_name", "")
    try:
        payload = json.loads(result)
        summary = str(payload.get("summary", ""))
        output_text = str(payload.get("output", ""))
        if "[NOT_RELEVANT]" in summary or "[NOT_RELEVANT]" in output_text:
            state.blocked_skills.add(skill_name)
        if not payload.get("ok", False):
            state.execute_failures += 1
            state.last_execute_error = summary
            error_type = str(payload.get("error_code", "UNKNOWN"))
            state.skill_failure_tracker.setdefault(skill_name, []).append(error_type)
        else:
            state.execute_failures = 0
            state.skill_failure_tracker.pop(skill_name, None)
    except Exception:
        state.execute_failures += 1
        state.last_execute_error = result[:200]
        state.skill_failure_tracker.setdefault(skill_name, []).append("PARSE_ERROR")
