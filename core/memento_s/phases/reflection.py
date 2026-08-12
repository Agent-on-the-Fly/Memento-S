# SPDX-License-Identifier: Apache-2.0
"""Phase: Reflection — the *Supervisor* role.

Responsibilities:
  - Evaluate execution results
  - Make decisions under global resource budget constraints
  - Detect when user input is needed (ASK_USER)

Does NOT: execute, understand intent.
"""

from __future__ import annotations

import json
import re
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..state import AgentRunState

from pydantic import BaseModel, ValidationError

from core.prompts.templates import REFLECTION_PROMPT
from middleware.llm import LLMClient
from utils.debug_logger import log_agent_phase
from utils.logger import get_logger

from ..schemas import AgentRuntimeConfig
from ..utils import extract_json
from .planning import PlanStep, TaskPlan

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════


class ReflectionDecision(StrEnum):
    CONTINUE = "continue"
    IN_PROGRESS = "in_progress"
    REPLAN = "replan"
    FINALIZE = "finalize"
    ASK_USER = "ask_user"


class ReflectionResult(BaseModel):
    """Output of step-level reflection."""

    decision: ReflectionDecision
    reason: str = ""
    next_step_hint: str | None = None
    completed_step_id: int | None = None
    ask_user_question: str = ""


# ═══════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════


async def reflect(
    plan: TaskPlan,
    current_step: PlanStep,
    step_result: str,
    remaining_steps: list[PlanStep],
    llm: LLMClient,
    config: AgentRuntimeConfig | None = None,
    context_messages: list[dict[str, Any]] | None = None,
    react_iteration: int = 0,
    reflection_history: list[str] | None = None,
    state: AgentRunState | None = None,
) -> ReflectionResult:
    """Reflect on step execution with budget-aware constraints.

    Parameters:
        react_iteration: Current react iteration within the step.
        reflection_history: Previous reflection decisions for dedup.
        state: AgentRunState — when provided, max_react_per_step, replan_count,
            and max_replans are read from state.config; otherwise defaults are used.

    Constraint overrides (applied internally):
        - IN_PROGRESS → CONTINUE when react budget exhausted
        - REPLAN → CONTINUE when replan budget exhausted
    """
    cfg = config or AgentRuntimeConfig()
    _reflection_history = reflection_history or []

    # Derive constraint values from state when available
    if state is not None:
        max_react_per_step = state.config.max_react_per_step
        react_budget_exhausted = react_iteration >= max_react_per_step
        replan_count = state.replan_count
        max_replans = state.config.max_replans
    else:
        max_react_per_step = 5
        react_budget_exhausted = False
        replan_count = 0
        max_replans = 2

    plan_str = "\n".join(
        f"  Step {s.step_id}: {s.action} -> {s.expected_output}"
        for s in plan.steps
    )
    remaining_str = (
        "\n".join(f"  Step {s.step_id}: {s.action}" for s in remaining_steps)
        or "(none — all steps completed)"
    )

    exec_state_lines = [
        "## Execution State",
        f"- React: {react_iteration}/{max_react_per_step}"
        + (" [EXHAUSTED]" if react_budget_exhausted else ""),
        f"- Replan: {replan_count}/{max_replans}"
        + (" [EXHAUSTED]" if replan_count >= max_replans else ""),
    ]
    if _reflection_history:
        exec_state_lines.append(f"- Previous decisions: {', '.join(_reflection_history[-5:])}")

    exec_state_block = "\n".join(exec_state_lines)

    prompt = REFLECTION_PROMPT.format(
        plan=f"Goal: {plan.goal}\n{plan_str}",
        current_step=(
            f"Step {current_step.step_id}: {current_step.action} "
            f"(expected: {current_step.expected_output})"
        ),
        step_result=step_result[: cfg.reflection_input_chars],
        remaining_steps=remaining_str,
        execution_state=exec_state_block,
    )

    if context_messages:
        messages = list(context_messages) + [{"role": "user", "content": prompt}]
    else:
        messages = [{"role": "user", "content": prompt}]

    try:
        resp = await llm.async_chat(
            messages=messages,
            temperature=0,
            max_tokens=cfg.reflection_max_tokens,
        )
        raw = (resp.content or "").strip()
        data = extract_json(raw)

        if "completed_step_id" in data:
            step_id = data["completed_step_id"]
            if isinstance(step_id, str):
                match = re.search(r"\d+", step_id)
                data["completed_step_id"] = int(match.group()) if match else None

        # ── Field guard: if decision missing, try to infer from other fields ──
        if "decision" not in data:
            logger.warning(
                "[Reflection] LLM response missing 'decision' field. "
                "Got keys: {}. Falling back to CONTINUE.",
                list(data.keys()),
            )
            is_error = _looks_like_error(step_result)
            fallback = ReflectionDecision.REPLAN if is_error else ReflectionDecision.CONTINUE
            return ReflectionResult(
                decision=fallback,
                reason=(
                    f"LLM response missing 'decision'. "
                    f"Falling back to {fallback}. Response keys: {list(data.keys())}"
                ),
                completed_step_id=current_step.step_id if current_step else None,
            )

        result = ReflectionResult(**data)

        # ── Hard constraints override ──
        if react_budget_exhausted and result.decision == ReflectionDecision.IN_PROGRESS:
            logger.info("Overriding IN_PROGRESS → CONTINUE (react budget exhausted)")
            result.decision = ReflectionDecision.CONTINUE
            result.reason = f"React budget exhausted. {result.reason}"

        if replan_count >= max_replans and result.decision == ReflectionDecision.REPLAN:
            logger.info("Overriding REPLAN → CONTINUE (replan budget exhausted)")
            result.decision = ReflectionDecision.CONTINUE
            result.reason = f"Replan budget exhausted. {result.reason}"

        log_agent_phase(
            "REFLECTION_RESULT", "system",
            f"decision={result.decision}, step={result.completed_step_id}",
        )
        return result

    except ValidationError as e:
        # Pydantic validation error — LLM returned malformed JSON structure
        logger.warning(
            "[Reflection] ValidationError: {}. Falling back.",
            e,
        )
        is_error = _looks_like_error(step_result)
        fallback = ReflectionDecision.REPLAN if is_error else ReflectionDecision.CONTINUE
        return ReflectionResult(
            decision=fallback,
            reason=f"Reflection ValidationError ({e}), falling back to {fallback}",
            completed_step_id=current_step.step_id if current_step else None,
        )

    except (ValueError, json.JSONDecodeError) as e:
        # extract_json failed — LLM didn't return valid JSON
        logger.warning(
            "[Reflection] LLM did not return valid JSON ({}). Falling back to CONTINUE.",
            e,
        )
        is_error = _looks_like_error(step_result)
        fallback = ReflectionDecision.REPLAN if is_error else ReflectionDecision.CONTINUE
        return ReflectionResult(
            decision=fallback,
            reason=f"Reflection LLM response invalid JSON ({e}), falling back to {fallback}",
            completed_step_id=current_step.step_id if current_step else None,
        )

    except Exception as e:
        logger.warning("Reflection failed, defaulting: {}", e)
        is_error = _looks_like_error(step_result)
        fallback = ReflectionDecision.REPLAN if is_error else ReflectionDecision.CONTINUE
        if remaining_steps:
            return ReflectionResult(
                decision=fallback,
                reason=f"Reflection error ({e}), falling back to {fallback}",
                completed_step_id=current_step.step_id if current_step else None,
            )
        return ReflectionResult(
            decision=ReflectionDecision.FINALIZE,
            reason=f"Reflection error ({e}), no remaining steps",
            completed_step_id=current_step.step_id if current_step else None,
        )


def _looks_like_error(text: str) -> bool:
    """Heuristic: check if step output is dominated by error signals."""
    stripped = text.strip()
    if not stripped:
        return True
    lower = stripped.lower()
    if lower.startswith(("error", "traceback", "exception", "fatal")):
        return True
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict) and parsed.get("ok") is False:
            return True
    except (json.JSONDecodeError, TypeError):
        pass
    return False
