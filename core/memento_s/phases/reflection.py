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
from typing import Any

from pydantic import BaseModel

from core.prompts.templates import REFLECTION_PROMPT
from middleware.llm import LLMClient
from utils.debug_logger import log_agent_phase
from utils.logger import get_logger

from ..schemas import AgentConfig
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
    config: AgentConfig | None = None,
    context_messages: list[dict[str, Any]] | None = None,
    react_budget_exhausted: bool = False,
    react_iteration: int = 0,
    max_react_per_step: int = 5,
    replan_count: int = 0,
    max_replans: int = 2,
    reflection_history: list[str] | None = None,
) -> ReflectionResult:
    """Reflect on step execution with budget-aware constraints.

    Parameters:
        react_iteration / max_react_per_step: Per-step react loop bounds.
        replan_count / max_replans: Global replan counters.
        reflection_history: Previous reflection decisions for dedup.
    """
    cfg = config or AgentConfig()
    _reflection_history = reflection_history or []

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

    except Exception as e:
        logger.warning("Reflection failed, defaulting: {}", e)
        if remaining_steps:
            fallback = ReflectionDecision.REPLAN if _looks_like_error(step_result) else ReflectionDecision.CONTINUE
            return ReflectionResult(
                decision=fallback,
                reason=f"Reflection error ({e}), falling back to {fallback}",
                completed_step_id=current_step.step_id,
            )
        return ReflectionResult(
            decision=ReflectionDecision.FINALIZE,
            reason=f"Reflection error ({e}), no remaining steps",
            completed_step_id=current_step.step_id,
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
