"""Phase: Reflection — post-execution decision making.

Owns ReflectionResult (co-located to avoid circular imports).
"""

from __future__ import annotations

import re
from enum import StrEnum

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
    REPLAN = "replan"
    FINALIZE = "finalize"


class ReflectionResult(BaseModel):
    """Output of step-level reflection."""

    decision: ReflectionDecision
    reason: str = ""
    next_step_hint: str | None = None
    completed_step_id: int | None = None


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
) -> ReflectionResult:
    """Reflect on step execution and decide next action.

    Returns one of: continue / replan / finalize.
    """
    cfg = config or AgentConfig()

    plan_str = "\n".join(
        f"  Step {s.step_id}: {s.action} -> {s.expected_output}"
        for s in plan.steps
    )
    remaining_str = (
        "\n".join(f"  Step {s.step_id}: {s.action}" for s in remaining_steps)
        or "(none — all steps completed)"
    )

    prompt = REFLECTION_PROMPT.format(
        plan=f"Goal: {plan.goal}\n{plan_str}",
        current_step=(
            f"Step {current_step.step_id}: {current_step.action} "
            f"(expected: {current_step.expected_output})"
        ),
        step_result=step_result[: cfg.reflection_input_chars],
        remaining_steps=remaining_str,
    )

    try:
        resp = await llm.async_chat(
            messages=[{"role": "user", "content": prompt}],
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
    return lower.startswith("error") or lower.startswith("traceback")
