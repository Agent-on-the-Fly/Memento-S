"""Phase: Plan generation — the *Architect* role.

Responsibilities:
  - Decompose tasks into executable steps at single-skill granularity
  - Assign skills and pre-fill skill_request per step
  - Define data flow between steps (input_from)
  - Flatten nested skill dependencies into linear steps
  - Validate the resulting plan

Does NOT: execute, understand intent.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from core.prompts.templates import PLAN_GENERATION_PROMPT
from middleware.llm import LLMClient
from utils.debug_logger import log_agent_phase
from utils.logger import get_logger

from ..utils import extract_json

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════


class PlanStep(BaseModel):
    step_id: int
    action: str
    expected_output: str = ""
    skill_name: str | None = None
    skill_request: str = ""
    input_from: list[int] = Field(default_factory=list)
    requires_user_input: bool = False


class TaskPlan(BaseModel):
    goal: str
    steps: list[PlanStep] = Field(default_factory=list)

    def to_event_payload(self) -> dict:
        """Canonical dict for PLAN_GENERATED events — single source of truth."""
        return {
            "goal": self.goal,
            "steps": [
                {
                    "step_id": s.step_id,
                    "action": s.action,
                    "expected_output": s.expected_output,
                    "skill_name": s.skill_name,
                    "skill_request": s.skill_request,
                    "input_from": s.input_from,
                }
                for s in self.steps
            ],
        }


@dataclass
class SkillBrief:
    """Lightweight skill descriptor for the planner prompt."""

    name: str
    description: str
    parameters: dict[str, Any] | None = None


@dataclass
class PlanContext:
    """All context the planner needs to generate a good plan."""

    environment_summary: str = ""
    available_skills: list[SkillBrief] = dc_field(default_factory=list)
    history_summary: str = ""
    replan_context: str | None = None


# ═══════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════


def validate_plan(plan: TaskPlan, available_skill_names: set[str]) -> TaskPlan:
    """Post-process and validate a generated plan.

    - Nullify unknown skill_name values
    - Remove invalid input_from references
    - Re-number step_id sequentially
    """
    valid_ids = {s.step_id for s in plan.steps}
    for step in plan.steps:
        if step.skill_name and step.skill_name not in available_skill_names:
            logger.warning(
                "Plan step %d references unknown skill '%s', clearing",
                step.step_id, step.skill_name,
            )
            step.skill_name = None
        step.input_from = [ref for ref in step.input_from if ref in valid_ids and ref < step.step_id]

    for i, step in enumerate(plan.steps):
        step.step_id = i + 1

    return plan


# ═══════════════════════════════════════════════════════════════════
# Skill catalog formatting
# ═══════════════════════════════════════════════════════════════════


def _format_skill_catalog(skills: list[SkillBrief]) -> str:
    """Render skill briefs into a prompt-ready catalog."""
    if not skills:
        return "(no skills installed)"
    lines: list[str] = []
    for s in skills:
        line = f"- **{s.name}**: {s.description}"
        if s.parameters:
            params = ", ".join(
                f"{k}: {v}" for k, v in list(s.parameters.items())[:5]
            )
            line += f" (params: {params})"
        lines.append(line)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════


async def generate_plan(
    goal: str,
    context: str | PlanContext,
    llm: LLMClient,
) -> TaskPlan:
    """Generate a task plan with skill assignments and data flow."""
    log_agent_phase("PLAN_LLM_CALL", "system", f"goal_len={len(goal)}")
    now = datetime.now()

    if isinstance(context, PlanContext):
        skill_catalog = _format_skill_catalog(context.available_skills)
        ctx_text = context.environment_summary
        if context.history_summary:
            ctx_text += f"\n\nRecent conversation:\n{context.history_summary}"
        if context.replan_context:
            ctx_text += f"\n\nReplan context:\n{context.replan_context}"
    else:
        skill_catalog = "(no skill catalog provided)"
        ctx_text = context or "(no additional context)"

    prompt = PLAN_GENERATION_PROMPT.format(
        goal=goal,
        context=ctx_text,
        skill_catalog=skill_catalog,
        current_datetime=now.strftime("%Y-%m-%d %H:%M:%S"),
        current_year=str(now.year),
    )

    try:
        resp = await llm.async_chat(messages=[{"role": "user", "content": prompt}])
        raw = (resp.content or "").strip()
        data = extract_json(raw)
        plan = TaskPlan(**data)
        log_agent_phase(
            "PLAN_RESULT", "system",
            f"steps={len(plan.steps)}, goal={plan.goal[:60]}",
        )
        return plan

    except Exception as e:
        logger.warning("Plan generation failed, single-step fallback: {}", e)
        return TaskPlan(
            goal=goal,
            steps=[PlanStep(step_id=1, action=goal, expected_output="Complete user request")],
        )
