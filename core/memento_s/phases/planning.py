"""Phase: Plan generation — decompose a task into executable steps.

Owns PlanStep and TaskPlan (co-located to avoid circular imports).
"""

from __future__ import annotations

from datetime import datetime

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


class TaskPlan(BaseModel):
    goal: str
    steps: list[PlanStep] = Field(default_factory=list)

    def to_event_payload(self) -> dict:
        """Canonical dict for PLAN_GENERATED events — single source of truth."""
        return {
            "goal": self.goal,
            "steps": [
                {"step_id": s.step_id, "action": s.action, "expected_output": s.expected_output}
                for s in self.steps
            ],
        }


# ═══════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════


async def generate_plan(goal: str, context: str, llm: LLMClient) -> TaskPlan:
    """Generate a task plan with human-readable action steps."""
    log_agent_phase("PLAN_LLM_CALL", "system", f"goal_len={len(goal)}")
    now = datetime.now()
    prompt = PLAN_GENERATION_PROMPT.format(
        goal=goal,
        context=context or "(no additional context)",
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
