"""Mutable state for a single agent run (one ``reply_stream`` invocation)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..schemas import AgentConfig
from ..tools import TOOL_EXECUTE_SKILL
from .intent import IntentMode
from .planning import PlanStep, TaskPlan


@dataclass
class AgentRunState:
    """Encapsulates the mutable state tracked across execution iterations."""

    config: AgentConfig = field(default_factory=AgentConfig)

    mode: IntentMode = IntentMode.AGENTIC

    # Plan tracking
    task_plan: TaskPlan | None = None
    current_plan_step_idx: int = 0
    replan_count: int = 0

    # Accumulated results within the current plan step
    step_accumulated_results: list[str] = field(default_factory=list)

    # Skill management
    blocked_skills: set[str] = field(default_factory=set)
    explicit_skill_name: str | None = None
    explicit_skill_retry_done: bool = False

    # Error tracking
    execute_failures: int = 0
    last_execute_error: str = ""

    # Message accumulator (the evolving conversation)
    messages: list[dict[str, Any]] = field(default_factory=list)

    # ── Helpers ──────────────────────────────────────────────────────

    def should_stop_for_failures(self) -> bool:
        return self.execute_failures >= self.config.max_consecutive_exec_failures

    def current_plan_step(self) -> PlanStep | None:
        if self.task_plan and self.current_plan_step_idx < len(self.task_plan.steps):
            return self.task_plan.steps[self.current_plan_step_idx]
        return None

    def remaining_plan_steps(self) -> list[PlanStep]:
        if not self.task_plan:
            return []
        return self.task_plan.steps[self.current_plan_step_idx + 1 :]

    def advance_plan_step(self) -> None:
        """Mark the current step as done and move to the next."""
        self.current_plan_step_idx += 1
        self.step_accumulated_results = []

    def reset_for_replan(self, new_plan: TaskPlan) -> None:
        """Replace the current plan and reset step tracking."""
        self.task_plan = new_plan
        self.current_plan_step_idx = 0
        self.step_accumulated_results = []
        self.replan_count += 1

    def can_replan(self) -> bool:
        return self.replan_count < self.config.max_replans

    def check_duplicate_call(self, tool_name: str, args: dict[str, Any]) -> int:
        """Track consecutive identical tool calls. Returns current dup count."""
        key_parts = [tool_name]
        if tool_name == TOOL_EXECUTE_SKILL:
            key_parts.append(str(args.get("skill_name", "")))
            inner = args.get("args", {})
            if isinstance(inner, dict):
                key_parts.append(str(inner.get("operation", "")))
                key_parts.append(str(inner.get("path", "")))
        sig = "|".join(key_parts)
        if sig == self._last_tool_sig:
            self._dup_count += 1
        else:
            self._last_tool_sig = sig
            self._dup_count = 1
        return self._dup_count
