"""Mutable state for a single agent run (one ``reply_stream`` invocation)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from core.protocol.types import PlanStepStatus

from ..schemas import AgentConfig
from .intent import IntentMode
from .planning import PlanStep, TaskPlan


def _build_tool_signature(tool_name: str, args: dict[str, Any]) -> str:
    """Build a stable signature string for duplicate-call detection.

    Two calls are duplicates only when tool name AND all arguments are identical.
    """
    try:
        args_str = json.dumps(args, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        args_str = str(args)
    return f"{tool_name}|{args_str}"


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

    # Plan step statuses (single source of truth)
    plan_step_statuses: list[PlanStepStatus] = field(default_factory=list)

    # HITL support
    pending_ask_user_call_id: str | None = None

    # Reflection decision history (for deduplication / context)
    reflection_history: list[str] = field(default_factory=list)

    # Duplicate tool call detection
    _last_tool_sig: str = field(default="", repr=False)
    _dup_count: int = field(default=0, repr=False)
    _last_tool_success: bool = field(default=True, repr=False)
    _last_result_hash: int | None = field(default=None, repr=False)

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

    def can_replan(self) -> bool:
        return self.replan_count < self.config.max_replans

    def check_duplicate_call(self, tool_name: str, args: dict[str, Any]) -> tuple[int, bool]:
        """Track consecutive identical tool calls.

        Returns:
            (consecutive_dup_count, last_call_succeeded)
        """
        sig = _build_tool_signature(tool_name, args)
        if sig == self._last_tool_sig:
            self._dup_count += 1
        else:
            self._last_tool_sig = sig
            self._dup_count = 1
            self._last_tool_success = True
            self._last_result_hash = None
        return self._dup_count, self._last_tool_success

    def record_tool_result(
        self, tool_name: str, args: dict[str, Any], success: bool, result: str = "",
    ) -> None:
        """Update success state and reset dup counter when result content changes.

        Same call + different result means the skill is making progress
        (e.g. first list_dir, then file_create), so it should not be blocked.
        Only truly stuck loops (same call → same result) accumulate the counter.
        """
        sig = _build_tool_signature(tool_name, args)
        if sig == self._last_tool_sig:
            self._last_tool_success = success
            rh = hash(result[:1000]) if result else None
            if self._last_result_hash is not None and rh != self._last_result_hash:
                self._dup_count = 1
            self._last_result_hash = rh

    # ── Plan status (single source of truth) ─────────────────────

    def _ensure_statuses(self) -> None:
        """Lazily initialise ``plan_step_statuses`` from ``task_plan``."""
        if self.task_plan and len(self.plan_step_statuses) != len(self.task_plan.steps):
            self.plan_step_statuses = [PlanStepStatus.PENDING] * len(self.task_plan.steps)

    def to_plan_prompt_section(self) -> str:
        """Render the plan as a markdown checklist for injection into prompts."""
        if not self.task_plan or not self.task_plan.steps:
            return ""
        self._ensure_statuses()
        _MARKERS = {
            PlanStepStatus.PENDING: "[ ]",
            PlanStepStatus.IN_PROGRESS: "[~]",
            PlanStepStatus.DONE: "[x]",
            PlanStepStatus.FAILED: "[!]",
        }
        lines = ["## Task Plan"]
        for i, step in enumerate(self.task_plan.steps):
            status = (
                self.plan_step_statuses[i]
                if i < len(self.plan_step_statuses)
                else PlanStepStatus.PENDING
            )
            marker = _MARKERS.get(status, "[ ]")
            lines.append(f"  {marker} Step {step.step_id}: {step.action}")
        done = sum(1 for s in self.plan_step_statuses if s == PlanStepStatus.DONE)
        lines.append(f"\nProgress: {done}/{len(self.task_plan.steps)}")
        return "\n".join(lines)

    def sync_plan_state(self, session_ctx: Any) -> None:
        """Push current plan into ``SessionContext`` (backward-compat bridge).

        Keeps ``SessionContext._plan_steps/_plan_statuses`` in sync so that
        ``to_prompt_section`` on the session still renders correctly.
        """
        if not self.task_plan:
            return
        self._ensure_statuses()
        steps_text = [s.action for s in self.task_plan.steps]
        session_ctx.set_plan(steps_text)
        session_ctx._plan_statuses = list(self.plan_step_statuses)

    def advance_plan_step(self) -> None:
        """Mark the current step as done and move to the next."""
        self._ensure_statuses()
        if self.current_plan_step_idx < len(self.plan_step_statuses):
            self.plan_step_statuses[self.current_plan_step_idx] = PlanStepStatus.DONE
        self.current_plan_step_idx += 1
        self.step_accumulated_results = []
        if self.current_plan_step_idx < len(self.plan_step_statuses):
            self.plan_step_statuses[self.current_plan_step_idx] = PlanStepStatus.IN_PROGRESS

    def reset_for_replan(self, new_plan: TaskPlan) -> None:
        """Replace the current plan and reset step tracking."""
        self.task_plan = new_plan
        self.current_plan_step_idx = 0
        self.step_accumulated_results = []
        self.replan_count += 1
        self.plan_step_statuses = [PlanStepStatus.PENDING] * len(new_plan.steps)
        if self.plan_step_statuses:
            self.plan_step_statuses[0] = PlanStepStatus.IN_PROGRESS

    # ── Serialization (HITL resume) ──────────────────────────────

    def serialize(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict for persistence / HITL resume."""
        data: dict[str, Any] = {
            "mode": self.mode.value,
            "current_plan_step_idx": self.current_plan_step_idx,
            "replan_count": self.replan_count,
            "step_accumulated_results": self.step_accumulated_results,
            "blocked_skills": sorted(self.blocked_skills),
            "explicit_skill_name": self.explicit_skill_name,
            "explicit_skill_retry_done": self.explicit_skill_retry_done,
            "execute_failures": self.execute_failures,
            "last_execute_error": self.last_execute_error,
            "plan_step_statuses": [s.value for s in self.plan_step_statuses],
            "pending_ask_user_call_id": self.pending_ask_user_call_id,
            "reflection_history": self.reflection_history,
            "messages": self.messages,
        }
        if self.task_plan:
            data["task_plan"] = self.task_plan.model_dump()
        return data

    @classmethod
    def deserialize(cls, data: dict[str, Any], config: AgentConfig | None = None) -> "AgentRunState":
        """Reconstruct from a serialized dict."""
        cfg = config or AgentConfig()
        state = cls(config=cfg)
        state.mode = IntentMode(data.get("mode", "agentic"))
        state.current_plan_step_idx = data.get("current_plan_step_idx", 0)
        state.replan_count = data.get("replan_count", 0)
        state.step_accumulated_results = data.get("step_accumulated_results", [])
        state.blocked_skills = set(data.get("blocked_skills", []))
        state.explicit_skill_name = data.get("explicit_skill_name")
        state.explicit_skill_retry_done = data.get("explicit_skill_retry_done", False)
        state.execute_failures = data.get("execute_failures", 0)
        state.last_execute_error = data.get("last_execute_error", "")
        state.plan_step_statuses = [
            PlanStepStatus(s) for s in data.get("plan_step_statuses", [])
        ]
        state.pending_ask_user_call_id = data.get("pending_ask_user_call_id")
        state.reflection_history = data.get("reflection_history", [])
        state.messages = data.get("messages", [])
        plan_data = data.get("task_plan")
        if plan_data:
            state.task_plan = TaskPlan(**plan_data)
        return state
