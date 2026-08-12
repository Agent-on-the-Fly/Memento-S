# SPDX-License-Identifier: Apache-2.0
"""core.context.runtime — RuntimeState + RuntimeStateStore (per-session persistence).

This module defines the session-level runtime state that survives across reply_stream
invocations and is persisted to disk. The RuntimeState tracks:
  - Goal / turn count
  - Plan progress (version, active step, completed steps)
  - Execution state (open loops, blocked actions, replan flag)
  - Lifecycle status (awaiting_user / planning / executing / sealed)

All state is programmatic (no LLM summaries) and designed for small serialized size.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from core.context.session import SessionStatus
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RuntimeState:
    """Session-level persistent control state — all fields programmatic.

    Design goals:
      - Small footprint (typically < 1KB)
      - No large text blobs or LLM summaries
      - Programmatic updates only (no LLM writes)
      - Written after each status change
    """

    session_id: str = ""

    current_goal_text: str = ""

    # Plan tracking
    plan_version: int = 0
    active_plan_step: int = 0
    completed_plan_steps: list[int] = field(default_factory=list)

    # Execution state
    open_loops: list[str] = field(default_factory=list)
    blocked_actions: list[str] = field(default_factory=list)
    need_replan: bool = False
    last_effective_action: str | None = None

    # Lifecycle status
    current_status: SessionStatus = SessionStatus.AWAITING_USER

    # Metadata
    turn_count: int = 0
    updated_at: str = ""

    # ── Serialization ──────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {
            "session_id": self.session_id,
            "current_goal_text": self.current_goal_text,
            "plan_version": self.plan_version,
            "active_plan_step": self.active_plan_step,
            "completed_plan_steps": self.completed_plan_steps,
            "open_loops": self.open_loops,
            "blocked_actions": self.blocked_actions,
            "need_replan": self.need_replan,
            "last_effective_action": self.last_effective_action,
            "current_status": self.current_status.value,
            "turn_count": self.turn_count,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuntimeState:
        """Reconstruct from a serialized dict."""
        status_val = data.get("current_status", "awaiting_user")
        # Accept both str and SessionStatus
        status = SessionStatus(status_val) if isinstance(status_val, str) else status_val

        return cls(
            session_id=data.get("session_id", ""),
            current_goal_text=data.get("current_goal_text", ""),
            plan_version=data.get("plan_version", 0),
            active_plan_step=data.get("active_plan_step", 0),
            completed_plan_steps=data.get("completed_plan_steps", []),
            open_loops=data.get("open_loops", []),
            blocked_actions=data.get("blocked_actions", []),
            need_replan=data.get("need_replan", False),
            last_effective_action=data.get("last_effective_action"),
            current_status=status,
            turn_count=data.get("turn_count", 0),
            updated_at=data.get("updated_at", ""),
        )

# ── Persistence layer ──────────────────────────────────────────────────


class RuntimeStateStore:
    """Handles persistence of RuntimeState to/from disk.

    File location: {session_dir}/runtime_state.json
    """

    def __init__(self, session_id: str, state_dir: Path) -> None:
        self._session_id = session_id
        state_dir.mkdir(parents=True, exist_ok=True)
        self._path = state_dir / "runtime_state.json"
        self._state: RuntimeState | None = None

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> RuntimeState:
        """Load from file, or create fresh state."""
        if self._state is not None:
            return self._state

        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self._state = RuntimeState.from_dict(data)
                logger.info(
                    "RuntimeState loaded: session={}, status={}",
                    self._state.session_id,
                    self._state.current_status,
                )
                return self._state
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load runtime state, creating fresh: {}", e)

        self._state = RuntimeState(session_id=self._ctx.session_id)
        return self._state

    def save(self, state: RuntimeState | None = None) -> None:
        """Write state to disk. Uses cached state if no argument given."""
        st = state or self._state
        if st is None:
            return
        self._state = st
        try:
            self._path.write_text(
                json.dumps(st.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            logger.opt(exception=True).warning(
                "Failed to save runtime state: {}", self._path
            )

    def get(self) -> RuntimeState:
        """Get cached state (loads if needed)."""
        if self._state is None:
            return self.load()
        return self._state


def sync_from_agent_run(
    runtime_state: RuntimeState,
    agent_run_state: Any,
    session_ctx: Any,
) -> None:
    """Sync AgentRunState + SessionGoal → RuntimeState.

    Called at the end of each ``reply_stream`` invocation to persist
    the in-memory run state into the persistent runtime state.

    Args:
        runtime_state: Target RuntimeState (in-place mutation)
        agent_run_state: AgentRunState (per-run memory)
        session_ctx: SessionGoal (per-session goal state)
    """
    if hasattr(session_ctx, "text") and session_ctx.text:
        runtime_state.current_goal_text = session_ctx.text
    if hasattr(session_ctx, "turn_count"):
        runtime_state.turn_count = session_ctx.turn_count

    # From AgentRunState
    if agent_run_state is None:
        return

    if hasattr(agent_run_state, "task_plan") and agent_run_state.task_plan:
        runtime_state.active_plan_step = agent_run_state.current_plan_step_idx
        completed = []
        if hasattr(agent_run_state, "plan_step_statuses"):
            for i, status in enumerate(agent_run_state.plan_step_statuses):
                # Accept either PlanStepStatus enum or string
                status_val = status.value if hasattr(status, "value") else str(status)
                if status_val == "done":
                    completed.append(i)
        runtime_state.completed_plan_steps = completed

    if hasattr(agent_run_state, "blocked_skills"):
        runtime_state.blocked_actions = sorted(agent_run_state.blocked_skills)

    if hasattr(agent_run_state, "execute_failures"):
        if agent_run_state.execute_failures >= 2:
            runtime_state.need_replan = True

    if hasattr(agent_run_state, "replan_count"):
        runtime_state.plan_version = agent_run_state.replan_count + 1

    runtime_state.updated_at = datetime.now().isoformat(timespec="seconds")
