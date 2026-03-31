"""RuntimeState — session 级别的程序化状态持久化。

与 AgentRunState（per-run 的内存态）不同，RuntimeState 是跨 reply_stream
调用持久化到文件的 session 级控制对象。

设计原则:
  - 小体积（通常 < 1KB），不含 messages 和完整 tool outputs
  - 程序化更新，不依赖 LLM
  - 每次状态变更后写盘，支持崩溃恢复

文件位置: {context_dir}/{YYYY-MM-DD}/runtime_state_{session_id}.json
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RuntimeState:
    """Session 级别的持久化控制状态。

    字段全部可程序化维护，不含自然语言 summary。
    """

    session_id: str = ""
    active_block_id: str = ""
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

    # Artifact refs
    recent_refs: list[str] = field(default_factory=list)

    # Status
    current_status: str = "awaiting_user"
    # Valid statuses: awaiting_user, planning, executing, sealed

    # Metadata
    turn_count: int = 0
    updated_at: str = ""

    # ── Serialization ─────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {
            "session_id": self.session_id,
            "active_block_id": self.active_block_id,
            "current_goal_text": self.current_goal_text,
            "plan_version": self.plan_version,
            "active_plan_step": self.active_plan_step,
            "completed_plan_steps": self.completed_plan_steps,
            "open_loops": self.open_loops,
            "blocked_actions": self.blocked_actions,
            "need_replan": self.need_replan,
            "last_effective_action": self.last_effective_action,
            "recent_refs": self.recent_refs,
            "current_status": self.current_status,
            "turn_count": self.turn_count,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeState":
        """Reconstruct from a serialized dict."""
        return cls(
            session_id=data.get("session_id", ""),
            active_block_id=data.get("active_block_id", ""),
            current_goal_text=data.get("current_goal_text", ""),
            plan_version=data.get("plan_version", 0),
            active_plan_step=data.get("active_plan_step", 0),
            completed_plan_steps=data.get("completed_plan_steps", []),
            open_loops=data.get("open_loops", []),
            blocked_actions=data.get("blocked_actions", []),
            need_replan=data.get("need_replan", False),
            last_effective_action=data.get("last_effective_action"),
            recent_refs=data.get("recent_refs", []),
            current_status=data.get("current_status", "awaiting_user"),
            turn_count=data.get("turn_count", 0),
            updated_at=data.get("updated_at", ""),
        )

    # ── Programmatic updates ──────────────────────────────────

    def on_user_input(self, goal_text: str, block_id: str = "") -> None:
        """User input arrived — update goal and status."""
        self.current_goal_text = goal_text
        self.current_status = "planning"
        self.turn_count += 1
        if block_id:
            self.active_block_id = block_id
        self._touch()

    def on_plan_generated(self, step_count: int) -> None:
        """Planner produced a plan — reset plan tracking."""
        self.plan_version += 1
        self.active_plan_step = 0
        self.completed_plan_steps = []
        self.need_replan = False
        self.current_status = "executing"
        self._touch()

    def on_step_completed(self, step_idx: int) -> None:
        """A plan step was completed successfully."""
        if step_idx not in self.completed_plan_steps:
            self.completed_plan_steps.append(step_idx)
        self.active_plan_step = step_idx + 1
        self._touch()

    def on_effective_action(self, action_name: str) -> None:
        """A tool call completed effectively."""
        self.last_effective_action = action_name
        # Remove from blocked if it was previously blocked
        if action_name in self.blocked_actions:
            self.blocked_actions.remove(action_name)
        self._touch()

    def on_ineffective_action(
        self, action_name: str, *, replan_threshold: int = 2,
    ) -> None:
        """A tool call was ineffective."""
        if action_name not in self.blocked_actions:
            self.blocked_actions.append(action_name)
        # Check if we need replan
        if len(self.blocked_actions) >= replan_threshold:
            self.need_replan = True
        self._touch()

    def on_new_artifact(self, ref_path: str) -> None:
        """A new artifact was created — track in refs.

        保留全部 artifact 引用，不做滑动窗口截断。
        block.py 已对 tool_result 做了 fold，上下文本身已精炼。
        """
        if ref_path not in self.recent_refs:
            self.recent_refs.append(ref_path)
        self._touch()

    def on_block_sealed(self) -> None:
        """Active block was sealed — transition to awaiting."""
        self.current_status = "awaiting_user"
        # Clear block-local short-term fields
        self.open_loops = []
        self.blocked_actions = []
        self.need_replan = False
        self._touch()

    def on_run_finished(self) -> None:
        """A reply_stream run completed."""
        self.current_status = "awaiting_user"
        self._touch()

    def _touch(self) -> None:
        """Update the timestamp."""
        self.updated_at = datetime.now().isoformat(timespec="seconds")


# ═══════════════════════════════════════════════════════════════
# File I/O
# ═══════════════════════════════════════════════════════════════


class RuntimeStateStore:
    """Handles persistence of RuntimeState to/from disk.

    File location: {state_dir}/runtime_state.json
    (state_dir is typically the session directory)
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
                logger.warning(
                    "Failed to load runtime state, creating fresh: {}", e
                )

        self._state = RuntimeState(session_id=self._session_id)
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
            logger.warning(
                "Failed to save runtime state: {}", self._path, exc_info=True
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
    """从 AgentRunState + SessionContext 同步到 RuntimeState。

    这是 AgentRunState (per-run) → RuntimeState (per-session) 的桥接函数。
    在每次 reply_stream 结束时调用。
    """
    # From SessionContext
    if hasattr(session_ctx, "session_goal"):
        runtime_state.current_goal_text = session_ctx.session_goal
    if hasattr(session_ctx, "turn_count"):
        runtime_state.turn_count = session_ctx.turn_count

    # From AgentRunState
    if agent_run_state is None:
        return

    if hasattr(agent_run_state, "task_plan") and agent_run_state.task_plan:
        plan = agent_run_state.task_plan
        runtime_state.active_plan_step = agent_run_state.current_plan_step_idx

        # Rebuild completed steps from statuses
        completed = []
        if hasattr(agent_run_state, "plan_step_statuses"):
            for i, status in enumerate(agent_run_state.plan_step_statuses):
                if hasattr(status, "value") and status.value == "done":
                    completed.append(i)
                elif status == "done":
                    completed.append(i)
        runtime_state.completed_plan_steps = completed

    if hasattr(agent_run_state, "blocked_skills"):
        runtime_state.blocked_actions = sorted(agent_run_state.blocked_skills)

    if hasattr(agent_run_state, "execute_failures"):
        if agent_run_state.execute_failures >= 2:
            runtime_state.need_replan = True

    if hasattr(agent_run_state, "replan_count"):
        runtime_state.plan_version = agent_run_state.replan_count + 1

    runtime_state._touch()
