"""SessionContext — per-session dynamic state awareness.

Four layers:
1. EnvironmentSnapshot: local context (cwd, git, project type, OS)
2. Progressive Goal Refinement: evolving session goal
3. Task Plan Tracking: plain-string step descriptions + status
4. Action History: tool call records for "what happened" narrative
"""
from __future__ import annotations

import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from middleware.config import g_config


@dataclass
class EnvironmentSnapshot:
    """Zero-LLM local environment context."""

    cwd: str = ""
    git_branch: str | None = None
    git_dirty_files: list[str] = field(default_factory=list)
    project_type: str = ""
    os_info: str = ""

    @classmethod
    def capture(cls) -> "EnvironmentSnapshot":
        """Capture current environment state (no LLM calls)."""
        cwd = str(g_config.paths.workspace_dir)
        os_info = f"{platform.system()} {platform.machine()}"

        git_branch = None
        git_dirty: list[str] = []
        try:
            branch_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=cwd,
            )
            if branch_result.returncode == 0:
                git_branch = branch_result.stdout.strip()

            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=cwd,
            )
            if status_result.returncode == 0 and status_result.stdout.strip():
                git_dirty = [
                    line.strip()
                    for line in status_result.stdout.strip().split("\n")
                    if line.strip()
                ][:10]
        except Exception:
            pass

        project_type = _detect_project_type(Path(cwd))

        return cls(
            cwd=cwd,
            git_branch=git_branch,
            git_dirty_files=git_dirty,
            project_type=project_type,
            os_info=os_info,
        )


def _detect_project_type(path: Path) -> str:
    """Detect project type from marker files."""
    markers = {
        "pyproject.toml": "python",
        "setup.py": "python",
        "requirements.txt": "python",
        "package.json": "node",
        "Cargo.toml": "rust",
        "go.mod": "go",
        "pom.xml": "java",
        "build.gradle": "java",
        "Gemfile": "ruby",
        "composer.json": "php",
    }
    for marker, ptype in markers.items():
        if (path / marker).exists():
            return ptype
    return ""


@dataclass
class ActionRecord:
    """Record of a single tool/skill execution."""

    tool_name: str
    skill_name: str = ""
    args_summary: str = ""
    result_summary: str = ""
    success: bool = True
    timestamp: float = 0.0

    @classmethod
    def from_tool_call(
        cls,
        tool_name: str,
        args: dict,
        result: str,
        success: bool = True,
    ) -> "ActionRecord":
        skill_name = ""
        if tool_name == "execute_skill":
            skill_name = args.get("skill_name", "")

        args_str = str(args)
        if len(args_str) > 100:
            args_str = args_str[:97] + "..."

        result_str = str(result)
        if len(result_str) > 200:
            result_str = result_str[:197] + "..."

        return cls(
            tool_name=tool_name,
            skill_name=skill_name,
            args_summary=args_str,
            result_summary=result_str,
            success=success,
            timestamp=time.time(),
        )


@dataclass
class SessionContext:
    """Per-session dynamic state.

    Plan tracking 只存纯字符串描述，不依赖 memento_s 类型。
    上层通过 set_plan() / mark_step_done() 操作，通过
    has_active_plan / plan_step_count 查询。
    """

    session_id: str
    environment: EnvironmentSnapshot = field(default_factory=EnvironmentSnapshot)
    session_goal: str = ""
    action_history: list[ActionRecord] = field(default_factory=list)
    turn_count: int = 0

    _plan_steps: list[str] = field(default_factory=list)
    _plan_statuses: list[str] = field(default_factory=list)

    # ── Plan tracking ────────────────────────────────────────────

    def set_plan(self, steps: list[str]) -> None:
        """设置/替换任务计划（纯字符串描述列表）。"""
        self._plan_steps = list(steps)
        self._plan_statuses = ["pending"] * len(steps)

    def mark_step_done(self, idx: int) -> None:
        """标记第 idx 步完成。"""
        if 0 <= idx < len(self._plan_statuses):
            self._plan_statuses[idx] = "done"

    @property
    def has_active_plan(self) -> bool:
        return bool(self._plan_steps and any(s == "pending" for s in self._plan_statuses))

    @property
    def plan_step_count(self) -> int:
        return len(self._plan_steps)

    # ── Goal & actions ───────────────────────────────────────────

    def update_goal(self, user_msg: str):
        """Progressive goal refinement — 首轮从 user_msg 提取目标。"""
        self.turn_count += 1

        if self.turn_count == 1:
            goal = user_msg.strip()
            if len(goal) > 200:
                goal = goal[:197] + "..."
            self.session_goal = goal

    def add_action(self, record: ActionRecord) -> None:
        self.action_history.append(record)

    # ── Prompt rendering ─────────────────────────────────────────

    def to_prompt_section(self) -> str:
        """Format for injection into system prompt tail."""
        lines = ["## Session State"]

        env = self.environment
        if env.cwd:
            lines.append(f"Working directory: {env.cwd}")
        if env.project_type:
            lines.append(f"Project type: {env.project_type}")
        if env.git_branch:
            branch_info = f"Git branch: {env.git_branch}"
            if env.git_dirty_files:
                branch_info += f" ({len(env.git_dirty_files)} uncommitted changes)"
            lines.append(branch_info)

        if self.session_goal:
            lines.append(f"\nCurrent goal: {self.session_goal}")

        if self._plan_steps:
            lines.append("\nTask plan:")
            for i, step_desc in enumerate(self._plan_steps):
                status = (
                    self._plan_statuses[i]
                    if i < len(self._plan_statuses)
                    else "pending"
                )
                marker = {"pending": "[ ]", "done": "[x]", "failed": "[!]"}.get(
                    status, "[ ]"
                )
                lines.append(f"  {marker} {step_desc}")

        recent = self.action_history[-5:]
        if recent:
            lines.append(f"\nRecent actions ({len(self.action_history)} total):")
            for r in recent:
                name = r.skill_name or r.tool_name
                status = "OK" if r.success else "FAIL"
                lines.append(f"  - {name}: {status} — {r.result_summary[:80]}")

        return "\n".join(lines)

    def to_summary(self) -> str:
        """Generate a short summary for session persistence."""
        parts = []
        if self.session_goal:
            parts.append(self.session_goal[:100])

        skill_names = list({r.skill_name for r in self.action_history if r.skill_name})
        if skill_names:
            parts.append(f"Skills: {', '.join(skill_names[:5])}")

        total = len(self.action_history)
        success = sum(1 for r in self.action_history if r.success)
        if total:
            parts.append(f"Actions: {success}/{total} succeeded")

        return " | ".join(parts) if parts else "Empty session"
