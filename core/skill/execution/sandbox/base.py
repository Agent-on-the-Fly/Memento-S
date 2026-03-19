"""沙箱抽象基类 + 工厂函数"""

from __future__ import annotations

import abc
import sys
from pathlib import Path

from core.skill.schema import Skill, SkillExecutionOutcome


class BaseSandbox(abc.ABC):
    @property
    def python_executable(self) -> Path:
        """Return the Python executable path for this sandbox."""
        return Path(sys.executable)

    @abc.abstractmethod
    def run_code(
        self,
        code: str,
        skill: Skill,
        deps: list[str] | None = None,
        session_id: str = "",
    ) -> SkillExecutionOutcome: ...

    @abc.abstractmethod
    def run(
        self,
        cmd: list[str],
        *,
        cwd: str | Path,
        pythonpath: str | Path | None = None,
        timeout: int | None = None,
        skill_name: str = "",
        check_syntax: str | None = None,
        env: dict[str, str] | None = None,
    ) -> SkillExecutionOutcome: ...

    def get_python_executable(self) -> Path | None:
        return None


def get_sandbox() -> BaseSandbox:
    """获取沙箱实例（当前统一使用 UvLocalSandbox）。"""
    from .uv import UvLocalSandbox

    return UvLocalSandbox()
