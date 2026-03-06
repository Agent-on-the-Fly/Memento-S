
from __future__ import annotations

import abc
from pathlib import Path

from core.config import g_settings
from core.config.logging import get_logger
from ...schema import Skill, SkillExecutionResult

logger = get_logger(__name__)


class BaseSandbox(abc.ABC):

    @abc.abstractmethod
    def run_code(
        self,
        code: str,
        skill: Skill,
        deps: list[str] | None = None,
    ) -> SkillExecutionResult: ...

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
    ) -> SkillExecutionResult: ...


def get_sandbox() -> BaseSandbox:
    from .local import LocalSandbox
    provider = g_settings.sandbox_provider
    if provider == "e2b":
        if not g_settings.e2b_api_key:
            logger.warning("E2B API key not set, falling back to local sandbox")
            return LocalSandbox()
        from .e2b import E2BSandbox
        return E2BSandbox()
    elif provider == "modal":
        logger.warning("Modal sandbox not yet implemented, falling back to local")
        return LocalSandbox()
    else:
        return LocalSandbox()
