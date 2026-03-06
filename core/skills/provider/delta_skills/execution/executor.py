
from __future__ import annotations

from pathlib import Path
from typing import Any

from core.config.logging import get_logger
from core.llm import LLM

from .sandbox import get_sandbox
from .tracks import CodeTrackExecutor, KnowledgeExecutor, PlaybookExecutor
from ..schema import Skill, SkillExecutionResult

logger = get_logger(__name__)

class SkillExecutor:

    def __init__(self, *, sandbox=None, max_retries: int = 2, llm: Any = None):
        _llm = llm if llm is not None else LLM()
        _sandbox = sandbox if sandbox is not None else get_sandbox()

        self.knowledge = KnowledgeExecutor(_sandbox, max_retries, _llm)
        self.playbook = PlaybookExecutor(_sandbox, max_retries, _llm)
        self.code = CodeTrackExecutor(_sandbox, max_retries, _llm)

    async def execute(
        self,
        skill: Skill,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> tuple[SkillExecutionResult, str]:
        params = params or {}

        mode = skill.execution_mode

        if mode == "knowledge" or (mode is None and skill.is_knowledge_skill):
            return await self.knowledge.execute(skill, query, params)

        if mode == "playbook" or (mode is None and skill.is_playbook):
            return await self.playbook.execute(skill, params)

        if (
            mode is None
            and ("script" in params or "args" in params)
            and self._has_script_files(skill)
        ):
            logger.info(
                "Param-driven fallback to playbook for skill '%s'", skill.name,
            )
            return await self.playbook.execute(skill, params)

        return await self.code.execute(skill, query, params)

    @staticmethod
    def _has_script_files(skill: Skill) -> bool:
        if not skill.source_dir:
            return False
        scripts_dir = Path(skill.source_dir) / "scripts"
        if not scripts_dir.exists():
            return False
        return any(
            p.name != "__init__.py"
            for p in scripts_dir.glob("*.py")
        )
