
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from core.config.logging import get_logger
from .executor import SkillExecutor
from ..schema import Skill

logger = get_logger(__name__)


@dataclass
class VerifyResult:
    success: bool
    output: str = ""
    error: str = ""
    generated_code: str = ""
    sample_params: dict[str, Any] = field(default_factory=dict)


class SkillVerifier:

    def __init__(self, executor: SkillExecutor | None = None):
        self.executor = executor or SkillExecutor()

    async def verify(
        self,
        skill: Skill,
        params: dict[str, Any] | None = None,
    ) -> VerifyResult:
        logger.info("Verifying skill '%s' via LLM executor...", skill.name)
        if params:
            query = (
                f"Verify/test the skill '{skill.name}' by calling it with these parameters: "
                f"{json.dumps(params, ensure_ascii=False)}. "
                f"Print the result clearly."
            )
        else:
            query = (
                f"Verify/test the skill '{skill.name}' by generating realistic sample parameters "
                f"and calling the function. Print the result clearly. "
                f"Choose reasonable test values based on the function signature and description."
            )
        exec_result, generated_code = await self.executor.execute(
            skill, query=query, params=params,
        )
        if exec_result.success:
            logger.info(
                "Skill '%s' verified OK: %s",
                skill.name, str(exec_result.result)[:100],
            )
        else:
            logger.warning(
                "Skill '%s' verification failed: %s",
                skill.name, str(exec_result.error)[:200],
            )

        return VerifyResult(
            success=exec_result.success,
            output=str(exec_result.result or "")[:2000],
            error=str(exec_result.error or "")[:2000],
            generated_code=generated_code,
            sample_params=params or {},
        )
