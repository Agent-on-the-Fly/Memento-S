
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.config.logging import get_logger

from ..analyzer import find_entry_function, parse_code
from ...schema import Skill, SkillExecutionResult
from .base import BaseTrackExecutor

logger = get_logger(__name__)


class CodeTrackExecutor(BaseTrackExecutor):
    async def execute(
        self,
        skill: Skill,
        query: str,
        params: dict[str, Any],
    ) -> tuple[SkillExecutionResult, str]:
        logger.info("Code skill '%s' received: query=%r, params=%s", skill.name, query, params)

        det_result = await self._execute_deterministic(skill, params)
        if det_result.success:
            logger.info("Deterministic execution succeeded for '%s'", skill.name)
            return det_result, ""

        logger.info(
            "Deterministic failed for '%s' (error=%s), falling back to LLM",
            skill.name, (det_result.error or "")[:200],
        )
        return await self._execute_via_llm(skill, query, params)

    async def _execute_deterministic(self, skill: Skill, params: dict[str, Any]) -> SkillExecutionResult:
        runner_code = self._build_deterministic_runner(skill, params)
        if not runner_code:
            return SkillExecutionResult(
                success=False, result=None,
                error="Cannot build deterministic runner (no code or syntax error)",
                skill_name=skill.name,
            )
        return await self._run_in_sandbox(skill, runner_code)

    def _build_deterministic_runner(self, skill: Skill, params: dict[str, Any]) -> str | None:
        code = skill.code
        if not code or not code.strip():
            return None

        tree = parse_code(code)
        if tree is None:
            return None

        params_json = json.dumps(params, ensure_ascii=False, indent=2) if params else "{}"

        is_multi_file = False
        main_module_name: str | None = None
        if skill.source_dir:
            scripts_dir = Path(skill.source_dir) / "scripts"
            if scripts_dir.exists() and any(scripts_dir.glob("*.py")):
                is_multi_file = True
                skill_dir_name = Path(skill.source_dir).name.replace("-", "_")
                py_files = sorted(p for p in scripts_dir.glob("*.py") if p.name != "__init__.py")
                main_script = next(
                    (p for p in py_files if p.stem == skill_dir_name),
                    py_files[0] if py_files else None,
                )
                if main_script:
                    main_module_name = main_script.stem

        func_name = find_entry_function(code, skill.name, tree=tree)

        if is_multi_file and main_module_name and func_name:
            return self._gen_runner_import(main_module_name, func_name, params_json)
        elif func_name:
            return self._gen_runner_embed(code, func_name, params_json)
        else:
            return self._gen_runner_script(code, params_json)

    @staticmethod
    def _gen_runner_import(module_name: str, func_name: str, params_json: str) -> str:
        return f'''\
import json
import importlib

_params = json.loads({params_json!r})

_module = importlib.import_module("{module_name}")
_func = getattr(_module, "{func_name}")
_result = _func(**_params)
if _result is not None:
    if isinstance(_result, str):
        print(_result)
    else:
        print(json.dumps(_result, ensure_ascii=False, default=str))
'''

    @staticmethod
    def _gen_runner_embed(skill_code: str, func_name: str, params_json: str) -> str:
        return f'''\
import json

{skill_code}

_params = json.loads({params_json!r})
_result = {func_name}(**_params)
if _result is not None:
    if isinstance(_result, str):
        print(_result)
    else:
        print(json.dumps(_result, ensure_ascii=False, default=str))
'''

    @staticmethod
    def _gen_runner_script(skill_code: str, params_json: str) -> str:
        return f'''\
import json

_params = json.loads({params_json!r})

{skill_code}
'''

    async def _execute_via_llm(
        self,
        skill: Skill,
        query: str,
        params: dict[str, Any],
    ) -> tuple[SkillExecutionResult, str]:
        generated_code = await self._generate_execution_code(skill, query, params)
        if not generated_code or not generated_code.strip():
            return SkillExecutionResult(
                success=False, result=None,
                error="LLM failed to generate execution code",
                skill_name=skill.name,
            ), ""

        logger.info("LLM generated execution code for '%s' (%d bytes)", skill.name, len(generated_code))
        result = await self._run_in_sandbox(skill, generated_code)
        if not result.success and self.max_retries > 0:
            logger.info("LLM execution failed, triggering reflection (max_retries=%d)", self.max_retries)
            result, generated_code = await self._reflect_and_retry(
                skill, query, generated_code, result.error or "Unknown error", params,
            )
        out_preview = (result.result[:300] + "…") if result.result and len(result.result) > 300 else (result.result or "")
        logger.info(
            "Code skill '%s' result: success=%s, output_len=%s, preview=%s",
            skill.name, result.success, len(result.result or ""), out_preview,
        )
        return result, generated_code
