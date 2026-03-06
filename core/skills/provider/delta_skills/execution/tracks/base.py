
from __future__ import annotations

import asyncio
import functools
import json
import time
from typing import Any

from core.config.logging import get_logger

from ..analyzer import ensure_dependencies
from ..sandbox import BaseSandbox
from ...observability import track_reflection
from ...schema import Skill, SkillExecutionResult
from .prompts import EXECUTE_PROMPT, REFLECT_PROMPT
from .utils import (
    build_available_modules_section,
    build_skill_content,
    parse_diagnosis,
    refresh_skill_metadata,
    strip_code_fences,
)

logger = get_logger(__name__)


class BaseTrackExecutor:

    def __init__(self, sandbox: BaseSandbox, max_retries: int, llm: Any):
        self._llm = llm
        self.max_retries = max_retries
        self.sandbox = sandbox

    async def _chat_async(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> str:
        resp = await self._llm.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return (getattr(resp, "content", None) or "").strip()

    async def _call_llm(self, prompt: str) -> str:
        try:
            content = await self._chat_async(
                [{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=4096,
            )
            return strip_code_fences(content)
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return ""

    async def _run_in_sandbox(self, skill: Skill, code: str) -> SkillExecutionResult:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            functools.partial(self._run_in_sandbox_sync, skill, code),
        )

    def _run_in_sandbox_sync(self, skill: Skill, code: str) -> SkillExecutionResult:
        all_deps = ensure_dependencies(skill)
        return self.sandbox.run_code(code, skill, deps=all_deps)

    async def _generate_execution_code(
        self,
        skill: Skill,
        query: str,
        params: dict[str, Any],
    ) -> str:
        prompt = EXECUTE_PROMPT.format(
            skill_name=skill.name,
            description=skill.description or "(no description)",
            skill_content=build_skill_content(skill),
            parameters=json.dumps(skill.parameters, ensure_ascii=False, indent=2)
            if skill.parameters
            else "(none)",
            query=query,
            params_json=json.dumps(params, ensure_ascii=False, indent=2)
            if params
            else "{}",
            available_modules_section=build_available_modules_section(skill),
        )
        return await self._call_llm(prompt)

    async def _reflect_and_retry(
        self,
        skill: Skill,
        query: str,
        failed_code: str,
        error: str,
        params: dict[str, Any] | None = None,
    ) -> tuple[SkillExecutionResult, str]:
        params = params or {}
        current_exec = failed_code
        current_error = error
        last_diagnosis = "UNKNOWN"
        error_history: list[str] = []
        t0 = time.monotonic()

        original_code = skill.code
        original_params = skill.parameters
        original_deps = list(skill.dependencies)
        original_desc = skill.description

        for i in range(self.max_retries):
            logger.info("Reflection %d/%d for '%s'", i + 1, self.max_retries, skill.name)

            if "timed out" in (current_error or "").lower():
                logger.warning(
                    "Reflection %d: skipping retry for '%s' — execution timed out "
                    "(code fix cannot resolve a timeout; agent should use tmux or increase timeout)",
                    i + 1, skill.name,
                )
                last_diagnosis = "TIMEOUT"
                break

            error_section = current_error
            if error_history:
                history_text = "\n---\n".join(
                    f"Attempt {j+1}: {err}" for j, err in enumerate(error_history)
                )
                error_section = (
                    f"{current_error}\n\n"
                    f"## Previous Failed Attempts\n{history_text}"
                )

            prompt = REFLECT_PROMPT.format(
                skill_name=skill.name,
                query=query,
                skill_code=skill.code,
                execution_code=current_exec,
                error=error_section,
                available_modules_section=build_available_modules_section(skill),
            )

            response = await self._call_llm(prompt)
            if not response or not response.strip():
                logger.warning("Reflection %d: empty response", i + 1)
                last_diagnosis = "TIMEOUT"
                continue

            diagnosis, fixed_code = parse_diagnosis(response)
            last_diagnosis = diagnosis

            if diagnosis == "SKILL_FIX":
                logger.info("Reflection %d: SKILL_FIX", i + 1)
                skill.code = fixed_code
                refresh_skill_metadata(skill, fixed_code)
                new_exec = await self._generate_execution_code(skill, query, params)
                if not new_exec or not new_exec.strip():
                    logger.warning("Reflection %d: failed to regenerate exec code", i + 1)
                    continue
                result = await self._run_in_sandbox(skill, new_exec)
                current_exec = new_exec
            else:
                logger.info("Reflection %d: %s", i + 1, diagnosis)
                result = await self._run_in_sandbox(skill, fixed_code)
                current_exec = fixed_code

            if result.success:
                logger.info("Reflection succeeded on attempt %d (%s)", i + 1, diagnosis)
                elapsed = (time.monotonic() - t0) * 1000
                track_reflection(skill.name, i + 1, diagnosis, True, elapsed)
                return result, current_exec

            error_history.append(f"[{diagnosis}] {current_error}")
            current_error = result.error or ""

        skill.code = original_code
        skill.parameters = original_params
        skill.dependencies = original_deps
        skill.description = original_desc
        skill.invalidate_cache()
        logger.warning(
            "All %d reflection retries failed for '%s', restored original skill code",
            self.max_retries, skill.name,
        )

        elapsed = (time.monotonic() - t0) * 1000
        track_reflection(skill.name, self.max_retries, last_diagnosis, False, elapsed)
        return SkillExecutionResult(
            success=False, result=None,
            error=f"Failed after {self.max_retries} reflection retries. Last error: {current_error}",
            skill_name=skill.name,
        ), current_exec
