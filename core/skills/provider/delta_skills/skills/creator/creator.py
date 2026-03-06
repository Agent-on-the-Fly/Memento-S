
from __future__ import annotations

import re

from core.config.logging import get_logger
from core.llm import LLM
from ...execution.analyzer import extract_dependencies, validate_syntax
from ...schema import Skill, SkillCreationError
from .extraction import extract_description, extract_parameters
from .prompts import (
    EVOLVE_PROMPT,
    GENERATE_PROMPT,
    KNOWLEDGE_SKILL_PROMPT,
    REFLECTION_PROMPT,
    SKILL_TYPE_PROMPT,
)


_RE_OUTER_FENCE = re.compile(
    r"^\s*```[\w-]*\s*\n(.*?)\n\s*```\s*$",
    re.DOTALL,
)

logger = get_logger(__name__)


class SkillCreator:

    def __init__(self, llm: LLM | None = None):
        self.llm = llm or LLM()

    async def create_skill(self, name: str, description: str, force_type: str = None) -> Skill:
        skill_type = force_type or await self._determine_skill_type(description)
        logger.info("Creating %s skill '%s'...", skill_type, name)

        if skill_type == "knowledge":
            skill = await self._create_knowledge_skill(name, description)
        else:
            skill = await self._create_code_skill(name, description)

        self._post_creation_audit(skill)
        return skill

    async def evolve_skill(self, existing_skill: Skill, new_requirement: str) -> Skill:
        logger.info(
            "Evolving skill '%s' for requirement: '%s'",
            existing_skill.name, new_requirement[:60],
        )

        prompt = EVOLVE_PROMPT.format(
            name=existing_skill.name,
            current_description=existing_skill.description,
            current_code=existing_skill.code,
            new_requirement=new_requirement,
        )
        code = await self._call_llm(prompt)

        parameters = extract_parameters(code, existing_skill.name)
        dependencies = extract_dependencies(code)
        final_description = extract_description(code, existing_skill.name) or existing_skill.description

        evolved = Skill(
            name=existing_skill.name,
            description=final_description,
            code=code,
            parameters=parameters,
            dependencies=dependencies,
            version=existing_skill.version,
            execution_mode=existing_skill.execution_mode,
            entry_script=existing_skill.entry_script,
        )

        logger.info(
            "Skill '%s' evolved (code=%d→%d bytes, deps=%s)",
            evolved.name, len(existing_skill.code), len(code), dependencies,
        )
        return evolved

    async def reflect_and_fix(self, skill: Skill, error_msg: str) -> Skill:
        logger.info("Reflection fix for '%s': %s", skill.name, error_msg[:200])
        prompt = REFLECTION_PROMPT.format(
            name=skill.name,
            error=error_msg,
            code=skill.code,
        )
        new_code = await self._call_llm(prompt)
        skill.code = new_code
        skill.parameters = extract_parameters(new_code, skill.name)
        skill.dependencies = extract_dependencies(new_code)

        if not validate_syntax(new_code):
            raise SkillCreationError(
                skill.name, f"Syntax still invalid after reflection fix: {error_msg}",
            )

        logger.info("Syntax validation passed after fix for '%s'", skill.name)
        return skill


    async def _determine_skill_type(self, description: str) -> str:
        result = (await self._call_llm(
            SKILL_TYPE_PROMPT.format(request=description),
            timeout=15,
        )).strip().lower()
        return result if result in ("code", "knowledge") else "code"

    async def _create_code_skill(self, name: str, description: str) -> Skill:
        prompt = GENERATE_PROMPT.format(name=name, request=description)
        code = await self._call_llm(prompt)

        parameters = extract_parameters(code, name)
        dependencies = extract_dependencies(code)
        final_description = extract_description(code, name) or description

        skill = Skill(
            name=name,
            description=final_description,
            code=code,
            parameters=parameters,
            dependencies=dependencies,
            execution_mode="function",
        )
        logger.info(
            "Code skill '%s' created (code=%d bytes, params=%d, deps=%s)",
            name, len(code), len(parameters), dependencies,
        )
        return skill

    async def _create_knowledge_skill(self, name: str, description: str) -> Skill:
        kebab_name = name.replace("_", "-")
        prompt = KNOWLEDGE_SKILL_PROMPT.format(
            name=name, kebab_name=kebab_name, request=description,
        )
        content = await self._call_llm(prompt)

        if not content.lstrip().startswith("---"):
            content = f"---\nname: {kebab_name}\ndescription: {description}\n---\n\n{content}"

        skill = Skill(
            name=name,
            description=description,
            code=content,
            execution_mode="knowledge",
        )
        logger.info("Knowledge skill '%s' created (content=%d bytes)", name, len(content))
        return skill

    async def _call_llm(self, prompt: str, timeout: float = 90) -> str:
        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                timeout=timeout,
            )
            content = (response.content or "").strip()
            m = _RE_OUTER_FENCE.match(content)
            if m:
                content = m.group(1).strip()
            return content
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            raise

    @staticmethod
    def _post_creation_audit(skill: Skill):
        issues: list[str] = []

        if skill.is_knowledge_skill:
            if not skill.code.lstrip().startswith("---"):
                issues.append("Knowledge skill missing YAML frontmatter")
            kebab_name = skill.name.replace("_", "-")
            if not re.match(r'^[a-z0-9]+(-[a-z0-9]+)*$', kebab_name):
                issues.append(f"Name '{skill.name}' is not valid kebab-case")
        else:
            if not validate_syntax(skill.code):
                issues.append("Generated code has syntax errors")

        if not skill.description or len(skill.description.strip()) < 10:
            issues.append("Description is too short (< 10 chars)")
        if len(skill.description) > 1024:
            issues.append(f"Description too long ({len(skill.description)} chars, max 1024)")

        for issue in issues:
            logger.warning("Post-creation audit for '%s': %s", skill.name, issue)
