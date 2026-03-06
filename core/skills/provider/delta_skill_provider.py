
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from core.config import g_settings
from core.config.logging import get_logger
from .delta_skills.importers.adapter import SkillAdapter
from .delta_skills.importers.utils import download_with_strategy
from .delta_skills.retrieval.cloud_catalog import CloudSkillEntry
from .protocol import (
    SkillExecuteResult,
    SkillInfo,
    SkillProvider,
    SkillResolveResult,
)

logger = get_logger(__name__)


class DeltaSkillsProvider(SkillProvider):

    def __init__(self, app_context: "AppContext"):  # noqa: F821
        self._library = app_context.library
        self._resolver = app_context.resolver
        self._creator = app_context.creator
        self._executor = app_context.resolver.executor
        self._cloud_catalog = app_context.cloud_catalog


    @staticmethod
    def _skill_name_matches(skill_name: str, query: str) -> bool:
        def _norm(s: str) -> str:
            return s.lower().replace("-", "_").replace(" ", "_")

        a = _norm(skill_name)
        b = _norm(query)
        return a == b or a.startswith(b + "_") or b.startswith(a + "_")

    async def discover_skill(
        self,
        name: str,
    ) -> SkillInfo | None:
        try:
            result = await self._resolver.resolve(
                query=name,
                execute=False,
            )

            if (
                result.skill
                and result.source == "local"
                and not self._skill_name_matches(result.skill.name, name)
            ):
                logger.info(
                    "discover_skill: local returned '%s' for query '%s' "
                    "(name mismatch, likely false-positive); skipping to LLM",
                    result.skill.name,
                    name,
                )
                result = await self._resolver.resolve(
                    query=name,
                    execute=False,
                    skip_stages={"local"},
                )

            if not result.skill:
                logger.info("discover_skill: '%s' not found via any channel", name)
                return None

            if result.source != "local":
                try:
                    self._library.add_skill(result.skill)
                    logger.info(
                        "discover_skill: '%s' (source=%s) added to library",
                        result.skill.name, result.source,
                    )
                except Exception as e:
                    logger.warning(
                        "discover_skill: failed to add '%s' to library: %s",
                        result.skill.name, e,
                    )

            return self._to_skill_info(result.skill, source=result.source or "local")

        except Exception as e:
            logger.warning("discover_skill failed for '%s': %s", name, e)
            return None

    def list_skills(self) -> list[SkillInfo]:
        return [
            self._to_skill_info(skill)
            for skill in self._library.local_cache.values()
        ]


    def retrieve_top_k(self, query: str, k: int = 3) -> list[SkillInfo]:
        local_infos = [self._to_skill_info(skill) for skill in self._library.local_cache.values()]
        local_names = {info.name for info in local_infos}

        try:
            cloud_results = self._cloud_catalog.search_embedding(query, k=k)
        except Exception as e:
            logger.warning("Cloud retrieval failed: %s", e)
            cloud_results = []

        cloud_infos = [
            self._cloud_entry_to_skill_info(entry)
            for entry, _ in cloud_results
            if entry.name not in local_names
        ]

        logger.info(
            "retrieve_top_k: %d local + %d cloud (embedding_ready=%s)",
            len(local_infos), len(cloud_infos),
            self._cloud_catalog.embedding_ready,
        )

        return local_infos + cloud_infos

    @staticmethod
    def _cloud_entry_to_skill_info(entry: CloudSkillEntry) -> SkillInfo:
        return SkillInfo(
            name=entry.name,
            description=(entry.description or "")[:500],
            tags=[],
            parameters={
                "type": "object",
                "properties": {
                    "request": {
                        "type": "string",
                        "description": (
                            "Describe clearly what you need this skill to do. "
                            "Include the user's original request and any relevant context."
                        ),
                    },
                },
                "required": ["request"],
            },
            source="cloud",
            github_url=entry.github_url,
        )

    async def resolve(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        context: list[str] | None = None,
    ) -> SkillResolveResult:
        params = params or {}

        enriched_query = query
        if context:
            context_text = "\n".join(context[-3:])
            enriched_query = f"{query}\n\nContext:\n{context_text}"

        skill = None
        source = "not_found"

        norm_query = query.strip().split("\n")[0].strip()
        local_match = self._library.local_cache.get(norm_query)
        if local_match:
            skill = local_match
            source = "local"
            logger.info("resolve: found '%s' in local library (direct match)", norm_query)

        if not skill:
            cloud_skill = await self._try_download_cloud_skill(query)
            if cloud_skill:
                skill = cloud_skill
                source = "cloud"

        if not skill:
            try:
                resolve_result = await self._resolver.resolve(
                    query=enriched_query,
                    execute=False,
                )
                skill = resolve_result.skill
                source = resolve_result.source or "not_found"
            except Exception as e:
                logger.error("resolve search failed: %s", e)

        if not skill:
            logger.info("resolve: no skill found for query '%s'", query[:80])
            return SkillResolveResult(source="not_found")

        logger.info(
            "resolve: found skill '%s' (source=%s, knowledge=%s, playbook=%s)",
            skill.name, source, skill.is_knowledge_skill, skill.is_playbook,
        )

        skill_info = self._to_skill_info(skill, source=source)

        exec_result, generated_code = await self._executor.execute(skill, query, params)

        if exec_result.success and source != "local":
            try:
                self._library.add_skill(skill)
                logger.info("Skill '%s' (source=%s) stored to local library", skill.name, source)
            except Exception as e:
                logger.warning("Failed to store skill '%s': %s", skill.name, e)

        return SkillResolveResult(
            skill=skill_info,
            execute_result=self._map_result(exec_result, generated_code),
            source=source,
        )

    async def _try_download_cloud_skill(self, query: str) -> "Skill | None":  # noqa: F821
        norm_query = query.strip().split("\n")[0].strip()  #  context 
        cloud_entry = self._cloud_catalog.get_by_name(norm_query)
        if not cloud_entry or not cloud_entry.github_url:
            return None

        logger.info(
            "resolve: downloading cloud skill '%s' from %s",
            cloud_entry.name, cloud_entry.github_url,
        )

        try:
            local_path = download_with_strategy(
                cloud_entry.github_url,
                g_settings.skills_directory,
                cloud_entry.name,
            )
            if not local_path:
                logger.warning("resolve: download failed for cloud skill '%s'", cloud_entry.name)
                return None

            skill = SkillAdapter.from_directory(local_path)

            if not skill.code or not skill.code.strip():
                skill_md = local_path / "SKILL.md"
                if skill_md.exists():
                    skill.code = skill_md.read_text(encoding="utf-8")

            logger.info("resolve: cloud skill '%s' downloaded and loaded", skill.name)
            return skill

        except Exception as e:
            logger.warning("resolve: failed to download/load cloud skill '%s': %s", cloud_entry.name, e)
            return None


    @staticmethod
    def _map_result(
        r: "SkillExecutionResult",  # noqa: F821
        generated_code: str,
    ) -> SkillExecuteResult:
        return SkillExecuteResult(
            success=r.success,
            output=r.result,
            error=r.error,
            skill_name=r.skill_name,
            generated_code=generated_code,
            artifacts=r.artifacts,
        )


    @staticmethod
    def _to_skill_info(skill: "Skill", source: str = "local") -> SkillInfo:  # noqa: F821
        is_knowledge = skill.is_knowledge_skill
        is_playbook = skill.is_playbook

        if is_knowledge:
            params = DeltaSkillsProvider._build_knowledge_params(skill)
            description = DeltaSkillsProvider._build_knowledge_description(skill)
        elif is_playbook:
            params, description = DeltaSkillsProvider._build_playbook_params(skill)
        else:
            params = skill.parameters
            description = skill.description or ""

        deps = skill.dependencies or []
        if deps:
            deps_str = ", ".join(deps)
            dep_note = (
                f"\n\n⚠ Required Python packages: {deps_str}. "
                "Ensure these are installed in the current environment "
                "before invoking this skill."
            )
            description = (description or "") + dep_note

        return SkillInfo(
            name=skill.name,
            description=description,
            tags=skill.tags,
            parameters=params,
            is_knowledge=is_knowledge,
            dependencies=skill.dependencies,
            source=source,
        )

    @staticmethod
    def _build_knowledge_params(skill: "Skill") -> dict[str, Any]:  # noqa: F821
        return {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": (
                        "Describe clearly what you need this skill to do. "
                        "Include the user's original request and any relevant context."
                    ),
                },
                "code": {
                    "type": "string",
                    "description": (
                        "Optional Python code to execute for this skill's operations. "
                        "If omitted, the skill will handle the request using its "
                        "built-in instructions. Only provide code when the task "
                        "clearly requires custom programmatic execution."
                    ),
                },
            },
            "required": ["request"],
        }

    @staticmethod
    def _build_knowledge_description(skill: "Skill") -> str:  # noqa: F821
        md_content = skill.code if skill.code else ""

        md_content = re.sub(
            r"^---\s*\n.*?\n---\s*\n?", "", md_content, count=1, flags=re.DOTALL,
        )

        base_desc = skill.description or ""
        if md_content.strip():
            return f"{base_desc}\n\n{md_content.strip()}"
        return base_desc

    @staticmethod
    def _build_playbook_params(skill: "Skill") -> tuple[dict[str, Any], str]:  # noqa: F821
        scripts_dir = Path(skill.source_dir) / "scripts"
        script_names = sorted(
            p.stem for p in scripts_dir.glob("*.py")
            if p.name != "__init__.py" and not p.name.endswith("_test.py")
        )

        script_list = ", ".join(script_names)
        description = f"{skill.description} Available scripts: {script_list}"

        params: dict[str, Any] = {
            "type": "object",
            "properties": {
                "script": {
                    "type": "string",
                    "description": f"Script to run (without .py). Available: {script_list}",
                    "enum": script_names,
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "CLI arguments to pass to the script",
                },
            },
            "required": ["script"],
        }
        return params, description
