
from __future__ import annotations

import asyncio
import json
from typing import Any

from core.config.logging import get_logger
from .metadata import (
    clean_description,
    compress_description,
    normalize_parameters,
    normalize_skill_name,
    skill_to_tool_metadata,
)
from .provider.protocol import SkillExecuteResult, SkillInfo, SkillProvider
from .schema import SkillCall, SkillResult

logger = get_logger(__name__)


class _InfoAdapter:

    __slots__ = ("name", "description", "parameters")

    def __init__(self, info: SkillInfo) -> None:
        self.name = info.name
        self.description = info.description or ""
        self.parameters = info.parameters if isinstance(info.parameters, dict) else {"type": "object", "properties": {}}


class SkillManager:

    def __init__(
        self,
        provider: SkillProvider,
        *,
        name_prefix: str | None = None,
        max_description_chars: int = 500,
    ) -> None:
        self._provider = provider
        self._always_skill_names: list[str] | None = None
        self._normalized_to_original: dict[str, str] = {}
        self._skill_source_map: dict[str, dict[str, str]] = {}
        self._name_prefix = name_prefix
        self._max_description_chars = max_description_chars

    def register(self, skill: Any) -> None:
        if hasattr(self._provider, "register"):
            self._provider.register(skill)
            logger.debug("Registered skill via provider: %s", getattr(skill, "name", skill))
        else:
            logger.debug("Provider does not support register(); no-op.")

    def unregister(self, name: str) -> bool:
        if hasattr(self._provider, "unregister"):
            return bool(self._provider.unregister(name))
        return False

    def get(self, name: str) -> _InfoAdapter | None:
        original = self._normalized_to_original.get(name, name)
        infos = self._provider.list_skills()
        match = next((i for i in infos if i.name == original), None)
        return _InfoAdapter(match) if match else None

    @property
    def skill_names(self) -> list[str]:
        return [info.name for info in self._provider.list_skills()]

    def get_schemas(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        self._normalized_to_original.clear()
        infos = self._provider.list_skills()
        if names is not None:
            name_set = set(names)
            infos = [i for i in infos if i.name in name_set]
        result = []
        seen_names: set[str] = set()
        for info in infos:
            adapter = _InfoAdapter(info)
            meta = skill_to_tool_metadata(
                adapter,
                name_prefix=self._name_prefix,
                max_description_chars=self._max_description_chars,
            )
            tool_name = meta["function"]["name"]
            if tool_name in seen_names:
                continue
            seen_names.add(tool_name)
            self._normalized_to_original[tool_name] = info.name
            result.append(meta)
        return result

    def get_schemas_for_query(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        if not hasattr(self._provider, "retrieve_top_k"):
            logger.debug("Provider does not support retrieve_top_k(); falling back to get_schemas()")
            return self.get_schemas()

        self._normalized_to_original.clear()
        self._skill_source_map.clear()

        infos = self._provider.retrieve_top_k(query, k=k)
        result = []
        seen_names: set[str] = set()
        for info in infos:
            adapter = _InfoAdapter(info)
            meta = skill_to_tool_metadata(
                adapter,
                name_prefix=self._name_prefix,
                max_description_chars=self._max_description_chars,
            )
            tool_name = meta["function"]["name"]
            if tool_name in seen_names:
                logger.debug("Skipping duplicate tool name: %s (source=%s)", tool_name, info.source)
                continue
            seen_names.add(tool_name)
            self._normalized_to_original[tool_name] = info.name
            self._skill_source_map[tool_name] = {
                "source": info.source,
                "github_url": info.github_url,
            }
            result.append(meta)

        sources = [self._skill_source_map.get(m["function"]["name"], {}).get("source", "?") for m in result]
        logger.info(
            "get_schemas_for_query: %d skill(s) [%s]",
            len(result),
            ", ".join(
                f"{m['function']['name']}({s})"
                for m, s in zip(result, sources)
            ),
        )
        return result

    def get_always_skill_names(self) -> list[str]:
        if self._always_skill_names is not None:
            available = set(self.skill_names)
            return [n for n in self._always_skill_names if n in available]
        return self.skill_names

    def set_always_skill_names(self, names: list[str] | None) -> None:
        self._always_skill_names = names

    def get_matched_skills_context(self, query: str, k: int = 5) -> str:
        if not hasattr(self._provider, "retrieve_top_k"):
            return ""

        infos = self._provider.retrieve_top_k(query, k=k)
        if not infos:
            return ""

        lines = [
            "[Matched Skills]",
            "Skills relevant to your query have been pre-selected. "
            "You MUST call `read_skill(skill_name)` first to get the correct path and usage. "
            "NEVER guess import paths like `from skills.xxx import ...` — they will fail.",
            "",
        ]
        for info in infos:
            name = info.name.strip()
            desc = (info.description or "").strip()
            entry = f"- {name}: {desc}" if desc else f"- {name}"
            lines.append(entry)
        lines.append("[/Matched Skills]")
        return "\n".join(lines)

    def build_skills_summary(self) -> str:
        lines = []
        for info in self._provider.list_skills():
            norm_name = normalize_skill_name(info.name, prefix=self._name_prefix)
            desc = compress_description(
                info.description or "",
                max_chars=self._max_description_chars,
            )
            lines.append(f"- **{norm_name}**: {desc}")
        return "\n".join(sorted(lines)) if lines else ""

    def load_skills_for_context(self, skill_names: list[str]) -> str:
        parts = []
        for name in skill_names:
            adapter = self.get(name)
            if not adapter:
                continue
            norm_name = normalize_skill_name(adapter.name, prefix=self._name_prefix)
            desc = clean_description(getattr(adapter, "description", "") or "")
            parts.append(f"### {norm_name}\n{desc}")
            params = normalize_parameters(getattr(adapter, "parameters", None) or {})
            if params.get("properties"):
                parts.append("Parameters: " + json.dumps(params.get("properties", {}), ensure_ascii=False))
        return "\n\n".join(parts) if parts else ""

    async def discover_skill(
        self,
        name: str,
    ) -> bool:
        if hasattr(self._provider, "discover_skill"):
            info = await self._provider.discover_skill(name)
            return info is not None
        logger.debug("Provider does not support discover_skill(); skipping.")
        return False

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        original = self._normalized_to_original.get(name, name)

        result = await self._provider.resolve(
            query=original,
            params=arguments,
        )
        if result.execute_result and result.execute_result.success:
            output = str(result.execute_result.output) if result.execute_result.output is not None else ""
            if not output.strip():
                return "Skill executed successfully."
            return output
        if result.execute_result:
            return f"Skill execution error: {result.execute_result.error or 'Unknown error'}"
        return f"Skill '{original}' not found."

    async def call_many(
        self,
        skill_calls: list[SkillCall] | list[dict[str, Any]],
    ) -> list[SkillResult]:
        normalized: list[tuple[str, str, dict[str, Any]]] = []
        for sc in skill_calls:
            if isinstance(sc, SkillCall):
                normalized.append((sc.id, sc.name, sc.arguments))
            else:
                normalized.append(
                    (
                        sc.get("id", ""),
                        sc.get("name", ""),
                        sc.get("arguments", {}),
                    )
                )

        async def _run_one(tuple_item: tuple[str, str, dict[str, Any]]) -> SkillResult:
            call_id, name, arguments = tuple_item
            try:
                result = await self.call(name, arguments)
                return SkillResult(skill_call_id=call_id, name=name, result=result, error=False)
            except Exception as exc:
                return SkillResult(skill_call_id=call_id, name=name, result=str(exc), error=True)

        return list(await asyncio.gather(*[_run_one(t) for t in normalized]))

    def __len__(self) -> int:
        return len(self.skill_names)

    def __repr__(self) -> str:
        return f"<SkillManager provider={type(self._provider).__name__} skills={self.skill_names}>"
