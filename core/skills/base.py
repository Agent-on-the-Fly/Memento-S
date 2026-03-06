
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable

from .schema import skill_to_openai_tool


class Skill(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        ...

    def get_deep_context(self) -> str | None:
        return None

    def to_openai_schema(self) -> dict[str, Any]:
        return skill_to_openai_tool(self.name, self.description, self.parameters)

    def __repr__(self) -> str:
        return f"<Skill: {self.name}>"


class _SkillFromTool(Skill):

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        executor: Callable[..., Awaitable[str]],
    ) -> None:
        self._name = name
        self._description = description
        self._parameters = parameters
        self._executor = executor

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    async def execute(self, **kwargs: Any) -> str:
        return await self._executor(**kwargs)


def openai_tool_to_skill(
    schema: dict[str, Any],
    executor: Callable[..., Awaitable[str]],
) -> Skill:
    fn = schema.get("function", schema)
    name = str(fn.get("name", ""))
    description = str(fn.get("description", ""))
    parameters = fn.get("parameters") or {"type": "object", "properties": {}}
    return _SkillFromTool(name=name, description=description, parameters=parameters, executor=executor)
