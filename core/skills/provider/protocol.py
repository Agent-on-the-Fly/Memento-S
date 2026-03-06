
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class SkillInfo:

    name: str
    description: str
    tags: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    is_knowledge: bool = False
    dependencies: list[str] = field(default_factory=list)
    source: str = "local"
    github_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "parameters": self.parameters,
            "is_knowledge": self.is_knowledge,
            "dependencies": self.dependencies,
            "source": self.source,
            "github_url": self.github_url,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillInfo:
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            tags=data.get("tags", []),
            parameters=data.get("parameters", {}),
            is_knowledge=data.get("is_knowledge", False),
            dependencies=data.get("dependencies", []),
            source=data.get("source", "local"),
            github_url=data.get("github_url", ""),
        )


@dataclass
class SkillExecuteResult:

    success: bool
    output: Any = None
    error: str | None = None
    skill_name: str = ""
    generated_code: str = ""
    artifacts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "skill_name": self.skill_name,
            "generated_code": self.generated_code,
            "artifacts": self.artifacts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillExecuteResult:
        return cls(
            success=data["success"],
            output=data.get("output"),
            error=data.get("error"),
            skill_name=data.get("skill_name", ""),
            generated_code=data.get("generated_code", ""),
            artifacts=data.get("artifacts", []),
        )


@dataclass
class SkillResolveResult:

    skill: SkillInfo | None = None
    dependencies: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    execute_result: SkillExecuteResult | None = None
    source: str = "not_found"

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill.to_dict() if self.skill else None,
            "dependencies": self.dependencies,
            "parameters": self.parameters,
            "execute_result": self.execute_result.to_dict() if self.execute_result else None,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillResolveResult:
        return cls(
            skill=SkillInfo.from_dict(data["skill"]) if data.get("skill") else None,
            dependencies=data.get("dependencies", []),
            parameters=data.get("parameters", {}),
            execute_result=SkillExecuteResult.from_dict(data["execute_result"]) if data.get("execute_result") else None,
            source=data.get("source", "not_found"),
        )


@runtime_checkable
class SkillProvider(Protocol):

    def list_skills(self) -> list[SkillInfo]:
        ...

    async def resolve(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        context: list[str] | None = None,
    ) -> SkillResolveResult:
        ...
