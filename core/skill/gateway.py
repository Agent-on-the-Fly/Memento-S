"""Agent-Skill 契约层：Protocol + DTO。"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from core.skill.schema import ExecutionMode


# 默认 skill 参数 schema - 单个自然语言请求
# 仅作为兼容层，新 skill 应自行定义 parameters
DEFAULT_SKILL_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "request": {
            "type": "string",
            "description": "Describe clearly what you need this skill to do.",
        },
    },
    "required": ["request"],
}


class SkillStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    BLOCKED = "blocked"
    TIMEOUT = "timeout"


class SkillErrorCode(str, Enum):
    SKILL_NOT_FOUND = "SKILL_NOT_FOUND"
    INVALID_INPUT = "INVALID_INPUT"
    POLICY_DENIED = "POLICY_DENIED"
    DEPENDENCY_MISSING = "DEPENDENCY_MISSING"
    KEY_MISSING = "KEY_MISSING"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    TIMEOUT = "TIMEOUT"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class SkillGovernanceMeta(BaseModel):
    source: Literal["local", "cloud", "builtin"] = "local"


class SkillExecOptions(BaseModel):
    """Skill 执行选项"""

    workdir: str | None = None
    timeout: int | None = None
    env: dict[str, str] = Field(default_factory=dict)


class SkillManifest(BaseModel):
    """Skill 元数据 - 用于发现和注册"""

    name: str
    description: str
    execution_mode: ExecutionMode
    # parameters 为 None 表示 skill 自描述，不由 manifest 强制指定
    parameters: dict[str, Any] | None = None
    dependencies: list[str] = Field(default_factory=list)
    governance: SkillGovernanceMeta = Field(default_factory=SkillGovernanceMeta)


class SkillExecutionResponse(BaseModel):
    """Agent契约：Skill执行响应。

    这是SkillGateway对外暴露的统一响应格式，由Provider层转换执行层结果后返回。
    """

    ok: bool
    status: SkillStatus
    error_code: SkillErrorCode | None = None
    summary: str = ""
    output: Any = None
    outputs: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    skill_name: str = ""


class SkillGateway(Protocol):
    """Agent 层依赖的唯一 Skill 契约。"""

    def discover(self) -> list[SkillManifest]: ...

    async def search(
        self, query: str, k: int = 5, cloud_only: bool = False
    ) -> list[SkillManifest]: ...

    async def execute(
        self,
        skill_name: str,
        params: dict[str, Any],
        options: SkillExecOptions | None = None,
    ) -> SkillExecutionResponse: ...


async def create_gateway() -> SkillGateway:
    """创建 SkillGateway 实例（工厂函数）。

    内部调用 Provider 的工厂方法。Skill 同步应由调用方（如 bootstrap）
    在使用此函数前完成。

    Returns:
        SkillGateway 实例
    """
    from core.skill.provider import SkillProvider

    return await SkillProvider.create_default()
