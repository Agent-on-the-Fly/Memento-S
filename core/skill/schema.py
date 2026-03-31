"""技能领域模型（含 Agent-Skill 契约 DTO）。

所有 Skill 相关的数据模型集中定义在此文件。
"""

from __future__ import annotations

from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


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


class ExecutionMode(str, Enum):
    KNOWLEDGE = "knowledge"
    PLAYBOOK = "playbook"


class DiscoverStrategy(StrEnum):
    LOCAL_ONLY = "local_only"
    MULTI_RECALL = "multi_recall"


def _check_is_playbook(source_dir: str | None) -> bool:
    """Playbook = 目录里除了 SKILL.md 还有其他文件。"""
    if not source_dir:
        return False
    d = Path(source_dir)
    if not d.is_dir():
        return False
    for p in d.rglob("*"):
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if p.name == "SKILL.md" and p.parent == d:
            continue
        return True
    return False


class Skill(BaseModel):
    """技能定义。"""

    name: str = Field(..., description="技能名称，如 calculate_sum")
    description: str = Field(..., description="技能功能描述")
    content: str = Field(..., description="SKILL.md 内容")
    dependencies: list[str] = Field(default_factory=list, description="依赖包列表")
    version: int = Field(0, description="当前版本号")
    files: dict[str, str] = Field(default_factory=dict, description="技能文件")
    references: dict[str, str] = Field(
        default_factory=dict,
        description="references/ 目录下的文件（按 agentskills.io 规范单独存储）",
    )
    source_dir: Optional[str] = Field(None, description="技能目录路径")
    execution_mode: Optional[ExecutionMode] = Field(
        None,
        description="显式执行模式。None 时由目录结构推断",
    )
    entry_script: Optional[str] = Field(
        None,
        description="playbook 默认入口脚本名（无 .py）",
    )
    required_keys: list[str] = Field(
        default_factory=list,
        description="此 skill 运行所需的 API key 环境变量名，如 ['SERPER_API_KEY']",
    )
    parameters: Optional[dict[str, Any]] = Field(
        None,
        description="OpenAI/Anthropic 兼容的参数 schema。为 None 时由执行层推断",
    )
    allowed_tools: list[str] = Field(
        default_factory=list,
        description="此 skill 允许使用的工具列表（按 agentskills.io 规范，实验性功能）",
    )

    @property
    def is_playbook(self) -> bool:
        """判断是否为 playbook 类型 skill。

        execution_mode 在初始化时已确定，直接比较即可。
        """
        return self.execution_mode == ExecutionMode.PLAYBOOK

    @model_validator(mode="after")
    def _infer_execution_mode(self) -> "Skill":
        """如果未显式设置 execution_mode，通过目录结构推断。"""
        if self.execution_mode is None:
            is_pb = _check_is_playbook(self.source_dir)
            self.execution_mode = (
                ExecutionMode.PLAYBOOK if is_pb else ExecutionMode.KNOWLEDGE
            )
        return self

    def to_embedding_text(self) -> str:
        """生成用于 embedding 的文本（name + description）"""
        return f"{self.name.replace('_', ' ')} | {self.description}"


class ErrorType(str, Enum):
    """通用错误分类。"""

    INPUT_REQUIRED = "input_required"
    INPUT_INVALID = "input_invalid"
    RESOURCE_MISSING = "resource_missing"
    PERMISSION_DENIED = "permission_denied"
    TIMEOUT = "timeout"
    DEPENDENCY_ERROR = "dependency_error"
    EXECUTION_ERROR = "execution_error"
    TOOL_NOT_FOUND = "tool_not_found"
    POLICY_BLOCKED = "policy_blocked"
    PATH_VALIDATION_FAILED = "path_validation_failed"
    ENVIRONMENT_ERROR = "environment_error"
    UNAVAILABLE = "unavailable"
    INTERNAL_ERROR = "internal_error"


class SkillExecutionOutcome(BaseModel):
    """执行层内部结果。

    由SkillExecutor和Sandbox返回，包含详细的执行信息。
    在Provider层转换为SkillExecutionResponse对外暴露。
    """

    success: bool
    result: Any
    error: str | None = None
    error_type: ErrorType | None = None
    error_detail: dict[str, Any] | None = None
    skill_name: str
    artifacts: list[str] = []
    operation_results: list[dict[str, Any]] | None = (
        None  # 已执行的 builtin tool 调用明细
    )


# ------------------------------------------------------------------------------
# Gateway 契约 DTO（从 gateway.py 迁移过来）
# ------------------------------------------------------------------------------


class SkillStatus(str, Enum):
    """Skill 执行状态。"""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    BLOCKED = "blocked"
    TIMEOUT = "timeout"


class SkillErrorCode(str, Enum):
    """Skill 执行错误码。"""

    SKILL_NOT_FOUND = "SKILL_NOT_FOUND"
    INVALID_INPUT = "INVALID_INPUT"
    POLICY_DENIED = "POLICY_DENIED"
    DEPENDENCY_MISSING = "DEPENDENCY_MISSING"
    KEY_MISSING = "KEY_MISSING"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    TIMEOUT = "TIMEOUT"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class SkillGovernanceMeta(BaseModel):
    """Skill 治理元数据。"""

    source: Literal["local", "cloud"] = "local"


class SkillExecOptions(BaseModel):
    """Skill 执行选项。"""

    workdir: str | None = None
    timeout: int | None = None
    env: dict[str, str] = Field(default_factory=dict)


class SkillManifest(BaseModel):
    """Skill 元数据 - 用于发现和注册。"""

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
