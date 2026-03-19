"""技能领域模型（不含 Agent-Skill 契约 DTO）。"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Optional

from enum import Enum

from pydantic import BaseModel, Field


class ExecutionMode(str, Enum):
    KNOWLEDGE = "knowledge"
    PLAYBOOK = "playbook"


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
        if self.execution_mode is not None:
            return self.execution_mode == ExecutionMode.PLAYBOOK
        return _check_is_playbook(self.source_dir)

    def to_embedding_text(self) -> str:
        parts = [self.name.replace("_", " "), self.description]
        if self.content:
            parts.append(self.content)
        if self.dependencies:
            parts.append(f"dependencies: {' '.join(self.dependencies)}")
        return " | ".join(parts)


class ErrorType(str, Enum):
    """通用错误分类。"""

    INPUT_REQUIRED = "input_required"
    INPUT_INVALID = "input_invalid"
    RESOURCE_MISSING = "resource_missing"
    PERMISSION_DENIED = "permission_denied"
    TIMEOUT = "timeout"
    DEPENDENCY_ERROR = "dependency_error"
    EXECUTION_ERROR = "execution_error"
    POLICY_BLOCKED = "policy_blocked"
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
