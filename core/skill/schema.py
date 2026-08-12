# SPDX-License-Identifier: Apache-2.0
"""技能领域模型（含 Agent-Skill 契约 DTO）。

所有 Skill 相关的数据模型集中定义在此文件。

注意：ExecutionMode 已统一迁移到 shared/schema/skill.py，
避免 shared/schema 与 core/skill 之间的循环依赖。
"""

from __future__ import annotations

from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

# ExecutionMode 从 shared.schema 导入（避免与 shared/schema/skill.py 重复定义）
from shared.schema.skill import ExecutionMode


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


class DiscoverStrategy(StrEnum):
    LOCAL_ONLY = "local_only"
    MULTI_RECALL = "multi_recall"


def _doc_only_skill_root_file(filename: str) -> bool:
    """Root files that do not imply a runnable playbook (LICENSE, README, etc.)."""
    low = filename.lower()
    if low == "skill.md":
        return True
    if low == "license" or low.startswith("license."):
        return True
    if low.startswith("readme"):
        return True
    if low.startswith("changelog"):
        return True
    if low.startswith("contributing"):
        return True
    if low.startswith("code_of_conduct"):
        return True
    if low in {"copying", "authors", "notice", "security.md", "maintainers.md"}:
        return True
    return False


def _is_executable_file(filename: str) -> bool:
    """判断是否为可执行/可运行的入口文件。"""
    low = filename.lower()

    # 1. 常见脚本语言后缀
    if low.endswith(
        (".py", ".js", ".mjs", ".ts", ".sh", ".bash", ".zsh", ".rb", ".go", ".java", ".rs", ".php", ".pl")
    ):
        return True

    # 2. 约定俗成的入口文件名（无扩展名）
    if low in {"run", "main", "agent", "execute", "skill", "start", "dev", "index"}:
        return True

    # 3. 带扩展名的约定入口文件
    if low in {"index.js", "index.ts", "index.mjs", "main.js", "main.ts", "server.js", "app.js", "entry.js", "entry.ts"}:
        return True

    # 4. 项目配置文件（表明是运行时项目）
    if low in {"package.json", "makefile", "dockerfile", "justfile"}:
        return True

    return False


def _check_is_playbook(source_dir: str | None) -> bool:
    """Playbook = 根目录下存在可执行/可运行文件。"""
    if not source_dir:
        return False
    d = Path(source_dir)
    if not d.is_dir():
        return False
    for p in d.iterdir():
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if _doc_only_skill_root_file(p.name):
            continue
        if _is_executable_file(p.name):
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
    license: Optional[str] = Field(
        None,
        description="许可证名称或指向 bundled LICENSE 文件的引用（agentskills.io 规范）",
    )
    compatibility: Optional[str] = Field(
        None,
        description="环境依赖说明，如所需系统包、网络访问等（agentskills.io 规范）",
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
