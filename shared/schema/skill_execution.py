# SPDX-License-Identifier: Apache-2.0
"""shared/schema/skill_execution.py — Skill 执行层契约类型（跨层共享）。

从 core/skill/schema.py 迁移：
  - SkillExecutionResponse（被 middleware/storage/ 重导出）
  - SkillErrorCode（与 SkillExecutionResponse 强绑定）
  - SkillStatus（SkillExecutionResponse 的字段类型）

保留在 core/skill/schema.py 的类型：
  - SkillExecutionOutcome（仅 execution 内部使用）
  - ErrorType（执行层内部错误分类）
  - SkillExecOptions（gateway 内部参数）
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


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


class SkillExecutionResponse(BaseModel):
    """Agent 契约：Skill 执行响应。

    这是 SkillGateway 对外暴露的统一响应格式，由 Provider 层转换执行层结果后返回。
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


__all__ = ["SkillStatus", "SkillErrorCode", "SkillExecutionResponse"]
