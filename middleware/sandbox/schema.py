"""Sandbox schema definitions.

Base types for sandbox execution results.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


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


class SandboxExecutionOutcome:
    """沙箱执行结果。

    由 Sandbox 返回，包含详细的执行信息。
    """

    def __init__(
        self,
        success: bool,
        result: Any,
        error: str | None = None,
        error_type: ErrorType | None = None,
        error_detail: dict[str, Any] | None = None,
        skill_name: str = "",
        artifacts: list[str] | None = None,
    ):
        self.success = success
        self.result = result
        self.error = error
        self.error_type = error_type
        self.error_detail = error_detail
        self.skill_name = skill_name
        self.artifacts = artifacts or []

    def __repr__(self) -> str:
        return f"SandboxExecutionOutcome(success={self.success}, skill_name={self.skill_name})"


__all__ = [
    "ErrorType",
    "SandboxExecutionOutcome",
]
