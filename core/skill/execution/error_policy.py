"""Error policy matrix for skill execution outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from core.skill.schema import ErrorType


class ErrorAction(str, Enum):
    ABORT = "abort"
    RETRY = "retry"
    AUTO_FIX = "auto_fix"
    PROMPT_USER = "prompt_user"


@dataclass(frozen=True)
class ErrorPolicyDecision:
    action: ErrorAction
    reason: str
    detail: dict[str, Any] | None = None


class ErrorPolicy:
    """Error handling policy matrix for agent decisions."""

    _MATRIX: dict[ErrorType, ErrorAction] = {
        ErrorType.INPUT_REQUIRED: ErrorAction.PROMPT_USER,
        ErrorType.INPUT_INVALID: ErrorAction.AUTO_FIX,
        ErrorType.RESOURCE_MISSING: ErrorAction.AUTO_FIX,
        ErrorType.DEPENDENCY_ERROR: ErrorAction.AUTO_FIX,
        ErrorType.PERMISSION_DENIED: ErrorAction.PROMPT_USER,
        ErrorType.TIMEOUT: ErrorAction.RETRY,
        ErrorType.ENVIRONMENT_ERROR: ErrorAction.PROMPT_USER,
        ErrorType.UNAVAILABLE: ErrorAction.RETRY,
        ErrorType.EXECUTION_ERROR: ErrorAction.AUTO_FIX,
        ErrorType.POLICY_BLOCKED: ErrorAction.ABORT,
        ErrorType.INTERNAL_ERROR: ErrorAction.ABORT,
    }

    @staticmethod
    def decide_from_diagnostics(
        diagnostics: dict[str, Any] | None,
        *,
        success: bool,
        fallback_error: str | None = None,
    ) -> ErrorPolicyDecision | None:
        """Return a decision based on diagnostics from SkillExecutionResponse."""
        if success:
            return None
        if not diagnostics:
            return None

        error_type_value = diagnostics.get("error_type")
        if not error_type_value:
            return None

        try:
            error_type = ErrorType(error_type_value)
        except Exception:
            return None

        action = ErrorPolicy._MATRIX.get(error_type, ErrorAction.ABORT)
        detail = diagnostics.get("error_detail") or {}
        reason = detail.get("message") or fallback_error or error_type.value
        return ErrorPolicyDecision(action=action, reason=reason, detail=detail)
