"""Post-execute recovery policy decisions."""

from __future__ import annotations

from typing import Any

from core.skill.execution.policy.types import RecoveryAction, RecoveryDecision
from core.skill.schema import ErrorType


class RecoveryPolicy:
    """Error handling policy matrix for agent decisions."""

    _MATRIX: dict[ErrorType, RecoveryAction] = {
        ErrorType.INPUT_REQUIRED: RecoveryAction.PROMPT_USER,
        ErrorType.INPUT_INVALID: RecoveryAction.AUTO_FIX,
        ErrorType.RESOURCE_MISSING: RecoveryAction.AUTO_FIX,
        ErrorType.DEPENDENCY_ERROR: RecoveryAction.AUTO_FIX,
        ErrorType.PERMISSION_DENIED: RecoveryAction.PROMPT_USER,
        ErrorType.TIMEOUT: RecoveryAction.RETRY,
        ErrorType.ENVIRONMENT_ERROR: RecoveryAction.PROMPT_USER,
        ErrorType.UNAVAILABLE: RecoveryAction.RETRY,
        ErrorType.EXECUTION_ERROR: RecoveryAction.AUTO_FIX,
        ErrorType.TOOL_NOT_FOUND: RecoveryAction.AUTO_FIX,
        ErrorType.PATH_VALIDATION_FAILED: RecoveryAction.AUTO_FIX,
        ErrorType.POLICY_BLOCKED: RecoveryAction.ABORT,
        ErrorType.INTERNAL_ERROR: RecoveryAction.ABORT,
    }

    @staticmethod
    def decide_from_diagnostics(
        diagnostics: dict[str, Any] | None,
        *,
        success: bool,
        fallback_error: str | None = None,
    ) -> RecoveryDecision | None:
        if success or not diagnostics:
            return None

        if RecoveryPolicy._looks_like_success(diagnostics, fallback_error):
            return None

        error_type_value = diagnostics.get("error_type")
        if not error_type_value:
            return None

        try:
            error_type = ErrorType(error_type_value)
        except Exception:
            return None

        detail = diagnostics.get("error_detail") or {}
        if not isinstance(detail, dict):
            detail = {"raw_detail": detail}

        action = RecoveryPolicy._MATRIX.get(error_type, RecoveryAction.ABORT)

        retryable = bool(detail.get("retryable", False))
        if retryable and action in {RecoveryAction.AUTO_FIX, RecoveryAction.PROMPT_USER}:
            action = RecoveryAction.RETRY

        category = str(detail.get("category", "")).strip().lower()
        if category in {"permission", "environment"}:
            action = RecoveryAction.PROMPT_USER
        elif category in {"input_invalid", "resource_missing", "dependency", "tool_not_found"}:
            if action != RecoveryAction.ABORT:
                action = RecoveryAction.AUTO_FIX

        message = detail.get("message")
        hint = detail.get("hint")
        tool = detail.get("tool")

        reason_parts = [p for p in [message, fallback_error, error_type.value] if p]
        reason = reason_parts[0] if reason_parts else "skill execution failed"

        enriched_detail = {
            **detail,
            "resolved_action": action.value,
            "error_type": error_type.value,
        }
        if hint and "hint" not in enriched_detail:
            enriched_detail["hint"] = hint
        if tool and "tool" not in enriched_detail:
            enriched_detail["tool"] = tool

        return RecoveryDecision(action=action, reason=reason, detail=enriched_detail)

    @staticmethod
    def _looks_like_success(
        diagnostics: dict[str, Any],
        fallback_error: str | None,
    ) -> bool:
        detail = diagnostics.get("error_detail") or {}
        if not isinstance(detail, dict):
            return False

        decision_basis = detail.get("decision_basis") or {}
        if not isinstance(decision_basis, dict):
            return False

        state = str(decision_basis.get("state") or "").strip().lower()
        if state == "succeeded":
            return True
        if state == "failed":
            return False

        # unknown state should not bypass recovery
        return False
