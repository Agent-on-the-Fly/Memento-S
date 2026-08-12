# SPDX-License-Identifier: Apache-2.0
"""Post-execute recovery policy decisions."""

from __future__ import annotations

from typing import Any

from core.skill.execution.policy.types import RecoveryAction, RecoveryDecision
from core.skill.schema import ErrorType


# ---------------------------------------------------------------------------
# Retry decision helper — evaluates whether a tool error should be retried
# ---------------------------------------------------------------------------

DEFAULT_MAX_RETRIES = 2


def decide_and_act(
    error_type: ErrorType,
    error_detail: dict[str, Any],
    attempt: int,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> tuple[RecoveryAction, bool]:
    """
    根据 error_type 决定 RecoveryAction 并判断是否应继续重试。

    Args:
        error_type: 分类后的错误类型。
        error_detail: classify_error() 产出的错误详情字典。
        attempt: 当前尝试次数（0-based）。
        max_retries: 最大重试次数。

    Returns:
        (action, should_continue) — action 指示应采取的行动，
        should_continue 指示是否继续重试。
    """
    action = RecoveryPolicy._MATRIX.get(error_type, RecoveryAction.AUTO_FIX)

    retryable = bool(error_detail.get("retryable", False))

    if action == RecoveryAction.RETRY:
        should_continue = attempt < max_retries
        return action, should_continue

    if action == RecoveryAction.ABORT:
        return action, False

    # AUTO_FIX / PROMPT_USER: 仅当 retryable=True 时升级为 RETRY
    if retryable and action in {RecoveryAction.AUTO_FIX, RecoveryAction.PROMPT_USER}:
        should_continue = attempt < max_retries
        return RecoveryAction.RETRY, should_continue

    return action, False


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

        error_detail = diagnostics.get("error_detail") or {}
        if not isinstance(error_detail, dict):
            error_detail = {"raw_detail": error_detail}

        # 关键修复：如果有产物，即使错误也应该继续执行后续步骤
        # 这是打通产物传递链的核心：允许 PARTIAL 状态（有产物但未完全成功）继续
        has_artifacts = diagnostics.get("has_artifacts", False)
        artifacts = diagnostics.get("artifacts") or []
        if has_artifacts or (isinstance(artifacts, list) and len(artifacts) > 0):
            return RecoveryDecision(
                action=RecoveryAction.CONTINUE,
                reason="Has artifacts — allow next step to continue",
                detail={**error_detail, "continue_reason": "artifacts_exist"},
            )

        error_type_value = diagnostics.get("error_type")
        if not error_type_value:
            return None

        try:
            error_type = ErrorType(error_type_value)
        except Exception:
            return None

        action = RecoveryPolicy._MATRIX.get(error_type, RecoveryAction.ABORT)

        retryable = bool(error_detail.get("retryable", False))
        if retryable and action in {RecoveryAction.AUTO_FIX, RecoveryAction.PROMPT_USER}:
            action = RecoveryAction.RETRY

        category = str(error_detail.get("category", "")).strip().lower()
        if category in {"permission", "environment"}:
            action = RecoveryAction.PROMPT_USER
        elif category in {"input_invalid", "resource_missing", "dependency", "tool_not_found"}:
            if action != RecoveryAction.ABORT:
                action = RecoveryAction.AUTO_FIX

        message = error_detail.get("message")
        hint = error_detail.get("hint")
        tool = error_detail.get("tool")

        reason_parts = [p for p in [message, fallback_error, error_type.value] if p]
        reason = reason_parts[0] if reason_parts else "skill execution failed"

        enriched_detail = {
            **error_detail,
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
