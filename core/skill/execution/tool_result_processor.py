# SPDX-License-Identifier: Apache-2.0
"""Tool result processing: warnings, retry decisions, error classification."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

from core.skill.execution.policy.recovery import DEFAULT_MAX_RETRIES, decide_and_act
from core.skill.execution.policy.types import RecoveryAction
from core.skill.schema import ErrorType
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class _Rule:
    error_type: ErrorType
    patterns: tuple[str, ...]
    category: str
    hint: str
    retryable: bool = False


@dataclass(frozen=True)
class ToolResultProcessingOutput:
    warning: str | None
    classified_error: tuple[ErrorType, dict[str, Any]] | None
    summary: str
    decision_basis: dict[str, Any]


class ToolResultProcessor:
    """Process tool outputs into warning + classified error signals."""

    _RULES: tuple[_Rule, ...] = (
        _Rule(
            error_type=ErrorType.TIMEOUT,
            patterns=(
                r"\btimeout\b",
                r"timed out",
                r"deadline exceeded",
                r"operation timed out",
                r"read timed out",
            ),
            category="timeout",
            hint="Increase timeout, split task, or retry with lighter command.",
            retryable=True,
        ),
        _Rule(
            error_type=ErrorType.PERMISSION_DENIED,
            patterns=(
                r"permission denied",
                r"operation not permitted",
                r"access denied",
                r"eacces",
                r"eprem",
                r"authorization failed",
            ),
            category="permission",
            hint="Adjust file/path permissions or request a permitted execution path.",
        ),
        _Rule(
            error_type=ErrorType.TOOL_NOT_FOUND,
            patterns=(
                r"command not found",
                r"not found",
                r"no such file or directory",
                r"unknown command",
                r"executable file not found",
            ),
            category="tool_not_found",
            hint="Check command/tool name, installation, and absolute path.",
        ),
        _Rule(
            error_type=ErrorType.DEPENDENCY_ERROR,
            patterns=(
                r"module not found",
                r"modulenotfounderror",
                r"cannot import name",
                r"importerror",
                r"no matching distribution found",
                r"failed building wheel",
                r"unsatisfied requirement",
            ),
            category="dependency",
            hint="Install missing dependencies or pin compatible versions.",
        ),
        _Rule(
            error_type=ErrorType.RESOURCE_MISSING,
            patterns=(
                r"file not found",
                r"no such file",
                r"does not exist",
                r"cannot find",
                r"not a directory",
            ),
            category="resource_missing",
            hint="Check path correctness and ensure required files exist.",
        ),
        _Rule(
            error_type=ErrorType.INPUT_INVALID,
            patterns=(
                r"invalid argument",
                r"unrecognized arguments?",
                r"bad request",
                r"invalid request",
                r"validation error",
                r"json decode error",
                r"expecting value",
            ),
            category="input_invalid",
            hint="Validate tool arguments and data format before retrying.",
        ),
        _Rule(
            error_type=ErrorType.UNAVAILABLE,
            patterns=(
                r"service unavailable",
                r"temporarily unavailable",
                r"connection refused",
                r"network is unreachable",
                r"name or service not known",
                r"dns",
                r"502 bad gateway",
                r"503",
                r"429",
                r"rate limit",
                r"too many requests",
            ),
            category="service_unavailable",
            hint="Retry with backoff or switch endpoint/network.",
            retryable=True,
        ),
        _Rule(
            error_type=ErrorType.ENVIRONMENT_ERROR,
            patterns=(
                r"environment variable",
                r"keyerror",
                r"api[_ ]?key",
                r"missing credentials",
                r"not configured",
            ),
            category="environment",
            hint="Provide required environment variables and credentials.",
        ),
    )

    async def process(
        self,
        *,
        tool_name: str,
        tool_result: Any,
        args: dict[str, Any],
        runner,
    ) -> tuple[Any, ToolResultProcessingOutput]:
        """Process result and optionally perform internal retry for bash/http errors."""
        warning: str | None = None
        final_result = tool_result
        retry_performed = False

        if tool_name == "bash" and isinstance(tool_result, str):
            retry_result = await self._retry_bash_on_http_error(
                runner=runner,
                args=args,
                tool_output=tool_result,
            )
            if retry_result is not None:
                final_result = retry_result
                retry_performed = True
            elif self._is_nonfatal_http_error(tool_result):
                warning = "Non-fatal HTTP error detected (request failure)."

        state = self.assess_execution_state(final_result)
        classified = None
        if not warning and state["state"] == "failed":
            classified = self.classify(tool_name, final_result)

        summary = self.summarize(tool_name, final_result)

        decision_basis = {
            "tool": tool_name,
            "retry_performed": retry_performed,
            "warning": warning,
            "state": state["state"],
            "state_reason": state["reason"],
            "classified_error": classified[0].value if classified else None,
        }

        return final_result, ToolResultProcessingOutput(
            warning=warning,
            classified_error=classified,
            summary=summary,
            decision_basis=decision_basis,
        )

    async def _retry_bash_on_http_error(
        self,
        *,
        runner,
        args: dict[str, Any],
        tool_output: str,
    ) -> Any | None:
        if not self._is_nonfatal_http_error(tool_output):
            return None

        command = str(args.get("command", ""))
        if not command or "http" not in command or "__http_retry__" in command:
            return None

        retry_args = dict(args)
        retry_args["command"] = f"__http_retry__=1 {command}"
        logger.info("Retrying bash command after HTTP client error")
        return await runner.run("bash", retry_args)

    async def execute_with_retry(
        self,
        runner,
        tool_name: str,
        args: dict[str, Any],
        error_detail: dict[str, Any] | None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> tuple[Any, dict[str, Any] | None]:
        """
        根据 classify_error() 产出的 error_detail，决定是否自动重试。

        利用已有的 `retryable` 标志和 `RecoveryPolicy._MATRIX` 做决策。
        采用指数退避：delay = min(1.0 * 2**attempt, 30.0) 秒。

        Args:
            runner: SkillToolAdapter 实例（提供 .run() 方法）。
            tool_name: 工具名。
            args: 工具参数。
            error_detail: classify_error() 产出的错误详情，若为 None 表示无错误。
            max_retries: 最大重试次数。

        Returns:
            (final_result, remaining_error_dict) — 若重试成功则 second 为 None。
        """
        if error_detail is None:
            return None, None  # 无错误，无需重试

        error_type = ErrorType(error_detail.get("error_type", "execution_error"))
        action, should_continue = decide_and_act(
            error_type, error_detail, attempt=0, max_retries=max_retries
        )

        if action != RecoveryAction.RETRY:
            # 不重试，返回原错误
            return None, error_detail

        # 指数退避重试循环
        for attempt in range(max_retries):
            delay = min(1.0 * (2**attempt), 30.0)
            if delay > 0:
                logger.info(
                    f"[execute_with_retry] retry {attempt + 1}/{max_retries} for "
                    f"tool={tool_name} after {delay:.1f}s"
                )
                await asyncio.sleep(delay)

            result = await runner.run(tool_name, args)
            if isinstance(result, tuple) and len(result) == 2:
                retry_obs, retry_err = result
            else:
                retry_obs, retry_err = str(result), None

            if retry_err is None:
                logger.info(f"[execute_with_retry] success on attempt {attempt + 1}")
                return retry_obs, None

            # 重新分类，判断是否继续
            classified = self.classify(tool_name, retry_obs)
            if classified is None:
                retry_error_detail = {
                    "tool": tool_name,
                    "category": "execution",
                    "message": str(retry_obs)[:200],
                    "hint": "Inspect tool output.",
                    "retryable": False,
                }
            else:
                retry_error_detail = classified[1]

            _, should_continue = decide_and_act(
                classified[0] if classified else ErrorType.EXECUTION_ERROR,
                retry_error_detail,
                attempt + 1,
                max_retries=max_retries,
            )

            if not should_continue:
                break

        # 所有重试均失败，返回最后一次错误
        final_obs = retry_obs if "retry_obs" in dir() else None
        final_err = retry_error_detail if "retry_error_detail" in dir() else error_detail
        return final_obs, final_err

    @staticmethod
    def _is_nonfatal_http_error(tool_output: str) -> bool:
        if not isinstance(tool_output, str):
            return False
        lower = tool_output.lower()
        return "http error" in lower and "client error" in lower

    def classify(self, tool_name: str, tool_output: Any):
        # Structured payload classification first (e.g., python_repl JSON string).
        structured = self._extract_structured_error(tool_output)
        if structured is not None:
            return structured

        text = self._extract_text(tool_output)
        if not text:
            return None

        normalized = text.strip()
        lower = normalized.lower()

        if lower.startswith("err:") or lower.startswith("error:"):
            normalized = (
                normalized.split(":", 1)[1].strip() if ":" in normalized else normalized
            )
            lower = normalized.lower()

        for idx, rule in enumerate(self._RULES, start=1):
            if any(re.search(pattern, lower, re.IGNORECASE) for pattern in rule.patterns):
                return rule.error_type, {
                    "tool": tool_name,
                    "category": rule.category,
                    "message": normalized,
                    "hint": rule.hint,
                    "retryable": rule.retryable,
                    "matched_rule": idx,
                    "raw_excerpt": normalized[:500],
                }

        if lower.startswith("exit code:"):
            return ErrorType.EXECUTION_ERROR, {
                "tool": tool_name,
                "category": "exit_code",
                "message": normalized,
                "hint": "Inspect command output and exit code details.",
                "retryable": False,
                "raw_excerpt": normalized[:500],
            }

        return ErrorType.EXECUTION_ERROR, {
            "tool": tool_name,
            "category": "execution",
            "message": normalized,
            "hint": "Inspect tool output and adjust command/arguments.",
            "retryable": False,
            "raw_excerpt": normalized[:500],
        }

    @staticmethod
    def summarize(tool_name: str, tool_output: Any) -> str:
        if tool_name == "search_web":
            if isinstance(tool_output, str):
                return tool_output[:200] + "..." if len(tool_output) > 200 else tool_output
            if isinstance(tool_output, dict) and "results" in tool_output:
                results_count = len(tool_output.get("results", []))
                return f"[搜索完成，返回 {results_count} 条结果]"
            return "[搜索完成]"

        text = ToolResultProcessor._extract_text(tool_output).strip()
        if text:
            return text[:200] + "..." if len(text) > 200 else text

        try:
            rendered = str(tool_output)
        except Exception:
            rendered = ""
        if not rendered:
            return "[no output]"
        return rendered[:200] + "..." if len(rendered) > 200 else rendered

    def assess_execution_state(self, tool_output: Any) -> dict[str, str]:
        """Assess tool execution state from normalized transport output.

        Rules are transport-oriented instead of business-case keywords:
        - explicit error prefixes -> failed
        - explicit non-zero exit marker -> failed
        - known success transport prefixes -> succeeded
        - python_repl json payload with success=false -> failed
        - otherwise unknown (do not classify as error by default)
        """
        # Special-case structured json payloads returned by python_repl.
        if isinstance(tool_output, str):
            parsed = self._try_parse_json(tool_output)
            if isinstance(parsed, dict) and isinstance(parsed.get("success"), bool):
                if parsed["success"] is False:
                    return {
                        "state": "failed",
                        "reason": "python_repl_structured_failure",
                    }
                return {"state": "succeeded", "reason": "python_repl_structured_success"}

        if isinstance(tool_output, dict) and isinstance(tool_output.get("success"), bool):
            if tool_output["success"] is False:
                return {"state": "failed", "reason": "structured_failure"}
            return {"state": "succeeded", "reason": "structured_success"}

        text = self._extract_text(tool_output).strip()
        if not text:
            return {"state": "unknown", "reason": "empty_output"}

        lower = text.lower()
        if lower.startswith("err:") or lower.startswith("error:"):
            return {"state": "failed", "reason": "explicit_error_prefix"}

        if lower.startswith("exit code:"):
            return {"state": "failed", "reason": "explicit_exit_code"}

        if lower.startswith("stdout:") or lower.startswith("success:"):
            return {"state": "succeeded", "reason": "explicit_transport_success"}

        return {"state": "unknown", "reason": "transport_unknown"}

    @staticmethod
    def _extract_text(tool_output: Any) -> str:
        if tool_output is None:
            return ""
        if isinstance(tool_output, str):
            return tool_output
        if isinstance(tool_output, dict):
            for key in ("error", "message", "stderr", "output", "detail"):
                value = tool_output.get(key)
                if isinstance(value, str) and value.strip():
                    return value
            return str(tool_output)
        if isinstance(tool_output, (list, tuple)):
            return "\n".join(str(item) for item in tool_output)
        return str(tool_output)

    def _extract_structured_error(
        self, tool_output: Any
    ) -> tuple[ErrorType, dict[str, Any]] | None:
        payload = None

        if isinstance(tool_output, dict):
            payload = tool_output
        elif isinstance(tool_output, str):
            parsed = self._try_parse_json(tool_output)
            if isinstance(parsed, dict):
                payload = parsed

        if not isinstance(payload, dict):
            return None

        if payload.get("success") is not False:
            return None

        raw_error_type = str(payload.get("error_type") or "").strip().lower()
        mapped_error_type = {
            "dependency_error": ErrorType.DEPENDENCY_ERROR,
            "timeout": ErrorType.TIMEOUT,
            "input_invalid": ErrorType.INPUT_INVALID,
            "resource_missing": ErrorType.RESOURCE_MISSING,
            "environment_error": ErrorType.ENVIRONMENT_ERROR,
            "unavailable": ErrorType.UNAVAILABLE,
        }.get(raw_error_type, ErrorType.EXECUTION_ERROR)

        message = str(payload.get("error") or payload.get("result") or "").strip()
        detail = payload.get("error_detail") if isinstance(payload.get("error_detail"), dict) else {}

        category = str(detail.get("category") or raw_error_type or "execution")
        hint = str(detail.get("hint") or "Inspect tool error and retry with corrected arguments/dependencies.")
        retryable = bool(detail.get("retryable", False))

        return mapped_error_type, {
            "tool": "python_repl",
            "category": category,
            "message": message,
            "hint": hint,
            "retryable": retryable,
            "raw_excerpt": message[:500],
            "structured_error": payload,
        }

    @staticmethod
    def _try_parse_json(text: str) -> dict[str, Any] | None:
        candidate = (text or "").strip()
        if not candidate or not candidate.startswith("{"):
            return None
        try:
            parsed = json.loads(candidate)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None
