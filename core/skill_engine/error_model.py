"""Shared execution/error modeling helpers for skill engine modules."""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any

_ERR_LINE_RE = re.compile(r"(^|\n)\s*ERR:", re.IGNORECASE)


@dataclass(frozen=True)
class ErrorDetail:
    code: str
    message: str
    retryable: bool = False


def format_error(message: str, *, code: str = "error") -> str:
    text = str(message or "").strip()
    if not text:
        text = code or "error"
    if text.upper().startswith("ERR:"):
        return text
    return f"ERR: {text}"


def is_error_output(output: Any) -> bool:
    text = str(output or "")
    if not text.strip():
        return False
    return _ERR_LINE_RE.search(text) is not None


def infer_ok_from_output(output: Any, *, default: bool = True) -> bool:
    text = str(output or "")
    if not text.strip():
        return bool(default)
    return not is_error_output(text)


def build_execution_result_payload(
    *,
    ok: bool,
    output: Any,
    skill_name: str,
    code: str = "",
    normalized_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_text = str(output or "")
    resolved_ok = bool(ok)
    resolved_code = str(code or ("ok" if resolved_ok else "error"))
    errors: list[dict[str, Any]] = []
    if not resolved_ok:
        errors.append(
            asdict(
                ErrorDetail(
                    code=resolved_code,
                    message=out_text.strip() or format_error(resolved_code, code=resolved_code),
                    retryable=False,
                )
            )
        )

    return {
        "ok": resolved_ok,
        "code": resolved_code,
        "skill_name": str(skill_name or "").strip(),
        "output": out_text,
        "errors": errors,
        "normalized_plan": normalized_plan if isinstance(normalized_plan, dict) else {},
    }


__all__ = [
    "ErrorDetail",
    "format_error",
    "is_error_output",
    "infer_ok_from_output",
    "build_execution_result_payload",
]

