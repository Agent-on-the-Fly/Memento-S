"""Shared plan-shape helpers for skill engine modules."""

from __future__ import annotations

from typing import Any


def extract_final_text(plan: Any) -> str | None:
    """Return final/handled text from a plan dict, else ``None``."""
    if not isinstance(plan, dict):
        return None

    if plan.get("_handled"):
        result = str(plan.get("result", "")).strip()
        return result if result else ""

    final = plan.get("final")
    if not isinstance(final, str) or not final.strip():
        final = plan.get("result")
    if isinstance(final, str) and final.strip():
        return final.strip()
    return None


def has_executable_calls(plan: Any) -> bool:
    """Return whether a plan contains executable tool calls/ops."""
    if not isinstance(plan, dict):
        return False
    tool_calls = plan.get("tool_calls")
    ops = plan.get("ops")
    return (isinstance(tool_calls, list) and bool(tool_calls)) or (isinstance(ops, list) and bool(ops))
