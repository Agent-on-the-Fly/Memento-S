from __future__ import annotations

import json
from typing import Any, Optional

from core.utils.path_utils import _stringify_result


class SkillExecutionError(RuntimeError):
    pass


def _normalize_plan(ops_or_plan: Any) -> dict:
    """
    Accept either:
      - list[dict] operations -> {"ops": [...]}
      - a full plan dict -> plan
      - a single op dict -> {"ops": [op]}
    """
    if isinstance(ops_or_plan, list):
        return {"ops": ops_or_plan}
    if isinstance(ops_or_plan, dict):
        if "ops" in ops_or_plan:
            return dict(ops_or_plan)
        if "type" in ops_or_plan:
            return {"ops": [ops_or_plan]}
        # Some skills accept shorthand at the top-level (e.g. query/url).
        if any(k in ops_or_plan for k in ("query", "url")):
            return dict(ops_or_plan)
        return {"ops": [ops_or_plan]}
    raise SkillExecutionError(f"Invalid plan/ops type: {type(ops_or_plan).__name__}")


def _try_get_logger():
    # TUI logger integration removed in CLI-only mode.
    return None


def call_skill(
    skill_name: str,
    ops_or_plan: Any,
    *,
    caller: Optional[str] = None,
) -> str:
    """
    Call another skill through the agent bridge runtime.
    """
    if not isinstance(skill_name, str) or not skill_name.strip():
        raise SkillExecutionError("call_skill: skill_name must be a non-empty string")
    name = skill_name.strip()
    plan = _normalize_plan(ops_or_plan)

    log = _try_get_logger()
    if log:
        log.event("call_skill_start", caller=caller, skill=name, plan=plan)

    try:
        from agent import execute_skill_plan, normalize_plan_shape

        result = execute_skill_plan(name, normalize_plan_shape(plan))
    except Exception as exc:
        if log:
            log.exception("call_skill_error", caller=caller, skill=name, error=str(exc), plan=plan)
        raise SkillExecutionError(f"call_skill: skill '{name}' failed: {exc}") from exc

    out = _stringify_result(result).strip()
    if log:
        log.event("call_skill_done", caller=caller, skill=name, output=out, output_len=len(out))
    return out
