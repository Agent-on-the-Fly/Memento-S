"""Skill execution loops and continuation heuristics."""

from __future__ import annotations

import json
import re

from core.config import BUILTIN_BRIDGE_SKILLS, SKILL_LOOP_FEEDBACK_CHARS
from core.utils.logging_utils import log_event
from core.utils.path_utils import _truncate, _truncate_middle
from core.skill_engine.planning import ask_for_plan
from core.skill_engine.skill_executor import execute_skill_plan
from core.skill_engine.skill_resolver import openskills_read


_MISSING_PLAN_FEEDBACK = (
    "Your previous response had no executable `tool_calls` and no `final` answer. "
    'Return either {"final":"..."} or a valid OpenAI-style `tool_calls` array.'
)
_AUTO_CONTINUE_FEEDBACK = (
    "Previous tool output appears to be an intermediate step, not final completion. "
    "Continue following SKILL.md and execute the next concrete step "
    "(e.g. download/fetch/unpack/read/summarize), and avoid repeating only existence checks."
)


def _extract_loop_result(plan: dict | Any) -> tuple[str | None, str | None]:
    """Return (mode, result) when plan is terminal; otherwise (None, None)."""
    if not isinstance(plan, dict):
        return None, None

    if plan.get("_handled"):
        return "handled", str(plan.get("result", "")).strip()

    final = plan.get("final")
    if not isinstance(final, str) or not final.strip():
        final = plan.get("result")
    if isinstance(final, str) and final.strip():
        return "final", final.strip()
    return None, None


def _plan_has_calls(plan: dict | Any) -> bool:
    if not isinstance(plan, dict):
        return False
    tool_calls = plan.get("tool_calls")
    ops = plan.get("ops")
    return (isinstance(tool_calls, list) and bool(tool_calls)) or (isinstance(ops, list) and bool(ops))


def _build_exec_error_feedback(plan: dict, exc: Exception) -> tuple[str, str, str]:
    plan_preview = json.dumps(plan, indent=2, ensure_ascii=False)[:1500]
    error_msg = f"{type(exc).__name__}: {exc}"
    feedback = (
        f"ERROR executing tool_calls: {error_msg}\n\nYour plan was:\n{plan_preview}\n\n"
        "Please fix the plan structure and try again."
    )
    return error_msg, feedback, plan_preview


def _build_max_rounds_error(max_rounds: int, last_outputs: list[str], *, include_limit: bool) -> str:
    hint = "\n".join(_truncate(x) for x in last_outputs[:8])
    if include_limit:
        return f"ERR: exceeded max_rounds={max_rounds}\n\nLast outputs:\n{hint}".strip()
    return f"ERR: exceeded max_rounds\n\nLast outputs:\n{hint}".strip()


def _run_skill_loop_common(
    user_text: str,
    skill_name: str,
    skill_md: str,
    *,
    max_rounds: int,
    emit_logs: bool,
    include_round_limit_in_error: bool,
) -> tuple[str, dict]:
    """Shared multi-round loop used by public execution helpers."""
    messages: list[dict] = [
        {"role": "user", "content": f"# Loaded SKILL.md\n\n{skill_md}"},
        {"role": "user", "content": user_text},
    ]
    last_plan: dict = {}
    last_outputs: list[str] = []

    for round_no in range(1, max_rounds + 1):
        if emit_logs:
            log_event("run_one_skill_loop_round_start", skill_name=skill_name, round=round_no)

        plan = ask_for_plan(user_text, skill_md, skill_name, messages=messages)
        if isinstance(plan, dict):
            last_plan = plan

        if emit_logs:
            log_event("run_one_skill_loop_round_plan", skill_name=skill_name, round=round_no, plan=plan)

        terminal_mode, terminal_result = _extract_loop_result(plan)
        if terminal_mode and terminal_result is not None:
            if emit_logs:
                log_event(
                    "run_one_skill_loop_end",
                    skill_name=skill_name,
                    round=round_no,
                    result=terminal_result,
                    mode=terminal_mode,
                )
            return terminal_result, last_plan

        if _plan_has_calls(last_plan):
            try:
                result_str = execute_skill_plan(skill_name, last_plan).strip()
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                error_msg, feedback, plan_preview = _build_exec_error_feedback(last_plan, exc)
                if emit_logs:
                    log_event(
                        "run_one_skill_loop_exec_error",
                        skill_name=skill_name,
                        round=round_no,
                        error=error_msg,
                        plan_preview=plan_preview,
                    )
                messages.append({"role": "user", "content": feedback})
                last_outputs.append(error_msg)
                continue

            if result_str.startswith("CONTINUE:"):
                if emit_logs:
                    log_event(
                        "run_one_skill_loop_continue",
                        skill_name=skill_name,
                        round=round_no,
                        output=result_str,
                    )
                continued_output = result_str[9:].strip()
                last_outputs.append(continued_output)
                messages.append(
                    {
                        "role": "user",
                        "content": "Previous tool output:\n"
                        + _truncate_middle(continued_output, SKILL_LOOP_FEEDBACK_CHARS),
                    }
                )
                continue

            if should_auto_continue_skill_result(skill_name, result_str):
                if emit_logs:
                    log_event(
                        "run_one_skill_loop_auto_continue",
                        skill_name=skill_name,
                        round=round_no,
                        output=result_str,
                        feedback=_AUTO_CONTINUE_FEEDBACK,
                    )
                last_outputs.append(result_str)
                messages.append(
                    {
                        "role": "user",
                        "content": _AUTO_CONTINUE_FEEDBACK
                        + "\n\nPrevious tool output:\n"
                        + _truncate_middle(result_str, SKILL_LOOP_FEEDBACK_CHARS),
                    }
                )
                continue

            if emit_logs:
                log_event(
                    "run_one_skill_loop_end",
                    skill_name=skill_name,
                    round=round_no,
                    result=result_str,
                    mode="tool_calls_result",
                )
            return result_str, last_plan

        messages.append({"role": "user", "content": _MISSING_PLAN_FEEDBACK})

    result = _build_max_rounds_error(
        max_rounds,
        last_outputs,
        include_limit=include_round_limit_in_error,
    )
    if emit_logs:
        log_event("run_one_skill_loop_end", skill_name=skill_name, round=max_rounds, result=result, mode="max_rounds")
    return result, last_plan


def run_one_skill(user_text: str, skill_name: str) -> str:
    """Single-shot skill execution (no multi-round loop)."""
    skill_md = openskills_read(skill_name)
    plan = ask_for_plan(user_text, skill_md, skill_name)
    if isinstance(plan, dict) and plan.get("_handled"):
        return str(plan.get("result", "")).strip()

    final = plan.get("final") if isinstance(plan, dict) else None
    if not isinstance(final, str) or not final.strip():
        final = plan.get("result") if isinstance(plan, dict) else None
    if isinstance(final, str) and final.strip():
        return final.strip()

    try:
        return execute_skill_plan(skill_name, plan if isinstance(plan, dict) else {})
    except Exception as exc:
        return f"ERR: {type(exc).__name__}: {exc}"


def run_one_skill_loop(user_text: str, skill_name: str, max_rounds: int = 50) -> str:
    """
    Run one skill with multi-round planning until completion or *max_rounds*.
    Skills execute via the SKILL.md bridge path (``tool_calls``/``final``) only.
    """
    skill_md = openskills_read(skill_name)
    log_event(
        "run_one_skill_loop_start",
        skill_name=skill_name,
        user_text=user_text,
        max_rounds=max_rounds,
    )

    result, _last_plan = _run_skill_loop_common(
        user_text,
        skill_name,
        skill_md,
        max_rounds=max_rounds,
        emit_logs=True,
        include_round_limit_in_error=True,
    )
    return result


def run_skill_once_with_plan(
    user_text: str,
    skill_name: str,
    *,
    max_rounds: int = 20,
) -> tuple[str, dict]:
    """
    Run one skill and return ``(result, last_plan)``.
    Mirrors TUI workflow behaviour.
    """
    skill_md = openskills_read(skill_name)
    return _run_skill_loop_common(
        user_text,
        skill_name,
        skill_md,
        max_rounds=max_rounds,
        emit_logs=False,
        include_round_limit_in_error=False,
    )


# ---------------------------------------------------------------------------
# Continuation logic
# ---------------------------------------------------------------------------

def should_auto_continue_skill_result(skill_name: str, result_str: str) -> bool:
    """
    Heuristic continuation trigger for non-bridge skills.
    If a skill only returns intermediate bridge tool-call output, ask it to continue
    instead of stopping.
    """
    name = str(skill_name or "").strip()
    if not name or name in BUILTIN_BRIDGE_SKILLS:
        return False
    text = str(result_str or "").strip()
    if not text:
        return False

    if text == "NOT_FOUND":
        return True

    # Common bridge wrapper style: [op#1:shell]\nNOT_FOUND
    if re.fullmatch(r"\[op#\d+:[^\]]+\]\s*NOT_FOUND", text, re.DOTALL):
        return True

    # Non-bridge skills should keep iterating when they still return bridge tool-call blocks.
    if re.search(r"\[op#\d+:[^\]]+\]", text):
        return True

    return False
