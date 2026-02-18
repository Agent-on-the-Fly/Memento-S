"""Skill execution loops and continuation heuristics."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from core.config import BUILTIN_BRIDGE_SKILLS, SKILL_LOOP_FEEDBACK_CHARS
from core.skill_engine.common import extract_final_text, has_executable_calls
from core.skill_engine.planning import ask_for_plan
from core.skill_engine.skill_executor import execute_skill_plan
from core.skill_engine.skill_resolver import openskills_read
from core.utils.logging_utils import log_event
from core.utils.path_utils import _truncate, _truncate_middle

_PLAN_RETRY_HINT = (
    "Your previous response had no executable `tool_calls` and no `final` answer. "
    'Return either {"final":"..."} or a valid OpenAI-style `tool_calls` array.'
)
_AUTO_CONTINUE_FEEDBACK = (
    "Previous tool output appears to be an intermediate step, not final completion. "
    "Continue following SKILL.md and execute the next concrete step "
    "(e.g. download/fetch/unpack/read/summarize), and avoid repeating only existence checks."
)


@dataclass(frozen=True)
class _LoopOptions:
    """Configuration for the shared multi-round execution loop."""

    max_rounds: int
    include_max_rounds_value: bool
    emit_logs: bool


@dataclass(frozen=True)
class _LoopResult:
    """Final loop result with optional plan snapshot."""

    output: str
    last_plan: dict[str, Any]


def _initial_messages(user_text: str, skill_md: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": f"# Loaded SKILL.md\n\n{skill_md}"},
        {"role": "user", "content": user_text},
    ]



def _append_execution_error_feedback(messages: list[dict[str, str]], plan: Any, error_msg: str) -> None:
    plan_preview = json.dumps(plan, indent=2, ensure_ascii=False)[:1500]
    feedback = (
        f"ERROR executing tool_calls: {error_msg}\n\nYour plan was:\n{plan_preview}\n\n"
        "Please fix the plan structure and try again."
    )
    messages.append({"role": "user", "content": feedback})


def _append_previous_output_feedback(messages: list[dict[str, str]], output: str, *, prefix: str = "") -> None:
    messages.append(
        {
            "role": "user",
            "content": prefix + "Previous tool output:\n" + _truncate_middle(output, SKILL_LOOP_FEEDBACK_CHARS),
        }
    )


def _run_skill_loop(
    user_text: str,
    skill_name: str,
    *,
    skill_md: str,
    options: _LoopOptions,
) -> _LoopResult:
    messages = _initial_messages(user_text, skill_md)
    last_plan: dict[str, Any] = {}
    last_outputs: list[str] = []

    for round_no in range(1, options.max_rounds + 1):
        if options.emit_logs:
            log_event("run_one_skill_loop_round_start", skill_name=skill_name, round=round_no)

        plan = ask_for_plan(user_text, skill_md, skill_name, messages=messages)
        last_plan = plan if isinstance(plan, dict) else {}

        if options.emit_logs:
            log_event("run_one_skill_loop_round_plan", skill_name=skill_name, round=round_no, plan=plan)

        final_text = extract_final_text(plan)
        if final_text is not None:
            if options.emit_logs:
                mode = "handled" if isinstance(plan, dict) and plan.get("_handled") else "final"
                log_event("run_one_skill_loop_end", skill_name=skill_name, round=round_no, result=final_text, mode=mode)
            return _LoopResult(output=final_text, last_plan=last_plan)

        if has_executable_calls(plan):
            try:
                result_str = execute_skill_plan(skill_name, plan if isinstance(plan, dict) else {}).strip()
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                error_msg = f"{type(exc).__name__}: {exc}"
                if options.emit_logs:
                    plan_preview = json.dumps(plan, indent=2, ensure_ascii=False)[:1500]
                    log_event(
                        "run_one_skill_loop_exec_error",
                        skill_name=skill_name,
                        round=round_no,
                        error=error_msg,
                        plan_preview=plan_preview,
                    )
                _append_execution_error_feedback(messages, plan, error_msg)
                last_outputs.append(error_msg)
                continue

            if result_str.startswith("CONTINUE:"):
                out = result_str[9:].strip()
                if options.emit_logs:
                    log_event("run_one_skill_loop_continue", skill_name=skill_name, round=round_no, output=result_str)
                last_outputs.append(out)
                _append_previous_output_feedback(messages, out)
                continue

            if should_auto_continue_skill_result(skill_name, result_str):
                if options.emit_logs:
                    log_event(
                        "run_one_skill_loop_auto_continue",
                        skill_name=skill_name,
                        round=round_no,
                        output=result_str,
                        feedback=_AUTO_CONTINUE_FEEDBACK,
                    )
                last_outputs.append(result_str)
                _append_previous_output_feedback(messages, result_str, prefix=_AUTO_CONTINUE_FEEDBACK + "\n\n")
                continue

            if options.emit_logs:
                log_event(
                    "run_one_skill_loop_end",
                    skill_name=skill_name,
                    round=round_no,
                    result=result_str,
                    mode="tool_calls_result",
                )
            return _LoopResult(output=result_str, last_plan=last_plan)

        messages.append({"role": "user", "content": _PLAN_RETRY_HINT})

    hint = "\n".join(_truncate(x) for x in last_outputs[:8])
    if options.include_max_rounds_value:
        output = f"ERR: exceeded max_rounds={options.max_rounds}\n\nLast outputs:\n{hint}".strip()
    else:
        output = f"ERR: exceeded max_rounds\n\nLast outputs:\n{hint}".strip()

    if options.emit_logs:
        log_event(
            "run_one_skill_loop_end",
            skill_name=skill_name,
            round=options.max_rounds,
            result=output,
            mode="max_rounds",
        )
    return _LoopResult(output=output, last_plan=last_plan)


def run_one_skill(user_text: str, skill_name: str) -> str:
    """Single-shot skill execution (no multi-round loop)."""
    skill_md = openskills_read(skill_name)
    plan = ask_for_plan(user_text, skill_md, skill_name)

    final_text = extract_final_text(plan)
    if final_text is not None:
        return final_text

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
    options = _LoopOptions(max_rounds=max_rounds, include_max_rounds_value=True, emit_logs=True)
    return _run_skill_loop(user_text, skill_name, skill_md=skill_md, options=options).output


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
    options = _LoopOptions(max_rounds=max_rounds, include_max_rounds_value=False, emit_logs=False)
    loop_result = _run_skill_loop(user_text, skill_name, skill_md=skill_md, options=options)
    return loop_result.output, loop_result.last_plan


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
