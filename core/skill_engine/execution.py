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
    Skills execute via the SKILL.md bridge path (``ops``/``final``) only.
    """
    skill_md = openskills_read(skill_name)
    log_event(
        "run_one_skill_loop_start",
        skill_name=skill_name,
        user_text=user_text,
        max_rounds=max_rounds,
    )

    # Keep a running history so the model can see prior outputs.
    messages: list[dict] = [
        {"role": "user", "content": f"# Loaded SKILL.md\n\n{skill_md}"},
        {"role": "user", "content": user_text},
    ]

    last_outputs: list[str] = []
    for round_no in range(1, max_rounds + 1):
        log_event("run_one_skill_loop_round_start", skill_name=skill_name, round=round_no)
        plan = ask_for_plan(user_text, skill_md, skill_name, messages=messages)
        log_event("run_one_skill_loop_round_plan", skill_name=skill_name, round=round_no, plan=plan)

        # ---- handled (rejected / error) ----
        if isinstance(plan, dict) and plan.get("_handled"):
            result = str(plan.get("result", "")).strip()
            log_event("run_one_skill_loop_end", skill_name=skill_name, round=round_no, result=result, mode="handled")
            return result

        # ---- final answer ----
        if isinstance(plan, dict):
            final = plan.get("final")
            if not isinstance(final, str) or not final.strip():
                final = plan.get("result")
            if isinstance(final, str) and final.strip():
                result = final.strip()
                log_event("run_one_skill_loop_end", skill_name=skill_name, round=round_no, result=result, mode="final")
                return result

        # ---- execute ops via the bridge executor ----
        ops = plan.get("ops") if isinstance(plan, dict) else None
        if isinstance(ops, list) and ops:
            try:
                result_str = execute_skill_plan(skill_name, plan).strip()
            except (KeyError, TypeError, ValueError, AttributeError) as e:
                plan_preview = json.dumps(plan, indent=2, ensure_ascii=False)[:1500]
                error_msg = f"{type(e).__name__}: {e}"
                log_event(
                    "run_one_skill_loop_exec_error",
                    skill_name=skill_name,
                    round=round_no,
                    error=error_msg,
                    plan_preview=plan_preview,
                )
                feedback = (
                    f"ERROR executing ops: {error_msg}\n\nYour plan was:\n{plan_preview}\n\n"
                    "Please fix the plan structure and try again."
                )
                messages.append({"role": "user", "content": feedback})
                last_outputs.append(error_msg)
                continue

            # -- CONTINUE: protocol --
            if result_str.startswith("CONTINUE:"):
                log_event(
                    "run_one_skill_loop_continue",
                    skill_name=skill_name,
                    round=round_no,
                    output=result_str,
                )
                last_outputs.append(result_str[9:].strip())
                messages.append(
                    {
                        "role": "user",
                        "content": "Previous ops output:\n" + _truncate_middle(result_str[9:], SKILL_LOOP_FEEDBACK_CHARS),
                    }
                )
                continue

            # -- auto-continue heuristic --
            if should_auto_continue_skill_result(skill_name, result_str):
                feedback = (
                    "Previous ops output appears to be an intermediate step, not final completion. "
                    "Continue following SKILL.md and execute the next concrete step "
                    "(e.g. download/fetch/unpack/read/summarize), and avoid repeating only existence checks."
                )
                log_event(
                    "run_one_skill_loop_auto_continue",
                    skill_name=skill_name,
                    round=round_no,
                    output=result_str,
                    feedback=feedback,
                )
                last_outputs.append(result_str)
                messages.append(
                    {
                        "role": "user",
                        "content": feedback
                        + "\n\nPrevious ops output:\n"
                        + _truncate_middle(result_str, SKILL_LOOP_FEEDBACK_CHARS),
                    }
                )
                continue

            # -- done --
            log_event("run_one_skill_loop_end", skill_name=skill_name, round=round_no, result=result_str, mode="ops_result")
            return result_str

        # ---- no ops and no final – ask the model to try again ----
        messages.append(
            {
                "role": "user",
                "content": (
                    "Your previous response had no executable `ops` and no `final` answer. "
                    'Return either {"final":"..."} or a valid ops array.'
                ),
            }
        )
        continue

    hint = "\n".join(_truncate(x) for x in last_outputs[:8])
    result = f"ERR: exceeded max_rounds={max_rounds}\n\nLast outputs:\n{hint}".strip()
    log_event("run_one_skill_loop_end", skill_name=skill_name, round=max_rounds, result=result, mode="max_rounds")
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
    messages: list[dict] = [
        {"role": "user", "content": f"# Loaded SKILL.md\n\n{skill_md}"},
        {"role": "user", "content": user_text},
    ]

    last_plan: dict = {}
    last_outputs: list[str] = []
    for _round in range(1, max_rounds + 1):
        plan = ask_for_plan(user_text, skill_md, skill_name, messages=messages)
        last_plan = plan if isinstance(plan, dict) else {}

        if isinstance(plan, dict) and plan.get("_handled"):
            return str(plan.get("result", "")).strip(), last_plan

        if isinstance(plan, dict):
            final = plan.get("final")
            if not isinstance(final, str) or not final.strip():
                final = plan.get("result")
            if isinstance(final, str) and final.strip():
                return final.strip(), last_plan

        ops = plan.get("ops") if isinstance(plan, dict) else None
        if isinstance(ops, list) and ops:
            try:
                result_str = execute_skill_plan(skill_name, plan).strip()
            except (KeyError, TypeError, ValueError, AttributeError) as e:
                plan_preview = json.dumps(plan, indent=2, ensure_ascii=False)[:1500]
                error_msg = f"{type(e).__name__}: {e}"
                feedback = (
                    f"ERROR executing ops: {error_msg}\n\nYour plan was:\n{plan_preview}\n\n"
                    "Please fix the plan structure and try again."
                )
                messages.append({"role": "user", "content": feedback})
                last_outputs.append(error_msg)
                continue

            if result_str.startswith("CONTINUE:"):
                out = result_str[9:].strip()
                last_outputs.append(out)
                messages.append(
                    {
                        "role": "user",
                        "content": "Previous ops output:\n" + _truncate_middle(out, SKILL_LOOP_FEEDBACK_CHARS),
                    }
                )
                continue

            if should_auto_continue_skill_result(skill_name, result_str):
                feedback = (
                    "Previous ops output appears to be an intermediate step, not final completion. "
                    "Continue following SKILL.md and execute the next concrete step "
                    "(e.g. download/fetch/unpack/read/summarize), and avoid repeating only existence checks."
                )
                last_outputs.append(result_str)
                messages.append(
                    {
                        "role": "user",
                        "content": feedback
                        + "\n\nPrevious ops output:\n"
                        + _truncate_middle(result_str, SKILL_LOOP_FEEDBACK_CHARS),
                    }
                )
                continue

            return result_str, last_plan

        messages.append(
            {
                "role": "user",
                "content": (
                    "Your previous response had no executable `ops` and no `final` answer. "
                    'Return either {"final":"..."} or a valid ops array.'
                ),
            }
        )

    hint = "\n".join(_truncate(x) for x in last_outputs[:8])
    return f"ERR: exceeded max_rounds\n\nLast outputs:\n{hint}".strip(), last_plan


# ---------------------------------------------------------------------------
# Continuation logic
# ---------------------------------------------------------------------------

def should_auto_continue_skill_result(skill_name: str, result_str: str) -> bool:
    """
    Heuristic continuation trigger for non-bridge skills.
    If a skill only returns intermediate bridge op output, ask it to continue
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

    # Non-bridge skills should keep iterating when they still return bridge op blocks.
    if re.search(r"\[op#\d+:[^\]]+\]", text):
        return True

    return False
