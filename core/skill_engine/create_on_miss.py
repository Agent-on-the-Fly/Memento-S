"""Create-on-miss decision and orchestration helpers."""

from __future__ import annotations

import re

from core.llm import openrouter_messages
from core.utils.json_utils import parse_json_output
from core.utils.logging_utils import log_event
from core.skill_engine.execution import run_skill_once_with_plan
from core.skill_engine.skill_resolver import has_local_skill_dir, install_or_update_skill

def _should_create_skill_on_miss_fallback(user_text: str) -> tuple[bool, str]:
    text = (user_text or "").strip()
    if not text:
        return False, "empty_input"
    if re.search(r"(\bskill\b|\u6280\u80fd|\u5199\u4e2a\u6280\u80fd|\u521b\u5efa\u6280\u80fd|\u505a\u4e2a\u6280\u80fd|new\s+skill|create\s+skill)", text, re.IGNORECASE):
        return True, "explicit_skill_request"
    return False, "fallback_default"


def _normalize_skill_names(available_skill_names: list[str] | None) -> list[str]:
    """Normalize skill names by stripping, filtering empties, and de-duplicating."""
    names: list[str] = []
    for raw_name in available_skill_names or []:
        if not isinstance(raw_name, str):
            continue
        name = raw_name.strip()
        if not name or name in names:
            continue
        names.append(name)
    return names


def should_create_skill_on_miss(
    user_text: str,
    *,
    router_reason: str | None = None,
    available_skill_names: list[str] | None = None,
) -> tuple[bool, str]:
    """
    Decide whether to create a new skill on router miss.  Mirrors TUI behaviour.
    """
    text = (user_text or "").strip()
    if not text:
        return False, "empty_input"

    skill_names = _normalize_skill_names(available_skill_names)
    system_prompt = """You are a skill routing assistant. Decide if a user request should trigger creating a NEW reusable skill.

Answer with a JSON object: {"create": true/false, "reason": "brief explanation"}

Guidelines for when to CREATE a skill (create=true):
- The task involves a repeatable workflow (automation, data processing, file operations)
- The task requires external tools, APIs, or specialized operations
- The task could benefit future similar requests
- User explicitly asks to create a skill/tool

Guidelines for when NOT to create (create=false):
- Simple greetings, small talk, or acknowledgments
- General knowledge questions that don't need tools
- One-off factual questions or explanations
- The existing skills should already handle it (check available skills list)
- Requests better served by direct LLM chat

Be conservative: only create skills for genuinely reusable operational workflows."""

    user_prompt = f"""User request: {text}

Router reason for no match: {router_reason or 'none'}

Available skills: {', '.join(skill_names) if skill_names else 'none'}

Should we create a new skill for this request? Reply with JSON only."""

    try:
        response = openrouter_messages(system_prompt, [{"role": "user", "content": user_prompt}])
        parsed = parse_json_output(response)
        if isinstance(parsed, dict):
            should_create = bool(parsed.get("create", False))
            reason = str(parsed.get("reason") or "llm_decision").strip() or "llm_decision"
            return should_create, f"llm:{reason}"
    except Exception as exc:
        log_event("create_on_miss_decision_error", error=str(exc), router_reason=router_reason, user_text=text)

    return _should_create_skill_on_miss_fallback(text)


def create_skill_on_miss(
    user_text: str,
    *,
    router_reason: str | None = None,
    available_skill_names: list[str] | None = None,
) -> tuple[bool, str | None, str]:
    """
    Try to create a skill via skill-creator for a routing miss.
    Returns ``(created, skill_name, report)``.
    """
    names = _normalize_skill_names(available_skill_names)
    if "skill-creator" not in names and not has_local_skill_dir("skill-creator"):
        return False, None, "skill-creator unavailable"

    should_create, why = should_create_skill_on_miss(
        user_text,
        router_reason=router_reason,
        available_skill_names=names,
    )
    log_event(
        "create_on_miss_decision",
        should_create=should_create,
        reason=why,
        router_reason=router_reason,
        user_text=user_text,
    )
    if not should_create:
        return False, None, why

    prompt = (
        "Create a new skill to solve tasks like the user request below. "
        "Keep it concise but operational. Prefer generalizable workflows over one-off hacks.\n\n"
        f"User request:\n{user_text}\n"
    )
    try:
        result, plan = run_skill_once_with_plan(prompt, "skill-creator", max_rounds=20)
    except Exception as exc:
        return False, None, f"skill-creator failed: {exc}"
    skill_name = plan.get("skill_name") if isinstance(plan, dict) else None
    if not isinstance(skill_name, str) or not skill_name.strip():
        return False, None, "skill-creator did not return a skill_name"
    skill_name = skill_name.strip()

    ok, install_msg = install_or_update_skill(skill_name)
    if not ok:
        if has_local_skill_dir(skill_name):
            return True, skill_name, f"{result}\n[warn] install/sync failed: {install_msg}"
        return False, None, f"install/sync failed: {install_msg}"

    return True, skill_name, result or f"created skill {skill_name}"
