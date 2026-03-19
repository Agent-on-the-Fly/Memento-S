"""Memento-S Agent utility functions.

Contains only pure utility functions used by agent.py and phases/.
All dead code and hardcoded classification logic has been removed.
"""

from __future__ import annotations

import json
import re
from typing import Any

from middleware.llm.schema import ToolCall


# =============================================================================
# JSON extraction
# =============================================================================


def extract_json(text: str) -> dict:
    """Extract a JSON object from LLM output, tolerating markdown code blocks."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if m:
        return json.loads(m.group(1))
    if text.startswith("{"):
        return json.loads(text)
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"No JSON object found in: {text[:200]}")


# =============================================================================
# Skill name utilities
# =============================================================================


def normalize_skill_name(value: str) -> str:
    """Normalize a skill name to lowercase snake_case."""
    return (value or "").strip().lower().replace("-", "_").replace(" ", "_")


def mentions_skill_name(user_text: str, skill_name: str) -> bool:
    """Check if user text mentions a specific skill name (case-insensitive)."""
    if not user_text or not skill_name:
        return False
    text = user_text.lower()
    name = skill_name.lower()
    return (
        name in text
        or name.replace("-", "_") in text
        or name.replace("_", "-") in text
    )


def can_direct_execute_skill(user_content: str, args: dict[str, Any]) -> bool:
    """Allow direct execute only when user gives a concrete skill name + request context."""
    skill_name = str(args.get("skill_name", "")).strip()
    request = str(args.get("request", "")).strip()
    if not skill_name or not request:
        return False
    lowered = user_content.lower()
    if skill_name.lower() in lowered:
        return True
    if "skill_name" in lowered:
        return True
    return False


def extract_explicit_skill_name(
    user_content: str,
    discover_fn: Any,
) -> str | None:
    """Extract an explicitly mentioned skill name from user content.

    Args:
        user_content: The raw user message.
        discover_fn: A callable that returns a list of skill manifests
                     (each with a ``.name`` attribute), e.g. ``gateway.discover``.
    """
    text = normalize_skill_name(user_content)
    if not text:
        return None

    try:
        local_names = [m.name for m in discover_fn()]
    except Exception:
        return None

    for name in local_names:
        n = normalize_skill_name(name)
        if n and n in text:
            return name
    return None


# =============================================================================
# Tool call format conversion
# =============================================================================


def skill_call_to_openai_payload(skill_call: ToolCall) -> dict[str, Any]:
    """Convert a ToolCall into the OpenAI-style tool_calls item format."""
    return {
        "id": skill_call.id,
        "type": "function",
        "function": {
            "name": skill_call.name,
            "arguments": json.dumps(skill_call.arguments, ensure_ascii=False),
        },
    }
