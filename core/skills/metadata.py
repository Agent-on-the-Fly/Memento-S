
from __future__ import annotations

import re
from typing import Any, Protocol

from .base import Skill


class SkillLike(Protocol):

    name: str
    description: str
    parameters: dict[str, Any]

DEFAULT_MAX_DESCRIPTION_CHARS = 500

DESCRIPTION_SAFETY_CAP = 4000

_PARAM_TYPE_MAP = {
    "str": "string",
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "list": "array",
    "array": "array",
    "dict": "object",
    "object": "object",
}


def normalize_skill_name(name: str, *, prefix: str | None = None) -> str:
    s = name.strip()
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    s = s.lower()
    if not s:
        s = "skill"
    if prefix:
        p = prefix.strip().lower()
        p = re.sub(r"[^\w]+", "_", p).strip("_")
        if p:
            s = f"{p}_{s}" if not s.startswith(p + "_") else s
    return s or "skill"


def clean_description(
    raw: str,
    safety_cap: int = DESCRIPTION_SAFETY_CAP,
) -> str:
    raw_str = raw if isinstance(raw, str) else (str(raw) if raw is not None else "")
    lines = raw_str.strip().splitlines()
    text = "\n".join(re.sub(r"[^\S\n]+", " ", line).strip() for line in lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) <= safety_cap:
        return text
    truncated_lines: list[str] = []
    total = 0
    for line in text.splitlines():
        if total + len(line) + 1 > safety_cap - 4:  #  4 chars  "\n..."
            break
        truncated_lines.append(line)
        total += len(line) + 1
    return "\n".join(truncated_lines) + "\n..."


_RE_TEMPLATE_SEP = re.compile(
    r"(?:Action|Pre-condition|Outcome|Use when|Returns?)\s*[:\-]\s*",
    re.IGNORECASE,
)
_RE_TEMPLATE_HEAD = re.compile(
    r"^(?:Action|Pre-condition|Outcome|Use when|Returns?)",
    re.IGNORECASE,
)


def compress_description(
    raw: str,
    max_chars: int = DEFAULT_MAX_DESCRIPTION_CHARS,
    *,
    use_template: bool = True,
) -> str:
    raw_str = raw if isinstance(raw, str) else (str(raw) if raw is not None else "")
    text = raw_str.strip().replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    if len(text) <= max_chars:
        return text

    if use_template:
        action = pre = outcome = ""
        try:
            parts = _RE_TEMPLATE_SEP.split(text)
        except (TypeError, ValueError, re.error):
            parts = [text]
        for part in parts:
            part = (part.strip() if isinstance(part, str) else str(part)).strip()
            if not part:
                continue
            try:
                is_header = _RE_TEMPLATE_HEAD.match(part) is not None
            except (TypeError, ValueError, re.error):
                is_header = False
            if not action and not is_header:
                action = part
            elif not pre and "pre" not in part.lower()[:20]:
                pre = part[:160] + "..." if len(part) > 160 else part
            elif not outcome:
                outcome = part[:160] + "..." if len(part) > 160 else part
                break
        if action or pre or outcome:
            parts = []
            if action:
                parts.append(f"Action: {action[:200]}")
            if pre:
                parts.append(f"Use when: {pre}")
            if outcome:
                parts.append(f"Returns: {outcome}")
            candidate = ". ".join(parts)
            if len(candidate) <= max_chars:
                return candidate

    for sep in ".!?!":
        idx = text.rfind(sep, 0, max_chars + 1)
        if idx > 0:
            return text[: idx + 1].strip()
    truncated = text[: max_chars - 3].rsplit(maxsplit=1)[0] if len(text) > max_chars else text
    return (truncated or text[:max_chars]) + "..."


def normalize_parameters(params: dict[str, Any]) -> dict[str, Any]:
    if not params:
        return {"type": "object", "properties": {}}
    out = dict(params)
    if "properties" not in out:
        out["properties"] = {}
    props = out.get("properties") or {}
    normalized_props = {}
    for k, v in list(props.items()):
        if not isinstance(v, dict):
            normalized_props[k] = {"description": str(v) if v else ""}
            continue
        p = dict(v)
        t = (p.get("type") or "").lower().strip()
        if t in _PARAM_TYPE_MAP:
            p["type"] = _PARAM_TYPE_MAP[t]
        elif t and t not in ("object", "array", "string", "number", "boolean", "integer"):
            p["type"] = _PARAM_TYPE_MAP.get(t, "string")
        if "enum" in p and p["enum"]:
            pass  # 
        normalized_props[k] = p
    out["properties"] = normalized_props
    return out


def skill_to_tool_metadata(
    skill: Skill | SkillLike,
    *,
    name_prefix: str | None = None,
    max_description_chars: int = DEFAULT_MAX_DESCRIPTION_CHARS,
    normalize_name: bool = True,
    full_description: bool = True,
) -> dict[str, Any]:
    name = skill.name
    if normalize_name:
        name = normalize_skill_name(name, prefix=name_prefix)
    if full_description:
        description = clean_description(skill.description or "")
    else:
        description = compress_description(
            skill.description or "",
            max_chars=max_description_chars,
        )
    parameters = normalize_parameters(skill.parameters or {})
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }
