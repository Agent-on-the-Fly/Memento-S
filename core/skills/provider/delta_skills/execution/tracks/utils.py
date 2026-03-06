
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from core.config.logging import get_logger

logger = get_logger(__name__)

_RE_OUTER_FENCE = re.compile(
    r"^\s*```[\w-]*\s*\n(.*?)\n\s*```\s*$",
    re.DOTALL,
)


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    m = _RE_OUTER_FENCE.match(stripped)
    return m.group(1).strip() if m else stripped


def parse_diagnosis(response: str) -> tuple[str, str]:
    stripped = response.strip()
    first_line, _, rest = stripped.partition("\n")
    tag = first_line.strip().upper()
    if tag in ("SKILL_FIX", "EXEC_FIX"):
        return tag, strip_code_fences(rest)
    return "UNKNOWN", strip_code_fences(stripped)


def build_available_modules_section(skill: Any) -> str:
    modules: list[str] = []
    if skill.source_dir:
        scripts_dir = Path(skill.source_dir) / "scripts"
        if scripts_dir.exists():
            for py_file in sorted(scripts_dir.rglob("*.py")):
                if py_file.name == "__init__.py":
                    continue
                rel = py_file.relative_to(scripts_dir)
                mod_name = str(rel).replace(".py", "").replace("/", ".").replace("\\", ".")
                modules.append(f"- {mod_name} ({rel})")
    elif skill.files:
        for filename in sorted(skill.files.keys()):
            if filename == "__init__.py" or not filename.endswith(".py"):
                continue
            mod_name = filename.replace(".py", "").replace("/", ".").replace("\\", ".")
            modules.append(f"- {mod_name} ({filename})")
    if not modules:
        return ""
    return (
        "\n\n## Available Modules in Sandbox\n"
        "The following modules from this skill are available in the execution environment. "
        "You can import them directly:\n"
        + "\n".join(modules)
    )


def build_skill_content(skill: Any) -> str:
    if not skill.code:
        return "(empty skill)"
    parts: list[str] = []
    if skill.code.lstrip().startswith("---"):
        parts.append("=== SKILL.md (Full Document) ===")
        parts.append(skill.code)
    else:
        parts.append("=== Python Function (INCLUDE this code and call it) ===")
        parts.append(f"```python\n{skill.code}\n```")
        parts.append(
            f"=== How to Use ===\n"
            f"1. Include the function definition above in your script.\n"
            f"2. Call `{skill.name}(...)` with the appropriate parameters.\n"
            f"3. Print the return value using `print()`."
        )
    if skill.dependencies:
        parts.append(f"=== Required Dependencies (pip install) ===\n{', '.join(skill.dependencies)}")
    return "\n\n".join(parts)


def parse_knowledge_response(raw: str) -> dict | None:
    text = raw.strip()

    not_relevant_match = re.match(
        r"^\[NOT_RELEVANT\]\s*(.*)", text, re.IGNORECASE | re.DOTALL,
    )
    if not_relevant_match:
        reason = not_relevant_match.group(1).strip() or "skill not relevant"
        return {"relevant": False, "reason": reason}

    md_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    candidate = md_match.group(1).strip() if md_match else text
    json_match = re.search(r'\{[^{}]*"relevant"\s*:', candidate)
    if json_match:
        start = json_match.start()
        brace_count = 0
        end = start
        for i in range(start, len(candidate)):
            if candidate[i] == '{':
                brace_count += 1
            elif candidate[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break
        try:
            parsed = json.loads(candidate[start:end])
            if isinstance(parsed, dict) and "relevant" in parsed:
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def detect_skill_mismatch(llm_output: str, skill_name: str, query: str) -> bool:
    text = llm_output[:500].lower()
    patterns = [
        r"(?:this|the)\s+(?:skill|tool)\s+(?:was|is)\s+(?:triggered|activated|selected).*?(?:but|however)",
        r"(?:doesn'?t|does not|don'?t|do not)\s+(?:involve|relate|match|apply|pertain)",
        r"(?:not|isn'?t|is not)\s+(?:relevant|related|applicable|appropriate)\s+(?:to|for)",
        r"(?:wrong|incorrect|mismatched)\s+skill",
        r"(?:||skill).*?(?:||||)",
        r"(?:|).*?(?:||).*?(?:||)",
    ]
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def refresh_skill_metadata(skill: Any, new_code: str) -> None:
    from ..analyzer import extract_dependencies
    from ...skills.creator import extract_description, extract_parameters

    try:
        skill.parameters = extract_parameters(new_code, skill.name)
        skill.dependencies = extract_dependencies(new_code)
        desc = extract_description(new_code, skill.name)
        if desc:
            skill.description = desc
        skill.invalidate_cache()
    except (ImportError, AttributeError, SyntaxError, ValueError) as e:
        logger.warning("Failed to refresh skill metadata after SKILL_FIX: %s", e)
