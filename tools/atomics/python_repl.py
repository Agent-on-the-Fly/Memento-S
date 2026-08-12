# SPDX-License-Identifier: Apache-2.0
"""Python REPL tool powered by UV sandbox."""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from typing import Any

from middleware.utils.parsing import parse_code
from middleware.sandbox.uv import UvLocalSandbox
from shared.tools.dependency_aliases import normalize_dependency_name

_NAME = "python_repl"
_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {
            "type": "string",
            "description": "Python code to execute.",
        },
        "skill_name": {
            "type": "string",
            "description": "Name of the skill owning this execution. Auto-filled from execution context.",
        },
        "deps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional Python dependencies to install.",
        },
        "skill_deps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Skill metadata dependencies. Auto-filled by SkillAgent via tool_props.",
        },
        "session_id": {
            "type": "string",
            "description": "Session identifier for sandbox paths. Auto-filled from execution context.",
        },
        "work_dir": {
            "type": "string",
            "description": "Run directory (@ROOT) for the UV sandbox. Auto-filled from the execution workspace.",
        },
    },
    "required": ["code"],
}


def extract_python_code(llm_output: str) -> str:
    """Extract Python code from Markdown code blocks."""
    pattern = r"```(?:python)?\s*(.*?)\s*```"
    match = re.search(pattern, llm_output, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return llm_output.strip()


def _get_hint_for_invalid_char(error_msg: str, code: str, lineno: int) -> str:
    """从 SyntaxError 中提取无效字符，给出基于 Unicode 范围的智能提示。

    适用于所有非 ASCII Unicode 字符，包括中文引号、Box-Drawing 字符、CJK 字符等。
    """
    # 从错误信息中提取字符和码点: "invalid character '┌' (U+250C)"
    match = re.search(r"'([^']+)'\s*\(U\+([0-9A-Fa-f]+)\)", error_msg)
    if not match:
        return (
            "This Unicode character is not valid in Python code. "
            "Use standard ASCII characters only."
        )

    char, codepoint_hex = match.groups()
    try:
        codepoint = int(codepoint_hex, 16)
    except ValueError:
        return (
            "This Unicode character is not valid in Python code. "
            "Use standard ASCII characters only."
        )

    # Box-Drawing 字符 (U+2500-U+257F): ┌─│└ 等
    if 0x2500 <= codepoint <= 0x257F:
        return (
            "You used Unicode box-drawing characters (e.g., ┌─│└) which are "
            "not valid in Python code. "
            "If you need to write a file containing diagrams or formatted text, "
            "use the 'file_create' tool instead of python_repl. "
            "If you must use python_repl, remove all box-drawing characters."
        )

    # CJK 标点/特殊符号: 中文引号「」等
    if codepoint in (0x201C, 0x201D, 0x2018, 0x2019, 0xFF02, 0xFF07):
        return (
            "You used Chinese typographic quotes as string delimiters. "
            "Please rewrite using strictly standard ASCII quotes (single ' or double \")."
        )

    # CJK Unified Ideographs (U+4E00-U+9FFF): 中日韩文字
    if 0x4E00 <= codepoint <= 0x9FFF:
        return (
            f"You used non-ASCII character '{char}' (U+{codepoint:04X}) in code. "
            "Python identifiers must be ASCII letters/digits/underscores. "
            "Put any non-ASCII text inside string literals, or remove it."
        )

    # 其他 Unicode 范围
    if codepoint >= 0x80:
        return (
            f"Unicode character U+{codepoint:04X} ('{char}') is not valid in Python code. "
            "Use standard ASCII characters only, or put non-ASCII text in string literals."
        )

    return (
        "This Unicode character is not valid in Python code. "
        "Use standard ASCII characters only."
    )


def validate_python_syntax(code: str) -> tuple[bool, str]:
    """Validate Python code syntax using AST."""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        error_msg = str(e)
        if "invalid character" in error_msg:
            lineno = e.lineno or 1
            hint = _get_hint_for_invalid_char(error_msg, code, lineno)
            return False, f"SYNTAX ERROR at line {lineno}: {error_msg}\nHint: {hint}"
        if "unicode error" in error_msg.lower() or "(unicode error)" in error_msg.lower():
            return False, (
                f"SYNTAX ERROR at line {e.lineno}: {error_msg}\n"
                "Hint: Watch out for unescaped backslashes in strings or paths. "
                'Use raw strings (e.g., r"C:\\\\path") or forward slashes.'
            )
        return False, f"SYNTAX ERROR at line {e.lineno}: {error_msg}. Please fix the syntax and try again."


async def python_repl(
    code: str,
    skill_name: str | None = None,
    deps: list[str] | None = None,
    skill_deps: list[str] | None = None,
    session_id: str = "",
    work_dir: str | None = None,
) -> str:
    """
    Execute Python code using the UV sandbox.

    Args:
        code: Python code to execute.
        skill_name: Name of the skill owning this execution. Auto-filled from context.
        deps: Optional dependencies to install.
        skill_deps: Skill metadata dependencies (auto-filled by SkillToolAdapter).
        session_id: Optional session identifier for sandbox paths.
        work_dir: Run directory for the UV sandbox.
    """
    sandbox_name = skill_name or (
        f"python_{session_id}" if session_id else "python_repl"
    )
    error_skill_name = skill_name or "python_repl"

    try:
        clean_code = extract_python_code(code)
        is_valid, error_hint = validate_python_syntax(clean_code)
        if not is_valid:
            payload: dict[str, Any] = {
                "success": False,
                "result": None,
                "error": error_hint,
                "error_type": "syntax_error",
                "error_detail": {"hint": error_hint},
                "artifacts": [],
                "skill_name": error_skill_name,
            }
            return json.dumps(payload, ensure_ascii=False)

        sandbox = UvLocalSandbox()
        resolved_deps = _collect_dependencies(clean_code, deps, skill_deps)
        env: dict[str, str] = {}
        env_value = os.environ.get("PRIMARY_ARTIFACT_PATH")
        if env_value:
            env["PRIMARY_ARTIFACT_PATH"] = env_value

        result = sandbox.run_code(
            clean_code,
            name=sandbox_name,
            deps=resolved_deps,
            session_id=session_id,
            work_dir=work_dir,
            extra_env=env or None,
        )
        payload: dict[str, Any] = {
            "success": result.success,
            "result": result.result,
            "error": result.error,
            "error_type": result.error_type.value if result.error_type else None,
            "error_detail": result.error_detail,
            "artifacts": result.artifacts or [],
            "skill_name": result.skill_name or sandbox_name,
        }
        return json.dumps(payload, ensure_ascii=False)
    except Exception as e:
        return f"ERR: python_repl failed: {e}"


def _collect_dependencies(
    code: str,
    deps: list[str] | None,
    skill_deps: list[str] | None,
) -> list[str] | None:
    """Merge code imports + explicit deps + skill metadata deps into unified install list."""
    base_deps = {
        normalized
        for dep in (deps or [])
        if dep and (normalized := normalize_dependency_name(dep))
    }
    module_deps = {
        normalized
        for mod in _extract_import_modules(code)
        if mod and (normalized := normalize_dependency_name(mod))
    }
    skill_meta_deps = {
        normalized
        for dep in (skill_deps or [])
        if dep and (normalized := normalize_dependency_name(dep))
    }
    merged = sorted(base_deps.union(module_deps).union(skill_meta_deps))
    return merged or None


def _extract_import_modules(code: str) -> set[str]:
    tree = parse_code(code)
    if tree is None:
        return set()
    stdlib = getattr(sys, "stdlib_module_names", set())
    modules: set[str] = set()
    for node in tree.body:
        cls_name = node.__class__.__name__
        if cls_name == "Import":
            for alias in node.names:
                name = alias.name.split(".", 1)[0]
                if name and name not in stdlib:
                    modules.add(name)
        elif cls_name == "ImportFrom":
            if node.level and node.level > 0:
                continue
            module = (node.module or "").split(".", 1)[0]
            if module and module not in stdlib:
                modules.add(module)
    return modules


python_repl._schema = _SCHEMA  # type: ignore[attr-defined]
