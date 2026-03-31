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
from core.shared.dependency_aliases import normalize_dependency_name


def extract_python_code(llm_output: str) -> str:
    """Extract Python code from Markdown code blocks.

    Removes markdown wrapper (```python ... ```) if present.
    """
    # Match ```python ... ``` or ``` ... ```
    pattern = r"```(?:python)?\s*(.*?)\s*```"
    match = re.search(pattern, llm_output, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return llm_output.strip()


def validate_python_syntax(code: str) -> tuple[bool, str]:
    """Validate Python code syntax using AST.

    Returns:
        (is_valid, error_message_or_hint)
    """
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        error_msg = str(e)

        # Check for Chinese quotes in the error message
        if "invalid character" in error_msg:
            # Look for Chinese quotes in the problematic line
            lineno = e.lineno or 1
            lines = code.split("\n")
            if lineno <= len(lines):
                problematic_line = lines[lineno - 1]
                # Chinese quotes to check
                chinese_quotes = [
                    "\u201c",
                    "\u201d",  # Left/right double quotation marks
                    "\u2018",
                    "\u2019",  # Left/right single quotation marks
                    "\uff02",
                    "\uff07",  # Fullwidth double/single quotation marks
                ]
                if any(q in problematic_line for q in chinese_quotes):
                    return False, (
                        f"SYNTAX ERROR at line {lineno}: {error_msg}\n"
                        "System Hint: You used Chinese typographic quotes (e.g., \" \" or ' ') "
                        "as string delimiters. Please rewrite the code using strictly "
                        "standard ASCII quotes (\" or ') for Python syntax."
                    )

        # Check for Windows path escape issues
        if (
            "unicode error" in error_msg.lower()
            or "(unicode error)" in error_msg.lower()
        ):
            return False, (
                f"SYNTAX ERROR at line {e.lineno}: {error_msg}\n"
                "System Hint: Watch out for unescaped backslashes in strings or paths. "
                'Use raw strings (e.g., r"C:\\\\path") or forward slashes.'
            )

        return (
            False,
            f"SYNTAX ERROR at line {e.lineno}: {error_msg}. Please fix the syntax and try again.",
        )


async def python_repl_tool(
    code: str,
    deps: list[str] | None = None,
    session_id: str = "",
    work_dir: str | None = None,
    primary_artifact_path: str | None = None,
) -> str:
    """Execute Python code using the UV sandbox.

    Args:
        code: Python code to execute.
        deps: Optional dependencies to install (in addition to detected imports).
        session_id: Optional session identifier for sandbox paths.
    """
    try:
        # Step 1: Extract clean code from Markdown
        clean_code = extract_python_code(code)

        # Step 2: Validate syntax before execution (Fail Fast)
        is_valid, error_hint = validate_python_syntax(clean_code)
        if not is_valid:
            # Return error with smart hint for LLM to fix
            payload: dict[str, Any] = {
                "success": False,
                "result": None,
                "error": error_hint,
                "error_type": "syntax_error",
                "error_detail": {"hint": error_hint},
                "artifacts": [],
                "skill_name": "python_repl",
            }
            return json.dumps(payload, ensure_ascii=False)

        # Step 3: Execute in sandbox
        sandbox = UvLocalSandbox()
        resolved_deps = _collect_dependencies(clean_code, deps)
        env: dict[str, str] = {}
        if primary_artifact_path:
            env["PRIMARY_ARTIFACT_PATH"] = primary_artifact_path
        else:
            env_value = os.environ.get("PRIMARY_ARTIFACT_PATH")
            if env_value:
                env["PRIMARY_ARTIFACT_PATH"] = env_value

        result = sandbox.run_code(
            clean_code,
            name="python_repl",
            deps=resolved_deps,
            session_id=session_id,
            work_dir=work_dir,
            extra_env=env or None,
        )
        payload = {
            "success": result.success,
            "result": result.result,
            "error": result.error,
            "error_type": result.error_type.value if result.error_type else None,
            "error_detail": result.error_detail,
            "artifacts": result.artifacts or [],
            "skill_name": result.skill_name,
        }
        return json.dumps(payload, ensure_ascii=False)
    except Exception as e:
        return f"ERR: python_repl failed: {e}"


def _collect_dependencies(code: str, deps: list[str] | None) -> list[str] | None:
    base_deps = set(
        normalized
        for dep in (deps or [])
        if dep and (normalized := normalize_dependency_name(dep))
    )
    module_deps = set(
        normalized
        for mod in _extract_import_modules(code)
        if mod and (normalized := normalize_dependency_name(mod))
    )
    merged = sorted(base_deps.union(module_deps))
    return merged or None


def _extract_import_modules(code: str) -> set[str]:
    tree = parse_code(code)
    if tree is None:
        return set()

    stdlib = getattr(sys, "stdlib_module_names", set())
    modules: set[str] = set()

    for node in tree.body:
        if node.__class__.__name__ == "Import":
            for alias in node.names:
                name = alias.name.split(".", 1)[0]
                if name and name not in stdlib:
                    modules.add(name)
        elif node.__class__.__name__ == "ImportFrom":
            if node.level and node.level > 0:
                continue
            module = (node.module or "").split(".", 1)[0]
            if module and module not in stdlib:
                modules.add(module)

    return modules
