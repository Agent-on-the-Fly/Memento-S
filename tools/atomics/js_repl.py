# SPDX-License-Identifier: Apache-2.0
"""JavaScript/TypeScript REPL tool powered by NodeSandbox."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Literal

from middleware.sandbox.node_sandbox import NodeSandbox
from middleware.utils.platform import SUBPROCESS_TEXT_KWARGS

_NAME = "js_repl"
_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {
            "type": "string",
            "description": "JavaScript or TypeScript code to execute.",
        },
        "deps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional npm packages to install before execution.",
        },
        "session_id": {
            "type": "string",
            "description": "Optional session identifier for sandbox paths.",
            "default": "",
        },
        "runtime": {
            "type": "string",
            "enum": ["node", "bun", "deno"],
            "description": "Preferred JS/TS runtime: node (with tsx for TS), bun, or deno.",
            "default": "node",
        },
        "work_dir": {
            "type": "string",
            "description": "Run directory for the Node sandbox. Auto-filled from the execution workspace.",
        },
        "source_dir": {
            "type": "string",
            "description": "Skill source root for NODE_PATH / vendor resolution. Auto-filled when running inside a skill.",
        },
    },
    "required": ["code"],
}


def extract_js_code(llm_output: str) -> tuple[str, str]:
    """Extract JS/TS code from Markdown code blocks."""
    pattern = r"```((?:javascript|js|typescript|ts|tsx|jsx)?)\s*(.*?)```"
    match = re.search(pattern, llm_output, re.DOTALL | re.IGNORECASE)
    if match:
        lang = match.group(1).lower() if match.group(1) else ""
        lang_hint = {
            "javascript": "js", "js": "js", "jsx": "jsx",
            "typescript": "ts", "ts": "ts", "tsx": "tsx",
        }.get(lang, "")
        return match.group(2).strip(), lang_hint

    unmarked_pattern = r"```\s*(.*?)\s*```"
    unmarked_match = re.search(unmarked_pattern, llm_output, re.DOTALL)
    if unmarked_match:
        return unmarked_match.group(1).strip(), ""
    return llm_output.strip(), ""


def validate_js_syntax(
    code: str, runtime: Literal["node", "bun", "deno"] = "node"
) -> tuple[bool, str]:
    """Validate JavaScript/TypeScript syntax via node --check."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
        f.write(code)
        temp_path = f.name
    try:
        result = subprocess.run(
            ["node", "--check", temp_path],
            capture_output=True,
            text=True,
            timeout=15,
            **SUBPROCESS_TEXT_KWARGS,
        )
        if result.returncode == 0:
            return True, ""
        stderr = result.stderr.strip()
        return False, stderr or f"Syntax error (node --check failed with code {result.returncode})"
    except FileNotFoundError:
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "Syntax check timed out"
    except Exception as e:
        return True, f"(syntax validation skipped: {e})"
    finally:
        Path(temp_path).unlink(missing_ok=True)


async def js_repl(
    code: str,
    deps: list[str] | None = None,
    session_id: str = "",
    runtime: Literal["node", "bun", "deno"] = "node",
    work_dir: str | None = None,
    source_dir: str | None = None,
) -> str:
    """
    Execute JavaScript/TypeScript code using the NodeSandbox.
    Supports .js, .ts, .tsx, and .jsx. Auto-detects the appropriate
    runtime (node/tsx, bun, or deno).

    Args:
        code: JS/TS code to execute.
        deps: Optional npm packages to install before execution.
        session_id: Optional session identifier for sandbox paths.
        runtime: Preferred runtime (node/bun/deno). Default is "node".
        work_dir: Optional execution working directory.
        source_dir: Optional skill source directory for NODE_PATH resolution.
    """
    try:
        clean_code, lang_hint = extract_js_code(code)
        is_valid, syntax_error = validate_js_syntax(clean_code, runtime=runtime)
        if not is_valid:
            payload: dict[str, Any] = {
                "success": False,
                "result": None,
                "error": f"SYNTAX ERROR in JavaScript/TypeScript: {syntax_error}\nPlease fix the syntax and try again.",
                "error_type": "syntax_error",
                "error_detail": {"hint": syntax_error},
                "artifacts": [],
                "skill_name": "js_repl",
            }
            return json.dumps(payload, ensure_ascii=False)

        sandbox = NodeSandbox()
        exec_name = f"js_repl_{session_id or 'default'}"
        env: dict[str, str] = {}
        env_value = os.environ.get("PRIMARY_ARTIFACT_PATH")
        if env_value:
            env["PRIMARY_ARTIFACT_PATH"] = env_value

        result = sandbox.run_code(
            code=clean_code,
            name=exec_name,
            deps=deps,
            session_id=session_id,
            source_dir=source_dir,
            work_dir=work_dir,
            extra_env=env or None,
            runtime=runtime,
        )
        payload: dict[str, Any] = {
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
        return f"ERR: js_repl failed: {e}"


js_repl._schema = _SCHEMA  # type: ignore[attr-defined]
