# SPDX-License-Identifier: Apache-2.0
"""Bash command execution tool."""

from __future__ import annotations

import fnmatch
import logging
import os
import re
import sys
from pathlib import Path

from shared.tools.path_boundary import get_boundary
from middleware.sandbox import execute_shell
from middleware.sandbox.node_env import build_node_path_from_dir

_NAME = "bash"
_logger = logging.getLogger("middleware.sandbox.atomics.bash")


_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": "The shell command to run.",
        },
        "stdin": {
            "type": "string",
            "description": "Optional standard input passed to the command.",
        },
        "work_dir": {
            "type": "string",
            "description": "Working directory for the subprocess (absolute path).",
        },
        "source_dir": {
            "type": "string",
            "description": "Skill source root for NODE_PATH / vendor resolution.",
        },
        "env": {
            "type": "object",
            "description": "Optional extra environment variables for this invocation.",
            "additionalProperties": {"type": "string"},
        },
        "skill_deps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Skill metadata dependencies. Auto-filled by SkillAgent via tool_props.",
        },
    },
    "required": ["command"],
}


# ---------------------------------------------------------------------------
# Cross-platform PATH tightening
# ---------------------------------------------------------------------------

_UNSAFE_PATHS_UNIX: frozenset[str] = frozenset({
    "/usr/local/bin",    # user-writable, may contain custom tools
    "/opt/local/bin",   # MacPorts
    "/snap/bin",        # Snap packages
    "/usr/games",
    "/usr/local/games",
})

_UNSAFE_PATH_PREFIXES_UNIX: tuple[str, ...] = (
    "/home/",           # user home directories
    "/Users/",          # macOS user home
    "/.local/bin",      # pip install --user
)

_SAFE_PATHS_UNIX: tuple[str, ...] = ("/usr/bin", "/bin", "/usr/sbin", "/sbin")

_SAFE_PATHS_WINDOWS: tuple[str, ...] = (r"C:\Windows\System32", r"C:\Windows")


# Pattern: commands that invoke Python interpreter
_PYTHON_CMD_PATTERNS = (
    r"^\s*python",
    r"^\s*python3",
    r"\bpython\s",
    r"\bpython3\s",
    r"\bpython\s",
    r"\bpython3\s",
    r"\bpython3?\s",
)


def _looks_like_python_command(command: str) -> bool:
    """Check if the command invokes a Python interpreter."""
    import re
    for pattern in _PYTHON_CMD_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return True
    return False


def _parse_cli_dependency(dep: str) -> tuple[str, str, str]:
    """Parse dependency spec to extract CLI tool name.

    Mirrors the CLI branch of parse_dependency from pre_execute.py.
    Returns (kind, name, install_spec).
    """
    from middleware.sandbox.utils import _parse_cli_dependency as _parse

    return _parse(dep)


def _get_safe_env_paths() -> tuple[str, ...]:
    """
    收紧 PATH 环境变量，移除跨平台不安全路径，保留系统安全路径。

    Unix (Linux/macOS): 过滤 /usr/local/bin, /opt/local/bin, /snap/bin 等，
        替换为 /usr/bin, /bin, /usr/sbin, /sbin。
    Windows: 过滤 C:\\Users, C:\\Temp, C:\\ProgramData，
        替换为 C:\\Windows\\System32, C:\\Windows。
    """
    current_path = os.environ.get("PATH", "")

    if sys.platform == "win32":
        raw_paths = [p.strip() for p in current_path.split(";") if p.strip()]
        safe: list[str] = []

        for p in raw_paths:
            p_lower = p.lower()
            # Skip user-writable or temp paths
            if any(p_lower.startswith(prefix.lower()) for prefix in ("c:\\users", "c:\\temp", "c:\\programdata")):
                continue
            safe.append(p)

        # Ensure system-safe paths are present
        safe_lower = {p.lower() for p in safe}
        for sp in _SAFE_PATHS_WINDOWS:
            if sp.lower() not in safe_lower:
                safe.append(sp)

        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for p in safe:
            norm = os.path.normpath(p).lower()
            if norm not in seen:
                seen.add(norm)
                deduped.append(p)

        return tuple(deduped)

    else:
        # Unix: split by colon, filter unsafe paths
        raw_paths = [p.strip() for p in current_path.split(":") if p.strip()]
        safe: list[str] = []

        for p in raw_paths:
            if p in _UNSAFE_PATHS_UNIX:
                continue
            # Also skip paths under user home dirs
            if any(p.startswith(prefix) for prefix in _UNSAFE_PATH_PREFIXES_UNIX):
                continue
            safe.append(p)

        # Ensure system-safe paths are present
        safe_set = set(safe)
        for sp in _SAFE_PATHS_UNIX:
            if sp not in safe_set:
                safe.append(sp)

        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for p in safe:
            norm = os.path.normpath(p)
            if norm not in seen:
                seen.add(norm)
                deduped.append(p)

        return tuple(deduped)


# Regex to extract the final cd target from a command chain
_CD_TARGET_RE = re.compile(
    r"^\s*cd\s+(?P<path>(?:'[^']*'|\"[^\"]*\"|[^\s;&|])+)",
    re.MULTILINE,
)


def _extract_final_cd_target(command: str) -> str | None:
    """Extract the final cd target from a command chain."""
    segments = re.split(r"\s*(?:&&|\|\|)\s*", command.strip())
    for seg in reversed(segments):
        seg = seg.strip()
        if seg.startswith("cd "):
            match = _CD_TARGET_RE.match(seg)
            if match:
                path = match.group("path")
                return path.strip("'\"")
    return None


async def bash(
    command: str,
    work_dir: str | None = None,
    source_dir: str | None = None,
    stdin: str | None = None,
    env: dict[str, str] | None = None,
    skill_deps: list[str] | None = None,
) -> str:
    """
    Execute a shell command in the run_dir.
    IMPORTANT: This is a STATELESS environment. Environment variables or `cd`
    will not persist across calls. Use `&&` to chain commands (e.g., `cd src && ls`).
    Interactive commands (like vim, nano, top) are strictly prohibited.

    Args:
        command: The shell command to run.
        work_dir: Working directory for the subprocess (absolute path).
        source_dir: Skill source root for NODE_PATH / vendor resolution.
        stdin: Optional standard input passed to the command.
        env: Optional extra environment variables.
        skill_deps: Skill metadata dependencies (auto-installed for Python commands).
    """
    try:
        default_cwd = Path(work_dir) if work_dir else None
        if default_cwd is None:
            raise RuntimeError(
                "bash requires work_dir to be provided by the execution context. "
                "Ensure the args_processor pipeline injects work_dir for the bash tool."
            )

        boundary = get_boundary()

        cd_target = _extract_final_cd_target(command)
        if cd_target:
            cd_abs = Path(cd_target).expanduser().is_absolute()
            if cd_abs:
                resolved = Path(cd_target).expanduser().resolve()
                if boundary.is_system_path(resolved):
                    return (
                        "ERR: bash command blocked by policy\n"
                        f"Reason: cd target '{cd_target}' is a system directory.\n"
                        "Hint: Use paths under workspace or user directories."
                    )
                if resolved.exists() and resolved.is_dir():
                    effective_cwd = resolved
                else:
                    effective_cwd = default_cwd
            else:
                # Relative cd: resolve from work_dir, no system path check needed
                resolved = (default_cwd / Path(cd_target)).resolve()
                if resolved.exists() and resolved.is_dir():
                    effective_cwd = resolved
                else:
                    effective_cwd = default_cwd
        else:
            effective_cwd = default_cwd

        node_search_root: Path | None = None
        if source_dir:
            node_search_root = Path(source_dir)
        elif effective_cwd != default_cwd:
            node_search_root = effective_cwd

        extra_env = dict(env) if env else {}

        # Tighten PATH: remove unsafe paths, ensure system-safe paths
        safe_path_parts = _get_safe_env_paths()
        path_sep = ";" if sys.platform == "win32" else ":"
        extra_env["PATH"] = path_sep.join(safe_path_parts)

        if node_search_root:
            node_env = build_node_path_from_dir(node_search_root)
            if node_env:
                for k, v in node_env.items():
                    extra_env.setdefault(k, v)

        # Validate command for dangerous patterns
        sanitized_command, warnings, reject_reason = _sanitize_bash_command(command)
        if reject_reason:
            return (
                "ERR: bash command blocked by policy\n"
                f"Reason: {reject_reason}\n"
                "Hint: Use paths under @ROOT (workspace), and avoid system paths or traversal patterns."
            )

        # Auto-install CLI tool warnings for missing system tools (ffmpeg etc.)
        install_warnings: list[str] = []
        if skill_deps and _looks_like_python_command(sanitized_command):
            # Note: Python package installation for bash is intentionally skipped.
            # Python packages installed to sandbox venv are NOT accessible to
            # bash commands running in the system environment. CLI tool detection
            # (ffmpeg, git, etc.) is still useful.
            for dep in skill_deps:
                kind, name, _ = _parse_cli_dependency(dep)
                if kind == "cli":
                    import shutil
                    if shutil.which(name) is None:
                        install_warnings.append(
                            f"CLI tool '{name}' not found in PATH. "
                            f"Install it if the command requires it."
                        )

        result = execute_shell(
            command=sanitized_command,
            extra_env=extra_env if extra_env else None,
            work_dir=effective_cwd,
        )

        out = str(result.result) if result.result else ""
        err = result.error or ""

        # [ANALYSIS-LOG] Log command execution summary
        _logger.info(
            "[ANALYSIS-LOG] bash exec: command_preview='{}', success={}, "
            "out_len={}, err_len={}, work_dir='{}'",
            command[:120] if command else "(empty)",
            result.success,
            len(out),
            len(err),
            str(effective_cwd) if effective_cwd else "(none)",
        )

        if len(out) > 50000:
            out = out[:50000] + "\n... [STDOUT TRUNCATED]"
        if len(err) > 50000:
            err = err[:50000] + "\n... [STDERR TRUNCATED]"

        warning_msg = ""
        if install_warnings:
            warning_msg += "DEPENDENCY WARNINGS:\n" + "\n".join(f"  - {w}" for w in install_warnings) + "\n\n"
        if warnings:
            warning_msg += (
                "WARNINGS:\n" + "\n".join(f"  - {w}" for w in warnings) + "\n\n"
            )

        if not result.success:
            return (
                f"{warning_msg}EXIT CODE: {result.error_type.value if result.error_type else '1'}\n"
                f"STDOUT:\n{out}\nSTDERR:\n{err}"
            )
        return f"{warning_msg}STDOUT:\n{out}" if out else f"{warning_msg}SUCCESS: (No output)"

    except Exception as e:
        return f"ERR: bash execution failed: {e}"


def _sanitize_bash_command(command: str) -> tuple[str, list[str], str | None]:
    """Validate bash command using PathBoundary."""
    warnings: list[str] = []
    boundary = get_boundary()
    command, reject_reason = boundary.validate_bash_command(command)
    if reject_reason:
        return command, warnings, reject_reason
    return command, warnings, None


bash._schema = _SCHEMA  # type: ignore[attr-defined]
