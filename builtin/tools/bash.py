"""Bash command execution tool with sandbox environment support."""

from __future__ import annotations

import os
import re
from pathlib import Path

from middleware.config import g_config
from middleware.sandbox import execute_shell


def _get_dangerous_patterns() -> list[str]:
    """Get dangerous command patterns for current platform."""
    if os.name == "nt":  # Windows
        return [
            # Windows file deletion
            r"rmdir\s+/[sq]\s+[a-zA-Z]:\\",
            r"del\s+/[fq]\s+[a-zA-Z]:\\",
            r"Remove-Item\s+-Recurse\s+-Force",
            # Format commands
            r"format\s+[a-zA-Z]:",
            # Direct writes to Windows system directories
            r">\s*[a-zA-Z]:\\Windows",
            r">\s*[a-zA-Z]:\\Program\s+Files",
            r">\s*C:\\Windows\\System32",
            # Registry operations
            r"reg\s+(add|delete|import)",
            r"Set-ItemProperty\s+-Path\s+.*Registry",
        ]
    else:  # Unix/Linux/macOS
        return [
            # System-level file operations outside workspace
            r"rm\s+-rf\s+/[^\s]*",
            r"rm\s+-rf\s+\$HOME",
            r"rm\s+-rf\s+~",
            # Direct writes to system directories
            r">\s*/etc/\w+",
            r">\s*/usr/\w+",
            r">\s*/bin/\w+",
            r">\s*/sbin/\w+",
            r">\s*/lib.*/\w+",
        ]


def _get_system_paths() -> frozenset[str]:
    """Get system paths for current platform."""
    if os.name == "nt":  # Windows
        return frozenset(
            {
                "C:\\Windows",
                "C:\\Windows\\System32",
                "C:\\Program Files",
                "C:\\Program Files (x86)",
                "C:\\Users",
                "C:\\ProgramData",
                "C:\\",
            }
        )
    else:  # Unix/Linux/macOS
        return frozenset(
            {
                "/etc",
                "/usr",
                "/bin",
                "/sbin",
                "/lib",
                "/lib64",
                "/opt",
                "/var",
                "/tmp",
                "/root",
                "/home",
                "/sys",
                "/proc",
                "/dev",
                "/boot",
                "/run",
            }
        )


# Compile dangerous patterns for current platform
_DANGEROUS_REGEX = [
    re.compile(pattern, re.IGNORECASE) for pattern in _get_dangerous_patterns()
]
_SYSTEM_PATHS = _get_system_paths()


def _sanitize_bash_command(command: str) -> tuple[str, list[str], str | None]:
    """Validate bash command and return explanatory rejection when needed.

    Returns:
        Tuple of (sanitized_command, warnings, reject_reason)
    """
    warnings = []

    # Check for dangerous patterns
    for pattern in _DANGEROUS_REGEX:
        if pattern.search(command):
            return (
                command,
                warnings,
                f"Command rejected: matched dangerous pattern `{pattern.pattern}`.",
            )

    # Check for path traversal attempts (cross-platform)
    if re.search(r"\.\./\.\./\.\.", command):
        return (
            command,
            warnings,
            "Command rejected: path traversal (`../../../`) is not allowed.",
        )

    # Check for absolute paths to system directories
    for sys_path in _SYSTEM_PATHS:
        normalized_cmd = command.replace("\\", "/")
        normalized_sys = sys_path.replace("\\", "/")
        pattern = rf"(?:\s|^|\"|'|=){re.escape(normalized_sys)}(?:/|\\|\s|$|\"|')"
        if re.search(pattern, normalized_cmd, re.IGNORECASE):
            return (
                command,
                warnings,
                f"Command rejected: references protected system path `{sys_path}`.",
            )

    return command, warnings, None


async def bash_tool(
    command: str,
    env: dict[str, str] | None = None,
    stdin: str | None = None,
    work_dir: str | None = None,
) -> str:
    """
    Execute a shell command in the workspace.

    This tool supports sandbox isolation - if a sandbox is configured,
    it will automatically use the sandbox's virtual environment.

    IMPORTANT: This is a STATELESS environment. Environment variables or `cd`
    will not persist across calls. Use `&&` to chain commands (e.g., `cd src && ls`).
    Interactive commands (like vim, nano, top) are strictly prohibited.

    Args:
        command: The shell command to run.
        env: Optional custom environment variables to inject.
        stdin: Optional standard input to pass to the command.
        work_dir: The working directory for command execution (absolute path, pre-validated).
    """
    try:
        # work_dir is pre-validated by ToolContext as absolute path
        # It defaults to context.root_dir via enrich_args() if not provided
        cwd = Path(work_dir) if work_dir else None

        if cwd is None:
            # Fallback only when called directly without ToolContext
            if not hasattr(g_config, "_config") or g_config._config is None:
                raise RuntimeError(
                    "g_config not initialized. Please ensure bootstrap() is called before using tools."
                )
            if g_config.paths.workspace_dir is None:
                raise RuntimeError(
                    "g_config.paths.workspace_dir is None. Please check configuration."
                )
            cwd = Path(g_config.paths.workspace_dir)

        # Validate command for dangerous patterns and system paths
        sanitized_command, warnings, reject_reason = _sanitize_bash_command(command)
        if reject_reason:
            return (
                "ERR: bash command blocked by policy\n"
                f"Reason: {reject_reason}\n"
                "Hint: Use paths under @ROOT (workspace), and avoid system paths or traversal patterns."
            )

        # Path validation is handled by ToolContext.resolve_path() in the execution layer
        # The _sanitize_bash_command above only checks for dangerous patterns and system paths
        # as a defense-in-depth measure, but does not re-implement boundary checking

        # Execute command through sandbox (which handles environment setup and path isolation)
        result = execute_shell(
            command=sanitized_command,
            extra_env=env,
            work_dir=cwd,
            timeout=300,
        )

        # Format output from SkillExecutionOutcome
        out = str(result.result) if result.result else ""
        err = result.error or ""

        if len(out) > 50000:
            out = out[:50000] + "\n... [STDOUT TRUNCATED]"
        if len(err) > 50000:
            err = err[:50000] + "\n... [STDERR TRUNCATED]"

        # Add warnings to output if any
        warning_msg = ""
        if warnings:
            warning_msg = (
                "WARNINGS:\n" + "\n".join(f"  - {w}" for w in warnings) + "\n\n"
            )

        if not result.success:
            return f"{warning_msg}EXIT CODE: {result.error_type.value if result.error_type else '1'}\nSTDOUT:\n{out}\nSTDERR:\n{err}"
        return (
            f"{warning_msg}STDOUT:\n{out}"
            if out
            else f"{warning_msg}SUCCESS: (No output)"
        )

    except Exception as e:
        return f"ERR: bash execution failed: {e}"
