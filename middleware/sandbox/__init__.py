# SPDX-License-Identifier: Apache-2.0
"""Sandbox — UV-based local execution helpers.

This module provides a unified interface for executing code and shell commands
in isolated sandbox environments.

Supports two sandbox types:
- Python: UvLocalSandbox (uv + virtual environment)
- Node.js: NodeSandbox (system Node.js + native module resolution)

The execute_shell() function auto-detects the appropriate sandbox based on
the command content (Node.js commands like npm/pnpm/yarn/node/deno automatically
use the NodeSandbox; everything else uses the Python sandbox).
"""

from __future__ import annotations

from pathlib import Path

from shared.fs.snapshot import SandboxSnapshot as SandboxArtifactCollector
from .base import BaseSandbox, RuntimeType, detect_runtime, get_sandbox
from .env_builder import build_env
from .schema import ErrorType, SandboxExecutionOutcome
from .uv import UvLocalSandbox

__all__ = [
    "BaseSandbox",
    "get_sandbox",
    "RuntimeType",
    "detect_runtime",
    "UvLocalSandbox",
    "SandboxArtifactCollector",
    "build_env",
    "execute_shell",
    "execute_python",
    "SandboxExecutionOutcome",
    "ErrorType",
]


def _default_bash_timeout() -> int:
    """Read bash_timeout_sec from config, falling back to 300."""
    try:
        from middleware.config import g_config

        return getattr(g_config.skills.execution, "bash_timeout_sec", 300) or 300
    except Exception:
        return 300


def execute_shell(
    command: str,
    extra_env: dict[str, str] | None = None,
    work_dir: str | Path | None = None,
    timeout: int | None = None,
    use_sandbox: bool = True,
    collect_artifacts: bool = False,
    session_id: str = "",
) -> SandboxExecutionOutcome:
    """Execute a shell command in sandbox environment.

    This function auto-detects whether the command is a Node.js ecosystem
    command (npm/pnpm/yarn/bun/node/deno/etc.) and routes it to the
    appropriate sandbox (NodeSandbox or UvLocalSandbox).

    Args:
        command: Shell command to execute
        extra_env: Additional environment variables
        work_dir: Working directory for the command
        timeout: Timeout in seconds (default: bash_timeout_sec from config)
        use_sandbox: Whether to use sandbox environment
        collect_artifacts: Whether to collect generated files as artifacts
        session_id: Session identifier for artifact collection

    Returns:
        SandboxExecutionOutcome with execution results and artifacts
    """
    if timeout is None:
        timeout = _default_bash_timeout()

    if not use_sandbox:
        import subprocess

        cwd = Path(work_dir) if work_dir else None
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=timeout,
            )

            if result.returncode != 0:
                return SandboxExecutionOutcome(
                    success=False,
                    result=result.stdout.strip() or None,
                    error=f"Command failed with code {result.returncode}\nSTDERR:\n{result.stderr}",
                    error_type=ErrorType.EXECUTION_ERROR,
                    skill_name="shell",
                )

            return SandboxExecutionOutcome(
                success=True,
                result=result.stdout.strip(),
                skill_name="shell",
            )

        except subprocess.TimeoutExpired:
            return SandboxExecutionOutcome(
                success=False,
                result=None,
                error=f"Command timed out after {timeout}s",
                error_type=ErrorType.TIMEOUT,
                skill_name="shell",
            )

    # Auto-detect runtime from command content
    runtime_type = detect_runtime(command)
    sandbox = get_sandbox(runtime=runtime_type)

    return sandbox.execute_shell(
        command=command,
        extra_env=extra_env,
        work_dir=Path(work_dir) if work_dir else None,
        timeout=timeout,
        collect_artifacts=collect_artifacts,
        session_id=session_id,
    )


def execute_python(
    code: str,
    deps: list[str] | None = None,
    session_id: str = "",
) -> SandboxExecutionOutcome:
    """Execute Python code in sandbox environment.

    Args:
        code: Python code to execute
        deps: Optional dependencies to install
        session_id: Session identifier for sandbox paths

    Returns:
        SandboxExecutionOutcome with execution results and artifacts
    """
    sandbox = get_sandbox(runtime=RuntimeType.PYTHON)
    return sandbox.run_code(
        code=code,
        name="python_exec",
        deps=deps,
        session_id=session_id,
    )
