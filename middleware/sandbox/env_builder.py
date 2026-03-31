"""Sandbox environment builder.

Builds environment variables for sandbox execution, combining:
1. System environment (whitelist filtered)
2. Config environment variables (pip mirrors, etc.)
3. Sandbox-specific variables (VIRTUAL_ENV, UV_PYTHON, PATH)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from middleware.utils.environment import filter_env_by_whitelist, get_config_env_vars
from middleware.utils.platform import venv_bin_dir


def build_env(
    extra: dict[str, str] | None = None,
    use_sandbox: bool = False,
    venv_path: Path | None = None,
    python_executable: Path | None = None,
    full_system_env: bool = False,
    config: Any = None,
) -> dict[str, str]:
    """Build environment variables for execution.

    This unified function supports multiple use cases:
    - Basic execution environment (default)
    - Sandbox-aware execution (use_sandbox=True)
    - Direct sandbox path specification (venv_path + python_executable)
    - UV subprocess (full_system_env=True, for pip mirror settings)

    Args:
        extra: Extra environment variables (highest priority)
        use_sandbox: Auto-detect and use sandbox environment
        venv_path: Direct venv path (overrides use_sandbox)
        python_executable: Direct python executable path (required with venv_path)
        full_system_env: Use full os.environ instead of whitelist (for uv commands)
        config: Optional config instance

    Returns:
        Environment variables dictionary
    """
    # Determine sandbox paths
    resolved_venv_path = venv_path
    resolved_python = python_executable

    if use_sandbox and resolved_venv_path is None:
        # Auto-detect sandbox
        try:
            from .base import get_sandbox

            sandbox = get_sandbox()
            if hasattr(sandbox, "python_executable"):
                resolved_python = Path(sandbox.python_executable)
                resolved_venv_path = resolved_python.parent.parent
        except Exception:
            pass

    # Build base environment
    if full_system_env:
        # For uv commands: full system env + config
        env = dict(os.environ)
    else:
        # For sandbox execution: whitelist filtered
        env = filter_env_by_whitelist()

    # Add config environment variables
    env.update(get_config_env_vars(config))

    # Add sandbox-specific variables if available
    if resolved_venv_path and resolved_python:
        env["VIRTUAL_ENV"] = str(resolved_venv_path)
        env["UV_PYTHON"] = str(resolved_python)

        # Update PATH to include venv bin directory
        venv_bin = venv_bin_dir(resolved_venv_path)
        current_path = env.get("PATH", os.environ.get("PATH", ""))
        env["PATH"] = f"{venv_bin}{os.pathsep}{current_path}"

    # Add extra variables (highest priority)
    if extra:
        env.update(extra)

    return env
