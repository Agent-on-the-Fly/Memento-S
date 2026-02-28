"""Subprocess / Git helpers used by the skill engine."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from core.config import SKILL_DYNAMIC_FETCH_TIMEOUT_SEC


# ===================================================================
# Git environment
# ===================================================================

def _no_git_prompt_env() -> dict[str, str]:
    """Return a copy of ``os.environ`` with variables that suppress
    interactive Git/SSH credential prompts so subprocesses never hang
    waiting for input.
    """
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_ASKPASS"] = "/bin/echo"
    env["GIT_SSH_COMMAND"] = "ssh -oBatchMode=yes"
    return env


_NO_GIT_PROMPT_ENV: dict[str, str] = _no_git_prompt_env()


# ===================================================================
# Subprocess capture
# ===================================================================

def _run_command_capture(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = SKILL_DYNAMIC_FETCH_TIMEOUT_SEC,
) -> tuple[bool, str]:
    """Run *cmd* and return ``(success, output_or_error)``.

    Uses ``_NO_GIT_PROMPT_ENV`` to prevent interactive prompts.
    """
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            env=_NO_GIT_PROMPT_ENV,
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if p.returncode == 0:
        return True, (p.stdout or "").strip()
    err = (p.stderr or p.stdout or "").strip()
    return False, err or f"command failed: {' '.join(cmd)}"
