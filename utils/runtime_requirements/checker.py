# SPDX-License-Identifier: Apache-2.0
"""Runtime requirements checking — verify presence and version of system-level tools."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from middleware.utils.platform import node_install_hint, uv_install_hint, venv_python

# -------------------------------------------------------------------------- #
#  Result types
# -------------------------------------------------------------------------- #


@dataclass
class RequirementStatus:
    """Status of a single system dependency."""

    name: str
    found: bool
    version: str | None = None
    path: str | None = None
    install_hint: str | None = None
    required: bool = False

    @property
    def status_icon(self) -> str:
        return "✅" if self.found else "❌"


# -------------------------------------------------------------------------- #
#  Core check helpers
# -------------------------------------------------------------------------- #


def _which_path(name: str) -> str | None:
    path = shutil.which(name)
    return path


def _run_version(args: list[str], timeout: int = 5) -> str | None:
    """Run a command and return its stripped stdout, or None on failure."""
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0]
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


# -------------------------------------------------------------------------- #
#  Individual dependency checkers
# -------------------------------------------------------------------------- #

_REQUIRED: bool = True
_OPTIONAL: bool = False


def check_node() -> RequirementStatus:
    """Check Node.js (node command)."""
    path = _which_path("node")
    version = None
    if path:
        version = _run_version(["node", "--version"])
    return RequirementStatus(
        name="node",
        found=bool(path),
        version=version,
        path=path,
        install_hint=node_install_hint(),
        required=_REQUIRED,
    )


def check_npm() -> RequirementStatus:
    """Check npm (Node package manager)."""
    path = _which_path("npm")
    version = None
    if path:
        version = _run_version(["npm", "--version"])
    return RequirementStatus(
        name="npm",
        found=bool(path),
        version=version,
        path=path,
        required=_REQUIRED,
    )


def check_bun() -> RequirementStatus:
    """Check Bun JS runtime."""
    path = _which_path("bun")
    version = None
    if path:
        version = _run_version(["bun", "--version"])
    return RequirementStatus(
        name="bun",
        found=bool(path),
        version=version,
        path=path,
        install_hint="brew install bun  # macOS\ncurl -fsSL https://bun.sh/install | bash  # Linux",
        required=_OPTIONAL,
    )


def check_tsx() -> RequirementStatus:
    """Check tsx — TypeScript executor."""
    path = _which_path("tsx")
    version = None
    if path:
        version = _run_version(["tsx", "--version"])
    return RequirementStatus(
        name="tsx",
        found=bool(path),
        version=version,
        path=path,
        install_hint="npm install -g tsx",
        required=_OPTIONAL,
    )


def check_uv() -> RequirementStatus:
    """Check uv Python package manager."""
    path = _which_path("uv")
    version = None
    if path:
        version = _run_version(["uv", "--version"])
    return RequirementStatus(
        name="uv",
        found=bool(path),
        version=version,
        path=path,
        install_hint=uv_install_hint(),
        required=_REQUIRED,
    )


def check_python() -> RequirementStatus:
    """Check Python executable used by the uv sandbox (venv python)."""
    try:
        from middleware.config import g_config

        if not g_config.is_loaded():
            return RequirementStatus(
                name="python",
                found=False,
                install_hint="Unable to check — application config not initialized",
                required=_REQUIRED,
            )

        if not g_config.paths.venv_dir:
            return RequirementStatus(
                name="python",
                found=False,
                install_hint="venv_dir not configured in settings",
                required=_REQUIRED,
            )

        venv_dir = Path(g_config.paths.venv_dir).expanduser().resolve()
        python_exe = venv_python(venv_dir)

        if python_exe.exists():
            version = _run_version([str(python_exe), "--version"])
            return RequirementStatus(
                name="python",
                found=True,
                version=version,
                path=str(python_exe),
                required=_REQUIRED,
            )
        else:
            return RequirementStatus(
                name="python",
                found=False,
                path=str(python_exe),
                install_hint="Run the app once — uv will auto-create the virtual environment",
                required=_REQUIRED,
            )
    except RuntimeError:
        # g_config.paths raises when config not loaded
        return RequirementStatus(
            name="python",
            found=False,
            install_hint="Unable to check — application config not initialized",
            required=_REQUIRED,
        )
    except Exception:
        return RequirementStatus(
            name="python",
            found=False,
            install_hint="Unable to check — error during config access",
            required=_REQUIRED,
        )


def check_git() -> RequirementStatus:
    """Check git."""
    path = _which_path("git")
    version = None
    if path:
        version = _run_version(["git", "--version"])
    return RequirementStatus(
        name="git",
        found=bool(path),
        version=version,
        path=path,
        required=_REQUIRED,
    )


def check_ffmpeg() -> RequirementStatus:
    """Check FFmpeg (used for audio transcoding)."""
    path = _which_path("ffmpeg")
    version = None
    if path:
        version = _run_version(["ffmpeg", "-version"])
    return RequirementStatus(
        name="ffmpeg",
        found=bool(path),
        version=version,
        path=path,
        required=_OPTIONAL,
    )


# -------------------------------------------------------------------------- #
#  Aggregated reports
# -------------------------------------------------------------------------- #

# Node.js ecosystem
NODE_DEPS: list = [
    check_node,
    check_npm,
    check_bun,
    check_tsx,
]

# Python ecosystem
PYTHON_DEPS: list = [
    check_uv,
    check_python,
    check_git,
]

# Optional tools used by specific skills
OPTIONAL_DEPS: list = [
    check_ffmpeg,
]


def check_all() -> list[RequirementStatus]:
    """Run all dependency checks and return a flat list."""
    return [fn() for fn in NODE_DEPS + PYTHON_DEPS + OPTIONAL_DEPS]


def check_required() -> list[RequirementStatus]:
    """Return only required dependencies."""
    return [s for s in check_all() if s.required]


def summarize() -> str:
    """Return a compact human-readable summary of required dependencies."""
    lines = ["[Diagnostics] System dependency summary:"]
    for dep in check_required():
        icon = "✅" if dep.found else "❌"
        ver = dep.version or "not found"
        lines.append(f"  {icon} {dep.name}: {ver}")
    return "\n".join(lines)
