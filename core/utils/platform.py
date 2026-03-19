"""
Platform capability detection for the execution layer.
"""

from __future__ import annotations

import os
import shutil
import sys
import sysconfig
import tempfile
from pathlib import Path

# ------------------------------------------------------------------ #
#  Python executable
# ------------------------------------------------------------------ #

def python_executable() -> str:
    """Return the current Python interpreter path."""
    return sys.executable


# ------------------------------------------------------------------ #
#  Temp directory
# ------------------------------------------------------------------ #

def temp_dir() -> str:
    """Return the platform temp directory (cross-platform)."""
    return tempfile.gettempdir()


# ------------------------------------------------------------------ #
#  venv layout  (probed from sysconfig, not hardcoded)
# ------------------------------------------------------------------ #


def _detect_venv_layout() -> tuple[str, str]:
    if os.name == "nt":
        return "Scripts", "python.exe"
    return "bin", "python"


_VENV_BIN_NAME, _PYTHON_NAME = _detect_venv_layout()


def venv_bin_dir(venv_path: Path) -> Path:
    """Return the bin/Scripts directory inside a venv."""
    return venv_path / _VENV_BIN_NAME


def venv_python(venv_path: Path) -> Path:
    """Return the python executable path inside a venv."""
    return venv_bin_dir(venv_path) / _PYTHON_NAME


# ------------------------------------------------------------------ #
#  Script file extensions  (probed from PATHEXT on Windows)
# ------------------------------------------------------------------ #

def _detect_script_extensions() -> set[str]:
    exts: set[str] = {".py"}
    pathext = os.environ.get("PATHEXT", "")
    if pathext:
        for ext in pathext.split(os.pathsep):
            if ext.lower() in {".bat", ".cmd", ".ps1"}:
                exts.add(ext.lower())
    else:
        exts.add(".sh")
    return exts


SCRIPT_EXTENSIONS: set[str] = _detect_script_extensions()


# ------------------------------------------------------------------ #
#  File permissions
# ------------------------------------------------------------------ #

def chmod_executable(path: Path) -> None:
    """chmod +x on POSIX; no-op elsewhere."""
    if os.name == "posix":
        path.chmod(0o755)


# ------------------------------------------------------------------ #
#  Shell capability detection
# ------------------------------------------------------------------ #

def has_bash() -> bool:
    return shutil.which("bash") is not None


def has_powershell() -> bool:
    return (
        shutil.which("pwsh") is not None
        or shutil.which("powershell") is not None
    )


# ------------------------------------------------------------------ #
#  Path safety
# ------------------------------------------------------------------ #

def is_path_within(child: Path, parent: Path) -> bool:
    """Cross-platform check whether *child* resides inside *parent*."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


# ------------------------------------------------------------------ #
#  Environment variable whitelist
#  All platform entries merged; missing vars simply won't match.
# ------------------------------------------------------------------ #

_ENV_WHITELIST_PATTERNS: set[str] = {
    # universal
    "PATH", "LANG", "LC_ALL", "LC_CTYPE",
    "VIRTUAL_ENV", "PYTHONPATH", "UV_*", "MEMENTO_*",
    # pip / PyPI mirror
    "PIP_*",
    # proxy / certs (cross-platform)
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
    "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
    # POSIX-common (absent on Windows → ignored)
    "HOME", "USER", "LOGNAME", "TERM", "SHELL",
    "TMPDIR", "XDG_RUNTIME_DIR",
    # Windows-common (absent on POSIX → ignored)
    "USERPROFILE", "TEMP", "TMP", "COMSPEC", "PATHEXT",
    "APPDATA", "LOCALAPPDATA", "SystemRoot",
    "HOMEDRIVE", "HOMEPATH",
    "SYSTEMROOT", "WINDIR", "SYSTEMDRIVE",
}


def filter_env_by_whitelist(
    source: dict[str, str] | None = None,
) -> dict[str, str]:
    """Filter environment variables through the whitelist.

    Supports trailing ``*`` as a prefix-match wildcard.
    """
    src = source if source is not None else dict(os.environ)
    return {
        k: v
        for k, v in src.items()
        if any(
            k == w or (w.endswith("*") and k.startswith(w[:-1]))
            for w in _ENV_WHITELIST_PATTERNS
        )
    }


# ------------------------------------------------------------------ #
#  pip shim helpers
# ------------------------------------------------------------------ #

def pip_shim_path(venv_path: Path) -> Path:
    name = "pip" if os.name == "posix" else "pip.bat"
    return venv_bin_dir(venv_path) / name


def pip_shim_content(python_path: Path) -> str:
    if os.name == "posix":
        return f'#!/bin/sh\nexec "{python_path}" -m pip "$@"\n'
    return f'@echo off\n"{python_path}" -m pip %*\n'


# ------------------------------------------------------------------ #
#  Hint messages  (based on detected shell capabilities)
# ------------------------------------------------------------------ #

def background_hint() -> str:
    if has_bash():
        return "nohup ... &"
    return "start /b ..."


def uv_install_hint() -> str:
    if has_powershell():
        return 'powershell -c "irm https://astral.sh/uv/install.ps1 | iex"'
    return "curl -LsSf https://astral.sh/uv/install.sh | sh"


# ------------------------------------------------------------------ #
#  Subprocess text encoding
#  Windows Chinese locale defaults to GBK, which chokes on UTF-8 output.
# ------------------------------------------------------------------ #

SUBPROCESS_TEXT_KWARGS: dict[str, object] = {
    "text": True,
    "encoding": "utf-8",
    "errors": "replace",
}
