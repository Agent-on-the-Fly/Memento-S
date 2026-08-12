# SPDX-License-Identifier: Apache-2.0
"""Cross-platform path boundary model.

Single source of truth for all path and permission logic.
Handles Windows / Linux / macOS with platform-specific system paths,
external path prefixes, and permission checks.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import logging as _logging

_platform_logger = _logging.getLogger("path_boundary")
_platform_logger.setLevel(_logging.DEBUG)

Platform = Literal["windows", "linux", "darwin"]


def detect_platform() -> Platform:
    """Detect current platform from sys.platform."""
    if sys.platform == "win32":
        return "windows"
    elif sys.platform == "darwin":
        return "darwin"
    else:
        return "linux"


# ---------------------------------------------------------------------------
# Platform-aware system paths
# ---------------------------------------------------------------------------


def _make_safe_filename(path: Path, platform: Platform) -> str:
    """Generate a safe cross-platform filename for auto-copy targets."""
    name = str(path).lstrip("/\\")
    if platform == "windows":
        name = re.sub(r'[<>:"\'|?*]', "_", name)
        name = name.replace(":", "_")
    name = name.replace("/", "_").replace("\\", "_")
    if len(name) > 200:
        name = name[:200]
    return name


def _get_system_paths_for_platform(platform: Platform) -> frozenset[str]:
    """Return platform-specific system directory prefixes."""
    if platform == "windows":
        system_root = os.environ.get("SystemRoot", "C:\\Windows")
        program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
        prog_files_x86 = os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")
        return frozenset({
            system_root,
            program_files,
            prog_files_x86,
            os.environ.get("ProgramData", "C:\\ProgramData"),
            "C:\\Windows\\System32",
            "C:\\Users",
            "C:\\",
            "D:\\",
        })
    elif platform == "darwin":
        return frozenset({
            "/System",
            "/usr",
            "/bin",
            "/sbin",
            "/lib",
            "/lib64",
            "/etc",
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
            "/Applications",
            "/Library",
            "/private",
        })
    else:  # linux
        return frozenset({
            "/usr",
            "/bin",
            "/sbin",
            "/lib",
            "/lib64",
            "/etc",
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
        })


# ---------------------------------------------------------------------------
# Platform-aware dangerous command patterns
# ---------------------------------------------------------------------------


def _get_dangerous_patterns_for_platform(platform: Platform) -> list[str]:
    """Return platform-specific dangerous command regex patterns."""
    base = [
        r"\brsync\s+--delete",
        r"\bdd\s+.*of=",
        r"\bmkfifo\b",
        r"\bgit\s+push\s+--force\b",
    ]
    if platform == "windows":
        return base + [
            r"rmdir\s+/[sq]\s+[a-zA-Z]:\\",
            r"del\s+/[fq]\s+[a-zA-Z]:\\",
            r"Remove-Item\s+-Recurse\s+-Force",
            r"format\s+[a-zA-Z]:",
            r">\s*[a-zA-Z]:\\Windows",
            r">\s*[a-zA-Z]:\\Program\s+Files",
            r">\s*C:\\Windows\\System32",
            r"\breg\s+(add|delete|import)",
            r"Set-ItemProperty\s+-Path\s+.*Registry",
        ]
    else:
        return base + [
            r"\bsudo\s+rm\s+-rf",
            r"\brm\s+-rf\s+/[^\s]*",
            r"\brm\s+-rf\s+\$HOME",
            r"\brm\s+-rf\s+~",
            r"\bchmod\s+777\s+/",
            r">\s*/etc/\w+",
            r">\s*/usr/\w+",
            r">\s*/bin/\w+",
            r">\s*/sbin/\w+",
            r">\s*/lib.*/\w+",
            r"\|\s*/etc/",
        ]


def _get_protected_dirs_for_platform(platform: Platform) -> frozenset[str]:
    """Return platform-specific protected directory prefixes."""
    if platform == "windows":
        system_root = os.environ.get("SystemRoot", "C:\\Windows")
        program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
        return frozenset({
            system_root,
            program_files,
            "C:\\Windows",
            "C:\\Program Files",
            "C:\\Program Files (x86)",
            "C:\\Windows\\System32",
        })
    else:
        return frozenset({
            "/etc", "/usr", "/bin", "/sbin", "/lib", "/lib64", "/opt",
        })


# ---------------------------------------------------------------------------
# PathBoundary — single source of truth
# ---------------------------------------------------------------------------


class PathBoundary:
    """Unified path boundary and permission model (cross-platform).

    This is the single source of truth for all path and permission logic
    in the skill execution pipeline.
    """

    # Ignored directories during traversal (platform-agnostic)
    IGNORE_DIRS: frozenset[str] = frozenset({
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "build",
        "dist",
        ".pytest_cache",
        ".mypy_cache",
        ".tox",
        ".idea",
    })

    def __init__(self, platform: Platform | None = None) -> None:
        object.__setattr__(self, "_platform", platform or detect_platform())

    @property
    def platform(self) -> Platform:
        return self._platform

    @property
    def SYSTEM_PATHS(self) -> frozenset[str]:
        return _get_system_paths_for_platform(self._platform)

    @property
    def DANGEROUS_PATTERNS(self) -> list[str]:
        return _get_dangerous_patterns_for_platform(self._platform)

    @property
    def DANGEROUS_REGEX(self) -> list[re.Pattern]:
        return [re.compile(p, re.IGNORECASE) for p in self.DANGEROUS_PATTERNS]

    @property
    def PROTECTED_DIRS(self) -> frozenset[str]:
        return _get_protected_dirs_for_platform(self._platform)

    # ------------------------------------------------------------------
    # System directory check
    # ------------------------------------------------------------------

    def is_system_path(self, path: Path) -> bool:
        """Check if path is under a system directory."""
        resolved = path.resolve()
        for sys_prefix in self.SYSTEM_PATHS:
            sys_path = Path(sys_prefix).resolve()
            try:
                resolved.relative_to(sys_path)
                return True
            except ValueError:
                pass
        return False

    # ------------------------------------------------------------------
    # Path normalization (cross-platform)
    # ------------------------------------------------------------------

    def normalize_path(self, raw: str) -> Path:
        """Normalize a raw path string for the current platform."""
        if not raw:
            return Path(".")
        if self._platform == "windows":
            p = Path(str(raw).replace("/", "\\"))
        else:
            p = Path(raw)
        return p

    def expand_path(self, path: Path) -> Path:
        """Expand ~ and environment variables (cross-platform)."""
        path_str = str(path)
        if self._platform == "windows":
            expanded = os.path.expandvars(path_str)
            expanded = os.path.expanduser(expanded)
        else:
            expanded = os.path.expanduser(path_str)
            expanded = os.path.expandvars(expanded)
        return Path(expanded)

    # ------------------------------------------------------------------
    # Bash command validation
    # ------------------------------------------------------------------

    def validate_bash_command(self, command: str) -> tuple[str, str | None]:
        """Validate a bash command for dangerous patterns and protected paths.

        Returns:
            Tuple of (command, reject_reason). If reject_reason is not None,
            the command should be blocked.
        """
        for pattern in self.DANGEROUS_REGEX:
            if pattern.search(command):
                _platform_logger.info(
                    "[ANALYSIS-LOG] validate_bash_command: MATCHED pattern='{}', "
                    "command_preview='{}'",
                    pattern.pattern,
                    command[:200] if command else "(empty)",
                )
                return (
                    command,
                    f"Command rejected: matched dangerous pattern `{pattern.pattern}`.",
                )

        if re.search(r"\.\./\.\./\.\.", command):
            return command, "Command rejected: path traversal (`../../../`) is not allowed."

        normalized_cmd = command.replace("\\", "/")
        for sys_path in self.PROTECTED_DIRS:
            normalized_sys = sys_path.replace("\\", "/")
            pattern = rf"(?:\s|^|\"|'|=){re.escape(normalized_sys)}(?:/|\\|\s|$|\"|')"
            if re.search(pattern, normalized_cmd, re.IGNORECASE):
                return (
                    command,
                    f"Command rejected: references protected path `{sys_path}`.",
                )

        return command, None

    # ------------------------------------------------------------------
    # Path argument validation (before resolution)
    # ------------------------------------------------------------------

    @staticmethod
    def validate_path_arg(path: str) -> str | None:
        """Validate raw path argument shape before resolution.

        Returns error message if invalid, None if valid.
        """
        if not isinstance(path, str):
            return "ERR: Path must be a string."
        if "\n" in path or "\r" in path:
            return "ERR: Path must not contain newlines."
        if path.lstrip().startswith("#"):
            return "ERR: Path looks like Markdown content."
        if len(path) > 4096:
            return "ERR: Path is too long."
        return None


# ---------------------------------------------------------------------------
# Module-level singleton (lazy, per-platform)
# ---------------------------------------------------------------------------

_boundaries: dict[Platform, PathBoundary] = {}


def get_boundary(platform: Platform | None = None) -> PathBoundary:
    """Get or create a PathBoundary for the given platform (default: current)."""
    p = platform or detect_platform()
    if p not in _boundaries:
        _boundaries[p] = PathBoundary(platform=p)
    return _boundaries[p]
