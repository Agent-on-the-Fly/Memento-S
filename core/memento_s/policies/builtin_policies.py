"""Built-in policy implementations."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from core.utils.platform import is_path_within, temp_dir
from middleware.config import g_config

# --- Dangerous bash patterns (multi-layer) ---
# All platform patterns merged: a POSIX pattern won't false-positive on
# Windows and vice versa, so keeping them together is both simpler and safer.

_BASH_EXACT_PATTERNS = [
    # POSIX
    "rm -rf /",
    "rm -rf /*",
    "mkfs",
    "dd if=",
    ":(){ :|:& };:",
    # Windows
    "format c:",
    "format d:",
    "rd /s /q c:\\",
    "rd /s /q \\",
]

_BASH_SUBSTRING_PATTERNS = [
    # POSIX
    "sudo rm -rf",
    "chmod 777",
    "chmod -R 777",
    "> /dev/sda",
    "mv /* ",
    "rm -rf ~",
    "rm -rf $HOME",
    "shutdown",
    "reboot",
    "init 0",
    "init 6",
    # Windows
    "del /f /s /q",
    "rd /s /q",
    "reg delete",
    "bcdedit",
    # Cross-platform
    "git push --force",
    "git push -f",
    "drop table",
    "drop database",
    "truncate table",
]

_BASH_INJECTION_PATTERNS = [
    re.compile(r"\$\(curl\s"),
    re.compile(r"`curl\s"),
    re.compile(r"base64\s.*\|\s*sh"),
    re.compile(r"base64\s.*\|\s*bash"),
    re.compile(r"eval\s+[\"']"),
    re.compile(r"\|\s*sh\b"),
    re.compile(r"\|\s*bash\b"),
    re.compile(r"curl.*\|\s*(sh|bash)"),
    re.compile(r"wget.*\|\s*(sh|bash)"),
    re.compile(r"\\x[0-9a-f]{2}", re.IGNORECASE),
]


def block_dangerous_bash(action_name: str, args: dict[str, Any]) -> bool:
    """Block destructive shell patterns with multi-layer detection."""
    if action_name != "bash":
        return True

    command = str(args.get("command", ""))
    lowered = command.lower()

    # Layer 1: exact patterns
    if any(p in lowered for p in _BASH_EXACT_PATTERNS):
        return False

    # Layer 2: substring patterns
    if any(p in lowered for p in _BASH_SUBSTRING_PATTERNS):
        return False

    # Layer 3: regex injection detection
    if any(pat.search(command) for pat in _BASH_INJECTION_PATTERNS):
        return False

    return True


def restrict_file_ops(action_name: str, args: dict[str, Any]) -> bool:
    """Restrict file operations to workspace directory."""
    if action_name not in (
        "edit_file_by_lines",
        "file_create",
        "read_file",
        "list_dir",
    ):
        return True

    raw_path = str(args.get("path", ""))
    if not raw_path:
        return True

    base_dir = Path(g_config.paths.workspace_dir)
    p = Path(raw_path)
    if not p.is_absolute():
        p = base_dir / p
    resolved = p.resolve()

    if is_path_within(resolved, base_dir):
        return True
    if is_path_within(resolved, Path(temp_dir())):
        return True

    return False
