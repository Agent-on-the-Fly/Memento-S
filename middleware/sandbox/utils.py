# SPDX-License-Identifier: Apache-2.0
"""Shared utilities for sandbox management."""

from __future__ import annotations

import shutil
from typing import Any


def check_missing_cli_tools(
    dependencies: list[str],
) -> list[str]:
    """Check which CLI tools from a dependency list are missing from PATH.

    Returns warning messages for each missing CLI tool.
    """
    import re

    warnings: list[str] = []
    for dep in dependencies:
        kind, name, _ = _parse_cli_dependency(dep)
        if kind == "cli" and name:
            if shutil.which(name) is None:
                warnings.append(
                    f"CLI tool '{name}' not found in PATH. "
                    f"Please install it if the command requires it."
                )
    return warnings


def _parse_cli_dependency(dep: str) -> tuple[str, str, str]:
    """Parse a dependency spec string to extract CLI tool information.

    Mirrors the CLI detection logic from pre_execute.py.

    Args:
        dep: A dependency spec string, e.g. "cli:ffmpeg", "ffmpeg", "pip:numpy"

    Returns:
        A tuple of (kind, name, install_spec).
        kind is one of "python", "cli", "none"
    """
    raw = (dep or "").strip()
    if not raw:
        return ("none", "", "")
    lowered = raw.lower()
    if lowered.startswith("cli:"):
        tool = raw.split(":", 1)[1].strip()
        return ("cli", tool, tool)
    if lowered.startswith("pip:") or lowered.startswith("py:"):
        mod = raw.split(":", 1)[1].strip()
        return ("python", mod, mod)
    # Known CLI tools without prefix
    if raw.lower() in {"ffmpeg"}:
        return ("cli", raw, raw)
    # Otherwise treat as python package
    return ("python", raw, raw)
