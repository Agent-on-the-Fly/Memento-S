# SPDX-License-Identifier: Apache-2.0
"""List directory tool."""

from __future__ import annotations

import asyncio
from pathlib import Path

from shared.tools.tool_security import IGNORE_DIRS

_NAME = "list_dir"
_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "The directory path to list. Auto-filled from execution context if not provided.",
        },
        "max_depth": {
            "type": "integer",
            "description": "Maximum recursion depth (default 2).",
            "default": 2,
        },
    },
    "required": [],
}


async def list_dir(path: str = ".", max_depth: int = 2) -> str:
    """
    List the contents of a directory as a tree.
    Use this to understand the project structure and find files.

    Args:
        path: The directory path to list (absolute path, pre-validated).
        max_depth: Maximum recursion depth (default 2).
    """
    def _run() -> str:
        try:
            target = Path(path)
            if not target.exists() or not target.is_dir():
                return f"ERR: Directory not found: {target}"

            lines = [f"Directory Tree for: {target}"]

            def walk(current_path, current_depth: int, prefix: str = ""):
                if current_depth > max_depth:
                    return
                try:
                    entries = sorted(
                        current_path.iterdir(),
                        key=lambda x: (not x.is_dir(), x.name.lower()),
                    )
                    entries = [
                        e
                        for e in entries
                        if e.name not in IGNORE_DIRS
                        and not e.name.startswith("._")
                        and e.name != ".DS_Store"
                    ]
                except PermissionError:
                    lines.append(f"{prefix}[Permission Denied]")
                    return

                for i, entry in enumerate(entries):
                    is_last = i == len(entries) - 1
                    connector = "└── " if is_last else "├── "
                    lines.append(
                        f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}"
                    )
                    if entry.is_dir():
                        extension = "    " if is_last else "│   "
                        walk(entry, current_depth + 1, prefix + extension)

            walk(target, 1)
            return "\n".join(lines)
        except Exception as e:
            return f"ERR: list_dir failed: {e}"

    return await asyncio.to_thread(_run)


list_dir._schema = _SCHEMA  # type: ignore[attr-defined]
