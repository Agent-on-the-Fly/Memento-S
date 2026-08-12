# SPDX-License-Identifier: Apache-2.0
"""File create tool."""

from __future__ import annotations

import asyncio
from pathlib import Path

_NAME = "file_create"
_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path to the file.",
        },
        "content": {
            "type": "string",
            "description": "The content to write into the file.",
            "default": "",
        },
        "overwrite": {
            "type": "boolean",
            "description": "If true, overwrite existing file. Default false.",
            "default": False,
        },
    },
    "required": ["path"],
}


async def file_create(
    path: str,
    content: str = "",
    overwrite: bool = False,
) -> str:
    """
    Create a new file, or overwrite an existing file if overwrite=True.

    Args:
        path: Path to the new file (absolute path, pre-validated).
        content: The initial content to write into the file.
        overwrite: If True, overwrite existing file. Default False.
    """
    def _run() -> str:
        try:
            target = Path(path)

            if target.is_dir():
                if not target.exists():
                    target.mkdir(parents=True, exist_ok=True)
                return f"SUCCESS: Directory already exists at {target}"

            if target.exists() and not overwrite:
                return (
                    f"ERR: File already exists at {target}. "
                    f"Use overwrite=true to replace it, or edit_file_by_lines to modify it."
                )

            target.parent.mkdir(parents=True, exist_ok=True)
            action = "Overwrote" if target.exists() else "Created"
            target.write_text(content, encoding="utf-8")
            return f"SUCCESS: {action} file {target}"
        except Exception as e:
            return f"ERR: file_create failed: {e}"

    return await asyncio.to_thread(_run)


file_create._schema = _SCHEMA  # type: ignore[attr-defined]
