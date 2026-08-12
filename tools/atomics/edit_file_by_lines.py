# SPDX-License-Identifier: Apache-2.0
"""Edit file by line range tool."""

from __future__ import annotations

import asyncio
from pathlib import Path

_NAME = "edit_file_by_lines"
_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path to the file. Must be a file path, not file content.",
        },
        "start_line": {
            "type": "integer",
            "description": "The first line number to replace (1-indexed).",
        },
        "end_line": {
            "type": "integer",
            "description": "The last line number to replace (inclusive).",
        },
        "new_content": {
            "type": "string",
            "description": "The exact new text to put in place of the replaced lines.",
        },
    },
    "required": ["path", "start_line", "end_line", "new_content"],
}


async def edit_file_by_lines(
    path: str,
    start_line: int,
    end_line: int,
    new_content: str,
) -> str:
    """
    Replace specific lines in a file with new content.
    This is extremely robust. To INSERT code, replace a line with itself + new code.
    To DELETE lines, pass an empty string to new_content.
    IMPORTANT: You must ensure the indentation of new_content matches the original file!

    Args:
        path: Path to the file (absolute path, pre-validated).
        start_line: Starting line number (1-indexed).
        end_line: Ending line number (inclusive).
        new_content: New content to replace the lines.
    """
    def _run() -> str:
        try:
            target = Path(path)

            if not target.exists():
                return f"ERR: File not found: {target}"

            backup_path = target.with_suffix(target.suffix + ".bak")
            target.parent.joinpath(backup_path.name).write_bytes(target.read_bytes())

            content = target.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines(keepends=True)

            if start_line < 1 or end_line < start_line:
                return (
                    f"ERR: Invalid range start_line={start_line}, end_line={end_line}."
                )

            new_lines = new_content.splitlines(keepends=True)
            if new_content and not new_content.endswith("\n"):
                new_lines[-1] = new_lines[-1] + "\n"

            prefix = lines[: start_line - 1]
            suffix = lines[end_line:] if end_line <= len(lines) else []
            final_lines = prefix + new_lines + suffix

            target.write_text("".join(final_lines), encoding="utf-8")

            show_start = max(1, start_line - 3)
            show_end = min(len(final_lines), start_line + len(new_lines) + 3)

            context_snippet = []
            for i in range(show_start - 1, show_end):
                marker = (
                    ">> "
                    if (start_line - 1 <= i < start_line - 1 + len(new_lines))
                    else "   "
                )
                context_snippet.append(
                    f"{marker}{i + 1:5d} | {final_lines[i].rstrip()}"
                )

            snippet_str = "\n".join(context_snippet)
            return (
                f"SUCCESS: Replaced lines {start_line} to {end_line}.\n"
                f"Please verify the indentation and syntax in the resulting snippet below:\n"
                f"-----------------------------------\n{snippet_str}\n-----------------------------------"
            )
        except Exception as e:
            return f"ERR: edit_file_by_lines failed: {e}"

    return await asyncio.to_thread(_run)


edit_file_by_lines._schema = _SCHEMA  # type: ignore[attr-defined]
