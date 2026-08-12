# SPDX-License-Identifier: Apache-2.0
"""Read file tool."""

from __future__ import annotations

import asyncio
from pathlib import Path

_NAME = "read_file"
_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path to the file to read. Must be a file path, not file content.",
        },
        "start_line": {
            "type": "integer",
            "description": "Line number to start reading from (1-indexed, default 1).",
            "default": 1,
        },
        "end_line": {
            "type": "integer",
            "description": "Line number to stop reading (inclusive). Use -1 to read to the end.",
            "default": -1,
        },
    },
    "required": ["path"],
}


async def read_file(
    path: str,
    start_line: int = 1,
    end_line: int = -1,
) -> str:
    """
    Read the contents of a file with line numbers.
    Always read files before editing them to get the exact line numbers.

    Args:
        path: Path to the file to read (absolute path, pre-validated).
        start_line: Line number to start reading from (1-indexed, default 1).
        end_line: Line number to stop reading (inclusive). Use -1 to read to the end.
    """
    def _run() -> str:
        try:
            target = Path(path)

            if not target.is_file():
                return f"ERR: File not found or is a directory: {target}"

            _SIZE_LIMIT = 10 * 1024 * 1024
            _CHUNK_HINT = 300
            file_size = target.stat().st_size
            is_large = file_size > _SIZE_LIMIT
            range_specified = not (start_line == 1 and end_line == -1)

            if is_large and not range_specified:
                with target.open(encoding="utf-8", errors="replace") as f:
                    total_lines = sum(1 for _ in f)
                size_mb = file_size / 1024 / 1024
                return (
                    f"INFO: File '{path}' is large ({size_mb:.1f} MB, {total_lines} lines total). "
                    f"Use start_line/end_line to read in chunks (suggested: {_CHUNK_HINT} lines each).\n"
                    f"Example: read_file(path='{path}', start_line=1, end_line={_CHUNK_HINT})"
                )

            if is_large and range_specified:
                _start = max(1, start_line)
                lines_buf = []
                total_lines = 0
                with target.open(encoding="utf-8", errors="replace") as f:
                    for line_num, line in enumerate(f, 1):
                        total_lines = line_num
                        if line_num < _start:
                            continue
                        if end_line != -1 and line_num > end_line:
                            continue
                        lines_buf.append(line.rstrip("\n\r"))
                _end = total_lines if end_line == -1 else min(end_line, total_lines)
                if _start > total_lines:
                    return f"ERR: start_line ({_start}) is beyond the file length ({total_lines})."
                numbered = [
                    f"{_start + i:5d} | {line}" for i, line in enumerate(lines_buf)
                ]
                header = f"--- File: {path} (Lines {_start} to {_end} of {total_lines}) ---\n"
                return header + "\n".join(numbered)

            content = target.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            total_lines = len(lines)

            _end = total_lines if end_line == -1 else min(end_line, total_lines)
            _start = max(1, start_line)

            if _start > total_lines:
                return f"ERR: start_line ({_start}) is beyond the file length ({total_lines})."

            sliced = lines[_start - 1 : _end]
            numbered = [f"{_start + i:5d} | {line}" for i, line in enumerate(sliced)]

            header = (
                f"--- File: {path} (Lines {_start} to {_end} of {total_lines}) ---\n"
            )
            return header + "\n".join(numbered)
        except Exception as e:
            return f"ERR: read_file failed: {e}"

    return await asyncio.to_thread(_run)


read_file._schema = _SCHEMA  # type: ignore[attr-defined]
