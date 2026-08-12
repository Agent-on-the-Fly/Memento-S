# SPDX-License-Identifier: Apache-2.0
"""Grep/search tool."""

from __future__ import annotations

import asyncio
import os
import re
import fnmatch
from pathlib import Path
from typing import Optional

from shared.tools.tool_security import IGNORE_DIRS

_NAME = "grep"
_SCHEMA = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": 'The Python regex pattern to search for (e.g., "def process_data", "error|Error").',
        },
        "dir_path": {
            "type": "string",
            "description": "The directory to search in. Auto-filled from execution context if not provided.",
        },
        "file_pattern": {
            "type": "string",
            "description": 'Glob pattern to filter files (e.g., "*.py", "*.ts", default is "*").',
            "default": "*",
        },
        "text": {
            "type": "string",
            "description": "If provided, search within this text string instead of files. Useful for searching in fetched content, command output, or loaded file content.",
        },
        "show_line_numbers": {
            "type": "boolean",
            "description": "Whether to show line numbers in results (default True). Only applies when searching text.",
            "default": True,
        },
    },
    "required": ["pattern"],
}


async def grep(
    pattern: str,
    dir_path: str = ".",
    file_pattern: str = "*",
    text: Optional[str] = None,
    show_line_numbers: bool = True,
) -> str:
    """
    Search for a regex pattern in files or text.

    If 'text' is provided, search within that text string.
    Otherwise, search across text files in 'dir_path'.

    Args:
        pattern: The Python regex pattern to search for.
        dir_path: The directory to search in (absolute path, pre-validated).
        file_pattern: Glob pattern to filter files (e.g., "*.py", "*.ts").
        text: If provided, search within this text string instead of files.
        show_line_numbers: Whether to show line numbers in results (default True).
    """
    def _run() -> str:
        if text is not None:
            try:
                regex = re.compile(pattern, re.MULTILINE)
                lines = text.splitlines()
                results = []
                max_matches = 50

                for i, line in enumerate(lines):
                    if regex.search(line):
                        if show_line_numbers:
                            results.append(f"{i + 1}: {line}")
                        else:
                            results.append(line)
                        if len(results) >= max_matches:
                            results.append(f"... (truncated at {max_matches} matches)")
                            break

                if not results:
                    return f"No matches found for '{pattern}' in text"
                return "\n".join(results)
            except re.error as e:
                return f"ERR: Invalid regex pattern: {e}"
            except Exception as e:
                return f"ERR: grep failed: {e}"

        # Search in files
        try:
            target = Path(dir_path)
            regex = re.compile(pattern)
            results = []
            max_matches = 100

            for root, dirs, files in os.walk(target):
                dirs[:] = [
                    d for d in dirs
                    if d not in IGNORE_DIRS and not d.startswith(".")
                ]
                for file in files:
                    if not fnmatch.fnmatch(file, file_pattern):
                        continue
                    if file.startswith("."):
                        continue
                    filepath = Path(root) / file

                    if filepath.suffix.lower() in {
                        ".png", ".jpg", ".pyc", ".pdf", ".zip",
                    }:
                        continue

                    try:
                        lines = filepath.read_text(
                            encoding="utf-8", errors="ignore"
                        ).splitlines()
                        for i, line in enumerate(lines):
                            if regex.search(line):
                                rel_path = filepath.relative_to(target)
                                results.append(f"{rel_path}:{i + 1}: {line.strip()}")
                                if len(results) >= max_matches:
                                    results.append(
                                        f"... (truncated at {max_matches} matches)"
                                    )
                                    return "\n".join(results)
                    except Exception:
                        pass

            if not results:
                return f"No matches found for '{pattern}' in {dir_path}"
            return "\n".join(results)
        except re.error as e:
            return f"ERR: Invalid regex pattern: {e}"
        except Exception as e:
            return f"ERR: grep failed: {e}"

    return await asyncio.to_thread(_run)


grep._schema = _SCHEMA  # type: ignore[attr-defined]
