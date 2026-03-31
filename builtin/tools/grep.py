"""Search tool: grep for files or text."""

from __future__ import annotations

import asyncio
import os
import re
import fnmatch
from pathlib import Path
from typing import Optional

from middleware.config import g_config
from middleware.utils.path_security import IGNORE_DIRS


async def grep_tool(
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
        # Search in text string
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
            # Path is pre-validated by ToolContext as absolute path
            target = Path(dir_path)
            regex = re.compile(pattern)
            results = []
            max_matches = 100

            for root, dirs, files in os.walk(target):
                dirs[:] = [
                    d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")
                ]
                for file in files:
                    if not fnmatch.fnmatch(file, file_pattern):
                        continue
                    if file.startswith("."):
                        continue
                    filepath = Path(root) / file

                    if filepath.suffix.lower() in {
                        ".png",
                        ".jpg",
                        ".pyc",
                        ".pdf",
                        ".zip",
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
