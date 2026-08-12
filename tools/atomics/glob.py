# SPDX-License-Identifier: Apache-2.0
"""Glob tool — find files matching a pattern."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from shared.tools.tool_security import IGNORE_DIRS

_NAME = "glob"
_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Root directory to search from. Auto-filled from execution context if not provided.",
        },
        "pattern": {
            "type": "string",
            "description": 'Glob pattern (e.g., "**/*.py", "*.json", "src/**/*.ts").',
        },
        "max_depth": {
            "type": "integer",
            "description": "Maximum directory traversal depth (relative to path). -1 means unlimited.",
            "default": -1,
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of results to return.",
            "default": 100,
        },
    },
    "required": ["pattern"],
}


async def glob(
    path: str = ".",
    pattern: str = "*",
    max_depth: int = -1,
    max_results: int = 100,
) -> str:
    """
    Find files matching a glob pattern within a directory tree.

    Supports standard glob patterns:
    - * matches everything within one directory level
    - ** matches any subdirectory depth
    - ? matches single character
    - [abc] matches character sets

    Args:
        path: Root directory to search from (absolute path).
        pattern: Glob pattern (e.g., "**/*.py", "*.json").
        max_depth: Maximum directory traversal depth. -1 means unlimited.
        max_results: Maximum number of results to return.
    """
    def _run() -> str:
        try:
            root = Path(path)
            if not root.exists():
                return f"ERR: Directory not found: {root}"
            if not root.is_dir():
                return f"ERR: Path is not a directory: {root}"

            results: list[str] = []

            if "**" in pattern:
                # Recursive glob
                if pattern.startswith("**/"):
                    # Normalize src/**/foo to just **/foo
                    search_pattern = pattern
                else:
                    search_pattern = "**/" + pattern

                try:
                    matched = root.glob(search_pattern)
                    for p in matched:
                        if p.name in IGNORE_DIRS or p.name.startswith("."):
                            continue
                        rel = p.relative_to(root)
                        if max_depth != -1 and len(rel.parts) > max_depth:
                            continue
                        results.append(str(rel))
                        if len(results) >= max_results:
                            results.append(f"... (truncated at {max_results} results)")
                            break
                except ValueError:
                    return f"ERR: Invalid glob pattern: {pattern}"
            else:
                # Non-recursive glob at current level
                try:
                    for p in root.glob(pattern):
                        if p.name in IGNORE_DIRS or p.name.startswith("."):
                            continue
                        rel = p.relative_to(root)
                        results.append(str(rel))
                        if len(results) >= max_results:
                            results.append(f"... (truncated at {max_results} results)")
                            break
                except ValueError:
                    return f"ERR: Invalid glob pattern: {pattern}"

            if not results:
                return f"No files match pattern '{pattern}' in {root}"

            return "\n".join(results)

        except Exception as e:
            return f"ERR: glob failed: {e}"

    return await asyncio.to_thread(_run)


glob._schema = _SCHEMA  # type: ignore[attr-defined]
