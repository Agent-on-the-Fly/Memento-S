"""Tools for file and directory operations.

Note: Path validation and coercion is handled by ToolContext in the execution layer.
These tools receive pre-processed absolute paths that are guaranteed to be within allowed boundaries.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from middleware.utils.path_security import IGNORE_DIRS, validate_path_arg


async def list_dir_tool(
    path: str = ".",
    max_depth: int = 2,
) -> str:
    """
    List the contents of a directory as a tree.
    Use this to understand the project structure and find files.

    Args:
        path: The directory path to list (absolute path, pre-validated).
        max_depth: Maximum recursion depth (default 2).
    """

    def _run() -> str:
        try:
            error = validate_path_arg(path)
            if error:
                return error

            # Path is pre-validated by ToolContext as absolute path
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


async def read_file_tool(
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
            error = validate_path_arg(path)
            if error:
                return error

            # Path is pre-validated by ToolContext as absolute path
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


async def file_create_tool(
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
            error = validate_path_arg(path)
            if error:
                return error

            # Path is pre-validated by ToolContext as absolute path
            target = Path(path)

            if target.exists() and not overwrite:
                return f"ERR: File already exists at {target}. Use overwrite=true to replace it, or edit_file_by_lines to modify it."

            target.parent.mkdir(parents=True, exist_ok=True)
            action = "Overwrote" if target.exists() else "Created"
            target.write_text(content, encoding="utf-8")
            return f"SUCCESS: {action} file {target}"
        except Exception as e:
            return f"ERR: file_create failed: {e}"

    return await asyncio.to_thread(_run)


async def edit_file_by_lines_tool(
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
            error = validate_path_arg(path)
            if error:
                return error

            # Path is pre-validated by ToolContext as absolute path
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
