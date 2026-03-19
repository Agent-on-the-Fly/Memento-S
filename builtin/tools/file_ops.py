"""
Tools for file and directory operations.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from ._base import _resolve_path, _IGNORE_DIRS


def _validate_path_arg(path: str) -> str | None:
    if not isinstance(path, str):
        return "ERR: Path must be a string."
    if "\n" in path or "\r" in path:
        return "ERR: Path must not contain newlines. Provide a valid file path, not file content."
    if path.lstrip().startswith("#"):
        return "ERR: Path looks like Markdown content. Provide a file path, not content."
    if len(path) > 4096:
        return "ERR: Path is too long. Provide a valid file path, not content."
    return None


async def list_dir_tool(
    path: str = ".",
    max_depth: int = 2,
    base_dir: str | None = None,
) -> str:
    """
    List the contents of a directory as a tree.
    Use this to understand the project structure and find files.
    
    Args:
        path: The directory path to list (default is current workspace).
        max_depth: Maximum recursion depth (default 2).
    """
    def _run() -> str:
        try:
            error = _validate_path_arg(path)
            if error:
                return error
            base = Path(base_dir).resolve() if base_dir else None
            target = _resolve_path(path, base)
            if not target.exists() or not target.is_dir():
                return f"ERR: Directory not found: {path}"

            lines = [f"Directory Tree for: {target.absolute()}"]
            
            def walk(current_path, current_depth: int, prefix: str = ""):
                if current_depth > max_depth:
                    return
                try:
                    entries = sorted(current_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
                    entries =[e for e in entries if e.name not in _IGNORE_DIRS and not e.name.startswith("._") and e.name != ".DS_Store"]
                except PermissionError:
                    lines.append(f"{prefix}[Permission Denied]")
                    return

                for i, entry in enumerate(entries):
                    is_last = (i == len(entries) - 1)
                    connector = "└── " if is_last else "├── "
                    lines.append(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")
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
    base_dir: str | None = None,
) -> str:
    """
    Read the contents of a file with line numbers.
    Always read files before editing them to get the exact line numbers.
    
    Args:
        path: Path to the file to read.
        start_line: Line number to start reading from (1-indexed, default 1).
        end_line: Line number to stop reading (inclusive). Use -1 to read to the end.
    """
    def _run() -> str:
        try:
            error = _validate_path_arg(path)
            if error:
                return error
            base = Path(base_dir).resolve() if base_dir else None
            target = _resolve_path(path, base)
            if not target.is_file():
                return f"ERR: File not found or is a directory: {path}"

            if target.stat().st_size > 10 * 1024 * 1024:
                return "ERR: File is too large (>10MB). Cannot read."

            content = target.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            total_lines = len(lines)

            _end = total_lines if end_line == -1 else min(end_line, total_lines)
            _start = max(1, start_line)

            if _start > total_lines:
                return f"ERR: start_line ({_start}) is beyond the file length ({total_lines})."

            sliced = lines[_start - 1 : _end]
            numbered = [f"{_start + i:5d} | {line}" for i, line in enumerate(sliced)]
            
            header = f"--- File: {path} (Lines {_start} to {_end} of {total_lines}) ---\n"
            return header + "\n".join(numbered)
        except Exception as e:
            return f"ERR: read_file failed: {e}"

    return await asyncio.to_thread(_run)


async def file_create_tool(
    path: str,
    content: str = "",
    base_dir: str | None = None,
) -> str:
    """
    Create a new file. Fails if the file already exists.
    
    Args:
        path: Path to the new file.
        content: The initial content to write into the file.
    """
    def _run() -> str:
        try:
            error = _validate_path_arg(path)
            if error:
                return error
            base = Path(base_dir).resolve() if base_dir else None
            target = _resolve_path(path, base)
            if target.exists():
                return f"ERR: File already exists at {path}. Use edit_file_by_lines to modify it."
            
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"SUCCESS: Created file {path}"
        except Exception as e:
            return f"ERR: file_create failed: {e}"

    return await asyncio.to_thread(_run)


async def edit_file_by_lines_tool(
    path: str,
    start_line: int,
    end_line: int,
    new_content: str,
    base_dir: str | None = None,
) -> str:
    """
    Replace specific lines in a file with new content.
    This is extremely robust. To INSERT code, replace a line with itself + new code.
    To DELETE lines, pass an empty string to new_content.
    IMPORTANT: You must ensure the indentation of new_content matches the original file!
    """
    def _run() -> str:
        try:
            error = _validate_path_arg(path)
            if error:
                return error
            base = Path(base_dir).resolve() if base_dir else None
            target = _resolve_path(path, base)
            if not target.exists():
                return f"ERR: File not found: {path}"

            backup_path = target.with_suffix(target.suffix + ".bak")
            target.parent.joinpath(backup_path.name).write_bytes(target.read_bytes())

            content = target.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines(keepends=True)
            
            if start_line < 1 or end_line < start_line:
                return f"ERR: Invalid range start_line={start_line}, end_line={end_line}."

            new_lines = new_content.splitlines(keepends=True)
            if new_content and not new_content.endswith("\n"):
                new_lines[-1] = new_lines[-1] + "\n"

            prefix = lines[:start_line - 1]
            suffix = lines[end_line:] if end_line <= len(lines) else[]
            final_lines = prefix + new_lines + suffix
            
            target.write_text("".join(final_lines), encoding="utf-8")
            
            show_start = max(1, start_line - 3)
            show_end = min(len(final_lines), start_line + len(new_lines) + 3)
            
            context_snippet =[]
            for i in range(show_start - 1, show_end):
                marker = ">> " if (start_line - 1 <= i < start_line - 1 + len(new_lines)) else "   "
                context_snippet.append(f"{marker}{i + 1:5d} | {final_lines[i].rstrip()}")
                
            snippet_str = "\n".join(context_snippet)
            return (f"SUCCESS: Replaced lines {start_line} to {end_line}.\n"
                    f"Please verify the indentation and syntax in the resulting snippet below:\n"
                    f"-----------------------------------\n{snippet_str}\n-----------------------------------")
        except Exception as e:
            return f"ERR: edit_file_by_lines failed: {e}"

    return await asyncio.to_thread(_run)
