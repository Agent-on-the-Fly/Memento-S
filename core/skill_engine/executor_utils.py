"""Shared utility helpers for executor modules."""

from __future__ import annotations

from typing import Any

_OP_TYPE_ALIASES: dict[str, str] = {
    # Shared bridge aliases
    "shell": "run_command",
    "google_search": "web_search",
    "search": "web_search",
    "fetch_url": "fetch",
    "fetch_markdown": "fetch",
    # Filesystem aliases
    "read_text_file": "read_file",
    "write_text_file": "write_file",
    "get_file_info": "file_info",
    "list_dir": "list_directory",
    "dir_tree": "directory_tree",
    "mkdir": "create_directory",
    "rm": "delete_file",
    "mv": "move_file",
    "cp": "copy_file",
}


def canonicalize_op_type(raw_type: Any) -> str:
    op_type = str(raw_type or "").strip().lower()
    if not op_type:
        return ""
    return _OP_TYPE_ALIASES.get(op_type, op_type)


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if not lowered:
            return default
        if lowered in {"1", "true", "yes", "on", "y", "t"}:
            return True
        if lowered in {"0", "false", "no", "off", "n", "f"}:
            return False
        return default
    return bool(value)


def parse_int(
    value: Any,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    parsed = default
    try:
        if isinstance(value, bool):
            raise ValueError("bool is not accepted as int")
        parsed = int(value)
    except Exception:
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed
