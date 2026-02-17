"""Skill execution functions extracted from agent.py.

This module contains all plan normalization, skill context helpers,
and executor functions for filesystem, terminal, web, uv-pip, and
skill-creator operations, plus the bridge dispatcher.
"""

import asyncio
import json
import os
import re
import requests
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Set

from core.config import (
    BUILTIN_BRIDGE_SKILLS,
    FILESYSTEM_OP_TYPES,
    TERMINAL_OP_TYPES,
    WORKSPACE_DIR,
    UV_PIP_OP_TYPES,
    WEB_OP_TYPES,
)
from core.utils.logging_utils import log_event
from core.utils.path_utils import (
    _find_venv,
    _parse_json_object,
    _rewrite_command_paths_for_skill,
    _resolve_dir,
    _resolve_path,
    _safe_subpath,
    _shell_command,
    _truncate_text,
    _venv_bin_dir,
)
from core.skill_engine.skill_resolver import _resolve_skill_dir

# ---------------------------------------------------------------------------
# Terminal toolkit (lazy import, moved from core/config.py)
# ---------------------------------------------------------------------------
_TERMINAL_IMPORT_ERROR: Exception | None = None
try:
    from camel.toolkits.terminal_toolkit import utils as terminal_utils
except Exception as exc:  # pragma: no cover - runtime environment dependent
    terminal_utils = None  # type: ignore[assignment]
    _TERMINAL_IMPORT_ERROR = exc


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


def _canonicalize_op_type(raw_type: Any) -> str:
    """Return canonical op type by applying shared alias rules."""
    op_type = str(raw_type or "").strip().lower()
    if not op_type:
        return ""
    return _OP_TYPE_ALIASES.get(op_type, op_type)


def _parse_bool(value: Any, default: bool = False) -> bool:
    """Parse booleans robustly so string values like 'false' stay false."""
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


def _parse_int(
    value: Any,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Parse integers with fallback/clamp semantics."""
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


def _coerce_existing_dir(raw_dir: Any) -> Path | None:
    """Resolve raw directory path and return it only when it exists."""
    if not isinstance(raw_dir, str) or not raw_dir.strip():
        return None
    path = Path(raw_dir.strip())
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    if path.exists() and path.is_dir():
        return path
    return None


def _read_skill_context(raw_ctx: Any) -> tuple[str | None, Path | None]:
    """Parse raw _skill_context payload into (name, dir)."""
    if not isinstance(raw_ctx, dict):
        return None, None
    raw_name = raw_ctx.get("name")
    skill_name = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else None
    return skill_name, _coerce_existing_dir(raw_ctx.get("dir"))




# ===================================================================
# 1. Plan normalization
# ===================================================================

def _normalize_op_dict(op: Any) -> dict[str, Any] | None:
    """Normalize one operation dict, handling legacy wrapper formats."""
    if not isinstance(op, dict):
        return None

    out = dict(op)
    op_type = out.get("type") or out.get("op") or out.get("action")
    if isinstance(op_type, str) and op_type.strip():
        out["type"] = op_type.strip()

    wrapper_type = (out.get("type") or "").strip().lower()
    if wrapper_type in {"mcp_tool", "mcp_call", "mcp"}:
        actual_tool = out.get("tool") or out.get("name")
        args = _parse_json_object(out.get("args") or out.get("arguments") or out.get("parameters"))
        merged: dict[str, Any] = {}
        merged.update(args)
        merged.update({k: v for k, v in out.items() if k not in {"args", "arguments", "parameters"}})
        if isinstance(actual_tool, str) and actual_tool.strip():
            merged["type"] = actual_tool.strip()
        out = merged

    if isinstance(out.get("arguments"), str):
        parsed_args = _parse_json_object(out.get("arguments"))
        if parsed_args:
            merged = dict(parsed_args)
            merged.update({k: v for k, v in out.items() if k != "arguments"})
            out = merged

    if "type" not in out and isinstance(op_type, str) and op_type.strip():
        out["type"] = op_type.strip()

    return out


def _tool_call_to_op(call: Any) -> dict[str, Any] | None:
    """Convert a tool_calls-style entry to a normalized op dict."""
    if not isinstance(call, dict):
        return None

    name = call.get("name") or call.get("tool")
    args = call.get("args") or call.get("arguments") or call.get("parameters")

    fn = call.get("function")
    if isinstance(fn, dict):
        fn_name = fn.get("name")
        if isinstance(fn_name, str) and fn_name.strip():
            name = fn_name
        args = args or fn.get("arguments")
    if not isinstance(name, str) or not name.strip():
        fallback = call.get("type")
        if isinstance(fallback, str) and fallback.strip() and fallback.strip().lower() != "function":
            name = fallback

    args_dict = _parse_json_object(args)
    if not isinstance(name, str) or not name.strip():
        return None

    op: dict[str, Any] = {"type": name.strip()}
    op.update(args_dict)
    op.update(
        {
            k: v
            for k, v in call.items()
            if k not in {"name", "tool", "type", "function", "args", "arguments", "parameters"}
        }
    )
    return op


def _op_to_tool_call(op: Any, *, call_id: str) -> dict[str, Any] | None:
    """Convert one op-style entry into an OpenAI-style tool_call object."""
    normalized = _normalize_op_dict(op)
    if not isinstance(normalized, dict):
        return None

    op_type = str(normalized.get("type") or "").strip()
    if not op_type:
        return None

    raw_args = normalized.get("args") or normalized.get("arguments") or normalized.get("parameters")
    args: dict[str, Any] = _parse_json_object(raw_args)
    for key, value in normalized.items():
        if key in {
            "type",
            "op",
            "action",
            "name",
            "tool",
            "function",
            "id",
            "args",
            "arguments",
            "parameters",
        }:
            continue
        args[key] = value

    try:
        args_json = json.dumps(args, ensure_ascii=False)
    except TypeError:
        args_json = json.dumps({k: str(v) for k, v in args.items()}, ensure_ascii=False)

    raw_id = normalized.get("id")
    resolved_id = str(raw_id).strip() if isinstance(raw_id, str) and raw_id.strip() else call_id
    tool_call: dict[str, Any] = {
        "id": resolved_id,
        "type": "function",
        "function": {
            "name": op_type,
            "arguments": args_json,
        },
    }
    depends_on = normalized.get("depends_on")
    if isinstance(depends_on, list):
        tool_call["depends_on"] = depends_on
    policy = normalized.get("policy")
    if isinstance(policy, dict):
        tool_call["policy"] = policy
    protocol_version = normalized.get("protocol_version")
    if isinstance(protocol_version, str) and protocol_version.strip():
        tool_call["protocol_version"] = protocol_version.strip()
    return tool_call


def normalize_plan_shape(plan: Any) -> dict:
    """Normalize plan into both OpenAI `tool_calls` and internal `ops` lists."""
    if not isinstance(plan, dict):
        return {}

    normalized = dict(plan)
    normalized_ops: list[dict[str, Any]] = []

    def _append_from_tool_calls(items: list[Any]) -> None:
        for raw_call in items:
            op = _tool_call_to_op(raw_call)
            if op:
                normalized_ops.append(op)

    def _append_from_ops(items: list[Any]) -> None:
        for raw_op in items:
            op = _normalize_op_dict(raw_op)
            if op:
                normalized_ops.append(op)

    tool_calls = normalized.get("tool_calls")
    ops = normalized.get("ops")
    calls = normalized.get("calls")
    if isinstance(tool_calls, list) and tool_calls:
        _append_from_tool_calls(tool_calls)
    elif isinstance(ops, list) and ops:
        _append_from_ops(ops)
    elif isinstance(calls, list) and calls:
        _append_from_tool_calls(calls)

    normalized["ops"] = normalized_ops

    normalized_calls: list[dict[str, Any]] = []
    for idx, op in enumerate(normalized_ops, start=1):
        call = _op_to_tool_call(op, call_id=f"call_{idx}")
        if call:
            normalized_calls.append(call)
    normalized["tool_calls"] = normalized_calls

    return normalized


# ===================================================================
# 2. Skill context helpers
# ===================================================================

def _coerce_skill_context(plan: dict, fallback_skill: str) -> dict[str, str]:
    """Build/coerce _skill_context metadata on a plan dict."""
    raw = plan.get("_skill_context")
    ctx: dict[str, str] = dict(raw) if isinstance(raw, dict) else {}
    raw_name, skill_dir = _read_skill_context(ctx)
    name = raw_name or fallback_skill.strip()
    ctx["name"] = name

    if skill_dir is None:
        skill_dir = _resolve_skill_dir(name)
    if skill_dir is not None:
        ctx["dir"] = str(skill_dir)

    return ctx


def _extract_skill_context(plan: dict) -> tuple[str | None, Path | None, bool]:
    """Extract (skill_name, skill_dir, prefer_skill_paths) from plan."""
    skill_name, skill_dir = _read_skill_context(plan.get("_skill_context"))

    if skill_dir is None and skill_name:
        skill_dir = _resolve_skill_dir(skill_name)

    prefer_skill_paths = bool(skill_name and skill_name not in BUILTIN_BRIDGE_SKILLS)
    return skill_name, skill_dir, prefer_skill_paths


# ===================================================================
# 3. Skill creator executor
# ===================================================================

def _execute_skill_creator_plan(plan: dict) -> str:
    """Execute a skill-creator plan (create/update a skill directory)."""
    action = plan.get("action")
    skills_dir = str(plan.get("skills_dir") or "skills")
    skill_name = plan.get("skill_name")
    ops = plan.get("ops", [])

    if action not in {"create", "update"}:
        return f"Invalid action: {action}"
    if not isinstance(skill_name, str) or not skill_name.strip():
        return "Missing skill_name"
    if not isinstance(ops, list):
        return "Invalid tool_calls"

    base = (Path(skills_dir) / skill_name.strip()).resolve()
    report: list[str] = []

    if action == "create":
        base.mkdir(parents=True, exist_ok=True)
        report.append(f"ensure_dir OK: {base}")
    elif action == "update" and not base.exists():
        return f"Skill not found: {base}"

    for op in ops:
        if not isinstance(op, dict):
            report.append("SKIP: op is not a dict")
            continue
        op_type = str(op.get("type") or "").strip()
        rel_path = str(op.get("path") or "").strip()
        if op_type in {"mkdir", "write_file", "append_file", "replace_text"} and not rel_path:
            report.append(f"{op_type} SKIP: missing path")
            continue

        try:
            if op_type == "mkdir":
                p = _safe_subpath(base, rel_path)
                p.mkdir(parents=True, exist_ok=True)
                report.append(f"mkdir OK: {p}")
            elif op_type == "write_file":
                p = _safe_subpath(base, rel_path)
                overwrite = _parse_bool(op.get("overwrite"), True)
                if p.exists() and not overwrite:
                    report.append(f"write_file SKIP (exists): {p}")
                    continue
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(str(op.get("content", "")), encoding="utf-8")
                report.append(f"write_file OK: {p}")
            elif op_type == "append_file":
                p = _safe_subpath(base, rel_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                with p.open("a", encoding="utf-8") as f:
                    f.write(str(op.get("content", "")))
                report.append(f"append_file OK: {p}")
            elif op_type == "replace_text":
                p = _safe_subpath(base, rel_path)
                if not p.exists():
                    report.append(f"replace_text SKIP (missing): {p}")
                    continue
                old = str(op.get("old", ""))
                new = str(op.get("new", ""))
                max_n = _parse_int(op.get("max"), 1, minimum=1)
                text = p.read_text(encoding="utf-8")
                if old not in text:
                    report.append(f"replace_text NOOP (not found): {p}")
                    continue
                p.write_text(text.replace(old, new, max_n), encoding="utf-8")
                report.append(f"replace_text OK: {p}")
            else:
                report.append(f"unknown op: {op_type}")
        except Exception as exc:
            report.append(f"{op_type} ERR: {exc}")

    return "\n".join(report) if report else "No tool_calls"


# ===================================================================
# 4. Filesystem executor
# ===================================================================

def _default_workspace_base_dir() -> Path:
    """Return workspace dir for intermediate outputs, creating it if needed."""
    try:
        WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return WORKSPACE_DIR.resolve()


def _resolve_working_dir_or_workspace(
    raw_working_dir: Any,
    *,
    base_dir: Path | None = None,
) -> tuple[Path, bool]:
    """
    Resolve working_dir and fallback to workspace when path is missing/invalid.

    Returns:
        (resolved_dir, used_fallback)
    """
    workspace_dir = _default_workspace_base_dir()
    if raw_working_dir is None:
        return workspace_dir, False
    if isinstance(raw_working_dir, str) and not raw_working_dir.strip():
        return workspace_dir, False

    anchor = base_dir.resolve() if isinstance(base_dir, Path) else Path.cwd().resolve()
    resolved = _resolve_dir(anchor, raw_working_dir)
    if resolved.exists() and resolved.is_dir():
        return resolved, False
    return workspace_dir, True


def _filesystem_tree(
    path: Path,
    prefix: str = "",
    depth: int = 3,
    current_depth: int = 0,
) -> list[str]:
    """Build a tree-style directory listing."""
    if current_depth >= depth:
        return []
    lines: list[str] = []
    try:
        entries = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    except PermissionError:
        return [f"{prefix}[permission denied]"]
    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
        lines.append(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")
        if entry.is_dir():
            extension = "    " if is_last else "\u2502   "
            lines.extend(_filesystem_tree(entry, prefix + extension, depth, current_depth + 1))
    return lines


@dataclass(frozen=True)
class _FilesystemContext:
    base_dir: Path
    skill_dir: Path | None = None
    prefer_skill_paths: bool = False


def _filesystem_path(ctx: _FilesystemContext, raw_path: Any) -> Path:
    return _resolve_path(
        ctx.base_dir,
        raw_path,
        skill_dir=ctx.skill_dir,
        prefer_skill_paths=ctx.prefer_skill_paths,
    )


def _filesystem_op_path(ctx: _FilesystemContext, op: dict[str, Any], key: str = "path") -> Path:
    return _filesystem_path(ctx, op.get(key))


def _filesystem_source_destination(
    ctx: _FilesystemContext,
    op: dict[str, Any],
) -> tuple[Path, Path]:
    src = _filesystem_path(ctx, op.get("src") or op.get("source"))
    dst = _filesystem_path(ctx, op.get("dst") or op.get("destination"))
    return src, dst


def _normalize_filesystem_op(op: dict[str, Any]) -> tuple[dict[str, Any], str]:
    normalized = dict(op)
    op_type = _canonicalize_op_type(normalized.get("type"))
    if op_type == "replace_text":
        op_type = "edit_file"
        if "old_text" not in normalized and "old" in normalized:
            normalized["old_text"] = normalized.get("old")
        if "new_text" not in normalized and "new" in normalized:
            normalized["new_text"] = normalized.get("new")
    if op_type:
        normalized["type"] = op_type
    return normalized, op_type


def _fs_read_file(op: dict[str, Any], ctx: _FilesystemContext) -> str:
    path = _filesystem_op_path(ctx, op)
    if not path.exists():
        return f"read_file ERR: not found: {path}"
    if not path.is_file():
        return f"read_file ERR: not a file: {path}"
    content = path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    head = op.get("head")
    tail = op.get("tail")
    if isinstance(head, int):
        lines = lines[:head]
    elif isinstance(tail, int):
        lines = lines[-tail:]
    return "\n".join(lines)


def _fs_write_file(op: dict[str, Any], ctx: _FilesystemContext) -> str:
    path = _filesystem_op_path(ctx, op)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(op.get("content", "")), encoding="utf-8")
    return f"write_file OK: {path}"


def _fs_edit_file(op: dict[str, Any], ctx: _FilesystemContext) -> str:
    path = _filesystem_op_path(ctx, op)
    if not path.exists():
        return f"edit_file ERR: not found: {path}"
    old_text = op.get("old_text")
    new_text = op.get("new_text") if "new_text" in op else op.get("new")
    if old_text is None:
        return "edit_file ERR: missing old_text"
    content = path.read_text(encoding="utf-8", errors="replace")
    if str(old_text) not in content:
        return f"edit_file ERR: old_text not found in {path}"
    new_content = content.replace(str(old_text), str(new_text or ""), 1)
    if _parse_bool(op.get("dry_run"), False):
        return f"edit_file DRY_RUN: would replace in {path}"
    path.write_text(new_content, encoding="utf-8")
    return f"edit_file OK: {path}"


def _fs_append_file(op: dict[str, Any], ctx: _FilesystemContext) -> str:
    path = _filesystem_op_path(ctx, op)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(str(op.get("content", "")))
    return f"append_file OK: {path}"


def _fs_list_directory(op: dict[str, Any], ctx: _FilesystemContext) -> str:
    path = _filesystem_op_path(ctx, op)
    if not path.exists():
        return f"list_directory ERR: not found: {path}"
    if not path.is_dir():
        return f"list_directory ERR: not a directory: {path}"
    entries = [
        f"{entry.name}{'/' if entry.is_dir() else ''}"
        for entry in sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    ]
    return "\n".join(entries) if entries else "(empty)"


def _fs_directory_tree(op: dict[str, Any], ctx: _FilesystemContext) -> str:
    path = _filesystem_op_path(ctx, op)
    if not path.exists():
        return f"directory_tree ERR: not found: {path}"
    depth = _parse_int(op.get("depth"), 3, minimum=1)
    lines = [str(path) + "/"]
    lines.extend(_filesystem_tree(path, "", depth))
    return "\n".join(lines)


def _fs_create_directory(op: dict[str, Any], ctx: _FilesystemContext) -> str:
    path = _filesystem_op_path(ctx, op)
    path.mkdir(parents=True, exist_ok=True)
    return f"create_directory OK: {path}"


def _fs_move_file(op: dict[str, Any], ctx: _FilesystemContext) -> str:
    src, dst = _filesystem_source_destination(ctx, op)
    if not src.exists():
        return f"move_file ERR: source not found: {src}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return f"move_file OK: {src} -> {dst}"


def _fs_copy_file(op: dict[str, Any], ctx: _FilesystemContext) -> str:
    src, dst = _filesystem_source_destination(ctx, op)
    if not src.exists():
        return f"copy_file ERR: source not found: {src}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))
    return f"copy_file OK: {src} -> {dst}"


def _fs_delete_file(op: dict[str, Any], ctx: _FilesystemContext) -> str:
    path = _filesystem_op_path(ctx, op)
    if not path.exists():
        return f"delete_file OK: already not exists: {path}"
    if path.is_dir():
        shutil.rmtree(str(path))
    else:
        path.unlink()
    return f"delete_file OK: {path}"


def _fs_file_info(op: dict[str, Any], ctx: _FilesystemContext) -> str:
    path = _filesystem_op_path(ctx, op)
    if not path.exists():
        return f"file_info ERR: not found: {path}"
    stat = path.stat()
    info = {
        "path": str(path),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "size": stat.st_size,
        "mtime": stat.st_mtime,
    }
    return "\n".join(f"{k}: {v}" for k, v in info.items())


def _fs_search_files(op: dict[str, Any], ctx: _FilesystemContext) -> str:
    path = _filesystem_op_path(ctx, op)
    pattern = str(op.get("pattern", "*"))
    if not path.exists():
        return f"search_files ERR: not found: {path}"
    matches = list(path.rglob(pattern))[:100]
    return "\n".join(str(m) for m in matches) if matches else "(no matches)"


def _fs_file_exists(op: dict[str, Any], ctx: _FilesystemContext) -> str:
    path = _filesystem_op_path(ctx, op)
    return f"exists: {path.exists()}"


_FILESYSTEM_OP_HANDLERS: dict[str, Callable[[dict[str, Any], _FilesystemContext], str]] = {
    "read_file": _fs_read_file,
    "write_file": _fs_write_file,
    "edit_file": _fs_edit_file,
    "append_file": _fs_append_file,
    "list_directory": _fs_list_directory,
    "directory_tree": _fs_directory_tree,
    "create_directory": _fs_create_directory,
    "move_file": _fs_move_file,
    "copy_file": _fs_copy_file,
    "delete_file": _fs_delete_file,
    "file_info": _fs_file_info,
    "search_files": _fs_search_files,
    "file_exists": _fs_file_exists,
}


def _execute_filesystem_op(
    op: dict,
    base_dir: Path,
    *,
    skill_dir: Path | None = None,
    prefer_skill_paths: bool = False,
) -> str:
    """Execute a single filesystem operation."""
    normalized_op, op_type = _normalize_filesystem_op(op)
    if not op_type:
        return "unknown op_type: "

    handler = _FILESYSTEM_OP_HANDLERS.get(op_type)
    if handler is None:
        return f"unknown op_type: {op_type}"

    ctx = _FilesystemContext(
        base_dir=base_dir,
        skill_dir=skill_dir,
        prefer_skill_paths=prefer_skill_paths,
    )
    return handler(normalized_op, ctx)


def _execute_filesystem_ops(plan: dict) -> str:
    """Execute all filesystem ops in a plan."""
    ops = plan.get("ops", [])
    if not isinstance(ops, list) or not ops:
        return "ERR: no tool_calls provided"
    raw_working_dir = plan.get("working_dir")
    if raw_working_dir is None or (isinstance(raw_working_dir, str) and not raw_working_dir.strip()):
        base_dir = _default_workspace_base_dir()
    else:
        base_dir = _resolve_dir(Path.cwd().resolve(), raw_working_dir)
    _, skill_dir, prefer_skill_paths = _extract_skill_context(plan)
    results: list[str] = []
    for op in ops:
        if not isinstance(op, dict):
            results.append("SKIP: op is not a dict")
            continue
        try:
            results.append(
                _execute_filesystem_op(
                    dict(op),
                    base_dir,
                    skill_dir=skill_dir,
                    prefer_skill_paths=prefer_skill_paths,
                )
            )
        except Exception as exc:
            op_type = str(op.get("type") or "unknown")
            results.append(f"{op_type} ERR: {exc}")
    return "\n".join(results) if results else "OK"


# ===================================================================
# 5. Terminal executor
# ===================================================================

def _convert_pip_to_uv(command: str, working_dir: Path) -> str:
    """Rewrite pip commands to use uv pip when inside a venv."""
    current = working_dir.resolve()
    for _ in range(5):
        if (current / ".venv").exists():
            command = re.sub(
                r"(^|&&\s*|;\s*|\|\s*)(?:[^\s\"']*python(?:\d+(?:\.\d+)*)?)\s+-m\s+uv\s+pip\b",
                r"\1uv pip",
                command,
            )
            command = re.sub(
                r"(^|&&\s*|;\s*|\|\s*)(?:[^\s\"']*python(?:\d+(?:\.\d+)*)?)\s+-m\s+pip\b",
                r"\1uv pip",
                command,
            )
            command = re.sub(r"(^|&&\s*|;\s*|\|\s*)(?:[^\s\"']*/)?pip(?:3)?\s+", r"\1uv pip ", command)
            break
        parent = current.parent
        if parent == current:
            break
        current = parent
    return command


def _callback(report: list[str], prefix: str):
    """Create a callback that appends messages to report."""

    def _cb(message: str | None):
        if message:
            report.append(f"{prefix}{message}")

    return _cb


def _resolve_terminal_working_dir(
    base_dir: Path,
    op: dict[str, Any],
    *,
    op_name: str,
    report: list[str],
) -> Path:
    raw_op_working_dir = op.get("working_dir")
    working_dir, op_wd_fallback = _resolve_working_dir_or_workspace(
        raw_op_working_dir,
        base_dir=base_dir,
    )
    if op_wd_fallback:
        report.append(
            f"{op_name} WARN: invalid working_dir={raw_op_working_dir!r}; fallback to {working_dir}"
        )
    return working_dir


def _append_terminal_call_result(
    report: list[str],
    *,
    op_name: str,
    call: Callable[[], Any],
) -> None:
    try:
        result = call()
        report.append(f"{op_name} result: {result}")
    except Exception as exc:
        report.append(f"{op_name} ERR: {exc}")


def _prepare_terminal_env_operation(
    op: dict[str, Any],
    *,
    op_name: str,
    base_dir: Path,
    report: list[str],
) -> tuple[str, Path, Callable[[str | None], None]] | None:
    env_path = op.get("env_path")
    if not env_path:
        report.append(f"{op_name} SKIP: missing env_path")
        return None
    working_dir = _resolve_terminal_working_dir(
        base_dir,
        op,
        op_name=op_name,
        report=report,
    )
    resolved_env_path = str(_resolve_dir(base_dir, str(env_path)))
    cb = _callback(report, f"{op_name}: ")
    return resolved_env_path, working_dir, cb


def _execute_terminal_ops(plan: dict) -> str:
    """Execute terminal/shell operations."""
    if terminal_utils is None:
        return f"ERR: camel is not available: {_TERMINAL_IMPORT_ERROR}"
    ops = plan.get("ops", [])
    if not isinstance(ops, list) or not ops:
        return "Invalid tool_calls"

    raw_plan_working_dir = plan.get("working_dir")
    base_dir, plan_wd_fallback = _resolve_working_dir_or_workspace(raw_plan_working_dir)
    skill_name, skill_dir, prefer_skill_paths = _extract_skill_context(plan)
    report: list[str] = []
    if plan_wd_fallback:
        report.append(
            f"terminal WARN: invalid working_dir={raw_plan_working_dir!r}; fallback to {base_dir}"
        )

    for op in ops:
        if not isinstance(op, dict):
            report.append("SKIP op (not a dict)")
            continue
        op_type = _canonicalize_op_type(op.get("type"))

        if op_type == "run_command":
            command = str(op.get("command") or "").strip()
            if not command:
                report.append("run_command SKIP: missing command")
                continue

            working_dir = _resolve_terminal_working_dir(
                base_dir,
                op,
                op_name="run_command",
                report=report,
            )
            command = _convert_pip_to_uv(command, working_dir)
            command = _rewrite_command_paths_for_skill(
                command,
                working_dir=working_dir,
                skill_dir=skill_dir,
                prefer_skill_paths=prefer_skill_paths,
            )

            allowed_commands: Optional[Set[str]] = None
            safe_mode = _parse_bool(op.get("safe_mode"), True)
            use_docker_backend = _parse_bool(op.get("use_docker_backend"), False)
            timeout = _parse_int(op.get("timeout"), 60, minimum=1)

            try:
                is_safe, reason = terminal_utils.check_command_safety(command, allowed_commands)
            except Exception as exc:
                report.append(f"run_command ERR: check_command_safety failed: {exc}")
                continue

            try:
                ok, msg_or_cmd = terminal_utils.sanitize_command(
                    command=command,
                    use_docker_backend=use_docker_backend,
                    safe_mode=safe_mode,
                    working_dir=str(working_dir),
                    allowed_commands=allowed_commands,
                )
            except Exception as exc:
                report.append(f"run_command ERR: sanitize_command failed: {exc}")
                continue

            if not is_safe:
                report.append(f"run_command REFUSED: {reason}")
                continue
            if not ok:
                report.append(f"run_command REFUSED: {msg_or_cmd}")
                continue
            if use_docker_backend:
                report.append("run_command REFUSED: docker backend not supported")
                continue

            venv_dir = _find_venv(working_dir)
            final_cmd = str(msg_or_cmd)
            env = os.environ.copy()
            if venv_dir:
                venv_bin = str(_venv_bin_dir(venv_dir))
                env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
                env["VIRTUAL_ENV"] = str(venv_dir)
            if skill_name:
                env["MEMENTO_SKILL_NAME"] = skill_name
            if skill_dir:
                env["MEMENTO_SKILL_DIR"] = str(skill_dir)

            try:
                proc = subprocess.run(
                    _shell_command(final_cmd),
                    cwd=str(working_dir),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=env,
                )
            except subprocess.TimeoutExpired:
                report.append(f"run_command TIMEOUT after {timeout}s: {msg_or_cmd}")
                continue
            except FileNotFoundError as exc:
                report.append(f"run_command ERR: shell not found: {exc}")
                continue
            except Exception as exc:
                report.append(f"run_command ERR: {exc}")
                continue

            stdout = (proc.stdout or "").strip()
            stderr = (proc.stderr or "").strip()
            if proc.returncode != 0:
                report.append(f"run_command ERR ({proc.returncode}): {stderr or stdout}")
            else:
                report.append(stdout or stderr or "OK")
            continue

        if op_type == "is_uv_environment":
            try:
                result = terminal_utils.is_uv_environment()
                report.append(f"is_uv_environment: {result}")
            except Exception as exc:
                report.append(f"is_uv_environment ERR: {exc}")
            continue

        if op_type == "ensure_uv_available":
            cb = _callback(report, "ensure_uv_available: ")
            try:
                success, uv_path = terminal_utils.ensure_uv_available(cb)
                report.append(f"ensure_uv_available result: {success} {uv_path or ''}".strip())
            except Exception as exc:
                report.append(f"ensure_uv_available ERR: {exc}")
            continue

        if op_type == "setup_initial_env_with_uv":
            prepared = _prepare_terminal_env_operation(
                op,
                op_name="setup_initial_env_with_uv",
                base_dir=base_dir,
                report=report,
            )
            if prepared is None:
                continue
            resolved_env_path, working_dir, cb = prepared
            uv_path = op.get("uv_path")
            if not uv_path:
                try:
                    success, uv_path = terminal_utils.ensure_uv_available(cb)
                except Exception as exc:
                    report.append(f"setup_initial_env_with_uv ERR: {exc}")
                    continue
                if not success or not uv_path:
                    report.append("setup_initial_env_with_uv ERR: uv not available")
                    continue
            _append_terminal_call_result(
                report,
                op_name="setup_initial_env_with_uv",
                call=lambda: terminal_utils.setup_initial_env_with_uv(
                    resolved_env_path,
                    str(uv_path),
                    str(working_dir),
                    cb,
                ),
            )
            continue

        if op_type == "setup_initial_env_with_venv":
            prepared = _prepare_terminal_env_operation(
                op,
                op_name="setup_initial_env_with_venv",
                base_dir=base_dir,
                report=report,
            )
            if prepared is None:
                continue
            resolved_env_path, working_dir, cb = prepared
            _append_terminal_call_result(
                report,
                op_name="setup_initial_env_with_venv",
                call=lambda: terminal_utils.setup_initial_env_with_venv(
                    resolved_env_path,
                    str(working_dir),
                    cb,
                ),
            )
            continue

        if op_type == "clone_current_environment":
            prepared = _prepare_terminal_env_operation(
                op,
                op_name="clone_current_environment",
                base_dir=base_dir,
                report=report,
            )
            if prepared is None:
                continue
            resolved_env_path, working_dir, cb = prepared
            _append_terminal_call_result(
                report,
                op_name="clone_current_environment",
                call=lambda: terminal_utils.clone_current_environment(
                    resolved_env_path,
                    str(working_dir),
                    cb,
                ),
            )
            continue

        if op_type == "check_nodejs_availability":
            cb = _callback(report, "check_nodejs_availability: ")
            try:
                result = terminal_utils.check_nodejs_availability(cb)
                report.append(f"check_nodejs_availability: {result}")
            except Exception as exc:
                report.append(f"check_nodejs_availability ERR: {exc}")
            continue

        report.append(f"unknown op: {op_type}")

    return "\n".join(report) if report else "OK"


# ===================================================================
# 6. UV pip executor
# ===================================================================

def _run_uv_pip(
    args: list[str],
    working_dir: Path,
    venv_dir: Path,
) -> tuple[int, str, str]:
    """Run a `uv pip` subcommand inside the given venv."""
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(venv_dir)
    env["PATH"] = f"{_venv_bin_dir(venv_dir)}{os.pathsep}{env.get('PATH', '')}"
    cmd = ["uv", "pip"] + args
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(working_dir),
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except FileNotFoundError:
        return -1, "", "uv command not found. Please install uv first."
    except Exception as exc:
        return -1, "", str(exc)


def _execute_uv_pip_ops(plan: dict) -> str:
    """Execute uv-pip operations (check/install/list)."""
    ops = plan.get("ops", [])
    if not isinstance(ops, list) or not ops:
        return "Invalid tool_calls"
    report: list[str] = []
    raw_working_dir = plan.get("working_dir")
    working_dir, wd_fallback = _resolve_working_dir_or_workspace(raw_working_dir)
    if wd_fallback:
        report.append(
            f"uv-pip-install WARN: invalid working_dir={raw_working_dir!r}; fallback to {working_dir}"
        )
    venv_dir = _find_venv(working_dir)
    if not venv_dir:
        report.append(f"ERR: No .venv or venv found in {working_dir} or parent directories")
        return "\n".join(report)
    report.append(f"Using venv: {venv_dir}")

    for op in ops:
        if not isinstance(op, dict):
            report.append("SKIP op (not a dict)")
            continue
        op_type = str(op.get("type") or "").strip()

        if op_type == "check":
            package = str(op.get("package", "")).strip()
            if not package:
                report.append("check SKIP: missing package name")
                continue
            returncode, stdout, _stderr = _run_uv_pip(["show", package], working_dir, venv_dir)
            if returncode == 0:
                version = "unknown"
                for line in stdout.split("\n"):
                    if line.startswith("Version:"):
                        version = line.split(":", 1)[1].strip()
                        break
                report.append(f"check OK: {package} is installed (version {version})")
            else:
                report.append(f"check: {package} is NOT installed")
            continue

        if op_type == "install":
            package = str(op.get("package", "")).strip()
            if not package:
                report.append("install SKIP: missing package name")
                continue
            extras = str(op.get("extras", "")).strip()
            pkg_spec = f"{package}{extras}" if extras else package
            returncode, stdout, stderr = _run_uv_pip(["install", pkg_spec], working_dir, venv_dir)
            if returncode == 0:
                output = stdout or stderr or "OK"
                if len(output) > 1000:
                    output = output[:1000] + "\n...[truncated]"
                report.append(f"install OK: {pkg_spec}\n{output}")
            else:
                report.append(f"install ERR: {pkg_spec}\n{stderr or stdout}")
            continue

        if op_type == "list":
            returncode, stdout, stderr = _run_uv_pip(["list"], working_dir, venv_dir)
            if returncode == 0:
                output = stdout or "No packages installed"
                if len(output) > 3000:
                    output = output[:3000] + "\n...[truncated]"
                report.append(f"list OK:\n{output}")
            else:
                report.append(f"list ERR: {stderr or stdout}")
            continue

        report.append(f"unknown op: {op_type}")

    return "\n".join(report) if report else "OK"


# ===================================================================
# 7. Web executor
# ===================================================================

def _normalize_organic_results(items: list[Any], *, limit: int) -> list[dict]:
    """Normalize search-result rows into a stable title/link/snippet/position shape."""
    out: list[dict] = []
    for idx, item in enumerate(items[:limit]):
        row = item if isinstance(item, dict) else {}
        position_raw = row.get("position")
        try:
            position = int(position_raw)
        except Exception:
            position = idx + 1
        out.append(
            {
                "title": str(row.get("title") or "N/A"),
                "link": str(row.get("link") or "N/A"),
                "snippet": str(row.get("snippet") or ""),
                "position": position,
            }
        )
    return out


def _web_google_search(query: str, num_results: int = 10) -> list[dict]:
    """Run a Google search via Serper (fallback: SerpAPI)."""
    q = str(query or "").strip()
    if not q:
        return []
    n = _parse_int(num_results, 10, minimum=1)

    serper_key = (os.getenv("SERPER_API_KEY") or os.getenv("SERPER_DEV_API_KEY") or "").strip()
    serpapi_key = (os.getenv("SERPAPI_API_KEY") or "").strip()
    if serper_key:
        try:
            endpoint = (os.getenv("SERPER_BASE_URL") or "https://google.serper.dev/search").strip()
            payload = {"q": q, "num": max(1, min(n, 20))}
            headers = {
                "X-API-KEY": serper_key,
                "Content-Type": "application/json",
            }
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=20)
            body_preview = str(resp.text or "")[:1000]
            if resp.status_code >= 400:
                raise RuntimeError(f"Serper error {resp.status_code}: {body_preview}")

            try:
                data = resp.json()
            except Exception as exc:
                raise RuntimeError(f"Serper returned non-JSON response: {body_preview}") from exc

            if not isinstance(data, dict):
                raise RuntimeError(f"Serper returned non-dict response: {type(data).__name__}")
            if data.get("error"):
                raise RuntimeError(f"Serper error: {data.get('error')}")
            organic = data.get("organic")
            if organic is None:
                return []
            if not isinstance(organic, list):
                raise RuntimeError(f"Serper organic field is not a list: {type(organic).__name__}")
            return _normalize_organic_results(organic, limit=n)
        except requests.RequestException as exc:
            if not serpapi_key:
                raise RuntimeError(f"Serper request failed: {type(exc).__name__}: {exc}") from exc
        except Exception as serper_exc:
            if not serpapi_key:
                raise RuntimeError(
                    "Serper search failed and no SerpAPI fallback key is set: "
                    f"{type(serper_exc).__name__}: {serper_exc}"
                ) from serper_exc

    if not serpapi_key:
        raise RuntimeError("Missing search API key: set SERPER_API_KEY (or fallback SERPAPI_API_KEY)")

    from serpapi import GoogleSearch

    params = {
        "engine": "google",
        "q": q,
        "api_key": serpapi_key,
        "num": n,
    }
    search = GoogleSearch(params)
    results = search.get_dict() or {}
    if not isinstance(results, dict):
        raise RuntimeError(f"SerpAPI returned non-dict response: {type(results).__name__}")
    if results.get("error"):
        raise RuntimeError(f"SerpAPI error: {results.get('error')}")
    organic = results.get("organic_results")
    if organic is None:
        meta = results.get("search_metadata") if isinstance(results.get("search_metadata"), dict) else {}
        status = meta.get("status") or meta.get("api_status") or "unknown"
        raise RuntimeError(f"SerpAPI returned no organic_results (status={status})")
    if not isinstance(organic, list):
        raise RuntimeError(f"SerpAPI organic_results is not a list: {type(organic).__name__}")
    return _normalize_organic_results(organic, limit=n)


async def _fetch_async(url: str, max_length: int = 50000, raw: bool = False) -> str:
    """Fetch a URL asynchronously using crawl4ai."""
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        content = result.html if raw else result.markdown
        return str(content or "")[:max_length]


def _web_fetch(url: str, max_length: int = 50000, raw: bool = False) -> str:
    """Fetch a URL, handling both running and non-running event loops."""
    try:
        try:
            asyncio.get_running_loop()
            has_loop = True
        except RuntimeError:
            has_loop = False

        if has_loop:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _fetch_async(url, max_length, raw))
                return future.result(timeout=60)
        return asyncio.run(_fetch_async(url, max_length, raw))
    except Exception as exc:
        return f"Error fetching {url}: {exc}"


def _execute_web_ops(plan: dict) -> str:
    """Execute web operations (search, fetch)."""
    ops = plan.get("ops", [])
    if not isinstance(ops, list) or not ops:
        return "ERR: no tool_calls provided. Expected 'query' for search or 'url' for fetch."
    results: list[str] = []

    for op in ops:
        if not isinstance(op, dict):
            results.append("SKIP: op is not a dict")
            continue
        op_type = _canonicalize_op_type(op.get("type"))

        if op_type == "web_search":
            query = str(op.get("query", ""))
            num_results = _parse_int(op.get("num_results"), 10, minimum=1)
            try:
                search_results = _web_google_search(query, num_results=num_results)
                output_parts = []
                for r in search_results:
                    output_parts.append(f"Title: {r.get('title', 'N/A')}")
                    output_parts.append(f"Link: {r.get('link', 'N/A')}")
                    output_parts.append(f"Snippet: {r.get('snippet', 'N/A')}")
                    output_parts.append("---")
                text = "\n".join(output_parts)
                results.append(f"[web_search]\n{text or 'No results found'}")
            except Exception as exc:
                results.append(f"[web_search]\nERR: {exc}")
            continue

        if op_type == "fetch":
            url = str(op.get("url", ""))
            max_length = _parse_int(op.get("max_length"), 50000, minimum=1)
            raw_flag = _parse_bool(op.get("raw"), False)
            content = _web_fetch(url, max_length=max_length, raw=raw_flag)
            results.append(f"[fetch]\n{_truncate_text(content, 50000) or 'No content fetched'}")
            continue

        results.append(f"unknown op_type: {op_type}")

    return "\n\n".join(results) if results else "OK"


# ===================================================================
# 8. Bridge dispatcher & execute_skill_plan
# ===================================================================

SkillPlanHandler = Callable[[dict], str]
TOOL_PROTOCOL_VERSION = "1.0"


@dataclass(frozen=True)
class ToolSchema:
    """Lightweight schema used to validate typed tool-call args."""

    required: tuple[str, ...] = ()
    typed: dict[str, tuple[type, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolSpec:
    """Registry entry describing one callable tool."""

    target_skill: str | None
    forward_working_dir: bool = False
    schema: ToolSchema = field(default_factory=ToolSchema)


@dataclass(frozen=True)
class ToolCall:
    """Typed tool-call envelope used internally by the bridge."""

    id: str
    tool: str
    args: dict[str, Any]
    depends_on: tuple[str, ...] = ()
    policy: dict[str, Any] = field(default_factory=dict)
    protocol_version: str = TOOL_PROTOCOL_VERSION


@dataclass(frozen=True)
class ToolCallResult:
    """Standardized result contract for all tool-call dispatches."""

    ok: bool
    data: str = ""
    error_code: str | None = None
    retryable: bool = False


def _schema(
    *,
    required: tuple[str, ...] = (),
    typed: dict[str, tuple[type, ...]] | None = None,
) -> ToolSchema:
    return ToolSchema(required=required, typed=typed or {})


_SKILL_HANDLERS: dict[str, SkillPlanHandler] = {
    "skill-creator": _execute_skill_creator_plan,
    "filesystem": _execute_filesystem_ops,
    "terminal": _execute_terminal_ops,
    "web-search": _execute_web_ops,
    "uv-pip-install": _execute_uv_pip_ops,
}


def _build_tool_registry() -> dict[str, ToolSpec]:
    """Build the canonical tool registry for bridge dispatch."""
    registry: dict[str, ToolSpec] = {"call_skill": ToolSpec(target_skill=None)}

    def _register_many(
        op_types: set[str],
        target_skill: str,
        *,
        forward_working_dir: bool = False,
    ) -> None:
        for op_type in op_types:
            registry[op_type] = ToolSpec(
                target_skill=target_skill,
                forward_working_dir=forward_working_dir,
            )

    _register_many(FILESYSTEM_OP_TYPES, "filesystem", forward_working_dir=True)
    _register_many(TERMINAL_OP_TYPES, "terminal", forward_working_dir=True)
    _register_many(WEB_OP_TYPES, "web-search")
    _register_many(UV_PIP_OP_TYPES, "uv-pip-install", forward_working_dir=True)

    schema_overrides: dict[str, ToolSchema] = {
        "run_command": _schema(
            required=("command",),
            typed={
                "command": (str,),
                "timeout": (int,),
            },
        ),
        "web_search": _schema(
            required=("query",),
            typed={
                "query": (str,),
                "num_results": (int,),
            },
        ),
        "fetch": _schema(
            required=("url",),
            typed={
                "url": (str,),
                "max_length": (int,),
            },
        ),
        "check": _schema(
            required=("package",),
            typed={"package": (str,)},
        ),
        "install": _schema(
            required=("package",),
            typed={
                "package": (str,),
                "extras": (str,),
            },
        ),
        "read_file": _schema(required=("path",), typed={"path": (str,)}),
        "write_file": _schema(required=("path",), typed={"path": (str,)}),
        "edit_file": _schema(required=("path",), typed={"path": (str,)}),
        "replace_text": _schema(required=("path",), typed={"path": (str,)}),
        "append_file": _schema(required=("path",), typed={"path": (str,)}),
        "list_directory": _schema(required=("path",), typed={"path": (str,)}),
        "directory_tree": _schema(required=("path",), typed={"path": (str,)}),
        "create_directory": _schema(required=("path",), typed={"path": (str,)}),
        "mkdir": _schema(required=("path",), typed={"path": (str,)}),
        "delete_file": _schema(required=("path",), typed={"path": (str,)}),
        "file_info": _schema(required=("path",), typed={"path": (str,)}),
        "file_exists": _schema(required=("path",), typed={"path": (str,)}),
        "setup_initial_env_with_uv": _schema(required=("env_path",), typed={"env_path": (str,)}),
        "setup_initial_env_with_venv": _schema(required=("env_path",), typed={"env_path": (str,)}),
        "clone_current_environment": _schema(required=("env_path",), typed={"env_path": (str,)}),
    }

    for tool_name, schema in schema_overrides.items():
        spec = registry.get(tool_name)
        if spec is None:
            continue
        registry[tool_name] = ToolSpec(
            target_skill=spec.target_skill,
            forward_working_dir=spec.forward_working_dir,
            schema=schema,
        )
    return registry


_TOOL_REGISTRY = _build_tool_registry()


def _normalize_bridge_op_type(op: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Normalize bridge op aliases to canonical op types."""
    normalized_op = dict(op)
    canonical = _canonicalize_op_type(normalized_op.get("type"))
    if canonical:
        normalized_op["type"] = canonical
    return normalized_op, canonical


def _extract_tool_args(op: dict[str, Any]) -> dict[str, Any]:
    """Extract typed-call args from a legacy op payload."""
    parsed_args = _parse_json_object(op.get("args") or op.get("arguments") or op.get("parameters"))
    args = dict(parsed_args)
    for key, value in op.items():
        if key in {
            "type",
            "op",
            "action",
            "tool",
            "id",
            "depends_on",
            "policy",
            "protocol_version",
            "args",
            "arguments",
            "parameters",
        }:
            continue
        args[key] = value
    return args


def _legacy_op_to_tool_call(op: dict[str, Any], *, call_id: str) -> ToolCall:
    """Compatibility layer: convert legacy op entries into typed calls."""
    normalized_op, canonical_tool = _normalize_bridge_op_type(op)
    depends_on_raw = normalized_op.get("depends_on")
    depends_on: tuple[str, ...] = ()
    if isinstance(depends_on_raw, list):
        depends_on = tuple(
            str(item).strip()
            for item in depends_on_raw
            if isinstance(item, (str, int, float)) and str(item).strip()
        )
    policy = dict(normalized_op.get("policy")) if isinstance(normalized_op.get("policy"), dict) else {}
    raw_id = normalized_op.get("id")
    resolved_id = str(raw_id).strip() if isinstance(raw_id, str) and raw_id.strip() else call_id
    raw_protocol = normalized_op.get("protocol_version")
    protocol_version = (
        str(raw_protocol).strip()
        if isinstance(raw_protocol, str) and raw_protocol.strip()
        else TOOL_PROTOCOL_VERSION
    )
    return ToolCall(
        id=resolved_id,
        tool=canonical_tool,
        args=_extract_tool_args(normalized_op),
        depends_on=depends_on,
        policy=policy,
        protocol_version=protocol_version,
    )


def _tool_error(message: str, *, code: str, retryable: bool = False) -> ToolCallResult:
    return ToolCallResult(ok=False, data=message, error_code=code, retryable=retryable)


def _validate_tool_call(call: ToolCall) -> ToolCallResult | None:
    """Validate tool name and args using the registry schema."""
    tool = str(call.tool or "").strip()
    if not tool:
        return _tool_error("ERR: tool_call missing required function.name", code="missing_tool")

    spec = _TOOL_REGISTRY.get(tool)
    if spec is None:
        return _tool_error(f"unknown op type: {tool}", code="unknown_tool")

    if not isinstance(call.args, dict):
        return _tool_error(f"{tool} ERR: args must be an object", code="invalid_args")

    for key in spec.schema.required:
        value = call.args.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            return _tool_error(
                f"{tool} ERR: missing required arg '{key}'",
                code="invalid_args",
            )

    for key, allowed_types in spec.schema.typed.items():
        if key not in call.args or call.args.get(key) is None:
            continue
        value = call.args.get(key)
        if not isinstance(value, allowed_types):
            expected = "/".join(t.__name__ for t in allowed_types)
            return _tool_error(
                f"{tool} ERR: arg '{key}' must be {expected}",
                code="invalid_args",
            )

    return None


def _dispatch_typed_tool_call(call: ToolCall, parent_plan: dict, caller_skill: str) -> ToolCallResult:
    """Dispatch one typed tool call through the registry."""
    invalid = _validate_tool_call(call)
    if invalid is not None:
        return invalid

    tool = call.tool
    if tool == "call_skill":
        target = call.args.get("skill") or call.args.get("name")
        if not isinstance(target, str) or not target.strip():
            return _tool_error("call_skill ERR: missing 'skill' name", code="invalid_args")
        target_name = target.strip()
        if target_name == caller_skill:
            return _tool_error(
                f"call_skill ERR: recursive self-call blocked for '{caller_skill}'",
                code="recursive_call",
            )

        sub_plan = call.args.get("plan") or _parse_json_object(
            call.args.get("args") or call.args.get("arguments")
        )
        if isinstance(sub_plan, list):
            sub_plan = {"tool_calls": sub_plan}
        if not isinstance(sub_plan, dict):
            sub_plan = {}
        if "tool_calls" not in sub_plan and isinstance(call.args.get("tool_calls"), list):
            sub_plan["tool_calls"] = call.args.get("tool_calls")
        if "ops" not in sub_plan and isinstance(call.args.get("ops"), list):
            sub_plan["ops"] = call.args.get("ops")
        if "working_dir" not in sub_plan and parent_plan.get("working_dir"):
            sub_plan["working_dir"] = parent_plan.get("working_dir")
        if (
            target_name in BUILTIN_BRIDGE_SKILLS
            and "_skill_context" not in sub_plan
            and isinstance(parent_plan.get("_skill_context"), dict)
        ):
            sub_plan["_skill_context"] = dict(parent_plan["_skill_context"])
        if not sub_plan:
            return _tool_error("call_skill ERR: missing plan/tool_calls payload", code="invalid_args")
        output = execute_skill_plan(target_name, normalize_plan_shape(sub_plan))
        return ToolCallResult(ok=True, data=output)

    spec = _TOOL_REGISTRY.get(tool)
    if spec is None or not spec.target_skill:
        return _tool_error(f"{tool} ERR: no target skill registered", code="dispatch_config", retryable=True)

    forwarded_op: dict[str, Any] = {"type": tool}
    forwarded_op.update(call.args)
    forwarded_call = _op_to_tool_call(forwarded_op, call_id=f"{call.id}:1")
    if forwarded_call is None:
        return _tool_error(f"{tool} ERR: failed to build forwarded tool_call", code="dispatch_config", retryable=True)
    forwarded: dict[str, Any] = {"tool_calls": [forwarded_call]}
    if parent_plan.get("working_dir") and spec.forward_working_dir:
        forwarded["working_dir"] = parent_plan.get("working_dir")
    if isinstance(parent_plan.get("_skill_context"), dict):
        forwarded["_skill_context"] = dict(parent_plan["_skill_context"])
    output = execute_skill_plan(spec.target_skill, normalize_plan_shape(forwarded))
    return ToolCallResult(ok=True, data=output)


def _format_tool_call_result(result: ToolCallResult) -> str:
    """Convert the structured tool result back to legacy text output."""
    if result.ok:
        return result.data or "OK"
    return result.data or f"ERR: {result.error_code or 'tool_error'}"


def _dispatch_bridge_op(
    op: dict,
    parent_plan: dict,
    caller_skill: str,
    *,
    call_id: str,
) -> str:
    """Route one legacy op via the typed-call protocol and registry."""
    call = _legacy_op_to_tool_call(op, call_id=call_id)
    result = _dispatch_typed_tool_call(call, parent_plan, caller_skill)
    return _format_tool_call_result(result)


def execute_skill_plan(skill_name: str, plan: dict) -> str:
    """Top-level entry point: execute a skill plan by name."""
    normalized = normalize_plan_shape(plan)
    skill = str(skill_name or "").strip()
    if skill:
        normalized["_skill_context"] = _coerce_skill_context(normalized, skill)
    log_event("execute_skill_plan_input", skill_name=skill_name, normalized_plan=normalized)
    ops = normalized.get("ops")
    if not isinstance(ops, list) or not ops:
        result = "ERR: no tool_calls provided"
        log_event("execute_skill_plan_output", skill_name=skill_name, result=result)
        return result

    handler = _SKILL_HANDLERS.get(skill)
    if handler is not None:
        result = handler(normalized)
        log_event("execute_skill_plan_output", skill_name=skill_name, result=result)
        return result

    # Generic skill: dispatch each op individually through the bridge
    outputs: list[str] = []
    for idx, raw_op in enumerate(ops, start=1):
        op = _normalize_op_dict(raw_op)
        if not op:
            outputs.append(f"[op#{idx}] SKIP: op is not a dict")
            continue
        op_type = str(op.get("type") or "unknown")
        out = _dispatch_bridge_op(op, normalized, skill, call_id=f"op#{idx}")
        outputs.append(f"[op#{idx}:{op_type}]\n{out}")

    result = "\n\n".join(outputs) if outputs else "ERR: no executable tool_calls"
    log_event("execute_skill_plan_output", skill_name=skill_name, result=result)
    return result
