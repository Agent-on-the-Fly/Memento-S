"""Skill execution functions extracted from agent.py.

This module contains all plan normalization, skill context helpers,
and executor functions for filesystem, terminal, web, uv-pip, and
skill-creator operations, plus the bridge dispatcher.
"""

import asyncio
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional, Set

from core.config import (
    BUILTIN_BRIDGE_SKILLS,
    FILESYSTEM_OP_TYPES,
    PROJECT_ROOT,
    SKILL_LOCAL_DIR_PREFIXES,
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
    _resolve_runtime_path,
    _safe_subpath,
    _shell_command,
    _skill_local_rel_path,
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




# ===================================================================
# 1. Plan normalization
# ===================================================================

def _normalize_op_dict(op: Any) -> dict[str, Any] | None:
    """Normalize a single op dict, handling wrapper formats like mcp_call."""
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

    name = call.get("name")
    args = call.get("args") or call.get("arguments") or call.get("parameters")

    fn = call.get("function")
    if isinstance(fn, dict):
        name = name or fn.get("name")
        args = args or fn.get("arguments")

    args_dict = _parse_json_object(args)
    if not isinstance(name, str) or not name.strip():
        return None

    op: dict[str, Any] = {"type": name.strip()}
    op.update(args_dict)
    return op


def normalize_plan_shape(plan: Any) -> dict:
    """Ensure plan has a well-formed 'ops' list."""
    if not isinstance(plan, dict):
        return {}

    normalized = dict(plan)

    if not isinstance(normalized.get("ops"), list):
        normalized["ops"] = []

    tool_calls = normalized.get("tool_calls")
    if not normalized["ops"] and isinstance(tool_calls, list):
        converted = []
        for call in tool_calls:
            op = _tool_call_to_op(call)
            if op:
                converted.append(op)
        if converted:
            normalized["ops"] = converted

    if isinstance(normalized.get("ops"), list):
        normalized_ops = []
        for raw_op in normalized["ops"]:
            op = _normalize_op_dict(raw_op)
            if op:
                normalized_ops.append(op)
        normalized["ops"] = normalized_ops

    return normalized


# ===================================================================
# 2. Skill context helpers
# ===================================================================

def _coerce_skill_context(plan: dict, fallback_skill: str) -> dict[str, str]:
    """Build/coerce _skill_context metadata on a plan dict."""
    raw = plan.get("_skill_context")
    ctx: dict[str, str] = dict(raw) if isinstance(raw, dict) else {}

    raw_name = ctx.get("name")
    name = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else fallback_skill.strip()
    ctx["name"] = name

    skill_dir: Path | None = None
    raw_dir = ctx.get("dir")
    if isinstance(raw_dir, str) and raw_dir.strip():
        p = Path(raw_dir.strip())
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        else:
            p = p.resolve()
        if p.exists() and p.is_dir():
            skill_dir = p

    if skill_dir is None:
        skill_dir = _resolve_skill_dir(name)
    if skill_dir is not None:
        ctx["dir"] = str(skill_dir)

    return ctx


def _extract_skill_context(plan: dict) -> tuple[str | None, Path | None, bool]:
    """Extract (skill_name, skill_dir, prefer_skill_paths) from plan."""
    raw = plan.get("_skill_context")
    skill_name: str | None = None
    skill_dir: Path | None = None

    if isinstance(raw, dict):
        raw_name = raw.get("name")
        if isinstance(raw_name, str) and raw_name.strip():
            skill_name = raw_name.strip()
        raw_dir = raw.get("dir")
        if isinstance(raw_dir, str) and raw_dir.strip():
            p = Path(raw_dir.strip())
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            else:
                p = p.resolve()
            if p.exists() and p.is_dir():
                skill_dir = p

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
        return "Invalid ops"

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
                overwrite = bool(op.get("overwrite", True))
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
                max_n = int(op.get("max", 1))
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

    return "\n".join(report) if report else "No ops"


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


def _execute_filesystem_op(
    op: dict,
    base_dir: Path,
    *,
    skill_dir: Path | None = None,
    prefer_skill_paths: bool = False,
) -> str:
    """Execute a single filesystem operation."""
    op_type = str(op.get("type") or "").strip()

    tool_name_mapping = {
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
    op_type = tool_name_mapping.get(op_type, op_type)

    if op_type in {"replace_text"}:
        op_type = "edit_file"
        if "old_text" not in op and "old" in op:
            op["old_text"] = op.get("old")
        if "new_text" not in op and "new" in op:
            op["new_text"] = op.get("new")

    if op_type == "read_file":
        path = _resolve_path(base_dir, op.get("path"), skill_dir=skill_dir, prefer_skill_paths=prefer_skill_paths)
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

    if op_type == "write_file":
        path = _resolve_path(base_dir, op.get("path"), skill_dir=skill_dir, prefer_skill_paths=prefer_skill_paths)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(op.get("content", "")), encoding="utf-8")
        return f"write_file OK: {path}"

    if op_type == "edit_file":
        path = _resolve_path(base_dir, op.get("path"), skill_dir=skill_dir, prefer_skill_paths=prefer_skill_paths)
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
        if bool(op.get("dry_run", False)):
            return f"edit_file DRY_RUN: would replace in {path}"
        path.write_text(new_content, encoding="utf-8")
        return f"edit_file OK: {path}"

    if op_type == "append_file":
        path = _resolve_path(base_dir, op.get("path"), skill_dir=skill_dir, prefer_skill_paths=prefer_skill_paths)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(str(op.get("content", "")))
        return f"append_file OK: {path}"

    if op_type == "list_directory":
        path = _resolve_path(base_dir, op.get("path"), skill_dir=skill_dir, prefer_skill_paths=prefer_skill_paths)
        if not path.exists():
            return f"list_directory ERR: not found: {path}"
        if not path.is_dir():
            return f"list_directory ERR: not a directory: {path}"
        entries = [
            f"{entry.name}{'/' if entry.is_dir() else ''}"
            for entry in sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        ]
        return "\n".join(entries) if entries else "(empty)"

    if op_type == "directory_tree":
        path = _resolve_path(base_dir, op.get("path"), skill_dir=skill_dir, prefer_skill_paths=prefer_skill_paths)
        if not path.exists():
            return f"directory_tree ERR: not found: {path}"
        depth = int(op.get("depth", 3))
        lines = [str(path) + "/"]
        lines.extend(_filesystem_tree(path, "", depth))
        return "\n".join(lines)

    if op_type == "create_directory":
        path = _resolve_path(base_dir, op.get("path"), skill_dir=skill_dir, prefer_skill_paths=prefer_skill_paths)
        path.mkdir(parents=True, exist_ok=True)
        return f"create_directory OK: {path}"

    if op_type == "move_file":
        src = _resolve_path(
            base_dir,
            op.get("src") or op.get("source"),
            skill_dir=skill_dir,
            prefer_skill_paths=prefer_skill_paths,
        )
        dst = _resolve_path(
            base_dir,
            op.get("dst") or op.get("destination"),
            skill_dir=skill_dir,
            prefer_skill_paths=prefer_skill_paths,
        )
        if not src.exists():
            return f"move_file ERR: source not found: {src}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return f"move_file OK: {src} -> {dst}"

    if op_type == "copy_file":
        src = _resolve_path(
            base_dir,
            op.get("src") or op.get("source"),
            skill_dir=skill_dir,
            prefer_skill_paths=prefer_skill_paths,
        )
        dst = _resolve_path(
            base_dir,
            op.get("dst") or op.get("destination"),
            skill_dir=skill_dir,
            prefer_skill_paths=prefer_skill_paths,
        )
        if not src.exists():
            return f"copy_file ERR: source not found: {src}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))
        return f"copy_file OK: {src} -> {dst}"

    if op_type == "delete_file":
        path = _resolve_path(base_dir, op.get("path"), skill_dir=skill_dir, prefer_skill_paths=prefer_skill_paths)
        if not path.exists():
            return f"delete_file OK: already not exists: {path}"
        if path.is_dir():
            shutil.rmtree(str(path))
        else:
            path.unlink()
        return f"delete_file OK: {path}"

    if op_type == "file_info":
        path = _resolve_path(base_dir, op.get("path"), skill_dir=skill_dir, prefer_skill_paths=prefer_skill_paths)
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

    if op_type == "search_files":
        path = _resolve_path(base_dir, op.get("path"), skill_dir=skill_dir, prefer_skill_paths=prefer_skill_paths)
        pattern = str(op.get("pattern", "*"))
        if not path.exists():
            return f"search_files ERR: not found: {path}"
        matches = list(path.rglob(pattern))[:100]
        return "\n".join(str(m) for m in matches) if matches else "(no matches)"

    if op_type == "file_exists":
        path = _resolve_path(base_dir, op.get("path"), skill_dir=skill_dir, prefer_skill_paths=prefer_skill_paths)
        return f"exists: {path.exists()}"

    return f"unknown op_type: {op_type}"


def _execute_filesystem_ops(plan: dict) -> str:
    """Execute all filesystem ops in a plan."""
    ops = plan.get("ops", [])
    if not isinstance(ops, list) or not ops:
        return "ERR: no ops provided"
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


def _execute_terminal_ops(plan: dict) -> str:
    """Execute terminal/shell operations."""
    if terminal_utils is None:
        return f"ERR: camel is not available: {_TERMINAL_IMPORT_ERROR}"
    ops = plan.get("ops", [])
    if not isinstance(ops, list) or not ops:
        return "Invalid ops"

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
        op_type = str(op.get("type") or "").strip()

        if op_type in {"shell"}:
            op_type = "run_command"

        if op_type == "run_command":
            command = str(op.get("command") or "").strip()
            if not command:
                report.append("run_command SKIP: missing command")
                continue

            raw_op_working_dir = op.get("working_dir")
            working_dir, op_wd_fallback = _resolve_working_dir_or_workspace(
                raw_op_working_dir,
                base_dir=base_dir,
            )
            if op_wd_fallback:
                report.append(
                    f"run_command WARN: invalid working_dir={raw_op_working_dir!r}; "
                    f"fallback to {working_dir}"
                )
            command = _convert_pip_to_uv(command, working_dir)
            command = _rewrite_command_paths_for_skill(
                command,
                working_dir=working_dir,
                skill_dir=skill_dir,
                prefer_skill_paths=prefer_skill_paths,
            )

            allowed_commands: Optional[Set[str]] = None
            safe_mode = bool(op.get("safe_mode", True))
            use_docker_backend = bool(op.get("use_docker_backend", False))
            timeout = int(op.get("timeout", 60))

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
            env_path = op.get("env_path")
            if not env_path:
                report.append("setup_initial_env_with_uv SKIP: missing env_path")
                continue
            raw_op_working_dir = op.get("working_dir")
            working_dir, op_wd_fallback = _resolve_working_dir_or_workspace(
                raw_op_working_dir,
                base_dir=base_dir,
            )
            if op_wd_fallback:
                report.append(
                    f"setup_initial_env_with_uv WARN: invalid working_dir={raw_op_working_dir!r}; "
                    f"fallback to {working_dir}"
                )
            cb = _callback(report, "setup_initial_env_with_uv: ")
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
            try:
                result = terminal_utils.setup_initial_env_with_uv(
                    str(_resolve_dir(base_dir, env_path)),
                    str(uv_path),
                    str(working_dir),
                    cb,
                )
                report.append(f"setup_initial_env_with_uv result: {result}")
            except Exception as exc:
                report.append(f"setup_initial_env_with_uv ERR: {exc}")
            continue

        if op_type == "setup_initial_env_with_venv":
            env_path = op.get("env_path")
            if not env_path:
                report.append("setup_initial_env_with_venv SKIP: missing env_path")
                continue
            raw_op_working_dir = op.get("working_dir")
            working_dir, op_wd_fallback = _resolve_working_dir_or_workspace(
                raw_op_working_dir,
                base_dir=base_dir,
            )
            if op_wd_fallback:
                report.append(
                    f"setup_initial_env_with_venv WARN: invalid working_dir={raw_op_working_dir!r}; "
                    f"fallback to {working_dir}"
                )
            cb = _callback(report, "setup_initial_env_with_venv: ")
            try:
                result = terminal_utils.setup_initial_env_with_venv(
                    str(_resolve_dir(base_dir, env_path)),
                    str(working_dir),
                    cb,
                )
                report.append(f"setup_initial_env_with_venv result: {result}")
            except Exception as exc:
                report.append(f"setup_initial_env_with_venv ERR: {exc}")
            continue

        if op_type == "clone_current_environment":
            env_path = op.get("env_path")
            if not env_path:
                report.append("clone_current_environment SKIP: missing env_path")
                continue
            raw_op_working_dir = op.get("working_dir")
            working_dir, op_wd_fallback = _resolve_working_dir_or_workspace(
                raw_op_working_dir,
                base_dir=base_dir,
            )
            if op_wd_fallback:
                report.append(
                    f"clone_current_environment WARN: invalid working_dir={raw_op_working_dir!r}; "
                    f"fallback to {working_dir}"
                )
            cb = _callback(report, "clone_current_environment: ")
            try:
                result = terminal_utils.clone_current_environment(
                    str(_resolve_dir(base_dir, env_path)),
                    str(working_dir),
                    cb,
                )
                report.append(f"clone_current_environment result: {result}")
            except Exception as exc:
                report.append(f"clone_current_environment ERR: {exc}")
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
        return "Invalid ops"
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

def _web_google_search(query: str, num_results: int = 10) -> list[dict]:
    """Run a Google search via SerpAPI."""
    from serpapi import GoogleSearch

    params = {
        "engine": "google",
        "q": query,
        "api_key": os.getenv("SERPAPI_API_KEY", ""),
        "num": num_results,
    }
    if not params["api_key"]:
        raise RuntimeError("SERPAPI_API_KEY is not set")
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
    return organic


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
        return "ERR: no ops provided. Expected 'query' for search or 'url' for fetch."
    results: list[str] = []

    for op in ops:
        if not isinstance(op, dict):
            results.append("SKIP: op is not a dict")
            continue
        op_type = str(op.get("type") or "").strip()
        if op_type in {"search", "google_search"}:
            op_type = "web_search"
        elif op_type in {"fetch_markdown", "fetch_url"}:
            op_type = "fetch"

        if op_type == "web_search":
            query = str(op.get("query", ""))
            num_results = int(op.get("num_results", 10))
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
            max_length = int(op.get("max_length", 50000))
            raw_flag = bool(op.get("raw", False))
            content = _web_fetch(url, max_length=max_length, raw=raw_flag)
            results.append(f"[fetch]\n{_truncate_text(content, 50000) or 'No content fetched'}")
            continue

        results.append(f"unknown op_type: {op_type}")

    return "\n\n".join(results) if results else "OK"


# ===================================================================
# 8. Bridge dispatcher & execute_skill_plan
# ===================================================================

def _dispatch_bridge_op(op: dict, parent_plan: dict, caller_skill: str) -> str:
    """Route a single op to the appropriate skill executor."""
    op_type = str(op.get("type") or "").strip()
    if not op_type:
        return "ERR: op missing required key 'type'"

    lowered = op_type.lower()
    normalized_op = dict(op)

    if lowered == "call_skill":
        target = op.get("skill") or op.get("name")
        if not isinstance(target, str) or not target.strip():
            return "call_skill ERR: missing 'skill' name"
        target_name = target.strip()
        if target_name == caller_skill:
            return f"call_skill ERR: recursive self-call blocked for '{caller_skill}'"

        sub_plan = op.get("plan") or _parse_json_object(op.get("args") or op.get("arguments"))
        if isinstance(sub_plan, list):
            sub_plan = {"ops": sub_plan}
        if not isinstance(sub_plan, dict):
            sub_plan = {}
        if "ops" not in sub_plan and isinstance(op.get("ops"), list):
            sub_plan["ops"] = op.get("ops")
        if "working_dir" not in sub_plan and parent_plan.get("working_dir"):
            sub_plan["working_dir"] = parent_plan.get("working_dir")
        if (
            target_name in BUILTIN_BRIDGE_SKILLS
            and "_skill_context" not in sub_plan
            and isinstance(parent_plan.get("_skill_context"), dict)
        ):
            sub_plan["_skill_context"] = dict(parent_plan["_skill_context"])
        if not sub_plan:
            return "call_skill ERR: missing plan/ops payload"
        return execute_skill_plan(target_name, normalize_plan_shape(sub_plan))

    if lowered in {"shell"}:
        normalized_op["type"] = "run_command"
        lowered = "run_command"
    elif lowered in {"google_search", "search"}:
        normalized_op["type"] = "web_search"
        lowered = "web_search"
    elif lowered in {"fetch_url", "fetch_markdown"}:
        normalized_op["type"] = "fetch"
        lowered = "fetch"

    target_skill = None
    if lowered in FILESYSTEM_OP_TYPES:
        target_skill = "filesystem"
    elif lowered in TERMINAL_OP_TYPES:
        target_skill = "terminal"
    elif lowered in WEB_OP_TYPES:
        target_skill = "web-search"
    elif lowered in UV_PIP_OP_TYPES:
        target_skill = "uv-pip-install"

    if not target_skill:
        return f"unknown op type: {op_type}"

    forwarded: dict[str, Any] = {"ops": [normalized_op]}
    if parent_plan.get("working_dir") and target_skill in {"filesystem", "terminal", "uv-pip-install"}:
        forwarded["working_dir"] = parent_plan.get("working_dir")
    if isinstance(parent_plan.get("_skill_context"), dict):
        forwarded["_skill_context"] = dict(parent_plan["_skill_context"])
    return execute_skill_plan(target_skill, normalize_plan_shape(forwarded))


def execute_skill_plan(skill_name: str, plan: dict) -> str:
    """Top-level entry point: execute a skill plan by name."""
    normalized = normalize_plan_shape(plan)
    skill = str(skill_name or "").strip()
    if skill:
        normalized["_skill_context"] = _coerce_skill_context(normalized, skill)
    log_event("execute_skill_plan_input", skill_name=skill_name, normalized_plan=normalized)
    ops = normalized.get("ops")
    if not isinstance(ops, list) or not ops:
        result = "ERR: no ops provided"
        log_event("execute_skill_plan_output", skill_name=skill_name, result=result)
        return result

    if skill == "skill-creator":
        result = _execute_skill_creator_plan(normalized)
        log_event("execute_skill_plan_output", skill_name=skill_name, result=result)
        return result
    if skill == "filesystem":
        result = _execute_filesystem_ops(normalized)
        log_event("execute_skill_plan_output", skill_name=skill_name, result=result)
        return result
    if skill == "terminal":
        result = _execute_terminal_ops(normalized)
        log_event("execute_skill_plan_output", skill_name=skill_name, result=result)
        return result
    if skill == "web-search":
        result = _execute_web_ops(normalized)
        log_event("execute_skill_plan_output", skill_name=skill_name, result=result)
        return result
    if skill == "uv-pip-install":
        result = _execute_uv_pip_ops(normalized)
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
        out = _dispatch_bridge_op(op, normalized, skill)
        outputs.append(f"[op#{idx}:{op_type}]\n{out}")

    result = "\n\n".join(outputs) if outputs else "ERR: no executable ops"
    log_event("execute_skill_plan_output", skill_name=skill_name, result=result)
    return result
