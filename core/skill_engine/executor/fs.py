"""Filesystem executor split from skill_executor.py."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from core.config import BUILTIN_BRIDGE_SKILLS, WORKSPACE_DIR
from core.utils.path_utils import _resolve_dir, _resolve_path
from core.skill_engine.skill_resolver import _resolve_skill_dir

from ..executor_utils import canonicalize_op_type, parse_bool, parse_int


def _coerce_existing_dir(raw_dir: Any) -> Path | None:
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
    if not isinstance(raw_ctx, dict):
        return None, None
    raw_name = raw_ctx.get("name")
    skill_name = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else None
    return skill_name, _coerce_existing_dir(raw_ctx.get("dir"))


def _extract_skill_context(plan: dict) -> tuple[str | None, Path | None, bool]:
    skill_name, skill_dir = _read_skill_context(plan.get("_skill_context"))
    if skill_dir is None and skill_name:
        skill_dir = _resolve_skill_dir(skill_name)
    prefer_skill_paths = bool(skill_name and skill_name not in BUILTIN_BRIDGE_SKILLS)
    return skill_name, skill_dir, prefer_skill_paths


def _default_workspace_base_dir() -> Path:
    try:
        WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return WORKSPACE_DIR.resolve()


def _filesystem_tree(
    path: Path,
    prefix: str = "",
    depth: int = 3,
    current_depth: int = 0,
) -> list[str]:
    if current_depth >= depth:
        return []
    lines: list[str] = []
    try:
        entries = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    except PermissionError:
        return [f"{prefix}[permission denied]"]
    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")
        if entry.is_dir():
            extension = "    " if is_last else "│   "
            lines.extend(_filesystem_tree(entry, prefix + extension, depth, current_depth + 1))
    return lines


def filesystem_tree(
    path: Path,
    prefix: str = "",
    depth: int = 3,
    current_depth: int = 0,
) -> list[str]:
    return _filesystem_tree(path, prefix, depth, current_depth)


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
    op_type = canonicalize_op_type(normalized.get("type"))
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
    if parse_bool(op.get("dry_run"), False):
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
    depth = parse_int(op.get("depth"), 3, minimum=1)
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


def execute_filesystem_op(
    op: dict[str, Any],
    base_dir: Path,
    *,
    skill_dir: Path | None = None,
    prefer_skill_paths: bool = False,
) -> str:
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


def execute_filesystem_ops(plan: dict[str, Any]) -> str:
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
                execute_filesystem_op(
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


__all__ = [
    "filesystem_tree",
    "execute_filesystem_op",
    "execute_filesystem_ops",
]
