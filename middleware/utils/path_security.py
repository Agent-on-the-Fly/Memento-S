"""Path security utilities for the middleware layer.

These utilities provide the foundation for path validation and sanitization
used by both builtin tools and core modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Directories to ignore during directory traversal
IGNORE_DIRS = frozenset(
    {
        ".git",
        "node_modules",
        "__pycache__",
        "venv",
        ".venv",
        ".tox",
        ".idea",
        ".mypy_cache",
        ".pytest_cache",
        ".gitignore",
    }
)


def validate_path_arg(path: str) -> str | None:
    """Validate raw path argument shape before path resolution.

    Returns error message if invalid, None if valid.
    """
    if not isinstance(path, str):
        return "ERR: Path must be a string."
    if "\n" in path or "\r" in path:
        return "ERR: Path must not contain newlines. Provide a valid file path, not file content."
    if path.lstrip().startswith("#"):
        return (
            "ERR: Path looks like Markdown content. Provide a file path, not content."
        )
    if len(path) > 4096:
        return "ERR: Path is too long. Provide a valid file path, not content."
    return None


def is_path_within(child: Path, parent: Path) -> bool:
    """Cross-platform check whether *child* resides inside *parent*."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def resolve_path(
    raw: str,
    *,
    workspace_dir: Path,
    base_dir: Path | None = None,
    allow_roots: list[Path] | None = None,
    path_validation_enabled: bool = True,
) -> Path:
    """Resolve a path under configured roots and enforce boundary if enabled.

    Raises:
        PermissionError: If path is outside allowed roots and validation is enabled.
    """
    p = Path(raw)
    resolved_base = base_dir.resolve() if base_dir else workspace_dir.resolve()
    allowed_roots = allow_roots or [workspace_dir.resolve()]

    def _is_allowed(path: Path) -> bool:
        return any(is_path_within(path, root) for root in allowed_roots)

    if not p.is_absolute():
        p = resolved_base / p
        resolved = p.resolve()
        if path_validation_enabled and not _is_allowed(resolved):
            raise PermissionError(
                f"Access denied: Path '{raw}' is outside allowed roots."
            )
        return resolved

    resolved = p.resolve()
    if path_validation_enabled and not _is_allowed(resolved):
        raise PermissionError(f"Access denied: Path '{raw}' is outside allowed roots.")
    return resolved


def coerce_path_to_root(
    raw: str,
    *,
    work_dir: Path,
    allow_roots: list[Path] | None = None,
) -> Path:
    """Coerce any path to be within allowed roots, rewriting if necessary.

    This is the final safety net for builtin tools. Unlike resolve_path which
    raises on violation, this function silently rewrites out-of-bound paths
    to the work_dir.

    Args:
        raw: Raw path string from tool arguments
        work_dir: Base working directory for relative paths and fallback
        allow_roots: Optional list of allowed root directories

    Returns:
        Resolved Path guaranteed to be within work_dir or allow_roots
    """
    if not isinstance(raw, str):
        raw = str(raw) if raw else "."

    p = Path(raw.strip())

    # Relative path: resolve under work_dir
    if not p.is_absolute():
        return (work_dir / p).resolve()

    # Absolute path: check if within allowed roots
    try:
        resolved = p.resolve()
    except (OSError, ValueError):
        return work_dir

    allowed = allow_roots or [work_dir]
    for root in allowed:
        try:
            if is_path_within(resolved, root):
                return resolved
        except (OSError, ValueError):
            continue

    # Out of bounds: rewrite to work_dir (filename only to avoid traversal)
    safe_name = p.name if p.name and p.name != "/" else "file"
    return (work_dir / safe_name).resolve()


def build_allow_roots(
    *,
    workspace_dir: Path,
    skill_root: Path | None = None,
    extra_roots: list[Path] | None = None,
) -> list[Path]:
    """Build ordered, deduplicated allow-roots for path boundary checks."""
    roots: list[Path] = [workspace_dir]
    if skill_root:
        roots.append(skill_root)
    if extra_roots:
        roots.extend(extra_roots)

    seen = set()
    ordered: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(resolved)
    return ordered
