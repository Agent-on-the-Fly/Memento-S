"""Runtime path validation for file operations.

Validates file paths against allow_roots boundaries before tool execution.
This replaces the static restrict_file_ops policy with dynamic path checking.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.skill.execution.tool_bridge.context import PATH_LIKE_KEYS
from middleware.utils.path_security import is_path_within
from utils.logger import get_logger

logger = get_logger(__name__)


# File operation tools that require path validation
FILE_TOOLS = frozenset(
    [
        "read_file",
        "write_file",
        "edit_file",
        "edit_file_by_lines",
        "file_create",
        "list_dir",
    ]
)

# Write-like operations should stay in the primary execution root (allow_roots[0]).
WRITE_TOOLS = frozenset(
    [
        "write_file",
        "edit_file",
        "edit_file_by_lines",
        "file_create",
    ]
)


@dataclass
class ValidationResult:
    """Path validation result."""

    valid: bool
    reason: str = ""
    resolved_path: Path | None = None


def validate_path(
    tool_name: str,
    args: dict[str, Any],
    allow_roots: list[Path],
) -> ValidationResult:
    """Validate all path-like args against allow_roots boundaries.

    Args:
        tool_name: Name of the tool being called
        args: Tool arguments
        allow_roots: List of allowed root directories (from ToolContext)

    Returns:
        ValidationResult with valid status and error reason if invalid
    """
    # Skip non-file tools
    if tool_name not in FILE_TOOLS:
        return ValidationResult(valid=True)

    path_items = [
        (k, v) for k, v in args.items() if k in PATH_LIKE_KEYS and isinstance(v, str)
    ]
    if not path_items:
        return ValidationResult(valid=True)

    def _resolve(raw_path: str) -> tuple[Path | None, str | None]:
        p = Path(raw_path)
        if not p.is_absolute():
            if allow_roots:
                p = allow_roots[0] / p
            else:
                return None, f"Relative path '{raw_path}' cannot be resolved: no allow_roots defined"
        try:
            return p.resolve(), None
        except (OSError, ValueError) as e:
            return None, f"Invalid path '{raw_path}': {e}"

    for key, raw_path in path_items:
        # Guardrail: reject literal '@ROOT' directory segments.
        if "/@ROOT/" in raw_path or raw_path.endswith("/@ROOT"):
            return ValidationResult(
                valid=False,
                reason=(
                    f"[path:{key}] Detected literal '@ROOT' directory in path '{raw_path}'. "
                    "Use @ROOT as an alias (e.g., @ROOT/file.txt) or plain relative paths."
                ),
            )

        resolved, err = _resolve(raw_path)
        if err:
            return ValidationResult(valid=False, reason=f"[path:{key}] {err}")

        if resolved is None:
            return ValidationResult(valid=False, reason=f"[path:{key}] unresolved path")

        # For write-like tools, all path-like args must stay in primary root.
        if tool_name in WRITE_TOOLS and allow_roots:
            primary_root = allow_roots[0]
            try:
                if not is_path_within(resolved, primary_root):
                    return ValidationResult(
                        valid=False,
                        reason=(
                            f"[path:{key}] Write path '{raw_path}' (resolved: {resolved}) must stay under primary root: {primary_root}. "
                            "Hint: use @ROOT-relative paths."
                        ),
                        resolved_path=resolved,
                    )
            except (OSError, ValueError):
                return ValidationResult(
                    valid=False,
                    reason=f"[path:{key}] Failed to validate resolved path: {resolved}",
                    resolved_path=resolved,
                )
            continue

        # Read/list can use any allowed root.
        if allow_roots:
            in_any_root = False
            for root in allow_roots:
                try:
                    if is_path_within(resolved, root):
                        in_any_root = True
                        break
                except (OSError, ValueError):
                    continue
            if not in_any_root:
                allowed_paths = [str(r) for r in allow_roots]
                return ValidationResult(
                    valid=False,
                    reason=(
                        f"[path:{key}] Path '{raw_path}' (resolved: {resolved}) is outside allowed roots: {allowed_paths}. "
                        "Hint: use @ROOT-relative paths."
                    ),
                    resolved_path=resolved,
                )

    return ValidationResult(valid=True)
