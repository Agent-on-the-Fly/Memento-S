"""
Base components for built-in tools, including security and path resolution.
"""

from __future__ import annotations

import logging
from pathlib import Path

from middleware.config import g_config
from core.utils.platform import is_path_within

logger = logging.getLogger(__name__)

# --- 1. Global Configuration & Security Core ---

_IGNORE_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    "venv",
    ".venv",
    ".tox",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
}


def _resolve_path(raw: str, base_dir: Path | None = None) -> Path:
    """Safely resolve a path, preventing traversal outside the user data directory.

    When path validation is disabled, resolve paths normally without enforcing
    the data directory boundary.
    """
    p = Path(raw)
    workspace_dir = Path(g_config.paths.workspace_dir)
    resolved_base = base_dir.resolve() if base_dir else workspace_dir

    if not p.is_absolute():
        p = resolved_base / p
        resolved = p.resolve()
        if g_config.paths.path_validation_enabled:
            if not is_path_within(resolved, workspace_dir):
                raise PermissionError(
                    f"Access denied: Path '{raw}' is outside the workspace directory '{workspace_dir}'."
                )
        return resolved

    resolved = p.resolve()
    if g_config.paths.path_validation_enabled:
        if is_path_within(resolved, workspace_dir):
            return resolved
        raise PermissionError(
            f"Access denied: Path '{raw}' is outside the workspace directory '{workspace_dir}'."
        )
    return resolved
