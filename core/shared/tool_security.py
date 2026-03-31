"""Shared tool security utilities (utility layer).

This module re-exports path security utilities from middleware layer.
For builtin tools, import directly from middleware.utils.path_security to avoid
circular dependencies.
"""

from __future__ import annotations

# Re-export from middleware to maintain single source of truth
from middleware.utils.path_security import (
    IGNORE_DIRS,
    build_allow_roots,
    coerce_path_to_root,
    is_path_within,
    resolve_path,
    validate_path_arg,
)

__all__ = [
    "IGNORE_DIRS",
    "build_allow_roots",
    "coerce_path_to_root",
    "is_path_within",
    "resolve_path",
    "validate_path_arg",
]
