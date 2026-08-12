# SPDX-License-Identifier: Apache-2.0
"""Shared tools module — cross-module utility layer for tool-related logic.

Single source of truth for:
  - path_boundary: Path boundary and permission logic (cross-platform)
  - tool_security: IGNORE_DIRS for file traversal
  - dependency_aliases: Canonical dependency name normalization

注意：shared/tools/facade.py 已删除，tool 访问入口统一使用 from tools import ...
"""

from __future__ import annotations

from shared.tools.path_boundary import PathBoundary, get_boundary, detect_platform, Platform
from shared.tools.tool_security import IGNORE_DIRS
from shared.tools.dependency_aliases import (
    normalize_dependency_name,
    normalize_dependency_spec,
    strip_version_extras,
    get_dependency_aliases,
)
# tool registry 入口（直接从 tools/ 重导出，避免 shared.tools 消费者改 import 路径）
from tools import (
    get_registry,
    get_tool_schemas,
    get_tools_summary,
    get_tool_stats,
    init_registry,
    bootstrap,
)

__all__ = [
    "PathBoundary",
    "get_boundary",
    "detect_platform",
    "Platform",
    "IGNORE_DIRS",
    "normalize_dependency_name",
    "normalize_dependency_spec",
    "strip_version_extras",
    "get_dependency_aliases",
    # tool registry re-exports（从 tools/ 重导出，保持 shared.tools 消费者的兼容性）
    "get_registry",
    "get_tool_schemas",
    "get_tools_summary",
    "get_tool_stats",
    "init_registry",
    "bootstrap",
]
