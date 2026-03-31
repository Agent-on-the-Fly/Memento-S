"""Facade for builtin tools access.

集中封装 builtin.tools 的访问入口，避免业务层直接依赖 builtin 层实现细节。
"""

from __future__ import annotations

from typing import Any

from builtin.tools import execute_builtin_tool as _execute_builtin_tool
from builtin.tools.registry import (
    BUILTIN_TOOL_REGISTRY,
    BUILTIN_TOOL_SCHEMAS,
    get_tool_schema,
    get_tools_summary,
    is_builtin_tool,
)


async def execute_tool(tool_name: str, args: dict[str, Any]) -> Any:
    """Execute a builtin tool through unified facade."""
    return await _execute_builtin_tool(tool_name, args)


__all__ = [
    "execute_tool",
    "is_builtin_tool",
    "get_tool_schema",
    "get_tools_summary",
    "BUILTIN_TOOL_SCHEMAS",
    "BUILTIN_TOOL_REGISTRY",
]
